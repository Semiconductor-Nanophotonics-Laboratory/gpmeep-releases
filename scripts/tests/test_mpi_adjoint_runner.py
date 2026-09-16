from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import pathlib
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gpmeep_benchmark_evidence import (  # noqa: E402
    canonical_float64_sha256,
    sha256_file,
)
from gpmeep_provenance import canonical_sha256, file_record, tree_manifest  # noqa: E402


def load_runner():
    path = SCRIPTS / "run-mpi-adjoint-benchmark.py"
    spec = importlib.util.spec_from_file_location("gpmeep_mpi_adjoint_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import MPI adjoint runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepublication_inventory(output: pathlib.Path) -> dict:
    manifest = {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file()
        ],
        "runtime_directories": sorted(
            path.name for path in output.iterdir() if path.is_dir()
        ),
    }
    return {
        "pass": True,
        "epoch": "prepublication-running",
        "manifest": manifest,
        "sha256": canonical_sha256(manifest),
    }


def runtime_closure(root: pathlib.Path):
    package = root / "meep"
    package.mkdir(parents=True, exist_ok=True)
    paths = {
        "python_executable": root / "python",
        "meep_module": package / "__init__.py",
        "extension": package / "_meep.so",
        "libmeep": root / "libmeep.so.38",
    }
    result = {}
    for name, path in paths.items():
        path.write_text(name + "\n", encoding="utf-8")
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = tree_manifest(package, package.parent)
    result["python_package"] = {
        "root": str(package),
        "file_count": manifest["file_count"],
        "sha256": manifest["sha256"],
    }
    loaded = {}
    library_names = {
        "mpi": "libmpi.so",
        "cuda_driver": "libcuda.so.1",
        "cuda_runtime": "libcudart.so.12",
        "hdf5": "libhdf5.so",
    }
    for group in ("mpi", "cuda_driver", "cuda_runtime", "hdf5"):
        path = root / library_names[group]
        path.write_text(group + "\n", encoding="utf-8")
        loaded[group] = [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        ]
    loaded["environment"] = [
        dict(record)
        for group in ("mpi", "cuda_runtime", "hdf5")
        for record in loaded[group]
    ]
    result["loaded_libraries"] = loaded
    return result


def result_payload(gradient=None):
    if gradient is None:
        gradient = [0.25, -0.5]
    raw_direction = [math.sin(index * 0.61803398875) for index in range(len(gradient))]
    direction_norm = math.sqrt(math.fsum(value * value for value in raw_direction))
    direction = [value / direction_norm for value in raw_direction]
    return {
        "schema_version": 4,
        "workload": {
            "resolution": 64,
            "run_time": 40.0,
            "cell_size": 32.0,
            "design_resolution": 20,
            "design_shape": [41, 41],
            "design_variables": len(gradient),
            "yee_cells": 4194304,
            "weights_sha256": "a" * 64,
            "direction_sha256": "b" * 64,
        },
        "result": {
            "objective": 1.5,
            "gradient": gradient,
            "gradient_shape": [len(gradient)],
            "gradient_count": len(gradient),
            "gradient_sha256": canonical_float64_sha256(gradient),
            "gradient_l2": sum(value * value for value in gradient) ** 0.5,
            "gradient_projection": math.fsum(
                value * direction_value
                for value, direction_value in zip(gradient, direction)
            ),
            "gradient_sum": sum(gradient),
        },
    }


def coverage():
    return {
        "transport": "pinned-staging",
        "device_inventory_sha256": "d" * 64,
        "process_environment_sha256": "e" * 64,
        "phases": {
            "forward": {"dft": {"calls": 152, "points": 1314268}},
            "adjoint": {"dft": {"calls": 211, "points": 3620971}},
        }
    }


def lazy_import_gate():
    return {
        "schema_version": 1,
        "capture_point": "immediately-after-meep-and-meep-adjoint-import",
        "deferred_module_roots": ["jax", "matplotlib", "scipy"],
        "deferred_module_loaded": {
            "jax": False,
            "matplotlib": False,
            "scipy": False,
        },
        "core_surface": {
            "meep.Simulation": True,
            "meep.adjoint.DesignRegion": True,
            "meep.adjoint.FourierFields": True,
            "meep.adjoint.OptimizationProblem": True,
        },
        "lazy_surface_discoverable": {
            "meep.Animate2D": True,
            "meep.plot2D": True,
            "meep.adjoint.FilteredSource": True,
            "meep.adjoint.MeepJaxWrapper": True,
            "meep.adjoint.conic_filter": True,
        },
        "pass": True,
    }


def cuda_sample(lane: str, kind: str, iteration: int, elapsed: float, nonce: str):
    identifiers = [f"{1:032x}"] if lane == "cuda-single" else [f"{1:032x}", f"{2:032x}"]
    all_identifiers = [f"{1:032x}", f"{2:032x}"]
    inventory = [
        {
            "ordinal": index,
            "identifier": identifier,
            "name": "gpu",
            "compute_capability": [8, 6],
            "multiprocessor_count": 84,
            "max_threads_per_block": 1024,
            "global_memory_bytes": 1024,
            "compatible": True,
        }
        for index, identifier in enumerate(all_identifiers)
    ]
    inventory_digest = canonical_sha256(inventory)
    inventories = [
        {
            "world_rank": rank,
            "backend": {
                "devices": copy.deepcopy(inventory)
            },
        }
        for rank in range(len(identifiers))
    ]
    rank_coverage = []
    for rank_index, _ in enumerate(identifiers):
        record = coverage()
        record["device_inventory_sha256"] = inventory_digest
        record["transport"] = "none" if lane == "cuda-single" else "pinned-staging"
        if lane == "cuda-multi":
            record["phases"]["forward"]["dft"]["points"] = (
                657134 if rank_index == 0 else 657135
            )
            record["phases"]["adjoint"]["dft"]["points"] = (
                1810485 if rank_index == 0 else 1810486
            )
        rank_coverage.append(record)
    return {
        "lane": lane,
        "sample_kind": kind,
        "iteration": iteration,
        "nonce": nonce,
        "process_seconds": elapsed + 0.3,
        "lazy_import_gate": lazy_import_gate(),
        "result": {
            **result_payload(),
            "distributed": {"ranks": inventories},
        },
        "validation": {
            "objective": 1.5,
            "elapsed_seconds": elapsed,
            "workload_wall_seconds": elapsed,
            "device_identifiers": identifiers,
            "rank_coverage": rank_coverage,
            "rank_runtimes": [
                {
                    "loaded_libraries": {
                        "environment": [
                            {
                                "path": f"/synthetic/{lane}/rank-{rank_index}/libmpi.so",
                                "sha256": f"{rank_index + 1:064x}",
                            }
                        ]
                    }
                }
                for rank_index in range(len(identifiers))
            ],
            "rank_driver": [
                {
                    "hostname": f"node-{rank_index}",
                    "executable": {
                        "path": "/usr/bin/nvidia-smi",
                        "size_bytes": 1,
                        "sha256": "9" * 64,
                    },
                    "inventory_identifiers": [identifier],
                    "selected_identifier": identifier,
                }
                for rank_index, identifier in enumerate(identifiers)
            ],
        },
    }


def cpu_sample(kind: str, iteration: int, elapsed: float, nonce: str):
    rank_coverage = []
    for rank_index in range(2):
        record = coverage()
        record["transport"] = "none"
        record["phases"]["forward"]["dft"]["points"] = (
            657134 if rank_index == 0 else 657135
        )
        record["phases"]["adjoint"]["dft"]["points"] = (
            1810485 if rank_index == 0 else 1810486
        )
        rank_coverage.append(record)
    return {
        "lane": "cpu-2x4",
        "sample_kind": kind,
        "iteration": iteration,
        "nonce": nonce,
        "process_seconds": elapsed + 0.3,
        "lazy_import_gate": lazy_import_gate(),
        "result": result_payload(),
        "validation": {
            "objective": 1.5,
            "elapsed_seconds": elapsed - 0.1,
            "workload_wall_seconds": elapsed,
            "affinity": {
                "pass": True,
                "hostname": "node-0",
                "binding": "core",
                "omp_threads": 4,
                "used_physical_cores": [[0, index] for index in range(8)],
                "rank_affinity": [
                    {"world_rank": 0, "physical_cores": [[0, index] for index in range(4)]},
                    {"world_rank": 1, "physical_cores": [[0, index] for index in range(4, 8)]},
                ],
            },
            "rank_coverage": rank_coverage,
        },
    }


FIXED_MEASURED_REPEATS = 5


def sample_matrix():
    samples = [
        {
            "lane": "cpu-legacy-oracle",
            "sample_kind": "measured",
            "iteration": 0,
            "nonce": "oracle",
            "process_seconds": 1.0,
            "lazy_import_gate": lazy_import_gate(),
            "result": result_payload(),
            "validation": {
                "objective": 1.5,
                "gradient": [0.25, -0.5],
            },
            "material_gradient_gate": {
                "pass": True,
                "legacy_requests": 2,
            },
        }
    ]
    samples.extend(
        (
            cpu_sample("warmup", 0, 4.0, "cpu-warmup"),
            cuda_sample("cuda-single", "warmup", 0, 2.0, "single-warmup"),
            cuda_sample("cuda-multi", "warmup", 0, 1.5, "multi-warmup"),
        )
    )
    orders = (
        ("cpu-2x4", "cuda-single", "cuda-multi"),
        ("cuda-multi", "cuda-single", "cpu-2x4"),
        ("cuda-single", "cuda-multi", "cpu-2x4"),
        ("cpu-2x4", "cuda-multi", "cuda-single"),
    )
    for iteration in range(FIXED_MEASURED_REPEATS):
        records = {
            "cpu-2x4": cpu_sample(
                "measured", iteration, 4.0, f"cpu-{iteration}"
            ),
            "cuda-single": cuda_sample(
                "cuda-single", "measured", iteration, 2.0, f"single-{iteration}"
            ),
            "cuda-multi": cuda_sample(
                "cuda-multi", "measured", iteration, 1.5, f"multi-{iteration}"
            ),
        }
        samples.extend(records[lane] for lane in orders[iteration % len(orders)])
    return samples


def synthetic_host_audit(runner, samples):
    logical_cpus = list(range(runner.PROFILE["cpu_required_physical_cores"]))
    power_state = {
        "logical_cpus": logical_cpus,
        "global": {},
        "per_cpu": {
            str(cpu): {
                "cpufreq": {},
                "thermal_throttle": {
                    "core_throttle_count": {"status": "absent"},
                    "package_throttle_count": {"status": "absent"},
                },
            }
            for cpu in logical_cpus
        },
        "loadavg": {"load_1m": 1.0, "load_5m": 1.0, "load_15m": 1.0},
    }
    topology = {
        "schema_version": 1,
        "captured_at_utc": "2026-08-13T00:00:00Z",
        "parent_pid": 1,
        "parent_affinity": logical_cpus,
        "normalized": {
            "available_logical_cpus": logical_cpus,
            "physical_core_count": len(logical_cpus),
            "cores": [
                {
                    "socket": 0,
                    "core": cpu,
                    "node": 0,
                    "logical_cpus": [cpu],
                }
                for cpu in logical_cpus
            ],
        },
        "power_load_state_start": copy.deepcopy(power_state),
        "summary": {"Model name": "synthetic-cpu"},
        "lscpu_executable": {
            "path": "/usr/bin/lscpu",
            "size_bytes": 1,
            "sha256": "c" * 64,
        },
        "probes": {},
    }
    module = runner._cpu_host_audit_module()
    contention_segments = []
    for sample_index, sample in enumerate(samples):
        process_usec = float(sample.get("process_seconds", 1.0)) * 1.0e6
        elapsed_usec = process_usec + 1000.0
        start_ns = 1_000_000_000 + sample_index * 2_000_000
        end_ns = start_ns + int(elapsed_usec * 1000)
        contention_segments.append(
            {
                "label": sample.get(
                    "artifact_label",
                    f"{sample['lane']}-{sample['sample_kind']}-{sample['iteration']}",
                ),
                "lane": sample["lane"],
                "sample_kind": sample["sample_kind"],
                "iteration": sample["iteration"],
                "start": {
                    "monotonic_ns": start_ns,
                    "some_total_usec": 1000 + sample_index,
                    "full_total_usec": 100,
                },
                "end": {
                    "monotonic_ns": end_ns,
                    "some_total_usec": 1000 + sample_index,
                    "full_total_usec": 100,
                },
                "elapsed_usec": elapsed_usec,
                "some_pressure_fraction": 0.0,
                "full_pressure_fraction": 0.0,
            }
        )
    return {
        "schema_version": 1,
        "scope": runner.PROFILE["cpu_baseline_scope"],
        "hostname": "node-0",
        "cpu_model": "synthetic-cpu",
        "topology": topology,
        "topology_identity_sha256": canonical_sha256(
            module._topology_identity(topology)
        ),
        "contention_segments": contention_segments,
        "power_load_state_end": copy.deepcopy(power_state),
        "power_load_comparability": module.validate_power_load_comparability(
            topology["power_load_state_start"], power_state
        ),
    }


def bind_synthetic_host(runner, comparison, samples):
    record = synthetic_host_audit(runner, samples)
    runner.bind_host_audit(
        comparison,
        record,
        samples=samples,
        repo=SCRIPTS.parent,
        verify_live=False,
    )
    return record


def cpu_backend_counters():
    counter = {"calls": 1, "points": 16}
    zero = {"calls": 0, "points": 0}
    names = ("curl", "update_eh", "source", "boundary", "dft", "polarization")
    return {
        "cpu": {
            name: dict(zero if name == "polarization" else counter)
            for name in names
        },
        "cuda": {name: dict(zero) for name in names},
    }


def cpu_statistics_delta(active=False):
    calls = 1 if active else 0
    points = 16 if active else 0
    return {
        "boundaries": {
            "cpu_boundary_calls": calls,
            "cpu_boundary_points": points,
            "cuda_boundary_calls": 0,
            "cuda_boundary_points": 0,
        },
        "boundary_eh_overlap": {
            name: 0
            for name in (
                "boundary_eh_overlap_checks",
                "boundary_eh_overlap_eligible",
                "boundary_eh_overlap_launched_e",
                "boundary_eh_overlap_launched_h",
                "boundary_eh_overlap_rejected",
                "boundary_eh_overlap_skipped_cold_topology",
                "boundary_eh_overlap_skipped_disabled",
                "boundary_eh_overlap_skipped_no_remote",
                "boundary_eh_overlap_skipped_unsupported_schedule",
            )
        },
        "dfts": {
            "cpu_dft_calls": calls,
            "cpu_dft_points": points,
            "cuda_dft_calls": 0,
            "cuda_dft_points": 0,
        },
        "dispatch": {
            "cpu_curl_calls": calls,
            "cpu_curl_points": points,
            "cuda_curl_calls": 0,
            "cuda_curl_points": 0,
            "device_to_host_bytes": 0,
            "host_to_device_bytes": 0,
        },
        "field_updates": {
            "cpu_update_eh_calls": calls,
            "cpu_update_eh_points": points,
            "cuda_update_eh_calls": 0,
            "cuda_update_eh_points": 0,
        },
        "halo_curl_overlap": {
            name: 0
            for name in (
                "halo_curl_overlap_checks",
                "halo_curl_overlap_eligible",
                "halo_curl_overlap_full_points",
                "halo_curl_overlap_interior_points",
                "halo_curl_overlap_launches",
                "halo_curl_overlap_rejected_feature",
                "halo_curl_overlap_rejected_small",
                "halo_curl_overlap_shell_points",
                "halo_curl_overlap_skipped_cold_topology",
                "halo_curl_overlap_skipped_disabled",
                "halo_curl_overlap_skipped_no_remote",
                "halo_curl_overlap_skipped_unsupported_schedule",
            )
        },
        "ldos": {
            name: 0
            for name in (
                "cpu_ldos_reduction_calls",
                "cpu_ldos_source_points",
                "cuda_ldos_descriptor_uploads",
                "cuda_ldos_kernel_launches",
                "cuda_ldos_reduction_calls",
                "cuda_ldos_result_device_to_host_bytes",
                "cuda_ldos_source_points",
                "cuda_ldos_submitted_profiles",
                "ldos_full_field_device_to_host_bytes_avoided",
            )
        },
        "mpi_completion": {
            "mpi_waitall_executions": 0,
            "mpi_waitsome_executions": 0,
        },
        "multi_gpu": {
            "mpi_messages": 0,
            "mpi_scalars": 0,
            "cuda_aware_bytes": 0,
            "pinned_staging_bytes": 0,
            "pinned_device_to_host_bytes": 0,
            "pinned_host_to_device_bytes": 0,
        },
        "polarizations": {
            "cpu_polarization_calls": 0,
            "cpu_polarization_points": 0,
            "cuda_polarization_calls": 0,
            "cuda_polarization_points": 0,
        },
        "resident": {
            name: 0
            for name in (
                "device_buffer_allocations",
                "device_buffer_reuses",
                "device_to_host_bytes_avoided",
                "host_to_device_bytes_avoided",
                "live_resident_device_buffers",
            )
        },
        "runtime": {
            "runtime_availability_probes": 0,
            "runtime_device_enumerations": 0,
            "runtime_device_selections": 0,
        },
        "sources": {
            "cpu_source_calls": calls,
            "cpu_source_points": points,
            "cuda_source_calls": 0,
            "cuda_source_points": 0,
        },
        "tile_coalescing": {
            "tile_coalesced_curl_chunk_phases": 0,
            "tile_coalesced_curl_input_tiles": 0,
            "tile_coalesced_update_eh_chunk_phases": 0,
            "tile_coalesced_update_eh_input_tiles": 0,
        },
    }


def cpu_final_statistics():
    return {
        group: {field: 2 * value for field, value in values.items()}
        for group, values in cpu_statistics_delta(active=True).items()
    }


def cpu_affinity_raw():
    ranks = []
    for rank_index in range(2):
        first_core = 4 * rank_index
        threads = [
            {
                "thread_id": 1000 + rank_index * 10 + core,
                "logical_cpus": [2 * core, 2 * core + 1],
            }
            for core in range(first_core, first_core + 4)
        ]
        logical_union = sorted(
            cpu for thread in threads for cpu in thread["logical_cpus"]
        )
        ranks.append(
            {
                "hostname": "cpu-node",
                "cpu_affinity": logical_union,
                "completion_thread_cpu_affinity": threads[0]["logical_cpus"],
                "process_thread_cpu_affinities": {
                    "thread_count": 4,
                    "logical_cpu_union": logical_union,
                    "threads": threads,
                },
            }
        )
    producer = {
        name: copy.deepcopy(ranks[0][name])
        for name in (
            "cpu_affinity",
            "completion_thread_cpu_affinity",
            "process_thread_cpu_affinities",
        )
    }
    producer["hostname"] = "cpu-node"
    return {
        "producer": producer,
        "distributed": {"ranks": ranks},
    }


def cpu_benchmark_raw():
    payload = result_payload()
    backend = {
        "requested": "cpu",
        "active": "cpu",
        "compiled": True,
        "single_precision": True,
        "strict_cuda_marker": False,
    }
    ranks = []
    phase_times = (
        {
            "update_design_seconds": 0.1,
            "forward_seconds": 1.0,
            "adjoint_seconds": 2.0,
            "gradient_seconds": 0.5,
            "elapsed_seconds": 3.6,
            "workload_wall_seconds": 4.0,
        },
        {
            "update_design_seconds": 0.2,
            "forward_seconds": 1.1,
            "adjoint_seconds": 2.1,
            "gradient_seconds": 0.6,
            "elapsed_seconds": 4.0,
            "workload_wall_seconds": 4.1,
        },
    )
    for rank_index in range(2):
        phases = {
            phase: {
                "elapsed_seconds": phase_times[rank_index][f"{phase}_seconds"],
                "backend_counters": cpu_backend_counters(),
                "statistics_delta": cpu_statistics_delta(active=True),
            }
            for phase in ("forward", "adjoint")
        }
        phases.update(
            {
                phase: {
                    "elapsed_seconds": phase_times[rank_index][f"{phase}_seconds"],
                    "backend_counters": {
                        backend_name: {
                            counter_name: {"calls": 0, "points": 0}
                            for counter_name in (
                                "curl",
                                "update_eh",
                                "source",
                                "boundary",
                                "dft",
                                "polarization",
                            )
                        }
                        for backend_name in ("cpu", "cuda")
                    },
                    "statistics_delta": cpu_statistics_delta(active=False),
                }
                for phase in ("update_design", "gradient")
            }
        )
        ranks.append(
            {
                "world_rank": rank_index,
                "objective": payload["result"]["objective"],
                "gradient": list(payload["result"]["gradient"]),
                "backend": copy.deepcopy(backend),
                "active_backend": "cpu",
                "final_statistics": cpu_final_statistics(),
                "phases": phases,
                "timing": phase_times[rank_index],
            }
        )
    top_phases = copy.deepcopy(ranks[0]["phases"])
    for phase_name in ("update_design", "forward", "adjoint", "gradient"):
        top_phases[phase_name]["elapsed_seconds"] = max(
            rank["timing"][f"{phase_name}_seconds"] for rank in ranks
        )
    return {
        "schema_version": 4,
        "workload": {
            **payload["workload"],
            "final_statistics": copy.deepcopy(ranks[0]["final_statistics"]),
            "phases": top_phases,
        },
        "result": payload["result"],
        "backend": copy.deepcopy(backend),
        "timing": {
            "update_design_seconds": 0.2,
            "forward_seconds": 1.1,
            "adjoint_seconds": 2.1,
            "gradient_seconds": 0.6,
            "elapsed_seconds": 4.0,
            "rank_local_elapsed_seconds": 3.6,
            "rank_local_workload_wall_seconds": 4.0,
            "workload_wall_seconds": 4.1,
        },
        "distributed": {
            "world_size": 2,
            "timing_semantics": "barrier-delimited maximum rank elapsed per phase",
            "gate": {
                "pass": True,
                "objective": True,
                "gradient": True,
                "strict_cuda_coverage": True,
            },
            "ranks": ranks,
        },
    }


def fd_oracle():
    records = [
        {
            "name": name,
            "finite_difference": 0.25,
            "adjoint_projection": 0.25,
            "rtol": 8.0e-2,
            "atol": 2.0e-5,
            "pass": True,
        }
        for name in (
            "cosine",
            "quasiperiodic-sine",
            "component-6",
            "component-12",
            "component-18",
        )
    ]
    return {
        "pass": True,
        "kind": "installed-multi-direction-and-component-finite-difference",
        "directions": len(records),
        "direction_records": records,
    }


def publication_capability(runner, output, run_id, samples=None, oracle=None):
    capability = runner._RunPublicationCapability(output, run_id)
    for sample in samples or []:
        label = sample["artifact_label"]
        key = f"sample:{label}"
        capability.record_measurement(
            key,
            {
                "role": "sample",
                "run_id": run_id,
                "output": str(output.resolve()),
                "label": label,
                "lane": sample["lane"],
                "sample_kind": sample["sample_kind"],
                "iteration": sample["iteration"],
                "nonce": sample["nonce"],
                "process_seconds": sample["process_seconds"],
                "command_sha256": canonical_sha256(sample["command"]),
                "environment_sha256": canonical_sha256(sample["environment"]),
                "artifact_paths": {
                    name: sample[name]
                    for name in (
                        "result_file",
                        "stdout_log",
                        "stderr_log",
                        "timing_file",
                    )
                },
            },
        )
        capability.record_artifacts(
            key,
            {
                "role": "sample",
                "label": label,
                "artifact_sha256": {
                    "result_file": sample["result_sha256"],
                    "stdout_log": sample["stdout_sha256"],
                    "stderr_log": sample["stderr_sha256"],
                    "timing_file": sample["timing_sha256"],
                },
            },
        )
    if oracle is not None:
        label = f"{run_id}-directional-fd-oracle"
        key = f"fd:{label}"
        capability.record_measurement(
            key,
            {
                "role": "finite-difference-oracle",
                "run_id": run_id,
                "output": str(output.resolve()),
                "label": label,
                "process_seconds": oracle["process_seconds"],
                "started_at_utc": oracle["started_at_utc"],
                "completed_at_utc": oracle["completed_at_utc"],
                "command_sha256": canonical_sha256(oracle["command"]),
                "environment_sha256": canonical_sha256(oracle["environment"]),
                "artifact_paths": {
                    name: oracle[name]
                    for name in ("stdout_log", "stderr_log", "timing_file")
                },
            },
        )
        capability.record_artifacts(
            key,
            {
                "role": "finite-difference-oracle",
                "label": label,
                "artifact_sha256": {
                    "stdout_log": oracle["stdout_sha256"],
                    "stderr_log": oracle["stderr_sha256"],
                    "timing_file": oracle["timing_sha256"],
                },
            },
        )
    capability.seal()
    return capability


class MpiAdjointRunnerTests(unittest.TestCase):
    def test_lazy_import_contract_requires_deferred_optional_stacks(self):
        payload = {"producer": {"lazy_import_contract": lazy_import_gate()}}
        validated = self.runner._validate_lazy_import_contract(
            payload, label="fixture"
        )
        self.assertIs(validated["pass"], True)

        for section, field, value in (
            ("deferred_module_loaded", "scipy", True),
            ("core_surface", "meep.Simulation", False),
            ("lazy_surface_discoverable", "meep.plot2D", False),
        ):
            changed = copy.deepcopy(payload)
            changed["producer"]["lazy_import_contract"][section][field] = value
            with self.subTest(section=section, field=field), self.assertRaisesRegex(
                RuntimeError, "lazy import contract"
            ):
                self.runner._validate_lazy_import_contract(
                    changed, label="fixture"
                )

    def test_receipt_runtime_cache_propagates_across_every_rank(self):
        cache = object()
        validation = {"rank_runtimes": [{"rank": 0}, {"rank": 1}]}
        gates = [
            {"loaded_library_origins": {"rank": 0}, "receipt_id": "r"},
            {"loaded_library_origins": {"rank": 1}, "receipt_id": "r"},
        ]
        with mock.patch.object(
            self.runner.COMPARATOR,
            "validate_receipt_runtime",
            side_effect=gates,
        ) as validate:
            result = self.runner._validate_receipt_rank_runtimes(
                {"receipt_id": "r"}, validation, hash_cache=cache
            )
        self.assertEqual(result["rank_count"], 2)
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(
            all(call.kwargs.get("hash_cache") is cache for call in validate.call_args_list)
        )

    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_cli_does_not_expose_release_thresholds(self):
        with self.assertRaises(SystemExit):
            self.runner.parse_args(["--minimum-speedup", "1.0"])
        with self.assertRaises(TypeError):
            self.runner.PROFILE["minimum_median_speedup"] = 0.01
        self.runner._assert_fixed_profile()

    def test_actual_build_runs_only_in_clean_audited_worker(self):
        source = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        self.assertTrue(source.startswith("#!/bin/bash -p\n"))
        self.assertIn('if [[ "$-" != *p* ]]', source)
        self.assertIn('/bin/bash -p "${BASH_SOURCE[0]}" audited-worker', source)
        self.assertNotIn("MEEP_GPU_ENV_ACTIVE", source)
        self.assertNotIn('BUILD_STAGE}" == "prepare', source)
        worker_entry = source.index(
            'if [[ "${BUILD_STAGE}" != "audited-worker" || "$#" -ne 0 ]]'
        )
        force_reinstall = source.index("--force-reinstall", worker_entry)
        no_pyc = source.index("--no-pyc", force_reinstall)
        normalize_bytecode = source.index(
            '"${SCRIPT_DIR}/normalize-conda-generated-bytecode.py"', no_pyc
        )
        byte_audit = source.index('"${SCRIPT_DIR}/audit-conda-prefix.py"', force_reinstall)
        issue_attestation = source.index(
            '"${FRESH_ATTESTATION_SCRIPT}" create', byte_audit
        )
        canonical_capture = source.index(
            '"${SCRIPT_DIR}/capture-build-environment.py"', issue_attestation
        )
        worker_prefix = source[worker_entry:canonical_capture]
        self.assertIn('if [[ "${PWD}" != "${REPO_ROOT}" ]]', worker_prefix)
        self.assertNotIn('cd "${REPO_ROOT}"', worker_prefix)
        actual_capture = source.index(
            '"${SCRIPT_DIR}/capture-build-environment.py"', canonical_capture + 1
        )
        exact_compare = source.index("/usr/bin/cmp -s", actual_capture)
        worker_verification = source.index(
            '"${FRESH_ATTESTATION_SCRIPT}" verify', exact_compare
        )
        configure = source.index('"../../configure"', worker_verification)
        build = source.index('make -j"${MAKE_JOBS}"', configure)
        cuda_cache_disable = source.index("export CUDA_CACHE_DISABLE=1", build)
        fontconfig_export = source.index(
            'export FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}"',
            cuda_cache_disable,
        )
        first_qualification = source.index(
            'in-place-singleton-import.log', fontconfig_export
        )
        self.assertLess(force_reinstall, byte_audit)
        self.assertLess(no_pyc, byte_audit)
        self.assertLess(no_pyc, normalize_bytecode)
        self.assertLess(normalize_bytecode, byte_audit)
        self.assertLess(byte_audit, issue_attestation)
        self.assertLess(issue_attestation, canonical_capture)
        self.assertLess(canonical_capture, actual_capture)
        self.assertLess(actual_capture, exact_compare)
        self.assertLess(exact_compare, worker_verification)
        self.assertLess(worker_verification, configure)
        self.assertLess(configure, build)
        self.assertLess(build, cuda_cache_disable)
        self.assertLess(cuda_cache_disable, fontconfig_export)
        self.assertLess(fontconfig_export, first_qualification)
        self.assertIn("--safety-checks enabled --extra-safety-checks", source)
        self.assertIn(
            '/usr/bin/grep -aFq -- "${required_diagnostic}" "${log}"', source
        )
        self.assertIn(
            '--configuration-file "qualification_fontconfig=${QUALIFICATION_FONTCONFIG}"',
            source,
        )

    def test_m810_qualifications_bind_actual_libtool_elfs(self):
        source = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        explicit_check_program_build = source.index(
            'make -C tests -j"${MAKE_JOBS}"'
        )
        materializations = [
            source.index(
                f"gpu-{name}-libtool-materialization.log",
                explicit_check_program_build,
            )
            for name in ("backend", "step-db", "mpi-performance")
        ]
        actual_elfs = [
            source.index(f".libs/lt-gpu-{name}", materializations[-1])
            for name in ("mpi-performance", "step-db", "backend")
        ]
        provenance = source.index(
            "record_direct_elf_provenance gpu-mpi-performance",
            max(actual_elfs),
        )
        make_check = source.index("make -C tests check", provenance)
        first_direct_qualification = source.index(
            '"${GPU_MPI_PERFORMANCE_ELF}"', provenance + 1
        )
        self.assertTrue(
            all(explicit_check_program_build < index for index in materializations)
        )
        self.assertTrue(all(materializations[-1] < index for index in actual_elfs))
        self.assertLess(max(actual_elfs), provenance)
        self.assertLess(provenance, make_check)
        self.assertLess(provenance, first_direct_qualification)
        self.assertIn(
            '--artifact "gpu_step_db_test=${GPU_STEP_DB_ELF}"', source
        )
        self.assertIn(
            '--artifact "gpu_mpi_performance=${GPU_MPI_PERFORMANCE_ELF}"',
            source,
        )
        self.assertNotIn(
            '--artifact "gpu_mpi_performance=${BUILD_DIR}/tests/.libs/gpu-mpi-performance"',
            source,
        )
        self.assertIn(
            r"/usr/bin/awk '$1 ~ /^libmeep\.so/ {print $3; exit}'",
            source,
        )
        self.assertNotIn(r"/^libmeep\\.so/", source)

    def test_direct_multi_gpu_qualifications_pin_release_initial_condition(self):
        source = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'RELEASE_INITIAL_CONDITION="trigonometric-v1"', source
        )
        self.assertEqual(
            self.runner.RELEASE_INITIAL_CONDITION, "trigonometric-v1"
        )
        self.assertIn('RELEASE_SOURCE_PROFILE="single-ez-v1"', source)
        self.assertEqual(self.runner.RELEASE_SOURCE_PROFILE, "single-ez-v1")
        self.assertIn(
            "MEEP_GPU_MULTI_INITIAL_CONDITION",
            self.runner.ALLOWED_CANONICAL_BUILD_ENVIRONMENT_KEYS,
        )
        self.assertIn(
            "MEEP_GPU_MULTI_SOURCE_PROFILE",
            self.runner.ALLOWED_CANONICAL_BUILD_ENVIRONMENT_KEYS,
        )
        self.assertIn(
            '--env "MEEP_GPU_MULTI_INITIAL_CONDITION=${RELEASE_INITIAL_CONDITION}"',
            source,
        )
        self.assertEqual(
            source.count(
                '--env "MEEP_GPU_MULTI_INITIAL_CONDITION=${RELEASE_INITIAL_CONDITION}"'
            ),
            2,
        )
        self.assertEqual(
            source.count(
                '--env "MEEP_GPU_MULTI_SOURCE_PROFILE=${RELEASE_SOURCE_PROFILE}"'
            ),
            2,
        )
        bootstrap_environment = source[
            source.index("  RUN_ENVIRONMENT=(") : source.index(
                "  for variable in MEEP_GPU_MAKE_JOBS", source.index("  RUN_ENVIRONMENT=(")
            )
        ]
        worker_environment = source[
            source.index("WORKER_ENVIRONMENT=(") : source.index(
                "for variable in MEEP_GPU_MAKE_JOBS",
                source.index("WORKER_ENVIRONMENT=("),
            )
        ]
        injection = (
            '--env "MEEP_GPU_MULTI_INITIAL_CONDITION=${RELEASE_INITIAL_CONDITION}"'
        )
        source_injection = (
            '--env "MEEP_GPU_MULTI_SOURCE_PROFILE=${RELEASE_SOURCE_PROFILE}"'
        )
        self.assertEqual(bootstrap_environment.count(injection), 1)
        self.assertEqual(worker_environment.count(injection), 1)
        self.assertEqual(bootstrap_environment.count(source_injection), 1)
        self.assertEqual(worker_environment.count(source_injection), 1)
        bootstrap_forwarding = source[
            source.index("  for variable in MEEP_GPU_MAKE_JOBS") : source.index(
                '  cd "${REPO_ROOT}"',
                source.index("  for variable in MEEP_GPU_MAKE_JOBS"),
            )
        ]
        worker_forwarding = source[
            source.index(
                "for variable in MEEP_GPU_MAKE_JOBS",
                source.index("WORKER_ENVIRONMENT=("),
            ) : source.index(
                'if [[ "${PWD}" != "${REPO_ROOT}" ]]',
                source.index("WORKER_ENVIRONMENT=("),
            )
        ]
        self.assertNotIn("MEEP_GPU_MULTI_INITIAL_CONDITION", bootstrap_forwarding)
        self.assertNotIn("MEEP_GPU_MULTI_INITIAL_CONDITION", worker_forwarding)
        self.assertNotIn("MEEP_GPU_MULTI_SOURCE_PROFILE", bootstrap_forwarding)
        self.assertNotIn("MEEP_GPU_MULTI_SOURCE_PROFILE", worker_forwarding)
        export = source.index(
            'export MEEP_GPU_MULTI_INITIAL_CONDITION="${RELEASE_INITIAL_CONDITION}"'
        )
        first_direct = source.index('"${GPU_MPI_PERFORMANCE_ELF}"', export)
        self.assertLess(export, first_direct)
        source_export = source.index(
            'export MEEP_GPU_MULTI_SOURCE_PROFILE="${RELEASE_SOURCE_PROFILE}"'
        )
        self.assertLess(source_export, first_direct)

    def test_clean_build_reproduction_pins_release_initial_condition(self):
        captured: dict[str, object] = {}

        def reproduce(command, **kwargs):
            captured["command"] = command
            captured["environment"] = kwargs["env"]
            pathlib.Path(command[-1]).write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": {
                            "MEEP_GPU_MULTI_INITIAL_CONDITION": (
                                "trigonometric-v1"
                            ),
                            "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
                        },
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            self.runner.subprocess, "run", side_effect=reproduce
        ):
            root = pathlib.Path(temporary)
            environment = self.runner._reproduce_clean_build_environment(
                repo=root,
                environment_prefix=root / "prefix",
                build_dir=root / "build",
                recorded_environment={
                    "GPMEEP_FRESH_ENV_NONCE": "a" * 64,
                },
            )
        self.assertEqual(
            environment["MEEP_GPU_MULTI_INITIAL_CONDITION"],
            "trigonometric-v1",
        )
        self.assertEqual(
            environment["MEEP_GPU_MULTI_SOURCE_PROFILE"], "single-ez-v1"
        )
        self.assertEqual(
            captured["command"].count(
                "MEEP_GPU_MULTI_INITIAL_CONDITION=trigonometric-v1"
            ),
            1,
        )
        self.assertEqual(
            captured["command"].count(
                "MEEP_GPU_MULTI_SOURCE_PROFILE=single-ez-v1"
            ),
            1,
        )
        command = captured["command"]
        process_root = pathlib.Path(command[command.index("--root-prefix") + 1])
        self.assertEqual(process_root.name, "micromamba-root")
        self.assertTrue(process_root.parent.name.startswith("gpmeep-build-env-"))
        self.assertEqual(process_root.parent.parent, pathlib.Path("/tmp"))
        self.assertNotEqual(process_root, root / ".micromamba")
        process_environment = captured["environment"]
        self.assertEqual(
            process_environment["MAMBA_ROOT_PREFIX"], str(process_root)
        )
        self.assertEqual(
            process_environment["XDG_CACHE_HOME"], str(process_root / "cache")
        )
        self.assertIn(
            f"MAMBA_ROOT_PREFIX={root / '.micromamba'}", command
        )

    def test_fixed_workload_rank_mismatch_preflights_are_qualified(self):
        source = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        for gate in (
            "MEEP_GPU_TEST_INITIAL_CONDITION_RANK_MISMATCH",
            "MEEP_GPU_TEST_INITIAL_CONDITION_RANK_INVALID",
            "MEEP_GPU_TEST_INITIAL_CONDITION_PREFLIGHT_RANK_MISMATCH",
            "MEEP_GPU_TEST_SOURCE_ENABLED_RANK_MISMATCH",
            "MEEP_GPU_TEST_SOURCE_PROFILE_RANK_MISMATCH",
            "MEEP_GPU_TEST_SOURCE_PROFILE_RANK_INVALID",
            "MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE_RANK_MISMATCH",
        ):
            with self.subTest(gate=gate):
                self.assertIn(gate, source)
        self.assertGreaterEqual(
            source.count(
                "fixed workload profile differs across MPI ranks before allocation"
            ),
            5,
        )

    def test_boundary_graph_lifecycle_reinitialization_is_qualified(self):
        source = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        lane = source.index(
            'BOUNDARY_GRAPH_LIFECYCLE_LOG="${TEST_LOG_DIR}/'
            'boundary-graph-lifecycle-two-rank.log"'
        )
        marker = source.index(
            "validate-multi-gpu-initialization-markers.py", lane
        )
        qualification = source.index(
            "gpmeep-qualification:boundary-graph-lifecycle-two-rank.log:PASS",
            marker,
        )
        segment = source[lane:qualification]
        self.assertIn("MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE=1", segment)
        self.assertIn("MEEP_GPU_MULTI_PIXELS=16", segment)
        self.assertIn("MEEP_GPU_MULTI_WARMUP_STEPS=2", segment)
        self.assertIn("MEEP_GPU_MULTI_STEPS=2", segment)
        self.assertIn("--expected-profile trigonometric-v1", segment)
        self.assertIn("--expected-applications-per-rank 2", segment)
        self.assertIn("--expected-ranks 2", segment)
        self.assertIn("--expected-pixels 16", segment)
        self.assertIn("--expected-warmup-steps 2", segment)
        self.assertIn("--expected-steps 2", segment)
        self.assertIn("--expected-transport cuda-aware", segment)

    def test_build_environment_capture_records_unlisted_influence_variables(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "environment.json"
            environment = dict(os.environ)
            environment.update(
                {
                    "LIBS": "-lpoison",
                    "NVCC_APPEND_FLAGS": "--use_fast_math",
                    "MAKEFILES": "/tmp/poison.mk",
                    "ACLOCAL_PATH": "/tmp/poison-m4",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "capture-build-environment.py"),
                    "--output",
                    str(output),
                ],
                env=environment,
                check=True,
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["normalized_shell_keys"], ["SHLVL", "_"])
            captured = document["environment"]
            for name, value in environment.items():
                if name in {"SHLVL", "_"}:
                    self.assertEqual(captured.get(name), "<shell-managed>")
                else:
                    self.assertEqual(captured.get(name), value)

    def test_protected_bash_verifier_requires_startup_dash_p(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = pathlib.Path(temporary) / "protected-driver.sh"
            script.write_text(
                "#!/bin/bash -p\n"
                'stage="${1:-bootstrap}"\n'
                "/usr/bin/python3 -I -E "
                + str(SCRIPTS / "verify-protected-bash.py")
                + ' --script "$0" --stage "$stage"\n',
                encoding="utf-8",
            )
            script.chmod(0o755)
            direct = subprocess.run([str(script)], capture_output=True, text=True)
            explicit = subprocess.run(
                ["/bin/bash", "-p", str(script)], capture_output=True, text=True
            )
            worker = subprocess.run(
                ["/bin/bash", "-p", str(script), "audited-worker"],
                capture_output=True,
                text=True,
            )
            late = subprocess.run(
                ["/bin/bash", str(script)], capture_output=True, text=True
            )
            crafted = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    '/usr/bin/python3 -I -E "$1" --script "$3" --stage bootstrap',
                    "benign",
                    str(SCRIPTS / "verify-protected-bash.py"),
                    "-p",
                    str(script),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            self.assertEqual(worker.returncode, 0, worker.stderr)
            self.assertNotEqual(late.returncode, 0)
            self.assertIn("not started with protected argv", late.stderr)
            self.assertNotEqual(crafted.returncode, 0)
            self.assertIn("not started with protected argv", crafted.stderr)

    def test_sample_matrix_requires_interleaving_repeats_and_unique_nonces(self):
        samples = sample_matrix()
        self.assertEqual(
            self.runner.PROFILE["repeats"], FIXED_MEASURED_REPEATS
        )
        self.assertEqual(
            len(samples),
            1
            + len(self.runner.MEASURED_LANES)
            * (
                self.runner.PROFILE["warmups"]
                + self.runner.PROFILE["repeats"]
            ),
        )
        self.assertEqual(
            [
                sample["lane"]
                for sample in samples
                if sample["sample_kind"] == "measured"
                and sample["lane"] != "cpu-legacy-oracle"
            ],
            [
                lane
                for iteration in range(self.runner.PROFILE["repeats"])
                for lane in self.runner.MEASURED_ORDERS[
                    iteration % len(self.runner.MEASURED_ORDERS)
                ]
            ],
        )
        first_positions = {
            lane: sum(order[0] == lane for order in self.runner.MEASURED_ORDERS)
            for lane in self.runner.MEASURED_LANES
        }
        last_positions = {
            lane: sum(order[-1] == lane for order in self.runner.MEASURED_ORDERS)
            for lane in self.runner.MEASURED_LANES
        }
        self.assertEqual(first_positions, {"cpu-2x4": 2, "cuda-single": 1, "cuda-multi": 1})
        self.assertEqual(last_positions, {"cpu-2x4": 2, "cuda-single": 1, "cuda-multi": 1})
        self.assertTrue(self.runner.validate_sample_matrix(samples)["pass"])
        with self.assertRaisesRegex(RuntimeError, "incomplete or not interleaved"):
            self.runner.validate_sample_matrix(samples[:-1])
        duplicated = copy.deepcopy(samples)
        duplicated[-1]["nonce"] = duplicated[-2]["nonce"]
        with self.assertRaisesRegex(RuntimeError, "nonce was reused"):
            self.runner.validate_sample_matrix(duplicated)

    def test_cpu_lane_is_excluded_from_cuda_device_maps(self):
        samples = sample_matrix()
        for sample in samples:
            if sample["lane"] == "cpu-2x4":
                sample["validation"]["device_identifiers"] = ["poison-cpu-device"]
                for rank in sample["validation"]["rank_coverage"]:
                    rank["device_inventory_sha256"] = "0" * 64
        compared = self.runner.compare_measured_samples(
            samples, fd_oracle=fd_oracle()
        )
        for field in (
            "inventory_sha256_by_lane_and_rank",
            "process_environment_sha256_by_lane_and_rank",
            "mapped_environment_sha256_by_lane_and_rank",
            "rank_local_driver_by_lane_and_rank",
        ):
            self.assertEqual(
                set(compared["devices"][field]), set(self.runner.CUDA_LANES)
            )
        self.assertNotIn("poison-cpu-device", compared["devices"]["identifiers"])

    def test_cpu_lane_pins_two_ranks_four_omp_threads_and_core_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix"
            runtime = {
                "python": prefix / "bin" / "python",
                "mpiexec": prefix / "bin" / "mpiexec",
                "libmeep": root / "install" / "lib" / "libmeep.so",
                "package_parent": root / "install" / "site-packages",
                "qualification_fontconfig": root / "fontconfig.conf",
            }
            for path in (runtime["python"], runtime["mpiexec"], runtime["libmeep"]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            runtime["package_parent"].mkdir(parents=True)
            runtime["qualification_fontconfig"].write_text(
                "<fontconfig></fontconfig>\n", encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-should-not-leak"}, clear=False
            ):
                environment = self.runner._child_environment(
                    runtime,
                    lane="cpu-2x4",
                    output=root / "output",
                    pycache_namespace="cpu-lane",
                )
            self.assertEqual(environment["MEEP_GPU_BACKEND"], "cpu")
            self.assertEqual(environment["MEEP_MATERIAL_GRADIENT_PATH"], "analytic-required")
            self.assertEqual(environment["OMP_NUM_THREADS"], "4")
            self.assertEqual(environment["OMP_PLACES"], "cores")
            self.assertEqual(environment["OMP_PROC_BIND"], "close")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
            self.assertNotIn("MEEP_GPU_STRICT", environment)

            command = self.runner._expected_sample_command(
                repo=SCRIPTS.parent,
                runtime=runtime,
                receipt_id="receipt",
                snapshot_sha256="1" * 64,
                producer_sha256="2" * 64,
                run_id="run",
                nonce="nonce",
                lane="cpu-2x4",
                sample_kind="measured",
                iteration=0,
                result_path=root / "result.json",
                ranks=2,
                capture_field_times=False,
            )
            self.assertEqual(
                command[:9],
                [
                    str(runtime["mpiexec"]),
                    "--nooversubscribe",
                    "--report-bindings",
                    "--map-by",
                    "ppr:2:node:PE=4",
                    "--bind-to",
                    "core",
                    "-n",
                    "2",
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "rank count"):
                self.runner._expected_sample_command(
                    repo=SCRIPTS.parent,
                    runtime=runtime,
                    receipt_id="receipt",
                    snapshot_sha256="1" * 64,
                    producer_sha256="2" * 64,
                    run_id="run",
                    nonce="nonce",
                    lane="cpu-2x4",
                    sample_kind="measured",
                    iteration=0,
                    result_path=root / "result.json",
                    ranks=1,
                    capture_field_times=False,
                )

    def test_cpu_benchmark_backend_counters_and_workload_wall_fail_closed(self):
        baseline = cpu_benchmark_raw()
        patches = (
            mock.patch.object(
                self.runner.COMPARATOR, "validate_gradient_summaries"
            ),
            mock.patch.object(
                self.runner.COMPARATOR,
                "validate_runtime_closure",
                return_value={"runtime": "same"},
            ),
            mock.patch.object(
                self.runner.COMPARATOR,
                "runtime_environment_independent_projection",
                side_effect=lambda value: value,
            ),
            mock.patch.object(
                self.runner,
                "_validate_cpu_affinity",
                return_value={"pass": True, "binding": "core"},
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            self.assertEqual(
                self.runner.validate_cpu_benchmark_record(
                    copy.deepcopy(baseline), label="cpu"
                )["workload_wall_seconds"],
                4.1,
            )
            mutations = []
            changed = copy.deepcopy(baseline)
            changed["distributed"]["ranks"][0]["backend"]["active"] = "cuda"
            mutations.append(("backend", changed, "FP32 CPU backend"))
            changed = copy.deepcopy(baseline)
            changed["distributed"]["ranks"][1]["final_statistics"]["dfts"][
                "cuda_dft_calls"
            ] = 1
            mutations.append(("cuda-counter", changed, "contains CUDA work"))
            changed = copy.deepcopy(baseline)
            changed["distributed"]["ranks"][0]["phases"]["forward"][
                "backend_counters"
            ]["cpu"]["dft"] = {"calls": 123456789, "points": 987654321}
            mutations.append(
                ("unbound-phase-counter", changed, "view/statistics disagree")
            )
            changed = copy.deepcopy(baseline)
            changed["timing"]["workload_wall_seconds"] = 4.0
            mutations.append(("max-rank-wall", changed, "exact max-rank"))
            changed = copy.deepcopy(baseline)
            changed["distributed"]["ranks"][1]["timing"][
                "workload_wall_seconds"
            ] = 3.9
            changed["timing"]["workload_wall_seconds"] = 4.0
            mutations.append(("wall-shorter-than-phase", changed, "shorter than phase"))
            changed = copy.deepcopy(baseline)
            changed["timing"]["rank_local_elapsed_seconds"] += 0.1
            mutations.append(("rank-zero-elapsed", changed, "rank-local elapsed"))
            changed = copy.deepcopy(baseline)
            changed["distributed"]["ranks"][0]["timing"][
                "elapsed_seconds"
            ] += 0.1
            mutations.append(("rank-phase-sum", changed, "phase sum"))
            for name, invalid, pattern in mutations:
                with self.subTest(name=name), self.assertRaisesRegex(
                    RuntimeError, pattern
                ):
                    self.runner.validate_cpu_benchmark_record(invalid, label="cpu")

    def test_cpu_affinity_rejects_overlapping_physical_cores(self):
        raw = cpu_affinity_raw()

        def topology(cpu, **_kwargs):
            core = cpu // 2
            return 0, core, (2 * core, 2 * core + 1)

        with mock.patch.object(self.runner, "_linux_cpu_core", side_effect=topology):
            self.assertTrue(self.runner._validate_cpu_affinity(raw, label="cpu")["pass"])
            overlapping = copy.deepcopy(raw)
            first = overlapping["distributed"]["ranks"][0]
            second = overlapping["distributed"]["ranks"][1]
            for field in (
                "cpu_affinity",
                "completion_thread_cpu_affinity",
                "process_thread_cpu_affinities",
            ):
                second[field] = copy.deepcopy(first[field])
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                self.runner._validate_cpu_affinity(overlapping, label="cpu")

    def test_host_audit_rejects_cpu_scheduler_contention(self):
        samples = sample_matrix()
        record = synthetic_host_audit(self.runner, samples)
        cpu_segment = next(
            segment
            for segment in record["contention_segments"]
            if segment["lane"] == self.runner.CPU_BENCHMARK_LANE
        )
        cpu_segment["end"]["some_total_usec"] += int(
            cpu_segment["elapsed_usec"] * 0.5
        )
        cpu_segment["some_pressure_fraction"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "scheduler contention"):
            self.runner.validate_host_audit(
                record,
                samples=samples,
                repo=SCRIPTS.parent,
                verify_live=False,
            )

    def test_cuda_rank_hostname_is_bound_to_driver_and_rank_zero_producer(self):
        result = {
            "producer": {"hostname": "node-a"},
            "distributed": {
                "ranks": [
                    {"hostname": "node-b", "nvidia_driver": {}}
                ]
            },
        }
        validation = {"device_identifiers": ["1" * 32]}
        with mock.patch.object(
            self.runner.COMPARATOR,
            "validate_rank_nvidia_driver",
            return_value={"hostname": "node-a"},
        ), self.assertRaisesRegex(RuntimeError, "hostname binding"):
            self.runner._validate_cuda_rank_driver_evidence(
                result, validation, label="cuda"
            )

    def test_final_closure_binds_current_run_paths_and_recomputed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "evidence"
            run_id = "a" * 32
            self.runner.prepare_output(output, run_id)
            repo = SCRIPTS.parent.resolve()
            runtime = {
                "mpiexec": root / "mpiexec",
                "python": root / "python",
            }
            for path in runtime.values():
                path.write_text("fixture\n", encoding="utf-8")
            receipt_path = root / "receipt.json"
            receipt_path.write_text('{"fixture": true}\n', encoding="utf-8")
            snapshot = {"sha256": "1" * 64}
            receipt = {
                "receipt_id": "receipt",
                "build_input_id": "build-input",
                "artifact_set_id": "artifact-set",
                "source_end": snapshot,
            }
            producer_sha256 = sha256_file(repo / "scripts/benchmark-adjoint.py")
            fixed_environment = {
                "BASE": "fixed",
                "MEEP_GPU_MPI_COMPLETION": "waitsome",
            }
            samples = []
            for matrix_sample in sample_matrix():
                lane = matrix_sample["lane"]
                kind = matrix_sample["sample_kind"]
                iteration = matrix_sample["iteration"]
                nonce = matrix_sample["nonce"]
                label = f"{run_id}-{lane}-{kind}-{iteration}"
                ranks = self.runner._lane_ranks(lane)
                result_path = output / f"{label}.json"
                stdout_path = output / f"{label}.stdout.log"
                stderr_path = output / f"{label}.stderr.log"
                timing_path = output / f"{label}.parent-timing.json"
                command = self.runner._expected_sample_command(
                    repo=repo,
                    runtime=runtime,
                    receipt_id="receipt",
                    snapshot_sha256=snapshot["sha256"],
                    producer_sha256=producer_sha256,
                    run_id=run_id,
                    nonce=nonce,
                    lane=lane,
                    sample_kind=kind,
                    iteration=iteration,
                    result_path=result_path,
                    ranks=ranks,
                    capture_field_times=False,
                )
                if lane == "cpu-legacy-oracle":
                    validation = {"kind": "cpu-oracle"}
                elif lane == "cpu-2x4":
                    validation = {
                        "kind": "cpu-benchmark",
                        "workload_wall_seconds": matrix_sample["validation"][
                            "workload_wall_seconds"
                        ],
                    }
                else:
                    validation = {
                        "kind": "cuda",
                        "rank_driver": [],
                        "workload_wall_seconds": matrix_sample["validation"][
                            "workload_wall_seconds"
                        ],
                    }
                raw = {
                    "schema_version": 4,
                    "timing": {
                        "workload_wall_seconds": matrix_sample.get(
                            "validation", {}
                        ).get("workload_wall_seconds", 1.0)
                    },
                    "producer": {
                        "command": command[command.index(str(runtime["python"])) :]
                    },
                    "result": {"gradient": [1.0]},
                }
                self.runner.atomic_write_json(result_path, raw)
                pointer = {
                    "result_file": str(result_path.resolve()),
                    "sha256": sha256_file(result_path),
                }
                self.runner.atomic_write_text(
                    stdout_path,
                    self.runner.PREFIX + json.dumps(pointer, sort_keys=True) + "\n",
                )
                self.runner.atomic_write_text(stderr_path, "")
                process_seconds = max(
                    2.0, validation.get("workload_wall_seconds", 0.0) + 0.3
                )
                self.runner.atomic_write_json(
                    timing_path,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "label": label,
                        "lane": lane,
                        "sample_kind": kind,
                        "iteration": iteration,
                        "nonce": nonce,
                        "process_seconds": process_seconds,
                        "command_sha256": canonical_sha256(command),
                    },
                )
                samples.append(
                    {
                        "lane": lane,
                        "sample_kind": kind,
                        "iteration": iteration,
                        "completion_policy": "waitsome",
                        "capture_field_times": False,
                        "artifact_label": label,
                        "ranks": ranks,
                        "nonce": nonce,
                        "process_seconds": process_seconds,
                        "result": raw,
                        "validation": validation,
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "lazy_import_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                        "result_file": str(result_path.resolve()),
                        "result_sha256": sha256_file(result_path),
                        "stdout_log": str(stdout_path.resolve()),
                        "stdout_sha256": sha256_file(stdout_path),
                        "stderr_log": str(stderr_path.resolve()),
                        "stderr_sha256": sha256_file(stderr_path),
                        "timing_file": str(timing_path.resolve()),
                        "timing_sha256": sha256_file(timing_path),
                        "command": command,
                        "environment": fixed_environment,
                    }
                )

            test_path = repo / "python/tests/test_adjoint_default_material_grid.py"
            fd_label = f"{run_id}-directional-fd-oracle"
            fd_stdout = output / f"{fd_label}.stdout.log"
            fd_stderr = output / f"{fd_label}.stderr.log"
            fd_timing = output / f"{fd_label}.parent-timing.json"
            directions = fd_oracle()["direction_records"]
            fd_text = "\n".join(
                [*self.runner.FD_TEST_METHODS, "Ran 3 tests in 1.0s", "OK"]
                + [
                    self.runner.FD_DIRECTION_PREFIX + json.dumps(record)
                    for record in directions
                ]
            ) + "\n"
            self.runner.atomic_write_text(fd_stdout, fd_text)
            self.runner.atomic_write_text(fd_stderr, "")
            fd_command = [str(runtime["python"]), str(test_path), "-v"]
            fd_environment = {"BASE": "fixed"}
            started = "2026-08-12T00:00:00Z"
            completed = "2026-08-12T00:00:01Z"
            self.runner.atomic_write_json(
                fd_timing,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "label": fd_label,
                    "process_seconds": 1.0,
                    "started_at_utc": started,
                    "completed_at_utc": completed,
                    "command_sha256": canonical_sha256(fd_command),
                    "environment_sha256": canonical_sha256(fd_environment),
                },
            )
            finite_difference = {
                "pass": True,
                "kind": "installed-multi-direction-and-component-finite-difference",
                "receipt_id": "receipt",
                "source_snapshot_sha256": snapshot["sha256"],
                "test_file": str(test_path),
                "test_sha256": sha256_file(test_path),
                "methods": list(self.runner.FD_TEST_METHODS),
                "directions": len(directions),
                "direction_records": directions,
                "started_at_utc": started,
                "completed_at_utc": completed,
                "process_seconds": 1.0,
                "command": fd_command,
                "environment": fd_environment,
                "stdout_log": str(fd_stdout.resolve()),
                "stdout_sha256": sha256_file(fd_stdout),
                "stderr_log": str(fd_stderr.resolve()),
                "stderr_sha256": sha256_file(fd_stderr),
                "timing_file": str(fd_timing.resolve()),
                "timing_sha256": sha256_file(fd_timing),
            }

            patches = (
                mock.patch.object(self.runner, "verify_build_receipt", return_value=receipt),
                mock.patch.object(
                    self.runner,
                    "validate_build_receipt_contract",
                    return_value={
                        "pass": True,
                        "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                    },
                ),
                mock.patch.object(self.runner, "source_snapshot", return_value=snapshot),
                mock.patch.object(self.runner, "_receipt_runtime", return_value=runtime),
                mock.patch.object(
                    self.runner.COMPARATOR, "validate_qualification_binding", return_value={}
                ),
                mock.patch.object(self.runner, "_fixed_workload_gate"),
                mock.patch.object(
                    self.runner, "_validate_lazy_import_contract", return_value={"pass": True}
                ),
                mock.patch.object(
                    self.runner,
                    "validate_cpu_oracle",
                    return_value={"kind": "cpu-oracle"},
                ),
                mock.patch.object(
                    self.runner,
                    "validate_cpu_benchmark_record",
                    side_effect=lambda raw, **_kwargs: {
                        "kind": "cpu-benchmark",
                        "workload_wall_seconds": raw["timing"][
                            "workload_wall_seconds"
                        ],
                    },
                ),
                mock.patch.object(
                    self.runner.COMPARATOR,
                    "validate_distributed_record",
                    side_effect=lambda raw, **_kwargs: {
                        "kind": "cuda",
                        "workload_wall_seconds": raw["timing"][
                            "workload_wall_seconds"
                        ],
                    },
                ),
                mock.patch.object(
                    self.runner, "_validate_cuda_rank_driver_evidence", return_value=[]
                ),
                mock.patch.object(
                    self.runner, "_validate_receipt_rank_runtimes", return_value={"pass": True}
                ),
                mock.patch.object(
                    self.runner, "_extract_material_gradient_stats", return_value={"pass": True}
                ),
                mock.patch.object(
                    self.runner, "_validate_actual_process_environments", return_value={"pass": True}
                ),
                mock.patch.object(self.runner, "_validate_lane_process_environment"),
                mock.patch.object(
                    self.runner,
                    "_child_environment",
                    side_effect=lambda *_args, **kwargs: (
                        {"BASE": "fixed"}
                    ),
                ),
            )

            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                 patches[5], patches[6], patches[7], patches[8], patches[9], \
                 patches[10], patches[11], patches[12], patches[13], patches[14], \
                 patches[15]:
                capability = publication_capability(
                    self.runner, output, run_id, samples, finite_difference
                )
                kwargs = {
                    "output": output,
                    "expected_run_id": run_id,
                    "repo": repo,
                    "receipt_path": receipt_path,
                    "expected_receipt_id": "receipt",
                    "expected_snapshot": snapshot,
                    "expected_producer_sha256": producer_sha256,
                    "samples": samples,
                    "fd_oracle": finite_difference,
                    "publication_capability": capability,
                }
                self.assertTrue(self.runner.reverify_final_closure(**kwargs)["pass"])

                mutations = []
                changed = copy.deepcopy(samples)
                changed[1]["artifact_label"] = "foreign-label"
                mutations.append(("identity", changed, finite_difference))
                changed = copy.deepcopy(samples)
                changed[1]["validation"] = {"kind": "forged"}
                mutations.append(("derived", changed, finite_difference))
                changed = copy.deepcopy(samples)
                changed[1]["process_seconds"] += 1.0
                mutations.append(("parent-timing", changed, finite_difference))
                changed = copy.deepcopy(samples)
                changed[1]["command"] = ["forged"]
                mutations.append(("command", changed, finite_difference))
                changed_fd = copy.deepcopy(finite_difference)
                changed_fd["direction_records"][0]["finite_difference"] = 0.26
                changed_fd["direction_records"][0]["adjoint_projection"] = 0.26
                mutations.append(("fd-log-replay", samples, changed_fd))
                for name, changed_samples, changed_oracle in mutations:
                    with self.subTest(mutation=name), self.assertRaises(RuntimeError):
                        self.runner.reverify_final_closure(
                            **{
                                **kwargs,
                                "samples": changed_samples,
                                "fd_oracle": changed_oracle,
                            }
                        )

                changed = copy.deepcopy(samples)
                timing_path = pathlib.Path(changed[1]["timing_file"])
                original_timing = json.loads(timing_path.read_text(encoding="utf-8"))
                changed_timing = copy.deepcopy(original_timing)
                changed_timing["process_seconds"] += 1.0
                changed[1]["process_seconds"] += 1.0
                self.runner.atomic_write_json(timing_path, changed_timing)
                changed[1]["timing_sha256"] = sha256_file(timing_path)
                try:
                    with self.assertRaisesRegex(RuntimeError, "controller-owned"):
                        self.runner.reverify_final_closure(
                            **{**kwargs, "samples": changed}
                        )
                finally:
                    self.runner.atomic_write_json(timing_path, original_timing)

                changed = copy.deepcopy(samples)
                changed[0]["unexpected"] = True
                with self.assertRaisesRegex(RuntimeError, "exact schema"):
                    self.runner.reverify_final_closure(
                        **{**kwargs, "samples": changed}
                    )

                unbound = output / "old-run-unbound.json"
                unbound.write_text("{}\n", encoding="utf-8")
                try:
                    with self.assertRaisesRegex(RuntimeError, "unbound artifacts"):
                        self.runner.reverify_final_closure(**kwargs)
                finally:
                    unbound.unlink()

                external = root / "external.json"
                shutil.copyfile(samples[0]["result_file"], external)
                changed = copy.deepcopy(samples)
                changed[0]["result_file"] = str(external)
                changed[0]["result_sha256"] = sha256_file(external)
                with self.assertRaisesRegex(RuntimeError, "current-run artifact path"):
                    self.runner.reverify_final_closure(
                        **{**kwargs, "samples": changed}
                    )

                owned = pathlib.Path(samples[0]["result_file"])
                backup = root / "owned-backup.json"
                owned.rename(backup)
                owned.symlink_to(backup)
                try:
                    with self.assertRaisesRegex(RuntimeError, "owned regular"):
                        self.runner.reverify_final_closure(**kwargs)
                finally:
                    owned.unlink()
                    backup.rename(owned)

    def test_failed_marker_hash_binds_complete_unqualified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "evidence"
            run_id = "failure-run"
            self.runner.prepare_output(output, run_id)
            samples = []
            matrix_samples = sample_matrix()
            for sample in matrix_samples:
                if sample["lane"] == "cuda-multi" and sample["sample_kind"] == "measured":
                    sample["validation"]["workload_wall_seconds"] = 1.9
                    sample["process_seconds"] = 2.2
            for sample_index, sample in enumerate(matrix_samples):
                result_file = output / f"sample-{sample_index}.json"
                stdout_log = output / f"sample-{sample_index}.stdout.log"
                stderr_log = output / f"sample-{sample_index}.stderr.log"
                timing_file = output / f"sample-{sample_index}.parent-timing.json"
                result_file.write_text(
                    json.dumps(sample["result"], sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                stdout_log.write_text("stdout\n", encoding="utf-8")
                stderr_log.write_text("", encoding="utf-8")
                timing_file.write_text("{}\n", encoding="utf-8")
                full_sample = copy.deepcopy(sample)
                full_sample.update(
                    {
                        "process_seconds": sample.get("process_seconds", 1.0),
                        "completion_policy": "waitsome",
                        "capture_field_times": False,
                        "artifact_label": f"failure-run-sample-{sample_index}",
                        "ranks": self.runner._lane_ranks(sample["lane"]),
                        "receipt_gate": {"receipt_id": "receipt"},
                        "process_environment_gate": {"pass": True},
                        "material_gradient_gate": sample.get(
                            "material_gradient_gate", {"pass": True}
                        ),
                        "result_file": str(result_file.resolve()),
                        "result_sha256": sha256_file(result_file),
                        "stdout_log": str(stdout_log.resolve()),
                        "stdout_sha256": sha256_file(stdout_log),
                        "stderr_log": str(stderr_log.resolve()),
                        "stderr_sha256": sha256_file(stderr_log),
                        "timing_file": str(timing_file.resolve()),
                        "timing_sha256": sha256_file(timing_file),
                        "command": ["/synthetic/python", "/synthetic/producer.py"],
                        "environment": {
                            "MEEP_GPU_BACKEND": (
                                "cuda"
                                if sample["lane"] in self.runner.CUDA_LANES
                                else "cpu"
                            )
                        },
                    }
                )
                samples.extend(self.runner.summarize_samples([full_sample]))
            finite_difference = fd_oracle()
            fd_label = f"{run_id}-directional-fd-oracle"
            fd_stdout = output / f"{fd_label}.stdout.log"
            fd_stderr = output / f"{fd_label}.stderr.log"
            fd_timing = output / f"{fd_label}.parent-timing.json"
            fd_stdout.write_text("stdout\n", encoding="utf-8")
            fd_stderr.write_text("", encoding="utf-8")
            fd_timing.write_text("{}\n", encoding="utf-8")
            finite_difference.update(
                {
                    "receipt_id": "receipt",
                    "source_snapshot_sha256": "6" * 64,
                    "test_file": str(
                        (
                            SCRIPTS.parent
                            / "python"
                            / "tests"
                            / "test_adjoint_default_material_grid.py"
                        ).resolve()
                    ),
                    "test_sha256": sha256_file(
                        SCRIPTS.parent
                        / "python"
                        / "tests"
                        / "test_adjoint_default_material_grid.py"
                    ),
                    "methods": list(self.runner.FD_TEST_METHODS),
                    "stdout_log": str(fd_stdout.resolve()),
                    "stdout_sha256": sha256_file(fd_stdout),
                    "stderr_log": str(fd_stderr.resolve()),
                    "stderr_sha256": sha256_file(fd_stderr),
                    "timing_file": str(fd_timing.resolve()),
                    "timing_sha256": sha256_file(fd_timing),
                }
            )
            receipt = {
                "path": "/synthetic/build-provenance.json",
                "sha256": "2" * 64,
                "receipt_id": "receipt",
                "build_input_id": "build-input",
                "artifact_set_id": "artifacts",
                "qualification_contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
            }
            comparison = self.runner.compare_measured_samples(
                matrix_samples, fd_oracle=finite_difference
            )
            host_audit = bind_synthetic_host(
                self.runner, comparison, samples
            )
            semantic_samples = []
            for summary, raw_sample in zip(samples, matrix_samples):
                reconstructed = {
                    key: value
                    for key, value in summary.items()
                    if key not in {"raw_result_schema", "raw_gradient_sha256"}
                }
                reconstructed["result"] = raw_sample["result"]
                semantic_samples.append(reconstructed)
            semantic_patcher = mock.patch.object(
                self.runner,
                "_reconstruct_unqualified_samples",
                return_value=semantic_samples,
            )
            semantic_patcher.start()
            self.addCleanup(semantic_patcher.stop)
            fd_semantic_patcher = mock.patch.object(
                self.runner,
                "_validate_unqualified_fd_semantics",
                return_value=None,
            )
            fd_semantic_patcher.start()
            self.addCleanup(fd_semantic_patcher.stop)
            gate = comparison["gate"]
            self.assertIs(gate["performance"], False)
            report = {
                "schema_version": 1,
                "state": "UNQUALIFIED",
                "evidence_class": "failed-release-qualification",
                "run_id": run_id,
                "profile": dict(self.runner.PROFILE),
                "build_receipt": receipt,
                "finite_difference_oracle": finite_difference,
                "host_audit": host_audit,
                "integrity_reverification": {
                    "pass": True,
                    "verified_at_utc": "2026-08-10T00:00:00Z",
                    "build_receipt": receipt,
                    "receipt_id": receipt["receipt_id"],
                    "receipt_sha256": receipt["sha256"],
                    "source_snapshot": {"sha256": "6" * 64},
                    "contract": {
                        "pass": True,
                        "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                    },
                    "samples": [
                        {
                            "label": (
                                f"{sample['lane']}-{sample['sample_kind']}-"
                                f"{sample['iteration']}"
                            ),
                            "nonce": sample["nonce"],
                            "process_seconds": sample["process_seconds"],
                            "summary_sha256": canonical_sha256(sample),
                            "files": {
                                path_name: {
                                    "path": sample[path_name],
                                    "sha256": sample[digest_name],
                                }
                                for path_name, digest_name in (
                                    ("result_file", "result_sha256"),
                                    ("stdout_log", "stdout_sha256"),
                                    ("stderr_log", "stderr_sha256"),
                                    ("timing_file", "timing_sha256"),
                                )
                            },
                        }
                        for sample in samples
                    ],
                    "finite_difference_logs": {
                        "stdout_log": {
                            "path": finite_difference["stdout_log"],
                            "sha256": finite_difference["stdout_sha256"],
                        },
                        "stderr_log": {
                            "path": finite_difference["stderr_log"],
                            "sha256": finite_difference["stderr_sha256"],
                        },
                        "timing_file": {
                            "path": finite_difference["timing_file"],
                            "sha256": finite_difference["timing_sha256"],
                        },
                    },
                },
                "sample_matrix": self.runner.validate_sample_matrix(samples),
                "samples": samples,
                "comparison": comparison,
                "gate": gate,
            }
            inventory_files = [
                record
                for verified in report["integrity_reverification"]["samples"]
                for record in verified["files"].values()
            ] + list(
                report["integrity_reverification"][
                    "finite_difference_logs"
                ].values()
            )
            inventory_files.append(
                {
                    "path": str((output / "state.json").resolve()),
                    "sha256": sha256_file(output / "state.json"),
                }
            )
            inventory_manifest = {
                "files": sorted(inventory_files, key=lambda record: record["path"]),
                "runtime_directories": [
                    "cache",
                    "home",
                    "matplotlib",
                    "pycache",
                    "tmp",
                ],
            }
            report["integrity_reverification"]["controller_ledger"] = {
                "pass": True,
                "entries": len(samples) + 1,
                "sha256": "b" * 64,
            }
            report["integrity_reverification"]["output_inventory"] = {
                "pass": True,
                "epoch": "prepublication-running",
                "manifest": inventory_manifest,
                "sha256": canonical_sha256(inventory_manifest),
            }
            record = self.runner.write_unqualified_report(output, run_id, report)
            failure_error = self.runner.QualificationGateFailure(
                "performance", record
            )
            self.runner.mark_failed(output, run_id, failure_error)
            failure = json.loads(
                (output / "FAILED.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["unqualified_report"], record)
            self.assertEqual(
                record["sha256"],
                sha256_file(output / self.runner.UNQUALIFIED_REPORT_NAME),
            )

            passing = copy.deepcopy(report)
            passing["gate"]["performance"] = True
            passing["gate"]["pass"] = True
            with self.assertRaisesRegex(RuntimeError, "incomplete unqualified"):
                self.runner.write_unqualified_report(output, run_id, passing)

            for mutation in ("truncate", "identity-preserving"):
                with self.subTest(mutation=mutation):
                    self.runner.prepare_output(output, run_id)
                    record = self.runner.write_unqualified_report(
                        output, run_id, report
                    )
                    unqualified_path = output / self.runner.UNQUALIFIED_REPORT_NAME
                    if mutation == "truncate":
                        unqualified_path.write_bytes(b'{"schema_version": 1')
                    else:
                        modified = copy.deepcopy(report)
                        modified["samples"][0]["process_seconds"] = 2.0
                        self.runner.atomic_write_json(unqualified_path, modified)
                    failure_error = self.runner.QualificationGateFailure(
                        "performance", record
                    )
                    self.runner.mark_failed(output, run_id, failure_error)
                    failure = json.loads(
                        (output / "FAILED.json").read_text(encoding="utf-8")
                    )
                    self.assertNotIn("unqualified_report", failure)
                    self.assertIn("changed after publication", failure[
                        "unqualified_report_rejected"
                    ])

            incomplete = copy.deepcopy(report)
            del incomplete["integrity_reverification"]["samples"][0]["files"][
                "stdout_log"
            ]
            with self.assertRaisesRegex(RuntimeError, "incomplete unqualified"):
                self.runner.write_unqualified_report(output, run_id, incomplete)

            incomplete = copy.deepcopy(report)
            del incomplete["comparison"]["performance"]["fresh_process_wall"]
            with self.assertRaisesRegex(RuntimeError, "incomplete unqualified"):
                self.runner.write_unqualified_report(output, run_id, incomplete)

            incomplete = copy.deepcopy(report)
            incomplete["integrity_reverification"]["contract"]["contract"] = (
                "wrong-contract"
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete unqualified"):
                self.runner.write_unqualified_report(output, run_id, incomplete)

            self.runner.prepare_output(output, run_id)
            record = self.runner.write_unqualified_report(output, run_id, report)
            (output / self.runner.UNQUALIFIED_REPORT_NAME).unlink()
            failure_error = self.runner.QualificationGateFailure(
                "performance", record
            )
            self.runner.mark_failed(output, run_id, failure_error)
            failure = json.loads(
                (output / "FAILED.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("unqualified_report", failure)
            self.assertIn(
                "file is missing", failure["unqualified_report_rejected"]
            )

    def test_build_receipt_report_record_keeps_contract_identity_scalar(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = pathlib.Path(temporary) / "build-provenance.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            receipt = {
                "receipt_id": "receipt-id",
                "build_input_id": "build-input-id",
                "artifact_set_id": "artifact-set-id",
            }
            validation = {
                "pass": True,
                "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                "qualification_logs": ["detailed-validation-stays-separate"],
            }

            record = self.runner.build_receipt_report_record(
                receipt_path, receipt, validation
            )

            self.assertEqual(
                record,
                {
                    "path": str(receipt_path.resolve()),
                    "sha256": sha256_file(receipt_path),
                    "receipt_id": "receipt-id",
                    "build_input_id": "build-input-id",
                    "artifact_set_id": "artifact-set-id",
                    "qualification_contract": (
                        self.runner.BUILD_QUALIFICATION_CONTRACT
                    ),
                },
            )
            self.assertIsInstance(record["qualification_contract"], str)
            self.assertNotIn("qualification_logs", record)
            self.assertTrue(
                self.runner.valid_build_receipt_report_record(record)
            )

            object_contract = copy.deepcopy(record)
            object_contract["qualification_contract"] = validation
            self.assertFalse(
                self.runner.valid_build_receipt_report_record(object_contract)
            )

            invalid = copy.deepcopy(validation)
            invalid["contract"] = "wrong-contract"
            with self.assertRaisesRegex(
                RuntimeError, "qualification-contract validation is incomplete"
            ):
                self.runner.build_receipt_report_record(
                    receipt_path, receipt, invalid
                )

            incomplete_receipt = copy.deepcopy(receipt)
            del incomplete_receipt["artifact_set_id"]
            with self.assertRaisesRegex(
                RuntimeError, "build receipt identity is incomplete"
            ):
                self.runner.build_receipt_report_record(
                    receipt_path, incomplete_receipt, validation
                )

    def test_execute_performance_failure_uses_scalar_receipt_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            receipt_path = root / "build-provenance.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            snapshot = {"sha256": "a" * 64}
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "source_end": snapshot,
                "receipt_id": "receipt-id",
                "build_input_id": "build-input-id",
                "artifact_set_id": "artifact-set-id",
            }
            contract = {
                "pass": True,
                "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                "qualification_logs": ["detailed-contract-evidence"],
            }
            samples = sample_matrix()
            for sample in samples:
                sample["receipt_gate"] = {"receipt_id": "receipt-id"}
                sample["environment"] = {"FIXED": "environment"}
                if sample["sample_kind"] != "measured":
                    continue
                if sample["lane"] == "cuda-single":
                    sample["validation"]["workload_wall_seconds"] = 2.0
                    sample["process_seconds"] = 2.0
                elif sample["lane"] == "cuda-multi":
                    sample["validation"]["workload_wall_seconds"] = 1.9
                    sample["process_seconds"] = 1.9
            finite_difference = fd_oracle()
            integrity = {"pass": True, "contract": contract}
            captured = {}

            def capture_unqualified(output, run_id, report):
                captured["output"] = output
                captured["run_id"] = run_id
                captured["report"] = copy.deepcopy(report)
                return {"path": str(output / "UNQUALIFIED_REPORT.json"), "sha256": "b" * 64}

            args = argparse.Namespace(
                build_receipt=receipt_path,
                output=root / "evidence",
                profile=self.runner.PROFILE_ID,
            )
            args.output.mkdir()
            with mock.patch.object(
                self.runner, "source_snapshot", return_value=snapshot
            ), mock.patch.object(
                self.runner, "verify_build_receipt", return_value=receipt
            ), mock.patch.object(
                self.runner,
                "validate_build_receipt_contract",
                return_value=contract,
            ), mock.patch.object(
                self.runner, "_receipt_runtime", return_value={}
            ), mock.patch.object(
                self.runner, "capture_host_audit_start", return_value={}
            ), mock.patch.object(
                self.runner,
                "complete_host_audit",
                side_effect=lambda _start, host_samples, repo: synthetic_host_audit(
                    self.runner, host_samples
                ),
            ), mock.patch.object(
                self.runner, "validate_host_audit", return_value={"pass": True}
            ), mock.patch.object(
                self.runner,
                "_run_fd_oracle_test",
                return_value=finite_difference,
            ), mock.patch.object(
                self.runner, "_run_raw_sample", side_effect=samples
            ), mock.patch.object(
                self.runner,
                "reverify_final_closure",
                return_value=integrity,
            ), mock.patch.object(
                self.runner, "summarize_samples", return_value=[]
            ), mock.patch.object(
                self.runner,
                "_nvidia_smi_probe",
                return_value={"available": True},
            ), mock.patch.object(
                self.runner,
                "validate_driver_device_binding",
                return_value={"pass": True},
            ), mock.patch.object(
                self.runner,
                "write_unqualified_report",
                side_effect=capture_unqualified,
            ):
                with self.assertRaises(self.runner.QualificationGateFailure) as error:
                    self.runner.execute(
                        args, SCRIPTS.parent, "performance-failure"
                    )

            self.assertIn("performance", str(error.exception))
            report = captured["report"]
            self.assertEqual(
                report["build_receipt"]["qualification_contract"],
                self.runner.BUILD_QUALIFICATION_CONTRACT,
            )
            self.assertIsInstance(
                report["build_receipt"]["qualification_contract"], str
            )
            self.assertEqual(report["integrity_reverification"], integrity)
            self.assertIs(report["gate"]["performance"], False)

    def test_execute_success_capability_binds_scalar_receipt_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            receipt_path = root / "build-provenance.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            snapshot = {"sha256": "a" * 64}
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "source_end": snapshot,
                "receipt_id": "receipt-id",
                "build_input_id": "build-input-id",
                "artifact_set_id": "artifact-set-id",
            }
            contract = {
                "pass": True,
                "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
            }
            samples = sample_matrix()
            for sample in samples:
                sample["receipt_gate"] = {"receipt_id": "receipt-id"}
                sample["environment"] = {"FIXED": "environment"}
            finite_difference = fd_oracle()
            integrity = {
                "pass": True,
                "receipt_id": "receipt-id",
                "receipt_sha256": sha256_file(receipt_path),
                "source_snapshot": snapshot,
                "contract": contract,
            }
            captured = {}

            def capture_publish(output, run_id, report, markdown, **kwargs):
                captured["report"] = copy.deepcopy(report)
                captured["kwargs"] = copy.deepcopy(kwargs)

            args = argparse.Namespace(
                build_receipt=receipt_path,
                output=root / "evidence",
                profile=self.runner.PROFILE_ID,
            )
            args.output.mkdir()
            with mock.patch.object(
                self.runner, "source_snapshot", return_value=snapshot
            ), mock.patch.object(
                self.runner, "verify_build_receipt", return_value=receipt
            ), mock.patch.object(
                self.runner,
                "validate_build_receipt_contract",
                return_value=contract,
            ), mock.patch.object(
                self.runner, "_receipt_runtime", return_value={}
            ), mock.patch.object(
                self.runner, "capture_host_audit_start", return_value={}
            ), mock.patch.object(
                self.runner,
                "complete_host_audit",
                side_effect=lambda _start, host_samples, repo: synthetic_host_audit(
                    self.runner, host_samples
                ),
            ), mock.patch.object(
                self.runner, "validate_host_audit", return_value={"pass": True}
            ), mock.patch.object(
                self.runner,
                "_run_fd_oracle_test",
                return_value=finite_difference,
            ), mock.patch.object(
                self.runner, "_run_raw_sample", side_effect=samples
            ), mock.patch.object(
                self.runner,
                "reverify_final_closure",
                return_value=integrity,
            ), mock.patch.object(
                self.runner, "summarize_samples", return_value=[]
            ), mock.patch.object(
                self.runner,
                "_nvidia_smi_probe",
                return_value={"available": True},
            ), mock.patch.object(
                self.runner,
                "validate_driver_device_binding",
                return_value={"pass": True},
            ), mock.patch.object(
                self.runner, "git_output", return_value=""
            ), mock.patch.object(
                self.runner, "publish_complete", side_effect=capture_publish
            ):
                self.assertEqual(
                    self.runner.execute(args, SCRIPTS.parent, "success"), 0
                )

            report_record = captured["report"]["build_receipt"]
            self.assertEqual(
                report_record["qualification_contract"],
                self.runner.BUILD_QUALIFICATION_CONTRACT,
            )
            self.assertEqual(
                captured["kwargs"]["expected_build_receipt_record"],
                report_record,
            )

    def test_fixed_workload_hashes_and_phase_counts_cannot_be_relaxed(self):
        workload = {
            "resolution": 64,
            "run_time": 40.0,
            "cell_size": 32.0,
            "design_resolution": 20,
            "design_shape": [41, 41],
            "design_variables": 1681,
            "yee_cells": 4194304,
            "weights_sha256": self.runner.PROFILE["weights_sha256"],
            "direction_sha256": self.runner.PROFILE["direction_sha256"],
            "material_gradient_path": "analytic-required",
            "phases": {
                "forward": {
                    "timesteps": self.runner.PROFILE["forward_timesteps"],
                    "end_time": self.runner.PROFILE["forward_end_time"],
                },
                "adjoint": {
                    "timesteps": self.runner.PROFILE["adjoint_timesteps"],
                    "end_time": self.runner.PROFILE["adjoint_end_time"],
                },
            },
        }
        result = {"workload": workload}
        self.runner._fixed_workload_gate(result, lane="cuda-single")
        stale_hash = copy.deepcopy(result)
        stale_hash["workload"]["weights_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "weights_sha256"):
            self.runner._fixed_workload_gate(stale_hash, lane="cuda-single")
        stale_steps = copy.deepcopy(result)
        stale_steps["workload"]["phases"]["adjoint"]["timesteps"] -= 1
        with self.assertRaisesRegex(RuntimeError, "adjoint timesteps"):
            self.runner._fixed_workload_gate(stale_steps, lane="cuda-single")

    def test_fixed_comparison_recomputes_median_and_conservative_speedups(self):
        compared = self.runner.compare_measured_samples(
            sample_matrix(), fd_oracle=fd_oracle()
        )
        self.assertTrue(compared["gate"]["pass"])
        self.assertAlmostEqual(
            compared["performance"]["workload_wall"]["comparisons"][
                "cuda_single_over_cuda_multi"
            ]["median_speedup"],
            4 / 3,
        )
        self.assertAlmostEqual(
            compared["performance"]["workload_wall"]["comparisons"][
                "cuda_single_over_cuda_multi"
            ]["conservative_speedup"],
            4 / 3,
        )

        slow = sample_matrix()
        for sample in slow:
            if sample["lane"] == "cuda-multi" and sample["sample_kind"] == "measured":
                sample["validation"]["workload_wall_seconds"] = 2.1
        compared = self.runner.compare_measured_samples(slow, fd_oracle=fd_oracle())
        self.assertFalse(compared["gate"]["performance"])
        self.assertFalse(compared["gate"]["pass"])

    def test_device_mapping_drift_fails_closed(self):
        samples = sample_matrix()
        samples[-1]["validation"]["device_identifiers"] = [f"{2:032x}"]
        with self.assertRaisesRegex(RuntimeError, "physical GPU mapping"):
            self.runner.compare_measured_samples(samples, fd_oracle=fd_oracle())

    def test_rank_local_inventory_and_visibility_are_stable_not_globally_identical(self):
        samples = sample_matrix()
        for sample in samples:
            if sample["lane"] == "cuda-multi":
                sample["validation"]["rank_coverage"][1][
                    "device_inventory_sha256"
                ] = "c" * 64
                sample["validation"]["rank_coverage"][1][
                    "process_environment_sha256"
                ] = "f" * 64
        self.assertTrue(
            self.runner.compare_measured_samples(samples, fd_oracle=fd_oracle())["gate"][
                "pass"
            ]
        )
        drifted = copy.deepcopy(samples)
        changed = next(
            sample
            for sample in drifted
            if sample["lane"] == "cuda-multi" and sample["sample_kind"] == "measured"
        )
        changed["validation"]["rank_coverage"][1][
            "device_inventory_sha256"
        ] = "a" * 64
        with self.assertRaisesRegex(RuntimeError, "mapping|inventory"):
            self.runner.compare_measured_samples(drifted, fd_oracle=fd_oracle())

    def test_mapped_environment_closure_is_stable_per_lane_and_rank(self):
        samples = sample_matrix()
        changed = next(
            sample
            for sample in samples
            if sample["lane"] == "cuda-multi"
            and sample["sample_kind"] == "measured"
            and sample["iteration"] == 1
        )
        changed["validation"]["rank_runtimes"][1]["loaded_libraries"][
            "environment"
        ].append(
            {
                "path": "/synthetic/cuda-multi/rank-1/mca_btl_smcuda.so",
                "sha256": "f" * 64,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "dependency closure changed"):
            self.runner.compare_measured_samples(samples, fd_oracle=fd_oracle())

    def test_common_dft_point_inflation_and_transport_drift_fail_closed(self):
        inflated = sample_matrix()
        for sample in inflated:
            if sample["lane"].startswith("cuda"):
                for rank in sample["validation"]["rank_coverage"]:
                    for phase in rank["phases"].values():
                        phase["dft"]["points"] *= 10**9
        compared = self.runner.compare_measured_samples(
            inflated, fd_oracle=fd_oracle()
        )
        self.assertFalse(compared["gate"]["absolute_dft_decimation"])
        self.assertFalse(compared["gate"]["pass"])

        drifted = sample_matrix()
        changed = next(
            sample
            for sample in drifted
            if sample["lane"] == "cuda-multi" and sample["sample_kind"] == "measured"
        )
        for rank in changed["validation"]["rank_coverage"]:
            rank["transport"] = "cuda-aware"
        with self.assertRaisesRegex(RuntimeError, "transport changed"):
            self.runner.compare_measured_samples(drifted, fd_oracle=fd_oracle())

    def test_fd_and_fresh_process_wall_are_independent_release_gates(self):
        with self.assertRaisesRegex(RuntimeError, "finite-difference oracle"):
            self.runner.compare_measured_samples(sample_matrix(), fd_oracle={"pass": False})
        incomplete_fd = fd_oracle()
        incomplete_fd["direction_records"].pop()
        with self.assertRaisesRegex(RuntimeError, "direction inventory"):
            self.runner.compare_measured_samples(
                sample_matrix(), fd_oracle=incomplete_fd
            )

        slow_wall = sample_matrix()
        for sample in slow_wall:
            if sample["lane"] == "cuda-multi" and sample["sample_kind"] == "measured":
                sample["process_seconds"] = 3.0
        compared = self.runner.compare_measured_samples(
            slow_wall, fd_oracle=fd_oracle()
        )
        self.assertGreater(
            compared["performance"]["workload_wall"]["comparisons"][
                "cuda_single_over_cuda_multi"
            ]["median_speedup"],
            self.runner.PROFILE["minimum_median_speedup"],
        )
        self.assertFalse(
            compared["performance"]["fresh_process_wall"]["comparisons"][
                "cuda_single_over_cuda_multi"
            ]["median_speedup"]
            >= self.runner.PROFILE["minimum_median_speedup"]
        )
        self.assertFalse(compared["gate"]["performance"])
        self.assertFalse(compared["gate"]["pass"])

    def test_child_environment_scrubs_unrecorded_runtime_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix"
            python = prefix / "bin" / "python"
            libmeep = root / "install" / "lib" / "libmeep.so"
            package = root / "install" / "site-packages" / "meep"
            python.parent.mkdir(parents=True)
            libmeep.parent.mkdir(parents=True)
            package.mkdir(parents=True)
            qualification_fontconfig = root / "qualification-fontconfig.conf"
            python.write_text("python\n", encoding="utf-8")
            libmeep.write_text("lib\n", encoding="utf-8")
            qualification_fontconfig.write_text(
                "<fontconfig></fontconfig>\n", encoding="utf-8"
            )
            runtime = {
                "python": python,
                "libmeep": libmeep,
                "package_parent": package.parent,
                "qualification_fontconfig": qualification_fontconfig,
            }
            with mock.patch.dict(
                os.environ,
                {
                    "HOME": str(root),
                    "LD_PRELOAD": "/tmp/unrecorded.so",
                    "OMPI_MCA_btl": "self",
                    "UCX_TLS": "tcp",
                },
                clear=False,
            ):
                environment = self.runner._child_environment(
                    runtime,
                    lane="cuda-single",
                    output=root / "output",
                    pycache_namespace="test-run",
                )
            for name in ("LD_PRELOAD", "OMPI_MCA_btl", "UCX_TLS"):
                self.assertNotIn(name, environment)
            self.assertEqual(environment["CUDA_CACHE_DISABLE"], "1")
            self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(
                environment["XDG_CACHE_HOME"],
                str((root / "output" / "cache" / "test-run").resolve()),
            )
            self.assertEqual(
                environment["FONTCONFIG_FILE"], str(qualification_fontconfig)
            )
            fontconfig_cache = (
                pathlib.Path(environment["XDG_CACHE_HOME"]) / "fontconfig"
            )
            fontconfig_cache.mkdir()
            (fontconfig_cache / "cache-entry").write_text(
                "cache\n", encoding="utf-8"
            )
            repeated_environment = self.runner._child_environment(
                runtime,
                lane="cuda-single",
                output=root / "output",
                pycache_namespace="test-run",
            )
            self.assertEqual(repeated_environment, environment)
            before_read_only = sorted(
                (
                    str(path.relative_to(root / "output")),
                    path.lstat().st_mode,
                    path.lstat().st_size,
                    path.lstat().st_mtime_ns,
                    sha256_file(path) if path.is_file() else None,
                )
                for path in (root / "output").rglob("*")
            )
            replay_environment = self.runner._child_environment(
                runtime,
                lane="cuda-single",
                output=root / "output",
                pycache_namespace="test-run",
                prepare_filesystem=False,
            )
            after_read_only = sorted(
                (
                    str(path.relative_to(root / "output")),
                    path.lstat().st_mode,
                    path.lstat().st_size,
                    path.lstat().st_mtime_ns,
                    sha256_file(path) if path.is_file() else None,
                )
                for path in (root / "output").rglob("*")
            )
            self.assertEqual(replay_environment, environment)
            self.assertEqual(after_read_only, before_read_only)
            with self.assertRaisesRegex(RuntimeError, "absent during read-only replay"):
                self.runner._child_environment(
                    runtime,
                    lane="cuda-single",
                    output=root / "output",
                    pycache_namespace="missing-run",
                    prepare_filesystem=False,
                )
            self.assertFalse((root / "output" / "cache" / "missing-run").exists())
            self.assertFalse((root / "output" / "home" / "missing-run").exists())
            self.assertFalse((root / "output" / "pycache" / "missing-run").exists())
            self.assertEqual(
                sorted(
                    path.name
                    for path in pathlib.Path(environment["HOME"]).iterdir()
                ),
                [".gpmeep-empty-home"],
            )
            actual = dict(environment)
            actual.pop("PRTE_MCA_mca_base_param_files")
            actual.update(
                {
                    "OMPI_COMM_WORLD_RANK": "0",
                    "OMPI_COMM_WORLD_SIZE": "1",
                    "PMIX_RANK": "0",
                    "OMPI_MCA_cpu_type": "x86_64",
                }
            )
            result = {
                "producer": {"environment": dict(actual)},
                "distributed": {
                    "ranks": [{"process_environment": dict(actual)}]
                },
            }
            self.assertTrue(
                self.runner._validate_actual_process_environments(
                    result, environment, label="clean"
                )["pass"]
            )
            changed_prte = copy.deepcopy(result)
            changed_prte["producer"]["environment"][
                "PRTE_MCA_mca_base_param_files"
            ] = "/tmp/poisoned-prte.conf"
            changed_prte["distributed"]["ranks"][0]["process_environment"][
                "PRTE_MCA_mca_base_param_files"
            ] = "/tmp/poisoned-prte.conf"
            with self.assertRaisesRegex(RuntimeError, "launcher-consumed"):
                self.runner._validate_actual_process_environments(
                    changed_prte, environment, label="changed-prte"
                )
            poisoned = copy.deepcopy(result)
            poisoned["producer"]["environment"][
                "MEEP_GPU_DISABLE_DFT_NORM_REDUCTION"
            ] = "1"
            poisoned["distributed"]["ranks"][0]["process_environment"][
                "MEEP_GPU_DISABLE_DFT_NORM_REDUCTION"
            ] = "1"
            with self.assertRaisesRegex(RuntimeError, "unlaunched runtime controls"):
                self.runner._validate_actual_process_environments(
                    poisoned, environment, label="poisoned"
                )
            poisoned_fontconfig = copy.deepcopy(result)
            poisoned_fontconfig["producer"]["environment"][
                "FONTCONFIG_PATH"
            ] = "/tmp/poisoned-fontconfig"
            poisoned_fontconfig["distributed"]["ranks"][0][
                "process_environment"
            ]["FONTCONFIG_PATH"] = "/tmp/poisoned-fontconfig"
            with self.assertRaisesRegex(RuntimeError, "unlaunched runtime controls"):
                self.runner._validate_actual_process_environments(
                    poisoned_fontconfig, environment, label="fontconfig-poisoned"
                )
            qualification_home = pathlib.Path(environment["HOME"])
            (qualification_home / ".prte").mkdir()
            with self.assertRaisesRegex(RuntimeError, "HOME was modified"):
                self.runner._validate_actual_process_environments(
                    result, environment, label="home-poisoned"
                )

    def test_driver_uuid_binding_and_publish_capability_fail_closed(self):
        identifiers = [f"{1:032x}", f"{2:032x}"]
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            executable = output / "nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            probe = {
                "available": True,
                "exit_code": 0,
                "stdout": (
                    "0, GPU-00000000-0000-0000-0000-000000000001, A, 1, 8.6, 1 MiB\n"
                    "1, GPU-00000000-0000-0000-0000-000000000002, B, 1, 8.6, 1 MiB"
                ),
                "executable": file_record(executable, output),
            }
            with mock.patch.object(
                self.runner,
                "TRUSTED_NVIDIA_SMI_DIRECTORIES",
                (output,),
            ):
                self.assertTrue(
                    self.runner.validate_driver_device_binding(
                        probe, identifiers, repo=output
                    )["pass"]
                )
                with self.assertRaisesRegex(RuntimeError, "do not match"):
                    self.runner.validate_driver_device_binding(
                        probe, [f"{3:032x}"], repo=output
                    )
            for name in ("COMPLETE", "report.json", "report.md"):
                (output / name).write_text("stale\n", encoding="utf-8")
            self.runner.atomic_write_json(
                output / "state.json",
                {"schema_version": 1, "state": "COMPLETE", "run_id": "old"},
            )
            with self.assertRaisesRegex(RuntimeError, "capability"):
                self.runner.publish_complete(
                    output,
                    "run",
                    {},
                    "bad\n",
                    capability=None,
                    repo=output,
                    receipt_path=output / "missing.json",
                    expected_receipt_id="receipt",
                    expected_build_receipt_record={},
                    expected_snapshot={},
                    expected_producer_sha256="0" * 64,
                    samples=[],
                    fd_oracle={},
                )
            self.assertFalse((output / "COMPLETE").exists())
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "report.md").exists())
            failed_state = json.loads(
                (output / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_state["state"], "FAILED")

    def test_publisher_recomputes_sample_matrix_before_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            receipt_path = output / "receipt.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            build_receipt = self.runner.build_receipt_report_record(
                receipt_path,
                {
                    "receipt_id": "receipt",
                    "build_input_id": "build-input",
                    "artifact_set_id": "artifact-set",
                },
                {
                    "pass": True,
                    "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                },
            )
            duplicate_samples = [sample_matrix()[0]] * (
                1
                + 2 * self.runner.PROFILE["warmups"]
                + 2 * self.runner.PROFILE["repeats"]
            )
            fresh = {
                "pass": True,
                "verified_at_utc": "fresh",
                "build_receipt": build_receipt,
                "receipt_id": "receipt",
                "receipt_sha256": build_receipt["sha256"],
                "source_snapshot": {},
            }
            self.runner.prepare_output(output, "forged-run")
            with mock.patch.object(
                self.runner, "reverify_final_closure", return_value=fresh
            ), self.assertRaisesRegex(RuntimeError, "schema|matrix|nonce"):
                self.runner.publish_complete(
                    output,
                    "forged-run",
                    {
                        "state": "COMPLETE",
                        "run_id": "forged-run",
                        "profile": dict(self.runner.PROFILE),
                        "build_receipt": dict(build_receipt),
                        "samples": [{}] * len(duplicate_samples),
                        "gate": {"pass": True},
                    },
                    "forged\n",
                    capability=publication_capability(
                        self.runner, output, "forged-run"
                    ),
                    repo=output,
                    receipt_path=receipt_path,
                    expected_receipt_id="receipt",
                    expected_build_receipt_record=build_receipt,
                    expected_snapshot={},
                    expected_producer_sha256="0" * 64,
                    samples=duplicate_samples,
                    fd_oracle=fd_oracle(),
                )
            self.assertFalse((output / "COMPLETE").exists())

    def test_partial_publication_failure_leaves_no_success_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            run_id = "partial-run"
            self.runner.prepare_output(output, run_id)
            receipt_path = output / "receipt.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            receipt_record = self.runner.build_receipt_report_record(
                receipt_path,
                {
                    "receipt_id": "receipt",
                    "build_input_id": "build-input",
                    "artifact_set_id": "artifact-set",
                },
                {
                    "pass": True,
                    "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                },
            )
            comparison = self.runner.compare_measured_samples(
                sample_matrix(), fd_oracle=fd_oracle()
            )
            comparison["gate"]["driver_runtime_device_binding"] = True
            comparison["gate"]["pass"] = all(
                value
                for name, value in comparison["gate"].items()
                if name != "pass"
            )
            fresh = {
                "pass": True,
                "build_receipt": receipt_record,
                "receipt_id": "receipt",
                "receipt_sha256": receipt_record["sha256"],
                "source_snapshot": {},
                "output_inventory": prepublication_inventory(output),
            }
            markdown = self.runner.render_complete_markdown(comparison, "receipt")
            original_atomic_write_json = self.runner.atomic_write_json
            observed_states = []

            def fail_complete(path, value):
                if pathlib.Path(path).name == "state.json":
                    observed_states.append(value.get("state"))
                    if value.get("state") == "COMPLETE":
                        raise OSError("simulated final state write failure")
                return original_atomic_write_json(path, value)

            with mock.patch.object(
                self.runner, "reverify_final_closure", return_value=fresh
            ), mock.patch.object(
                self.runner,
                "validate_complete_report",
                return_value=(comparison, {"pass": True}),
            ), mock.patch.object(
                self.runner,
                "atomic_write_json",
                side_effect=fail_complete,
            ), self.assertRaisesRegex(OSError, "final state write failure"):
                self.runner.publish_complete(
                    output,
                    run_id,
                    {},
                    markdown,
                    capability=publication_capability(self.runner, output, run_id),
                    repo=output,
                    receipt_path=receipt_path,
                    expected_receipt_id="receipt",
                    expected_build_receipt_record=receipt_record,
                    expected_snapshot={},
                    expected_producer_sha256="0" * 64,
                    samples=[],
                    fd_oracle={},
                )
            for name in ("COMPLETE", "report.json", "report.md"):
                self.assertFalse((output / name).exists())
            failed_state = json.loads(
                (output / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_state["state"], "FAILED")
            self.assertEqual(
                observed_states,
                ["FINALIZING", "COMPLETE", "FAILED"],
            )

    def test_publisher_accepts_only_freshly_recomputed_positive_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            samples = sample_matrix()
            matrix = self.runner.validate_sample_matrix(samples)
            comparison = self.runner.compare_measured_samples(
                samples, fd_oracle=fd_oracle()
            )
            host_audit = bind_synthetic_host(self.runner, comparison, samples)
            comparison["gate"]["driver_runtime_device_binding"] = True
            comparison["gate"]["pass"] = all(
                value
                for name, value in comparison["gate"].items()
                if name != "pass"
            )
            receipt_path = output / "receipt.json"
            receipt_path.write_text('{"receipt": "fixture"}\n', encoding="utf-8")
            build_receipt = self.runner.build_receipt_report_record(
                receipt_path,
                {
                    "receipt_id": "receipt",
                    "build_input_id": "build-input",
                    "artifact_set_id": "artifact-set",
                },
                {
                    "pass": True,
                    "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                },
            )
            oracle = fd_oracle()
            fresh_integrity = {
                "pass": True,
                "verified_at_utc": "2026-08-12T01:00:00Z",
                "build_receipt": build_receipt,
                "receipt_id": "receipt",
                "receipt_sha256": build_receipt["sha256"],
                "source_snapshot": {},
                "contract": {
                    "pass": True,
                    "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                },
                "samples": [
                    {
                        "label": (
                            f"{sample['lane']}-{sample['sample_kind']}-"
                            f"{sample['iteration']}"
                        ),
                        "nonce": sample["nonce"],
                        "process_seconds": sample.get("process_seconds"),
                        "summary_sha256": canonical_sha256(
                            self.runner.summarize_samples([sample])[0]
                        ),
                        "files": {},
                    }
                    for sample in samples
                ],
                "finite_difference_logs": {},
            }
            first_integrity = copy.deepcopy(fresh_integrity)
            first_integrity["verified_at_utc"] = "2026-08-12T00:59:00Z"
            postprocessor = {"fixture": "exactly-bound"}
            report = {
                "schema_version": 1,
                "state": "COMPLETE",
                "run_id": "positive-run",
                "profile": dict(self.runner.PROFILE),
                "build_receipt": build_receipt,
                "finite_difference_oracle": oracle,
                "host_audit": host_audit,
                "integrity_reverification": first_integrity,
                "samples": self.runner.summarize_samples(samples),
                "sample_matrix": matrix,
                "comparison": comparison,
                "gate": comparison["gate"],
                "postprocessor": postprocessor,
            }
            markdown = self.runner.render_complete_markdown(comparison, "receipt")

            def validate_postprocessor(value, **_kwargs):
                if value != postprocessor:
                    raise RuntimeError("postprocessor mutation")
                return {"pass": True, "rank_local": {}}

            self.runner.prepare_output(output, "positive-run")
            inventory = prepublication_inventory(output)
            fresh_integrity["output_inventory"] = copy.deepcopy(inventory)
            first_integrity["output_inventory"] = copy.deepcopy(inventory)
            report["integrity_reverification"] = first_integrity
            base_report = copy.deepcopy(report)
            with mock.patch.object(
                self.runner,
                "_validate_complete_postprocessor",
                side_effect=validate_postprocessor,
            ), mock.patch.object(
                self.runner,
                "validate_host_audit",
                return_value=copy.deepcopy(comparison["host"]),
            ), mock.patch.object(
                self.runner,
                "reverify_final_closure",
                return_value=copy.deepcopy(fresh_integrity),
            ):
                self.runner.publish_complete(
                    output,
                    "positive-run",
                    copy.deepcopy(base_report),
                    markdown,
                    capability=publication_capability(
                        self.runner, output, "positive-run"
                    ),
                    repo=output,
                    receipt_path=receipt_path,
                    expected_receipt_id="receipt",
                    expected_build_receipt_record=build_receipt,
                    expected_snapshot={},
                    expected_producer_sha256="0" * 64,
                    samples=samples,
                    fd_oracle=oracle,
                )
            self.assertTrue((output / "COMPLETE").is_file())
            published = json.loads((output / "report.json").read_text(encoding="utf-8"))
            complete_marker = json.loads(
                (output / "COMPLETE").read_text(encoding="utf-8")
            )
            state_marker = json.loads(
                (output / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(complete_marker, state_marker)
            self.assertEqual(
                complete_marker["report_sha256"],
                sha256_file(output / "report.json"),
            )
            self.assertEqual(
                complete_marker["report_markdown_sha256"],
                sha256_file(output / "report.md"),
            )
            self.assertEqual(
                published["build_receipt"]["qualification_contract"],
                self.runner.BUILD_QUALIFICATION_CONTRACT,
            )
            self.assertIsInstance(
                published["build_receipt"]["qualification_contract"], str
            )

            def verify_terminal():
                return self.runner.verify_terminal_publication(
                    output,
                    expected_prepublication_inventory=inventory,
                    expected_run_id="positive-run",
                )

            replayed_marker = self.runner.verify_terminal_publication(
                output,
                expected_prepublication_inventory=inventory,
                expected_run_id="positive-run",
            )
            self.assertEqual(replayed_marker, complete_marker)
            terminal_manifest = complete_marker["terminal_output_inventory"][
                "manifest"
            ]
            self.assertEqual(
                terminal_manifest["top_level_files"],
                [
                    "COMPLETE",
                    "receipt.json",
                    "report.json",
                    "report.md",
                    "state.json",
                ],
            )
            self.assertEqual(
                [record["path"] for record in terminal_manifest["hashed_files"]],
                ["receipt.json", "report.json", "report.md"],
            )

            complete_bytes = (output / "COMPLETE").read_bytes()
            state_bytes = (output / "state.json").read_bytes()
            report_bytes = (output / "report.json").read_bytes()
            (output / "report.json").write_bytes(report_bytes + b" ")
            with self.assertRaisesRegex(RuntimeError, "hashed file changed"):
                verify_terminal()
            (output / "report.json").write_bytes(report_bytes)

            (output / "unexpected.bin").write_bytes(b"unbound")
            with self.assertRaisesRegex(RuntimeError, "unbound files"):
                verify_terminal()
            (output / "unexpected.bin").unlink()

            (output / "unexpected.bin").write_bytes(b"coherently-resigned")
            resigned = copy.deepcopy(complete_marker)
            resigned_manifest = resigned["terminal_output_inventory"]["manifest"]
            resigned_manifest["hashed_files"].append(
                {
                    "path": "unexpected.bin",
                    "size_bytes": (output / "unexpected.bin").stat().st_size,
                    "sha256": sha256_file(output / "unexpected.bin"),
                }
            )
            resigned_manifest["hashed_files"].sort(key=lambda record: record["path"])
            resigned_manifest["top_level_files"].append("unexpected.bin")
            resigned_manifest["top_level_files"].sort()
            resigned["terminal_output_inventory"]["sha256"] = canonical_sha256(
                resigned_manifest
            )
            self.runner.atomic_write_json(output / "COMPLETE", resigned)
            self.runner.atomic_write_json(output / "state.json", resigned)
            with self.assertRaisesRegex(RuntimeError, "exact cross-epoch closure"):
                verify_terminal()
            (output / "unexpected.bin").unlink()
            (output / "COMPLETE").write_bytes(complete_bytes)
            (output / "state.json").write_bytes(state_bytes)

            (output / "state.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "terminal markers differ"):
                verify_terminal()
            (output / "state.json").write_bytes(state_bytes)

            (output / "unexpected-link").symlink_to(output / "report.json")
            with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
                verify_terminal()
            (output / "unexpected-link").unlink()

            forged = copy.deepcopy(complete_marker)
            forged_manifest = forged["terminal_output_inventory"]["manifest"]
            forged_manifest["hashed_files"] = [
                record
                for record in forged_manifest["hashed_files"]
                if record["path"] != "report.md"
            ]
            forged_manifest["top_level_files"].remove("report.md")
            forged["terminal_output_inventory"]["sha256"] = canonical_sha256(
                forged_manifest
            )
            self.runner.atomic_write_json(output / "COMPLETE", forged)
            self.runner.atomic_write_json(output / "state.json", forged)
            with self.assertRaisesRegex(RuntimeError, "exact cross-epoch closure"):
                verify_terminal()
            (output / "COMPLETE").write_bytes(complete_bytes)
            (output / "state.json").write_bytes(state_bytes)
            self.assertEqual(verify_terminal(), complete_marker)

            receipt_mutations = {
                "path": str(output / "other-receipt.json"),
                "sha256": "f" * 64,
                "receipt_id": "other-receipt",
                "build_input_id": "other-build-input",
                "artifact_set_id": "other-artifact-set",
                "qualification_contract": {
                    "pass": True,
                    "contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                },
            }
            for field, value in receipt_mutations.items():
                invalid = copy.deepcopy(base_report)
                invalid["build_receipt"][field] = value
                self.runner.prepare_output(output, "positive-run")
                with self.subTest(receipt_field=field), mock.patch.object(
                    self.runner,
                    "reverify_final_closure",
                    return_value=copy.deepcopy(fresh_integrity),
                ), self.assertRaisesRegex(RuntimeError, "schema or inputs"):
                    self.runner.publish_complete(
                        output,
                        "positive-run",
                        invalid,
                        markdown,
                        capability=publication_capability(
                            self.runner, output, "positive-run"
                        ),
                        repo=output,
                        receipt_path=receipt_path,
                        expected_receipt_id="receipt",
                        expected_build_receipt_record=build_receipt,
                        expected_snapshot={},
                        expected_producer_sha256="0" * 64,
                        samples=samples,
                        fd_oracle=oracle,
                    )

            mutation_cases = {}
            extra = copy.deepcopy(base_report)
            extra["unexpected"] = True
            mutation_cases["extra-top-level"] = extra
            missing = copy.deepcopy(base_report)
            del missing["finite_difference_oracle"]
            mutation_cases["missing-top-level"] = missing
            changed_samples = copy.deepcopy(base_report)
            changed_samples["samples"][1]["process_seconds"] += 1.0
            mutation_cases["sample-timing"] = changed_samples
            changed_fd = copy.deepcopy(base_report)
            changed_fd["finite_difference_oracle"]["pass"] = False
            mutation_cases["fd-oracle"] = changed_fd
            changed_integrity = copy.deepcopy(base_report)
            changed_integrity["integrity_reverification"]["receipt_id"] = "other"
            mutation_cases["first-integrity"] = changed_integrity
            changed_postprocessor = copy.deepcopy(base_report)
            changed_postprocessor["postprocessor"] = {"fixture": "mutated"}
            mutation_cases["postprocessor"] = changed_postprocessor
            changed_comparison = copy.deepcopy(base_report)
            changed_comparison["comparison"]["performance"]["fresh_process_wall"][
                "comparisons"
            ]["cuda_single_over_cuda_multi"][
                "median_speedup"
            ] += 1.0
            mutation_cases["comparison"] = changed_comparison
            changed_gate = copy.deepcopy(base_report)
            changed_gate["gate"]["gradient"] = False
            mutation_cases["gate"] = changed_gate
            for name, invalid in mutation_cases.items():
                self.runner.prepare_output(output, "positive-run")
                with self.subTest(report_mutation=name), mock.patch.object(
                    self.runner,
                    "_validate_complete_postprocessor",
                    side_effect=validate_postprocessor,
                ), mock.patch.object(
                    self.runner,
                    "reverify_final_closure",
                    return_value=copy.deepcopy(fresh_integrity),
                ), self.assertRaises(RuntimeError):
                    self.runner.publish_complete(
                        output,
                        "positive-run",
                        invalid,
                        markdown,
                        capability=publication_capability(
                            self.runner, output, "positive-run"
                        ),
                        repo=output,
                        receipt_path=receipt_path,
                        expected_receipt_id="receipt",
                        expected_build_receipt_record=build_receipt,
                        expected_snapshot={},
                        expected_producer_sha256="0" * 64,
                        samples=samples,
                        fd_oracle=oracle,
                    )

            jointly_mutated_receipt = copy.deepcopy(build_receipt)
            jointly_mutated_receipt["build_input_id"] = "jointly-mutated"
            jointly_mutated_report = copy.deepcopy(base_report)
            jointly_mutated_report["build_receipt"] = jointly_mutated_receipt
            self.runner.prepare_output(output, "positive-run")
            with mock.patch.object(
                self.runner,
                "reverify_final_closure",
                return_value=copy.deepcopy(fresh_integrity),
            ), self.assertRaisesRegex(RuntimeError, "reverification is not bound"):
                self.runner.publish_complete(
                    output,
                    "positive-run",
                    jointly_mutated_report,
                    markdown,
                    capability=publication_capability(
                        self.runner, output, "positive-run"
                    ),
                    repo=output,
                    receipt_path=receipt_path,
                    expected_receipt_id="receipt",
                    expected_build_receipt_record=jointly_mutated_receipt,
                    expected_snapshot={},
                    expected_producer_sha256="0" * 64,
                    samples=samples,
                    fd_oracle=oracle,
                )

    def test_nvidia_probe_ignores_parent_path_injection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            trusted = root / "trusted"
            injected = root / "injected"
            trusted.mkdir()
            injected.mkdir()
            trusted.chmod(0o755)
            for directory, marker in ((trusted, "trusted"), (injected, "injected")):
                executable = directory / "nvidia-smi"
                executable.write_text(
                    f"#!/bin/sh\necho {marker}\n", encoding="utf-8"
                )
                executable.chmod(0o755)
            with mock.patch.object(
                self.runner,
                "TRUSTED_NVIDIA_SMI_DIRECTORIES",
                (trusted,),
            ), mock.patch.dict(os.environ, {"PATH": str(injected)}):
                probe = self.runner._nvidia_smi_probe([], root)
            self.assertEqual(probe["stdout"], "trusted")
            recorded_executable = pathlib.Path(probe["executable"]["path"])
            if not recorded_executable.is_absolute():
                recorded_executable = root / recorded_executable
            self.assertEqual(
                recorded_executable.resolve(),
                (trusted / "nvidia-smi").resolve(),
            )

    def test_build_receipt_contract_rejects_generic_or_drifted_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo = root / "repo"
            (repo / "scripts").mkdir(parents=True)
            (repo / "environment").mkdir()
            (repo / ".tools").mkdir()
            prefix = root / "prefix"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "etc").mkdir()
            prte_params = prefix / "etc" / "prte-mca-params.conf"
            prte_hostfile = prefix / "etc" / "prte-default-hostfile"
            prte_params.write_text("# synthetic PRRTE params\n", encoding="utf-8")
            prte_hostfile.write_text("# synthetic PRRTE hostfile\n", encoding="utf-8")
            build_dir = repo / "build" / "meep-cuda-mpi-python-fp32"
            qualification_home = build_dir / "qualification-home"
            qualification_home.mkdir(parents=True)
            home_marker = qualification_home / ".gpmeep-empty-home"
            home_marker.write_text(
                "gpmeep isolated MPI qualification home\n", encoding="utf-8"
            )
            dependency_groups = {}
            for group, relative in (
                ("mpi", "lib/libmpi.so"),
                ("cuda_runtime", "lib/libcudart.so"),
                ("hdf5", "lib/libhdf5.so"),
            ):
                path = prefix / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(group + "\n", encoding="utf-8")
                dependency_groups[group] = [
                    {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                ]
            driver_directory = root / "driver"
            driver_directory.mkdir()
            driver_directory.chmod(0o755)
            driver = driver_directory / "libcuda.so"
            driver.write_text("cuda_driver\n", encoding="utf-8")
            driver.chmod(0o755)
            dependency_groups["cuda_driver"] = [
                {
                    "path": str(driver.resolve()),
                    "size_bytes": driver.stat().st_size,
                    "sha256": sha256_file(driver),
                }
            ]
            dependency_groups["environment"] = [
                dict(record)
                for group in ("mpi", "cuda_runtime", "hdf5")
                for record in dependency_groups[group]
            ]
            expected_install = repo / "install" / "meep-cuda-mpi-python-fp32"
            installed_runtime_paths = (
                expected_install
                / "lib/python3.11/site-packages/meep/_meep.so.38.0.0",
                expected_install
                / "lib/python3.11/site-packages/meep/mpb/_mpb.so.38.0.0",
                expected_install / "lib/libmeep.so.38.0.0",
                expected_install / "lib/libpympb.so.38.0.0",
            )
            for path, label in zip(
                installed_runtime_paths,
                ("_meep", "_mpb", "libmeep", "libpympb"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(label + " synthetic runtime\n", encoding="utf-8")
            dependency_groups["installed"] = [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in installed_runtime_paths
            ]
            dependency_closure = build_dir / "runtime-dependency-closure.json"
            dependency_closure.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "environment_prefix": str(prefix.resolve()),
                        "installed_prefix": str(expected_install.resolve()),
                        "groups": dependency_groups,
                    }
                ),
                encoding="utf-8",
            )
            old_driver_roots = self.runner.COMPARATOR.TRUSTED_CUDA_DRIVER_ROOTS
            self.runner.COMPARATOR.TRUSTED_CUDA_DRIVER_ROOTS = (root,)
            self.addCleanup(
                setattr,
                self.runner.COMPARATOR,
                "TRUSTED_CUDA_DRIVER_ROOTS",
                old_driver_roots,
            )
            tools = {}
            for name in self.runner.REQUIRED_TOOLS:
                path = prefix / "bin" / name
                path.write_text(name + "\n", encoding="utf-8")
                tools[name] = {"path": str(path)}
            trusted_timeout = next(
                path.resolve()
                for path in self.runner.TRUSTED_TIMEOUT_PATHS
                if path.is_file()
            )
            tools["timeout"] = {"path": str(trusted_timeout)}
            lock = root / "canonical.lock"
            explicit = root / "environment.lock"
            package = "https://example.invalid/pkg.conda#" + "0" * 64
            lock.write_text("@EXPLICIT\n" + package + "\n", encoding="utf-8")
            explicit.write_text("@EXPLICIT\n" + package + "\n", encoding="utf-8")
            lock_record = file_record(lock, repo)
            fresh_nonce = secrets.token_hex(32)
            build_home = (
                repo / ".micromamba" / "cache" / f"build-home-{fresh_nonce}"
            )
            build_home.mkdir(mode=0o700, parents=True)
            (build_home / ".gpmeep-empty-build-home").write_text(
                "gpmeep isolated build home\n", encoding="utf-8"
            )
            (build_home / "cache").mkdir()
            (build_home / "cache" / "fontconfig").mkdir()
            (build_home / "matplotlib").mkdir()
            qualification_fontconfig = build_dir / "qualification-fontconfig.conf"
            qualification_fontconfig.write_text(
                self.runner._qualification_fontconfig_text(prefix, build_home),
                encoding="utf-8",
            )
            self.addCleanup(shutil.rmtree, build_home, True)
            installed_environment_manifest = {
                "root": str(prefix),
                "files": [{"path": "bin/python"}],
            }
            fresh_attestation = build_dir / f"cuda-mpi-{fresh_nonce}.json"
            fresh_attestation.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "fresh-exact-lock-environment",
                        "nonce": fresh_nonce,
                        "created_at_unix_ns": 1_700_000_000_000_000_000,
                        "creator_pid": 1234,
                        "repo": str(repo.resolve()),
                        "environment_prefix": str(prefix.resolve()),
                        "lock": lock_record,
                        "explicit_package_count": 1,
                        "environment_manifest": installed_environment_manifest,
                    }
                ),
                encoding="utf-8",
            )
            fresh_attestation.chmod(0o600)
            package_cache = repo / ".micromamba" / "pkgs"
            package_cache.mkdir(parents=True)
            prefix_content_audit = build_dir / "conda-prefix-content-audit.json"
            prefix_content_audit.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "environment_prefix": str(prefix.resolve()),
                        "python_version": "3.11",
                        "package_cache": {
                            "path": str(package_cache.resolve()),
                            "archive_count": 1,
                        },
                        "archive_snapshot": {
                            "open_flags": [
                                "O_RDONLY",
                                "O_CLOEXEC",
                                "O_NOFOLLOW",
                            ],
                            "hash_and_parse_same_private_snapshot": True,
                            "descriptor_retained_until_terminal_verification": True,
                            "pathname_fingerprint_rechecked": True,
                        },
                        "archive_decoder": {
                            "path": "/usr/bin/zstd",
                            "sha256": sha256_file(pathlib.Path("/usr/bin/zstd")),
                            "version": "fixture-zstd",
                            "timeout_seconds": 120,
                        },
                        "lock": {
                            "path": str(lock.resolve()),
                            "package_count": 1,
                            "sha256": lock_record["sha256"],
                        },
                        "metadata_package_count": 1,
                        "owned_path_count": 1,
                        "hashed_installed_path_count": 1,
                        "archive_bound_hardlink_count": 1,
                        "relocated_hardlink_count": 0,
                        "relocated_source_bytecode_count": 1,
                        "relocated_source_bytecode": [
                            {
                                "current_state": "relocated-source-compiled",
                                "current_sha256": "1" * 64,
                                "derived_sha256": "1" * 64,
                            }
                        ],
                        "generated_source_bytecode_count": 1,
                        "generated_source_bytecode": [
                            {
                                "current_state": "source-compiled",
                                "current_sha256": "2" * 64,
                                "derived_sha256": "2" * 64,
                            }
                        ],
                        "declared_generated_bytecode_path_count": 1,
                        "generated_entry_points": [],
                        "softlink_count": 0,
                        "softlinks": [],
                        "unowned_allowed_path_count": 0,
                        "package_records": [
                            {
                                "url": package.rpartition("#")[0],
                                "sha256": package.rpartition("#")[2],
                            }
                        ],
                        "pass": True,
                    }
                ),
                encoding="utf-8",
            )
            logs = root / "logs"
            logs.mkdir()
            log_records = []
            for name in self.runner.REQUIRED_QUALIFICATION_LOGS:
                path = logs / name
                expected_diagnostic = self.runner.EXPECTED_MPI_FAILURE_LOGS.get(name)
                expected_failure = ""
                if expected_diagnostic is not None:
                    expected_failure = (
                        expected_diagnostic
                        + "\n"
                        + f"gpmeep-expected-mpi-failure:{name}:status=1:"
                        + f"diagnostic={expected_diagnostic}\n"
                    )
                path.write_text(
                    expected_failure + f"gpmeep-qualification:{name}:PASS\n",
                    encoding="utf-8",
                )
                log_records.append({"path": name})
            configure_argv = [
                "--enable-maintainer-mode",
                "--enable-shared",
                "--enable-single",
                "--enable-cuda",
                "--with-cuda-arch=AUTO",
                "--with-openmp",
                "--with-mpi",
                "--with-python",
                "--without-scheme",
                f"--prefix={expected_install}",
                "--disable-cuda-fast-math",
            ]
            config_h = root / "config.h"
            config_h.write_text(
                "#define HAVE_CUDA 1\n#define HAVE_MPI 1\n#define HAVE_OPENMP 1\n",
                encoding="utf-8",
            )
            config_status = build_dir / "config.status"
            quoted_configure = " ".join(f"'{value}'" for value in configure_argv)
            config_status.write_text(
                "  set X /bin/bash '../../configure' "
                + quoted_configure
                + " 'CC=/synthetic'\n"
                + '\n'.join(
                    (
                        'S["MEEP_HAVE_CUDA"]="1"',
                        'S["MEEP_SINGLE"]="1"',
                        'S["WITH_PYTHON_FALSE"]="#"',
                        'S["WITH_PYTHON_TRUE"]=""',
                        'S["WITH_SCHEME_FALSE"]=""',
                        'S["WITH_SCHEME_TRUE"]="#"',
                        'D["HAVE_MPI"]=" 1"',
                        'D["HAVE_CUDA"]=" 1"',
                        'D["HAVE_OPENMP"]=" 1"',
                        'S["NVCCFLAGS"]="-O3"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            compiler_prefix = prefix / "bin" / "x86_64-conda-linux-gnu"
            pathlib.Path(f"{compiler_prefix}-c++").write_text(
                "synthetic host compiler\n", encoding="utf-8"
            )
            mamba_root = repo / ".micromamba"
            fixed_mpi_environment = {
                "OMPI_MCA_mca_base_param_files": str(
                    repo
                    / "environment"
                    / "openmpi-qualification-mca-params.conf"
                ),
                "OMPI_MCA_mca_base_component_path": str(
                    prefix / "lib" / "openmpi"
                ),
                "PMIX_MCA_mca_base_component_path": str(prefix / "lib" / "pmix"),
                "PMIX_MCA_mca_base_param_files": str(
                    repo
                    / "environment"
                    / "openmpi-qualification-mca-params.conf"
                ),
                "PRTE_MCA_mca_base_param_files": str(
                    repo
                    / "environment"
                    / "openmpi-qualification-mca-params.conf"
                ),
            }
            canonical_environment = {
                **fixed_mpi_environment,
                "CC": f"{compiler_prefix}-cc",
                "CC_FOR_BUILD": f"{compiler_prefix}-cc",
                "CFLAGS": f"-O2 -I{prefix}/include",
                "CPPFLAGS": f"-O2 -I{prefix}/include",
                "CXX": f"{compiler_prefix}-c++",
                "CXX_FOR_BUILD": f"{compiler_prefix}-c++",
                "CXXFLAGS": f"-O2 -I{prefix}/include",
                "F77": f"{compiler_prefix}-gfortran",
                "FC": f"{compiler_prefix}-gfortran",
                "FC_FOR_BUILD": f"{compiler_prefix}-gfortran",
                "FFLAGS": f"-O2 -I{prefix}/include",
                "GPMEEP_FRESH_ENV_NONCE": fresh_nonce,
                "HOME": str(build_home),
                "LDFLAGS": f"-O2 -L{prefix}/lib",
                "MAMBA_EXE": str(repo / ".tools" / "micromamba"),
                "MAMBA_ROOT_PREFIX": str(mamba_root),
                "MEEP_GPU_MULTI_INITIAL_CONDITION": "trigonometric-v1",
                "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
                "MEEP_GPU_FAST_MATH": "OFF",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(build_home / "matplotlib"),
                "NVCC_PREPEND_FLAGS": f" -ccbin={compiler_prefix}-c++",
                "PATH": ":".join(
                    (
                        str(prefix / "bin"),
                        str(mamba_root / "condabin"),
                        "/usr/local/sbin",
                        "/usr/local/bin",
                        "/usr/sbin",
                        "/usr/bin",
                        "/sbin",
                        "/bin",
                    )
                ),
                "PWD": str(repo.resolve()),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPYCACHEPREFIX": "/dev/null",
                "PYTHONSAFEPATH": "1",
                "SHLVL": "<shell-managed>",
                "_": "<shell-managed>",
                "XDG_CACHE_HOME": str(build_home / "cache"),
                "CONDA_DEFAULT_ENV": str(prefix),
                "CONDA_PKGS_DIRS": str(mamba_root / "pkgs"),
                "CONDA_PREFIX": str(prefix),
                "CONDA_SHLVL": "1",
            }
            canonical_build_environment = (
                build_dir / "canonical-build-environment.json"
            )
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": canonical_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            cuda_runtime_flags_stamp = (
                build_dir / "src" / "cuda-runtime-flags.stamp"
            )
            cuda_runtime_flags_stamp.parent.mkdir()
            cuda_runtime_flags_stamp.write_text(
                f"NVCC={prefix / 'bin' / 'nvcc'}\n"
                f"CUDAHOSTCXX={compiler_prefix}-c++\n"
                "NVCCFLAGS=-O3\n"
                "CUDA_ARCH_FLAGS=-gencode=arch=compute_80,code=sm_80\n",
                encoding="utf-8",
            )
            cuda_runtime_cmake_cache = (
                build_dir / "cuda-runtime-qualification" / "CMakeCache.txt"
            )
            cuda_runtime_cmake_cache.parent.mkdir()
            cuda_runtime_cmake_cache.write_text(
                "MEEP_GPU_FAST_MATH:BOOL=OFF\n"
                "CMAKE_CUDA_FLAGS:STRING=\n"
                "CMAKE_CUDA_FLAGS_DEBUG:STRING=-g\n"
                "CMAKE_CUDA_FLAGS_MINSIZEREL:STRING=-O1 -DNDEBUG\n"
                "CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
                "CMAKE_CUDA_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG\n"
                f"CMAKE_CUDA_COMPILER:UNINITIALIZED={prefix / 'bin' / 'nvcc'}\n"
                f"CMAKE_CUDA_HOST_COMPILER:UNINITIALIZED={compiler_prefix}-c++\n",
                encoding="utf-8",
            )
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "build_dir": str(build_dir),
                "started_at_utc": "2023-11-14T22:13:21Z",
                "configuration": {
                    "qualification_contract": self.runner.BUILD_QUALIFICATION_CONTRACT,
                    "builder": {
                        "path": str(repo / "scripts" / "build-meep-cuda-mpi-python.sh")
                    },
                    "configure_argv": configure_argv,
                    "environment": {
                        **{
                            name: None
                            for name in self.runner.RECORDED_BUILD_ENVIRONMENT_KEYS
                        },
                        **fixed_mpi_environment,
                        "CC": canonical_environment["CC"],
                        "CXX": canonical_environment["CXX"],
                        "FC": canonical_environment["FC"],
                        "F77": canonical_environment["F77"],
                        "GPMEEP_FRESH_ENV_NONCE": fresh_nonce,
                        "MEEP_GPU_FAST_MATH": "OFF",
                    },
                    "lockfiles": {"environment_lock": lock_record},
                },
                "configuration_files": {
                    "micromamba": {
                        "path": str((repo / ".tools" / "micromamba").resolve()),
                        "sha256": self.runner.PINNED_MICROMAMBA_SHA256,
                    },
                    "config_h": {"path": str(config_h)},
                    "config_status": {"path": str(config_status)},
                    "environment_explicit": {"path": str(explicit)},
                    "openmpi_qualification_params": {
                        "path": str(
                            repo
                            / "environment"
                            / "openmpi-qualification-mca-params.conf"
                        )
                    },
                    "prte_mca_params": {"path": str(prte_params)},
                    "prte_default_hostfile": {"path": str(prte_hostfile)},
                    "runtime_dependency_closure": {
                        "path": str(dependency_closure)
                    },
                    "fresh_environment_attestation": {
                        "path": str(fresh_attestation)
                    },
                    "conda_prefix_content_audit": {
                        "path": str(prefix_content_audit)
                    },
                    "canonical_build_environment": {
                        "path": str(canonical_build_environment)
                    },
                    "cuda_runtime_flags_stamp": {
                        "path": str(cuda_runtime_flags_stamp)
                    },
                    "cuda_runtime_cmake_cache": {
                        "path": str(cuda_runtime_cmake_cache)
                    },
                    "qualification_fontconfig": {
                        "path": str(qualification_fontconfig)
                    },
                },
                "artifacts": {
                    **{name: {} for name in self.runner.REQUIRED_ARTIFACTS},
                    "installed_python_extension": file_record(
                        installed_runtime_paths[0], repo
                    ),
                    "installed_mpb_extension": file_record(
                        installed_runtime_paths[1], repo
                    ),
                    "installed_libmeep": file_record(
                        installed_runtime_paths[2], repo
                    ),
                    "installed_libpympb": file_record(
                        installed_runtime_paths[3], repo
                    ),
                },
                "manifests": {
                    "in_place_python": {},
                    "installed_python": {},
                    "installed_environment": installed_environment_manifest,
                    "installed_prefix": tree_manifest(expected_install, repo),
                    "qualification_logs": {
                        "root": str(logs),
                        "files": log_records,
                    },
                    "qualification_home": {
                        "root": str(qualification_home),
                        "files": [{"path": ".gpmeep-empty-home"}],
                    },
                    "build_home": tree_manifest(build_home, repo),
                },
                "toolchain": tools,
            }
            reproduction = mock.patch.object(
                self.runner,
                "_reproduce_clean_build_environment",
                return_value=canonical_environment,
            )
            reproduction_mock = reproduction.start()
            self.addCleanup(reproduction.stop)
            prefix_reaudit = mock.patch.object(
                self.runner,
                "_rerun_conda_prefix_audit",
                return_value=json.loads(
                    prefix_content_audit.read_text(encoding="utf-8")
                ),
            )
            prefix_reaudit_mock = prefix_reaudit.start()
            self.addCleanup(prefix_reaudit.stop)
            validated_contract = self.runner.validate_build_receipt_contract(
                receipt, repo
            )
            self.assertTrue(validated_contract["pass"])
            self.assertEqual(
                validated_contract["qualification_logs"],
                sorted(record["path"] for record in log_records),
            )
            fast_math_configure = copy.deepcopy(receipt)
            fast_math_configure["configuration"]["configure_argv"][-1] = (
                "--enable-cuda-fast-math"
            )
            with self.assertRaisesRegex(RuntimeError, "exact CUDA/MPI"):
                self.runner.validate_build_receipt_contract(
                    fast_math_configure, repo
                )
            fast_math_receipt_environment = copy.deepcopy(receipt)
            fast_math_receipt_environment["configuration"]["environment"][
                "MEEP_GPU_FAST_MATH"
            ] = "ON"
            with self.assertRaisesRegex(RuntimeError, "CUDA math policy"):
                self.runner.validate_build_receipt_contract(
                    fast_math_receipt_environment, repo
                )
            canonical_fast_math = dict(canonical_environment)
            canonical_fast_math["MEEP_GPU_FAST_MATH"] = "ON"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": canonical_fast_math,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA math policy"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": canonical_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            stamp_off = cuda_runtime_flags_stamp.read_text(encoding="utf-8")
            cuda_runtime_flags_stamp.write_text(
                stamp_off.replace("NVCCFLAGS=-O3", "NVCCFLAGS=-O3 --use_fast_math"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA math policy"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            cuda_runtime_flags_stamp.write_text(stamp_off, encoding="utf-8")
            cache_off = cuda_runtime_cmake_cache.read_text(encoding="utf-8")
            for cache_payload in (
                cache_off.replace(
                    "MEEP_GPU_FAST_MATH:BOOL=OFF",
                    "MEEP_GPU_FAST_MATH:BOOL=ON",
                ),
                cache_off.replace(
                    "CMAKE_CUDA_FLAGS:STRING=",
                    "CMAKE_CUDA_FLAGS:STRING=--use_fast_math",
                ),
            ):
                cuda_runtime_cmake_cache.write_text(
                    cache_payload, encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "CUDA math policy"):
                    self.runner.validate_build_receipt_contract(receipt, repo)
            cuda_runtime_cmake_cache.write_text(cache_off, encoding="utf-8")
            qualification_fontconfig.write_text(
                self.runner._qualification_fontconfig_text(prefix, build_home).replace(
                    '<cachedir prefix="xdg">fontconfig</cachedir>',
                    "<cachedir>/tmp/unbound-font-cache</cachedir>",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "fontconfig is not isolated"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            qualification_fontconfig.write_text(
                self.runner._qualification_fontconfig_text(prefix, build_home),
                encoding="utf-8",
            )
            protected_mismatch = copy.deepcopy(prefix_reaudit_mock.return_value)
            protected_mismatch["owned_path_count"] += 1
            prefix_reaudit_mock.return_value = protected_mismatch
            with self.assertRaisesRegex(RuntimeError, "protected re-audit"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            prefix_reaudit_mock.return_value = json.loads(
                prefix_content_audit.read_text(encoding="utf-8")
            )
            poisoned_environment = dict(canonical_environment)
            poisoned_environment["NVCC_APPEND_FLAGS"] = "--use_fast_math"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": poisoned_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unapproved variables"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            missing_initial_condition = dict(canonical_environment)
            del missing_initial_condition["MEEP_GPU_MULTI_INITIAL_CONDITION"]
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": missing_initial_condition,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            invalid_initial_condition = dict(canonical_environment)
            invalid_initial_condition["MEEP_GPU_MULTI_INITIAL_CONDITION"] = None
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": invalid_initial_condition,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "invalid entries"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            wrong_initial_condition = dict(canonical_environment)
            wrong_initial_condition[
                "MEEP_GPU_MULTI_INITIAL_CONDITION"
            ] = "affine-v1"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": wrong_initial_condition,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "clean worker environment"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            missing_source_profile = dict(canonical_environment)
            del missing_source_profile["MEEP_GPU_MULTI_SOURCE_PROFILE"]
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": missing_source_profile,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            wrong_source_profile = dict(canonical_environment)
            wrong_source_profile[
                "MEEP_GPU_MULTI_SOURCE_PROFILE"
            ] = "dual-electric-v1"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": wrong_source_profile,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "clean worker environment"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": canonical_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            reproduction_mock.return_value = wrong_initial_condition
            with self.assertRaisesRegex(RuntimeError, "fresh clean activation"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            reproduction_mock.return_value = canonical_environment
            poisoned_environment = dict(canonical_environment)
            poisoned_environment["PYTHONSAFEPATH"] = "0"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": poisoned_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "clean worker environment"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            poisoned_environment = dict(canonical_environment)
            poisoned_environment["NVCC_PREPEND_FLAGS"] += " --use_fast_math"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": poisoned_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "NVCC implicit flags"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            poisoned_environment = dict(canonical_environment)
            poisoned_environment["CFLAGS"] += " -O0 -include /tmp/poison.h"
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": poisoned_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "fresh clean activation"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            canonical_build_environment.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": canonical_environment,
                        "normalized_shell_keys": ["SHLVL", "_"],
                    }
                ),
                encoding="utf-8",
            )
            generic = copy.deepcopy(receipt)
            generic["configuration"]["qualification_contract"] = "generic"
            with self.assertRaisesRegex(RuntimeError, "wrong qualification"):
                self.runner.validate_build_receipt_contract(generic, repo)
            conflicting = copy.deepcopy(receipt)
            conflicting["configuration"]["configure_argv"].append("--disable-cuda")
            with self.assertRaisesRegex(RuntimeError, "exact CUDA/MPI"):
                self.runner.validate_build_receipt_contract(conflicting, repo)
            expected_failure_log = logs / "dft-norm-rank-failure-two-rank.log"
            missing_raw_diagnostic = self.runner.EXPECTED_MPI_FAILURE_LOGS[
                "dft-norm-rank-failure-two-rank.log"
            ]
            expected_failure_log.write_text(
                "gpmeep-expected-mpi-failure:dft-norm-rank-failure-two-rank.log:"
                + f"status=1:diagnostic={missing_raw_diagnostic}\n"
                + "gpmeep-qualification:dft-norm-rank-failure-two-rank.log:PASS\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unbound reason"):
                self.runner.validate_build_receipt_contract(receipt, repo)
            diagnostic = self.runner.EXPECTED_MPI_FAILURE_LOGS[
                "dft-norm-rank-failure-two-rank.log"
            ]
            expected_failure_log.write_bytes(
                diagnostic.encode("utf-8")
                + b"\n\0\n"
                + (
                    "gpmeep-expected-mpi-failure:"
                    "dft-norm-rank-failure-two-rank.log:"
                    f"status=1:diagnostic={diagnostic}\n"
                    "gpmeep-qualification:"
                    "dft-norm-rank-failure-two-rank.log:PASS\n"
                ).encode("utf-8")
            )
            self.runner.validate_build_receipt_contract(receipt, repo)
            expected_failure_log.write_text(
                diagnostic
                + "\n"
                + "gpmeep-expected-mpi-failure:dft-norm-rank-failure-two-rank.log:"
                + f"status=1:diagnostic={diagnostic}\n"
                + "gpmeep-qualification:dft-norm-rank-failure-two-rank.log:PASS\n",
                encoding="utf-8",
            )
            drifted = copy.deepcopy(receipt)
            explicit.write_text(
                "@EXPLICIT\nhttps://example.invalid/other.conda#" + "1" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                self.runner.validate_build_receipt_contract(drifted, repo)

    def test_timeout_terminates_owned_process_group_and_preserves_text_logs(self):
        code = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(child.pid, flush=True); time.sleep(30)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(self.runner.CommandTimeout) as caught:
                self.runner._run_command(
                    [sys.executable, "-c", code],
                    cwd=pathlib.Path(temporary),
                    environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    timeout_seconds=1,
                )
            self.assertIsInstance(caught.exception.stdout, str)
            child_pid = int(caught.exception.stdout.strip())
            terminated = False
            for _ in range(20):
                status = pathlib.Path(f"/proc/{child_pid}/status")
                if not status.exists() or "State:\tZ" in status.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    terminated = True
                    break
                time.sleep(0.05)
            self.assertTrue(terminated, "timed-out descendant remained live")

    def test_cpu_oracle_rejects_all_zero_gradient(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = runtime_closure(pathlib.Path(temporary))
            gradient = [0.25, -0.5]
            statistics = {
                "dispatch": {"cpu_curl_calls": 1, "cuda_curl_calls": 0},
                "field_updates": {
                    "cpu_update_eh_calls": 1,
                    "cuda_update_eh_calls": 0,
                },
                "sources": {"cpu_source_calls": 1, "cuda_source_calls": 0},
                "boundaries": {
                    "cpu_boundary_calls": 1,
                    "cuda_boundary_calls": 0,
                },
                "dfts": {"cpu_dft_calls": 1, "cuda_dft_calls": 0},
                "multi_gpu": {
                    "mpi_messages": 0,
                    "mpi_scalars": 0,
                    "cuda_aware_bytes": 0,
                    "pinned_staging_bytes": 0,
                    "pinned_device_to_host_bytes": 0,
                    "pinned_host_to_device_bytes": 0,
                },
            }
            raw = {
                "schema_version": 4,
                "runtime": runtime,
                "result": result_payload(gradient)["result"],
                "distributed": {
                    "world_size": 1,
                    "ranks": [
                        {
                            "runtime": copy.deepcopy(runtime),
                            "backend": {
                                "requested": "cpu",
                                "active": "cpu",
                                "single_precision": True,
                                "strict_cuda_marker": False,
                            },
                            "gradient": gradient,
                            "gradient_sha256": canonical_float64_sha256(gradient),
                            "final_statistics": statistics,
                        }
                    ],
                },
            }
            self.assertEqual(self.runner.validate_cpu_oracle(raw)["gradient"], gradient)
            zero = copy.deepcopy(raw)
            zero["result"].update(result_payload([0.0, 0.0])["result"])
            zero["distributed"]["ranks"][0]["gradient"] = [0.0, 0.0]
            with self.assertRaisesRegex(RuntimeError, "all zero"):
                self.runner.validate_cpu_oracle(zero)


if __name__ == "__main__":
    unittest.main()
