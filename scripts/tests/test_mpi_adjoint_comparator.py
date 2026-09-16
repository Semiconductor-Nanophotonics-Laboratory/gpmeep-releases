from __future__ import annotations

import copy
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gpmeep_benchmark_evidence import (  # noqa: E402
    canonical_float64_sha256,
    sha256_file,
)
import gpmeep_provenance  # noqa: E402
from gpmeep_provenance import StatHashCache, tree_manifest  # noqa: E402


def load_comparator():
    path = SCRIPTS / "compare-mpi-adjoint-benchmarks.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_mpi_adjoint_comparator", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import MPI adjoint comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_closure(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    package = root / "meep"
    package.mkdir(exist_ok=True)
    paths = {
        "python_executable": root / "python_executable",
        "meep_module": package / "__init__.py",
        "extension": package / "_meep.so",
        "libmeep": root / "libmeep.so.38",
    }
    result = {}
    for name, path in paths.items():
        path.write_bytes((name + "\n").encode())
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
        path.write_bytes((group + "\n").encode())
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


def statistics(*, mpi: bool, cpu_fallback: int = 0):
    return {
        "dispatch": {
            "cuda_curl_calls": 10,
            "cuda_curl_points": 100,
            "cpu_curl_calls": cpu_fallback,
            "cpu_curl_points": 0,
        },
        "field_updates": {
            "cuda_update_eh_calls": 10,
            "cuda_update_eh_points": 100,
            "cpu_update_eh_calls": 0,
            "cpu_update_eh_points": 0,
        },
        "polarizations": {
            "cuda_polarization_calls": 0,
            "cuda_polarization_points": 0,
            "cpu_polarization_calls": 0,
            "cpu_polarization_points": 0,
        },
        "sources": {
            "cuda_source_calls": 10,
            "cuda_source_points": 100,
            "cpu_source_calls": 0,
            "cpu_source_points": 0,
        },
        "boundaries": {
            "cuda_boundary_calls": 10,
            "cuda_boundary_points": 100,
            "cpu_boundary_calls": 0,
            "cpu_boundary_points": 0,
        },
        "dfts": {
            "cuda_dft_calls": 10,
            "cuda_dft_points": 100,
            "cpu_dft_calls": 0,
            "cpu_dft_points": 0,
        },
        "multi_gpu": {
            "mpi_messages": 6 if mpi else 0,
            "mpi_scalars": 14 if mpi else 0,
            "cuda_aware_bytes": 0,
            "pinned_staging_bytes": 56 if mpi else 0,
            "pinned_device_to_host_bytes": 24 if mpi else 0,
            "pinned_host_to_device_bytes": 32 if mpi else 0,
        },
    }


def phase_record(*, mpi: bool, elapsed: float, timesteps: int = 10):
    names = ("curl", "update_eh", "polarization", "source", "boundary", "dft")
    cpu = {name: {"calls": 0, "points": 0} for name in names}
    cuda = {
        name: {
            "calls": 5 if name != "polarization" else 0,
            "points": 50 if name != "polarization" else 0,
        }
        for name in names
    }
    group_mapping = {
        "curl": ("dispatch", "curl"),
        "update_eh": ("field_updates", "update_eh"),
        "polarization": ("polarizations", "polarization"),
        "source": ("sources", "source"),
        "boundary": ("boundaries", "boundary"),
        "dft": ("dfts", "dft"),
    }
    statistics_delta = {}
    for name, (group, stem) in group_mapping.items():
        statistics_delta[group] = {
            f"{backend}_{stem}_{field}": values[name][field]
            for backend, values in (("cpu", cpu), ("cuda", cuda))
            for field in ("calls", "points")
        }
    statistics_delta["multi_gpu"] = {
        "mpi_messages": 3 if mpi else 0,
        "mpi_scalars": 7 if mpi else 0,
        "cuda_aware_bytes": 0,
        "pinned_staging_bytes": 28 if mpi else 0,
        "pinned_device_to_host_bytes": 12 if mpi else 0,
        "pinned_host_to_device_bytes": 16 if mpi else 0,
    }
    return {
        "elapsed_seconds": elapsed,
        "timesteps": timesteps,
        "end_time": 2.5,
        "backend_counters": {"cpu": cpu, "cuda": cuda},
        "statistics_delta": statistics_delta,
    }


def synthetic_result(root: pathlib.Path, world_size: int):
    runtime = runtime_closure(root)
    gradient = [0.25, -0.5]
    digest = canonical_float64_sha256(gradient)
    ranks = []
    inventory = [
        {
            "ordinal": index,
            "identifier": f"{index + 1:032x}",
            "name": "Synthetic GPU",
            "compute_capability": [8, 6],
            "multiprocessor_count": 84,
            "max_threads_per_block": 1024,
            "global_memory_bytes": 24 * 1024**3,
            "compatible": True,
        }
        for index in range(world_size)
    ]
    for rank_index in range(world_size):
        phase_times = {
            "update_design_seconds": 0.10 + 0.01 * rank_index,
            "forward_seconds": 1.00 + 0.02 * rank_index,
            "adjoint_seconds": 0.80 + 0.03 * rank_index,
            "gradient_seconds": 0.10 + 0.01 * rank_index,
        }
        phase_times["elapsed_seconds"] = sum(phase_times.values())
        phase_times["workload_wall_seconds"] = (
            phase_times["elapsed_seconds"] + 0.25
        )
        backend = {
            "requested": "cuda",
            "active": "cuda",
            "compiled": True,
            "runtime_available": True,
            "single_precision": True,
            "strict_cuda_marker": True,
            "selected_device": rank_index,
            "selected_device_identifier": f"{rank_index + 1:032x}",
            "devices": copy.deepcopy(inventory),
        }
        ranks.append(
            {
                "world_rank": rank_index,
                "backend": backend,
                "runtime": copy.deepcopy(runtime),
                "active_backend": "cuda",
                "selected_device": rank_index,
                "selected_device_identifier": f"{rank_index + 1:032x}",
                "process_environment": {
                    "PATH": "/synthetic",
                    "OMPI_COMM_WORLD_RANK": str(rank_index),
                    "OMPI_COMM_WORLD_SIZE": str(world_size),
                    "OMPI_COMM_WORLD_LOCAL_RANK": str(rank_index),
                    "OMPI_COMM_WORLD_LOCAL_SIZE": str(world_size),
                    "PMIX_HOSTNAME": "synthetic-node",
                    "PMIX_RANK": str(rank_index),
                },
                "backend_diagnostic": "",
                "objective": 1.5,
                "gradient": gradient,
                "gradient_sha256": digest,
                "phases": {
                    "update_design": {
                        "elapsed_seconds": phase_times["update_design_seconds"]
                    },
                    "forward": phase_record(
                        mpi=world_size > 1,
                        elapsed=phase_times["forward_seconds"],
                    ),
                    "adjoint": phase_record(
                        mpi=world_size > 1,
                        elapsed=phase_times["adjoint_seconds"],
                    ),
                    "gradient": {
                        "elapsed_seconds": phase_times["gradient_seconds"]
                    },
                },
                "final_statistics": statistics(mpi=world_size > 1),
                "timing": phase_times,
            }
        )
    maxima = {
        field: max(rank["timing"][field] for rank in ranks)
        for field in (
            "update_design_seconds",
            "forward_seconds",
            "adjoint_seconds",
            "gradient_seconds",
        )
    }
    top_timing = dict(maxima)
    top_timing["elapsed_seconds"] = sum(maxima.values())
    top_timing["workload_wall_seconds"] = max(
        rank["timing"]["workload_wall_seconds"] for rank in ranks
    )
    top_timing["rank_local_elapsed_seconds"] = ranks[0]["timing"][
        "elapsed_seconds"
    ]
    top_timing["rank_local_workload_wall_seconds"] = ranks[0]["timing"][
        "workload_wall_seconds"
    ]
    top_phases = {
        phase_name: {
            "elapsed_seconds": maxima[f"{phase_name}_seconds"]
        }
        for phase_name in ("update_design", "forward", "adjoint", "gradient")
    }
    return {
        "schema_version": 4,
        "backend": copy.deepcopy(ranks[0]["backend"]),
        "runtime": runtime,
        "workload": {
            "resolution": 10,
            "run_time": 2.0,
            "cell_size": 4.0,
            "design_resolution": 2,
            "design_shape": [1, 2],
            "design_variables": 2,
            "yee_cells": 1600,
            "weights_sha256": "a" * 64,
            "direction_sha256": "b" * 64,
            "material_gradient_path": "analytic-required",
            "phases": {
                "update_design": top_phases["update_design"],
                "forward": {
                    "timesteps": 10,
                    "end_time": 2.5,
                    **top_phases["forward"],
                },
                "adjoint": {
                    "timesteps": 10,
                    "end_time": 2.5,
                    **top_phases["adjoint"],
                },
                "gradient": top_phases["gradient"],
            },
            "final_statistics": copy.deepcopy(ranks[0]["final_statistics"]),
        },
        "result": {
            "objective": 1.5,
            "gradient": gradient,
            "gradient_shape": [2],
            "gradient_count": 2,
            "gradient_sha256": digest,
            "gradient_l2": 0.5590169943749475,
            "gradient_projection": math.fsum(
                value * direction_value
                for value, direction_value in zip(gradient, [0.0, 1.0])
            ),
            "gradient_sum": -0.25,
        },
        "timing": top_timing,
        "distributed": {
            "world_size": world_size,
            "timing_semantics": (
                "barrier-delimited maximum rank elapsed per phase"
            ),
            "selected_devices_unique": True,
            "objective_max_abs_difference": 0.0,
            "gradient_max_abs_difference": 0.0,
            "gradient_max_relative_l2_difference": 0.0,
            "gate": {
                "pass": True,
                "objective": True,
                "gradient": True,
                "strict_cuda_coverage": True,
            },
            "ranks": ranks,
        },
    }


class MpiAdjointComparatorTests(unittest.TestCase):
    def test_runtime_verification_cache_hashes_once_and_rechecks_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "runtime.so"
            path.write_bytes(b"first")
            record = {"path": str(path), "sha256": sha256_file(path)}
            cache = StatHashCache()
            original = gpmeep_provenance.sha256_file
            with mock.patch.object(
                gpmeep_provenance, "sha256_file", wraps=original
            ) as hashing:
                self.comparator._validate_runtime_record(
                    record, "runtime", hash_cache=cache
                )
                self.comparator._validate_runtime_record(
                    record, "runtime", hash_cache=cache
                )
                self.assertEqual(hashing.call_count, 1)
                path.write_bytes(b"other")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    self.comparator._validate_runtime_record(
                        record, "runtime", hash_cache=cache
                    )
                self.assertEqual(hashing.call_count, 2)

    @classmethod
    def setUpClass(cls):
        cls.comparator = load_comparator()

    def test_valid_record_requires_rank_runtime_uuid_coverage_and_max_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            validated = self.comparator.validate_distributed_record(
                result, expected_world_size=2, label="multi"
            )
            self.assertEqual(validated["device_identifiers"], [f"{1:032x}", f"{2:032x}"])
            self.assertEqual(validated["rank_coverage"][0]["cpu_fallback_calls"], 0)
            self.assertEqual(
                validated["elapsed_seconds"], result["timing"]["elapsed_seconds"]
            )
            self.assertEqual(
                validated["workload_wall_seconds"],
                result["timing"]["workload_wall_seconds"],
            )

    def test_workload_wall_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mutations = []

            for value in (True, math.nan, 0.0):
                result = synthetic_result(root, 2)
                result["timing"]["workload_wall_seconds"] = value
                mutations.append((f"top-level {value!r}", result))

            result = synthetic_result(root, 2)
            del result["timing"]["workload_wall_seconds"]
            mutations.append(("missing top-level", result))

            for value in (True, math.inf, -1.0):
                result = synthetic_result(root, 2)
                result["distributed"]["ranks"][0]["timing"][
                    "workload_wall_seconds"
                ] = value
                mutations.append((f"rank {value!r}", result))

            result = synthetic_result(root, 2)
            del result["distributed"]["ranks"][0]["timing"][
                "workload_wall_seconds"
            ]
            mutations.append(("missing rank", result))

            result = synthetic_result(root, 2)
            result["timing"]["workload_wall_seconds"] += 1.0
            mutations.append(("rank max mismatch", result))

            result = synthetic_result(root, 2)
            rank_zero_timing = result["distributed"]["ranks"][0]["timing"]
            rank_zero_timing["workload_wall_seconds"] = (
                rank_zero_timing["elapsed_seconds"] - 0.01
            )
            mutations.append(("shorter than rank elapsed", result))

            result = synthetic_result(root, 2)
            result["timing"]["rank_local_workload_wall_seconds"] += 0.1
            mutations.append(("rank-zero workload binding", result))

            result = synthetic_result(root, 2)
            result["timing"]["rank_local_elapsed_seconds"] += 0.1
            mutations.append(("rank-zero elapsed binding", result))

            result = synthetic_result(root, 2)
            result["distributed"]["ranks"][0]["timing"][
                "elapsed_seconds"
            ] += 0.1
            mutations.append(("rank phase-sum binding", result))

            for case, result in mutations:
                with self.subTest(case=case):
                    with self.assertRaises(RuntimeError):
                        self.comparator.validate_distributed_record(
                            result, expected_world_size=2, label="multi"
                        )

    def test_phase_backend_counter_view_is_bound_to_statistics_and_final_sum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            valid = synthetic_result(root, 2)
            self.assertEqual(
                self.comparator.validate_distributed_record(
                    valid, expected_world_size=2, label="multi"
                )["rank_coverage"][0]["phases"]["forward"]["dft"]["calls"],
                5,
            )
            changed = synthetic_result(root, 2)
            changed["distributed"]["ranks"][0]["phases"]["forward"][
                "backend_counters"
            ]["cuda"]["dft"] = {"calls": 123456789, "points": 987654321}
            with self.assertRaisesRegex(RuntimeError, "view/statistics disagree"):
                self.comparator.validate_distributed_record(
                    changed, expected_world_size=2, label="multi"
                )
            changed = synthetic_result(root, 2)
            changed["distributed"]["ranks"][0]["final_statistics"]["dfts"][
                "cuda_dft_calls"
            ] += 1
            changed["workload"]["final_statistics"] = copy.deepcopy(
                changed["distributed"]["ranks"][0]["final_statistics"]
            )
            with self.assertRaisesRegex(RuntimeError, "differ from phase sum"):
                self.comparator.validate_distributed_record(
                    changed, expected_world_size=2, label="multi"
                )

    def test_rank_local_single_gpu_inventories_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            for rank_index, rank in enumerate(result["distributed"]["ranks"]):
                device = copy.deepcopy(rank["backend"]["devices"][rank_index])
                device["ordinal"] = 0
                rank["backend"]["devices"] = [device]
                rank["backend"]["selected_device"] = 0
                rank["selected_device"] = 0
            validated = self.comparator.validate_distributed_record(
                result, expected_world_size=2, label="rank-local"
            )
            self.assertNotEqual(
                validated["rank_coverage"][0]["device_inventory_sha256"],
                validated["rank_coverage"][1]["device_inventory_sha256"],
            )

    def test_volatile_mpi_session_values_do_not_break_semantic_env_stability(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            first = self.comparator.validate_rank_cuda_coverage(
                result["distributed"]["ranks"][0], world_size=2, label="first"
            )
            changed = copy.deepcopy(result["distributed"]["ranks"][0])
            changed["process_environment"].update(
                {
                    "PMIX_NAMESPACE": "fresh-job-999",
                    "PMIX_SERVER_URI2": "fresh-job-999;tcp4://127.0.0.1:50000",
                    "OMPI_FILE_LOCATION": "/tmp/fresh-job-999/0",
                }
            )
            second = self.comparator.validate_rank_cuda_coverage(
                changed, world_size=2, label="second"
            )
            self.assertEqual(
                first["process_environment_sha256"],
                second["process_environment_sha256"],
            )
            self.assertNotEqual(
                first["process_environment_full_sha256"],
                second["process_environment_full_sha256"],
            )
            changed["process_environment"]["PMIX_HOSTNAME"] = "different-node"
            third = self.comparator.validate_rank_cuda_coverage(
                changed, world_size=2, label="third"
            )
            self.assertNotEqual(
                first["process_environment_sha256"],
                third["process_environment_sha256"],
            )

    def test_duplicate_physical_device_identifier_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            result["distributed"]["ranks"][1][
                "selected_device_identifier"
            ] = f"{1:032x}"
            result["distributed"]["ranks"][1]["backend"][
                "selected_device_identifier"
            ] = f"{1:032x}"
            result["distributed"]["selected_devices_unique"] = False
            with self.assertRaisesRegex(RuntimeError, "identifier|distinct physical"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_rank_zero_timing_cannot_masquerade_as_distributed_wall_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            result["timing"]["forward_seconds"] = result["distributed"][
                "ranks"
            ][0]["timing"]["forward_seconds"]
            with self.assertRaisesRegex(RuntimeError, "exact max-rank"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_cpu_fallback_and_missing_mpi_traffic_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            result["distributed"]["ranks"][0]["final_statistics"]["dispatch"][
                "cpu_curl_calls"
            ] = 1
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

            result = synthetic_result(pathlib.Path(temporary), 2)
            result["distributed"]["ranks"][0]["final_statistics"]["multi_gpu"][
                "mpi_messages"
            ] = 0
            with self.assertRaisesRegex(RuntimeError, "MPI GPU-halo traffic"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_phase_points_transport_and_absolute_dft_bound_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = synthetic_result(root, 2)
            result["distributed"]["ranks"][0]["phases"]["forward"][
                "backend_counters"
            ]["cuda"]["dft"]["points"] = 0
            with self.assertRaisesRegex(RuntimeError, "view/statistics disagree"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

            result = synthetic_result(root, 2)
            result["distributed"]["ranks"][0]["phases"]["forward"][
                "statistics_delta"
            ]["multi_gpu"]["cuda_aware_bytes"] = 28
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

            result = synthetic_result(root, 2)
            result["distributed"]["ranks"][0]["phases"]["adjoint"][
                "backend_counters"
            ]["cuda"]["dft"]["calls"] = 257
            result["distributed"]["ranks"][0]["phases"]["adjoint"][
                "statistics_delta"
            ]["dfts"]["cuda_dft_calls"] = 257
            result["distributed"]["ranks"][0]["final_statistics"]["dfts"][
                "cuda_dft_calls"
            ] = 262
            with self.assertRaisesRegex(RuntimeError, "fixed limit"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_python_package_manifest_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 1)
            package = pathlib.Path(result["runtime"]["python_package"]["root"])
            (package / "tampered.py").write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Python package changed"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=1, label="single"
                )

    def test_loaded_mpi_cuda_hdf5_library_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 1)
            library = pathlib.Path(
                result["runtime"]["loaded_libraries"]["cuda_runtime"][0]["path"]
            )
            library.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after measurement"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=1, label="single"
                )

    def test_receipt_runtime_allows_core_hdf5_subset_from_meep_only_process(self):
        core = ("/prefix/lib/libhdf5.so.310.5.1", "a" * 64)
        high_level = ("/prefix/lib/libhdf5_hl.so.310.0.6", "b" * 64)
        captured = [core, high_level]
        matches = self.comparator.measured_dependency_projection_matches

        self.assertTrue(matches("hdf5", captured, [core]))
        self.assertTrue(matches("hdf5", captured, captured))
        self.assertFalse(matches("hdf5", captured, [high_level]))
        self.assertFalse(matches("hdf5", captured, [core, core]))
        self.assertFalse(
            matches("hdf5", captured, [(core[0], "c" * 64)])
        )
        self.assertFalse(
            matches(
                "hdf5",
                [*captured, ("/prefix/lib/libz.so.1", "d" * 64)],
                [core],
            )
        )
        self.assertFalse(
            matches(
                "hdf5",
                [("/prefix/lib/libhdf5.so.evil", "e" * 64)],
                [("/prefix/lib/libhdf5.so.evil", "e" * 64)],
            )
        )
        self.assertFalse(
            matches(
                "hdf5",
                [("/prefix/lib/libhdf5.so.١", "f" * 64)],
                [("/prefix/lib/libhdf5.so.١", "f" * 64)],
            )
        )
        self.assertFalse(matches("hdf5", [core, core], [core]))
        self.assertFalse(matches("mpi", captured, [core]))

    def test_loaded_library_origins_are_isolated_and_driver_is_system_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix"
            runtime = runtime_closure(prefix)
            prefix.chmod(0o755)
            pathlib.Path(
                runtime["loaded_libraries"]["cuda_driver"][0]["path"]
            ).chmod(0o444)
            with mock.patch.object(
                self.comparator, "TRUSTED_CUDA_DRIVER_ROOTS", (prefix,)
            ):
                self.assertEqual(
                    self.comparator.validate_loaded_library_origins(runtime, prefix)[
                        "environment_prefix"
                    ],
                    str(prefix.resolve()),
                )
                outside = root / "outside" / "libmpi.so"
                outside.parent.mkdir()
                outside.write_text("outside\n", encoding="utf-8")
                runtime["loaded_libraries"]["mpi"][0]["path"] = str(outside)
                with self.assertRaisesRegex(RuntimeError, "outside the isolated"):
                    self.comparator.validate_loaded_library_origins(runtime, prefix)

    def test_rank_local_driver_uuid_and_executable_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            trusted = root / "trusted"
            trusted.mkdir()
            trusted.chmod(0o755)
            executable = trusted / "nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            identifier = f"{1:032x}"
            evidence = {
                "hostname": "node-a",
                "executable": {
                    "path": str(executable),
                    "size_bytes": executable.stat().st_size,
                    "sha256": sha256_file(executable),
                },
                "inventory": {
                    "command": [str(executable), "--query-gpu=uuid"],
                    "exit_code": 0,
                    "stdout": (
                        "0, GPU-00000000-0000-0000-0000-000000000001, "
                        "GPU, 1, 8.6, 1 MiB"
                    ),
                    "stderr": "",
                },
                "topology": {
                    "command": [str(executable), "-L"],
                    "exit_code": 0,
                    "stdout": (
                        "GPU 0: GPU (UUID: "
                        "GPU-00000000-0000-0000-0000-000000000001)"
                    ),
                    "stderr": "",
                },
            }
            with mock.patch.object(
                self.comparator, "TRUSTED_NVIDIA_SMI_ROOTS", (trusted,)
            ):
                self.assertTrue(
                    self.comparator.validate_rank_nvidia_driver(
                        evidence, identifier, "rank0"
                    )["selected_identifier"]
                    == identifier
                )
                tampered = copy.deepcopy(evidence)
                tampered["executable"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "path/hash"):
                    self.comparator.validate_rank_nvidia_driver(
                        tampered, identifier, "rank0"
                    )

    def test_stale_gradient_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 1)
            result["result"]["gradient_sum"] = 0.0
            with self.assertRaisesRegex(RuntimeError, "gradient_sum summary"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=1, label="single"
                )

    def test_qualification_binding_rejects_nonce_and_producer_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            producer = pathlib.Path(temporary) / "producer.py"
            producer.write_text("print('producer')\n", encoding="utf-8")
            digest = sha256_file(producer)
            qualification = {
                "mode": "qualification",
                "profile": self.comparator.QUALIFICATION_PROFILE,
                "run_nonce": "nonce",
                "run_id": "run-id",
                "build_receipt_id": "receipt",
                "source_snapshot_sha256": "snapshot",
                "producer_sha256": digest,
                "sample_kind": "measured",
                "sample_lane": "cuda-single",
                "sample_iteration": 0,
            }
            result = {
                "qualification": qualification,
                "producer": {
                    "path": str(producer),
                    "sha256": digest,
                    "command": ["python", str(producer)],
                    "environment": {},
                    "started_at_utc": "2026-08-04T00:00:00Z",
                    "completed_at_utc": "2026-08-04T00:00:01Z",
                },
            }
            validated = self.comparator.validate_qualification_binding(
                result,
                expected_nonce="nonce",
                expected_run_id="run-id",
                expected_receipt_id="receipt",
                expected_source_snapshot_sha256="snapshot",
                expected_producer_sha256=digest,
                expected_sample_kind="measured",
                expected_sample_lane="cuda-single",
                expected_sample_iteration=0,
                label="sample",
            )
            self.assertEqual(validated, qualification)
            with self.assertRaisesRegex(RuntimeError, "stale or forged"):
                self.comparator.validate_qualification_binding(
                    result,
                    expected_nonce="different",
                    expected_run_id="run-id",
                    expected_receipt_id="receipt",
                    expected_source_snapshot_sha256="snapshot",
                    expected_producer_sha256=digest,
                    expected_sample_kind="measured",
                    expected_sample_lane="cuda-single",
                    expected_sample_iteration=0,
                    label="sample",
                )
            changed_run = copy.deepcopy(result)
            changed_run["qualification"]["run_id"] = "old-run"
            with self.assertRaisesRegex(RuntimeError, "stale or forged"):
                self.comparator.validate_qualification_binding(
                    changed_run,
                    expected_nonce="nonce",
                    expected_run_id="run-id",
                    expected_receipt_id="receipt",
                    expected_source_snapshot_sha256="snapshot",
                    expected_producer_sha256=digest,
                    expected_sample_kind="measured",
                    expected_sample_lane="cuda-single",
                    expected_sample_iteration=0,
                    label="sample",
                )
            producer.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after measurement"):
                self.comparator.validate_qualification_binding(
                    result,
                    expected_nonce="nonce",
                    expected_run_id="run-id",
                    expected_receipt_id="receipt",
                    expected_source_snapshot_sha256="snapshot",
                    expected_producer_sha256=digest,
                    expected_sample_kind="measured",
                    expected_sample_lane="cuda-single",
                    expected_sample_iteration=0,
                    label="sample",
                )

    def test_workload_comparison_includes_gradient_path_and_timesteps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            single = synthetic_result(root / "single", 1)
            multi = synthetic_result(root / "multi", 2)
            self.assertTrue(self.comparator.compare_workloads(single, multi)["pass"])
            multi["workload"]["material_gradient_path"] = "legacy"
            self.assertFalse(self.comparator.compare_workloads(single, multi)["pass"])

    def test_failure_lifecycle_never_leaves_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "COMPLETE").write_text("stale", encoding="utf-8")
            self.comparator.prepare_output(output, "new")
            self.assertFalse((output / "COMPLETE").exists())
            self.comparator.mark_failed(output, "new", RuntimeError("slow"))
            self.assertFalse((output / "COMPLETE").exists())
            failure = json.loads((output / "FAILED.json").read_text())
            self.assertEqual(failure["state"], "FAILED")

    def test_publish_rejects_failed_performance_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            report = {
                "state": "COMPLETE",
                "run_id": "bad",
                "gate": {"pass": False, "performance": False},
            }
            with self.assertRaisesRegex(RuntimeError, "cannot publish COMPLETE"):
                self.comparator.publish_complete(output, "bad", report, "bad\n")
            self.assertFalse((output / "COMPLETE").exists())

    def test_standalone_cli_can_never_publish_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            exit_code = self.comparator.main(
                [
                    "--single-report",
                    str(output / "single.json"),
                    "--multi-report",
                    str(output / "multi.json"),
                    "--build-receipt",
                    str(output / "receipt.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 2)
            self.assertFalse((output / "COMPLETE").exists())
            self.assertTrue((output / "FAILED.json").is_file())

    def test_transport_byte_direction_and_cross_rank_closure_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = synthetic_result(root, 2)
            result["distributed"]["ranks"][0]["final_statistics"]["multi_gpu"][
                "pinned_staging_bytes"
            ] = 24
            with self.assertRaisesRegex(RuntimeError, "MPI GPU-halo traffic"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

            result = synthetic_result(root, 2)
            rank = result["distributed"]["ranks"][1]
            final = rank["final_statistics"]["multi_gpu"]
            final.update(
                {
                    "cuda_aware_bytes": 56,
                    "pinned_staging_bytes": 0,
                    "pinned_device_to_host_bytes": 0,
                    "pinned_host_to_device_bytes": 0,
                }
            )
            for phase in rank["phases"].values():
                if "statistics_delta" not in phase:
                    continue
                transport = phase["statistics_delta"]["multi_gpu"]
                transport.update(
                    {
                        "cuda_aware_bytes": 28,
                        "pinned_staging_bytes": 0,
                        "pinned_device_to_host_bytes": 0,
                        "pinned_host_to_device_bytes": 0,
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "different MPI"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_phase_schema_and_aggregate_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = synthetic_result(root, 2)
            del result["distributed"]["ranks"][0]["phases"]["forward"][
                "backend_counters"
            ]["cpu"]["source"]
            with self.assertRaisesRegex(RuntimeError, "counter schema"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

            result = synthetic_result(root, 2)
            for rank in result["distributed"]["ranks"]:
                rank["phases"]["forward"]["backend_counters"]["cuda"][
                    "source"
                ] = {"calls": 0, "points": 0}
                rank["phases"]["forward"]["statistics_delta"]["sources"].update(
                    {"cuda_source_calls": 0, "cuda_source_points": 0}
                )
                rank["final_statistics"]["sources"].update(
                    {"cuda_source_calls": 5, "cuda_source_points": 50}
                )
            result["workload"]["final_statistics"] = copy.deepcopy(
                result["distributed"]["ranks"][0]["final_statistics"]
            )
            with self.assertRaisesRegex(RuntimeError, "aggregate CUDA source"):
                self.comparator.validate_distributed_record(
                    result, expected_world_size=2, label="multi"
                )

    def test_unselected_incompatible_device_is_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_result(pathlib.Path(temporary), 2)
            for rank in result["distributed"]["ranks"]:
                rank["backend"]["devices"].append(
                    {
                        "ordinal": 2,
                        "identifier": f"{3:032x}",
                        "name": "Old GPU",
                        "compute_capability": [3, 0],
                        "multiprocessor_count": 1,
                        "max_threads_per_block": 256,
                        "global_memory_bytes": 1024,
                        "compatible": False,
                    }
                )
            validated = self.comparator.validate_distributed_record(
                result, expected_world_size=2, label="multi"
            )
            self.assertEqual(len(validated["device_identifiers"]), 2)


if __name__ == "__main__":
    unittest.main()
