from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gpmeep_benchmark_evidence import canonical_float64_sha256, sha256_file  # noqa: E402


def load_runner():
    path = SCRIPTS / "run-cpu-parallel-benchmark.py"
    name = "gpmeep_cpu_parallel_runner_test"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import CPU parallel benchmark runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lscpu_rows():
    return {
        "cpus": [
            {
                "cpu": cpu,
                "core": cpu % 8,
                "socket": 0,
                "node": 0,
                "online": True,
            }
            for cpu in range(16)
        ]
    }


def topology(runner):
    return runner.parse_lscpu_extended(lscpu_rows(), set(range(16)))


def phase_counters(active: str | None):
    names = ("curl", "update_eh", "polarization", "source", "boundary", "dft")
    return {
        backend: {
            name: {
                "calls": 2 if backend == active and name != "polarization" else 0,
                "points": 100 if backend == active and name != "polarization" else 0,
            }
            for name in names
        }
        for backend in ("cpu", "cuda")
    }


def statistics_record(calls: int = 4, points: int = 200, timesteps: int = 0):
    record = {
        "boundaries": {
            "cpu_boundary_calls": calls,
            "cpu_boundary_points": points,
            "cuda_boundary_calls": 0,
            "cuda_boundary_points": 0,
        },
        "boundary_eh_overlap": {
            "boundary_eh_overlap_checks": 0,
            "boundary_eh_overlap_eligible": 0,
            "boundary_eh_overlap_launched_e": 0,
            "boundary_eh_overlap_launched_h": 0,
            "boundary_eh_overlap_rejected": 0,
            "boundary_eh_overlap_skipped_cold_topology": 0,
            "boundary_eh_overlap_skipped_disabled": 0,
            "boundary_eh_overlap_skipped_no_remote": 0,
            "boundary_eh_overlap_skipped_unsupported_schedule": 0,
        },
        "dfts": {
            "cpu_dft_calls": calls,
            "cpu_dft_points": points,
            "cuda_dft_calls": 0,
            "cuda_dft_points": 0,
        },
        "ldos": {
            "cpu_ldos_reduction_calls": 0,
            "cpu_ldos_source_points": 0,
            "cuda_ldos_reduction_calls": 0,
            "cuda_ldos_submitted_profiles": 0,
            "cuda_ldos_source_points": 0,
            "cuda_ldos_descriptor_uploads": 0,
            "cuda_ldos_kernel_launches": 0,
            "cuda_ldos_result_device_to_host_bytes": 0,
            "ldos_full_field_device_to_host_bytes_avoided": 0,
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
            "halo_curl_overlap_checks": 0,
            "halo_curl_overlap_eligible": 0,
            "halo_curl_overlap_full_points": 0,
            "halo_curl_overlap_interior_points": 0,
            "halo_curl_overlap_launches": 0,
            "halo_curl_overlap_rejected_feature": 0,
            "halo_curl_overlap_rejected_small": 0,
            "halo_curl_overlap_shell_points": 0,
            "halo_curl_overlap_skipped_cold_topology": 0,
            "halo_curl_overlap_skipped_disabled": 0,
            "halo_curl_overlap_skipped_no_remote": 0,
            "halo_curl_overlap_skipped_unsupported_schedule": 0,
        },
        "multi_gpu": {
            "cuda_aware_bytes": 0,
            "mpi_messages": 0,
            "mpi_scalars": 0,
            "pinned_device_to_host_bytes": 0,
            "pinned_host_to_device_bytes": 0,
            "pinned_staging_bytes": 0,
        },
        "mpi_completion": {
            "mpi_waitall_executions": 0,
            "mpi_waitsome_executions": 0,
        },
        "polarizations": {
            "cpu_polarization_calls": 0,
            "cpu_polarization_points": 0,
            "cuda_polarization_calls": 0,
            "cuda_polarization_points": 0,
        },
        "resident": {
            "device_buffer_allocations": 0,
            "device_buffer_reuses": 0,
            "device_to_host_bytes_avoided": 0,
            "host_to_device_bytes_avoided": 0,
            "live_resident_device_buffers": 0,
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
    record["boundary_eh_overlap"]["boundary_eh_overlap_checks"] = 2 * timesteps
    record["boundary_eh_overlap"][
        "boundary_eh_overlap_skipped_disabled"
    ] = 2 * timesteps
    record["halo_curl_overlap"]["halo_curl_overlap_checks"] = timesteps
    record["halo_curl_overlap"]["halo_curl_overlap_skipped_disabled"] = timesteps
    return record


def config_affinities(config):
    # Synthetic evidence must exercise the exact policy contract, including
    # hwloc's core-major hardware-thread rank order.
    return [list(logical_cpus) for logical_cpus in config.rank_logical_plan]


def affinity_evidence(logical_cpus, thread_id):
    # The Python/master thread may be pinned to only its first OpenMP place;
    # the per-process thread union is the authoritative hybrid rank cpuset.
    master = logical_cpus[:1] if len(logical_cpus) == 1 else [logical_cpus[0], logical_cpus[-1]]
    cores = {}
    for cpu in logical_cpus:
        cores.setdefault(cpu % 8, []).append(cpu)
    threads = [
        {"thread_id": thread_id + index, "logical_cpus": sorted(cpus)}
        for index, (_core, cpus) in enumerate(sorted(cores.items()))
    ]
    return {
        "cpu_affinity": master,
        "completion_thread_cpu_affinity": master,
        "process_thread_cpu_affinities": {
            "thread_count": len(threads),
            "logical_cpu_union": logical_cpus,
            "threads": threads,
        },
    }


def gradient_payload():
    gradient = [0.01 + index * 1.0e-6 for index in range(1681)]
    raw_direction = [math.sin(index * 0.61803398875) for index in range(len(gradient))]
    direction_norm = math.sqrt(math.fsum(value * value for value in raw_direction))
    projection = math.fsum(
        value * direction_value / direction_norm
        for value, direction_value in zip(gradient, raw_direction)
    )
    return {
        "objective": 1.25,
        "gradient": gradient,
        "gradient_shape": [41, 41],
        "gradient_count": len(gradient),
        "gradient_sha256": canonical_float64_sha256(gradient),
        "gradient_l2": math.sqrt(math.fsum(value * value for value in gradient)),
        "gradient_projection": projection,
        "gradient_sum": math.fsum(gradient),
    }


def synthetic_result(runner, config):
    payload = gradient_payload()
    backend = {
        "requested": "cpu",
        "active": "cpu",
        "compiled": True,
        "runtime_available": True,
        "single_precision": True,
        "strict_cuda_marker": False,
    }
    rank_affinities = config_affinities(config)
    ranks = []
    for rank, logical_cpus in enumerate(rank_affinities):
        phase_times = {
            "update_design_seconds": 1.0 + 0.1 * rank,
            "forward_seconds": 2.0 + 0.1 * rank,
            "adjoint_seconds": 3.0 + 0.1 * rank,
            "gradient_seconds": 4.0 + 0.1 * rank,
        }
        affinity = affinity_evidence(logical_cpus, 1000 + rank)
        ranks.append(
            {
                "world_rank": rank,
                "hostname": "test-host",
                **affinity,
                "backend": copy.deepcopy(backend),
                "active_backend": "cpu",
                "runtime": {"synthetic": True},
                "objective": payload["objective"],
                "gradient": list(payload["gradient"]),
                "gradient_sha256": payload["gradient_sha256"],
                "final_statistics": statistics_record(
                    timesteps=runner.PROFILE["forward_timesteps"]
                    + runner.PROFILE["adjoint_timesteps"]
                ),
                "phases": {
                    name: {
                        **(
                            {
                                "timesteps": runner.PROFILE[f"{name}_timesteps"],
                                "end_time": runner.PROFILE[f"{name}_end_time"],
                            }
                            if name in ("forward", "adjoint")
                            else {}
                        ),
                        "elapsed_seconds": phase_times[f"{name}_seconds"],
                        "backend_counters": phase_counters(
                            "cpu" if name in ("forward", "adjoint") else None
                        ),
                        "statistics_delta": statistics_record(
                            2 if name in ("forward", "adjoint") else 0,
                            100 if name in ("forward", "adjoint") else 0,
                            runner.PROFILE[f"{name}_timesteps"]
                            if name in ("forward", "adjoint")
                            else 0,
                        ),
                    }
                    for name in ("update_design", "forward", "adjoint", "gradient")
                },
                "timing": {
                    **phase_times,
                    "elapsed_seconds": sum(phase_times.values()),
                    "workload_wall_seconds": 12.0 + rank,
                },
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
    producer_affinity = {
        name: copy.deepcopy(ranks[0][name])
        for name in (
            "cpu_affinity",
            "completion_thread_cpu_affinity",
            "process_thread_cpu_affinities",
        )
    }
    return {
        "schema_version": 4,
        "producer": producer_affinity,
        "backend": copy.deepcopy(backend),
        "runtime": {"synthetic": True},
        "workload": {
            "resolution": 64,
            "run_time": 40.0,
            "cell_size": 16.0,
            "design_resolution": 20,
            "design_shape": [41, 41],
            "design_variables": 1681,
            "yee_cells": 1048576,
            "weights_sha256": runner.PROFILE["weights_sha256"],
            "direction_sha256": runner.PROFILE["direction_sha256"],
            "material_gradient_path": "analytic-required",
            "final_statistics": copy.deepcopy(ranks[0]["final_statistics"]),
            "phases": {
                "update_design": {
                    "elapsed_seconds": maxima["update_design_seconds"],
                    "backend_counters": copy.deepcopy(
                        ranks[0]["phases"]["update_design"]["backend_counters"]
                    ),
                    "statistics_delta": copy.deepcopy(
                        ranks[0]["phases"]["update_design"]["statistics_delta"]
                    ),
                },
                "forward": {
                    "timesteps": 6614,
                    "end_time": 51.671875,
                    "elapsed_seconds": maxima["forward_seconds"],
                    "backend_counters": copy.deepcopy(
                        ranks[0]["phases"]["forward"]["backend_counters"]
                    ),
                    "statistics_delta": copy.deepcopy(
                        ranks[0]["phases"]["forward"]["statistics_delta"]
                    ),
                },
                "adjoint": {
                    "timesteps": 19840,
                    "end_time": 155.0,
                    "elapsed_seconds": maxima["adjoint_seconds"],
                    "backend_counters": copy.deepcopy(
                        ranks[0]["phases"]["adjoint"]["backend_counters"]
                    ),
                    "statistics_delta": copy.deepcopy(
                        ranks[0]["phases"]["adjoint"]["statistics_delta"]
                    ),
                },
                "gradient": {
                    "elapsed_seconds": maxima["gradient_seconds"],
                    "backend_counters": copy.deepcopy(
                        ranks[0]["phases"]["gradient"]["backend_counters"]
                    ),
                    "statistics_delta": copy.deepcopy(
                        ranks[0]["phases"]["gradient"]["statistics_delta"]
                    ),
                },
            },
        },
        "result": payload,
        "timing": {
            **maxima,
            "elapsed_seconds": sum(maxima.values()),
            "workload_wall_seconds": max(
                rank["timing"]["workload_wall_seconds"] for rank in ranks
            ),
            "rank_local_workload_wall_seconds": ranks[0]["timing"]["workload_wall_seconds"],
            "rank_local_elapsed_seconds": ranks[0]["timing"]["elapsed_seconds"],
        },
        "distributed": {
            "world_size": config.ranks,
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


def selection_validation(workload_wall):
    elapsed = workload_wall * 0.5
    return {
        "pass": True,
        "workload_wall_seconds": workload_wall,
        "elapsed_seconds": elapsed,
        "phase_seconds": {
            "update_design": elapsed * 0.1,
            "forward": elapsed * 0.3,
            "adjoint": elapsed * 0.5,
            "gradient": elapsed * 0.1,
        },
    }


class TopologyAndMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_fixed_profile_and_exact_matrix(self):
        self.runner._assert_fixed_profile()
        self.assertGreaterEqual(self.runner.PROFILE["repeats"], 5)
        self.assertLessEqual(self.runner.PROFILE["maximum_workload_cv"], 0.05)
        self.assertEqual(
            [config.config_id for config in self.runner.CONFIGS],
            ["1x1", "2x1", "4x1", "8x1", "16x1", "1x8", "2x4", "4x2"],
        )
        schedule = self.runner.sample_schedule()
        self.assertEqual(len(schedule), 48)
        for config in self.runner.CONFIGS:
            self.assertEqual(sum(row[0] == config.config_id for row in schedule), 6)
        self.assertEqual(schedule[8][0], "1x1")
        self.assertEqual(schedule[16][0], "8x1")
        self.assertEqual(schedule[24][0], "2x4")

    def test_legacy_smt_plan_matches_hwloc_core_major_rank_order(self):
        config = self.runner._config_by_id("16x1")
        self.assertEqual(
            config.rank_logical_plan,
            [[0], [8], [1], [9], [2], [10], [3], [11],
             [4], [12], [5], [13], [6], [14], [7], [15]],
        )
        self.assertEqual(
            config.rank_core_plan,
            [[[0, 0, core]] for core in range(8) for _ in range(2)],
        )

    def test_parses_complete_eight_core_smt_topology(self):
        parsed = topology(self.runner)
        self.assertEqual(parsed["physical_core_count"], 8)
        self.assertEqual(parsed["logical_cpu_count"], 16)
        self.assertEqual(parsed["cores"][0]["logical_cpus"], [0, 8])

    def test_accepts_partial_cpuset_and_rejects_duplicate_topology(self):
        partial = self.runner.parse_lscpu_extended(lscpu_rows(), set(range(15)))
        self.assertEqual(partial["logical_cpu_count"], 15)
        self.assertEqual(partial["physical_core_count"], 8)
        self.assertEqual(partial["allowed_threads_per_core"], [1, 2])
        with self.assertRaisesRegex(RuntimeError, "host-specific-interim"):
            self.runner._validate_legacy_interim_topology(partial)
        bad = lscpu_rows()
        bad["cpus"][15]["cpu"] = 14
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.runner.parse_lscpu_extended(bad, set(range(16)))

    def test_power_and_load_state_is_captured_and_policy_drift_rejected(self):
        def fake_record(path):
            path = pathlib.Path(path)
            if path == pathlib.Path("/proc/loadavg"):
                text = "0.25 0.50 0.75 1/100 123\n"
            elif path.name.endswith("throttle_count"):
                text = "0\n"
            else:
                text = path.name + "\n"
            return {
                "path": str(path),
                "resolved_path": str(path),
                "status": "ok",
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }

        with mock.patch.object(
            self.runner, "_read_system_text_record", side_effect=fake_record
        ) as reader:
            start = self.runner.capture_power_load_state([0, 1])
        self.assertGreater(reader.call_count, 10)
        self.assertEqual(start["loadavg"]["load_1m"], 0.25)
        end = copy.deepcopy(start)
        current = end["per_cpu"]["0"]["cpufreq"]["scaling_cur_freq"]
        current["text"] = "999999\n"
        current["sha256"] = hashlib.sha256(current["text"].encode()).hexdigest()
        end["loadavg"]["load_1m"] = 0.5
        self.assertTrue(
            self.runner.validate_power_load_comparability(start, end)["pass"]
        )
        drifted = copy.deepcopy(end)
        drifted["per_cpu"]["0"]["cpufreq"]["scaling_governor"]["text"] = "performance\n"
        with self.assertRaisesRegex(RuntimeError, "governor"):
            self.runner.validate_power_load_comparability(start, drifted)
        throttled = copy.deepcopy(end)
        throttled["per_cpu"]["0"]["thermal_throttle"]["core_throttle_count"]["text"] = "1\n"
        with self.assertRaisesRegex(RuntimeError, "thermal-throttle"):
            self.runner.validate_power_load_comparability(start, throttled)


class CommandEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_exact_binding_argv_for_every_configuration(self):
        runtime = {"mpiexec": pathlib.Path("/prefix/bin/mpiexec")}
        producer = ["/prefix/bin/python", "/repo/scripts/benchmark-adjoint.py"]
        for config in self.runner.CONFIGS:
            argv = self.runner.launch_arguments(
                runtime=runtime, config=config, producer_argv=producer
            )
            self.assertEqual(
                argv[:3],
                ["/prefix/bin/mpiexec", "--nooversubscribe", "--report-bindings"],
            )
            self.assertNotIn("--host", argv)
            if config.binding == "hwthread":
                self.assertIn("--use-hwthread-cpus", argv)
                self.assertEqual(argv[argv.index("--bind-to") + 1], "hwthread")
            else:
                self.assertNotIn("--use-hwthread-cpus", argv)
                self.assertEqual(argv[argv.index("--bind-to") + 1], "core")
            if config.binding == "hwthread":
                expected_mapping = "hwthread"
            elif config.omp_threads == 1:
                expected_mapping = "core"
            else:
                expected_mapping = f"ppr:{config.ranks}:node:PE={config.omp_threads}"
            self.assertEqual(argv[argv.index("--map-by") + 1], expected_mapping)
            self.assertEqual(argv[argv.index("-n") + 1], str(config.ranks))
            self.assertEqual(argv[-2:], producer)

    def test_cpu_raw_qualification_is_bound_to_controller_run_id(self):
        producer = pathlib.Path(self.runner.__file__).resolve().with_name(
            "benchmark-adjoint.py"
        )
        producer_sha256 = sha256_file(producer)
        run_id = "d" * 32
        qualification = {
            "mode": "qualification",
            "profile": self.runner.PRODUCER_QUALIFICATION_PROFILE_ID,
            "run_nonce": "c" * 32,
            "run_id": run_id,
            "build_receipt_id": "receipt",
            "source_snapshot_sha256": "a" * 64,
            "producer_sha256": producer_sha256,
            "sample_kind": "measured",
            "sample_lane": "cpu-parallel",
            "sample_iteration": 0,
        }
        result = {
            "qualification": qualification,
            "producer": {
                "path": str(producer),
                "sha256": producer_sha256,
                "started_at_utc": "2026-08-12T00:00:00Z",
                "completed_at_utc": "2026-08-12T00:00:01Z",
            },
        }
        self.assertEqual(
            self.runner.validate_qualification_binding(
                result,
                nonce="c" * 32,
                run_id=run_id,
                receipt_id="receipt",
                snapshot_sha256="a" * 64,
                producer_sha256=producer_sha256,
                sample_kind="measured",
                iteration=0,
                label="cpu-run",
            ),
            qualification,
        )
        stale = copy.deepcopy(result)
        stale["qualification"]["run_id"] = "e" * 32
        with self.assertRaisesRegex(RuntimeError, "stale or forged"):
            self.runner.validate_qualification_binding(
                stale,
                nonce="c" * 32,
                run_id=run_id,
                receipt_id="receipt",
                snapshot_sha256="a" * 64,
                producer_sha256=producer_sha256,
                sample_kind="measured",
                iteration=0,
                label="cpu-run",
            )

    def test_clean_controlled_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib" / "openmpi").mkdir(parents=True)
            (prefix / "lib" / "pmix").mkdir(parents=True)
            fontconfig = root / "qualification-fontconfig.conf"
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            runtime = {
                "python": prefix / "bin" / "python",
                "package_parent": prefix / "lib" / "python",
                "libmeep": prefix / "lib" / "libmeep.so",
                "qualification_fontconfig": fontconfig,
            }
            (root / "output").mkdir()
            environment = self.runner._child_environment(
                runtime,
                config=self.runner._config_by_id("2x4"),
                output=root / "output",
                run_id="0" * 32,
                sample_kind="warmup",
                iteration=0,
                nonce="1" * 32,
                prepare_xdg_cache=True,
            )
            self.assertEqual(environment["OMP_NUM_THREADS"], "4")
            self.assertEqual(environment["OMP_PLACES"], "cores")
            self.assertEqual(environment["OMP_PROC_BIND"], "close")
            self.assertEqual(environment["MEEP_GPU_BACKEND"], "cpu")
            self.assertEqual(environment["MEEP_MATERIAL_GRADIENT_PATH"], "analytic-required")
            self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(environment["FONTCONFIG_FILE"], str(fontconfig.resolve()))
            self.assertEqual(
                environment["XDG_CACHE_HOME"],
                str(
                    (
                        root
                        / "output"
                        / "cache"
                        / ("0" * 32)
                        / ("2x4-warmup-0-" + "1" * 32)
                    ).resolve()
                ),
            )
            self.assertNotIn("LD_PRELOAD", environment)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
            self.assertEqual(
                self.runner._require_exact_recomputed_launch_environment(
                    dict(sorted(environment.items())), environment, label="exact"
                ),
                dict(sorted(environment.items())),
            )
            forged_stored_and_raw_environment = dict(sorted(environment.items()))
            forged_stored_and_raw_environment["OMP_SCHEDULE"] = "dynamic"
            with self.assertRaisesRegex(RuntimeError, "fixed recomputation"):
                self.runner._require_exact_recomputed_launch_environment(
                    forged_stored_and_raw_environment,
                    environment,
                    label="forged",
                )

    def test_xdg_cache_is_fresh_and_symlink_confined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib" / "openmpi").mkdir(parents=True)
            (prefix / "lib" / "pmix").mkdir(parents=True)
            fontconfig = root / "qualification-fontconfig.conf"
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            runtime = {
                "python": prefix / "bin" / "python",
                "package_parent": prefix / "lib" / "python",
                "libmeep": prefix / "lib" / "libmeep.so",
                "qualification_fontconfig": fontconfig,
            }
            common = {
                "config": self.runner._config_by_id("2x4"),
                "run_id": "0" * 32,
                "sample_kind": "warmup",
                "iteration": 0,
                "nonce": "1" * 32,
            }
            output = root / "output"
            output.mkdir()
            environment = self.runner._child_environment(
                runtime,
                output=output,
                prepare_xdg_cache=True,
                **common,
            )
            cache = pathlib.Path(environment["XDG_CACHE_HOME"])
            self.assertEqual(
                cache,
                output
                / "cache"
                / ("0" * 32)
                / ("2x4-warmup-0-" + "1" * 32),
            )
            self.assertEqual(list(cache.iterdir()), [])
            with self.assertRaisesRegex(RuntimeError, "not fresh"):
                self.runner._child_environment(
                    runtime,
                    output=output,
                    prepare_xdg_cache=True,
                    **common,
                )
            retry = self.runner._child_environment(
                runtime,
                output=output,
                prepare_xdg_cache=True,
                **{**common, "nonce": "2" * 32},
            )
            retry_cache = pathlib.Path(retry["XDG_CACHE_HOME"])
            self.assertNotEqual(retry_cache, cache)
            self.assertEqual(list(retry_cache.iterdir()), [])
            self.assertEqual(
                self.runner._child_environment(
                    runtime,
                    output=output,
                    prepare_xdg_cache=False,
                    **common,
                )["XDG_CACHE_HOME"],
                str(cache),
            )

        for symlink_at_leaf in (False, True):
            with self.subTest(symlink_at_leaf=symlink_at_leaf), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                prefix = root / "prefix"
                (prefix / "bin").mkdir(parents=True)
                (prefix / "lib" / "openmpi").mkdir(parents=True)
                (prefix / "lib" / "pmix").mkdir(parents=True)
                fontconfig = root / "qualification-fontconfig.conf"
                fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
                runtime = {
                    "python": prefix / "bin" / "python",
                    "package_parent": prefix / "lib" / "python",
                    "libmeep": prefix / "lib" / "libmeep.so",
                    "qualification_fontconfig": fontconfig,
                }
                output = root / "output"
                output.mkdir()
                outside = root / "outside"
                outside.mkdir()
                if symlink_at_leaf:
                    cache_parent = output / "cache" / ("0" * 32)
                    cache_parent.mkdir(parents=True)
                    (
                        cache_parent / ("2x4-warmup-0-" + "1" * 32)
                    ).symlink_to(outside, target_is_directory=True)
                else:
                    (output / "cache").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    self.runner._child_environment(
                        runtime,
                        output=output,
                        prepare_xdg_cache=True,
                        **common,
                    )

    def test_receipt_bound_fontconfig_path_and_digest_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fontconfig = root / "qualification-fontconfig.conf"
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            receipt = {
                "configuration_files": {
                    "qualification_fontconfig": {
                        "path": str(fontconfig),
                        "size_bytes": fontconfig.stat().st_size,
                        "sha256": sha256_file(fontconfig),
                    }
                }
            }
            runtime = {"qualification_fontconfig": fontconfig}
            self.assertEqual(
                self.runner._qualification_fontconfig(receipt, root, runtime),
                fontconfig.resolve(),
            )
            wrong_runtime = {"qualification_fontconfig": root / "other.conf"}
            with self.assertRaisesRegex(RuntimeError, "paths disagree"):
                self.runner._qualification_fontconfig(receipt, root, wrong_runtime)
            fontconfig.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.runner._qualification_fontconfig(receipt, root, runtime)

    def test_probe_digests_are_raw_utf8_stream_hashes(self):
        completed = mock.Mock(returncode=0, stdout="plain output\n", stderr="warning\n")
        with mock.patch.object(self.runner.subprocess, "run", return_value=completed):
            record = self.runner._probe(["/usr/bin/lscpu", "--json"])
        self.assertEqual(
            record["stdout_sha256"],
            hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            record["stderr_sha256"],
            hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        )


class CommandExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_keyboard_interrupt_terminates_process_group_reaps_and_reraises(self):
        process = mock.Mock(pid=4321, returncode=-15)
        process.poll.return_value = None
        process.communicate.side_effect = [KeyboardInterrupt(), ("captured", "warning")]
        with mock.patch.object(
            self.runner.subprocess, "Popen", return_value=process
        ), mock.patch.object(self.runner.os, "killpg") as killpg:
            with self.assertRaises(KeyboardInterrupt):
                self.runner._run_command(
                    ["/prefix/bin/mpiexec", "-n", "2", "/prefix/bin/python"],
                    cwd=pathlib.Path("/repo"),
                    environment={"OMP_NUM_THREADS": "4"},
                    timeout_seconds=123,
                )
        self.assertEqual(
            process.communicate.call_args_list,
            [
                mock.call(timeout=123),
                mock.call(timeout=self.runner.PROFILE["timeout_kill_grace_seconds"]),
            ],
        )
        killpg.assert_called_once_with(4321, self.runner.signal.SIGTERM)

    def test_timeout_escalates_to_sigkill_and_preserves_captured_streams(self):
        process = mock.Mock(pid=8765, returncode=-9)
        process.poll.return_value = None
        process.communicate.side_effect = [
            self.runner.subprocess.TimeoutExpired(["mpiexec"], 123),
            self.runner.subprocess.TimeoutExpired(
                ["mpiexec"], self.runner.PROFILE["timeout_kill_grace_seconds"]
            ),
            ("captured stdout", "captured stderr"),
        ]
        with mock.patch.object(
            self.runner.subprocess, "Popen", return_value=process
        ), mock.patch.object(self.runner.os, "killpg") as killpg:
            with self.assertRaises(self.runner.CommandTimeout) as raised:
                self.runner._run_command(
                    ["/prefix/bin/mpiexec", "-n", "2", "/prefix/bin/python"],
                    cwd=pathlib.Path("/repo"),
                    environment={"OMP_NUM_THREADS": "4"},
                    timeout_seconds=123,
                )
        self.assertEqual(raised.exception.stdout, "captured stdout")
        self.assertEqual(raised.exception.stderr, "captured stderr")
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(8765, self.runner.signal.SIGTERM),
                mock.call(8765, self.runner.signal.SIGKILL),
            ],
        )
        self.assertEqual(process.communicate.call_args_list[-1], mock.call())

    def test_cleanup_signals_group_even_if_mpiexec_leader_already_exited(self):
        process = mock.Mock(pid=2468, returncode=0)
        process.poll.return_value = 0
        process.communicate.return_value = ("", "")
        with mock.patch.object(self.runner.os, "killpg") as killpg:
            self.runner._terminate_process_group(process)
        killpg.assert_called_once_with(2468, self.runner.signal.SIGTERM)
        process.communicate.assert_called_once_with(
            timeout=self.runner.PROFILE["timeout_kill_grace_seconds"]
        )


class CpuRecordGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.topology = topology(cls.runner)

    @staticmethod
    def runtime_validator(_record, _label):
        return {"synthetic": True, "loaded_libraries": {}}

    def test_accepts_world_backend_counters_timing_and_hybrid_affinity(self):
        for config_id in ("2x1", "16x1", "1x8", "2x4", "4x2"):
            config = self.runner._config_by_id(config_id)
            result = synthetic_result(self.runner, config)
            validated = self.runner.validate_cpu_record(
                result,
                config,
                self.topology,
                label=config_id,
                runtime_validator=self.runtime_validator,
            )
            self.assertEqual(validated["affinity"]["used_logical_cpus"], sorted({cpu for affinity in config_affinities(config) for cpu in affinity}))
            self.assertEqual(validated["workload_wall_seconds"], 12.0 + config.ranks - 1)

    def test_rejects_numeric_cpu_id_order_for_smt_ranks(self):
        config = self.runner._config_by_id("16x1")
        result = synthetic_result(self.runner, config)
        for rank, record in enumerate(result["distributed"]["ranks"]):
            record.update(affinity_evidence([rank], 5000 + rank))
        with self.assertRaisesRegex(RuntimeError, r"16x1\.rank1 differs"):
            self.runner.validate_cpu_record(
                result,
                config,
                self.topology,
                label="16x1",
                runtime_validator=self.runtime_validator,
            )

    def test_accepts_domain_decomposition_sparse_source_and_dft_ranks(self):
        config = self.runner._config_by_id("8x1")
        result = synthetic_result(self.runner, config)
        for rank_index in (0, 2, 4, 6):
            rank = result["distributed"]["ranks"][rank_index]
            rank["final_statistics"]["sources"]["cpu_source_calls"] = 0
            rank["final_statistics"]["sources"]["cpu_source_points"] = 0
            rank["final_statistics"]["dfts"]["cpu_dft_calls"] = 0
            rank["final_statistics"]["dfts"]["cpu_dft_points"] = 0
            for phase_name in ("forward", "adjoint"):
                cpu_phase = rank["phases"][phase_name]["backend_counters"]["cpu"]
                cpu_phase["source"] = {"calls": 0, "points": 0}
                cpu_phase["dft"] = {"calls": 0, "points": 0}
                statistics = rank["phases"][phase_name]["statistics_delta"]
                statistics["sources"]["cpu_source_calls"] = 0
                statistics["sources"]["cpu_source_points"] = 0
                statistics["dfts"]["cpu_dft_calls"] = 0
                statistics["dfts"]["cpu_dft_points"] = 0
        result["workload"]["final_statistics"] = copy.deepcopy(
            result["distributed"]["ranks"][0]["final_statistics"]
        )
        for phase_name in self.runner.PHASES:
            for field in ("backend_counters", "statistics_delta"):
                result["workload"]["phases"][phase_name][field] = copy.deepcopy(
                    result["distributed"]["ranks"][0]["phases"][phase_name][field]
                )
        validated = self.runner.validate_cpu_record(
            result,
            config,
            self.topology,
            label="sparse-distributed",
            runtime_validator=self.runtime_validator,
        )
        self.assertTrue(validated["pass"])

    def test_rejects_missing_aggregate_or_inconsistent_sparse_coverage(self):
        config = self.runner._config_by_id("8x1")
        missing_final = synthetic_result(self.runner, config)
        for rank in missing_final["distributed"]["ranks"]:
            rank["final_statistics"]["dfts"]["cpu_dft_calls"] = 0
            rank["final_statistics"]["dfts"]["cpu_dft_points"] = 0
            for phase_name in ("forward", "adjoint"):
                rank["phases"][phase_name]["backend_counters"]["cpu"]["dft"] = {
                    "calls": 0,
                    "points": 0,
                }
                statistics = rank["phases"][phase_name]["statistics_delta"]["dfts"]
                statistics["cpu_dft_calls"] = 0
                statistics["cpu_dft_points"] = 0
        missing_final["workload"]["final_statistics"] = copy.deepcopy(
            missing_final["distributed"]["ranks"][0]["final_statistics"]
        )
        for phase_name in self.runner.PHASES:
            for field in ("backend_counters", "statistics_delta"):
                missing_final["workload"]["phases"][phase_name][field] = copy.deepcopy(
                    missing_final["distributed"]["ranks"][0]["phases"][phase_name][field]
                )
        with self.assertRaisesRegex(RuntimeError, "aggregate CPU cpu_dft_calls"):
            self.runner.validate_cpu_record(
                missing_final,
                config,
                self.topology,
                label="missing-final-dft",
                runtime_validator=self.runtime_validator,
            )

        missing_phase = synthetic_result(self.runner, config)
        for rank in missing_phase["distributed"]["ranks"]:
            rank["phases"]["forward"]["backend_counters"]["cpu"]["source"] = {
                "calls": 0,
                "points": 0,
            }
            statistics = rank["phases"]["forward"]["statistics_delta"]["sources"]
            statistics["cpu_source_calls"] = 0
            statistics["cpu_source_points"] = 0
            rank["final_statistics"]["sources"]["cpu_source_calls"] = 2
            rank["final_statistics"]["sources"]["cpu_source_points"] = 100
        missing_phase["workload"]["final_statistics"] = copy.deepcopy(
            missing_phase["distributed"]["ranks"][0]["final_statistics"]
        )
        for phase_name in self.runner.PHASES:
            for field in ("backend_counters", "statistics_delta"):
                missing_phase["workload"]["phases"][phase_name][field] = copy.deepcopy(
                    missing_phase["distributed"]["ranks"][0]["phases"][phase_name][field]
                )
        with self.assertRaisesRegex(RuntimeError, "forward has no aggregate CPU source"):
            self.runner.validate_cpu_record(
                missing_phase,
                config,
                self.topology,
                label="missing-phase-source",
                runtime_validator=self.runtime_validator,
            )

        inconsistent = synthetic_result(self.runner, config)
        inconsistent["distributed"]["ranks"][0]["phases"]["forward"][
            "backend_counters"
        ]["cpu"]["dft"] = {"calls": 0, "points": 100}
        inconsistent["distributed"]["ranks"][0]["phases"]["forward"][
            "statistics_delta"
        ]["dfts"]["cpu_dft_calls"] = 0
        with self.assertRaisesRegex(RuntimeError, "calls/points disagree"):
            self.runner.validate_cpu_record(
                inconsistent,
                config,
                self.topology,
                label="inconsistent-sparse",
                runtime_validator=self.runtime_validator,
            )

    def test_rejects_world_backend_counter_affinity_and_wall_tampering(self):
        config = self.runner._config_by_id("2x4")
        mutations = []
        world = synthetic_result(self.runner, config)
        world["distributed"]["world_size"] = 1
        mutations.append(world)
        backend = synthetic_result(self.runner, config)
        backend["distributed"]["ranks"][0]["backend"]["active"] = "cuda"
        mutations.append(backend)
        counter = synthetic_result(self.runner, config)
        counter["distributed"]["ranks"][0]["final_statistics"]["dispatch"]["cuda_curl_calls"] = 1
        mutations.append(counter)
        ldos = synthetic_result(self.runner, config)
        ldos["distributed"]["ranks"][0]["final_statistics"]["ldos"][
            "cuda_ldos_reduction_calls"
        ] = 1
        mutations.append(ldos)
        affinity = synthetic_result(self.runner, config)
        affinity["distributed"]["ranks"][1]["process_thread_cpu_affinities"]["logical_cpu_union"] = [0, 8]
        mutations.append(affinity)
        timing = synthetic_result(self.runner, config)
        timing["timing"]["workload_wall_seconds"] += 1
        mutations.append(timing)
        too_short = synthetic_result(self.runner, config)
        rank = too_short["distributed"]["ranks"][0]
        rank["timing"]["workload_wall_seconds"] = (
            rank["timing"]["elapsed_seconds"] - 0.1
        )
        too_short["timing"]["rank_local_workload_wall_seconds"] = rank["timing"][
            "workload_wall_seconds"
        ]
        mutations.append(too_short)
        for result in mutations:
            with self.assertRaises(RuntimeError):
                self.runner.validate_cpu_record(
                    result,
                    config,
                    self.topology,
                    label="tampered",
                    runtime_validator=self.runtime_validator,
                )

    def test_rejects_counter_accounting_top_level_and_gpu_transfer_tampering(self):
        config = self.runner._config_by_id("2x1")

        accounting = synthetic_result(self.runner, config)
        accounting["distributed"]["ranks"][1]["final_statistics"]["dispatch"][
            "cpu_curl_calls"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "exact phase sum"):
            self.runner.validate_cpu_record(
                accounting,
                config,
                self.topology,
                label="accounting",
                runtime_validator=self.runtime_validator,
            )

        top = synthetic_result(self.runner, config)
        top["workload"]["final_statistics"]["dispatch"]["cpu_curl_calls"] += 1
        with self.assertRaisesRegex(RuntimeError, "top-level statistics"):
            self.runner.validate_cpu_record(
                top,
                config,
                self.topology,
                label="top",
                runtime_validator=self.runtime_validator,
            )

        transfer = synthetic_result(self.runner, config)
        rank = transfer["distributed"]["ranks"][1]
        rank["phases"]["forward"]["statistics_delta"]["dispatch"][
            "host_to_device_bytes"
        ] = 4
        rank["final_statistics"]["dispatch"]["host_to_device_bytes"] = 4
        with self.assertRaisesRegex(RuntimeError, "GPU/dispersive work"):
            self.runner.validate_cpu_record(
                transfer,
                config,
                self.topology,
                label="transfer",
                runtime_validator=self.runtime_validator,
            )

        non_fdtd = synthetic_result(self.runner, config)
        rank = non_fdtd["distributed"]["ranks"][1]
        rank["phases"]["gradient"]["statistics_delta"]["dispatch"][
            "cpu_curl_calls"
        ] = 1
        rank["phases"]["gradient"]["statistics_delta"]["dispatch"][
            "cpu_curl_points"
        ] = 1
        rank["phases"]["gradient"]["backend_counters"]["cpu"]["curl"] = {
            "calls": 1,
            "points": 1,
        }
        rank["final_statistics"]["dispatch"]["cpu_curl_calls"] += 1
        rank["final_statistics"]["dispatch"]["cpu_curl_points"] += 1
        with self.assertRaisesRegex(RuntimeError, "unexpectedly recorded FDTD work"):
            self.runner.validate_cpu_record(
                non_fdtd,
                config,
                self.topology,
                label="non-fdtd",
                runtime_validator=self.runtime_validator,
            )

    def test_accepts_only_disabled_overlap_and_zero_gpu_control_statistics(self):
        valid = statistics_record()
        valid["boundary_eh_overlap"]["boundary_eh_overlap_checks"] = 8
        valid["boundary_eh_overlap"]["boundary_eh_overlap_skipped_disabled"] = 8
        valid["halo_curl_overlap"]["halo_curl_overlap_checks"] = 4
        valid["halo_curl_overlap"]["halo_curl_overlap_skipped_disabled"] = 4
        flattened = self.runner._validate_statistics_record(valid, label="valid")
        self.assertEqual(
            flattened["boundary_eh_overlap.boundary_eh_overlap_checks"], 8
        )
        self.assertEqual(flattened["halo_curl_overlap.halo_curl_overlap_checks"], 4)

        missing = copy.deepcopy(valid)
        missing.pop("runtime")
        with self.assertRaisesRegex(RuntimeError, "statistics schema is incomplete"):
            self.runner._validate_statistics_record(missing, label="missing")

        for group, field in (
            ("ldos", "ldos_full_field_device_to_host_bytes_avoided"),
            ("mpi_completion", "mpi_waitsome_executions"),
            ("runtime", "runtime_availability_probes"),
            ("tile_coalescing", "tile_coalesced_curl_chunk_phases"),
        ):
            tampered = copy.deepcopy(valid)
            tampered[group][field] = 1
            with self.assertRaisesRegex(RuntimeError, "GPU/dispersive work"):
                self.runner._validate_statistics_record(tampered, label=group)

        for group, field in (
            ("boundary_eh_overlap", "boundary_eh_overlap_launched_e"),
            ("halo_curl_overlap", "halo_curl_overlap_launches"),
        ):
            tampered = copy.deepcopy(valid)
            tampered[group][field] = 1
            with self.assertRaisesRegex(RuntimeError, "GPU overlap work"):
                self.runner._validate_statistics_record(tampered, label=group)

        mismatched = copy.deepcopy(valid)
        mismatched["halo_curl_overlap"]["halo_curl_overlap_skipped_disabled"] -= 1
        with self.assertRaisesRegex(RuntimeError, "GPU overlap work"):
            self.runner._validate_statistics_record(mismatched, label="mismatch")

    def test_rejects_coherent_gpu_ldos_and_overlap_timestep_tampering(self):
        config = self.runner._config_by_id("1x1")

        ldos = synthetic_result(self.runner, config)
        ldos_rank = ldos["distributed"]["ranks"][0]
        ldos_field = "ldos_full_field_device_to_host_bytes_avoided"
        ldos_rank["phases"]["forward"]["statistics_delta"]["ldos"][ldos_field] = 1
        ldos_rank["final_statistics"]["ldos"][ldos_field] = 1
        ldos["workload"]["phases"]["forward"]["statistics_delta"]["ldos"][
            ldos_field
        ] = 1
        ldos["workload"]["final_statistics"]["ldos"][ldos_field] = 1
        with self.assertRaisesRegex(RuntimeError, "GPU/dispersive work"):
            self.runner.validate_cpu_record(
                ldos,
                config,
                self.topology,
                label="ldos",
                runtime_validator=self.runtime_validator,
            )

        for mode in ("undersized", "zero"):
            with self.subTest(mode=mode):
                overlap = synthetic_result(self.runner, config)
                overlap_rank = overlap["distributed"]["ranks"][0]
                for phase_name in ("forward", "adjoint"):
                    timesteps = (
                        1
                        if mode == "undersized" and phase_name == "forward"
                        else 0
                    )
                    statistics = overlap_rank["phases"][phase_name][
                        "statistics_delta"
                    ]
                    statistics["boundary_eh_overlap"][
                        "boundary_eh_overlap_checks"
                    ] = 2 * timesteps
                    statistics["boundary_eh_overlap"][
                        "boundary_eh_overlap_skipped_disabled"
                    ] = 2 * timesteps
                    statistics["halo_curl_overlap"]["halo_curl_overlap_checks"] = (
                        timesteps
                    )
                    statistics["halo_curl_overlap"][
                        "halo_curl_overlap_skipped_disabled"
                    ] = timesteps
                    overlap["workload"]["phases"][phase_name][
                        "statistics_delta"
                    ] = copy.deepcopy(statistics)
                for group, fields in (
                    (
                        "boundary_eh_overlap",
                        (
                            "boundary_eh_overlap_checks",
                            "boundary_eh_overlap_skipped_disabled",
                        ),
                    ),
                    (
                        "halo_curl_overlap",
                        (
                            "halo_curl_overlap_checks",
                            "halo_curl_overlap_skipped_disabled",
                        ),
                    ),
                ):
                    for field in fields:
                        total = sum(
                            overlap_rank["phases"][phase]["statistics_delta"][group][
                                field
                            ]
                            for phase in (
                                "update_design",
                                "forward",
                                "adjoint",
                                "gradient",
                            )
                        )
                        overlap_rank["final_statistics"][group][field] = total
                        overlap["workload"]["final_statistics"][group][field] = total
                with self.assertRaisesRegex(
                    RuntimeError, "overlap checks do not match fixed timesteps"
                ):
                    self.runner.validate_cpu_record(
                        overlap,
                        config,
                        self.topology,
                        label=mode,
                        runtime_validator=self.runtime_validator,
                    )

    def test_hybrid_rejects_single_wide_or_extra_spanning_thread(self):
        config = self.runner._config_by_id("1x8")
        for mode in ("single-wide", "extra-spanning"):
            result = synthetic_result(self.runner, config)
            rank = result["distributed"]["ranks"][0]
            evidence = rank["process_thread_cpu_affinities"]
            if mode == "single-wide":
                evidence["threads"] = [
                    {"thread_id": 9999, "logical_cpus": list(range(16))}
                ]
            else:
                evidence["threads"].append(
                    {"thread_id": 9999, "logical_cpus": list(range(16))}
                )
            evidence["thread_count"] = len(evidence["threads"])
            result["producer"]["process_thread_cpu_affinities"] = copy.deepcopy(
                evidence
            )
            with self.assertRaisesRegex(RuntimeError, "exact OpenMP"):
                self.runner.validate_cpu_record(
                    result,
                    config,
                    self.topology,
                    label=mode,
                    runtime_validator=self.runtime_validator,
                )

    def test_full_gradient_and_objective_comparison_rejects_late_element(self):
        config = self.runner._config_by_id("1x1")
        reference = synthetic_result(self.runner, config)
        candidate = copy.deepcopy(reference)
        self.assertTrue(self.runner.compare_to_reference(reference, candidate, label="same")["pass"])
        candidate["result"]["gradient"][-1] += 1.0
        candidate["result"]["gradient_sha256"] = canonical_float64_sha256(candidate["result"]["gradient"])
        with self.assertRaisesRegex(RuntimeError, "full-gradient"):
            self.runner.compare_to_reference(reference, candidate, label="changed")


class SelectionAndTamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_median_selects_fastest_parallel_configuration(self):
        medians = {"1x1": 20, "2x1": 12, "4x1": 8, "8x1": 6, "16x1": 7, "1x8": 5, "2x4": 4, "4x2": 5}
        samples = []
        for config in self.runner.CONFIGS:
            for iteration, delta in enumerate((-0.1, -0.05, 0, 0.05, 0.1)):
                samples.append(
                    {
                        "config_id": config.config_id,
                        "sample_kind": "measured",
                        "iteration": iteration,
                        "validation": selection_validation(
                            medians[config.config_id] + delta
                        ),
                        "correctness": {"pass": True},
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                        "process_seconds": medians[config.config_id] + delta + 1,
                    }
                )
        selection = self.runner.select_fastest(samples)
        self.assertEqual(selection["selected_config_id"], "2x4")
        self.assertEqual(selection["selected"]["workload_wall_median_seconds"], 4)
        self.assertEqual(selection["headline_lane"], "physical")
        self.assertEqual(
            selection["canonical_full_physical"]["config_id"], "1x8"
        )
        self.assertEqual(
            selection["fastest_passing_physical"]["config_id"], "2x4"
        )
        self.assertEqual(selection["single_core_reference"]["config_id"], "1x1")
        self.assertIn("coefficient_of_variation", selection["selected"]["workload_wall_statistics"])
        self.assertEqual(set(selection["selected"]["phase_statistics"]), set(self.runner.PHASES))
        markdown = self.runner._markdown_timing_table(selection)
        self.assertIn("| Config | MPI | OMP |", markdown)
        for config in self.runner.CONFIGS:
            self.assertIn(f"| {config.config_id} |", markdown)

    def test_single_core_is_diagnostic_only_even_when_fastest(self):
        samples = []
        for config in self.runner.CONFIGS:
            wall = 1.0 if config.config_id == "1x1" else 10.0 + config.ranks
            for iteration in range(self.runner.PROFILE["repeats"]):
                samples.append(
                    {
                        "config_id": config.config_id,
                        "sample_kind": "measured",
                        "iteration": iteration,
                        "validation": selection_validation(wall),
                        "correctness": {"pass": True},
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                    }
                )
        selection = self.runner.select_fastest(samples)
        self.assertNotEqual(selection["selected_config_id"], "1x1")
        self.assertFalse(selection["single_core_reference"]["selection_eligible"])

    def test_smt_lane_is_never_a_physical_headline(self):
        samples = []
        for config in self.runner.CONFIGS:
            wall = 0.5 if config.lane == "smt" else 10.0 + config.ranks
            for iteration in range(self.runner.PROFILE["repeats"]):
                samples.append(
                    {
                        "config_id": config.config_id,
                        "sample_kind": "measured",
                        "iteration": iteration,
                        "validation": selection_validation(wall),
                        "correctness": {"pass": True},
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                    }
                )
        selection = self.runner.select_fastest(samples)
        self.assertEqual(selection["fastest_passing_smt"]["config_id"], "16x1")
        self.assertEqual(selection["fastest_passing_physical"]["lane"], "physical")
        self.assertNotEqual(selection["selected_config_id"], "16x1")

    def test_failed_fast_configuration_is_never_selected(self):
        samples = []
        for config in self.runner.CONFIGS:
            for iteration in range(self.runner.PROFILE["repeats"]):
                sample = {
                    "config_id": config.config_id,
                    "sample_kind": "measured",
                    "iteration": iteration,
                    "validation": selection_validation(10.0),
                    "correctness": {"pass": True},
                    "receipt_gate": {"pass": True},
                    "process_environment_gate": {"pass": True},
                    "material_gradient_gate": {"pass": True},
                    "process_seconds": 11.0,
                }
                if config.config_id == "2x4":
                    sample["validation"] = selection_validation(1.0)
                    sample["correctness"]["pass"] = False
                samples.append(sample)
        selection = self.runner.select_fastest(samples)
        self.assertNotEqual(selection["selected_config_id"], "2x4")
        rejected = next(
            item for item in selection["configurations"] if item["config_id"] == "2x4"
        )
        self.assertFalse(rejected["eligible"])
        self.assertIn("correctness", rejected["failed_gates"])

    def test_unstable_fast_looking_samples_fail_fixed_cv_gate(self):
        samples = []
        for config in self.runner.CONFIGS:
            walls = (
                (1.0, 1.0, 1.0, 1.0, 100.0)
                if config.config_id == "2x4"
                else (10.0,) * self.runner.PROFILE["repeats"]
            )
            for iteration, wall in enumerate(walls):
                samples.append(
                    {
                        "config_id": config.config_id,
                        "sample_kind": "measured",
                        "iteration": iteration,
                        "validation": selection_validation(wall),
                        "correctness": {"pass": True},
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                    }
                )
        with self.assertRaisesRegex(RuntimeError, "CV threshold"):
            self.runner.select_fastest(samples)

    def test_publication_recomputes_selection_and_rejects_tampering(self):
        medians = {config.config_id: 20.0 - index for index, config in enumerate(self.runner.CONFIGS)}
        samples = []
        for config_id, sample_kind, iteration in self.runner.sample_schedule():
            config = self.runner._config_by_id(config_id)
            signature_record = {
                "world_size": config.ranks,
                "final": {"synthetic.calls": config.ranks},
                "phases": {},
            }
            validation = selection_validation(medians[config_id])
            validation["aggregate_counter_signature"] = {
                "record": signature_record,
                "sha256": self.runner.canonical_sha256(signature_record),
            }
            sample = {
                "config_id": config_id,
                "sample_kind": sample_kind,
                "iteration": iteration,
                "nonce": f"{config_id}-{sample_kind}-{iteration}",
                "validation": validation,
                "correctness": {"pass": True},
                "receipt_gate": {"pass": True},
                "process_environment_gate": {"pass": True},
                "material_gradient_gate": {"pass": True},
                "process_seconds": medians[config_id] + 1,
            }
            samples.append(sample)
        matrix = self.runner.validate_sample_matrix(samples)
        counter_drift = copy.deepcopy(samples)
        drifted_sample = next(
            sample
            for sample in counter_drift
            if sample["config_id"] == "1x8" and sample["sample_kind"] == "measured"
        )
        drifted_record = drifted_sample["validation"]["aggregate_counter_signature"][
            "record"
        ]
        drifted_record["final"]["synthetic.calls"] += 1
        drifted_sample["validation"]["aggregate_counter_signature"][
            "sha256"
        ] = self.runner.canonical_sha256(drifted_record)
        with self.assertRaisesRegex(RuntimeError, "changed for world size 1"):
            self.runner.validate_sample_matrix(counter_drift)
        selection = self.runner.select_fastest(samples)
        checkpoint_closure = {
            "pass": True,
            "record": {"entry_count": len(samples), "head_sha256": "a" * 64},
            "sha256": "b" * 64,
        }
        resume_closure = {
            "pass": True,
            "record": {"event_count": 0, "head_sha256": None},
            "sha256": "c" * 64,
        }
        recovery_closure = {
            "pass": True,
            "record": {"event_count": 0, "head_sha256": None, "pending_orphan_count": 0},
            "sha256": "d" * 64,
        }
        report = {
            "state": "COMPLETE",
            "profile": dict(self.runner.PROFILE),
            "sample_matrix": matrix,
            "selection": selection,
            "integrity_reverification": {
                "pass": True,
                "sample_matrix": matrix,
                "selection": selection,
                "checkpoint_closure": checkpoint_closure,
                "resume_closure": resume_closure,
                "recovery_closure": recovery_closure,
            },
            "checkpoint_closure": checkpoint_closure,
            "resume_closure": resume_closure,
            "recovery_closure": recovery_closure,
            "gate": {
                "pass": True,
                "receipt_and_source": True,
                "topology": True,
                "sample_matrix": True,
                "world_backend_counters_affinity": True,
                "objective_and_full_gradient": True,
                "raw_evidence_integrity": True,
                "durable_checkpoint_journal": True,
                "durable_recovery_journal": True,
                "selection": True,
            },
        }
        self.assertTrue(
            self.runner.validate_publication_report(
                report,
                samples,
                authoritative_recomputation=report["integrity_reverification"],
            )["pass"]
        )
        tampered = copy.deepcopy(report)
        tampered["selection"]["selected_config_id"] = "1x1"
        with self.assertRaisesRegex(RuntimeError, "recomputation"):
            self.runner.validate_publication_report(
                tampered,
                samples,
                authoritative_recomputation=report["integrity_reverification"],
            )

    def test_forged_stored_wall_time_differs_from_raw_recomputation(self):
        raw_recomputed = {
            "validation": selection_validation(10.0),
            "correctness": {"pass": True},
            "receipt_gate": {"pass": True},
            "process_environment_gate": {"pass": True},
            "material_gradient_gate": {"pass": True},
        }
        stored = copy.deepcopy(raw_recomputed)
        stored["validation"]["workload_wall_seconds"] = 0.1
        with self.assertRaisesRegex(RuntimeError, "raw recomputation"):
            self.runner._assert_stored_gates_match_raw_recomputation(
                stored, raw_recomputed, label="forged"
            )

    def test_process_wall_cannot_be_shorter_than_raw_workload_wall(self):
        validation = selection_validation(10.0)
        self.assertEqual(
            self.runner._validate_process_wall_seconds(10.0, validation, label="ok"),
            10.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shorter"):
            self.runner._validate_process_wall_seconds(9.9, validation, label="bad")

    def test_raw_json_and_log_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = root / "raw.json"
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            result.write_text(json.dumps({"value": 1}), encoding="utf-8")
            stdout.write_text("stdout\n", encoding="utf-8")
            stderr.write_text("stderr\n", encoding="utf-8")
            sample = {
                "config_id": "1x1",
                "sample_kind": "measured",
                "iteration": 0,
                "nonce": "nonce",
                "result": {"value": 1},
                "result_file": str(result),
                "result_sha256": sha256_file(result),
                "stdout_log": str(stdout),
                "stdout_sha256": sha256_file(stdout),
                "stderr_log": str(stderr),
                "stderr_sha256": sha256_file(stderr),
            }
            self.assertEqual(len(self.runner.reverify_raw_files([sample])), 1)
            stderr.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.runner.reverify_raw_files([sample])


class VerificationHashCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_execute_shares_one_cache_for_source_receipt_and_contract_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            snapshot = {"sha256": "a" * 64}
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "source_end": snapshot,
            }
            args = types.SimpleNamespace(
                host_specific_interim=True,
                build_receipt=root / "receipt.json",
                output=root / "output",
                resume=False,
            )
            with mock.patch.object(
                self.runner, "source_snapshot", return_value=snapshot
            ) as source, mock.patch.object(
                self.runner, "verify_build_receipt", return_value=receipt
            ) as receipt_verifier, mock.patch.object(
                self.runner._MPI_RUNNER,
                "validate_build_receipt_contract",
                return_value={},
            ) as contract, mock.patch.object(
                self.runner._MPI_RUNNER, "_receipt_runtime", return_value={}
            ), mock.patch.object(
                self.runner,
                "_qualification_fontconfig",
                side_effect=RuntimeError("stop after cache-bound preflight"),
            ) as fontconfig:
                with self.assertRaisesRegex(RuntimeError, "cache-bound preflight"):
                    self.runner.execute(args, root)
            cache = source.call_args.kwargs["hash_cache"]
            self.assertIsInstance(cache, self.runner.StatHashCache)
            self.assertIs(receipt_verifier.call_args.kwargs["hash_cache"], cache)
            self.assertIs(contract.call_args.kwargs["hash_cache"], cache)
            self.assertIs(fontconfig.call_args.kwargs["hash_cache"], cache)

    def test_sample_runtime_closure_receives_epoch_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            config = self.runner._config_by_id("1x1")
            run_id = "1" * 32
            nonce = "2" * 32
            result_path, stdout_path, stderr_path = self.runner._sample_artifact_paths(
                output, run_id, config.config_id, "measured", 0
            )
            result = {"producer": {"command": ["producer"]}}
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            stdout_path.write_text(
                self.runner.PREFIX
                + json.dumps(
                    {
                        "result_file": str(result_path.resolve()),
                        "sha256": sha256_file(result_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text("\n", encoding="utf-8")
            cache = self.runner.StatHashCache()
            validation = {"pass": True, "workload_wall_seconds": 1.0}
            with mock.patch.object(
                self.runner, "producer_arguments", return_value=["producer"]
            ), mock.patch.object(
                self.runner, "validate_qualification_binding", return_value={}
            ), mock.patch.object(
                self.runner, "validate_cpu_record", return_value=validation
            ), mock.patch.object(
                self.runner._MPI_RUNNER,
                "_validate_receipt_rank_runtimes",
                return_value={"pass": True},
            ) as runtime_gate, mock.patch.object(
                self.runner, "_child_environment", return_value={}
            ), mock.patch.object(
                self.runner,
                "_validate_process_environments",
                return_value={"pass": True},
            ), mock.patch.object(
                self.runner, "_material_gradient_gate", return_value={"pass": True}
            ), mock.patch.object(
                self.runner, "launch_arguments", return_value=["launch"]
            ):
                sample = self.runner._validate_sample_artifacts(
                    repo=output,
                    output=output,
                    runtime={},
                    receipt={"receipt_id": "receipt"},
                    snapshot_sha256="3" * 64,
                    producer_sha256="4" * 64,
                    topology={},
                    run_id=run_id,
                    config=config,
                    sample_kind="measured",
                    iteration=0,
                    nonce=nonce,
                    process_seconds=2.0,
                    hash_cache=cache,
                )
            self.assertTrue(sample["receipt_gate"]["pass"])
            self.assertIs(runtime_gate.call_args.kwargs["hash_cache"], cache)

    def test_stat_change_rehashes_and_concurrent_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "evidence.bin"
            path.write_bytes(b"original")
            cache = self.runner.StatHashCache()
            provenance_hasher = self.runner._PROVENANCE.sha256_file
            with mock.patch.object(
                self.runner._PROVENANCE,
                "sha256_file",
                wraps=provenance_hasher,
            ) as hasher:
                first = self.runner._regular_evidence_record(
                    path, hash_cache=cache
                )["sha256"]
                self.assertEqual(
                    self.runner._regular_evidence_record(path, hash_cache=cache)[
                        "sha256"
                    ],
                    first,
                )
                self.assertEqual(hasher.call_count, 1)
                path.write_bytes(b"changed!")
                second = self.runner._regular_evidence_record(
                    path, hash_cache=cache
                )["sha256"]
                self.assertNotEqual(second, first)
                self.assertEqual(hasher.call_count, 2)

            class MutatingCache(self.runner.StatHashCache):
                mutate_during_lookup = False

                def _validate_open_file(
                    self, lexical, descriptor, fingerprint, context
                ):
                    if self.mutate_during_lookup and context == "during cache lookup":
                        lexical.write_bytes(b"raced!!!")
                    return super()._validate_open_file(
                        lexical, descriptor, fingerprint, context
                    )

            racing_cache = MutatingCache()
            self.runner._regular_evidence_record(path, hash_cache=racing_cache)
            racing_cache.mutate_during_lookup = True
            with self.assertRaisesRegex(RuntimeError, "changed during cache lookup"):
                self.runner._regular_evidence_record(path, hash_cache=racing_cache)


class DurableCheckpointJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def anchor(self, root, run_id="1" * 32):
        topology_record = {
            "normalized": topology(self.runner),
            "power_load_state_start": {},
        }
        record = {
            "run_id": run_id,
            "receipt": {"receipt_id": "receipt", "sha256": "a" * 64},
            "source_snapshot": {"sha256": "b" * 64},
            "producer": {"sha256": "c" * 64},
            "runner_provenance": {"sha256": "d" * 64},
            "topology": {"sha256": "e" * 64, "identity_sha256": "f" * 64},
            "boot": {"boot_id": "00000000-0000-0000-0000-000000000001"},
            "host": {"hostname": "test", "uname_nodename": "test"},
            "parent_affinity": list(range(16)),
            "power_policy": {"record": {}, "sha256": self.runner.canonical_sha256({})},
            "matrix": self.runner._matrix_binding(topology_record),
        }
        anchor_path = root / "checkpoint" / "run.json"
        if anchor_path.exists():
            evidence = {
                "path": str(anchor_path.resolve()),
                "sha256": sha256_file(anchor_path),
                "record": json.loads(anchor_path.read_text()),
            }
        else:
            evidence = self.runner._publish_run_anchor(root, record)
        return {
            **evidence,
            "topology_evidence": {"record": topology_record},
            "resume_power_continuity": {"pass": True},
            "resume_live_state": {
                "loadavg": {"load_1m": 0.1, "load_5m": 0.2, "load_15m": 0.3}
            },
        }

    def sample(self, root, anchor, index, reference=None):
        config_id, sample_kind, iteration = self.runner.sample_schedule()[index]
        config = self.runner._config_by_id(config_id)
        result_path, stdout_path, stderr_path = self.runner._sample_artifact_paths(
            root, anchor["record"]["run_id"], config_id, sample_kind, iteration
        )
        result_path.write_text(json.dumps({"index": index}) + "\n", encoding="utf-8")
        stdout_path.write_text(f"stdout {index}\n", encoding="utf-8")
        stderr_path.write_text(f"stderr {index}\n", encoding="utf-8")
        signature_record = {
            "world_size": config.ranks,
            "final": {"synthetic.calls": config.ranks},
            "phases": {},
        }
        validation = selection_validation(20.0 + index)
        validation["aggregate_counter_signature"] = {
            "record": signature_record,
            "sha256": self.runner.canonical_sha256(signature_record),
        }
        sample = {
            "config_id": config_id,
            "config": self.runner.asdict(config),
            "sample_kind": sample_kind,
            "iteration": iteration,
            "nonce": f"{index + 1:032x}",
            "process_seconds": 100.0 + index,
            "validation": validation,
            "receipt_gate": {"pass": True, "value": "receipt"},
            "process_environment_gate": {"pass": True, "value": "environment"},
            "material_gradient_gate": {"pass": True, "value": "gradient"},
            "result": synthetic_result(self.runner, config),
            "result_file": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "stdout_log": str(stdout_path.resolve()),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_log": str(stderr_path.resolve()),
            "stderr_sha256": sha256_file(stderr_path),
        }
        if index >= self.runner.REFERENCE_SCHEDULE_INDEX:
            reference_result = reference or sample["result"]
            sample["correctness"] = self.runner.compare_to_reference(
                reference_result, sample["result"], label=f"sample-{index}"
            )
        return sample

    def publish_prefix(self, root, count):
        anchor = self.anchor(root)
        samples = []
        closure = self.runner._empty_journal_closure()
        reference = None
        for index in range(count):
            sample = self.sample(root, anchor, index, reference)
            if index == self.runner.REFERENCE_SCHEDULE_INDEX:
                reference = sample["result"]
            entry = self.runner._make_checkpoint_entry(
                output=root,
                anchor=anchor,
                index=index,
                previous_sha256=closure["record"]["head_sha256"],
                sample=sample,
            )
            evidence = self.runner._publish_checkpoint_entry(root, entry)
            closure = self.runner._extend_journal_closure(closure, evidence)
            samples.append(sample)
        return anchor, samples, closure

    def publication_quartet(self, root, anchor):
        checkpoint = self.runner._structural_checkpoint_closure(root, anchor)
        self.assertTrue(checkpoint["record"]["complete"])
        resumes = self.runner._load_resume_events(root, anchor)
        failures = self.runner._load_failure_events(root, anchor)
        recoveries = self.runner._load_recovery_events(root, anchor)
        report_path = root / "report.json"
        markdown_path = root / "report.md"
        report_path.write_text('{"synthetic": true}\n', encoding="utf-8")
        markdown_path.write_text("synthetic report\n", encoding="utf-8")
        publishing = {
            "schema_version": 1,
            "state": "PUBLISHING",
            "run_id": anchor["record"]["run_id"],
            "profile_id": self.runner.PROFILE_ID,
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "report_markdown": str(markdown_path.resolve()),
            "report_markdown_sha256": sha256_file(markdown_path),
            "run_anchor": anchor["path"],
            "run_anchor_sha256": anchor["sha256"],
            "checkpoint_entry_count": checkpoint["record"]["entry_count"],
            "checkpoint_head_sha256": checkpoint["record"]["head_sha256"],
            "checkpoint_closure_sha256": checkpoint["sha256"],
            "resume_event_count": resumes["record"]["event_count"],
            "resume_closure_sha256": resumes["sha256"],
            "failure_event_count": failures["record"]["event_count"],
            "failure_closure_sha256": failures["sha256"],
            "recovery_event_count": recoveries["record"]["event_count"],
            "recovery_closure_sha256": recoveries["sha256"],
        }
        publishing_path = root / "PUBLISHING.json"
        self.runner._atomic_create_json(publishing_path, publishing)
        complete = {
            **publishing,
            "state": "COMPLETE",
            "publishing_marker": self.runner._regular_evidence_record(
                publishing_path,
                output=root,
                expected_parent=root,
                label="synthetic PUBLISHING marker",
            ),
        }
        self.runner._atomic_create_json(root / "COMPLETE", complete)
        self.runner.atomic_write_json(root / "state.json", complete)
        return complete

    def test_atomic_no_clobber_and_fresh_directory_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "immutable.json"
            self.runner._atomic_create_json(target, {"value": 1})
            self.assertEqual(target.stat().st_nlink, 1)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.runner._atomic_create_json(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text()), {"value": 1})
            output = root / "output"
            output.mkdir()
            (output / "stale.txt").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                self.runner._require_fresh_output(output)

    def test_atomic_anonymous_publication_failure_leaves_no_named_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "never-visible.json"
            with mock.patch.object(
                self.runner,
                "_link_anonymous_file_noreplace",
                side_effect=RuntimeError("injected link failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected link failure"):
                    self.runner._atomic_create_json(target, {"value": 1})
            self.assertFalse(target.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_lock_contention_fails_without_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.runner.output_lock(root):
                with self.assertRaisesRegex(RuntimeError, "another CPU benchmark writer"):
                    with self.runner.output_lock(root):
                        self.fail("second writer acquired the output lock")

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            external = root / "external-lock-target"
            external.write_text("external\n", encoding="utf-8")
            lock = root / self.runner.RUN_LOCK_FILE
            lock.symlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "safely open"):
                with self.runner.output_lock(root):
                    self.fail("symlink lock was accepted")
            self.assertEqual(external.read_text(encoding="utf-8"), "external\n")

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            external = root / "external-lock-target"
            external.write_text("external\n", encoding="utf-8")
            lock = root / self.runner.RUN_LOCK_FILE
            lock.hardlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "private single-link"):
                with self.runner.output_lock(root):
                    self.fail("hardlink lock was accepted")

    def test_checkpoint_entry_records_process_triad_provenance_and_deferred_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, samples, closure = self.publish_prefix(root, 1)
            entry_path = self.runner._entry_path(root, 0)
            entry = json.loads(entry_path.read_text())
            self.assertEqual(entry["previous_entry_sha256"], None)
            self.assertEqual(entry["sample"]["process_seconds"], 100.0)
            self.assertEqual(entry["provenance"], self.runner._checkpoint_provenance(anchor))
            self.assertEqual(
                entry["gates"]["correctness"],
                {
                    "status": "deferred",
                    "reference_schedule_index": self.runner.REFERENCE_SCHEDULE_INDEX,
                },
            )
            self.assertEqual(closure["record"]["entry_count"], 1)
            for record in entry["files"].values():
                self.assertEqual(record["sha256"], sha256_file(pathlib.Path(record["path"])))
                self.assertGreater(record["size_bytes"], 0)
            bad = copy.deepcopy(samples[0])
            bad.pop("process_seconds")
            with self.assertRaisesRegex(RuntimeError, "process_seconds"):
                self.runner._make_checkpoint_entry(
                    output=root,
                    anchor=anchor,
                    index=0,
                    previous_sha256=None,
                    sample=bad,
                )

    def test_sample_summarization_is_concrete_and_non_recursive(self):
        config = self.runner._config_by_id("1x1")
        raw = synthetic_result(self.runner, config)
        sample = {
            "config_id": "1x1",
            "sample_kind": "measured",
            "iteration": 0,
            "result": raw,
        }
        expected_gradient_sha256 = self.runner.canonical_float64_sha256(
            raw["result"]["gradient"]
        )
        with mock.patch.object(
            self.runner,
            "canonical_float64_sha256",
            wraps=self.runner.canonical_float64_sha256,
        ) as gradient_hasher:
            summarized = self.runner._summarize_samples([sample])
        self.assertEqual(gradient_hasher.call_count, 1)
        self.assertNotIn("result", summarized[0])
        self.assertEqual(summarized[0]["raw_result_schema"], raw["schema_version"])
        self.assertEqual(
            summarized[0]["raw_gradient_sha256"], expected_gradient_sha256
        )
        with self.assertRaisesRegex(RuntimeError, "without its raw result"):
            self.runner._summarize_samples([{"config_id": "1x1"}])

    def test_gap_order_duplicate_nonce_and_hash_chain_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            entries = root / "checkpoint" / "entries"
            entries.mkdir(parents=True)
            (entries / "000000.json").write_text("{}\n", encoding="utf-8")
            (entries / "000002.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "gap, duplicate"):
                self.runner._enumerate_indexed_json(entries, label="checkpoint entries")

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, samples, _closure = self.publish_prefix(root, 2)
            second_path = self.runner._entry_path(root, 1)
            second = json.loads(second_path.read_text())
            for mutation in ("order", "chain", "nonce"):
                changed = copy.deepcopy(second)
                if mutation == "order":
                    changed["index"] = 0
                elif mutation == "chain":
                    changed["previous_entry_sha256"] = "0" * 64
                else:
                    changed["sample"]["nonce"] = samples[0]["nonce"]
                second_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                with mock.patch.object(
                    self.runner,
                    "_validate_sample_artifacts",
                    side_effect=lambda **kwargs: samples[
                        self.runner.sample_schedule().index(
                            (kwargs["config"].config_id, kwargs["sample_kind"], kwargs["iteration"])
                        )
                    ],
                ):
                    with self.assertRaises(RuntimeError):
                        self.runner._load_checkpoint_entries(
                            repo=root,
                            output=root,
                            runtime={},
                            receipt={},
                            anchor=anchor,
                        )
                second_path.write_text(json.dumps(second) + "\n", encoding="utf-8")

    def test_partial_future_foreign_and_tampered_triad_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, _samples, _closure = self.publish_prefix(root, 1)
            result, stdout, stderr = self.runner._sample_artifact_paths(
                root, anchor["record"]["run_id"], *self.runner.sample_schedule()[1]
            )
            result.write_text("partial\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "future/uncheckpointed"):
                self.runner._scan_sample_artifacts(root, anchor["record"]["run_id"], 1)
            result.unlink()
            stale = root / ("2" * 32 + "-1x1-warmup-0.json")
            stale.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
                self.runner._scan_sample_artifacts(root, anchor["record"]["run_id"], 1)
            stale.unlink()
            first_result = pathlib.Path(
                json.loads(self.runner._entry_path(root, 0).read_text())["files"]["result"]["path"]
            )
            first_result.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw triad changed"):
                with mock.patch.object(self.runner, "_validate_sample_artifacts"):
                    self.runner._load_checkpoint_entries(
                        repo=root, output=root, runtime={}, receipt={}, anchor=anchor
                    )

    def test_valid_prefix_resume_revalidates_and_recomputes_deferred_correctness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, samples, _closure = self.publish_prefix(root, 9)
            mapping = {
                (sample["config_id"], sample["sample_kind"], sample["iteration"]): sample
                for sample in samples
            }
            cache = self.runner.StatHashCache()

            def validate_with_epoch_cache(**kwargs):
                self.assertIs(kwargs["hash_cache"], cache)
                return copy.deepcopy(
                    mapping[
                        (
                            kwargs["config"].config_id,
                            kwargs["sample_kind"],
                            kwargs["iteration"],
                        )
                    ]
                )

            with mock.patch.object(
                self.runner,
                "_validate_sample_artifacts",
                side_effect=validate_with_epoch_cache,
            ) as validator:
                loaded, closure = self.runner._load_checkpoint_entries(
                    repo=root,
                    output=root,
                    runtime={},
                    receipt={},
                    anchor=anchor,
                    hash_cache=cache,
                )
            self.assertEqual(validator.call_count, 9)
            self.assertEqual(closure["record"]["entry_count"], 9)
            self.assertEqual(self.runner.sample_schedule()[9], ("2x1", "measured", 0))
            self.assertTrue(all(sample["correctness"]["pass"] for sample in loaded))
            first = json.loads(self.runner._entry_path(root, 0).read_text())
            self.assertEqual(first["gates"]["correctness"]["status"], "deferred")
            reference = json.loads(self.runner._entry_path(root, 8).read_text())
            self.assertEqual(reference["gates"]["correctness"]["status"], "complete")

    def test_stale_final_artifacts_are_never_deleted_for_resume(self):
        for name in ("report.json", "report.md", "COMPLETE"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                path = root / name
                path.write_text("stale\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "final artifacts"):
                    self.runner._reject_stale_publication_artifacts(root)
                self.assertTrue(path.exists())

    def test_trusted_failed_state_can_resume_without_destroying_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            self.runner._write_running_state(
                root, anchor, closure, resumed=False, prior_failure=None
            )
            self.runner.mark_failed(
                root,
                anchor["record"]["run_id"],
                KeyboardInterrupt(),
                anchor=anchor,
                journal_closure=closure,
            )
            state = self.runner._validate_resume_state(root, anchor)
            self.assertEqual(state["failed"]["sha256"], sha256_file(root / "FAILED.json"))
            self.assertTrue((root / "FAILED.json").exists())
            changed = json.loads((root / "state.json").read_text())
            changed["run_id"] = "2" * 32
            self.runner.atomic_write_json(root / "state.json", changed)
            with self.assertRaisesRegex(RuntimeError, "wrong run identity"):
                self.runner._validate_resume_state(root, anchor)

    def test_correctness_failure_is_terminal_and_failed_genesis_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            self.runner._write_running_state(
                root, anchor, closure, resumed=False, prior_failure=None
            )
            self.runner.mark_failed(
                root,
                anchor["record"]["run_id"],
                self.runner.CorrectnessGateFailure(
                    "objective/full-gradient correctness gate failed",
                    {"pass": False, "objective": {"absolute_error": 1e-5}},
                ),
                anchor=anchor,
                journal_closure=closure,
            )
            failed_bytes = (root / "FAILED.json").read_bytes()
            raw = root / "uncommitted-raw-evidence"
            raw.write_text("must remain\n", encoding="utf-8")
            with self.assertRaises(self.runner.NonResumableRun):
                self.runner._validate_resume_state(root, anchor)
            self.assertEqual((root / "FAILED.json").read_bytes(), failed_bytes)
            self.assertEqual(raw.read_text(encoding="utf-8"), "must remain\n")

    def test_multiple_resumable_failures_append_without_overwriting_genesis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            self.runner._write_running_state(
                root, anchor, closure, resumed=False, prior_failure=None
            )
            self.runner.mark_failed(
                root,
                anchor["record"]["run_id"],
                KeyboardInterrupt(),
                anchor=anchor,
                journal_closure=closure,
            )
            genesis = (root / "FAILED.json").read_bytes()
            first = self.runner._validate_resume_state(root, anchor)
            self.runner._write_running_state(
                root,
                anchor,
                closure,
                resumed=True,
                prior_failure=first["failed"],
            )
            self.runner.mark_failed(
                root,
                anchor["record"]["run_id"],
                self.runner.ResumableRunFailure("clean timeout"),
                anchor=anchor,
                journal_closure=closure,
            )
            self.assertEqual((root / "FAILED.json").read_bytes(), genesis)
            self.assertTrue(self.runner._failure_path(root, 1).is_file())
            loaded = self.runner._load_failure_events(root, anchor)
            self.assertEqual(loaded["record"]["event_count"], 2)
            self.assertFalse(loaded["record"]["terminal"])

    def test_unknown_failure_defaults_nonresumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            self.runner._write_running_state(
                root, anchor, closure, resumed=False, prior_failure=None
            )
            self.runner.mark_failed(
                root,
                anchor["record"]["run_id"],
                RuntimeError("unknown internal failure"),
                anchor=anchor,
                journal_closure=closure,
            )
            with self.assertRaises(self.runner.NonResumableRun):
                self.runner._validate_resume_state(root, anchor)

    def test_failure_event_state_crash_window_is_reconciled_fail_closed(self):
        for error, resumable in (
            (KeyboardInterrupt(), True),
            (
                self.runner.CorrectnessGateFailure(
                    "correctness failed", {"pass": False}
                ),
                False,
            ),
        ):
            with self.subTest(resumable=resumable), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor = self.anchor(root)
                closure = self.runner._empty_journal_closure()
                self.runner._write_running_state(
                    root, anchor, closure, resumed=False, prior_failure=None
                )
                self.runner._publish_failure_event(root, anchor, error)
                self.assertEqual(
                    json.loads((root / "state.json").read_text())["state"],
                    "RUNNING",
                )
                if resumable:
                    state = self.runner._validate_resume_state(root, anchor)
                    self.assertIsNotNone(state["failed"])
                else:
                    with self.assertRaises(self.runner.NonResumableRun):
                        self.runner._validate_resume_state(root, anchor)

    def test_terminal_failure_rejects_later_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            self.runner._write_running_state(
                root, anchor, closure, resumed=False, prior_failure=None
            )
            self.runner._publish_failure_event(
                root,
                anchor,
                self.runner.CorrectnessGateFailure(
                    "correctness failed", {"pass": False}
                ),
            )
            genesis = (root / "FAILED.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "terminal failure"):
                self.runner._publish_failure_event(
                    root, anchor, KeyboardInterrupt()
                )
            self.assertEqual((root / "FAILED.json").read_bytes(), genesis)
            self.assertFalse(self.runner._failure_path(root, 1).exists())

    def test_resume_event_is_append_only_and_hash_chained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            closure = self.runner._empty_journal_closure()
            first = self.runner._publish_resume_event(root, anchor, closure)
            second = self.runner._publish_resume_event(root, anchor, closure)
            loaded = self.runner._load_resume_events(root, anchor)
            self.assertEqual(loaded["record"]["event_count"], 2)
            self.assertEqual(loaded["record"]["head_sha256"], second["sha256"])
            second_record = json.loads(pathlib.Path(second["path"]).read_text())
            self.assertEqual(second_record["previous_event_sha256"], first["sha256"])
            second_record["previous_event_sha256"] = "0" * 64
            pathlib.Path(second["path"]).write_text(json.dumps(second_record) + "\n")
            with self.assertRaisesRegex(RuntimeError, "hash-chain"):
                self.runner._load_resume_events(root, anchor)

    def test_run_anchor_rejects_wrong_receipt_source_boot_and_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo = root / "repo"
            output = root / "output"
            (repo / "scripts").mkdir(parents=True)
            output.mkdir()
            producer = repo / "scripts" / "benchmark-adjoint.py"
            producer.write_text("# producer\n", encoding="utf-8")
            receipt_path = root / "receipt.json"
            receipt_path.write_text("{}\n", encoding="utf-8")
            receipt = {
                "receipt_id": "receipt-id",
                "build_input_id": "input-id",
                "artifact_set_id": "artifact-id",
            }
            snapshot = {"file_count": 1, "sha256": "1" * 64}
            power_state = {
                "global": {},
                "per_cpu": {},
                "loadavg": {"load_1m": 0.1, "load_5m": 0.2, "load_15m": 0.3},
            }
            topology_record = {
                "schema_version": 1,
                "captured_at_utc": "2026-01-01T00:00:00Z",
                "parent_pid": 1,
                "parent_affinity": list(range(16)),
                "normalized": topology(self.runner),
                "power_load_state_start": power_state,
                "summary": {"Model name": "test-cpu"},
                "lscpu_executable": {"path": "/usr/bin/lscpu", "sha256": "2" * 64},
                "probes": {},
            }
            topology_path = output / "topology.json"
            self.runner.atomic_write_json(topology_path, topology_record)
            topology_evidence = {
                "path": str(topology_path.resolve()),
                "sha256": sha256_file(topology_path),
                "record": topology_record,
            }
            boot = {
                "path": "/proc/sys/kernel/random/boot_id",
                "boot_id": "00000000-0000-0000-0000-000000000001",
                "sha256": "3" * 64,
            }
            host = {"hostname": "test-host", "uname_nodename": "test-host"}
            runner_provenance = {"record": {"runner": True}, "sha256": "4" * 64}
            with mock.patch.object(self.runner, "_boot_identity", return_value=boot), mock.patch.object(
                self.runner, "_host_identity", return_value=host
            ), mock.patch.object(
                self.runner, "_runner_provenance", return_value=runner_provenance
            ):
                payload = self.runner._anchor_payload(
                    repo=repo,
                    output=output,
                    run_id="5" * 32,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    snapshot=snapshot,
                    producer_sha256=sha256_file(producer),
                    topology_evidence=topology_evidence,
                )
                self.runner._publish_run_anchor(output, payload)

            def validate(**overrides):
                current_receipt = overrides.get("receipt", receipt)
                current_snapshot = overrides.get("snapshot", snapshot)
                current_boot = overrides.get("boot", boot)
                live = copy.deepcopy(topology_record)
                if overrides.get("topology") == "wrong":
                    live["parent_affinity"] = list(range(15))
                with mock.patch.object(self.runner, "_boot_identity", return_value=current_boot), mock.patch.object(
                    self.runner, "_host_identity", return_value=host
                ), mock.patch.object(
                    self.runner, "_runner_provenance", return_value=runner_provenance
                ), mock.patch.object(
                    self.runner, "_capture_topology_record", return_value=live
                ):
                    return self.runner._validate_run_anchor(
                        repo=repo,
                        output=output,
                        receipt_path=receipt_path,
                        receipt=current_receipt,
                        snapshot=current_snapshot,
                        producer_sha256=sha256_file(producer),
                    )

            self.assertEqual(validate()["record"]["run_id"], "5" * 32)
            wrong_receipt = copy.deepcopy(receipt)
            wrong_receipt["receipt_id"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "receipt/source"):
                validate(receipt=wrong_receipt)
            with self.assertRaisesRegex(RuntimeError, "receipt/source"):
                validate(snapshot={"file_count": 1, "sha256": "0" * 64})
            wrong_boot = copy.deepcopy(boot)
            wrong_boot["boot_id"] = "00000000-0000-0000-0000-000000000002"
            with self.assertRaisesRegex(RuntimeError, "receipt/source"):
                validate(boot=wrong_boot)
            with self.assertRaisesRegex(RuntimeError, "topology/affinity"):
                validate(topology="wrong")

    def test_publication_rejects_checkpoint_closure_tamper(self):
        samples = []
        for config_id, sample_kind, iteration in self.runner.sample_schedule():
            config = self.runner._config_by_id(config_id)
            signature_record = {"world_size": config.ranks, "final": {}, "phases": {}}
            validation = selection_validation(10.0 + config.ranks)
            validation["aggregate_counter_signature"] = {
                "record": signature_record,
                "sha256": self.runner.canonical_sha256(signature_record),
            }
            samples.append(
                {
                    "config_id": config_id,
                    "sample_kind": sample_kind,
                    "iteration": iteration,
                    "nonce": f"{len(samples) + 1:032x}",
                    "validation": validation,
                    "correctness": {"pass": True},
                    "receipt_gate": {"pass": True},
                    "process_environment_gate": {"pass": True},
                    "material_gradient_gate": {"pass": True},
                    "process_seconds": 20.0,
                }
            )
        matrix = self.runner.validate_sample_matrix(samples)
        selection = self.runner.select_fastest(samples)
        checkpoint = {"pass": True, "record": {"entry_count": 32}, "sha256": "a" * 64}
        resumes = {"pass": True, "record": {"event_count": 0}, "sha256": "b" * 64}
        failures = {
            "pass": True,
            "record": {
                "event_count": 0,
                "head_sha256": None,
                "terminal": False,
                "events": [],
            },
            "sha256": self.runner.canonical_sha256(
                {
                    "event_count": 0,
                    "head_sha256": None,
                    "terminal": False,
                    "events": [],
                }
            ),
        }
        recoveries = {
            "pass": True,
            "record": {"event_count": 0, "pending_orphan_count": 0},
            "sha256": "c" * 64,
        }
        authoritative = {
            "pass": True,
            "sample_matrix": matrix,
            "selection": selection,
            "checkpoint_closure": checkpoint,
            "resume_closure": resumes,
            "failure_closure": failures,
            "recovery_closure": recoveries,
        }
        gate = {
            "pass": True,
            "receipt_and_source": True,
            "topology": True,
            "sample_matrix": True,
            "world_backend_counters_affinity": True,
            "objective_and_full_gradient": True,
            "raw_evidence_integrity": True,
            "durable_checkpoint_journal": True,
            "durable_recovery_journal": True,
            "selection": True,
        }
        report = {
            "state": "COMPLETE",
            "profile": dict(self.runner.PROFILE),
            "sample_matrix": matrix,
            "selection": selection,
            "checkpoint_closure": copy.deepcopy(checkpoint),
            "resume_closure": resumes,
            "failure_closure": failures,
            "recovery_closure": recoveries,
            "integrity_reverification": authoritative,
            "gate": gate,
        }
        serialized_report = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(serialized_report["sample_matrix"], matrix)
        self.runner.validate_publication_report(
            serialized_report,
            samples,
            authoritative_recomputation=authoritative,
        )
        report["checkpoint_closure"]["record"]["entry_count"] = 31
        with self.assertRaisesRegex(RuntimeError, "publication recomputation"):
            self.runner.validate_publication_report(
                report, samples, authoritative_recomputation=authoritative
            )

    def test_complete_marker_reverification_detects_marker_and_report_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            report_path = root / "report.json"
            markdown_path = root / "report.md"
            report_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("ok\n", encoding="utf-8")
            marker = {
                "schema_version": 1,
                "state": "COMPLETE",
                "run_id": "1" * 32,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "report_markdown": str(markdown_path.resolve()),
                "report_markdown_sha256": sha256_file(markdown_path),
            }
            self.runner.atomic_write_json(root / "COMPLETE", marker)
            self.runner.atomic_write_json(root / "state.json", marker)
            self.runner._validate_complete_marker(root, marker, require_state=True)
            report_path.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "report closure"):
                self.runner._validate_complete_marker(root, marker, require_state=True)

    def test_publish_deep_verifies_before_complete_and_survives_post_verify_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sample = {
                "config_id": "synthetic",
                "result": {
                    "schema_version": 1,
                    "result": {"gradient": [1.0, 2.0]},
                },
            }
            closure = {
                "checkpoint_closure": {
                    "record": {"entry_count": 32, "head_sha256": "a" * 64},
                    "sha256": "b" * 64,
                },
                "resume_closure": {
                    "record": {"event_count": 0},
                    "sha256": "c" * 64,
                },
                "failure_closure": {
                    "record": {"event_count": 0},
                    "sha256": "e" * 64,
                },
                "recovery_closure": {
                    "record": {"event_count": 0},
                    "sha256": "d" * 64,
                },
            }
            report = {
                "run_id": "1" * 32,
                "samples": self.runner._summarize_samples([sample]),
                "execution_continuity": {"prior_failure": None},
            }
            anchor = {"path": str(root / "checkpoint" / "run.json"), "sha256": "e" * 64}
            verifier_calls = []
            cache = self.runner.StatHashCache()

            def verifier(**kwargs):
                self.assertIs(kwargs["hash_cache"], cache)
                verifier_calls.append(kwargs["marker_name"] if "marker_name" in kwargs else "COMPLETE")
                if kwargs.get("marker_name") == "PUBLISHING.json":
                    self.assertTrue((root / "PUBLISHING.json").is_file())
                    self.assertFalse((root / "COMPLETE").exists())
                    return {"pass": True}
                raise RecursionError("injected post-publication verifier regression")

            with mock.patch.object(
                self.runner, "reverify_final_closure", return_value=closure
            ) as final_reverify, mock.patch.object(
                self.runner, "validate_publication_report", return_value={"pass": True}
            ), mock.patch.object(
                self.runner, "verify_build_receipt", return_value={}
            ) as receipt_verifier, mock.patch.object(
                self.runner, "_verify_anchor_evidence_on_disk", return_value={}
            ), mock.patch.object(
                self.runner, "_verify_published_complete", side_effect=verifier
            ), mock.patch.object(
                self.runner, "_verify_report_failure_consistency", return_value=None
            ):
                self.runner.publish_complete(
                    root,
                    report["run_id"],
                    report,
                    "synthetic report\n",
                    capability=self.runner._PUBLISH_CAPABILITY,
                    repo=root,
                    receipt_path=root / "receipt.json",
                    expected_receipt_id="receipt",
                    expected_snapshot={"sha256": "f" * 64},
                    expected_producer_sha256="0" * 64,
                    topology_evidence={},
                    samples=[sample],
                    anchor_evidence=anchor,
                    hash_cache=cache,
                )
            self.assertEqual(verifier_calls, ["PUBLISHING.json", "COMPLETE"])
            self.assertTrue(
                all(
                    call.kwargs["hash_cache"] is cache
                    for call in final_reverify.call_args_list
                )
            )
            self.assertIs(receipt_verifier.call_args.kwargs["hash_cache"], cache)
            self.assertEqual(
                json.loads((root / "PUBLISHING.json").read_text())["state"],
                "PUBLISHING",
            )
            complete = json.loads((root / "COMPLETE").read_text())
            self.assertEqual(complete["state"], "COMPLETE")
            self.assertEqual(json.loads((root / "state.json").read_text()), complete)
            self.assertFalse((root / "FAILED.json").exists())

    def test_post_verify_fallback_rehashes_publishing_before_accepting_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sample = {
                "config_id": "synthetic",
                "result": {
                    "schema_version": 1,
                    "result": {"gradient": [1.0, 2.0]},
                },
            }
            closure = {
                "checkpoint_closure": {
                    "record": {"entry_count": 32, "head_sha256": "a" * 64},
                    "sha256": "b" * 64,
                },
                "resume_closure": {
                    "record": {"event_count": 0},
                    "sha256": "c" * 64,
                },
                "failure_closure": {
                    "record": {"event_count": 0},
                    "sha256": "e" * 64,
                },
                "recovery_closure": {
                    "record": {"event_count": 0},
                    "sha256": "d" * 64,
                },
            }
            report = {
                "run_id": "1" * 32,
                "samples": self.runner._summarize_samples([sample]),
                "execution_continuity": {"prior_failure": None},
            }
            anchor = {
                "path": str(root / "checkpoint" / "run.json"),
                "sha256": "e" * 64,
            }
            verifier_calls = []

            def verifier(**kwargs):
                marker_name = kwargs.get("marker_name", "COMPLETE")
                verifier_calls.append(marker_name)
                if marker_name == "PUBLISHING.json":
                    return {"pass": True}
                (root / "PUBLISHING.json").write_text(
                    '{"tampered": true}\n', encoding="utf-8"
                )
                raise RuntimeError("injected post-publication evidence tamper")

            with mock.patch.object(
                self.runner, "reverify_final_closure", return_value=closure
            ), mock.patch.object(
                self.runner, "validate_publication_report", return_value={"pass": True}
            ), mock.patch.object(
                self.runner, "verify_build_receipt", return_value={}
            ), mock.patch.object(
                self.runner, "_verify_anchor_evidence_on_disk", return_value={}
            ), mock.patch.object(
                self.runner, "_verify_published_complete", side_effect=verifier
            ), mock.patch.object(
                self.runner, "_verify_report_failure_consistency", return_value=None
            ), mock.patch.object(
                self.runner, "_recover_interrupted_evidence", return_value={"pass": True}
            ) as recovery:
                with self.assertRaisesRegex(
                    RuntimeError, "final markers were quarantined"
                ):
                    self.runner.publish_complete(
                        root,
                        report["run_id"],
                        report,
                        "synthetic report\n",
                        capability=self.runner._PUBLISH_CAPABILITY,
                        repo=root,
                        receipt_path=root / "receipt.json",
                        expected_receipt_id="receipt",
                        expected_snapshot={"sha256": "f" * 64},
                        expected_producer_sha256="0" * 64,
                        topology_evidence={},
                        samples=[sample],
                        anchor_evidence=anchor,
                    )
            self.assertEqual(verifier_calls, ["PUBLISHING.json", "COMPLETE"])
            recovery.assert_called_once()
            self.assertNotEqual(
                self.runner.sha256_file(root / "PUBLISHING.json"),
                json.loads((root / "COMPLETE").read_text())["publishing_marker"]["sha256"],
            )

    def test_on_disk_anchor_closure_rejects_missing_tamper_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            self.assertEqual(
                self.runner._verify_anchor_evidence_on_disk(root, anchor)["sha256"],
                anchor["sha256"],
            )
            path = pathlib.Path(anchor["path"])
            original = path.read_bytes()
            path.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "anchor binding"):
                self.runner._verify_anchor_evidence_on_disk(root, anchor)
            path.write_bytes(original)
            alias = root / "anchor-hardlink.json"
            alias.hardlink_to(path)
            with self.assertRaisesRegex(RuntimeError, "single-link"):
                self.runner._verify_anchor_evidence_on_disk(root, anchor)
            alias.unlink()
            path.unlink()
            with self.assertRaises(RuntimeError):
                self.runner._verify_anchor_evidence_on_disk(root, anchor)

    def test_interruption_recovery_quarantines_timeout_nonzero_and_full_triad(self):
        for mode, artifact_count in (("sigkill", 1), ("timeout", 2), ("nonzero", 3)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor = self.anchor(root)
                paths = self.runner._sample_artifact_paths(
                    root, anchor["record"]["run_id"], *self.runner.sample_schedule()[0]
                )
                for index, path in enumerate(paths[:artifact_count]):
                    path.write_text(f"{mode}-{index}\n", encoding="utf-8")
                closure = self.runner._recover_interrupted_evidence(root, anchor, 0)
                self.assertEqual(closure["record"]["event_count"], 1)
                self.assertEqual(closure["record"]["pending_orphan_count"], 0)
                self.assertTrue(all(not path.exists() for path in paths))
                event = json.loads(self.runner._recovery_path(root, 0).read_text())
                self.assertEqual(len(event["artifacts"]), artifact_count)
                self.assertTrue(
                    all(
                        artifact["reason"] == "interrupted-next-sample"
                        for artifact in event["artifacts"]
                    )
                )
                for artifact in event["artifacts"]:
                    orphan = pathlib.Path(artifact["orphan_path"])
                    self.assertTrue(orphan.is_file())
                    self.assertEqual(sha256_file(orphan), artifact["sha256"])

    def test_recovery_resumes_each_durable_crash_boundary(self):
        stages = (
            "before-intent",
            "after-intent",
            "after-orphan-mkdir",
            "after-manifest",
            "after-artifact-move:0",
            "before-event",
        )
        for crash_stage in stages:
            with self.subTest(stage=crash_stage), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor = self.anchor(root)
                raw_paths = self.runner._sample_artifact_paths(
                    root,
                    anchor["record"]["run_id"],
                    *self.runner.sample_schedule()[0],
                )
                for index, path in enumerate(raw_paths):
                    path.write_text(f"raw-{index}\n", encoding="utf-8")

                def inject(stage):
                    if stage == crash_stage:
                        raise RuntimeError(f"crash at {stage}")

                with self.assertRaisesRegex(RuntimeError, "crash at"):
                    self.runner._recover_interrupted_evidence(
                        root, anchor, 0, fault_injector=inject
                    )
                if crash_stage == "before-intent":
                    self.assertFalse(
                        self.runner._recovery_intent_path(root, 0).exists()
                    )
                closure = self.runner._recover_interrupted_evidence(root, anchor, 0)
                self.assertEqual(closure["record"]["event_count"], 1)
                self.assertEqual(closure["record"]["intent_count"], 1)
                self.assertEqual(closure["record"]["pending_intent_count"], 0)
                self.assertEqual(closure["record"]["pending_orphan_count"], 0)
                self.assertTrue(all(not path.exists() for path in raw_paths))
                event = self.runner._read_json_object(
                    self.runner._recovery_path(root, 0), label="completed recovery"
                )
                self.assertEqual(len(event["artifacts"]), len(raw_paths))
                self.assertTrue(
                    all(
                        pathlib.Path(artifact["orphan_path"]).is_file()
                        for artifact in event["artifacts"]
                    )
                )

    def test_complete_state_publication_recovery_resumes_every_crash_boundary(self):
        stages = (
            "after-intent",
            "after-orphan-mkdir",
            "after-manifest",
            "after-artifact-move:0",
            "after-artifact-move:1",
            "after-artifact-move:2",
            "after-artifact-move:3",
            "before-event",
            "after-event",
        )
        for crash_stage in stages:
            with self.subTest(stage=crash_stage), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor, samples, _closure = self.publish_prefix(
                    root, len(self.runner.sample_schedule())
                )
                complete = self.publication_quartet(root, anchor)

                def inject(stage):
                    if stage == crash_stage:
                        raise RuntimeError(f"crash at {stage}")

                with self.assertRaisesRegex(RuntimeError, "crash at"):
                    self.runner._recover_interrupted_evidence(
                        root,
                        anchor,
                        len(self.runner.sample_schedule()),
                        allow_complete_publication_triad=True,
                        fault_injector=inject,
                    )
                context = self.runner._publication_recovery_context(root, anchor)
                self.assertIsNotNone(context)
                state = self.runner._validate_resume_state(
                    root, anchor, publication_recovery=context
                )
                self.assertEqual(state["state"], complete)
                recovered = self.runner._recover_interrupted_evidence(
                    root,
                    anchor,
                    len(self.runner.sample_schedule()),
                    allow_complete_publication_triad=True,
                )
                self.assertEqual(recovered["record"]["pending_intent_count"], 0)
                self.assertEqual(recovered["record"]["pending_orphan_count"], 0)
                self.assertTrue(
                    all(
                        not (root / name).exists()
                        for name in self.runner.RECOVERABLE_PUBLICATION_ARTIFACTS
                    )
                )

                mapping = {
                    (
                        sample["config_id"],
                        sample["sample_kind"],
                        sample["iteration"],
                    ): sample
                    for sample in samples
                }
                with mock.patch.object(
                    self.runner,
                    "_validate_sample_artifacts",
                    side_effect=lambda **kwargs: copy.deepcopy(
                        mapping[
                            (
                                kwargs["config"].config_id,
                                kwargs["sample_kind"],
                                kwargs["iteration"],
                            )
                        ]
                    ),
                ):
                    _loaded, journal = self.runner._load_checkpoint_entries(
                        repo=root,
                        output=root,
                        runtime={},
                        receipt={},
                        anchor=anchor,
                    )
                self.runner._publish_resume_event(root, anchor, journal)
                self.runner._write_running_state(
                    root,
                    anchor,
                    journal,
                    resumed=True,
                    prior_failure=None,
                )
                self.assertEqual(
                    json.loads((root / "state.json").read_text())["state"],
                    "RUNNING",
                )

    def test_complete_state_recovery_rejects_marker_state_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, _samples, _closure = self.publish_prefix(
                root, len(self.runner.sample_schedule())
            )
            self.publication_quartet(root, anchor)

            def stop_after_intent(stage):
                if stage == "after-intent":
                    raise RuntimeError("stop after intent")

            with self.assertRaisesRegex(RuntimeError, "stop after intent"):
                self.runner._recover_interrupted_evidence(
                    root,
                    anchor,
                    len(self.runner.sample_schedule()),
                    allow_complete_publication_triad=True,
                    fault_injector=stop_after_intent,
                )
            state = json.loads((root / "state.json").read_text())
            state["checkpoint_head_sha256"] = "0" * 64
            self.runner.atomic_write_json(root / "state.json", state)
            with self.assertRaisesRegex(RuntimeError, "byte-identical"):
                self.runner._publication_recovery_context(root, anchor)

    def test_complete_state_recovery_accepts_only_valid_later_journal_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, _samples, journal = self.publish_prefix(
                root, len(self.runner.sample_schedule())
            )
            self.publication_quartet(root, anchor)

            def stop_after_intent(stage):
                if stage == "after-intent":
                    raise RuntimeError("stop after intent")

            with self.assertRaisesRegex(RuntimeError, "stop after intent"):
                self.runner._recover_interrupted_evidence(
                    root,
                    anchor,
                    len(self.runner.sample_schedule()),
                    allow_complete_publication_triad=True,
                    fault_injector=stop_after_intent,
                )
            failure = self.runner._publish_failure_event(
                root, anchor, KeyboardInterrupt()
            )
            context = self.runner._publication_recovery_context(root, anchor)
            state = self.runner._validate_resume_state(
                root, anchor, publication_recovery=context
            )
            self.assertEqual(state["failed"], failure["event"])

            self.runner._recover_interrupted_evidence(
                root,
                anchor,
                len(self.runner.sample_schedule()),
                allow_complete_publication_triad=True,
            )
            self.runner._publish_resume_event(root, anchor, journal)
            context_after_resume_event = self.runner._publication_recovery_context(
                root, anchor
            )
            state_after_resume_event = self.runner._validate_resume_state(
                root,
                anchor,
                publication_recovery=context_after_resume_event,
            )
            self.assertEqual(state_after_resume_event["failed"], failure["event"])

            terminal = self.runner._publish_failure_event(
                root, anchor, RuntimeError("terminal after recovery")
            )
            self.assertTrue(terminal["closure"]["record"]["terminal"])
            with self.assertRaises(self.runner.NonResumableRun):
                self.runner._publication_recovery_context(root, anchor)

    def test_recovery_rejects_external_orphan_destination_before_any_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            raw = self.runner._sample_artifact_paths(
                root,
                anchor["record"]["run_id"],
                *self.runner.sample_schedule()[0],
            )[0]
            raw.write_text("must remain\n", encoding="utf-8")

            def stop_after_manifest(stage):
                if stage == "after-manifest":
                    raise RuntimeError("stop after manifest")

            with self.assertRaisesRegex(RuntimeError, "stop after manifest"):
                self.runner._recover_interrupted_evidence(
                    root, anchor, 0, fault_injector=stop_after_manifest
                )
            orphan = self.runner._orphan_path(root, 0)
            manifest_path = orphan / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            outside = pathlib.Path("/tmp") / f"gpmeep-path-escape-{root.name}"
            self.assertFalse(outside.exists())
            manifest["artifacts"][0]["orphan_path"] = str(outside)
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact recoverable set"):
                self.runner._complete_pending_recovery(
                    root, anchor, orphan, manifest
                )
            self.assertEqual(raw.read_text(encoding="utf-8"), "must remain\n")
            self.assertFalse(outside.exists())

    def test_arbitrary_empty_orphan_without_intent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            orphan = self.runner._orphan_path(root, 0)
            orphan.parent.mkdir(parents=True)
            orphan.mkdir(mode=0o700)
            with self.assertRaisesRegex(RuntimeError, "counts disagree"):
                self.runner._load_recovery_events(root, anchor)

    def test_recovery_rejects_next_plus_one_foreign_symlink_and_hardlink(self):
        attacks = ("future", "foreign", "symlink", "hardlink")
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor = self.anchor(root)
                if attack == "future":
                    path = self.runner._sample_artifact_paths(
                        root, anchor["record"]["run_id"], *self.runner.sample_schedule()[1]
                    )[0]
                    path.write_text("future\n", encoding="utf-8")
                elif attack == "foreign":
                    path = root / ("f" * 32 + "-1x1-warmup-0.json")
                    path.write_text("foreign\n", encoding="utf-8")
                else:
                    path = self.runner._sample_artifact_paths(
                        root, anchor["record"]["run_id"], *self.runner.sample_schedule()[0]
                    )[0]
                    target = root / "target"
                    target.write_text("alias\n", encoding="utf-8")
                    if attack == "symlink":
                        path.symlink_to(target)
                    else:
                        path.hardlink_to(target)
                with self.assertRaises(RuntimeError):
                    self.runner._recover_interrupted_evidence(root, anchor, 0)

    def test_complete_prefix_partial_publication_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, _samples, closure = self.publish_prefix(
                root, len(self.runner.sample_schedule())
            )
            self.assertTrue(closure["record"]["complete"])
            report_md = root / "report.md"
            report_json = root / "report.json"
            report_md.write_text("partial markdown\n", encoding="utf-8")
            report_json.write_text('{"partial": true}\n', encoding="utf-8")
            recovered = self.runner._recover_interrupted_evidence(
                root, anchor, len(self.runner.sample_schedule())
            )
            self.assertEqual(recovered["record"]["event_count"], 1)
            self.assertFalse(report_md.exists())
            self.assertFalse(report_json.exists())
            event = json.loads(self.runner._recovery_path(root, 0).read_text())
            self.assertEqual(
                {artifact["reason"] for artifact in event["artifacts"]},
                {"partial-publication"},
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, _samples, _closure = self.publish_prefix(
                root, len(self.runner.sample_schedule())
            )
            for name in self.runner.RECOVERABLE_PUBLICATION_ARTIFACTS:
                (root / name).write_text(f"partial {name}\n", encoding="utf-8")
            recovered = self.runner._recover_interrupted_evidence(
                root,
                anchor,
                len(self.runner.sample_schedule()),
                allow_complete_publication_triad=True,
            )
            self.assertEqual(recovered["record"]["event_count"], 1)
            self.assertTrue(
                all(
                    not (root / name).exists()
                    for name in self.runner.RECOVERABLE_PUBLICATION_ARTIFACTS
                )
            )

    def test_output_and_checkpoint_ancestor_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                self.runner._trusted_output_path(alias / "output", create=True)

        for child, loader in (
            ("entries", "_inspect_checkpoint_chain"),
            ("resumes", "_load_resume_events"),
            ("failures", "_load_failure_events"),
            ("recoveries", "_load_recovery_events"),
            ("recovery-intents", "_load_recovery_events"),
            ("orphans", "_load_recovery_events"),
        ):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                anchor = self.anchor(root)
                external = root / f"external-{child}"
                external.mkdir()
                (root / "checkpoint" / child).symlink_to(
                    external, target_is_directory=True
                )
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    getattr(self.runner, loader)(root, anchor)

    def test_raw_hardlink_alias_and_recovery_event_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor, samples, _closure = self.publish_prefix(root, 1)
            raw = pathlib.Path(samples[0]["result_file"])
            alias = root / "raw-hardlink-alias"
            alias.hardlink_to(raw)
            with mock.patch.object(self.runner, "_validate_sample_artifacts"):
                with self.assertRaisesRegex(RuntimeError, "single-link"):
                    self.runner._load_checkpoint_entries(
                        repo=root, output=root, runtime={}, receipt={}, anchor=anchor
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            next_result = self.runner._sample_artifact_paths(
                root, anchor["record"]["run_id"], *self.runner.sample_schedule()[0]
            )[0]
            next_result.write_text("interrupted\n", encoding="utf-8")
            self.runner._recover_interrupted_evidence(root, anchor, 0)
            event_path = self.runner._recovery_path(root, 0)
            event = json.loads(event_path.read_text())
            event["previous_event_sha256"] = "0" * 64
            event_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash-chain"):
                self.runner._load_recovery_events(root, anchor)

    def test_failed_marker_report_contradictions_and_complete_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchor = self.anchor(root)
            failure_closure = self.runner._load_failure_events(root, anchor)
            report = {
                "run_id": anchor["record"]["run_id"],
                "failure_closure": failure_closure,
                "execution_continuity": {
                    "prior_failure": None,
                    "failure_closure": failure_closure,
                },
            }
            self.assertIsNone(
                self.runner._verify_report_failure_consistency(
                    root, report, anchor
                )
            )
            report["execution_continuity"]["prior_failure"] = {
                "path": str(root / "FAILED.json"),
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "contradicts"):
                self.runner._verify_report_failure_consistency(
                    root, report, anchor
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            report_path = root / "report.json"
            markdown_path = root / "report.md"
            report_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("ok\n", encoding="utf-8")
            marker = {
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "report_markdown": str(markdown_path.resolve()),
                "report_markdown_sha256": sha256_file(markdown_path),
            }
            self.runner.atomic_write_json(root / "COMPLETE", marker)
            self.runner.atomic_write_json(root / "state.json", {"state": "FAILED"})
            with self.assertRaisesRegex(RuntimeError, "differs from COMPLETE"):
                self.runner._validate_complete_marker(root, marker, require_state=True)


if __name__ == "__main__":
    unittest.main()
