"""Self-tests for the standard-library gpmeep validation harness."""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


RUNNER_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_validation.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_validation_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from gpmeep_provenance import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_record,
    source_snapshot,
    tree_manifest,
)
from gpmeep_qualification_contract import (  # noqa: E402
    AUDIT_LOG_NAME,
    ALL_QUALIFICATION_DIRECTORY_LOGS,
    DIRECT_ELF_LOG_BINDINGS,
    INSTALLED_CUDA_QUALIFICATION_SPECS,
    INSTALLED_LAZY_API_TEST_SPECS,
    PYTHON_RUNTIME_LOG_BINDINGS,
    REQUIRED_ARTIFACT_BINDINGS,
    REQUIRED_QUALIFICATION_LOGS,
    STEP_FINITE_FAILURE_DIAGNOSTIC,
    STEP_FINITE_FAILURE_LOG_NAME,
    create_identity_snapshot,
    marker_for,
    seal_contract,
)
from qualification_fixture import (  # noqa: E402
    write_synthetic_auxiliary_logs,
    write_synthetic_release_cuda_math_configuration,
)


FAKE_MEEP = """
import ctypes
import os
import pathlib

_package = pathlib.Path(__file__).resolve().parent
_extension_path = _package / "_meep-self-test.so"
_libmeep_path = _package.parent / "libmeep.so.38.0.0"
ctypes.CDLL(str(_extension_path), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(_libmeep_path), mode=ctypes.RTLD_GLOBAL)

class _Extension:
    __file__ = str(_extension_path)

    @staticmethod
    def is_single_precision():
        return True

    @staticmethod
    def with_mpi():
        return True

    @staticmethod
    def _gpu_backend_compiled():
        return True

    @staticmethod
    def _gpu_runtime_available():
        return True

    @staticmethod
    def _gpu_runtime_diagnostic():
        return ""

    @staticmethod
    def _gpu_compiled_architectures():
        return "self-test"

    @staticmethod
    def _gpu_requested_backend():
        return {"cpu": 0, "auto": 1, "cuda": 2}[
            os.environ["MEEP_GPU_BACKEND"]
        ]

    @staticmethod
    def _gpu_active_backend():
        return _Extension._gpu_requested_backend()

    @staticmethod
    def _gpu_backend_diagnostic():
        return "synthetic native CUDA" if os.environ[
            "MEEP_GPU_BACKEND"
        ] == "cuda" else "synthetic native CPU"

    @staticmethod
    def _gpu_selected_device():
        return 0 if os.environ["MEEP_GPU_BACKEND"] == "cuda" else -1

    @staticmethod
    def _gpu_selected_device_identifier():
        return "GPU-self-test" if os.environ[
            "MEEP_GPU_BACKEND"
        ] == "cuda" else ""

    @staticmethod
    def _gpu_device_count():
        return 1

    @staticmethod
    def _gpu_device_integer_property(index, property_index):
        if index != 0:
            raise IndexError(index)
        return (0, 8, 0, 1, 1024, 1)[property_index]

    @staticmethod
    def _gpu_device_memory(index):
        return 1 << 30

    @staticmethod
    def _gpu_device_memory_bandwidth(index):
        return 1 << 30

    @staticmethod
    def _gpu_device_identifier(index):
        return "GPU-self-test"

    @staticmethod
    def _gpu_device_name(index):
        return "fake"

    @staticmethod
    def _gpu_statistics():
        class Statistics:
            pass

        value = Statistics()
        fields = '''
            runtime_availability_probes runtime_device_enumerations
            runtime_device_selections cpu_curl_calls cpu_curl_points
            cuda_curl_calls cuda_curl_points host_to_device_bytes
            device_to_host_bytes host_to_device_bytes_avoided
            device_to_host_bytes_avoided device_buffer_allocations
            device_buffer_reuses live_resident_device_buffers
            cpu_update_eh_calls cpu_update_eh_points cuda_update_eh_calls
            cuda_update_eh_points cpu_polarization_calls
            cpu_polarization_points cuda_polarization_calls
            cuda_polarization_points cpu_source_calls cpu_source_points
            cuda_source_calls cuda_source_points cpu_boundary_calls
            cpu_boundary_points cuda_boundary_calls cuda_boundary_points
            cpu_dft_calls cpu_dft_points cuda_dft_calls cuda_dft_points
            dft_batch_calls dft_submitted_updates
            dft_phase_preparation_launches dft_phase_reuses
            dft_update_kernel_launches dft_maximum_batch_size
            dft_multi_monitor_automatic_checks
            dft_multi_monitor_automatic_selected
            dft_multi_monitor_automatic_rejected
            dft_multi_monitor_forced_batches
            dft_multi_monitor_batched_updates
            dft_multi_monitor_unbatched_updates
            dft_multi_monitor_plan_uploads
            dft_multi_monitor_plan_reuses
            dft_multi_monitor_metadata_host_to_device_bytes
            cpu_dft_reduction_calls cpu_dft_reduction_pairs
            cpu_dft_reduction_terms cuda_dft_reduction_calls
            cuda_dft_reduction_pairs cuda_dft_reduction_terms
            cuda_dft_reduction_descriptor_uploads
            cuda_dft_reduction_plan_reuses
            cuda_dft_reduction_kernel_launches
            cuda_dft_reduction_result_device_to_host_bytes
            dft_reduction_full_dft_device_to_host_bytes_avoided
            dft_reduction_mpi_allreduce_calls
            dft_reduction_mpi_allreduce_bytes
            cpu_dft_array_materialization_calls
            cpu_dft_array_materialization_points
            cuda_dft_array_materialization_calls
            cuda_dft_array_materialization_points
            host_synthetic_material_array_calls
            host_synthetic_material_array_points
            cpu_dft_output_calls cpu_dft_output_points
            cuda_dft_output_calls cuda_dft_output_points
            cuda_dft_output_staging_calls
            cuda_dft_output_staging_points
            cuda_dft_output_staging_frequencies
            cuda_dft_output_staging_descriptor_uploads
            cuda_dft_output_staging_plan_reuses
            cuda_dft_output_staging_kernel_launches
            cuda_dft_output_staging_result_device_to_host_bytes
            cuda_dft_output_staging_full_dft_device_to_host_bytes_avoided
            cuda_dft_output_staging_workspace_ceiling_bytes
            cuda_dft_materialization_kernel_launches
            cuda_dft_materialization_result_device_to_host_bytes
            dft_materialization_full_dft_device_to_host_bytes_avoided
            dft_array_mpi_allreduce_calls dft_array_mpi_allreduce_bytes
            cpu_dft_checkpoint_save_calls
            cpu_dft_checkpoint_save_values
            cuda_dft_checkpoint_save_calls
            cuda_dft_checkpoint_save_values
            cuda_dft_checkpoint_save_device_to_host_bytes
            cuda_dft_checkpoint_save_full_cache_device_to_host_bytes_avoided
            cpu_dft_checkpoint_load_calls
            cpu_dft_checkpoint_load_values
            cuda_dft_checkpoint_load_calls
            cuda_dft_checkpoint_load_values
            cuda_dft_checkpoint_load_host_to_device_bytes
            cpu_dft_scale_calls cpu_dft_scale_values
            cuda_dft_scale_calls cuda_dft_scale_values
            cuda_dft_scale_kernel_launches
            cuda_dft_scale_host_to_device_bytes
            cpu_eigenmode_overlap_calls cpu_eigenmode_overlap_terms
            cuda_eigenmode_overlap_calls cuda_eigenmode_overlap_terms
            cuda_eigenmode_mode_flux_calls
            cuda_eigenmode_mode_mode_calls
            cuda_eigenmode_submitted_pairs
            cuda_eigenmode_descriptor_uploads
            cuda_eigenmode_plan_reuses
            cuda_eigenmode_kernel_launches
            cuda_eigenmode_result_device_to_host_bytes
            eigenmode_full_dft_device_to_host_bytes_avoided
            host_mode_profile_sampling_calls
            host_mode_profile_sampling_points
            eigenmode_zero_rank_channels_skipped
            host_mode_profile_host_to_device_bytes
            eigenmode_mpi_allreduce_calls eigenmode_mpi_allreduce_bytes
            cpu_ldos_reduction_calls cpu_ldos_source_points
            cuda_ldos_reduction_calls cuda_ldos_submitted_profiles
            cuda_ldos_source_points cuda_ldos_descriptor_uploads
            cuda_ldos_kernel_launches
            cuda_ldos_result_device_to_host_bytes
            ldos_full_field_device_to_host_bytes_avoided
            cpu_near2far_transform_calls cpu_near2far_terms
            cuda_near2far_transform_calls cuda_near2far_terms
            cuda_near2far_submitted_chunks cuda_near2far_source_points
            cuda_near2far_output_points cuda_near2far_frequencies
            cuda_near2far_periodic_copies
            cuda_near2far_fast_precision_calls
            cuda_near2far_mixed_precision_calls
            cuda_near2far_cancellation_retries cuda_near2far_target_tiles
            cuda_near2far_frequency_tiles cuda_near2far_operation_tiles
            cuda_near2far_maximum_workspace_bytes
            cuda_near2far_descriptor_uploads cuda_near2far_kernel_launches
            cuda_near2far_result_device_to_host_bytes
            cuda_near2far_condition_device_to_host_bytes
            near2far_dft_device_to_host_bytes_avoided
            near2far_mpi_allreduce_calls near2far_mpi_allreduce_bytes
            cpu_near2far_adjoint_calls cpu_near2far_adjoint_terms
            cuda_near2far_adjoint_calls cuda_near2far_adjoint_terms
            cuda_near2far_adjoint_submitted_chunks
            cuda_near2far_adjoint_source_points
            cuda_near2far_adjoint_far_points
            cuda_near2far_adjoint_frequencies
            cuda_near2far_adjoint_periodic_copies
            cuda_near2far_adjoint_fast_precision_calls
            cuda_near2far_adjoint_mixed_precision_calls
            cuda_near2far_adjoint_cancellation_retries
            cuda_near2far_adjoint_maximum_workspace_bytes
            cuda_near2far_adjoint_descriptor_uploads
            cuda_near2far_adjoint_kernel_launches
            cuda_near2far_adjoint_host_to_device_bytes
            cuda_near2far_adjoint_result_device_to_host_bytes
            cuda_near2far_adjoint_condition_device_to_host_bytes
            mpi_messages mpi_scalars cuda_aware_bytes pinned_staging_bytes
            pinned_device_to_host_bytes pinned_host_to_device_bytes
            mpi_waitsome_executions mpi_waitall_executions
            boundary_eh_overlap_checks boundary_eh_overlap_eligible
            boundary_eh_overlap_launched_h boundary_eh_overlap_launched_e
            boundary_eh_overlap_skipped_disabled
            boundary_eh_overlap_skipped_unsupported_schedule
            boundary_eh_overlap_skipped_no_remote
            boundary_eh_overlap_skipped_cold_topology
            boundary_eh_overlap_rejected halo_curl_overlap_checks
            halo_curl_overlap_eligible halo_curl_overlap_launches
            halo_curl_overlap_skipped_disabled
            halo_curl_overlap_skipped_unsupported_schedule
            halo_curl_overlap_skipped_no_remote
            halo_curl_overlap_skipped_cold_topology
            halo_curl_overlap_rejected_feature
            halo_curl_overlap_rejected_small halo_curl_overlap_full_points
            halo_curl_overlap_interior_points halo_curl_overlap_shell_points
            tile_coalesced_curl_chunk_phases
            tile_coalesced_curl_input_tiles
            tile_coalesced_update_eh_chunk_phases
            tile_coalesced_update_eh_input_tiles
            phase_curl_automatic_checks phase_curl_automatic_selected
            phase_curl_automatic_rejected phase_curl_forced_batches
            phase_curl_batched_operations phase_curl_unbatched_operations
            phase_curl_replay_checks phase_curl_replay_hits
            phase_curl_replay_unready phase_curl_replay_generation_misses
            phase_curl_replay_mirror_misses
            phase_update_eh_automatic_checks
            phase_update_eh_automatic_selected
            phase_update_eh_automatic_rejected
            phase_update_eh_forced_batches
            phase_update_eh_batched_operations
            phase_update_eh_unbatched_operations
        '''.split()
        for name in fields:
            setattr(value, name, 0)
        prefix = "cuda" if os.environ["MEEP_GPU_BACKEND"] == "cuda" else "cpu"
        for stem in (
            "curl", "update_eh", "source", "boundary",
        ):
            setattr(value, f"{prefix}_{stem}_calls", 2)
            setattr(value, f"{prefix}_{stem}_points", 20)
        setattr(value, f"{prefix}_near2far_transform_calls", 1)
        setattr(value, f"{prefix}_near2far_terms", 42)
        setattr(value, f"{prefix}_dft_reduction_calls", 1)
        setattr(value, f"{prefix}_dft_reduction_pairs", 3)
        setattr(value, f"{prefix}_dft_reduction_terms", 84)
        return value

_meep = _Extension()

__version__ = "self-test"

def is_single_precision():
    return True

def with_mpi():
    return True

class _Gpu:
    compiled = True
    runtime_available = True
    runtime_diagnostic = ""
    compiled_architectures = "self-test"
    backend_diagnostic = ""
    selected_device = 0

    @property
    def requested_backend(self):
        return os.environ["MEEP_GPU_BACKEND"]

    @property
    def active_backend(self):
        return os.environ["MEEP_GPU_BACKEND"]

    def devices(self):
        return [{"ordinal": 0, "name": "fake", "compatible": True}]

    def statistics(self):
        backend = os.environ["MEEP_GPU_BACKEND"]
        return {
            "dispatch": {
                "cpu_curl_calls": 3 if backend == "cpu" else 0,
                "cuda_curl_calls": 3 if backend == "cuda" else 0,
            },
            "field_updates": {
                "cpu_update_eh_calls": 2 if backend == "cpu" else 0,
                "cuda_update_eh_calls": 2 if backend == "cuda" else 0,
            },
        }

gpu = _Gpu()
"""


def base_manifest() -> dict:
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "inventory": [
            {
                "glob": "cases/*.py",
                "kind": "unittest",
                "disposition": "run",
                "reason": "Synthetic embedded oracle for harness self-test.",
                "tier": ["smoke"],
                "timeout_seconds": 30,
                "comparison": {"mode": "embedded_oracle"},
                "compute_scope": "fdtd_cuda",
                "gpu_contract": "cuda_dispatch",
                "allowed_unittest_skips": 0,
                "expected_unittest": {
                    "count": 1,
                    "identities": [
                        "pass_case.SyntheticPass.test_pass"
                    ],
                },
            }
        ],
        "overrides": [],
    }


class RunnerSelfTest(unittest.TestCase):
    def test_case_command_expands_case_path_placeholder(self):
        case = {
            "path": "cases/example.py",
            "kind": "example",
            "command": ["{python}", "{repo}/driver.py", "{path}"],
        }
        command = runner.case_command(
            case, pathlib.Path("/env/bin/python"), pathlib.Path("/repo")
        )
        self.assertEqual(
            command,
            [
                "/env/bin/python",
                "/repo/driver.py",
                "/repo/cases/example.py",
            ],
        )

    def test_checked_in_wrapper_uses_authoritative_cuda_mpi_build(self):
        wrapper = RUNNER_PATH.parents[1] / "run-python-validation.sh"
        source = wrapper.read_text(encoding="utf-8")
        self.assertIn('.envs/meep-gpu-cuda-mpi"', source)
        self.assertIn('build/meep-cuda-mpi-python-fp32/python"', source)
        self.assertIn('install/meep-cuda-mpi-python-fp32"', source)
        self.assertIn("--env PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("qualification-fontconfig.conf", source)
        self.assertIn('--env "HOME=${USER_HOME}"', source)
        self.assertIn('--env "XDG_CONFIG_HOME=${QUALIFICATION_XDG_CONFIG}"', source)
        self.assertIn('--env "FONTCONFIG_FILE=${QUALIFICATION_FONTCONFIG}"', source)
        self.assertIn("--env CUDA_CACHE_DISABLE=1", source)
        self.assertIn("--env JAX_PLATFORMS=cpu", source)
        for argument in (
            "--repo=/tmp/override",
            "--manifest=/tmp/override",
            "--build-python=/tmp/override",
            "--install-prefix=/tmp/override",
            "--python=/tmp/override",
        ):
            with self.subTest(argument=argument):
                process = subprocess.run(
                    [str(wrapper), argument],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(process.returncode, 2)
                self.assertIn(
                    "is fixed by the authoritative validation wrapper",
                    process.stderr,
                )

    def test_runner_defaults_route_to_cuda_mpi_build(self):
        args = runner.parse_args([])
        self.assertEqual(
            args.build_python,
            RUNNER_PATH.parents[2]
            / "build"
            / "meep-cuda-mpi-python-fp32"
            / "python",
        )
        self.assertEqual(
            args.install_prefix,
            RUNNER_PATH.parents[2]
            / "install"
            / "meep-cuda-mpi-python-fp32",
        )

    def test_unknown_tier_selection_fails_closed(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest, "--tier", "ghost-tier")
        self.assertEqual(status, runner.EXIT_MANIFEST_INVALID)
        self.assertFalse((root / "evidence" / "COMPLETE").exists())

    def write_fixture_receipt(self, root: pathlib.Path) -> None:
        build_python = root / "build-python"
        snapshot = source_snapshot(root)
        release_build = root / "build" / "meep-cuda-mpi-python-fp32"
        install_prefix = root / "install" / "meep-cuda-mpi-python-fp32"
        extension_path = build_python / "meep" / "_meep-self-test.so"
        libmeep_path = build_python / "libmeep.so.38.0.0"
        installed_extension = (
            install_prefix / "lib/python3.11/site-packages/meep/_meep.so.38.0.0"
        )
        installed_mpb_extension = (
            install_prefix
            / "lib/python3.11/site-packages/meep/mpb/_mpb.so.38.0.0"
        )
        installed_libmeep = install_prefix / "lib/libmeep.so.38.0.0"
        installed_libpympb = install_prefix / "lib/libpympb.so.38.0.0"
        for path, contents in (
            (installed_extension, b"installed extension\n"),
            (installed_mpb_extension, b"installed MPB extension\n"),
            (installed_libmeep, b"installed libmeep\n"),
            (installed_libpympb, b"installed libpympb\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        artifact_paths = {
            "cuda_architecture_test": extension_path,
            "cuda_formula_test": extension_path,
            "cuda_near2far_runtime_validation": extension_path,
            "cuda_runtime_validation": extension_path,
            "cuda_smoke": extension_path,
            "fd_allocation_shim": extension_path,
            "python_extension": extension_path,
            "libmeep": libmeep_path,
            "installed_python_extension": installed_extension,
            "installed_libmeep": installed_libmeep,
            "installed_mpb_extension": installed_mpb_extension,
            "installed_libpympb": installed_libpympb,
            "gpu_backend_test": extension_path,
            "gpu_step_db_test": extension_path,
            "gpu_mpi_performance": extension_path,
        }
        self.assertEqual(tuple(sorted(artifact_paths)), REQUIRED_ARTIFACT_BINDINGS)
        qualification_root = build_python / "qualification-logs"
        qualification_root.mkdir(parents=True, exist_ok=True)
        (qualification_root / AUDIT_LOG_NAME).unlink(missing_ok=True)
        for name in REQUIRED_QUALIFICATION_LOGS:
            if name == STEP_FINITE_FAILURE_LOG_NAME:
                lines = [
                    "gpmeep-finite-abort-probe:rank=0,stage=before-failing-step",
                    "gpmeep-finite-abort-probe:rank=1,stage=before-failing-step",
                    f"meep: {STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    f"meep: {STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    f"gpmeep-expected-mpi-failure:{name}:status=1:"
                    f"diagnostic={STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    marker_for(name).decode("ascii"),
                ]
                (qualification_root / name).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            else:
                (qualification_root / name).write_bytes(marker_for(name) + b"\n")
        for name, (extension_name, libmeep_name) in (
            PYTHON_RUNTIME_LOG_BINDINGS.items()
        ):
            if name in INSTALLED_CUDA_QUALIFICATION_SPECS:
                continue
            attestation = {
                "python_extension": file_record(
                    artifact_paths[extension_name], root
                ),
                "libmeep": file_record(artifact_paths[libmeep_name], root),
            }
            (qualification_root / name).write_text(
                json.dumps(
                    attestation, sort_keys=True, separators=(",", ":")
                )
                + "\n"
                + marker_for(name).decode("ascii")
                + "\n",
                encoding="utf-8",
            )
        write_synthetic_auxiliary_logs(qualification_root, root, artifact_paths)
        for name, artifact_name in DIRECT_ELF_LOG_BINDINGS.items():
            executable = artifact_paths[artifact_name].resolve()
            libmeep = artifact_paths["libmeep"].resolve()
            executable_record = file_record(executable, root)
            libmeep_record = file_record(libmeep, root)
            (qualification_root / name).write_text(
                f"executable={executable}\n"
                f"loaded_libmeep={libmeep}\n"
                f"{executable_record['sha256']}  {executable}\n"
                f"{libmeep_record['sha256']}  {libmeep}\n"
                "synthetic ldd output\n"
                + marker_for(name).decode("ascii")
                + "\n",
                encoding="utf-8",
            )
        qualification_pycache = build_python / "qualification-pycache"
        qualification_pycache.mkdir(mode=0o700, exist_ok=True)
        qualification_pycache.chmod(0o700)
        if any(qualification_pycache.iterdir()):
            raise AssertionError("qualification pycache fixture is not empty")
        cuda_math = write_synthetic_release_cuda_math_configuration(
            root, release_build, install_prefix
        )
        configure_argv = cuda_math["configure_argv"]
        nvcc_path = cuda_math["nvcc"]
        cuda_math_files = cuda_math["configuration_files"]
        canonical_build_environment = cuda_math_files[
            "canonical_build_environment"
        ]
        config_status = cuda_math_files["config_status"]
        cuda_runtime_cmake_cache = cuda_math_files[
            "cuda_runtime_cmake_cache"
        ]
        cuda_runtime_flags_stamp = cuda_math_files[
            "cuda_runtime_flags_stamp"
        ]
        identity = create_identity_snapshot(
            root,
            root / ".envs" / "meep-gpu-cuda-mpi",
            artifact_paths,
            installed_prefix=install_prefix,
            immutable_empty_directories={
                "qualification_pycache": qualification_pycache,
            },
        )
        sentinel_name = "qualification-mutation-sentinel.log"
        (qualification_root / sentinel_name).write_text(
            json.dumps(
                {
                    "identity_sha256": canonical_sha256(identity),
                    "watch_count": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            + marker_for(sentinel_name).decode("ascii")
            + "\n",
            encoding="utf-8",
        )
        identity_before = build_python / "qualification-identity-before.json"
        identity_after = build_python / "qualification-identity-after.json"
        qualification_seal = build_python / "qualification-contract-v2.json"
        atomic_write_json(identity_before, identity)
        atomic_write_json(identity_after, identity)
        seal_contract(
            root, qualification_root, identity_before, identity_after,
            qualification_seal,
        )
        self.assertEqual(
            sorted(path.name for path in qualification_root.iterdir()),
            list(ALL_QUALIFICATION_DIRECTORY_LOGS),
        )
        configuration = {
            "builder": file_record(
                root / "scripts" / "build-meep-cuda-mpi-python.sh", root
            ),
            "qualification_contract": "gpmeep-cuda-mpi-python-fp32-v2",
            "configure_argv": configure_argv,
            "environment": {"MEEP_GPU_FAST_MATH": "OFF"},
            "lockfiles": {},
        }
        receipt = {
            "schema_version": 1,
            "state": "complete",
            "build_kind": "cuda-mpi-python-fp32",
            "started_at_utc": "2000-01-01T00:00:00Z",
            "completed_at_utc": "2000-01-01T00:00:01Z",
            "repo": str(root.resolve()),
            "build_dir": str(release_build.resolve()),
            "source_start": snapshot,
            "source_end": snapshot,
            "source_unchanged": True,
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip(),
            "git_status_porcelain": subprocess.run(
                [
                    "git", "status", "--porcelain=v1",
                    "--untracked-files=all",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.splitlines(),
            "configuration": configuration,
            "configuration_files": {
                "canonical_build_environment": file_record(
                    canonical_build_environment, root
                ),
                "config_status": file_record(config_status, root),
                "cuda_runtime_cmake_cache": file_record(
                    cuda_runtime_cmake_cache, root
                ),
                "cuda_runtime_flags_stamp": file_record(
                    cuda_runtime_flags_stamp, root
                ),
                "qualification_fontconfig": file_record(
                    root / "qualification-fontconfig.conf", root
                ),
                "qualification_identity_before": file_record(
                    identity_before, root
                ),
                "qualification_identity_after": file_record(
                    identity_after, root
                ),
                "qualification_contract_v2": file_record(
                    qualification_seal, root
                ),
            },
            "toolchain": {
                "nvcc": file_record(nvcc_path, root),
                "python": file_record(pathlib.Path(sys.executable).resolve(), root),
            },
            "artifacts": {
                name: file_record(path, root)
                for name, path in artifact_paths.items()
            },
            "manifests": {
                "in_place_python": tree_manifest(build_python / "meep", root),
                "installed_environment": tree_manifest(
                    root / ".envs" / "meep-gpu-cuda-mpi", root,
                    excluded_suffixes=(".pyc",),
                ),
                "installed_prefix": tree_manifest(install_prefix, root),
                "build_home": tree_manifest(
                    root / ".micromamba" / "cache" / "build-home-fixture",
                    root,
                ),
                "qualification_home": tree_manifest(
                    root / "qualification-home", root
                ),
                "qualification_logs": tree_manifest(
                    qualification_root, root
                ),
            },
        }
        receipt["build_input_id"] = canonical_sha256(
            {
                "schema_version": receipt["schema_version"],
                "build_kind": receipt["build_kind"],
                "source_start": snapshot,
                "configuration": configuration,
            }
        )
        receipt["artifact_set_id"] = canonical_sha256(
            {
                "configuration_files": receipt["configuration_files"],
                "toolchain": receipt["toolchain"],
                "artifacts": receipt["artifacts"],
                "manifests": receipt["manifests"],
            }
        )
        receipt["receipt_id"] = canonical_sha256(receipt)
        atomic_write_json(root / "build-provenance.json", receipt)

    def reseal_fixture_receipt(
        self, root: pathlib.Path, receipt: dict
    ) -> None:
        receipt["build_input_id"] = canonical_sha256(
            {
                "schema_version": receipt["schema_version"],
                "build_kind": receipt["build_kind"],
                "source_start": receipt["source_start"],
                "configuration": receipt["configuration"],
            }
        )
        receipt["artifact_set_id"] = canonical_sha256(
            {
                "configuration_files": receipt["configuration_files"],
                "toolchain": receipt["toolchain"],
                "artifacts": receipt["artifacts"],
                "manifests": receipt["manifests"],
            }
        )
        receipt.pop("receipt_id", None)
        receipt["receipt_id"] = canonical_sha256(receipt)
        atomic_write_json(root / "build-provenance.json", receipt)

    def make_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        (root / "cases").mkdir()
        (root / "scripts").mkdir()
        (root / "scripts" / "build-meep-cuda-mpi-python.sh").write_text(
            "#!/bin/bash -p\nexit 0\n", encoding="utf-8"
        )
        (root / "python" / "tests").mkdir(parents=True)
        (root / "python" / "tests" / "test_gpu_backend.py").write_text(
            "# GPU backend fixture\n", encoding="utf-8"
        )
        (
            root
            / "python"
            / "tests"
            / "test_adjoint_default_material_grid.py"
        ).write_text("# adjoint fixture\n", encoding="utf-8")
        for relative, _selectors, _tests_run in INSTALLED_LAZY_API_TEST_SPECS:
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                source.write_text(
                    "# installed lazy API fixture\n", encoding="utf-8"
                )
        environment_prefix = root / ".envs" / "meep-gpu-cuda-mpi"
        environment_prefix.mkdir(parents=True)
        (environment_prefix / "fixture-package.txt").write_text(
            "fixture\n", encoding="utf-8"
        )
        build_home = root / ".micromamba" / "cache" / "build-home-fixture"
        (build_home / "cache" / "fontconfig").mkdir(parents=True)
        (build_home / "cache" / "fontconfig" / "fixture-cache.txt").write_text(
            "fixture\n", encoding="utf-8"
        )
        (root / "qualification-home").mkdir()
        (root / "qualification-home" / ".gpmeep-empty-home").write_text(
            "fixture\n", encoding="utf-8"
        )
        (root / "qualification-fontconfig.conf").write_text(
            textwrap.dedent(
                f"""
                <?xml version="1.0"?>
                <fontconfig>
                  <dir>{environment_prefix / "fonts"}</dir>
                  <include ignore_missing="yes">{environment_prefix / "etc" / "fonts" / "conf.d"}</include>
                  <cachedir prefix="xdg">fontconfig</cachedir>
                </fontconfig>
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (root / "build-python" / "meep").mkdir(parents=True)
        (root / "build-python" / "meep" / "__init__.py").write_text(
            textwrap.dedent(FAKE_MEEP), encoding="utf-8"
        )
        synthetic_source = root / "build-python" / "synthetic.c"
        synthetic_source.write_text("int gpmeep_fixture(void) { return 1; }\n")
        subprocess.run(
            [
                "cc", "-shared", "-fPIC", str(synthetic_source), "-o",
                str(root / "build-python" / "meep" / "_meep-self-test.so"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "cc", "-shared", "-fPIC", str(synthetic_source), "-o",
                str(root / "build-python" / "libmeep.so.38.0.0"),
            ],
            check=True,
        )
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(base_manifest()), encoding="utf-8")
        (root / ".gitignore").write_text(
            "evidence/\nbuild-python/\ninstall/\nbuild-provenance.json\n"
            "build/\n.envs/meep-gpu-cuda-mpi/bin/\n"
            "__pycache__/\n*.pyc\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "init", "--quiet"], cwd=root, check=True
        )
        subprocess.run(
            ["git", "add", "."], cwd=root, check=True
        )
        subprocess.run(
            [
                "git", "-c", "user.email=test@example.invalid", "-c",
                "user.name=Validation Test", "commit", "-qm", "fixture",
            ],
            cwd=root,
            check=True,
        )
        self.write_fixture_receipt(root)
        return temporary, root, manifest_path

    def invoke(
        self,
        root: pathlib.Path,
        manifest: pathlib.Path,
        *extra: str,
        refresh_receipt: bool = True,
    ) -> int:
        # Individual self-tests intentionally rewrite tracked fixture inputs.
        # Model a real build after those edits so validation starts from a
        # source-bound receipt; tests which deliberately remove a build
        # artifact opt out below.
        if refresh_receipt:
            self.write_fixture_receipt(root)
        return runner.main(
            [
                "--repo",
                str(root),
                "--manifest",
                str(manifest),
                "--build-python",
                str(root / "build-python"),
                "--install-prefix",
                str(root / "install" / "meep-cuda-mpi-python-fp32"),
                "--python",
                sys.executable,
                "--output",
                str(root / "evidence"),
                *extra,
            ]
        )

    def test_self_consistent_outer_receipt_cannot_forge_terminal_pass_log(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        log_root = root / "build-python" / "qualification-logs"
        target = log_root / REQUIRED_QUALIFICATION_LOGS[0]
        target.write_bytes(target.read_bytes() + b"forged-after-pass\n")
        receipt_path = root / "build-provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["manifests"]["qualification_logs"] = tree_manifest(
            log_root, root
        )
        self.reseal_fixture_receipt(root, receipt)

        status = self.invoke(
            root, manifest, "--backend", "cpu", refresh_receipt=False
        )

        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["provenance"]["validation_window"]["problems"]
        self.assertTrue(
            any("terminal PASS marker" in problem for problem in problems),
            problems,
        )

    def test_outer_receipt_cannot_forge_python_runtime_artifact_hash(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        log_root = root / "build-python" / "qualification-logs"
        name = "in-place-python-runtime-provenance.log"
        target = log_root / name
        lines = target.read_text(encoding="utf-8").splitlines()
        attestation = json.loads(lines[-2])
        attestation["python_extension"]["sha256"] = "0" * 64
        target.write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":"))
            + "\n"
            + marker_for(name).decode("ascii")
            + "\n",
            encoding="utf-8",
        )
        receipt_path = root / "build-provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["manifests"]["qualification_logs"] = tree_manifest(
            log_root, root
        )
        self.reseal_fixture_receipt(root, receipt)

        status = self.invoke(
            root, manifest, "--backend", "cpu", refresh_receipt=False
        )

        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["provenance"]["validation_window"]["problems"]
        self.assertTrue(
            any(
                "Python runtime loaded artifact is not receipt-bound" in problem
                for problem in problems
            ),
            problems,
        )

    def test_both_backends_capture_stats_and_compare(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest)
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(report["schema_version"], runner.REPORT_SCHEMA_VERSION)
        result = report["results"][0]
        self.assertEqual(result["compute_scope"], "fdtd_cuda")
        self.assertEqual(result["comparison"]["outcome"], "PASS")
        self.assertTrue(result["runs"]["cpu"]["backend_contract_ok"])
        self.assertTrue(result["runs"]["cuda"]["backend_contract_ok"])
        self.assertEqual(
            result["runs"]["cuda"]["statistics"]["active_backend"], "cuda"
        )
        self.assertEqual(
            result["runs"]["cuda"]["statistics"]["statistics"]["dispatch"][
                "cpu_curl_calls"
            ],
            0,
        )
        self.assertTrue((root / "evidence" / "report.md").is_file())
        complete = json.loads((root / "evidence" / "COMPLETE").read_text())
        self.assertEqual(
            complete["schema_version"], runner.REPORT_SCHEMA_VERSION
        )
        self.assertEqual(complete["exit_code"], runner.EXIT_OK)
        self.assertEqual(
            complete["report_sha256"],
            hashlib.sha256(
                (root / "evidence" / "report.json").read_bytes()
            ).hexdigest(),
        )
        self.assertTrue(
            report["provenance"]["validation_window"]["unchanged"]
        )
        cpu_environment = result["runs"]["cpu"]["environment_contract"]
        cuda_environment = result["runs"]["cuda"]["environment_contract"]
        self.assertNotEqual(
            cpu_environment["XDG_CACHE_HOME"],
            cuda_environment["XDG_CACHE_HOME"],
        )
        for environment in (cpu_environment, cuda_environment):
            self.assertEqual(
                environment["FONTCONFIG_FILE"],
                str(root / "qualification-fontconfig.conf"),
            )
            cache = pathlib.Path(environment["XDG_CACHE_HOME"])
            self.assertTrue(cache.is_relative_to(root / "evidence"))
            self.assertFalse(
                cache.is_relative_to(
                    root / ".micromamba" / "cache" / "build-home-fixture"
                )
            )
        for run in result["runs"].values():
            for record in run["evidence_files"].values():
                self.assertTrue(record["available"])
                path = pathlib.Path(record["path"])
                self.assertFalse(path.is_absolute())
                if not path.is_absolute():
                    path = root / "evidence" / path
                self.assertEqual(record["size_bytes"], path.stat().st_size)
                self.assertEqual(
                    record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
        probe = report["provenance"]["build_package_probe"]
        self.assertTrue(probe["isolated_runtime"])
        for value in probe["runtime_environment"].values():
            if value is not None and "qualification-fontconfig" not in value:
                self.assertTrue(
                    pathlib.Path(value).is_relative_to(root / "evidence")
                )
        performance = report["configuration"]["performance_evidence"]
        self.assertFalse(performance["valid_for_speed_gate"])
        self.assertFalse(performance["concurrency_detected"])
        markdown = (root / "evidence" / "report.md").read_text()
        self.assertIn(
            "Observed CPU/CUDA ratio (non-gating)",
            markdown,
        )
        self.assertIn("## Follow-up performance candidates", markdown)
        self.assertIn(
            "explicit optimization candidates, not speed-gate conclusions",
            markdown,
        )
        self.assertNotIn("| Speedup |", markdown)

    def test_required_near2far_cuda_counter_is_enforced_end_to_end(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["inventory"][0]["required_cuda_call_counters"] = [
            "near2far.cuda_near2far_transform_calls"
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")

        status = self.invoke(root, manifest)

        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = report["results"][0]
        self.assertEqual(
            result["runs"]["cpu"]["statistics"]["statistics"]["near2far"][
                "cpu_near2far_transform_calls"
            ],
            1,
        )
        self.assertEqual(
            result["runs"]["cuda"]["statistics"]["statistics"]["near2far"][
                "cuda_near2far_transform_calls"
            ],
            1,
        )

        module = root / "build-python" / "meep" / "__init__.py"
        source = module.read_text(encoding="utf-8")
        module.write_text(
            source.replace(
                '        setattr(value, f"{prefix}_near2far_transform_calls", 1)\n',
                '        setattr(value, f"{prefix}_near2far_transform_calls", 0)\n',
            ),
            encoding="utf-8",
        )

        failing_output = root / "evidence" / "near2far-missing"
        status = self.invoke(
            root, manifest, "--output", str(failing_output)
        )

        self.assertEqual(status, runner.EXIT_BACKEND_CONTRACT)
        report = json.loads((failing_output / "report.json").read_text())
        cpu_problems = report["results"][0]["runs"]["cpu"][
            "backend_contract_problems"
        ]
        cuda_problems = report["results"][0]["runs"]["cuda"][
            "backend_contract_problems"
        ]
        self.assertIn(
            "CPU reference did not execute paired counter "
            "'near2far.cpu_near2far_transform_calls'",
            cpu_problems,
        )
        self.assertIn(
            "required CUDA activity counter "
            "'near2far.cuda_near2far_transform_calls' did not execute",
            cuda_problems,
        )

    def test_concurrent_gpu_work_invalidates_timing_explicitly(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest, "--concurrent-gpu-work")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        performance = report["configuration"]["performance_evidence"]
        self.assertFalse(performance["valid_for_speed_gate"])
        self.assertTrue(performance["concurrency_detected"])
        self.assertIn("concurrency_detected", performance["invalid_reasons"])

    def test_required_dft_reduction_counter_is_enforced_end_to_end(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["inventory"][0]["required_cuda_call_counters"] = [
            "dft_reductions.cuda_dft_reduction_calls"
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")

        status = self.invoke(root, manifest)
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = report["results"][0]
        self.assertEqual(
            result["runs"]["cpu"]["statistics"]["statistics"]
            ["dft_reductions"]["cpu_dft_reduction_calls"],
            1,
        )
        self.assertEqual(
            result["runs"]["cuda"]["statistics"]["statistics"]
            ["dft_reductions"]["cuda_dft_reduction_calls"],
            1,
        )

        module = root / "build-python" / "meep" / "__init__.py"
        source = module.read_text(encoding="utf-8")
        module.write_text(
            source.replace(
                '        setattr(value, f"{prefix}_dft_reduction_calls", 1)\n',
                '        setattr(value, f"{prefix}_dft_reduction_calls", 0)\n',
            ),
            encoding="utf-8",
        )
        failing_output = root / "evidence" / "dft-reduction-missing"
        status = self.invoke(root, manifest, "--output", str(failing_output))
        self.assertEqual(status, runner.EXIT_BACKEND_CONTRACT)
        report = json.loads((failing_output / "report.json").read_text())
        cpu_problems = report["results"][0]["runs"]["cpu"][
            "backend_contract_problems"
        ]
        cuda_problems = report["results"][0]["runs"]["cuda"][
            "backend_contract_problems"
        ]
        self.assertIn(
            "CPU reference did not execute paired counter "
            "'dft_reductions.cpu_dft_reduction_calls'",
            cpu_problems,
        )
        self.assertIn(
            "required CUDA activity counter "
            "'dft_reductions.cuda_dft_reduction_calls' did not execute",
            cuda_problems,
        )

    def test_markdown_labels_non_gating_ratios_and_lists_slow_candidates(self):
        report = {
            "generated_at_utc": "2026-08-07T00:00:00Z",
            "configuration": {
                "backends": ["cpu", "cuda"],
                "performance_evidence": {
                    "valid_for_speed_gate": False,
                    "concurrency_detected": False,
                    "invalid_reasons": ["single sample"],
                },
            },
            "provenance": {
                "git_head": {"stdout": "deadbeef"},
                "manifest": "manifest.json",
                "python_executable": "/python",
                "build_python": "/build/python",
                "validation_window": {"unchanged": True},
            },
            "exit_code": 0,
            "summary": {"PASS": 3},
            "results": [
                {
                    "path": "examples/slow.py",
                    "compute_scope": "fdtd_cuda",
                    "outcome": "PASS",
                    "selected": True,
                    "runs": {
                        "cpu": {"outcome": "PASS"},
                        "cuda": {"outcome": "PASS"},
                    },
                    "comparison": {
                        "outcome": "PASS",
                        "speedup_cpu_over_cuda": 0.25,
                    },
                },
                {
                    "path": "examples/fast.py",
                    "compute_scope": "fdtd_cuda",
                    "outcome": "PASS",
                    "selected": True,
                    "runs": {
                        "cpu": {"outcome": "PASS"},
                        "cuda": {"outcome": "PASS"},
                    },
                    "comparison": {
                        "outcome": "PASS",
                        "speedup_cpu_over_cuda": 2.0,
                    },
                },
                {
                    "path": "examples/host.py",
                    "compute_scope": "host_only",
                    "outcome": "PASS",
                    "selected": True,
                    "runs": {
                        "cpu": {"outcome": "PASS"},
                        "cuda": {"outcome": "PASS"},
                    },
                    "comparison": {
                        "outcome": "PASS",
                        "speedup_cpu_over_cuda": 0.1,
                    },
                },
            ],
        }
        markdown = runner.markdown_report(report)
        self.assertIn("Observed CPU/CUDA ratio (non-gating)", markdown)
        self.assertNotIn("| Speedup |", markdown)
        candidates = markdown.split(
            "## Follow-up performance candidates", 1
        )[1].split("## Interpretation", 1)[0]
        self.assertIn("`examples/slow.py`", candidates)
        self.assertIn("observed non-gating ratio `0.250×`", candidates)
        self.assertNotIn("`examples/fast.py`", candidates)
        self.assertNotIn("`examples/host.py`", candidates)

    def test_raw_evidence_is_fsynced_before_report_and_complete(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        events: list[str] = []
        original_fsync = runner.fsync_evidence_directories
        original_write = runner.atomic_write_text

        def recording_fsync(output, results):
            events.append("fsync:raw-evidence")
            return original_fsync(output, results)

        def recording_write(path, value):
            events.append(f"write:{path.name}")
            return original_write(path, value)

        with (
            mock.patch.object(
                runner,
                "fsync_evidence_directories",
                side_effect=recording_fsync,
            ),
            mock.patch.object(
                runner, "atomic_write_text", side_effect=recording_write
            ),
        ):
            status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_OK)
        raw_index = events.index("fsync:raw-evidence")
        report_index = events.index("write:report.json")
        complete_index = events.index("write:COMPLETE")
        self.assertLess(raw_index, report_index)
        self.assertLess(report_index, complete_index)

    def test_nonempty_evidence_output_cannot_be_reused(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import os
                import sys
                import unittest
                import meep

                if os.environ.get("GPMEEP_SELF_TEST_OS_EXIT") == "1":
                    sys.stderr.write(
                        "test_pass (pass_case.SyntheticPass.test_pass) ... ok\\n"
                        "\\n----------------------------------------"
                        "------------------------------\\n"
                        "Ran 1 test in 0.000s\\n\\nOK\\n"
                    )
                    sys.stderr.flush()
                    os._exit(0)

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        fixed_uuid = mock.Mock(hex="1" * 32)
        with mock.patch.object(runner.uuid, "uuid4", return_value=fixed_uuid):
            first_status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(first_status, runner.EXIT_OK)
        stale_stats = (
            root
            / "evidence"
            / "runs"
            / runner.case_directory_id("cases/pass_case.py")
            / "cpu"
            / "gpu-statistics.json"
        )
        self.assertTrue(stale_stats.is_file())

        with (
            mock.patch.dict(
                os.environ, {"GPMEEP_SELF_TEST_OS_EXIT": "1"}, clear=False
            ),
            mock.patch.object(runner.uuid, "uuid4", return_value=fixed_uuid),
        ):
            second_status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(second_status, runner.EXIT_MANIFEST_INVALID)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cpu"]
        self.assertEqual(run["outcome"], "PASS")
        self.assertTrue(run["evidence_files"]["statistics"]["available"])
        self.assertTrue(stale_stats.exists())

    def test_nonce_mismatch_is_rejected_by_full_runner(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import os
                import unittest
                import meep

                os.environ["GPMEEP_VALIDATION_RUN_NONCE"] = "wrong-nonce"

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_BACKEND_CONTRACT)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cpu"]
        self.assertEqual(run["outcome"], "BACKEND_CONTRACT_FAILED")
        self.assertIn(
            "GPU evidence nonce does not match this child run",
            run["backend_contract_problems"],
        )

    def test_python_gpu_controller_monkeypatch_cannot_forge_native_statistics(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                meep.gpu.statistics = lambda: {
                    "dispatch": {
                        "cpu_curl_calls": 0,
                        "cuda_curl_calls": 999999,
                    }
                }

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertEqual(meep.gpu.statistics()["dispatch"]["cuda_curl_calls"], 999999)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cuda")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cuda"]
        native_dispatch = run["statistics"]["statistics"]["dispatch"]
        self.assertEqual(native_dispatch["cuda_curl_calls"], 2)
        self.assertNotEqual(native_dispatch["cuda_curl_calls"], 999999)

    def test_native_gpu_callable_monkeypatch_fails_closed(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                meep._meep._gpu_statistics = lambda: object()

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cuda")
        self.assertEqual(status, runner.EXIT_BACKEND_CONTRACT)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cuda"]
        self.assertEqual(
            run["statistics"]["capture_status"], "capture_error"
        )
        self.assertIn(
            "sealed native Meep callable identity changed: _gpu_statistics",
            run["statistics"]["capture_error"],
        )

    def test_runtime_and_precision_tampering_fail_closed(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest, "--backend", "cuda")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = next(item for item in report["results"] if item["selected"])
        case = runner.materialize_cases(runner.load_json(manifest), root)[0]
        run = result["runs"]["cuda"]
        stats = run["statistics"]
        expected_runtime = report["provenance"]["validation_window"]["start"][
            "runtime_contract"
        ]

        mutations = []
        for artifact in (
            "python_executable",
            "meep_module",
            "extension",
            "libmeep",
        ):
            mutations.extend(
                (
                    (
                        f"{artifact}_path",
                        lambda value, name=artifact: value[name].__setitem__(
                            "path", "/tmp/not-the-loaded-artifact"
                        ),
                        f"loaded {artifact} path differs",
                    ),
                    (
                        f"{artifact}_sha256",
                        lambda value, name=artifact: value[name].__setitem__(
                            "sha256", "0" * 64
                        ),
                        f"loaded {artifact} SHA-256 differs",
                    ),
                )
            )
        mutations.extend(
            (
                (
                    "single_precision",
                    lambda value: value.__setitem__("single_precision", False),
                    "not single precision",
                ),
                (
                    "with_mpi",
                    lambda value: value.__setitem__("with_mpi", False),
                    "not MPI-enabled",
                ),
                (
                    "gpu_compiled",
                    lambda value: value.__setitem__("gpu_compiled", False),
                    "not CUDA-enabled",
                ),
                (
                    "strict_marker",
                    lambda value: value.__setitem__("strict_cuda_marker", False),
                    "strict CUDA marker",
                ),
                (
                    "receipt_id",
                    lambda value: value.__setitem__(
                        "build_receipt_id", "0" * 64
                    ),
                    "build receipt ID mismatch",
                ),
            )
        )
        for label, mutate, expected_problem in mutations:
            with self.subTest(label=label):
                candidate = copy.deepcopy(stats)
                mutate(candidate)
                ok, problems = runner.verify_backend_contract(
                    case,
                    "cuda",
                    candidate,
                    expected_runtime,
                    run["run_nonce"],
                )
                self.assertFalse(ok)
                self.assertTrue(
                    any(expected_problem in problem for problem in problems),
                    problems,
                )

    def test_cuda_contract_can_intentionally_release_backend_at_exit(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest, "--backend", "cuda")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = next(item for item in report["results"] if item["selected"])
        run = result["runs"]["cuda"]
        stats = copy.deepcopy(run["statistics"])
        stats["selected_device"] = -1
        stats["requested_backend"] = "cpu"
        stats["active_backend"] = "cpu"
        case = runner.materialize_cases(runner.load_json(manifest), root)[0]
        case["require_active_backend_at_exit"] = False
        case["gpu_contract"] = "none"
        expected_runtime = report["provenance"]["validation_window"]["start"][
            "runtime_contract"
        ]

        ok, problems = runner.verify_backend_contract(
            case,
            "cuda",
            stats,
            expected_runtime,
            run["run_nonce"],
        )

        self.assertTrue(ok, problems)
        self.assertEqual(problems, [])

    def test_required_cuda_only_and_shared_activity_backend_semantics(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest)
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = next(item for item in report["results"] if item["selected"])
        expected_runtime = report["provenance"]["validation_window"]["start"][
            "runtime_contract"
        ]
        case = runner.materialize_cases(runner.load_json(manifest), root)[0]
        cuda_only = (
            "dft_materializations."
            "cuda_dft_materialization_kernel_launches"
        )
        shared = "dft_materializations.dft_array_mpi_allreduce_calls"
        case["required_cuda_call_counters"] = [cuda_only, shared]

        candidates = {}
        for backend in ("cpu", "cuda"):
            run = result["runs"][backend]
            stats = copy.deepcopy(run["statistics"])
            materializations = stats["statistics"]["dft_materializations"]
            materializations[
                "cuda_dft_materialization_kernel_launches"
            ] = 2 if backend == "cuda" else 0
            materializations["dft_array_mpi_allreduce_calls"] = 1
            ok, problems = runner.verify_backend_contract(
                case,
                backend,
                stats,
                expected_runtime,
                run["run_nonce"],
            )
            self.assertTrue(ok, problems)
            candidates[backend] = (run, stats)

        cuda_run, cuda_stats = candidates["cuda"]
        cuda_stats["statistics"]["dft_materializations"][
            "cuda_dft_materialization_kernel_launches"
        ] = True
        ok, problems = runner.verify_backend_contract(
            case,
            "cuda",
            cuda_stats,
            expected_runtime,
            cuda_run["run_nonce"],
        )
        self.assertFalse(ok)
        self.assertIn(
            f"required CUDA activity counter {cuda_only!r} did not execute",
            problems,
        )

        cpu_run, cpu_stats = candidates["cpu"]
        cpu_stats["statistics"]["dft_materializations"][
            "dft_array_mpi_allreduce_calls"
        ] = 0
        ok, problems = runner.verify_backend_contract(
            case,
            "cpu",
            cpu_stats,
            expected_runtime,
            cpu_run["run_nonce"],
        )
        self.assertFalse(ok)
        self.assertIn(
            "CPU reference did not execute shared required activity "
            f"{shared!r}",
            problems,
        )

    def test_phase_call_counters_reject_cancellation_and_schema_forgery(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        status = self.invoke(root, manifest, "--backend", "cuda")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = next(item for item in report["results"] if item["selected"])
        run = result["runs"]["cuda"]
        expected_runtime = report["provenance"]["validation_window"]["start"][
            "runtime_contract"
        ]
        case = runner.materialize_cases(runner.load_json(manifest), root)[0]

        attacks = {
            "negative-cancellation": {
                "cpu_forged_positive_calls": 7,
                "cpu_forged_negative_calls": -7,
            },
            "boolean": {"cpu_forged_calls": False},
            "uint64-overflow": {"cpu_forged_calls": 1 << 64},
            "unknown-zero": {"cpu_forged_calls": 0},
        }
        for label, additions in attacks.items():
            with self.subTest(label=label):
                stats = copy.deepcopy(run["statistics"])
                stats["statistics"]["dispatch"].update(additions)
                ok, problems = runner.verify_backend_contract(
                    case,
                    "cuda",
                    stats,
                    expected_runtime,
                    run["run_nonce"],
                )
                self.assertFalse(ok)
                self.assertTrue(
                    any(
                        "phase call counter" in problem
                        for problem in problems
                    ),
                    problems,
                )

    def test_inherited_backend_python_omp_and_cuda_tuning_are_removed(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                f"""
                import os
                import pathlib
                import unittest
                import meep

                class SyntheticCleanEnvironment(unittest.TestCase):
                    def test_environment(self):
                        for key in (
                            "MEEP_GPU_POISON",
                            "GPMEEP_VALIDATION_POISON",
                            "OMP_POISON",
                            "CUDA_TUNING_POISON",
                            "JAX_POISON",
                            "LD_PRELOAD",
                            "LD_AUDIT",
                            "PYTHONHOME",
                            "BASH_ENV",
                            "ENV",
                        ):
                            self.assertNotIn(key, os.environ)
                        self.assertNotIn("FONTCONFIG_PATH", os.environ)
                        self.assertNotIn("FONTCONFIG_SYSROOT", os.environ)
                        self.assertNotIn("poison-python-path", os.environ["PYTHONPATH"])
                        self.assertEqual(os.environ["MEEP_GPU_BACKEND"], "cpu")
                        self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
                        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
                        self.assertEqual(os.environ["CUDA_CACHE_DISABLE"], "1")
                        self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
                        runtime_home = pathlib.Path(os.environ["HOME"])
                        runtime_cache = pathlib.Path(os.environ["XDG_CACHE_HOME"])
                        runtime_config = pathlib.Path(os.environ["XDG_CONFIG_HOME"])
                        runtime_matplotlib = pathlib.Path(os.environ["MPLCONFIGDIR"])
                        for path in (
                            runtime_home,
                            runtime_cache,
                            runtime_config,
                            runtime_matplotlib,
                        ):
                            self.assertTrue(path.is_relative_to({str(root / "evidence")!r}))
                            self.assertFalse(path.is_relative_to({str(root / ".micromamba" / "cache" / "build-home-fixture")!r}))
                        (runtime_cache / "fontconfig").mkdir(parents=True)
                        (runtime_cache / "fontconfig" / "synthetic-cache-9").write_text("cache\\n")
                        self.assertEqual(
                            os.environ["FONTCONFIG_FILE"],
                            {str(root / "qualification-fontconfig.conf")!r},
                        )
                """
            ),
            encoding="utf-8",
        )
        inherited = {
            "PYTHONPATH": "poison-python-path",
            "MEEP_GPU_POISON": "1",
            "GPMEEP_VALIDATION_POISON": "1",
            "OMP_POISON": "1",
            "CUDA_TUNING_POISON": "1",
            "CUDA_VISIBLE_DEVICES": "7",
            "JAX_POISON": "1",
            "JAX_PLATFORMS": "cuda",
            "FONTCONFIG_FILE": "/poison/fonts.conf",
            "FONTCONFIG_PATH": "/poison/fonts",
            "FONTCONFIG_SYSROOT": "/poison/root",
            "XDG_CACHE_HOME": "/poison/cache",
            "XDG_CONFIG_HOME": "/poison/config",
            "MPLCONFIGDIR": "/poison/matplotlib",
            "LD_PRELOAD": "/poison/preload.so",
            "LD_AUDIT": "/poison/audit.so",
            "PYTHONHOME": "/poison/python-home",
            "BASH_ENV": "/poison/bash-env",
            "ENV": "/poison/sh-env",
        }
        value = json.loads(manifest.read_text())
        value["inventory"][0]["expected_unittest"] = {
            "count": 1,
            "identities": [
                "pass_case.SyntheticCleanEnvironment.test_environment"
            ],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with mock.patch.dict(os.environ, inherited, clear=False):
            status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_OK)

    def test_isolated_environment_is_an_explicit_allowlist(self):
        hostile = {
            "LD_PRELOAD": "/poison/preload.so",
            "LD_AUDIT": "/poison/audit.so",
            "PYTHONHOME": "/poison/python",
            "PYTHONPATH": "/poison/path",
            "BASH_ENV": "/poison/bash",
            "ENV": "/poison/sh",
            "SSL_CERT_FILE": "/poison/cert",
            "MODULEPATH": "/poison/modules",
            "PATH": "/trusted/path",
            "LANG": "C.UTF-8",
        }
        with mock.patch.dict(os.environ, hostile, clear=True):
            environment = runner.isolated_process_environment()
        self.assertEqual(
            environment,
            {"PATH": "/trusted/path", "LANG": "C.UTF-8"},
        )
        self.assertEqual(
            runner.environment_sha256(environment),
            runner.environment_sha256(dict(reversed(list(environment.items())))),
        )

    def test_source_snapshots_ignore_path_injected_fake_git(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = pathlib.Path(temporary_name)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["/usr/bin/git", "init", "-q"], cwd=repo, check=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("trusted\n", encoding="utf-8")
            subprocess.run(
                ["/usr/bin/git", "add", "tracked.txt"], cwd=repo, check=True
            )
            expected_runner = runner.source_snapshot(repo)
            expected_shared = source_snapshot(repo)
            self.assertEqual(
                expected_runner,
                {
                    "available": True,
                    "algorithm": expected_shared["algorithm"],
                    "file_count": expected_shared["file_count"],
                    "sha256": expected_shared["sha256"],
                },
            )
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            fake_git = fake_bin / "git"
            fake_git.write_text(
                "#!/bin/sh\nprintf 'forged\\0'\n",
                encoding="utf-8",
            )
            os.chmod(fake_git, 0o755)
            with mock.patch.dict(
                os.environ,
                {"PATH": str(fake_bin)},
                clear=False,
            ):
                self.assertEqual(runner.source_snapshot(repo), expected_runner)
                self.assertEqual(source_snapshot(repo), expected_shared)

            os.chmod(tracked, 0o755)
            mode_changed = runner.source_snapshot(repo)
            self.assertNotEqual(mode_changed["sha256"], expected_runner["sha256"])

    def test_atomic_text_publish_does_not_follow_fixed_temp_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = pathlib.Path(temporary_name)
            victim = root / "victim"
            victim.write_text("SAFE", encoding="utf-8")
            legacy_temp = root / ".report.json.tmp"
            legacy_temp.symlink_to(victim)
            destination = root / "report.json"
            runner.atomic_write_text(destination, "ATTACK")
            self.assertEqual(victim.read_text(encoding="utf-8"), "SAFE")
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(encoding="utf-8"), "ATTACK")

    def test_missing_or_tampered_receipt_fails_closed(self):
        for mode in ("missing", "tampered"):
            with self.subTest(mode=mode):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                receipt_path = root / "build-provenance.json"
                if mode == "missing":
                    receipt_path.unlink()
                else:
                    receipt = json.loads(receipt_path.read_text())
                    receipt["receipt_id"] = "0" * 64
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                status = self.invoke(
                    root,
                    manifest,
                    "--backend",
                    "cpu",
                    refresh_receipt=False,
                )
                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any("build receipt" in problem for problem in problems),
                    problems,
                )

    def test_legacy_non_mpi_build_receipt_fails_closed(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        receipt_path = root / "build-provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["build_kind"] = "cuda-python-fp32"
        self.reseal_fixture_receipt(root, receipt)

        status = self.invoke(
            root,
            manifest,
            "--backend",
            "cpu",
            refresh_receipt=False,
        )

        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["provenance"]["validation_window"]["problems"]
        self.assertTrue(
            any(
                "release receipt has the wrong build kind" in problem
                for problem in problems
            ),
            problems,
        )

    def test_non_authoritative_mpi_receipt_contract_fails_closed(self):
        mutations = (
            (
                "qualification",
                lambda receipt, _root: receipt["configuration"].__setitem__(
                    "qualification_contract", "generic"
                ),
                "wrong qualification contract",
            ),
            (
                "builder",
                lambda receipt, root: receipt["configuration"].__setitem__(
                    "builder", file_record(root / "cases" / "pass_case.py", root)
                ),
                "authoritative builder",
            ),
            (
                "configure",
                lambda receipt, _root: receipt["configuration"].__setitem__(
                    "configure_argv",
                    [
                        value
                        for value in receipt["configuration"]["configure_argv"]
                        if value != "--with-mpi"
                    ],
                ),
                "receipt and config.status configure arguments disagree",
            ),
        )
        for label, mutate, expected_problem in mutations:
            with self.subTest(label=label):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                receipt_path = root / "build-provenance.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                mutate(receipt, root)
                self.reseal_fixture_receipt(root, receipt)

                status = self.invoke(
                    root,
                    manifest,
                    "--backend",
                    "cpu",
                    refresh_receipt=False,
                )

                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any(expected_problem in problem for problem in problems),
                    problems,
                )

    def test_cache_prefix_qualification_closure_fails_closed(self):
        mutations = (
            (
                "missing_fontconfig",
                lambda receipt, _root: receipt["configuration_files"].pop(
                    "qualification_fontconfig"
                ),
                "cache/prefix qualification closure is incomplete",
            ),
            (
                "missing_build_home",
                lambda receipt, _root: receipt["manifests"].pop("build_home"),
                "cache/prefix qualification closure is incomplete",
            ),
            (
                "missing_installed_environment",
                lambda receipt, _root: receipt["manifests"].pop(
                    "installed_environment"
                ),
                "qualification-v2 receipt lacks log/environment manifests",
            ),
            (
                "missing_qualification_home",
                lambda receipt, _root: receipt["manifests"].pop(
                    "qualification_home"
                ),
                "cache/prefix qualification closure is incomplete",
            ),
            (
                "wrong_fontconfig_path",
                lambda receipt, root: receipt["configuration_files"].__setitem__(
                    "qualification_fontconfig",
                    file_record(root / "cases" / "pass_case.py", root),
                ),
                "qualification Fontconfig path is not authoritative",
            ),
            (
                "wrong_build_home_path",
                lambda receipt, root: receipt["manifests"].__setitem__(
                    "build_home", tree_manifest(root / "cases", root)
                ),
                "build-home manifest path is not authoritative",
            ),
            (
                "wrong_installed_environment_path",
                lambda receipt, root: receipt["manifests"].__setitem__(
                    "installed_environment", tree_manifest(root / "cases", root)
                ),
                "qualification prefix is not receipt-bound",
            ),
            (
                "wrong_qualification_home_path",
                lambda receipt, root: receipt["manifests"].__setitem__(
                    "qualification_home", tree_manifest(root / "cases", root)
                ),
                "qualification-home manifest path is not authoritative",
            ),
        )
        for label, mutate, expected_problem in mutations:
            with self.subTest(label=label):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                receipt_path = root / "build-provenance.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                mutate(receipt, root)
                self.reseal_fixture_receipt(root, receipt)

                status = self.invoke(
                    root,
                    manifest,
                    "--backend",
                    "cpu",
                    refresh_receipt=False,
                )

                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any(expected_problem in problem for problem in problems),
                    problems,
                )

    def test_runtime_cache_or_prefix_mutation_fails_integrity(self):
        targets = (
            ("fontconfig", "qualification-fontconfig.conf"),
            (
                "build_home",
                ".micromamba/cache/build-home-fixture/cache/fontconfig/fixture-cache.txt",
            ),
            (
                "installed_environment",
                ".envs/meep-gpu-cuda-mpi/fixture-package.txt",
            ),
            (
                "qualification_home_xdg_config",
                "qualification-home/.config/fontconfig/fonts.conf",
            ),
        )
        for label, relative_target in targets:
            with self.subTest(label=label):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                target = root / relative_target
                (root / "cases" / "pass_case.py").write_text(
                    textwrap.dedent(
                        f"""
                        import pathlib
                        import unittest
                        import meep

                        target = pathlib.Path({str(target)!r})
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(
                            "runtime mutation\\n", encoding="utf-8"
                        )

                        class SyntheticPass(unittest.TestCase):
                            def test_pass(self):
                                self.assertTrue(True)
                        """
                    ),
                    encoding="utf-8",
                )

                status = self.invoke(root, manifest, "--backend", "cpu")

                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any(
                        "changed" in problem
                        or "source tree changed" in problem
                        for problem in problems
                    ),
                    problems,
                )

    def test_qualification_fontconfig_semantics_fail_closed(self):
        for label, element in (
            ("font_directory", "dir"),
            ("include_directory", "include"),
            ("cache_directory", "cachedir"),
        ):
            with self.subTest(label=label):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                fontconfig = root / "qualification-fontconfig.conf"
                source = fontconfig.read_text(encoding="utf-8")
                opening = (
                    '<include ignore_missing="yes">'
                    if element == "include"
                    else (
                        '<cachedir prefix="xdg">'
                        if element == "cachedir"
                        else f"<{element}>"
                    )
                )
                start = source.index(opening) + len(opening)
                end = source.index(f"</{element}>", start)
                fontconfig.write_text(
                    source[:start] + "/tmp/not-authoritative" + source[end:],
                    encoding="utf-8",
                )

                status = self.invoke(root, manifest, "--backend", "cpu")

                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any(
                        "qualification Fontconfig" in problem
                        for problem in problems
                    ),
                    problems,
                )

        structural_mutations = (
            (
                "root_attribute",
                lambda source: source.replace(
                    "<fontconfig>", '<fontconfig prefix="xdg">', 1
                ),
            ),
            (
                "reset_dirs",
                lambda source: source.replace(
                    "  <dir>", "  <reset-dirs/>\n  <dir>", 1
                ),
            ),
            (
                "dir_attribute",
                lambda source: source.replace(
                    "<dir>", '<dir prefix="xdg">', 1
                ),
            ),
            (
                "include_attribute",
                lambda source: source.replace(
                    '<include ignore_missing="yes">',
                    '<include ignore_missing="yes" prefix="xdg">',
                    1,
                ),
            ),
            (
                "cachedir_wrong_prefix",
                lambda source: source.replace(
                    '<cachedir prefix="xdg">', '<cachedir prefix="cwd">', 1
                ),
            ),
            (
                "cachedir_missing_prefix",
                lambda source: source.replace(
                    '<cachedir prefix="xdg">', "<cachedir>", 1
                ),
            ),
        )
        for label, mutate in structural_mutations:
            with self.subTest(label=label):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                fontconfig = root / "qualification-fontconfig.conf"
                fontconfig.write_text(
                    mutate(fontconfig.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )

                status = self.invoke(root, manifest, "--backend", "cpu")

                self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
                report = json.loads(
                    (root / "evidence" / "report.json").read_text()
                )
                problems = report["provenance"]["validation_window"][
                    "problems"
                ]
                self.assertTrue(
                    any(
                        "qualification Fontconfig" in problem
                        for problem in problems
                    ),
                    problems,
                )

    def test_non_mpi_runtime_fails_backend_contract(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        module = root / "build-python" / "meep" / "__init__.py"
        source = module.read_text(encoding="utf-8")
        module.write_text(
            source.replace(
                "def with_mpi():\n    return True",
                "def with_mpi():\n    return False",
            ).replace(
                "    def with_mpi():\n        return True",
                "    def with_mpi():\n        return False",
            ),
            encoding="utf-8",
        )

        status = self.invoke(root, manifest, "--backend", "cpu")

        self.assertEqual(status, runner.EXIT_BACKEND_CONTRACT)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["results"][0]["runs"]["cpu"][
            "backend_contract_problems"
        ]
        self.assertIn("Meep runtime is not MPI-enabled", problems)


    def test_missing_dependency_is_explicit_blocker(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["dependencies"] = [
            "gpmeep_validation_module_that_does_not_exist"
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_BLOCKED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(report["results"][0]["outcome"], "BLOCKED_DEPENDENCY")

    def test_unittest_skip_text_is_not_silent(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                print("OK (skipped=1)")

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EXECUTION_FAILED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(report["results"][0]["outcome"], "UNDECLARED_SKIP")

    def test_declared_unittest_skip_identity_passes(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                class SyntheticSkip(unittest.TestCase):
                    @unittest.skip("expected optional feature")
                    def test_expected_skip(self):
                        pass
                """
            ),
            encoding="utf-8",
        )
        value = json.loads(manifest.read_text())
        value["inventory"][0]["allowed_unittest_skips"] = 1
        value["inventory"][0]["allowed_unittest_skip_details"] = [
            {
                "test": "*test_expected_skip*",
                "reason": "expected optional feature",
            }
        ]
        value["inventory"][0]["expected_unittest"] = {
            "count": 1,
            "identities": [
                "pass_case.SyntheticSkip.test_expected_skip"
            ],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(
            report["results"][0]["runs"]["cpu"]["outcome"],
            "PASS_WITH_DECLARED_SKIPS",
        )

    def test_declared_docstring_skip_uses_method_identity(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                class SyntheticSkip(unittest.TestCase):
                    @unittest.skip("expected optional feature")
                    def test_expected_skip(self):
                        \"""Human-readable prose that is not a test identity.\"""
                        pass

                if __name__ == "__main__":
                    unittest.main(verbosity=2)
                """
            ),
            encoding="utf-8",
        )
        value = json.loads(manifest.read_text())
        value["inventory"][0]["allowed_unittest_skips"] = 1
        value["inventory"][0]["allowed_unittest_skip_details"] = [
            {
                "test": "*test_expected_skip*",
                "reason": "expected optional feature",
            }
        ]
        value["inventory"][0]["expected_unittest"] = {
            "count": 1,
            "identities": [
                "pass_case.SyntheticSkip.test_expected_skip"
            ],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cpu"]
        self.assertEqual(run["outcome"], "PASS_WITH_DECLARED_SKIPS")
        self.assertIn("test_expected_skip", run["unittest_skip_details"][0]["test"])
        self.assertNotIn(
            "Human-readable prose", run["unittest_skip_details"][0]["test"]
        )

    def test_same_skip_count_with_wrong_identity_fails(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import meep

                class SyntheticSkip(unittest.TestCase):
                    @unittest.skip("unexpected reason")
                    def test_wrong_skip(self):
                        pass
                """
            ),
            encoding="utf-8",
        )
        value = json.loads(manifest.read_text())
        value["inventory"][0]["allowed_unittest_skips"] = 1
        value["inventory"][0]["allowed_unittest_skip_details"] = [
            {
                "test": "*test_expected_skip*",
                "reason": "expected optional feature",
            }
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EXECUTION_FAILED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        result = report["results"][0]
        self.assertEqual(result["outcome"], "UNDECLARED_SKIP")
        self.assertIn(
            "unapproved or ambiguous unittest skip",
            result["runs"]["cpu"]["unittest_skip_policy_problems"][0],
        )

    def test_allowed_skip_policy_is_consumed_at_most_once(self):
        case = {
            "allowed_unittest_skip_details": [
                {"test": "*test_duplicate*", "reason": "expected"},
                {"test": "*test_never*", "reason": "expected"},
            ]
        }
        ok, problems = runner.verify_unittest_skip_policy(
            case,
            2,
            [
                {"test": "test_duplicate", "reason": "expected"},
                {"test": "test_duplicate", "reason": "expected"},
            ],
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("unapproved or ambiguous" in problem for problem in problems)
        )
        optional_ok, optional_problems = runner.verify_unittest_skip_policy(
            case, 0, []
        )
        self.assertTrue(optional_ok, optional_problems)

    def test_expected_unittest_count_and_identities_pass(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["expected_unittest"] = {
            "count": 1,
            "identities": ["pass_case.SyntheticPass.test_pass"],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cpu"]
        self.assertTrue(run["unittest_test_contract_ok"])
        self.assertEqual(run["unittest_test_count"], 1)
        self.assertEqual(
            run["unittest_test_identities"],
            ["pass_case.SyntheticPass.test_pass"],
        )

    def test_same_unittest_count_with_wrong_identity_fails(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["expected_unittest"] = {
            "count": 1,
            "identities": ["pass_case.SyntheticPass.test_missing"],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EXECUTION_FAILED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        run = report["results"][0]["runs"]["cpu"]
        self.assertEqual(run["outcome"], "UNITTEST_CONTRACT_FAILED")
        self.assertFalse(run["unittest_test_contract_ok"])

    def test_unittest_contract_parses_multiline_docstring_identity(self):
        case = {
            "expected_unittest": {
                "count": 1,
                "identities": ["test_module.Example.test_value"],
            }
        }
        output = textwrap.dedent(
            """
            test_value (test_module.Example.test_value)
            Human-readable prose ... ok

            ----------------------------------------------------------------------
            Ran 1 test in 0.001s

            OK
            """
        )
        ok, evidence, problems = runner.verify_unittest_test_contract(
            case, output
        )
        self.assertTrue(ok, problems)
        self.assertEqual(
            evidence["identities"], ["test_module.Example.test_value"]
        )

    def test_unittest_identity_parser_counts_mpi_interleaved_records(self):
        output = (
            "test_value (test_module.Example.test_value) ... "
            "test_value (test_module.Example.test_value) ... ok\n"
            "ok\n"
        )
        self.assertEqual(
            runner.parse_unittest_test_identities(output),
            [
                "test_module.Example.test_value",
                "test_module.Example.test_value",
            ],
        )

    def test_unittest_contract_rejects_failed_terminal_summary(self):
        case = {
            "expected_unittest": {
                "count": 1,
                "identities": ["test_module.Example.test_value"],
            }
        }
        for summary in ("FAIL", "ERROR", "UNEXPECTED SUCCESS"):
            with self.subTest(summary=summary):
                output = textwrap.dedent(
                    f"""
                    test_value (test_module.Example.test_value) ... {summary.lower()}

                    ======================================================================
                    {summary}: test_value (test_module.Example.test_value)
                    ----------------------------------------------------------------------
                    Ran 1 test in 0.001s

                    FAILED
                    """
                )
                ok, evidence, problems = runner.verify_unittest_test_contract(
                    case, output
                )
                self.assertFalse(ok)
                self.assertTrue(
                    any("terminal summary reports failure" in item for item in problems)
                )
                self.assertEqual(
                    evidence["identities"],
                    ["test_module.Example.test_value"],
                )

    def test_unittest_contract_still_rejects_actual_duplicate_execution(self):
        output = textwrap.dedent(
            """
            test_value (test_module.Example.test_value) ... ok
            test_value (test_module.Example.test_value) ... ok

            ----------------------------------------------------------------------
            Ran 2 tests in 0.001s

            OK
            """
        )
        ok, evidence, problems = runner.verify_unittest_test_contract(
            {}, output
        )
        self.assertFalse(ok)
        self.assertEqual(len(evidence["identities"]), 2)
        self.assertTrue(
            any("duplicate test identities" in problem for problem in problems)
        )

    def test_unittest_contract_rejects_zero_tests(self):
        ok, _, problems = runner.verify_unittest_test_contract(
            {}, "Ran 0 tests in 0.000s\n\nOK\n"
        )
        self.assertFalse(ok)
        self.assertIn("unittest discovered no tests", problems)

    def test_manifest_rejects_malformed_expected_unittest(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["expected_unittest"] = {
            "count": 2,
            "identities": ["pass_case.SyntheticPass.test_pass"],
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(runner.ManifestError, "must equal"):
            runner.materialize_cases(runner.load_json(manifest), root)

    def test_manifest_rejects_nonboolean_active_backend_contract(self):
        for invalid in ("false", 0, 1, None):
            with self.subTest(invalid=invalid):
                temporary, root, manifest = self.make_fixture()
                self.addCleanup(temporary.cleanup)
                value = json.loads(manifest.read_text())
                value["inventory"][0][
                    "require_active_backend_at_exit"
                ] = invalid
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(
                    runner.ManifestError, "must be a boolean"
                ):
                    runner.materialize_cases(
                        runner.load_json(manifest), root
                    )

    def test_manifest_rejects_unpinned_backend_release_waiver(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        case = value["inventory"][0]
        case["gpu_contract"] = "none"
        case["require_active_backend_at_exit"] = False
        case.pop("expected_unittest")
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            runner.ManifestError, "pinned expected_unittest identities"
        ):
            runner.materialize_cases(runner.load_json(manifest), root)

    def test_manifest_rejects_unpinned_runnable_unittest(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0].pop("expected_unittest")
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            runner.ManifestError,
            "every runnable unittest requires pinned",
        ):
            runner.materialize_cases(runner.load_json(manifest), root)

    def test_source_change_during_validation_fails_integrity(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        mutation_target = root / "cases" / "tracked_source.txt"
        mutation_target.write_text("before\n", encoding="utf-8")
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                f"""
                import pathlib
                import unittest
                import meep

                pathlib.Path({str(mutation_target)!r}).write_text(
                    "after\\n", encoding="utf-8"
                )

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        window = report["provenance"]["validation_window"]
        self.assertFalse(window["unchanged"])
        self.assertIn("source tree changed during validation", window["problems"])
        complete = json.loads((root / "evidence" / "COMPLETE").read_text())
        self.assertEqual(complete["exit_code"], runner.EXIT_EVIDENCE_INTEGRITY)

    def test_extension_change_during_validation_fails_integrity(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        extension = root / "build-python" / "meep" / "_meep-self-test.so"
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                f"""
                import pathlib
                import unittest
                import meep

                pathlib.Path({str(extension)!r}).write_bytes(b"changed")

                class SyntheticPass(unittest.TestCase):
                    def test_pass(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["provenance"]["validation_window"]["problems"]
        self.assertIn("built extension changed during validation", problems)

    def test_missing_start_extension_fails_with_completed_report(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "build-python" / "meep" / "_meep-self-test.so").unlink()
        status = self.invoke(
            root,
            manifest,
            "--backend",
            "cpu",
            refresh_receipt=False,
        )
        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(report["exit_code"], runner.EXIT_EVIDENCE_INTEGRITY)
        self.assertEqual(
            report["results"][0]["outcome"], "EVIDENCE_INTEGRITY_FAILED"
        )
        self.assertTrue((root / "evidence" / "COMPLETE").is_file())

    def test_extension_snapshot_rejects_multiple_candidates(self):
        temporary, root, _ = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "build-python" / "meep" / "_meep-second.so").write_bytes(
            b"second synthetic extension"
        )
        snapshot = runner.build_extension_snapshot(root / "build-python")
        self.assertFalse(snapshot["available"])
        self.assertIn("found 2", snapshot["error"])

    def test_unavailable_source_snapshot_fails_with_completed_report(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        unavailable = {"available": False, "error": "synthetic git failure"}
        with mock.patch.object(
            runner, "source_snapshot", return_value=unavailable
        ):
            status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_EVIDENCE_INTEGRITY)
        report = json.loads((root / "evidence" / "report.json").read_text())
        problems = report["provenance"]["validation_window"]["problems"]
        self.assertTrue(
            any("synthetic git failure" in problem for problem in problems)
        )
        self.assertTrue((root / "evidence" / "COMPLETE").is_file())

    def test_nonempty_output_rejection_preserves_prior_authority_files(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        evidence = root / "evidence"
        evidence.mkdir()
        for name in ("report.json", "report.md", "COMPLETE"):
            (evidence / name).write_text("stale\n", encoding="utf-8")
        with mock.patch.object(runner, "run_backend") as run_backend:
            status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_MANIFEST_INVALID)
        run_backend.assert_not_called()
        for name in ("report.json", "report.md", "COMPLETE"):
            self.assertEqual(
                (evidence / name).read_text(encoding="utf-8"), "stale\n"
            )

    def test_manifest_rejects_example_without_observable_oracle(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["kind"] = "example"
        value["inventory"][0].pop("expected_unittest")
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(runner.ManifestError, "embedded oracle"):
            runner.materialize_cases(runner.load_json(manifest), root)

    def test_json_metrics_compare_with_per_metric_tolerances(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "angle": {"atol": 0.02},
                "spectrum[0]": {"rtol": 1e-3},
            },
        }
        result = runner.compare_json_metrics(
            'noise\ngpmeep-metrics: {"angle": 12.0, "spectrum": [2.0]}\n',
            'gpmeep-metrics: {"spectrum": [2.001], "angle": 12.01}\n',
            comparison,
        )
        self.assertEqual(result["outcome"], "PASS")
        self.assertLess(
            result["differences"]["angle"]["absolute_error"],
            result["differences"]["angle"]["allowed_error"],
        )

    def test_json_metrics_supports_unambiguous_metric_patterns(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "objective*": {"rtol": 1e-3},
                "*derivative*": {"atol": 1e-6},
            },
        }
        result = runner.compare_json_metrics(
            'gpmeep-metrics: {"objective": [2.0], "derivative": 1e-5}\n',
            'gpmeep-metrics: {"objective": [2.001], "derivative": 1.05e-5}\n',
            comparison,
        )
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(
            result["differences"]["objective[0]"]["tolerance_rule"],
            "objective*",
        )

    def test_json_metrics_diagnostic_only_is_recorded_without_deciding(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "robust": {"atol": 0.0},
                "pure_diagnostic": {
                    "atol": 0.0,
                    "diagnostic_only": True,
                },
            },
        }
        cpu_text = (
            'gpmeep-metrics:{"robust":2.0,"pure_diagnostic":1.0}\n'
        )
        cuda_text = (
            'gpmeep-metrics:{"robust":2.0,"pure_diagnostic":9.0}\n'
        )

        expected = runner.compare_json_metrics(
            cpu_text, cuda_text, comparison
        )

        self.assertEqual(expected["outcome"], "PASS")
        self.assertEqual(expected["failure_count"], 0)
        self.assertEqual(
            expected["metric_evidence"]["diagnostic_only_count"], 1
        )
        self.assertEqual(
            expected["metric_evidence"][
                "diagnostic_only_out_of_tolerance_count"
            ],
            1,
        )
        diagnostic = expected["differences"]["pure_diagnostic"]
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertFalse(diagnostic["within_tolerance"])
        self.assertIn("diagnostic-only", expected["reason"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu_path = root / "cpu.stdout"
            cuda_path = root / "cuda.stdout"
            failure_sidecar = root / "failures.ndjson"
            cpu_path.write_text(cpu_text, encoding="utf-8")
            cuda_path.write_text(cuda_text, encoding="utf-8")
            actual = runner.compare_json_metric_files(
                cpu_path,
                cuda_path,
                comparison,
                failure_sidecar=failure_sidecar,
            )
            self.assertEqual(actual, expected)
            self.assertFalse(failure_sidecar.exists())

        robust_mismatch = runner.compare_json_metrics(
            cpu_text,
            'gpmeep-metrics:{"robust":3.0,"pure_diagnostic":9.0}\n',
            comparison,
        )
        self.assertEqual(robust_mismatch["outcome"], "MISMATCH")
        self.assertEqual(robust_mismatch["failure_count"], 1)
        self.assertEqual(
            robust_mismatch["failures"][0]["metric"], "robust"
        )

    def test_json_metrics_diagnostic_only_does_not_hide_comparison_overflow(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "robust": {"atol": 0.0},
                "diagnostic": {
                    "atol": 0.0,
                    "diagnostic_only": True,
                },
            },
        }
        maximum = sys.float_info.max
        result = runner.compare_json_metrics(
            "gpmeep-metrics:"
            + json.dumps({"robust": 1.0, "diagnostic": maximum})
            + "\n",
            "gpmeep-metrics:"
            + json.dumps({"robust": 1.0, "diagnostic": -maximum})
            + "\n",
            comparison,
        )
        self.assertEqual(result["outcome"], "MISMATCH")
        self.assertEqual(result["failure_count"], 1)
        self.assertTrue(result["failures"][0]["arithmetic_overflow"])

    def test_json_metrics_rejects_invalid_diagnostic_only_rules(self):
        base = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"metric": {"atol": 0.0}},
        }
        invalid_boolean = copy.deepcopy(base)
        invalid_boolean["tolerances"]["metric"]["diagnostic_only"] = 1
        with self.assertRaisesRegex(
            runner.ManifestError, "diagnostic_only must be boolean"
        ):
            runner.validate_json_tolerance_rules(invalid_boolean)

        unknown_field = copy.deepcopy(base)
        unknown_field["tolerances"]["metric"]["diagnositic_only"] = True
        with self.assertRaisesRegex(
            runner.ManifestError, "unknown tolerance fields"
        ):
            runner.validate_json_tolerance_rules(unknown_field)

        all_diagnostic = copy.deepcopy(base)
        all_diagnostic["tolerances"]["metric"]["diagnostic_only"] = True
        with self.assertRaisesRegex(
            runner.ManifestError, "verdict-bearing tolerance"
        ):
            runner.validate_json_tolerance_rules(all_diagnostic)

    def test_large_json_metrics_are_digest_bound_without_report_duplication(self):
        count = runner.MAX_INLINE_JSON_METRICS + 1
        values = [float(index) / 8 for index in range(count)]
        payload = "gpmeep-metrics:" + json.dumps({"values": values}) + "\n"
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"values*": {"rtol": 0.0, "atol": 0.0}},
        }

        result = runner.compare_json_metrics(payload, payload, comparison)

        self.assertEqual(result["outcome"], "PASS")
        self.assertFalse(result["metric_evidence"]["inlined"])
        self.assertEqual(result["metric_evidence"]["count"], count)
        self.assertEqual(result["metric_evidence"]["passed_count"], count)
        self.assertNotIn("cpu_metrics", result)
        self.assertNotIn("cuda_metrics", result)
        self.assertNotIn("differences", result)
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(
            result["metric_evidence"]["cpu_sha256"],
            result["metric_evidence"]["cuda_sha256"],
        )
        self.assertLess(len(json.dumps(result)), 10000)

    def test_streaming_json_metrics_preserves_unordered_full_semantics(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "angle": {"atol": 0.02},
                "spectrum*": {"rtol": 1e-3},
            },
        }
        cpu_text = (
            'noise\ngpmeep-metrics:{"angle":12.0,'
            '"spectrum":[2.0,-0.0]}\n'
        )
        cuda_text = (
            'gpmeep-metrics:{"spectrum":[2.001,-0.0],'
            '"angle":12.01}\n'
        )
        expected = runner.compare_json_metrics(
            cpu_text, cuda_text, comparison
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu_path = root / "cpu.stdout"
            cuda_path = root / "cuda.stdout"
            cpu_path.write_text(cpu_text, encoding="utf-8")
            cuda_path.write_text(cuda_text, encoding="utf-8")
            actual = runner.compare_json_metric_files(
                cpu_path, cuda_path, comparison
            )
            mismatch_comparison = copy.deepcopy(comparison)
            mismatch_comparison["tolerances"] = {
                "angle": {"atol": 0.0},
                "spectrum*": {"atol": 0.0},
            }
            expected_mismatch = runner.compare_json_metrics(
                cpu_text,
                cuda_text,
                mismatch_comparison,
                failure_sidecar=root / "legacy-failures.ndjson",
            )
            actual_mismatch = runner.compare_json_metric_files(
                cpu_path,
                cuda_path,
                mismatch_comparison,
                failure_sidecar=root / "stream-failures.ndjson",
            )
            self.assertEqual(actual_mismatch, expected_mismatch)
            self.assertEqual(
                (root / "stream-failures.ndjson").read_bytes(),
                (root / "legacy-failures.ndjson").read_bytes(),
            )
        self.assertEqual(actual, expected)

    def test_direct_json_metrics_fully_compare_without_sort_artifacts(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        cpu_text = (
            comparison["prefix"]
            + '{"early":1,"nested":{"values":[2,3]},"late":4}\n'
        )
        cuda_text = (
            comparison["prefix"]
            + '{"early":9,"nested":{"values":[2,3]},"late":8}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu_path = root / "cpu.stdout"
            cuda_path = root / "cuda.stdout"
            cpu_path.write_text(cpu_text, encoding="utf-8")
            cuda_path.write_text(cuda_text, encoding="utf-8")
            with mock.patch.object(
                runner, "DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES", 0
            ), mock.patch.object(
                runner,
                "_external_sort_records",
                side_effect=AssertionError("direct path created sort files"),
            ):
                result = runner.compare_json_metric_files(
                    cpu_path, cuda_path, comparison
                )

        self.assertEqual(result["outcome"], "MISMATCH")
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["metric_evidence"]["count"], 4)
        self.assertEqual(
            result["metric_evidence"]["schema"],
            runner.DIRECT_METRIC_EVIDENCE_SCHEMA,
        )
        self.assertEqual(
            result["metric_evidence"]["comparison_strategy"],
            "aligned-direct",
        )
        self.assertEqual(
            result["metric_evidence"]["digest_order"], "document-order"
        )

    def test_direct_json_metrics_order_difference_uses_external_sort(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.write_text(
                comparison["prefix"] + '{"a":1,"b":2}\n',
                encoding="utf-8",
            )
            cuda.write_text(
                comparison["prefix"] + '{"b":2,"a":1}\n',
                encoding="utf-8",
            )
            original_sort = runner._external_sort_records
            with mock.patch.object(
                runner, "DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES", 0
            ), mock.patch.object(
                runner, "_external_sort_records", wraps=original_sort
            ) as external_sort:
                result = runner.compare_json_metric_files(
                    cpu, cuda, comparison
                )

        self.assertEqual(result["outcome"], "PASS")
        self.assertGreater(external_sort.call_count, 0)
        self.assertEqual(
            result["metric_evidence"]["schema"],
            runner.METRIC_EVIDENCE_SCHEMA,
        )
        self.assertNotIn("comparison_strategy", result["metric_evidence"])

    def test_direct_json_metrics_preserve_duplicate_and_collision_checks(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name, payload in (
                ("duplicate", '{"x":1,"x":2}'),
                ("collision", '{"a.b":1,"a":{"b":2}}'),
            ):
                cpu = root / f"{name}-cpu"
                cuda = root / f"{name}-cuda"
                cpu.write_text(
                    comparison["prefix"] + payload + "\n", encoding="utf-8"
                )
                cuda.write_text(
                    comparison["prefix"] + payload + "\n", encoding="utf-8"
                )
                error = "duplicate key" if name == "duplicate" else "ambiguous"
                with mock.patch.object(
                    runner, "DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES", 0
                ), self.assertRaisesRegex(runner.ManifestError, error):
                    runner.compare_json_metric_files(
                        cpu, cuda, comparison
                    )

    def test_direct_json_metrics_parse_late_invalid_value_on_both_sides(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.write_text(
                comparison["prefix"] + '{"a":1,"late":2}\n',
                encoding="utf-8",
            )
            cuda.write_text(
                comparison["prefix"] + '{"a":1,"late":1e999}\n',
                encoding="utf-8",
            )
            with mock.patch.object(
                runner, "DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES", 0
            ), self.assertRaisesRegex(runner.ManifestError, "must be finite"):
                runner.compare_json_metric_files(cpu, cuda, comparison)

    def test_direct_json_metrics_bind_output_to_parser_origin(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        original_iterator = runner._DirectMetricSource.iter_metrics

        def omit_metric(source):
            for key, value in original_iterator(source):
                if key == b"b":
                    continue
                yield key, value

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.write_text(
                comparison["prefix"] + '{"a":1,"b":999}\n',
                encoding="utf-8",
            )
            cuda.write_text(
                comparison["prefix"] + '{"a":1,"b":-999}\n',
                encoding="utf-8",
            )
            with mock.patch.object(
                runner, "DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES", 0
            ), mock.patch.object(
                runner._DirectMetricSource,
                "iter_metrics",
                autospec=True,
                side_effect=omit_metric,
            ), self.assertRaisesRegex(
                runner.ManifestError, "differs from parser origin"
            ):
                runner.compare_json_metric_files(cpu, cuda, comparison)

    def test_streaming_json_metrics_matches_integer_negative_zero_semantics(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        cpu_text = (
            'gpmeep-metrics:{"unicode_β":0,"nested":{"x":1e-300},'
            '"values":[2,3]}\n'
        )
        cuda_text = (
            'gpmeep-metrics:{"values":[2.0,3e0],"nested":{"x":1e-300},'
            '"unicode_\\u03b2":-0}\n'
        )
        expected = runner.compare_json_metrics(
            cpu_text, cuda_text, comparison
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu_path = root / "cpu.stdout"
            cuda_path = root / "cuda.stdout"
            cpu_path.write_text(cpu_text, encoding="utf-8")
            cuda_path.write_text(cuda_text, encoding="utf-8")
            actual = runner.compare_json_metric_files(
                cpu_path, cuda_path, comparison
            )

        self.assertEqual(actual, expected)
        self.assertEqual(actual["outcome"], "PASS")
        self.assertEqual(
            actual["metric_evidence"]["cpu_sha256"],
            actual["metric_evidence"]["cuda_sha256"],
        )

    def test_streaming_json_metrics_rejects_duplicate_and_ambiguous_paths(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        for payload, error in (
            ('{"x":1,"x":2}', "duplicate key"),
            ('{"a.b":1,"a":{"b":2}}', "ambiguous"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                for backend in ("cpu", "cuda"):
                    (root / backend).write_text(
                        comparison["prefix"] + payload + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaisesRegex(runner.ManifestError, error):
                    runner.compare_json_metric_files(
                        root / "cpu", root / "cuda", comparison
                    )

    def test_streaming_json_metrics_checks_both_complete_documents(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "cpu").write_text(
                comparison["prefix"] + '{"x":[1,2]}\n', encoding="utf-8"
            )
            (root / "cuda").write_text(
                comparison["prefix"] + '{"x":[1,2],"late":1e999}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(runner.ManifestError, "must be finite"):
                runner.compare_json_metric_files(
                    root / "cpu", root / "cuda", comparison
                )

            (root / "cuda").write_text(
                comparison["prefix"] + '{"x":[1,2],"late":3}\n',
                encoding="utf-8",
            )
            result = runner.compare_json_metric_files(
                root / "cpu", root / "cuda", comparison
            )
            self.assertEqual(result["outcome"], "MISMATCH")
            self.assertEqual(result["metric_evidence"]["cuda_only_count"], 1)
            self.assertEqual(
                result["metric_evidence"]["cuda_only_sample"], ["late"]
            )

    def test_streaming_json_metrics_is_bound_to_recorded_stdout_digest(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            original = comparison["prefix"] + '{"value":1}\n'
            cpu.write_text(original, encoding="utf-8")
            cuda.write_text(original, encoding="utf-8")

            def record(path):
                payload = path.read_bytes()
                return {
                    "available": True,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

            cpu_record = record(cpu)
            cuda_record = record(cuda)
            # Preserve the byte count to ensure this is a digest rather than a
            # file-size-only binding test.
            cpu.write_text(
                comparison["prefix"] + '{"value":2}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                runner.ManifestError, "recorded evidence digest"
            ):
                runner.compare_json_metric_files(
                    cpu,
                    cuda,
                    comparison,
                    expected_cpu_stdout=cpu_record,
                    expected_cuda_stdout=cuda_record,
                )

    def test_streaming_json_metrics_rejects_symlink_stdout(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target"
            target.write_text(
                comparison["prefix"] + '{"value":1}\n', encoding="utf-8"
            )
            symlink = root / "stdout"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(
                runner.ManifestError, "cannot open json_metrics stdout"
            ):
                runner.compare_json_metric_files(
                    symlink, target, comparison
                )

    def test_streaming_json_metrics_rejects_changed_sorted_replay(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 10.0}},
        }

        def changing_source():
            calls = 0

            def iterator():
                nonlocal calls
                calls += 1
                value = 1.0 if calls == 1 else 2.0
                return iter([(b"value", value)])

            return runner._MetricSource(iterator)

        with self.assertRaisesRegex(
            runner.ManifestError,
            "replay changed between comparison passes",
        ):
            runner._compare_metric_sources(
                changing_source(), changing_source(), comparison, None
            )

    def test_streaming_json_sort_records_fail_closed_on_bit_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            record_path = pathlib.Path(temporary) / "record.spool"
            with record_path.open("xb") as output:
                runner._write_sort_record(output, b"example_results.value", 1.0)
            with record_path.open("r+b") as output:
                output.seek(4 + len(b"example_"))
                original = output.read(1)
                output.seek(-1, os.SEEK_CUR)
                output.write(bytes([original[0] ^ 0x20]))

            with record_path.open("rb") as source, self.assertRaisesRegex(
                runner.ManifestError, "failed its checksum"
            ):
                runner._read_sort_record(source)

        self.assertEqual(runner.JSON_SORT_RECORD_READ_ATTEMPTS, 1)

    def test_streaming_json_metrics_bind_sorted_set_to_parser_origin(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        original_iterator = runner._iter_sorted_chunks

        def omit_metric(chunks, description):
            for key, value in original_iterator(chunks, description):
                if description == "metric" and key == b"b":
                    continue
                yield key, value

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.write_text(
                comparison["prefix"] + '{"a":1,"b":999}\n',
                encoding="utf-8",
            )
            cuda.write_text(
                comparison["prefix"] + '{"a":1,"b":-999}\n',
                encoding="utf-8",
            )
            with mock.patch.object(
                runner, "_iter_sorted_chunks", side_effect=omit_metric
            ), self.assertRaisesRegex(
                runner.ManifestError, "differs from parser origin"
            ):
                runner.compare_json_metric_files(cpu, cuda, comparison)

    def test_streaming_json_snapshot_transient_rewrite_is_rejected(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        original_source = runner._streaming_metric_source

        def transient_source(snapshot, policy, temporary_root, stem):
            original = snapshot.path.read_bytes()
            snapshot.path.chmod(0o600)
            snapshot.path.write_text(
                policy["prefix"] + '{"a":1,"b":0}\n', encoding="utf-8"
            )
            try:
                return original_source(
                    snapshot, policy, temporary_root, stem
                )
            finally:
                snapshot.path.write_bytes(original)
                snapshot.path.chmod(0o400)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.write_text(
                comparison["prefix"] + '{"a":1,"b":999}\n',
                encoding="utf-8",
            )
            cuda.write_text(
                comparison["prefix"] + '{"a":1,"b":-999}\n',
                encoding="utf-8",
            )
            with mock.patch.object(
                runner,
                "_streaming_metric_source",
                side_effect=transient_source,
            ), self.assertRaisesRegex(
                runner.ManifestError,
                "private CPU json_metrics snapshot changed",
            ):
                runner.compare_json_metric_files(cpu, cuda, comparison)

    def test_json_metric_digest_is_order_invariant_and_value_sensitive(self):
        first = {"beta": -0.0, "alpha": 1.25}
        reordered = {"alpha": 1.25, "beta": -0.0}
        changed = {"alpha": 1.25, "beta": 0.0}

        self.assertEqual(
            runner.flattened_metrics_sha256(first),
            "6f484410747faeb55b7b271ad60fe6c479c15f6fd4a8941e6cfcd59a97c58bd2",
        )
        self.assertEqual(
            runner.flattened_metric_keys_sha256(first),
            "be6c4de06a406f90e5b899b38f282d615801508b4c757ba24fe26a6f0a4c0179",
        )
        self.assertEqual(
            runner.flattened_metrics_sha256(first),
            runner.flattened_metrics_sha256(reordered),
        )
        self.assertNotEqual(
            runner.flattened_metrics_sha256(first),
            runner.flattened_metrics_sha256(changed),
        )
        self.assertEqual(
            runner.flattened_metric_keys_sha256(first),
            runner.flattened_metric_keys_sha256(changed),
        )

    def test_json_metrics_rejects_ambiguous_metric_patterns(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {
                "*value": {"atol": 1e-6},
                "objective*": {"atol": 1e-6},
            },
        }
        with self.assertRaisesRegex(runner.ManifestError, "ambiguous"):
            runner.compare_json_metrics(
                'gpmeep-metrics: {"objective_value": 1.0}\n',
                'gpmeep-metrics: {"objective_value": 1.0}\n',
                comparison,
            )

    def test_json_metrics_mismatch_is_a_comparison_failure(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["inventory"][0]["kind"] = "example"
        value["inventory"][0].pop("expected_unittest")
        value["inventory"][0]["comparison"] = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"value": {"atol": 0.01}},
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        (root / "cases" / "pass_case.py").write_text(
            textwrap.dedent(
                """
                import json
                import os
                import meep
                value = 1.0 if os.environ["MEEP_GPU_BACKEND"] == "cpu" else 1.1
                print("gpmeep-metrics:" + json.dumps({"value": value}))
                """
            ),
            encoding="utf-8",
        )
        status = self.invoke(root, manifest)
        self.assertEqual(status, runner.EXIT_COMPARISON_FAILED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        comparison = report["results"][0]["comparison"]
        self.assertEqual(comparison["outcome"], "MISMATCH")
        sidecar = root / "evidence" / comparison["failure_evidence"]["path"]
        self.assertTrue(sidecar.is_file())
        self.assertEqual(len(sidecar.read_text().splitlines()), 1)
        self.assertEqual(
            comparison["failure_evidence"]["sha256"],
            hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        )

    def test_invalid_json_metric_comparison_retains_raw_evidence_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            case_directory = "case"
            runs = {}
            for backend, payload in (
                ("cpu", 'gpmeep-metrics:{"value":1e999}\n'),
                ("cuda", 'gpmeep-metrics:{"value":1.0}\n'),
            ):
                stdout = (
                    root / "runs" / case_directory / backend / "stdout.log"
                )
                stdout.parent.mkdir(parents=True)
                stdout.write_text(payload, encoding="utf-8")
                record = {
                    "available": True,
                    **runner.evidence_file_record(stdout, root),
                }
                runs[backend] = {
                    "outcome": "PASS",
                    "duration_seconds": 1.0,
                    "stdout_log": record["path"],
                    "evidence_case_directory": case_directory,
                    "evidence_files": {"stdout": record},
                }
            case = {
                "comparison": {
                    "mode": "json_metrics",
                    "prefix": "gpmeep-metrics:",
                    "tolerances": {"*": {"atol": 0.0}},
                }
            }

            comparison = runner.compare_runs(case, runs, root)

        self.assertEqual(comparison["outcome"], "MISMATCH")
        self.assertIn("must be finite", comparison["reason"])
        self.assertEqual(
            comparison["raw_metric_evidence"],
            {
                "cpu_stdout": runs["cpu"]["evidence_files"]["stdout"],
                "cuda_stdout": runs["cuda"]["evidence_files"]["stdout"],
            },
        )

    def test_json_metrics_rejects_duplicate_and_ambiguous_paths(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0}},
        }
        with self.assertRaisesRegex(
            runner.ManifestError, "duplicate key"
        ):
            runner.extract_json_metrics(
                'gpmeep-metrics:{"value":1,"value":2}\n',
                comparison,
            )
        with self.assertRaisesRegex(
            runner.ManifestError, "ambiguous"
        ):
            runner.extract_json_metrics(
                'gpmeep-metrics:{"a.b":1,"a":{"b":2}}\n',
                comparison,
            )

    def test_json_metrics_enforces_nesting_depth_without_recursion_crash(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"rtol": 0.0, "atol": 0.0}},
        }
        accepted_depth = runner.MAX_JSON_NESTING_DEPTH - 1
        accepted = (
            comparison["prefix"]
            + '{"x":'
            + "[" * accepted_depth
            + "1"
            + "]" * accepted_depth
            + "}\n"
        )
        self.assertEqual(len(runner.extract_json_metrics(accepted, comparison)), 1)

        rejected_depth = runner.MAX_JSON_NESTING_DEPTH
        rejected = (
            comparison["prefix"]
            + '{"x":'
            + "[" * rejected_depth
            + "1"
            + "]" * rejected_depth
            + "}\n"
        )
        with self.assertRaisesRegex(runner.ManifestError, "nesting depth"):
            runner.extract_json_metrics(rejected, comparison)

        pathological = (
            comparison["prefix"]
            + '{"x":'
            + "[" * 2000
            + "1"
            + "]" * 2000
            + "}\n"
        )
        with self.assertRaisesRegex(runner.ManifestError, "nesting depth"):
            runner.extract_json_metrics(pathological, comparison)

    def test_bounded_process_streams_and_fails_closed_at_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            with (
                mock.patch.object(runner, "MAX_BACKEND_STDOUT_BYTES", 64),
                mock.patch.object(runner, "MAX_BACKEND_STDERR_BYTES", 64),
                mock.patch.object(runner, "MAX_BACKEND_OUTPUT_BYTES", 96),
            ):
                result = runner.run_bounded_process(
                    [
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'x' * 65)",
                    ],
                    cwd=root,
                    env=runner.isolated_process_environment(),
                    timeout_seconds=5,
                    stdout_file=stdout,
                    stderr_file=stderr,
                )
            self.assertEqual(result["output_limit"], "stdout")
            self.assertLessEqual(stdout.stat().st_size, 64)
            self.assertLessEqual(
                stdout.stat().st_size + stderr.stat().st_size, 96
            )

    def test_bounded_process_preserves_partial_timeout_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            result = runner.run_bounded_process(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,time; os.write(1,b'partial\\n'); "
                        "time.sleep(5)"
                    ),
                ],
                cwd=root,
                env=runner.isolated_process_environment(),
                timeout_seconds=1,
                stdout_file=stdout,
                stderr_file=stderr,
            )
            self.assertTrue(result["timeout"])
            self.assertEqual(stdout.read_bytes(), b"partial\n")

    def test_bounded_process_times_out_after_child_closes_both_pipes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            started = runner.time.monotonic()
            result = runner.run_bounded_process(
                [
                    sys.executable,
                    "-c",
                    "import os,time; os.close(1); os.close(2); time.sleep(5)",
                ],
                cwd=root,
                env=runner.isolated_process_environment(),
                timeout_seconds=0.2,
                stdout_file=stdout,
                stderr_file=stderr,
            )
            elapsed = runner.time.monotonic() - started
            self.assertTrue(result["timeout"])
            self.assertLess(elapsed, 2.0)

    def test_bounded_process_does_not_wait_for_escaped_pipe_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            child = (
                "import os,subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',"
                "'import os,time; os.setsid(); time.sleep(2)']); "
                "time.sleep(5)"
            )
            started = runner.time.monotonic()
            result = runner.run_bounded_process(
                [sys.executable, "-c", child],
                cwd=root,
                env=runner.isolated_process_environment(),
                timeout_seconds=0.2,
                stdout_file=stdout,
                stderr_file=stderr,
            )
            elapsed = runner.time.monotonic() - started
            self.assertTrue(result["timeout"])
            self.assertLess(elapsed, 2.0)

    def test_json_metrics_rejects_integer_too_large_for_float(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0}},
        }
        payload = "gpmeep-metrics:" + json.dumps({"value": 10**400}) + "\n"
        with self.assertRaisesRegex(runner.ManifestError, "must be finite"):
            runner.extract_json_metrics(payload, comparison)

    def test_json_metrics_derived_overflow_fails_closed(self):
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"value": {"rtol": 2.0, "atol": 0.0}},
        }
        result = runner.compare_json_metrics(
            'gpmeep-metrics:{"value":1e308}\n',
            'gpmeep-metrics:{"value":-1e308}\n',
            comparison,
        )
        self.assertEqual(result["outcome"], "MISMATCH")
        self.assertTrue(result["failures"][0]["arithmetic_overflow"])
        self.assertIsNone(result["failures"][0]["absolute_error"])
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("Infinity", serialized)
        self.assertNotIn("NaN", serialized)

    def test_json_metrics_enforces_tolerance_and_path_byte_budgets(self):
        excessive_rules = {
            f"value[{index}]": {"atol": 0.0}
            for index in range(runner.MAX_JSON_TOLERANCE_RULES + 1)
        }
        with self.assertRaisesRegex(runner.ManifestError, "tolerance rules"):
            runner.compare_json_metrics(
                'gpmeep-metrics:{"value":[1]}\n',
                'gpmeep-metrics:{"value":[1]}\n',
                {
                    "mode": "json_metrics",
                    "prefix": "gpmeep-metrics:",
                    "tolerances": excessive_rules,
                },
            )
        long_key = "k" * (runner.MAX_JSON_METRIC_PATH_BYTES + 1)
        payload = "gpmeep-metrics:" + json.dumps({long_key: 1.0}) + "\n"
        with self.assertRaisesRegex(runner.ManifestError, "path exceeds"):
            runner.extract_json_metrics(
                payload,
                {
                    "mode": "json_metrics",
                    "prefix": "gpmeep-metrics:",
                    "tolerances": {"*": {"atol": 0.0}},
                },
            )

    def test_json_metrics_inline_policy_has_a_byte_budget(self):
        values = {
            (f"metric_{index:04d}_" + "x" * 380): float(index)
            for index in range(3000)
        }
        payload = "gpmeep-metrics:" + json.dumps(values) + "\n"
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"*": {"atol": 0.0}},
        }
        result = runner.compare_json_metrics(payload, payload, comparison)
        self.assertEqual(result["outcome"], "PASS")
        self.assertLess(len(values), runner.MAX_INLINE_JSON_METRICS)
        self.assertGreater(
            2 * result["metric_evidence"]["flattened_storage_bytes"],
            runner.MAX_INLINE_JSON_BYTES,
        )
        self.assertFalse(result["metric_evidence"]["inlined"])
        self.assertNotIn("cpu_metrics", result)

    def test_normalized_stdout_report_is_hash_bound_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "cpu.log").write_text("x" * 20000, encoding="utf-8")
            (root / "cuda.log").write_text("x" * 20000, encoding="utf-8")
            runs = {
                backend: {
                    "outcome": "PASS",
                    "duration_seconds": 1.0,
                    "stdout_log": f"{backend}.log",
                }
                for backend in ("cpu", "cuda")
            }
            result = runner.compare_runs(
                {
                    "comparison": {
                        "mode": "normalized_stdout",
                        "ignore_regex": [],
                    }
                },
                runs,
                root,
            )
        self.assertEqual(result["outcome"], "PASS")
        evidence = result["normalized_stdout_evidence"]
        self.assertTrue(evidence["cpu"]["truncated"])
        self.assertLessEqual(
            evidence["cpu"]["preview_bytes"],
            runner.MAX_NORMALIZED_STDOUT_PREVIEW_BYTES,
        )
        self.assertNotIn("cpu_normalized_stdout", result)

    def test_case_directory_id_separates_equal_slugs(self):
        self.assertNotEqual(
            runner.case_directory_id("a/b"),
            runner.case_directory_id("a_b"),
        )
        self.assertEqual(
            runner.case_directory_id("a/b"),
            runner.case_directory_id("a/b"),
        )

    def test_prepare_output_rejects_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "evidence"
            output.mkdir()
            (output / "stale").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.ManifestError, "new or empty"):
                runner.prepare_output(output)

    def test_manifest_rejects_duplicate_json_keys(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        manifest.write_text(
            '{"schema_version":1,"schema_version":1}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            runner.ManifestError, "duplicate key"
        ):
            runner.load_json(manifest)

    def test_manifest_requires_known_compute_scope(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        del value["inventory"][0]["compute_scope"]
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(runner.ManifestError, "compute_scope"):
            runner.materialize_cases(runner.load_json(manifest), root)

        value["inventory"][0]["compute_scope"] = "accelerate_everything"
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(runner.ManifestError, "compute_scope"):
            runner.materialize_cases(runner.load_json(manifest), root)

    def test_every_disposition_rejects_incompatible_compute_scope(self):
        allowed = runner.DISPOSITION_COMPUTE_SCOPES
        self.assertSetEqual(set(allowed), runner.KNOWN_DISPOSITIONS)
        base = {
            "id": "cases/scope.py",
            "path": "cases/scope.py",
            "kind": "example",
            "reason": "Synthetic scope contract.",
            "tier": ["smoke"],
            "milestone": "M-self-test",
        }
        for disposition, permitted in sorted(allowed.items()):
            forbidden = sorted(runner.KNOWN_COMPUTE_SCOPES - permitted)[0]
            case = {**base, "disposition": disposition, "compute_scope": forbidden}
            with self.subTest(disposition=disposition, scope=forbidden):
                with self.assertRaisesRegex(
                    runner.ManifestError, "requires compute_scope"
                ):
                    runner.validate_case(
                        case,
                        pathlib.Path("/does-not-matter"),
                        require_source_file=False,
                    )

    def test_host_only_run_rejects_cuda_dispatch_claim(self):
        case = {
            "id": "cases/host.py",
            "path": "cases/host.py",
            "kind": "example",
            "disposition": "run",
            "reason": "Synthetic host-only runnable.",
            "tier": ["smoke"],
            "timeout_seconds": 10,
            "comparison": {"mode": "normalized_stdout", "ignore_regex": []},
            "compute_scope": "host_only",
            "gpu_contract": "cuda_dispatch",
        }
        with self.assertRaisesRegex(
            runner.ManifestError, "host_only.*gpu_contract none"
        ):
            runner.validate_case(
                case,
                pathlib.Path("/does-not-matter"),
                require_source_file=False,
            )

    def test_required_cuda_activity_categories_are_explicitly_registered(self):
        base = {
            "id": "cases/activity.py",
            "path": "cases/activity.py",
            "kind": "example",
            "disposition": "run",
            "reason": "Synthetic required CUDA activity contract.",
            "tier": ["smoke"],
            "timeout_seconds": 10,
            "comparison": {"mode": "normalized_stdout", "ignore_regex": []},
            "compute_scope": "fdtd_cuda",
            "gpu_contract": "cuda_dispatch",
        }
        accepted = (
            "dft_materializations.cuda_dft_output_calls",
            "dft_materializations.cuda_dft_materialization_kernel_launches",
            "dft_materializations.cuda_dft_materialization_result_device_to_host_bytes",
            "dft_materializations.dft_array_mpi_allreduce_calls",
            "near2far.near2far_mpi_allreduce_calls",
        )
        for counter in accepted:
            with self.subTest(counter=counter):
                case = {**base, "required_cuda_call_counters": [counter]}
                runner.validate_case(
                    case,
                    pathlib.Path("/does-not-matter"),
                    require_source_file=False,
                )

        for counter in (
            "dft_materializations.host_synthetic_material_array_calls",
            "dft_materializations.dft_array_mpi_allreduce_bytes",
            "not_registered.cuda_activity_calls",
        ):
            with self.subTest(rejected=counter), self.assertRaisesRegex(
                runner.ManifestError, "registered CUDA activity counters"
            ):
                case = {**base, "required_cuda_call_counters": [counter]}
                runner.validate_case(
                    case,
                    pathlib.Path("/does-not-matter"),
                    require_source_file=False,
                )

    def test_mpi_aggregate_and_per_rank_activity_schema_is_fail_closed(self):
        base = {
            "id": "cases/mpi-activity.py",
            "path": "cases/mpi-activity.py",
            "kind": "unittest",
            "disposition": "run",
            "reason": "Synthetic MPI aggregate and rank participation contract.",
            "tier": ["full"],
            "timeout_seconds": 10,
            "comparison": {"mode": "embedded_oracle"},
            "compute_scope": "fdtd_cuda",
            "gpu_contract": "cuda_dispatch",
            "expected_unittest": {
                "count": 1,
                "identities": [
                    "test_mpi_activity.MpiActivityTest.test_contract"
                ],
            },
            "required_cuda_call_counters": [
                "near2far.cuda_near2far_adjoint_calls"
            ],
            "mpi_aggregate_cuda_call_counters": [
                "near2far.cuda_near2far_adjoint_calls"
            ],
            "mpi_required_per_rank_call_counters": [
                "near2far.near2far_mpi_allreduce_calls"
            ],
        }
        runner.validate_case(
            base,
            pathlib.Path("/does-not-matter"),
            require_source_file=False,
        )

        invalid = copy.deepcopy(base)
        invalid["mpi_aggregate_cuda_call_counters"] = [
            "near2far.cuda_near2far_transform_calls"
        ]
        with self.assertRaisesRegex(
            runner.ManifestError, "subset of required_cuda_call_counters"
        ):
            runner.validate_case(
                invalid,
                pathlib.Path("/does-not-matter"),
                require_source_file=False,
            )

        invalid = copy.deepcopy(base)
        invalid["mpi_required_per_rank_call_counters"] = [
            "near2far.cuda_near2far_adjoint_calls"
        ]
        with self.assertRaisesRegex(
            runner.ManifestError, "registered shared activity counters"
        ):
            runner.validate_case(
                invalid,
                pathlib.Path("/does-not-matter"),
                require_source_file=False,
            )

        invalid = copy.deepcopy(base)
        invalid["mpi_required_per_rank_call_counters"] = []
        with self.assertRaisesRegex(
            runner.ManifestError, "requires explicit per-rank shared activity"
        ):
            runner.validate_case(
                invalid,
                pathlib.Path("/does-not-matter"),
                require_source_file=False,
            )

    def test_covered_case_contract_rejects_self_duplicate_and_bad_relation(self):
        base = {
            "id": "cases/covered.py",
            "path": "cases/covered.py",
            "kind": "example",
            "disposition": "covered_by_test",
            "reason": "Synthetic structured coverage.",
            "tier": ["smoke"],
            "compute_scope": "coverage_only",
            "covered_by": ["cases/target.py"],
            "coverage_relation": "same_workflow",
        }
        mutations = {
            "self": (
                lambda case: case.update(covered_by=[case["id"]]),
                "cannot reference itself",
            ),
            "duplicate": (
                lambda case: case.update(
                    covered_by=["cases/target.py", "cases/target.py"]
                ),
                "must be unique",
            ),
            "bad-relation": (
                lambda case: case.update(coverage_relation="exact-ish"),
                "known coverage_relation",
            ),
            "bad-scope": (
                lambda case: case.update(compute_scope="fdtd_cuda"),
                "requires compute_scope (?:in )?coverage_only",
            ),
        }
        for label, (mutate, diagnostic) in mutations.items():
            with self.subTest(label=label):
                case = copy.deepcopy(base)
                mutate(case)
                with self.assertRaisesRegex(runner.ManifestError, diagnostic):
                    runner.validate_case(
                        case, pathlib.Path("/does-not-matter"),
                        require_source_file=False,
                    )

    def test_coverage_graph_rejects_missing_nonterminal_and_cycle(self):
        def run(case_id):
            return {"id": case_id, "path": case_id, "disposition": "run"}

        def covered(case_id, targets):
            return {
                "id": case_id,
                "path": case_id,
                "disposition": "covered_by_test",
                "covered_by": targets,
            }

        invalid = {
            "missing": (
                [covered("covered", ["missing"]), run("target")],
                "target does not exist",
            ),
            "nonterminal": (
                [
                    covered("covered", ["delegate"]),
                    covered("delegate", ["target"]),
                    run("target"),
                ],
                "must be terminal runnable",
            ),
            "cycle": (
                [covered("left", ["right"]), covered("right", ["left"])],
                "coverage cycle",
            ),
        }
        for label, (cases, diagnostic) in invalid.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(runner.ManifestError, diagnostic):
                    runner.validate_coverage_graph(cases)

    def test_coverage_outcomes_require_all_selected_targets_to_pass(self):
        cases = [
            {
                "id": "covered",
                "disposition": "covered_by_test",
                "covered_by": ["target-a", "target-b"],
            },
            {"id": "target-a", "disposition": "run"},
            {"id": "target-b", "disposition": "run"},
        ]

        def results(second_outcome="PASS", second_selected=True):
            return [
                {"id": "covered", "selected": True, "outcome": "COVERED_BY_TEST"},
                {"id": "target-a", "selected": True, "outcome": "PASS"},
                {
                    "id": "target-b",
                    "selected": second_selected,
                    "outcome": second_outcome,
                },
            ]

        passing = results()
        runner.resolve_coverage_outcomes(cases, passing)
        self.assertEqual(passing[0]["outcome"], runner.COVERED_PASS_OUTCOME)

        failed = results(second_outcome="MISMATCH")
        runner.resolve_coverage_outcomes(cases, failed)
        self.assertEqual(failed[0]["outcome"], runner.COVERAGE_FAILED_OUTCOME)
        self.assertEqual(
            runner.outcome_exit_code([failed[0]]), runner.EXIT_BLOCKED
        )

        unresolved = results(second_selected=False)
        runner.resolve_coverage_outcomes(cases, unresolved)
        self.assertEqual(
            unresolved[0]["outcome"], runner.COVERAGE_UNRESOLVED_OUTCOME
        )
        self.assertEqual(
            runner.outcome_exit_code([unresolved[0]]), runner.EXIT_BLOCKED
        )

    def test_full_runner_derives_covered_pass_from_runnable_target(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (root / "cases" / "covered_case.py").write_text(
            "raise AssertionError('covered case must not execute')\n",
            encoding="utf-8",
        )
        value = json.loads(manifest.read_text())
        value["overrides"] = [
            {
                "glob": "cases/covered_case.py",
                "disposition": "covered_by_test",
                "reason": "Synthetic full-runner coverage dependency.",
                "compute_scope": "coverage_only",
                "covered_by": ["cases/pass_case.py"],
                "coverage_relation": "same_workflow",
            }
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")

        status = self.invoke(root, manifest)

        self.assertEqual(status, runner.EXIT_OK)
        report = json.loads((root / "evidence" / "report.json").read_text())
        results = {result["id"]: result for result in report["results"]}
        self.assertEqual(results["cases/pass_case.py"]["outcome"], "PASS")
        self.assertEqual(
            results["cases/covered_case.py"]["outcome"],
            runner.COVERED_PASS_OUTCOME,
        )
        self.assertEqual(results["cases/covered_case.py"]["runs"], {})

    def test_classified_nonrun_case_remains_in_report(self):
        temporary, root, manifest = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        value = json.loads(manifest.read_text())
        value["overrides"] = [
            {
                "glob": "cases/pass_case.py",
                "disposition": "expected_feature_gap",
                "reason": "Synthetic known gap.",
                "milestone": "M-self-test",
            }
        ]
        manifest.write_text(json.dumps(value), encoding="utf-8")
        status = self.invoke(root, manifest, "--backend", "cpu")
        self.assertEqual(status, runner.EXIT_BLOCKED)
        report = json.loads((root / "evidence" / "report.json").read_text())
        self.assertEqual(report["results"][0]["outcome"], "EXPECTED_FEATURE_GAP")
        self.assertEqual(report["inventory"]["selected"], 1)

    def test_checked_in_manifest_accounts_for_complete_python_scope(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = runner.materialize_cases(runner.load_json(manifest), repo)
        materialized = {case["path"] for case in cases}
        expected = {
            path.relative_to(repo).as_posix()
            for path in (repo / "python" / "tests").glob("test_*.py")
        }
        expected.update(
            path.relative_to(repo).as_posix()
            for suffix in ("*.py", "*.ipynb")
            for path in (repo / "python" / "examples").rglob(suffix)
        )
        self.assertSetEqual(materialized, expected)
        self.assertEqual(len(cases), 207)
        self.assertFalse(any(path.startswith("scheme/") for path in materialized))

    def test_checked_in_manifest_has_structured_compute_and_coverage_scope(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["id"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }
        self.assertTrue(cases)
        self.assertTrue(
            all(case["compute_scope"] in runner.KNOWN_COMPUTE_SCOPES
                for case in cases.values())
        )
        expected = {
            "python/examples/adjoint_optimization/01-Introduction.ipynb": [
                "python/tests/test_adjoint_solver.py",
                "python/examples/waveguide_crossing.py",
            ],
            "python/examples/adjoint_optimization/02-Waveguide_Bend.ipynb": [
                "python/tests/test_adjoint_solver.py",
                "python/examples/waveguide_crossing.py",
            ],
            "python/examples/adjoint_optimization/03-Filtered_Waveguide_Bend.ipynb": [
                "python/examples/waveguide_crossing.py",
                "python/tests/test_adjoint_jax.py",
                "python/examples/adjoint_optimization/mode_converter.py",
            ],
            "python/examples/adjoint_optimization/04-Splitter.ipynb": [
                "python/examples/waveguide_crossing.py",
                "python/tests/test_adjoint_solver.py",
                "python/examples/adjoint_optimization/mode_converter.py",
            ],
            "python/examples/adjoint_optimization/05-Near2Far.ipynb": [
                "python/examples/adjoint_optimization/near2far_epigraph_validation.py"
            ],
            "python/examples/adjoint_optimization/06-Near2Far-Epigraph.ipynb": [
                "python/examples/adjoint_optimization/near2far_epigraph_validation.py"
            ],
            "python/examples/adjoint_optimization/07-Connectivity-Constraint.ipynb": [
                "python/examples/adjoint_optimization/connectivity_validation.py"
            ],
            "python/examples/adjoint_optimization/Bend Minimax.ipynb": [
                "python/examples/adjoint_optimization/mode_converter.py"
            ],
            "python/examples/adjoint_optimization/Fourier-Bend.ipynb": [
                "python/tests/test_adjoint_solver.py",
                "python/examples/waveguide_crossing.py",
            ],
            "python/examples/adjoint_optimization/Fourier-Metalens.ipynb": [
                "python/examples/adjoint_optimization/multilayer_opt.py",
                "python/examples/adjoint_optimization/near2far_epigraph_validation.py",
            ],
            "python/examples/bend-flux.py": ["python/tests/test_bend_flux.py"],
            "python/examples/cavity_arrayslice.py": [
                "python/tests/test_cavity_arrayslice.py"
            ],
            "python/examples/cyl-ellipsoid.py": [
                "python/tests/test_cyl_ellipsoid.py"
            ],
            "python/examples/extraction_eff_ldos.py": ["python/tests/test_ldos.py"],
            "python/examples/gaussian-beam.py": [
                "python/tests/test_gaussianbeam.py"
            ],
            "python/examples/holey-wvg-bands.py": [
                "python/tests/test_holey_wvg_bands.py"
            ],
            "python/examples/holey-wvg-cavity.py": [
                "python/tests/test_holey_wvg_cavity.py"
            ],
            "python/examples/oblique-source.py": [
                "python/tests/test_oblique_source.py"
            ],
            "python/examples/pw-source.py": ["python/tests/test_pw_source.py"],
            "python/examples/refl-angular.py": [
                "python/tests/test_refl_angular.py"
            ],
            "python/examples/refl_angular_bfast.ipynb": [
                "python/tests/test_refl_angular.py"
            ],
            "python/examples/ring-cyl.py": ["python/tests/test_ring_cyl.py"],
            "python/examples/ring.py": ["python/tests/test_ring.py"],
            "python/examples/wvg-src.py": ["python/tests/test_wvg_src.py"],
        }
        actual = {
            case_id: case["covered_by"]
            for case_id, case in cases.items()
            if case["disposition"] == "covered_by_test"
        }
        self.assertDictEqual(actual, expected)
        for case_id, targets in actual.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(cases[case_id]["compute_scope"], "coverage_only")
                self.assertIn(
                    cases[case_id]["coverage_relation"],
                    runner.KNOWN_COVERAGE_RELATIONS,
                )
                self.assertTrue(all(cases[target]["disposition"] == "run"
                                    for target in targets))

    def test_checked_in_source_eigenmode_suite_pins_exact_unittest_contracts(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }
        expected_counts = {
            "python/tests/test_boundaries_1D.py": 1,
            "python/tests/test_diffracted_planewave.py": 1,
            "python/tests/test_gaussianbeam.py": 1,
            "python/tests/test_integrated_source.py": 1,
            "python/tests/test_mode_coeffs.py": 6,
            "python/tests/test_mode_decomposition.py": 6,
            "python/tests/test_oblique_source.py": 1,
            "python/tests/test_planewave_1D.py": 1,
            "python/tests/test_pw_source.py": 1,
            "python/tests/test_source.py": 9,
            "python/tests/test_wvg_src.py": 1,
        }
        for path, count in expected_counts.items():
            with self.subTest(path=path):
                contract = cases[path].get("expected_unittest")
                self.assertIsInstance(contract, dict)
                self.assertEqual(contract.get("count"), count)
                identities = contract.get("identities")
                self.assertIsInstance(identities, list)
                self.assertEqual(len(identities), count)
                self.assertEqual(len(set(identities)), count)

    def test_checked_in_gpu_backend_contract_matches_source_tests(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        path = "python/tests/test_gpu_backend.py"
        tree = ast.parse((repo / path).read_text(encoding="utf-8"))
        source_identities = sorted(
            f"test_gpu_backend.{class_node.name}.{method_node.name}"
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef)
            for method_node in class_node.body
            if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method_node.name.startswith("test")
        )
        contract = cases[path]["expected_unittest"]
        self.assertEqual(contract["count"], len(source_identities))
        self.assertListEqual(contract["identities"], source_identities)

    def test_checked_in_gpu_eigenmode_contract_matches_source_tests(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        path = "python/tests/test_gpu_eigenmode_overlap.py"
        tree = ast.parse((repo / path).read_text(encoding="utf-8"))
        source_identities = sorted(
            f"test_gpu_eigenmode_overlap.{class_node.name}.{method_node.name}"
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef)
            for method_node in class_node.body
            if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method_node.name.startswith("test")
        )
        contract = cases[path]["expected_unittest"]
        self.assertEqual(contract["count"], len(source_identities))
        self.assertListEqual(contract["identities"], source_identities)
        self.assertEqual(cases[path]["allowed_unittest_skips"], 1)
        self.assertListEqual(
            cases[path]["allowed_unittest_skip_details"],
            [
                {
                    "test": (
                        "*test_injected_failures_retry_without_leaking_"
                        "resident_plans*"
                    ),
                    "reason": "CUDA failure-atomicity test",
                }
            ],
        )

    def test_checked_in_dft_fields_requires_all_resident_consumers(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        self.assertSetEqual(
            set(cases["python/tests/test_dft_fields.py"][
                "required_cuda_call_counters"
            ]),
            {
                "dft_materializations.cuda_dft_output_calls",
                "dft_materializations.cuda_dft_array_materialization_calls",
                "dft_materializations.cuda_dft_output_staging_calls",
                "dft_materializations.cuda_dft_materialization_kernel_launches",
                "dft_materializations.cuda_dft_materialization_result_device_to_host_bytes",
                "dft_materializations.dft_array_mpi_allreduce_calls",
            },
        )

    def test_checked_in_top_level_example_duplicate_allowlist_is_pinned(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }
        approved = {
            "python/examples/bend-flux.py",
            "python/examples/cavity_arrayslice.py",
            "python/examples/cyl-ellipsoid.py",
            "python/examples/extraction_eff_ldos.py",
            "python/examples/gaussian-beam.py",
            "python/examples/holey-wvg-bands.py",
            "python/examples/holey-wvg-cavity.py",
            "python/examples/oblique-source.py",
            "python/examples/pw-source.py",
            "python/examples/refl-angular.py",
            "python/examples/ring.py",
            "python/examples/ring-cyl.py",
            "python/examples/wvg-src.py",
        }
        rejected = {
            "python/examples/3rd-harm-1d.py",
            "python/examples/absorber-1d.py",
            "python/examples/antenna-radiation.py",
            "python/examples/cavity-farfield.py",
            "python/examples/diffracted_planewave.py",
            "python/examples/material-dispersion.py",
            "python/examples/mode-decomposition.py",
        }
        actual = {
            path
            for path, case in cases.items()
            if pathlib.PurePosixPath(path).parent
            == pathlib.PurePosixPath("python/examples")
            and path.endswith(".py")
            and case["disposition"] == "covered_by_test"
        }
        self.assertSetEqual(actual, approved)
        for path in rejected:
            with self.subTest(path=path):
                self.assertNotEqual(cases[path]["disposition"], "covered_by_test")

    def test_near2far_examples_fail_closed_on_host_transform_fallback(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        counter = "near2far.cuda_near2far_transform_calls"
        actual = {
            path
            for path, case in cases.items()
            if counter in case.get("required_cuda_call_counters", [])
        }
        self.assertSetEqual(
            actual,
            {
                "python/examples/adjoint_optimization/near2far_3d_validation.py",
                "python/examples/antenna-radiation.py",
                "python/examples/antenna_pec_ground_plane.py",
                "python/examples/binary_grating_n2f.py",
                "python/examples/cavity-farfield.py",
                "python/examples/differential_cross_section.py",
                "python/examples/dipole_in_vacuum_cyl_off_axis.py",
                "python/examples/dipole_in_vacuum_cyl_on_axis.py",
                "python/examples/disc_extraction_efficiency.py",
                "python/examples/disc_radiation_pattern.py",
                "python/examples/metasurface_lens.py",
                "python/examples/zone_plate.py",
            },
        )
        for path in actual:
            with self.subTest(path=path):
                self.assertEqual(cases[path]["gpu_contract"], "cuda_dispatch")
                self.assertTrue(cases[path]["reason"].strip())
        adjoint_counter = "near2far.cuda_near2far_adjoint_calls"
        self.assertSetEqual(
            {
                path
                for path, case in cases.items()
                if adjoint_counter
                in case.get("required_cuda_call_counters", [])
            },
            {
                "python/examples/adjoint_optimization/"
                "near2far_3d_validation.py",
                "python/examples/adjoint_optimization/"
                "near2far_epigraph_validation.py",
                "python/examples/adjoint_optimization/"
                "connectivity_validation.py",
                "python/tests/test_adjoint_cyl.py",
            },
        )

    def test_direct_presentation_output_examples_use_validation_mode(self):
        """Full-suite work directories must contain only declared evidence.

        These examples normally write plots or HDF5/NPZ presentation artifacts.
        Their checked-in validation profiles retain the numerical workflow while
        suppressing the persistent plot/data files (or, for bent-waveguide,
        writing HDF5 into an automatically cleaned temporary directory).
        """
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }
        expected = {
            "python/examples/3rd-harm-1d.py",
            "python/examples/absorbed_power_density.py",
            "python/examples/antenna_pec_ground_plane.py",
            "python/examples/bent-waveguide.py",
            "python/examples/cylinder_cross_section.py",
            "python/examples/dipole_in_vacuum_cyl_on_axis.py",
            "python/examples/oblique-planewave.py",
            "python/examples/solve-cw.py",
        }
        for path in expected:
            with self.subTest(path=path):
                case = cases[path]
                self.assertEqual(case["disposition"], "run")
                self.assertIn("--validation", case["command"])
                self.assertIn(
                    "--validation",
                    (repo / path).read_text(encoding="utf-8"),
                )

    def test_oblique_planewave_fp32_floor_is_local_and_bounded(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        case = cases["python/examples/oblique-planewave.py"]
        tolerances = case["comparison"]["tolerances"]
        self.assertEqual(
            {"rtol": 1e-4, "atol": 2e-5},
            tolerances["example_results.oblique_field_samples.*"],
        )
        self.assertEqual(
            {"rtol": 1e-4, "atol": 1e-5},
            tolerances["example_results.oblique_y_field.*"],
        )
        self.assertIn("local 2e-5 absolute FP32 floor", case["reason"])

    def test_checked_in_generic_example_tolerances_cover_per_run_vectors(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = runner.materialize_cases(runner.load_json(manifest), repo)
        metrics = {
            "run_count": 1,
            "timestep": {"final": 10, "sum": 10, "weighted_checksum": 10},
            "timestep_delta": {
                "final": 10,
                "sum": 10,
                "weighted_checksum": 10,
            },
            "meep_time": {
                "final": 0.5,
                "sum": 0.5,
                "l2": 0.5,
                "weighted_checksum": 0.5,
            },
            "field_energy": {
                "final": 2.0,
                "sum": 2.0,
                "l2": 2.0,
                "weighted_checksum": 2.0,
            },
            "dft_norm": {
                "final": 3.0,
                "sum": 3.0,
                "l2": 3.0,
                "weighted_checksum": 3.0,
            },
            "per_run": {
                "timestep": [10],
                "timestep_delta": [10],
                "meep_time": [0.5],
                "field_energy": [2.0],
                "dft_norm": [3.0],
            },
        }
        generic_cases = [
            case
            for case in cases
            if any(
                str(item).endswith("run_example_oracle.py")
                for item in case.get("command", [])
            )
        ]
        self.assertSetEqual(
            {case["path"] for case in generic_cases},
            {
                "python/examples/absorbed_power_density.py",
                "python/examples/antenna_pec_ground_plane.py",
                "python/examples/3rd-harm-1d.py",
                "python/examples/absorber-1d.py",
                "python/examples/antenna_pec_ground_plane_1D.py",
                "python/examples/binary_grating.py",
                "python/examples/binary_grating_phasemap.py",
                "python/examples/binary_grating_n2f.py",
                "python/examples/binary_grating_oblique.py",
                "python/examples/bent-waveguide.py",
                "python/examples/cavity-farfield.py",
                "python/examples/cherenkov-radiation.py",
                "python/examples/chirped_pulse.py",
                "python/examples/cylinder_cross_section.py",
                "python/examples/diffracted_planewave.py",
                "python/examples/disc_extraction_efficiency.py",
                "python/examples/disc_radiation_pattern.py",
                "python/examples/grating2d_triangular_lattice.py",
                "python/examples/dipole_in_vacuum_cyl_off_axis.py",
                "python/examples/dipole_in_vacuum_cyl_on_axis.py",
                "python/examples/dipole_in_vacuum_1D.py",
                "python/examples/edge_emitter_2D.py",
                "python/examples/edge_emitter_3D.py",
                "python/examples/finite_grating.py",
                "python/examples/metal-cavity-ldos.py",
                "python/examples/metasurface_lens.py",
                "python/examples/material-dispersion.py",
                "python/examples/mode_coeff_phase.py",
                "python/examples/mode-decomposition.py",
                "python/examples/oblique-planewave.py",
                "python/examples/parallel-wvgs-force.py",
                "python/examples/phase_in_material.py",
                "python/examples/planar_cavity_ldos.py",
                "python/examples/polarization_grating.py",
                "python/examples/perturbation_theory.py",
                "python/examples/perturbation_theory_2d.py",
                "python/examples/refl-angular-kz2d.py",
                "python/examples/refl-quartz.py",
                "python/examples/ring-mode-overlap.py",
                "python/examples/solve-cw.py",
                "python/examples/stochastic_emitter_line.py",
                "python/examples/stochastic_emitter_reciprocity.py",
                "python/examples/stochastic_emitter.py",
                "python/examples/straight-waveguide.py",
                "python/examples/antenna-radiation.py",
                "python/examples/zone_plate.py",
            },
        )
        def write_synthetic_payload(
            path, comparison, base_metrics, result_names, result_shapes
        ):
            def write_member(output, name, value, first):
                if not first:
                    output.write(",")
                output.write(json.dumps(name) + ":")
                output.write(json.dumps(value, separators=(",", ":")))
                return False

            def write_repeated_array(output, value, count):
                token = json.dumps(value, allow_nan=False)
                output.write("[")
                written = 0
                while written < count:
                    batch = min(8192, count - written)
                    if written:
                        output.write(",")
                    output.write(",".join([token] * batch))
                    written += batch
                output.write("]")

            with path.open("w", encoding="utf-8") as output:
                output.write(comparison["prefix"] + "{")
                first = True
                for name, value in base_metrics.items():
                    first = write_member(output, name, value, first)
                if result_names:
                    if not first:
                        output.write(",")
                    output.write('"example_results":{')
                    for result_index, name in enumerate(result_names):
                        if result_index:
                            output.write(",")
                        shape = result_shapes[name]
                        count = math.prod(shape) if shape else 1
                        real_value = 0.1
                        output.write(json.dumps(name) + ":{")
                        output.write(f'"count":{count},"real":')
                        write_repeated_array(output, real_value, count)
                        output.write(',"imag":')
                        write_repeated_array(output, 0.0, count)
                        output.write(
                            ',"l2":'
                            + json.dumps(real_value * math.sqrt(count))
                            + ',"max_abs":'
                            + json.dumps(real_value)
                            + ',"weighted_real":'
                            + json.dumps(real_value * count * (count + 1) / 2)
                            + ',"weighted_imag":0.0'
                        )
                        if shape:
                            output.write(
                                ',"shape":'
                                + json.dumps(list(shape), separators=(",", ":"))
                            )
                        else:
                            output.write(',"rank":0')
                        output.write("}")
                    output.write("}")
                output.write("}\n")

        for case in generic_cases:
            with self.subTest(path=case["path"]):
                comparison = case["comparison"]
                command = case["command"]
                result_names = [
                    command[index + 1]
                    for index, item in enumerate(command[:-1])
                    if item == "--result-vector"
                ]
                result_shape_specs = [
                    command[index + 1]
                    for index, item in enumerate(command[:-1])
                    if item == "--expected-result-shape"
                ]
                self.assertEqual(len(result_shape_specs), len(result_names))
                result_shapes = {}
                for spec in result_shape_specs:
                    name, raw_shape = spec.split("=", 1)
                    self.assertNotIn(name, result_shapes)
                    result_shapes[name] = (
                        ()
                        if raw_shape == "scalar"
                        else tuple(int(item) for item in raw_shape.split(","))
                    )
                self.assertSetEqual(set(result_shapes), set(result_names))
                expected_run_count = int(
                    command[command.index("--expected-run-count") + 1]
                )
                for range_option in (
                    "--expected-run-timestep-range",
                    "--expected-run-timestep-delta-range",
                ):
                    range_count = command.count(range_option)
                    if range_count:
                        self.assertEqual(range_count, expected_run_count)
                with tempfile.TemporaryDirectory(
                    prefix="gpmeep-static-vector-contract-"
                ) as temporary:
                    root = pathlib.Path(temporary)
                    cpu_payload = root / "cpu.stdout"
                    cuda_payload = root / "cuda.stdout"
                    write_synthetic_payload(
                        cpu_payload,
                        comparison,
                        metrics,
                        result_names,
                        result_shapes,
                    )
                    # A physical copy intentionally exercises independent
                    # complete CPU and CUDA payload parsing without retaining
                    # either multi-million-value document in Python memory.
                    shutil.copyfile(cpu_payload, cuda_payload)
                    result = runner.compare_json_metric_files(
                        cpu_payload, cuda_payload, comparison
                    )
                    self.assertEqual(result["outcome"], "PASS")
                    self.assertGreater(result["metric_evidence"]["count"], 0)

        binary_case = next(
            case
            for case in generic_cases
            if case["path"] == "python/examples/binary_grating.py"
        )
        binary_metrics = copy.deepcopy(metrics)
        binary_angle = [float(index) for index in range(210)]
        binary_tran = [0.1 + index * 1e-5 for index in range(210)]
        binary_metrics["example_results"] = {
            "mode_angle": {
                "shape": [210],
                "count": 210,
                "real": binary_angle,
                "imag": [0.0] * 210,
                "l2": math.sqrt(math.fsum(value * value for value in binary_angle)),
                "max_abs": max(binary_angle),
                "weighted_real": math.fsum(
                    (index + 1) * value
                    for index, value in enumerate(binary_angle)
                ),
                "weighted_imag": 0.0,
            },
            "mode_tran": {
                "shape": [210],
                "count": 210,
                "real": binary_tran,
                "imag": [0.0] * 210,
                "l2": math.sqrt(math.fsum(value * value for value in binary_tran)),
                "max_abs": max(binary_tran),
                "weighted_real": math.fsum(
                    (index + 1) * value
                    for index, value in enumerate(binary_tran)
                ),
                "weighted_imag": 0.0,
            },
            "direction_cosine_max_overshoot": {
                "rank": 0,
                "count": 1,
                "real": [0.0],
                "imag": [0.0],
                "l2": 0.0,
                "max_abs": 0.0,
                "weighted_real": 0.0,
                "weighted_imag": 0.0,
            },
        }
        comparison = binary_case["comparison"]
        cpu_payload = comparison["prefix"] + json.dumps(binary_metrics) + "\n"
        self.assertEqual(
            runner.compare_json_metrics(
                cpu_payload, cpu_payload, comparison
            )["outcome"],
            "PASS",
        )
        bad_metrics = copy.deepcopy(binary_metrics)
        bad_metrics["example_results"]["mode_tran"]["real"][0] = 0.3
        cuda_payload = comparison["prefix"] + json.dumps(bad_metrics) + "\n"
        self.assertEqual(
            runner.compare_json_metrics(
                cpu_payload, cuda_payload, comparison
            )["outcome"],
            "MISMATCH",
        )

    def test_feature_expansion_multi_run_cases_require_signal_from_every_run(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }
        expected = {
            "python/examples/3rd-harm-1d.py": ("--min-each-dft-norm", "1"),
            "python/examples/metal-cavity-ldos.py": (
                "--min-each-field-energy",
                "1e-7",
            ),
            "python/examples/dipole_in_vacuum_cyl_on_axis.py": (
                "--min-each-dft-norm",
                "0.1",
            ),
            "python/examples/parallel-wvgs-force.py": (
                "--min-each-dft-norm",
                "100000",
            ),
            "python/examples/stochastic_emitter.py": (
                "--min-each-dft-norm",
                "10",
            ),
        }
        for path, (option, threshold) in expected.items():
            with self.subTest(path=path):
                command = cases[path]["command"]
                self.assertEqual(command.count(option), 1)
                self.assertEqual(command[command.index(option) + 1], threshold)

    def test_phase_map_profiles_have_nontrivial_circular_phase_contracts(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(runner.load_json(manifest), repo)
        }

        binary_command = cases[
            "python/examples/binary_grating_phasemap.py"
        ]["command"]
        binary_thresholds = {
            binary_command[index + 1]
            for index, item in enumerate(binary_command[:-1])
            if item == "--min-result-l2"
        }
        self.assertSetEqual(
            {
                "binary_phasemap_phase_duty_spread=1",
                "binary_phasemap_phase_frequency_spread=1",
                "binary_phasemap_phase_polarization_difference=0.5",
            },
            {
                item
                for item in binary_thresholds
                if item.startswith("binary_phasemap_phase_")
                and not item.startswith("binary_phasemap_phase_unit_")
            },
        )

        metasurface_command = cases[
            "python/examples/metasurface_lens.py"
        ]["command"]
        shape_specs = {
            metasurface_command[index + 1]
            for index, item in enumerate(metasurface_command[:-1])
            if item == "--expected-result-shape"
        }
        self.assertTrue(
            {
                "metasurface_phase_unwrapped=9",
                "metasurface_profile_offsets=3",
                "metasurface_profile_duty_cycles=422",
                "metasurface_profile_target_phase_unit_real=422",
                "metasurface_profile_target_phase_unit_imag=422",
                "metasurface_profile_achieved_phase_unit_real=422",
                "metasurface_profile_achieved_phase_unit_imag=422",
                "metasurface_profile_circular_error=422",
            }.issubset(shape_specs)
        )
        error_spec = "metasurface_profile_circular_error=0.03"
        self.assertIn(error_spec, metasurface_command)
        self.assertEqual(
            metasurface_command[metasurface_command.index(error_spec) - 1],
            "--max-result-abs",
        )
        self.assertIn("metasurface_phase_range=6.28", metasurface_command)

    def test_extended_stochastic_profiles_pin_physics_contracts(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }

        line_command = cases[
            "python/examples/stochastic_emitter_line.py"
        ]["command"]
        self.assertIn("stochastic_line_basis_matrix.py", " ".join(line_command))
        self.assertEqual(
            line_command[line_command.index("--expected-run-count") + 1], "60"
        )
        self.assertEqual(
            line_command[
                line_command.index("--expected-total-timestep-delta") + 1
            ],
            "960000",
        )
        self.assertTrue(
            {
                "line_flat_basis_flux=16,2",
                "line_flat_basis_flux_full=16,15",
                "line_flat_rotated_flux=16,2",
                "line_textured_basis_flux=16,2",
                "line_textured_basis_flux_full=16,15",
                "line_textured_rotated_flux=16,2",
                "line_scaled_high_mode_indices=13",
                "line_flat_scaled_high_mode_flux=16,13",
                "line_textured_scaled_high_mode_flux=16,13",
                "line_scaled_high_mode_callback_counts=2,13",
                "line_scaled_high_mode_residual=2,16,13",
                "line_scaled_high_mode_reference_l2=2,13",
                "line_scaled_high_mode_residual_l2=2,13",
                "line_scaled_high_mode_relative_l2=2,13",
                "line_basis_callback_counts=2,15",
                "line_unitary_closure_residual=2,16",
                "line_basis_gram_error=2,2",
                "line_full_basis_gram_error=15,15",
                "line_full_basis_active_mode_mask=2,15",
                "line_full_basis_active_mode_count=2",
                "line_full_basis_pairwise_relative_l2=2,15,15",
                "line_full_basis_min_pairwise_relative_l2=2",
                "line_m12_m15_convergence_relative_l2=1",
            }.issubset(
                {
                    line_command[index + 1]
                    for index, item in enumerate(line_command[:-1])
                    if item == "--expected-result-shape"
                }
            )
        )
        for gate in (
            "line_scaled_high_mode_residual=0.01",
            "line_scaled_high_mode_residual_l2=0.01",
            "line_scaled_high_mode_relative_l2=0.01",
        ):
            self.assertIn(gate, line_command)
            self.assertEqual(
                line_command[line_command.index(gate) - 1], "--max-result-abs"
            )
        pairwise_gate = "line_full_basis_min_pairwise_relative_l2=0.05"
        self.assertIn(pairwise_gate, line_command)
        self.assertEqual(
            line_command[line_command.index(pairwise_gate) - 1], "--min-result-l2"
        )
        active_count_gate = "line_full_basis_active_mode_count=15"
        self.assertIn(active_count_gate, line_command)
        self.assertEqual(
            line_command[line_command.index(active_count_gate) - 1],
            "--min-result-l2",
        )
        line_tolerances = cases[
            "python/examples/stochastic_emitter_line.py"
        ]["comparison"]["tolerances"]
        self.assertEqual(
            line_tolerances["example_results.line_full_basis_active_mode_mask.*"],
            {"rtol": 0.0, "atol": 0.0},
        )
        self.assertEqual(
            line_tolerances["example_results.line_full_basis_active_mode_count.*"],
            {"rtol": 0.0, "atol": 0.0},
        )
        self.assertEqual(
            line_tolerances["example_results.line_scaled_high_mode_residual.*"][
                "atol"
            ],
            1e-4,
        )
        self.assertEqual(
            line_tolerances[
                "example_results.line_scaled_high_mode_residual_l2.*"
            ]["atol"],
            5e-6,
        )
        line_wrapper = (
            repo / "scripts/python-validation/stochastic_line_basis_matrix.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ACTIVE_MODE_RELATIVE_L2_THRESHOLD = 1e-5", line_wrapper)
        self.assertIn(
            "MIN_FULL_BASIS_ACTIVE_MODE_COUNTS = (9, 13)", line_wrapper
        )

        reciprocity_command = cases[
            "python/examples/stochastic_emitter_reciprocity.py"
        ]["command"]
        self.assertIn(
            "stochastic_reciprocity_matrix.py", " ".join(reciprocity_command)
        )
        self.assertEqual(
            reciprocity_command[
                reciprocity_command.index("--expected-run-count") + 1
            ],
            "64",
        )
        self.assertEqual(
            reciprocity_command[
                reciprocity_command.index("--expected-final-timestep") + 1
            ],
            "517000",
        )
        self.assertNotIn(
            "--expected-total-timestep-delta", reciprocity_command
        )
        timestep_delta_flag = "--expected-run-timestep-delta-range"
        actual_timestep_delta_sequence = [
            reciprocity_command[index + 1]
            for index, value in enumerate(reciprocity_command)
            if value == timestep_delta_flag
        ]
        expected_timestep_delta_sequence = (
            ["69000:69000"] * 2
            + ["261000:261000"] * 29
            + [
                "69000:69000",
                "261000:261000",
                "133000:133000",
            ]
            + ["517000:517000"] * 28
            + ["133000:133000", "517000:517000"]
        )
        self.assertEqual(
            actual_timestep_delta_sequence,
            expected_timestep_delta_sequence,
        )
        self.assertEqual(
            cases["python/examples/stochastic_emitter_reciprocity.py"][
                "timeout_seconds"
            ],
            14400,
        )
        self.assertIn(
            "reciprocity_forward_textured_unique_flux=4,28",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_forward_textured_unique_doubled_flux=4,28",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_backward_metadata_raw_power=4,4,58",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_convergence_relative_l2=4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_convergence_pointwise_relative_error=4,4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_convergence_scale_aware_pointwise_relative_error=4,4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_forward_normalized_convergence_scale_aware_pointwise_relative_error=4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_backward_normalized_convergence_scale_aware_pointwise_relative_error=4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_scale_aware_pointwise_relative_error=4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_max_scale_aware_pointwise_relative_error=1",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_backward_expected_recomputed_flux=4,4",
            reciprocity_command,
        )
        self.assertIn(
            "reciprocity_backward_metadata_weighted_ghost_fraction=4,4",
            reciprocity_command,
        )
        for gate in (
            "reciprocity_convergence_scale_aware_pointwise_relative_error=0.05",
            "reciprocity_forward_normalized_convergence_relative_l2=0.02",
            "reciprocity_forward_normalized_convergence_scale_aware_pointwise_relative_error=0.05",
            "reciprocity_backward_normalized_convergence_relative_l2=0.02",
            "reciprocity_backward_normalized_convergence_scale_aware_pointwise_relative_error=0.05",
            "reciprocity_scale_aware_pointwise_relative_error=0.05",
            "reciprocity_relative_l2=0.05",
            "reciprocity_max_scale_aware_pointwise_relative_error=0.05",
            "reciprocity_backward_metadata_ghost_weight_max_abs=1e-14",
            "reciprocity_backward_metadata_weighted_ghost_contribution=1e-12",
            "reciprocity_backward_metadata_weighted_ghost_fraction=1e-12",
        ):
            self.assertIn(gate, reciprocity_command)
            self.assertEqual(
                reciprocity_command[reciprocity_command.index(gate) - 1],
                "--max-result-abs",
            )
        for forbidden_pure_gate in (
            "reciprocity_convergence_pointwise_relative_error=0.1",
            "reciprocity_forward_normalized_convergence_pointwise_relative_error=0.05",
            "reciprocity_backward_normalized_convergence_pointwise_relative_error=0.05",
            "reciprocity_pointwise_relative_error=0.05",
            "reciprocity_max_relative_error=0.05",
        ):
            self.assertNotIn(forbidden_pure_gate, reciprocity_command)

        reciprocity_tolerances = cases[
            "python/examples/stochastic_emitter_reciprocity.py"
        ]["comparison"]["tolerances"]
        diagnostic_only_rules = {
            "example_results.reciprocity_forward_normalized_convergence_pointwise_relative_error.*",
            "example_results.reciprocity_backward_normalized_convergence_pointwise_relative_error.*",
            "example_results.reciprocity_base_pointwise_relative_error.*",
            "example_results.reciprocity_pointwise_relative_error.*",
            "example_results.reciprocity_max_relative_error.*",
            "example_results.reciprocity_convergence_pointwise_relative_error.*",
        }
        self.assertEqual(
            {
                rule
                for rule, tolerance in reciprocity_tolerances.items()
                if tolerance.get("diagnostic_only", False)
            },
            diagnostic_only_rules,
        )
        for scale_aware_rule in (
            "example_results.reciprocity_forward_normalized_convergence_scale_aware_pointwise_relative_error.*",
            "example_results.reciprocity_backward_normalized_convergence_scale_aware_pointwise_relative_error.*",
            "example_results.reciprocity_scale_aware_pointwise_relative_error.*",
            "example_results.reciprocity_convergence_scale_aware_pointwise_relative_error.*",
        ):
            self.assertFalse(
                reciprocity_tolerances[scale_aware_rule].get(
                    "diagnostic_only", False
                )
            )

    def test_absorber_complete_trace_remains_gating_while_checksum_is_diagnostic(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        cases = {
            case["path"]: case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
        }
        absorber = cases["python/examples/absorber-1d.py"]
        tolerances = absorber["comparison"]["tolerances"]
        checksum = tolerances[
            "example_results.absorber_trace_fields.weighted_real"
        ]
        complete_trace = tolerances[
            "example_results.absorber_trace_fields.*"
        ]

        self.assertTrue(checksum["diagnostic_only"])
        self.assertEqual(checksum["rtol"], 0.02)
        self.assertEqual(checksum["atol"], 1e-8)
        self.assertFalse(complete_trace.get("diagnostic_only", False))
        self.assertEqual(complete_trace, {"rtol": 0.02, "atol": 1e-8})
        self.assertIn("complete pointwise trace", absorber["reason"])
        self.assertIn("ill-conditioned", absorber["reason"])

    def test_finite_grating_pins_full_result_shape_before_sampling(self):
        source_path = (
            RUNNER_PATH.parents[2] / "python/examples/finite_grating.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
            and target.id == "expected_scattered_field_shape"
        }
        self.assertEqual(
            assignments.get("expected_scattered_field_shape"), (501, 352)
        )
        shape_guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and ast.unparse(node.test)
            == "scattered_field.shape != expected_scattered_field_shape"
            and any(
                isinstance(child, ast.Raise) for child in ast.walk(node)
            )
        ]
        self.assertEqual(len(shape_guards), 1)

    def test_checked_in_runnable_unittests_pin_exact_contracts(self):
        repo = RUNNER_PATH.parents[2]
        manifest = RUNNER_PATH.parent / "manifest.json"
        runnable = [
            case
            for case in runner.materialize_cases(
                runner.load_json(manifest), repo
            )
            if case["kind"] == "unittest"
            and case["disposition"] == "run"
        ]
        self.assertEqual(len(runnable), 72)
        total_identities = 0
        for case in runnable:
            with self.subTest(path=case["path"]):
                contract = case.get("expected_unittest")
                self.assertIsInstance(contract, dict)
                identities = contract.get("identities")
                self.assertIsInstance(identities, list)
                self.assertEqual(contract.get("count"), len(identities))
                self.assertEqual(len(set(identities)), len(identities))
                total_identities += len(identities)
        self.assertEqual(total_identities, 387)


if __name__ == "__main__":
    unittest.main()
