"""Capture process-local gpmeep GPU evidence for the validation runner.

Python imports ``sitecustomize`` automatically when its directory is on
``PYTHONPATH``.  This hook does not import Meep itself, so it cannot perturb
import order or test setup.  At interpreter shutdown it records provenance and
dispatch counters if the case imported Meep.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import pathlib
import sys
import tempfile
import time
import traceback
from fractions import Fraction


_STARTED = time.time()
_COMPLETION_PATH_AT_START = os.environ.get(
    "GPMEEP_VALIDATION_COMPLETE_FILE"
)
_COMPLETION_PREEXISTED = bool(
    _COMPLETION_PATH_AT_START
    and pathlib.Path(_COMPLETION_PATH_AT_START).exists()
)
_REGISTERED_INVOCATION_SHA256 = None
_REGISTERED_RESULT_SHA256 = None


_GPU_NATIVE_FUNCTIONS = (
    "_gpu_active_backend",
    "_gpu_backend_compiled",
    "_gpu_backend_diagnostic",
    "_gpu_compiled_architectures",
    "_gpu_device_count",
    "_gpu_device_identifier",
    "_gpu_device_integer_property",
    "_gpu_device_memory",
    "_gpu_device_memory_bandwidth",
    "_gpu_device_name",
    "_gpu_requested_backend",
    "_gpu_runtime_available",
    "_gpu_runtime_diagnostic",
    "_gpu_selected_device",
    "_gpu_selected_device_identifier",
    "_gpu_statistics",
    "is_single_precision",
    "with_mpi",
)


_GPU_STATISTIC_GROUPS = {
    "runtime": (
        "runtime_availability_probes",
        "runtime_device_enumerations",
        "runtime_device_selections",
    ),
    "dispatch": (
        "cpu_curl_calls",
        "cpu_curl_points",
        "cuda_curl_calls",
        "cuda_curl_points",
        "host_to_device_bytes",
        "device_to_host_bytes",
    ),
    "resident": (
        "host_to_device_bytes_avoided",
        "device_to_host_bytes_avoided",
        "device_buffer_allocations",
        "device_buffer_reuses",
        "live_resident_device_buffers",
    ),
    "field_updates": (
        "cpu_update_eh_calls",
        "cpu_update_eh_points",
        "cuda_update_eh_calls",
        "cuda_update_eh_points",
    ),
    "polarizations": (
        "cpu_polarization_calls",
        "cpu_polarization_points",
        "cuda_polarization_calls",
        "cuda_polarization_points",
    ),
    "sources": (
        "cpu_source_calls",
        "cpu_source_points",
        "cuda_source_calls",
        "cuda_source_points",
    ),
    "boundaries": (
        "cpu_boundary_calls",
        "cpu_boundary_points",
        "cuda_boundary_calls",
        "cuda_boundary_points",
    ),
    "dfts": (
        "cpu_dft_calls",
        "cpu_dft_points",
        "cuda_dft_calls",
        "cuda_dft_points",
    ),
    "dft_batches": (
        "dft_batch_calls",
        "dft_submitted_updates",
        "dft_phase_preparation_launches",
        "dft_phase_reuses",
        "dft_update_kernel_launches",
        "dft_maximum_batch_size",
        "dft_multi_monitor_automatic_checks",
        "dft_multi_monitor_automatic_selected",
        "dft_multi_monitor_automatic_rejected",
        "dft_multi_monitor_forced_batches",
        "dft_multi_monitor_batched_updates",
        "dft_multi_monitor_unbatched_updates",
        "dft_multi_monitor_plan_uploads",
        "dft_multi_monitor_plan_reuses",
        "dft_multi_monitor_metadata_host_to_device_bytes",
    ),
    "dft_reductions": (
        "cpu_dft_reduction_calls",
        "cpu_dft_reduction_pairs",
        "cpu_dft_reduction_terms",
        "cuda_dft_reduction_calls",
        "cuda_dft_reduction_pairs",
        "cuda_dft_reduction_terms",
        "cuda_dft_reduction_descriptor_uploads",
        "cuda_dft_reduction_plan_reuses",
        "cuda_dft_reduction_kernel_launches",
        "cuda_dft_reduction_result_device_to_host_bytes",
        "dft_reduction_full_dft_device_to_host_bytes_avoided",
        "dft_reduction_mpi_allreduce_calls",
        "dft_reduction_mpi_allreduce_bytes",
    ),
    "dft_materializations": (
        "cpu_dft_array_materialization_calls",
        "cpu_dft_array_materialization_points",
        "cuda_dft_array_materialization_calls",
        "cuda_dft_array_materialization_points",
        "host_synthetic_material_array_calls",
        "host_synthetic_material_array_points",
        "cpu_dft_output_calls",
        "cpu_dft_output_points",
        "cuda_dft_output_calls",
        "cuda_dft_output_points",
        "cuda_dft_output_staging_calls",
        "cuda_dft_output_staging_points",
        "cuda_dft_output_staging_frequencies",
        "cuda_dft_output_staging_descriptor_uploads",
        "cuda_dft_output_staging_plan_reuses",
        "cuda_dft_output_staging_kernel_launches",
        "cuda_dft_output_staging_result_device_to_host_bytes",
        "cuda_dft_output_staging_full_dft_device_to_host_bytes_avoided",
        "cuda_dft_output_staging_workspace_ceiling_bytes",
        "cuda_dft_materialization_kernel_launches",
        "cuda_dft_materialization_result_device_to_host_bytes",
        "dft_materialization_full_dft_device_to_host_bytes_avoided",
        "dft_array_mpi_allreduce_calls",
        "dft_array_mpi_allreduce_bytes",
    ),
    "dft_checkpoints": (
        "cpu_dft_checkpoint_save_calls",
        "cpu_dft_checkpoint_save_values",
        "cuda_dft_checkpoint_save_calls",
        "cuda_dft_checkpoint_save_values",
        "cuda_dft_checkpoint_save_device_to_host_bytes",
        "cuda_dft_checkpoint_save_full_cache_device_to_host_bytes_avoided",
        "cpu_dft_checkpoint_load_calls",
        "cpu_dft_checkpoint_load_values",
        "cuda_dft_checkpoint_load_calls",
        "cuda_dft_checkpoint_load_values",
        "cuda_dft_checkpoint_load_host_to_device_bytes",
    ),
    "dft_scales": (
        "cpu_dft_scale_calls",
        "cpu_dft_scale_values",
        "cuda_dft_scale_calls",
        "cuda_dft_scale_values",
        "cuda_dft_scale_kernel_launches",
        "cuda_dft_scale_host_to_device_bytes",
    ),
    "eigenmode_overlaps": (
        "cpu_eigenmode_overlap_calls",
        "cpu_eigenmode_overlap_terms",
        "cuda_eigenmode_overlap_calls",
        "cuda_eigenmode_overlap_terms",
        "cuda_eigenmode_mode_flux_calls",
        "cuda_eigenmode_mode_mode_calls",
        "cuda_eigenmode_submitted_pairs",
        "cuda_eigenmode_descriptor_uploads",
        "cuda_eigenmode_plan_reuses",
        "cuda_eigenmode_kernel_launches",
        "cuda_eigenmode_result_device_to_host_bytes",
        "eigenmode_full_dft_device_to_host_bytes_avoided",
        "host_mode_profile_sampling_calls",
        "host_mode_profile_sampling_points",
        "eigenmode_zero_rank_channels_skipped",
        "host_mode_profile_host_to_device_bytes",
        "eigenmode_mpi_allreduce_calls",
        "eigenmode_mpi_allreduce_bytes",
    ),
    "ldos": (
        "cpu_ldos_reduction_calls",
        "cpu_ldos_source_points",
        "cuda_ldos_reduction_calls",
        "cuda_ldos_submitted_profiles",
        "cuda_ldos_source_points",
        "cuda_ldos_descriptor_uploads",
        "cuda_ldos_kernel_launches",
        "cuda_ldos_result_device_to_host_bytes",
        "ldos_full_field_device_to_host_bytes_avoided",
    ),
    "near2far": (
        "cpu_near2far_transform_calls",
        "cpu_near2far_terms",
        "cuda_near2far_transform_calls",
        "cuda_near2far_terms",
        "cuda_near2far_submitted_chunks",
        "cuda_near2far_source_points",
        "cuda_near2far_output_points",
        "cuda_near2far_frequencies",
        "cuda_near2far_periodic_copies",
        "cuda_near2far_fast_precision_calls",
        "cuda_near2far_mixed_precision_calls",
        "cuda_near2far_cancellation_retries",
        "cuda_near2far_target_tiles",
        "cuda_near2far_frequency_tiles",
        "cuda_near2far_operation_tiles",
        "cuda_near2far_maximum_workspace_bytes",
        "cuda_near2far_descriptor_uploads",
        "cuda_near2far_kernel_launches",
        "cuda_near2far_result_device_to_host_bytes",
        "cuda_near2far_condition_device_to_host_bytes",
        "near2far_dft_device_to_host_bytes_avoided",
        "near2far_mpi_allreduce_calls",
        "near2far_mpi_allreduce_bytes",
        "cpu_near2far_adjoint_calls",
        "cpu_near2far_adjoint_terms",
        "cuda_near2far_adjoint_calls",
        "cuda_near2far_adjoint_terms",
        "cuda_near2far_adjoint_submitted_chunks",
        "cuda_near2far_adjoint_source_points",
        "cuda_near2far_adjoint_far_points",
        "cuda_near2far_adjoint_frequencies",
        "cuda_near2far_adjoint_periodic_copies",
        "cuda_near2far_adjoint_fast_precision_calls",
        "cuda_near2far_adjoint_mixed_precision_calls",
        "cuda_near2far_adjoint_cancellation_retries",
        "cuda_near2far_adjoint_maximum_workspace_bytes",
        "cuda_near2far_adjoint_descriptor_uploads",
        "cuda_near2far_adjoint_kernel_launches",
        "cuda_near2far_adjoint_host_to_device_bytes",
        "cuda_near2far_adjoint_result_device_to_host_bytes",
        "cuda_near2far_adjoint_condition_device_to_host_bytes",
    ),
    "multi_gpu": (
        "mpi_messages",
        "mpi_scalars",
        "cuda_aware_bytes",
        "pinned_staging_bytes",
        "pinned_device_to_host_bytes",
        "pinned_host_to_device_bytes",
    ),
    "mpi_completion": (
        "mpi_waitsome_executions",
        "mpi_waitall_executions",
    ),
    "boundary_eh_overlap": (
        "boundary_eh_overlap_checks",
        "boundary_eh_overlap_eligible",
        "boundary_eh_overlap_launched_h",
        "boundary_eh_overlap_launched_e",
        "boundary_eh_overlap_skipped_disabled",
        "boundary_eh_overlap_skipped_unsupported_schedule",
        "boundary_eh_overlap_skipped_no_remote",
        "boundary_eh_overlap_skipped_cold_topology",
        "boundary_eh_overlap_rejected",
    ),
    "halo_curl_overlap": (
        "halo_curl_overlap_checks",
        "halo_curl_overlap_eligible",
        "halo_curl_overlap_launches",
        "halo_curl_overlap_skipped_disabled",
        "halo_curl_overlap_skipped_unsupported_schedule",
        "halo_curl_overlap_skipped_no_remote",
        "halo_curl_overlap_skipped_cold_topology",
        "halo_curl_overlap_rejected_feature",
        "halo_curl_overlap_rejected_small",
        "halo_curl_overlap_full_points",
        "halo_curl_overlap_interior_points",
        "halo_curl_overlap_shell_points",
    ),
    "tile_coalescing": (
        "tile_coalesced_curl_chunk_phases",
        "tile_coalesced_curl_input_tiles",
        "tile_coalesced_update_eh_chunk_phases",
        "tile_coalesced_update_eh_input_tiles",
    ),
    "phase_batch_policy": (
        "phase_curl_automatic_checks",
        "phase_curl_automatic_selected",
        "phase_curl_automatic_rejected",
        "phase_curl_forced_batches",
        "phase_curl_batched_operations",
        "phase_curl_unbatched_operations",
        "phase_curl_replay_checks",
        "phase_curl_replay_hits",
        "phase_curl_replay_unready",
        "phase_curl_replay_generation_misses",
        "phase_curl_replay_mirror_misses",
        "phase_update_eh_automatic_checks",
        "phase_update_eh_automatic_selected",
        "phase_update_eh_automatic_rejected",
        "phase_update_eh_forced_batches",
        "phase_update_eh_batched_operations",
        "phase_update_eh_unbatched_operations",
    ),
}


def _install_meep_native_seal():
    """Seal native Meep callables before validation code can replace them.

    The hook runs immediately after the real package loader completes, before
    the import statement returns to an example.  State lives in this closure
    and is exposed only through an immutable snapshot returned to the atexit
    evidence collector.
    """

    state = {"seal": None, "error": None}

    def seal(module):
        if state["seal"] is not None:
            if state["seal"]["module"] is not module:
                raise RuntimeError("two different meep modules were imported")
            return
        extension = getattr(module, "_meep", None)
        gpu = getattr(module, "gpu", None)
        if extension is None or gpu is None:
            raise RuntimeError("meep import did not expose _meep and gpu")
        functions = []
        for name in _GPU_NATIVE_FUNCTIONS:
            value = getattr(extension, name, None)
            if not callable(value):
                raise RuntimeError(
                    f"native Meep extension is missing callable {name}"
                )
            functions.append((name, value))
        state["seal"] = {
            "module": module,
            "extension": extension,
            "gpu": gpu,
            "functions": tuple(functions),
        }

    class SealingLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            creator = getattr(self._wrapped, "create_module", None)
            return creator(spec) if creator is not None else None

        def exec_module(self, module):
            self._wrapped.exec_module(module)
            try:
                seal(module)
            except BaseException:
                state["error"] = traceback.format_exc()
                raise

    class SealingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "meep":
                return None
            spec = importlib.machinery.PathFinder.find_spec(
                fullname, path, target
            )
            if spec is None or spec.loader is None:
                return spec
            spec.loader = SealingLoader(spec.loader)
            return spec

    finder = SealingFinder()
    sys.meta_path.insert(0, finder)

    def snapshot():
        if state["error"] is not None:
            raise RuntimeError(
                "native Meep API sealing failed:\n" + state["error"]
            )
        sealed = state["seal"]
        if sealed is None:
            return None
        module = sealed["module"]
        extension = sealed["extension"]
        if sys.modules.get("meep") is not module:
            raise RuntimeError("sealed meep module identity changed")
        if getattr(module, "_meep", None) is not extension:
            raise RuntimeError("sealed native Meep extension identity changed")
        if getattr(module, "gpu", None) is not sealed["gpu"]:
            raise RuntimeError("sealed meep.gpu controller identity changed")
        functions = dict(sealed["functions"])
        for name, value in sealed["functions"]:
            if getattr(extension, name, None) is not value:
                raise RuntimeError(
                    f"sealed native Meep callable identity changed: {name}"
                )
        return {
            "module": module,
            "extension": extension,
            "gpu": sealed["gpu"],
            "functions": functions,
        }

    return snapshot


_NATIVE_MEEP_SNAPSHOT = _install_meep_native_seal()


def register_gpmeep_benchmark_evidence(invocation_sha256, result_sha256):
    """Bind benchmark-produced digests in process memory before atexit.

    The result file cannot replace these values by rewriting its own hash.
    Registration is single-use so a second writer cannot supersede the
    benchmark payload later in the same interpreter.
    """

    global _REGISTERED_INVOCATION_SHA256, _REGISTERED_RESULT_SHA256
    for name, value in (
        ("invocation", invocation_sha256),
        ("result", result_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError(f"invalid registered {name} SHA-256")
    if (
        _REGISTERED_INVOCATION_SHA256 is not None
        or _REGISTERED_RESULT_SHA256 is not None
    ):
        raise RuntimeError("gpmeep benchmark evidence was registered twice")
    _REGISTERED_INVOCATION_SHA256 = invocation_sha256
    _REGISTERED_RESULT_SHA256 = result_sha256


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    resolved = pathlib.Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _file_record_from_bytes(path, payload):
    resolved = pathlib.Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _mapped_library(name_prefix):
    candidates = set()
    for line in pathlib.Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            candidate = pathlib.Path(columns[5])
            if candidate.name.startswith(name_prefix):
                candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one mapped {name_prefix} library, found {len(candidates)}"
        )
    return next(iter(candidates))


def _mapped_file_identity(path):
    resolved = pathlib.Path(path).resolve()
    identities = set()
    deleted_mapping = False
    for line in pathlib.Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        raw_path = columns[5]
        if raw_path.endswith(" (deleted)"):
            raw_path = raw_path[: -len(" (deleted)")]
            deleted = True
        else:
            deleted = False
        if pathlib.Path(raw_path).resolve() != resolved:
            continue
        deleted_mapping = deleted_mapping or deleted
        major_text, minor_text = columns[3].split(":", 1)
        identities.add(
            (int(major_text, 16), int(minor_text, 16), int(columns[4]))
        )
    if deleted_mapping or len(identities) != 1:
        raise RuntimeError(
            f"loaded mapping identity is ambiguous or deleted: {resolved}"
        )
    stat = resolved.stat()
    current = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    if identities != {current}:
        raise RuntimeError(
            f"loaded mapping inode differs from current pathname: {resolved}"
        )
    return {
        "device_major": current[0],
        "device_minor": current[1],
        "inode": current[2],
    }


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _native_gpu_statistics(functions):
    raw = functions["_gpu_statistics"]()
    return {
        group: {name: int(getattr(raw, name)) for name in names}
        for group, names in _GPU_STATISTIC_GROUPS.items()
    }


def _native_gpu_devices(functions):
    devices = []
    for index in range(int(functions["_gpu_device_count"]())):
        devices.append(
            {
                "ordinal": int(
                    functions["_gpu_device_integer_property"](index, 0)
                ),
                "compute_capability": (
                    int(functions["_gpu_device_integer_property"](index, 1)),
                    int(functions["_gpu_device_integer_property"](index, 2)),
                ),
                "multiprocessor_count": int(
                    functions["_gpu_device_integer_property"](index, 3)
                ),
                "max_threads_per_block": int(
                    functions["_gpu_device_integer_property"](index, 4)
                ),
                "compatible": bool(
                    functions["_gpu_device_integer_property"](index, 5)
                ),
                "global_memory_bytes": int(
                    functions["_gpu_device_memory"](index)
                ),
                "memory_bandwidth_bytes_per_second": int(
                    functions["_gpu_device_memory_bandwidth"](index)
                ),
                "identifier": str(
                    functions["_gpu_device_identifier"](index)
                ),
                "name": str(functions["_gpu_device_name"](index)),
            }
        )
    return devices


def _capture_value(result, key, function) -> None:
    try:
        result[key] = function()
    except BaseException:
        result.setdefault("capture_errors", {})[key] = traceback.format_exc()


def _write_json_atomically(target, payload) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(
            _jsonable(payload),
            temporary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        temporary.write("\n")
        temporary_path = pathlib.Path(temporary.name)
    temporary_path.replace(target)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_invocation_hash(contract) -> bool:
    if not isinstance(contract, dict):
        return False
    unsigned_contract = dict(contract)
    recorded = unsigned_contract.pop("invocation_sha256", None)
    return (
        isinstance(recorded, str)
        and len(recorded) == 64
        and all(character in "0123456789abcdef" for character in recorded)
        and _canonical_sha256(unsigned_contract) == recorded
    )


def _expected_moving_source_workload(resolution):
    if (
        not isinstance(resolution, int)
        or isinstance(resolution, bool)
        or resolution <= 0
    ):
        return None
    sx = Fraction("40.0")
    sy = Fraction("40.0")
    speed = Fraction("0.7")
    courant = Fraction(1, 2)
    requested = sx / speed
    timestep = courant / resolution
    ratio = requested / timestep
    steps = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return {
        "sx": float(sx),
        "sy": float(sy),
        "dpml": 1.0,
        "source_speed": float(speed),
        "source_frequency": 1e-10,
        "courant": float(courant),
        "requested_until": float(requested),
        "expected_timesteps": steps,
        "expected_meep_time": float(steps * timestep),
        "expected_source_updates": steps + 1,
        "expected_field_shape": [int(sx * resolution), int(sy * resolution)],
    }


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_benchmark_shape(
    benchmark, expected_requested_backend, expected_backend
) -> bool:
    if not isinstance(benchmark, dict):
        return False
    repeat = benchmark.get("repeat")
    decisions = benchmark.get("backend_decisions")
    durations = benchmark.get("solver_seconds")
    metrics = benchmark.get("metrics")
    workload = _expected_moving_source_workload(benchmark.get("resolution"))
    if not (
        benchmark.get("schema_version") == 2
        and isinstance(repeat, int)
        and repeat > 0
        and isinstance(durations, list)
        and len(durations) == repeat
        and all(
            _finite_number(value) and value > 0
            for value in durations
        )
        and isinstance(metrics, list)
        and len(metrics) == repeat
        and isinstance(decisions, list)
        and len(decisions) == repeat
        and workload is not None
        and benchmark.get("workload_contract") == workload
        and _finite_number(benchmark.get("solver_seconds_median"))
        and math.isclose(
            float(benchmark["solver_seconds_median"]),
            float(sorted(durations)[repeat // 2]
                  if repeat % 2
                  else 0.5 * (sorted(durations)[repeat // 2 - 1] +
                              sorted(durations)[repeat // 2])),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and benchmark.get("parallelism_contract")
        == (
            "serial-single-rank-diagnostic"
            if expected_backend == "cpu"
            else "single-rank-gpu"
        )
        and (
            benchmark.get("performance_claim_eligible") is False
            if expected_backend == "cpu"
            else benchmark.get("performance_claim_eligible") is True
        )
    ):
        return False
    reference_metric = None
    for metric in metrics:
        if not isinstance(metric, dict):
            return False
        if (
            metric.get("shape") != workload["expected_field_shape"]
            or metric.get("source_updates")
            != workload["expected_source_updates"]
            or not _finite_number(metric.get("meep_time"))
            or not math.isclose(
                float(metric["meep_time"]),
                workload["expected_meep_time"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not all(
                _finite_number(metric.get(name))
                for name in ("sum", "l1", "l2", "maximum_absolute")
            )
            or any(
                metric[name] < 0
                for name in ("l1", "l2", "maximum_absolute")
            )
        ):
            return False
        if reference_metric is None:
            reference_metric = metric
        elif metric != reference_metric:
            return False
    signatures = set()
    for decision in decisions:
        expected_marker = (
            "selected CPU" if expected_backend == "cpu" else "CUDA"
        )
        if not (
            isinstance(decision, dict)
            and decision.get("requested_backend")
            == expected_requested_backend
            and decision.get("active_backend") == expected_backend
            and decision.get("process_active_backend") == expected_backend
            and isinstance(decision.get("backend_diagnostic"), str)
            and expected_marker in decision.get("backend_diagnostic")
            and decision.get("process_backend_diagnostic")
            == decision.get("backend_diagnostic")
        ):
            return False
        signatures.add(
            (
                decision.get("requested_backend"),
                decision.get("active_backend"),
                decision.get("backend_diagnostic"),
                json.dumps(decision.get("selected_device"), sort_keys=True),
            )
        )
    return len(signatures) == 1


def _valid_selected_backend_statistics(
    statistics, expected_requested_backend, expected_backend
) -> bool:
    if not isinstance(statistics, dict):
        return False
    required_groups = (
        "runtime",
        "dispatch",
        "resident",
        "field_updates",
        "polarizations",
        "sources",
        "boundaries",
        "dfts",
        "dft_reductions",
        "dft_materializations",
        "eigenmode_overlaps",
        "ldos",
        "near2far",
    )
    if any(not isinstance(statistics.get(group), dict) for group in required_groups):
        return False
    dispatch = statistics["dispatch"]
    fields = statistics["field_updates"]
    polarizations = statistics["polarizations"]
    sources = statistics["sources"]
    boundaries = statistics["boundaries"]
    dfts = statistics["dfts"]
    dft_reductions = statistics["dft_reductions"]
    dft_materializations = statistics["dft_materializations"]
    eigenmode_overlaps = statistics["eigenmode_overlaps"]
    ldos = statistics["ldos"]
    near2far = statistics["near2far"]
    if expected_backend == "cuda":
        return (
            dispatch.get("cuda_curl_calls", 0) > 0
            and fields.get("cuda_update_eh_calls", 0) > 0
            and sources.get("cuda_source_calls", 0) > 0
            and boundaries.get("cuda_boundary_calls", 0) > 0
            and dispatch.get("cpu_curl_calls") == 0
            and fields.get("cpu_update_eh_calls") == 0
            and polarizations.get("cpu_polarization_calls") == 0
            and sources.get("cpu_source_calls") == 0
            and boundaries.get("cpu_boundary_calls") == 0
            and dfts.get("cpu_dft_calls") == 0
            and dft_reductions.get("cpu_dft_reduction_calls") == 0
            and dft_materializations.get(
                "cpu_dft_array_materialization_calls"
            )
            == 0
            and dft_materializations.get("cpu_dft_output_calls") == 0
            and eigenmode_overlaps.get("cpu_eigenmode_overlap_calls") == 0
            and ldos.get("cpu_ldos_reduction_calls") == 0
            and near2far.get("cpu_near2far_adjoint_calls") == 0
        )
    if expected_backend != "cpu":
        return False
    cpu_executed = (
        dispatch.get("cpu_curl_calls", 0) > 0
        and fields.get("cpu_update_eh_calls", 0) > 0
        and sources.get("cpu_source_calls", 0) > 0
        and boundaries.get("cpu_boundary_calls", 0) > 0
    )
    cuda_unused = (
        dispatch.get("cuda_curl_calls") == 0
        and fields.get("cuda_update_eh_calls") == 0
        and polarizations.get("cuda_polarization_calls") == 0
        and sources.get("cuda_source_calls") == 0
        and boundaries.get("cuda_boundary_calls") == 0
        and dfts.get("cuda_dft_calls") == 0
        and dft_reductions.get("cuda_dft_reduction_calls") == 0
        and dft_materializations.get(
            "cuda_dft_array_materialization_calls"
        )
        == 0
        and dft_materializations.get("cuda_dft_output_calls") == 0
        and eigenmode_overlaps.get("cuda_eigenmode_overlap_calls") == 0
        and ldos.get("cuda_ldos_reduction_calls") == 0
        and near2far.get("cuda_near2far_transform_calls") == 0
        and near2far.get("cuda_near2far_adjoint_calls") == 0
    )
    if expected_requested_backend != "auto":
        return cpu_executed and cuda_unused
    return (
        cpu_executed
        and cuda_unused
        and dispatch.get("host_to_device_bytes") == 0
        and dispatch.get("device_to_host_bytes") == 0
        and statistics["resident"].get("device_buffer_allocations") == 0
        and statistics["runtime"]
        == {
            "runtime_availability_probes": 0,
            "runtime_device_enumerations": 0,
            "runtime_device_selections": 0,
        }
    )


def _valid_statistics_by_repeat(
    benchmark, expected_requested_backend, expected_backend
) -> bool:
    repeat = benchmark.get("repeat") if isinstance(benchmark, dict) else None
    snapshots = (
        benchmark.get("statistics_by_repeat")
        if isinstance(benchmark, dict)
        else None
    )
    final = benchmark.get("gpu_statistics") if isinstance(benchmark, dict) else None
    if not (
        isinstance(repeat, int)
        and repeat > 0
        and isinstance(snapshots, list)
        and len(snapshots) == repeat
        and snapshots[-1] == final
    ):
        return False
    backend_prefix = "cuda" if expected_backend == "cuda" else "cpu"
    progress_fields = (
        ("dispatch", f"{backend_prefix}_curl_calls"),
        ("field_updates", f"{backend_prefix}_update_eh_calls"),
        ("sources", f"{backend_prefix}_source_calls"),
        ("boundaries", f"{backend_prefix}_boundary_calls"),
    )
    previous = {field: -1 for field in progress_fields}
    for snapshot in snapshots:
        if not _valid_selected_backend_statistics(
            snapshot, expected_requested_backend, expected_backend
        ):
            return False
        for group, name in progress_fields:
            value = snapshot.get(group, {}).get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= previous[(group, name)]
            ):
                return False
            previous[(group, name)] = value
    return True


def _completion_path_is_fresh(preexisted, completion_path) -> bool:
    return bool(completion_path) and not preexisted and not pathlib.Path(
        completion_path
    ).exists()


def _registered_evidence_matches(invocation_sha256, result_sha256) -> bool:
    return (
        invocation_sha256 == _REGISTERED_INVOCATION_SHA256
        and result_sha256 == _REGISTERED_RESULT_SHA256
    )


def _receipt_artifact_record(receipt, name):
    repository = pathlib.Path(receipt["repo"]).resolve()
    record = receipt["artifacts"][name]
    recorded_path = pathlib.Path(record["path"])
    resolved = (
        recorded_path.resolve()
        if recorded_path.is_absolute()
        else (repository / recorded_path).resolve()
    )
    return {
        "path": str(resolved),
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def _capture(_native_snapshot=_NATIVE_MEEP_SNAPSHOT) -> None:
    destination = os.environ.get("GPMEEP_VALIDATION_STATS_FILE")
    if not destination:
        return
    result = {
        "capture_status": "captured",
        "pid": os.getpid(),
        "elapsed_seconds": time.time() - _STARTED,
        "expected_backend": os.environ.get("GPMEEP_VALIDATION_EXPECTED_BACKEND"),
        "expected_requested_backend": os.environ.get(
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND",
            os.environ.get("GPMEEP_VALIDATION_EXPECTED_BACKEND"),
        ),
        "run_nonce": os.environ.get("GPMEEP_VALIDATION_RUN_NONCE"),
        "build_receipt_id": os.environ.get(
            "GPMEEP_VALIDATION_BUILD_RECEIPT_ID"
        ),
        "build_receipt_file": os.environ.get(
            "GPMEEP_VALIDATION_BUILD_RECEIPT_FILE"
        ),
        "strict_cuda_marker": os.environ.get("GPMEEP_VALIDATION_STRICT_CUDA") == "1",
        "python_executable": _file_record(sys.executable),
        "python_version": sys.version,
        "meep_imported": "meep" in sys.modules,
        "gpu_api_available": False,
    }
    benchmark_payload = None
    try:
        receipt_environment_path = os.environ.get(
            "GPMEEP_VALIDATION_BUILD_RECEIPT_FILE"
        )
        if receipt_environment_path:
            from gpmeep_provenance import StatHashCache, verify_build_receipt

            receipt_environment_path = pathlib.Path(
                receipt_environment_path
            ).resolve()
            receipt_for_repo = json.loads(
                receipt_environment_path.read_text(encoding="utf-8")
            )
            verified_at_exit = verify_build_receipt(
                receipt_environment_path,
                pathlib.Path(receipt_for_repo["repo"]).resolve(),
                verify_source=True,
                hash_cache=StatHashCache(),
            )
            result["receipt_exit_verification"] = {
                "receipt_id": verified_at_exit["receipt_id"],
                "artifact_set_id": verified_at_exit["artifact_set_id"],
                "build_input_id": verified_at_exit["build_input_id"],
            }
        sealed_api = _native_snapshot()
        meep = sys.modules.get("meep")
        if meep is not None:
            if sealed_api is None or sealed_api["module"] is not meep:
                raise RuntimeError(
                    "imported Meep was not bound by the native API seal"
                )
            extension_module = sealed_api["extension"]
            gpu = sealed_api["gpu"]
            native = sealed_api["functions"]
            result.update(
                {
                    "meep_module": _file_record(meep.__file__),
                    "meep_version": getattr(meep, "__version__", None),
                    "single_precision": bool(native["is_single_precision"]()),
                    "with_mpi": bool(native["with_mpi"]()),
                }
            )
            if extension_module is not None and getattr(
                extension_module, "__file__", None
            ):
                result["extension"] = _file_record(extension_module.__file__)
                result["extension_mapping"] = _mapped_file_identity(
                    extension_module.__file__
                )
            libmeep_path = _mapped_library("libmeep.so")
            result["libmeep"] = _file_record(libmeep_path)
            result["libmeep_mapping"] = _mapped_file_identity(libmeep_path)
            result["gpu_api_available"] = True
            backend_names = ("cpu", "auto", "cuda")
            _capture_value(
                result,
                "gpu_compiled",
                lambda: bool(native["_gpu_backend_compiled"]()),
            )
            _capture_value(
                result,
                "compiled_architectures",
                lambda: str(native["_gpu_compiled_architectures"]()),
            )
            _capture_value(
                result,
                "requested_backend",
                lambda: backend_names[int(native["_gpu_requested_backend"]())],
            )
            _capture_value(
                result,
                "active_backend",
                lambda: backend_names[int(native["_gpu_active_backend"]())],
            )
            _capture_value(
                result,
                "backend_diagnostic",
                lambda: str(native["_gpu_backend_diagnostic"]()),
            )
            _capture_value(
                result,
                "selected_device",
                lambda: int(native["_gpu_selected_device"]()),
            )
            _capture_value(
                result,
                "selected_device_identifier",
                lambda: str(native["_gpu_selected_device_identifier"]()),
            )
            _capture_value(
                result,
                "statistics",
                lambda: _native_gpu_statistics(native),
            )
            automatic_cpu = (
                result.get("expected_requested_backend") == "auto"
                and result.get("expected_backend") == "cpu"
            )
            result["runtime_discovery_skipped_for_automatic_cpu"] = (
                automatic_cpu
            )
            if not automatic_cpu:
                _capture_value(
                    result,
                    "runtime_available",
                    lambda: bool(native["_gpu_runtime_available"]()),
                )
                _capture_value(
                    result,
                    "runtime_diagnostic",
                    lambda: str(native["_gpu_runtime_diagnostic"]()),
                )
                _capture_value(
                    result, "devices", lambda: _native_gpu_devices(native)
                )
        benchmark_result = os.environ.get("GPMEEP_VALIDATION_RESULT_FILE")
        if benchmark_result:
            benchmark_payload = pathlib.Path(benchmark_result).read_bytes()
            result["benchmark_result"] = _file_record_from_bytes(
                benchmark_result, benchmark_payload
            )
    except BaseException:
        result["capture_status"] = "capture_error"
        result["capture_error"] = traceback.format_exc()

    target = pathlib.Path(destination)
    try:
        _write_json_atomically(target, result)

        completion_path = os.environ.get("GPMEEP_VALIDATION_COMPLETE_FILE")
        benchmark_path = os.environ.get("GPMEEP_VALIDATION_RESULT_FILE")
        if completion_path and benchmark_path:
            if benchmark_payload is None:
                raise RuntimeError("benchmark result bytes were not captured")
            benchmark = json.loads(benchmark_payload)
            contract = benchmark["runtime_contract"]
            recorded_invocation = contract.get("invocation_sha256")
            contract_environment = contract["environment"]
            expected_backend = os.environ.get(
                "GPMEEP_VALIDATION_EXPECTED_BACKEND"
            )
            expected_requested_backend = os.environ.get(
                "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND", expected_backend
            )
            expected_nonce = os.environ.get("GPMEEP_VALIDATION_RUN_NONCE")
            expected_receipt = os.environ.get(
                "GPMEEP_VALIDATION_BUILD_RECEIPT_ID"
            )
            receipt_path = pathlib.Path(
                os.environ["GPMEEP_VALIDATION_BUILD_RECEIPT_FILE"]
            ).resolve()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            unsigned_receipt = dict(receipt)
            recorded_receipt_id = unsigned_receipt.pop("receipt_id", None)
            receipt_identity = contract.get("verified_build_receipt", {})
            ending_receipt_identity = contract.get(
                "verified_build_receipt_after_runs", {}
            )
            run_statistics = benchmark.get("gpu_statistics", {})
            runtime_statistics = (
                run_statistics.get("runtime", {})
                if isinstance(run_statistics, dict)
                else {}
            )
            automatic_cpu_runtime_untouched = True
            if (
                expected_requested_backend == "auto"
                and expected_backend == "cpu"
            ):
                automatic_cpu_runtime_untouched = runtime_statistics == {
                    "runtime_availability_probes": 0,
                    "runtime_device_enumerations": 0,
                    "runtime_device_selections": 0,
                }
            from gpmeep_provenance import StatHashCache, verify_build_receipt

            verified_before_seal = verify_build_receipt(
                receipt_path,
                pathlib.Path(receipt["repo"]).resolve(),
                verify_source=True,
                hash_cache=StatHashCache(),
            )
            preseal_receipt_identity = {
                "receipt_id": verified_before_seal["receipt_id"],
                "artifact_set_id": verified_before_seal["artifact_set_id"],
                "build_input_id": verified_before_seal["build_input_id"],
            }
            checks = {
                "capture": result.get("capture_status") == "captured"
                and not result.get("capture_errors")
                and _completion_path_is_fresh(
                    _COMPLETION_PREEXISTED, completion_path
                ),
                "invocation_hash": _valid_invocation_hash(contract),
                "in_process_invocation_binding": _registered_evidence_matches(
                    recorded_invocation,
                    result.get("benchmark_result", {}).get("sha256"),
                ),
                "in_process_result_binding": _registered_evidence_matches(
                    recorded_invocation,
                    result.get("benchmark_result", {}).get("sha256"),
                ),
                "result_schema": _valid_benchmark_shape(
                    benchmark, expected_requested_backend, expected_backend
                ),
                "selected_backend_statistics": (
                    _valid_selected_backend_statistics(
                        run_statistics,
                        expected_requested_backend,
                        expected_backend,
                    )
                ),
                "all_repeat_backend_statistics": (
                    _valid_statistics_by_repeat(
                        benchmark,
                        expected_requested_backend,
                        expected_backend,
                    )
                ),
                "native_final_statistics_binding": (
                    result.get("statistics") == run_statistics
                ),
                "automatic_cpu_cuda_runtime_untouched": (
                    automatic_cpu_runtime_untouched
                    and (
                        expected_requested_backend != "auto"
                        or expected_backend != "cpu"
                        or (
                            result.get(
                                "runtime_discovery_skipped_for_automatic_cpu"
                            )
                            is True
                            and result.get("statistics", {}).get("runtime")
                            == {
                                "runtime_availability_probes": 0,
                                "runtime_device_enumerations": 0,
                                "runtime_device_selections": 0,
                            }
                        )
                    )
                ),
                "backend": benchmark.get("backend") == expected_backend
                and result.get("active_backend") == expected_backend,
                "requested_backend": benchmark.get("requested_backend", expected_backend)
                == expected_requested_backend
                and result.get("requested_backend") == expected_requested_backend
                and contract.get("requested_backend") == expected_requested_backend
                and contract_environment.get(
                    "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND",
                    contract_environment.get("GPMEEP_VALIDATION_EXPECTED_BACKEND"),
                )
                == expected_requested_backend,
                "nonce": result.get("run_nonce") == expected_nonce
                and contract_environment.get("GPMEEP_VALIDATION_RUN_NONCE")
                == expected_nonce,
                "receipt": result.get("build_receipt_id") == expected_receipt
                and contract_environment.get(
                    "GPMEEP_VALIDATION_BUILD_RECEIPT_ID"
                )
                == expected_receipt
                and recorded_receipt_id == expected_receipt
                and _canonical_sha256(unsigned_receipt) == expected_receipt
                and receipt_identity.get("receipt_id") == expected_receipt
                and receipt_identity.get("artifact_set_id")
                == receipt.get("artifact_set_id")
                and receipt_identity.get("build_input_id")
                == receipt.get("build_input_id")
                and ending_receipt_identity == receipt_identity
                and result.get("receipt_exit_verification")
                == receipt_identity
                and preseal_receipt_identity == receipt_identity
                and contract_environment.get(
                    "GPMEEP_VALIDATION_BUILD_RECEIPT_FILE"
                )
                == str(receipt_path)
                and contract.get("build_receipt") == _file_record(receipt_path),
                "receipt_runtime_artifacts": result.get("extension")
                == _receipt_artifact_record(receipt, "python_extension")
                and result.get("libmeep")
                == _receipt_artifact_record(receipt, "libmeep")
                and contract.get("python_extension")
                == _receipt_artifact_record(receipt, "python_extension")
                and contract.get("libmeep")
                == _receipt_artifact_record(receipt, "libmeep"),
                "runtime_mapping_identity": result.get("extension_mapping")
                == contract.get("python_extension_mapping")
                and result.get("libmeep_mapping")
                == contract.get("libmeep_mapping"),
                "result_hash": result.get("benchmark_result")
                == _file_record(benchmark_path),
                "extension": result.get("extension")
                == contract.get("python_extension"),
                "libmeep": result.get("libmeep") == contract.get("libmeep"),
            }
            failed = sorted(name for name, passed in checks.items() if not passed)
            if failed:
                raise RuntimeError(
                    "benchmark completion contract failed: " + ", ".join(failed)
                )
            completion = {
                "schema_version": 2,
                "status": "complete",
                "created_at_unix_seconds": time.time(),
                "backend": expected_backend,
                "requested_backend": expected_requested_backend,
                "run_nonce": expected_nonce,
                "build_receipt_id": expected_receipt,
                "runtime_invocation_sha256": contract.get("invocation_sha256"),
                "checks": checks,
                "result": result.get("benchmark_result"),
                "statistics": _file_record(target),
                "artifacts": {
                    "python_extension": contract.get("python_extension"),
                    "python_extension_mapping": contract.get(
                        "python_extension_mapping"
                    ),
                    "libmeep": contract.get("libmeep"),
                    "libmeep_mapping": contract.get("libmeep_mapping"),
                },
            }
            completion_target = pathlib.Path(completion_path)
            if completion_target.exists():
                raise RuntimeError(
                    "benchmark completion path existed before atomic seal"
                )
            _write_json_atomically(completion_target, completion)
    except BaseException:
        # Never mask a test result because evidence persistence failed.  The
        # parent runner treats the missing file as an explicit contract failure.
        completion_path = os.environ.get("GPMEEP_VALIDATION_COMPLETE_FILE")
        if completion_path:
            try:
                pathlib.Path(completion_path).unlink(missing_ok=True)
            except BaseException:
                pass
        result["completion_status"] = "failed"
        result["completion_error"] = traceback.format_exc()
        try:
            _write_json_atomically(target, result)
        except BaseException:
            pass


atexit.register(_capture)
