"""Evidence helpers shared by gpmeep benchmark producers and gates.

This module intentionally depends only on the Python standard library.  It is
therefore usable by the isolated benchmark interpreter and by GPU-free unit
tests which exercise the release gates with synthetic records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import struct
import tempfile
from typing import Any, Iterable


PHASE_COUNTERS = {
    "curl": ("dispatch", "curl"),
    "update_eh": ("field_updates", "update_eh"),
    "polarization": ("polarizations", "polarization"),
    "source": ("sources", "source"),
    "boundary": ("boundaries", "boundary"),
    "dft": ("dfts", "dft"),
}
STEPPING_PHASES = ("forward", "adjoint")
REQUIRED_STEPPING_OPERATIONS = (
    "curl",
    "update_eh",
    "source",
    "boundary",
    "dft",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_float64_bytes(values: Iterable[float]) -> bytes:
    materialized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in materialized):
        raise ValueError("floating-point evidence contains NaN or infinity")
    return struct.pack(f"<{len(materialized)}d", *materialized)


def canonical_float64_sha256(values: Iterable[float]) -> str:
    return hashlib.sha256(canonical_float64_bytes(values)).hexdigest()


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Commit a complete text file with a same-directory atomic rename."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = pathlib.Path(temporary.name)
        temporary_path.replace(path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(
        pathlib.Path(path),
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def statistics_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, int]]:
    """Subtract integer GPU statistics while preserving their group layout."""

    result: dict[str, dict[str, int]] = {}
    for group_name, after_group in after.items():
        if not isinstance(after_group, dict):
            continue
        before_group = before.get(group_name, {})
        if not isinstance(before_group, dict):
            before_group = {}
        group_delta: dict[str, int] = {}
        for counter_name, after_value in after_group.items():
            if isinstance(after_value, bool) or not isinstance(after_value, int):
                continue
            before_value = before_group.get(counter_name, 0)
            if isinstance(before_value, bool) or not isinstance(before_value, int):
                raise ValueError(
                    f"statistics counter changed type: {group_name}.{counter_name}"
                )
            group_delta[counter_name] = after_value - before_value
        result[group_name] = group_delta
    return result


def phase_counter_view(
    statistics: dict[str, Any], backend: str
) -> dict[str, dict[str, int]]:
    """Return normalized logical call/point counters for one backend."""

    if backend not in ("cpu", "cuda"):
        raise ValueError(f"unsupported backend {backend!r}")
    result: dict[str, dict[str, int]] = {}
    for operation, (group_name, counter_stem) in PHASE_COUNTERS.items():
        group = statistics.get(group_name, {})
        if not isinstance(group, dict):
            raise ValueError(f"statistics group {group_name!r} is absent")
        calls_name = f"{backend}_{counter_stem}_calls"
        points_name = f"{backend}_{counter_stem}_points"
        calls = group.get(calls_name)
        points = group.get(points_name)
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or isinstance(points, bool)
            or not isinstance(points, int)
        ):
            raise ValueError(
                f"statistics counters {calls_name!r}/{points_name!r} are absent"
            )
        result[operation] = {"calls": calls, "points": points}
    return result


def loaded_library_path(name_prefix: str) -> pathlib.Path:
    """Resolve one actually mapped shared object from Linux /proc evidence."""

    maps = pathlib.Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("/proc/self/maps is required for loaded-library evidence")
    candidates: set[pathlib.Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        candidate = pathlib.Path(columns[5])
        if candidate.name.startswith(name_prefix):
            candidates.add(candidate.resolve())
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            f"expected one loaded {name_prefix} shared object, found "
            f"{len(candidates)}: {rendered}"
        )
    return next(iter(candidates))


def loaded_library_records(
    groups: dict[str, tuple[str, ...]],
) -> dict[str, list[dict[str, Any]]]:
    """Hash the actual mapped shared libraries for required runtime groups."""

    maps = pathlib.Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("/proc/self/maps is required for loaded-library evidence")
    mapped: set[pathlib.Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        candidate = pathlib.Path(columns[5]).resolve()
        if candidate.is_file():
            mapped.add(candidate)
    result: dict[str, list[dict[str, Any]]] = {}
    for group, prefixes in groups.items():
        matches = sorted(
            path for path in mapped if any(path.name.startswith(prefix) for prefix in prefixes)
        )
        if not matches:
            raise RuntimeError(f"no mapped shared library found for {group}")
        result[group] = [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in matches
        ]
    return result


def loaded_mapped_file_records(root: pathlib.Path) -> list[dict[str, Any]]:
    """Hash every regular file currently mapped from ``root``.

    This is deliberately broader than a basename allowlist: MPI, PMIx, PRRTE,
    UCX, libfabric, and their MCA/provider plugins can be loaded lazily.  The
    qualification parent binds these records to the complete installed
    environment manifest in the build receipt.
    """

    maps = pathlib.Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("/proc/self/maps is required for mapped-file evidence")
    resolved_root = root.resolve()
    mapped: set[pathlib.Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        raw_path = columns[5]
        deleted = raw_path.endswith(" (deleted)")
        if deleted:
            raw_path = raw_path[: -len(" (deleted)")]
        candidate = pathlib.Path(raw_path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if deleted or not candidate.is_file():
            raise RuntimeError(
                f"mapped environment file is deleted or missing: {candidate}"
            )
        mapped.add(candidate)
    if not mapped:
        raise RuntimeError(f"no mapped regular files found below {resolved_root}")
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(mapped)
    ]


def validate_gradient_record(result: dict[str, Any]) -> list[float]:
    payload = result.get("result")
    if not isinstance(payload, dict):
        raise ValueError("benchmark result payload is absent")
    gradient = payload.get("gradient")
    shape = payload.get("gradient_shape")
    count = payload.get("gradient_count")
    if not isinstance(gradient, list):
        raise ValueError("full gradient vector is absent")
    if not isinstance(shape, list) or not shape or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in shape
    ):
        raise ValueError("gradient shape must contain positive integers")
    expected_count = math.prod(shape)
    if count != expected_count or len(gradient) != expected_count:
        raise ValueError(
            "gradient vector/count/shape disagree: "
            f"len={len(gradient)} count={count!r} shape={shape!r}"
        )
    values = [float(value) for value in gradient]
    digest = canonical_float64_sha256(values)
    if payload.get("gradient_sha256") != digest:
        raise ValueError("gradient canonical SHA-256 does not match its vector")
    return values


def compare_gradient_vectors(
    cpu: dict[str, Any],
    cuda: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    expected_count: int,
) -> dict[str, Any]:
    """Compare every gradient element and an independent L2 error bound."""

    if atol < 0 or rtol < 0:
        raise ValueError("gradient tolerances must be nonnegative")
    cpu_values = validate_gradient_record(cpu)
    cuda_values = validate_gradient_record(cuda)
    cpu_shape = cpu["result"]["gradient_shape"]
    cuda_shape = cuda["result"]["gradient_shape"]
    structural_pass = (
        cpu_shape == cuda_shape
        and len(cpu_values) == len(cuda_values) == expected_count
    )
    if len(cpu_values) != len(cuda_values):
        return {
            "pass": False,
            "structural_pass": False,
            "expected_count": expected_count,
            "cpu_count": len(cpu_values),
            "cuda_count": len(cuda_values),
            "cpu_shape": cpu_shape,
            "cuda_shape": cuda_shape,
            "reason": "CPU/CUDA gradient lengths differ",
        }

    errors = [abs(left - right) for left, right in zip(cpu_values, cuda_values)]
    allowed = [
        atol + rtol * max(abs(left), abs(right))
        for left, right in zip(cpu_values, cuda_values)
    ]
    scaled = [
        error / limit if limit > 0 else (0.0 if error == 0 else math.inf)
        for error, limit in zip(errors, allowed)
    ]
    max_absolute_index = max(range(len(errors)), key=errors.__getitem__)
    max_scaled_index = max(range(len(scaled)), key=scaled.__getitem__)
    l2_error = math.sqrt(sum(error * error for error in errors))
    cpu_l2 = math.sqrt(sum(value * value for value in cpu_values))
    cuda_l2 = math.sqrt(sum(value * value for value in cuda_values))
    l2_allowed = atol * math.sqrt(len(errors)) + rtol * max(cpu_l2, cuda_l2)
    component_pass = all(error <= limit for error, limit in zip(errors, allowed))
    l2_pass = l2_error <= l2_allowed
    return {
        "pass": structural_pass and component_pass and l2_pass,
        "structural_pass": structural_pass,
        "component_pass": component_pass,
        "l2_pass": l2_pass,
        "compared_count": len(errors),
        "expected_count": expected_count,
        "cpu_shape": cpu_shape,
        "cuda_shape": cuda_shape,
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": errors[max_absolute_index],
        "max_absolute_error_index": max_absolute_index,
        "max_absolute_allowed": allowed[max_absolute_index],
        "max_scaled_error": scaled[max_scaled_index],
        "max_scaled_error_index": max_scaled_index,
        "l2_error": l2_error,
        "l2_allowed": l2_allowed,
        "relative_l2_error": l2_error / max(cpu_l2, cuda_l2, 1e-300),
        "cpu_gradient_sha256": cpu["result"]["gradient_sha256"],
        "cuda_gradient_sha256": cuda["result"]["gradient_sha256"],
    }


def validate_backend_and_paths(
    result: dict[str, Any],
    *,
    expected_backend: str,
    expected_python: pathlib.Path,
    expected_python_sha256: str,
    expected_meep_module: pathlib.Path,
    expected_meep_module_sha256: str,
    expected_extension: pathlib.Path,
    expected_extension_sha256: str,
    expected_libmeep: pathlib.Path,
    expected_libmeep_sha256: str,
) -> dict[str, Any]:
    """Enforce backend exclusivity and the exact loaded runtime closure."""

    problems: list[str] = []
    backend = result.get("backend", {})
    runtime = result.get("runtime", {})
    if not isinstance(backend, dict) or not isinstance(runtime, dict):
        raise ValueError("backend/runtime evidence is absent")

    if backend.get("requested") != expected_backend:
        problems.append("requested backend mismatch")
    if backend.get("active") != expected_backend:
        problems.append("active backend mismatch")
    if backend.get("compiled") is not True:
        problems.append("CUDA-capable build marker is false")
    if backend.get("single_precision") is not True:
        problems.append("single-precision marker is false")
    expected_strict = expected_backend == "cuda"
    if backend.get("strict_cuda_marker") is not expected_strict:
        problems.append("strict-CUDA marker mismatch")
    if expected_backend == "cuda" and backend.get("runtime_available") is not True:
        problems.append("CUDA runtime is unavailable")

    path_expectations = (
        ("python_executable", expected_python, expected_python_sha256),
        ("meep_module", expected_meep_module, expected_meep_module_sha256),
        ("extension", expected_extension, expected_extension_sha256),
        ("libmeep", expected_libmeep, expected_libmeep_sha256),
    )
    for key, expected_path, expected_hash in path_expectations:
        value = runtime.get(key)
        if isinstance(value, dict):
            actual_path = (
                pathlib.Path(str(value.get("path"))).resolve()
                if value.get("path")
                else None
            )
            actual_hash = value.get("sha256")
        else:
            actual_path = None
            actual_hash = None
        if actual_path != pathlib.Path(expected_path).resolve():
            problems.append(f"loaded {key} path mismatch")
        if actual_hash != expected_hash:
            problems.append(f"loaded {key} SHA-256 mismatch")

    phases = result.get("workload", {}).get("phases", {})
    if not isinstance(phases, dict):
        problems.append("phase evidence is absent")
    else:
        inactive_backend = "cpu" if expected_backend == "cuda" else "cuda"
        for phase_name in STEPPING_PHASES:
            phase = phases.get(phase_name, {})
            counters = phase.get("backend_counters", {}) if isinstance(phase, dict) else {}
            active = counters.get(expected_backend, {}) if isinstance(counters, dict) else {}
            inactive = counters.get(inactive_backend, {}) if isinstance(counters, dict) else {}
            for operation in REQUIRED_STEPPING_OPERATIONS:
                active_values = active.get(operation, {}) if isinstance(active, dict) else {}
                if (
                    active_values.get("calls", 0) <= 0
                    or active_values.get("points", 0) <= 0
                ):
                    problems.append(
                        f"{phase_name} has no {expected_backend} {operation} work"
                    )
            for operation in PHASE_COUNTERS:
                inactive_values = (
                    inactive.get(operation, {}) if isinstance(inactive, dict) else {}
                )
                if inactive_values.get("calls", 0) != 0 or inactive_values.get(
                    "points", 0
                ) != 0:
                    problems.append(
                        f"{phase_name} recorded {inactive_backend} {operation} fallback"
                    )

        gradient_phase = phases.get("gradient", {})
        gradient_counters = (
            gradient_phase.get("backend_counters", {})
            if isinstance(gradient_phase, dict)
            else {}
        )
        for phase_backend in ("cpu", "cuda"):
            operations = (
                gradient_counters.get(phase_backend, {})
                if isinstance(gradient_counters, dict)
                else {}
            )
            for operation, values in operations.items():
                if values.get("calls", 0) != 0 or values.get("points", 0) != 0:
                    problems.append(
                        f"gradient postprocessing unexpectedly recorded "
                        f"{phase_backend} {operation} FDTD work"
                    )

    return {"pass": not problems, "problems": problems}


def compare_workloads(cpu: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    """Require equal inputs, timesteps, and phase point workloads."""

    comparisons: list[dict[str, Any]] = []
    cpu_workload = cpu.get("workload", {})
    cuda_workload = cuda.get("workload", {})
    for key in (
        "resolution",
        "run_time",
        "cell_size",
        "design_resolution",
        "design_shape",
        "design_variables",
        "yee_cells",
        "weights_sha256",
        "direction_sha256",
    ):
        left = cpu_workload.get(key)
        right = cuda_workload.get(key)
        comparisons.append(
            {
                "phase": "input",
                "metric": key,
                "cpu": left,
                "cuda": right,
                "pass": left is not None and left == right,
            }
        )

    cpu_phases = cpu_workload.get("phases", {})
    cuda_phases = cuda_workload.get("phases", {})
    for phase_name in STEPPING_PHASES:
        cpu_phase = cpu_phases.get(phase_name, {})
        cuda_phase = cuda_phases.get(phase_name, {})
        cpu_steps = cpu_phase.get("timesteps")
        cuda_steps = cuda_phase.get("timesteps")
        comparisons.append(
            {
                "phase": phase_name,
                "metric": "timesteps",
                "cpu": cpu_steps,
                "cuda": cuda_steps,
                "pass": (
                    isinstance(cpu_steps, int)
                    and not isinstance(cpu_steps, bool)
                    and cpu_steps > 0
                    and cpu_steps == cuda_steps
                ),
            }
        )
        cpu_end_time = cpu_phase.get("end_time")
        cuda_end_time = cuda_phase.get("end_time")
        comparisons.append(
            {
                "phase": phase_name,
                "metric": "end_time",
                "cpu": cpu_end_time,
                "cuda": cuda_end_time,
                "pass": (
                    isinstance(cpu_end_time, (int, float))
                    and not isinstance(cpu_end_time, bool)
                    and math.isfinite(float(cpu_end_time))
                    and float(cpu_end_time) > 0.0
                    and cpu_end_time == cuda_end_time
                ),
            }
        )
        cpu_counters = cpu_phase.get("backend_counters", {}).get("cpu", {})
        cuda_counters = cuda_phase.get("backend_counters", {}).get("cuda", {})
        for operation in PHASE_COUNTERS:
            cpu_points = cpu_counters.get(operation, {}).get("points")
            cuda_points = cuda_counters.get(operation, {}).get("points")
            comparisons.append(
                {
                    "phase": phase_name,
                    "metric": f"{operation}_points",
                    "cpu": cpu_points,
                    "cuda": cuda_points,
                    "pass": (
                        isinstance(cpu_points, int)
                        and not isinstance(cpu_points, bool)
                        and cpu_points >= 0
                        and cpu_points == cuda_points
                    ),
                }
            )
    return {
        "pass": all(item["pass"] for item in comparisons),
        "comparisons": comparisons,
    }
