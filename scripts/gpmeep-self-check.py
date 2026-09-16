#!/usr/bin/env python3
"""Installed-runtime CPU, CUDA, and two-GPU qualification for gpmeep.

The public command is an orchestrator.  Every backend lane runs in a fresh
child process so CPU qualification cannot accidentally inherit CUDA state and
single-GPU qualification cannot inherit a prior device claim.  The private
worker interface is intentionally undocumented and emits one prefixed JSON
record for the orchestrator to validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))
from gpmeep_release_manifest import (  # noqa: E402
    DISTRIBUTION_VERSION,
    RELEASE_MANIFEST_FIELDS,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    SEALED_M3_ANCHOR,
    ReleaseManifestError,
    validate_cuda_architecture_audit,
    validate_glibc_compatibility_audit,
    validate_release_manifest,
)

SCHEMA_VERSION = 1
MARKER = "gpmeep-self-check-v1:"
PUBLIC_MARKER = "gpmeep-self-check-result-v1:"
RELEASE_MANIFEST = pathlib.Path("share/gpmeep/release.json")
DEFAULT_TIMEOUT_SECONDS = 180
FIELD_ATOL = 5.0e-4
FIELD_RTOL = 5.0e-4


class SelfCheckError(RuntimeError):
    """Raised when an installed-runtime contract is not satisfied."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SelfCheckError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise SelfCheckError(f"non-finite JSON constant {value!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SelfCheckError(f"expected a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _flatten_counters(value: object) -> dict[str, int]:
    counters: dict[str, int] = {}

    def visit(node: object, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, (*prefix, str(key)))
        elif isinstance(node, bool):
            counters[".".join(prefix)] = int(node)
        elif isinstance(node, int):
            counters[".".join(prefix)] = node

    visit(value, ())
    return counters


def _counter(counters: dict[str, int], suffix: str) -> int:
    matches = [value for key, value in counters.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise SelfCheckError(
            f"expected exactly one statistics counter ending in {suffix!r}"
        )
    return matches[0]


def _cpu_execution_counters(counters: dict[str, int]) -> dict[str, int]:
    return {
        key: value
        for key, value in counters.items()
        if key.rsplit(".", 1)[-1].startswith("cpu_") and value != 0
    }


def _cuda_execution_counters(counters: dict[str, int]) -> dict[str, int]:
    return {
        key: value
        for key, value in counters.items()
        if key.rsplit(".", 1)[-1].startswith("cuda_") and value != 0
    }


def _parse_marker(output: str) -> dict[str, Any]:
    records = []
    for line in output.splitlines():
        if line.startswith(MARKER):
            try:
                value = json.loads(line[len(MARKER) :])
            except json.JSONDecodeError as error:
                raise SelfCheckError("worker marker contains invalid JSON") from error
            if not isinstance(value, dict):
                raise SelfCheckError("worker marker must contain a JSON object")
            records.append(value)
    if len(records) != 1:
        raise SelfCheckError(
            f"worker emitted {len(records)} terminal markers; expected one"
        )
    return records[0]


def _normalized_visible_devices(specification: str | None) -> list[str]:
    if specification is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited is not None:
            specification = inherited
    if specification is not None:
        values = [item.strip() for item in specification.split(",")]
        if any(not item for item in values):
            raise SelfCheckError("GPU device list contains an empty entry")
        if len(set(values)) != len(values):
            raise SelfCheckError("GPU device list contains duplicates")
        return values

    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise SelfCheckError(
            "cannot discover a GPU; pass --gpu-devices or install nvidia-smi"
        )
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise SelfCheckError("nvidia-smi GPU discovery failed")
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values or len(set(values)) != len(values):
        raise SelfCheckError("nvidia-smi returned no unique GPU indices")
    return values


def _release_manifest() -> dict[str, Any]:
    prefix = pathlib.Path(sys.prefix).resolve(strict=True)
    lexical_path = prefix / RELEASE_MANIFEST
    try:
        path = lexical_path.resolve(strict=True)
        path.relative_to(prefix)
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise OSError("release manifest is not regular")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SelfCheckError(
            f"installed release manifest is unreadable: {lexical_path}"
        ) from error
    try:
        validate_release_manifest(value)
        validate_cuda_architecture_audit(
            prefix,
            value["cuda_architecture_audit_sha256"],
            value["cuda_real_architectures"],
            value["cuda_virtual_architectures"],
        )
        validate_glibc_compatibility_audit(
            prefix, value["glibc_compatibility_audit_sha256"]
        )
    except ReleaseManifestError as error:
        raise SelfCheckError(str(error)) from error
    value["file"] = _file_record(path)
    return value


def _mapped_libmeep() -> pathlib.Path:
    candidates: set[pathlib.Path] = set()
    maps = pathlib.Path("/proc/self/maps")
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise SelfCheckError("cannot inspect mapped gpmeep libraries") from error
    for line in lines:
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = pathlib.Path(columns[5])
            if path.name.startswith("libmeep.so") and path.is_file():
                candidates.add(path.resolve(strict=True))
    if len(candidates) != 1:
        raise SelfCheckError(f"expected one mapped libmeep, found {len(candidates)}")
    return next(iter(candidates))


def _run_fixed_workload(lane: str) -> dict[str, Any]:
    import numpy as np
    from mpi4py import MPI
    import meep as mp
    import meep._meep as native

    communicator = MPI.COMM_WORLD
    expected_size = 2 if lane == "gpu2" else 1
    if communicator.size != expected_size or mp.count_processors() != expected_size:
        raise SelfCheckError(
            f"{lane} worker requires exactly {expected_size} aligned MPI rank(s)"
        )
    backend = "cpu" if lane == "cpu" else "cuda"
    if not mp.with_mpi() or not mp.is_single_precision():
        raise SelfCheckError("installed gpmeep must be MPI-enabled FP32")
    if not mp.gpu.compiled:
        raise SelfCheckError("installed gpmeep was not compiled with CUDA")

    mp.gpu.set_backend(backend)
    mp.gpu.reset_statistics()
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3.0, 2.5),
        resolution=20,
        boundary_layers=[mp.PML(0.4)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.3, fwidth=0.12),
                component=mp.Ez,
                center=mp.Vector3(-0.65),
            )
        ],
    )
    flux = simulation.add_flux(
        0.3,
        0.12,
        3,
        mp.FluxRegion(center=mp.Vector3(0.65), size=mp.Vector3(y=1.2)),
    )
    communicator.Barrier()
    started = time.perf_counter()
    simulation.run(until=8)
    communicator.Barrier()
    elapsed = time.perf_counter() - started
    field = np.asarray(
        simulation.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(3.0, 2.5),
            component=mp.Ez,
        ),
        dtype=np.float64,
    )
    field_values = [float(value) for value in field.ravel(order="C")]
    flux_values = [float(value) for value in mp.get_fluxes(flux)]
    if not field_values or not all(math.isfinite(value) for value in field_values):
        raise SelfCheckError("field observable is empty or non-finite")
    if not flux_values or not all(math.isfinite(value) for value in flux_values):
        raise SelfCheckError("flux observable is empty or non-finite")
    record = {
        "rank": communicator.rank,
        "world_size": communicator.size,
        "lane": lane,
        "backend": backend,
        "requested_backend": mp.gpu.requested_backend,
        "active_backend": mp.gpu.active_backend,
        "cuda_execution_selected": bool(
            simulation.fields.gpu_cuda_execution_selected()
        ),
        "execution_diagnostic": simulation.fields.gpu_execution_diagnostic(),
        "selected_device": int(mp.gpu.selected_device),
        "selected_device_identifier": mp.gpu.selected_device_identifier,
        "elapsed_seconds": elapsed,
        "timesteps": int(simulation.timestep()),
        "field_shape": list(field.shape),
        "field_values": field_values,
        "field_sha256": hashlib.sha256(
            np.asarray(field_values, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "field_l2": float(np.linalg.norm(field)),
        "flux_values": flux_values,
        "statistics": mp.gpu.statistics(),
        "python": _file_record(pathlib.Path(sys.executable)),
        "extension": _file_record(pathlib.Path(native.__file__)),
        "libmeep": _file_record(_mapped_libmeep()),
        "meep_version": mp.__version__,
    }
    simulation.reset_meep()
    records = communicator.gather(record, root=0)
    communicator.Barrier()
    if communicator.rank != 0:
        return {}
    assert records is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "lane": lane,
        "rank_records": sorted(records, key=lambda item: item["rank"]),
    }


def _validate_cpu_record(record: dict[str, Any]) -> None:
    if (
        record.get("backend") != "cpu"
        or record.get("requested_backend") != "cpu"
        or record.get("active_backend") != "cpu"
        or record.get("cuda_execution_selected") is not False
    ):
        raise SelfCheckError("CPU worker did not execute the strict CPU backend")
    counters = _flatten_counters(record.get("statistics"))
    if _counter(counters, "cpu_update_eh_points") <= 0:
        raise SelfCheckError("CPU worker recorded no field-update work")
    if _counter(counters, "cpu_curl_points") <= 0:
        raise SelfCheckError("CPU worker recorded no curl work")
    active_cuda = _cuda_execution_counters(counters)
    if active_cuda:
        raise SelfCheckError(f"CPU worker recorded CUDA execution: {active_cuda}")


def _validate_cuda_record(record: dict[str, Any]) -> None:
    if (
        record.get("backend") != "cuda"
        or record.get("requested_backend") != "cuda"
        or record.get("active_backend") != "cuda"
        or record.get("cuda_execution_selected") is not True
    ):
        raise SelfCheckError("CUDA worker did not execute the strict CUDA backend")
    identifier = record.get("selected_device_identifier")
    if not isinstance(identifier, str) or not identifier:
        raise SelfCheckError("CUDA worker has no physical device identifier")
    counters = _flatten_counters(record.get("statistics"))
    if _counter(counters, "cuda_update_eh_points") <= 0:
        raise SelfCheckError("CUDA worker recorded no field-update work")
    if _counter(counters, "cuda_curl_points") <= 0:
        raise SelfCheckError("CUDA worker recorded no curl work")
    active_cpu = _cpu_execution_counters(counters)
    if active_cpu:
        raise SelfCheckError(f"CUDA worker recorded CPU fallback: {active_cpu}")


def _validate_lane(value: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    if value.get("schema_version") != SCHEMA_VERSION or value.get("lane") != lane:
        raise SelfCheckError(f"{lane} worker schema or lane is invalid")
    records = value.get("rank_records")
    expected = 2 if lane == "gpu2" else 1
    if not isinstance(records, list) or len(records) != expected:
        raise SelfCheckError(f"{lane} worker returned the wrong rank count")
    if [record.get("rank") for record in records] != list(range(expected)):
        raise SelfCheckError(f"{lane} worker ranks are incomplete or unordered")
    for record in records:
        if record.get("world_size") != expected or record.get("lane") != lane:
            raise SelfCheckError(f"{lane} worker rank identity is invalid")
        if lane == "cpu":
            _validate_cpu_record(record)
        else:
            _validate_cuda_record(record)
    if lane == "gpu2":
        identifiers = [record["selected_device_identifier"] for record in records]
        if len(set(identifiers)) != 2:
            raise SelfCheckError("two-GPU workers did not select distinct devices")
        for record in records:
            counters = _flatten_counters(record["statistics"])
            if _counter(counters, "pinned_staging_bytes") <= 0:
                raise SelfCheckError("two-GPU worker recorded no pinned MPI traffic")
            if _counter(counters, "mpi_waitsome_executions") <= 0:
                raise SelfCheckError("two-GPU worker recorded no waitsome completion")
    return records


def _compare_observables(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | int]:
    if reference.get("field_shape") != candidate.get("field_shape"):
        raise SelfCheckError("CPU and CUDA field shapes differ")
    lhs = reference.get("field_values")
    rhs = candidate.get("field_values")
    if not isinstance(lhs, list) or not isinstance(rhs, list) or len(lhs) != len(rhs):
        raise SelfCheckError("CPU and CUDA field vectors differ in length")
    differences = [abs(float(a) - float(b)) for a, b in zip(lhs, rhs)]
    maximum_absolute = max(differences, default=0.0)
    reference_norm = math.sqrt(sum(float(value) ** 2 for value in lhs))
    difference_norm = math.sqrt(sum(value * value for value in differences))
    relative_l2 = difference_norm / max(reference_norm, 1.0e-30)
    violations = sum(
        abs(float(a) - float(b)) > FIELD_ATOL + FIELD_RTOL * abs(float(a))
        for a, b in zip(lhs, rhs)
    )
    reference_flux = reference.get("flux_values")
    candidate_flux = candidate.get("flux_values")
    if (
        not isinstance(reference_flux, list)
        or not isinstance(candidate_flux, list)
        or len(reference_flux) != len(candidate_flux)
    ):
        raise SelfCheckError("CPU and CUDA flux vectors differ in length")
    flux_maximum_absolute = max(
        (abs(float(a) - float(b)) for a, b in zip(reference_flux, candidate_flux)),
        default=0.0,
    )
    flux_violations = sum(
        abs(float(a) - float(b)) > FIELD_ATOL + FIELD_RTOL * abs(float(a))
        for a, b in zip(reference_flux, candidate_flux)
    )
    if violations or flux_violations:
        raise SelfCheckError(
            "CPU/CUDA numerical comparison failed: "
            f"field violations={violations}, flux violations={flux_violations}"
        )
    return {
        "field_count": len(lhs),
        "field_maximum_absolute_error": maximum_absolute,
        "field_relative_l2_error": relative_l2,
        "field_violations": violations,
        "flux_count": len(reference_flux),
        "flux_maximum_absolute_error": flux_maximum_absolute,
        "flux_violations": flux_violations,
        "atol": FIELD_ATOL,
        "rtol": FIELD_RTOL,
    }


def _sanitized_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "MEEP_GPU_DEVICE",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
        "MEEP_GPU_MPI_TRANSPORT",
        "MEEP_GPU_MPI_COMPLETION",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    return environment


def _run_worker(lane: str, devices: list[str], timeout_seconds: int) -> dict[str, Any]:
    environment = _sanitized_environment()
    script = pathlib.Path(__file__).resolve(strict=True)
    command = [sys.executable, str(script), "--_worker", lane]
    if lane == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["MEEP_GPU_BACKEND"] = "cpu"
    elif lane == "gpu1":
        environment["CUDA_VISIBLE_DEVICES"] = devices[0]
        environment["MEEP_GPU_BACKEND"] = "cuda"
        environment["MEEP_GPU_DEVICE"] = "0"
    elif lane == "gpu2":
        mpiexec = shutil.which("mpiexec", path=environment.get("PATH"))
        if mpiexec is None:
            raise SelfCheckError("mpiexec is absent from the installed environment")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:2])
        environment["MEEP_GPU_BACKEND"] = "cuda"
        environment["MEEP_GPU_ALLOW_OVERSUBSCRIBE"] = "0"
        environment["MEEP_GPU_MPI_TRANSPORT"] = "pinned"
        environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
        command = [mpiexec, "-n", "2", sys.executable, str(script), "--_worker", lane]
    else:
        raise SelfCheckError(f"unknown worker lane {lane!r}")
    result = subprocess.run(
        command,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-30:])
        raise SelfCheckError(
            f"{lane} worker exited with status {result.returncode}:\n{tail}"
        )
    return _parse_marker(result.stdout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify an installed gpmeep CPU/CUDA runtime"
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu1", "gpu2"),
        default="gpu1",
        help="cpu; CPU+GPU1; or CPU+GPU1+GPU2 (default: gpu1)",
    )
    parser.add_argument(
        "--gpu-devices",
        help="comma-separated physical CUDA indices or UUIDs",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        help="write the canonical JSON report to a new file",
    )
    parser.add_argument(
        "--_worker", choices=("cpu", "gpu1", "gpu2"), help=argparse.SUPPRESS
    )
    return parser.parse_args()


def _write_report(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise SelfCheckError(f"refusing to overwrite report: {path}") from error


def _public_receipt(report: dict[str, Any]) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for lane, value in report["lanes"].items():
        records = value["rank_records"]
        lanes[lane] = {
            "rank_count": len(records),
            "device_identifiers": [
                record["selected_device_identifier"]
                for record in records
                if record.get("selected_device_identifier")
            ],
            "maximum_elapsed_seconds": max(
                (float(record["elapsed_seconds"]) for record in records),
                default=0.0,
            ),
            "timesteps": sorted({int(record["timesteps"]) for record in records}),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "state": report["state"],
        "mode": report["mode"],
        "report_id": report["report_id"],
        "wall_seconds": report["wall_seconds"],
        "valid_for_speed_gate": report["valid_for_speed_gate"],
        "comparisons": report["comparisons"],
        "lanes": lanes,
    }


def _public_main(args: argparse.Namespace) -> int:
    if args.timeout_seconds < 10 or args.timeout_seconds > 3600:
        raise SelfCheckError("--timeout-seconds must be in [10, 3600]")
    release = _release_manifest()
    devices: list[str] = []
    if args.mode != "cpu":
        devices = _normalized_visible_devices(args.gpu_devices)
        required = 2 if args.mode == "gpu2" else 1
        if len(devices) < required:
            raise SelfCheckError(
                f"{args.mode} qualification requires {required} visible GPU(s)"
            )
    lanes = ["cpu"]
    if args.mode in {"gpu1", "gpu2"}:
        lanes.append("gpu1")
    if args.mode == "gpu2":
        lanes.append("gpu2")
    outputs: dict[str, Any] = {}
    validated: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for lane in lanes:
        outputs[lane] = _run_worker(lane, devices, args.timeout_seconds)
        validated[lane] = _validate_lane(outputs[lane], lane)
    comparisons: dict[str, Any] = {}
    if "gpu1" in validated:
        comparisons["cpu_vs_gpu1"] = _compare_observables(
            validated["cpu"][0], validated["gpu1"][0]
        )
    if "gpu2" in validated:
        comparisons["cpu_vs_gpu2"] = _compare_observables(
            validated["cpu"][0], validated["gpu2"][0]
        )
        comparisons["gpu1_vs_gpu2"] = _compare_observables(
            validated["gpu1"][0], validated["gpu2"][0]
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "mode": args.mode,
        "release": release,
        "lanes": outputs,
        "comparisons": comparisons,
        "wall_seconds": time.perf_counter() - started,
        "valid_for_speed_gate": False,
        "speed_gate_reason": "small installed-runtime correctness smoke",
    }
    report["report_id"] = hashlib.sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    if args.report is not None:
        _write_report(args.report, report)
    print(f"gpmeep self-check PASS: mode={args.mode}, report_id={report['report_id']}")
    print(PUBLIC_MARKER + _canonical_json(_public_receipt(report)))
    return 0


def main() -> int:
    args = _parse_args()
    if args._worker is not None:
        value = _run_fixed_workload(args._worker)
        if value:
            print(MARKER + _canonical_json(value))
        return 0
    return _public_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelfCheckError, subprocess.TimeoutExpired) as error:
        print(f"gpmeep self-check FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
