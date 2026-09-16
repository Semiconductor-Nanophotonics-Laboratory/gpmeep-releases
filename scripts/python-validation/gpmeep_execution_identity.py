#!/usr/bin/env python3
"""Fail-closed runtime identity records for gpmeep validation probes."""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
import pathlib
import socket
import sys
from collections.abc import Sequence
from typing import Any


DIAGNOSTICS_PREFIX = "gpmeep-execution-identity:"
ENVIRONMENT_KEYS = (
    "CUDA_CACHE_DISABLE",
    "CUDA_VISIBLE_DEVICES",
    "GPMEEP_VALIDATION_STRICT_CUDA",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "MEEP_GPU_BACKEND",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_RANK",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "PMI_RANK",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = pathlib.Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"runtime identity path is not a file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
        "device_major": os.major(stat.st_dev),
        "device_minor": os.minor(stat.st_dev),
        "inode": stat.st_ino,
    }


def _mapped_paths() -> tuple[
    dict[pathlib.Path, set[tuple[int, int, int]]], set[pathlib.Path]
]:
    result: dict[pathlib.Path, set[tuple[int, int, int]]] = {}
    deleted_paths: set[pathlib.Path] = set()
    for line in pathlib.Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        raw_path = columns[5]
        if raw_path.endswith(" (deleted)"):
            deleted_paths.add(
                pathlib.Path(raw_path[: -len(" (deleted)")]).resolve(strict=False)
            )
            continue
        major, minor = columns[3].split(":", 1)
        path = pathlib.Path(raw_path).resolve(strict=True)
        result.setdefault(path, set()).add(
            (int(major, 16), int(minor, 16), int(columns[4]))
        )
    return result, deleted_paths


def mapped_file_record(
    path: str | os.PathLike[str],
    mappings: dict[pathlib.Path, set[tuple[int, int, int]]],
    deleted_paths: set[pathlib.Path],
) -> dict[str, Any]:
    record = file_record(path)
    resolved = pathlib.Path(record["path"])
    if resolved in deleted_paths:
        raise RuntimeError(f"required runtime mapping was deleted: {resolved}")
    identities = mappings.get(resolved, set())
    expected = {
        (
            int(record["device_major"]),
            int(record["device_minor"]),
            int(record["inode"]),
        )
    }
    if identities != expected:
        raise RuntimeError(
            f"loaded mapping identity differs from the current file: {resolved}"
        )
    record["mapped_identity_verified"] = True
    return record


def mapped_library_record(
    prefix: str,
    mappings: dict[pathlib.Path, set[tuple[int, int, int]]],
    deleted_paths: set[pathlib.Path],
) -> dict[str, Any]:
    candidates = sorted(path for path in mappings if path.name.startswith(prefix))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one mapped {prefix} library, found {len(candidates)}"
        )
    return mapped_file_record(candidates[0], mappings, deleted_paths)


def _optional_mapped_library_record(
    prefix: str,
    mappings: dict[pathlib.Path, set[tuple[int, int, int]]],
    deleted_paths: set[pathlib.Path],
) -> dict[str, Any] | None:
    candidates = sorted(path for path in mappings if path.name.startswith(prefix))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected at most one mapped {prefix} library, found {len(candidates)}"
        )
    return mapped_file_record(candidates[0], mappings, deleted_paths)


def _proc_cmdline() -> list[str]:
    payload = pathlib.Path("/proc/self/cmdline").read_bytes()
    if not payload.endswith(b"\0"):
        raise RuntimeError("process command line is not NUL terminated")
    return [
        item.decode("utf-8", errors="strict")
        for item in payload[:-1].split(b"\0")
    ]


def _loaded_source_manifest(
    mp: Any, validation_sources: Sequence[str | os.PathLike[str]]
) -> dict[str, Any]:
    package_root = pathlib.Path(mp.__file__).resolve(strict=True).parent
    loaded_modules = []
    for name in sorted(sys.modules):
        if name != "meep" and not name.startswith("meep."):
            continue
        module = sys.modules[name]
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = pathlib.Path(module_file).resolve(strict=True)
        if not resolved.is_relative_to(package_root):
            raise RuntimeError(
                f"loaded Meep module is outside the selected package root: {name}"
            )
        loaded_modules.append({"module": name, **file_record(resolved)})
    required_modules = {
        "meep",
        "meep._meep",
        "meep.geom",
        "meep.materials",
        "meep.simulation",
        "meep.source",
    }
    loaded_names = {record["module"] for record in loaded_modules}
    missing = sorted(required_modules - loaded_names)
    if missing:
        raise RuntimeError(
            "required loaded Meep source modules are missing: " + ", ".join(missing)
        )
    source_paths = sorted(
        {pathlib.Path(path).resolve(strict=True) for path in validation_sources},
        key=str,
    )
    if not source_paths:
        raise RuntimeError("execution identity requires validation source files")
    manifest = {
        "selected_meep_package_root": str(package_root),
        "loaded_meep_modules": loaded_modules,
        "validation_sources": [file_record(path) for path in source_paths],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _cuda_version_record(libcudart_path: str) -> dict[str, int]:
    library = ctypes.CDLL(libcudart_path)
    result = {}
    for record_name, function_name in (
        ("runtime_version", "cudaRuntimeGetVersion"),
        ("driver_api_version", "cudaDriverGetVersion"),
    ):
        value = ctypes.c_int()
        function = getattr(library, function_name)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        status = int(function(ctypes.byref(value)))
        if status != 0 or value.value <= 0:
            raise RuntimeError(
                f"{function_name} failed while capturing CUDA provenance: {status}"
            )
        result[record_name] = int(value.value)
    return result


def _local_identity(
    mp: Any,
    label: str,
    simulation: Any,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    extension = getattr(mp, "_meep", None)
    if not getattr(mp, "__file__", None) or not getattr(extension, "__file__", None):
        raise RuntimeError("Meep module or native extension path is unavailable")
    if simulation.fields is None:
        raise RuntimeError("simulation must be initialized before identity capture")
    timestep = int(simulation.fields.t)
    if timestep != 0:
        raise RuntimeError("execution identity must be captured before stepping")
    mappings, deleted_paths = _mapped_paths()
    rank = int(mp.my_rank())
    size = int(mp.count_processors())
    devices = list(mp.gpu.devices())
    selected_ordinal = int(mp.gpu.selected_device)
    selected_identifier = str(mp.gpu.selected_device_identifier)
    active_backend = str(mp.gpu.active_backend)
    requested_backend = str(mp.gpu.requested_backend)
    if active_backend == "cuda":
        selected = [
            item for item in devices if int(item["ordinal"]) == selected_ordinal
        ]
        if len(selected) != 1 or str(selected[0]["identifier"]) != selected_identifier:
            raise RuntimeError("selected CUDA device identity is inconsistent")
    python_executable = file_record(sys.executable)
    proc_self_exe = mapped_file_record("/proc/self/exe", mappings, deleted_paths)
    if python_executable["path"] != proc_self_exe["path"]:
        raise RuntimeError("sys.executable and /proc/self/exe resolve differently")
    critical_libraries = {
        "libmpb": mapped_library_record("libmpb.so", mappings, deleted_paths)
    }
    for name, prefix in (
        ("libcudart", "libcudart.so"),
        ("libcuda_driver", "libcuda.so"),
    ):
        record = _optional_mapped_library_record(prefix, mappings, deleted_paths)
        if record is not None:
            critical_libraries[name] = record
    cuda_versions = None
    if active_backend == "cuda":
        if "libcudart" not in critical_libraries or "libcuda_driver" not in critical_libraries:
            raise RuntimeError("active CUDA execution lacks mapped runtime/driver DSOs")
        cuda_versions = _cuda_version_record(
            str(critical_libraries["libcudart"]["path"])
        )
    return {
        "label": label,
        "rank": rank,
        "world_size": size,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cwd": str(pathlib.Path.cwd().resolve()),
        "sys_argv": list(sys.argv),
        "proc_cmdline": _proc_cmdline(),
        "python_executable": python_executable,
        "proc_self_exe": proc_self_exe,
        "meep_module": file_record(mp.__file__),
        "meep_extension": mapped_file_record(
            extension.__file__, mappings, deleted_paths
        ),
        "libmeep": mapped_library_record(
            "libmeep.so", mappings, deleted_paths
        ),
        "critical_libraries": critical_libraries,
        "cuda_versions": cuda_versions,
        "source_manifest_sha256": source_manifest_sha256,
        "precision": "fp32" if bool(mp.is_single_precision()) else "fp64",
        "with_mpi": bool(mp.with_mpi()),
        "backend": {
            "compiled": bool(mp.gpu.compiled),
            "compiled_architectures": str(mp.gpu.compiled_architectures),
            "requested": requested_backend,
            "active": active_backend,
            "selected_ordinal": selected_ordinal,
            "selected_identifier": selected_identifier,
            "devices": devices,
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "simulation": {
            "initial_timestep": timestep,
            "dt": float(simulation.fields.dt),
            "last_source_time": float(simulation.fields.last_source_time()),
        },
    }


def capture_execution_identity(
    mp: Any,
    label: str,
    simulation: Any,
    validation_sources: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Capture and all-rank replicate a pre-step runtime identity document."""

    source_manifest = _loaded_source_manifest(mp, validation_sources)
    local = _local_identity(
        mp, label, simulation, str(source_manifest["manifest_sha256"])
    )
    world_size = int(mp.count_processors())
    if bool(mp.with_mpi()) and world_size > 1:
        from mpi4py import MPI

        communicator = MPI.COMM_WORLD
        if communicator.size != world_size or communicator.rank != int(mp.my_rank()):
            raise RuntimeError("mpi4py and Meep MPI worlds disagree")
        gathered = communicator.gather(local, root=0)
        document = (
            {
                "schema": "gpmeep-execution-identity-v2",
                "label": label,
                "records": gathered,
                "source_manifest": source_manifest,
            }
            if communicator.rank == 0
            else None
        )
        document = communicator.bcast(document, root=0)
    else:
        document = {
            "schema": "gpmeep-execution-identity-v2",
            "label": label,
            "records": [local],
            "source_manifest": source_manifest,
        }
    ranks = [int(record["rank"]) for record in document["records"]]
    if sorted(ranks) != list(range(world_size)) or len(set(ranks)) != world_size:
        raise RuntimeError("execution identity does not contain every MPI rank once")
    manifest_sha256 = str(document["source_manifest"]["manifest_sha256"])
    unsigned_manifest = dict(document["source_manifest"])
    unsigned_manifest.pop("manifest_sha256")
    if manifest_sha256 != canonical_sha256(unsigned_manifest) or any(
        record["source_manifest_sha256"] != manifest_sha256
        for record in document["records"]
    ):
        raise RuntimeError("execution source manifest differs across MPI ranks")
    document["identity_sha256"] = canonical_sha256(document)
    return document


def emit_execution_identity(identity: dict[str, Any], mp: Any) -> None:
    unsigned = dict(identity)
    recorded = unsigned.pop("identity_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("execution identity self-hash is invalid")
    if bool(mp.am_master()):
        print(
            DIAGNOSTICS_PREFIX
            + json.dumps(
                identity,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
