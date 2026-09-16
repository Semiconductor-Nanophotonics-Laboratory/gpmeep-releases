#!/usr/bin/env python3
"""Capture the exact shared-library closure used by the installed package."""

from __future__ import annotations

import argparse
import pathlib
import sys
import types
from typing import Any


def _load_evidence_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name("gpmeep_benchmark_evidence.py")
    module = types.ModuleType("gpmeep_benchmark_evidence")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


_EVIDENCE = _load_evidence_from_source()


def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def capture(
    output: pathlib.Path,
    environment_prefix: pathlib.Path,
    installed_prefix: pathlib.Path,
) -> None:
    import meep as mp
    from meep import mpb

    if not callable(mpb.ModeSolver):
        raise RuntimeError("installed MPB API is unavailable")

    if not mp.with_mpi() or not mp.gpu.compiled or not mp.is_single_precision():
        raise RuntimeError("runtime closure requires the installed FP32 CUDA/MPI package")
    if not bool(mp.gpu.runtime_available):
        raise RuntimeError("CUDA runtime is unavailable while capturing dependencies")
    groups: dict[str, list[dict[str, Any]]] = _EVIDENCE.loaded_library_records(
        {
            "mpi": ("libmpi.so",),
            "cuda_driver": ("libcuda.so",),
            "cuda_runtime": ("libcudart.so",),
            "hdf5": ("libhdf5",),
        }
    )
    prefix = environment_prefix.resolve()
    installed = installed_prefix.resolve()
    groups["environment"] = _EVIDENCE.loaded_mapped_file_records(prefix)
    groups["installed"] = _EVIDENCE.loaded_mapped_file_records(installed)
    for group in ("mpi", "cuda_runtime", "hdf5"):
        if not all(_under(pathlib.Path(record["path"]), prefix) for record in groups[group]):
            raise RuntimeError(f"captured {group} dependency lies outside the environment")
    required_installed = ("_meep.so", "_mpb.so", "libmeep.so", "libpympb.so")
    installed_names = [pathlib.Path(record["path"]).name for record in groups["installed"]]
    if any(
        not any(name.startswith(prefix_name) for name in installed_names)
        for prefix_name in required_installed
    ):
        raise RuntimeError("captured installed runtime closure is incomplete")
    _EVIDENCE.atomic_write_json(
        output.resolve(),
        {
            "schema_version": 3,
            "environment_prefix": str(prefix),
            "installed_prefix": str(installed),
            "groups": groups,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--environment-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--installed-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        capture(args.output, args.environment_prefix, args.installed_prefix)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
