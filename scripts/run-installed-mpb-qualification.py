#!/usr/bin/env python3
"""Execute and attest the installed MPB Python/runtime closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys


LOG_NAME = "installed-mpb-runtime.log"
QUALIFICATION = "gpmeep-installed-mpb-runtime-v1"


def _record(path: pathlib.Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _mapped(prefix: str) -> set[pathlib.Path]:
    result: set[pathlib.Path] = set()
    for line in pathlib.Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            candidate = pathlib.Path(columns[5]).resolve()
            if candidate.name.startswith(prefix):
                result.add(candidate)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-extension", required=True, type=pathlib.Path)
    parser.add_argument("--expected-libpympb", required=True, type=pathlib.Path)
    parser.add_argument("--expected-python-extension", required=True, type=pathlib.Path)
    parser.add_argument("--expected-libmeep", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expected_extension = args.expected_extension.resolve(strict=True)
    expected_libpympb = args.expected_libpympb.resolve(strict=True)
    expected_python_extension = args.expected_python_extension.resolve(strict=True)
    expected_libmeep = args.expected_libmeep.resolve(strict=True)
    expected_pythonpath = str(expected_extension.parents[2])
    if os.environ.get("PYTHONPATH") != expected_pythonpath:
        raise RuntimeError("installed MPB qualification has an unexpected PYTHONPATH")

    import meep as mp
    from meep import mpb
    from meep.mpb import _mpb
    from mpi4py import MPI

    actual_extension = pathlib.Path(str(_mpb.__file__)).resolve(strict=True)
    if actual_extension != expected_extension:
        raise RuntimeError(f"Python loaded unexpected _mpb: {actual_extension}")
    if not mp.with_mpi() or not mp.is_single_precision() or not mp.gpu.compiled:
        raise RuntimeError("installed MPB did not load from the FP32 CUDA/MPI build")
    expected_mappings = {
        "libpympb": ("libpympb.so", expected_libpympb),
        "libmeep": ("libmeep.so", expected_libmeep),
        "python_extension": ("_meep.so", expected_python_extension),
    }
    for label, (prefix, expected) in expected_mappings.items():
        observed = _mapped(prefix)
        if observed != {expected}:
            raise RuntimeError(
                f"installed MPB loaded unexpected {label} mapping(s): "
                + ", ".join(str(path) for path in sorted(observed))
            )

    def solve() -> list[float]:
        solver = mpb.ModeSolver(
            num_bands=2,
            k_points=[mp.Vector3(0.5)],
            geometry_lattice=mp.Lattice(size=mp.Vector3(1)),
            geometry=[
                mp.Block(
                    size=mp.Vector3(0.5, mp.inf, mp.inf),
                    material=mp.Medium(epsilon=4),
                )
            ],
            resolution=8,
            deterministic=True,
            tolerance=1e-7,
        )
        solver.run_te()
        values = [float(value) for value in solver.all_freqs[0]]
        if (
            len(values) != 2
            or not all(math.isfinite(value) and value > 0 for value in values)
            or values != sorted(values)
        ):
            raise RuntimeError("installed MPB produced invalid eigenfrequencies")
        return values

    first = solve()
    second = solve()
    maximum_repeat_error = max(abs(left - right) for left, right in zip(first, second))
    if maximum_repeat_error > 1e-7:
        raise RuntimeError("installed MPB deterministic repeatability check failed")
    value = {
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "log_name": LOG_NAME,
        "world_size": int(MPI.COMM_WORLD.Get_size()),
        "single_precision_meep": True,
        "mpb_extension": _record(expected_extension),
        "libpympb": _record(expected_libpympb),
        "python_extension": _record(expected_python_extension),
        "libmeep": _record(expected_libmeep),
        "frequencies": first,
        "repeat_frequencies": second,
        "maximum_repeat_error": maximum_repeat_error,
    }
    mp._gpu_finalize_distributed_runtime()
    MPI.Finalize()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    print(f"gpmeep-qualification:{LOG_NAME}:PASS")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
