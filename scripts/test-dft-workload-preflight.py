#!/usr/bin/env python3
"""Driver-independent MPI contract test for the M27 DFT workload worker."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


PREFIX = "gpmeep-dft-workload-v1:"
SUMMARY_PREFIX = "gpmeep-dft-workload-preflight-suite-v1:"


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key in DFT workload marker: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise RuntimeError(f"non-finite JSON value in DFT workload marker: {value}")


def parse_success_marker(output: str, expected: dict[str, int]) -> dict[str, int]:
    marker_lines = [
        line for line in output.replace("\r", "\n").splitlines()
        if line.startswith(PREFIX)
    ]
    if len(marker_lines) != 1:
        raise RuntimeError(
            f"expected exactly one DFT workload marker, found {len(marker_lines)}"
        )
    for line in output.replace("\r", "\n").splitlines():
        if "gpmeep-" in line and not line.startswith(PREFIX):
            raise RuntimeError("preflight output contains an unexpected gpmeep marker")
    record = json.loads(
        marker_lines[0][len(PREFIX):],
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
    if (
        not isinstance(record, dict)
        or set(record) != {"components", "frequencies", "monitors"}
        or any(type(record.get(key)) is not int for key in record)
        or record != expected
    ):
        raise RuntimeError("DFT workload marker differs from the exact contract")
    return record


def controlled_environment(home: pathlib.Path, prefix: pathlib.Path) -> dict[str, str]:
    environment = {
        "HOME": str(home),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(prefix / "lib"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
    }
    if any(key.startswith("MEEP_GPU") for key in environment):
        raise RuntimeError("controlled preflight environment leaked a GPU control")
    return environment


def run_case(
    executable: pathlib.Path,
    mpiexec: pathlib.Path,
    base_environment: dict[str, str],
    *,
    name: str,
    ranks: int,
    additions: dict[str, str],
    expected: dict[str, int] | None = None,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    environment = dict(base_environment)
    environment.update(additions)
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "30s",
        str(mpiexec),
        "--bind-to",
        "none",
        "-n",
        str(ranks),
        str(executable),
    ]
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=40,
    )
    if expected is not None:
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} failed with status {result.returncode}: {result.stdout}"
            )
        marker = parse_success_marker(result.stdout, expected)
    else:
        if result.returncode == 0:
            raise RuntimeError(f"{name} unexpectedly accepted an invalid contract")
        if diagnostic is None or diagnostic not in result.stdout:
            raise RuntimeError(
                f"{name} failed without the required diagnostic {diagnostic!r}"
            )
        if PREFIX in result.stdout:
            raise RuntimeError(f"{name} emitted a workload marker after rejection")
        marker = None
    return {
        "name": name,
        "ranks": ranks,
        "returncode": result.returncode,
        "marker": marker,
        "required_diagnostic": diagnostic,
        "pass": True,
    }


def run_suite(executable: pathlib.Path, mpiexec: pathlib.Path) -> dict[str, Any]:
    executable = executable.resolve()
    mpiexec = mpiexec.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("gpu-mpi-performance is absent or not executable")
    if not mpiexec.is_file() or not os.access(mpiexec, os.X_OK):
        raise RuntimeError("mpiexec is absent or not executable")
    prefix = mpiexec.parent.parent
    with tempfile.TemporaryDirectory(prefix="gpmeep-m27-preflight-") as home_value:
        base = controlled_environment(pathlib.Path(home_value), prefix)
        common = {"MEEP_GPU_TEST_DFT_WORKLOAD_PREFLIGHT_ONLY": "1"}
        cases = [
            run_case(
                executable, mpiexec, base,
                name="one-rank-launch-dominated", ranks=1,
                additions={
                    **common,
                    "MEEP_GPU_MULTI_DFT_MONITORS": "64",
                    "MEEP_GPU_MULTI_DFT_FREQUENCIES": "2",
                },
                expected={"components": 2, "frequencies": 2, "monitors": 64},
            ),
            run_case(
                executable, mpiexec, base,
                name="two-rank-frequency-heavy", ranks=2,
                additions={
                    **common,
                    "MEEP_GPU_MULTI_DFT_MONITORS": "8",
                    "MEEP_GPU_MULTI_DFT_FREQUENCIES": "32",
                },
                expected={"components": 2, "frequencies": 32, "monitors": 8},
            ),
            run_case(
                executable, mpiexec, base,
                name="monitor-limit", ranks=1,
                additions={**common, "MEEP_GPU_MULTI_DFT_MONITORS": "257"},
                diagnostic="MEEP_GPU_MULTI_DFT_MONITORS must not exceed 256",
            ),
            run_case(
                executable, mpiexec, base,
                name="frequency-limit", ranks=1,
                additions={**common, "MEEP_GPU_MULTI_DFT_FREQUENCIES": "1025"},
                diagnostic="MEEP_GPU_MULTI_DFT_FREQUENCIES must not exceed 1024",
            ),
            run_case(
                executable, mpiexec, base,
                name="nonpositive-monitor-count", ranks=1,
                additions={**common, "MEEP_GPU_MULTI_DFT_MONITORS": "0"},
                diagnostic="MEEP_GPU_MULTI_DFT_MONITORS must be a positive integer",
            ),
            run_case(
                executable, mpiexec, base,
                name="mutually-exclusive-preflight", ranks=1,
                additions={
                    **common,
                    "MEEP_GPU_TEST_INITIAL_CONDITION_PREFLIGHT_ONLY": "1",
                },
                diagnostic=(
                    "initial-condition and DFT-workload preflight gates are "
                    "mutually exclusive"
                ),
            ),
            run_case(
                executable, mpiexec, base,
                name="two-rank-preflight-mismatch", ranks=2,
                additions={
                    "MEEP_GPU_TEST_DFT_WORKLOAD_PREFLIGHT_RANK_MISMATCH": "1"
                },
                diagnostic=(
                    "fixed workload profile differs across MPI ranks before allocation"
                ),
            ),
        ]
    return {
        "schema_version": 1,
        "driver_required": False,
        "case_count": len(cases),
        "cases": cases,
        "pass": all(case["pass"] for case in cases),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--mpiexec", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_suite(args.executable, args.mpiexec)
    print(SUMMARY_PREFIX + json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
