#!/usr/bin/env python3
"""Run an installed-tree CUDA suite and provenance probe in one process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import unittest
from typing import Any

from mpi4py import MPI

import meep as mp


def _load_qualification_contract():
    path = pathlib.Path(__file__).resolve().with_name(
        "gpmeep_qualification_contract.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "gpmeep_qualification_contract_for_installed_cuda", path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load qualification contract: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


_QUALIFICATION_CONTRACT = _load_qualification_contract()
INSTALLED_CUDA_QUALIFICATION_NAME = (
    _QUALIFICATION_CONTRACT.INSTALLED_CUDA_QUALIFICATION_NAME
)
INSTALLED_CUDA_QUALIFICATION_SPECS = (
    _QUALIFICATION_CONTRACT.INSTALLED_CUDA_QUALIFICATION_SPECS
)
QualificationContractError = _QUALIFICATION_CONTRACT.QualificationContractError
marker_for = _QUALIFICATION_CONTRACT.marker_for
python_runtime_attestation = (
    _QUALIFICATION_CONTRACT.python_runtime_attestation
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_attestation(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_suite(path: pathlib.Path, log_name: str) -> unittest.TestSuite:
    module_name = "_gpmeep_installed_cuda_" + log_name.replace("-", "_").replace(
        ".", "_"
    )
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise QualificationContractError(f"cannot load CUDA suite: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def _suite_summary(result: unittest.TestResult) -> dict[str, Any]:
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "successful": result.wasSuccessful(),
    }


def _cuda_execution_probe() -> dict[str, Any]:
    mp.gpu.set_backend("cuda")
    mp.gpu.reset_statistics()
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.6, 1.4),
        resolution=10,
        boundary_layers=[mp.PML(0.2)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.3, fwidth=0.1),
                component=mp.Ez,
                center=mp.Vector3(),
            )
        ],
    )
    simulation.run(until=2)
    value = {
        "requested_backend": mp.gpu.requested_backend,
        "active_backend": mp.gpu.active_backend,
        "selected_device": int(mp.gpu.selected_device),
        "selected_device_identifier": mp.gpu.selected_device_identifier,
        "cuda_execution_selected": simulation.fields.gpu_cuda_execution_selected(),
        "execution_diagnostic": simulation.fields.gpu_execution_diagnostic(),
        "statistics": mp.gpu.statistics(),
    }
    simulation.reset_meep()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--expected-extension", required=True, type=pathlib.Path)
    parser.add_argument("--expected-libmeep", required=True, type=pathlib.Path)
    parser.add_argument(
        "--log-name",
        required=True,
        choices=tuple(sorted(INSTALLED_CUDA_QUALIFICATION_SPECS)),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    spec = INSTALLED_CUDA_QUALIFICATION_SPECS[args.log_name]
    expected_environment = dict(spec["environment"])
    actual_environment = {
        name: os.environ.get(name) for name in expected_environment
    }
    if actual_environment != expected_environment:
        raise QualificationContractError(
            "installed CUDA qualification environment is not exact: "
            f"expected={expected_environment!r}, actual={actual_environment!r}"
        )

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    world_size = communicator.Get_size()
    if world_size != spec["world_size"] or mp.count_processors() != world_size:
        raise QualificationContractError(
            "installed CUDA qualification has the wrong MPI world size"
        )

    suite_path = (repo / str(spec["test_file"])).resolve()
    try:
        suite_path.relative_to(repo)
    except ValueError as exc:
        raise QualificationContractError(
            "installed CUDA suite escaped the source repository"
        ) from exc
    if not suite_path.is_file() or suite_path.is_symlink():
        raise QualificationContractError(
            f"installed CUDA suite is absent or unsafe: {suite_path}"
        )
    suite_file = _file_attestation(suite_path)
    suite = _load_suite(suite_path, args.log_name)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)

    probe = _cuda_execution_probe()
    runtime = python_runtime_attestation(
        args.expected_extension.resolve(), args.expected_libmeep.resolve()
    )
    rank_record = {
        "world_rank": rank,
        "world_size": world_size,
        "suite": _suite_summary(result),
        "python_extension": runtime["python_extension"],
        "libmeep": runtime["libmeep"],
        "probe": probe,
    }
    rank_records = communicator.gather(rank_record, root=0)

    success = result.wasSuccessful() and result.testsRun == spec["tests_run"]
    all_success = communicator.gather(success, root=0)
    complete = False
    serialized: str | None = None
    if rank == 0:
        assert rank_records is not None
        assert all_success is not None
        complete = all(all_success) and len(rank_records) == world_size
        if complete:
            value = {
                "schema_version": 1,
                "qualification": INSTALLED_CUDA_QUALIFICATION_NAME,
                "log_name": args.log_name,
                "mode": spec["mode"],
                "expected_world_size": world_size,
                "environment": actual_environment,
                "suite_file": suite_file,
                "python_extension": rank_records[0]["python_extension"],
                "libmeep": rank_records[0]["libmeep"],
                "rank_records": sorted(
                    rank_records, key=lambda record: record["world_rank"]
                ),
            }
            serialized = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            )
    complete = communicator.bcast(complete, root=0)
    communicator.Barrier()
    # Meep's Python module normally releases its world-wide CUDA assignment
    # claim window from an atexit handler.  This runner deliberately bypasses
    # Python atexit below, so perform that collective cleanup explicitly while
    # MPI is still live, then finalize MPI before writing terminal evidence.
    mp._gpu_finalize_distributed_runtime()
    MPI.Finalize()
    if rank == 0:
        if complete:
            assert serialized is not None
            print(serialized)
            print(marker_for(args.log_name).decode("ascii"))
        else:
            print(
                "installed CUDA qualification suite was incomplete or failed",
                file=sys.stderr,
            )
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(0 if complete else 1)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"installed CUDA qualification error: {exc}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        # A rank-local failure cannot safely enter Meep's collective claim-
        # window cleanup while peers may be in a different operation.  Abort
        # the initialized world explicitly instead of leaking it via _exit or
        # hanging until the builder timeout.
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.Abort(1)
        os._exit(1)
