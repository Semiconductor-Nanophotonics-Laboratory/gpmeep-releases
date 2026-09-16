#!/usr/bin/env python3
"""Validate CUDA DFT decay decisions with reduction enabled and disabled."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import uuid
from typing import Any

import numpy as np

from gpmeep_benchmark_evidence import (
    atomic_write_json,
    atomic_write_text,
    loaded_library_path,
    sha256_file,
)
from gpmeep_provenance import (
    ProvenanceError,
    file_record,
    source_snapshot,
    verify_build_receipt,
)


PREFIX = "gpmeep-dft-decay-worker:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/dft-norm-decay"),
    )
    return parser.parse_args()


def phase_calls(statistics: dict[str, Any], backend: str) -> int:
    prefix = f"{backend}_"
    return sum(
        value
        for group in statistics.values()
        for name, value in group.items()
        if name.startswith(prefix)
        and name.endswith("_calls")
        and isinstance(value, int)
    )


def worker() -> int:
    import meep as mp

    mp.verbosity(0)
    if mp.gpu.active_backend != "cuda":
        raise RuntimeError(
            f"strict CUDA worker selected {mp.gpu.active_backend!r}"
        )
    mp.gpu.reset_statistics()
    resolution = 24
    cell_width = 4.0
    fcen = 0.8
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width),
        resolution=resolution,
        boundary_layers=[mp.PML(0.5)],
        geometry=[
            mp.Block(
                center=mp.Vector3(),
                size=mp.Vector3(1.4, 1.0),
                material=mp.Medium(index=1.7),
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.35),
                component=mp.Ez,
                center=mp.Vector3(-0.7, 0.1),
            )
        ],
    )
    monitor = sim.add_dft_fields(
        [mp.Ez],
        fcen,
        0,
        1,
        where=mp.Volume(
            center=mp.Vector3(0.35, 0),
            size=mp.Vector3(1.4, 1.2),
        ),
    )
    sim.run(
        until_after_sources=mp.stop_when_dft_decayed(
            tol=1e-5,
            minimum_run_time=20,
            maximum_run_time=500,
        )
    )
    dft = np.asarray(sim.get_dft_array(monitor, mp.Ez, 0))
    sample = sim.get_field_point(mp.Ez, mp.Vector3(0.25, -0.15))
    statistics = mp.gpu.statistics()
    extension = pathlib.Path(mp._meep.__file__).resolve()
    libmeep = loaded_library_path("libmeep.so")
    metrics = {
        "active_backend": mp.gpu.active_backend,
        "build_receipt_id": os.environ.get("GPMEEP_DFT_BUILD_RECEIPT_ID"),
        "compiled": bool(mp.gpu.compiled),
        "cpu_phase_calls": phase_calls(statistics, "cpu"),
        "cuda_curl_calls": statistics["dispatch"]["cuda_curl_calls"],
        "cuda_dft_calls": statistics["dfts"]["cuda_dft_calls"],
        "cuda_phase_calls": phase_calls(statistics, "cuda"),
        "dft_imag_sum": float(np.imag(dft).sum()),
        "dft_l2": float(np.linalg.norm(dft)),
        "dft_real_sum": float(np.real(dft).sum()),
        "field_sample_imag": float(np.imag(sample)),
        "field_sample_real": float(np.real(sample)),
        "round_time": float(sim.round_time()),
        "run_nonce": os.environ.get("GPMEEP_DFT_RUN_NONCE"),
        "runtime": {
            "extension": {
                "path": str(extension),
                "sha256": sha256_file(extension),
            },
            "libmeep": {
                "path": str(libmeep),
                "sha256": sha256_file(libmeep),
            },
            "meep_module": {
                "path": str(pathlib.Path(mp.__file__).resolve()),
                "sha256": sha256_file(pathlib.Path(mp.__file__).resolve()),
            },
            "python": {
                "path": str(pathlib.Path(sys.executable).resolve()),
                "sha256": sha256_file(pathlib.Path(sys.executable).resolve()),
            },
        },
        "single_precision": bool(mp.is_single_precision()),
        "strict_cuda_marker": os.environ.get("MEEP_GPU_STRICT") == "1",
        "timestep": int(sim.fields.t),
    }
    print(PREFIX + json.dumps(metrics, sort_keys=True))
    if (
        metrics["cpu_phase_calls"] != 0
        or metrics["cuda_phase_calls"] <= 0
        or metrics["compiled"] is not True
        or metrics["single_precision"] is not True
        or metrics["strict_cuda_marker"] is not True
    ):
        raise RuntimeError("worker violated the strict CUDA dispatch contract")
    return 0


def extract_metrics(stdout: str) -> dict[str, Any]:
    records = [
        json.loads(line[len(PREFIX) :])
        for line in stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one worker metrics record, found {len(records)}"
        )
    return records[0]


def run_worker(
    args: argparse.Namespace,
    disable_reduction: bool,
    *,
    receipt_id: str,
    expected_runtime: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    label = "full-sync" if disable_reduction else "resident-reduction"
    env = os.environ.copy()
    preserved_cuda = {
        key: env[key]
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        if key in env
    }
    for key in list(env):
        if (
            key == "PYTHONPATH"
            or key.startswith("MEEP_GPU_")
            or key.startswith("GPMEEP_")
            or key.startswith("OMP_")
            or key.startswith("CUDA_")
        ):
            env.pop(key, None)
    env.update(preserved_cuda)
    run_nonce = uuid.uuid4().hex
    env.update(
        {
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_AUTO_MIN_CELLS": "0",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(args.output / "matplotlib"),
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(
                pathlib.Path(expected_runtime["meep_module"]["path"]).parent.parent
            ),
            "PYTHONNOUSERSITE": "1",
            "GPMEEP_DFT_BUILD_RECEIPT_ID": receipt_id,
            "GPMEEP_DFT_RUN_NONCE": run_nonce,
        }
    )
    if disable_reduction:
        env["MEEP_GPU_DISABLE_DFT_NORM_REDUCTION"] = "1"
    else:
        env.pop("MEEP_GPU_DISABLE_DFT_NORM_REDUCTION", None)
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
    ]
    proc = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = args.output / f"{label}.stdout.log"
    stderr_path = args.output / f"{label}.stderr.log"
    atomic_write_text(stdout_path, proc.stdout)
    atomic_write_text(stderr_path, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} worker exited {proc.returncode}; "
            f"see {args.output / f'{label}.stderr.log'}"
        )
    metrics = extract_metrics(proc.stdout)
    contract_problems = []
    if metrics.get("run_nonce") != run_nonce:
        contract_problems.append("worker nonce mismatch")
    if metrics.get("build_receipt_id") != receipt_id:
        contract_problems.append("worker receipt ID mismatch")
    if metrics.get("runtime") != expected_runtime:
        contract_problems.append("worker runtime closure mismatch")
    if metrics.get("active_backend") != "cuda":
        contract_problems.append("worker did not retain strict CUDA")
    if metrics.get("compiled") is not True or metrics.get("single_precision") is not True:
        contract_problems.append("worker is not a CUDA-enabled FP32 build")
    if metrics.get("strict_cuda_marker") is not True:
        contract_problems.append("worker strict CUDA marker is absent")
    if contract_problems:
        raise RuntimeError(f"{label} runtime contract failed: " + "; ".join(contract_problems))
    return {
        "command": command,
        "disable_reduction": disable_reduction,
        "metrics": metrics,
        "stdout": file_record(stdout_path, pathlib.Path(__file__).resolve().parent.parent),
        "stderr": file_record(stderr_path, pathlib.Path(__file__).resolve().parent.parent),
    }


def controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    complete_path = args.output / "COMPLETE"
    report_path.unlink(missing_ok=True)
    complete_path.unlink(missing_ok=True)
    repo = pathlib.Path(__file__).resolve().parent.parent
    source_before = source_snapshot(repo)
    import meep as mp

    extension = pathlib.Path(mp._meep.__file__).resolve()
    libmeep = loaded_library_path("libmeep.so")
    meep_module = pathlib.Path(mp.__file__).resolve()
    python = pathlib.Path(sys.executable).resolve()
    build_root = extension.parents[2]
    receipt_path = build_root / "build-provenance.json"
    try:
        receipt = verify_build_receipt(receipt_path, repo)
    except ProvenanceError as error:
        raise RuntimeError(f"DFT evidence requires a valid build receipt: {error}") from error
    if receipt.get("build_kind") != "cuda-python-fp32":
        raise RuntimeError("DFT evidence build receipt has the wrong build_kind")
    receipt_artifacts = receipt.get("artifacts", {})
    for name, actual in (("python_extension", extension), ("libmeep", libmeep)):
        record = receipt_artifacts.get(name, {})
        recorded_path = pathlib.Path(str(record.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = repo / recorded_path
        if recorded_path.resolve() != actual or record.get("sha256") != sha256_file(actual):
            raise RuntimeError(f"DFT evidence receipt {name} does not match runtime")
    expected_runtime = {
        "extension": {"path": str(extension), "sha256": sha256_file(extension)},
        "libmeep": {"path": str(libmeep), "sha256": sha256_file(libmeep)},
        "meep_module": {
            "path": str(meep_module),
            "sha256": sha256_file(meep_module),
        },
        "python": {"path": str(python), "sha256": sha256_file(python)},
    }
    enabled = run_worker(
        args,
        False,
        receipt_id=receipt["receipt_id"],
        expected_runtime=expected_runtime,
    )
    disabled = run_worker(
        args,
        True,
        receipt_id=receipt["receipt_id"],
        expected_runtime=expected_runtime,
    )
    enabled_metrics = enabled["metrics"]
    disabled_metrics = disabled["metrics"]

    exact_metrics = (
        "active_backend",
        "cpu_phase_calls",
        "cuda_curl_calls",
        "cuda_dft_calls",
        "round_time",
        "timestep",
    )
    exact_comparisons = {
        name: {
            "resident_reduction": enabled_metrics[name],
            "full_sync": disabled_metrics[name],
            "pass": enabled_metrics[name] == disabled_metrics[name],
        }
        for name in exact_metrics
    }
    observable_names = (
        "dft_imag_sum",
        "dft_l2",
        "dft_real_sum",
        "field_sample_imag",
        "field_sample_real",
    )
    observable_comparisons = {}
    for name in observable_names:
        resident = float(enabled_metrics[name])
        full_sync = float(disabled_metrics[name])
        absolute_error = abs(resident - full_sync)
        allowed_error = 1e-10 + 1e-7 * max(
            abs(resident), abs(full_sync)
        )
        observable_comparisons[name] = {
            "resident_reduction": resident,
            "full_sync": full_sync,
            "absolute_error": absolute_error,
            "allowed_error": allowed_error,
            "pass": absolute_error <= allowed_error,
        }
    passed = all(
        item["pass"]
        for item in (
            *exact_comparisons.values(),
            *observable_comparisons.values(),
        )
    )
    source_after = source_snapshot(repo)
    receipt_after = verify_build_receipt(receipt_path, repo)
    provenance_unchanged = (
        source_before == source_after and receipt_after == receipt
    )
    passed = passed and provenance_unchanged
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "gate": {
            "pass": passed,
            "provenance_unchanged": provenance_unchanged,
        },
        "provenance": {
            "build_receipt": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "receipt_id": receipt["receipt_id"],
                "build_input_id": receipt["build_input_id"],
                "artifact_set_id": receipt["artifact_set_id"],
            },
            "runtime": expected_runtime,
            "source_before": source_before,
            "source_after": source_after,
            "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "runs": {
            "resident_reduction": enabled,
            "full_sync": disabled,
        },
        "exact_comparisons": exact_comparisons,
        "observable_comparisons": observable_comparisons,
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        complete_path,
        {
            "schema_version": 1,
            "state": "COMPLETE",
            "gate_pass": passed,
            "report": file_record(report_path, repo),
        },
    )
    print(f"DFT norm decay report: {report_path}")
    print(f"DFT norm decay gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker()
    return controller(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
