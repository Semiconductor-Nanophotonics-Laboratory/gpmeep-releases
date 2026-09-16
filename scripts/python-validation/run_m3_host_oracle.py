#!/usr/bin/env python3
"""Run one of the four bounded, CPU-only Milestone 3 compatibility oracles."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import re
import runpy
import stat
import sys
import unittest
from typing import Any

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import archive_validation  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    atomic_write_json,
    atomic_write_text,
    file_record,
)


SCHEMA = "gpmeep-m3-host-oracle-v1"
COMPLETE_SCHEMA = "gpmeep-m3-host-oracle-complete-v1"
METRIC_PREFIX = "gpmeep-eps-fit-metrics:"
GENERIC_PREFIX = "gpmeep-generic-example-metrics:"
OUTPUT_LIMIT_BYTES = 64 * 1024**2
ORACLES = {
    "host-eps-fit-lorentzian": {
        "target": "python/examples/eps_fit_lorentzian.py",
        "driver": "python-validation-exact-case",
    },
    "host-mpb-regression-suite": {
        "target": "python/tests/test_mpb.py",
        "driver": "python-unittest-file",
    },
    "host-parallel-wvgs-mpb": {
        "target": "python/examples/parallel-wvgs-mpb.py",
        "driver": "runpy-finite-array-adapter",
    },
    "host-plot-radiation-pattern": {
        "target": "python/examples/plot_radiation_pattern_dipole.py",
        "driver": "render-from-sealed-producer-metrics",
    },
}


class BoundedText(io.TextIOBase):
    def __init__(self, label: str, limit_bytes: int = OUTPUT_LIMIT_BYTES):
        self.label = label
        self.limit_bytes = limit_bytes
        self.parts: list[str] = []
        self.size_bytes = 0

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("bounded text accepts strings only")
        encoded = value.encode("utf-8", errors="strict")
        if self.size_bytes + len(encoded) > self.limit_bytes:
            raise WorkloadError(f"{self.label} exceeded its output limit")
        self.parts.append(value)
        self.size_bytes += len(encoded)
        return len(value)

    def flush(self) -> None:
        return None

    def value(self) -> str:
        return "".join(self.parts)


@contextlib.contextmanager
def working_directory(path: pathlib.Path):
    previous = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def argv_context(values: list[str]):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    if path.is_symlink():
        raise WorkloadError("M3 host output is a symlink")
    path = pathlib.Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise WorkloadError("M3 host output parent contains a symlink")
    path.mkdir(exist_ok=False)
    return path


def _overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _target(repo: pathlib.Path, unit_id: str) -> pathlib.Path:
    expected = ORACLES[unit_id]["target"]
    path = repo.joinpath(*pathlib.PurePosixPath(expected).parts)
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise WorkloadError("M3 host target is not a regular non-symlink file")
    return path


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadError(f"M3 host metric JSON repeats key {key!r}")
        result[key] = value
    return result


def _prefixed_json(text: str, prefix: str, label: str) -> dict[str, Any]:
    lines = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1 or len(lines[0].encode("utf-8")) > 8 * 1024**2:
        raise WorkloadError(f"{label} metric record inventory differs")
    try:
        value = json.loads(
            lines[0],
            object_pairs_hook=_unique_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                WorkloadError(f"{label} contains nonfinite JSON {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise WorkloadError(f"{label} metric JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} metric payload is not an object")
    return value


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise WorkloadError(f"{label} contains a nonfinite value")
        return
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item, label)
        return
    raise WorkloadError(f"{label} contains an unsupported value")


def run_eps_fit(target: pathlib.Path, stdout: BoundedText) -> dict[str, Any]:
    with contextlib.redirect_stdout(stdout), argv_context(
        [str(target), "--validation"]
    ):
        runpy.run_path(str(target), run_name="__main__")
    metrics = _prefixed_json(stdout.value(), METRIC_PREFIX, "Lorentzian fit")
    validate_eps_result(metrics)
    return metrics


def validate_eps_result(metrics: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "frequency_count",
        "initial_error",
        "final_error",
        "improvement_ratio",
        "fit_relative_l2",
        "gradient_relative_error",
        "material_relative_l2",
        "parameter_relative_l2",
        "fitted_parameters",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_keys:
        raise WorkloadError("Lorentzian fit metric schema differs")
    _finite_tree(metrics, "Lorentzian fit")
    if (
        metrics["schema_version"] != 1
        or metrics["frequency_count"] != 96
        or metrics["improvement_ratio"] <= 1e12
        or metrics["fit_relative_l2"] > 1e-8
        or metrics["parameter_relative_l2"] > 1e-7
        or metrics["gradient_relative_error"] > 1e-7
        or metrics["material_relative_l2"] > 1e-12
        or not isinstance(metrics["fitted_parameters"], list)
        or len(metrics["fitted_parameters"]) != 6
    ):
        raise WorkloadError("Lorentzian fit oracle thresholds failed")
    return metrics


def run_mpb_suite(target: pathlib.Path, stdout: BoundedText) -> dict[str, Any]:
    module_name = "gpmeep_m3_test_mpb"
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise WorkloadError("could not load the MPB regression suite")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            result = unittest.TextTestRunner(stream=stdout, verbosity=2).run(suite)
    finally:
        sys.modules.pop(module_name, None)
    if (
        not result.wasSuccessful()
        or result.testsRun != 47
        or len(result.skipped) != 0
        or len(result.expectedFailures) != 0
        or len(result.unexpectedSuccesses) != 0
    ):
        raise WorkloadError("MPB full regression suite did not pass 47 tests")
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
    }


def _array(namespace: dict[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(namespace.get(name), dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise WorkloadError(f"parallel-waveguide array {name} is invalid")
    return value


def _array_record(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value, dtype="<f8")
    return {
        "shape": list(contiguous.shape),
        "l2": float(np.linalg.norm(contiguous)),
        "minimum": float(np.min(contiguous)),
        "maximum": float(np.max(contiguous)),
        "sha256_le_f64": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "values": contiguous.tolist(),
    }


def run_parallel_waveguides(
    target: pathlib.Path, output: pathlib.Path, stdout: BoundedText
) -> dict[str, Any]:
    with contextlib.redirect_stdout(stdout), working_directory(output):
        namespace = runpy.run_path(str(target), run_name="__main__")
    separation = _array(namespace, "ss", (21,))
    arrays = {
        name: _array(namespace, name, shape)
        for name, shape in (
            ("f_odd", (21,)),
            ("vg_odd", (21,)),
            ("f_even", (21,)),
            ("vg_even", (21,)),
            ("force_odd", (20,)),
            ("force_even", (20,)),
        )
    }
    if (
        not np.all(np.diff(separation) > 0)
        or not np.allclose(np.diff(separation), 0.05, rtol=0, atol=1e-12)
        or any(np.linalg.norm(value) <= 0 for value in arrays.values())
        or np.linalg.norm(arrays["f_odd"] - arrays["f_even"]) <= 1e-8
        or np.linalg.norm(arrays["force_odd"] - arrays["force_even"]) <= 1e-8
        or np.any(arrays["f_odd"] <= 0)
        or np.any(arrays["f_even"] <= 0)
        or np.any(np.abs(arrays["vg_odd"]) <= 1e-12)
        or np.any(np.abs(arrays["vg_even"]) <= 1e-12)
    ):
        raise WorkloadError("parallel-waveguide finite-array oracle failed")
    result = {
        "separation": _array_record(separation),
        **{name: _array_record(value) for name, value in arrays.items()},
    }
    validate_parallel_result(result)
    return result


def _validate_array_record(
    record: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    expected_keys = {
        "shape",
        "l2",
        "minimum",
        "maximum",
        "sha256_le_f64",
        "values",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise WorkloadError(f"{label} array record schema differs")
    try:
        value = np.asarray(record["values"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"{label} array values are invalid") from exc
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise WorkloadError(f"{label} array shape or values differ")
    if record != _array_record(value):
        raise WorkloadError(f"{label} array record was not re-derived")
    return value


def validate_parallel_result(result: Any) -> dict[str, Any]:
    expected = {
        "separation": (21,),
        "f_odd": (21,),
        "vg_odd": (21,),
        "f_even": (21,),
        "vg_even": (21,),
        "force_odd": (20,),
        "force_even": (20,),
    }
    if not isinstance(result, dict) or set(result) != set(expected):
        raise WorkloadError("parallel-waveguide result schema differs")
    arrays = {
        name: _validate_array_record(result[name], shape, f"parallel {name}")
        for name, shape in expected.items()
    }
    separation = arrays["separation"]
    if (
        not np.all(np.diff(separation) > 0)
        or not np.allclose(np.diff(separation), 0.05, rtol=0, atol=1e-12)
        or any(
            np.linalg.norm(value) <= 0
            for name, value in arrays.items()
            if name != "separation"
        )
        or np.linalg.norm(arrays["f_odd"] - arrays["f_even"]) <= 1e-8
        or np.linalg.norm(arrays["force_odd"] - arrays["force_even"]) <= 1e-8
        or np.any(arrays["f_odd"] <= 0)
        or np.any(arrays["f_even"] <= 0)
        or np.any(np.abs(arrays["vg_odd"]) <= 1e-12)
        or np.any(np.abs(arrays["vg_even"]) <= 1e-12)
    ):
        raise WorkloadError("parallel-waveguide retained oracle failed")
    return result


def _safe_member(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or "." in pure.parts
        or ".." in pure.parts
        or "\\" in relative
    ):
        raise WorkloadError(f"{label} path is unsafe")
    path = root.joinpath(*pure.parts)
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise WorkloadError(f"{label} is not a regular non-symlink file")
    return path


def _producer_metrics(
    archive: pathlib.Path,
    expected_manifest_sha256: str,
    expected_receipt_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete = archive_validation.verify_archive(
        archive,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_receipt_id=expected_receipt_id,
    )
    report_path = archive.joinpath(*archive_validation.RUNNER_ROOT.parts) / "report.json"
    with StableFile(report_path, "M3 plot producer report") as stable:
        with stable.file_object() as handle:
            report = json.load(handle, object_pairs_hook=_unique_json_pairs)
    selected = [item for item in report.get("results", []) if item.get("selected")]
    if (
        len(selected) != 1
        or selected[0].get("path") != "python/examples/dipole_in_vacuum_1D.py"
        or selected[0].get("outcome") != "PASS"
        or not selected[0]
        .get("runs", {})
        .get("cpu", {})
        .get("outcome", "")
        .startswith("PASS")
    ):
        raise WorkloadError("plot producer archive is not the CPU dipole PASS")
    stdout_relative = selected[0]["runs"]["cpu"]["stdout_log"]
    stdout_path = _safe_member(
        archive.joinpath(*archive_validation.RUNNER_ROOT.parts),
        stdout_relative,
        "plot producer stdout",
    )
    with StableFile(stdout_path, "M3 plot producer stdout") as stable:
        with stable.file_object() as handle:
            try:
                text = handle.read().decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise WorkloadError("plot producer stdout is not UTF-8") from exc
    metrics = _prefixed_json(text, GENERIC_PREFIX, "plot producer")
    return metrics, complete


def _result_array(
    metrics: dict[str, Any], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        record = metrics["example_results"][name]
        values = record["real"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(f"plot producer result is absent: {name}") from exc
    record_keys = {
        "count",
        "real",
        "imag",
        "l2",
        "max_abs",
        "weighted_real",
        "weighted_imag",
        "shape",
    }
    if (
        not isinstance(record, dict)
        or set(record) != record_keys
        or record.get("shape") != list(shape)
        or record.get("count") != math.prod(shape)
        or not isinstance(record.get("imag"), list)
        or len(record["imag"]) != math.prod(shape)
    ):
        raise WorkloadError(f"plot producer result shape differs: {name}")
    value = np.asarray(values, dtype=float)
    if value.shape != (math.prod(shape),) or not np.all(np.isfinite(value)):
        raise WorkloadError(f"plot producer result values differ: {name}")
    if any(not math.isfinite(float(item)) or float(item) != 0.0 for item in record["imag"]):
        raise WorkloadError(f"plot producer result is unexpectedly complex: {name}")
    return value.reshape(shape)


def run_plotter(
    target: pathlib.Path,
    output: pathlib.Path,
    stdout: BoundedText,
    producer_archive: pathlib.Path,
    expected_manifest_sha256: str,
    expected_receipt_id: str,
) -> dict[str, Any]:
    metrics, complete = _producer_metrics(
        producer_archive, expected_manifest_sha256, expected_receipt_id
    )
    radial_flux = _result_array(metrics, "dipole_1d_radial_flux", (7, 8))
    np.savez(
        output / "dipole_radiation_pattern.npz",
        NUM_POLAR=7,
        NUM_AZIMUTH=8,
        dipole_pol="x",
        radial_flux=radial_flux,
    )
    with contextlib.redirect_stdout(stdout), working_directory(output), argv_context(
        [str(target)]
    ):
        runpy.run_path(str(target), run_name="__main__")
    matches = re.findall(
        r"relative error in radiation pattern \(φ = 0\):,\s*([^\s]+)",
        stdout.value(),
    )
    if len(matches) != 1:
        raise WorkloadError("plotter did not report exactly one relative error")
    try:
        relative_error = float(matches[0])
    except ValueError as exc:
        raise WorkloadError("plotter relative error is invalid") from exc
    if not math.isfinite(relative_error) or relative_error < 0:
        raise WorkloadError("plotter relative error is nonfinite or negative")
    images = {}
    for name in (
        "dipole_radiation_pattern_phi0.png",
        "dipole_radiation_pattern_3D.png",
    ):
        path = output / name
        payload = path.read_bytes()
        if len(payload) < 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise WorkloadError(f"plotter image is not a nonempty PNG: {name}")
        images[name] = {
            "path": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    replayed = archive_validation.verify_archive(
        producer_archive,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_receipt_id=expected_receipt_id,
    )
    if replayed != complete:
        raise WorkloadError("plot producer archive changed during rendering")
    return {
        "producer_archive_manifest_sha256": complete["archive_manifest_sha256"],
        "producer_receipt_id": complete["build_receipt_id"],
        "radial_flux": _array_record(radial_flux),
        "relative_error": relative_error,
        "input_npz": file_record(
            output / "dipole_radiation_pattern.npz", output
        ),
        "images": images,
    }


def _recorded_file(
    root: pathlib.Path, record: Any, expected_path: str, label: str
) -> pathlib.Path:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "size_bytes", "sha256"}
        or record.get("path") != expected_path
        or type(record.get("size_bytes")) is not int
        or record["size_bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise WorkloadError(f"{label} file record differs")
    path = _safe_member(root, expected_path, label)
    with StableFile(path, label, expected=record):
        pass
    return path


def _load_stable_json(
    root: pathlib.Path, record: Any, expected_path: str, label: str
) -> dict[str, Any]:
    path = _recorded_file(root, record, expected_path, label)
    with StableFile(path, label, expected=record) as stable:
        with stable.file_object() as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_json_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    WorkloadError(f"{label} contains nonfinite JSON {token}")
                ),
            )
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _validate_mpb_result(result: Any, stdout: str) -> dict[str, Any]:
    expected = {
        "tests_run": 47,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "expected_failures": 0,
        "unexpected_successes": 0,
    }
    if result != expected:
        raise WorkloadError("MPB retained regression result differs")
    counts = re.findall(r"^Ran\s+(\d+)\s+tests?\s+in\s+.+$", stdout, re.MULTILINE)
    if counts != ["47"] or "OK" not in stdout:
        raise WorkloadError("MPB retained unittest output differs")
    return result


def _validate_plot_result(
    result: Any,
    output: pathlib.Path,
    stdout: str,
    producer_archive: pathlib.Path,
    expected_manifest_sha256: str,
    expected_receipt_id: str,
) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {
        "producer_archive_manifest_sha256",
        "producer_receipt_id",
        "radial_flux",
        "relative_error",
        "input_npz",
        "images",
    }:
        raise WorkloadError("plotter retained result schema differs")
    if (
        result["producer_archive_manifest_sha256"]
        != expected_manifest_sha256
        or result["producer_receipt_id"] != expected_receipt_id
    ):
        raise WorkloadError("plotter retained producer identity differs")
    radial = _validate_array_record(result["radial_flux"], (7, 8), "plot radial")
    relative_error = result["relative_error"]
    if (
        isinstance(relative_error, bool)
        or not isinstance(relative_error, (int, float))
        or not math.isfinite(float(relative_error))
        or relative_error < 0
    ):
        raise WorkloadError("plotter retained relative error differs")
    matches = re.findall(
        r"relative error in radiation pattern \(φ = 0\):,\s*([^\s]+)", stdout
    )
    if len(matches) != 1 or float(matches[0]) != relative_error:
        raise WorkloadError("plotter retained stdout relative error differs")
    npz_path = _recorded_file(
        output,
        result["input_npz"],
        "dipole_radiation_pattern.npz",
        "plotter input NPZ",
    )
    with StableFile(
        npz_path, "plotter input NPZ", expected=result["input_npz"]
    ) as stable:
        with stable.file_object() as handle, np.load(
            handle, allow_pickle=False
        ) as data:
            if set(data.files) != {
                "NUM_POLAR",
                "NUM_AZIMUTH",
                "dipole_pol",
                "radial_flux",
            }:
                raise WorkloadError("plotter input NPZ inventory differs")
            if (
                int(data["NUM_POLAR"]) != 7
                or int(data["NUM_AZIMUTH"]) != 8
                or str(data["dipole_pol"]) != "x"
                or not np.array_equal(np.asarray(data["radial_flux"]), radial)
            ):
                raise WorkloadError("plotter input NPZ content differs")
    image_names = {
        "dipole_radiation_pattern_phi0.png",
        "dipole_radiation_pattern_3D.png",
    }
    if not isinstance(result["images"], dict) or set(result["images"]) != image_names:
        raise WorkloadError("plotter retained image inventory differs")
    for name in image_names:
        path = _recorded_file(output, result["images"][name], name, f"plotter {name}")
        with StableFile(
            path, f"plotter {name}", expected=result["images"][name]
        ) as stable:
            with stable.file_object() as handle:
                payload = handle.read()
        if len(payload) < 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise WorkloadError(f"plotter retained image is not PNG: {name}")
    metrics, producer_complete = _producer_metrics(
        producer_archive, expected_manifest_sha256, expected_receipt_id
    )
    producer_radial = _result_array(
        metrics, "dipole_1d_radial_flux", (7, 8)
    )
    if (
        producer_complete["archive_manifest_sha256"]
        != expected_manifest_sha256
        or producer_complete["build_receipt_id"] != expected_receipt_id
        or not np.array_equal(producer_radial, radial)
    ):
        raise WorkloadError("plotter producer metrics were not re-derived")
    return result


def validate_complete(
    output: pathlib.Path,
    repo: pathlib.Path,
    unit_id: str,
    *,
    terminal_name: str = "COMPLETE",
    producer_archive: pathlib.Path | None = None,
    expected_manifest_sha256: str | None = None,
    expected_receipt_id: str | None = None,
) -> dict[str, Any]:
    if unit_id not in ORACLES:
        raise WorkloadError("M3 host unit ID differs")
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M3 host terminal filename differs")
    output = pathlib.Path(os.path.abspath(output))
    repo = pathlib.Path(os.path.abspath(repo))
    if output.resolve(strict=True) != output or repo.resolve(strict=True) != repo:
        raise WorkloadError("M3 host replay root contains a symlink")
    complete_path = _safe_member(output, terminal_name, "M3 host COMPLETE")
    with StableFile(complete_path, "M3 host COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle, object_pairs_hook=_unique_json_pairs)
    if (
        not isinstance(complete, dict)
        or set(complete) != {"schema", "outcome", "report"}
        or complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 host COMPLETE is not an exact PASS")
    report = _load_stable_json(
        output, complete["report"], "report.json", "M3 host report"
    )
    if (
        set(report)
        != {"schema", "outcome", "unit_id", "driver", "target", "adapter", "result", "stdout"}
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or report.get("unit_id") != unit_id
        or report.get("driver") != ORACLES[unit_id]["driver"]
    ):
        raise WorkloadError("M3 host report identity differs")
    target_relative = ORACLES[unit_id]["target"]
    _recorded_file(repo, report["target"], target_relative, "M3 host target")
    adapter_relative = pathlib.Path(__file__).resolve(strict=True).relative_to(repo).as_posix()
    _recorded_file(repo, report["adapter"], adapter_relative, "M3 host adapter")
    stdout_path = _recorded_file(
        output, report["stdout"], "stdout.log", "M3 host stdout"
    )
    with StableFile(
        stdout_path, "M3 host stdout", expected=report["stdout"]
    ) as stable:
        with stable.file_object() as handle:
            try:
                stdout = handle.read().decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise WorkloadError("M3 host stdout is not UTF-8") from exc
    if unit_id == "host-eps-fit-lorentzian":
        result = validate_eps_result(report["result"])
        if _prefixed_json(stdout, METRIC_PREFIX, "Lorentzian fit") != result:
            raise WorkloadError("Lorentzian retained stdout metrics differ")
    elif unit_id == "host-mpb-regression-suite":
        result = _validate_mpb_result(report["result"], stdout)
    elif unit_id == "host-parallel-wvgs-mpb":
        result = validate_parallel_result(report["result"])
    else:
        if (
            producer_archive is None
            or expected_manifest_sha256 is None
            or expected_receipt_id is None
        ):
            raise WorkloadError("plotter replay requires producer identities")
        result = _validate_plot_result(
            report["result"],
            output,
            stdout,
            producer_archive,
            expected_manifest_sha256,
            expected_receipt_id,
        )
    if complete != {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
    }:
        raise WorkloadError("M3 host COMPLETE payload differs")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--unit-id", required=True, choices=tuple(ORACLES))
    parser.add_argument("--producer-archive", type=pathlib.Path)
    parser.add_argument("--expected-producer-manifest-sha256")
    parser.add_argument("--expected-receipt-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve(strict=True)
    output_absolute = pathlib.Path(os.path.abspath(args.output))
    if _overlap(output_absolute, repo):
        raise WorkloadError("M3 host output overlaps the immutable repository")
    producer_values = (
        args.producer_archive,
        args.expected_producer_manifest_sha256,
        args.expected_receipt_id,
    )
    if args.unit_id == "host-plot-radiation-pattern":
        if any(value is None for value in producer_values):
            raise WorkloadError("plotter requires sealed producer identities")
    elif any(value is not None for value in producer_values):
        raise WorkloadError("non-plot host oracle received producer arguments")
    output = _safe_output(args.output)
    target = _target(repo, args.unit_id)
    target_stable = StableFile(target, f"M3 host target {args.unit_id}")
    adapter_stable = StableFile(pathlib.Path(__file__), "M3 host adapter")
    stdout = BoundedText("M3 host stdout")
    try:
        with contextlib.redirect_stderr(stdout):
            if args.unit_id == "host-eps-fit-lorentzian":
                result = run_eps_fit(target, stdout)
            elif args.unit_id == "host-mpb-regression-suite":
                result = run_mpb_suite(target, stdout)
            elif args.unit_id == "host-parallel-wvgs-mpb":
                result = run_parallel_waveguides(target, output, stdout)
            else:
                assert args.producer_archive is not None
                assert args.expected_producer_manifest_sha256 is not None
                assert args.expected_receipt_id is not None
                result = run_plotter(
                    target,
                    output,
                    stdout,
                    args.producer_archive.resolve(strict=True),
                    args.expected_producer_manifest_sha256,
                    args.expected_receipt_id,
                )
        target_stable.verify_unchanged()
        adapter_stable.verify_unchanged()
        atomic_write_text(output / "stdout.log", stdout.value())
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "unit_id": args.unit_id,
            "driver": ORACLES[args.unit_id]["driver"],
            "target": file_record(target, repo),
            "adapter": file_record(pathlib.Path(__file__), repo),
            "result": result,
            "stdout": file_record(output / "stdout.log", output),
        }
        atomic_write_json(output / "report.json", report)
        atomic_write_json(
            output / "PENDING_COMPLETE",
            {
                "schema": COMPLETE_SCHEMA,
                "outcome": "PASS",
                "report": file_record(output / "report.json", output),
            },
        )
        validate_complete(
            output,
            repo,
            args.unit_id,
            terminal_name="PENDING_COMPLETE",
            producer_archive=(
                args.producer_archive.resolve(strict=True)
                if args.producer_archive is not None
                else None
            ),
            expected_manifest_sha256=args.expected_producer_manifest_sha256,
            expected_receipt_id=args.expected_receipt_id,
        )
        os.replace(output / "PENDING_COMPLETE", output / "COMPLETE")
        return 0
    finally:
        target_stable.close(verify=False)
        adapter_stable.close(verify=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 host oracle error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
