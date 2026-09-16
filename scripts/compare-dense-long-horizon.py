#!/usr/bin/env python3
"""Strictly compare full-array CPU-FP64, CPU-FP32, and CUDA-FP32 evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib
import re
import stat
import tempfile
from typing import BinaryIO


MAGIC = "GPMEEP_DENSE_LONG_HORIZON_EVIDENCE_V1"
EXPECTED_TIMESTEPS = 1024
EXPECTED_GRID = (2.5, 2.0, 16.0)
EXPECTED_SOURCE_FREQUENCY = 0.27
EXPECTED_SOURCES = (
    (4, 0.83, 0.91, 0.8, -0.31),
    (0, 1.57, 1.19, -0.23, 0.17),
)
REQUIRED_COMPONENTS = (0, 1, 4, 5, 6, 9, 10, 11, 14, 15, 16, 19)
EXPECTED_VALUES_PER_CHUNK = (64, 64, 152, 64, 64, 152, 216, 216, 513)
EXPECTED_KEYS = tuple(
    (chunk, component, part)
    for chunk in range(len(EXPECTED_VALUES_PER_CHUNK))
    for component in REQUIRED_COMPONENTS
    for part in (0, 1)
)
EXPECTED_ARRAY_COUNT = len(EXPECTED_KEYS)
EXPECTED_SCALAR_COUNT = (
    sum(EXPECTED_VALUES_PER_CHUNK) * len(REQUIRED_COMPONENTS) * 2
)
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 4096
INTEGER_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")

LIMITS = {
    "cpu_fp32_vs_cuda_fp32": {
        "linf_absolute": 5e-5,
        "linf_relative": 1e-6,
        "l2_relative": 8e-6,
        "energy_relative": 4e-6,
    },
    "cpu_fp64_vs_cpu_fp32": {
        "linf_absolute": 5e-3,
        "linf_relative": 2e-4,
        "l2_relative": 2e-4,
        "energy_relative": 2e-4,
    },
    "cpu_fp64_vs_cuda_fp32": {
        "linf_absolute": 5e-3,
        "linf_relative": 2e-4,
        "l2_relative": 2e-4,
        "energy_relative": 2e-4,
    },
}


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    return _stable_file_snapshot(path, capture=False)[1]


def _stable_file_snapshot(
    path_value: os.PathLike[str] | str,
    *,
    capture: bool,
    maximum_bytes: int | None = None,
) -> tuple[pathlib.Path, str, bytes | None]:
    """Read/hash one regular file through a stable O_NOFOLLOW descriptor."""

    provided = pathlib.Path(path_value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceError("this verifier requires O_NOFOLLOW support")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(provided, flags)
    except OSError as error:
        raise EvidenceError(
            f"{provided}: cannot securely open regular non-symlink file"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"{provided}: is not a regular file")
        if before.st_size <= 0:
            raise EvidenceError(f"{provided}: file is empty")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise EvidenceError(f"{provided}: file exceeds the allowed size")
        proc_link = pathlib.Path(f"/proc/self/fd/{descriptor}")
        link_text = os.readlink(proc_link)
        if link_text.endswith(" (deleted)"):
            raise EvidenceError(f"{provided}: file was deleted while being inspected")
        canonical = pathlib.Path(link_text)
        if not canonical.is_absolute():
            raise EvidenceError(f"{provided}: descriptor path is not absolute")

        digest = hashlib.sha256()
        captured = bytearray() if capture else None
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if maximum_bytes is not None and total > maximum_bytes:
                raise EvidenceError(f"{provided}: file exceeds the allowed size")
            digest.update(block)
            if captured is not None:
                captured.extend(block)

        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise EvidenceError(f"{provided}: file changed while being inspected")
        try:
            current = os.stat(canonical, follow_symlinks=False)
        except OSError as error:
            raise EvidenceError(f"{provided}: file disappeared while being inspected") from error
        if any(getattr(after, field) != getattr(current, field) for field in stable_fields):
            raise EvidenceError(f"{provided}: path identity changed while being inspected")
        return canonical, digest.hexdigest(), bytes(captured) if captured is not None else None
    finally:
        os.close(descriptor)


def _line(stream: BinaryIO, path: pathlib.Path, line_number: list[int]) -> str:
    raw = stream.readline(MAX_LINE_BYTES + 1)
    line_number[0] += 1
    if not raw:
        raise EvidenceError(f"{path}: unexpected EOF at line {line_number[0]}")
    if len(raw) > MAX_LINE_BYTES:
        raise EvidenceError(f"{path}: overlong line {line_number[0]}")
    if not raw.endswith(b"\n"):
        raise EvidenceError(f"{path}: unterminated line {line_number[0]}")
    try:
        text = raw[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"{path}: non-ASCII data at line {line_number[0]}"
        ) from error
    if not text or "\r" in text or "\x00" in text:
        raise EvidenceError(f"{path}: invalid line {line_number[0]}")
    return text


def _named(
    stream: BinaryIO, path: pathlib.Path, line_number: list[int], name: str
) -> str:
    text = _line(stream, path, line_number)
    prefix = name + "="
    if not text.startswith(prefix) or text.count("=") != 1:
        raise EvidenceError(
            f"{path}: expected unique key {name!r} at line {line_number[0]}"
        )
    value = text[len(prefix) :]
    if not value:
        raise EvidenceError(f"{path}: empty {name!r} value")
    return value


def _integer(value: str, path: pathlib.Path, name: str) -> int:
    if not INTEGER_RE.fullmatch(value):
        raise EvidenceError(f"{path}: {name} is not a canonical integer")
    return int(value)


def _finite_hex(value: str, path: pathlib.Path, name: str) -> float:
    # Hexadecimal input avoids locale and decimal-rounding ambiguity.  Requiring
    # the 0x marker also rejects Python spellings for NaN and infinity.
    normalized = value.lower()
    if not (
        normalized.startswith("0x")
        or normalized.startswith("-0x")
        or normalized.startswith("+0x")
    ):
        raise EvidenceError(f"{path}: {name} is not a hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise EvidenceError(f"{path}: {name} is not a valid float") from error
    if not math.isfinite(parsed):
        raise EvidenceError(f"{path}: {name} is non-finite")
    return parsed


def _runtime_artifact(value: str, path: pathlib.Path, name: str) -> dict[str, str]:
    if not value.startswith("/"):
        raise EvidenceError(f"{path}: {name} path is not absolute")
    canonical, digest, _ = _stable_file_snapshot(value, capture=False)
    if str(canonical) != value:
        raise EvidenceError(f"{path}: {name} path is not a canonical regular file")
    return {"path": str(canonical), "sha256": digest}


def parse_evidence(path_value: os.PathLike[str] | str) -> dict:
    path, evidence_sha256, payload = _stable_file_snapshot(
        path_value, capture=True, maximum_bytes=MAX_EVIDENCE_BYTES
    )
    assert payload is not None

    line_number = [0]
    with io.BytesIO(payload) as stream:
        if _line(stream, path, line_number) != MAGIC:
            raise EvidenceError(f"{path}: invalid evidence magic")
        lane = _named(stream, path, line_number, "lane")
        precision_bits = _integer(
            _named(stream, path, line_number, "precision_bits"),
            path,
            "precision_bits",
        )
        backend = _named(stream, path, line_number, "backend")
        runtime = {
            "executable": _runtime_artifact(
                _named(stream, path, line_number, "executable_path"),
                path,
                "executable",
            ),
            "libmeep": _runtime_artifact(
                _named(stream, path, line_number, "libmeep_path"),
                path,
                "loaded libmeep",
            ),
        }
        timesteps = _integer(
            _named(stream, path, line_number, "timesteps"), path, "timesteps"
        )
        grid = tuple(
            _finite_hex(_named(stream, path, line_number, key), path, key)
            for key in ("grid_sx", "grid_sy", "resolution")
        )
        source_count = _integer(
            _named(stream, path, line_number, "source_count"),
            path,
            "source_count",
        )
        source_frequency = _finite_hex(
            _named(stream, path, line_number, "source_frequency"),
            path,
            "source_frequency",
        )
        sources = tuple(
            (
                _integer(
                    _named(
                        stream,
                        path,
                        line_number,
                        f"source{source_index}_component",
                    ),
                    path,
                    f"source{source_index}_component",
                ),
                *(
                    _finite_hex(
                        _named(
                            stream,
                            path,
                            line_number,
                            f"source{source_index}_{field}",
                        ),
                        path,
                        f"source{source_index}_{field}",
                    )
                    for field in ("x", "y", "real", "imag")
                ),
            )
            for source_index in range(source_count)
        )
        array_count = _integer(
            _named(stream, path, line_number, "array_count"), path, "array_count"
        )
        declared_scalar_count = _integer(
            _named(stream, path, line_number, "scalar_count"),
            path,
            "scalar_count",
        )
        statistics = {
            key: _integer(_named(stream, path, line_number, key), path, key)
            for key in (
                "cpu_curl_calls",
                "cuda_curl_calls",
                "cpu_update_eh_calls",
                "cuda_update_eh_calls",
            )
        }
        energy = _finite_hex(
            _named(stream, path, line_number, "energy"), path, "energy"
        )

        if lane not in {"cpu-fp64", "cpu-fp32", "cuda-fp32"}:
            raise EvidenceError(f"{path}: unknown lane {lane!r}")
        expected_identity = {
            "cpu-fp64": (64, "cpu"),
            "cpu-fp32": (32, "cpu"),
            "cuda-fp32": (32, "cuda"),
        }[lane]
        if (precision_bits, backend) != expected_identity:
            raise EvidenceError(f"{path}: lane/precision/backend identity mismatch")
        if timesteps != EXPECTED_TIMESTEPS or grid != EXPECTED_GRID:
            raise EvidenceError(f"{path}: workload identity mismatch")
        if (
            source_count != len(EXPECTED_SOURCES)
            or source_frequency != EXPECTED_SOURCE_FREQUENCY
            or sources != EXPECTED_SOURCES
        ):
            raise EvidenceError(f"{path}: source identity mismatch")
        if (
            array_count != EXPECTED_ARRAY_COUNT
            or declared_scalar_count != EXPECTED_SCALAR_COUNT
        ):
            raise EvidenceError(f"{path}: array or scalar count is incomplete")
        if energy <= 0.0:
            raise EvidenceError(f"{path}: energy must be positive")

        arrays: dict[tuple[int, int, int], tuple[float, ...]] = {}
        previous_key: tuple[int, int, int] | None = None
        scalar_count = 0
        for _ in range(array_count):
            metadata = _named(stream, path, line_number, "array").split(",")
            if len(metadata) != 4:
                raise EvidenceError(f"{path}: invalid array metadata")
            values = tuple(
                _integer(item, path, "array metadata") for item in metadata
            )
            key = values[:3]
            count = values[3]
            if previous_key is not None and key <= previous_key:
                raise EvidenceError(f"{path}: duplicate or noncanonical array key")
            if (
                key not in EXPECTED_KEYS
                or count != EXPECTED_VALUES_PER_CHUNK[key[0]]
            ):
                raise EvidenceError(f"{path}: unexpected array key or count")
            previous_key = key
            array_values = tuple(
                _finite_hex(
                    _named(stream, path, line_number, "value"),
                    path,
                    f"field value for {key}",
                )
                for _ in range(count)
            )
            arrays[key] = array_values
            scalar_count += count

        if tuple(arrays) != EXPECTED_KEYS or scalar_count != declared_scalar_count:
            raise EvidenceError(f"{path}: full-array topology is incomplete")
        if _named(stream, path, line_number, "end") != "1":
            raise EvidenceError(f"{path}: invalid evidence footer")
        if stream.read(1):
            raise EvidenceError(f"{path}: trailing data after evidence footer")

    cpu_expected = backend == "cpu"
    if cpu_expected:
        expected_cpu_statistics = {
            "cpu_curl_calls": 344064,
            "cuda_curl_calls": 0,
            "cpu_update_eh_calls": 79872,
            "cuda_update_eh_calls": 0,
        }
        if statistics != expected_cpu_statistics:
            raise EvidenceError(f"{path}: CPU lane dispatch attestation failed")
    else:
        expected_cuda_statistics = {
            "cpu_curl_calls": 0,
            "cuda_curl_calls": 2048,
            "cpu_update_eh_calls": 0,
            "cuda_update_eh_calls": 2048,
        }
        if statistics != expected_cuda_statistics:
            raise EvidenceError(f"{path}: CUDA lane dispatch attestation failed")

    return {
        "path": str(path),
        "sha256": evidence_sha256,
        "lane": lane,
        "precision_bits": precision_bits,
        "backend": backend,
        "timesteps": timesteps,
        "grid": grid,
        "source_frequency": source_frequency,
        "sources": sources,
        "array_count": array_count,
        "scalar_count": scalar_count,
        "statistics": statistics,
        "runtime": runtime,
        "energy": energy,
        "arrays": arrays,
    }


def compare_arrays(reference: dict, candidate: dict) -> dict[str, float | int]:
    if tuple(reference["arrays"]) != tuple(candidate["arrays"]):
        raise EvidenceError("comparison array keys differ")
    if reference["scalar_count"] != candidate["scalar_count"]:
        raise EvidenceError("comparison scalar counts differ")

    errors: list[float] = []
    reference_values: list[float] = []
    for key in EXPECTED_KEYS:
        expected = reference["arrays"][key]
        actual = candidate["arrays"][key]
        if len(expected) != len(actual):
            raise EvidenceError(f"comparison array count differs for {key}")
        for expected_value, actual_value in zip(expected, actual, strict=True):
            error = abs(expected_value - actual_value)
            if not math.isfinite(error):
                raise EvidenceError("comparison produced a non-finite error")
            errors.append(error)
            reference_values.append(expected_value)

    reference_linf = max(abs(value) for value in reference_values)
    reference_l2_squared = math.fsum(value * value for value in reference_values)
    if reference_linf <= 0.0 or reference_l2_squared <= 0.0:
        raise EvidenceError("comparison reference field is identically zero")
    squared_error = math.fsum(error * error for error in errors)
    energy_scale = abs(reference["energy"])
    if energy_scale <= 0.0:
        raise EvidenceError("comparison reference energy is zero")
    metrics: dict[str, float | int] = {
        "scalar_count": len(errors),
        "linf_absolute": max(errors),
        "linf_relative": max(errors) / reference_linf,
        "l2_relative": math.sqrt(squared_error / reference_l2_squared),
        "energy_relative": abs(reference["energy"] - candidate["energy"])
        / energy_scale,
    }
    if not all(
        isinstance(value, int) or math.isfinite(value) for value in metrics.values()
    ):
        raise EvidenceError("comparison produced a non-finite norm")
    return metrics


def compare_evidence(
    fp64_cpu_path: os.PathLike[str] | str,
    fp32_cpu_path: os.PathLike[str] | str,
    fp32_cuda_path: os.PathLike[str] | str,
) -> dict:
    evidence = {
        "cpu-fp64": parse_evidence(fp64_cpu_path),
        "cpu-fp32": parse_evidence(fp32_cpu_path),
        "cuda-fp32": parse_evidence(fp32_cuda_path),
    }
    for expected_lane, record in evidence.items():
        if record["lane"] != expected_lane:
            raise EvidenceError(
                f"expected {expected_lane} evidence, got {record['lane']}"
            )
    for artifact in ("executable", "libmeep"):
        identities = {
            (
                record["runtime"][artifact]["path"],
                record["runtime"][artifact]["sha256"],
            )
            for record in evidence.values()
        }
        if len(identities) != len(evidence):
            raise EvidenceError(
                f"three precision/backend lanes do not bind distinct {artifact} artifacts"
            )

    pairs = {
        "cpu_fp32_vs_cuda_fp32": ("cpu-fp32", "cuda-fp32"),
        "cpu_fp64_vs_cpu_fp32": ("cpu-fp64", "cpu-fp32"),
        "cpu_fp64_vs_cuda_fp32": ("cpu-fp64", "cuda-fp32"),
    }
    comparisons = {}
    for name, (reference_lane, candidate_lane) in pairs.items():
        metrics = compare_arrays(
            evidence[reference_lane], evidence[candidate_lane]
        )
        limits = LIMITS[name]
        failed = [key for key, limit in limits.items() if metrics[key] > limit]
        if failed:
            detail = ", ".join(
                f"{key}={metrics[key]:.9g}>{limits[key]:.9g}" for key in failed
            )
            raise EvidenceError(f"{name} exceeded numerical gate: {detail}")
        comparisons[name] = {
            "reference_lane": reference_lane,
            "candidate_lane": candidate_lane,
            "metrics": metrics,
            "limits": limits,
            "pass": True,
        }

    public_inputs = {
        lane: {key: value for key, value in record.items() if key != "arrays"}
        for lane, record in evidence.items()
    }
    return {
        "schema_version": 1,
        "state": "COMPLETE",
        "gate": {"pass": True},
        "workload": {
            "timesteps": EXPECTED_TIMESTEPS,
            "grid": list(EXPECTED_GRID),
            "source_frequency": EXPECTED_SOURCE_FREQUENCY,
            "sources": [list(source) for source in EXPECTED_SOURCES],
            "array_count": EXPECTED_ARRAY_COUNT,
            "values_per_chunk_array": list(EXPECTED_VALUES_PER_CHUNK),
            "scalar_count": EXPECTED_SCALAR_COUNT,
            "coverage": "all real/imaginary Ex/Ey/Ez/Hx/Hy/Hz/Dx/Dy/Dz/Bx/By/Bz arrays",
        },
        "inputs": public_inputs,
        "comparisons": comparisons,
        "claims": {
            "cpu_fp64_reference": True,
            "cpu_fp32_candidate": True,
            "cuda_fp32_candidate": True,
            "cuda_fp64": False,
        },
    }


def atomic_write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp64-cpu", type=pathlib.Path, required=True)
    parser.add_argument("--fp32-cpu", type=pathlib.Path, required=True)
    parser.add_argument("--fp32-cuda", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    report = compare_evidence(
        arguments.fp64_cpu, arguments.fp32_cpu, arguments.fp32_cuda
    )
    if arguments.output:
        atomic_write_json(arguments.output.resolve(), report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
