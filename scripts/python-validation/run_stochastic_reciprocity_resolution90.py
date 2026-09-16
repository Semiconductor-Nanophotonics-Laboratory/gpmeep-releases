#!/usr/bin/env python3
"""Run the hash-bound resolution-90 stochastic reciprocity profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

import run_example_oracle


REFERENCE_SCHEMA = "gpmeep-reciprocity-resolution-reference-v1"
METRICS_PREFIX = "gpmeep-generic-example-metrics:"
RESOLUTION = 90
FREQUENCIES = 4
POSITION = -0.25
FLAT_RUNTIME = 32
TEXTURED_RUNTIME = 128
EXPECTED_TIMESTEPS = (
    124200,
    469800,
    124200,
    469800,
    239400,
    930600,
    239400,
    930600,
)


RESULT_SHAPES = {
    "reciprocity_resolution_probe_resolution": "1",
    "reciprocity_resolution_probe_dipole_count": "1",
    "reciprocity_resolution_probe_position": "1",
    "reciprocity_resolution_probe_source_index": "1",
    "reciprocity_resolution_probe_metadata_index": "4",
    "reciprocity_resolution_probe_frequencies": "4",
    "reciprocity_resolution_probe_forward_flat_flux": "2,4",
    "reciprocity_resolution_probe_forward_textured_flux": "2,4",
    "reciprocity_resolution_probe_backward_flat_local_power": "2,4",
    "reciprocity_resolution_probe_backward_textured_local_power": "2,4",
    "reciprocity_resolution_probe_forward_normalized": "2,4",
    "reciprocity_resolution_probe_backward_normalized": "2,4",
    "reciprocity_resolution_probe_raw_time_base": "4,4",
    "reciprocity_resolution_probe_raw_time_doubled": "4,4",
    "reciprocity_resolution_probe_raw_time_residual": "4,4",
    "reciprocity_resolution_probe_raw_time_relative_l2": "4",
    "reciprocity_resolution_probe_raw_time_pointwise": "4,4",
    "reciprocity_resolution_probe_raw_time_scale_aware": "4,4",
    "reciprocity_resolution_probe_normalized_time_residual": "2,4",
    "reciprocity_resolution_probe_normalized_time_relative_l2": "2",
    "reciprocity_resolution_probe_normalized_time_pointwise": "2,4",
    "reciprocity_resolution_probe_normalized_time_scale_aware": "2,4",
    "reciprocity_resolution_probe_base_closure_residual": "4",
    "reciprocity_resolution_probe_base_closure_relative_l2": "1",
    "reciprocity_resolution_probe_doubled_closure_residual": "4",
    "reciprocity_resolution_probe_doubled_closure_pointwise": "4",
    "reciprocity_resolution_probe_doubled_closure_scale_aware": "4",
    "reciprocity_resolution_probe_doubled_closure_relative_l2": "1",
    "reciprocity_resolution_probe_reference_res50_relative_l2": "1",
    "reciprocity_resolution_probe_refinement_ratio": "1",
    "reciprocity_resolution_probe_metadata_position_error": "4",
    "reciprocity_resolution_probe_metadata_local_weight": "4",
    "reciprocity_resolution_probe_metadata_return_recompute_relative_l2": "4",
    "reciprocity_resolution_probe_metadata_canonical_recompute_relative_l2": "4",
    "reciprocity_resolution_probe_metadata_ghost_weight_max_abs": "4",
    "reciprocity_resolution_probe_metadata_ghost_fraction": "4,4",
}


MINIMUMS = {
    "reciprocity_resolution_probe_frequencies": 1.0,
    "reciprocity_resolution_probe_forward_flat_flux": 1e-10,
    "reciprocity_resolution_probe_forward_textured_flux": 1e-10,
    "reciprocity_resolution_probe_backward_flat_local_power": 1e-10,
    "reciprocity_resolution_probe_backward_textured_local_power": 1e-10,
    "reciprocity_resolution_probe_forward_normalized": 1e-3,
    "reciprocity_resolution_probe_backward_normalized": 1e-3,
    "reciprocity_resolution_probe_metadata_local_weight": 1e-3,
    "reciprocity_resolution_probe_reference_res50_relative_l2": 1e-3,
}


MAXIMUMS = {
    "reciprocity_resolution_probe_raw_time_relative_l2": 0.1,
    "reciprocity_resolution_probe_raw_time_scale_aware": 0.05,
    "reciprocity_resolution_probe_normalized_time_relative_l2": 0.02,
    "reciprocity_resolution_probe_normalized_time_scale_aware": 0.05,
    "reciprocity_resolution_probe_doubled_closure_scale_aware": 0.05,
    "reciprocity_resolution_probe_doubled_closure_relative_l2": 0.03,
    "reciprocity_resolution_probe_refinement_ratio": 0.8,
    "reciprocity_resolution_probe_metadata_position_error": 1e-12,
    "reciprocity_resolution_probe_metadata_return_recompute_relative_l2": 1e-12,
    "reciprocity_resolution_probe_metadata_canonical_recompute_relative_l2": 5e-8,
    "reciprocity_resolution_probe_metadata_ghost_weight_max_abs": 1e-14,
    "reciprocity_resolution_probe_metadata_ghost_fraction": 1e-12,
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metric_array(results: dict[str, Any], name: str) -> np.ndarray:
    value = results.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"resolution reference lacks {name}")
    shape = value.get("shape")
    real = value.get("real")
    if not isinstance(shape, list) or not isinstance(real, list):
        raise RuntimeError(f"resolution reference has invalid {name}")
    array = np.asarray(real, dtype=float)
    try:
        array = array.reshape(tuple(int(item) for item in shape))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"resolution reference has invalid {name} shape"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"resolution reference {name} is not finite")
    return array


def load_reference(path: pathlib.Path) -> tuple[float, dict[str, Any]]:
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read resolution reference {path}") from exc
    if not isinstance(reference, dict) or reference.get("schema") != REFERENCE_SCHEMA:
        raise RuntimeError("unsupported resolution reference schema")
    source = reference.get("source")
    if not isinstance(source, dict) or source.get("backend") != "cuda":
        raise RuntimeError("resolution reference is not CUDA evidence")
    source_path_value = source.get("path")
    source_sha256 = source.get("sha256")
    if not isinstance(source_path_value, str) or not isinstance(
        source_sha256, str
    ):
        raise RuntimeError("resolution reference source binding is invalid")
    source_path = pathlib.Path(source_path_value).resolve()
    if not source_path.is_file() or sha256_file(source_path) != source_sha256:
        raise RuntimeError("resolution reference source hash does not match")
    if (
        reference.get("resolution") != 50
        or reference.get("dipole_count") != 55
        or reference.get("source_index") != 15
        or reference.get("metadata_index") != 16
        or not np.isclose(
            float(reference.get("position", np.nan)),
            POSITION,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError("resolution reference has the wrong grid point")

    payloads = [
        line[len(METRICS_PREFIX) :]
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(METRICS_PREFIX)
    ]
    if len(payloads) != 1:
        raise RuntimeError("resolution reference source has invalid metrics")
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("resolution reference metrics are invalid") from exc
    results = payload.get("example_results")
    if not isinstance(results, dict):
        raise RuntimeError("resolution reference lacks example results")

    forward_textured = _metric_array(
        results, "reciprocity_forward_textured_unique_doubled_flux"
    )[:, 15]
    forward_flat = _metric_array(
        results, "reciprocity_forward_doubled_summed_flux"
    )[0] / 55
    backward_raw = _metric_array(
        results, "reciprocity_backward_metadata_raw_power"
    )
    metadata_x = _metric_array(results, "reciprocity_backward_metadata_x")
    if not np.allclose(
        metadata_x[[2, 3], 16], POSITION, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("resolution reference metadata point is wrong")
    forward = forward_textured / forward_flat
    backward = backward_raw[3, :, 16] / backward_raw[2, :, 16]
    residual = forward - backward
    relative_l2 = float(np.linalg.norm(residual) / np.linalg.norm(backward))
    if not np.isfinite(relative_l2) or relative_l2 <= 0:
        raise RuntimeError("resolution reference closure is invalid")

    for key, recomputed in (
        ("forward_normalized_doubled", forward),
        ("backward_normalized_doubled", backward),
        ("residual", residual),
    ):
        recorded = np.asarray(reference.get(key), dtype=float)
        if recorded.shape != (FREQUENCIES,) or not np.allclose(
            recorded, recomputed, rtol=0.0, atol=1e-14
        ):
            raise RuntimeError(f"resolution reference {key} does not recompute")
    if not np.isclose(
        float(reference.get("relative_l2", np.nan)),
        relative_l2,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("resolution reference relative L2 does not recompute")

    audit = {
        "schema": "gpmeep-resolution-reference-audit-v1",
        "reference_path": str(path.resolve()),
        "reference_sha256": sha256_file(path),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "resolution": 50,
        "position": POSITION,
        "relative_l2": relative_l2,
    }
    return relative_l2, audit


def build_oracle_argv(
    example: pathlib.Path, reference_relative_l2: float
) -> list[str]:
    wrapper = pathlib.Path(__file__).resolve().with_name(
        "stochastic_reciprocity_resolution_probe.py"
    )
    arguments = [
        "--expected-run-count",
        "8",
        "--expected-final-timestep",
        str(EXPECTED_TIMESTEPS[-1]),
    ]
    for timestep in EXPECTED_TIMESTEPS:
        arguments.extend(
            [
                "--expected-run-timestep-delta-range",
                f"{timestep}:{timestep}",
            ]
        )
    arguments.extend(["--min-each-dft-norm", "1e-8"])
    for name, shape in RESULT_SHAPES.items():
        arguments.extend(
            [
                "--result-vector",
                name,
                "--expected-result-shape",
                f"{name}={shape}",
            ]
        )
    for name, threshold in MINIMUMS.items():
        arguments.extend(["--min-result-l2", f"{name}={threshold}"])
    for name, threshold in MAXIMUMS.items():
        arguments.extend(["--max-result-abs", f"{name}={threshold}"])
    arguments.extend(
        [
            str(wrapper),
            "--",
            str(example.resolve()),
            "--resolution",
            str(RESOLUTION),
            "--frequencies",
            str(FREQUENCIES),
            "--position",
            str(POSITION),
            "--flat-runtime",
            str(FLAT_RUNTIME),
            "--textured-runtime",
            str(TEXTURED_RUNTIME),
            "--reference-res50-relative-l2",
            repr(reference_relative_l2),
        ]
    )
    return arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", required=True, type=pathlib.Path)
    parser.add_argument("example", type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.example.is_file() or args.example.suffix != ".py":
        parser.error(f"example is not a Python file: {args.example}")
    if not args.reference_json.is_file():
        parser.error(
            f"resolution reference is not a file: {args.reference_json}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference_relative_l2, audit = load_reference(args.reference_json)
    print(
        "gpmeep-resolution-reference-audit:"
        + json.dumps(audit, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return run_example_oracle.main(
        build_oracle_argv(args.example, reference_relative_l2)
    )


if __name__ == "__main__":
    raise SystemExit(main())
