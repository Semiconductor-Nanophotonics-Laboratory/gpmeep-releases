#!/usr/bin/env python3
"""Run the hash-bound resolution-90 textured temporal-extension profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

import run_example_oracle
import stochastic_reciprocity_matrix as reciprocity


V3_LOG_SHA256 = "55a690ccfe0b02c4557dc25a2c4e0b4cb7b3090211e08fd620f9f2f97895421b"
V3_FREEZE_SHA256 = "b28a2c02f56091d5aa419a8e4c1922fe5fd198ed773ef2b39e291289b547cc5a"
V3_EXPECTED_TIMESTEPS = (
    124200,
    469800,
    124200,
    469800,
    239400,
    930600,
    239400,
    930600,
)
EXPECTED_TIMESTEP = 1_852_200
DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-resolution-diagnostics:"


RESULT_SHAPES = {
    "reciprocity_temporal_extension_resolution": "1",
    "reciprocity_temporal_extension_dipole_count": "1",
    "reciprocity_temporal_extension_position": "1",
    "reciprocity_temporal_extension_source_index": "1",
    "reciprocity_temporal_extension_metadata_index": "1",
    "reciprocity_temporal_extension_frequencies": "4",
    "reciprocity_temporal_extension_reference_forward_flat": "4",
    "reciprocity_temporal_extension_reference_backward_flat": "4",
    "reciprocity_temporal_extension_reference_forward_textured": "4",
    "reciprocity_temporal_extension_reference_backward_textured": "4",
    "reciprocity_temporal_extension_forward_textured": "4",
    "reciprocity_temporal_extension_backward_textured": "4",
    "reciprocity_temporal_extension_raw_time_residual": "2,4",
    "reciprocity_temporal_extension_raw_time_relative_l2": "2",
    "reciprocity_temporal_extension_raw_time_pointwise": "2,4",
    "reciprocity_temporal_extension_raw_time_scale_aware": "2,4",
    "reciprocity_temporal_extension_reference_normalized": "2,4",
    "reciprocity_temporal_extension_extended_normalized": "2,4",
    "reciprocity_temporal_extension_normalized_time_residual": "2,4",
    "reciprocity_temporal_extension_normalized_time_relative_l2": "2",
    "reciprocity_temporal_extension_normalized_time_pointwise": "2,4",
    "reciprocity_temporal_extension_normalized_time_scale_aware": "2,4",
    "reciprocity_temporal_extension_normalized_time_symmetric": "2,4",
    "reciprocity_temporal_extension_closure_residual": "4",
    "reciprocity_temporal_extension_closure_relative_l2": "1",
    "reciprocity_temporal_extension_closure_pointwise": "4",
    "reciprocity_temporal_extension_closure_scale_aware": "4",
    "reciprocity_temporal_extension_closure_symmetric": "4",
    "reciprocity_temporal_extension_reference_res50_relative_l2": "1",
    "reciprocity_temporal_extension_refinement_ratio": "1",
    "reciprocity_temporal_extension_flat_condition_ratio": "2,4",
    "reciprocity_temporal_extension_metadata_position_error": "1",
    "reciprocity_temporal_extension_metadata_y_error": "1",
    "reciprocity_temporal_extension_metadata_z_error": "1",
    "reciprocity_temporal_extension_metadata_local_weight": "1",
    "reciprocity_temporal_extension_metadata_return_recompute_relative_l2": "1",
    "reciprocity_temporal_extension_metadata_canonical_recompute_relative_l2": "1",
    "reciprocity_temporal_extension_metadata_ghost_weight_max_abs": "1",
    "reciprocity_temporal_extension_metadata_ghost_fraction": "4",
}


MINIMUMS = {
    "reciprocity_temporal_extension_frequencies": 1.0,
    "reciprocity_temporal_extension_reference_forward_flat": 1e-10,
    "reciprocity_temporal_extension_reference_backward_flat": 1e-10,
    "reciprocity_temporal_extension_reference_forward_textured": 1e-10,
    "reciprocity_temporal_extension_reference_backward_textured": 1e-10,
    "reciprocity_temporal_extension_forward_textured": 1e-10,
    "reciprocity_temporal_extension_backward_textured": 1e-10,
    "reciprocity_temporal_extension_reference_normalized": 1e-3,
    "reciprocity_temporal_extension_extended_normalized": 1e-3,
    "reciprocity_temporal_extension_flat_condition_ratio": 1e-5,
    "reciprocity_temporal_extension_metadata_local_weight": 1e-3,
}


MAXIMUMS = {
    "reciprocity_temporal_extension_raw_time_relative_l2": 0.1,
    "reciprocity_temporal_extension_raw_time_scale_aware": 0.05,
    "reciprocity_temporal_extension_normalized_time_relative_l2": 0.02,
    "reciprocity_temporal_extension_normalized_time_scale_aware": 0.05,
    "reciprocity_temporal_extension_normalized_time_symmetric": 0.05,
    "reciprocity_temporal_extension_closure_relative_l2": 0.03,
    "reciprocity_temporal_extension_closure_scale_aware": 0.05,
    "reciprocity_temporal_extension_closure_symmetric": 0.05,
    "reciprocity_temporal_extension_refinement_ratio": 0.8,
    "reciprocity_temporal_extension_metadata_position_error": 1e-12,
    "reciprocity_temporal_extension_metadata_y_error": 1e-12,
    "reciprocity_temporal_extension_metadata_z_error": 1e-12,
    "reciprocity_temporal_extension_metadata_return_recompute_relative_l2": 1e-12,
    "reciprocity_temporal_extension_metadata_canonical_recompute_relative_l2": 5e-8,
    "reciprocity_temporal_extension_metadata_ghost_weight_max_abs": 1e-14,
    "reciprocity_temporal_extension_metadata_ghost_fraction": 1e-12,
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return reciprocity._finite_array(value, name, shape)


def _relative_l2_rows(residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(residual[row])
            / max(np.linalg.norm(reference[row]), np.finfo(float).tiny)
            for row in range(residual.shape[0])
        ]
    )


def _require_close(
    recorded: Any,
    recomputed: np.ndarray | float,
    name: str,
    *,
    atol: float = 1e-14,
) -> None:
    recorded_array = np.asarray(recorded, dtype=float)
    recomputed_array = np.asarray(recomputed, dtype=float)
    if recorded_array.shape != recomputed_array.shape or not np.allclose(
        recorded_array, recomputed_array, rtol=0.0, atol=atol
    ):
        raise RuntimeError(f"resolution-90 reference {name} does not recompute")


def load_reference(
    log_path: pathlib.Path, freeze_path: pathlib.Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    log_path = log_path.resolve()
    freeze_path = freeze_path.resolve()
    if not log_path.is_file() or sha256_file(log_path) != V3_LOG_SHA256:
        raise RuntimeError("resolution-90 v3 log hash does not match pinned evidence")
    if not freeze_path.is_file() or sha256_file(freeze_path) != V3_FREEZE_SHA256:
        raise RuntimeError(
            "resolution-90 v3 source-freeze hash does not match pinned evidence"
        )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("resolution-90 v3 source freeze is invalid") from exc
    expected_profile = {
        "resolution": 90,
        "period_pixels": 99,
        "position": -0.25,
        "source_index": 27,
        "expected_metadata_index": 28,
        "runs": 8,
        "expected_timestep_deltas": list(V3_EXPECTED_TIMESTEPS),
    }
    if (
        not isinstance(freeze, dict)
        or freeze.get("schema") != "gpmeep-development-source-freeze-v1"
        or freeze.get("git_head")
        != "b65efaa749ec04d76f469db0a408c22e003468b3"
        or freeze.get("release_eligible") is not False
        or freeze.get("profile") != expected_profile
    ):
        raise RuntimeError("resolution-90 v3 source freeze has the wrong profile")
    freeze_inputs = freeze.get("inputs")
    required_inputs = {
        "python/examples/stochastic_emitter_reciprocity.py",
        "scripts/python-validation/stochastic_reciprocity_resolution_probe.py",
        "scripts/python-validation/run_stochastic_reciprocity_resolution90.py",
        "scripts/python-validation/run_example_oracle.py",
        "build/meep-cuda-mpi-python-fp32/src/.libs/libmeep.so.38.0.0",
        "build/meep-cuda-mpi-python-fp32/python/meep/_meep.so",
    }
    if not isinstance(freeze_inputs, dict) or not required_inputs.issubset(
        freeze_inputs
    ):
        raise RuntimeError("resolution-90 v3 source freeze lacks required inputs")
    if any(
        not isinstance(freeze_inputs[name], str)
        or len(freeze_inputs[name]) != 64
        for name in required_inputs
    ):
        raise RuntimeError("resolution-90 v3 source-freeze input digest is invalid")

    log_text = log_path.read_text(encoding="utf-8")
    first_line = log_text.splitlines()[0] if log_text else ""
    for token in (
        "CUDA_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=1",
        "MEEP_GPU_BACKEND=cuda",
        "GPMEEP_VALIDATION_STRICT_CUDA=1",
        "run_stochastic_reciprocity_resolution90.py",
    ):
        if token not in first_line:
            raise RuntimeError(f"resolution-90 v3 command lacks {token}")
    if log_text.count("Using MPI version 3.1, 1 processes") != 1:
        raise RuntimeError("resolution-90 v3 log has the wrong MPI profile")
    if log_text.count('COMMAND_EXIT_CODE="1"') != 1:
        raise RuntimeError("resolution-90 v3 log did not preserve its expected failure")
    timestep_matches = [
        int(value)
        for value in re.findall(
            r"run 0 finished at t = [^\n]+ \(([0-9]+) timesteps\)",
            log_text,
        )
    ]
    if timestep_matches != list(V3_EXPECTED_TIMESTEPS):
        raise RuntimeError("resolution-90 v3 log has the wrong eight-run sequence")

    payloads = [
        line[len(DIAGNOSTICS_PREFIX) :]
        for line in log_text.splitlines()
        if line.startswith(DIAGNOSTICS_PREFIX)
    ]
    if len(payloads) != 1:
        raise RuntimeError("resolution-90 v3 log lacks one diagnostics payload")
    try:
        diagnostics = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("resolution-90 v3 diagnostics JSON is invalid") from exc
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("schema")
        != "gpmeep-reciprocity-resolution-diagnostics-v1"
        or diagnostics.get("resolution") != 90
        or diagnostics.get("dipole_count") != 99
        or diagnostics.get("source_index") != 27
        or not np.isclose(
            float(diagnostics.get("position", np.nan)),
            -0.25,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError("resolution-90 v3 diagnostics has the wrong profile")

    frequencies = _finite_array(diagnostics.get("frequencies"), "frequencies", (4,))
    forward_flat = _finite_array(diagnostics.get("forward_flat"), "forward flat", (2, 4))
    forward_textured = _finite_array(
        diagnostics.get("forward_textured"), "forward textured", (2, 4)
    )
    backward_flat = _finite_array(diagnostics.get("backward_flat"), "backward flat", (2, 4))
    backward_textured = _finite_array(
        diagnostics.get("backward_textured"), "backward textured", (2, 4)
    )
    for name, value in (
        ("frequencies", frequencies),
        ("forward flat", forward_flat),
        ("forward textured", forward_textured),
        ("backward flat", backward_flat),
        ("backward textured", backward_textured),
    ):
        if np.min(value) <= 0:
            raise RuntimeError(f"resolution-90 v3 {name} is not positive")
    if not np.all(np.diff(frequencies) > 0):
        raise RuntimeError("resolution-90 v3 frequencies are not increasing")

    raw_base = np.stack(
        (forward_flat[0], forward_textured[0], backward_flat[0], backward_textured[0])
    )
    raw_doubled = np.stack(
        (forward_flat[1], forward_textured[1], backward_flat[1], backward_textured[1])
    )
    raw_residual = raw_base - raw_doubled
    raw_relative_l2 = _relative_l2_rows(raw_residual, raw_doubled)
    _, raw_scale_aware = reciprocity._pointwise_relative_errors(
        raw_residual, raw_doubled, axis=1
    )
    forward_normalized = forward_textured / forward_flat
    backward_normalized = backward_textured / backward_flat
    normalized_residual = np.stack(
        (
            forward_normalized[0] - forward_normalized[1],
            backward_normalized[0] - backward_normalized[1],
        )
    )
    normalized_reference = np.stack(
        (forward_normalized[1], backward_normalized[1])
    )
    normalized_relative_l2 = _relative_l2_rows(
        normalized_residual, normalized_reference
    )
    _, normalized_scale_aware = reciprocity._pointwise_relative_errors(
        normalized_residual, normalized_reference, axis=1
    )
    closure_residual = forward_normalized[1] - backward_normalized[1]
    closure_relative_l2 = float(
        np.linalg.norm(closure_residual) / np.linalg.norm(backward_normalized[1])
    )
    _, closure_scale_aware = reciprocity._pointwise_relative_errors(
        closure_residual, backward_normalized[1]
    )
    reference_res50 = float(diagnostics.get("reference_res50_relative_l2", np.nan))
    if not np.isfinite(reference_res50) or reference_res50 <= 0:
        raise RuntimeError("resolution-90 v3 resolution-50 reference is invalid")

    _require_close(diagnostics.get("forward_normalized"), forward_normalized, "forward normalized")
    _require_close(diagnostics.get("backward_normalized"), backward_normalized, "backward normalized")
    _require_close(diagnostics.get("raw_time_relative_l2"), raw_relative_l2, "raw relative L2")
    _require_close(
        diagnostics.get("raw_time_scale_aware_max"),
        float(np.max(raw_scale_aware)),
        "raw scale-aware maximum",
    )
    _require_close(
        diagnostics.get("normalized_time_relative_l2"),
        normalized_relative_l2,
        "normalized relative L2",
    )
    _require_close(
        diagnostics.get("normalized_time_scale_aware_max"),
        float(np.max(normalized_scale_aware)),
        "normalized scale-aware maximum",
    )
    _require_close(
        diagnostics.get("doubled_closure_relative_l2"),
        closure_relative_l2,
        "doubled closure relative L2",
    )
    _require_close(
        diagnostics.get("doubled_closure_scale_aware_max"),
        float(np.max(closure_scale_aware)),
        "doubled closure scale-aware maximum",
    )
    _require_close(
        diagnostics.get("refinement_ratio"),
        closure_relative_l2 / reference_res50,
        "refinement ratio",
    )
    if (
        float(np.max(normalized_relative_l2)) <= 0.02
        or "normalized ratios did not converge" not in log_text
    ):
        raise RuntimeError("resolution-90 v3 evidence does not prove the target failure")

    audit = {
        "schema": "gpmeep-reciprocity-temporal-reference-audit-v1",
        "log_path": str(log_path),
        "log_sha256": V3_LOG_SHA256,
        "freeze_path": str(freeze_path),
        "freeze_sha256": V3_FREEZE_SHA256,
        "git_head": freeze["git_head"],
        "runs": 8,
        "timestep_deltas": timestep_matches,
        "reference_failure_normalized_relative_l2_max": float(
            np.max(normalized_relative_l2)
        ),
        "reference_failure_normalized_scale_aware_max": float(
            np.max(normalized_scale_aware)
        ),
        "reference_res50_relative_l2": reference_res50,
    }
    return diagnostics, audit


def build_oracle_argv(
    example: pathlib.Path, reference_diagnostics: dict[str, Any]
) -> list[str]:
    wrapper = pathlib.Path(__file__).resolve().with_name(
        "stochastic_reciprocity_temporal_extension_probe.py"
    )
    arguments = [
        "--expected-run-count",
        "2",
        "--expected-final-timestep",
        str(EXPECTED_TIMESTEP),
        "--expected-run-timestep-delta-range",
        f"{EXPECTED_TIMESTEP}:{EXPECTED_TIMESTEP}",
        "--expected-run-timestep-delta-range",
        f"{EXPECTED_TIMESTEP}:{EXPECTED_TIMESTEP}",
        "--min-each-dft-norm",
        "1e-8",
    ]
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
            "90",
            "--frequencies",
            "4",
            "--position",
            "-0.25",
            "--textured-runtime",
            "512",
            "--reference-diagnostics-json",
            json.dumps(
                reference_diagnostics,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    return arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-log", required=True, type=pathlib.Path)
    parser.add_argument("--reference-source-freeze", required=True, type=pathlib.Path)
    parser.add_argument("example", type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.example.is_file() or args.example.suffix != ".py":
        parser.error(f"example is not a Python file: {args.example}")
    if not args.reference_log.is_file():
        parser.error(f"reference log is not a file: {args.reference_log}")
    if not args.reference_source_freeze.is_file():
        parser.error(
            f"reference source freeze is not a file: {args.reference_source_freeze}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics, audit = load_reference(
        args.reference_log, args.reference_source_freeze
    )
    print(
        "gpmeep-reciprocity-temporal-reference-audit:"
        + json.dumps(
            audit, allow_nan=False, sort_keys=True, separators=(",", ":")
        ),
        flush=True,
    )
    return run_example_oracle.main(
        build_oracle_argv(args.example, diagnostics)
    )


if __name__ == "__main__":
    raise SystemExit(main())
