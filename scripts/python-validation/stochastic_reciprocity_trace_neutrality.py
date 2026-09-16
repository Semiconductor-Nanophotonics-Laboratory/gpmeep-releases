#!/usr/bin/env python3
"""Prove that segmented stability-trace getters do not perturb final state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import runpy
import struct
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import gpmeep_execution_identity as execution_identity
import stochastic_reciprocity_matrix as reciprocity
import stochastic_reciprocity_stability_trace as trace


DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-trace-neutrality:"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=trace.positive_int, default=90)
    parser.add_argument("--frequencies", type=trace.positive_int, default=4)
    parser.add_argument("--runtime", type=trace.positive_int, default=3)
    parser.add_argument(
        "--checkpoints", type=trace.checkpoint_list, default=(1, 2)
    )
    parser.add_argument("--courant", type=trace.positive_courant, default=0.5)
    parser.add_argument("--position", type=float, default=-0.25)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        parser.error(f"example is not a file: {args.example}")
    if len(args.checkpoints) < 2 or args.checkpoints[-1] >= args.runtime:
        parser.error(
            "neutrality requires at least two increasing checkpoints before runtime"
        )
    if args.resolution != 90 or args.frequencies != 4:
        parser.error("neutrality probe requires resolution=90 and frequencies=4")
    return args


def _signature_manifest(values: dict[str, np.ndarray]) -> dict[str, Any]:
    records = {}
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        records[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "size_bytes": int(array.nbytes),
            "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        }
    manifest = {
        "schema": "gpmeep-bytewise-array-signature-v1",
        "arrays": records,
    }
    manifest["manifest_sha256"] = execution_identity.canonical_sha256(manifest)
    return manifest


def _bytewise_signature_comparison(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    if left.keys() != right.keys():
        raise RuntimeError("neutrality signatures have different keys")
    unequal = []
    for name in left:
        left_array = np.ascontiguousarray(left[name])
        right_array = np.ascontiguousarray(right[name])
        if (
            left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape
            or not np.array_equal(
                left_array.view(np.uint8).reshape(-1),
                right_array.view(np.uint8).reshape(-1),
            )
        ):
            unequal.append(name)
    return unequal, _signature_manifest(left), _signature_manifest(right)


def _float64_bit_record(value: float) -> dict[str, Any]:
    payload = struct.pack(">d", float(value))
    return {
        "encoding": "IEEE-754-binary64-big-endian",
        "hex": payload.hex(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _getter_observation(
    simulation: Any,
    monitor: Any,
    mp: Any,
    sx: float,
    sy: float,
    field_buffers: dict[int, np.ndarray],
) -> dict[str, int]:
    metadata_calls = 0
    if not field_buffers:
        simulation.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(sx, sy))
        metadata_calls = 1
    allocations = 0
    reuses = 0
    for component in (mp.Ez, mp.Dz, mp.Hx, mp.Hy):
        previous = field_buffers.get(component)
        if previous is None:
            values = np.asarray(
                simulation.get_array(
                    component=component,
                    center=mp.Vector3(),
                    size=mp.Vector3(sx, sy),
                )
            )
            field_buffers[component] = values
            allocations += 1
        else:
            values = np.asarray(
                simulation.get_array(
                    component=component,
                    center=mp.Vector3(),
                    size=mp.Vector3(sx, sy),
                    arr=previous,
                )
            )
            if not np.shares_memory(values, previous):
                raise RuntimeError(
                    "neutrality observer did not reuse the supplied field buffer"
                )
            field_buffers[component] = values
            reuses += 1
        if not np.all(np.isfinite(values)):
            raise RuntimeError("neutrality observer encountered non-finite fields")
    flux = np.asarray(mp.get_fluxes(monitor), dtype=float)
    dft_norm = float(simulation.fields.dft_norm())
    if not np.all(np.isfinite(flux)) or not np.isfinite(dft_norm):
        raise RuntimeError("neutrality observer encountered non-finite DFT data")
    return {
        "metadata_calls": metadata_calls,
        "field_buffer_allocations": allocations,
        "field_buffer_reuses": reuses,
    }


def _final_signature(
    simulation: Any,
    monitor: Any,
    mp: Any,
    sx: float,
    sy: float,
    frequencies: int,
) -> tuple[dict[str, np.ndarray], float]:
    arrays = {
        f"field_{name}": np.asarray(
            simulation.get_array(
                component=component,
                center=mp.Vector3(),
                size=mp.Vector3(sx, sy),
            )
        ).copy()
        for name, component in (
            ("Ez", mp.Ez),
            ("Dz", mp.Dz),
            ("Hx", mp.Hx),
            ("Hy", mp.Hy),
        )
    }
    arrays["flux"] = np.asarray(mp.get_fluxes(monitor), dtype=float)
    for frequency_index in range(frequencies):
        for name, component in (("Ez", mp.Ez), ("Hx", mp.Hx)):
            arrays[f"dft_{name}_{frequency_index}"] = np.asarray(
                simulation.get_dft_array(monitor, component, frequency_index)
            ).copy()
    mode = simulation.get_eigenmode_coefficients(
        monitor, [1], eig_parity=mp.ODD_Z
    )
    arrays["mode_alpha"] = np.asarray(mode.alpha).copy()
    arrays["mode_spectrum"] = np.abs(arrays["mode_alpha"][0, :, 0]) ** 2
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"neutrality final signature {name} is non-finite")
    dft_norm = float(simulation.fields.dft_norm())
    if not np.isfinite(dft_norm):
        raise RuntimeError("neutrality final DFT norm is non-finite")
    return arrays, dft_norm


def run_neutrality_probe(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
    identity_capture: Callable[
        [Any, str, Any, Sequence[pathlib.Path]], dict[str, Any]
    ] = (
        execution_identity.capture_execution_identity
    ),
) -> dict[str, np.ndarray]:
    namespace = run_path(
        str(args.example), run_name="gpmeep_reciprocity_neutrality_target"
    )
    factory = namespace.get("_forward_problem")
    mp = namespace.get("mp")
    if not callable(factory) or mp is None:
        raise RuntimeError("neutrality target lacks the forward problem factory")
    sx = trace._finite_scalar(namespace.get("sx"), "target period")
    sy = trace._finite_scalar(namespace.get("sy"), "target height")
    df = trace._finite_scalar(namespace.get("df"), "target frequency width")
    dipole_count = int(round(sx * args.resolution))
    source_index = int(round((args.position + 0.5 * sx) / (sx / dipole_count)))
    if dipole_count != 99 or source_index != 27:
        raise RuntimeError("neutrality probe is not the shared-grid profile")
    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
        "courant": args.courant,
    }

    strict_cuda = os.environ.get("GPMEEP_VALIDATION_STRICT_CUDA") == "1"

    def execute(
        segmented: bool,
    ) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
        with reciprocity._patched_target_globals((factory,), updates):
            simulation, monitor = factory(source_index, True)
            try:
                simulation.init_sim()
                if simulation.fields is None or int(simulation.fields.t) != 0:
                    raise RuntimeError(
                        "neutrality case was not initialized at timestep zero"
                    )
                case_name = "segmented" if segmented else "uninterrupted"
                identity_document = identity_capture(
                    mp,
                    f"stochastic-reciprocity-trace-neutrality-{case_name}",
                    simulation,
                    (
                        pathlib.Path(__file__),
                        args.example,
                        pathlib.Path(reciprocity.__file__),
                        pathlib.Path(trace.__file__),
                        pathlib.Path(execution_identity.__file__),
                    ),
                )
                execution_identity.emit_execution_identity(identity_document, mp)
                before = trace._phase_call_totals(mp.gpu.statistics())
                observation_totals = {
                    "count": 0,
                    "metadata_calls": 0,
                    "field_buffer_allocations": 0,
                    "field_buffer_reuses": 0,
                }
                if segmented:
                    field_buffers: dict[int, np.ndarray] = {}
                    for checkpoint in args.checkpoints:
                        simulation.run(
                            until_after_sources=(
                                checkpoint * args.frequencies / df
                            )
                        )
                        expected_checkpoint = trace.expected_timestep(
                            checkpoint,
                            args.resolution,
                            args.courant,
                            frequencies=args.frequencies,
                            frequency_width=df,
                        )
                        if int(simulation.fields.t) != expected_checkpoint:
                            raise RuntimeError(
                                "neutrality observation ended on the wrong timestep"
                            )
                        observation = _getter_observation(
                            simulation,
                            monitor,
                            mp,
                            sx,
                            sy,
                            field_buffers,
                        )
                        observation_totals["count"] += 1
                        for name, value in observation.items():
                            observation_totals[name] += value
                    if observation_totals != {
                        "count": len(args.checkpoints),
                        "metadata_calls": 1,
                        "field_buffer_allocations": 4,
                        "field_buffer_reuses": 4 * (len(args.checkpoints) - 1),
                    }:
                        raise RuntimeError(
                            "neutrality observations did not mirror trace buffer reuse"
                        )
                simulation.run(
                    until_after_sources=args.runtime * args.frequencies / df
                )
                expected = trace.expected_timestep(
                    args.runtime,
                    args.resolution,
                    args.courant,
                    frequencies=args.frequencies,
                    frequency_width=df,
                )
                if int(simulation.fields.t) != expected:
                    raise RuntimeError("neutrality case ended on the wrong timestep")
                signature, dft_norm = _final_signature(
                    simulation,
                    monitor,
                    mp,
                    sx,
                    sy,
                    args.frequencies,
                )
                after = trace._phase_call_totals(mp.gpu.statistics())
                calls = trace._counter_delta(before, after, "neutrality case")
                active = str(mp.gpu.active_backend)
                requested = str(mp.gpu.requested_backend)
                other = "cuda" if active == "cpu" else "cpu"
                if active not in calls or calls[active] <= 0 or calls[other] != 0:
                    raise RuntimeError("neutrality case used an invalid backend path")
                precision = "fp32" if bool(mp.is_single_precision()) else "fp64"
                if strict_cuda and (
                    active != "cuda"
                    or requested != "cuda"
                    or calls["cpu"] != 0
                    or calls["cuda"] <= 0
                    or precision != "fp32"
                ):
                    raise RuntimeError(
                        "strict CUDA neutrality did not use an all-CUDA FP32 path"
                    )
                audit = {
                    "active_backend": active,
                    "requested_backend": requested,
                    "precision": precision,
                    "selected_device": int(mp.gpu.selected_device),
                    "selected_device_identifier": str(
                        mp.gpu.selected_device_identifier
                    ),
                    "phase_calls": calls,
                    "timestep": int(simulation.fields.t),
                    "meep_time": float(simulation.round_time()),
                    "observations": observation_totals,
                    "execution_identity_sha256": identity_document[
                        "identity_sha256"
                    ],
                }
                return signature, dft_norm, audit
            finally:
                simulation.reset_meep()

    uninterrupted, uninterrupted_norm, uninterrupted_audit = execute(False)
    segmented, segmented_norm, segmented_audit = execute(True)
    unequal, uninterrupted_manifest, segmented_manifest = (
        _bytewise_signature_comparison(uninterrupted, segmented)
    )
    uninterrupted_norm_bits = _float64_bit_record(uninterrupted_norm)
    segmented_norm_bits = _float64_bit_record(segmented_norm)
    if unequal or uninterrupted_norm_bits["hex"] != segmented_norm_bits["hex"]:
        raise RuntimeError(
            "segmented getter observation perturbed final state: "
            + ", ".join(unequal or ["dft_norm"])
        )
    diagnostics = {
        "schema": "gpmeep-reciprocity-trace-neutrality-v2",
        "resolution": args.resolution,
        "courant": args.courant,
        "runtime": args.runtime,
        "checkpoints": list(args.checkpoints),
        "strict_cuda": strict_cuda,
        "precision": "fp32" if bool(mp.is_single_precision()) else "fp64",
        "bitwise_equal": True,
        "signature_sha256": uninterrupted_manifest["manifest_sha256"],
        "uninterrupted_signature": uninterrupted_manifest,
        "segmented_signature": segmented_manifest,
        "uninterrupted_dft_norm_bits": uninterrupted_norm_bits,
        "segmented_dft_norm_bits": segmented_norm_bits,
        "dft_norm": uninterrupted_norm,
        "uninterrupted": uninterrupted_audit,
        "segmented": segmented_audit,
    }
    if bool(mp.am_master()):
        print(
            DIAGNOSTICS_PREFIX
            + json.dumps(
                diagnostics,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return {
        "reciprocity_trace_neutrality_bitwise_equal": np.asarray([1.0]),
        "reciprocity_trace_neutrality_dft_norm": np.asarray(
            [uninterrupted_norm]
        ),
        "reciprocity_trace_neutrality_timestep": np.asarray(
            [uninterrupted_audit["timestep"]], dtype=float
        ),
        "reciprocity_trace_neutrality_observations": np.asarray(
            [
                segmented_audit["observations"]["count"],
                segmented_audit["observations"]["metadata_calls"],
                segmented_audit["observations"]["field_buffer_allocations"],
                segmented_audit["observations"]["field_buffer_reuses"],
            ],
            dtype=float,
        ),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_neutrality_probe(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
