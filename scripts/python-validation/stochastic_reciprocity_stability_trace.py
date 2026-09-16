#!/usr/bin/env python3
"""Trace late-time stochastic-reciprocity field growth in one simulation.

The target example remains the single source of truth.  This probe uses the
same private problem factory as its production ``forward`` function, retains
one Simulation/DFT state, and advances it through increasing absolute
post-source endpoints before taking getter-only snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import runpy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import gpmeep_execution_identity as execution_identity
import stochastic_reciprocity_matrix as reciprocity


DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-stability-trace:"
FIELD_NAMES = ("Ez", "Dz", "Hx", "Hy", "Dz_minus_Ez")
REGION_NAMES = (
    "global",
    "nominal_silver_center",
    "high_index_including_rod",
    "pml",
    "air_non_pml",
)
DEFAULT_CHECKPOINTS = (128, 192, 256, 320, 384, 448, 512)
POSITION_ATOL = 1e-12


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_courant(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0 or result > 0.5:
        raise argparse.ArgumentTypeError("must be finite and in (0, 0.5]")
    return result


def checkpoint_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "checkpoints must be comma-separated integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("checkpoints must be positive")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("checkpoints must be unique and increasing")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=positive_int, default=90)
    parser.add_argument("--frequencies", type=positive_int, default=4)
    parser.add_argument("--runtime", type=positive_int, default=512)
    parser.add_argument("--courant", type=positive_courant, default=0.5)
    parser.add_argument("--position", type=float, default=-0.25)
    parser.add_argument(
        "--checkpoints",
        type=checkpoint_list,
        default=DEFAULT_CHECKPOINTS,
    )
    args = parser.parse_args(argv)
    if not args.example.is_file():
        parser.error(f"example is not a file: {args.example}")
    if not np.isfinite(args.position):
        parser.error("position must be finite")
    if args.checkpoints[-1] != args.runtime:
        parser.error("the final checkpoint must equal runtime")
    return args


def expected_timestep(
    runtime: int,
    resolution: int,
    courant: float,
    *,
    source_end: float = 50.0,
    frequencies: int = 4,
    frequency_width: float = 0.2,
) -> int:
    if frequencies <= 0 or not np.isfinite(frequency_width) or frequency_width <= 0:
        raise RuntimeError("frequency count and width must be positive")
    exact = (
        source_end + runtime * frequencies / frequency_width
    ) * resolution / courant
    rounded = int(round(exact))
    if not np.isclose(exact, rounded, rtol=0.0, atol=1e-9):
        raise RuntimeError("checkpoint does not end on an exact timestep")
    return rounded


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def _phase_call_totals(statistics: Any) -> dict[str, int]:
    totals = {"cpu": 0, "cuda": 0}

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key, value in item.items():
            if isinstance(value, dict):
                visit(value)
            elif key.endswith("_calls"):
                for backend in totals:
                    if key.startswith(f"{backend}_"):
                        count = int(value)
                        if count < 0:
                            raise RuntimeError(
                                f"negative {backend} phase-call counter {key}"
                            )
                        totals[backend] += count

    visit(statistics)
    return totals


def _counter_delta(
    before: dict[str, int], after: dict[str, int], label: str
) -> dict[str, int]:
    result = {backend: after[backend] - before[backend] for backend in before}
    if any(value < 0 for value in result.values()):
        raise RuntimeError(f"GPU statistics reset during {label}")
    return result


def _grid_partition(
    simulation: Any, mp: Any, namespace: dict[str, Any]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    dict[str, Any],
]:
    sx = _finite_scalar(namespace.get("sx"), "target period")
    sy = _finite_scalar(namespace.get("sy"), "target height")
    d_ag = _finite_scalar(namespace.get("dAg"), "target silver thickness")
    d_sub = _finite_scalar(namespace.get("dsub"), "target substrate thickness")
    dpml = _finite_scalar(namespace.get("dpml"), "target PML thickness")
    hrod = _finite_scalar(namespace.get("hrod"), "target rod height")
    wrod = _finite_scalar(namespace.get("wrod"), "target rod width")
    if min(sx, sy, d_ag, d_sub, dpml, hrod, wrod) <= 0:
        raise RuntimeError("target geometry dimensions must be positive")
    x, y, _z, raw_weights = simulation.get_array_metadata(
        center=mp.Vector3(), size=mp.Vector3(sx, sy)
    )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(raw_weights, dtype=float)
    if (
        x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
    ):
        raise RuntimeError("field-grid coordinates must be finite and increasing")
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    if (
        x_grid.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise RuntimeError("global field grid and cubature metadata differ")
    positive = weights > 0
    ag_top = -0.5 * sy + d_ag
    substrate_top = ag_top + d_sub
    rod_top = substrate_top + hrod
    pml_start = 0.5 * sy - dpml
    nominal_silver_center = positive & (y_grid < ag_top)
    high_index_including_rod = positive & (
        ((y_grid >= ag_top) & (y_grid < substrate_top))
        | (
            (y_grid >= substrate_top)
            & (y_grid < rod_top)
            & (np.abs(x_grid) <= 0.5 * wrod)
        )
    )
    pml = positive & (y_grid >= pml_start)
    assigned = nominal_silver_center | high_index_including_rod | pml
    air = positive & ~assigned
    masks = (
        positive,
        nominal_silver_center,
        high_index_including_rod,
        pml,
        air,
    )
    partition_count = (
        nominal_silver_center.astype(np.uint8)
        + high_index_including_rod.astype(np.uint8)
        + pml.astype(np.uint8)
        + air.astype(np.uint8)
    )
    if not np.array_equal(
        partition_count[positive], np.ones(np.count_nonzero(positive))
    ):
        raise RuntimeError("material/PML masks do not partition the field grid")
    for name, mask in zip(REGION_NAMES, masks):
        if not np.any(mask) or float(np.sum(weights[mask])) <= 0:
            raise RuntimeError(f"{name} field-grid region is empty")
    partition_weight = sum(float(np.sum(weights[mask])) for mask in masks[1:])
    global_weight = float(np.sum(weights[positive]))
    if not np.isclose(partition_weight, global_weight, rtol=1e-12, atol=1e-15):
        raise RuntimeError(
            "material/PML region weights do not sum to the global weight"
        )
    region_weight = {
        name: float(np.sum(weights[mask]))
        for name, mask in zip(REGION_NAMES, masks)
    }
    grid_metadata = {
        "mask_semantics": (
            "nominal Yee-center coordinate partition; it is not effective-material "
            "ownership under subpixel averaging"
        ),
        "boundaries": {
            "silver_top": ag_top,
            "substrate_top": substrate_top,
            "rod_top": rod_top,
            "pml_start": pml_start,
            "rod_half_width": 0.5 * wrod,
        },
        "coordinate_extents": {
            "x_min": float(x[0]),
            "x_max": float(x[-1]),
            "y_min": float(y[0]),
            "y_max": float(y[-1]),
        },
        "spacing": {
            "dx_min": float(np.min(np.diff(x))),
            "dx_max": float(np.max(np.diff(x))),
            "dy_min": float(np.min(np.diff(y))),
            "dy_max": float(np.max(np.diff(y))),
        },
        "positive_weight_points": int(np.count_nonzero(positive)),
        "region_weight": region_weight,
    }
    return x_grid, y_grid, weights, masks, grid_metadata


def _weighted_stats(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    name: str,
) -> tuple[float, float]:
    array = np.asarray(values)
    cubature = np.asarray(weights, dtype=float)
    region = np.asarray(mask, dtype=bool)
    if (
        array.shape != cubature.shape
        or region.shape != cubature.shape
        or array.size == 0
    ):
        raise RuntimeError(f"{name} field and cubature shapes differ")
    magnitude = np.abs(
        np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)
    )
    if not np.all(np.isfinite(magnitude)) or not np.all(np.isfinite(cubature)):
        raise RuntimeError(f"{name} checkpoint contains non-finite values")
    if np.any(cubature < 0):
        raise RuntimeError(f"{name} cubature weights are negative")
    weight_sum = float(np.sum(cubature[region]))
    if weight_sum <= 0:
        raise RuntimeError(f"{name} cubature weights are not positive")
    maximum = float(np.max(magnitude[region]))
    rms = float(
        np.sqrt(
            np.sum(cubature[region] * magnitude[region] * magnitude[region])
            / weight_sum
        )
    )
    if not np.isfinite(maximum) or not np.isfinite(rms):
        raise RuntimeError(f"{name} checkpoint statistics are not finite")
    return maximum, rms


def run_stability_trace(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
    identity_capture: Callable[
        [Any, str, Any, Sequence[pathlib.Path]], dict[str, Any]
    ] = (
        execution_identity.capture_execution_identity
    ),
) -> dict[str, np.ndarray]:
    namespace = run_path(str(args.example), run_name="gpmeep_reciprocity_trace_target")
    forward_problem = namespace.get("_forward_problem")
    mp = namespace.get("mp")
    if not callable(forward_problem) or mp is None:
        raise RuntimeError(
            "target does not expose the forward problem factory and Meep module"
        )

    sx = _finite_scalar(namespace.get("sx"), "target period")
    sy = _finite_scalar(namespace.get("sy"), "target height")
    df = _finite_scalar(namespace.get("df"), "target frequency width")
    if not np.isclose(df, 0.2, rtol=0.0, atol=1e-15):
        raise RuntimeError("stability trace requires target df=0.2")
    dipole_count = int(round(sx * args.resolution))
    if dipole_count != 99 or not np.isclose(
        sx * args.resolution, dipole_count, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("stability trace requires the 99-pixel period")
    grid_spacing = sx / dipole_count
    source_index = int(round((args.position + 0.5 * sx) / grid_spacing))
    source_x = sx * (-0.5 + source_index / dipole_count)
    if source_index != 27 or not np.isclose(
        source_x, args.position, rtol=0.0, atol=POSITION_ATOL
    ):
        raise RuntimeError("stability trace is not the shared grid point")

    captures: list[dict[str, Any]] = []
    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
        "courant": args.courant,
    }
    with reciprocity._patched_target_globals((forward_problem,), updates):
        simulation, flux_monitor = forward_problem(source_index, True)
        grid: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            tuple[np.ndarray, ...],
            dict[str, Any],
        ] | None = None
        field_buffers: dict[int, np.ndarray] = {}
        field_buffer_allocations = 0
        field_buffer_reuses = 0
        try:
            simulation.init_sim()
            if simulation.fields is None or int(simulation.fields.t) != 0:
                raise RuntimeError("stability trace was not initialized at timestep zero")
            identity_document = identity_capture(
                mp,
                "stochastic-reciprocity-stability-trace",
                simulation,
                (
                    pathlib.Path(__file__),
                    args.example,
                    pathlib.Path(reciprocity.__file__),
                    pathlib.Path(execution_identity.__file__),
                ),
            )
            execution_identity.emit_execution_identity(identity_document, mp)
            previous_expected_timestep = 0
            for runtime in args.checkpoints:
                pre_timestep = (
                    int(simulation.fields.t) if simulation.fields is not None else 0
                )
                pre_calls = _phase_call_totals(mp.gpu.statistics())
                simulation.run(
                    until_after_sources=runtime * args.frequencies / df
                )
                post_run_statistics = mp.gpu.statistics()
                post_run_calls = _phase_call_totals(post_run_statistics)
                run_phase_calls = _counter_delta(
                    pre_calls, post_run_calls, "stability trace segment"
                )
                active_backend = str(mp.gpu.active_backend)
                requested_backend = str(
                    getattr(mp.gpu, "requested_backend", active_backend)
                )
                if active_backend not in ("cpu", "cuda"):
                    raise RuntimeError("stability trace selected an unknown backend")
                other_backend = "cuda" if active_backend == "cpu" else "cpu"
                if run_phase_calls[active_backend] <= 0:
                    raise RuntimeError(
                        "stability trace segment lacks active-backend phase calls"
                    )
                if run_phase_calls[other_backend] != 0:
                    raise RuntimeError("stability trace segment used backend fallback")
                if os.environ.get("GPMEEP_VALIDATION_STRICT_CUDA") == "1" and (
                    active_backend != "cuda" or requested_backend != "cuda"
                    or not bool(mp.is_single_precision())
                ):
                    raise RuntimeError(
                        "strict CUDA trace did not select an all-CUDA FP32 path"
                    )
                source_end = float(simulation.fields.last_source_time())
                if not np.isclose(source_end, 50.0, rtol=0.0, atol=1e-12):
                    raise RuntimeError("target Gaussian source end time changed")
                target_time = source_end + runtime * args.frequencies / df
                target_timestep = expected_timestep(
                    runtime,
                    args.resolution,
                    args.courant,
                    source_end=source_end,
                    frequencies=args.frequencies,
                    frequency_width=df,
                )
                actual_time = float(simulation.round_time())
                timestep = int(simulation.fields.t)
                timestep_delta = timestep - pre_timestep
                expected_delta = target_timestep - previous_expected_timestep
                if timestep != target_timestep or not np.isclose(
                    actual_time,
                    target_time,
                    rtol=0.0,
                    atol=0.51 * float(simulation.fields.dt),
                ):
                    raise RuntimeError(
                        "stability checkpoint landed on the wrong timestep"
                    )
                if (
                    pre_timestep != previous_expected_timestep
                    or timestep_delta != expected_delta
                ):
                    raise RuntimeError("stability trace did not advance continuously")
                flux = np.asarray(mp.get_fluxes(flux_monitor), dtype=float)
                if flux.shape != (args.frequencies,) or not np.all(
                    np.isfinite(flux)
                ):
                    raise RuntimeError("stability checkpoint flux is invalid")
                if grid is None:
                    grid = _grid_partition(simulation, mp, namespace)
                x_grid, y_grid, weights, masks, grid_metadata = grid
                maxima = np.zeros(
                    (len(REGION_NAMES), len(FIELD_NAMES)), dtype=float
                )
                rms = np.zeros_like(maxima)
                components = (mp.Ez, mp.Dz, mp.Hx, mp.Hy)
                arrays = []
                for component in components:
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
                        field_buffer_allocations += 1
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
                                "get_array did not reuse the supplied field buffer"
                            )
                        field_buffers[component] = values
                        field_buffer_reuses += 1
                    if values.shape != weights.shape:
                        raise RuntimeError("field and global metadata shapes differ")
                    arrays.append(values)
                promoted = [
                    np.asarray(
                        values,
                        dtype=(
                            np.complex128
                            if np.iscomplexobj(values)
                            else np.float64
                        ),
                    )
                    for values in arrays
                ]
                promoted.append(promoted[1] - promoted[0])
                argmax_xy = np.zeros((len(FIELD_NAMES), 2), dtype=float)
                argmax_silver_interface_distance = np.zeros(
                    len(FIELD_NAMES), dtype=float
                )
                nonfinite_count = np.zeros(
                    (len(REGION_NAMES), len(FIELD_NAMES)), dtype=float
                )
                for field_index, values in enumerate(promoted):
                    magnitude = np.abs(values)
                    for region_index, mask in enumerate(masks):
                        nonfinite_count[region_index, field_index] = float(
                            np.count_nonzero(~np.isfinite(magnitude[mask]))
                        )
                    if np.any(nonfinite_count[:, field_index]):
                        raise RuntimeError(
                            f"{FIELD_NAMES[field_index]} became non-finite at "
                            f"runtime {runtime}"
                        )
                    flat_index = int(
                        np.argmax(np.where(masks[0], magnitude, -np.inf))
                    )
                    array_index = np.unravel_index(flat_index, magnitude.shape)
                    argmax_xy[field_index] = (
                        x_grid[array_index],
                        y_grid[array_index],
                    )
                    argmax_silver_interface_distance[field_index] = float(
                        y_grid[array_index]
                        - grid_metadata["boundaries"]["silver_top"]
                    )
                for region_index, mask in enumerate(masks):
                    for field_index, values in enumerate(promoted):
                        (
                            maxima[region_index, field_index],
                            rms[region_index, field_index],
                        ) = _weighted_stats(
                            values,
                            weights,
                            mask,
                            f"{REGION_NAMES[region_index]}/{FIELD_NAMES[field_index]}",
                        )
                dft_norm = float(simulation.fields.dft_norm())
                if not np.isfinite(dft_norm) or dft_norm <= 0:
                    raise RuntimeError("stability checkpoint DFT norm is invalid")
                measurement_phase_calls = _counter_delta(
                    post_run_calls,
                    _phase_call_totals(mp.gpu.statistics()),
                    "stability trace measurement",
                )
                if any(measurement_phase_calls.values()):
                    raise RuntimeError(
                        "stability trace getters invoked a stepping backend"
                    )
                captures.append(
                    {
                        "runtime": runtime,
                        "meep_time": actual_time,
                        "timestep": timestep,
                        "flux": flux,
                        "field_max": maxima,
                        "field_rms": rms,
                        "field_argmax_xy": argmax_xy,
                        "field_argmax_silver_interface_distance": (
                            argmax_silver_interface_distance
                        ),
                        "field_nonfinite_count": nonfinite_count,
                        "dft_norm": dft_norm,
                        "active_backend": active_backend,
                        "requested_backend": requested_backend,
                        "run_phase_calls": run_phase_calls,
                        "measurement_phase_calls": measurement_phase_calls,
                        "execution_diagnostic": str(
                            simulation.fields.gpu_execution_diagnostic()
                            if hasattr(
                                simulation.fields,
                                "gpu_execution_diagnostic",
                            )
                            else "unavailable"
                        ),
                        "timestep_delta": timestep_delta,
                        "dt": float(simulation.fields.dt),
                        "source_end": source_end,
                    }
                )
                previous_expected_timestep = target_timestep
            mode = simulation.get_eigenmode_coefficients(
                flux_monitor, [1], eig_parity=mp.ODD_Z
            )
            spectrum = np.abs(mode.alpha[0, :, 0]) ** 2
            frequencies = mp.get_flux_freqs(flux_monitor)
        finally:
            simulation.reset_meep()

    if len(captures) != len(args.checkpoints):
        raise RuntimeError("stability trace did not capture every checkpoint")
    captured_runtime = tuple(item["runtime"] for item in captures)
    if captured_runtime != args.checkpoints:
        raise RuntimeError("stability checkpoints are missing or out of order")
    if field_buffer_allocations != 4 or field_buffer_reuses != 4 * (
        len(args.checkpoints) - 1
    ):
        raise RuntimeError("stability trace field-buffer lifecycle is invalid")
    frequencies = reciprocity._finite_array(
        frequencies, "stability trace frequencies", (args.frequencies,)
    )
    spectrum = reciprocity._finite_array(
        spectrum, "stability trace endpoint spectrum", (args.frequencies,)
    )
    if np.min(spectrum) <= 0:
        raise RuntimeError("stability trace endpoint spectrum is not positive")

    runtime_array = np.asarray(captured_runtime, dtype=float)
    time_array = np.asarray([item["meep_time"] for item in captures], dtype=float)
    timestep_array = np.asarray([item["timestep"] for item in captures], dtype=float)
    flux_array = np.stack([item["flux"] for item in captures])
    field_max = np.stack([item["field_max"] for item in captures])
    field_rms = np.stack([item["field_rms"] for item in captures])
    field_argmax_xy = np.stack([item["field_argmax_xy"] for item in captures])
    field_argmax_silver_interface_distance = np.stack(
        [
            item["field_argmax_silver_interface_distance"]
            for item in captures
        ]
    )
    field_nonfinite_count = np.stack(
        [item["field_nonfinite_count"] for item in captures]
    )
    dft_norm = np.asarray([item["dft_norm"] for item in captures], dtype=float)
    timestep_delta = np.asarray(
        [item["timestep_delta"] for item in captures], dtype=float
    )
    dt = np.asarray([item["dt"] for item in captures], dtype=float)
    source_end = np.asarray(
        [item["source_end"] for item in captures], dtype=float
    )
    run_phase_calls = np.asarray(
        [
            [item["run_phase_calls"]["cpu"], item["run_phase_calls"]["cuda"]]
            for item in captures
        ],
        dtype=float,
    )
    measurement_phase_calls = np.asarray(
        [
            [
                item["measurement_phase_calls"]["cpu"],
                item["measurement_phase_calls"]["cuda"],
            ]
            for item in captures
        ],
        dtype=float,
    )
    backend_code = np.asarray(
        [0.0 if item["active_backend"] == "cpu" else 1.0 for item in captures]
    )
    diagnostics = {
        "schema": "gpmeep-reciprocity-stability-trace-v1",
        "resolution": args.resolution,
        "courant": args.courant,
        "source_index": source_index,
        "checkpoint_runtime": runtime_array.tolist(),
        "checkpoint_time": time_array.tolist(),
        "checkpoint_timestep": timestep_array.astype(int).tolist(),
        "region_names": list(REGION_NAMES),
        "field_names": list(FIELD_NAMES),
        "flux": flux_array.tolist(),
        "field_max": field_max.tolist(),
        "field_rms": field_rms.tolist(),
        "field_argmax_xy": field_argmax_xy.tolist(),
        "field_argmax_silver_interface_distance": (
            field_argmax_silver_interface_distance.tolist()
        ),
        "field_nonfinite_count": field_nonfinite_count.astype(int).tolist(),
        "dft_norm": dft_norm.tolist(),
        "timestep_delta": timestep_delta.astype(int).tolist(),
        "dt": dt.tolist(),
        "source_end": source_end.tolist(),
        "run_phase_calls": {
            "columns": ["cpu", "cuda"],
            "values": run_phase_calls.astype(int).tolist(),
        },
        "measurement_phase_calls": {
            "columns": ["cpu", "cuda"],
            "values": measurement_phase_calls.astype(int).tolist(),
        },
        "active_backend": [item["active_backend"] for item in captures],
        "requested_backend": [item["requested_backend"] for item in captures],
        "execution_diagnostic": [
            item["execution_diagnostic"] for item in captures
        ],
        "frequencies": frequencies.tolist(),
        "endpoint_spectrum": spectrum.tolist(),
        "precision": "fp32" if bool(mp.is_single_precision()) else "fp64",
        "grid_metadata": grid_metadata,
        "execution_identity_sha256": identity_document["identity_sha256"],
        "field_buffer_allocations": field_buffer_allocations,
        "field_buffer_reuses": field_buffer_reuses,
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
        "reciprocity_trace_resolution": np.asarray([args.resolution], dtype=float),
        "reciprocity_trace_courant": np.asarray([args.courant], dtype=float),
        "reciprocity_trace_source_index": np.asarray([source_index], dtype=float),
        "reciprocity_trace_checkpoint_runtime": runtime_array,
        "reciprocity_trace_checkpoint_time": time_array,
        "reciprocity_trace_checkpoint_timestep": timestep_array,
        "reciprocity_trace_flux": flux_array,
        "reciprocity_trace_field_max": field_max,
        "reciprocity_trace_field_rms": field_rms,
        "reciprocity_trace_field_argmax_xy": field_argmax_xy,
        "reciprocity_trace_field_argmax_silver_interface_distance": (
            field_argmax_silver_interface_distance
        ),
        "reciprocity_trace_field_nonfinite_count": field_nonfinite_count,
        "reciprocity_trace_dft_norm": dft_norm,
        "reciprocity_trace_timestep_delta": timestep_delta,
        "reciprocity_trace_dt": dt,
        "reciprocity_trace_source_end": source_end,
        "reciprocity_trace_run_phase_calls": run_phase_calls,
        "reciprocity_trace_measurement_phase_calls": measurement_phase_calls,
        "reciprocity_trace_backend_code": backend_code,
        "reciprocity_trace_frequencies": frequencies,
        "reciprocity_trace_endpoint_spectrum": spectrum,
        "reciprocity_trace_region_weight": np.asarray(
            [grid_metadata["region_weight"][name] for name in REGION_NAMES],
            dtype=float,
        ),
        "reciprocity_trace_field_buffer_allocations": np.asarray(
            [field_buffer_allocations], dtype=float
        ),
        "reciprocity_trace_field_buffer_reuses": np.asarray(
            [field_buffer_reuses], dtype=float
        ),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_stability_trace(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
