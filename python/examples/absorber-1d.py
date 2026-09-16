#!/usr/bin/env python3

"""Compare a scalar absorber with PML in a one-dimensional dispersive cell."""

import argparse

import numpy as np
from meep.materials import Al

import meep as mp


def main(args):
    # The dispersive Al fit is unstable at coarse resolution in FP32.  Keep
    # the upstream 40 pixels/unit stability requirement in validation too.
    resolution = args.resolution if args.resolution is not None else 40
    decay_window = (
        args.decay_window
        if args.decay_window is not None
        else (20 if args.validation else 50)
    )
    decay_tolerance = (
        args.decay_tolerance
        if args.decay_tolerance is not None
        else (1e-5 if args.validation else 1e-6)
    )
    if resolution <= 0 or decay_window <= 0 or not 0 < decay_tolerance < 1:
        raise RuntimeError("invalid absorber validation controls")

    cell_size = mp.Vector3(z=10)
    boundary_layers = [
        mp.PML(1, direction=mp.Z) if args.pml else mp.Absorber(1, direction=mp.Z)
    ]
    sources = [
        mp.Source(
            src=mp.GaussianSource(1 / 0.803, fwidth=0.1),
            center=mp.Vector3(),
            component=mp.Ex,
        )
    ]

    trace_times = []
    trace_fields = []

    def record_field(sim):
        value = complex(sim.get_field_point(mp.Ex, mp.Vector3()))
        time_value = float(sim.meep_time())
        if trace_times and abs(time_value - trace_times[-1]) < 1e-12:
            trace_fields[-1] = value
        else:
            trace_times.append(time_value)
            trace_fields.append(value)
        if not args.validation:
            print(f"ex:, {time_value}, {value.real}")

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        dimensions=1,
        default_material=Al,
        boundary_layers=boundary_layers,
        sources=sources,
    )
    # Resolve the 0.803-time-unit carrier with about 16 samples per period.
    # The previous two-time-unit validation cadence aliased the ringdown and
    # could miss both its true peak and branch differences.
    callback_period = 0.05 if args.validation else 10
    step_functions = [mp.at_every(callback_period, record_field)]
    if args.validation:
        step_functions.insert(0, mp.at_beginning(record_field))
    sim.run(
        *step_functions,
        until_after_sources=mp.stop_when_fields_decayed(
            decay_window, mp.Ex, mp.Vector3(), decay_tolerance
        ),
    )
    if args.validation:
        record_field(sim)

    times = np.asarray(trace_times, dtype=float)
    fields = np.asarray(trace_fields, dtype=complex)
    magnitudes = np.abs(fields)
    peak = float(np.max(magnitudes))
    late = magnitudes[times >= times[-1] - decay_window - 1e-9]
    tail_amplitude_ratio = float(np.max(late) / max(peak, 1e-30))
    tail_power_ratio = tail_amplitude_ratio**2
    result = {
        "times": times,
        "fields": fields,
        "peak": peak,
        "tail_amplitude_ratio": tail_amplitude_ratio,
        "tail_power_ratio": tail_power_ratio,
        "stop_time": float(sim.meep_time()),
        "timestep": int(sim.timestep()),
        "branch": 1 if args.pml else 0,
        "parameters": np.asarray(
            [resolution, decay_window, decay_tolerance, 1 / 0.803, 0.1],
            dtype=float,
        ),
    }
    if args.validation:
        if times.size < 8 or peak <= 1e-12 or not np.all(np.isfinite(fields)):
            raise RuntimeError("absorber validation produced an invalid field trace")
        if tail_power_ratio > 5 * decay_tolerance:
            raise RuntimeError("absorber validation did not independently confirm decay")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pml", action="store_true", default=False, help="Use PML as boundary layer"
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--decay-window", type=float, default=None)
    parser.add_argument("--decay-tolerance", type=float, default=None)
    args = parser.parse_args()
    absorber_result = main(args)
    absorber_trace_times = absorber_result["times"]
    absorber_trace_fields = absorber_result["fields"]
    absorber_summary = np.asarray(
        [
            absorber_result["branch"],
            absorber_result["stop_time"],
            absorber_result["timestep"],
            absorber_result["peak"],
            absorber_result["tail_power_ratio"],
        ],
        dtype=float,
    )
    absorber_parameters = absorber_result["parameters"]
