#!/usr/bin/env python3
"""Execute and validate both branches of the one-dimensional absorber example."""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys

import numpy as np

TRACE_SAMPLES = 4096


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("example_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.example_args[:1] == ["--"]:
        args.example_args = args.example_args[1:]
    if not args.example.is_file():
        parser.error(f"example does not exist: {args.example}")
    return args


def execute(example, common_args, use_pml, run_path=runpy.run_path):
    previous_argv = sys.argv
    sys.argv = [str(example), *common_args]
    if use_pml:
        sys.argv.append("-pml")
    try:
        return run_path(str(example), run_name="__main__")["absorber_result"]
    finally:
        sys.argv = previous_argv


def run_branch_matrix(args, run_path=runpy.run_path):
    results = [
        execute(args.example, args.example_args, use_pml=False, run_path=run_path),
        execute(args.example, args.example_args, use_pml=True, run_path=run_path),
    ]
    branch_codes = np.asarray([item["branch"] for item in results], dtype=float)
    stop_times = np.asarray([item["stop_time"] for item in results], dtype=float)
    timesteps = np.asarray([item["timestep"] for item in results], dtype=float)
    peaks = np.asarray([item["peak"] for item in results], dtype=float)
    tail_amplitude_ratios = np.asarray(
        [item["tail_amplitude_ratio"] for item in results], dtype=float
    )
    tail_power_ratios = np.asarray(
        [item["tail_power_ratio"] for item in results], dtype=float
    )
    parameters = np.stack([item["parameters"] for item in results])

    common_stop = float(np.min(stop_times))
    sample_times = np.linspace(0.0, common_stop, TRACE_SAMPLES)
    trace_times = np.stack([sample_times, sample_times])
    sampled_fields = []
    for item in results:
        times = np.asarray(item["times"], dtype=float)
        fields = np.asarray(item["fields"], dtype=complex)
        carrier_frequency = float(item["parameters"][3])
        if times.size < 2 or np.max(np.diff(times)) > 0.25 / carrier_frequency:
            raise RuntimeError("absorber field trace undersamples the optical carrier")
        sampled_fields.append(
            np.interp(sample_times, times, fields.real)
            + 1j * np.interp(sample_times, times, fields.imag)
        )
    trace_fields = np.stack(sampled_fields)
    denominator = max(float(np.linalg.norm(trace_fields[1])), 1e-30)
    branch_relative_error = np.asarray(
        [float(np.linalg.norm(trace_fields[0] - trace_fields[1]) / denominator)]
    )

    if not np.array_equal(branch_codes, [0.0, 1.0]):
        raise RuntimeError("absorber branch matrix did not execute Absorber and PML")
    if np.any(~np.isfinite(trace_fields)) or np.any(peaks <= 1e-12):
        raise RuntimeError("absorber branch matrix produced invalid fields")
    if np.any(tail_power_ratios > 5e-5):
        raise RuntimeError("absorber branch matrix failed its decay contract")
    if branch_relative_error[0] > 0.02:
        raise RuntimeError("one-dimensional Absorber and PML traces diverged")
    return {
        "absorber_branch_codes": branch_codes,
        "absorber_stop_times": stop_times,
        "absorber_timesteps": timesteps,
        "absorber_peaks": peaks,
        "absorber_tail_amplitude_ratios": tail_amplitude_ratios,
        "absorber_tail_power_ratios": tail_power_ratios,
        "absorber_parameters": parameters,
        "absorber_trace_times": trace_times,
        "absorber_trace_fields": trace_fields,
        "absorber_branch_relative_error": branch_relative_error,
    }


def main(argv=None):
    result = run_branch_matrix(parse_args(argv))
    globals().update(result)
    return 0


if __name__ == "__main__":
    main()
