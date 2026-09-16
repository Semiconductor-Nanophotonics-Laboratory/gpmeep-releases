#!/usr/bin/env python3
"""Linear down-chirped planewave from a complex CustomSource."""

import argparse

import numpy as np

import meep as mp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="retain dense source/probe traces and gate chirp propagation",
)
parser.add_argument("--resolution", type=int, default=None)
parser.add_argument("--until", type=float, default=None)
args = parser.parse_args()

resolution = args.resolution if args.resolution is not None else (20 if args.validation else 40)

dpml = 2
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

sx = 40
sy = 6
cell_size = mp.Vector3(sx + 2 * dpml, sy)

v0 = 1.0  # pulse center frequency
a = 0.2  # Gaussian envelope half-width
b = -0.5  # linear chirp rate (positive: up-chirp, negative: down-chirp)
t0 = 15  # peak time
run_until = args.until if args.until is not None else t0 + 50
if resolution <= 0 or run_until <= 0:
    raise RuntimeError("resolution and run duration must be positive")


def chirp(time):
    tau = time - t0
    return np.exp(1j * 2 * np.pi * v0 * tau) * np.exp(
        -a * tau**2 + 1j * b * tau**2
    )


source_x = -0.5 * sx
sources = [
    mp.Source(
        src=mp.CustomSource(src_func=chirp),
        center=mp.Vector3(source_x),
        size=mp.Vector3(y=sy),
        component=mp.Ez,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    k_point=mp.Vector3(),
    sources=sources,
    symmetries=[mp.Mirror(mp.Y)],
)


def analytic_signal(values):
    """Return the positive-frequency analytic signal of a real trace."""
    values = np.asarray(values, dtype=float)
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(values.size)
    multiplier[0] = 1
    if values.size % 2 == 0:
        multiplier[1 : values.size // 2] = 2
        multiplier[values.size // 2] = 1
    else:
        multiplier[1 : (values.size + 1) // 2] = 2
    return np.fft.ifft(spectrum * multiplier)


if args.validation:
    sample_period = 0.05
    probe_x = np.asarray([-4.0, 4.0])
    probe_y = np.asarray([-2.0, 0.0, 2.0])
    trace_times = []
    trace_fields = []

    def record_probe_fields(simulation):
        trace_times.append(float(simulation.meep_time()))
        trace_fields.append(
            [
                [
                    complex(
                        simulation.get_field_point(
                            mp.Ez, mp.Vector3(float(x_coord), float(y_coord))
                        )
                    )
                    for y_coord in probe_y
                ]
                for x_coord in probe_x
            ]
        )

    sim.run(mp.at_every(sample_period, record_probe_fields), until=run_until)

    chirp_probe_times = np.asarray(trace_times)
    # Shape: probe x, probe y, time.
    chirp_probe_fields = np.transpose(np.asarray(trace_fields), (1, 2, 0))
    chirp_probe_positions = np.stack(
        np.meshgrid(probe_x, probe_y, indexing="ij"), axis=-1
    )
    center_fields = np.real(chirp_probe_fields[:, 1, :])
    chirp_analytic_probe_fields = np.stack(
        [analytic_signal(values) for values in center_fields]
    )
    chirp_probe_envelopes = np.abs(chirp_analytic_probe_fields)
    arrival_times = t0 + probe_x - source_x
    unwrapped_phase = np.unwrap(np.angle(chirp_analytic_probe_fields), axis=1)
    raw_instantaneous_frequencies = np.gradient(
        unwrapped_phase, chirp_probe_times, axis=1
    ) / (2 * np.pi)
    # Instantaneous phase is undefined once the pulse envelope has decayed to
    # numerical noise.  Retain the full time axis while explicitly marking the
    # deterministic, theory-derived signal interval used for comparison.
    frequency_valid_mask = (
        np.abs(chirp_probe_times[None, :] - arrival_times[:, None]) <= 4.0
    )
    chirp_frequency_valid_mask = frequency_valid_mask.astype(float)
    chirp_frequency_valid_counts = np.sum(frequency_valid_mask, axis=1).astype(float)
    chirp_frequency_valid_time_bounds = np.asarray(
        [
            chirp_probe_times[np.flatnonzero(mask)[[0, -1]]]
            for mask in frequency_valid_mask
        ]
    )
    chirp_expected_frequency_valid_time_bounds = arrival_times[:, None] + np.asarray(
        [-4.0, 4.0]
    )
    chirp_instantaneous_frequencies = np.where(
        frequency_valid_mask, raw_instantaneous_frequencies, 0.0
    )
    window_offsets = ((-2.75, -0.75), (0.75, 2.75))
    window_frequency_means = np.zeros((probe_x.size, len(window_offsets)))
    for probe_index, arrival in enumerate(arrival_times):
        for window_index, (offset_start, offset_stop) in enumerate(window_offsets):
            mask = (chirp_probe_times >= arrival + offset_start) & (
                chirp_probe_times <= arrival + offset_stop
            )
            weights = np.square(chirp_probe_envelopes[probe_index, mask])
            if mask.sum() < 20 or np.sum(weights) <= 0:
                raise RuntimeError("chirped-pulse frequency window has no signal")
            window_frequency_means[probe_index, window_index] = np.sum(
                weights * chirp_instantaneous_frequencies[probe_index, mask]
            ) / np.sum(weights)

    window_centers = np.asarray(
        [0.5 * (window[0] + window[1]) for window in window_offsets]
    )
    chirp_expected_window_frequencies = v0 + b * window_centers / np.pi
    chirp_window_frequency_means = window_frequency_means
    chirp_window_frequency_errors = (
        chirp_window_frequency_means - chirp_expected_window_frequencies[None, :]
    )

    envelope_a = chirp_probe_envelopes[0] - np.mean(chirp_probe_envelopes[0])
    envelope_b = chirp_probe_envelopes[1] - np.mean(chirp_probe_envelopes[1])
    correlation = np.correlate(envelope_b, envelope_a, mode="full")
    lag_samples = int(np.argmax(correlation) - (envelope_a.size - 1))
    chirp_measured_delay = np.asarray([lag_samples * sample_period])
    chirp_expected_delay = np.asarray([probe_x[1] - probe_x[0]])
    chirp_delay_error = np.abs(chirp_measured_delay - chirp_expected_delay)

    shifted_reference = np.interp(
        chirp_probe_times - chirp_expected_delay[0],
        chirp_probe_times,
        chirp_analytic_probe_fields[0].real,
        left=0,
        right=0,
    ) + 1j * np.interp(
        chirp_probe_times - chirp_expected_delay[0],
        chirp_probe_times,
        chirp_analytic_probe_fields[0].imag,
        left=0,
        right=0,
    )
    waveform_mask = chirp_probe_envelopes[1] > 0.05 * np.max(chirp_probe_envelopes[1])
    reference = shifted_reference[waveform_mask]
    observed = chirp_analytic_probe_fields[1, waveform_mask]
    scale = np.vdot(reference, observed) / np.vdot(reference, reference)
    chirp_shifted_waveform_error = np.asarray(
        [np.linalg.norm(observed - scale * reference) / np.linalg.norm(observed)]
    )

    center_trace = chirp_probe_fields[:, 1:2, :]
    chirp_y_uniformity_error = np.asarray(
        [
            np.linalg.norm(chirp_probe_fields[index] - center_trace[index])
            / np.linalg.norm(center_trace[index])
            for index in range(probe_x.size)
        ]
    )
    chirp_source_sample_times = np.linspace(t0 - 4, t0 + 4, 161)
    chirp_source_samples = np.asarray([chirp(time) for time in chirp_source_sample_times])
    chirp_source_envelope_error = np.asarray(
        [
            np.max(
                np.abs(
                    np.abs(chirp_source_samples)
                    - np.exp(-a * np.square(chirp_source_sample_times - t0))
                )
            )
        ]
    )

    if not np.all(np.diff(chirp_probe_times) > 0):
        raise RuntimeError("chirped-pulse probe times are not strictly increasing")
    if not np.array_equal(chirp_frequency_valid_counts, np.asarray([161.0, 161.0])):
        raise RuntimeError(
            "chirped-pulse frequency-valid interval has the wrong sample count: "
            f"{chirp_frequency_valid_counts.tolist()}"
        )
    if not np.allclose(
        chirp_frequency_valid_time_bounds,
        chirp_expected_frequency_valid_time_bounds,
        rtol=0.0,
        atol=0.5 * sample_period + 1e-8,
    ):
        raise RuntimeError(
            "chirped-pulse frequency-valid interval has the wrong endpoints: "
            f"observed={chirp_frequency_valid_time_bounds.tolist()}, "
            f"expected={chirp_expected_frequency_valid_time_bounds.tolist()}"
        )
    if np.min(np.linalg.norm(chirp_probe_fields, axis=(1, 2))) <= 1e-4:
        raise RuntimeError("chirped-pulse probe traces are empty")
    if np.any(chirp_window_frequency_means[:, 0] <= chirp_window_frequency_means[:, 1] + 0.2):
        raise RuntimeError("chirped-pulse probe does not contain a down-chirp")
    if np.max(np.abs(chirp_window_frequency_errors)) > 0.12:
        raise RuntimeError("chirped-pulse instantaneous frequency disagrees with theory")
    if chirp_delay_error[0] > 0.2:
        raise RuntimeError("chirped-pulse propagation delay disagrees with vacuum transit")
    if chirp_shifted_waveform_error[0] > 0.12:
        raise RuntimeError("chirped-pulse probes are not related by vacuum propagation")
    if np.max(chirp_y_uniformity_error) > 2e-3:
        raise RuntimeError("chirped-pulse planewave is not uniform in y")
    if chirp_source_envelope_error[0] > 1e-12:
        raise RuntimeError("chirped-pulse source envelope changed")
else:
    sim.run(
        mp.in_volume(
            mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy)),
            mp.at_every(2.7, mp.output_efield_z),
        ),
        until=run_until,
    )
