#!/usr/bin/env python3
# Material dispersion example, from the Meep tutorial.  Here, we simply
# simulate homogenous space filled with a dispersive material, and compute
# its modes as a function of wavevector k.  Since omega/c = k/n, we can
# extract the dielectric function epsilon(omega) = (ck/omega)^2.
import argparse

import numpy as np

import meep as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="use a bounded k-point sweep while retaining both dispersive poles",
)
parser.add_argument("--k-interp", type=int, default=None)
parser.add_argument("--run-time", type=float, default=None)
args = parser.parse_args()

cell = mp.Vector3()
resolution = 20

# We'll use a dispersive material with two polarization terms, just for
# illustration.  The first one is a strong resonance at omega=1.1,
# which leads to a polaritonic gap in the dispersion relation.  The second
# one is a weak resonance at omega=0.5, whose main effect is to add a
# small absorption loss around that frequency.

susceptibilities = [
    mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5),
    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.1, sigma=2e-5),
]

default_material = mp.Medium(epsilon=2.25, E_susceptibilities=susceptibilities)

fcen = 1.0
df = 2.0

sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3())
]

kmin = 0.3
kmax = 2.2
k_interp = (
    args.k_interp if args.k_interp is not None else (5 if args.validation else 99)
)
run_time = (
    args.run_time if args.run_time is not None else (100 if args.validation else 200)
)
if k_interp < 1 or run_time <= 0:
    raise RuntimeError("k interpolation and run time must be positive")

kpts = mp.interpolate(k_interp, [mp.Vector3(kmin), mp.Vector3(kmax)])

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    default_material=default_material,
    resolution=resolution,
)

all_freqs = sim.run_k_points(run_time, kpts)  # a list of lists of frequencies

k_values = np.asarray([v.x for v in kpts])
dispersion_mode_counts = np.asarray([len(fs) for fs in all_freqs], dtype=np.int64)
dispersion_mode_offsets = np.concatenate(
    ([0], np.cumsum(dispersion_mode_counts, dtype=np.int64))
)
dispersion_frequencies = np.asarray(
    [frequency for frequencies in all_freqs for frequency in frequencies],
    dtype=np.complex128,
)
dispersion_epsilons = np.asarray(
    [
        (kx / frequency) ** 2
        for frequencies, kx in zip(all_freqs, k_values)
        for frequency in frequencies
    ],
    dtype=np.complex128,
)
expected_mode_counts = np.full(k_values.shape, 2, dtype=np.int64)
expected_mode_offsets = np.arange(0, 2 * len(k_values) + 1, 2, dtype=np.int64)
if (
    dispersion_frequencies.size == 0
    or np.any(dispersion_mode_counts <= 0)
    or not np.all(np.isfinite(dispersion_frequencies))
    or not np.all(np.isfinite(dispersion_epsilons))
):
    raise RuntimeError("material-dispersion sweep produced missing or non-finite modes")

if args.validation and (
    not np.array_equal(dispersion_mode_counts, expected_mode_counts)
    or not np.array_equal(dispersion_mode_offsets, expected_mode_offsets)
):
    raise RuntimeError(
        "material-dispersion sweep violated the two-pole rectangular mode topology "
        f"(counts={dispersion_mode_counts}, offsets={dispersion_mode_offsets})"
    )

if args.validation:
    dispersion_branches = dispersion_frequencies.reshape(len(k_values), 2)
    if (
        np.any(dispersion_branches.real <= 0)
        or np.any(dispersion_branches.imag >= 0)
        or np.any(dispersion_branches[:, 1].real <= dispersion_branches[:, 0].real)
        or np.any(np.diff(dispersion_branches.real, axis=0) <= 0)
    ):
        raise RuntimeError("material-dispersion branch ordering is non-physical")

for fs, kx in zip(all_freqs, k_values):
    for f in fs:
        print(f"eps:, {f.real:.6g}, {f.imag:.6g}, {(kx / f) ** 2:.6g}")
