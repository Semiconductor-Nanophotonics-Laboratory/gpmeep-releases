import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

import meep as mp


def metal_cavity(w):
    resolution = 50
    sxy = 2
    dpml = 1
    sxy += 2 * dpml
    cell = mp.Vector3(sxy, sxy)

    pml_layers = [mp.PML(dpml)]
    a = 1
    t = 0.1
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
    ]

    geometry.append(
        mp.Block(
            center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, w, mp.inf), material=mp.air
        )
    )

    fcen = math.sqrt(0.5) / a
    df = 0.2
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3()
        )
    ]

    symmetries = [mp.Mirror(mp.Y)]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        resolution=resolution,
    )

    h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=500)

    m = h.modes[0]
    f = m.freq
    Q = m.Q
    Vmode = 0.25 * a * a
    ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)

    sim.reset_meep()

    T = 2 * Q * (1 / f)
    sim.run(mp.dft_ldos(f, 0, 1), until_after_sources=T)
    ldos_2 = sim.ldos_data[0]

    physical_values = np.asarray([f, Q, T, ldos_1, ldos_2])
    if not np.all(np.isfinite(physical_values)) or np.any(physical_values <= 0):
        raise RuntimeError(
            "metal-cavity resonance, Q, duration, and LDOS values must be "
            f"positive and finite: {physical_values.tolist()}"
        )

    return ldos_1, ldos_2, f, Q, T


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help=(
        "run the lowest-Q aperture while retaining the complete "
        "Harminv/reset/dft_ldos workflow"
    ),
)
args = parser.parse_args()

ws = np.asarray([0.4]) if args.validation else np.arange(0.2, 0.5, 0.1)
ldos_1 = np.zeros(len(ws))
ldos_2 = np.zeros(len(ws))
resonance_frequencies = np.zeros(len(ws))
quality_factors = np.zeros(len(ws))
second_run_times = np.zeros(len(ws))

for j in range(len(ws)):
    (
        ldos_1[j],
        ldos_2[j],
        resonance_frequencies[j],
        quality_factors[j],
        second_run_times[j],
    ) = metal_cavity(ws[j])
    print(f"ldos:, {ldos_1[j]}, {ldos_2[j]}")

ldos_relative_errors = np.abs(ldos_1 - ldos_2) / np.maximum(
    np.abs(ldos_1), np.abs(ldos_2)
)
if not np.all(ldos_relative_errors < 0.25):
    raise RuntimeError(
        "Harminv Q/V and direct dft_ldos estimates disagree: "
        f"{ldos_relative_errors.tolist()}"
    )

plt.figure(dpi=150)
plt.semilogy(1 / ws, ldos_1, "bo-", label="2Q/(πωV)")
plt.semilogy(1 / ws, ldos_2, "rs-", label="LDOS")
plt.xlabel("a/w")
plt.ylabel("2Q/(πωW) or LDOS")
plt.show()
