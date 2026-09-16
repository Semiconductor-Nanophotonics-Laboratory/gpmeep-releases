"""Plot Faraday rotation or emit deterministic gpmeep validation metrics."""

import argparse
import json

import meep as mp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="skip plotting and emit deterministic CPU/CUDA comparison metrics",
)
args = parser.parse_args()

## Parameters for a gyrotropic Lorentzian medium
epsn = 1.5  # background permittivity
f0 = 1.0  # natural frequency
gamma = 1e-6  # damping rate
sn = 0.1  # sigma parameter
b0 = 0.15  # magnitude of bias vector

susc = [
    mp.GyrotropicLorentzianSusceptibility(
        frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0, 0, b0)
    )
]
mat = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)

## Set up and run the Meep simulation:
tmax = 100
L = 20.0
cell = mp.Vector3(0, 0, L)
fsrc, src_z = 0.8, -8.5
pml_layers = [mp.PML(thickness=1.0, direction=mp.Z)]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=fsrc),
        component=mp.Ex,
        center=mp.Vector3(0, 0, src_z),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    boundary_layers=pml_layers,
    default_material=mat,
    resolution=50,
)
sim.run(until=tmax)

ex_data = sim.get_efield_x().real
ey_data = sim.get_efield_y().real

z = np.linspace(-L / 2, L / 2, len(ex_data))

## Comparison with analytic result:
dfsq = f0**2 - 1j * fsrc * gamma - fsrc**2
eperp = epsn + sn * f0**2 * dfsq / (dfsq**2 - (fsrc * b0) ** 2)
eta = sn * f0**2 * fsrc * b0 / (dfsq**2 - (fsrc * b0) ** 2)

k_gyro = 2 * np.pi * fsrc * np.sqrt(0.5 * (eperp - np.sqrt(eperp**2 - eta**2)))
Ex_theory = 0.37 * np.cos(k_gyro * (z - src_z)).real
Ey_theory = 0.37 * np.sin(k_gyro * (z - src_z)).real

if args.validation:
    sample_positions = (-4.0, 0.0, 4.0, 8.0)
    sample_indices = [
        int(np.argmin(np.abs(z - position)))
        for position in sample_positions
    ]
    metrics = {
        "ex": [float(ex_data[index]) for index in sample_indices],
        "ey": [float(ey_data[index]) for index in sample_indices],
        "field_l2": [
            float(np.linalg.norm(ex_data)),
            float(np.linalg.norm(ey_data)),
        ],
    }
    print(
        "gpmeep-example-metrics:"
        + json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    )
else:
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(z, ex_data, label="Ex")
    plt.plot(z, ey_data, label="Ey")
    plt.xlim(-L / 2, L / 2)
    plt.xlabel("z")
    plt.legend()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(z, ex_data, label="Ex (MEEP)")
    plt.plot(z, Ex_theory, "k--")
    plt.plot(z, -Ex_theory, "k--", label="Ex envelope (theory)")
    plt.xlim(-L / 2, L / 2)
    plt.xlabel("z")
    plt.legend(loc="lower right")

    plt.subplot(2, 1, 2)
    plt.plot(z, ey_data, label="Ey (MEEP)")
    plt.plot(z, Ey_theory, "k--")
    plt.plot(z, -Ey_theory, "k--", label="Ey envelope (theory)")
    plt.xlim(-L / 2, L / 2)
    plt.xlabel("z")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
