#!/usr/bin/env python3
"""Waveguide-taper reflectance from mode coefficients and Poynting flux."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import meep as mp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="retain the checked-in four-taper workflow and publish physical references",
)
parser.add_argument("--resolution", type=int, default=None)
args = parser.parse_args()

resolution = args.resolution if args.resolution is not None else 25
if resolution <= 0:
    raise RuntimeError("resolution must be positive")

w1 = 1.0
w2 = 2.0
Lw = 10.0
Lts = [2**m for m in range(4)]
dair = 3.0
dpml_x = 6.0
dpml_y = 2.0
sy = dpml_y + dair + w2 + dair + dpml_y
Si = mp.Medium(epsilon=12.0)
boundary_layers = [mp.PML(dpml_x, direction=mp.X), mp.PML(dpml_y, direction=mp.Y)]
lcen = 6.67
fcen = 1 / lcen
symmetries = [mp.Mirror(mp.Y)]

R_coeffs = []
R_flux = []
mode_incident_coefficients = []
mode_taper_coefficients = []
mode_incident_fluxes = []
mode_taper_fluxes = []

for Lt in Lts:
    sx = dpml_x + Lw + Lt + Lw + dpml_x
    cell_size = mp.Vector3(sx, sy, 0)
    src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
            center=src_pt,
            size=mp.Vector3(y=sy - 2 * dpml_y),
            eig_match_freq=True,
            eig_parity=mp.ODD_Z + mp.EVEN_Y,
        )
    ]

    vertices = [
        mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
        mp.Vector3(0.5 * sx + 1, 0.5 * w1),
        mp.Vector3(0.5 * sx + 1, -0.5 * w1),
        mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
    ]
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
        sources=sources,
        symmetries=symmetries,
    )
    mon_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.7 * Lw)
    flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
    )
    decay_tolerance = 1e-9
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mon_pt, decay_tolerance
        )
    )
    res = sim.get_eigenmode_coefficients(
        flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
    )
    incident_coeffs = np.asarray(res.alpha).copy()
    incident_flux = float(mp.get_fluxes(flux)[0])
    incident_flux_data = sim.get_flux_data(flux)
    sim.reset_meep()

    vertices = [
        mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
        mp.Vector3(-0.5 * Lt, 0.5 * w1),
        mp.Vector3(0.5 * Lt, 0.5 * w2),
        mp.Vector3(0.5 * sx + 1, 0.5 * w2),
        mp.Vector3(0.5 * sx + 1, -0.5 * w2),
        mp.Vector3(0.5 * Lt, -0.5 * w2),
        mp.Vector3(-0.5 * Lt, -0.5 * w1),
        mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
    ]
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
        sources=sources,
        symmetries=symmetries,
    )
    flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
    )
    sim.load_minus_flux_data(flux, incident_flux_data)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mon_pt, decay_tolerance
        )
    )
    res2 = sim.get_eigenmode_coefficients(
        flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
    )
    taper_coeffs = np.asarray(res2.alpha).copy()
    taper_flux = float(mp.get_fluxes(flux)[0])

    coefficient_reflectance = (
        abs(taper_coeffs[0, 0, 1]) ** 2 / abs(incident_coeffs[0, 0, 0]) ** 2
    )
    flux_reflectance = -taper_flux / incident_flux
    R_coeffs.append(float(coefficient_reflectance))
    R_flux.append(float(flux_reflectance))
    mode_incident_coefficients.append(incident_coeffs)
    mode_taper_coefficients.append(taper_coeffs)
    mode_incident_fluxes.append(incident_flux)
    mode_taper_fluxes.append(taper_flux)
    print(f"refl:, {Lt}, {R_coeffs[-1]:.8f}, {R_flux[-1]:.8f}")

mode_taper_lengths = np.asarray(Lts, dtype=float)
mode_coefficient_reflectance = np.asarray(R_coeffs, dtype=float)
mode_flux_reflectance = np.asarray(R_flux, dtype=float)
mode_incident_coefficients = np.stack(mode_incident_coefficients)
mode_taper_coefficients = np.stack(mode_taper_coefficients)
mode_incident_coefficient_magnitudes = np.abs(mode_incident_coefficients)
mode_taper_coefficient_magnitudes = np.abs(mode_taper_coefficients)
mode_incident_fluxes = np.asarray(mode_incident_fluxes, dtype=float)
mode_taper_fluxes = np.asarray(mode_taper_fluxes, dtype=float)
mode_method_absolute_error = np.abs(mode_coefficient_reflectance - mode_flux_reflectance)
mode_method_relative_error = mode_method_absolute_error / np.maximum(
    mode_coefficient_reflectance, 1e-12
)
# Independent double-precision CPU values for this exact checked-in geometry,
# resolution, source, and 1e-9 decay contract.  The Poynting result includes
# all backward power whereas the coefficient result is the fundamental mode,
# so their relative difference is physical for the shortest abrupt taper.
mode_reference_coefficient_reflectance = np.asarray(
    [0.00033425, 0.00008612, 0.00004647, 0.00002143]
)
mode_reference_flux_reflectance = np.asarray(
    [0.00041190, 0.00009790, 0.00004959, 0.00002189]
)
mode_reference_relative_error = np.stack(
    [
        np.abs(
            mode_coefficient_reflectance - mode_reference_coefficient_reflectance
        )
        / mode_reference_coefficient_reflectance,
        np.abs(mode_flux_reflectance - mode_reference_flux_reflectance)
        / mode_reference_flux_reflectance,
    ]
)
mode_initial_pair_slope = np.asarray(
    [
        float(
            np.log(mode_coefficient_reflectance[1] / mode_coefficient_reflectance[0])
            / np.log(mode_taper_lengths[1] / mode_taper_lengths[0])
        )
    ]
)
mode_log_slope = np.asarray(
    [
        float(
            np.polyfit(
                np.log(mode_taper_lengths[-3:]),
                np.log(mode_coefficient_reflectance[-3:]),
                1,
            )[0]
        )
    ]
)

if args.validation:
    if np.any(mode_incident_fluxes <= 0) or np.any(~np.isfinite(mode_incident_coefficients)):
        raise RuntimeError("mode decomposition produced invalid incident power")
    if np.any(mode_coefficient_reflectance <= 0) or np.any(mode_coefficient_reflectance >= 1):
        raise RuntimeError("mode decomposition produced non-physical reflectance")
    if np.max(mode_method_absolute_error) > 1e-4 or np.max(mode_method_relative_error) > 0.3:
        raise RuntimeError("mode-decomposition reflectance methods changed unexpectedly")
    if np.max(mode_reference_relative_error) > 0.01:
        raise RuntimeError("mode-decomposition reflectance disagrees with the FP64 oracle")
    if np.any(np.diff(mode_coefficient_reflectance) > 1e-8):
        raise RuntimeError("waveguide-taper reflectance failed to decrease")
    if not -2.1 < mode_initial_pair_slope[0] < -1.8:
        raise RuntimeError("waveguide-taper leading inverse-square trend changed")
elif mp.am_master():
    plt.figure()
    plt.loglog(Lts, R_coeffs, "bo-", label="mode decomposition")
    plt.loglog(Lts, R_flux, "ro-", label="Poynting flux")
    plt.loglog(
        Lts,
        [0.005 / Lt**2 for Lt in Lts],
        "k-",
        label=r"quadratic reference (1/Lt$^2$)",
    )
    plt.legend(loc="upper right")
    plt.xlabel("taper length Lt (μm)")
    plt.ylabel("reflectance")
    plt.show()
