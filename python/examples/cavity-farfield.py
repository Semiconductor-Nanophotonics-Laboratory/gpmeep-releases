#!/usr/bin/env python3
"""Compare direct and near-to-far fields above a photonic-crystal cavity."""

import argparse

import matplotlib
import numpy as np

import meep as mp

matplotlib.use("agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run a bounded cavity and publish complete direct/N2F line observables",
)
args = parser.parse_args()

resolution = 15 if args.validation else 20  # pixels/μm

fcen = 0.25  # pulse center frequency
df = 0.2  # pulse width (in frequency)

eps = 13  # dielectric constant of waveguide
w = 1.2  # width of waveguide
r = 0.36  # radius of holes
d = 1.4  # defect spacing (ordinary spacing = 1)
N = 3  # number of holes on either side of defect

dpad = 8 if args.validation else 32  # padding between last hole and PML edge
dpml = 0.5 / (fcen - 0.5 * df)  # PML thickness (> half the largest wavelength)
sx = 2 * (dpad + dpml + N) + d - 1  # size of cell in x direction

d1 = 0.2  # y-distance from waveguide edge to near2far surface
d2 = 2.0  # y-distance from near2far surface to far-field line
sy = w + 2 * (d1 + d2 + dpml)  # size of cell in y direction (perpendicular to wvg.)

cell = mp.Vector3(sx, sy, 0)

geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
        material=mp.Medium(epsilon=eps),
    )
]

for i in range(N):
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / -2 - i)))

pml_layers = [mp.PML(dpml)]

sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df), component=mp.Hz, center=mp.Vector3()
    )
]

symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    symmetries=symmetries,
    boundary_layers=pml_layers,
    resolution=resolution,
)

nearfield = sim.add_near2far(
    fcen,
    0,
    1,
    mp.Near2FarRegion(mp.Vector3(0, 0.5 * w + d1), size=mp.Vector3(sx - 2 * dpml)),
    mp.Near2FarRegion(
        mp.Vector3(-0.5 * sx + dpml, 0.5 * w + 0.5 * d1),
        size=mp.Vector3(0, d1),
        weight=-1.0,
    ),
    mp.Near2FarRegion(
        mp.Vector3(0.5 * sx - dpml, 0.5 * w + 0.5 * d1), size=mp.Vector3(0, d1)
    ),
)

mon = sim.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Hz] if args.validation else [mp.Hz],
    fcen,
    0,
    1,
    center=mp.Vector3(0, 0.5 * w + d1 + d2),
    size=mp.Vector3(sx - 2 * (dpad + dpml), 0),
)

sim.run(until_after_sources=400 if args.validation else mp.stop_when_dft_decayed())

if not args.validation:
    sim.plot2D()
if not args.validation and mp.am_master():
    plt.savefig(
        f"cavity_farfield_plot2D_dpad{dpad}_{d1}_{d2}.png", bbox_inches="tight", dpi=150
    )

Hz_mon = np.asarray(sim.get_dft_array(mon, mp.Hz, 0), dtype=complex).reshape(-1)
if args.validation:
    Ex_mon = np.asarray(sim.get_dft_array(mon, mp.Ex, 0), dtype=complex).reshape(-1)
    Ey_mon = np.asarray(sim.get_dft_array(mon, mp.Ey, 0), dtype=complex).reshape(-1)
else:
    # Keep the upstream default monitor Hz-only; these placeholders are used
    # only by validation bookkeeping whose gates remain disabled here.
    Ex_mon = np.zeros_like(Hz_mon)
    Ey_mon = np.zeros_like(Hz_mon)

(x, y, z, metadata_weights) = sim.get_array_metadata(dft_cell=mon)
x = np.asarray(x, dtype=float).reshape(-1)
y = np.asarray(y, dtype=float).reshape(-1)
z = np.asarray(z, dtype=float).reshape(-1)

ff_all = []
for xc in x:
    ff_pt = sim.get_farfield(nearfield, mp.Vector3(xc, y[0]))
    ff_all.append(ff_pt)
ff_all = np.asarray(ff_all, dtype=complex)
# Retain the upstream Hz-only plotting variable and output behavior.
ff = ff_all[:, 5]

cavity_farfield_component_codes = np.asarray([mp.Ex, mp.Ey, mp.Hz], dtype=float)
cavity_farfield_coordinates = np.column_stack(
    [x, np.full_like(x, y[0]), np.full_like(x, z[0])]
)
cavity_farfield_parameters = np.asarray(
    [resolution, fcen, df, eps, w, r, d, N, dpad, dpml, d1, d2, sx, sy],
    dtype=float,
)
cavity_direct_fields = np.column_stack([Ex_mon, Ey_mon, Hz_mon])
cavity_n2f_fields = ff_all[:, [0, 1, 5]]
cavity_direct_magnitudes = np.abs(cavity_direct_fields)
cavity_n2f_magnitudes = np.abs(cavity_n2f_fields)
cavity_direct_intensity = np.sum(np.square(cavity_direct_magnitudes), axis=1)
cavity_n2f_intensity = np.sum(np.square(cavity_n2f_magnitudes), axis=1)

# For the upper line in this TM problem, (E x H*)_y = -Ex*conj(Hz).
# Retain the sign rather than comparing only a nonnegative magnitude so that
# a flipped or tangential field cannot masquerade as outward radiation.
cavity_direct_signed_poynting_y = -np.real(
    np.conj(cavity_direct_fields[:, 0]) * cavity_direct_fields[:, 2]
)
cavity_n2f_signed_poynting_y = -np.real(
    np.conj(cavity_n2f_fields[:, 0]) * cavity_n2f_fields[:, 2]
)


def relative_l2(reference, candidate):
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1e-30)
    )


cavity_field_relative_errors = np.asarray(
    [
        relative_l2(cavity_direct_fields[:, index], cavity_n2f_fields[:, index])
        for index in range(3)
    ]
)
cavity_magnitude_relative_errors = np.asarray(
    [
        relative_l2(
            cavity_direct_magnitudes[:, index], cavity_n2f_magnitudes[:, index]
        )
        for index in range(3)
    ]
)
cavity_intensity_relative_error = np.asarray(
    [relative_l2(cavity_direct_intensity, cavity_n2f_intensity)]
)
cavity_poynting_relative_error = np.asarray(
    [
        relative_l2(
            cavity_direct_signed_poynting_y, cavity_n2f_signed_poynting_y
        )
    ]
)
integrate_line = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
cavity_integrated_signed_poynting = np.asarray(
    [
        float(integrate_line(cavity_direct_signed_poynting_y, x=x)),
        float(integrate_line(cavity_n2f_signed_poynting_y, x=x)),
    ]
)
cavity_integrated_poynting_relative_error = np.asarray(
    [
        abs(
            cavity_integrated_signed_poynting[1]
            - cavity_integrated_signed_poynting[0]
        )
        / max(abs(cavity_integrated_signed_poynting[0]), 1e-30)
    ]
)

if args.validation:
    # Meep includes both Yee-grid endpoints and rounds this 6.4-unit line to
    # 98 samples at the fixed validation resolution.
    expected_points = 98
    if x.size != expected_points or cavity_farfield_coordinates.shape != (x.size, 3):
        raise RuntimeError(
            "cavity far-field line topology changed: "
            f"x={x.size}, expected={expected_points}, "
            f"coordinates={cavity_farfield_coordinates.shape}"
        )
    if x.size < 64 or np.any(np.diff(x) <= 0) or x[0] >= -3 or x[-1] <= 3:
        raise RuntimeError("cavity far-field line does not span the physical cavity")
    if cavity_direct_fields.shape != (x.size, 3) or cavity_n2f_fields.shape != (
        x.size,
        3,
    ):
        raise RuntimeError("cavity direct/N2F field component topology changed")
    observable_arrays = (
        cavity_farfield_coordinates,
        cavity_direct_fields,
        cavity_n2f_fields,
        cavity_direct_intensity,
        cavity_n2f_intensity,
        cavity_direct_signed_poynting_y,
        cavity_n2f_signed_poynting_y,
    )
    if any(np.any(~np.isfinite(values)) for values in observable_arrays):
        raise RuntimeError("cavity direct/N2F observables contain nonfinite values")
    direct_component_norms = np.linalg.norm(cavity_direct_fields, axis=0)
    n2f_component_norms = np.linalg.norm(cavity_n2f_fields, axis=0)
    if np.any(direct_component_norms <= 0.1) or np.any(n2f_component_norms <= 0.1):
        raise RuntimeError("cavity direct/N2F component fields are trivially zero")
    if np.max(cavity_field_relative_errors) > 0.04:
        raise RuntimeError(
            "cavity complex direct/N2F fields disagree: "
            f"{cavity_field_relative_errors.tolist()}"
        )
    if np.max(cavity_magnitude_relative_errors) > 0.03:
        raise RuntimeError("cavity direct/N2F field magnitudes disagree")
    if cavity_intensity_relative_error[0] > 0.04:
        raise RuntimeError("cavity direct/N2F intensities disagree")
    if cavity_poynting_relative_error[0] > 0.05:
        raise RuntimeError("cavity direct/N2F signed Poynting profiles disagree")
    if np.any(cavity_integrated_signed_poynting <= 1e-4):
        raise RuntimeError("cavity direct/N2F radiation is not outward and nonzero")
    if cavity_integrated_poynting_relative_error[0] > 0.03:
        raise RuntimeError("cavity direct/N2F integrated Poynting flux does not close")

if not args.validation and mp.am_master():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.real(Hz_mon), "bo-", label="DFT")
    plt.plot(np.real(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("real(Hz)")

    plt.subplot(1, 3, 2)
    plt.plot(np.imag(Hz_mon), "bo-", label="DFT")
    plt.plot(np.imag(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("imag(Hz)")

    plt.subplot(1, 3, 3)
    plt.plot(np.abs(Hz_mon), "bo-", label="DFT")
    plt.plot(np.abs(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("|Hz|")

    plt.suptitle(
        f"comparison of near2far and actual DFT fields\n dpad={dpad}, d1={d1}, d2={d2}"
    )
    plt.subplots_adjust(wspace=0.6)
    plt.savefig(
        f"test_Hz_dft_vs_n2f_res{resolution}_dpad{dpad}_d1{d1}_d2{d2}.png",
        bbox_inches="tight",
        dpi=150,
    )
