#!/usr/bin/env python3
"""Radiation pattern of Ex, Ey, and Ez dipole antennas in free space.

Tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#radiation-pattern-of-an-antenna
"""

import argparse
import math

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run all polarizations with bounded resolution and publish N2F observables",
)
parser.add_argument("--resolution", type=int, default=None)
parser.add_argument("--polar-points", type=int, default=None)
parser.add_argument("--farfield-radius", type=float, default=None)
args = parser.parse_args()

RESOLUTION_UM = args.resolution if args.resolution is not None else (25 if args.validation else 50)
PML_UM = 1.0
WAVELENGTH_UM = 1.0
NUM_POLAR = (
    args.polar_points
    if args.polar_points is not None
    else (64 if args.validation else 100)
)
FARFIELD_RADIUS_UM = (
    args.farfield_radius
    if args.farfield_radius is not None
    else (100 * WAVELENGTH_UM if args.validation else 1000 * WAVELENGTH_UM)
)
FARFIELD_RESOLUTION_UM = 1
GREENCYL_TOL = 1e-8

if RESOLUTION_UM <= 0 or NUM_POLAR < 16 or FARFIELD_RADIUS_UM <= 10:
    raise RuntimeError("invalid antenna-radiation validation controls")

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 2 * math.pi, NUM_POLAR)


def trapezoidal_integral(values, coordinates):
    """Use the nondeprecated NumPy 2 API while retaining NumPy 1 support."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=coordinates)
    return np.trapz(values, x=coordinates)


def radiation_pattern(sim: mp.Simulation, n2f_mon: mp.DftNear2Far):
    """Return radial power and every complex far-field component."""

    fields = np.zeros((NUM_POLAR, 6), dtype=np.complex128)
    for i, angle in enumerate(polar_rad):
        fields[i, :] = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                FARFIELD_RADIUS_UM * math.cos(angle),
                FARFIELD_RADIUS_UM * math.sin(angle),
                0,
            ),
            GREENCYL_TOL,
        )
    e_field = fields[:, :3]
    h_field = fields[:, 3:]
    flux_x = np.real(
        np.conj(e_field[:, 1]) * h_field[:, 2]
        - np.conj(e_field[:, 2]) * h_field[:, 1]
    )
    flux_y = np.real(
        np.conj(e_field[:, 2]) * h_field[:, 0]
        - np.conj(e_field[:, 0]) * h_field[:, 2]
    )
    signed_radial = flux_x * np.cos(polar_rad) + flux_y * np.sin(polar_rad)
    tangential = -flux_x * np.sin(polar_rad) + flux_y * np.cos(polar_rad)
    radial_magnitude = np.sqrt(np.square(flux_x) + np.square(flux_y))
    return radial_magnitude, signed_radial, tangential, fields


def antenna_radiation_pattern(dipole_polarization: int, return_details=False):
    """Return the radiation pattern for an Ex, Ey, or Ez point dipole."""

    if dipole_polarization not in (mp.Ex, mp.Ey, mp.Ez):
        raise ValueError("dipole_polarization must be Ex, Ey, or Ez.")

    cell_um = 4.0
    sxy = PML_UM + cell_um + PML_UM
    cell_size = mp.Vector3(sxy, sxy, 0)
    boundary_layers = [mp.PML(PML_UM)]
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            center=mp.Vector3(),
            component=dipole_polarization,
        )
    ]

    if dipole_polarization == mp.Ex:
        symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
    elif dipole_polarization == mp.Ey:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
    else:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        symmetries=symmetries,
    )
    n2f_regions = [
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0.5 * cell_um), size=mp.Vector3(cell_um, 0)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -0.5 * cell_um),
            size=mp.Vector3(cell_um, 0),
            weight=-1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0.5 * cell_um, 0), size=mp.Vector3(0, cell_um)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-0.5 * cell_um, 0),
            size=mp.Vector3(0, cell_um),
            weight=-1,
        ),
    ]
    nearfield_box = sim.add_near2far(frequency, 0, 1, *n2f_regions)
    flux_box = sim.add_flux(
        frequency,
        0,
        1,
        *[
            mp.FluxRegion(center=region.center, size=region.size, weight=region.weight)
            for region in n2f_regions
        ],
    )
    sim.run(until_after_sources=mp.stop_when_dft_decayed())
    flux_near = float(mp.get_fluxes(flux_box)[0])

    flux_far = float(
        nearfield_box.flux(
            mp.Y,
            mp.Volume(
                center=mp.Vector3(0, FARFIELD_RADIUS_UM, 0),
                size=mp.Vector3(2 * FARFIELD_RADIUS_UM, 0, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        - nearfield_box.flux(
            mp.Y,
            mp.Volume(
                center=mp.Vector3(0, -FARFIELD_RADIUS_UM, 0),
                size=mp.Vector3(2 * FARFIELD_RADIUS_UM, 0, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        + nearfield_box.flux(
            mp.X,
            mp.Volume(
                center=mp.Vector3(FARFIELD_RADIUS_UM, 0, 0),
                size=mp.Vector3(0, 2 * FARFIELD_RADIUS_UM, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        - nearfield_box.flux(
            mp.X,
            mp.Volume(
                center=mp.Vector3(-FARFIELD_RADIUS_UM, 0, 0),
                size=mp.Vector3(0, 2 * FARFIELD_RADIUS_UM, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
    )
    radial_flux, signed_radial_flux, tangential_flux, farfield = radiation_pattern(
        sim, nearfield_box
    )
    integration_flux = signed_radial_flux if args.validation else radial_flux
    flux_radiation_pattern = float(
        trapezoidal_integral(integration_flux * FARFIELD_RADIUS_UM, polar_rad)
    )
    print(
        f"flux:, {mp.component_name(dipole_polarization)} (dipole), "
        f"{flux_near:.6f} (near), {flux_far:.6f} (far), "
        f"{flux_radiation_pattern:.6f} (radiation pattern)"
    )
    details = {
        "radial": radial_flux,
        "signed_radial": signed_radial_flux,
        "tangential": tangential_flux,
        "farfield": farfield,
        "fluxes": np.asarray([flux_near, flux_far, flux_radiation_pattern]),
    }
    return details if return_details else radial_flux


if __name__ == "__main__":
    polarizations = [mp.Ex, mp.Ey, mp.Ez]
    results = [antenna_radiation_pattern(item, return_details=True) for item in polarizations]
    antenna_polarizations = np.asarray(polarizations, dtype=float)
    antenna_polar_angles = np.asarray(polar_rad)
    antenna_farfield_components = np.stack([item["farfield"] for item in results])
    antenna_radial_flux = np.stack([item["radial"] for item in results])
    antenna_signed_radial_flux = np.stack([item["signed_radial"] for item in results])
    antenna_tangential_flux = np.stack([item["tangential"] for item in results])
    antenna_fluxes = np.stack([item["fluxes"] for item in results])
    antenna_theory = np.stack(
        [np.sin(polar_rad) ** 2, np.cos(polar_rad) ** 2, np.ones(NUM_POLAR)]
    )
    antenna_relative_errors = []
    antenna_directional_errors = []
    for index in range(3):
        normalized = antenna_signed_radial_flux[index] / np.max(
            antenna_signed_radial_flux[index]
        )
        theory_error = np.linalg.norm(normalized - antenna_theory[index]) / np.linalg.norm(
            antenna_theory[index]
        )
        flux_scale = max(abs(antenna_fluxes[index, 0]), 1e-30)
        antenna_relative_errors.append(
            [
                theory_error,
                abs(antenna_fluxes[index, 1] - antenna_fluxes[index, 0]) / flux_scale,
                abs(antenna_fluxes[index, 2] - antenna_fluxes[index, 0]) / flux_scale,
            ]
        )
        radial_norm = max(float(np.linalg.norm(antenna_radial_flux[index])), 1e-30)
        antenna_directional_errors.append(
            [
                float(
                    np.linalg.norm(
                        antenna_signed_radial_flux[index] - antenna_radial_flux[index]
                    )
                    / radial_norm
                ),
                float(np.linalg.norm(antenna_tangential_flux[index]) / radial_norm),
            ]
        )
    antenna_relative_errors = np.asarray(antenna_relative_errors)
    antenna_directional_errors = np.asarray(antenna_directional_errors)

    if args.validation:
        if np.any(~np.isfinite(antenna_farfield_components)) or np.any(antenna_fluxes <= 0):
            raise RuntimeError("antenna radiation validation produced invalid fields or flux")
        if np.max(antenna_relative_errors[:, 0]) > 0.04:
            raise RuntimeError(
                "antenna radiation pattern disagrees with dipole theory: "
                f"{antenna_relative_errors[:, 0].tolist()}"
            )
        if np.max(antenna_relative_errors[:, 1:]) > 0.03:
            raise RuntimeError("antenna near/far/radiation fluxes do not close")
        if np.max(antenna_directional_errors) > 3e-4:
            raise RuntimeError(
                "antenna Poynting flux is not outward and radial: "
                f"{antenna_directional_errors.tolist()}"
            )
        outward_scale = np.maximum(np.max(antenna_radial_flux, axis=1), 1e-30)
        minimum_signed = np.min(antenna_signed_radial_flux, axis=1)
        if np.any(minimum_signed < -1e-5 * outward_scale):
            raise RuntimeError("antenna far fields contain inward radial power")
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(18, 6), ncols=3)
        for i, dipole_polarization in enumerate(polarizations):
            ax[i].plot(
                polar_rad,
                antenna_radial_flux[i] / max(antenna_radial_flux[i]),
                "b-",
                label="Meep",
            )
            ax[i].plot(polar_rad, antenna_theory[i], "r--", label="theory")
            ax[i].set_rmax(1)
            ax[i].set_rticks([0, 0.5, 1])
            ax[i].grid(True)
            ax[i].set_rlabel_position(22)
            ax[i].set_title(
                r"$\mathcal{J}$"
                f"$_{{{mp.direction_name(mp.component_direction(dipole_polarization))}}}$"
                " dipole"
            )
            ax[i].legend(loc="upper right")
        fig.subplots_adjust(wspace=0.2)
        if mp.am_master():
            fig.savefig("antenna_radiation_pattern.png", dpi=150, bbox_inches="tight")
