"""Radiation pattern of a dipole in vacuum using Brillouin-zone integration."""

import argparse
from enum import Enum
from typing import Tuple
import math

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
NUM_POLAR = 30
NUM_AZIMUTH = 50
FIELD_DECAY_THRESHOLD = 1e-6
FIELD_DECAY_PERIOD = 25

frequency = 1 / WAVELENGTH_UM


def planewave_in_vacuum(dipole_pol: str, kx: float, ky: float, kz: float) -> float:
    """
    Returns the Poynting flux of a linearly polarized planewave in vacuum.

    Args:
        dipole_pol: the polarization of the electric dipole. Either x or y.
        kx, ky, kz: the wavevector components of the planewave.

    Returns:
        The Poynting flux in z.
    """
    pml_um = 1.0
    air_um = 10.0
    size_z_um = pml_um + air_um + pml_um
    cell_size = mp.Vector3(0, 0, size_z_um)
    pml_layers = [mp.PML(thickness=pml_um, direction=mp.Z)]
    k_point = mp.Vector3(kx, ky, kz)

    if dipole_pol == "x":
        src_cmpt = mp.Ex
    elif dipole_pol == "y":
        src_cmpt = mp.Ey

    src_pt = mp.Vector3(0, 0, -0.5 * air_um)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            component=src_cmpt,
            center=src_pt,
            size=mp.Vector3(),
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        sources=sources,
        boundary_layers=pml_layers,
        k_point=k_point,
    )

    mon_pt = mp.Vector3(0, 0, 0.5 * air_um)
    dft_flux_z = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mon_pt, size=mp.Vector3(), direction=mp.Z),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, src_cmpt, mon_pt, FIELD_DECAY_THRESHOLD
        )
    )

    flux_z = mp.get_fluxes(dft_flux_z)[0]

    return flux_z


def spherical_to_cartesian(polar_rad, azimuth_rad) -> Tuple[float, float, float]:
    """Converts a point on the unit sphere from spherical to Cartesian coords.

    Args:
        polar_rad: polar angle of the point. 0° is +z.
        azimuth_rad: azimuthal angle of the point. 0° is +x.

    Returns:
        The x,y,z coordinates of the point as a 3-tuple.
    """
    x = np.sin(polar_rad) * np.cos(azimuth_rad)
    y = np.sin(polar_rad) * np.sin(azimuth_rad)
    z = np.cos(polar_rad)

    return x, y, z


def analytic_dipole_pattern(dipole_pol, polar_rad, azimuth_rad):
    """Returns the normalized free-space radiation law on an angular grid."""
    x_direction = np.sin(polar_rad[:, np.newaxis]) * np.cos(azimuth_rad)
    y_direction = np.sin(polar_rad[:, np.newaxis]) * np.sin(azimuth_rad)
    return 1.0 - (
        x_direction**2 if dipole_pol == "x" else y_direction**2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dipole_pol",
        type=str,
        nargs="?",
        default="x",
        choices=["x", "y"],
        help="polarization of the electric dipole (x or y)",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded Brillouin-zone CPU/CUDA physics oracle",
    )
    args = parser.parse_args()

    if args.validation:
        RESOLUTION_UM = 20
        NUM_POLAR = 7
        NUM_AZIMUTH = 8
        FIELD_DECAY_THRESHOLD = 1e-5
        FIELD_DECAY_PERIOD = 15

    # Radial flux is defined on the surface of a hemisphere.
    polar_rad = np.linspace(0, 0.5 * np.pi, NUM_POLAR)
    azimuth_rad = np.linspace(
        0, 2 * np.pi, NUM_AZIMUTH, endpoint=not args.validation
    )
    radial_flux = np.zeros((NUM_POLAR, NUM_AZIMUTH))
    sampled_mask = np.zeros((NUM_POLAR, NUM_AZIMUTH), dtype=bool)

    for i in range(NUM_POLAR):
        for j in range(NUM_AZIMUTH):
            # Every azimuth represents the same +z wavevector at the pole.
            # In validation mode, execute it once and copy the physical value
            # without assigning duplicate weight to the analytic error.
            if args.validation and i == 0 and j > 0:
                radial_flux[i, j] = radial_flux[i, 0]
                continue

            rx, ry, rz = spherical_to_cartesian(polar_rad[i], azimuth_rad[j])
            # Specify the components of the wavevector of the outgoing
            # planewave in vacuum.
            kx = frequency * rx
            ky = frequency * ry
            kz = frequency * rz

            # Skip wavevectors which are close to the light cone
            # due to poor absorption by PML (i.e. glancing-angle waves).
            if np.sqrt(kx**2 + ky**2) > (0.95 * frequency):
                continue

            flux_z = planewave_in_vacuum(args.dipole_pol, kx, ky, kz)
            radial_flux[i, j] = rz * flux_z
            sampled_mask[i, j] = True

    dipole_1d_analytic_pattern = analytic_dipole_pattern(
        args.dipole_pol, polar_rad, azimuth_rad
    )
    scale = np.max(radial_flux[sampled_mask])
    if scale <= 0:
        raise RuntimeError("dipole 1D validation has no positive radial flux")
    dipole_1d_normalized_pattern = radial_flux / scale
    dipole_1d_relative_error = np.asarray(
        [
            np.linalg.norm(
                dipole_1d_normalized_pattern[sampled_mask]
                - dipole_1d_analytic_pattern[sampled_mask]
            )
            / np.linalg.norm(dipole_1d_analytic_pattern[sampled_mask])
        ]
    )
    dipole_1d_polar_angles = np.asarray(polar_rad)
    dipole_1d_azimuth_angles = np.asarray(azimuth_rad)
    dipole_1d_sampled_mask = sampled_mask.astype(float)
    dipole_1d_radial_flux = np.asarray(radial_flux)

    if args.validation:
        control_polar = np.pi / 4
        control_azimuth = 0.0
        rx, ry, rz = spherical_to_cartesian(control_polar, control_azimuth)
        dipole_1d_y_control_flux = np.asarray(
            [
                rz
                * planewave_in_vacuum(
                    "y", frequency * rx, frequency * ry, frequency * rz
                )
            ]
        )
        polar_idx = int(np.argmin(np.abs(polar_rad - control_polar)))
        rotated_idx = int(np.argmin(np.abs(azimuth_rad - 0.5 * np.pi)))
        rotated_x_flux = radial_flux[polar_idx, rotated_idx]
        analytic_y = analytic_dipole_pattern(
            "y", np.asarray([control_polar]), np.asarray([control_azimuth])
        )[0, 0]
        analytic_rotated_x = dipole_1d_analytic_pattern[polar_idx, rotated_idx]
        expected_ratio = analytic_y / analytic_rotated_x
        dipole_1d_y_control_relative_error = np.asarray(
            [abs(dipole_1d_y_control_flux[0] / rotated_x_flux - expected_ratio)]
        )

    if args.validation:
        retained = (
            dipole_1d_polar_angles,
            dipole_1d_azimuth_angles,
            dipole_1d_sampled_mask,
            dipole_1d_radial_flux,
            dipole_1d_normalized_pattern,
            dipole_1d_analytic_pattern,
            dipole_1d_relative_error,
            dipole_1d_y_control_flux,
            dipole_1d_y_control_relative_error,
        )
        if not all(np.all(np.isfinite(values)) for values in retained):
            raise RuntimeError("dipole 1D validation produced nonfinite data")
        if np.count_nonzero(sampled_mask) != 1 + 4 * NUM_AZIMUTH:
            raise RuntimeError("dipole 1D validation sampled an unexpected light cone")
        if dipole_1d_relative_error[0] > 0.04:
            raise RuntimeError(
                "dipole 1D radiation disagrees with the analytic pattern: "
                f"{dipole_1d_relative_error[0]}"
            )
        if dipole_1d_y_control_flux[0] <= 0:
            raise RuntimeError("dipole 1D y-control flux is not outward and positive")
        if dipole_1d_y_control_relative_error[0] > 0.02:
            raise RuntimeError(
                "dipole 1D y-control violates rotational symmetry: "
                f"{dipole_1d_y_control_relative_error[0]}"
            )

    if mp.am_master() and not args.validation:
        np.savez(
            "dipole_radiation_pattern.npz",
            NUM_AZIMUTH=NUM_AZIMUTH,
            NUM_POLAR=NUM_POLAR,
            RESOLUTION_UM=RESOLUTION_UM,
            WAVELENGTH_UM=WAVELENGTH_UM,
            azimuth_rad=azimuth_rad,
            dipole_pol=args.dipole_pol,
            polar_rad=polar_rad,
            radial_flux=radial_flux,
        )
