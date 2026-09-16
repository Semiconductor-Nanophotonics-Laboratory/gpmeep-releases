"""Radiated flux of a lossless dielectric disc in cylindrical coordinates.

Tutorial Reference:

https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#radiation-pattern-of-a-disc-in-cylindrical-coordinates
"""

import argparse
import math
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RESOLUTION_UM = 100
PML_UM = 0.5
AIR_UM = 1.0
DISC_RADIUS_UM = 1.2
N_DISC = 2.4
WAVELENGTH_UM = 1.0
NUM_AZIMUTH = 50
NUM_POLAR = 100
FARFIELD_RADIUS_UM = 1e6 * WAVELENGTH_UM
FIELD_DECAY_TOL = 1e-8
FIELD_DECAY_PERIOD = 50
GREENCYL_TOL = 1e-8

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 0.5 * math.pi, NUM_POLAR)
azimuth_rad = np.linspace(0, 2 * math.pi, NUM_AZIMUTH)


def plot_radiation_pattern_polar(
    radial_flux: np.ndarray, polar_angles: np.ndarray = polar_rad
):
    """Plots the radiation pattern in polar coordinates.

    The angles increase clockwise with zero at the top (+z direction).

    Args:
        radial_flux: radial flux of the far fields in polar coordinates.
    """
    radial_flux = np.asarray(radial_flux)
    polar_angles = np.asarray(polar_angles)
    if radial_flux.shape != polar_angles.shape:
        raise ValueError("radial flux and polar angle arrays must have matching shapes")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(
        polar_angles,
        radial_flux,
        "b-",
    )
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0.5 * math.pi)
    ax.set_thetalim(0, 0.5 * math.pi)
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.set_ylabel("radial flux (a.u.)")
    ax.set_title("radiation pattern in polar coordinates")

    if mp.am_master():
        fig.savefig(
            "disc_radiation_pattern_polar.png",
            dpi=150,
            bbox_inches="tight",
        )


def plot_radiation_pattern_3d(
    radial_flux: np.ndarray, polar_angles: np.ndarray = polar_rad
):
    """Plots the radiation pattern in 3d Cartesian coordinates.

    Args:
        radial_flux: radial flux of the far fields in polar coordinates.
    """
    radial_flux = np.asarray(radial_flux)
    polar_angles = np.asarray(polar_angles)
    if radial_flux.shape != polar_angles.shape:
        raise ValueError("radial flux and polar angle arrays must have matching shapes")
    num_polar = polar_angles.size
    x_coord = np.zeros((num_polar, NUM_AZIMUTH))
    y_coord = np.zeros((num_polar, NUM_AZIMUTH))
    z_coord = np.zeros((num_polar, NUM_AZIMUTH))

    for i in range(num_polar):
        for j in range(NUM_AZIMUTH):
            x_coord[i, j] = (
                radial_flux[i] * np.sin(polar_angles[i]) * np.cos(azimuth_rad[j])
            )
            y_coord[i, j] = (
                radial_flux[i] * np.sin(polar_angles[i]) * np.sin(azimuth_rad[j])
            )
            z_coord[i, j] = radial_flux[i] * np.cos(polar_angles[i])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.plot_surface(x_coord, y_coord, z_coord, cmap="inferno")
    ax.set_title("radiation pattern in 3d")
    ax.set_box_aspect((np.amax(x_coord), np.amax(y_coord), np.amax(z_coord)))
    ax.set_zlabel("radial flux (a.u.)")
    ax.set(xticklabels=[], yticklabels=[])

    if mp.am_master():
        fig.savefig(
            "disc_radiation_pattern_3d.png",
            dpi=150,
            bbox_inches="tight",
        )


def radiation_pattern(
    sim: mp.Simulation,
    n2f_mon: mp.DftNear2Far,
    *,
    polar_angles: np.ndarray = polar_rad,
    farfield_radius_um: float = FARFIELD_RADIUS_UM,
    greencyl_tolerance: float = GREENCYL_TOL,
    return_details: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Computes the radiation pattern from the far fields.

    Args:
        sim: a `Simulation` object.
        n2f_mon: a `DftNear2Far` object returned by `Simulation.add_near2far`.

    Returns:
        Array of radial Poynting flux, one for each point on the circumference of
        a quarter circle with angular range of [0, π/2] rad. 0 rad is the +z
        direction and π/2 is +r.
    """
    polar_angles = np.asarray(polar_angles, dtype=float)
    if (
        polar_angles.ndim != 1
        or polar_angles.size < 2
        or np.any(~np.isfinite(polar_angles))
        or np.any(np.diff(polar_angles) <= 0)
        or polar_angles[0] < 0
        or polar_angles[-1] > 0.5 * math.pi
    ):
        raise ValueError(
            "polar_angles must be a finite increasing 1d quarter-circle grid"
        )
    if farfield_radius_um <= 0 or greencyl_tolerance <= 0:
        raise ValueError(
            "far-field radius and cylindrical Green tolerance must be positive"
        )

    e_field = np.zeros((polar_angles.size, 3), dtype=np.complex128)
    h_field = np.zeros((polar_angles.size, 3), dtype=np.complex128)
    for i, angle in enumerate(polar_angles):
        far_field = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                farfield_radius_um * math.sin(angle),
                0,
                farfield_radius_um * math.cos(angle),
            ),
            greencyl_tolerance,
        )
        e_field[i, :] = [far_field[j] for j in range(3)]
        h_field[i, :] = [far_field[j + 3] for j in range(3)]

    flux_x = np.real(
        np.conj(e_field[:, 1]) * h_field[:, 2] - np.conj(e_field[:, 2]) * h_field[:, 1]
    )
    flux_z = np.real(
        np.conj(e_field[:, 0]) * h_field[:, 1] - np.conj(e_field[:, 1]) * h_field[:, 0]
    )
    flux_r = np.sqrt(np.square(flux_x) + np.square(flux_z))

    if return_details:
        return {
            "e_field": e_field,
            "h_field": h_field,
            "flux_x": flux_x,
            "flux_z": flux_z,
            "flux_radial_signed": flux_x * np.sin(polar_angles)
            + flux_z * np.cos(polar_angles),
            "flux_radial_magnitude": flux_r,
        }

    return flux_r


def disc_radiated_flux(
    disc_um: float,
    source_zpos: float,
    *,
    resolution_um: int = RESOLUTION_UM,
    num_polar: int = NUM_POLAR,
    field_decay_period: float = FIELD_DECAY_PERIOD,
    field_decay_tolerance: float = FIELD_DECAY_TOL,
    greencyl_tolerance: float = GREENCYL_TOL,
    plot: bool = True,
    return_details: bool = False,
) -> Union[Tuple[float, float], Dict[str, object]]:
    """Computes the  radiated flux from a "ring" current source within a disc.

    Args:
        disc_um: thickness of dielectric disc.
        source_zpos: height of the dipole source above the ground plane as
            a fraction of disc_um.

    Returns:
        A 2-tuple of the total flux computed using the near and far fields,
        respectively.
    """
    if disc_um <= 0 or not 0 < source_zpos < 1:
        raise ValueError("disc thickness and interior source position must be positive")
    if resolution_um <= 0 or num_polar < 8:
        raise ValueError("resolution and polar sample count are too small")
    if field_decay_period <= 0 or not 0 < field_decay_tolerance < 1:
        raise ValueError("invalid field-decay controls")
    if greencyl_tolerance <= 0:
        raise ValueError("cylindrical Green tolerance must be positive")

    cell_r_um = 6.0
    sr = cell_r_um + PML_UM
    sz = disc_um + AIR_UM + PML_UM
    cell_size = mp.Vector3(sr, 0, sz)

    boundary_layers = [
        mp.PML(PML_UM, direction=mp.R),
        mp.PML(PML_UM, direction=mp.Z, side=mp.High),
    ]

    src_cmpt = mp.Er
    src_pt = mp.Vector3(0.5 * DISC_RADIUS_UM, 0, -0.5 * sz + source_zpos * disc_um)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            component=src_cmpt,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_DISC),
            center=mp.Vector3(0.5 * DISC_RADIUS_UM, 0, -0.5 * sz + 0.5 * disc_um),
            size=mp.Vector3(DISC_RADIUS_UM, mp.inf, disc_um),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution_um,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
    )

    # flux monitor
    flux_mon = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * cell_r_um, 0, 0.5 * sz - PML_UM),
            size=mp.Vector3(cell_r_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                cell_r_um, 0, 0.5 * sz - PML_UM - 0.5 * (AIR_UM + disc_um)
            ),
            size=mp.Vector3(0, 0, AIR_UM + disc_um),
        ),
    )

    # near-field monitor
    n2f_mon = sim.add_near2far(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * cell_r_um, 0, 0.5 * sz - PML_UM),
            size=mp.Vector3(cell_r_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                cell_r_um, 0, 0.5 * sz - PML_UM - 0.5 * (AIR_UM + disc_um)
            ),
            size=mp.Vector3(0, 0, AIR_UM + disc_um),
        ),
    )

    if plot:
        fig, ax = plt.subplots()
        sim.plot2D(ax=ax)
        if mp.am_master():
            fig.savefig("disc_simulation_layout.png", dpi=150, bbox_inches="tight")

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            field_decay_period,
            src_cmpt,
            src_pt,
            field_decay_tolerance,
        ),
    )

    flux_near = mp.get_fluxes(flux_mon)[0]

    validation_angles = np.linspace(0, 0.5 * math.pi, num_polar)
    radiation = radiation_pattern(
        sim,
        n2f_mon,
        polar_angles=validation_angles,
        greencyl_tolerance=greencyl_tolerance,
        return_details=return_details,
    )
    radial_flux = (
        np.asarray(radiation["flux_radial_magnitude"])
        if return_details
        else np.asarray(radiation)
    )
    radial_flux_scaled = FARFIELD_RADIUS_UM * FARFIELD_RADIUS_UM * radial_flux
    if plot:
        plot_radiation_pattern_polar(radial_flux_scaled, validation_angles)
        plot_radiation_pattern_3d(radial_flux_scaled, validation_angles)

    flux_far = (
        2
        * math.pi
        * FARFIELD_RADIUS_UM**2
        * np.trapezoid(radial_flux * np.sin(validation_angles), validation_angles)
    )

    if return_details:
        result = dict(radiation)
        result.update(
            {
                "angles": validation_angles,
                "flux_near": float(flux_near),
                "flux_far": float(flux_far),
                "timestep": int(sim.timestep()),
                "meep_time": float(sim.meep_time()),
                "cell_size": np.asarray([sr, sz], dtype=float),
            }
        )
        sim.reset_meep()
        return result

    return flux_near, flux_far


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded full-field cylindrical near-to-far validation",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-polar", type=int, default=None)
    parser.add_argument("--decay-period", type=float, default=None)
    parser.add_argument("--decay-tolerance", type=float, default=None)
    parser.add_argument("--greencyl-tolerance", type=float, default=None)
    args = parser.parse_args()

    disc_thickness_um = 0.7 * WAVELENGTH_UM / N_DISC
    dipole_height = 0.5

    if not args.validation:
        flux_near, flux_far = disc_radiated_flux(disc_thickness_um, dipole_height)

        err = abs(flux_near - flux_far) / flux_near
        print(
            f"total_flux:, {flux_near:.5f} (near), {flux_far:.5f} (far), "
            f"{100 * err:.2f}% (error)"
        )
    else:
        validation_resolution = 30 if args.resolution is None else args.resolution
        validation_num_polar = 48 if args.num_polar is None else args.num_polar
        validation_decay_period = (
            20.0 if args.decay_period is None else args.decay_period
        )
        validation_decay_tolerance = (
            1e-5 if args.decay_tolerance is None else args.decay_tolerance
        )
        validation_greencyl_tolerance = (
            1e-6 if args.greencyl_tolerance is None else args.greencyl_tolerance
        )
        if validation_resolution < 20 or validation_num_polar < 32:
            raise RuntimeError("disc validation resolution/sampling is too small")
        details = disc_radiated_flux(
            disc_thickness_um,
            dipole_height,
            resolution_um=validation_resolution,
            num_polar=validation_num_polar,
            field_decay_period=validation_decay_period,
            field_decay_tolerance=validation_decay_tolerance,
            greencyl_tolerance=validation_greencyl_tolerance,
            plot=False,
            return_details=True,
        )

        disc_radiation_angles = np.asarray(details["angles"])
        disc_radiation_e_fields = np.asarray(details["e_field"])
        disc_radiation_h_fields = np.asarray(details["h_field"])
        disc_radiation_flux_components = np.stack(
            [
                details["flux_x"],
                details["flux_z"],
                details["flux_radial_signed"],
                details["flux_radial_magnitude"],
            ]
        )
        flux_near = float(details["flux_near"])
        flux_far = float(details["flux_far"])
        flux_closure_error = abs(flux_near - flux_far) / max(abs(flux_near), 1e-30)
        signed_flux = np.asarray(details["flux_radial_signed"])
        radial_flux = np.asarray(details["flux_radial_magnitude"])
        e_norm = float(np.linalg.norm(disc_radiation_e_fields))
        h_norm = float(np.linalg.norm(disc_radiation_h_fields))
        disc_radiation_near_far = np.asarray(
            [flux_near, flux_far, flux_closure_error], dtype=float
        )
        disc_radiation_independent_checks = np.asarray(
            [
                e_norm,
                h_norm,
                h_norm / max(e_norm, 1e-30),
                np.min(signed_flux) / max(np.max(radial_flux), 1e-30),
                np.trapezoid(
                    radial_flux * np.sin(disc_radiation_angles),
                    disc_radiation_angles,
                ),
            ],
            dtype=float,
        )
        disc_radiation_work_summary = np.asarray(
            [
                details["timestep"],
                details["meep_time"],
                *details["cell_size"],
            ],
            dtype=float,
        )
        disc_radiation_validation_parameters = np.asarray(
            [
                validation_resolution,
                validation_num_polar,
                validation_decay_period,
                validation_decay_tolerance,
                validation_greencyl_tolerance,
                disc_thickness_um,
                dipole_height,
                FARFIELD_RADIUS_UM,
            ],
            dtype=float,
        )

        retained = (
            disc_radiation_e_fields,
            disc_radiation_h_fields,
            disc_radiation_flux_components,
            disc_radiation_near_far,
            disc_radiation_independent_checks,
        )
        if any(np.any(~np.isfinite(value)) for value in retained):
            raise RuntimeError("disc radiation validation produced nonfinite data")
        if flux_near <= 0 or flux_far <= 0 or e_norm <= 0 or h_norm <= 0:
            raise RuntimeError("disc radiation validation has nonphysical zero power")
        if flux_closure_error > 0.15:
            raise RuntimeError(
                f"disc near/far flux closure is too large: {flux_closure_error}"
            )
        if not 0.7 < h_norm / e_norm < 1.3:
            raise RuntimeError("disc far-field H/E norm ratio is non-radiative")
        if np.min(signed_flux) < -0.1 * np.max(radial_flux):
            raise RuntimeError(
                "disc far field contains a large inward radial component"
            )
        print(
            "disc-radiation-validation:, "
            f"near={flux_near:.8g}, far={flux_far:.8g}, "
            f"closure={flux_closure_error:.6g}, h/e={h_norm / e_norm:.6g}"
        )
