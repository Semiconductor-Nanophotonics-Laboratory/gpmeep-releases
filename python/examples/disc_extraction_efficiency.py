"""Computes the extraction efficiency of a collection of dipoles in a disc.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#extraction-efficiency-of-a-disc-in-cylindrical-coordinates
"""

import argparse
import math
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
N_DISC = 2.4
DISC_RADIUS_UM = 1.2
DISC_THICKNESS_UM = 0.7 * WAVELENGTH_UM / N_DISC
NUM_FARFIELD_PTS = 200
FARFIELD_RADIUS_UM = 1e6 * WAVELENGTH_UM
NUM_DIPOLES = 11
GREENCYL_TOL = 1e-6
FLUX_DECAY_THRESHOLD = 1e-2

farfield_angles = np.linspace(0, 0.5 * math.pi, NUM_FARFIELD_PTS)


def plot_radiation_pattern_polar(
    radial_flux: np.ndarray, polar_angles: np.ndarray = farfield_angles
):
    """Plots the radiation pattern in polar coordinates.

    Args:
      radial_flux: radial flux of the far fields at each angle.
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
    radial_flux: np.ndarray, polar_angles: np.ndarray = farfield_angles
):
    """Plots the radiation pattern in 3d Cartesian coordinates.

    Args:
      radial_flux: radial flux of the far fields at each angle.
    """
    radial_flux = np.asarray(radial_flux)
    polar_angles = np.asarray(polar_angles)
    if radial_flux.shape != polar_angles.shape:
        raise ValueError("radial flux and polar angle arrays must have matching shapes")
    num_farfield_pts = polar_angles.size
    phis = np.linspace(0, 2 * np.pi, num_farfield_pts)

    xs = np.zeros((num_farfield_pts, num_farfield_pts))
    ys = np.zeros((num_farfield_pts, num_farfield_pts))
    zs = np.zeros((num_farfield_pts, num_farfield_pts))

    for i, theta in enumerate(polar_angles):
        for j, phi in enumerate(phis):
            xs[i, j] = radial_flux[i] * np.sin(theta) * np.cos(phi)
            ys[i, j] = radial_flux[i] * np.sin(theta) * np.sin(phi)
            zs[i, j] = radial_flux[i] * np.cos(theta)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.plot_surface(xs, ys, zs, cmap="inferno")
    ax.set_title("radiation pattern in 3d")
    ax.set_box_aspect((np.amax(xs), np.amax(ys), np.amax(zs)))
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
    polar_angles: np.ndarray = farfield_angles,
    farfield_radius_um: float = FARFIELD_RADIUS_UM,
    greencyl_tolerance: float = GREENCYL_TOL,
    return_details: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Computes the radiation pattern from the near fields.

    Args:
      sim: a `Simulation` object.
      n2f_mon: a `DftNear2Far` object returned by `Simulation.add_near2far`.

    Returns:
      The radiation pattern (radial flux at each angle) as a 1d array.
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
    for n, angle in enumerate(polar_angles):
        far_field = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                farfield_radius_um * math.sin(angle),
                0,
                farfield_radius_um * math.cos(angle),
            ),
            greencyl_tolerance,
        )
        e_field[n, :] = [far_field[j] for j in range(3)]
        h_field[n, :] = [far_field[j + 3] for j in range(3)]

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


def radiation_pattern_flux(
    radial_flux: np.ndarray,
    *,
    polar_angles: np.ndarray = farfield_angles,
    farfield_radius_um: float = FARFIELD_RADIUS_UM,
) -> float:
    """Computes the total flux from the radiation pattern.

    Based on integrating the radiation pattern over solid angles
    spanned by polar angles in the range of [0, π/2].

    Args:
      radial_flux: radial flux of the far fields at each angle.
    """
    radial_flux = np.asarray(radial_flux)
    polar_angles = np.asarray(polar_angles)
    if (
        radial_flux.ndim != 1
        or radial_flux.shape != polar_angles.shape
        or np.any(~np.isfinite(radial_flux))
        or np.any(~np.isfinite(polar_angles))
        or farfield_radius_um <= 0
    ):
        raise ValueError(
            "finite radial flux and angle arrays must have matching shapes"
        )
    total_flux = (
        2
        * math.pi
        * farfield_radius_um**2
        * np.trapezoid(radial_flux * np.sin(polar_angles), polar_angles)
    )

    return total_flux


def dipole_in_disc(
    zpos: float,
    rpos_um: float,
    m: int,
    *,
    resolution_um: int = RESOLUTION_UM,
    num_farfield_pts: int = NUM_FARFIELD_PTS,
    dft_decay_threshold: float = 1e-4,
    post_source_time: Optional[float] = None,
    greencyl_tolerance: float = GREENCYL_TOL,
    return_details: bool = False,
) -> Union[Tuple[float, np.ndarray], Dict[str, object]]:
    """Computes the total flux and radiation pattern of a dipole in a disc.

    Args:
      zpos: height of dipole above ground plane as fraction of disc thickness.
      rpos_um: radial position of dipole.
      m: angular φ dependence of the fields exp(imφ).

    Returns:
      A 2-tuple of the total flux and the radiation pattern.
    """
    pml_um = 1.0  # thickness of PML
    padding_um = 1.0  # thickness of air padding above disc
    r_um = 4.0  # length of cell in r

    frequency = 1 / WAVELENGTH_UM  # center frequency of source/monitor

    if not 0 < zpos < 1 or not 0 < rpos_um <= DISC_RADIUS_UM:
        raise ValueError("dipole must be inside the finite disc")
    if resolution_um <= 0 or num_farfield_pts < 8:
        raise ValueError("resolution and far-field sample count are too small")
    if not 0 < dft_decay_threshold < 1 or greencyl_tolerance <= 0:
        raise ValueError("invalid DFT-decay or Green-function tolerance")
    if post_source_time is not None and post_source_time <= 0:
        raise ValueError("post_source_time must be positive when provided")

    size_r = r_um + pml_um
    size_z = DISC_THICKNESS_UM + padding_um + pml_um
    cell_size = mp.Vector3(size_r, 0, size_z)

    boundary_layers = [
        mp.PML(pml_um, direction=mp.R),
        mp.PML(pml_um, direction=mp.Z, side=mp.High),
    ]

    src_pt = mp.Vector3(rpos_um, 0, -0.5 * size_z + zpos * DISC_THICKNESS_UM)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.05 * frequency),
            component=mp.Er,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_DISC),
            center=mp.Vector3(
                0.5 * DISC_RADIUS_UM, 0, -0.5 * size_z + 0.5 * DISC_THICKNESS_UM
            ),
            size=mp.Vector3(DISC_RADIUS_UM, mp.inf, DISC_THICKNESS_UM),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution_um,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=m,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        force_complex_fields=True,
    )

    n2f_mon = sim.add_near2far(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * r_um, 0, 0.5 * size_z - pml_um),
            size=mp.Vector3(r_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                r_um, 0, 0.5 * size_z - pml_um - 0.5 * (padding_um + DISC_THICKNESS_UM)
            ),
            size=mp.Vector3(0, 0, padding_um + DISC_THICKNESS_UM),
        ),
    )

    if post_source_time is None:
        sim.run(
            mp.dft_ldos(frequency, 0, 1),
            until_after_sources=mp.stop_when_dft_decayed(
                tol=dft_decay_threshold,
            ),
        )
    else:
        sim.run(
            mp.dft_ldos(frequency, 0, 1),
            until_after_sources=post_source_time,
        )

    delta_vol = 2 * np.pi * rpos_um / (resolution_um**2)
    dipole_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * delta_vol

    validation_angles = np.linspace(0, 0.5 * math.pi, num_farfield_pts)
    radiation = radiation_pattern(
        sim,
        n2f_mon,
        polar_angles=validation_angles,
        greencyl_tolerance=greencyl_tolerance,
        return_details=return_details,
    )
    dipole_radiation_pattern = (
        np.asarray(radiation["flux_radial_magnitude"])
        if return_details
        else np.asarray(radiation)
    )

    if return_details:
        result = dict(radiation)
        result.update(
            {
                "angles": validation_angles,
                "dipole_flux": float(dipole_flux),
                "radiation_flux": float(
                    radiation_pattern_flux(
                        dipole_radiation_pattern, polar_angles=validation_angles
                    )
                ),
                "ldos": np.asarray(sim.ldos_data, dtype=float).copy(),
                "ldos_fdata": np.asarray(sim.ldos_Fdata, dtype=np.complex128).copy(),
                "ldos_jdata": np.asarray(sim.ldos_Jdata, dtype=np.complex128).copy(),
                "work_summary": np.asarray(
                    [
                        resolution_um,
                        rpos_um,
                        m,
                        sim.meep_time(),
                        sim.timestep(),
                        round(resolution_um * size_r)
                        * round(resolution_um * size_z),
                    ],
                    dtype=float,
                ),
            }
        )
        sim.reset_meep()
        return result

    return dipole_flux, dipole_radiation_pattern


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded radial and azimuthal-mode extraction oracle",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-positions", type=int, default=None)
    parser.add_argument("--max-m", type=int, default=None)
    parser.add_argument("--num-farfield", type=int, default=None)
    parser.add_argument("--dft-decay-tolerance", type=float, default=None)
    parser.add_argument("--post-source-time", type=float, default=None)
    parser.add_argument("--greencyl-tolerance", type=float, default=None)
    args = parser.parse_args()

    dipole_height = 0.5
    if not args.validation:
        dipole_rpos_um = np.linspace(0, DISC_RADIUS_UM, NUM_DIPOLES)
        delta_rpos_um = DISC_RADIUS_UM / (NUM_DIPOLES - 1)

        # 1. Er source at r = 0 requires a single simulation with m = ±1.

        # An Er source at r = 0 needs to be slighty offset due to a bug.
        # https://github.com/NanoComp/meep/issues/2704
        dipole_rpos_um[0] = 1.5 / RESOLUTION_UM

        m = -1
        dipole_flux, dipole_radiation_pattern = dipole_in_disc(
            dipole_height,
            dipole_rpos_um[0],
            m,
        )

        flux_total = dipole_flux * dipole_rpos_um[0] * delta_rpos_um
        radiation_pattern_total = (
            dipole_radiation_pattern * dipole_rpos_um[0] * delta_rpos_um
        )

        print(
            f"dipole:, {dipole_rpos_um[0]:.4f}, "
            f"{radiation_pattern_flux(dipole_radiation_pattern):.6f}"
        )

        # 2. Er source at r > 0 requires Fourier-series expansion of φ.

        # Threshold flux to determine when to truncate expansion.
        flux_decay_threshold = FLUX_DECAY_THRESHOLD

        for rpos_um in dipole_rpos_um[1:]:
            dipole_flux_total = 0
            dipole_radiation_pattern_total = np.zeros(NUM_FARFIELD_PTS)
            dipole_radiation_pattern_flux_max = 0
            m = 0
            while True:
                dipole_flux, dipole_radiation_pattern = dipole_in_disc(
                    dipole_height, rpos_um, m
                )
                dipole_flux_total += dipole_flux * (1 if m == 0 else 2)
                dipole_radiation_pattern_total += dipole_radiation_pattern * (
                    1 if m == 0 else 2
                )

                dipole_radiation_pattern_flux = radiation_pattern_flux(
                    dipole_radiation_pattern
                )
                print(
                    f"dipole:, {rpos_um:.4f}, {m}, "
                    f"{dipole_radiation_pattern_flux:.6f}"
                )

                if dipole_radiation_pattern_flux > dipole_radiation_pattern_flux_max:
                    dipole_radiation_pattern_flux_max = dipole_radiation_pattern_flux

                if (
                    m > 0
                    and (
                        dipole_radiation_pattern_flux
                        / dipole_radiation_pattern_flux_max
                    )
                    < flux_decay_threshold
                ):
                    break
                else:
                    m += 1

            dipole_position_scale_factor = 0.5 * (dipole_rpos_um[0] / rpos_um) ** 2
            flux_total += (
                dipole_flux_total
                * dipole_position_scale_factor
                * rpos_um
                * delta_rpos_um
            )
            radiation_pattern_total += (
                dipole_radiation_pattern_total
                * dipole_position_scale_factor
                * rpos_um
                * delta_rpos_um
            )

        radiation_pattern_total_flux = radiation_pattern_flux(radiation_pattern_total)
        extraction_efficiency = radiation_pattern_total_flux / flux_total
        print(f"exteff:, {extraction_efficiency:.6f}")

        radiation_pattern_scaled = radiation_pattern_total * FARFIELD_RADIUS_UM**2
        plot_radiation_pattern_polar(radiation_pattern_scaled)
        plot_radiation_pattern_3d(radiation_pattern_scaled)
    else:
        validation_resolution = 20 if args.resolution is None else args.resolution
        validation_num_positions = (
            5 if args.num_positions is None else args.num_positions
        )
        validation_max_m = 14 if args.max_m is None else args.max_m
        validation_num_farfield = (
            40 if args.num_farfield is None else args.num_farfield
        )
        validation_dft_decay = (
            1e-4 if args.dft_decay_tolerance is None else args.dft_decay_tolerance
        )
        validation_post_source = (
            120.0 if args.post_source_time is None else args.post_source_time
        )
        validation_greencyl = (
            1e-6 if args.greencyl_tolerance is None else args.greencyl_tolerance
        )
        if validation_resolution < 20 or validation_num_positions < 4:
            raise RuntimeError("disc extraction validation radial grid is too small")
        if validation_max_m < 8 or validation_num_farfield < 32:
            raise RuntimeError(
                "disc extraction validation mode/far-field grid is too small"
            )

        disc_extraction_positions = np.linspace(
            0, DISC_RADIUS_UM, validation_num_positions
        )
        delta_rpos_um = DISC_RADIUS_UM / (validation_num_positions - 1)
        disc_extraction_positions[0] = 1.5 / validation_resolution
        disc_extraction_m_values = np.arange(validation_max_m + 1, dtype=float)
        common_kwargs = {
            "resolution_um": validation_resolution,
            "num_farfield_pts": validation_num_farfield,
            "dft_decay_threshold": validation_dft_decay,
            "post_source_time": validation_post_source,
            "greencyl_tolerance": validation_greencyl,
            "return_details": True,
        }

        center_result = dipole_in_disc(
            dipole_height,
            float(disc_extraction_positions[0]),
            -1,
            **common_kwargs,
        )
        offaxis_results = [
            [
                dipole_in_disc(
                    dipole_height,
                    float(position),
                    mode,
                    **common_kwargs,
                )
                for mode in range(validation_max_m + 1)
            ]
            for position in disc_extraction_positions[1:]
        ]

        disc_extraction_angles = np.asarray(center_result["angles"])
        disc_extraction_center_e_fields = np.asarray(center_result["e_field"])
        disc_extraction_center_h_fields = np.asarray(center_result["h_field"])
        disc_extraction_center_radial_flux = np.stack(
            [
                center_result["flux_radial_signed"],
                center_result["flux_radial_magnitude"],
            ]
        )
        disc_extraction_center_source_flux = np.asarray(
            [center_result["dipole_flux"]], dtype=float
        )
        disc_extraction_center_radiation_flux = np.asarray(
            [center_result["radiation_flux"]], dtype=float
        )
        disc_extraction_offaxis_e_fields = np.asarray(
            [[result["e_field"] for result in row] for row in offaxis_results]
        )
        disc_extraction_offaxis_h_fields = np.asarray(
            [[result["h_field"] for result in row] for row in offaxis_results]
        )
        disc_extraction_mode_radial_flux = np.asarray(
            [
                [
                    [
                        result["flux_radial_signed"],
                        result["flux_radial_magnitude"],
                    ]
                    for result in row
                ]
                for row in offaxis_results
            ]
        )
        disc_extraction_mode_source_flux = np.asarray(
            [[result["dipole_flux"] for result in row] for row in offaxis_results]
        )
        disc_extraction_mode_radiation_flux = np.asarray(
            [[result["radiation_flux"] for result in row] for row in offaxis_results]
        )
        disc_extraction_mode_ldos = np.asarray(
            [[result["ldos"] for result in row] for row in offaxis_results]
        )
        disc_extraction_mode_ldos_fdata = np.asarray(
            [[result["ldos_fdata"] for result in row] for row in offaxis_results]
        )
        disc_extraction_mode_ldos_jdata = np.asarray(
            [[result["ldos_jdata"] for result in row] for row in offaxis_results]
        )
        disc_extraction_work_summary = np.asarray(
            [center_result["work_summary"]]
            + [result["work_summary"] for row in offaxis_results for result in row]
        )

        # Keep every requested mode in the raw validation vectors, but reconstruct
        # the inclusive stopping rule used by the published example for physical
        # position totals. The first mode below 1% is included because the default
        # loop accumulates each mode before testing the truncation condition.
        mode_inclusion_mask = np.ones_like(
            disc_extraction_mode_radiation_flux, dtype=bool
        )
        disc_extraction_mode_stop_indices = np.full(
            validation_num_positions - 1, validation_max_m, dtype=float
        )
        disc_extraction_mode_stop_ratios = np.ones(
            validation_num_positions - 1, dtype=float
        )
        mode_stop_found = np.zeros(validation_num_positions - 1, dtype=bool)
        for position_index, radiation_fluxes in enumerate(
            disc_extraction_mode_radiation_flux
        ):
            maximum_radiation_flux = 0.0
            for mode, radiation_flux in enumerate(radiation_fluxes):
                maximum_radiation_flux = max(
                    maximum_radiation_flux, float(radiation_flux)
                )
                if maximum_radiation_flux > 0:
                    decay_ratio = float(radiation_flux) / maximum_radiation_flux
                else:
                    decay_ratio = 1.0
                if mode > 0 and decay_ratio < FLUX_DECAY_THRESHOLD:
                    disc_extraction_mode_stop_indices[position_index] = mode
                    disc_extraction_mode_stop_ratios[position_index] = decay_ratio
                    mode_inclusion_mask[position_index, mode + 1 :] = False
                    mode_stop_found[position_index] = True
                    break
        if not np.all(mode_stop_found):
            raise RuntimeError(
                "disc extraction validation mode range does not reach the "
                "published 1% radiation-flux stopping condition"
            )

        modal_weights = np.ones(validation_max_m + 1)
        modal_weights[1:] = 2
        truncated_modal_weights = modal_weights[None, :] * mode_inclusion_mask
        disc_extraction_position_source_flux = np.sum(
            disc_extraction_mode_source_flux * truncated_modal_weights, axis=1
        )
        disc_extraction_position_patterns = np.sum(
            disc_extraction_mode_radial_flux[:, :, 1, :]
            * truncated_modal_weights[:, :, None],
            axis=1,
        )
        disc_extraction_position_radiation_flux = np.asarray(
            [
                radiation_pattern_flux(pattern, polar_angles=disc_extraction_angles)
                for pattern in disc_extraction_position_patterns
            ]
        )

        center_weight = disc_extraction_positions[0] * delta_rpos_um
        offaxis_weights = (
            0.5
            * (
                disc_extraction_positions[0] / disc_extraction_positions[1:]
            )
            ** 2
            * disc_extraction_positions[1:]
            * delta_rpos_um
        )
        total_source_flux = float(
            center_result["dipole_flux"] * center_weight
            + np.dot(disc_extraction_position_source_flux, offaxis_weights)
        )
        disc_extraction_total_pattern = (
            np.asarray(center_result["flux_radial_magnitude"]) * center_weight
            + np.sum(
                disc_extraction_position_patterns * offaxis_weights[:, None], axis=0
            )
        )
        total_radiation_flux = float(
            radiation_pattern_flux(
                disc_extraction_total_pattern, polar_angles=disc_extraction_angles
            )
        )
        extraction_efficiency = total_radiation_flux / total_source_flux
        disc_extraction_integrated_results = np.asarray(
            [total_source_flux, total_radiation_flux, extraction_efficiency]
        )

        maximum_mode_power = np.maximum(
            np.max(disc_extraction_mode_radiation_flux, axis=1), 1e-30
        )
        disc_extraction_tail_power_ratios = (
            disc_extraction_mode_radiation_flux[:, -3:]
            / maximum_mode_power[:, None]
        )
        all_e_fields = np.concatenate(
            [
                disc_extraction_center_e_fields[None, :, :],
                disc_extraction_offaxis_e_fields.reshape(
                    -1, validation_num_farfield, 3
                ),
            ]
        )
        all_h_fields = np.concatenate(
            [
                disc_extraction_center_h_fields[None, :, :],
                disc_extraction_offaxis_h_fields.reshape(
                    -1, validation_num_farfield, 3
                ),
            ]
        )
        e_field_norms = np.linalg.norm(all_e_fields, axis=(1, 2))
        h_field_norms = np.linalg.norm(all_h_fields, axis=(1, 2))
        field_activity = np.maximum(e_field_norms, h_field_norms)
        disc_extraction_farfield_significant_mask = (
            field_activity > 1e-8 * np.max(field_activity)
        )
        disc_extraction_farfield_impedance_ratios = np.ones_like(e_field_norms)
        disc_extraction_farfield_impedance_ratios[
            disc_extraction_farfield_significant_mask
        ] = h_field_norms[disc_extraction_farfield_significant_mask] / np.maximum(
            e_field_norms[disc_extraction_farfield_significant_mask], 1e-30
        )
        signed_fluxes = np.concatenate(
            [
                np.asarray(center_result["flux_radial_signed"])[None, :],
                disc_extraction_mode_radial_flux[:, :, 0, :].reshape(
                    -1, validation_num_farfield
                ),
            ]
        )
        magnitude_fluxes = np.concatenate(
            [
                np.asarray(center_result["flux_radial_magnitude"])[None, :],
                disc_extraction_mode_radial_flux[:, :, 1, :].reshape(
                    -1, validation_num_farfield
                ),
            ]
        )
        flux_activity = np.max(magnitude_fluxes, axis=1)
        directional_mask = flux_activity > 1e-10 * np.max(flux_activity)
        disc_extraction_minimum_signed_ratios = np.zeros_like(flux_activity)
        disc_extraction_minimum_signed_ratios[directional_mask] = np.min(
            signed_fluxes[directional_mask], axis=1
        ) / np.maximum(flux_activity[directional_mask], 1e-30)
        disc_extraction_maximum_inward_ratios = np.maximum(
            -disc_extraction_minimum_signed_ratios, 0
        )
        disc_extraction_independent_checks = np.asarray(
            [
                np.max(disc_extraction_tail_power_ratios),
                np.min(
                    disc_extraction_farfield_impedance_ratios[
                        disc_extraction_farfield_significant_mask
                    ]
                ),
                np.max(
                    disc_extraction_farfield_impedance_ratios[
                        disc_extraction_farfield_significant_mask
                    ]
                ),
                np.min(disc_extraction_minimum_signed_ratios[directional_mask]),
                np.linalg.norm(np.diff(disc_extraction_position_radiation_flux)),
            ],
            dtype=float,
        )
        disc_extraction_run_counters = np.asarray(
            [
                1 + (validation_num_positions - 1) * (validation_max_m + 1),
                np.sum(disc_extraction_work_summary[:, 4]),
                np.sum(disc_extraction_work_summary[:, 5]),
                np.sum(
                    disc_extraction_work_summary[:, 4]
                    * disc_extraction_work_summary[:, 5]
                ),
            ],
            dtype=float,
        )
        disc_extraction_validation_parameters = np.asarray(
            [
                dipole_height,
                validation_resolution,
                validation_num_positions,
                validation_max_m,
                validation_num_farfield,
                validation_dft_decay,
                validation_post_source,
                validation_greencyl,
                delta_rpos_um,
                FLUX_DECAY_THRESHOLD,
            ],
            dtype=float,
        )

        retained = (
            disc_extraction_center_e_fields,
            disc_extraction_center_h_fields,
            disc_extraction_offaxis_e_fields,
            disc_extraction_offaxis_h_fields,
            disc_extraction_mode_radial_flux,
            disc_extraction_mode_source_flux,
            disc_extraction_mode_radiation_flux,
            disc_extraction_mode_stop_indices,
            disc_extraction_mode_stop_ratios,
            mode_inclusion_mask,
            disc_extraction_mode_ldos,
            disc_extraction_mode_ldos_fdata,
            disc_extraction_mode_ldos_jdata,
            disc_extraction_position_source_flux,
            disc_extraction_position_radiation_flux,
            disc_extraction_total_pattern,
            disc_extraction_integrated_results,
            disc_extraction_tail_power_ratios,
        )
        if any(np.any(~np.isfinite(value)) for value in retained):
            raise RuntimeError("disc extraction validation produced nonfinite data")
        if total_source_flux <= 0 or total_radiation_flux <= 0:
            raise RuntimeError("disc extraction total power must be positive")
        if not 0 < extraction_efficiency < 1.2:
            raise RuntimeError("disc extraction efficiency is nonphysical")
        if np.max(disc_extraction_tail_power_ratios) > 0.05:
            raise RuntimeError(
                "disc extraction azimuthal expansion has not reached its modal tail"
            )
        if np.count_nonzero(disc_extraction_farfield_significant_mask) < 20:
            raise RuntimeError(
                "disc extraction has too few significant far-field modes"
            )
        significant_impedance = disc_extraction_farfield_impedance_ratios[
            disc_extraction_farfield_significant_mask
        ]
        if np.min(significant_impedance) < 0.7 or np.max(significant_impedance) > 1.3:
            raise RuntimeError(
                "disc extraction far fields have a non-radiative H/E ratio"
            )
        if np.count_nonzero(directional_mask) < 20:
            raise RuntimeError(
                "disc extraction has too few directional far-field modes"
            )
        if np.min(disc_extraction_minimum_signed_ratios[directional_mask]) < -0.1:
            raise RuntimeError(
                "disc extraction contains a large inward far-field component"
            )
        if np.max(disc_extraction_maximum_inward_ratios) > 0.1:
            raise RuntimeError(
                "disc extraction inward-ratio oracle exceeds its physical bound"
            )
        if np.linalg.norm(np.diff(disc_extraction_position_radiation_flux)) <= 1e-9:
            raise RuntimeError(
                "disc extraction radial samples are not physically distinct"
            )
        # The generic validation collector deliberately accepts numeric arrays only.
        # Retain the mask as a 0/1 result vector after all boolean indexing is done.
        disc_extraction_farfield_significant_mask = (
            disc_extraction_farfield_significant_mask.astype(np.float64)
        )
        disc_extraction_mode_inclusion_mask = mode_inclusion_mask.astype(np.float64)
        print(
            "disc-extraction-validation:, "
            f"efficiency={extraction_efficiency:.8g}, "
            f"tail={np.max(disc_extraction_tail_power_ratios):.6g}, "
            f"runs={int(disc_extraction_run_counters[0])}"
        )
