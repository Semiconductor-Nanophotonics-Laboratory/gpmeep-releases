"""Radiation pattern of an off-axis dipole in cylindrical coordinates.

Tutorial Reference:

https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#radiation-pattern-of-an-antenna-in-cylindrical-coordinates
"""

import argparse
import cmath
import math
from typing import Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
PML_UM = 1.0
FARFIELD_RADIUS_UM = 1e6 * WAVELENGTH_UM
NUM_FARFIELD_PTS = 50
AZIMUTHAL_RAD = 0
POWER_DECAY_THRESHOLD = 1e-4
GREENCYL_TOL = 1e-6
FREQUENCY_POWER_RELATIVE_TOLERANCE = 1e-5
# This absolute-to-peak check primarily covers modes below the significant-mode
# mask. Keep a small FP32 compiler/architecture margin while the physically
# significant modes retain a separate relative-error gate plus full-field and
# analytic-pattern checks.
FREQUENCY_POWER_PEAK_NORMALIZED_TOLERANCE = 1.5e-8

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 0.5 * math.pi, NUM_FARFIELD_PTS)


def plot_radiation_pattern(dipole_pol: str, radial_flux: np.ndarray):
    """Plots the radiation pattern in polar coordinates.

    The angles increase clockwise with zero in the +z direction (the "pole")
    and π/2 in the +r direction (the "equator").

    Args:
        dipole_pol: the dipole polarization.
        radial_flux: the radial flux in polar coordinates.
    """
    normalized_radial_flux = radial_flux / np.max(radial_flux)
    if dipole_pol == "x":
        dipole_radial_flux = np.square(np.cos(polar_rad))
        dipole_radial_flux_label = r"$\cos^2θ$"
        dipole_name = "$E_x$"
    else:
        dipole_radial_flux = np.ones(NUM_FARFIELD_PTS)
        dipole_radial_flux_label = "constant (1.0)"
        dipole_name = "$E_y$"

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(polar_rad, normalized_radial_flux, "b-", label="Meep")
    ax.plot(polar_rad, dipole_radial_flux, "r--", label=dipole_radial_flux_label)
    ax.legend()
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0.5 * math.pi)
    ax.set_thetalim(0, 0.5 * math.pi)
    ax.set_rmax(1)
    ax.set_rticks([0, 0.5, 1])
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.set_ylabel("radial flux (a.u.)")
    ax.set_title(f"radiation pattern (φ = 0) of an off-axis {dipole_name} dipole")

    if mp.am_master():
        fig.savefig(
            "dipole_radiation_pattern_off_axis.png",
            dpi=150,
            bbox_inches="tight",
        )

    relative_error = np.linalg.norm(
        normalized_radial_flux - dipole_radial_flux
    ) / np.linalg.norm(dipole_radial_flux)
    print(f"relative error in radiation pattern:, {relative_error}")


def radiation_pattern(e_field: np.ndarray, h_field: np.ndarray) -> np.ndarray:
    """Computes the radiation pattern from the far fields.

    Args:
        e_field, h_field: the electric (Er, Ep, Ez) and magnetic (Hr, Hp, Hz)
          far fields, respectively.

    Returns:
        The radial Poynting flux as a 1D array. One element for each point on
        the circumference of a quarter circle with angular range of
        [0, π/2] rad. 0 radians is the +z direction (the "pole") and π/2 is
        the +r direction (the "equator").
    """
    flux_x = np.real(
        e_field[:, 1] * np.conj(h_field[:, 2]) - e_field[:, 2] * np.conj(h_field[:, 1])
    )
    flux_z = np.real(
        e_field[:, 0] * np.conj(h_field[:, 1]) - e_field[:, 1] * np.conj(h_field[:, 0])
    )
    flux_r = np.sqrt(np.square(flux_x) + np.square(flux_z))

    return flux_r


def poynting_components(
    e_field: np.ndarray, h_field: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the magnitude, signed-radial, and tangential Poynting flux."""
    flux_x = np.real(
        e_field[:, 1] * np.conj(h_field[:, 2])
        - e_field[:, 2] * np.conj(h_field[:, 1])
    )
    flux_z = np.real(
        e_field[:, 0] * np.conj(h_field[:, 1])
        - e_field[:, 1] * np.conj(h_field[:, 0])
    )
    radial_magnitude = np.sqrt(np.square(flux_x) + np.square(flux_z))
    signed_radial = flux_x * np.sin(polar_rad) + flux_z * np.cos(polar_rad)
    tangential = flux_x * np.cos(polar_rad) - flux_z * np.sin(polar_rad)
    return radial_magnitude, signed_radial, tangential


def get_farfields(
    sim: mp.Simulation, n2f_mon: mp.DftNear2Far
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the far fields from the near fields for φ = 0 (rz plane).

    Args:
        sim: a `Simulation` object.
        n2f_mon: a `DftNear2Far` object returned by `Simulation.add_near2far`.

    Returns:
        The electric (Er, Ep, Ez) and magnetic (Hr, Hp, Hz) far fields. One row
        for each point on the circumference of a quarter circle with angular
        range of [0, π/2] rad. Each row has six columns for the fields.
        0 radians is the +z direction (the "pole") and π/2 is the +r direction
        (the "equator").
    """
    e_field = np.zeros((NUM_FARFIELD_PTS, 3), dtype=np.complex128)
    h_field = np.zeros((NUM_FARFIELD_PTS, 3), dtype=np.complex128)
    for n in range(NUM_FARFIELD_PTS):
        far_field = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                FARFIELD_RADIUS_UM * math.sin(polar_rad[n]),
                0,
                FARFIELD_RADIUS_UM * math.cos(polar_rad[n]),
            ),
            GREENCYL_TOL,
        )
        e_field[n, :] = [far_field[j] for j in range(3)]
        h_field[n, :] = [far_field[j + 3] for j in range(3)]

    return e_field, h_field


def dipole_in_vacuum(
    dipole_pol: str, dipole_pos_r: mp.Vector3, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the far fields of an off-axis point source.

    Args:
        dipole_pol: the dipole polarization.
        dipole_pos_r: the radial position of the dipole.
        m: angular φ dependence of the fields exp(imφ).

    Returns:
        A 4-tuple containing the electric and magnetic far fields at positive
        and negative frequencies, respectively, as 1D arrays.
    """
    sr = 2.0
    sz = 4.0
    cell_size = mp.Vector3(sr + PML_UM, 0, sz + 2 * PML_UM)

    boundary_layers = [mp.PML(thickness=PML_UM)]

    src_cmpt = mp.Er if dipole_pol == "x" else mp.Ep

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            component=src_cmpt,
            center=mp.Vector3(dipole_pos_r, 0, 0),
        ),
        mp.Source(
            src=mp.GaussianSource(-frequency, fwidth=0.1 * frequency),
            component=src_cmpt,
            center=mp.Vector3(dipole_pos_r, 0, 0),
        ),
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=m,
        boundary_layers=boundary_layers,
        sources=sources,
        force_complex_fields=True,
    )

    nearfields_monitor_plus = sim.add_near2far(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * sr, 0, 0.5 * sz), size=mp.Vector3(sr, 0, 0)
        ),
        mp.FluxRegion(center=mp.Vector3(sr, 0, 0), size=mp.Vector3(0, 0, sz)),
        mp.FluxRegion(
            center=mp.Vector3(0.5 * sr, 0, -0.5 * sz),
            size=mp.Vector3(sr, 0, 0),
            weight=-1.0,
        ),
    )

    nearfields_monitor_minus = sim.add_near2far(
        -frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * sr, 0, 0.5 * sz), size=mp.Vector3(sr, 0, 0)
        ),
        mp.FluxRegion(center=mp.Vector3(sr, 0, 0), size=mp.Vector3(0, 0, sz)),
        mp.FluxRegion(
            center=mp.Vector3(0.5 * sr, 0, -0.5 * sz),
            size=mp.Vector3(sr, 0, 0),
            weight=-1.0,
        ),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20.0, src_cmpt, mp.Vector3(dipole_pos_r, 0, 0), 1e-6
        )
    )

    e_field_plus, h_field_plus = get_farfields(sim, nearfields_monitor_plus)
    e_field_minus, h_field_minus = get_farfields(sim, nearfields_monitor_minus)

    return e_field_plus, h_field_plus, e_field_minus, h_field_minus


def flux_from_farfields(e_field: np.ndarray, h_field: np.ndarray) -> float:
    """Computes the flux from the far fields.

    Args:
        e_field, h_field: the electric (Er, Ep, Ez) and magnetic (Hr, Hp, Hz)
          far fields, respectively.

    Returns:
        The Poynting flux obtained from the far fields.
    """
    dipole_radiation_pattern = radiation_pattern(e_field, h_field)
    flux = (
        2
        * math.pi
        * FARFIELD_RADIUS_UM**2
        * np.trapezoid(dipole_radiation_pattern * np.sin(polar_rad), polar_rad)
    )

    return flux


def validation_mode_sweep(dipole_pos_r: float, maximum_m: int):
    """Runs a fixed, converged Fourier-mode sweep for both dipole polarizations."""
    polarizations = ("x", "y")
    m_values = np.arange(maximum_m + 1, dtype=int)
    num_polarizations = len(polarizations)
    num_modes = len(m_values)
    field_shape = (num_polarizations, num_modes, NUM_FARFIELD_PTS, 3)

    e_fields_plus = np.zeros(field_shape, dtype=np.complex128)
    h_fields_plus = np.zeros_like(e_fields_plus)
    e_fields_minus = np.zeros_like(e_fields_plus)
    h_fields_minus = np.zeros_like(e_fields_plus)
    mode_radial_magnitude = np.zeros(
        (num_polarizations, num_modes, 2, NUM_FARFIELD_PTS)
    )
    mode_signed_radial = np.zeros_like(mode_radial_magnitude)
    mode_tangential = np.zeros_like(mode_radial_magnitude)
    mode_powers = np.zeros((num_polarizations, num_modes, 2))
    cumulative_e_fields = np.zeros(field_shape, dtype=np.complex128)
    cumulative_h_fields = np.zeros_like(cumulative_e_fields)
    cumulative_radial_magnitude = np.zeros(
        (num_polarizations, num_modes, NUM_FARFIELD_PTS)
    )
    cumulative_signed_radial = np.zeros_like(cumulative_radial_magnitude)
    cumulative_tangential = np.zeros_like(cumulative_radial_magnitude)

    for polarization_index, dipole_pol in enumerate(polarizations):
        e_field_total = np.zeros((NUM_FARFIELD_PTS, 3), dtype=np.complex128)
        h_field_total = np.zeros_like(e_field_total)
        for mode_index, m_value in enumerate(m_values):
            (
                e_field_plus,
                h_field_plus,
                e_field_minus,
                h_field_minus,
            ) = dipole_in_vacuum(dipole_pol, dipole_pos_r, int(m_value))
            e_fields_plus[polarization_index, mode_index] = e_field_plus
            h_fields_plus[polarization_index, mode_index] = h_field_plus
            e_fields_minus[polarization_index, mode_index] = e_field_minus
            h_fields_minus[polarization_index, mode_index] = h_field_minus

            for frequency_index, (e_field, h_field) in enumerate(
                (
                    (e_field_plus, h_field_plus),
                    (e_field_minus, h_field_minus),
                )
            ):
                radial_magnitude, signed_radial, tangential = poynting_components(
                    e_field, h_field
                )
                mode_radial_magnitude[
                    polarization_index, mode_index, frequency_index
                ] = radial_magnitude
                mode_signed_radial[
                    polarization_index, mode_index, frequency_index
                ] = signed_radial
                mode_tangential[
                    polarization_index, mode_index, frequency_index
                ] = tangential
                # Use the nonnegative magnitude for the modal convergence test,
                # matching the original example's stopping observable.  Signed
                # outward and tangential closure are checked on the reconstructed
                # physical dipole below.
                mode_powers[polarization_index, mode_index, frequency_index] = (
                    2
                    * math.pi
                    * FARFIELD_RADIUS_UM**2
                    * np.trapezoid(
                        radial_magnitude * np.sin(polar_rad), polar_rad
                    )
                )

            phase_plus = cmath.exp(1j * m_value * AZIMUTHAL_RAD)
            e_field_total += e_field_plus * phase_plus
            h_field_total += h_field_plus * phase_plus
            if m_value > 0:
                phase_minus = cmath.exp(-1j * m_value * AZIMUTHAL_RAD)
                e_field_total += np.conj(e_field_minus) * phase_minus
                h_field_total += np.conj(h_field_minus) * phase_minus

            cumulative_e_fields[polarization_index, mode_index] = e_field_total
            cumulative_h_fields[polarization_index, mode_index] = h_field_total
            (
                cumulative_radial_magnitude[polarization_index, mode_index],
                cumulative_signed_radial[polarization_index, mode_index],
                cumulative_tangential[polarization_index, mode_index],
            ) = poynting_components(e_field_total, h_field_total)

    power_scales = np.maximum(np.max(mode_powers, axis=(1, 2)), 1e-300)
    mode_power_ratios = mode_powers / power_scales[:, np.newaxis, np.newaxis]
    tail_power_ratios = mode_power_ratios[:, -3:, :]
    consecutive_tail_counts = np.zeros(num_polarizations, dtype=int)
    for polarization_index in range(num_polarizations):
        for mode_index in range(num_modes - 1, -1, -1):
            if np.max(mode_power_ratios[polarization_index, mode_index]) >= (
                POWER_DECAY_THRESHOLD
            ):
                break
            consecutive_tail_counts[polarization_index] += 1

    final_radial_magnitude = cumulative_radial_magnitude[:, -1]
    final_signed_radial = cumulative_signed_radial[:, -1]
    final_tangential = cumulative_tangential[:, -1]
    final_scaled_patterns = final_signed_radial * FARFIELD_RADIUS_UM**2
    pattern_scales = np.maximum(np.max(final_scaled_patterns, axis=1), 1e-300)
    normalized_patterns = final_scaled_patterns / pattern_scales[:, np.newaxis]
    analytic_patterns = np.stack(
        [np.square(np.cos(polar_rad)), np.ones(NUM_FARFIELD_PTS)]
    )
    analytic_relative_errors = np.linalg.norm(
        normalized_patterns - analytic_patterns, axis=1
    ) / np.linalg.norm(analytic_patterns, axis=1)

    radial_norms = np.maximum(
        np.linalg.norm(final_radial_magnitude, axis=1), 1e-300
    )
    directional_errors = np.column_stack(
        [
            np.linalg.norm(final_signed_radial - final_radial_magnitude, axis=1)
            / radial_norms,
            np.linalg.norm(final_tangential, axis=1) / radial_norms,
        ]
    )
    minimum_signed_radial_ratios = np.min(final_signed_radial, axis=1) / np.maximum(
        np.max(final_radial_magnitude, axis=1), 1e-300
    )

    last_mode_relative_changes = np.zeros((num_polarizations, 3))
    for polarization_index in range(num_polarizations):
        for quantity_index, quantity in enumerate(
            (
                cumulative_e_fields,
                cumulative_h_fields,
                cumulative_signed_radial,
            )
        ):
            final = quantity[polarization_index, -1]
            previous = quantity[polarization_index, -2]
            last_mode_relative_changes[polarization_index, quantity_index] = (
                np.linalg.norm(final - previous)
                / max(float(np.linalg.norm(final)), 1e-300)
            )

    frequency_power_relative_differences = np.abs(
        mode_powers[:, :, 0] - mode_powers[:, :, 1]
    ) / np.maximum(np.max(mode_powers, axis=2), 1e-300)
    frequency_power_absolute_differences = np.abs(
        mode_powers[:, :, 0] - mode_powers[:, :, 1]
    )
    frequency_power_peaks = np.maximum(
        np.max(mode_powers, axis=(1, 2)), 1e-300
    )
    frequency_power_peak_normalized_differences = (
        frequency_power_absolute_differences / frequency_power_peaks[:, None]
    )
    frequency_power_significant_mask = (
        np.max(mode_powers, axis=2) / frequency_power_peaks[:, None] >= 1e-4
    )
    frequency_power_significant_relative_differences = np.where(
        frequency_power_significant_mask,
        frequency_power_relative_differences,
        0.0,
    )

    arrays_to_check = (
        e_fields_plus,
        h_fields_plus,
        e_fields_minus,
        h_fields_minus,
        mode_radial_magnitude,
        mode_signed_radial,
        mode_tangential,
        mode_powers,
        cumulative_e_fields,
        cumulative_h_fields,
        cumulative_radial_magnitude,
        cumulative_signed_radial,
        cumulative_tangential,
        normalized_patterns,
        analytic_relative_errors,
        directional_errors,
        last_mode_relative_changes,
        frequency_power_relative_differences,
        frequency_power_absolute_differences,
        frequency_power_peak_normalized_differences,
        frequency_power_significant_relative_differences,
    )
    if any(np.any(~np.isfinite(array)) for array in arrays_to_check):
        raise RuntimeError("off-axis dipole validation produced nonfinite observables")
    if np.any(mode_powers < 0) or np.any(np.max(mode_powers, axis=(1, 2)) <= 0):
        raise RuntimeError(
            "off-axis dipole modal powers must be nonnegative with nonzero total power"
        )
    if np.any(consecutive_tail_counts < 3) or np.any(
        tail_power_ratios >= POWER_DECAY_THRESHOLD
    ):
        raise RuntimeError(
            "off-axis dipole fixed mode sweep did not contain three converged tail modes: "
            f"counts={consecutive_tail_counts.tolist()}, "
            f"ratios={tail_power_ratios.tolist()}"
        )
    if np.any(np.diff(np.max(tail_power_ratios, axis=2), axis=1) > 0):
        raise RuntimeError("off-axis dipole tail powers did not decrease monotonically")
    if np.max(last_mode_relative_changes) > 5e-3:
        raise RuntimeError(
            "off-axis dipole reconstruction changed after its final mode: "
            f"{last_mode_relative_changes.tolist()}"
        )
    if (
        np.max(frequency_power_relative_differences)
        > FREQUENCY_POWER_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            "off-axis dipole positive/negative-frequency powers disagree: "
            f"{frequency_power_relative_differences.tolist()}"
        )
    if np.max(frequency_power_significant_relative_differences) > 1e-6:
        raise RuntimeError(
            "off-axis dipole significant positive/negative-frequency powers "
            "disagree: "
            f"{frequency_power_significant_relative_differences.tolist()}"
        )
    if (
        np.max(frequency_power_peak_normalized_differences)
        > FREQUENCY_POWER_PEAK_NORMALIZED_TOLERANCE
    ):
        raise RuntimeError(
            "off-axis dipole positive/negative-frequency absolute imbalance "
            "is too large relative to peak modal power: "
            f"{frequency_power_peak_normalized_differences.tolist()}"
        )
    if np.max(analytic_relative_errors) > 0.06:
        raise RuntimeError(
            "off-axis dipole radiation pattern disagrees with dipole theory: "
            f"{analytic_relative_errors.tolist()}"
        )
    if np.max(directional_errors) > 1e-3 or np.any(
        minimum_signed_radial_ratios < -1e-4
    ):
        raise RuntimeError(
            "off-axis dipole far-field power is not outward and radial: "
            f"errors={directional_errors.tolist()}, "
            f"minimum={minimum_signed_radial_ratios.tolist()}"
        )

    return {
        "offaxis_polarization_codes": np.asarray([0, 1], dtype=float),
        "offaxis_m_values": m_values.astype(float),
        "offaxis_parameters": np.asarray(
            [
                dipole_pos_r,
                RESOLUTION_UM,
                maximum_m,
                NUM_FARFIELD_PTS,
                FARFIELD_RADIUS_UM,
                POWER_DECAY_THRESHOLD,
            ],
            dtype=float,
        ),
        "offaxis_polar_angles": np.asarray(polar_rad),
        "offaxis_e_fields_plus": e_fields_plus,
        "offaxis_h_fields_plus": h_fields_plus,
        "offaxis_e_fields_minus": e_fields_minus,
        "offaxis_h_fields_minus": h_fields_minus,
        "offaxis_mode_radial_magnitude": mode_radial_magnitude,
        "offaxis_mode_signed_radial": mode_signed_radial,
        "offaxis_mode_tangential": mode_tangential,
        "offaxis_mode_powers": mode_powers,
        "offaxis_mode_power_ratios": mode_power_ratios,
        "offaxis_frequency_power_relative_differences": (
            frequency_power_relative_differences
        ),
        "offaxis_frequency_power_peak_normalized_differences": (
            frequency_power_peak_normalized_differences
        ),
        "offaxis_frequency_power_significant_mask": (
            frequency_power_significant_mask.astype(float)
        ),
        "offaxis_frequency_power_significant_relative_differences": (
            frequency_power_significant_relative_differences
        ),
        "offaxis_tail_power_ratios": tail_power_ratios,
        "offaxis_consecutive_tail_counts": consecutive_tail_counts.astype(float),
        "offaxis_cumulative_e_fields": cumulative_e_fields,
        "offaxis_cumulative_h_fields": cumulative_h_fields,
        "offaxis_cumulative_radial_magnitude": cumulative_radial_magnitude,
        "offaxis_cumulative_signed_radial": cumulative_signed_radial,
        "offaxis_cumulative_tangential": cumulative_tangential,
        "offaxis_final_scaled_patterns": final_scaled_patterns,
        "offaxis_normalized_patterns": normalized_patterns,
        "offaxis_analytic_patterns": analytic_patterns,
        "offaxis_analytic_relative_errors": analytic_relative_errors,
        "offaxis_directional_errors": directional_errors,
        "offaxis_minimum_signed_radial_ratios": minimum_signed_radial_ratios,
        "offaxis_last_mode_relative_changes": last_mode_relative_changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dipole_pol",
        nargs="?",
        type=str,
        choices=["x", "y"],
        help="polarization of the electric dipole (x or y)",
    )
    parser.add_argument(
        "dipole_pos_r",
        nargs="?",
        type=float,
        help="radial position of the dipole",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run both polarizations through a fixed converged Fourier-mode sweep",
    )
    args = parser.parse_args()

    if args.validation:
        if args.dipole_pol is not None or args.dipole_pos_r is not None:
            parser.error("--validation does not accept positional arguments")
        # A radial offset of 0.3 wavelength is far enough from the axis to
        # require several azimuthal modes.  m=0..8 includes three modes below
        # the 1e-4 power threshold without relying on the first low-power term.
        RESOLUTION_UM = 25
        NUM_FARFIELD_PTS = 32
        polar_rad = np.linspace(0, 0.5 * math.pi, NUM_FARFIELD_PTS)
        validation_results = validation_mode_sweep(0.3, 8)
        globals().update(validation_results)
        frequency_power_max = float(
            np.max(offaxis_frequency_power_relative_differences)
        )
        significant_frequency_power_max = float(
            np.max(offaxis_frequency_power_significant_relative_differences)
        )
        peak_normalized_frequency_power_max = float(
            np.max(offaxis_frequency_power_peak_normalized_differences)
        )
        print(
            "offaxis_validation:,"
            f" analytic={offaxis_analytic_relative_errors.tolist()},"
            f" directional={offaxis_directional_errors.tolist()},"
            f" tail_counts={offaxis_consecutive_tail_counts.tolist()},"
            f" last_mode={offaxis_last_mode_relative_changes.tolist()},"
            f" frequency_power_max={frequency_power_max:.8g},"
            " significant_frequency_power_max="
            f"{significant_frequency_power_max:.8g},"
            " peak_normalized_frequency_power_max="
            f"{peak_normalized_frequency_power_max:.8g}"
        )
    else:
        if args.dipole_pol is None or args.dipole_pos_r is None:
            parser.error("dipole_pol and dipole_pos_r are required")
        if args.dipole_pos_r == 0:
            raise ValueError("dipole_pos_r must be nonzero.")

        # Fourier series expansion of the fields from a ring current source
        # used to generate a point dipole localized in the azimuthal direction.

        e_field_total = np.zeros((NUM_FARFIELD_PTS, 3), dtype=np.complex128)
        h_field_total = np.zeros((NUM_FARFIELD_PTS, 3), dtype=np.complex128)
        flux_max = 0
        m = 0
        while True:
            (
                e_field_plus,
                h_field_plus,
                e_field_minus,
                h_field_minus,
            ) = dipole_in_vacuum(args.dipole_pol, args.dipole_pos_r, m)
            e_field_total += e_field_plus * cmath.exp(1j * m * AZIMUTHAL_RAD)
            h_field_total += h_field_plus * cmath.exp(1j * m * AZIMUTHAL_RAD)

            if m > 0:
                e_field_total += np.conj(e_field_minus) * cmath.exp(
                    -1j * m * AZIMUTHAL_RAD
                )
                h_field_total += np.conj(h_field_minus) * cmath.exp(
                    -1j * m * AZIMUTHAL_RAD
                )

            flux = flux_from_farfields(e_field_plus, h_field_plus)
            if flux > flux_max:
                flux_max = flux
            power_decay = flux / flux_max
            print(f"power_decay:, {m}, {flux}, {flux_max}, {power_decay}")

            if m > 0 and power_decay < POWER_DECAY_THRESHOLD:
                break
            else:
                m += 1

        dipole_radiation_pattern = radiation_pattern(e_field_total, h_field_total)
        dipole_radiation_pattern_scaled = (
            dipole_radiation_pattern * FARFIELD_RADIUS_UM**2
        )
        plot_radiation_pattern(args.dipole_pol, dipole_radiation_pattern_scaled)

        if mp.am_master():
            np.savez(
                "dipole_farfields_off_axis.npz",
                AZIMUTHAL_RAD=AZIMUTHAL_RAD,
                FARFIELD_RADIUS_UM=FARFIELD_RADIUS_UM,
                PML_UM=PML_UM,
                POWER_DECAY_THRESHOLD=POWER_DECAY_THRESHOLD,
                RESOLUTION_UM=RESOLUTION_UM,
                WAVELENGTH_UM=WAVELENGTH_UM,
                dipole_pol=args.dipole_pol,
                dipole_pos_r=args.dipole_pos_r,
                dipole_radiation_pattern=dipole_radiation_pattern,
                e_field_total=e_field_total,
                h_field_total=h_field_total,
                m=m,
                polar_rad=polar_rad,
            )
