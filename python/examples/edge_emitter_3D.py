#!/usr/bin/env python3
"""Computes the extraction efficiency of a cleaved stack using a 3D sim."""

import argparse
from typing import Dict, Tuple, Union

import meep as mp
import matplotlib.pyplot as plt
import numpy as np


RESOLUTION_UM = 25
WAVELENGTH_UM = 1.0
CLADDING_UM = 0.5
ACTIVE_UM = 0.3
SUBSTRATE_UM = 1.0
PML_UM = 0.5
AIR_UM = 1.0
SIDE_UM = 3.0
SIDE_Z_UM = 10.0  # convergence parameter
N_SUBSTRATE = 3.2
N_CLADDING = 1.5
N_ACTIVE = 2.3
FIELD_DECAY_PERIOD = 25.0
FIELD_DECAY_TOL = 1e-6
DEBUG_OUTPUT = 0


def dipole_in_cleaved_stack(
    dipole_pol: str,
    dipole_pos_um: float,
    *,
    resolution_um: int = RESOLUTION_UM,
    side_z_um: float = SIDE_Z_UM,
    frequency_width: float = 0.0,
    num_frequencies: int = 1,
    field_decay_period: float = FIELD_DECAY_PERIOD,
    field_decay_tolerance: float = FIELD_DECAY_TOL,
    return_details: bool = False,
) -> Union[Tuple[float, float], Dict[str, object]]:
    """Computes the flux of a dipole in a cleaved multilayer stack.

    Args:
        dipole_pol: polarization of the dipole ("x", "y", or "z").
        dipole_pos_um: position of the dipole relative to the edge of the
          active region.
        return_details: return complete spectral monitors and work metadata
          instead of the original center-frequency flux tuple.

    Returns:
        The original emitted/air flux 2-tuple, or a spectral-details mapping
        when ``return_details`` is true.
    """
    if dipole_pol == "x":
        src_cmpt = mp.Ex
        symmetries = [mp.Mirror(direction=mp.Z)]
    elif dipole_pol == "y":
        src_cmpt = mp.Ey
        symmetries = [mp.Mirror(direction=mp.Z)]
    elif dipole_pol == "z":
        src_cmpt = mp.Ez
        symmetries = [mp.Mirror(direction=mp.Z, phase=-1)]
    else:
        raise ValueError("dipole_pol must be x, y, or z.")

    cell_x_um = PML_UM + SIDE_UM + AIR_UM + PML_UM
    cell_y_um = (
        PML_UM + SUBSTRATE_UM + CLADDING_UM + ACTIVE_UM + CLADDING_UM + AIR_UM + PML_UM
    )
    if resolution_um <= 0 or side_z_um <= 2 * PML_UM:
        raise ValueError("resolution_um and side_z_um must define a nonempty 3D cell")
    if num_frequencies <= 0 or frequency_width < 0:
        raise ValueError("invalid DFT frequency controls")
    if field_decay_period <= 0 or not 0 < field_decay_tolerance < 1:
        raise ValueError("invalid field-decay controls")

    cell_z_um = PML_UM + side_z_um + PML_UM
    cell_size = mp.Vector3(cell_x_um, cell_y_um, cell_z_um)

    boundary_layers = [mp.PML(thickness=PML_UM)]

    frequency = 1 / WAVELENGTH_UM

    src_pos = mp.Vector3(
        -0.5 * cell_x_um + PML_UM + SIDE_UM - dipole_pos_um,
        -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + CLADDING_UM + 0.5 * ACTIVE_UM,
        0,
    )
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            center=src_pos,
            component=src_cmpt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_SUBSTRATE),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um + 0.5 * (PML_UM + SUBSTRATE_UM),
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, PML_UM + SUBSTRATE_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_CLADDING),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + 0.5 * CLADDING_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_ACTIVE),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um
                + PML_UM
                + SUBSTRATE_UM
                + CLADDING_UM
                + 0.5 * ACTIVE_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, ACTIVE_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_CLADDING),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um
                + PML_UM
                + SUBSTRATE_UM
                + CLADDING_UM
                + ACTIVE_UM
                + 0.5 * CLADDING_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf),
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution_um,
        cell_size=cell_size,
        sources=sources,
        geometry=geometry,
        boundary_layers=boundary_layers,
        symmetries=symmetries,
    )

    flux_regions = [
        mp.FluxRegion(
            center=mp.Vector3(
                -0.5 * cell_x_um + PML_UM + 0.5 * (cell_x_um - 2 * PML_UM),
                0.5 * cell_y_um - PML_UM,
                0,
            ),
            size=mp.Vector3(cell_x_um - 2 * PML_UM, 0, cell_z_um - 2 * PML_UM),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM,
                0.5 * cell_y_um - PML_UM - 0.5 * (cell_y_um - 2 * PML_UM),
                0,
            ),
            size=mp.Vector3(0, cell_y_um - 2 * PML_UM, cell_z_um - 2 * PML_UM),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM - 0.5 * AIR_UM, -0.5 * cell_y_um + PML_UM, 0
            ),
            size=mp.Vector3(AIR_UM, 0, cell_z_um - 2 * PML_UM),
            weight=-1.0,
        ),
    ]
    flux_mon = sim.add_flux(
        frequency, frequency_width, num_frequencies, *flux_regions
    )
    face_flux_monitors = (
        [
            sim.add_flux(frequency, frequency_width, num_frequencies, region)
            for region in flux_regions
        ]
        if return_details
        else []
    )

    if DEBUG_OUTPUT:
        if mp.am_master():
            fig, ax = plt.subplots()
            sim.plot2D(
                ax=ax,
                output_plane=mp.Volume(
                    center=mp.Vector3(),
                    size=mp.Vector3(cell_x_um, cell_y_um, 0),
                ),
            )
            ax.set_title("z = 0")
            fig.savefig("edge_emitter_layout_3D.png", dpi=150, bbox_inches="tight")

    ldos_mon = mp.Ldos(frequency, frequency_width, num_frequencies)
    sim.run(
        mp.dft_ldos(ldos=ldos_mon),
        until_after_sources=mp.stop_when_fields_decayed(
            field_decay_period,
            src_cmpt,
            src_pos,
            field_decay_tolerance,
        ),
    )

    voxel_volume = (1 / resolution_um) ** 3
    ldos_fdata = np.asarray(sim.ldos_Fdata, dtype=np.complex128).copy()
    ldos_jdata = np.asarray(sim.ldos_Jdata, dtype=np.complex128).copy()
    ldos_data = np.asarray(sim.ldos_data, dtype=float).copy()
    dipole_flux_spectrum = (
        -np.real(ldos_fdata * np.conj(ldos_jdata)) * voxel_volume
    )
    air_flux_spectrum = np.asarray(mp.get_fluxes(flux_mon), dtype=float)
    center_index = num_frequencies // 2
    dipole_flux = float(dipole_flux_spectrum[center_index])
    air_flux = float(air_flux_spectrum[center_index])

    if not return_details:
        return dipole_flux, air_flux

    face_flux_spectrum = np.stack(
        [np.asarray(mp.get_fluxes(monitor), dtype=float) for monitor in face_flux_monitors]
    )
    flux_frequencies = np.asarray(mp.get_flux_freqs(flux_mon), dtype=float)
    ldos_frequencies = np.asarray(mp.get_ldos_freqs(ldos_mon), dtype=float)
    timestep = int(sim.timestep())
    meep_time = float(sim.meep_time())
    full_grid_shape = np.rint(
        resolution_um * np.asarray([cell_x_um, cell_y_um, cell_z_um])
    ).astype(int)
    full_grid_voxels = int(np.prod(full_grid_shape, dtype=np.int64))
    details = {
        "polarization": dipole_pol,
        "frequencies": flux_frequencies,
        "ldos_frequencies": ldos_frequencies,
        "dipole_flux": dipole_flux_spectrum,
        "air_flux": air_flux_spectrum,
        "air_face_flux": face_flux_spectrum,
        "extraction_efficiency": air_flux_spectrum / dipole_flux_spectrum,
        "ldos": ldos_data,
        "ldos_fdata": ldos_fdata,
        "ldos_jdata": ldos_jdata,
        # Columns: resolution, SIDE_Z, cell x/y/z, stop time, timesteps,
        # nominal full-cell voxels, and the z-mirror symmetry factor.
        "work_summary": np.asarray(
            [
                resolution_um,
                side_z_um,
                cell_x_um,
                cell_y_um,
                cell_z_um,
                meep_time,
                timestep,
                full_grid_voxels,
                2,
            ],
            dtype=float,
        ),
    }
    sim.reset_meep()
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dipole_pol",
        type=str,
        nargs="?",
        choices=["x", "y", "z"],
        help="polarization of the electric dipole (x, y, z)",
    )
    parser.add_argument(
        "dipole_pos_um",
        type=float,
        nargs="?",
        help="position of the dipole relative to the edge of the cleaved facet",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run all three dipole polarizations and publish 3D spectral observables",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--side-z", type=float, default=None)
    parser.add_argument("--control-side-z", type=float, default=None)
    parser.add_argument("--num-frequencies", type=int, default=None)
    parser.add_argument("--frequency-width", type=float, default=None)
    parser.add_argument("--decay-period", type=float, default=None)
    parser.add_argument("--decay-tolerance", type=float, default=None)
    args = parser.parse_args()

    if not args.validation:
        if args.dipole_pol is None or args.dipole_pos_um is None:
            parser.error("dipole_pol and dipole_pos_um are required without --validation")
        dipole_flux, air_flux = dipole_in_cleaved_stack(
            args.dipole_pol, args.dipole_pos_um
        )

        extraction_efficiency = air_flux / dipole_flux

        print(
            f"extraction_efficiency:, {args.dipole_pol}, {args.dipole_pos_um},"
            f" {100 * extraction_efficiency:.2f}%"
        )
    else:
        if args.dipole_pol is not None:
            parser.error("--validation always executes x, y, and z polarizations")
        dipole_position = 0.5 if args.dipole_pos_um is None else args.dipole_pos_um
        validation_resolution = 10 if args.resolution is None else args.resolution
        validation_side_z = 4.0 if args.side_z is None else args.side_z
        validation_control_side_z = (
            5.0 if args.control_side_z is None else args.control_side_z
        )
        validation_num_frequencies = (
            5 if args.num_frequencies is None else args.num_frequencies
        )
        validation_frequency_width = (
            0.1 if args.frequency_width is None else args.frequency_width
        )
        validation_decay_period = (
            20.0 if args.decay_period is None else args.decay_period
        )
        validation_decay_tolerance = (
            1e-5 if args.decay_tolerance is None else args.decay_tolerance
        )
        if validation_resolution < 10:
            raise RuntimeError("3D validation resolution must be at least 10 pixels/um")
        if validation_side_z < 4 or validation_control_side_z <= validation_side_z:
            raise RuntimeError("3D validation requires a larger SIDE_Z convergence control")
        if validation_num_frequencies < 5 or validation_num_frequencies % 2 != 1:
            raise RuntimeError(
                "3D validation requires at least five, and an odd number of, frequencies"
            )
        if not 0 < validation_frequency_width <= 0.4:
            raise RuntimeError("3D validation frequency width is out of range")
        if not 0 < dipole_position < SIDE_UM:
            raise RuntimeError("validation dipole must remain inside the active stack")

        validation_kwargs = {
            "resolution_um": validation_resolution,
            "side_z_um": validation_side_z,
            "frequency_width": validation_frequency_width,
            "num_frequencies": validation_num_frequencies,
            "field_decay_period": validation_decay_period,
            "field_decay_tolerance": validation_decay_tolerance,
            "return_details": True,
        }
        edge_emitter_results = [
            dipole_in_cleaved_stack(pol, dipole_position, **validation_kwargs)
            for pol in ("x", "y", "z")
        ]
        control_kwargs = dict(validation_kwargs)
        control_kwargs["side_z_um"] = validation_control_side_z
        edge_emitter_control_result = dipole_in_cleaved_stack(
            "x", dipole_position, **control_kwargs
        )

        edge_emitter_polarizations = np.asarray([0, 1, 2], dtype=float)
        edge_emitter_frequencies = np.asarray(
            edge_emitter_results[0]["frequencies"], dtype=float
        )
        edge_emitter_ldos_frequencies = np.stack(
            [result["ldos_frequencies"] for result in edge_emitter_results]
        )
        edge_emitter_dipole_flux = np.stack(
            [result["dipole_flux"] for result in edge_emitter_results]
        )
        edge_emitter_air_flux = np.stack(
            [result["air_flux"] for result in edge_emitter_results]
        )
        edge_emitter_air_face_flux = np.stack(
            [result["air_face_flux"] for result in edge_emitter_results]
        )
        edge_emitter_extraction_efficiency = np.stack(
            [result["extraction_efficiency"] for result in edge_emitter_results]
        )
        edge_emitter_ldos = np.stack(
            [result["ldos"] for result in edge_emitter_results]
        )
        edge_emitter_ldos_fdata = np.stack(
            [result["ldos_fdata"] for result in edge_emitter_results]
        )
        edge_emitter_ldos_jdata = np.stack(
            [result["ldos_jdata"] for result in edge_emitter_results]
        )
        edge_emitter_work_summary = np.stack(
            [result["work_summary"] for result in edge_emitter_results]
        )
        center_index = validation_num_frequencies // 2
        edge_emitter_center_fluxes = np.stack(
            [
                edge_emitter_dipole_flux[:, center_index],
                edge_emitter_air_flux[:, center_index],
                edge_emitter_extraction_efficiency[:, center_index],
            ],
            axis=1,
        )
        # This compares a monitor containing all three regions with the sum of
        # three monitors containing the identical regions individually.  It is
        # an additivity/API consistency check, not an independent energy
        # conservation law.
        edge_emitter_monitor_additivity_error = np.abs(
            np.sum(edge_emitter_air_face_flux, axis=1) - edge_emitter_air_flux
        ) / np.maximum(np.abs(edge_emitter_air_flux), 1e-30)
        edge_emitter_run_counters = np.asarray(
            [
                4,
                np.sum(edge_emitter_work_summary[:, 6])
                + edge_emitter_control_result["work_summary"][6],
                np.sum(edge_emitter_work_summary[:, 7])
                + edge_emitter_control_result["work_summary"][7],
                np.sum(
                    edge_emitter_work_summary[:, 6]
                    * edge_emitter_work_summary[:, 7]
                )
                + edge_emitter_control_result["work_summary"][6]
                * edge_emitter_control_result["work_summary"][7],
                validation_num_frequencies,
            ],
            dtype=float,
        )
        edge_emitter_validation_parameters = np.asarray(
            [
                dipole_position,
                validation_resolution,
                validation_side_z,
                validation_control_side_z,
                validation_num_frequencies,
                validation_frequency_width,
                validation_decay_period,
                validation_decay_tolerance,
            ],
            dtype=float,
        )

        edge_emitter_control_dipole_flux = np.asarray(
            edge_emitter_control_result["dipole_flux"]
        )
        edge_emitter_control_air_flux = np.asarray(
            edge_emitter_control_result["air_flux"]
        )
        edge_emitter_control_efficiency = np.asarray(
            edge_emitter_control_result["extraction_efficiency"]
        )
        edge_emitter_control_air_face_flux = np.asarray(
            edge_emitter_control_result["air_face_flux"]
        )
        edge_emitter_control_ldos = np.asarray(edge_emitter_control_result["ldos"])
        edge_emitter_control_work_summary = np.asarray(
            edge_emitter_control_result["work_summary"]
        )
        edge_emitter_control_monitor_additivity_error = np.abs(
            np.sum(edge_emitter_control_air_face_flux, axis=0)
            - edge_emitter_control_air_flux
        ) / np.maximum(np.abs(edge_emitter_control_air_flux), 1e-30)
        efficiency_scale = max(
            abs(edge_emitter_extraction_efficiency[0, center_index]), 1e-30
        )
        edge_emitter_convergence_summary = np.asarray(
            [
                validation_side_z,
                validation_control_side_z,
                abs(
                    edge_emitter_control_efficiency[center_index]
                    - edge_emitter_extraction_efficiency[0, center_index]
                )
                / efficiency_scale,
                np.linalg.norm(
                    edge_emitter_control_dipole_flux - edge_emitter_dipole_flux[0]
                )
                / max(np.linalg.norm(edge_emitter_dipole_flux[0]), 1e-30),
                np.linalg.norm(
                    edge_emitter_control_air_flux - edge_emitter_air_flux[0]
                )
                / max(np.linalg.norm(edge_emitter_air_flux[0]), 1e-30),
                edge_emitter_control_work_summary[7]
                / edge_emitter_work_summary[0, 7],
            ],
            dtype=float,
        )

        spectral_arrays = (
            edge_emitter_dipole_flux,
            edge_emitter_air_flux,
            edge_emitter_air_face_flux,
            edge_emitter_extraction_efficiency,
            edge_emitter_ldos,
            edge_emitter_ldos_fdata,
            edge_emitter_ldos_jdata,
            edge_emitter_control_dipole_flux,
            edge_emitter_control_air_flux,
            edge_emitter_control_air_face_flux,
            edge_emitter_control_efficiency,
            edge_emitter_control_ldos,
        )
        if any(np.any(~np.isfinite(array)) for array in spectral_arrays):
            raise RuntimeError("3D edge-emitter validation produced non-finite spectra")
        if not np.allclose(
            edge_emitter_ldos_frequencies,
            edge_emitter_frequencies[None, :],
            rtol=0,
            atol=1e-14,
        ) or np.any(np.diff(edge_emitter_frequencies) <= 0):
            raise RuntimeError("3D edge-emitter flux and LDOS frequency grids differ")
        if np.any(edge_emitter_dipole_flux <= 1e-10) or np.any(
            edge_emitter_air_flux <= 1e-10
        ):
            raise RuntimeError("3D edge-emitter emitted flux is zero or non-physical")
        if np.any(edge_emitter_ldos <= 0):
            raise RuntimeError("3D edge-emitter LDOS spectrum is non-physical")
        if np.any(edge_emitter_control_dipole_flux <= 1e-10) or np.any(
            edge_emitter_control_air_flux <= 1e-10
        ):
            raise RuntimeError("3D edge-emitter SIDE_Z control emitted non-physical flux")
        if np.any(edge_emitter_control_efficiency <= 0) or np.any(
            edge_emitter_control_efficiency >= 1
        ) or np.any(edge_emitter_control_ldos <= 0):
            raise RuntimeError("3D edge-emitter SIDE_Z control left its physical range")
        if np.any(edge_emitter_extraction_efficiency <= 0) or np.any(
            edge_emitter_extraction_efficiency >= 1
        ):
            raise RuntimeError("3D edge-emitter extraction efficiency left its physical range")
        if max(
            np.max(edge_emitter_monitor_additivity_error),
            np.max(edge_emitter_control_monitor_additivity_error),
        ) > 2e-5:
            raise RuntimeError(
                "individual air-face flux monitors are not additive with the "
                "combined monitor"
            )
        if np.any(edge_emitter_work_summary[:, 6] < 1000) or np.any(
            edge_emitter_work_summary[:, 7] < 50000
        ):
            raise RuntimeError("3D edge-emitter validation performed trivial work")
        normalized_dipole_spectra = edge_emitter_dipole_flux / np.linalg.norm(
            edge_emitter_dipole_flux, axis=1, keepdims=True
        )
        polarization_separation = [
            np.linalg.norm(normalized_dipole_spectra[i] - normalized_dipole_spectra[j])
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        if min(polarization_separation) < 1e-3:
            raise RuntimeError("3D edge-emitter polarization branches collapsed")
        if np.max(edge_emitter_convergence_summary[2:5]) > 0.35:
            raise RuntimeError("3D edge-emitter SIDE_Z convergence control changed excessively")
        if not 1.15 < edge_emitter_convergence_summary[5] < 1.4:
            raise RuntimeError("3D edge-emitter SIDE_Z control did not increase physical work")
        if max(edge_emitter_convergence_summary[3:5]) < 1e-4:
            raise RuntimeError("3D edge-emitter SIDE_Z control was not physically consumed")
