"""Computes the extraction efficiency of a cleaved stack using a 2D sim."""

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
N_SUBSTRATE = 3.2
N_CLADDING = 1.5
N_ACTIVE = 2.3
FIELD_DECAY_PERIOD = 25.0
FIELD_DECAY_TOL = 1e-6
DELTA_KZ = 0.02
DIPOLE_FLUX_TOL = 0.02
DEBUG_OUTPUT = 0


def line_current_in_cleaved_stack(
    dipole_pol: str,
    dipole_pos_um: float,
    kz: float,
    *,
    resolution_um: int = RESOLUTION_UM,
    field_decay_period: float = FIELD_DECAY_PERIOD,
    field_decay_tolerance: float = FIELD_DECAY_TOL,
    return_details: bool = False,
) -> Union[Tuple[float, float], Dict[str, object]]:
    """Computes the flux of a line current in a cleaved multilayer stack.

    Args:
        dipole_pol: polarization of the dipole ("x", "y", or "z").
        dipole_pos_um: position of the dipole relative to the edge of the
          active region.
        kz: z-component of the Bloch-periodic wavevector.

    Returns:
        The flux emitted by a line current in the active region and into air as
          a 2-tuple.
    """
    if dipole_pol == "x":
        src_cmpt = mp.Ex
    elif dipole_pol == "y":
        src_cmpt = mp.Ey
    elif dipole_pol == "z":
        src_cmpt = mp.Ez
    else:
        raise ValueError("dipole_pol must be x, y, or z.")

    if resolution_um <= 0 or not 0 < dipole_pos_um < SIDE_UM:
        raise ValueError("resolution and active-region dipole position are invalid")
    if not np.isfinite(kz) or kz < 0:
        raise ValueError("kz must be finite and nonnegative")
    if field_decay_period <= 0 or not 0 < field_decay_tolerance < 1:
        raise ValueError("invalid field-decay controls")

    cell_x_um = PML_UM + SIDE_UM + AIR_UM + PML_UM
    cell_y_um = (
        PML_UM + SUBSTRATE_UM + CLADDING_UM + ACTIVE_UM + CLADDING_UM + AIR_UM + PML_UM
    )
    cell_size = mp.Vector3(cell_x_um, cell_y_um, 0)

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
        k_point=mp.Vector3(0, 0, kz),
        kz_2d="real/imag",
        force_complex_fields=False,
        sources=sources,
        geometry=geometry,
        boundary_layers=boundary_layers,
    )

    flux_regions = [
        mp.FluxRegion(
            center=mp.Vector3(
                -0.5 * cell_x_um + PML_UM + 0.5 * (cell_x_um - 2 * PML_UM),
                0.5 * cell_y_um - PML_UM,
                0,
            ),
            size=mp.Vector3(cell_x_um - 2 * PML_UM, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM,
                0.5 * cell_y_um - PML_UM - 0.5 * (cell_y_um - 2 * PML_UM),
                0,
            ),
            size=mp.Vector3(0, cell_y_um - 2 * PML_UM, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM - +0.5 * AIR_UM, -0.5 * cell_y_um + PML_UM, 0
            ),
            size=mp.Vector3(AIR_UM, 0, 0),
            weight=-1.0,
        ),
    ]
    flux_mon = sim.add_flux(frequency, 0, 1, *flux_regions)
    face_flux_monitors = (
        [sim.add_flux(frequency, 0, 1, region) for region in flux_regions]
        if return_details
        else []
    )

    if DEBUG_OUTPUT:
        if mp.am_master():
            fig, ax = plt.subplots()
            sim.plot2D(ax=ax)
            fig.savefig("edge_emitter_layout.png", dpi=150, bbox_inches="tight")

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            field_decay_period,
            src_cmpt,
            src_pos,
            field_decay_tolerance,
        ),
    )

    pixel_area = (1 / resolution_um) ** 2
    ldos_fdata = np.asarray(sim.ldos_Fdata, dtype=np.complex128).copy()
    ldos_jdata = np.asarray(sim.ldos_Jdata, dtype=np.complex128).copy()
    ldos_data = np.asarray(sim.ldos_data, dtype=float).copy()
    dipole_flux = -np.real(ldos_fdata[0] * np.conj(ldos_jdata[0])) * pixel_area

    air_flux = mp.get_fluxes(flux_mon)[0]

    if return_details:
        face_flux = np.asarray(
            [mp.get_fluxes(monitor)[0] for monitor in face_flux_monitors],
            dtype=float,
        )
        details = {
            "polarization": dipole_pol,
            "kz": float(kz),
            "dipole_flux": float(dipole_flux),
            "air_flux": float(air_flux),
            "face_flux": face_flux,
            "ldos": ldos_data,
            "ldos_fdata": ldos_fdata,
            "ldos_jdata": ldos_jdata,
            "work_summary": np.asarray(
                [
                    resolution_um,
                    kz,
                    cell_x_um,
                    cell_y_um,
                    sim.meep_time(),
                    sim.timestep(),
                    round(resolution_um * cell_x_um)
                    * round(resolution_um * cell_y_um),
                ],
                dtype=float,
            ),
        }
        sim.reset_meep()
        return details

    return dipole_flux, air_flux


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
        help="run a bounded three-polarization fixed-kz integration oracle",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-kz", type=int, default=None)
    parser.add_argument("--max-kz", type=float, default=None)
    parser.add_argument("--decay-period", type=float, default=None)
    parser.add_argument("--decay-tolerance", type=float, default=None)
    args = parser.parse_args()

    if not args.validation:
        if args.dipole_pol is None or args.dipole_pos_um is None:
            parser.error(
                "dipole_pol and dipole_pos_um are required without --validation"
            )
        num_kz = 0
        max_dipole_flux = 0
        dipole_flux = []
        air_flux = []

        # Brillouin-zone integration over kz.
        while True:
            kz = num_kz * DELTA_KZ
            dipole_flux_kz, air_flux_kz = line_current_in_cleaved_stack(
                args.dipole_pol, args.dipole_pos_um, kz
            )
            print(
                f"kz:, {num_kz}, {kz:.2f}, {dipole_flux_kz:.2f}, "
                f"{air_flux_kz:.2f}"
            )

            dipole_flux.append(dipole_flux_kz)
            air_flux.append(air_flux_kz)
            num_kz += 1

            if dipole_flux_kz > max_dipole_flux:
                max_dipole_flux = dipole_flux_kz
            elif (dipole_flux_kz / max_dipole_flux) < DIPOLE_FLUX_TOL:
                break

        dipole_flux = np.array(dipole_flux)
        air_flux = np.array(air_flux)
        extraction_efficiency = (air_flux[0] + 2 * np.sum(air_flux[1:])) / (
            dipole_flux[0] + 2 * np.sum(dipole_flux[1:])
        )

        print(
            f"extraction_efficiency:, {args.dipole_pol}, {args.dipole_pos_um:.2f},"
            f" {100 * extraction_efficiency:.2f}%"
        )
    else:
        if args.dipole_pol is not None:
            parser.error("--validation always executes x, y, and z polarizations")
        dipole_position = 0.5 if args.dipole_pos_um is None else args.dipole_pos_um
        validation_resolution = 20 if args.resolution is None else args.resolution
        validation_num_kz = 14 if args.num_kz is None else args.num_kz
        validation_max_kz = 2.6 if args.max_kz is None else args.max_kz
        validation_decay_period = (
            20.0 if args.decay_period is None else args.decay_period
        )
        validation_decay_tolerance = (
            1e-4 if args.decay_tolerance is None else args.decay_tolerance
        )
        if validation_resolution < 15 or validation_num_kz < 12:
            raise RuntimeError("2D edge-emitter validation grid is too small")
        if validation_max_kz < 2.4:
            raise RuntimeError(
                "2D edge-emitter validation must include the evanescent tail"
            )
        if not 0 < dipole_position < SIDE_UM:
            raise RuntimeError("validation dipole must remain inside the active stack")

        edge_emitter_2d_kz = np.linspace(0, validation_max_kz, validation_num_kz)
        edge_emitter_2d_polarizations = np.asarray([0, 1, 2], dtype=float)
        result_matrix = []
        for polarization in ("x", "y", "z"):
            result_matrix.append(
                [
                    line_current_in_cleaved_stack(
                        polarization,
                        dipole_position,
                        float(kz),
                        resolution_um=validation_resolution,
                        field_decay_period=validation_decay_period,
                        field_decay_tolerance=validation_decay_tolerance,
                        return_details=True,
                    )
                    for kz in edge_emitter_2d_kz
                ]
            )

        edge_emitter_2d_dipole_flux = np.asarray(
            [[result["dipole_flux"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_air_flux = np.asarray(
            [[result["air_flux"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_face_flux = np.asarray(
            [[result["face_flux"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_ldos = np.asarray(
            [[result["ldos"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_ldos_fdata = np.asarray(
            [[result["ldos_fdata"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_ldos_jdata = np.asarray(
            [[result["ldos_jdata"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_work_summary = np.asarray(
            [[result["work_summary"] for result in row] for row in result_matrix]
        )
        edge_emitter_2d_monitor_additivity_error = np.abs(
            np.sum(edge_emitter_2d_face_flux, axis=2) - edge_emitter_2d_air_flux
        ) / np.maximum(np.abs(edge_emitter_2d_air_flux), 1e-12)
        integrated_dipole = np.trapezoid(
            edge_emitter_2d_dipole_flux, edge_emitter_2d_kz, axis=1
        )
        integrated_air = np.trapezoid(
            edge_emitter_2d_air_flux, edge_emitter_2d_kz, axis=1
        )
        integrated_efficiency = integrated_air / integrated_dipole
        edge_emitter_2d_integrated_flux = np.stack(
            [integrated_dipole, integrated_air, integrated_efficiency], axis=1
        )
        tail_mask = edge_emitter_2d_kz >= 1.4
        terminal_dipole_ratio = np.abs(edge_emitter_2d_dipole_flux[:, -1]) / np.maximum(
            np.max(np.abs(edge_emitter_2d_dipole_flux), axis=1), 1e-30
        )
        air_tail_ratio = np.max(
            np.abs(edge_emitter_2d_air_flux[:, tail_mask]), axis=1
        ) / np.maximum(np.max(np.abs(edge_emitter_2d_air_flux), axis=1), 1e-30)
        pairwise_distances = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            pairwise_distances.append(
                np.linalg.norm(
                    edge_emitter_2d_dipole_flux[first]
                    - edge_emitter_2d_dipole_flux[second]
                )
                / max(
                    0.5
                    * (
                        np.linalg.norm(edge_emitter_2d_dipole_flux[first])
                        + np.linalg.norm(edge_emitter_2d_dipole_flux[second])
                    ),
                    1e-30,
                )
            )
        edge_emitter_2d_independent_checks = np.concatenate(
            [terminal_dipole_ratio, air_tail_ratio, np.asarray(pairwise_distances)]
        )
        edge_emitter_2d_run_counters = np.asarray(
            [
                3 * validation_num_kz,
                np.sum(edge_emitter_2d_work_summary[:, :, 5]),
                np.sum(edge_emitter_2d_work_summary[:, :, 6]),
                np.sum(
                    edge_emitter_2d_work_summary[:, :, 5]
                    * edge_emitter_2d_work_summary[:, :, 6]
                ),
            ],
            dtype=float,
        )
        edge_emitter_2d_validation_parameters = np.asarray(
            [
                dipole_position,
                validation_resolution,
                validation_num_kz,
                validation_max_kz,
                validation_decay_period,
                validation_decay_tolerance,
            ],
            dtype=float,
        )

        retained = (
            edge_emitter_2d_dipole_flux,
            edge_emitter_2d_air_flux,
            edge_emitter_2d_face_flux,
            edge_emitter_2d_ldos,
            edge_emitter_2d_ldos_fdata,
            edge_emitter_2d_ldos_jdata,
            edge_emitter_2d_integrated_flux,
            edge_emitter_2d_monitor_additivity_error,
            edge_emitter_2d_independent_checks,
        )
        if any(np.any(~np.isfinite(value)) for value in retained):
            raise RuntimeError("2D edge-emitter validation produced nonfinite data")
        if np.any(integrated_dipole <= 0) or np.any(integrated_air <= 0):
            raise RuntimeError("2D edge-emitter integrated power must be positive")
        if np.any(integrated_efficiency <= 0) or np.any(integrated_efficiency >= 1.2):
            raise RuntimeError("2D edge-emitter extraction efficiency is nonphysical")
        if np.max(edge_emitter_2d_monitor_additivity_error) > 2e-5:
            raise RuntimeError("2D edge-emitter face monitors fail additivity")
        if np.max(terminal_dipole_ratio) > 1e-3:
            raise RuntimeError("2D edge-emitter kz grid does not reach the source tail")
        if np.max(air_tail_ratio) > 0.05:
            raise RuntimeError("2D edge-emitter air light-cone tail is too large")
        if np.min(pairwise_distances) < 0.02:
            raise RuntimeError(
                "2D edge-emitter polarizations are not physically distinct"
            )
        print(
            "edge-emitter-2d-validation:, "
            f"efficiency={integrated_efficiency.tolist()}, "
            f"terminal={terminal_dipole_ratio.tolist()}, "
            f"air-tail={air_tail_ratio.tolist()}"
        )
