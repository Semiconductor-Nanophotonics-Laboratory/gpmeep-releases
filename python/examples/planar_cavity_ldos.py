# Computes the Purcell enhancement factor of an in-plane dipole in a planar
# dielectric cavity with lossless metallic walls. The result is computed in
# cylindrical and 3D coordinates and compared with the analytic result from:
# I. Abram et al., IEEE J. Quantum Electronics, Vol. 34, pp. 71-76 (1998).

# tutorial reference:
# https://meep.readthedocs.io/en/latest/Python_Tutorials/Local_Density_of_States/#planar-cavity-with-lossless-metallic-walls

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="compare bounded cylindrical and 3D LDOS trajectories with theory",
)
parser.add_argument("--resolution", type=int, default=None)
args = parser.parse_args()

# Note: Meep may round the cell dimensions to an integer number of pixels which
# could modify the cavity structure.
RESOLUTION_UM = (
    args.resolution if args.resolution is not None else (20 if args.validation else 71)
)
if RESOLUTION_UM <= 0:
    raise RuntimeError("resolution must be positive")

PML_UM = 0.5
BULK_UM = 4.0 if args.validation else 6.0
N_CAVITY = 2.4
WAVELENGTH_UM = 1.0
FIELD_DECAY_TOL = 1e-5 if args.validation else 1e-6
FIELD_DECAY_PERIOD = 10 if args.validation else 20

frequency = 1 / WAVELENGTH_UM


def ldos_cyl(cavity_um: Optional[float] = None) -> float:
    """Computes the LDOS of a dipole in a cavity or bulk media in cyl. coords.

    Args:
        cavity_um: thickness of the cavity. If None, bulk media is used.

    Returns:
        The LDOS of the dipole.
    """
    if cavity_um is None:
        cell_z_um = BULK_UM + 2 * PML_UM
        pml_layers = [mp.PML(thickness=PML_UM)]
    else:
        cell_z_um = cavity_um
        pml_layers = [mp.PML(thickness=PML_UM, direction=mp.R)]

    cell_r_um = BULK_UM + PML_UM
    cell_size = mp.Vector3(cell_r_um, 0, cell_z_um)

    # An Er source at r = 0 and m=±1 needs to be slightly offset.
    # https://github.com/NanoComp/meep/issues/2704
    dipole_rpos_um = 1.5 / RESOLUTION_UM

    src_pt = mp.Vector3(dipole_rpos_um, 0, 0)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=mp.Er,
            center=src_pt,
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        default_material=mp.Medium(index=N_CAVITY),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, mp.Er, src_pt, FIELD_DECAY_TOL
        ),
    )

    return sim.ldos_data[0]


def ldos_3d(cavity_um: Optional[float] = None) -> float:
    """Computes the LDOS of a dipole in a cavity or bulk media in 3D coords.

    Args:
        cavity_um: thickness of the cavity. If None, bulk media is used.

    Returns:
        The LDOS of the dipole.
    """
    if cavity_um is None:
        size_z_um = BULK_UM + 2 * PML_UM
        pml_layers = [mp.PML(thickness=PML_UM)]
    else:
        size_z_um = cavity_um
        pml_layers = [
            mp.PML(thickness=PML_UM, direction=mp.X),
            mp.PML(thickness=PML_UM, direction=mp.Y),
        ]

    size_xy_um = BULK_UM + 2 * PML_UM
    cell_size = mp.Vector3(size_xy_um, size_xy_um, size_z_um)

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=mp.Ex,
            center=mp.Vector3(),
        )
    ]

    symmetries = [
        mp.Mirror(direction=mp.X, phase=-1),
        mp.Mirror(direction=mp.Y),
        mp.Mirror(direction=mp.Z),
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        default_material=mp.Medium(index=N_CAVITY),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, mp.Ex, mp.Vector3(), FIELD_DECAY_TOL
        ),
    )

    return sim.ldos_data[0]


if __name__ == "__main__":
    ldos_bulk_cyl = ldos_cyl()
    ldos_bulk_3d = ldos_3d()

    cavity_um = (
        np.asarray([0.75, 1.25, 2.0])
        if args.validation
        else np.arange(0.50, 2.55, 0.05)
    )
    vacuum_cavity_um = cavity_um * WAVELENGTH_UM / N_CAVITY

    num_cavity_um = cavity_um.shape[0]
    ldos_cavity_cyl = np.zeros(num_cavity_um)
    ldos_cavity_3d = np.zeros(num_cavity_um)

    for j in range(num_cavity_um):
        ldos_cavity_cyl[j] = ldos_cyl(vacuum_cavity_um[j])
        ldos_cavity_3d[j] = ldos_3d(vacuum_cavity_um[j])
        purcell_cyl = ldos_cavity_cyl[j] / ldos_bulk_cyl
        purcell_3d = ldos_cavity_3d[j] / ldos_bulk_3d
        print(f"purcell:, {cavity_um[j]:.3f}, {purcell_cyl:.6f}, {purcell_3d:.6f}")

    # Purcell enhancement factor (relative to bulk medium)
    purcell_meep_cyl = ldos_cavity_cyl / ldos_bulk_cyl
    purcell_meep_3d = ldos_cavity_3d / ldos_bulk_3d

    # Equation 7 of 1998 reference.
    purcell_theory = 3 * np.fix(cavity_um + 0.5) / (4 * cavity_um) + (
        4 * np.power(np.fix(cavity_um + 0.5), 3) - np.fix(cavity_um + 0.5)
    ) / (16 * np.power(cavity_um, 3))

    planar_ldos_bulk = np.asarray([ldos_bulk_cyl, ldos_bulk_3d])
    planar_cavity_thickness_media = np.asarray(cavity_um)
    planar_cavity_thickness_vacuum = np.asarray(vacuum_cavity_um)
    planar_ldos_cavity = np.stack([ldos_cavity_cyl, ldos_cavity_3d])
    planar_purcell = np.stack([purcell_meep_cyl, purcell_meep_3d])
    planar_purcell_theory = np.asarray(purcell_theory)
    planar_purcell_theory_relative_error = np.abs(
        planar_purcell - planar_purcell_theory[None, :]
    ) / planar_purcell_theory[None, :]
    planar_cyl_3d_relative_difference = np.abs(
        purcell_meep_cyl - purcell_meep_3d
    ) / np.maximum(
        0.5 * (np.abs(purcell_meep_cyl) + np.abs(purcell_meep_3d)), 1e-30
    )
    planar_validation_parameters = np.asarray(
        [
            RESOLUTION_UM,
            PML_UM,
            BULK_UM,
            N_CAVITY,
            FIELD_DECAY_TOL,
            FIELD_DECAY_PERIOD,
        ]
    )

    if args.validation:
        retained = (
            planar_ldos_bulk,
            planar_ldos_cavity,
            planar_purcell,
            planar_purcell_theory,
            planar_purcell_theory_relative_error,
            planar_cyl_3d_relative_difference,
        )
        if not all(np.all(np.isfinite(values)) for values in retained):
            raise RuntimeError("planar-cavity LDOS validation produced nonfinite data")
        if np.min(planar_ldos_bulk) <= 0 or np.min(planar_ldos_cavity) <= 0:
            raise RuntimeError("planar-cavity LDOS must be positive")
        if np.max(planar_purcell_theory_relative_error) > 0.18:
            raise RuntimeError(
                "planar-cavity Purcell factor disagrees with theory: "
                f"{planar_purcell_theory_relative_error.tolist()}"
            )
        if np.max(planar_cyl_3d_relative_difference) > 0.25:
            raise RuntimeError(
                "cylindrical and 3D planar-cavity LDOS disagree: "
                f"{planar_cyl_3d_relative_difference.tolist()}"
            )
        print(
            "planar-cavity-validation:, "
            f"purcell={planar_purcell.tolist()}, "
            f"theory-error={planar_purcell_theory_relative_error.tolist()}, "
            f"cyl-3d={planar_cyl_3d_relative_difference.tolist()}"
        )

    if mp.am_master() and not args.validation:
        fig, ax = plt.subplots()
        ax.plot(cavity_um, purcell_meep_3d, "b-", label="Meep (3d)")
        ax.plot(cavity_um, purcell_meep_cyl, "r-", label="Meep (cylin.)")
        ax.plot(cavity_um, purcell_theory, "g-", label="theory")
        ax.plot(cavity_um, np.ones(len(cavity_um)), "k--")
        ax.set_xlabel("cavity thickness (in media)")
        ax.set_ylabel("Purcell enhancement factor")
        ax.set_title(
            "in-plane dipole at λ=1.0 μm in a planar cavity\n"
            "with n=2.4 and lossless metallic walls"
        )
        ax.axis([0.5, 2.5, 0.4, 3.1])
        ax.legend()
        fig.savefig(
            "cavity_purcell_factor_vs_thickness.png", dpi=150, bbox_inches="tight"
        )
