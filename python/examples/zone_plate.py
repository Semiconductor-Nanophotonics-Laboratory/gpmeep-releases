"""Computes the diffraction spectra of a zone plate in cylindrical coords."""

import argparse
import math

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run a bounded cylindrical m=-1 Near2Far focusing oracle",
)
parser.add_argument("--resolution", type=int, default=None)
args = parser.parse_args()

resolution_um = (
    args.resolution if args.resolution is not None else (12 if args.validation else 25)
)
if resolution_um <= 0:
    raise RuntimeError("resolution must be positive")

pml_um = 1.0
substrate_um = 2.0
padding_um = 2.0
height_um = 0.5
focal_length_um = 30 if args.validation else 200
scan_length_z_um = 20 if args.validation else 100
farfield_resolution_um = 4 if args.validation else 10

pml_layers = [mp.PML(thickness=pml_um)]

wavelength_um = 0.5
frequency = 1 / wavelength_um
frequench_width = 0.2 * frequency

# The number of zones in the zone plate.
# Odd-numbered zones impart a π phase shift and
# even-numbered zones impart no phase shift.
num_zones = 8 if args.validation else 25

# Specify the radius of each zone using the equation
# from https://en.wikipedia.org/wiki/Zone_plate.
zone_radius_um = np.zeros(num_zones)
for n in range(1, num_zones + 1):
    zone_radius_um[n - 1] = math.sqrt(
        n * wavelength_um * (focal_length_um + n * wavelength_um / 4)
    )

size_r_um = zone_radius_um[-1] + padding_um + pml_um
size_z_um = pml_um + substrate_um + height_um + padding_um + pml_um
cell_size = mp.Vector3(size_r_um, 0, size_z_um)

# Specify a (linearly polarized) planewave at normal incidence.
sources = [
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Er,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
    ),
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Ep,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
        amplitude=-1j,
    ),
]

glass = mp.Medium(index=1.5)

# Add the substrate.
geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(size_r_um, 0, pml_um + substrate_um),
        center=mp.Vector3(
            0.5 * size_r_um, 0, -0.5 * size_z_um + 0.5 * (pml_um + substrate_um)
        ),
    )
]

# Add the zone plates starting with the ones with largest radius.
for n in range(num_zones - 1, -1, -1):
    geometry.append(
        mp.Block(
            material=glass if n % 2 == 0 else mp.vacuum,
            size=mp.Vector3(zone_radius_um[n], 0, height_um),
            center=mp.Vector3(
                0.5 * zone_radius_um[n],
                0,
                -0.5 * size_z_um + pml_um + substrate_um + 0.5 * height_um,
            ),
        )
    )

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution_um,
    sources=sources,
    geometry=geometry,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

# Add the near-field monitor (must be entirely in air).
n2f_monitor = sim.add_near2far(
    frequency,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(0.5 * (size_r_um - pml_um), 0, 0.5 * size_z_um - pml_um),
        size=mp.Vector3(size_r_um - pml_um, 0, 0),
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(
            size_r_um - pml_um,
            0,
            0.5 * size_z_um - pml_um - 0.5 * (height_um + padding_um),
        ),
        size=mp.Vector3(0, 0, height_um + padding_um),
    ),
)

if not args.validation:
    fig, ax = plt.subplots()
    sim.plot2D(ax=ax)
    if mp.am_master():
        fig.savefig("zone_plate_layout.png", bbox_inches="tight", dpi=150)

# Timestep the fields until they have sufficiently decayed away.
sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        20.0 if args.validation else 50.0,
        mp.Er,
        mp.Vector3(0.5 * size_r_um, 0, 0),
        1e-5 if args.validation else 1e-6,
    )
)

farfields_r = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0.5 * (size_r_um - pml_um),
        0,
        -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um,
    ),
    size=mp.Vector3(size_r_um - pml_um, 0, 0),
)

farfields_z = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0, 0, -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um
    ),
    size=mp.Vector3(0, 0, scan_length_z_um),
)

intensity_r = (
    np.absolute(farfields_r["Ex"]) ** 2
    + np.absolute(farfields_r["Ey"]) ** 2
    + np.absolute(farfields_r["Ez"]) ** 2
)
intensity_z = (
    np.absolute(farfields_z["Ex"]) ** 2
    + np.absolute(farfields_z["Ey"]) ** 2
    + np.absolute(farfields_z["Ez"]) ** 2
)

zone_plate_component_order = np.asarray(
    [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz], dtype=float
)
zone_plate_farfields_r = np.stack(
    [farfields_r[name] for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
)
zone_plate_farfields_z = np.stack(
    [farfields_z[name] for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
)
zone_plate_eh_group_norms = np.asarray(
    [
        np.linalg.norm(zone_plate_farfields_r[:3]),
        np.linalg.norm(zone_plate_farfields_r[3:]),
        np.linalg.norm(zone_plate_farfields_z[:3]),
        np.linalg.norm(zone_plate_farfields_z[3:]),
    ]
)
zone_plate_eh_group_ratios = np.asarray(
    [
        zone_plate_eh_group_norms[1] / max(zone_plate_eh_group_norms[0], 1e-30),
        zone_plate_eh_group_norms[3] / max(zone_plate_eh_group_norms[2], 1e-30),
    ]
)
zone_plate_eh_group_relative_mismatch = np.abs(
    zone_plate_eh_group_ratios - 1.0
)
zone_plate_intensity_r = np.asarray(intensity_r)
zone_plate_intensity_z = np.asarray(intensity_z)
zone_plate_r_coordinates = np.linspace(
    0, size_r_um - pml_um, zone_plate_intensity_r.size
)
zone_plate_z_coordinates = np.linspace(
    focal_length_um - 0.5 * scan_length_z_um,
    focal_length_um + 0.5 * scan_length_z_um,
    zone_plate_intensity_z.size,
)
zone_plate_peak_coordinates = np.asarray(
    [
        zone_plate_r_coordinates[int(np.argmax(zone_plate_intensity_r))],
        zone_plate_z_coordinates[int(np.argmax(zone_plate_intensity_z))],
    ]
)
zone_plate_focus_errors = np.asarray(
    [
        abs(zone_plate_peak_coordinates[0]),
        abs(zone_plate_peak_coordinates[1] - focal_length_um),
    ]
)
zone_plate_focus_contrast = np.asarray(
    [
        np.max(zone_plate_intensity_r)
        / max(zone_plate_intensity_r[-1], 1e-30),
        np.max(zone_plate_intensity_z)
        / max(
            0.5 * (zone_plate_intensity_z[0] + zone_plate_intensity_z[-1]),
            1e-30,
        ),
    ]
)
zone_plate_parameters = np.asarray(
    [
        resolution_um,
        num_zones,
        size_r_um,
        size_z_um,
        focal_length_um,
        scan_length_z_um,
        farfield_resolution_um,
    ]
)

if args.validation:
    retained = (
        zone_plate_farfields_r.real,
        zone_plate_farfields_r.imag,
        zone_plate_farfields_z.real,
        zone_plate_farfields_z.imag,
        zone_plate_eh_group_norms,
        zone_plate_eh_group_ratios,
        zone_plate_eh_group_relative_mismatch,
        zone_plate_intensity_r,
        zone_plate_intensity_z,
        zone_plate_peak_coordinates,
        zone_plate_focus_errors,
        zone_plate_focus_contrast,
    )
    if not all(np.all(np.isfinite(values)) for values in retained):
        raise RuntimeError("zone-plate validation produced nonfinite data")
    if np.linalg.norm(zone_plate_farfields_r) <= 1e-5 or np.linalg.norm(
        zone_plate_farfields_z
    ) <= 1e-5:
        raise RuntimeError("zone-plate far fields are empty")
    if np.min(zone_plate_eh_group_norms) <= 1e-3:
        raise RuntimeError(
            "zone-plate E/H far-field group is empty: "
            f"{zone_plate_eh_group_norms.tolist()}"
        )
    if np.min(zone_plate_eh_group_ratios) <= 0.05:
        raise RuntimeError(
            "zone-plate magnetic far field is inconsistent with electric radiation: "
            f"{zone_plate_eh_group_ratios.tolist()}"
        )
    if np.max(zone_plate_eh_group_relative_mismatch) > 0.05:
        raise RuntimeError(
            "zone-plate electric and magnetic far-field norms are imbalanced: "
            f"{zone_plate_eh_group_relative_mismatch.tolist()}"
        )
    if zone_plate_focus_errors[0] > 0.75 or zone_plate_focus_errors[1] > 2.0:
        raise RuntimeError(
            "zone-plate focus is outside the physical contract: "
            f"peak={zone_plate_peak_coordinates.tolist()}, "
            f"error={zone_plate_focus_errors.tolist()}"
        )
    if np.min(zone_plate_focus_contrast) <= 1.5:
        raise RuntimeError(
            "zone-plate field does not form a resolved focus: "
            f"{zone_plate_focus_contrast.tolist()}"
        )
    print(
        "zone-plate-validation:, "
        f"radial-samples={zone_plate_intensity_r.size}, "
        f"axial-samples={zone_plate_intensity_z.size}, "
        f"peak={zone_plate_peak_coordinates.tolist()}, "
        f"contrast={zone_plate_focus_contrast.tolist()}"
    )

# Plot the intensity data and save the result to disk.
if not args.validation:
    fig, ax = plt.subplots(ncols=2)

    ax[0].semilogy(np.linspace(0, size_r_um - pml_um, intensity_r.size), intensity_r, "bo-")
    ax[0].set_xlim(-2, 20)
    ax[0].set_xticks(np.arange(0, 25, 5))
    ax[0].grid(True, axis="y", which="both", ls="-")
    ax[0].set_xlabel(r"$r$ coordinate (μm)")
    ax[0].set_ylabel(r"energy density of far fields, |E|$^2$")

    ax[1].semilogy(
        np.linspace(
            focal_length_um - 0.5 * scan_length_z_um,
            focal_length_um + 0.5 * scan_length_z_um,
            intensity_z.size,
        ),
        intensity_z,
        "bo-",
    )
    ax[1].grid(True, axis="y", which="both", ls="-")
    ax[1].set_xlabel(r"$z$ coordinate (μm)")
    ax[1].set_ylabel(r"energy density of far fields, |E|$^2$")

    fig.suptitle(f"binary-phase zone plate with focal length $z$ = {focal_length_um} μm")

    if mp.am_master():
        fig.savefig("zone_plate_farfields.png", dpi=200, bbox_inches="tight")
