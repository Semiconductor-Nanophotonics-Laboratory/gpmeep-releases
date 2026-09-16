"""Demonstration of launching a planewave source at oblique incidence.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/#planewaves-in-homogeneous-media
"""

import argparse

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run the complete numerical workflow without writing plot files",
)
args = parser.parse_args()


mp.verbosity(2)

resolution_um = 50
pml_um = 2.0
size_um = 10.0
cell_size = mp.Vector3(size_um + 2 * pml_um, size_um, 0)
pml_layers = [mp.PML(thickness=pml_um, direction=mp.X)]

# Incident angle of planewave. 0 is +x with rotation in
# counter clockwise (CCW) direction around z axis.
incident_angle = np.radians(40.0)

wavelength_um = 1.0
frequency = 1 / wavelength_um

n_mat = 1.5  # refractive index of homogeneous material
default_material = mp.Medium(index=n_mat)

k_point = mp.Vector3(n_mat * frequency, 0, 0).rotate(
    mp.Vector3(0, 0, 1), incident_angle
)

if incident_angle == 0:
    direction = mp.AUTOMATIC
    eig_parity = mp.EVEN_Y + mp.ODD_Z
    symmetries = [mp.Mirror(mp.Y)]
    eig_vol = None
else:
    direction = mp.NO_DIRECTION
    eig_parity = mp.ODD_Z
    symmetries = []
    eig_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1 / resolution_um, 0))

sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency),
        center=mp.Vector3(),
        size=mp.Vector3(0, size_um, 0),
        direction=direction,
        eig_kpoint=k_point,
        eig_band=1,
        eig_parity=eig_parity,
        eig_vol=eig_vol,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution_um,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=k_point,
    default_material=default_material,
    symmetries=symmetries,
)

sim.run(until=23.56)

oblique_y_field = sim.get_array(
    center=mp.Vector3(3, 0),
    size=mp.Vector3(0, size_um),
    component=mp.Ez,
)
oblique_bloch_step = np.exp(2j * np.pi * k_point.y / resolution_um)
oblique_bloch_relative_error = np.linalg.norm(
    oblique_y_field[1:] - oblique_bloch_step * oblique_y_field[:-1]
) / np.linalg.norm(oblique_y_field[1:])

oblique_sample_x = np.linspace(-4.0, 4.0, 17)
oblique_field_samples = [
    sim.get_field_point(mp.Ez, mp.Vector3(x, 0, 0)) for x in oblique_sample_x
]

if not args.validation:
    output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(size_um, size_um, 0))

    fig, ax = plt.subplots()
    sim.plot2D(fields=mp.Ez, output_plane=output_plane, ax=ax)
    fig.savefig("planewave_source.png", bbox_inches="tight")
