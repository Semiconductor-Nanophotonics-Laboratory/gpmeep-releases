# From the Meep tutorial: plotting permittivity and fields of a bent waveguide
import argparse
import tempfile

import meep as mp

cell = mp.Vector3(16, 16, 0)
geometry = [
    mp.Block(
        mp.Vector3(12, 1, mp.inf),
        center=mp.Vector3(-2.5, -3.5),
        material=mp.Medium(epsilon=12),
    ),
    mp.Block(
        mp.Vector3(1, 12, mp.inf),
        center=mp.Vector3(3.5, 2),
        material=mp.Medium(epsilon=12),
    ),
]
pml_layers = [mp.PML(1.0)]
resolution = 10

sources = [
    mp.Source(
        mp.ContinuousSource(wavelength=2 * (11**0.5), width=20),
        component=mp.Ez,
        center=mp.Vector3(-7, -3.5),
        size=mp.Vector3(0, 1),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="exercise HDF5 output in a temporary directory that is cleaned on exit",
)
args = parser.parse_args()

run_steps = (
    mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
)
if args.validation:
    with tempfile.TemporaryDirectory(prefix="gpmeep-bent-waveguide-") as output_dir:
        sim.use_output_directory(output_dir)
        sim.run(*run_steps, until=200)
else:
    sim.run(*run_steps, until=200)

bend_field_sample_points = [
    mp.Vector3(-6, -3.5),
    mp.Vector3(-4, -3.5),
    mp.Vector3(-2, -3.5),
    mp.Vector3(0, -3.5),
    mp.Vector3(2, -3.5),
    mp.Vector3(3.5, -3),
    mp.Vector3(3.5, -1),
    mp.Vector3(3.5, 1),
    mp.Vector3(3.5, 3),
    mp.Vector3(3.5, 5),
    mp.Vector3(0, 0),
]
bend_field_samples = [
    sim.get_field_point(mp.Ez, point) for point in bend_field_sample_points
]
