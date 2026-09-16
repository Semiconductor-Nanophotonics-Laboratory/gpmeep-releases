# Calculating 2d ring-resonator modes, from the Meep tutorial.
import meep as mp

n = 3.4  # index of waveguide
w = 1  # width of waveguide
r = 1  # inner radius of ring

pad = 4  # padding between waveguide and edge of PML
dpml = 2  # thickness of PML

sxy = 2 * (r + w + pad + dpml)  # cell size
cell = mp.Vector3(sxy, sxy)

# Create a ring waveguide by two overlapping cylinders - later objects
# take precedence over earlier objects, so we put the outer cylinder first.
# and the inner (air) cylinder second.
geometry = [
    mp.Cylinder(radius=r + w, height=mp.inf, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r, height=mp.inf, material=mp.air),
]

pml_layers = [mp.PML(dpml)]
resolution = 20

# If we don't want to excite a specific mode symmetry, we can just
# put a single point source at some arbitrary place, pointing in some
# arbitrary direction. We will only look for Ez-polarized modes.

fcen_w = 0.118  # pulse center frequency
fcen_2w = 2 * fcen_w
df = 0.010  # pulse width (in frequency)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen_w, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    )
]

# exploit the mirror symmetry in structure+source:
symmetries = [mp.Mirror(mp.Y)]

sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
)

h1 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen_w, df)
sim.run(mp.after_sources(h1), until_after_sources=300)

fields_w = sim.fields
field_energy_w = sim.field_energy_in_box(box=sim.fields.total_volume())
sim.reset_meep()

sources_2w = [
    mp.Source(
        src=mp.GaussianSource(fcen_2w, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    )
]
sim.change_sources(sources_2w)
h2 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen_2w, df)
sim.run(mp.after_sources(h2), until_after_sources=300)
field_energy_2w = sim.field_energy_in_box(box=sim.fields.total_volume())

if not h1.modes or not h2.modes:
    raise RuntimeError("Harminv did not find both w and 2w ring modes")

source_center_frequencies = [sources[0].src.frequency, sources_2w[0].src.frequency]
source_frequency_ratio_error = abs(
    source_center_frequencies[1] / (2 * source_center_frequencies[0]) - 1
)
harminv_nearest_frequencies = [
    min(h1.modes, key=lambda mode: abs(mode.freq - fcen_w)).freq,
    min(h2.modes, key=lambda mode: abs(mode.freq - fcen_2w)).freq,
]
harminv_center_relative_errors = [
    abs(harminv_nearest_frequencies[0] - source_center_frequencies[0])
    / source_center_frequencies[0],
    abs(harminv_nearest_frequencies[1] - source_center_frequencies[1])
    / source_center_frequencies[1],
]


def overlap_integral(r, ez1, ez2):
    return ez1.conjugate() * ez2


res = sim.integrate2_field_function(fields_w, [mp.Ez], [mp.Ez], overlap_integral)
print(f"overlap integral of mode at w and 2w: {abs(res)}")
