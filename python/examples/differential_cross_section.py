import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import meep as mp
try:
    from meep.mie_theory import (
        mie_differential_cross_section,
        mie_scattering_cross_section,
    )
except ModuleNotFoundError as exc:
    if exc.name != "meep.mie_theory":
        raise
    from mie_theory import (
        mie_differential_cross_section,
        mie_scattering_cross_section,
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run a bounded deterministic differential-scattering oracle",
)
args = parser.parse_args()
if args.validation:
    mp.verbosity(0)

r = 0.5 if args.validation else 1.0  # radius of sphere

frq_cen = 1.0

resolution = 16 if args.validation else 20  # pixels/um

dpml = 0.4 if args.validation else 0.5
dair = 0.8 if args.validation else 1.5

pml_layers = [mp.PML(thickness=dpml)]

s = 2 * (dpml + dair + r)
cell_size = mp.Vector3(s, s, s)

# circularly-polarized source with propagation axis along x
# is_integrated=True necessary for any planewave source extending into PML
sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=0.2 * frq_cen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    ),
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=0.2 * frq_cen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ey,
        amplitude=1j,
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
)

box_flux = sim.add_flux(
    frq_cen,
    0,
    1,
    mp.FluxRegion(center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r)),
)

nearfield_box = sim.add_near2far(
    frq_cen,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(x=+2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=-2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=+2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=-2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=+2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=-1
    ),
)

sim.run(until_after_sources=10)

input_flux = mp.get_fluxes(box_flux)[0]
nearfield_box_data = sim.get_near2far_data(nearfield_box)

sim.reset_meep()

n_sphere = 2.0
geometry = [
    mp.Sphere(material=mp.Medium(index=n_sphere), center=mp.Vector3(), radius=r)
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    geometry=geometry,
)

nearfield_box = sim.add_near2far(
    frq_cen,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(x=+2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=-2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=+2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=-2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=+2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=-1
    ),
)

sim.load_minus_near2far_data(nearfield_box, nearfield_box_data)

sim.run(until_after_sources=60 if args.validation else 100)

npts = 32 if args.validation else 100
angles = np.pi / npts * np.arange(npts)

ff_r = (2000 if args.validation else 10000) * r

far_points = [
    ff_r * mp.Vector3(np.cos(angle), 0, np.sin(angle)) for angle in angles
]
far_fields = sim.get_farfields_points(nearfield_box, far_points)[:, 0, :]
E = np.conj(far_fields[:, :3])
H = far_fields[:, 3:]

Px = np.real(np.multiply(E[:, 1], H[:, 2]) - np.multiply(E[:, 2], H[:, 1]))
Py = np.real(np.multiply(E[:, 2], H[:, 0]) - np.multiply(E[:, 0], H[:, 2]))
Pz = np.real(np.multiply(E[:, 0], H[:, 1]) - np.multiply(E[:, 1], H[:, 0]))
Pr = np.sqrt(np.square(Px) + np.square(Py) + np.square(Pz))

intensity = input_flux / (4 * r) ** 2
diff_cross_section = ff_r**2 * Pr / intensity
scatt_cross_section_meep = (
    2 * np.pi * np.sum(np.multiply(diff_cross_section, np.sin(angles))) * np.pi / npts
)
scatt_cross_section_theory = mie_scattering_cross_section(
    n_sphere, 2 * np.pi * r * frq_cen, r
)
print(
    "scatt:, {:.16f} (meep), {:.16f} (theory)".format(
        scatt_cross_section_meep, scatt_cross_section_theory
    )
)

if args.validation:
    relative_error = float(
        abs(scatt_cross_section_meep - scatt_cross_section_theory)
        / abs(scatt_cross_section_theory)
    )
    normalized_pattern = diff_cross_section / np.max(diff_cross_section)
    theory_pattern = mie_differential_cross_section(
        n_sphere, 2 * np.pi * r * frq_cen, r, angles
    )
    normalized_theory_pattern = theory_pattern / np.max(theory_pattern)
    pattern_relative_l2 = float(
        np.linalg.norm(normalized_pattern - normalized_theory_pattern)
        / np.linalg.norm(normalized_theory_pattern)
    )
    pattern_scale_relative_max = float(
        np.max(np.abs(normalized_pattern - normalized_theory_pattern))
    )
    if (
        not np.all(np.isfinite(normalized_pattern))
        or not np.isfinite(scatt_cross_section_meep)
        or scatt_cross_section_meep <= 0
        or scatt_cross_section_theory <= 0
        or relative_error > 0.15
        or pattern_relative_l2 > 0.10
        or pattern_scale_relative_max > 0.08
        or np.max(normalized_pattern) != 1.0
    ):
        raise RuntimeError("bounded differential Mie validation failed")
    metrics = {
        "schema_version": 1,
        "resolution": resolution,
        "angle_count": npts,
        "angles": angles.tolist(),
        "normalized_pattern": normalized_pattern.tolist(),
        "normalized_theory_pattern": normalized_theory_pattern.tolist(),
        "meep_cross_section": float(scatt_cross_section_meep),
        "theory_cross_section": float(scatt_cross_section_theory),
        "relative_error": relative_error,
        "pattern_relative_l2": pattern_relative_l2,
        "pattern_scale_relative_max": pattern_scale_relative_max,
    }
    if mp.am_master():
        print(
            "gpmeep-differential-scattering-metrics:"
            + json.dumps(metrics, sort_keys=True, allow_nan=False)
        )
