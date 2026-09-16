#!/usr/bin/env python3

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import meep as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run a bounded but complete source/unit-cell/supercell N2F comparison",
)
parser.add_argument("--resolution", type=int, default=None)
parser.add_argument("--periods", type=int, default=None)
parser.add_argument("--farfield-points", type=int, default=None)
parser.add_argument("--frequencies", type=int, default=None)
args = parser.parse_args()

resolution = (
    args.resolution if args.resolution is not None else (10 if args.validation else 25)
)  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML
gp = 10.0  # grating period
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle

nperiods = args.periods if args.periods is not None else (2 if args.validation else 10)

ff_distance = 1e8  # far-field distance from near-field monitor
ff_angle = 20  # far-field cone angle
ff_npts = (
    args.farfield_points
    if args.farfield_points is not None
    else (64 if args.validation else 500)
)
if min(resolution, nperiods, ff_npts) <= 0:
    raise RuntimeError("resolution, periods, and far-field points must be positive")

ff_length = ff_distance * math.tan(math.radians(ff_angle))
ff_res = ff_npts / ff_length

sx = dpml + dsub + gh + dpad + dpml
cell_size = mp.Vector3(sx)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=src_pt)
]

k_point = mp.Vector3()

glass = mp.Medium(index=1.5)

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    k_point=k_point,
    default_material=glass,
    sources=sources,
)

nfreq = (
    args.frequencies if args.frequencies is not None else (9 if args.validation else 21)
)
if nfreq <= 1:
    raise RuntimeError("at least two near-to-far frequencies are required")
n2f_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
n2f_obj = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt))

if args.validation:
    sim.run(until_after_sources=60)
else:
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    )

ff_source = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### unit cell with periodic boundaries

sy = gp
cell_size = mp.Vector3(sx, sy)

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(y=sy),
    )
]

geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
        center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
    ),
    mp.Block(
        material=glass,
        size=mp.Vector3(gh, gdc * gp, mp.inf),
        center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen,
    df,
    nfreq,
    mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)),
    nperiods=nperiods,
)

if args.validation:
    sim.run(until_after_sources=60)
else:
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    )

ff_unitcell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### finite periodic grating with flat surface termination extending into PML

num_cells = 2 * nperiods + 1
sy = dpml + num_cells * gp + dpml
cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(thickness=dpml)]

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(y=sy),
    )
]

geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
        center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
    )
]

for j in range(num_cells):
    geometry.append(
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(
                -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + dpml + (j + 0.5) * gp
            ),
        )
    )

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy - 2 * dpml))
)

if args.validation:
    sim.run(until_after_sources=60)
else:
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    )

ff_supercell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

norm_err = LA.norm(ff_unitcell["Ez"] - ff_supercell["Ez"]) / nperiods
print(f"error:, {nperiods}, {norm_err}")

freqs = mp.get_near2far_freqs(n2f_obj)
wvl = np.divide(1, freqs)
ff_lengths = np.linspace(0, ff_length, ff_npts)
angles = [math.degrees(math.atan(f)) for f in ff_lengths / ff_distance]

wvl_slice = 0.5
idx_slice = int(np.argmin(np.abs(np.asarray(freqs) - 1 / wvl_slice)))

rel_enh = np.absolute(ff_unitcell["Ez"]) ** 2 / np.absolute(ff_source["Ez"]) ** 2
farfield_source_ez = np.asarray(ff_source["Ez"])
farfield_unitcell_ez = np.asarray(ff_unitcell["Ez"])
farfield_supercell_ez = np.asarray(ff_supercell["Ez"])
farfield_relative_enhancement = np.asarray(rel_enh)
farfield_frequencies = np.asarray(freqs)
farfield_angles = np.asarray(angles)
farfield_l2 = np.asarray(
    [
        LA.norm(farfield_source_ez),
        LA.norm(farfield_unitcell_ez),
        LA.norm(farfield_supercell_ez),
    ]
)
farfield_error_metrics = np.asarray(
    [
        norm_err,
        LA.norm(farfield_unitcell_ez - farfield_supercell_ez)
        / max(LA.norm(farfield_supercell_ez), 1e-30),
    ]
)
farfield_relative_closure = np.asarray([farfield_error_metrics[1]])
for name, value in (
    ("source", farfield_source_ez),
    ("unit-cell", farfield_unitcell_ez),
    ("supercell", farfield_supercell_ez),
    ("relative enhancement", farfield_relative_enhancement),
    ("error metrics", farfield_error_metrics),
    ("relative closure", farfield_relative_closure),
):
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"binary-grating N2F {name} result is non-finite")
if np.any(farfield_l2 <= 1e-10):
    raise RuntimeError(f"binary-grating N2F path produced no field: {farfield_l2}")
if args.validation and farfield_relative_closure[0] >= 1e-2:
    raise RuntimeError(
        "binary-grating N2F unit-cell/supercell closure exceeded 1%: "
        f"{farfield_relative_closure[0]:.6g}"
    )

plt.figure(dpi=150)

plt.subplot(1, 2, 1)
plt.pcolormesh(wvl, angles, rel_enh, cmap="Blues", shading="auto")
plt.axis([wvl_min, wvl_max, 0, ff_angle])
plt.xlabel("wavelength (μm)")
plt.ylabel("angle (degrees)")
plt.grid(linewidth=0.5, linestyle="--")
plt.xticks([t for t in np.arange(wvl_min, wvl_max + 0.1, 0.1)])
plt.yticks([t for t in range(0, ff_angle + 1, 10)])
plt.title("far-field spectra")

plt.subplot(1, 2, 2)
plt.plot(angles, rel_enh[:, idx_slice], "bo-")
plt.xlim(0, ff_angle)
plt.ylim(0)
plt.xticks([t for t in range(0, ff_angle + 1, 10)])
plt.xlabel("angle (degrees)")
plt.ylabel("relative enhancement")
plt.grid(axis="x", linewidth=0.5, linestyle="--")
plt.title(f"f.-f. spectra @  λ = {wvl_slice:.1} μm")

plt.tight_layout(pad=0.5)
plt.show()
