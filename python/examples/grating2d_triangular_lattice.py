#!/usr/bin/env python3
"""Diffraction orders of a triangular-lattice binary grating."""

# Computes the diffraction orders of a 2D binary grating with
# triangular lattice using a rectangular supercell and verifies
# that only the diffraction orders of the actual unit cell
# produce non-zero power (up to discretization error)
import argparse
import math

import numpy as np

import meep as mp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run a bounded 3D CUDA/CPU oracle and retain every propagating order",
)
parser.add_argument("--resolution", type=int, default=None)
args = parser.parse_args()

resolution = args.resolution if args.resolution is not None else (40 if args.validation else 100)
if resolution <= 0:
    raise RuntimeError("resolution must be positive")

ng = 1.5
glass = mp.Medium(index=ng)

wvl = 0.5  # wavelength
fcen = 1 / wvl

# rectangular supercell
sx = 1.0
sy = np.sqrt(3)

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dair = 2.0  # air padding
hcyl = 0.5  # cylinder height
rcyl = 0.1  # cylinder radius

sz = dpml + dsub + hcyl + dair + dpml

cell_size = mp.Vector3(sx, sy, sz)

boundary_layers = [mp.PML(thickness=dpml, direction=mp.Z)]

# periodic boundary conditions
k_point = mp.Vector3()

src_pt = mp.Vector3(0, 0, -0.5 * sz + dpml)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
        size=mp.Vector3(sx, sy, 0),
        center=src_pt,
        component=mp.Ex,
    )
]

substrate = [
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, dpml + dsub),
        center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * (dpml + dsub)),
        material=glass,
    )
]

cyl_grating = [
    mp.Cylinder(
        center=mp.Vector3(0, 0, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
]

geometry = substrate + cyl_grating

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    sources=sources,
    geometry=geometry,
    boundary_layers=boundary_layers,
    k_point=k_point,
)

tran_pt = mp.Vector3(0, 0, 0.5 * sz - dpml)
tran_flux = sim.add_mode_monitor(
    fcen, 0, 1, mp.ModeRegion(center=tran_pt, size=mp.Vector3(sx, sy, 0))
)

decay_tolerance = 1e-5 if args.validation else 1e-6
sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, src_pt, decay_tolerance
    )
)

# diffraction order of unit cell (triangular lattice)
mx = 0
my = 1

# check: for diffraction orders of supercell for which
#        nx = mx and ny = -mx + 2*my and thus
#        only even orders should produce nonzero power
nx = mx
triangular_orders = []
triangular_alpha = []
triangular_transmission = []
triangular_vgrp = []
triangular_kdom = []
for ny in range(4):
    kz2 = fcen**2 - (nx / sx) ** 2 - (ny / sy) ** 2
    if kz2 > 0:
        res = sim.get_eigenmode_coefficients(
            tran_flux, mp.DiffractedPlanewave((nx, ny, 0), mp.Vector3(0, 1, 0), 1, 0)
        )
        t_coeffs = res.alpha
        tran = abs(t_coeffs[0, 0, 0]) ** 2

        wavevector = res.kdom[0]
        triangular_orders.append([nx, ny])
        triangular_alpha.append(complex(t_coeffs[0, 0, 0]))
        triangular_transmission.append(float(tran))
        triangular_vgrp.append(float(res.vgrp[0]))
        triangular_kdom.append([wavevector.x, wavevector.y, wavevector.z])

        print(f"order:, {nx}, {ny}, {tran:.5f}")

triangular_orders = np.asarray(triangular_orders, dtype=float)
triangular_alpha = np.asarray(triangular_alpha)
triangular_transmission = np.asarray(triangular_transmission)
triangular_vgrp = np.asarray(triangular_vgrp)
triangular_kdom = np.asarray(triangular_kdom)
triangular_expected_kdom = np.column_stack(
    [
        triangular_orders[:, 0] / sx,
        triangular_orders[:, 1] / sy,
        np.sqrt(
            np.maximum(
                0.0,
                fcen**2
                - np.square(triangular_orders[:, 0] / sx)
                - np.square(triangular_orders[:, 1] / sy),
            )
        ),
    ]
)
triangular_expected_vgrp = triangular_expected_kdom[:, 2] / fcen
triangular_dispersion_error = np.asarray(
    [
        np.max(np.abs(triangular_kdom - triangular_expected_kdom)),
        np.max(np.abs(triangular_vgrp - triangular_expected_vgrp)),
    ]
)
triangular_even_odd_power = np.asarray(
    [
        np.sum(triangular_transmission[triangular_orders[:, 1] % 2 == 0]),
        np.sum(triangular_transmission[triangular_orders[:, 1] % 2 == 1]),
    ]
)
triangular_selection_ratio = np.asarray(
    [triangular_even_odd_power[1] / max(triangular_even_odd_power[0], 1e-30)]
)

if args.validation:
    if not np.array_equal(
        triangular_orders, np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]])
    ):
        raise RuntimeError("triangular-grating propagating-order topology changed")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            triangular_alpha.real,
            triangular_alpha.imag,
            triangular_transmission,
            triangular_vgrp,
            triangular_kdom,
        )
    ):
        raise RuntimeError("triangular-grating result is nonfinite")
    if triangular_even_odd_power[0] <= 1e-5:
        raise RuntimeError("triangular-grating allowed orders have no signal")
    if triangular_selection_ratio[0] > 0.03:
        raise RuntimeError(
            "triangular-lattice forbidden orders carry excessive power: "
            f"{triangular_selection_ratio[0]}"
        )
    if np.max(triangular_dispersion_error) > 2e-3:
        raise RuntimeError(
            "triangular-grating orders violate vacuum dispersion: "
            f"{triangular_dispersion_error.tolist()}"
        )
