"""
Verifies that the relative error in the fields of a resonant mode
of a 2d ring resonator is monotonically decreasing with decreasing
tolerance of the CW solver. Also visualizes the fields of the resonant
mode in the time and frequency domains.
"""

import argparse

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run the complete numerical workflow without writing plot files",
)
args = parser.parse_args()

resolution = 20  # pixels/μm
n = 3.4  # refractive index of ring
w = 1  # width of ring
r = 1  # inner radius of ring
pad = 4  # padding between outer ring and PML
dpml = 2  # PML thickness

sxy = 2 * (r + w + pad + dpml)
cell_size = mp.Vector3(sxy, sxy)

pml_layers = [mp.PML(dpml)]

nonpml_vol = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml),
)

geometry = [
    mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r),
]

fcen = 0.118  # frequency of resonant mode

src = [
    mp.Source(
        mp.ContinuousSource(fcen),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    ),
    mp.Source(
        mp.ContinuousSource(fcen),
        component=mp.Ez,
        center=mp.Vector3(-(r + 0.1)),
        amplitude=-1,
    ),
]

symmetries = [
    mp.Mirror(mp.X, phase=-1),
    mp.Mirror(mp.Y, phase=+1),
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    geometry=geometry,
    sources=src,
    force_complex_fields=True,
    symmetries=symmetries,
    boundary_layers=pml_layers,
)

# CW solver convergence properties
maxiters = 10000
L = 10
if mp.is_single_precision():
    # Requested residuals below FP32 resolution make the final reference field
    # solver-noise dominated instead of demonstrating convergence.
    num_tols = 4
    tols = np.logspace(-2, -5, num_tols)
else:
    num_tols = 5
    tols = np.logspace(-8, -8.0 - num_tols + 1, num_tols)

ez_dat = np.zeros(
    (
        int(nonpml_vol.size.x * resolution) + 2,
        int(nonpml_vol.size.y * resolution) + 2,
        num_tols,
    ),
    dtype=np.complex128,
)
cw_converged_int = np.zeros(num_tols, dtype=np.int64)

for i in range(num_tols):
    sim.init_sim()
    cw_converged_int[i] = int(sim.solve_cw(tols[i], maxiters, L))
    ez_dat[:, :, i] = sim.get_array(vol=nonpml_vol, component=mp.Ez)

if not np.all(cw_converged_int == 1):
    failed_tolerances = tols[cw_converged_int != 1]
    raise RuntimeError(
        "solve_cw failed to reach its true-residual target for tolerances "
        f"{failed_tolerances.tolist()}"
    )

err_dat = np.zeros(num_tols - 1)
for i in range(num_tols - 1):
    err_dat[i] = np.linalg.norm(ez_dat[:, :, i] - ez_dat[:, :, -1]) / np.linalg.norm(
        ez_dat[:, :, -1]
    )
    print(f"err:, {tols[i]}, {err_dat[i]}")

cw_ez_reference = ez_dat[:, :, -1]
cw_ez_reference_flat = cw_ez_reference.reshape(-1)
cw_ez_anchor_index = int(np.argmax(np.abs(cw_ez_reference_flat)))
cw_ez_anchor = cw_ez_reference_flat[cw_ez_anchor_index]
cw_ez_norm = float(np.linalg.norm(cw_ez_reference_flat))
if cw_ez_norm == 0 or abs(cw_ez_anchor) == 0:
    raise RuntimeError("solve_cw produced a zero reference field")
cw_ez_phase_aligned = (
    cw_ez_reference * np.exp(-1j * np.angle(cw_ez_anchor)) / cw_ez_norm
)
cw_ez_norm_max = np.asarray(
    [cw_ez_norm, np.max(np.abs(cw_ez_reference_flat))]
)
cw_error_monotonic = bool(np.all(np.diff(err_dat) < 0))
cw_error_monotonic_int = int(cw_error_monotonic)
cw_final_relative_error = float(err_dat[-1])

if not args.validation:
    plt.figure(dpi=150)
    plt.loglog(tols[: num_tols - 1], err_dat, "bo-")
    plt.xlabel("frequency-domain solver tolerance")
    plt.ylabel("relative error in fields of resonant mode")
    plt.title("2d ring resonator")
    plt.savefig("ring_err.png", dpi=150, bbox_inches="tight")

    eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
    ez_data = np.real(ez_dat[:, :, num_tols - 1])

    plt.figure()
    plt.imshow(
        eps_data.transpose(),
        interpolation="spline36",
        cmap="binary",
    )
    plt.imshow(
        ez_data.transpose(),
        interpolation="spline36",
        cmap="RdBu",
        alpha=0.9,
    )
    plt.title("time-domain fields ($E_z$)")
    plt.axis("off")
    plt.savefig("ring_ez.png", dpi=150, bbox_inches="tight")

if cw_error_monotonic:
    print(
        "PASSED solve_cw test: error in the fields is "
        "decreasing with tighter tolerance."
    )
else:
    raise RuntimeError(
        "solve_cw field error is not decreasing with tighter tolerance: "
        f"tolerances={tols.tolist()}, errors={err_dat.tolist()}"
    )

sim.reset_meep()

df = 0.08  # frequency width of pulsed source
src = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    ),
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-(r + 0.1)),
        amplitude=-1,
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=mp.Vector3(sxy, sxy),
    geometry=geometry,
    sources=src,
    symmetries=symmetries,
    boundary_layers=pml_layers,
)

dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=nonpml_vol)

sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        50,
        mp.Ez,
        mp.Vector3(r + 0.1523),
        1e-8,
    )
)

dft_ez_data = np.real(sim.get_dft_array(dft_obj, mp.Ez, 0))

if not args.validation:
    eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
    plt.figure()
    plt.imshow(
        eps_data.transpose(),
        interpolation="spline36",
        cmap="binary",
    )
    plt.imshow(
        dft_ez_data.transpose(),
        interpolation="spline36",
        cmap="RdBu",
        alpha=0.9,
    )
    plt.title("DFT fields ($E_z$)")
    plt.axis("off")
    plt.savefig("ring_ez_dft.png", dpi=150, bbox_inches="tight")
