#!/usr/bin/env python3
"""Transmitted orders of a binary grating by two mode-decomposition APIs."""

import argparse
import cmath
import math

import numpy as np

import meep as mp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="run both incidence branches at bounded resolution and export all orders",
)
parser.add_argument("--resolution", type=int, default=None)
args = parser.parse_args()

resolution = args.resolution if args.resolution is not None else (25 if args.validation else 50)
if resolution <= 0:
    raise RuntimeError("resolution must be positive")


def binary_grating_diffraction(gp, gh, gdc, theta):
    dpml = 1.0
    dsub = 3.0
    dpad = 3.0
    sx = dpml + dsub + gh + dpad + dpml
    sy = gp
    cell_size = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
    wvl = 0.5
    fcen = 1 / wvl
    df = 0.05 * fcen
    ng = 1.5
    glass = mp.Medium(index=ng)
    theta_in = math.radians(theta)
    eig_parity = mp.EVEN_Z
    k = mp.Vector3(fcen * ng).rotate(mp.Vector3(z=1), theta_in)
    symmetries = []
    if theta_in == 0:
        k = mp.Vector3()
        eig_parity += mp.ODD_Y
        symmetries = [mp.Mirror(direction=mp.Y, phase=-1)]

    def pw_amp(wavevector, x0):
        def _pw_amp(point):
            return cmath.exp(1j * 2 * math.pi * wavevector.dot(point + x0))

        return _pw_amp

    src_pt = mp.Vector3(-0.5 * sx + dpml, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
            amp_func=pw_amp(k, src_pt),
        )
    ]
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k,
        default_material=glass,
        sources=sources,
        symmetries=symmetries,
    )
    tran_pt = mp.Vector3(0.5 * sx - dpml, 0, 0)
    tran_mon = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0))
    )
    sim.run(until_after_sources=50)
    input_flux = float(mp.get_fluxes(tran_mon)[0])
    sim.reset_meep()

    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub), 0, 0),
        ),
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh, 0, 0),
        ),
    ]
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        k_point=k,
        sources=sources,
        symmetries=symmetries,
    )
    tran_mon = sim.add_mode_monitor(
        fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0))
    )
    sim.run(until_after_sources=100)

    minimum_order = int(np.ceil((-fcen - k.y) * gp))
    maximum_order = int(np.floor((fcen - k.y) * gp))
    if theta_in == 0:
        orders = list(range(0, maximum_order + 1))
    else:
        all_orders = list(range(minimum_order, maximum_order + 1))
        longitudinal_squared = [fcen**2 - (k.y + order / gp) ** 2 for order in all_orders]
        order_indices = np.flip(np.argsort(longitudinal_squared))
        orders = [all_orders[int(index)] for index in order_indices]

    eig_transmission = []
    dp_transmission = []
    eig_alpha = []
    dp_alpha = []
    eig_vgrp = []
    dp_vgrp = []
    eig_kdom = []
    dp_kdom = []
    for band, order in enumerate(orders, start=1):
        eig_result = sim.get_eigenmode_coefficients(
            tran_mon, [band], eig_parity=eig_parity
        )
        dp_result = sim.get_eigenmode_coefficients(
            tran_mon,
            mp.DiffractedPlanewave((0, order, 0), mp.Vector3(0, 1, 0), 0, 1),
        )
        if eig_result is None or dp_result is None:
            raise RuntimeError("non-evanescent diffraction order has no eigenmode result")
        eig_amplitude = complex(eig_result.alpha[0, 0, 0])
        dp_amplitude = complex(dp_result.alpha[0, 0, 0])
        tran_eig = abs(eig_amplitude) ** 2 / input_flux
        tran_dp = abs(dp_amplitude) ** 2 / input_flux
        if theta_in == 0:
            tran_eig *= 0.5
            if order == 0:
                tran_dp *= 0.5
        eig_wavevector = eig_result.kdom[0]
        dp_wavevector = dp_result.kdom[0]
        eig_transmission.append(float(tran_eig))
        dp_transmission.append(float(tran_dp))
        eig_alpha.append(eig_amplitude)
        dp_alpha.append(dp_amplitude)
        eig_vgrp.append(float(eig_result.vgrp[0]))
        dp_vgrp.append(float(dp_result.vgrp[0]))
        eig_kdom.append([eig_wavevector.x, eig_wavevector.y, eig_wavevector.z])
        dp_kdom.append([dp_wavevector.x, dp_wavevector.y, dp_wavevector.z])
        error = abs(tran_eig - tran_dp) / max(abs(tran_eig), 1e-30)
        print(f"tran:, {band:2d}, {tran_eig:.8f}, {order:2d}, {tran_dp:.8f}, {error:.8f}")

    output_flux = float(mp.get_fluxes(tran_mon)[0])
    t_flux = output_flux / input_flux
    if theta_in == 0:
        t_flux *= 0.5
    result = {
        "parameters": np.asarray([gp, gh, gdc, theta, k.x, k.y, k.z]),
        "orders": np.asarray(orders, dtype=float),
        "bands": np.arange(1, len(orders) + 1, dtype=float),
        "eig_transmission": np.asarray(eig_transmission),
        "dp_transmission": np.asarray(dp_transmission),
        "eig_alpha": np.asarray(eig_alpha),
        "dp_alpha": np.asarray(dp_alpha),
        "eig_vgrp": np.asarray(eig_vgrp),
        "dp_vgrp": np.asarray(dp_vgrp),
        "eig_kdom": np.asarray(eig_kdom),
        "dp_kdom": np.asarray(dp_kdom),
        "fluxes": np.asarray(
            [input_flux, output_flux, np.sum(eig_transmission), np.sum(dp_transmission), t_flux]
        ),
    }
    closure_scale = max(abs(t_flux), 1e-30)
    result["closure"] = np.asarray(
        [
            abs(np.sum(eig_transmission) - t_flux) / closure_scale,
            abs(np.sum(dp_transmission) - t_flux) / closure_scale,
            np.max(np.abs(result["eig_transmission"] - result["dp_transmission"])),
            np.max(np.abs(result["eig_vgrp"] - result["dp_vgrp"])),
        ]
    )
    expected_dp_ky = k.y + result["orders"] / gp
    # The normal-incidence branch uses an odd-Y mirror and therefore MPB
    # reports the negative member of each +/-ky degenerate pair.  The public
    # DiffractedPlanewave order retains the requested positive representative.
    expected_eig_ky = -expected_dp_ky if theta_in == 0 else expected_dp_ky
    expected_kx = np.sqrt(np.maximum(0.0, fcen**2 - expected_dp_ky**2))
    result["expected_eig_kdom"] = np.column_stack(
        [expected_kx, expected_eig_ky, np.zeros_like(expected_kx)]
    )
    result["expected_dp_kdom"] = np.column_stack(
        [expected_kx, expected_dp_ky, np.zeros_like(expected_kx)]
    )
    result["expected_vgrp"] = expected_kx / fcen
    result["dispersion_closure"] = np.asarray(
        [
            np.max(np.abs(result["eig_kdom"] - result["expected_eig_kdom"])),
            np.max(np.abs(result["dp_kdom"] - result["expected_dp_kdom"])),
            np.max(np.abs(result["eig_vgrp"] - result["expected_vgrp"])),
            np.max(np.abs(result["dp_vgrp"] - result["expected_vgrp"])),
        ]
    )
    print(
        f"flux:, {result['fluxes'][2]:.8f}, {result['fluxes'][3]:.8f}, "
        f"{t_flux:.8f}, {result['closure'][1]:.8f}"
    )
    return result


if __name__ == "__main__":
    diffraction_results = [
        binary_grating_diffraction(2.6, 0.4, 0.3, 0),
        binary_grating_diffraction(3.7, 0.6, 0.4, 13.5),
    ]
    diffraction_order_counts = np.asarray(
        [len(item["orders"]) for item in diffraction_results], dtype=float
    )
    diffraction_order_offsets = np.asarray([0, 6, 20], dtype=float)
    diffraction_parameters = np.stack([item["parameters"] for item in diffraction_results])
    diffraction_orders = np.concatenate([item["orders"] for item in diffraction_results])
    diffraction_bands = np.concatenate([item["bands"] for item in diffraction_results])
    diffraction_eig_transmission = np.concatenate(
        [item["eig_transmission"] for item in diffraction_results]
    )
    diffraction_dp_transmission = np.concatenate(
        [item["dp_transmission"] for item in diffraction_results]
    )
    diffraction_eig_alpha = np.concatenate([item["eig_alpha"] for item in diffraction_results])
    diffraction_eig_alpha_magnitude = np.abs(diffraction_eig_alpha)
    diffraction_dp_alpha = np.concatenate([item["dp_alpha"] for item in diffraction_results])
    diffraction_eig_vgrp = np.concatenate([item["eig_vgrp"] for item in diffraction_results])
    diffraction_dp_vgrp = np.concatenate([item["dp_vgrp"] for item in diffraction_results])
    diffraction_eig_kdom = np.concatenate([item["eig_kdom"] for item in diffraction_results])
    diffraction_dp_kdom = np.concatenate([item["dp_kdom"] for item in diffraction_results])
    diffraction_fluxes = np.stack([item["fluxes"] for item in diffraction_results])
    diffraction_closure = np.stack([item["closure"] for item in diffraction_results])
    diffraction_expected_eig_kdom = np.concatenate(
        [item["expected_eig_kdom"] for item in diffraction_results]
    )
    diffraction_expected_dp_kdom = np.concatenate(
        [item["expected_dp_kdom"] for item in diffraction_results]
    )
    diffraction_expected_vgrp = np.concatenate(
        [item["expected_vgrp"] for item in diffraction_results]
    )
    diffraction_dispersion_closure = np.stack(
        [item["dispersion_closure"] for item in diffraction_results]
    )

    if args.validation:
        if not np.array_equal(diffraction_order_counts, [6.0, 14.0]):
            raise RuntimeError("diffraction-order topology changed")
        if set(diffraction_orders[:6].astype(int)) != set(range(6)) or set(
            diffraction_orders[6:].astype(int)
        ) != set(range(-9, 5)):
            raise RuntimeError("diffraction-order set changed")
        if np.any(diffraction_fluxes[:, 0] <= 0) or np.any(diffraction_eig_transmission < 0):
            raise RuntimeError("diffraction workflow produced non-physical power")
        if np.max(diffraction_closure[:, :2]) > 0.03:
            raise RuntimeError("diffraction-order powers do not close to monitor flux")
        if np.max(diffraction_closure[:, 2]) > 2e-3:
            raise RuntimeError("band and DiffractedPlanewave order powers disagree")
        if np.max(diffraction_closure[:, 3]) > 2e-3:
            raise RuntimeError("band and DiffractedPlanewave group velocities disagree")
        if np.any(diffraction_eig_kdom[:, 0] <= 0) or np.any(
            diffraction_dp_kdom[:, 0] <= 0
        ):
            raise RuntimeError("diffracted planewaves are not outgoing")
        if np.max(diffraction_dispersion_closure) > 2e-3:
            raise RuntimeError(
                "diffracted planewaves violate the vacuum dispersion relation: "
                f"{diffraction_dispersion_closure.tolist()}"
            )
