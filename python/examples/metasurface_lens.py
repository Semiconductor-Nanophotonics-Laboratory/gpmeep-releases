import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import meep as mp

resolution = 50  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dpad = 2.0  # padding between grating and PML

lcen = 0.5  # center wavelength
fcen = 1 / lcen  # center frequency
df = 0.2 * fcen  # frequency width

focal_length = 200  # focal length of metalens
spot_length = 100  # far field line length
ff_res = 10  # far field resolution (points/μm)

unit_reference_after_sources = 50
unit_grating_after_sources = 200
lens_after_sources = 500

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]


def grating(gp, gh, gdc_list):
    sx = dpml + dsub + gh + dpad + dpml
    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
        )
    ]

    num_cells = len(gdc_list)
    if num_cells == 1:
        sy = gp
        cell_size = mp.Vector3(sx, sy, 0)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy),
            )
        ]

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            k_point=k_point,
            default_material=glass,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=unit_reference_after_sources)

        input_flux = mp.get_fluxes(flux_obj)

        sim.reset_meep()

        geometry.append(
            mp.Block(
                material=glass,
                size=mp.Vector3(gh, gdc_list[0] * gp, mp.inf),
                center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
            )
        )

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=unit_grating_after_sources)

        freqs = mp.get_eigenmode_freqs(flux_obj)
        res = sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha

        mode_tran = abs(coeffs[0, 0, 0]) ** 2 / input_flux[0]
        mode_phase = np.angle(coeffs[0, 0, 0])
        if mode_phase > 0:
            mode_phase -= 2 * np.pi

        return mode_tran, mode_phase

    else:
        sy = num_cells * gp
        cell_size = mp.Vector3(sx, sy, 0)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy),
            )
        ]

        geometry.extend(
            mp.Block(
                material=glass,
                size=mp.Vector3(gh, gdc_list[j] * gp, mp.inf),
                center=mp.Vector3(
                    -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + (j + 0.5) * gp
                ),
            )
            for j in range(num_cells)
        )
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        n2f_obj = sim.add_near2far(
            fcen, 0, 1, mp.Near2FarRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=lens_after_sources)

        return (
            abs(
                sim.get_farfields(
                    n2f_obj,
                    ff_res,
                    center=mp.Vector3(-0.5 * sx + dpml + dsub + gh + focal_length),
                    size=mp.Vector3(spot_length),
                )["Ez"]
            )
            ** 2
        )


def main(validation=False, benchmark=False):
    global resolution
    global focal_length
    global spot_length
    global ff_res
    global unit_reference_after_sources
    global unit_grating_after_sources
    global lens_after_sources

    gp = 0.3  # grating periodicity
    gh = 1.8  # grating height

    bounded = validation or benchmark

    if validation:
        resolution = 40
        focal_length = 20
        spot_length = 12
        ff_res = 5
        unit_reference_after_sources = 30
        unit_grating_after_sources = 80
        lens_after_sources = 120
        gdc = np.linspace(0.1, 0.9, 9)
        gdc_new = np.linspace(0.16, 0.68, 300)
        phase_error_limit = 3e-2
        num_cells = [60, 150]
    elif benchmark:
        resolution = 50
        focal_length = 100
        spot_length = 20
        ff_res = 5
        unit_reference_after_sources = 30
        unit_grating_after_sources = 80
        lens_after_sources = 200
        gdc = np.linspace(0.1, 0.9, 9)
        gdc_new = np.linspace(0.16, 0.68, 300)
        phase_error_limit = 3e-2
        num_cells = [400]
    else:
        gdc = np.linspace(0.1, 0.9, 30)  # grating duty cycle
        gdc_new = np.linspace(0.16, 0.68, 500)
        phase_error_limit = None
        num_cells = [100, 200, 400]

    mode_tran = np.empty(gdc.size)
    mode_phase = np.empty(gdc.size)
    for n in range(gdc.size):
        mode_tran[n], mode_phase[n] = grating(gp, gh, [gdc[n]])

    if not bounded:
        plt.figure(dpi=200)
        plt.subplot(1, 2, 1)
        plt.plot(gdc, mode_tran, "bo-")
        plt.xlim(gdc[0], gdc[-1])
        plt.xticks(list(np.linspace(0.1, 0.9, 5)))
        plt.xlabel("grating duty cycle")
        plt.ylim(0.96, 1.00)
        plt.yticks(list(np.linspace(0.96, 1.00, 5)))
        plt.title("transmittance")

        plt.subplot(1, 2, 2)
        plt.plot(gdc, mode_phase, "rs-")
        plt.grid(True)
        plt.xlim(gdc[0], gdc[-1])
        plt.xticks(list(np.linspace(0.1, 0.9, 5)))
        plt.xlabel("grating duty cycle")
        plt.ylim(-2 * np.pi, 0)
        plt.yticks(list(np.linspace(-6, 0, 7)))
        plt.title("phase (radians)")

        plt.tight_layout(pad=0.5)
        plt.show()

    mode_phase_unwrapped = np.unwrap(mode_phase)
    mode_phase_interp = np.interp(gdc_new, gdc, mode_phase_unwrapped)
    phase_range = mode_phase_interp.max() - mode_phase_interp.min()
    print(f"phase-range:, {phase_range:.6f}")

    ff_nc = np.empty((spot_length * ff_res, len(num_cells)))
    profile_offsets = [0]
    profile_duty_cycles = []
    profile_target_phases = []
    profile_achieved_phases = []
    profile_circular_errors = []

    for k in range(len(num_cells)):
        cell_indices = np.arange(-num_cells[k], num_cells[k] + 1)
        target_phases = (
            2
            * np.pi
            / lcen
            * (focal_length - np.hypot(cell_indices * gp, focal_length))
        )
        circular_distances = np.abs(
            np.angle(
                np.exp(
                    1j * (mode_phase_interp[:, np.newaxis] - target_phases[np.newaxis])
                )
            )
        )
        selected_indices = np.argmin(circular_distances, axis=0)
        selected_duties = gdc_new[selected_indices]
        achieved_phases = mode_phase_interp[selected_indices]
        circular_errors = circular_distances[
            selected_indices, np.arange(target_phases.size)
        ]

        gdc_list = selected_duties.tolist()
        profile_duty_cycles.append(selected_duties)
        profile_target_phases.append(target_phases)
        profile_achieved_phases.append(achieved_phases)
        profile_circular_errors.append(circular_errors)
        profile_offsets.append(profile_offsets[-1] + target_phases.size)

        ff_nc[:, k] = grating(gp, gh, gdc_list)

    x = np.linspace(
        focal_length - 0.5 * spot_length,
        focal_length + 0.5 * spot_length,
        ff_res * spot_length,
    )

    metasurface_duty_cycles = np.asarray(gdc)
    metasurface_transmission = np.asarray(mode_tran)
    metasurface_phase_unit_real = np.cos(mode_phase)
    metasurface_phase_unit_imag = np.sin(mode_phase)
    metasurface_phase_unwrapped = np.asarray(mode_phase_unwrapped)
    metasurface_phase_range = np.asarray([phase_range])
    metasurface_half_cells = np.asarray(num_cells, dtype=float)
    metasurface_profile_offsets = np.asarray(profile_offsets, dtype=float)
    metasurface_profile_duty_cycles = np.concatenate(profile_duty_cycles)
    concatenated_target_phases = np.concatenate(profile_target_phases)
    concatenated_achieved_phases = np.concatenate(profile_achieved_phases)
    metasurface_profile_target_phase_unit_real = np.cos(concatenated_target_phases)
    metasurface_profile_target_phase_unit_imag = np.sin(concatenated_target_phases)
    metasurface_profile_achieved_phase_unit_real = np.cos(
        concatenated_achieved_phases
    )
    metasurface_profile_achieved_phase_unit_imag = np.sin(
        concatenated_achieved_phases
    )
    metasurface_profile_circular_error = np.concatenate(profile_circular_errors)
    metasurface_farfield_x = np.asarray(x)
    metasurface_farfield_intensity = np.asarray(ff_nc)
    peak_intensity = np.max(metasurface_farfield_intensity, axis=0)
    metasurface_normalized_farfield = metasurface_farfield_intensity / peak_intensity
    focus_indices = np.argmax(metasurface_farfield_intensity, axis=0)
    metasurface_focus_positions = metasurface_farfield_x[focus_indices]
    metasurface_focus_offsets = metasurface_focus_positions - focal_length
    edge_samples = np.vstack(
        (
            metasurface_farfield_intensity[:5],
            metasurface_farfield_intensity[-5:],
        )
    )
    metasurface_focus_contrast = peak_intensity / np.mean(edge_samples, axis=0)
    metasurface_peak_intensity = peak_intensity

    if bounded:
        globals().update(
            {
                "metasurface_duty_cycles": metasurface_duty_cycles,
                "metasurface_transmission": metasurface_transmission,
                "metasurface_phase_unit_real": metasurface_phase_unit_real,
                "metasurface_phase_unit_imag": metasurface_phase_unit_imag,
                "metasurface_phase_unwrapped": metasurface_phase_unwrapped,
                "metasurface_phase_range": metasurface_phase_range,
                "metasurface_half_cells": metasurface_half_cells,
                "metasurface_profile_offsets": metasurface_profile_offsets,
                "metasurface_profile_duty_cycles": metasurface_profile_duty_cycles,
                "metasurface_profile_target_phase_unit_real": metasurface_profile_target_phase_unit_real,
                "metasurface_profile_target_phase_unit_imag": metasurface_profile_target_phase_unit_imag,
                "metasurface_profile_achieved_phase_unit_real": metasurface_profile_achieved_phase_unit_real,
                "metasurface_profile_achieved_phase_unit_imag": metasurface_profile_achieved_phase_unit_imag,
                "metasurface_profile_circular_error": metasurface_profile_circular_error,
                "metasurface_farfield_x": metasurface_farfield_x,
                "metasurface_farfield_intensity": metasurface_farfield_intensity,
                "metasurface_normalized_farfield": metasurface_normalized_farfield,
                "metasurface_focus_positions": metasurface_focus_positions,
                "metasurface_focus_offsets": metasurface_focus_offsets,
                "metasurface_focus_contrast": metasurface_focus_contrast,
                "metasurface_peak_intensity": metasurface_peak_intensity,
            }
        )
        retained = (
            metasurface_duty_cycles,
            metasurface_transmission,
            metasurface_phase_unit_real,
            metasurface_phase_unit_imag,
            metasurface_phase_unwrapped,
            metasurface_phase_range,
            metasurface_half_cells,
            metasurface_profile_offsets,
            metasurface_profile_duty_cycles,
            metasurface_profile_target_phase_unit_real,
            metasurface_profile_target_phase_unit_imag,
            metasurface_profile_achieved_phase_unit_real,
            metasurface_profile_achieved_phase_unit_imag,
            metasurface_profile_circular_error,
            metasurface_farfield_x,
            metasurface_farfield_intensity,
            metasurface_normalized_farfield,
            metasurface_focus_positions,
            metasurface_focus_offsets,
            metasurface_focus_contrast,
            metasurface_peak_intensity,
        )
        if not all(np.all(np.isfinite(values)) for values in retained):
            raise RuntimeError("metasurface validation produced nonfinite data")
        if np.min(metasurface_transmission) < -1e-6:
            raise RuntimeError("metasurface has negative modal transmission")
        if np.max(metasurface_transmission) > 1.05:
            raise RuntimeError("metasurface violates passive transmission")
        if np.min(np.diff(metasurface_phase_unwrapped)) <= 0:
            raise RuntimeError("metasurface unit-cell phase is not monotonic")
        if metasurface_phase_range[0] < 2 * np.pi:
            raise RuntimeError("metasurface unit cells span too little phase")
        if np.max(metasurface_profile_circular_error) > phase_error_limit:
            raise RuntimeError(
                "metasurface profile exceeds its circular phase error: "
                f"{np.max(metasurface_profile_circular_error):.6g} > "
                f"{phase_error_limit:.6g}"
            )
        if np.min(metasurface_peak_intensity) <= 0:
            raise RuntimeError("metasurface far field has no positive intensity")
        if np.max(np.abs(metasurface_focus_offsets)) > 4.0:
            raise RuntimeError("metasurface focus is too far from the design target")
        if np.min(metasurface_focus_contrast) < 1.2:
            raise RuntimeError("metasurface far field has no resolved focus")

    if benchmark and mp.am_master():
        print(
            "gpmeep-metasurface-benchmark:"
            + json.dumps(
                {
                    "active_backend": mp.gpu.active_backend,
                    "focus_contrast": metasurface_focus_contrast.tolist(),
                    "focus_offsets": metasurface_focus_offsets.tolist(),
                    "half_cells": metasurface_half_cells.tolist(),
                    "peak_intensity": metasurface_peak_intensity.tolist(),
                    "phase_range": metasurface_phase_range.tolist(),
                    "profile_phase_error_max": float(
                        np.max(metasurface_profile_circular_error)
                    ),
                    "profile_phase_error_rms": float(
                        np.sqrt(np.mean(metasurface_profile_circular_error**2))
                    ),
                    "requested_backend": mp.gpu.requested_backend,
                    "statistics": mp.gpu.statistics(),
                },
                sort_keys=True,
            )
        )

    if not bounded:
        colors = ["b", "r", "g"]
        plt.figure(dpi=200)
        for idx, half_cells in enumerate(num_cells):
            plt.semilogy(
                x,
                abs(ff_nc[:, idx]) ** 2,
                f"{colors[idx]}o-",
                label=f"num_cells = {2 * half_cells + 1}",
            )
        plt.xlabel("x coordinate (μm)")
        plt.ylabel(r"energy density of far-field electric fields, |E$_z$|$^2$")
        plt.title("focusing properties of a binary-grating metasurface lens")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded unit-cell-to-focus CPU/CUDA oracle",
    )
    profile.add_argument(
        "--benchmark",
        action="store_true",
        help="run a production-scale 801-cell metalens performance profile",
    )
    args = parser.parse_args()
    main(validation=args.validation, benchmark=args.benchmark)
