import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import meep as mp

resolution = 60  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width
nfreq = 21  # number of frequency bins

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)


def grating(gp, gh, gdc, oddz):
    sx = dpml + dsub + gh + dpad + dpml
    sy = gp

    cell_size = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez if oddz else mp.Hz,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        )
    ]

    symmetries = [mp.Mirror(mp.Y, phase=+1 if oddz else -1)]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        default_material=glass,
        sources=sources,
        symmetries=symmetries,
    )

    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
    flux_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=100)

    input_flux = mp.get_fluxes(flux_mon)

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
        k_point=k_point,
        sources=sources,
        symmetries=symmetries,
    )

    mode_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=300)

    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
    )
    coeffs = res.alpha

    mode_wvl = [1 / freqs[nf] for nf in range(nfreq)]
    mode_tran = [abs(coeffs[0, nf, 0]) ** 2 / input_flux[nf] for nf in range(nfreq)]
    mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(nfreq)]

    return mode_wvl, mode_tran, mode_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gp", type=float, default=0.35, help="grating periodicity (default: 0.35 μm)"
    )
    parser.add_argument(
        "-gh", type=float, default=0.6, help="grating height (default: 0.6 μm)"
    )
    parser.add_argument(
        "-oddz", action="store_true", default=False, help="oddz? (default: False)"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded two-polarization CPU/CUDA phase-map oracle",
    )
    args = parser.parse_args()

    if args.validation:
        resolution = 40
        nfreq = 7
        gdc = np.asarray([0.2, 0.5, 0.8])
        oddz_values = (False, True)
    else:
        gdc = np.arange(0.1, 1.0, 0.1)
        oddz_values = (args.oddz,)

    mode_tran = np.empty((len(oddz_values), gdc.size, nfreq))
    mode_phase = np.empty_like(mode_tran)
    for pol_idx, oddz in enumerate(oddz_values):
        for duty_idx, duty_cycle in enumerate(gdc):
            mode_wvl, mode_tran[pol_idx, duty_idx, :], mode_phase[
                pol_idx, duty_idx, :
            ] = grating(args.gp, args.gh, duty_cycle, oddz)

    binary_phasemap_wavelengths = np.asarray(mode_wvl)
    binary_phasemap_duty_cycles = np.asarray(gdc)
    binary_phasemap_oddz = np.asarray(oddz_values, dtype=float)
    binary_phasemap_transmission = np.asarray(mode_tran)
    binary_phasemap_phase_unit_real = np.cos(mode_phase)
    binary_phasemap_phase_unit_imag = np.sin(mode_phase)
    phase_phasors = np.exp(1j * mode_phase)
    binary_phasemap_phase_duty_spread = np.sqrt(
        np.mean(
            np.abs(phase_phasors - np.mean(phase_phasors, axis=1, keepdims=True))
            ** 2,
            axis=1,
        )
    )
    binary_phasemap_phase_frequency_spread = np.sqrt(
        np.mean(
            np.abs(phase_phasors - np.mean(phase_phasors, axis=2, keepdims=True))
            ** 2,
            axis=2,
        )
    )
    binary_phasemap_phase_polarization_difference = np.asarray(
        [
            np.linalg.norm(phase_phasors[0] - phase_phasors[-1])
            / np.sqrt(phase_phasors[0].size)
        ]
    )
    binary_phasemap_transmission_variation = np.ptp(mode_tran, axis=1)
    binary_phasemap_polarization_difference = np.asarray(
        [
            np.linalg.norm(mode_tran[0] - mode_tran[-1])
            / np.linalg.norm(mode_tran[0])
        ]
    )

    if args.validation:
        retained = (
            binary_phasemap_wavelengths,
            binary_phasemap_duty_cycles,
            binary_phasemap_oddz,
            binary_phasemap_transmission,
            binary_phasemap_phase_unit_real,
            binary_phasemap_phase_unit_imag,
            binary_phasemap_phase_duty_spread,
            binary_phasemap_phase_frequency_spread,
            binary_phasemap_phase_polarization_difference,
            binary_phasemap_transmission_variation,
            binary_phasemap_polarization_difference,
        )
        if not all(np.all(np.isfinite(values)) for values in retained):
            raise RuntimeError("binary phase-map validation produced nonfinite data")
        if np.min(binary_phasemap_transmission) < -1e-6:
            raise RuntimeError("binary phase-map has negative modal transmission")
        if np.max(binary_phasemap_transmission) > 1.05:
            raise RuntimeError("binary phase-map violates passive transmission")
        if np.min(np.linalg.norm(binary_phasemap_transmission_variation, axis=1)) < 0.02:
            raise RuntimeError("binary phase-map does not resolve duty-cycle variation")
        if binary_phasemap_polarization_difference[0] < 0.01:
            raise RuntimeError("binary phase-map polarizations are indistinguishable")
        phase_norm = np.hypot(
            binary_phasemap_phase_unit_real, binary_phasemap_phase_unit_imag
        )
        if np.max(np.abs(phase_norm - 1.0)) > 1e-12:
            raise RuntimeError("binary phase-map phase phasors are not unit magnitude")
        if np.min(binary_phasemap_phase_duty_spread) < 0.5:
            raise RuntimeError("binary phase-map has no duty-cycle phase response")
        if np.min(binary_phasemap_phase_frequency_spread) < 0.5:
            raise RuntimeError("binary phase-map has no frequency phase response")
        if binary_phasemap_phase_polarization_difference[0] < 0.5:
            raise RuntimeError("binary phase-map polarizations have identical phase")

    if not args.validation:
        mode_tran = mode_tran[0]
        mode_phase = mode_phase[0]

        plt.figure(dpi=150)

        plt.subplot(1, 2, 1)
        plt.pcolormesh(
            mode_wvl,
            gdc,
            mode_tran,
            cmap="hot_r",
            shading="gouraud",
            vmin=0,
            vmax=mode_tran.max(),
        )
        plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
        plt.xlabel("wavelength (μm)")
        plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
        plt.ylabel("grating duty cycle")
        plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
        plt.title("transmittance")
        cbar = plt.colorbar()
        cbar.set_ticks(list(np.arange(0, 1.2, 0.2)))
        cbar.set_ticklabels([f"{t:.1f}" for t in np.arange(0, 1.2, 0.2)])

        plt.subplot(1, 2, 2)
        plt.pcolormesh(
            mode_wvl,
            gdc,
            mode_phase,
            cmap="RdBu",
            shading="gouraud",
            vmin=mode_phase.min(),
            vmax=mode_phase.max(),
        )
        plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
        plt.xlabel("wavelength (μm)")
        plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
        plt.ylabel("grating duty cycle")
        plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
        plt.title("phase (radians)")
        cbar = plt.colorbar()
        cbar.set_ticks(list(range(-3, 4)))
        cbar.set_ticklabels([f"{t:.1f}" for t in range(-3, 4)])

        plt.tight_layout()
        plt.show()
