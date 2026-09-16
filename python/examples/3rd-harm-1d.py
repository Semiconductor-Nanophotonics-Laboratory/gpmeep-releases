"""
1d simulation of a plane wave propagating through a Kerr medium
and generating the third-harmonic frequency component.

ref: https://meep.readthedocs.io/en/latest/Python_Tutorials/Third_Harmonic_Generation/
"""

import argparse
from typing import Tuple, List, Union
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import meep as mp


def third_harmonic_generation(
    k: float, amp: float = 1.0, nfreq: int = 10, flux_spectrum: bool = True
) -> Union[Tuple[List[float], List[float]], Tuple[float, float]]:

    """Computes the transmission spectrum of a plane wave propagating
       through a Kerr medium.

    Args:
      k: strength of Kerr susceptibility.
      amp: amplitude of the incident planewave.
      nfreq: number of frequencies in flux spectrum.
      flux_spectrum: compute the flux spectrum over broad bandwidth (True) or
                     just the two harmonic frequencies ω and 3ω (False).

    Returns:
      The frequencies and transmitted flux over the flux spectrum or
      the transmitted flux at the harmonic frequencies ω and 3ω.
    """

    sz = 100  # size of cell in z direction
    fcen = 1 / 3.0  # center frequency of source
    df = fcen / 20.0  # frequency width of source
    dpml = 1.0  # PML thickness

    # We'll use an explicitly 1d simulation.  Setting dimensions=1 will actually
    # result in faster execution than just using two no-size dimensions.  However,
    # in this case Meep requires us to use E in the x direction (and H in y),
    # and our one no-size dimension must be z.
    dimensions = 1
    cell = mp.Vector3(0, 0, sz)
    pml_layers = [mp.PML(dpml)]
    resolution = 25

    # to put the same material in all space, we can just set the default material
    # and pass it to the Simulation constructor
    default_material = mp.Medium(index=1, chi3=k)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.5 * sz + dpml),
            amplitude=amp,
        )
    ]

    # frequency range for flux calculation
    fmin = fcen / 2.0
    fmax = fcen * 4

    sim = mp.Simulation(
        cell_size=cell,
        sources=sources,
        boundary_layers=pml_layers,
        default_material=default_material,
        resolution=resolution,
        dimensions=dimensions,
    )

    mon_pt = mp.Vector3(0, 0, 0.5 * sz - dpml - 0.5)

    if flux_spectrum:
        trans = sim.add_flux(
            0.5 * (fmin + fmax),
            fmax - fmin,
            nfreq,
            mp.FluxRegion(mon_pt),
        )
    else:
        trans1 = sim.add_flux(fcen, 0, 1, mp.FluxRegion(mon_pt))
        trans3 = sim.add_flux(3 * fcen, 0, 1, mp.FluxRegion(mon_pt))

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mon_pt, 1e-6))

    if flux_spectrum:
        freqs = mp.get_flux_freqs(trans)
        trans_flux = mp.get_fluxes(trans)
        return freqs, trans_flux
    else:
        print(
            f"harmonics:, {k}, {amp}, {mp.get_fluxes(trans1)[0]}, "
            f"{mp.get_fluxes(trans3)[0]}"
        )
        return mp.get_fluxes(trans1)[0], mp.get_fluxes(trans3)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run the complete numerical workflow without writing plot files",
    )
    args = parser.parse_args()

    # Part 1: plot transmitted power spectrum for several values of χ(3).
    nfreq = 400
    logk = range(-3, 1)
    tflux = np.zeros((nfreq, len(logk)))
    for i, lk in enumerate(logk):
        freqs, tflux[:, i] = third_harmonic_generation(
            10**lk, nfreq=nfreq, flux_spectrum=True
        )

    if not args.validation:
        fig, ax = plt.subplots()
        ax.semilogy(freqs, tflux[:, 0], "bo-", label=r"$\chi^{(3)}$=0.001")
        ax.semilogy(freqs, tflux[:, 1], "ro-", label=r"$\chi^{(3)}$=0.01")
        ax.semilogy(freqs, tflux[:, 2], "go-", label=r"$\chi^{(3)}$=0.1")
        ax.semilogy(freqs, tflux[:, 3], "co-", label=r"$\chi^{(3)}$=1")
        ax.set_xlabel("frequency")
        ax.set_ylabel("transmitted power (a.u.)")
        ax.set_xlim(0.2, 1.2)
        ax.set_ylim(1e-15, 1e2)
        ax.legend()
        ax.grid(True)
        fig.savefig(
            "transmitted_power_vs_frequency_vary_chi3.png",
            dpi=150,
            bbox_inches="tight",
        )

    # Part 2: plot transmittance vs. χ(3) for frequencies ω and 3ω.
    logk = np.arange(-6.0, 0.2, 0.2)
    first_order = np.zeros(len(logk))
    third_order = np.zeros(len(logk))
    for i, lk in enumerate(logk):
        first_order[i], third_order[i] = third_harmonic_generation(
            10**lk, flux_spectrum=False
        )

    input_flux = first_order[0]
    chi3_values = 10**logk
    normalized_fundamental_flux = first_order / input_flux
    normalized_third_harmonic_flux = third_order / input_flux

    # In the perturbative regime the generated third-harmonic power is
    # proportional to the square of chi3.  Keep the full published sweep as
    # machine-readable output and independently require this physical scaling
    # over a range which is above the FP32 noise floor and below saturation.
    third_harmonic_low_power_slopes = np.diff(
        np.log10(third_order[4:17])
    ) / np.diff(np.log10(chi3_values[4:17]))
    if not np.all(
        (third_harmonic_low_power_slopes > 1.9)
        & (third_harmonic_low_power_slopes < 2.1)
    ):
        raise RuntimeError(
            "third-harmonic power does not follow the expected quadratic "
            "low-chi3 scaling: "
            f"{third_harmonic_low_power_slopes.tolist()}"
        )

    spectrum_frequencies = np.asarray(freqs)
    spectrum_fluxes = np.asarray(tflux)
    if not np.all(np.isfinite(spectrum_frequencies)) or not np.all(
        np.isfinite(spectrum_fluxes)
    ):
        raise RuntimeError(
            "the third-harmonic spectra must contain only finite values"
        )
    expected_spectrum_frequencies = np.linspace(1 / 6.0, 4 / 3.0, nfreq)
    if not np.allclose(
        spectrum_frequencies,
        expected_spectrum_frequencies,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("third-harmonic spectrum frequency grid is incorrect")
    spectrum_column_l2 = np.linalg.norm(spectrum_fluxes, axis=0)
    if not np.all(np.isfinite(spectrum_column_l2)) or np.any(
        spectrum_column_l2 <= 1e-6
    ):
        raise RuntimeError(
            "one or more third-harmonic spectrum runs produced no signal: "
            f"{spectrum_column_l2.tolist()}"
        )
    if (
        not np.all(np.isfinite(first_order))
        or not np.all(np.isfinite(third_order))
        or np.any(first_order <= 0)
        or np.any(third_order <= 0)
    ):
        raise RuntimeError(
            "the chi3 sweep must produce positive finite fundamental and "
            "third-harmonic powers at every point"
        )
    fundamental_fluxes = np.asarray(first_order)
    third_harmonic_fluxes = np.asarray(third_order)

    if not args.validation:
        fig, ax = plt.subplots()
        ax.loglog(
            chi3_values, normalized_fundamental_flux, "ro-", label=r"$\omega$"
        )
        ax.loglog(
            chi3_values,
            normalized_third_harmonic_flux,
            "bo-",
            label=r"$3\omega$",
        )
        ax.loglog(chi3_values, chi3_values**2, "k-", label="quadratic line")
        ax.set_xlabel(r"$\chi^{(3)}$")
        ax.set_ylabel("transmission / incident power")
        ax.legend()
        ax.grid(True, "both")
        fig.savefig("transmittance_vs_chi3.png", dpi=150, bbox_inches="tight")
