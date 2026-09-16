import argparse

import numpy as np

import meep as mp


def mode_observables(mode):
    return (
        float(mode.decay),
        float(mode.Q),
        complex(mode.amp),
        complex(mode.err),
    )


def main(args):
    if args.perpendicular:
        src_cmpt = mp.Hz
        fcen = 0.21  # pulse center frequency
    else:
        src_cmpt = mp.Ez
        fcen = 0.17  # pulse center frequency

    n = 3.4  # index of waveguide
    w = 1  # ring width
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML
    m = 5  # angular dependence

    pml_layers = [mp.PML(dpml)]

    sr = r + w + pad + dpml  # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL  # coordinate system is (r,phi,z) instead of (x,y,z)
    cell = mp.Vector3(sr)

    geometry = [
        mp.Block(
            center=mp.Vector3(r + (w / 2)),
            size=mp.Vector3(w, mp.inf, mp.inf),
            material=mp.Medium(index=n),
        )
    ]

    # find resonant frequency of unperturbed geometry using broadband source

    df = 0.2 * fcen  # pulse width (in frequency)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        dimensions=dimensions,
        m=m,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=100)

    frq_unperturbed = h.modes[0].freq
    unperturbed_mode = mode_observables(h.modes[0])

    sim.reset_meep()

    # unperturbed geometry with narrowband source centered at resonant frequency

    fcen = frq_unperturbed
    df = 0.05 * fcen

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        dimensions=dimensions,
        m=m,
    )

    sim.run(until_after_sources=100)

    deps = 1 - n**2
    deps_inv = 1 - 1 / n**2

    if args.perpendicular:
        para_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ep, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ep, mp.Vector3(r + w))) ** 2
            )
        )
        perp_integral = (
            deps_inv
            * 2
            * np.pi
            * (
                -r * abs(sim.get_field_point(mp.Dr, mp.Vector3(r))) ** 2
                + (r + w) * abs(sim.get_field_point(mp.Dr, mp.Vector3(r + w))) ** 2
            )
        )
        numerator_integral = para_integral + perp_integral
    else:
        numerator_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ez, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ez, mp.Vector3(r + w))) ** 2
            )
        )

    denominator_integral = sim.electric_energy_in_box(
        center=mp.Vector3(0.5 * sr), size=mp.Vector3(sr)
    )
    perturb_theory_dw_dR = (
        -frq_unperturbed * numerator_integral / (4 * denominator_integral)
    )

    sim.reset_meep()

    # perturbed geometry with narrowband source

    dr = 0.01

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + dr + 0.1),
        )
    ]

    geometry = [
        mp.Block(
            center=mp.Vector3(r + dr + (w / 2)),
            size=mp.Vector3(w, mp.inf, mp.inf),
            material=mp.Medium(index=n),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        dimensions=dimensions,
        m=m,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + dr + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=100)

    frq_perturbed = h.modes[0].freq
    perturbed_mode = mode_observables(h.modes[0])

    finite_diff_dw_dR = (frq_perturbed - frq_unperturbed) / dr

    print(
        f"dwdR:, {perturb_theory_dw_dR} (pert. theory), {finite_diff_dw_dR} (finite diff.)"
    )
    perturbation_frequencies = np.asarray(
        [frq_unperturbed, frq_perturbed], dtype=np.complex128
    )
    perturbation_derivatives = np.asarray(
        [perturb_theory_dw_dR, finite_diff_dw_dR], dtype=np.complex128
    )
    perturbation_integrals = np.asarray(
        [numerator_integral, denominator_integral], dtype=np.complex128
    )
    perturbation_relative_error = np.asarray(
        [
            abs(perturb_theory_dw_dR - finite_diff_dw_dR)
            / max(abs(finite_diff_dw_dR), 1e-30)
        ]
    )
    perturbation_mode_decays = np.asarray(
        [unperturbed_mode[0], perturbed_mode[0]], dtype=float
    )
    perturbation_mode_q = np.asarray(
        [unperturbed_mode[1], perturbed_mode[1]], dtype=float
    )
    perturbation_mode_amplitudes = np.asarray(
        [unperturbed_mode[2], perturbed_mode[2]], dtype=np.complex128
    )
    perturbation_mode_errors = np.asarray(
        [unperturbed_mode[3], perturbed_mode[3]], dtype=np.complex128
    )
    if (
        not np.all(np.isfinite(perturbation_frequencies))
        or not np.all(np.isfinite(perturbation_derivatives))
        or not np.all(np.isfinite(perturbation_integrals))
        or not np.all(np.isfinite(perturbation_mode_decays))
        or not np.all(np.isfinite(perturbation_mode_q))
        or not np.all(np.isfinite(perturbation_mode_amplitudes))
        or not np.all(np.isfinite(perturbation_mode_errors))
        or denominator_integral <= 0
        or np.any(np.abs(perturbation_derivatives) <= 1e-10)
    ):
        raise RuntimeError("cylindrical perturbation calculation is non-physical")
    return {
        "perturbation_frequencies": perturbation_frequencies,
        "perturbation_derivatives": perturbation_derivatives,
        "perturbation_integrals": perturbation_integrals,
        "perturbation_relative_error": perturbation_relative_error,
        "perturbation_mode_decays": perturbation_mode_decays,
        "perturbation_mode_q": perturbation_mode_q,
        "perturbation_mode_amplitudes": perturbation_mode_amplitudes,
        "perturbation_mode_errors": perturbation_mode_errors,
        "perturbation_polarization": np.asarray(
            [1 if args.perpendicular else 0], dtype=np.int64
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-perpendicular",
        action="store_true",
        help="use perpendicular field source (default: parallel field source)",
    )
    parser.add_argument(
        "-res", type=int, default=100, help="resolution (default: 100 pixels/um)"
    )
    args = parser.parse_args()
    globals().update(main(args))
