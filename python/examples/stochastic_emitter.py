import argparse

import numpy as np
from meep.materials import Ag

import meep as mp

parser = argparse.ArgumentParser()
parser.add_argument("-res", type=int, default=50, help="resolution (pixels/um)")
parser.add_argument(
    "-nr", type=int, default=20, help="number of random trials (method 1)"
)
parser.add_argument("-nd", type=int, default=10, help="number of dipoles")
parser.add_argument("-nf", type=int, default=500, help="number of frequencies")
parser.add_argument(
    "-textured",
    action="store_true",
    default=False,
    help="flat (default) or textured surface",
)
parser.add_argument(
    "-method",
    type=int,
    choices=[1, 2],
    default=1,
    help="type of method: (1) random dipoles with nr trials or (2) single dipole with 1 run per dipole",
)
parser.add_argument(
    "-seed",
    type=int,
    default=None,
    help="optional deterministic seed for reproducible random-source validation",
)
parser.add_argument(
    "--no-save",
    action="store_true",
    help=argparse.SUPPRESS,
)
args = parser.parse_args()

resolution = args.res

dpml = 1.0
dair = 1.0
hrod = 0.7
wrod = 0.5
dsub = 5.0
dAg = 0.5

sx = 1.1
sy = dpml + dair + hrod + dsub + dAg

cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(direction=mp.Y, thickness=dpml, side=mp.High)]

fcen = 1.0
df = 0.2
nfreq = args.nf
ndipole = args.nd
ntrial = args.nr
run_time = 2 * nfreq / df

geometry = [
    mp.Block(
        material=mp.Medium(index=3.45),
        center=mp.Vector3(0, 0.5 * sy - dpml - dair - hrod - 0.5 * dsub),
        size=mp.Vector3(mp.inf, dsub, mp.inf),
    ),
    mp.Block(
        material=Ag,
        center=mp.Vector3(0, -0.5 * sy + 0.5 * dAg),
        size=mp.Vector3(mp.inf, dAg, mp.inf),
    ),
]


class GaussianNoiseSource:
    """Per-dipole random source with independently auditable call count."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.calls = 0

    def __call__(self, _time):
        self.calls += 1
        if self.rng is None:
            return np.random.randn()
        return self.rng.standard_normal()


random_source_callbacks = []
random_source_index = 0

if args.textured:
    geometry.append(
        mp.Block(
            material=mp.Medium(index=3.45),
            center=mp.Vector3(0, 0.5 * sy - dpml - dair - 0.5 * hrod),
            size=mp.Vector3(wrod, hrod, mp.inf),
        )
    )


def compute_flux(m=1, n=0):
    global random_source_index
    if m == 1:
        callbacks = []
        for _ in range(ndipole):
            callback_seed = (
                None if args.seed is None else args.seed + random_source_index
            )
            random_source_index += 1
            callback = GaussianNoiseSource(callback_seed)
            callbacks.append(callback)
            random_source_callbacks.append(callback)
        sources = [
            mp.Source(
                mp.CustomSource(src_func=callbacks[n]),
                component=mp.Ez,
                center=mp.Vector3(
                    sx * (-0.5 + n / ndipole), -0.5 * sy + dAg + 0.5 * dsub
                ),
            )
            for n in range(ndipole)
        ]

    else:
        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(
                    sx * (-0.5 + n / ndipole), -0.5 * sy + dAg + 0.5 * dsub
                ),
            )
        ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        k_point=mp.Vector3(),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
    )

    flux_mon = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0.5 * sy - dpml), size=mp.Vector3(sx)),
    )

    sim.run(until=run_time)

    flux = mp.get_fluxes(flux_mon)
    freqs = mp.get_flux_freqs(flux_mon)

    return freqs, flux


if args.method == 1:
    fluxes = np.zeros((nfreq, ntrial))
    for t in range(ntrial):
        freqs, fluxes[:, t] = compute_flux(m=1)
else:
    fluxes = np.zeros((nfreq, ndipole))
    for d in range(ndipole):
        freqs, fluxes[:, d] = compute_flux(m=2, n=d)

mean_flux = np.mean(fluxes, axis=1)
standard_deviation_flux = np.std(fluxes, axis=1)
trial_flux_l2 = np.linalg.norm(fluxes, axis=0)
source_callback_counts = np.asarray(
    [callback.calls for callback in random_source_callbacks], dtype=np.int64
)
validation_seed = -1 if args.seed is None else args.seed
if not np.all(np.isfinite(fluxes)) or not np.all(np.diff(freqs) > 0):
    raise RuntimeError("stochastic-emitter spectrum is non-finite or unordered")
if not np.any(np.abs(fluxes) > 0):
    raise RuntimeError("stochastic-emitter spectrum is identically zero")
if np.any(trial_flux_l2 <= 1e-10):
    raise RuntimeError(
        "one or more stochastic-emitter trials produced no spectrum: "
        f"{trial_flux_l2.tolist()}"
    )
if args.method == 1 and np.linalg.norm(standard_deviation_flux) <= 1e-10:
    raise RuntimeError("stochastic-emitter trials have zero ensemble variation")
if args.method == 1 and (
    source_callback_counts.size != ntrial * ndipole
    or np.any(source_callback_counts <= 0)
):
    raise RuntimeError(
        "stochastic source callbacks did not execute exactly once per source set"
    )


if mp.am_master() and not args.no_save:
    with open(
        f'method{args.method}_{"textured" if args.textured else "flat"}_res{resolution}_nfreq{nfreq}_ndipole{ndipole}.npz',
        "wb",
    ) as f:
        np.savez(f, freqs=freqs, fluxes=fluxes)
