"""The 3D adjoint gradient must not depend on the chunk division.

This regression is backported from NanoComp/meep PR #3274 (commit
897dbe06). It forces two different chunk layouts even in a serial run, so the
MPI-only manifestation of the DFT chunk-pairing/indexing defect remains covered
by every build configuration.
"""

import unittest

import numpy as np
from autograd import numpy as npa

import meep as mp
import meep.adjoint as mpa


RESOLUTION = 12
DESIGN_N = 6
SEED = 0


def _gradient(num_chunks):
    """Return objective and adjoint gradient for a forced chunk count."""
    si, clad = mp.Medium(index=3.48), mp.Medium(index=1.44)
    pml, port_pad, side_pad = 0.5, 0.8, 0.5
    design, thickness = 1.0, 0.22

    sx = 2 * pml + 2 * port_pad + design
    sy = 2 * pml + 2 * side_pad + design
    sz = 2 * pml + 2 * side_pad + thickness

    weights = np.random.default_rng(SEED).uniform(
        0.2, 0.8, size=DESIGN_N * DESIGN_N
    )
    grid = mp.MaterialGrid(
        mp.Vector3(DESIGN_N, DESIGN_N, 1),
        clad,
        si,
        weights=weights.reshape(DESIGN_N, DESIGN_N, 1),
        do_averaging=False,
    )
    region = mpa.DesignRegion(
        grid,
        volume=mp.Volume(
            center=mp.Vector3(), size=mp.Vector3(design, design, thickness)
        ),
    )

    fcen = 1 / 1.55
    port = mp.Vector3(0, sy - 2 * pml, sz - 2 * pml)
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        resolution=RESOLUTION,
        boundary_layers=[mp.PML(pml)],
        default_material=clad,
        geometry=[
            mp.Block(
                center=mp.Vector3(),
                size=mp.Vector3(mp.inf, 0.5, thickness),
                material=si,
            ),
            mp.Block(center=region.center, size=region.size, material=grid),
        ],
        sources=[
            mp.EigenModeSource(
                mp.GaussianSource(fcen, fwidth=0.1 * fcen),
                center=mp.Vector3(-(design / 2 + port_pad / 2)),
                size=port,
                eig_band=1,
            )
        ],
        eps_averaging=False,
        num_chunks=num_chunks,
    )
    monitor = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(center=mp.Vector3(design / 2 + port_pad / 2), size=port),
        mode=1,
    )
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[lambda c: npa.abs(c) ** 2],
        objective_arguments=[monitor],
        design_regions=[region],
        frequencies=[fcen],
        decay_by=1e-6,
    )
    f0, gradient = opt([weights], need_gradient=True)
    return (
        float(np.squeeze(f0)),
        np.asarray(np.real(np.squeeze(gradient)), dtype=np.float64).reshape(-1),
    )


class TestAdjointChunks(unittest.TestCase):
    def test_gradient_independent_of_chunk_division(self):
        process_count = mp.count_processors()
        f0_a, gradient_a = _gradient(num_chunks=process_count)
        f0_b, gradient_b = _gradient(num_chunks=3 * process_count)

        # FP32 collective summation order yields roughly 1e-7 variation. This
        # remains four orders below the pre-fix chunk-dependent regression.
        tolerance = 1e-5 if mp.is_single_precision() else 1e-9
        self.assertAlmostEqual(f0_a / f0_b, 1.0, delta=tolerance)

        relative_error = np.linalg.norm(gradient_a - gradient_b) / np.linalg.norm(
            gradient_a
        )
        self.assertLess(
            relative_error,
            tolerance,
            "gradient changed by "
            f"{relative_error:.3e} under a different chunk split",
        )


if __name__ == "__main__":
    unittest.main()
