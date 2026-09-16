#!/usr/bin/env python3
"""MPI negative-test driver for distributed eigenmode backend consensus.

Run this with at least two MPI ranks.  The launcher should expose one CUDA
device per rank and request the strict CUDA backend.  Rank 1 deliberately
switches to the CPU immediately before the public eigenmode-overlap call.
The only valid outcome is a collective fail-closed abort; reaching the final
marker means the backend-consensus guard regressed.
"""

import meep as mp


def main():
    if mp.count_processors() < 2:
        raise RuntimeError("mixed-backend consensus validation requires at least two MPI ranks")

    frequency = 1.0
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 1),
        resolution=10,
        default_material=mp.Medium(index=1.5),
        sources=[
            mp.Source(
                mp.GaussianSource(frequency, fwidth=0.1),
                component=mp.Ez,
                center=mp.Vector3(),
            )
        ],
    )
    monitor = simulation.add_mode_monitor(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mp.Vector3(), size=mp.Vector3(0, 1)),
    )
    simulation.run(until=0.5)
    mp.all_wait()
    if mp.my_rank() == 1:
        mp.gpu.set_backend("cpu")
    mp.all_wait()
    print(f"M33_MIXED_BACKEND_ENTER rank={mp.my_rank()}", flush=True)
    simulation.get_eigenmode_coefficients(
        monitor,
        bands=[1],
        direction=mp.X,
        eig_tolerance=1e-10,
    )
    print("M33_MIXED_BACKEND_UNEXPECTED_RETURN", flush=True)


if __name__ == "__main__":
    main()
