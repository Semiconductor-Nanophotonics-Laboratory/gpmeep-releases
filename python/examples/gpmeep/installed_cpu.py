#!/usr/bin/env python3
"""Minimal installed gpmeep CPU execution example."""

import json

import meep as mp


mp.gpu.set_backend("cpu")
mp.gpu.reset_statistics()
simulation = mp.Simulation(
    cell_size=mp.Vector3(4, 3),
    resolution=20,
    boundary_layers=[mp.PML(0.5)],
    sources=[
        mp.Source(
            mp.GaussianSource(0.3, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(-1),
        )
    ],
)
simulation.run(until=10)
record = {
    "requested_backend": mp.gpu.requested_backend,
    "active_backend": mp.gpu.active_backend,
    "timesteps": int(simulation.timestep()),
    "statistics": mp.gpu.statistics(),
}
if record["active_backend"] != "cpu":
    raise RuntimeError("strict CPU backend was not selected")
print(json.dumps(record, sort_keys=True))
simulation.reset_meep()
