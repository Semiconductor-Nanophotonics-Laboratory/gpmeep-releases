#!/usr/bin/env python3
"""Minimal installed gpmeep strict single-GPU execution example."""

import json

import meep as mp


if not mp.gpu.compiled:
    raise RuntimeError("this gpmeep installation has no CUDA backend")
mp.gpu.set_backend("cuda")
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
    "selected_device_identifier": mp.gpu.selected_device_identifier,
    "cuda_execution_selected": bool(
        simulation.fields.gpu_cuda_execution_selected()
    ),
    "timesteps": int(simulation.timestep()),
    "statistics": mp.gpu.statistics(),
}
if record["active_backend"] != "cuda" or not record["cuda_execution_selected"]:
    raise RuntimeError("strict CUDA work was not selected")
print(json.dumps(record, sort_keys=True))
simulation.reset_meep()
