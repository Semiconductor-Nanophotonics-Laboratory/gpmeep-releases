#!/usr/bin/env python3

"""Continuously phase one nondispersive material layout into another."""

import argparse

import numpy as np

import meep as mp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--validation",
    action="store_true",
    help="use a bounded run and publish material arrays for CPU/CUDA validation",
)
parser.add_argument("--resolution", type=int, default=None)
parser.add_argument("--phase-time", type=float, default=None)
args = parser.parse_args()

resolution = args.resolution if args.resolution is not None else (10 if args.validation else 20)
phase_time = args.phase_time if args.phase_time is not None else (2.0 if args.validation else 10.0)
if resolution <= 0 or phase_time <= 0:
    raise RuntimeError("resolution and phase time must be positive")
if args.validation and (resolution != 10 or abs(phase_time - 2.0) > 1e-12):
    raise RuntimeError(
        "--validation has a fixed resolution=10 and phase-time=2 physical contract"
    )

cell_size = mp.Vector3(6, 6, 0)
medium = mp.Medium(index=3.5)

geometry1 = [mp.Cylinder(center=mp.Vector3(), radius=1.0, material=medium)]
validation_sources = (
    [
        mp.Source(
            mp.ContinuousSource(0.5, end_time=phase_time),
            component=mp.Ez,
            # Keep the validation excitation inside the part of the original
            # cylinder which phases to vacuum.  A distant source can exercise
            # CUDA timestepping without ever making the changing coefficient
            # materially affect the bounded trajectory.
            center=mp.Vector3(-0.5, 0),
        )
    ]
    if args.validation
    else []
)
sim1 = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry1,
    sources=validation_sources,
    resolution=resolution,
)
sim1.init_sim()

geometry2 = [
    mp.Cylinder(center=mp.Vector3(1, 1), radius=1.0, material=medium)
]
sim2 = mp.Simulation(cell_size=cell_size, geometry=geometry2, resolution=resolution)
sim2.init_sim()


def epsilon_array(sim):
    return np.asarray(sim.get_array(component=mp.Dielectric), dtype=float).copy()


phase_initial_epsilon = epsilon_array(sim1)
phase_target_epsilon = epsilon_array(sim2)
phase_snapshot_times_list = []
phase_snapshot_epsilon_list = []
phase_snapshot_ez_list = []
phase_snapshot_dz_list = []


def capture_epsilon(sim):
    time_value = float(sim.meep_time())
    epsilon = epsilon_array(sim)
    ez = np.asarray(sim.get_array(component=mp.Ez), dtype=float).copy()
    dz = np.asarray(sim.get_array(component=mp.Dz), dtype=float).copy()
    if phase_snapshot_times_list and abs(time_value - phase_snapshot_times_list[-1]) < 1e-12:
        phase_snapshot_epsilon_list[-1] = epsilon
        phase_snapshot_ez_list[-1] = ez
        phase_snapshot_dz_list[-1] = dz
    else:
        phase_snapshot_times_list.append(time_value)
        phase_snapshot_epsilon_list.append(epsilon)
        phase_snapshot_ez_list.append(ez)
        phase_snapshot_dz_list.append(dz)


sim1.fields.phase_in_material(sim2.structure, phase_time)

if args.validation:
    sim1.run(
        mp.at_beginning(capture_epsilon),
        mp.at_every(0.5, capture_epsilon),
        until=phase_time,
    )
    capture_epsilon(sim1)
else:
    sim1.run(
        mp.at_beginning(mp.output_epsilon),
        mp.at_every(0.5, mp.output_epsilon),
        until=phase_time,
    )

phase_snapshot_times = np.asarray(phase_snapshot_times_list, dtype=float)
phase_snapshot_epsilon = np.asarray(phase_snapshot_epsilon_list, dtype=float)
phase_snapshot_ez = np.asarray(phase_snapshot_ez_list, dtype=float)
phase_snapshot_dz = np.asarray(phase_snapshot_dz_list, dtype=float)
phase_final_epsilon = epsilon_array(sim1)

if args.validation:
    if phase_snapshot_times.shape != (5,):
        raise RuntimeError(
            "phase_in_material validation expected snapshots at 0, 0.5, 1, 1.5, 2"
        )
    expected_times = np.linspace(0.0, phase_time, 5)
    if not np.allclose(phase_snapshot_times, expected_times, rtol=0.0, atol=1e-6):
        raise RuntimeError("phase_in_material snapshot times changed")

    # Material phasing interpolates chi1inv=1/epsilon, not epsilon itself.
    inverse_initial = np.reciprocal(phase_initial_epsilon)
    inverse_target = np.reciprocal(phase_target_epsilon)
    expected = []
    for time_value in phase_snapshot_times:
        alpha = time_value / phase_time
        expected_inverse = (1.0 - alpha) * inverse_initial + alpha * inverse_target
        expected.append(np.reciprocal(expected_inverse))
    expected = np.asarray(expected)
    scale = max(float(np.linalg.norm(expected)), 1e-30)
    phase_reciprocal_relative_error = np.asarray(
        [float(np.linalg.norm(phase_snapshot_epsilon - expected) / scale)]
    )
    endpoint_scale = max(float(np.linalg.norm(phase_target_epsilon)), 1e-30)
    phase_endpoint_relative_error = np.asarray(
        [float(np.linalg.norm(phase_final_epsilon - phase_target_epsilon) / endpoint_scale)]
    )
    phase_progress = np.asarray(
        [
            float(np.linalg.norm(snapshot - phase_initial_epsilon))
            for snapshot in phase_snapshot_epsilon
        ]
    )
    changed_material_mask = np.abs(inverse_target - inverse_initial) > 1e-6
    phase_changed_region_ez_l2 = np.asarray(
        [
            float(np.linalg.norm(snapshot[changed_material_mask]))
            for snapshot in phase_snapshot_ez
        ]
    )
    if phase_reciprocal_relative_error[0] > 2e-5:
        raise RuntimeError("phase_in_material violated reciprocal-permittivity interpolation")
    if phase_endpoint_relative_error[0] > 2e-5:
        raise RuntimeError("phase_in_material did not reach the target material")
    if np.any(np.diff(phase_progress) < -1e-7) or phase_progress[-1] <= 0:
        raise RuntimeError("phase_in_material did not progress monotonically")
    if phase_snapshot_ez.shape != (5, 60, 60) or phase_snapshot_dz.shape != (5, 60, 60):
        raise RuntimeError("phase_in_material field snapshot topology changed")
    if phase_changed_region_ez_l2[-1] <= 1e-3:
        raise RuntimeError(
            "phase_in_material did not excite the changing material region: "
            f"{phase_changed_region_ez_l2.tolist()}"
        )

    # A static-initial-material negative control uses the same source and
    # elapsed work.  If a CUDA implementation merely updates host epsilon
    # output while leaving its device coefficient frozen, these complete
    # final fields collapse onto the control and the validation fails.
    static_sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry1,
        sources=[
            mp.Source(
                mp.ContinuousSource(0.5, end_time=phase_time),
                component=mp.Ez,
                center=mp.Vector3(-0.5, 0),
            )
        ],
        resolution=resolution,
    )
    static_sim.run(until=phase_time)
    phase_static_initial_ez = np.asarray(
        static_sim.get_array(component=mp.Ez), dtype=float
    ).copy()
    phase_static_initial_dz = np.asarray(
        static_sim.get_array(component=mp.Dz), dtype=float
    ).copy()
    dynamic_scale = max(float(np.linalg.norm(phase_snapshot_ez[-1])), 1e-30)
    phase_dynamic_static_relative_difference = np.asarray(
        [
            float(
                np.linalg.norm(phase_snapshot_ez[-1] - phase_static_initial_ez)
                / dynamic_scale
            )
        ]
    )
    if phase_dynamic_static_relative_difference[0] < 0.05:
        raise RuntimeError(
            "phase_in_material fields are indistinguishable from the static-material control"
        )
else:
    phase_reciprocal_relative_error = np.asarray([], dtype=float)
    phase_endpoint_relative_error = np.asarray([], dtype=float)
    phase_progress = np.asarray([], dtype=float)
    phase_changed_region_ez_l2 = np.asarray([], dtype=float)
    phase_static_initial_ez = np.asarray([], dtype=float)
    phase_static_initial_dz = np.asarray([], dtype=float)
    phase_dynamic_static_relative_difference = np.asarray([], dtype=float)
