"""Deterministic 3D complex, multifrequency adjoint Near2Far qualification.

This example validates a physical three-dimensional MaterialGrid gradient
against an independent forward perturbation.  Its two arbitrary far points
are evaluated as one batch so a strict CUDA run must exercise the resident
3D Near2Far transform rather than repeated CPU Green-function calls.
"""

import json

from autograd import numpy as npa
import meep as mp
import meep.adjoint as mpa
import numpy as np


mp.verbosity(0)

RESOLUTION = 8
SOURCE_FREQUENCY = 0.85
FREQUENCIES = [0.75, 0.95]
CELL_SIZE = mp.Vector3(3.0, 3.0, 3.0)
DESIGN_SIZE = mp.Vector3(1.0, 1.0, 1.0)
DESIGN_SHAPE = (5, 5, 5)
RUN_TIME = 18.0
FINITE_DIFFERENCE_STEPS = (2.0e-3, 1.0e-3)
MAX_DERIVATIVE_ERROR = 0.08


def near2far_box(half_width: float) -> list[mp.Near2FarRegion]:
    diameter = 2 * half_width
    return [
        mp.Near2FarRegion(
            center=mp.Vector3(+half_width, 0, 0),
            size=mp.Vector3(0, diameter, diameter),
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-half_width, 0, 0),
            size=mp.Vector3(0, diameter, diameter),
            weight=-1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, +half_width, 0),
            size=mp.Vector3(diameter, 0, diameter),
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -half_width, 0),
            size=mp.Vector3(diameter, 0, diameter),
            weight=-1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0, +half_width),
            size=mp.Vector3(diameter, diameter, 0),
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0, -half_width),
            size=mp.Vector3(diameter, diameter, 0),
            weight=-1,
        ),
    ]


def main() -> None:
    low = mp.Medium(index=1.0)
    high = mp.Medium(index=2.1)
    background_weight = 0.5
    background_epsilon = (1.0 - background_weight) * 1.0**2 + (
        background_weight * 2.1**2
    )
    background = mp.Medium(epsilon=background_epsilon)
    weights = np.full(np.prod(DESIGN_SHAPE), background_weight)
    center_parameter = np.ravel_multi_index((2, 2, 2), DESIGN_SHAPE)
    weights[center_parameter] = 0.58
    material_grid = mp.MaterialGrid(
        mp.Vector3(*DESIGN_SHAPE),
        low,
        high,
        weights=weights.reshape(DESIGN_SHAPE),
        do_averaging=False,
    )
    design_region = mpa.DesignRegion(
        material_grid,
        volume=mp.Volume(center=mp.Vector3(), size=DESIGN_SIZE),
    )
    simulation = mp.Simulation(
        cell_size=CELL_SIZE,
        boundary_layers=[mp.PML(0.5)],
        geometry=[
            mp.Block(
                center=design_region.center,
                size=design_region.size,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(SOURCE_FREQUENCY, fwidth=0.45),
                component=mp.Ez,
                center=mp.Vector3(0.09, -0.07, -0.65),
            )
        ],
        default_material=background,
        resolution=RESOLUTION,
        # Exercise the complex-field storage/adjoint path independently of a
        # nonzero Bloch wave vector, so the physical configuration remains
        # directly comparable between CPU and strict CUDA lanes.
        force_complex_fields=True,
    )
    far_points = [
        mp.Vector3(2.7, 1.4, 1.8),
        mp.Vector3(-1.9, 2.5, 1.6),
    ]
    far_fields = mpa.Near2FarFields(
        simulation,
        near2far_box(0.75),
        far_points,
        decimation_factor=1,
    )

    def objective(fields):
        # Both targets, frequencies, and complex field quadratures contribute.
        first_target = npa.sum(fields[0, :, 2])
        second_target = npa.sum(fields[1, :, 2])
        return npa.real((1.0 + 0.05j) * first_target - 0.05 * second_target)

    problem = mpa.OptimizationProblem(
        simulation=simulation,
        objective_functions=objective,
        objective_arguments=[far_fields],
        design_regions=[design_region],
        frequencies=FREQUENCIES,
        decay_by=0.0,
        minimum_run_time=RUN_TIME,
        maximum_run_time=RUN_TIME,
    )

    objective_value, gradient = problem([weights], need_gradient=True)
    objective_value = float(np.asarray(objective_value).reshape(-1)[0])
    gradient_by_frequency = np.asarray(gradient, dtype=np.float64)
    if len(FREQUENCIES) == 1:
        gradient_by_frequency = gradient_by_frequency.reshape(-1, 1)
    expected_gradient_shape = (weights.size, len(FREQUENCIES))
    if gradient_by_frequency.shape != expected_gradient_shape:
        raise RuntimeError(
            "3D Near2Far multifrequency gradient layout changed: "
            f"got {gradient_by_frequency.shape}, "
            f"expected {expected_gradient_shape}"
        )
    gradient = np.sum(gradient_by_frequency, axis=1)
    gradient_l2 = float(np.linalg.norm(gradient))
    adjoint_source_scale_without_volume = complex(
        np.asarray(far_fields._adj_src_scale(False)).reshape(-1)[0]
    )
    adjoint_source_scale_with_volume = complex(
        np.asarray(far_fields._adj_src_scale(True)).reshape(-1)[0]
    )
    if not np.isfinite(objective_value) or objective_value == 0:
        raise RuntimeError("3D Near2Far objective is invalid")
    if not np.isfinite(gradient_l2) or gradient_l2 <= 0:
        raise RuntimeError("3D Near2Far adjoint gradient is non-finite or zero")

    center_direction = np.zeros_like(gradient)
    center_direction[center_parameter] = 1.0
    dense_direction = np.sin(np.arange(weights.size, dtype=np.float64) + 0.375)
    dense_direction /= np.linalg.norm(dense_direction)
    directions = (
        ("center_parameter", center_direction),
        ("dense_non_axis", dense_direction),
    )
    derivative_checks = []
    for direction_index, (direction_name, direction) in enumerate(directions):
        directional_derivative = float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative == 0:
            raise RuntimeError(
                f"3D Near2Far {direction_name} adjoint derivative is invalid"
            )
        for step in FINITE_DIFFERENCE_STEPS:
            plus_weights = weights + step * direction
            minus_weights = weights - step * direction
            if (
                np.any(plus_weights < 0)
                or np.any(plus_weights > 1)
                or np.any(minus_weights < 0)
                or np.any(minus_weights > 1)
            ):
                raise RuntimeError("finite-difference direction left the MaterialGrid domain")
            plus_value, _ = problem([plus_weights], need_gradient=False)
            minus_value, _ = problem([minus_weights], need_gradient=False)
            plus_value = float(np.asarray(plus_value).reshape(-1)[0])
            minus_value = float(np.asarray(minus_value).reshape(-1)[0])
            finite_difference_derivative = (plus_value - minus_value) / (2.0 * step)
            derivative_scale = max(
                abs(directional_derivative),
                abs(finite_difference_derivative),
                1.0e-14,
            )
            relative_derivative_error = abs(
                directional_derivative - finite_difference_derivative
            ) / derivative_scale
            check = {
                "direction_index": direction_index,
                "step": step,
                "adjoint_derivative": directional_derivative,
                "adjoint_derivative_by_frequency": [
                    float(np.dot(gradient_by_frequency[:, frequency_index], direction))
                    for frequency_index in range(len(FREQUENCIES))
                ],
                "finite_difference_derivative": finite_difference_derivative,
                "relative_derivative_error": relative_derivative_error,
                "objective_plus": plus_value,
                "objective_minus": minus_value,
            }
            derivative_checks.append(check)
            if relative_derivative_error > MAX_DERIVATIVE_ERROR:
                raise RuntimeError(
                    "3D Near2Far adjoint directional derivative disagrees "
                    f"with central difference: {check}, "
                    f"objective={objective_value:.12g}, "
                    f"gradient_l2={gradient_l2:.12g}"
                )

    metrics = {
        "schema_version": 1,
        "resolution": RESOLUTION,
        "design_parameter_count": int(weights.size),
        "complex_fields_enabled": 1,
        "frequency_count": len(FREQUENCIES),
        "frequencies": list(FREQUENCIES),
        "far_point_count": len(far_points),
        "objective": objective_value,
        "gradient_l2": gradient_l2,
        "gradient_by_frequency_l2": [
            float(np.linalg.norm(gradient_by_frequency[:, frequency_index]))
            for frequency_index in range(len(FREQUENCIES))
        ],
        "center_parameter_gradient": float(gradient[center_parameter]),
        "adjoint_source_scale_without_volume_abs": abs(
            adjoint_source_scale_without_volume
        ),
        "adjoint_source_scale_with_volume_abs": abs(
            adjoint_source_scale_with_volume
        ),
        "direction_count": len(directions),
        "finite_difference_steps": list(FINITE_DIFFERENCE_STEPS),
        "derivative_checks": derivative_checks,
        "maximum_relative_derivative_error": max(
            check["relative_derivative_error"] for check in derivative_checks
        ),
    }
    print("gpmeep-near2far-3d-adjoint-metrics:" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
