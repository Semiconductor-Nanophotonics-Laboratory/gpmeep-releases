"""Deterministic validation slice for ``06-Near2Far-Epigraph.ipynb``.

The notebook's long NLopt trajectory is intentionally replaced by one fixed
design evaluation and independent directional finite differences. The
simulation geometry, three-frequency vector objective, mapping, Cartesian
near-to-far adjoint monitor, and full ``f_i - t`` constraint Jacobian are
retained.
"""

import json

from autograd import numpy as npa
from autograd import tensor_jacobian_product
import meep as mp
import meep.adjoint as mpa
import numpy as np


mp.verbosity(0)

DESIGN_WIDTH = 15.0
DESIGN_HEIGHT = 2.0
PML_SIZE = 1.0
RESOLUTION = 30
FREQUENCIES = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
ETA = 0.5
BETA = 4.0
MINIMUM_LENGTH = 0.09


def main() -> None:
    silicon = mp.Medium(index=3.4)
    air = mp.Medium(index=1.0)
    filter_radius = mpa.get_conic_radius_from_eta_e(
        MINIMUM_LENGTH, 0.55
    )
    design_resolution = RESOLUTION
    nx = int(design_resolution * DESIGN_WIDTH) + 1
    ny = int(design_resolution * DESIGN_HEIGHT) + 1

    design_variables = mp.MaterialGrid(
        mp.Vector3(nx, ny),
        air,
        silicon,
        grid_type="U_MEAN",
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(DESIGN_WIDTH, DESIGN_HEIGHT),
        ),
    )

    def mapping(x):
        filtered = mpa.conic_filter(
            x,
            filter_radius,
            DESIGN_WIDTH,
            DESIGN_HEIGHT,
            design_resolution,
        )
        projected = mpa.tanh_projection(filtered, BETA, ETA)
        return (0.5 * (npa.flipud(projected) + projected)).flatten()

    fcen = 1 / 1.55
    fwidth = 0.2 * fcen
    source = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            component=mp.Ez,
            center=mp.Vector3(0, -(DESIGN_HEIGHT / 2 + 1.5)),
            size=mp.Vector3(DESIGN_WIDTH, 0),
        )
    ]
    simulation = mp.Simulation(
        cell_size=mp.Vector3(
            2 * PML_SIZE + DESIGN_WIDTH,
            2 * PML_SIZE + DESIGN_HEIGHT + 5,
        ),
        boundary_layers=[mp.PML(PML_SIZE)],
        geometry=[
            mp.Block(
                center=design_region.center,
                size=design_region.size,
                material=design_variables,
            )
        ],
        sources=source,
        default_material=air,
        symmetries=[mp.Mirror(direction=mp.X)],
        resolution=RESOLUTION,
    )
    far_fields = mpa.Near2FarFields(
        simulation,
        [
            mp.Near2FarRegion(
                center=mp.Vector3(0, DESIGN_HEIGHT / 2 + 1.5),
                size=mp.Vector3(DESIGN_WIDTH, 0),
                weight=1,
            )
        ],
        [mp.Vector3(0, 15)],
    )

    def objective(fields):
        return -npa.abs(fields[0, :, 2]) ** 2

    problem = mpa.OptimizationProblem(
        simulation=simulation,
        objective_functions=[objective],
        objective_arguments=[far_fields],
        design_regions=[design_region],
        frequencies=FREQUENCIES,
        maximum_run_time=2000,
    )

    def epigraph_constraint(result, variables, jacobian):
        epigraph = variables[0]
        raw_weights = variables[1:]
        need_gradient = jacobian.size > 0
        objective_value, density_gradient = problem(
            [mapping(raw_weights)],
            need_gradient=need_gradient,
        )
        objective_value = np.asarray(
            objective_value, dtype=np.float64
        ).reshape(-1)
        result[:] = objective_value - epigraph
        if need_gradient:
            density_gradient = np.asarray(
                density_gradient, dtype=np.float64
            )
            if density_gradient.ndim == 1:
                density_gradient = density_gradient[:, None]
            jacobian[:, 0] = -1
            for frequency in range(FREQUENCIES.size):
                jacobian[frequency, 1:] = tensor_jacobian_product(
                    mapping, 0
                )(
                    raw_weights,
                    density_gradient[:, frequency],
                )

    raw_design = np.full(nx * ny, 0.5, dtype=np.float64)
    epigraph = 0.25
    epigraph_variables = np.concatenate(([epigraph], raw_design))
    constraint_value = np.empty(FREQUENCIES.size)
    constraint_jacobian = np.empty(
        (FREQUENCIES.size, raw_design.size + 1)
    )
    epigraph_constraint(
        constraint_value,
        epigraph_variables,
        constraint_jacobian,
    )
    np.testing.assert_array_equal(
        constraint_jacobian[:, 0],
        -np.ones(FREQUENCIES.size),
        err_msg="epigraph Jacobian has an invalid dummy-variable column",
    )
    value = constraint_value + epigraph
    mapped_gradient = constraint_jacobian[:, 1:].T

    # Each frequency has a distinct far-field sensitivity. A single
    # oscillatory or uniform direction can almost cancel one or more of those
    # gradients and leave an FP32-sized difference. Use one normalized
    # steepest direction per frequency and independently rerun the forward
    # solve. The finite difference still comes solely from perturbed Maxwell
    # solves, while every objective receives a well-conditioned ~1e-3 signal.
    gradient_l2 = np.linalg.norm(mapped_gradient, axis=0)
    if np.any(gradient_l2 == 0):
        raise RuntimeError("Near2Far validation produced a zero gradient")
    directions = mapped_gradient / gradient_l2[None, :]
    step = 1e-3
    perturbed_value = np.empty(FREQUENCIES.size, dtype=np.float64)
    for frequency in range(FREQUENCIES.size):
        perturbed_constraint = np.empty(FREQUENCIES.size)
        epigraph_constraint(
            perturbed_constraint,
            np.concatenate(
                (
                    [epigraph],
                    raw_design + step * directions[:, frequency],
                )
            ),
            np.empty(0),
        )
        perturbed_value[frequency] = (
            perturbed_constraint[frequency] + epigraph
        )
    adjoint_directional = (
        step * np.einsum("if,if->f", directions, mapped_gradient)
    )
    finite_difference = perturbed_value - value
    np.testing.assert_allclose(
        adjoint_directional,
        finite_difference,
        rtol=0.03,
        atol=1e-6,
        err_msg="Near2Far mapped gradient disagrees with forward differences",
    )

    metrics = {
        "adjoint_directional_derivative": adjoint_directional.tolist(),
        "finite_difference_directional_derivative": finite_difference.tolist(),
        "epigraph_constraint": constraint_value.tolist(),
        "epigraph_jacobian_dummy_column": (
            constraint_jacobian[:, 0].tolist()
        ),
        "gradient_l2": gradient_l2.tolist(),
        "objective": value.tolist(),
        "objective_perturbed": perturbed_value.tolist(),
    }
    print("gpmeep-example-metrics:" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
