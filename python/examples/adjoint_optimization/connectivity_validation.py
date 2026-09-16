"""Headless fixed-point validation for ``07-Connectivity-Constraint.ipynb``."""

import json

from autograd import grad
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import meep as mp
import meep.adjoint as mpa
import numpy as np


mp.verbosity(0)


def connectivity_value_and_gradient(
    values: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    conductivity: float,
    threshold: float,
) -> tuple[float, np.ndarray]:
    _, value, gradient = mpa.constraint_connectivity(
        values,
        nx,
        ny,
        nz,
        p=3,
        cond_s=conductivity,
        thresh=threshold,
    )
    return float(value), np.asarray(gradient, dtype=np.float64).reshape(-1)


def phase_one_metrics() -> dict:
    nz, ny, nx = 100, 1, 50
    disconnected = np.zeros((nz, ny, nx))
    disconnected[30:65, :, 20:30] = 1
    disconnected[70:, :, 20:30] = 1
    connected = disconnected.copy()
    connected[30:, :, 40] = 1
    connected[40, :, 30:40] = 1
    step = 1e-7

    result = {}
    for name, design in (
        ("disconnected", disconnected),
        ("connected", connected),
    ):
        value, gradient = connectivity_value_and_gradient(
            design, nx, ny, nz, 1e4, 50
        )
        # Each Phase-I structure has a very different gradient scale.  Its
        # normalized gradient provides a well-conditioned directional oracle
        # for both cases without an arbitrary direction cancelling the signal.
        direction = gradient / np.linalg.norm(gradient)
        perturbed_value, _ = connectivity_value_and_gradient(
            design + step * direction.reshape(design.shape),
            nx,
            ny,
            nz,
            1e4,
            50,
        )
        adjoint_directional = float(
            step * np.dot(gradient, direction)
        )
        finite_difference = perturbed_value - value
        np.testing.assert_allclose(
            adjoint_directional,
            finite_difference,
            rtol=5e-4,
            atol=1e-10,
            err_msg=(
                f"Phase-I {name} connectivity gradient disagrees with "
                "its independent forward direction"
            ),
        )
        result[name] = {
            "adjoint_directional_derivative": adjoint_directional,
            "finite_difference_directional_derivative": finite_difference,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "value": value,
            "value_perturbed": perturbed_value,
        }
    return result


def integrated_metrics() -> dict:
    pad = 0.1
    design_width0 = 8.0
    design_height0 = 2.0
    design_width = design_width0 + 2 * pad
    design_height = design_height0 + 2 * pad
    pml_size = 1.0
    resolution = 30
    fcen = 1 / 1.55
    eta_i = 0.5
    eta_e = 0.75
    eta_d = 1 - eta_e
    beta = 16.0
    minimum_length = 0.15
    filter_radius = mpa.get_conic_radius_from_eta_e(
        minimum_length, eta_e
    )
    # ``conic_filter`` and ``MaterialGrid`` use a non-periodic design grid
    # which includes both endpoints of the design-region interval.
    nx = int(round(resolution * design_width)) + 1
    ny = int(round(resolution * design_height)) + 1

    x_grid = np.linspace(-design_width / 2, design_width / 2, nx)
    y_grid = np.linspace(-design_height / 2, design_height / 2, ny)
    x_mesh, y_mesh = np.meshgrid(
        x_grid, y_grid, sparse=True, indexing="ij"
    )
    left_mask = (x_mesh <= -(design_width / 2 - pad)) & (
        np.abs(y_mesh) <= design_height / 2
    )
    right_mask = (x_mesh >= design_width / 2 - pad) & (
        np.abs(y_mesh) <= design_height / 2
    )
    top_mask = (y_mesh >= design_height / 2 - pad) & (
        np.abs(x_mesh) <= design_width / 2
    )
    bottom_mask = (y_mesh <= -(design_height / 2 - pad)) & (
        np.abs(x_mesh) <= design_width / 2
    )
    air_mask = left_mask | top_mask | right_mask
    fixed_mask = (bottom_mask | air_mask).flatten()

    def filtered(values):
        return mpa.conic_filter(
            values,
            filter_radius,
            design_width,
            design_height,
            resolution,
        )

    def mapping(values):
        masked = npa.where(
            bottom_mask.flatten(),
            1,
            npa.where(air_mask.flatten(), 0, values),
        )
        return filtered(npa.reshape(masked, (nx, ny))).flatten()

    # The notebook describes the connectivity preprocessing as
    # mask -> filter -> projection -> rotation.  Its published ``mapping_s``
    # and ``mapping_v`` cells construct the masked array but accidentally pass
    # the unmasked input to ``conic_filter``; use the described (and intended)
    # fixed-boundary semantics here.
    def mapping_solid(values):
        masked = npa.where(
            bottom_mask.flatten(),
            1,
            npa.where(air_mask.flatten(), 0, values),
        )
        projected = mpa.tanh_projection(
            filtered(npa.reshape(masked, (nx, ny))), beta, eta_i
        )
        return npa.rot90(projected).flatten()

    def mapping_void(values):
        masked = npa.where(
            bottom_mask.flatten(),
            1,
            npa.where(air_mask.flatten(), 0, values),
        )
        projected = mpa.tanh_projection(
            filtered(npa.reshape(masked, (nx, ny))), beta, eta_i
        )
        return (1 - npa.rot90(projected, 3)).flatten()

    silicon = mp.Medium(index=3.4)
    air = mp.Medium(index=1.0)
    design_variables = mp.MaterialGrid(
        mp.Vector3(nx, ny),
        air,
        silicon,
        # Match the notebook's connectivity-enabled continuation stage.  The
        # Maxwell mapping intentionally stops after the conic filter because
        # MaterialGrid applies this projection and its adjoint derivative.
        beta=beta,
        eta=eta_i,
        damping=0.2 * 2 * np.pi * fcen,
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(design_width, design_height),
        ),
    )
    source = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=0.1 * fcen),
            component=mp.Ez,
            center=mp.Vector3(0, -(design_height / 2 + 1)),
            size=mp.Vector3(design_width, 0),
        )
    ]
    simulation = mp.Simulation(
        cell_size=mp.Vector3(
            2 * pml_size + design_width,
            2 * pml_size + design_height + 3,
        ),
        boundary_layers=[mp.PML(pml_size)],
        geometry=[
            mp.Block(
                center=design_region.center,
                size=design_region.size,
                material=design_variables,
            ),
            mp.Block(
                center=mp.Vector3(0, -design_height0 / 2 - 0.25),
                size=mp.Vector3(design_width, 0.5),
                material=silicon,
            ),
        ],
        sources=source,
        default_material=air,
        resolution=resolution,
    )
    far_fields = mpa.Near2FarFields(
        simulation,
        [
            mp.Near2FarRegion(
                center=mp.Vector3(0, design_height / 2 + 0.5),
                size=mp.Vector3(design_width, 0),
                weight=1,
            )
        ],
        [mp.Vector3(0, 5)],
    )

    def objective(fields):
        return npa.abs(fields[0, 0, 2]) ** 2

    problem = mpa.OptimizationProblem(
        simulation=simulation,
        objective_functions=[objective],
        objective_arguments=[far_fields],
        design_regions=[design_region],
        frequencies=[fcen],
        maximum_run_time=500,
    )

    raw_design = np.full(nx * ny, 0.5, dtype=np.float64)
    value, density_gradient = problem([mapping(raw_design)])
    value = float(np.asarray(value).reshape(-1)[0])
    objective_gradient = np.asarray(
        tensor_jacobian_product(mapping, 0)(
            raw_design,
            density_gradient,
        ),
        dtype=np.float64,
    ).reshape(-1)

    solid_mapped = mapping_solid(raw_design)
    solid_value, solid_density_gradient = connectivity_value_and_gradient(
        solid_mapped, nx, 1, ny, 500, 30
    )
    solid_gradient = np.asarray(
        tensor_jacobian_product(mapping_solid, 0)(
            raw_design, solid_density_gradient
        ),
        dtype=np.float64,
    ).reshape(-1)

    void_mapped = mapping_void(raw_design)
    void_value, void_density_gradient = connectivity_value_and_gradient(
        void_mapped, nx, 1, ny, 500, 30
    )
    void_gradient = np.asarray(
        tensor_jacobian_product(mapping_void, 0)(
            raw_design, void_density_gradient
        ),
        dtype=np.float64,
    ).reshape(-1)

    threshold = lambda values: mpa.tanh_projection(
        values, beta, eta_i
    )
    filter_function = lambda values: filtered(
        values.reshape(nx, ny)
    )
    masked_design = npa.where(
        bottom_mask.flatten(),
        1,
        npa.where(air_mask.flatten(), 0, raw_design),
    )
    coefficient = (filter_radius / resolution) ** 4
    solid_geometry_value = float(
        mpa.constraint_solid(
            masked_design,
            coefficient,
            eta_e,
            filter_function,
            threshold,
            1,
        )
    )
    void_geometry_value = float(
        mpa.constraint_void(
            masked_design,
            coefficient,
            eta_d,
            filter_function,
            threshold,
            1,
        )
    )
    solid_geometry_gradient = np.asarray(
        grad(mpa.constraint_solid, 0)(
            masked_design,
            coefficient,
            eta_e,
            filter_function,
            threshold,
            1,
        ),
        dtype=np.float64,
    ).reshape(-1)
    void_geometry_gradient = np.asarray(
        grad(mpa.constraint_void, 0)(
            masked_design,
            coefficient,
            eta_d,
            filter_function,
            threshold,
            1,
        ),
        dtype=np.float64,
    ).reshape(-1)
    solid_geometry_gradient[fixed_mask] = 0
    void_geometry_gradient[fixed_mask] = 0

    step = 1e-5
    def gradient_direction(gradient: np.ndarray, name: str) -> np.ndarray:
        direction = np.asarray(gradient, dtype=np.float64).reshape(-1)
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm == 0:
            raise RuntimeError(f"{name} gradient is not finite and nonzero")
        return direction / direction_norm

    objective_direction = gradient_direction(
        objective_gradient, "objective"
    )
    perturbed_design = raw_design + step * objective_direction
    perturbed_value, _ = problem(
        [mapping(perturbed_design)],
        need_gradient=False,
    )
    perturbed_value = float(
        np.asarray(perturbed_value).reshape(-1)[0]
    )
    solid_direction = gradient_direction(
        solid_gradient, "solid connectivity"
    )
    perturbed_solid_value, _ = connectivity_value_and_gradient(
        mapping_solid(raw_design + step * solid_direction),
        nx,
        1,
        ny,
        500,
        30,
    )
    void_direction = gradient_direction(
        void_gradient, "void connectivity"
    )
    perturbed_void_value, _ = connectivity_value_and_gradient(
        mapping_void(raw_design + step * void_direction),
        nx,
        1,
        ny,
        500,
        30,
    )

    def geometry_values(values):
        masked = npa.where(
            bottom_mask.flatten(),
            1,
            npa.where(air_mask.flatten(), 0, values),
        )
        return (
            float(
                mpa.constraint_solid(
                    masked,
                    coefficient,
                    eta_e,
                    filter_function,
                    threshold,
                    1,
                )
            ),
            float(
                mpa.constraint_void(
                    masked,
                    coefficient,
                    eta_d,
                    filter_function,
                    threshold,
                    1,
                )
            ),
        )

    solid_geometry_direction = gradient_direction(
        solid_geometry_gradient, "solid geometry"
    )
    perturbed_solid_geometry, _ = geometry_values(
        raw_design + step * solid_geometry_direction
    )
    void_geometry_direction = gradient_direction(
        void_geometry_gradient, "void geometry"
    )
    _, perturbed_void_geometry = geometry_values(
        raw_design + step * void_geometry_direction
    )

    def metric(
        baseline: float,
        perturbed: float,
        gradient: np.ndarray,
        direction: np.ndarray,
        name: str,
        rtol: float,
        atol: float,
    ) -> dict:
        adjoint_directional = float(
            step * np.dot(gradient, direction)
        )
        finite_difference = perturbed - baseline
        np.testing.assert_allclose(
            adjoint_directional,
            finite_difference,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"integrated {name} gradient disagrees with its "
                "independent forward direction"
            ),
        )
        return {
            "adjoint_directional_derivative": adjoint_directional,
            "finite_difference_directional_derivative": finite_difference,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "value": baseline,
            "value_perturbed": perturbed,
        }

    return {
        "material_grid": {
            "beta": float(design_variables.beta),
            "eta": float(design_variables.eta),
        },
        "objective": metric(
            value,
            perturbed_value,
            objective_gradient,
            objective_direction,
            "objective",
            0.03,
            1e-8,
        ),
        "solid_connectivity": metric(
            solid_value,
            perturbed_solid_value,
            solid_gradient,
            solid_direction,
            "solid connectivity",
            3e-3,
            1e-10,
        ),
        "void_connectivity": metric(
            void_value,
            perturbed_void_value,
            void_gradient,
            void_direction,
            "void connectivity",
            3e-3,
            1e-10,
        ),
        "solid_geometry": metric(
            solid_geometry_value,
            perturbed_solid_geometry,
            solid_geometry_gradient,
            solid_geometry_direction,
            "solid geometry",
            3e-3,
            1e-11,
        ),
        "void_geometry": metric(
            void_geometry_value,
            perturbed_void_geometry,
            void_geometry_gradient,
            void_geometry_direction,
            "void geometry",
            3e-3,
            1e-11,
        ),
    }


def main() -> None:
    metrics = {
        "phase_one": phase_one_metrics(),
        "integrated": integrated_metrics(),
    }
    print("gpmeep-example-metrics:" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
