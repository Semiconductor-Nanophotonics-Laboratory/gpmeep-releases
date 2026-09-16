"""waveguide_crossing.py - Using meep's adjoint solver, designs a waveguide
crossing that maximizes transmission at a single frequency.

Two approaches are demonstrated: (1) start with the nominal crossing shape and
perform shape optimization via meep's smoothed projection feature; (2) run a
full topology optimization starting from an initial grayscale. Evolve the design
until β=∞.

Importantly, this particular example highlights some of the ways one can use the
novel smoothed projection function to perform both shape and topology
optimization.
"""

import argparse
import json
from pathlib import Path
import tempfile
from typing import Callable, List, Optional, Tuple

import meep.adjoint as mpa
import nlopt
import numpy as np
from autograd import grad
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from matplotlib import pyplot as plt

import meep as mp

mp.quiet()

DEFAULT_MIN_LENGTH = 0.09
DEFAULT_DESIGN_REGION_WIDTH = 3.0
DEFAULT_DESIGN_REGION_HEIGHT = 3.0
DEFAULT_WAVEGUIDE_WIDTH = 0.5
DEFAULT_ETA = 0.5
DEFAULT_ETA_E = 0.75
DEFAULT_MAX_EVAL = 30


def build_optimization_problem(
    resolution: float,
    beta: float,
    use_smoothed_projection: bool,
    min_length: float = DEFAULT_MIN_LENGTH,
    dx: float = DEFAULT_DESIGN_REGION_WIDTH,
    dy: float = DEFAULT_DESIGN_REGION_HEIGHT,
    waveguide_width: float = DEFAULT_WAVEGUIDE_WIDTH,
    eta: float = DEFAULT_ETA,
    eta_e: float = DEFAULT_ETA_E,
    damping_factor: float = 0.0,
) -> Tuple[mpa.OptimizationProblem, Callable]:
    """Build the waveguide-crossing optimization problem.

    The waveguide crossing is a cononical inverse design problem with both
    shape- and topology-optimization implementations. The idea is to find the
    optimal structure that maximizes transmission from one side to the other. It
    exhibits C4 symmetry, and generally resembles the following structure:

         |  |
         |  |
    -----    ------
    -----    ------
         |  |
         |  |

    Args:
        resolution: Simulation resolution in pixels/micron.
        beta: Tanh function projection strength parameter, ranging from [0,∞].
        use_smoothed_projection: Whether or not to use the smoothed projection.
        min_length: Minimum length scale in microns.
        dx: Design region width in microns.
        dy: Design region height in microns.
        waveguide_width: Waveguide width in microns.
        eta: Projection function threshold parameter.
        eta_e: Projection function eroded threshold parameter.
        damping_factor: The material grid damping scalar factor.

    Returns:
        The corresponding optimization problem object and the mapping function
        that applies the linear and nonlinear transformations.
    """
    # Map the design region resolution to the yee grid, which is twice the standard resolution.
    design_region_resolution = int(2 * resolution)

    # pml thickness
    dpml = 1.0

    filter_radius = mpa.get_conic_radius_from_eta_e(min_length, eta_e)

    sxy = dx + 1 + 2 * dpml

    silicon = mp.Medium(epsilon=12)
    cell_size = mp.Vector3(sxy, sxy, 0)
    boundary_layers = [mp.PML(thickness=dpml)]

    eig_parity = mp.EVEN_Y + mp.ODD_Z

    design_region_size = mp.Vector3(dx, dy)
    Nx = int(design_region_resolution * design_region_size.x) + 1
    Ny = int(design_region_resolution * design_region_size.y) + 1

    waveguide_geometry = [
        mp.Block(material=silicon, size=mp.Vector3(mp.inf, waveguide_width, mp.inf)),
        mp.Block(material=silicon, size=mp.Vector3(waveguide_width, mp.inf, mp.inf)),
    ]

    # Source centered in optical c-band
    fcen = 1 / 1.55
    df = 0.23 * fcen
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(-0.5 * sxy + dpml + 0.1, 0),
            size=mp.Vector3(0, sxy - 2 * dpml),
            eig_band=1,
            eig_parity=eig_parity,
        )
    ]

    damping = damping_factor * fcen
    matgrid = mp.MaterialGrid(
        mp.Vector3(Nx, Ny),
        mp.air,
        silicon,
        weights=np.ones((Nx, Ny)),
        beta=0,  # disable meep's internal smoothing
        do_averaging=False,  # disable meep's internal mg smoothing
        damping=damping,
    )

    matgrid_region = mpa.DesignRegion(
        matgrid,
        volume=mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(design_region_size.x, design_region_size.y, 0),
        ),
    )

    matgrid_geometry = [
        mp.Block(
            center=matgrid_region.center, size=matgrid_region.size, material=matgrid
        )
    ]

    geometry = waveguide_geometry + matgrid_geometry

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
    )

    frequencies = [fcen]

    obj_list = [
        mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=mp.Vector3(-0.5 * sxy + dpml + 0.2),
                size=mp.Vector3(0, sxy - 2 * dpml, 0),
            ),
            1,
            eig_parity=eig_parity,
        ),
        mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=mp.Vector3(0.5 * sxy - dpml - 0.2),
                size=mp.Vector3(0, sxy - 2 * dpml, 0),
            ),
            1,
            eig_parity=eig_parity,
        ),
    ]

    def J(input, output):
        """Simple objective function to minimize loss."""
        return 1 - npa.power(npa.abs(output / input), 2)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        maximum_run_time=500,
        objective_functions=J,
        objective_arguments=obj_list,
        design_regions=[matgrid_region],
        frequencies=frequencies,
    )

    def mapping(x: npa.ndarray):
        """Applies the smoothing and projection."""
        x = x.reshape(Nx, Ny)
        x = mpa.conic_filter(
            x,
            filter_radius,
            design_region_size.x,
            design_region_size.y,
            design_region_resolution,
        )

        # Enforce the square symmetry
        x = (x + npa.rot90(x) + npa.rot90(x, 2) + npa.rot90(x, 3)) / 4

        # Only used the smoothed projection if prompted.
        if use_smoothed_projection:
            x = mpa.smoothed_projection(
                x, beta=beta, eta=eta, resolution=design_region_resolution
            )
        else:
            x = mpa.tanh_projection(x, beta=beta, eta=eta)

        return x.flatten()

    return opt, mapping


def nlopt_fom(
    x: np.ndarray,
    gradient: np.ndarray,
    opt: mpa.OptimizationProblem,
    mapping: Callable,
    data: List,
    results: List,
):
    """Wrapper for NLopt FOM.

    Args:
        x: Degrees of freedom array.
        gradient: Gradient of FOM.
        opt: Optimization problem object.
        mapping: Mapping function.
        data: Structure to store the simulated design each iteration.
        results: Structure to store the simulated FOM each iteration.

    Returns:
        The FOM value at the current iteration.
    """
    grid_size = opt.design_regions[0].design_parameters.grid_size
    Nx, Ny = int(grid_size.x), int(grid_size.y)

    f0, dJ_du = opt([mapping(x)])
    backprop_gradient = tensor_jacobian_product(mapping, 0)(x, dJ_du)
    if gradient.size > 0:
        gradient[:] = backprop_gradient

    data.append(
        [
            np.squeeze(x.copy().reshape(Nx, Ny)),
            np.squeeze(mapping(x).copy().reshape(Nx, Ny)),
            np.squeeze(backprop_gradient.reshape(Nx, Ny)),
        ]
    )

    print(
        f"FOM: {np.real(f0)} | x NaNs:{np.sum(np.isnan(backprop_gradient))} | grad NaNs: {np.sum(np.isnan(backprop_gradient))}"
    )
    results.append(np.real(f0))

    return float(np.asarray(np.real(f0)).reshape(-1)[0])


def _plot_optimization_results(
    data: List,
    results: List,
    num_samples: int = 4,
):
    samples = np.linspace(0, len(results), num_samples, dtype=int, endpoint=False)
    plt.figure(figsize=(5.25, 4.0))
    plt.subplot(2, 1, 1)
    plt.plot(results, "o-")
    plt.xlabel("Optimization Iteration")
    plt.ylabel("FOM")
    for k in range(len(samples)):
        plt.subplot(2, 4, 5 + k)
        plt.imshow(data[samples[k]], cmap="binary", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title(f"It. {samples[k]+1}")
    plt.tight_layout()
    plt.show()


def run_shape_optimization(
    resolution: float,
    beta: float,
    maxeval: int,
    use_smoothed_projection: bool = True,
    damping_factor: float = 0.0,
    dx: float = DEFAULT_DESIGN_REGION_WIDTH,
    dy: float = DEFAULT_DESIGN_REGION_HEIGHT,
    waveguide_width: float = DEFAULT_WAVEGUIDE_WIDTH,
    output_filename_prefix: Optional[str] = None,
    plot_results: bool = True,
):
    """Run shape optimization using a cross as a starting guess.

    Args:
        resolution: Simulation resolution in pixels/micron.
        beta: Tanh function projection strength parameter, ranging from [0,∞].
        maxeval: Maximum number of optimization iterations to run.
        use_smoothed_projection: Whether or not to use the smoothed projection.
        dx: Design region width in microns.
        dy: Design region height in microns.
        waveguide_width: Waveguide width in microns.
        output_filename_prefix: The filename prefix that will store the
            optimization results. If `None`, no file is saved.
        plot_results: Whether or not to plot results.

    Returns:
        The design and FOM result arrays, along with the optimization problem
        and mapping function.

    """
    # Initialize the optimization problem and mapping functions
    opt, mapping = build_optimization_problem(
        resolution=resolution,
        beta=beta,
        dx=dx,
        dy=dy,
        waveguide_width=waveguide_width,
        use_smoothed_projection=use_smoothed_projection,
        damping_factor=damping_factor,
    )

    # pull number of parameters from the design region
    n = opt.design_regions[0].design_parameters.weights.size
    grid_size = opt.design_regions[0].design_parameters.grid_size
    Nx, Ny = int(grid_size.x), int(grid_size.y)

    # Set up the optimizer
    algorithm = nlopt.LD_CCSAQ
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    solver.set_maxeval(maxeval)

    # initial guess, which is just a simple waveguide crossing
    x = np.linspace(-dx / 2, dy / 2, Nx)
    y = np.linspace(-dx / 2, dy / 2, Ny)
    X, Y = np.meshgrid(x, y)
    mask = (np.abs(Y) <= waveguide_width / 2.0) + (np.abs(X) <= waveguide_width / 2.0)
    x0 = np.zeros((Nx, Ny))
    x0[mask.T] = 1
    x0 = mapping(x0)

    # Create empty datastructures we can use to log the results
    data = []
    results = []

    # prepare the optimizer objective function. We need to wrap the above fom in
    # a format that nlopt expects.
    nlopt_fom_simple = lambda x, gradient: nlopt_fom(
        x, gradient, opt=opt, mapping=mapping, data=data, results=results
    )
    solver.set_min_objective(nlopt_fom_simple)

    # Run the optimization
    x_final = solver.optimize(x0)

    # Log the final results
    opt.update_design([x_final.copy(), mapping(x_final)])
    f0, final_grad = opt(need_gradient=False)
    final_backprop_gradient = tensor_jacobian_product(mapping, 0)(x_final, final_grad)
    results.append(np.real(f0))
    data.append(
        [
            np.squeeze(x_final.copy().reshape(Nx, Ny)),
            np.squeeze(mapping(x_final).copy().reshape(Nx, Ny)),
            np.squeeze(final_backprop_gradient.reshape(Nx, Ny)),
        ]
    )

    if plot_results:
        _plot_optimization_results(data, results)

    # Save to disk
    if mp.am_really_master() and (output_filename_prefix is not None):
        np.savez(output_filename_prefix + "_data.npz", data=data, results=results)

    return data, results, opt, mapping


def run_topology_optimization(
    resolution: float,
    beta_evolution: List[float],
    maxeval: int,
    use_smoothed_projection: bool = True,
    dx: float = DEFAULT_DESIGN_REGION_WIDTH,
    dy: float = DEFAULT_DESIGN_REGION_HEIGHT,
    waveguide_width: float = DEFAULT_WAVEGUIDE_WIDTH,
    output_filename_prefix: Optional[str] = None,
    damping_factor: float = 0.0,
):
    """Run shape optimization using a cross as a starting guess.

    Args:
        resolution: Simulation resolution in pixels/micron.
        beta_evolution: List of Tanh function projection strength parameter,
            ranging from [0,∞], for each optimization epoch.
        maxeval: Maximum number of optimization iterations to run.
        use_smoothed_projection: Whether or not to use the smoothed projection.
        dx: Design region width in microns.
        dy: Design region height in microns.
        waveguide_width: Waveguide width in microns.
        output_filename_prefix: The filename prefix that will store the
            optimization results. If `None`, no file is saved.

    Returns:
        The design and FOM result arrays, along with the optimization problem
        and mapping function.

    """
    # Initialize the optimization problem and mapping functions so that we can
    # set up the problem.
    opt, mapping = build_optimization_problem(
        resolution=resolution,
        beta=beta_evolution[0],
        dx=dx,
        dy=dy,
        waveguide_width=waveguide_width,
        use_smoothed_projection=use_smoothed_projection,
        damping_factor=damping_factor,
    )

    # pull number of parameters from the design region
    n = opt.design_regions[0].design_parameters.weights.size
    grid_size = opt.design_regions[0].design_parameters.grid_size
    Nx, Ny = int(grid_size.x), int(grid_size.y)

    # Set up the optimizer
    algorithm = nlopt.LD_CCSAQ
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    solver.set_maxeval(maxeval)

    # initial guess, which is just a uniform gray region
    x0 = 0.5 * np.ones((Nx, Ny))
    x0 = mapping(x0)

    # Create empty datastructures we can use to log the results
    data = []
    results = []

    for beta in beta_evolution:
        # Re-initialize the optimization problem and mapping functions
        opt, mapping = build_optimization_problem(
            resolution=resolution,
            beta=beta,
            dx=dx,
            dy=dy,
            waveguide_width=waveguide_width,
            use_smoothed_projection=use_smoothed_projection,
            damping_factor=damping_factor,
        )

        # prepare the optimizer objective function. We need to wrap the above fom in
        # a format that nlopt expects.
        nlopt_fom_simple = lambda x, gradient: nlopt_fom(
            x, gradient, opt=opt, mapping=mapping, data=data, results=results
        )
        solver.set_min_objective(nlopt_fom_simple)

        # Run the optimization
        x0 = solver.optimize(x0)

    x_final = x0

    # Log the final results
    opt.update_design([mapping(x_final)])
    f0, final_grad = opt(need_gradient=False)
    final_backprop_gradient = tensor_jacobian_product(mapping, 0)(x_final, final_grad)
    results.append(np.real(f0))
    data.append(
        [
            np.squeeze(x_final.copy().reshape(Nx, Ny)),
            np.squeeze(mapping(x_final).copy().reshape(Nx, Ny)),
            np.squeeze(final_backprop_gradient.reshape(Nx, Ny)),
        ]
    )

    _plot_optimization_results(data, results)

    # Save to disk
    if mp.am_really_master() and (output_filename_prefix is not None):
        np.savez(output_filename_prefix + "_data.npz", data=data, results=results)

    return data, results, opt, mapping


def analyze_gradient_convergence(
    resolution: float,
    beta_range: List[float],
    dx: float = DEFAULT_DESIGN_REGION_WIDTH,
    dy: float = DEFAULT_DESIGN_REGION_HEIGHT,
    waveguide_width: float = DEFAULT_WAVEGUIDE_WIDTH,
) -> Tuple[List[float], List[float]]:
    """Analyze the norm of the gradient vs beta.

    This experiment plots the norm of the gradient vector as a function of beta
    for both the smoothed projection and standard projection functions. As seen
    here, the smoothed projection function has a well-defined.

    Args:
        resolution: Simulation resolution in pixels/micron.
        beta_range: List of projection threshold parameters to test.
        dx: Design region width in microns.
        dy: Design region height in microns.
        waveguide_width: Waveguide width in microns.

    Returns:
        The norm of the gradient array when using smoothing and when not using
        smoothing as a function of beta.
    """

    # Initialize the optimization problem and mapping functions
    opt, mapping = build_optimization_problem(
        resolution=resolution,
        beta=beta_range[0],
        dx=dx,
        dy=dy,
        waveguide_width=waveguide_width,
        use_smoothed_projection=True,
    )

    # pull number of parameters from the design region
    n = opt.design_regions[0].design_parameters.weights.size
    grid_size = opt.design_regions[0].design_parameters.grid_size
    Nx, Ny = int(grid_size.x), int(grid_size.y)

    # initial guess, which is just a simple waveguide crossing
    x = np.linspace(-dy / 2, dy / 2, Nx)
    y = np.linspace(-dx / 2, dy / 2, Ny)
    X, Y = np.meshgrid(x, y)
    mask = (np.abs(Y) <= waveguide_width / 2.0) + (np.abs(X) <= waveguide_width / 2.0)
    x0 = np.zeros((Nx, Ny))
    x0[mask.T] = 1
    x0 = mapping(x0)

    # Initialize storage structures
    with_smoothing = []
    without_smoothing = []

    for beta in beta_range:
        # Compute the norm with smoothing
        opt, mapping = build_optimization_problem(
            resolution=resolution,
            beta=beta,
            dx=dx,
            dy=dy,
            waveguide_width=waveguide_width,
            use_smoothed_projection=True,
        )
        _, dJ_du = opt([mapping(x0)])
        backprop_gradient = tensor_jacobian_product(mapping, 0)(x0, dJ_du)
        norm_gradient_smoothing = np.linalg.norm(backprop_gradient.flatten())
        with_smoothing.append(norm_gradient_smoothing)

        # Compute the norm without smoothing
        opt, mapping = build_optimization_problem(
            resolution=resolution,
            beta=beta,
            dx=dx,
            dy=dy,
            waveguide_width=waveguide_width,
            use_smoothed_projection=False,
        )
        _, dJ_du = opt([mapping(x0)])
        backprop_gradient = tensor_jacobian_product(mapping, 0)(x0, dJ_du)
        norm_gradient_no_smoothing = np.linalg.norm(backprop_gradient.flatten())
        without_smoothing.append(norm_gradient_no_smoothing)

    # plot results
    plt.figure(figsize=(5.2, 2.5), constrained_layout=True)
    plt.loglog(beta_range, without_smoothing, "o-", label="W/o smoothing")
    plt.loglog(beta_range, with_smoothing, "o-", label="W/ smoothing")
    plt.legend()
    plt.xlabel("β")
    plt.ylabel("|df/dx|")
    plt.show()

    return with_smoothing, without_smoothing


def analyze_FOM_convergence(
    resolution: float, beta: float, maxeval: int, damping_factor: float = 0.0
) -> Tuple[List[float], List[float]]:
    """Analyze the convergence of the new projection method.

    Because the smoothed projection has a well-defined gradient for all values
    of beta, it should exhibit better convergence properties than the standard
    projection step, which grows increasingly more ill-conditioned as
    beta->infinity. This experiment runs the same shape optimization problem for
    both projection functions and plots the results.

    Args:
        resolution: Simulation resolution in pixels/micron.
        beta: Tanh function projection strength parameter, ranging from [0,∞].
        maxeval: Maximum number of optimization iterations to run.

    Returns:
        The FOM evolution for both the smoothed and not smoothed case as a
        function of optimization iteration.

    """
    print("Running shape optimization WITHOUT smoothing...")
    _, results, _, _ = run_shape_optimization(
        beta=beta,
        resolution=resolution,
        maxeval=maxeval,
        use_smoothed_projection=False,
        plot_results=False,
        output_filename_prefix=f"without_smoothing_grad_{beta}",
        damping_factor=damping_factor,
    )
    print("Running shape optimization WITH smoothing...")
    (_, results_smoothed, _, _,) = run_shape_optimization(
        beta=beta,
        resolution=resolution,
        maxeval=maxeval,
        use_smoothed_projection=True,
        plot_results=False,
        output_filename_prefix=f"with_smoothing_grad_{beta}",
        damping_factor=damping_factor,
    )

    plt.figure(figsize=(5.2, 2.0), constrained_layout=True)
    plt.loglog(results, "o-", label="W/o smoothing")
    plt.loglog(results_smoothed, "o-", label="W/ smoothing")
    plt.legend()
    plt.xlabel("Optimization iteration")
    plt.ylabel("FOM")
    plt.show()

    return results, results_smoothed


def _run_optimizer_smoke() -> dict:
    """Run a two-epoch NLopt continuation and verify persisted state.

    The fixed-point slices below exercise the individual objective and
    gradient operations.  This deliberately small optimization also executes
    the stateful workflow used by the full example: NLopt mutates a design,
    the callback supplies an adjoint gradient, a new solver is constructed
    after changing beta, and the intermediate/final results are persisted.
    """
    beta_history = (4.0, 8.0)
    max_evaluations_per_epoch = 2
    output_directory = tempfile.TemporaryDirectory(
        prefix="gpmeep-waveguide-optimizer-smoke-"
    )
    checkpoint_path = (
        Path(output_directory.name) / "optimizer-checkpoint.json"
    )
    result_path = Path(output_directory.name) / "optimizer-result.npz"
    design = None
    checkpoint_epochs = []
    scalar_metrics = {
        "beta_initial": beta_history[0],
        "beta_final": beta_history[-1],
        "epoch_count": len(beta_history),
    }

    for epoch, beta in enumerate(beta_history):
        problem, mapping = build_optimization_problem(
            resolution=10,
            beta=beta,
            use_smoothed_projection=True,
            dx=1.0,
            dy=1.0,
        )
        design_size = problem.design_regions[0].design_parameters.weights.size
        if design is None:
            design = np.full(design_size, 0.5, dtype=np.float64)
        elif design.size != design_size:
            raise AssertionError("continuation changed the design-vector size")

        callback_data = []
        objective_history = []
        gradient_norm_history = []

        def callback_factory(
            epoch_problem,
            epoch_mapping,
            epoch_data,
            epoch_objectives,
            epoch_gradient_norms,
        ):
            def objective_callback(values, gradient):
                if gradient.size != values.size:
                    raise AssertionError(
                        "NLopt did not supply a full-sized writable gradient"
                    )
                objective_value = nlopt_fom(
                    values,
                    gradient,
                    opt=epoch_problem,
                    mapping=epoch_mapping,
                    data=epoch_data,
                    results=epoch_objectives,
                )
                if not np.isfinite(objective_value) or not np.all(
                    np.isfinite(gradient)
                ):
                    raise AssertionError(
                        "optimizer callback produced a non-finite value"
                    )
                epoch_gradient_norms.append(float(np.linalg.norm(gradient)))
                return objective_value

            return objective_callback

        objective_callback = callback_factory(
            problem,
            mapping,
            callback_data,
            objective_history,
            gradient_norm_history,
        )

        solver = nlopt.opt(nlopt.LD_CCSAQ, design_size)
        solver.set_lower_bounds(0.0)
        solver.set_upper_bounds(1.0)
        solver.set_maxeval(max_evaluations_per_epoch)
        solver.set_min_objective(objective_callback)

        design_before = design.copy()
        design = np.asarray(solver.optimize(design), dtype=np.float64)
        objective_history = [
            float(np.asarray(value).reshape(-1)[0])
            for value in objective_history
        ]
        callback_count = len(objective_history)
        nlopt_evaluations = int(solver.get_numevals())
        if callback_count != max_evaluations_per_epoch:
            raise AssertionError(
                "NLopt continuation epoch did not execute the requested "
                f"callbacks: epoch={epoch}, callbacks={callback_count}"
            )
        if nlopt_evaluations != callback_count:
            raise AssertionError("NLopt evaluation and callback counts disagree")
        result_code = int(solver.last_optimize_result())
        if result_code != int(nlopt.MAXEVAL_REACHED):
            raise AssertionError(
                "NLopt smoke did not terminate at the deliberate evaluation "
                f"gate: epoch={epoch}, result_code={result_code}"
            )
        if not np.all(np.isfinite(design)):
            raise AssertionError("NLopt returned a non-finite design")
        if np.any(design < 0.0) or np.any(design > 1.0):
            raise AssertionError("NLopt returned a design outside its bounds")
        design_step_l2 = float(np.linalg.norm(design - design_before))
        if not np.isfinite(design_step_l2) or design_step_l2 <= 1e-10:
            raise AssertionError("NLopt did not mutate the continuation design")

        epoch_record = {
            "beta": beta,
            "callback_count": callback_count,
            "design": design.tolist(),
            "gradient_norm_history": gradient_norm_history,
            "nlopt_evaluations": nlopt_evaluations,
            "objective_history": objective_history,
            "result_code": result_code,
        }
        checkpoint_epochs.append(epoch_record)
        checkpoint_payload = {
            "completed_epochs": epoch + 1,
            "epochs": checkpoint_epochs,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint_payload, sort_keys=True),
            encoding="utf-8",
        )
        loaded_checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if loaded_checkpoint != checkpoint_payload:
            raise AssertionError("optimizer checkpoint did not round-trip exactly")

        metric_prefix = f"epoch{epoch}"
        scalar_metrics.update(
            {
                f"{metric_prefix}_beta": beta,
                f"{metric_prefix}_callback_count": callback_count,
                f"{metric_prefix}_design_step_l2": design_step_l2,
                f"{metric_prefix}_gradient_l2_initial": gradient_norm_history[0],
                f"{metric_prefix}_gradient_l2_final": gradient_norm_history[-1],
                f"{metric_prefix}_nlopt_evaluations": nlopt_evaluations,
                f"{metric_prefix}_objective_initial": objective_history[0],
                f"{metric_prefix}_objective_final": objective_history[-1],
                f"{metric_prefix}_result_code": result_code,
            }
        )
        if epoch == 0:
            scalar_metrics["epoch0_design"] = design.tolist()

    saved_objective_history = np.asarray(
        [
            value
            for epoch_record in checkpoint_epochs
            for value in epoch_record["objective_history"]
        ],
        dtype=np.float64,
    )
    np.savez(
        result_path,
        beta_history=np.asarray(beta_history, dtype=np.float64),
        design=design,
        objective_history=saved_objective_history,
    )
    with np.load(result_path) as loaded_result:
        result_design = np.asarray(loaded_result["design"], dtype=np.float64)
        result_beta_history = np.asarray(
            loaded_result["beta_history"],
            dtype=np.float64,
        )
        result_objective_history = np.asarray(
            loaded_result["objective_history"],
            dtype=np.float64,
        )
    np.testing.assert_array_equal(result_design, design)
    np.testing.assert_array_equal(
        result_beta_history,
        np.asarray(beta_history, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result_objective_history,
        saved_objective_history,
    )

    checkpoint_roundtrip_differences = []
    for original, loaded in zip(
        checkpoint_payload["epochs"],
        loaded_checkpoint["epochs"],
    ):
        for key in (
            "design",
            "gradient_norm_history",
            "objective_history",
        ):
            checkpoint_roundtrip_differences.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(original[key], dtype=np.float64)
                            - np.asarray(loaded[key], dtype=np.float64)
                        )
                    )
                )
            )
        for key in (
            "beta",
            "callback_count",
            "nlopt_evaluations",
            "result_code",
        ):
            checkpoint_roundtrip_differences.append(
                float(abs(original[key] - loaded[key]))
            )
    checkpoint_roundtrip_max_abs_difference = max(
        checkpoint_roundtrip_differences
    )
    result_roundtrip_max_abs_difference = max(
        float(np.max(np.abs(result_design - design))),
        float(
            np.max(
                np.abs(
                    result_beta_history
                    - np.asarray(beta_history, dtype=np.float64)
                )
            )
        ),
        float(
            np.max(
                np.abs(result_objective_history - saved_objective_history)
            )
        ),
    )

    scalar_metrics.update(
        {
            "checkpoint_callback_count": sum(
                record["callback_count"] for record in checkpoint_epochs
            ),
            "checkpoint_completed_epochs": len(checkpoint_epochs),
            "checkpoint_roundtrip_max_abs_difference": (
                checkpoint_roundtrip_max_abs_difference
            ),
            "final_design": design.tolist(),
            "final_design_l2": float(np.linalg.norm(design)),
            "result_roundtrip_max_abs_difference": (
                result_roundtrip_max_abs_difference
            ),
        }
    )
    output_directory.cleanup()
    return scalar_metrics


def run_validation() -> None:
    """Validate projection gradients and a reduced optimizer continuation."""
    resolution = 20
    configurations = [
        ("smoothed_beta10", True, 10.0),
        ("smoothed_beta1000", True, 1000.0),
        ("tanh_beta10", False, 10.0),
        ("tanh_beta1000", False, 1000.0),
    ]
    metrics = {}
    for name, use_smoothed_projection, beta in configurations:
        opt, mapping = build_optimization_problem(
            resolution=resolution,
            beta=beta,
            use_smoothed_projection=use_smoothed_projection,
            dx=2.0,
            dy=2.0,
        )
        grid_size = opt.design_regions[0].design_parameters.grid_size
        nx, ny = int(grid_size.x), int(grid_size.y)
        x_coordinates = np.linspace(-1.0, 1.0, nx)
        y_coordinates = np.linspace(-1.0, 1.0, ny)
        xv, yv = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
        raw_design = (
            (np.abs(xv) <= DEFAULT_WAVEGUIDE_WIDTH / 2)
            | (np.abs(yv) <= DEFAULT_WAVEGUIDE_WIDTH / 2)
        ).astype(np.float64)
        raw_design = raw_design.ravel()

        mapped_design = np.asarray(mapping(raw_design))
        objective, density_gradient = opt([mapped_design])
        backprop_gradient = tensor_jacobian_product(mapping, 0)(
            raw_design,
            density_gradient,
        )
        backprop_gradient = np.asarray(
            backprop_gradient, dtype=np.float64
        ).reshape(-1)

        # Follow a feasible descent direction so the independent forward
        # perturbation remains inside MaterialGrid's [0, 1] design bounds.
        # A generic oscillatory direction moves some binary pixels outside
        # those bounds and makes the finite difference unnecessarily small.
        direction = -backprop_gradient.copy()
        direction[(raw_design == 0.0) & (direction < 0.0)] = 0.0
        direction[(raw_design == 1.0) & (direction > 0.0)] = 0.0
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-30:
            oscillation = np.sin(
                np.arange(raw_design.size, dtype=np.float64)
                * np.float64(0.754877666)
            )
            direction = np.where(
                raw_design == 0.0,
                np.abs(oscillation),
                -np.abs(oscillation),
            )
            direction_norm = np.linalg.norm(direction)
        direction /= direction_norm
        # The high-beta smoothed gradient is roughly an order of magnitude
        # smaller than the low-beta cases. Use a correspondingly larger
        # forward perturbation so its objective delta stays above the FP32
        # DFT-decay floor while remaining well inside the feasible bounds.
        step = (
            1e-2
            if use_smoothed_projection and beta >= 1000.0
            else 1e-3
        )
        mapped_perturbed_design = np.asarray(
            mapping(raw_design + step * direction)
        )
        perturbed_objective, _ = opt(
            [mapped_perturbed_design],
            need_gradient=False,
        )
        objective_value = float(np.asarray(objective).reshape(-1)[0])
        perturbed_value = float(
            np.asarray(perturbed_objective).reshape(-1)[0]
        )
        adjoint_delta = float(step * np.dot(backprop_gradient, direction))
        finite_difference_delta = perturbed_value - objective_value
        gradient_l2 = float(np.linalg.norm(backprop_gradient))
        mapped_design_max_abs_difference = float(
            np.max(np.abs(mapped_perturbed_design - mapped_design))
        )
        if not use_smoothed_projection and beta >= 1000.0:
            # The ordinary tanh projection at beta=1000 is analytically
            # saturated at this binary cross. Its derivative should underflow
            # to a negligible value and the independent forward solve should
            # be unchanged. This is a saturation oracle, not a skipped
            # gradient comparison; the smoothed beta=1000 case above remains
            # nonzero and is checked by the standard directional oracle.
            if gradient_l2 >= 1e-20:
                raise AssertionError(
                    "high-beta tanh gradient did not saturate as expected"
                )
            if abs(adjoint_delta) >= 1e-30:
                raise AssertionError(
                    "high-beta tanh directional derivative is not negligible"
                )
            if not np.array_equal(mapped_design, mapped_perturbed_design):
                raise AssertionError(
                    "high-beta tanh perturbation changed the mapped design: "
                    f"max_abs_difference={mapped_design_max_abs_difference}"
                )
            saturation_objective_atol = float(
                32
                * np.finfo(np.float32).eps
                * max(abs(objective_value), abs(perturbed_value))
            )
            if abs(finite_difference_delta) > saturation_objective_atol:
                raise AssertionError(
                    "high-beta tanh repeated objective exceeded its FP32 "
                    "allowance: "
                    f"delta={finite_difference_delta}, "
                    f"atol={saturation_objective_atol}"
                )
        else:
            if abs(adjoint_delta) <= 1e-8:
                raise AssertionError(
                    f"{name} produced a degenerate directional oracle"
                )
            np.testing.assert_allclose(
                finite_difference_delta,
                adjoint_delta,
                rtol=5e-2,
                atol=1e-10,
            )
        metrics[name] = {
            "adjoint_directional_derivative": adjoint_delta,
            "finite_difference_directional_derivative": (
                finite_difference_delta
            ),
            "gradient_l2": gradient_l2,
            "objective": objective_value,
            "objective_perturbed": perturbed_value,
        }
        if not use_smoothed_projection and beta >= 1000.0:
            metrics[name].update(
                {
                    "mapped_design_max_abs_difference": (
                        mapped_design_max_abs_difference
                    ),
                    "saturation_objective_atol": saturation_objective_atol,
                }
            )
    metrics["optimizer_smoke"] = _run_optimizer_smoke()
    print("gpmeep-example-metrics:" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help=(
            "run deterministic projection-gradient slices and a reduced "
            "NLopt continuation"
        ),
    )
    args = parser.parse_args()
    if args.validation:
        run_validation()
        raise SystemExit(0)

    analyze_gradient_convergence(
        beta_range=np.logspace(1, 3, base=10, num=10), resolution=20
    )
