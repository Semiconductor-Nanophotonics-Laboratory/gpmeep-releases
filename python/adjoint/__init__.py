"""
Adjoint-based sensitivity-analysis module for pymeep.
Authors: Homer Reid <homer@homerreid.com>, Alec Hammond <alec.hammond@gatech.edu>, Ian Williamson <iwill@google.com>
"""

from .objective import *

from . import utils
from .utils import DesignRegion

from .optimization_problem import OptimizationProblem

_LAZY_EXPORTS = {
    # Preserve the historical module aliases without importing their optional
    # dependency stacks during an ordinary solver startup.
    "basis": (".basis", None),
    "filter_source": (".filter_source", None),
    "filters": (".filters", None),
    "connectivity": (".connectivity", None),
    "wrapper": (".wrapper", None),
    "BilinearInterpolationBasis": (".basis", "BilinearInterpolationBasis"),
    "FilteredSource": (".filter_source", "FilteredSource"),
    "MeepJaxWrapper": (".wrapper", "MeepJaxWrapper"),
    "cc_fd": (".connectivity", "cc_fd"),
    "constraint_connectivity": (".connectivity", "constraint_connectivity"),
    "convolve_design_weights_and_kernel": (
        ".filters",
        "convolve_design_weights_and_kernel",
    ),
    "mesh_grid": (".filters", "mesh_grid"),
    "cylindrical_filter": (".filters", "cylindrical_filter"),
    "conic_filter": (".filters", "conic_filter"),
    "gaussian_filter": (".filters", "gaussian_filter"),
    "exponential_erosion": (".filters", "exponential_erosion"),
    "exponential_dilation": (".filters", "exponential_dilation"),
    "heaviside_erosion": (".filters", "heaviside_erosion"),
    "heaviside_dilation": (".filters", "heaviside_dilation"),
    "geometric_erosion": (".filters", "geometric_erosion"),
    "geometric_dilation": (".filters", "geometric_dilation"),
    "harmonic_erosion": (".filters", "harmonic_erosion"),
    "harmonic_dilation": (".filters", "harmonic_dilation"),
    "tanh_projection": (".filters", "tanh_projection"),
    "smoothed_projection": (".filters", "smoothed_projection"),
    "heaviside_projection": (".filters", "heaviside_projection"),
    "get_threshold_wang": (".filters", "get_threshold_wang"),
    "get_eta_from_conic": (".filters", "get_eta_from_conic"),
    "get_conic_radius_from_eta_e": (".filters", "get_conic_radius_from_eta_e"),
    "length_indicator": (".filters", "length_indicator"),
    "indicator_solid": (".filters", "indicator_solid"),
    "constraint_solid": (".filters", "constraint_solid"),
    "indicator_void": (".filters", "indicator_void"),
    "constraint_void": (".filters", "constraint_void"),
    "gray_indicator": (".filters", "gray_indicator"),
    "unfilter_design": (".unfilter_design", "unfilter_design"),
    # The legacy wildcard surface included names imported by filters.py and
    # connectivity.py.  They remain public for compatibility, but are lazy.
    "ArrayLikeType": (".filters", "ArrayLikeType"),
    "Tuple": (".filters", "Tuple"),
    "Union": (".filters", "Union"),
    "signal": (".filters", "signal"),
    "special": (".filters", "special"),
    "sys": (".filters", "sys"),
    "cg": (".connectivity", "cg"),
    "spsolve": (".connectivity", "spsolve"),
    "kron": (".connectivity", "kron"),
    "diags": (".connectivity", "diags"),
    "csr_matrix": (".connectivity", "csr_matrix"),
    "eye": (".connectivity", "eye"),
    "csc_matrix": (".connectivity", "csc_matrix"),
    "solvers": (".connectivity", "solvers"),
    "npa": (".connectivity", "npa"),
    "grad": (".connectivity", "grad"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute_name = target
    try:
        module = import_module(module_name, __name__)
        value = module if attribute_name is None else getattr(module, attribute_name)
    except ModuleNotFoundError as error:
        if name in {"MeepJaxWrapper", "wrapper"}:
            raise AttributeError(
                f"{name} requires the optional JAX dependency stack"
            ) from error
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# Preserve ``from meep.adjoint import *`` behavior: wildcard users explicitly
# request the complete surface, so resolving every installed export is
# expected.  JAX itself remains optional, just as it was before lazy loading.
from importlib.util import find_spec as _find_spec


def _jax_stack_is_discoverable():
    try:
        return all(
            _find_spec(name) is not None
            for name in ("jax", "jaxlib", "ml_dtypes", "opt_einsum")
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

_WILDCARD_LAZY_EXPORTS = set(_LAZY_EXPORTS)
if not _jax_stack_is_discoverable():
    _WILDCARD_LAZY_EXPORTS.difference_update({"MeepJaxWrapper", "wrapper"})
__all__ = sorted(
    {name for name in globals() if not name.startswith("_")}
    | _WILDCARD_LAZY_EXPORTS
)
