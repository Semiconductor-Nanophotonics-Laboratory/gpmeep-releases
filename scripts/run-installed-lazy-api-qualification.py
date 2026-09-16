#!/usr/bin/env python3
"""Qualify installed lazy imports, historical API aliases, and real consumers."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import typing


def _load_contract():
    path = pathlib.Path(__file__).resolve().with_name(
        "gpmeep_qualification_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gpmeep_qualification_contract_for_lazy_api", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qualification contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract()
QualificationContractError = CONTRACT.QualificationContractError
HISTORICAL_ADJOINT_WILDCARD = frozenset(
    {
        "ArrayLikeType",
        "BilinearInterpolationBasis",
        "Callable",
        "DesignRegion",
        "EigenmodeCoefficient",
        "FilteredSource",
        "FluxData",
        "FourierFields",
        "Grid",
        "LDOS",
        "List",
        "MeepJaxWrapper",
        "Near2FarFields",
        "NearToFarData",
        "ObjectiveQuantity",
        "OptimizationProblem",
        "Optional",
        "Tuple",
        "Union",
        "abc",
        "basis",
        "cc_fd",
        "cg",
        "conic_filter",
        "connectivity",
        "constraint_connectivity",
        "constraint_solid",
        "constraint_void",
        "convolve_design_weights_and_kernel",
        "csc_matrix",
        "csr_matrix",
        "cylindrical_filter",
        "diags",
        "exponential_dilation",
        "exponential_erosion",
        "eye",
        "filter_source",
        "filters",
        "gaussian_filter",
        "geometric_dilation",
        "geometric_erosion",
        "get_conic_radius_from_eta_e",
        "get_eta_from_conic",
        "get_threshold_wang",
        "grad",
        "gray_indicator",
        "harmonic_dilation",
        "harmonic_erosion",
        "heaviside_dilation",
        "heaviside_erosion",
        "heaviside_projection",
        "indicator_solid",
        "indicator_void",
        "kron",
        "length_indicator",
        "mesh_grid",
        "mp",
        "namedtuple",
        "np",
        "npa",
        "objective",
        "optimization_problem",
        "py_v3_to_vec",
        "signal",
        "smoothed_projection",
        "solvers",
        "special",
        "spsolve",
        "sys",
        "tanh_projection",
        "unfilter_design",
        "utils",
        "wrapper",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--expected-extension", required=True, type=pathlib.Path)
    parser.add_argument("--expected-libmeep", required=True, type=pathlib.Path)
    return parser.parse_args()


def _module_root_loaded(name: str) -> bool:
    return any(
        module == name or module.startswith(name + ".")
        for module in sys.modules
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_test(
    repo: pathlib.Path,
    relative: str,
    selectors: tuple[str, ...],
    expected_tests: int,
    child_pythonpath: str,
) -> dict[str, object]:
    source = (repo / relative).resolve()
    try:
        source.relative_to(repo)
    except ValueError as exc:
        raise QualificationContractError(
            f"installed lazy API test escaped repository: {source}"
        ) from exc
    command = [sys.executable, str(source), *selectors, "-v"]
    started = time.perf_counter()
    child_environment = dict(os.environ)
    child_environment["PYTHONPATH"] = child_pythonpath
    completed = subprocess.run(
        command,
        cwd=repo,
        env=child_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )
    elapsed = time.perf_counter() - started
    print(f"gpmeep-lazy-api-test:{relative}:stdout")
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    print(f"gpmeep-lazy-api-test:{relative}:stderr")
    print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    combined = completed.stdout + "\n" + completed.stderr
    matches = re.findall(r"Ran\s+(\d+)\s+tests?\s+in", combined)
    if (
        completed.returncode != 0
        or matches != [str(expected_tests)]
        or re.search(r"(?m)^OK\s*$", combined) is None
    ):
        raise QualificationContractError(
            f"installed lazy API consumer test failed or changed inventory: {relative}"
        )
    return {
        "source": CONTRACT._file_record(source, repo),
        "selectors": list(selectors),
        "tests_run": expected_tests,
        "pythonpath": child_pythonpath,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": _digest_text(completed.stdout),
        "stderr_sha256": _digest_text(completed.stderr),
    }


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    expected_installed_python = str(
        args.expected_extension.resolve().parent.parent
    )
    if os.environ.get("PYTHONPATH") != expected_installed_python:
        raise QualificationContractError(
            "installed lazy API qualification has an unexpected parent PYTHONPATH"
        )
    test_support = str((repo / "python" / "tests").resolve())
    child_pythonpath = test_support + os.pathsep + expected_installed_python
    deferred_roots = ("jax", "matplotlib", "scipy")
    if any(_module_root_loaded(name) for name in deferred_roots):
        raise QualificationContractError(
            "qualification harness preloaded an optional module stack"
        )

    import meep as mp
    import meep.adjoint as mpa
    from mpi4py import MPI

    initial_modules = {
        name: _module_root_loaded(name) for name in deferred_roots
    }
    if any(initial_modules.values()):
        raise QualificationContractError(
            "ordinary installed Meep import eagerly loaded an optional stack"
        )
    core_surface = {
        "meep.Simulation": callable(mp.Simulation),
        "meep.adjoint.DesignRegion": callable(mpa.DesignRegion),
        "meep.adjoint.FourierFields": callable(mpa.FourierFields),
        "meep.adjoint.OptimizationProblem": callable(mpa.OptimizationProblem),
    }
    discoverable_names = {
        "meep.visualization": "visualization" in dir(mp),
        "meep.plot2D": "plot2D" in dir(mp),
        "meep.adjoint.basis": "basis" in dir(mpa),
        "meep.adjoint.filter_source": "filter_source" in dir(mpa),
        "meep.adjoint.filters": "filters" in dir(mpa),
        "meep.adjoint.connectivity": "connectivity" in dir(mpa),
        "meep.adjoint.wrapper": "wrapper" in dir(mpa),
        "meep.adjoint.FilteredSource": "FilteredSource" in dir(mpa),
        "meep.adjoint.MeepJaxWrapper": "MeepJaxWrapper" in dir(mpa),
        "meep.adjoint.conic_filter": "conic_filter" in dir(mpa),
    }
    if not all(core_surface.values()) or not all(discoverable_names.values()):
        raise QualificationContractError(
            "installed lazy API core or discoverable surface is incomplete"
        )

    aliases = {
        "meep.visualization": mp.visualization,
        "meep.adjoint.basis": mpa.basis,
        "meep.adjoint.filter_source": mpa.filter_source,
        "meep.adjoint.filters": mpa.filters,
        "meep.adjoint.connectivity": mpa.connectivity,
        "meep.adjoint.wrapper": mpa.wrapper,
    }
    expected_aliases = {
        "meep.visualization": "meep.visualization",
        "meep.adjoint.basis": "meep.adjoint.basis",
        "meep.adjoint.filter_source": "meep.adjoint.filter_source",
        "meep.adjoint.filters": "meep.adjoint.filters",
        "meep.adjoint.connectivity": "meep.adjoint.connectivity",
        "meep.adjoint.wrapper": "meep.adjoint.wrapper",
    }
    module_aliases = {
        name: value.__name__ for name, value in aliases.items()
    }
    if module_aliases != expected_aliases:
        raise QualificationContractError(
            f"installed module aliases disagree: {module_aliases!r}"
        )

    root_namespace: dict[str, object] = {}
    adjoint_namespace: dict[str, object] = {}
    exec("from meep import *", root_namespace)
    exec("from meep.adjoint import *", adjoint_namespace)
    actual_adjoint_all = frozenset(mpa.__all__)
    imported_adjoint_names = frozenset(
        name for name in adjoint_namespace if not name.startswith("__")
    )
    if (
        actual_adjoint_all != HISTORICAL_ADJOINT_WILDCARD
        or imported_adjoint_names != HISTORICAL_ADJOINT_WILDCARD
    ):
        raise QualificationContractError(
            "installed adjoint wildcard namespace differs from the historical API: "
            f"missing={sorted(HISTORICAL_ADJOINT_WILDCARD - actual_adjoint_all)!r}, "
            f"extra={sorted(actual_adjoint_all - HISTORICAL_ADJOINT_WILDCARD)!r}"
        )
    wildcard_required = {
        "meep.visualization": "visualization" in root_namespace,
        "meep.plot2D": "plot2D" in root_namespace,
        "meep.adjoint.filters": "filters" in adjoint_namespace,
        "meep.adjoint.connectivity": "connectivity" in adjoint_namespace,
        "meep.adjoint.wrapper": "wrapper" in adjoint_namespace,
        "meep.adjoint.ArrayLikeType": "ArrayLikeType" in adjoint_namespace,
        "meep.adjoint.cg": "cg" in adjoint_namespace,
        "meep.adjoint.grad": "grad" in adjoint_namespace,
        "meep.adjoint.npa": "npa" in adjoint_namespace,
        "meep.adjoint.FilteredSource": "FilteredSource" in adjoint_namespace,
        "meep.adjoint.MeepJaxWrapper": "MeepJaxWrapper" in adjoint_namespace,
    }
    if not all(wildcard_required.values()):
        raise QualificationContractError(
            "installed historical wildcard surface is incomplete"
        )

    axes_annotation = typing.get_type_hints(mp.Simulation.plot2D).get("ax")
    annotation_types = set(typing.get_args(axes_annotation))
    axes_type = importlib.import_module("matplotlib.axes").Axes
    annotation_is_axes = axes_type in annotation_types
    if not annotation_is_axes:
        raise QualificationContractError(
            "Simulation.plot2D no longer resolves to the Matplotlib Axes type"
        )

    tests = [
        _run_test(
            repo,
            relative,
            tuple(selectors),
            expected_tests,
            child_pythonpath,
        )
        for relative, selectors, expected_tests in (
            CONTRACT.INSTALLED_LAZY_API_TEST_SPECS
        )
    ]
    runtime = CONTRACT.python_runtime_attestation(
        args.expected_extension.resolve(), args.expected_libmeep.resolve()
    )
    value = {
        "schema_version": 1,
        "qualification": CONTRACT.INSTALLED_LAZY_API_QUALIFICATION_NAME,
        "log_name": CONTRACT.INSTALLED_LAZY_API_LOG_NAME,
        "deferred_module_roots": list(deferred_roots),
        "initial_modules": initial_modules,
        "core_surface": core_surface,
        "discoverable_surface": discoverable_names,
        "module_aliases": module_aliases,
        "wildcard_surface": wildcard_required,
        "adjoint_wildcard_names": sorted(actual_adjoint_all),
        "plot2d_axes_annotation_is_matplotlib_axes": annotation_is_axes,
        "tests": tests,
        "python_extension": runtime["python_extension"],
        "libmeep": runtime["libmeep"],
    }
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    # This evidence file has an exact terminal-marker contract.  Meep normally
    # prints elapsed time from an atexit handler, so release its collective
    # claim window and finalize MPI before emitting the final record.  _exit
    # then prevents any later atexit output from invalidating the sealed log.
    mp._gpu_finalize_distributed_runtime()
    MPI.Finalize()
    print(serialized)
    print(CONTRACT.marker_for(CONTRACT.INSTALLED_LAZY_API_LOG_NAME).decode("ascii"))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
