from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpmeep_qualification_contract as contract  # noqa: E402


def _load_lazy_qualifier():
    path = SCRIPTS / "run-installed-lazy-api-qualification.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_installed_lazy_api_for_tests", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAZY_QUALIFIER = _load_lazy_qualifier()


class LazyPythonApiTests(unittest.TestCase):
    def test_installed_inventory_executes_a_real_jax_wrapper_gradient(self):
        specs = {
            relative: (tuple(selectors), count)
            for relative, selectors, count in contract.INSTALLED_LAZY_API_TEST_SPECS
        }
        self.assertEqual(specs["python/tests/test_adjoint_utils.py"], ((), 7))
        selectors, count = specs["python/tests/test_adjoint_jax.py"]
        self.assertEqual(count, 3)
        self.assertIn(
            (
                "WrapperTest.test_wrapper_gradients_0_"
                "1500_1550bw_01relative_gaussian_port1"
            ),
            selectors,
        )

    def test_installed_qualifier_finalizes_before_exact_terminal_marker(self):
        runner = (
            SCRIPTS / "run-installed-lazy-api-qualification.py"
        ).read_text(encoding="utf-8")
        cleanup = runner.index("mp._gpu_finalize_distributed_runtime()")
        finalize = runner.index("MPI.Finalize()")
        terminal_print = runner.index(
            "print(CONTRACT.marker_for(CONTRACT.INSTALLED_LAZY_API_LOG_NAME)"
        )
        flush = runner.index("sys.stdout.flush()", terminal_print)
        hard_exit = runner.index("os._exit(0)", flush)
        self.assertLess(cleanup, finalize)
        self.assertLess(finalize, terminal_print)
        self.assertLess(terminal_print, flush)
        self.assertLess(flush, hard_exit)

    def _load_adjoint(self, find_spec):
        package_name = "_gpmeep_lazy_fixture_" + uuid.uuid4().hex
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO / "python" / "adjoint")]
        sys.modules[package_name] = package
        objective = types.ModuleType(package_name + ".objective")
        objective.ObjectiveQuantity = type("ObjectiveQuantity", (), {})
        utils = types.ModuleType(package_name + ".utils")
        utils.DesignRegion = type("DesignRegion", (), {})
        optimization = types.ModuleType(package_name + ".optimization_problem")
        optimization.OptimizationProblem = type("OptimizationProblem", (), {})
        sys.modules[objective.__name__] = objective
        sys.modules[utils.__name__] = utils
        sys.modules[optimization.__name__] = optimization
        self.addCleanup(
            lambda: [
                sys.modules.pop(name, None)
                for name in tuple(sys.modules)
                if name == package_name or name.startswith(package_name + ".")
            ]
        )
        path = REPO / "python" / "adjoint" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name,
            path,
            submodule_search_locations=[str(path.parent)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        with mock.patch("importlib.util.find_spec", side_effect=find_spec):
            spec.loader.exec_module(module)
        return module

    def test_partial_jax_stack_is_excluded_from_wildcard_but_discoverable(self):
        module = self._load_adjoint(
            lambda name: object() if name == "jax" else None
        )
        self.assertIn("MeepJaxWrapper", dir(module))
        self.assertIn("wrapper", dir(module))
        self.assertNotIn("MeepJaxWrapper", module.__all__)
        self.assertNotIn("wrapper", module.__all__)

    def test_lazy_module_alias_returns_module_object(self):
        module = self._load_adjoint(lambda _name: None)
        filters = types.ModuleType(module.__name__ + ".filters")
        sys.modules[filters.__name__] = filters
        self.assertIs(module.filters, filters)
        self.assertIn("filters", module.__dict__)

    def test_jax_wildcard_probe_fails_closed_on_find_spec_error(self):
        module = self._load_adjoint(
            lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("partial"))
        )
        self.assertNotIn("MeepJaxWrapper", module.__all__)
        self.assertNotIn("wrapper", module.__all__)

    def test_installed_lazy_api_log_is_exact_and_artifact_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            extension = root / "_meep.so"
            libmeep = root / "libmeep.so"
            extension.write_bytes(b"extension")
            libmeep.write_bytes(b"libmeep")
            artifacts = {
                "installed_python_extension": contract._file_record(
                    extension, REPO
                ),
                "installed_libmeep": contract._file_record(libmeep, REPO),
            }
            tests = []
            for relative, selectors, tests_run in (
                contract.INSTALLED_LAZY_API_TEST_SPECS
            ):
                tests.append(
                    {
                        "source": contract._file_record(REPO / relative, REPO),
                        "selectors": list(selectors),
                        "tests_run": tests_run,
                        "pythonpath": (
                            str((REPO / "python" / "tests").resolve())
                            + os.pathsep
                            + str(extension.parent.parent)
                        ),
                        "returncode": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_sha256": "1" * 64,
                        "stderr_sha256": "2" * 64,
                    }
                )
            value = {
                "schema_version": 1,
                "qualification": contract.INSTALLED_LAZY_API_QUALIFICATION_NAME,
                "log_name": contract.INSTALLED_LAZY_API_LOG_NAME,
                "deferred_module_roots": ["jax", "matplotlib", "scipy"],
                "initial_modules": {
                    "jax": False,
                    "matplotlib": False,
                    "scipy": False,
                },
                "core_surface": {"meep.Simulation": True},
                "discoverable_surface": {"meep.visualization": True},
                "module_aliases": {
                    "meep.visualization": "meep.visualization"
                },
                "wildcard_surface": {"meep.plot2D": True},
                "adjoint_wildcard_names": sorted(
                    LAZY_QUALIFIER.HISTORICAL_ADJOINT_WILDCARD
                ),
                "plot2d_axes_annotation_is_matplotlib_axes": True,
                "tests": tests,
                "python_extension": artifacts["installed_python_extension"],
                "libmeep": artifacts["installed_libmeep"],
            }
            log = root / contract.INSTALLED_LAZY_API_LOG_NAME

            def write(payload):
                log.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    + contract.marker_for(
                        contract.INSTALLED_LAZY_API_LOG_NAME
                    ).decode("ascii")
                    + "\n",
                    encoding="utf-8",
                )

            write(value)
            contract.validate_installed_lazy_api_qualification_log(
                root, artifacts, REPO
            )
            changed = dict(value)
            changed["unexpected"] = True
            write(changed)
            with self.assertRaisesRegex(RuntimeError, "schema"):
                contract.validate_installed_lazy_api_qualification_log(
                    root, artifacts, REPO
                )


if __name__ == "__main__":
    unittest.main()
