"""Self-tests for the generic unchanged-example physical oracle."""

from __future__ import annotations

import contextlib
import ast
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest


ORACLE_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_example_oracle.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_example_oracle_test", ORACLE_PATH)
assert SPEC and SPEC.loader
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class FakeFields:
    def __init__(self, simulation: "FakeSimulation") -> None:
        self.simulation = simulation

    def dft_norm(self) -> float:
        return 2.0 * self.simulation.runs

    @property
    def t(self) -> int:
        return 10 * self.simulation.runs

    def total_volume(self) -> object:
        return self

    def gpu_execution_diagnostic(self) -> str:
        return "synthetic execution diagnostic"


class FakeSimulation:
    gpu = None

    def __init__(self) -> None:
        self.runs = 0
        self.fields = FakeFields(self)

    def run(self, *step_funcs, **kwargs):
        self.runs += int(kwargs["until"])
        assert self.gpu is not None
        self.gpu.advance()

    def timestep(self) -> int:
        return 10 * self.runs

    def round_time(self) -> float:
        return 0.5 * self.runs

    def field_energy_in_box(self, box=None) -> float:
        return 1.5 * self.runs


class FakeGpu:
    def __init__(self, active_backend: str = "cpu") -> None:
        self.active_backend = active_backend
        self.requested_backend = active_backend
        self.cpu_calls = 0
        self.cuda_calls = 0

    def advance(self) -> None:
        if self.active_backend == "cpu":
            self.cpu_calls += 3
        else:
            self.cuda_calls += 3

    def statistics(self):
        return {
            "runtime": {
                "runtime_availability_probes": 0,
                "runtime_device_enumerations": 0,
                "runtime_device_selections": 0,
            },
            "dispatch": {
                "cpu_curl_calls": self.cpu_calls,
                "cuda_curl_calls": self.cuda_calls,
                "host_to_device_bytes": 0,
            }
        }


def fake_meep_module(
    simulation_class=FakeSimulation, active_backend: str = "cpu"
) -> types.ModuleType:
    fake_meep = types.ModuleType("meep")
    fake_meep.gpu = FakeGpu(active_backend)
    simulation_class.gpu = fake_meep.gpu
    fake_meep.Simulation = simulation_class
    return fake_meep


def contract_args(example: pathlib.Path, *extra: str) -> list[str]:
    return [
        "--expected-run-count",
        "2",
        "--expected-final-timestep",
        "50",
        "--expected-total-timestep-delta",
        "50",
        *extra,
        str(example),
    ]


class ExampleOracleTests(unittest.TestCase):
    def test_unchanged_example_emits_aggregated_run_metrics(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "example.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(ORACLE.main(contract_args(example)), 0)
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        lines = output.getvalue().splitlines()
        checkpoints = [
            json.loads(line[len(ORACLE.CHECKPOINT_PREFIX) :])
            for line in lines
            if line.startswith(ORACLE.CHECKPOINT_PREFIX)
        ]
        self.assertEqual([item["run_index"] for item in checkpoints], [0, 1])
        self.assertEqual(
            checkpoints[0]["record"]["timestep_delta"],
            20,
        )
        audit_line = next(
            line for line in lines if line.startswith(ORACLE.AUDIT_PREFIX)
        )
        audit = json.loads(audit_line[len(ORACLE.AUDIT_PREFIX) :])
        self.assertEqual(audit["runs"][0]["run_phase_calls"]["cpu"], 3)
        self.assertEqual(audit["runs"][0]["measurement_phase_calls"]["cpu"], 0)
        self.assertGreater(audit["runs"][0]["run_wall_seconds"], 0)
        self.assertEqual(audit["runs"][0]["requested_backend"], "cpu")
        self.assertEqual(
            audit["runs"][0]["execution_diagnostic"],
            "synthetic execution diagnostic",
        )
        self.assertEqual(
            audit["runs"][0]["runtime_counters"][
                "runtime_availability_probes"
            ],
            0,
        )
        line = next(
            item for item in lines if item.startswith(ORACLE.METRIC_PREFIX)
        )
        self.assertTrue(line.startswith(ORACLE.METRIC_PREFIX))
        metrics = json.loads(line[len(ORACLE.METRIC_PREFIX) :])
        self.assertEqual(metrics["run_count"], 2)
        self.assertEqual(metrics["timestep"]["final"], 50)
        self.assertEqual(metrics["timestep"]["sum"], 70)
        self.assertEqual(metrics["timestep_delta"]["sum"], 50)
        self.assertEqual(metrics["field_energy"]["weighted_checksum"], 18.0)
        self.assertEqual(metrics["dft_norm"]["l2"], (4.0**2 + 10.0**2) ** 0.5)
        self.assertEqual(metrics["per_run"]["timestep"], [20, 50])
        self.assertEqual(metrics["per_run"]["timestep_delta"], [20, 30])
        self.assertEqual(metrics["per_run"]["field_energy"], [3.0, 7.5])

    def test_example_without_run_is_rejected(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "empty.py"
                example.write_text("import meep\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "without calling"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "1",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_zero_timestep_run_is_rejected_even_if_measurement_dispatches(self) -> None:
        class EmptySimulation(FakeSimulation):
            def run(self, *step_funcs, **kwargs):
                return None

            def field_energy_in_box(self, box=None) -> float:
                assert self.gpu is not None
                self.gpu.advance()
                return 0.0

        fake_meep = fake_meep_module(EmptySimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "empty_run.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=0)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "zero timesteps"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "0",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_one_zero_work_run_cannot_hide_behind_a_later_run(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "partial_noop.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=0)\n"
                    "sim.run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "record indices \\[0\\]"
                ):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "2",
                            "--expected-final-timestep",
                            "10",
                            "--expected-total-timestep-delta",
                            "10",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_progress_without_run_dispatch_is_rejected(self) -> None:
        class NoDispatchSimulation(FakeSimulation):
            def run(self, *step_funcs, **kwargs):
                self.runs += int(kwargs["until"])

        fake_meep = fake_meep_module(NoDispatchSimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "no_dispatch.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "without selected-backend"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_contract_requires_expected_count_and_work_or_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ORACLE.parse_args([str(example)])
            with self.assertRaises(SystemExit):
                ORACLE.parse_args(
                    ["--expected-run-count", "1", str(example)]
                )

    def test_per_run_signal_contract_rejects_one_zero_record(self) -> None:
        records = [
            {
                "timestep": 10,
                "timestep_delta": 10,
                "field_energy": 0.0,
                "dft_norm": 2.0,
            },
            {
                "timestep": 20,
                "timestep_delta": 10,
                "field_energy": 3.0,
                "dft_norm": 4.0,
            },
        ]
        args = types.SimpleNamespace(
            expected_run_count=2,
            expected_final_timestep=20,
            expected_total_timestep_delta=20,
            min_field_energy=1.0,
            min_dft_norm=None,
            min_each_field_energy=1.0,
            min_each_dft_norm=None,
        )
        with self.assertRaisesRegex(RuntimeError, "record indices \\[0\\]"):
            ORACLE.validate_contract(records, args)

    def test_result_vectors_are_captured_with_direct_complex_values(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "results.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n"
                    "spectrum = [[1.0, 2.0 + 3.0j], [4.0, -5.0j]]\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                args = contract_args(
                    example,
                    "--result-vector",
                    "spectrum",
                    "--expected-result-shape",
                    "spectrum=2,2",
                    "--min-result-l2",
                    "spectrum=1",
                )
                with contextlib.redirect_stdout(output):
                    self.assertEqual(ORACLE.main(args), 0)
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        metric_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith(ORACLE.METRIC_PREFIX)
        )
        metrics = json.loads(metric_line[len(ORACLE.METRIC_PREFIX) :])
        result = metrics["example_results"]["spectrum"]
        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["real"], [1.0, 2.0, 4.0, 0.0])
        self.assertEqual(result["imag"], [0.0, 3.0, 0.0, -5.0])
        scalar = ORACLE.numeric_result_vector(0.25, "scalar")
        self.assertEqual(scalar["rank"], 0)
        self.assertNotIn("shape", scalar)

    def test_result_vector_contract_rejects_invalid_results(self) -> None:
        cases = (
            ("missing", {}, "did not define"),
            ("ragged", {"ragged": [[1.0], [2.0, 3.0]]}, "ragged"),
            ("bad", {"bad": [float("inf")]}, "non-finite"),
            ("weak", {"weak": [1e-9]}, "below required signal"),
            ("large", {"large": [2.0]}, "exceeds required limit"),
        )
        for name, namespace, expected in cases:
            with self.subTest(name=name):
                args = types.SimpleNamespace(
                    result_vector=[name],
                    expected_result_shape=[(name, (1,))],
                    min_result_l2=[(name, 1e-6)],
                    max_result_abs=[(name, 1.0)],
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    ORACLE.collect_example_results(namespace, args)

    def test_result_vector_cli_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            base = [
                "--expected-run-count",
                "1",
                "--expected-final-timestep",
                "1",
            ]
            for extra in (
                ["--result-vector", "bad.name"],
                ["--result-vector", "value", "--result-vector", "value"],
                ["--min-result-l2", "value=1"],
                ["--result-vector", "value"],
                ["--expected-result-shape", "value=1"],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1",
                    "--expected-result-shape",
                    "value=2",
                ],
                ["--result-vector", "value", "--expected-result-shape", "value=0"],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1,bad",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--min-result-l2", "other=1",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--min-result-l2", "value=inf",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--max-result-abs", "other=1",
                ],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1",
                    "--max-result-abs",
                    "value=1",
                    "--max-result-abs",
                    "value=2",
                ],
            ):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, *extra, str(example)])

    def test_result_shape_contract_rejects_shrink_and_rank_change(self) -> None:
        cases = (
            ([1.0], (2,), "shape"),
            (1.0, (1,), "shape"),
            ([1.0], (), "shape"),
        )
        for value, expected_shape, message in cases:
            with self.subTest(value=value, expected_shape=expected_shape):
                args = types.SimpleNamespace(
                    result_vector=["value"],
                    expected_result_shape=[("value", expected_shape)],
                    min_result_l2=[],
                    max_result_abs=[],
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    ORACLE.collect_example_results({"value": value}, args)

    def test_per_run_timestep_ranges_are_fail_closed(self) -> None:
        records = [
            {
                "timestep": 12,
                "timestep_delta": 10,
                "field_energy": 1.0,
                "dft_norm": 1.0,
            },
            {
                "timestep": 25,
                "timestep_delta": 20,
                "field_energy": 1.0,
                "dft_norm": 1.0,
            },
        ]
        args = types.SimpleNamespace(
            expected_run_count=2,
            expected_final_timestep=None,
            expected_total_timestep_delta=None,
            expected_run_timestep_range=[(10, 15), (20, 30)],
            expected_run_timestep_delta_range=[(8, 12), (18, 22)],
            min_field_energy=None,
            min_dft_norm=None,
            min_each_field_energy=None,
            min_each_dft_norm=None,
        )
        ORACLE.validate_contract(records, args)
        args.expected_run_timestep_range[1] = (26, 30)
        with self.assertRaisesRegex(RuntimeError, "outside required range"):
            ORACLE.validate_contract(records, args)

        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            base = ["--expected-run-count", "2"]
            invalid = (
                ["--expected-run-timestep-range", "1:2"],
                [
                    "--expected-final-timestep", "1",
                    "--expected-run-timestep-range", "1:2",
                    "--expected-run-timestep-range", "1:2",
                ],
                ["--expected-run-timestep-range", "2:1"],
                ["--expected-run-timestep-range", "0:1"],
                ["--expected-run-timestep-delta-range", "0:1"],
            )
            for extra in invalid:
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, *extra, str(example)])

    def test_oversized_array_is_rejected_before_tolist(self) -> None:
        class HugeArray:
            shape = (10**12,)
            size = 10**12
            tolist_called = False

            def tolist(self):
                self.tolist_called = True
                raise AssertionError("tolist must not be called")

        value = HugeArray()
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            ORACLE.numeric_result_vector(value, "huge")
        self.assertFalse(value.tolist_called)

    def test_binary_grating_acos_rejects_large_domain_violation(self) -> None:
        source_path = ORACLE_PATH.parents[2] / "python/examples/binary_grating.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "bounded_direction_cosine_acos"
        )
        eps = 1.1920928955078125e-7
        fake_numpy = types.SimpleNamespace(
            float32=object(),
            finfo=lambda _: types.SimpleNamespace(eps=eps),
        )
        namespace = {
            "math": __import__("math"),
            "np": fake_numpy,
            "direction_cosine_max_overshoot": 0.0,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(source_path), "exec"), namespace)
        bounded_acos = namespace["bounded_direction_cosine_acos"]
        bounded_acos(1.0 + 16 * eps)
        self.assertGreater(namespace["direction_cosine_max_overshoot"], 0.0)
        for invalid in (1.0 + 64 * eps, -1.0 - 64 * eps, float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                bounded_acos(invalid)

    def test_system_exit_and_run_exception_restore_process_state(self) -> None:
        fake_meep = fake_meep_module()
        original_run = fake_meep.Simulation.run
        previous_meep = sys.modules.get("meep")
        previous_argv = sys.argv
        previous_path = list(sys.path)
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory_path = pathlib.Path(directory)
                successful = directory_path / "successful.py"
                successful.write_text(
                    "import meep as mp\nmp.Simulation().run(until=1)\n"
                    "raise SystemExit(0)\n",
                    encoding="utf-8",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "10",
                                str(successful),
                            ]
                        ),
                        0,
                    )
                failing_exit = directory_path / "failing_exit.py"
                failing_exit.write_text("raise SystemExit(7)\n", encoding="utf-8")
                with self.assertRaises(SystemExit) as raised:
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(failing_exit),
                        ]
                    )
                self.assertEqual(raised.exception.code, 7)
                failing_run = directory_path / "failing_run.py"
                failing_run.write_text(
                    "import meep as mp\n"
                    "def fail(self, *args, **kwargs): raise RuntimeError('run failed')\n"
                    "mp.Simulation.run = fail\n"
                    "mp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "run failed"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(failing_run),
                        ]
                    )
        finally:
            self.assertIs(fake_meep.Simulation.run, original_run)
            self.assertIs(sys.argv, previous_argv)
            self.assertEqual(sys.path, previous_path)
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_original_run_and_metric_exceptions_restore_wrapper(self) -> None:
        class RaisingRunSimulation(FakeSimulation):
            def run(self, *args, **kwargs):
                raise RuntimeError("original run failed")

        class RaisingMetricSimulation(FakeSimulation):
            def field_energy_in_box(self, box=None):
                raise RuntimeError("metric failed")

        for label, simulation_class, expected in (
            ("run", RaisingRunSimulation, "original run failed"),
            ("metric", RaisingMetricSimulation, "metric failed"),
        ):
            with self.subTest(label=label):
                fake_meep = fake_meep_module(simulation_class)
                original_run = fake_meep.Simulation.run
                previous_meep = sys.modules.get("meep")
                previous_argv = sys.argv
                previous_path = list(sys.path)
                sys.modules["meep"] = fake_meep
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        example = pathlib.Path(directory) / "failing.py"
                        example.write_text(
                            "import meep as mp\n"
                            "mp.Simulation().run(until=1)\n",
                            encoding="utf-8",
                        )
                        with self.assertRaisesRegex(RuntimeError, expected):
                            ORACLE.main(
                                [
                                    "--expected-run-count",
                                    "1",
                                    "--expected-final-timestep",
                                    "10",
                                    str(example),
                                ]
                            )
                finally:
                    self.assertIs(fake_meep.Simulation.run, original_run)
                    self.assertIs(sys.argv, previous_argv)
                    self.assertEqual(sys.path, previous_path)
                    if previous_meep is None:
                        sys.modules.pop("meep", None)
                    else:
                        sys.modules["meep"] = previous_meep


if __name__ == "__main__":
    unittest.main()
