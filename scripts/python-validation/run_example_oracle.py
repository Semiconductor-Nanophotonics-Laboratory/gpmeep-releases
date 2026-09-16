#!/usr/bin/env python3
"""Run a Meep example with deterministic physical and result observables.

The driver wraps ``Simulation.run`` and records scalar observables immediately
after every completed run, before an example can reset or discard its fields.
It does not alter the timestep loop.  Selected top-level numerical result
vectors can also be captured after the example completes, allowing examples'
published spectra, coefficients, or other derived observables to be compared
without parsing presentation-oriented stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import pathlib
import runpy
import sys
import time
from typing import Any


METRIC_PREFIX = "gpmeep-generic-example-metrics:"
AUDIT_PREFIX = "gpmeep-generic-example-audit:"
CHECKPOINT_PREFIX = "gpmeep-generic-example-run-checkpoint:"
MAX_RESULT_VALUES = 1_000_000


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"generic example oracle produced non-numeric {name}"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError(f"generic example oracle produced non-finite {name}")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def positive_range(value: str) -> tuple[int, int]:
    raw_minimum, separator, raw_maximum = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("must use MIN:MAX")
    try:
        minimum = positive_int(raw_minimum)
        maximum = positive_int(raw_maximum)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "range endpoints must be positive integers"
        ) from exc
    if maximum < minimum:
        raise argparse.ArgumentTypeError("range maximum must be >= minimum")
    return minimum, maximum


def positive_finite(value: str) -> float:
    try:
        result = finite_float(value, "signal threshold")
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def result_name(value: str) -> str:
    if not value.isidentifier() or value.startswith("_"):
        raise argparse.ArgumentTypeError(
            "result name must be a public Python identifier"
        )
    return value


def result_threshold(value: str) -> tuple[str, float]:
    name, separator, raw_threshold = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must use NAME=POSITIVE_THRESHOLD")
    return result_name(name), positive_finite(raw_threshold)


def result_shape_contract(value: str) -> tuple[str, tuple[int, ...]]:
    name, separator, raw_shape = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must use NAME=scalar or NAME=DIM[,DIM...]")
    name = result_name(name)
    if raw_shape == "scalar":
        return name, ()
    if not raw_shape:
        raise argparse.ArgumentTypeError("result shape must not be empty")
    dimensions: list[int] = []
    for raw_dimension in raw_shape.split(","):
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "result dimensions must be positive integers"
            ) from exc
        if dimension <= 0:
            raise argparse.ArgumentTypeError(
                "result dimensions must be positive integers"
            )
        dimensions.append(dimension)
    if math.prod(dimensions) > MAX_RESULT_VALUES:
        raise argparse.ArgumentTypeError(
            f"result shape exceeds the {MAX_RESULT_VALUES}-value limit"
        )
    return name, tuple(dimensions)


def numeric_result_vector(value: Any, name: str) -> dict[str, Any]:
    """Convert a rectangular scalar/vector result into stable JSON metrics."""

    if hasattr(value, "tolist"):
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is not None and size is not None:
            try:
                dimensions = tuple(operator.index(item) for item in shape)
                item_count = operator.index(size)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                ) from exc
            if isinstance(size, bool) or any(
                isinstance(item, bool) for item in shape
            ):
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                )
            if item_count < 0 or any(dimension < 0 for dimension in dimensions):
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                )
            if item_count > MAX_RESULT_VALUES:
                raise RuntimeError(
                    f"example result {name!r} exceeds the "
                    f"{MAX_RESULT_VALUES}-value limit"
                )
            expected_count = math.prod(dimensions)
            if expected_count != item_count:
                raise RuntimeError(
                    f"example result {name!r} has inconsistent array shape and size"
                )
            if item_count == 0:
                raise RuntimeError(f"example result {name!r} is empty")
        try:
            value = value.tolist()
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"example result {name!r} could not be converted to a list"
            ) from exc

    flat: list[complex] = []

    def visit(item: Any, path: str) -> tuple[int, ...]:
        if hasattr(item, "item") and not isinstance(
            item, (str, bytes, list, tuple)
        ):
            try:
                item = item.item()
            except (TypeError, ValueError):
                pass
        if isinstance(item, (list, tuple)):
            if not item:
                raise RuntimeError(f"example result {name!r} is empty at {path}")
            child_shapes = [
                visit(child, f"{path}[{index}]")
                for index, child in enumerate(item)
            ]
            if any(shape != child_shapes[0] for shape in child_shapes[1:]):
                raise RuntimeError(
                    f"example result {name!r} is ragged at {path}"
                )
            return (len(item), *child_shapes[0])
        if isinstance(item, (str, bytes, bool)):
            raise RuntimeError(
                f"example result {name!r} contains a non-numeric value at {path}"
            )
        try:
            number = complex(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"example result {name!r} contains a non-numeric value at {path}"
            ) from exc
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            raise RuntimeError(
                f"example result {name!r} contains a non-finite value at {path}"
            )
        flat.append(number)
        if len(flat) > MAX_RESULT_VALUES:
            raise RuntimeError(
                f"example result {name!r} exceeds the "
                f"{MAX_RESULT_VALUES}-value limit"
            )
        return ()

    shape = visit(value, name)
    if not flat:
        raise RuntimeError(f"example result {name!r} contains no scalar values")
    real = [number.real for number in flat]
    imag = [number.imag for number in flat]
    magnitudes = [abs(number) for number in flat]
    result = {
        "count": len(flat),
        "real": real,
        "imag": imag,
        "l2": math.sqrt(math.fsum(value * value for value in magnitudes)),
        "max_abs": max(magnitudes),
        "weighted_real": math.fsum(
            (index + 1) * value for index, value in enumerate(real)
        ),
        "weighted_imag": math.fsum(
            (index + 1) * value for index, value in enumerate(imag)
        ),
    }
    if shape:
        result["shape"] = list(shape)
    else:
        result["rank"] = 0
    return result


def collect_example_results(
    namespace: dict[str, Any] | None, args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    if not args.result_vector:
        return {}
    if namespace is None:
        raise RuntimeError(
            "example exited through SystemExit before result vectors could be captured"
        )
    results: dict[str, dict[str, Any]] = {}
    minimum_l2 = dict(getattr(args, "min_result_l2", []))
    maximum_abs = dict(getattr(args, "max_result_abs", []))
    expected_shapes = dict(getattr(args, "expected_result_shape", []))
    for name in args.result_vector:
        if name not in namespace:
            raise RuntimeError(f"example did not define requested result {name!r}")
        result = numeric_result_vector(namespace[name], name)
        actual_shape = tuple(result.get("shape", ()))
        expected_shape = expected_shapes.get(name)
        if expected_shape is None:
            raise RuntimeError(
                f"example result {name!r} has no independent shape contract"
            )
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"example result {name!r} has shape {actual_shape}, expected "
                f"{expected_shape}"
            )
        threshold = minimum_l2.get(name)
        if threshold is not None and result["l2"] < threshold:
            raise RuntimeError(
                f"example result {name!r} L2 norm {result['l2']} is below "
                f"required signal {threshold}"
            )
        threshold = maximum_abs.get(name)
        if threshold is not None and result["max_abs"] > threshold:
            raise RuntimeError(
                f"example result {name!r} maximum absolute value "
                f"{result['max_abs']} exceeds required limit {threshold}"
            )
        results[name] = result
    return results


def phase_call_totals(statistics: Any) -> dict[str, int]:
    """Return backend phase-call totals without counting transfer/readback data."""

    totals = {"cpu": 0, "cuda": 0}

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key, value in item.items():
            if isinstance(value, dict):
                visit(value)
            elif key.endswith("_calls"):
                for backend in totals:
                    if key.startswith(f"{backend}_"):
                        count = int(value)
                        if count < 0:
                            raise RuntimeError(
                                f"negative {backend} phase-call counter {key}"
                            )
                        totals[backend] += count

    visit(statistics)
    return totals


def counter_delta(
    before: dict[str, int], after: dict[str, int], label: str
) -> dict[str, int]:
    result = {backend: after[backend] - before[backend] for backend in before}
    for backend, value in result.items():
        if value < 0:
            raise RuntimeError(
                f"GPU statistics reset during {label}: negative {backend} delta"
            )
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("example completed without calling Simulation.run")

    def scalar_summary(key: str) -> dict[str, float]:
        values = [float(record[key]) for record in records]
        return {
            "final": values[-1],
            "sum": math.fsum(values),
            "l2": math.sqrt(math.fsum(value * value for value in values)),
            "weighted_checksum": math.fsum(
                (index + 1) * value for index, value in enumerate(values)
            ),
        }

    timesteps = [int(record["timestep"]) for record in records]
    timestep_deltas = [int(record["timestep_delta"]) for record in records]
    return {
        "run_count": len(records),
        "timestep": {
            "final": timesteps[-1],
            "sum": sum(timesteps),
            "weighted_checksum": sum(
                (index + 1) * value
                for index, value in enumerate(timesteps)
            ),
        },
        "timestep_delta": {
            "final": timestep_deltas[-1],
            "sum": sum(timestep_deltas),
            "weighted_checksum": sum(
                (index + 1) * value
                for index, value in enumerate(timestep_deltas)
            ),
        },
        "meep_time": scalar_summary("meep_time"),
        "field_energy": scalar_summary("field_energy"),
        "dft_norm": scalar_summary("dft_norm"),
        "per_run": {
            "timestep": timesteps,
            "timestep_delta": timestep_deltas,
            "meep_time": [float(record["meep_time"]) for record in records],
            "field_energy": [
                float(record["field_energy"]) for record in records
            ],
            "dft_norm": [float(record["dft_norm"]) for record in records],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-run-count", required=True, type=positive_int)
    parser.add_argument("--expected-final-timestep", type=nonnegative_int)
    parser.add_argument("--expected-total-timestep-delta", type=positive_int)
    parser.add_argument(
        "--expected-run-timestep-range",
        action="append",
        default=[],
        type=positive_range,
    )
    parser.add_argument(
        "--expected-run-timestep-delta-range",
        action="append",
        default=[],
        type=positive_range,
    )
    parser.add_argument("--min-field-energy", type=positive_finite)
    parser.add_argument("--min-dft-norm", type=positive_finite)
    parser.add_argument("--min-each-field-energy", type=positive_finite)
    parser.add_argument("--min-each-dft-norm", type=positive_finite)
    parser.add_argument(
        "--result-vector", action="append", default=[], type=result_name
    )
    parser.add_argument(
        "--expected-result-shape",
        action="append",
        default=[],
        type=result_shape_contract,
    )
    parser.add_argument(
        "--min-result-l2", action="append", default=[], type=result_threshold
    )
    parser.add_argument(
        "--max-result-abs", action="append", default=[], type=result_threshold
    )
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("example_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.example_args[:1] == ["--"]:
        args.example_args = args.example_args[1:]
    args.example = args.example.resolve()
    if not args.example.is_file() or args.example.suffix != ".py":
        parser.error(f"example is not a Python file: {args.example}")
    if all(
        value is None
        for value in (
            args.expected_final_timestep,
            args.expected_total_timestep_delta,
            args.expected_run_timestep_range or None,
            args.expected_run_timestep_delta_range or None,
            args.min_field_energy,
            args.min_dft_norm,
            args.min_each_field_energy,
            args.min_each_dft_norm,
        )
    ):
        parser.error(
            "at least one timestep or physical-signal contract is required"
        )
    for option_name, ranges in (
        ("--expected-run-timestep-range", args.expected_run_timestep_range),
        (
            "--expected-run-timestep-delta-range",
            args.expected_run_timestep_delta_range,
        ),
    ):
        if ranges and len(ranges) != args.expected_run_count:
            parser.error(
                f"{option_name} must appear exactly --expected-run-count times"
            )
    if args.expected_final_timestep is not None and args.expected_run_timestep_range:
        parser.error(
            "--expected-final-timestep and --expected-run-timestep-range are "
            "mutually exclusive"
        )
    if (
        args.expected_total_timestep_delta is not None
        and args.expected_run_timestep_delta_range
    ):
        parser.error(
            "--expected-total-timestep-delta and "
            "--expected-run-timestep-delta-range are mutually exclusive"
        )
    if len(set(args.result_vector)) != len(args.result_vector):
        parser.error("--result-vector names must be unique")
    shape_names = [name for name, _ in args.expected_result_shape]
    if len(set(shape_names)) != len(shape_names):
        parser.error("--expected-result-shape names must be unique")
    missing_shapes = sorted(set(args.result_vector) - set(shape_names))
    unknown_shapes = sorted(set(shape_names) - set(args.result_vector))
    if missing_shapes or unknown_shapes:
        details = []
        if missing_shapes:
            details.append("missing for: " + ", ".join(missing_shapes))
        if unknown_shapes:
            details.append("unknown for: " + ", ".join(unknown_shapes))
        parser.error(
            "--expected-result-shape must match every --result-vector exactly ("
            + "; ".join(details)
            + ")"
        )
    threshold_names = [name for name, _ in args.min_result_l2]
    if len(set(threshold_names)) != len(threshold_names):
        parser.error("--min-result-l2 names must be unique")
    unknown_thresholds = sorted(set(threshold_names) - set(args.result_vector))
    if unknown_thresholds:
        parser.error(
            "--min-result-l2 requires a matching --result-vector for: "
            + ", ".join(unknown_thresholds)
        )
    maximum_names = [name for name, _ in args.max_result_abs]
    if len(set(maximum_names)) != len(maximum_names):
        parser.error("--max-result-abs names must be unique")
    unknown_maximums = sorted(set(maximum_names) - set(args.result_vector))
    if unknown_maximums:
        parser.error(
            "--max-result-abs requires a matching --result-vector for: "
            + ", ".join(unknown_maximums)
        )
    return args


def validate_contract(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not records:
        raise RuntimeError("example completed without calling Simulation.run")
    if len(records) != args.expected_run_count:
        raise RuntimeError(
            f"expected {args.expected_run_count} Simulation.run calls, got "
            f"{len(records)}"
        )
    timestep_ranges = getattr(args, "expected_run_timestep_range", [])
    timestep_delta_ranges = getattr(
        args, "expected_run_timestep_delta_range", []
    )
    for label, key, ranges in (
        ("timestep", "timestep", timestep_ranges),
        ("timestep delta", "timestep_delta", timestep_delta_ranges),
    ):
        for index, (record, bounds) in enumerate(zip(records, ranges)):
            value = int(record[key])
            minimum, maximum = bounds
            if not minimum <= value <= maximum:
                raise RuntimeError(
                    f"run {index} {label} {value} is outside required range "
                    f"[{minimum}, {maximum}]"
                )
    timestep_deltas = [int(record["timestep_delta"]) for record in records]
    zero_work = [index for index, value in enumerate(timestep_deltas) if value <= 0]
    if zero_work:
        raise RuntimeError(
            "Simulation.run advanced zero timesteps at record indices "
            f"{zero_work}"
        )
    total_delta = sum(timestep_deltas)
    if (
        args.expected_total_timestep_delta is not None
        and total_delta != args.expected_total_timestep_delta
    ):
        raise RuntimeError(
            f"expected total timestep delta {args.expected_total_timestep_delta}, "
            f"got {total_delta}"
        )
    final_timestep = int(records[-1]["timestep"])
    if (
        args.expected_final_timestep is not None
        and final_timestep != args.expected_final_timestep
    ):
        raise RuntimeError(
            f"expected final timestep {args.expected_final_timestep}, got "
            f"{final_timestep}"
        )
    if args.min_field_energy is not None:
        maximum = max(abs(float(record["field_energy"])) for record in records)
        if maximum < args.min_field_energy:
            raise RuntimeError(
                f"maximum field energy {maximum} is below required signal "
                f"{args.min_field_energy}"
            )
    if args.min_dft_norm is not None:
        maximum = max(abs(float(record["dft_norm"])) for record in records)
        if maximum < args.min_dft_norm:
            raise RuntimeError(
                f"maximum DFT norm {maximum} is below required signal "
                f"{args.min_dft_norm}"
            )
    for key, threshold in (
        ("field_energy", args.min_each_field_energy),
        ("dft_norm", args.min_each_dft_norm),
    ):
        if threshold is None:
            continue
        values = [abs(float(record[key])) for record in records]
        below = [index for index, value in enumerate(values) if value < threshold]
        if below:
            raise RuntimeError(
                f"{key} is below required per-run signal {threshold} at "
                f"record indices {below}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import meep as mp

    is_master = bool(mp.am_master()) if hasattr(mp, "am_master") else True

    records: list[dict[str, Any]] = []
    original_run = mp.Simulation.run

    def audited_run(simulation: Any, *step_funcs: Any, **kwargs: Any) -> Any:
        pre_timestep = (
            int(simulation.fields.t) if simulation.fields is not None else 0
        )
        pre_statistics = mp.gpu.statistics()
        pre_calls = phase_call_totals(pre_statistics)
        started_ns = time.monotonic_ns()
        result = original_run(simulation, *step_funcs, **kwargs)
        ended_ns = time.monotonic_ns()
        run_wall_seconds = (ended_ns - started_ns) / 1_000_000_000.0
        if not math.isfinite(run_wall_seconds) or run_wall_seconds <= 0:
            raise RuntimeError("Simulation.run wall time is not positive and finite")
        post_timestep = int(simulation.timestep())
        post_statistics = mp.gpu.statistics()
        post_calls = phase_call_totals(post_statistics)
        run_calls = counter_delta(pre_calls, post_calls, "Simulation.run")
        active_backend = str(mp.gpu.active_backend)
        if active_backend not in ("cpu", "cuda"):
            raise RuntimeError(
                f"Simulation.run selected unsupported backend {active_backend!r}"
            )
        other_backend = "cuda" if active_backend == "cpu" else "cpu"
        if post_timestep > pre_timestep and run_calls[active_backend] <= 0:
            raise RuntimeError(
                "Simulation.run advanced timesteps without selected-backend "
                f"phase dispatch ({active_backend})"
            )
        if run_calls[other_backend] != 0:
            raise RuntimeError(
                "Simulation.run used backend fallback: "
                f"{other_backend} calls={run_calls[other_backend]}"
            )
        record = {
            "timestep": post_timestep,
            "timestep_delta": post_timestep - pre_timestep,
            "meep_time": finite_float(simulation.round_time(), "Meep time"),
            "run_wall_seconds": run_wall_seconds,
            "field_energy": finite_float(
                simulation.field_energy_in_box(
                    box=simulation.fields.total_volume()
                ),
                "field energy",
            ),
            "dft_norm": finite_float(
                simulation.fields.dft_norm(), "DFT norm"
            ),
            "active_backend": active_backend,
            "requested_backend": str(
                getattr(mp.gpu, "requested_backend", active_backend)
            ),
            "execution_diagnostic": str(
                simulation.fields.gpu_execution_diagnostic()
                if hasattr(simulation.fields, "gpu_execution_diagnostic")
                else "unavailable"
            ),
            "runtime_counters": {
                str(name): int(value)
                for name, value in post_statistics.get("runtime", {}).items()
            },
            "run_phase_calls": run_calls,
        }
        measurement_calls = counter_delta(
            post_calls,
            phase_call_totals(mp.gpu.statistics()),
            "oracle measurement",
        )
        record["measurement_phase_calls"] = measurement_calls
        if record["timestep"] < 0 or record["timestep_delta"] < 0:
            raise RuntimeError(
                "generic example oracle produced a negative timestep or delta"
            )
        records.append(record)
        if is_master:
            print(
                CHECKPOINT_PREFIX
                + json.dumps(
                    {
                        "schema": "gpmeep-generic-example-run-checkpoint-v1",
                        "run_index": len(records) - 1,
                        "record": record,
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return result

    previous_argv = sys.argv
    previous_path = list(sys.path)
    mp.Simulation.run = audited_run
    sys.argv = [str(args.example), *args.example_args]
    sys.path.insert(0, str(args.example.parent))
    namespace: dict[str, Any] | None = None
    try:
        try:
            namespace = runpy.run_path(str(args.example), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        mp.Simulation.run = original_run
        sys.argv = previous_argv
        sys.path[:] = previous_path
    validate_contract(records, args)
    example_results = collect_example_results(namespace, args)
    if not is_master:
        return 0
    print(AUDIT_PREFIX + json.dumps({"runs": records}, sort_keys=True, allow_nan=False))
    metrics = aggregate(records)
    if example_results:
        metrics["example_results"] = example_results
    print(METRIC_PREFIX + json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
