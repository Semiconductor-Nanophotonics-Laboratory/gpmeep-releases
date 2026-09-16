#!/usr/bin/env python3
"""Pure schedule and statistics replay for M3 repeated example performance."""

from __future__ import annotations

import math
import pathlib
import statistics
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import m3_performance_plan as performance_plan  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
from common import WorkloadError  # noqa: E402


SCHEMA = "gpmeep-m3-repeated-performance-replay-v1"
REPEATED_UNIT_IDS = (
    "performance-edge-emitter-3d",
    "performance-metasurface-crossover",
    "performance-long-horizon-policy",
)
MAXIMUM_CV = 0.15
MINIMUM_GPU1_SPEEDUP = 1.5
MINIMUM_GPU2_SPEEDUP = 2.0
MINIMUM_GPU2_SCALING = 1.1
MAXIMUM_AUTO_CPU_SLOWDOWN = 1.10
MAXIMUM_AUTO_GPU_SLOWDOWN = 1.25


def repeated_units(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema") != performance_plan.SCHEMA:
        raise WorkloadError("M3 repeated performance plan schema differs")
    units = [
        unit
        for unit in plan.get("units", [])
        if isinstance(unit, dict) and unit.get("driver", "").startswith("repeated-")
    ]
    if (
        [unit.get("unit_id") for unit in units] != list(REPEATED_UNIT_IDS)
        or any(unit.get("warmup_cycles") != 1 for unit in units)
        or any(unit.get("measured_cycles") != 5 for unit in units)
    ):
        raise WorkloadError("M3 repeated performance unit inventory differs")
    return units


def build_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for unit in repeated_units(plan):
        topologies = unit["topologies"]
        if not isinstance(topologies, list) or not topologies:
            raise WorkloadError("M3 repeated topology inventory differs")
        cycle_count = unit["warmup_cycles"] + unit["measured_cycles"]
        for cycle in range(cycle_count):
            sample_kind = "warmup" if cycle < unit["warmup_cycles"] else "measured"
            measured_index = 0 if sample_kind == "warmup" else cycle
            shift = cycle % len(topologies)
            order = topologies[shift:] + topologies[:shift]
            for topology in order:
                tasks.append(
                    {
                        "ordinal": len(tasks) + 1,
                        "unit_id": unit["unit_id"],
                        "target": unit["target"],
                        "topology": topology,
                        "sample_kind": sample_kind,
                        "cycle_index": measured_index,
                        "timeout_seconds": unit["timeout_seconds"],
                        "timing_contract": unit["timing_contract"],
                    }
                )
    if len(tasks) != 72 or [task["ordinal"] for task in tasks] != list(
        range(1, 73)
    ):
        raise WorkloadError("M3 repeated physical task inventory differs")
    return tasks


def task_name(task: dict[str, Any]) -> str:
    kind = "w" if task["sample_kind"] == "warmup" else "m"
    return (
        f"{int(task['ordinal']):03d}-{task['unit_id']}-"
        f"{kind}{int(task['cycle_index']):02d}-{task['topology']}"
    )


def _sample_map(
    unit: dict[str, Any], reports: list[dict[str, Any]]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for report in reports:
        key = (
            report.get("sample_kind"),
            report.get("cycle_index"),
            report.get("topology"),
        )
        if (
            report.get("outcome") != "PASS"
            or report.get("case_path") != unit["target"]
            or report.get("timing", {}).get("contract")
            != unit["timing_contract"]
            or key in result
        ):
            raise WorkloadError("M3 repeated sample report identity differs")
        result[key] = report
    expected_keys = {
        ("warmup", 0, topology) for topology in unit["topologies"]
    } | {
        ("measured", cycle, topology)
        for cycle in range(1, unit["measured_cycles"] + 1)
        for topology in unit["topologies"]
    }
    if set(result) != expected_keys:
        raise WorkloadError("M3 repeated sample report inventory is incomplete")
    return result


def _values(
    samples: dict[tuple[str, int, str], dict[str, Any]],
    topology: str,
    field: str = "primary_seconds",
) -> list[float]:
    values = [
        samples[("measured", cycle, topology)]["timing"].get(field)
        for cycle in range(1, 6)
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in values
    ):
        raise WorkloadError("M3 repeated timing values differ")
    return [float(value) for value in values]


def _speed_gate(
    reference: list[float], candidate: list[float], minimum: float
) -> dict[str, Any]:
    result: dict[str, Any] = matrix.conservative_speedup(reference, candidate)
    result["minimum"] = minimum
    result["outcome"] = (
        "PASS" if result["conservative_speedup"] >= minimum else "FAIL"
    )
    return result


def _slowdown_gate(
    reference: list[float], candidate: list[float], maximum: float
) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate):
        raise WorkloadError("M3 repeated slowdown samples are incomplete")
    ratios = [right / left for left, right in zip(reference, candidate, strict=True)]
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise WorkloadError("M3 repeated slowdown ratios differ")
    return {
        "reference_median_seconds": statistics.median(reference),
        "candidate_median_seconds": statistics.median(candidate),
        "median_slowdown": statistics.median(candidate)
        / statistics.median(reference),
        "maximum_paired_slowdown": max(ratios),
        "minimum_paired_slowdown": min(ratios),
        "maximum": maximum,
        "outcome": "PASS" if max(ratios) <= maximum else "FAIL",
    }


def derive_statistics(
    plan: dict[str, Any], unit_id: str, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    units = {unit["unit_id"]: unit for unit in repeated_units(plan)}
    try:
        unit = units[unit_id]
    except KeyError as exc:
        raise WorkloadError("M3 repeated statistics unit differs") from exc
    samples = _sample_map(unit, reports)
    timings = {
        topology: _values(samples, topology) for topology in unit["topologies"]
    }
    stability = {
        topology: matrix.timing_stability(values, MAXIMUM_CV)
        for topology, values in timings.items()
    }
    speedups: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    required: list[dict[str, Any]] = list(stability.values())
    if unit_id == "performance-edge-emitter-3d":
        speedups = {
            "cpu8-to-cuda1": _speed_gate(
                timings["cpu8"], timings["cuda1"], MINIMUM_GPU1_SPEEDUP
            ),
            "cpu8-to-cuda2": _speed_gate(
                timings["cpu8"], timings["cuda2"], MINIMUM_GPU2_SPEEDUP
            ),
            "cuda1-to-cuda2": _speed_gate(
                timings["cuda1"], timings["cuda2"], MINIMUM_GPU2_SCALING
            ),
        }
        required.extend(speedups.values())
    elif unit_id == "performance-metasurface-crossover":
        speedups = {
            "cpu8-to-cuda1-large": _speed_gate(
                timings["cpu8"], timings["cuda1"], MINIMUM_GPU1_SPEEDUP
            ),
            "cpu8-to-cuda2-large": _speed_gate(
                timings["cpu8"], timings["cuda2"], MINIMUM_GPU2_SPEEDUP
            ),
            "cuda1-to-cuda2-large": _speed_gate(
                timings["cuda1"], timings["cuda2"], MINIMUM_GPU2_SCALING
            ),
            "cpu8-to-auto1-large": _speed_gate(
                timings["cpu8"], timings["auto1"], MINIMUM_GPU1_SPEEDUP
            ),
        }
        policy = {
            "auto-small-vs-cpu1": _slowdown_gate(
                _values(samples, "cpu1", "small_seconds"),
                _values(samples, "auto1", "small_seconds"),
                MAXIMUM_AUTO_CPU_SLOWDOWN,
            ),
            "auto-large-vs-cuda1": _slowdown_gate(
                _values(samples, "cuda1", "large_seconds"),
                _values(samples, "auto1", "large_seconds"),
                MAXIMUM_AUTO_GPU_SLOWDOWN,
            ),
        }
        required.extend(speedups.values())
        required.extend(policy.values())
    elif unit_id == "performance-long-horizon-policy":
        diagnostic = matrix.conservative_speedup(
            timings["cpu8"], timings["cuda1"]
        )
        policy = {
            "auto-vs-cpu1": _slowdown_gate(
                timings["cpu1"], timings["auto1"], MAXIMUM_AUTO_CPU_SLOWDOWN
            ),
            "forced-cuda-diagnostic": {
                **diagnostic,
                "release_gate": False,
                "missed_speedup_m4_gap": (
                    diagnostic["conservative_speedup"] >= MINIMUM_GPU1_SPEEDUP
                ),
            },
        }
        required.append(policy["auto-vs-cpu1"])
    else:  # pragma: no cover - repeated_units already seals this branch
        raise WorkloadError("M3 repeated statistics unit is unsupported")
    outcome = (
        "PASS"
        if all(record.get("outcome") == "PASS" for record in required)
        else "FAIL"
    )
    return {
        "schema": SCHEMA,
        "unit_id": unit_id,
        "outcome": outcome,
        "measured_cycles": unit["measured_cycles"],
        "timings_seconds": timings,
        "timing_stability": stability,
        "speedups": speedups,
        "automatic_policy": policy,
    }
