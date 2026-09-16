#!/usr/bin/env python3
"""Tests for deterministic M3 repeated-performance replay."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "scripts"
    / "python-validation"
    / "m3_repeated_performance_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m3_repeated_performance_replay", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


def release_plan() -> dict[str, object]:
    return {
        "schema": REPLAY.performance_plan.SCHEMA,
        "units": [
            {
                "unit_id": "performance-edge-emitter-3d",
                "driver": "repeated-mpi-example",
                "target": "python/examples/edge_emitter_3D.py",
                "topologies": ["cpu8", "cuda1", "cuda2"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": "sum-all-simulation-run-wall-seconds",
            },
            {
                "unit_id": "performance-metasurface-crossover",
                "driver": "repeated-mpi-example-auto",
                "target": "python/examples/metasurface_lens.py",
                "topologies": ["cpu8", "cpu1", "cuda1", "cuda2", "auto1"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": (
                    "run-index-18-small-and-19-large-wall-seconds"
                ),
            },
            {
                "unit_id": "performance-long-horizon-policy",
                "driver": "repeated-mpi-example-auto",
                "target": "python/examples/stochastic_emitter_line.py",
                "topologies": ["cpu8", "cpu1", "cuda1", "auto1"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": "sum-all-simulation-run-wall-seconds",
            },
        ],
    }


def reports_for(
    unit: dict[str, object],
    primary: dict[str, float],
    *,
    small: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    reports = []
    for kind, cycles in (("warmup", (0,)), ("measured", range(1, 6))):
        for cycle in cycles:
            for topology in unit["topologies"]:
                timing = {
                    "contract": unit["timing_contract"],
                    "primary_seconds": primary[topology],
                }
                if small is not None:
                    timing.update(
                        {
                            "small_seconds": small[topology],
                            "large_seconds": primary[topology],
                        }
                    )
                reports.append(
                    {
                        "outcome": "PASS",
                        "case_path": unit["target"],
                        "topology": topology,
                        "sample_kind": kind,
                        "cycle_index": cycle,
                        "timing": timing,
                    }
                )
    return reports


class M3RepeatedPerformanceReplayTests(unittest.TestCase):
    def test_schedule_is_complete_interleaved_and_deterministic(self) -> None:
        tasks = REPLAY.build_tasks(release_plan())
        self.assertEqual(len(tasks), 72)
        self.assertEqual([task["ordinal"] for task in tasks], list(range(1, 73)))
        self.assertEqual(
            [task["topology"] for task in tasks[:3]],
            ["cpu8", "cuda1", "cuda2"],
        )
        self.assertEqual(
            [task["topology"] for task in tasks[3:6]],
            ["cuda1", "cuda2", "cpu8"],
        )
        self.assertEqual(len({REPLAY.task_name(task) for task in tasks}), 72)

    def test_edge_speed_and_stability_gates_pass(self) -> None:
        plan = release_plan()
        unit = plan["units"][0]
        reports = reports_for(
            unit, {"cpu8": 10.0, "cuda1": 5.0, "cuda2": 4.0}
        )
        result = REPLAY.derive_statistics(plan, unit["unit_id"], reports)
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(
            result["speedups"]["cpu8-to-cuda1"]["conservative_speedup"],
            2.0,
        )
        reports[1]["timing"]["primary_seconds"] = 6.0
        with self.assertRaisesRegex(Exception, "inventory"):
            REPLAY.derive_statistics(plan, unit["unit_id"], reports[:-1])

    def test_metasurface_policy_uses_same_topology_owner_controls(self) -> None:
        plan = release_plan()
        unit = plan["units"][1]
        reports = reports_for(
            unit,
            {
                "cpu8": 10.0,
                "cpu1": 12.0,
                "cuda1": 5.0,
                "cuda2": 4.0,
                "auto1": 5.5,
            },
            small={
                "cpu8": 0.9,
                "cpu1": 1.0,
                "cuda1": 2.0,
                "cuda2": 2.5,
                "auto1": 1.05,
            },
        )
        result = REPLAY.derive_statistics(plan, unit["unit_id"], reports)
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(
            result["automatic_policy"]["auto-small-vs-cpu1"]["outcome"],
            "PASS",
        )
        self.assertEqual(
            result["automatic_policy"]["auto-large-vs-cuda1"]["outcome"],
            "PASS",
        )

    def test_long_horizon_speedup_is_an_explicit_nonblocking_gap(self) -> None:
        plan = release_plan()
        unit = plan["units"][2]
        reports = reports_for(
            unit, {"cpu8": 8.0, "cpu1": 10.0, "cuda1": 4.0, "auto1": 10.5}
        )
        result = REPLAY.derive_statistics(plan, unit["unit_id"], reports)
        self.assertEqual(result["outcome"], "PASS")
        self.assertTrue(
            result["automatic_policy"]["forced-cuda-diagnostic"][
                "missed_speedup_m4_gap"
            ]
        )
        self.assertFalse(
            result["automatic_policy"]["forced-cuda-diagnostic"][
                "release_gate"
            ]
        )

    def test_slowdown_gate_uses_worst_paired_sample(self) -> None:
        reference = [1.0] * 5
        candidate = [1.0, 1.0, 1.0, 1.0, 1.11]
        result = REPLAY._slowdown_gate(reference, candidate, 1.10)
        self.assertEqual(result["median_slowdown"], 1.0)
        self.assertEqual(result["outcome"], "FAIL")


if __name__ == "__main__":
    unittest.main()
