from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def load_source(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load_source(
    "gpmeep_cpu_topology_policy_test",
    SCRIPTS / "gpmeep_cpu_topology_policy.py",
)


def synthetic_topology(
    node_layout: list[tuple[int, int, int]],
    *,
    siblings: int = 1,
    logical_start: int = 0,
    logical_stride: int = 1,
):
    """Build (socket, node, core-count) topology with controllable sparse IDs."""

    cores = []
    cpu = logical_start
    core_ids: dict[int, int] = {}
    for socket, node, count in node_layout:
        first_core = core_ids.get(socket, 0)
        for offset in range(count):
            logical = [
                cpu + sibling * logical_stride for sibling in range(siblings)
            ]
            cpu += siblings * logical_stride
            cores.append(
                {
                    "socket": socket,
                    "node": node,
                    "core": first_core + offset,
                    "logical_cpus": logical,
                }
            )
        core_ids[socket] = first_core + count
    allowed = sorted(cpu for core in cores for cpu in core["logical_cpus"])
    logical_rows = [
        {
            "cpu": cpu,
            "socket": core["socket"],
            "node": core["node"],
            "core": core["core"],
        }
        for core in cores
        for cpu in core["logical_cpus"]
    ]
    return {
        "available_logical_cpus": allowed,
        "logical_cpu_count": len(allowed),
        "physical_core_count": len(cores),
        "cores": cores,
        "logical_cpus": sorted(logical_rows, key=lambda row: row["cpu"]),
    }


class TopologyPolicyTests(unittest.TestCase):
    def assert_plan_invariants(self, policy):
        ids = [config["config_id"] for config in policy["configs"]]
        self.assertEqual(len(ids), len(set(ids)))
        canonical = [
            config
            for config in policy["configs"]
            if config["canonical_full_physical"]
        ]
        self.assertEqual(len(canonical), 1)
        self.assertTrue(canonical[0]["full_physical"])
        for config in policy["configs"]:
            self.assertEqual(len(config["rank_core_plan"]), config["ranks"])
            self.assertEqual(len(config["rank_logical_plan"]), config["ranks"])
            self.assertEqual(len(config["rank_numa_plan"]), config["ranks"])
            for cores, node in zip(
                config["rank_core_plan"], config["rank_numa_plan"]
            ):
                self.assertEqual({core[1] for core in cores}, {node})
            logical = [cpu for rank in config["rank_logical_plan"] for cpu in rank]
            self.assertEqual(len(logical), len(set(logical)))
            if config["lane"] == "physical":
                physical = [
                    tuple(core)
                    for rank in config["rank_core_plan"]
                    for core in rank
                ]
                self.assertEqual(len(physical), len(set(physical)))
                self.assertEqual(len(physical), config["physical_budget"])

    def test_one_socket_eight_core_without_smt(self):
        policy = POLICY.generate_policy(synthetic_topology([(0, 0, 8)]))
        self.assertEqual(policy["physical_budgets"], [2, 4, 8])
        self.assertEqual(policy["canonical_full_physical_config_id"], "1x8")
        self.assertFalse(any(config["lane"] == "smt" for config in policy["configs"]))
        self.assertEqual(policy["configs"][0]["lane"], "diagnostic")
        self.assert_plan_invariants(policy)

    def test_one_socket_eight_core_smt_has_separate_lane(self):
        policy = POLICY.generate_policy(
            synthetic_topology([(0, 0, 8)], siblings=2)
        )
        ids = [config["config_id"] for config in policy["configs"]]
        self.assertEqual(
            ids,
            ["1x1", "2x1", "4x1", "8x1", "16x1", "1x8", "2x4", "4x2"],
        )
        smt = next(config for config in policy["configs"] if config["lane"] == "smt")
        self.assertFalse(smt["canonical_full_physical"])
        self.assertEqual(
            sorted(cpu for rank in smt["rank_logical_plan"] for cpu in rank),
            list(range(16)),
        )
        self.assert_plan_invariants(policy)

    def test_smt_rank_plan_follows_core_major_hwloc_order(self):
        topology = synthetic_topology([(0, 0, 4)], siblings=1)
        for core in topology["cores"]:
            core_id = core["core"]
            core["logical_cpus"] = [core_id, core_id + 4]
        topology["available_logical_cpus"] = list(range(8))
        topology["logical_cpu_count"] = 8
        topology["logical_cpus"] = sorted(
            (
                {
                    "cpu": cpu,
                    "socket": core["socket"],
                    "node": core["node"],
                    "core": core["core"],
                }
                for core in topology["cores"]
                for cpu in core["logical_cpus"]
            ),
            key=lambda row: row["cpu"],
        )
        policy = POLICY.generate_policy(topology)
        smt = next(config for config in policy["configs"] if config["lane"] == "smt")
        self.assertEqual(
            smt["rank_logical_plan"],
            [[0], [4], [1], [5], [2], [6], [3], [7]],
        )
        self.assertEqual(
            smt["rank_core_plan"],
            [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 1]], [[0, 0, 1]],
             [[0, 0, 2]], [[0, 0, 2]], [[0, 0, 3]], [[0, 0, 3]]],
        )
        self.assert_plan_invariants(policy)

    def test_two_socket_two_numa_full_plan_never_crosses_node(self):
        policy = POLICY.generate_policy(
            synthetic_topology([(0, 0, 4), (1, 1, 4)], siblings=2)
        )
        self.assertEqual(policy["socket_count"], 2)
        self.assertEqual(policy["numa_node_count"], 2)
        self.assertEqual(policy["canonical_full_physical_config_id"], "2x4")
        self.assertIn(4, policy["physical_budgets"])
        canonical = next(
            config for config in policy["configs"] if config["canonical_full_physical"]
        )
        self.assertEqual(canonical["rank_numa_plan"], [0, 1])
        self.assert_plan_invariants(policy)

    def test_four_socket_boundary_and_canonical_plan(self):
        policy = POLICY.generate_policy(
            synthetic_topology([(0, 0, 2), (1, 1, 2), (2, 2, 2), (3, 3, 2)])
        )
        self.assertEqual(policy["socket_count"], 4)
        self.assertEqual(policy["physical_budgets"], [2, 4, 6, 8])
        self.assertEqual(policy["canonical_full_physical_config_id"], "4x2")
        self.assert_plan_invariants(policy)

    def test_one_cpuset_core_per_numa_uses_unique_pure_mpi_canonical(self):
        for layout, expected_id in (
            ([(0, 0, 1), (1, 1, 1)], "2x1"),
            ([(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)], "4x1"),
        ):
            with self.subTest(layout=layout):
                policy = POLICY.generate_policy(
                    synthetic_topology(layout, siblings=2)
                )
                self.assertEqual(
                    policy["canonical_full_physical_config_id"], expected_id
                )
                self.assert_plan_invariants(policy)

    def test_sparse_partial_cpuset_is_deterministic_and_exact(self):
        topology = synthetic_topology(
            [(0, 3, 4)], siblings=2, logical_start=2, logical_stride=3
        )
        # Simulate a cpuset which exposes only one sibling of one physical core.
        removed = topology["cores"][-1]["logical_cpus"].pop()
        topology["available_logical_cpus"].remove(removed)
        topology["logical_cpu_count"] -= 1
        topology["logical_cpus"] = [
            row for row in topology["logical_cpus"] if row["cpu"] != removed
        ]
        first = POLICY.generate_policy(topology)
        second = POLICY.generate_policy(copy.deepcopy(topology))
        self.assertEqual(first, second)
        self.assertEqual(len(first["matrix_sha256"]), 64)
        self.assertEqual(len(first["topology_sha256"]), 64)
        smt = next(config for config in first["configs"] if config["lane"] == "smt")
        self.assertEqual(
            sorted(cpu for rank in smt["rank_logical_plan"] for cpu in rank),
            topology["available_logical_cpus"],
        )
        self.assert_plan_invariants(first)

    def test_topology_change_changes_matrix_hash(self):
        one = POLICY.generate_policy(synthetic_topology([(0, 0, 8)]))
        two = POLICY.generate_policy(synthetic_topology([(0, 0, 4), (1, 1, 4)]))
        self.assertNotEqual(one["matrix_sha256"], two["matrix_sha256"])

    def test_normal_publication_rejects_less_than_two_physical_cores(self):
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            POLICY.generate_policy(synthetic_topology([(0, 0, 1)], siblings=2))


class RunnerPolicyBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_source(
            "gpmeep_cpu_parallel_runner_policy_bridge_test",
            SCRIPTS / "run-cpu-parallel-benchmark.py",
        )

    def setUp(self):
        self.original = (
            self.runner.CONFIGS,
            self.runner.TOPOLOGY_MATRIX,
            self.runner.REFERENCE_SCHEDULE_INDEX,
        )

    def tearDown(self):
        (
            self.runner.CONFIGS,
            self.runner.TOPOLOGY_MATRIX,
            self.runner.REFERENCE_SCHEDULE_INDEX,
        ) = self.original

    def test_dynamic_schedule_and_recovery_names_follow_matrix(self):
        matrix = self.runner.activate_topology_policy(
            synthetic_topology([(0, 0, 8)], siblings=2)
        )
        schedule = self.runner.sample_schedule()
        self.assertEqual(len(schedule), len(matrix["configs"]) * 6)
        self.assertEqual(
            schedule[self.runner.REFERENCE_SCHEDULE_INDEX], ("1x1", "measured", 0)
        )
        run_id = "a" * 32
        for row in schedule:
            for path in self.runner._sample_artifact_paths(
                pathlib.Path("/tmp"), run_id, *row
            ):
                self.assertTrue(
                    self.runner._is_scheduled_sample_artifact_name(path.name)
                )

    def test_dynamic_multi_numa_launch_is_fail_closed_until_plan_realization(self):
        self.runner.activate_topology_policy(
            synthetic_topology([(0, 0, 4), (1, 1, 4)], siblings=2)
        )
        config = self.runner._config_by_id("2x4")
        with self.assertRaisesRegex(RuntimeError, "multi-NUMA launcher placement"):
            self.runner.binding_arguments(config)

    def test_default_runner_fails_closed_before_final_publication(self):
        args = type("Args", (), {"host_specific_interim": False})()
        with self.assertRaisesRegex(RuntimeError, "portable/final CPU publication is disabled"):
            self.runner.execute(args, pathlib.Path("/unused"))

    def test_v2_runner_uses_existing_receipt_bound_producer_contract(self):
        argv = self.runner.producer_arguments(
            repo=pathlib.Path("/repo"),
            runtime={"python": pathlib.Path("/prefix/bin/python")},
            result_path=pathlib.Path("/out/result.json"),
            receipt_id="receipt",
            snapshot_sha256="a" * 64,
            producer_sha256="b" * 64,
            nonce="c" * 32,
            run_id="d" * 32,
            sample_kind="measured",
            iteration=0,
        )
        self.assertEqual(
            argv[argv.index("--qualification-profile") + 1],
            self.runner.PRODUCER_QUALIFICATION_PROFILE_ID,
        )
        self.assertEqual(
            argv[argv.index("--qualification-run-id") + 1], "d" * 32
        )
        self.assertNotEqual(
            self.runner.PROFILE_ID,
            self.runner.PRODUCER_QUALIFICATION_PROFILE_ID,
        )


if __name__ == "__main__":
    unittest.main()
