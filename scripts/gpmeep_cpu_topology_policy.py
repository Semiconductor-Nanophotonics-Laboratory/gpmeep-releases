"""Pure, portable CPU topology policy for the gpmeep benchmark runner.

This module deliberately has no host probes.  It converts the normalized,
cpuset-filtered lscpu record supplied by the runner into an immutable benchmark
matrix with explicit per-rank core and logical-CPU placement plans.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


POLICY_SCHEMA_VERSION = 2


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _core_key(core: dict[str, Any]) -> tuple[int, int, int]:
    return (int(core["socket"]), int(core["node"]), int(core["core"]))


def _validate_topology(topology: dict[str, Any]) -> list[dict[str, Any]]:
    cores = topology.get("cores") if isinstance(topology, dict) else None
    allowed = topology.get("available_logical_cpus") if isinstance(topology, dict) else None
    if not isinstance(cores, list) or not isinstance(allowed, list):
        raise RuntimeError("normalized CPU topology is incomplete")
    if allowed != sorted(set(allowed)) or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in allowed
    ):
        raise RuntimeError("normalized CPU affinity is invalid")
    if len(cores) < 2:
        raise RuntimeError(
            "normal CPU benchmark publication requires at least two allowed physical cores"
        )
    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int]] = set()
    seen_logical: set[int] = set()
    for index, raw in enumerate(cores):
        if not isinstance(raw, dict):
            raise RuntimeError(f"normalized core {index} is invalid")
        key = _core_key(raw)
        cpus = raw.get("logical_cpus")
        if key in seen_keys or not isinstance(cpus, list) or not cpus:
            raise RuntimeError("normalized physical-core mapping is invalid")
        if cpus != sorted(set(cpus)) or any(cpu not in allowed for cpu in cpus):
            raise RuntimeError("normalized core contains invalid logical CPUs")
        if seen_logical.intersection(cpus):
            raise RuntimeError("logical CPU belongs to multiple physical cores")
        seen_keys.add(key)
        seen_logical.update(cpus)
        normalized.append(
            {
                "socket": key[0],
                "node": key[1],
                "core": key[2],
                "logical_cpus": list(cpus),
            }
        )
    if seen_logical != set(allowed):
        raise RuntimeError("normalized cores do not exactly cover the allowed cpuset")
    return sorted(normalized, key=_core_key)


def _budget_boundaries(cores: list[dict[str, Any]]) -> list[int]:
    physical = len(cores)
    budgets: set[int] = {physical}
    power = 2
    while power <= physical:
        budgets.add(power)
        power *= 2
    for field in ("node", "socket"):
        counts: dict[int, int] = {}
        for core in cores:
            counts[int(core[field])] = counts.get(int(core[field]), 0) + 1
        cumulative = 0
        for identity in sorted(counts):
            count = counts[identity]
            budgets.add(count)
            cumulative += count
            budgets.add(cumulative)
    return sorted(budget for budget in budgets if 2 <= budget <= physical)


def _placement(
    groups: list[list[dict[str, Any]]], *, smt_cpu_groups: list[list[int]] | None = None
) -> dict[str, Any]:
    core_plan = [
        [[core["socket"], core["node"], core["core"]] for core in group]
        for group in groups
    ]
    logical_plan = (
        smt_cpu_groups
        if smt_cpu_groups is not None
        else [
            sorted(cpu for core in group for cpu in core["logical_cpus"])
            for group in groups
        ]
    )
    nodes = [sorted({core["node"] for core in group}) for group in groups]
    if any(len(node_set) != 1 for node_set in nodes):
        raise RuntimeError("an MPI rank placement crosses a NUMA node")
    flattened_cores = [tuple(core) for group in core_plan for core in group]
    flattened_logical = [cpu for group in logical_plan for cpu in group]
    if smt_cpu_groups is None and len(flattened_cores) != len(set(flattened_cores)):
        raise RuntimeError("physical-lane MPI ranks overlap physical cores")
    if len(flattened_logical) != len(set(flattened_logical)):
        raise RuntimeError("MPI ranks overlap logical CPUs")
    return {
        "rank_core_plan": core_plan,
        "rank_logical_plan": [sorted(group) for group in logical_plan],
        "rank_numa_plan": [node_set[0] for node_set in nodes],
    }


def _config(
    *,
    config_id: str,
    ranks: int,
    omp_threads: int,
    binding: str,
    lane: str,
    physical_budget: int,
    physical_total: int,
    logical_total: int,
    groups: list[list[dict[str, Any]]],
    canonical_full_physical: bool = False,
    smt_cpu_groups: list[list[int]] | None = None,
) -> dict[str, Any]:
    record = {
        "config_id": config_id,
        "ranks": ranks,
        "omp_threads": omp_threads,
        "binding": binding,
        "lane": lane,
        "physical_budget": physical_budget,
        "full_physical": lane == "physical" and physical_budget == physical_total,
        "canonical_full_physical": canonical_full_physical,
        **_placement(groups, smt_cpu_groups=smt_cpu_groups),
    }
    if lane == "physical" and ranks * omp_threads != physical_budget:
        raise RuntimeError("physical configuration does not exactly cover its budget")
    if lane == "smt" and ranks * omp_threads != logical_total:
        raise RuntimeError("SMT endpoint does not exactly cover the allowed cpuset")
    return record


def generate_policy(topology: dict[str, Any]) -> dict[str, Any]:
    """Generate a deterministic matrix from a normalized allowed-CPU topology."""

    cores = _validate_topology(topology)
    physical = len(cores)
    allowed = list(topology["available_logical_cpus"])
    logical = len(allowed)
    budgets = _budget_boundaries(cores)
    by_node: dict[int, list[dict[str, Any]]] = {}
    for core in cores:
        by_node.setdefault(core["node"], []).append(core)

    configs: list[dict[str, Any]] = []
    configs.append(
        _config(
            config_id="1x1",
            ranks=1,
            omp_threads=1,
            binding="core",
            lane="diagnostic",
            physical_budget=1,
            physical_total=physical,
            logical_total=logical,
            groups=[[cores[0]]],
        )
    )

    # Every budget has a portable pure-MPI endpoint.  Prefix selection follows
    # the same deterministic core order encoded in the placement plan.
    for budget in budgets:
        selected = cores[:budget]
        configs.append(
            _config(
                config_id=f"{budget}x1",
                ranks=budget,
                omp_threads=1,
                binding="core",
                lane="physical",
                physical_budget=budget,
                physical_total=physical,
                logical_total=logical,
                groups=[[core] for core in selected],
                canonical_full_physical=False,
            )
        )

    # Keep SMT as a separate endpoint/lane.  A partial cpuset can expose one
    # sibling on some cores; all allowed logical CPUs are still covered once.
    if logical > physical:
        cpu_to_core = {
            cpu: core for core in cores for cpu in core["logical_cpus"]
        }
        # Open MPI's hwloc-backed `--map-by hwthread` traverses processing
        # units in topology order: every allowed sibling of one core before
        # moving to the next core.  Linux logical CPU numbers need not follow
        # that order (for example 0,8,1,9,... on an 8C/16T host), so a numeric
        # sort produces a false per-rank affinity contract.
        smt_cpus = [
            cpu for core in cores for cpu in core["logical_cpus"]
        ]
        configs.append(
            _config(
                config_id=f"{logical}x1",
                ranks=logical,
                omp_threads=1,
                binding="hwthread",
                lane="smt",
                physical_budget=physical,
                physical_total=physical,
                logical_total=logical,
                groups=[[cpu_to_core[cpu]] for cpu in smt_cpus],
                smt_cpu_groups=[[cpu] for cpu in smt_cpus],
            )
        )

    # Hybrid endpoints cover *all* physical cores exactly and never cross NUMA
    # nodes.  Uniform nodes permit a deterministic ppr:N:numa launch mapping.
    node_sizes = [len(by_node[node]) for node in sorted(by_node)]
    hybrid_threads: list[int] = []
    if len(set(node_sizes)) == 1 and node_sizes[0] > 1:
        per_node = node_sizes[0]
        candidate = 2
        while candidate <= per_node:
            if per_node % candidate == 0:
                hybrid_threads.append(candidate)
            candidate *= 2
        if per_node not in hybrid_threads:
            hybrid_threads.append(per_node)
    hybrid_threads = sorted(set(hybrid_threads), reverse=True)
    canonical_id: str | None = None
    for threads in hybrid_threads:
        groups: list[list[dict[str, Any]]] = []
        for node in sorted(by_node):
            node_cores = by_node[node]
            if len(node_cores) % threads:
                groups = []
                break
            groups.extend(
                node_cores[index : index + threads]
                for index in range(0, len(node_cores), threads)
            )
        if not groups:
            continue
        ranks = len(groups)
        config_id = f"{ranks}x{threads}"
        is_canonical = threads == node_sizes[0]
        configs.append(
            _config(
                config_id=config_id,
                ranks=ranks,
                omp_threads=threads,
                binding="core",
                lane="physical",
                physical_budget=physical,
                physical_total=physical,
                logical_total=logical,
                groups=groups,
                canonical_full_physical=is_canonical,
            )
        )
        if is_canonical:
            canonical_id = config_id

    # Heterogeneous NUMA nodes cannot be represented by one OMP thread count;
    # the full pure-MPI endpoint is then the canonical portable baseline.
    if canonical_id is None:
        canonical_id = f"{physical}x1"
        for config in configs:
            if config["config_id"] == canonical_id:
                config["canonical_full_physical"] = True
                break

    ids = [config["config_id"] for config in configs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("topology policy generated duplicate configuration IDs")
    canonical = [
        config for config in configs if config["canonical_full_physical"] is True
    ]
    if len(canonical) != 1 or canonical[0]["lane"] != "physical":
        raise RuntimeError("topology policy lacks one canonical full-physical endpoint")

    matrix = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "topology_sha256": _canonical_sha256(topology),
        "physical_core_count": physical,
        "logical_cpu_count": logical,
        "socket_count": len({core["socket"] for core in cores}),
        "numa_node_count": len(by_node),
        "physical_budgets": budgets,
        "diagnostic_config_id": "1x1",
        "canonical_full_physical_config_id": canonical_id,
        "configs": configs,
    }
    return {**matrix, "matrix_sha256": _canonical_sha256(matrix)}
