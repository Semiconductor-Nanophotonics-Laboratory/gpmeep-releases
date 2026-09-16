#!/usr/bin/env python3
"""Run and seal the fixed Waitsome/Waitall MPI completion-policy A/B.

This publisher deliberately treats performance promotion as a reported
decision rather than a prerequisite for publishing valid negative evidence.
Every raw child is receipt/nonce/source bound and revalidated before COMPLETE.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import re
import shutil
import statistics
import sys
import traceback
import types
import uuid
from typing import Any


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
RUNNER = _load_source_module(
    "gpmeep_mpi_adjoint_runner_for_completion_ab",
    SCRIPT_DIRECTORY / "run-mpi-adjoint-benchmark.py",
)
EVIDENCE = RUNNER._EVIDENCE
PROVENANCE = RUNNER._PROVENANCE

PROFILE_ID = "m19-mpi-completion-ab-v3"
PROFILE = types.MappingProxyType(
    {
        "profile_id": PROFILE_ID,
        "raw_profile_id": RUNNER.PROFILE_ID,
        "resolution": 64,
        "run_time": 40.0,
        "cell_size": 32.0,
        "design_resolution": 20,
        "multi_ranks": 2,
        "policies": ["waitsome", "waitall"],
        "warmups_per_policy": 1,
        "measured_repeats_per_policy": 6,
        "capture_field_times": True,
        "objective_atol": 1.0e-8,
        "gradient_atol": 2.0e-7,
        "gradient_rtol": 5.0e-5,
        "material_promotion_speedup": 1.02,
        "process_position_effect_warning_ratio": 1.05,
        "promotion_requires_workload_and_process_wall": True,
        "promotion_requires_majority_pairs_faster": True,
        "promotion_requires_both_process_position_strata": True,
    }
)
PROFILE_CANONICAL_SHA256 = (
    "d0e8067be972dc5db0d231f72266f5642dae255059322a005bc1c65ab0c46679"
)
FINAL_NAMES = {
    "artifacts.sha256.json",
    "build-provenance.json",
    "COMPLETE",
    "FAILED.json",
    "report.json",
    "report.md",
    "state.json",
}

# Open MPI/PMIx creates a fresh namespace, temporary directory, and local TCP
# rendezvous endpoint for every mpiexec invocation.  Those values identify the
# launcher session, not a workload control.  Keep this list exact (rather than
# accepting a prefix) so a newly injected MPI/PMIx variable fails the
# cross-sample environment gate until it is deliberately classified.
MPI_SESSION_ENVIRONMENT_KEYS = frozenset(
    {
        "OMPI_FILE_LOCATION",
        "PMIX_NAMESPACE",
        "PMIX_SERVER_TMPDIR",
        "PMIX_SERVER_URI2",
        "PMIX_SERVER_URI21",
        "PMIX_SERVER_URI3",
        "PMIX_SERVER_URI4",
        "PMIX_SERVER_URI41",
    }
)
MPI_SESSION_PRESENCE_SENTINEL = "__gpmeep_mpi_session_environment_keys__"


def _assert_fixed_profile() -> None:
    if (
        not isinstance(PROFILE, types.MappingProxyType)
        or PROVENANCE.canonical_sha256(dict(PROFILE))
        != PROFILE_CANONICAL_SHA256
    ):
        raise RuntimeError("fixed MPI completion A/B profile changed in memory")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def _prepare_output(output: pathlib.Path, run_id: str) -> None:
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise RuntimeError("completion A/B output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    EVIDENCE.atomic_write_json(
        output / "state.json",
        {
            "schema_version": 1,
            "state": "RUNNING",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
        },
    )


def _expected_matrix() -> list[tuple[str, str, int]]:
    expected: list[tuple[str, str, int]] = []
    for iteration in range(PROFILE["warmups_per_policy"]):
        expected.extend(
            (policy, "warmup", iteration) for policy in PROFILE["policies"]
        )
    for iteration in range(PROFILE["measured_repeats_per_policy"]):
        order = (
            PROFILE["policies"]
            if iteration % 2 == 0
            else list(reversed(PROFILE["policies"]))
        )
        expected.extend((policy, "measured", iteration) for policy in order)
    return expected


def validate_sample_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _expected_matrix()
    actual = [
        (
            sample.get("completion_policy"),
            sample.get("sample_kind"),
            sample.get("iteration"),
        )
        for sample in samples
    ]
    if actual != expected:
        raise RuntimeError("completion A/B sample matrix is incomplete or not interleaved")
    nonces = [sample.get("nonce") for sample in samples]
    if any(not isinstance(nonce, str) or not nonce for nonce in nonces):
        raise RuntimeError("completion A/B sample nonce is absent")
    if len(nonces) != len(set(nonces)):
        raise RuntimeError("completion A/B sample nonce was reused")
    if any(
        sample.get("lane") != "cuda-multi"
        or sample.get("ranks") != PROFILE["multi_ranks"]
        or sample.get("capture_field_times") is not True
        for sample in samples
    ):
        raise RuntimeError("completion A/B sample lane/profile is invalid")
    return {"pass": True, "expected": expected, "actual": actual}


def validate_raw_artifact_set(
    raw_output: pathlib.Path, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_sample_files = {
        pathlib.Path(sample[name]).resolve()
        for sample in samples
        for name in ("result_file", "stdout_log", "stderr_log", "timing_file")
    }
    run_ids = {
        pathlib.Path(sample["result_file"]).name.split("-", 1)[0]
        for sample in samples
    }
    if len(run_ids) != 1:
        raise RuntimeError("completion A/B raw artifacts use different run IDs")
    run_id = next(iter(run_ids))
    home_marker = (raw_output / "home" / run_id / ".gpmeep-empty-home").resolve()
    if (
        not home_marker.is_file()
        or home_marker.read_text(encoding="utf-8")
        != "gpmeep isolated MPI qualification home\n"
    ):
        raise RuntimeError("completion A/B isolated HOME marker is invalid")
    matplotlib_root = (raw_output / "matplotlib").resolve()
    font_cache_files = {
        path.resolve()
        for path in matplotlib_root.iterdir()
        if path.is_file()
        and re.fullmatch(r"fontlist-v[0-9.]+\.json", path.name) is not None
    }
    if len(font_cache_files) > 1:
        raise RuntimeError("completion A/B created multiple font-cache records")
    for path in font_cache_files:
        try:
            font_cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("completion A/B font cache is invalid") from error
        if not isinstance(font_cache, dict):
            raise RuntimeError("completion A/B font cache is not a JSON object")
    expected_files = expected_sample_files | {home_marker} | font_cache_files
    actual_files = {
        path.resolve() for path in raw_output.rglob("*") if path.is_file()
    }
    allowed_directories = {
        raw_output.resolve(),
        (raw_output / "home").resolve(),
        (raw_output / "home" / run_id).resolve(),
        matplotlib_root,
        (raw_output / "pycache").resolve(),
        (raw_output / "pycache" / run_id).resolve(),
        (raw_output / "tmp").resolve(),
    }
    actual_directories = {
        path.resolve() for path in raw_output.rglob("*") if path.is_dir()
    } | {raw_output.resolve()}
    if (
        actual_files != expected_files
        or actual_directories != allowed_directories
        or any(path.is_symlink() for path in raw_output.rglob("*"))
    ):
        raise RuntimeError("completion A/B raw artifact set is incomplete or duplicated")
    return {
        "pass": True,
        "paths": sorted(str(path) for path in actual_files),
    }


def _rank_environment_projection(sample: dict[str, Any]) -> list[dict[str, str]]:
    ranks = sample["result"]["distributed"]["ranks"]
    producer_command = sample["result"].get("producer", {}).get("command")
    result: list[dict[str, str]] = []
    for rank in ranks:
        environment = dict(rank["process_environment"])
        if MPI_SESSION_PRESENCE_SENTINEL in environment:
            raise RuntimeError("rank environment contains the reserved session sentinel")
        policy = environment.pop("MEEP_GPU_MPI_COMPLETION", None)
        if policy != sample["completion_policy"]:
            raise RuntimeError("rank did not execute its bound completion policy")
        argv = environment.pop("OMPI_ARGV", None)
        if argv is not None:
            if (
                not isinstance(producer_command, list)
                or len(producer_command) < 2
                or not all(isinstance(value, str) for value in producer_command)
                or argv != " ".join(producer_command[1:])
            ):
                raise RuntimeError("rank OMPI_ARGV differs from its validated producer command")
        elif producer_command is not None:
            raise RuntimeError("rank OMPI_ARGV is absent for a recorded producer command")
        present_session_keys = sorted(
            name for name in MPI_SESSION_ENVIRONMENT_KEYS if name in environment
        )
        for name in present_session_keys:
            environment.pop(name)
        environment[MPI_SESSION_PRESENCE_SENTINEL] = ",".join(
            present_session_keys
        )
        result.append(environment)
    return result


def _counter_signature(sample: dict[str, Any]) -> str:
    return PROVENANCE.canonical_sha256(
        [
            rank["final_statistics"]
            for rank in sample["result"]["distributed"]["ranks"]
        ]
    )


def _runtime_signature(sample: dict[str, Any]) -> str:
    return PROVENANCE.canonical_sha256(
        [
            RUNNER.COMPARATOR.runtime_environment_independent_projection(runtime)
            for runtime in sample["validation"]["rank_runtimes"]
        ]
    )


def _finite_wall(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return converted


def _validate_parent_timing_attestation(
    sample: dict[str, Any], *, expected_run_id: str, label: str, expected_command: list[str]
) -> dict[str, Any]:
    try:
        timing = json.loads(
            pathlib.Path(sample["timing_file"]).read_text(encoding="utf-8")
        )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} parent timing attestation is unreadable") from error
    expected = {
        "schema_version": 1,
        "run_id": expected_run_id,
        "label": label,
        "lane": sample["lane"],
        "sample_kind": sample["sample_kind"],
        "iteration": sample["iteration"],
        "nonce": sample["nonce"],
        "process_seconds": sample["process_seconds"],
        "command_sha256": PROVENANCE.canonical_sha256(expected_command),
    }
    if timing != expected:
        raise RuntimeError(f"{label} parent timing attestation changed")
    return timing


def compare_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    validate_sample_matrix(samples)
    measured: dict[str, list[dict[str, Any]]] = {
        policy: sorted(
            (
                sample
                for sample in samples
                if sample["completion_policy"] == policy
                and sample["sample_kind"] == "measured"
            ),
            key=lambda sample: sample["iteration"],
        )
        for policy in PROFILE["policies"]
    }
    expected_repeats = PROFILE["measured_repeats_per_policy"]
    if any(len(policy_samples) != expected_repeats for policy_samples in measured.values()):
        raise RuntimeError("completion A/B measured counts differ from the profile")

    reference = samples[0]
    reference_devices = reference["validation"]["device_identifiers"]
    reference_environment = _rank_environment_projection(reference)
    reference_runtime = _runtime_signature(reference)
    reference_counter = _counter_signature(reference)
    stable_devices = True
    stable_environment = True
    stable_runtime = True
    stable_counters = True
    stable_workload = True
    for sample in samples:
        stable_devices = stable_devices and (
            sample["validation"]["device_identifiers"] == reference_devices
        )
        stable_environment = stable_environment and (
            _rank_environment_projection(sample) == reference_environment
        )
        stable_runtime = stable_runtime and (
            _runtime_signature(sample) == reference_runtime
        )
        stable_counters = stable_counters and (
            _counter_signature(sample) == reference_counter
        )
        stable_workload = stable_workload and RUNNER._same_workload(
            reference["result"], sample["result"]
        )

    correctness: list[dict[str, Any]] = []
    objective_pass = True
    gradient_pass = True
    workload_seconds: dict[str, list[float]] = {
        policy: [] for policy in PROFILE["policies"]
    }
    process_seconds: dict[str, list[float]] = {
        policy: [] for policy in PROFILE["policies"]
    }
    process_position_seconds: dict[str, dict[str, list[float]]] = {
        policy: {"first": [], "second": []} for policy in PROFILE["policies"]
    }
    measured_order = {
        iteration: [
            policy
            for policy, sample_kind, sample_iteration in _expected_matrix()
            if sample_kind == "measured" and sample_iteration == iteration
        ]
        for iteration in range(expected_repeats)
    }
    if expected_repeats % 2 != 0 or any(
        sorted(order) != sorted(PROFILE["policies"]) or len(order) != 2
        for order in measured_order.values()
    ):
        raise RuntimeError("completion A/B process-wall schedule is not position-balanced")
    paired_speedups: list[float] = []
    waitall_faster_pairs = 0
    for waitsome, waitall in zip(measured["waitsome"], measured["waitall"]):
        if waitsome["iteration"] != waitall["iteration"]:
            raise RuntimeError("completion A/B measured pairs are misaligned")
        objective_error = abs(
            waitsome["validation"]["objective"]
            - waitall["validation"]["objective"]
        )
        gradient = EVIDENCE.compare_gradient_vectors(
            waitsome["result"],
            waitall["result"],
            atol=PROFILE["gradient_atol"],
            rtol=PROFILE["gradient_rtol"],
            expected_count=int(waitsome["result"]["workload"]["design_variables"]),
        )
        pair_objective_pass = objective_error <= PROFILE["objective_atol"]
        objective_pass = objective_pass and pair_objective_pass
        gradient_pass = gradient_pass and gradient["pass"]
        some_wall = _finite_wall(
            waitsome["result"]["timing"]["workload_wall_seconds"],
            "Waitsome workload wall",
        )
        all_wall = _finite_wall(
            waitall["result"]["timing"]["workload_wall_seconds"],
            "Waitall workload wall",
        )
        workload_seconds["waitsome"].append(some_wall)
        workload_seconds["waitall"].append(all_wall)
        some_process = _finite_wall(
            waitsome["process_seconds"], "Waitsome process wall"
        )
        all_process = _finite_wall(
            waitall["process_seconds"], "Waitall process wall"
        )
        process_seconds["waitsome"].append(some_process)
        process_seconds["waitall"].append(all_process)
        position_by_policy = {
            policy: "first" if index == 0 else "second"
            for index, policy in enumerate(measured_order[waitsome["iteration"]])
        }
        process_position_seconds["waitsome"][
            position_by_policy["waitsome"]
        ].append(some_process)
        process_position_seconds["waitall"][
            position_by_policy["waitall"]
        ].append(all_process)
        paired_speedups.append(some_wall / all_wall)
        waitall_faster_pairs += int(all_wall < some_wall)
        correctness.append(
            {
                "iteration": waitsome["iteration"],
                "objective_absolute_error": objective_error,
                "objective_pass": pair_objective_pass,
                "gradient": gradient,
                "waitsome_workload_wall_seconds": some_wall,
                "waitall_workload_wall_seconds": all_wall,
                "waitsome_over_waitall": some_wall / all_wall,
            }
        )

    workload_speedup = statistics.median(workload_seconds["waitsome"]) / statistics.median(
        workload_seconds["waitall"]
    )
    unstratified_process_speedup = statistics.median(
        process_seconds["waitsome"]
    ) / statistics.median(process_seconds["waitall"])
    expected_position_count = expected_repeats // 2
    if any(
        len(process_position_seconds[policy][position])
        != expected_position_count
        for policy in PROFILE["policies"]
        for position in ("first", "second")
    ):
        raise RuntimeError("completion A/B process-wall position strata are incomplete")
    process_position_medians = {
        policy: {
            position: statistics.median(values)
            for position, values in by_position.items()
        }
        for policy, by_position in process_position_seconds.items()
    }
    process_position_speedups = {
        position: process_position_medians["waitsome"][position]
        / process_position_medians["waitall"][position]
        for position in ("first", "second")
    }
    balanced_process_speedup = math.sqrt(
        process_position_speedups["first"]
        * process_position_speedups["second"]
    )
    process_position_effect_by_policy = {
        policy: max(
            medians["first"] / medians["second"],
            medians["second"] / medians["first"],
        )
        for policy, medians in process_position_medians.items()
    }
    maximum_process_position_effect = max(
        process_position_effect_by_policy.values()
    )
    process_position_warning = (
        maximum_process_position_effect
        > PROFILE["process_position_effect_warning_ratio"]
    )
    majority_required = expected_repeats // 2 + 1
    process_position_strata_pass = all(
        speedup >= PROFILE["material_promotion_speedup"]
        for speedup in process_position_speedups.values()
    )
    promotion_pass = (
        workload_speedup >= PROFILE["material_promotion_speedup"]
        and balanced_process_speedup >= PROFILE["material_promotion_speedup"]
        and process_position_strata_pass
        and waitall_faster_pairs >= majority_required
    )
    evidence_gates = {
        "sample_matrix": True,
        "fixed_workload": stable_workload,
        "objective": objective_pass,
        "gradient": gradient_pass,
        "exact_cuda_counter_signature": stable_counters,
        "stable_physical_device_mapping": stable_devices,
        "stable_environment_except_completion_policy": stable_environment,
        "stable_receipt_controlled_runtime": stable_runtime,
        "receipt_and_runtime_closure": True,
        "strict_cuda_coverage": True,
    }
    evidence_gates["pass"] = all(evidence_gates.values())
    if not evidence_gates["pass"]:
        failed = [name for name, passed in evidence_gates.items() if not passed]
        raise RuntimeError("completion A/B evidence gates failed: " + ", ".join(failed))
    return {
        "evidence_gate": evidence_gates,
        "promotion": {
            "pass": promotion_pass,
            "decision": "promote-waitall" if promotion_pass else "retain-waitsome-default",
            "fixed_minimum_speedup": PROFILE["material_promotion_speedup"],
            "waitall_faster_pairs": waitall_faster_pairs,
            "majority_pairs_required": majority_required,
            "balanced_process_wall_pass": balanced_process_speedup
            >= PROFILE["material_promotion_speedup"],
            "both_process_position_strata_pass": process_position_strata_pass,
        },
        "correctness": correctness,
        "performance": {
            "workload_wall": {
                "seconds": workload_seconds,
                "median_seconds": {
                    policy: statistics.median(values)
                    for policy, values in workload_seconds.items()
                },
                "waitsome_over_waitall_median_speedup": workload_speedup,
            },
            "fresh_process_wall": {
                "seconds": process_seconds,
                "median_seconds": {
                    policy: statistics.median(values)
                    for policy, values in process_seconds.items()
                },
                "unstratified_waitsome_over_waitall_median_speedup_diagnostic_only": (
                    unstratified_process_speedup
                ),
                "by_position": {
                    "seconds": process_position_seconds,
                    "median_seconds": process_position_medians,
                    "waitsome_over_waitall_speedup": process_position_speedups,
                    "balanced_geometric_mean_speedup": balanced_process_speedup,
                    "samples_per_policy_per_position": expected_position_count,
                },
                "position_effect_diagnostic": {
                    "warning_threshold_ratio": PROFILE[
                        "process_position_effect_warning_ratio"
                    ],
                    "effect_ratio_by_policy": process_position_effect_by_policy,
                    "maximum_effect_ratio": maximum_process_position_effect,
                    "warning": process_position_warning,
                },
            },
            "paired_workload_speedups": paired_speedups,
            "paired_median_speedup": statistics.median(paired_speedups),
            "paired_minimum_speedup": min(paired_speedups),
            "paired_maximum_speedup": max(paired_speedups),
        },
        "devices": reference_devices,
        "counter_signature_sha256": reference_counter,
        "runtime_signature_sha256": reference_runtime,
    }


def _reverify(
    *,
    expected_run_id: str,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    samples: list[dict[str, Any]],
    expected_controller_ledger: dict[str, Any],
) -> dict[str, Any]:
    cache = PROVENANCE.StatHashCache()
    receipt = PROVENANCE.verify_build_receipt(
        receipt_path, repo, hash_cache=cache
    )
    if receipt.get("receipt_id") != expected_receipt_id:
        raise RuntimeError("build receipt changed during completion A/B")
    contract = RUNNER.validate_build_receipt_contract(
        receipt, repo, hash_cache=cache
    )
    if (
        PROVENANCE.source_snapshot(repo, hash_cache=cache) != expected_snapshot
        or receipt.get("source_end") != expected_snapshot
    ):
        raise RuntimeError("source changed during completion A/B")
    producer = repo / "scripts" / "benchmark-adjoint.py"
    if cache.digest(producer) != expected_producer_sha256:
        raise RuntimeError("raw benchmark producer changed during completion A/B")
    runtime = RUNNER._receipt_runtime(receipt, repo)
    checked: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != RUNNER.EXPECTED_SAMPLE_KEYS:
            raise RuntimeError("completion A/B sample exact schema disagrees")
        expected_label = (
            f"{expected_run_id}-{sample['sample_kind']}-{sample['iteration']}-"
            f"{sample['completion_policy']}"
        )
        if (
            sample.get("lane") != "cuda-multi"
            or sample.get("ranks") != PROFILE["multi_ranks"]
            or sample.get("capture_field_times") is not True
            or sample.get("artifact_label") != expected_label
            or sample.get("completion_policy") not in PROFILE["policies"]
        ):
            raise RuntimeError("completion A/B sample identity disagrees")
        label = pathlib.Path(sample["result_file"]).stem
        if label != expected_label:
            raise RuntimeError("completion A/B artifact label disagrees")
        files: dict[str, dict[str, str]] = {}
        for path_name, digest_name in (
            ("result_file", "result_sha256"),
            ("stdout_log", "stdout_sha256"),
            ("stderr_log", "stderr_sha256"),
            ("timing_file", "timing_sha256"),
        ):
            path = pathlib.Path(sample[path_name]).resolve()
            if cache.digest(path) != sample[digest_name]:
                raise RuntimeError(f"{label} {path_name} changed before publication")
            files[path_name] = {"path": str(path), "sha256": sample[digest_name]}
        raw = json.loads(pathlib.Path(sample["result_file"]).read_text(encoding="utf-8"))
        if PROVENANCE.canonical_sha256(raw) != PROVENANCE.canonical_sha256(
            sample["result"]
        ):
            raise RuntimeError(f"{label} raw and in-memory evidence disagree")
        stdout = pathlib.Path(sample["stdout_log"]).read_text(encoding="utf-8")
        pointer = RUNNER._extract_pointer(stdout)
        if (
            pathlib.Path(str(pointer.get("result_file", ""))).resolve()
            != pathlib.Path(sample["result_file"]).resolve()
            or pointer.get("sha256") != sample["result_sha256"]
        ):
            raise RuntimeError(f"{label} raw result pointer changed")
        expected_command = RUNNER._expected_sample_command(
            repo=repo,
            runtime=runtime,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            run_id=expected_run_id,
            nonce=sample["nonce"],
            lane=sample["lane"],
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            result_path=pathlib.Path(sample["result_file"]),
            ranks=sample["ranks"],
            capture_field_times=True,
        )
        expected_environment = RUNNER._child_environment(
            runtime,
            lane=sample["lane"],
            output=pathlib.Path(sample["result_file"]).parent,
            pycache_namespace=expected_run_id,
            prepare_filesystem=False,
        )
        expected_environment["MEEP_GPU_MPI_COMPLETION"] = sample[
            "completion_policy"
        ]
        expected_environment = dict(sorted(expected_environment.items()))
        if (
            sample.get("command") != expected_command
            or sample.get("environment") != expected_environment
        ):
            raise RuntimeError(f"{label} launch command/environment changed")
        _validate_parent_timing_attestation(
            sample,
            expected_run_id=expected_run_id,
            label=label,
            expected_command=expected_command,
        )
        if raw.get("producer", {}).get("command") != RUNNER._expected_producer_command(
            repo=repo,
            runtime=runtime,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            run_id=expected_run_id,
            nonce=sample["nonce"],
            lane=sample["lane"],
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            result_path=pathlib.Path(sample["result_file"]),
            ranks=sample["ranks"],
            capture_field_times=True,
        ):
            raise RuntimeError(f"{label} raw producer command changed")
        RUNNER.COMPARATOR.validate_qualification_binding(
            raw,
            expected_nonce=sample["nonce"],
            expected_run_id=expected_run_id,
            expected_receipt_id=expected_receipt_id,
            expected_source_snapshot_sha256=expected_snapshot["sha256"],
            expected_producer_sha256=expected_producer_sha256,
            expected_sample_kind=sample["sample_kind"],
            expected_sample_lane=sample["lane"],
            expected_sample_iteration=sample["iteration"],
            label=label,
            hash_cache=cache,
        )
        RUNNER._fixed_workload_gate(raw, lane=sample["lane"])
        lazy_import_gate = RUNNER._validate_lazy_import_contract(raw, label=label)
        validation = RUNNER.COMPARATOR.validate_distributed_record(
            raw,
            expected_world_size=sample["ranks"],
            label=label,
            hash_cache=cache,
        )
        validation["rank_driver"] = RUNNER._validate_cuda_rank_driver_evidence(
            raw, validation, label=label
        )
        receipt_gate = RUNNER._validate_receipt_rank_runtimes(
            receipt, validation, hash_cache=cache
        )
        process_environment_gate = RUNNER._validate_actual_process_environments(
            raw, expected_environment, label=label
        )
        RUNNER._validate_lane_process_environment(
            raw, expected_environment, lane=sample["lane"], label=label
        )
        material_gradient_gate = RUNNER._extract_material_gradient_stats(
            stdout, lane=sample["lane"]
        )
        if (
            sample["process_seconds"]
            < validation.get("workload_wall_seconds", math.inf)
            or sample.get("validation") != validation
            or sample.get("receipt_gate") != receipt_gate
            or sample.get("process_environment_gate") != process_environment_gate
            or sample.get("lazy_import_gate") != lazy_import_gate
            or sample.get("material_gradient_gate") != material_gradient_gate
        ):
            raise RuntimeError(f"{label} fresh-derived validation evidence disagrees")
        checked.append(
            {
                "label": label,
                "nonce": sample["nonce"],
                "completion_policy": sample["completion_policy"],
                "summary_sha256": PROVENANCE.canonical_sha256(
                    RUNNER.summarize_samples([sample])[0]
                ),
                "files": files,
            }
        )
    validate_sample_matrix(samples)
    compare_samples(samples)
    reconstructed_ledger = _expected_controller_ledger(
        output=pathlib.Path(samples[0]["result_file"]).resolve().parent,
        run_id=expected_run_id,
        samples=samples,
    )
    if expected_controller_ledger != reconstructed_ledger:
        raise RuntimeError("completion A/B controller publication ledger disagrees")
    return {
        "pass": True,
        "verified_at_utc": RUNNER._utc_now(),
        "receipt_id": expected_receipt_id,
        "receipt_sha256": cache.digest(receipt_path),
        "source_snapshot": expected_snapshot,
        "contract": contract,
        "samples": checked,
        "controller_ledger": {
            "pass": True,
            "entries": len(samples),
            "sha256": PROVENANCE.canonical_sha256(reconstructed_ledger),
        },
    }


def _expected_controller_ledger(
    *, output: pathlib.Path, run_id: str, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    measurements: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    canonical_output = str(output.resolve())
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != RUNNER.EXPECTED_SAMPLE_KEYS:
            raise RuntimeError("completion A/B sample exact schema disagrees")
        label = sample["artifact_label"]
        key = f"sample:{label}"
        measurements[key] = {
            "role": "sample",
            "run_id": run_id,
            "output": canonical_output,
            "label": label,
            "lane": sample["lane"],
            "sample_kind": sample["sample_kind"],
            "iteration": sample["iteration"],
            "nonce": sample["nonce"],
            "process_seconds": sample["process_seconds"],
            "command_sha256": PROVENANCE.canonical_sha256(sample["command"]),
            "environment_sha256": PROVENANCE.canonical_sha256(
                sample["environment"]
            ),
            "artifact_paths": {
                name: sample[name]
                for name in (
                    "result_file",
                    "stdout_log",
                    "stderr_log",
                    "timing_file",
                )
            },
        }
        artifacts[key] = {
            "role": "sample",
            "label": label,
            "artifact_sha256": {
                "result_file": sample["result_sha256"],
                "stdout_log": sample["stdout_sha256"],
                "stderr_log": sample["stderr_sha256"],
                "timing_file": sample["timing_sha256"],
            },
        }
    return {"measurements": measurements, "artifacts": artifacts}


def _artifact_records(
    output: pathlib.Path, excluded_names: set[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.parent == output and path.name in excluded_names:
            continue
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": EVIDENCE.sha256_file(path),
            }
        )
    return records


def _summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in sample.items()
            if key not in {"result", "validation"}
        }
        for sample in samples
    ]


def _postprocessor_sources() -> tuple[str, ...]:
    return (
        "scripts/run-mpi-completion-ab.py",
        "scripts/run-mpi-adjoint-benchmark.py",
        "scripts/benchmark-adjoint.py",
        "scripts/compare-mpi-adjoint-benchmarks.py",
        "scripts/gpmeep_benchmark_evidence.py",
        "scripts/gpmeep_provenance.py",
    )


def _render_markdown(comparison: dict[str, Any], receipt_id: str) -> str:
    promotion = comparison["promotion"]
    performance = comparison["performance"]
    return (
        "# gpmeep fixed MPI completion-policy A/B\n\n"
        "- Evidence state: **COMPLETE**\n"
        f"- Decision: **{promotion['decision']}**\n"
        f"- Fixed promotion floor: `{promotion['fixed_minimum_speedup']:.3f}x`\n"
        f"- Workload-wall Waitsome/Waitall: "
        f"`{performance['workload_wall']['waitsome_over_waitall_median_speedup']:.6f}x`\n"
        f"- Balanced fresh-process Waitsome/Waitall: "
        f"`{performance['fresh_process_wall']['by_position']['balanced_geometric_mean_speedup']:.6f}x`\n"
        f"- Fresh-process first-position stratum: "
        f"`{performance['fresh_process_wall']['by_position']['waitsome_over_waitall_speedup']['first']:.6f}x`\n"
        f"- Fresh-process second-position stratum: "
        f"`{performance['fresh_process_wall']['by_position']['waitsome_over_waitall_speedup']['second']:.6f}x`\n"
        f"- Strong process position-effect warning: "
        f"`{performance['fresh_process_wall']['position_effect_diagnostic']['warning']}`\n"
        f"- Waitall faster pairs: `{promotion['waitall_faster_pairs']}/"
        f"{PROFILE['measured_repeats_per_policy']}`\n"
        f"- Receipt: `{receipt_id}`\n"
    )


def _validate_report_for_publish(
    *,
    report: dict[str, Any],
    markdown: str,
    run_id: str,
    repo: pathlib.Path,
    raw_output: pathlib.Path,
    expected_snapshot: dict[str, Any],
    expected_build_receipt: dict[str, Any],
    expected_integrity: dict[str, Any],
    fresh_integrity: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    expected_keys = {
        "schema_version",
        "state",
        "run_id",
        "profile",
        "profile_sha256",
        "source_snapshot",
        "build_receipt",
        "sample_matrix",
        "raw_artifact_set",
        "samples",
        "comparison",
        "integrity_reverification",
        "postprocessor",
    }
    comparison = compare_samples(samples)
    if (
        not isinstance(report, dict)
        or set(report) != expected_keys
        or report.get("schema_version") != 1
        or report.get("state") != "COMPLETE"
        or report.get("run_id") != run_id
        or report.get("profile") != dict(PROFILE)
        or report.get("profile_sha256") != PROFILE_CANONICAL_SHA256
        or report.get("source_snapshot") != expected_snapshot
        or report.get("build_receipt") != expected_build_receipt
        or report.get("sample_matrix") != validate_sample_matrix(samples)
        or report.get("raw_artifact_set")
        != validate_raw_artifact_set(raw_output, samples)
        or report.get("samples") != _summarize_samples(samples)
        or report.get("comparison") != comparison
        or report.get("integrity_reverification") != expected_integrity
    ):
        raise RuntimeError("completion A/B report is not freshly derived")
    first_projection = dict(expected_integrity)
    fresh_projection = dict(fresh_integrity)
    first_projection.pop("verified_at_utc", None)
    fresh_projection.pop("verified_at_utc", None)
    if first_projection != fresh_projection:
        raise RuntimeError("completion A/B final reverification changed")
    postprocessor = report.get("postprocessor")
    expected_postprocessor_keys = {
        "python",
        "platform",
        "source_sha256",
        "nvidia_driver",
        "nvidia_topology",
        "driver_device_binding",
    }
    if (
        not isinstance(postprocessor, dict)
        or set(postprocessor) != expected_postprocessor_keys
        or postprocessor.get("python")
        != str(pathlib.Path(sys.executable).resolve())
        or postprocessor.get("platform") != platform.platform()
        or postprocessor.get("source_sha256")
        != {
            name: EVIDENCE.sha256_file(repo / name)
            for name in _postprocessor_sources()
        }
    ):
        raise RuntimeError("completion A/B postprocessor is not source-bound")
    fresh_driver = RUNNER._nvidia_smi_probe(
        [
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ],
        repo,
    )
    fresh_topology = RUNNER._nvidia_smi_probe(["-L"], repo)
    if (
        postprocessor.get("nvidia_driver") != fresh_driver
        or postprocessor.get("nvidia_topology") != fresh_topology
        or postprocessor.get("driver_device_binding")
        != RUNNER.validate_driver_device_binding(
            fresh_driver, comparison["devices"], fresh_topology, repo=repo
        )
    ):
        raise RuntimeError("completion A/B driver binding changed before publication")
    if markdown != _render_markdown(
        comparison, expected_build_receipt["receipt_id"]
    ):
        raise RuntimeError("completion A/B Markdown is not freshly derived")


def _publish(
    *,
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
    markdown: str,
    publication_capability: RUNNER._RunPublicationCapability,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    expected_build_receipt: dict[str, Any],
    expected_integrity: dict[str, Any],
    expected_controller_ledger: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    _assert_fixed_profile()
    if not isinstance(publication_capability, RUNNER._RunPublicationCapability):
        raise RuntimeError("completion A/B publication capability is absent")
    controller_ledger = publication_capability.snapshot(
        output / "raw", run_id, consume=True
    )
    if controller_ledger != expected_controller_ledger:
        raise RuntimeError("completion A/B live controller ledger changed")
    fresh_integrity = _reverify(
        expected_run_id=run_id,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=expected_receipt_id,
        expected_snapshot=expected_snapshot,
        expected_producer_sha256=expected_producer_sha256,
        samples=samples,
        expected_controller_ledger=expected_controller_ledger,
    )
    _validate_report_for_publish(
        report=report,
        markdown=markdown,
        run_id=run_id,
        repo=repo,
        raw_output=output / "raw",
        expected_snapshot=expected_snapshot,
        expected_build_receipt=expected_build_receipt,
        expected_integrity=expected_integrity,
        fresh_integrity=fresh_integrity,
        samples=samples,
    )
    report["final_publish_reverification"] = fresh_integrity
    report_path = output / "report.json"
    markdown_path = output / "report.md"
    EVIDENCE.atomic_write_text(markdown_path, markdown)
    report["report_markdown"] = {
        "path": str(markdown_path.resolve()),
        "sha256": EVIDENCE.sha256_file(markdown_path),
    }
    EVIDENCE.atomic_write_json(report_path, report)
    records = _artifact_records(
        output, {"artifacts.sha256.json", "COMPLETE", "FAILED.json", "state.json"}
    )
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "records": records,
        "records_sha256": PROVENANCE.canonical_sha256(records),
    }
    manifest_path = output / "artifacts.sha256.json"
    EVIDENCE.atomic_write_json(manifest_path, manifest)
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "profile_sha256": PROFILE_CANONICAL_SHA256,
        "receipt_id": report["build_receipt"]["receipt_id"],
        "source_snapshot_sha256": report["source_snapshot"]["sha256"],
        "report": {
            "path": str(report_path.resolve()),
            "sha256": EVIDENCE.sha256_file(report_path),
        },
        "artifacts_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": EVIDENCE.sha256_file(manifest_path),
            "records_sha256": manifest["records_sha256"],
        },
    }
    EVIDENCE.atomic_write_json(output / "COMPLETE", marker)
    # COMPLETE is the sole authoritative success marker. The mutable RUNNING
    # convenience state is removed instead of publishing an unbound duplicate.
    (output / "state.json").unlink(missing_ok=True)
    expected_regular = {
        record["path"] for record in records
    } | {"artifacts.sha256.json", "COMPLETE"}
    actual_regular = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_regular != expected_regular:
        (output / "COMPLETE").unlink(missing_ok=True)
        raise RuntimeError("unbound artifact appeared during completion A/B publication")
    try:
        replay_marker = json.loads(
            (output / "COMPLETE").read_text(encoding="utf-8")
        )
        replay_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        replay_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            replay_marker != marker
            or replay_manifest != manifest
            or replay_report != report
            or marker["report"]["sha256"] != EVIDENCE.sha256_file(report_path)
            or marker["artifacts_manifest"]["sha256"]
            != EVIDENCE.sha256_file(manifest_path)
            or any(
                record.get("size_bytes") != (output / record["path"]).stat().st_size
                or record.get("sha256")
                != EVIDENCE.sha256_file(output / record["path"])
                for record in manifest["records"]
            )
        ):
            raise RuntimeError("completion A/B terminal artifact replay disagrees")
        replay_integrity = _reverify(
            expected_run_id=run_id,
            repo=repo,
            receipt_path=receipt_path,
            expected_receipt_id=expected_receipt_id,
            expected_snapshot=expected_snapshot,
            expected_producer_sha256=expected_producer_sha256,
            samples=samples,
            expected_controller_ledger=expected_controller_ledger,
        )
        _validate_report_for_publish(
            report={
                name: value
                for name, value in replay_report.items()
                if name not in {"final_publish_reverification", "report_markdown"}
            },
            markdown=markdown_path.read_text(encoding="utf-8"),
            run_id=run_id,
            repo=repo,
            raw_output=output / "raw",
            expected_snapshot=expected_snapshot,
            expected_build_receipt=expected_build_receipt,
            expected_integrity=expected_integrity,
            fresh_integrity=replay_integrity,
            samples=samples,
        )
    except BaseException:
        (output / "COMPLETE").unlink(missing_ok=True)
        raise


def execute(args: argparse.Namespace, repo: pathlib.Path, run_id: str) -> int:
    _assert_fixed_profile()
    cache = PROVENANCE.StatHashCache()
    snapshot = PROVENANCE.source_snapshot(repo, hash_cache=cache)
    receipt_path = args.build_receipt.resolve()
    receipt = PROVENANCE.verify_build_receipt(
        receipt_path, repo, hash_cache=cache
    )
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise RuntimeError("completion A/B requires a CUDA+MPI+Python FP32 receipt")
    contract = RUNNER.validate_build_receipt_contract(
        receipt, repo, hash_cache=cache
    )
    if receipt.get("source_end") != snapshot:
        raise RuntimeError("completion A/B receipt does not match current source")
    runtime = RUNNER._receipt_runtime(receipt, repo)
    producer = repo / "scripts" / "benchmark-adjoint.py"
    producer_sha256 = cache.digest(producer)
    copied_receipt = args.output / "build-provenance.json"
    shutil.copyfile(receipt_path, copied_receipt)
    if EVIDENCE.sha256_file(copied_receipt) != cache.digest(receipt_path):
        raise RuntimeError("copied completion A/B receipt is invalid")

    raw_output = args.output / "raw"
    raw_output.mkdir()
    publication_capability = RUNNER._RunPublicationCapability(raw_output, run_id)
    samples: list[dict[str, Any]] = []
    for policy, sample_kind, iteration in _expected_matrix():
        label = f"{run_id}-{sample_kind}-{iteration}-{policy}"
        samples.append(
            RUNNER._run_raw_sample(
                repo=repo,
                output=raw_output,
                runtime=runtime,
                receipt=receipt,
                snapshot_sha256=snapshot["sha256"],
                producer_sha256=producer_sha256,
                run_id=run_id,
                lane="cuda-multi",
                sample_kind=sample_kind,
                iteration=iteration,
                completion_policy=policy,
                artifact_label=label,
                hash_cache=cache,
                capture_field_times=True,
                publication_capability=publication_capability,
            )
        )
    publication_capability.seal()
    controller_ledger = publication_capability.snapshot(raw_output, run_id)
    if controller_ledger != _expected_controller_ledger(
        output=raw_output, run_id=run_id, samples=samples
    ):
        raise RuntimeError("completion A/B controller publication ledger disagrees")

    sample_matrix = validate_sample_matrix(samples)
    raw_artifact_set = validate_raw_artifact_set(raw_output, samples)
    comparison = compare_samples(samples)
    driver_probe = RUNNER._nvidia_smi_probe(
        [
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ],
        repo,
    )
    topology_probe = RUNNER._nvidia_smi_probe(["-L"], repo)
    driver_binding = RUNNER.validate_driver_device_binding(
        driver_probe,
        comparison["devices"],
        topology_probe,
        repo=repo,
    )
    final_reverification = _reverify(
        expected_run_id=run_id,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=snapshot,
        expected_producer_sha256=producer_sha256,
        samples=samples,
        expected_controller_ledger=controller_ledger,
    )
    if PROVENANCE.source_snapshot(repo, hash_cache=cache) != snapshot:
        raise RuntimeError("source changed before completion A/B publication")

    summarized_samples = _summarize_samples(samples)
    build_receipt_record = {
        "original_path": str(receipt_path),
        "copied_path": str(copied_receipt.resolve()),
        "sha256": EVIDENCE.sha256_file(copied_receipt),
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "contract": contract,
    }
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "profile": dict(PROFILE),
        "profile_sha256": PROFILE_CANONICAL_SHA256,
        "source_snapshot": snapshot,
        "build_receipt": build_receipt_record,
        "sample_matrix": sample_matrix,
        "raw_artifact_set": raw_artifact_set,
        "samples": summarized_samples,
        "comparison": comparison,
        "integrity_reverification": final_reverification,
        "postprocessor": {
            "python": str(pathlib.Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "source_sha256": {
                name: cache.digest(repo / name) for name in _postprocessor_sources()
            },
            "nvidia_driver": driver_probe,
            "nvidia_topology": topology_probe,
            "driver_device_binding": driver_binding,
        },
    }
    markdown = _render_markdown(comparison, receipt["receipt_id"])
    _publish(
        output=args.output,
        run_id=run_id,
        report=report,
        markdown=markdown,
        publication_capability=publication_capability,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=snapshot,
        expected_producer_sha256=producer_sha256,
        expected_build_receipt=build_receipt_record,
        expected_integrity=final_reverification,
        expected_controller_ledger=controller_ledger,
        samples=samples,
    )
    print(markdown, end="")
    return 0


def _mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    for name in ("COMPLETE", "report.json", "report.md", "artifacts.sha256.json"):
        (output / name).unlink(missing_ok=True)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "profile": dict(PROFILE),
        "profile_sha256": PROFILE_CANONICAL_SHA256,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    EVIDENCE.atomic_write_json(output / "FAILED.json", failure)
    EVIDENCE.atomic_write_json(output / "state.json", failure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = SCRIPT_DIRECTORY.parent
    run_id = uuid.uuid4().hex
    output = args.output.resolve()
    prepared = False
    try:
        _prepare_output(output, run_id)
        prepared = True
        args.output = output
        return execute(args, repo, run_id)
    except BaseException as error:
        if prepared:
            _mark_failed(output, run_id, error)
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
