#!/usr/bin/env python3
"""Compare paired, development-only MPI CPU and CUDA benchmark records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from fractions import Fraction


METRIC_SCALARS = ("l1", "l2", "maximum_absolute", "sum")
NATIVE_GROUPS = {
    "dispatch": ("curl_calls", "curl_points"),
    "field_updates": ("update_eh_calls", "update_eh_points"),
    "polarizations": ("polarization_calls", "polarization_points"),
    "sources": ("source_calls", "source_points"),
    "boundaries": ("boundary_calls", "boundary_points"),
    "dfts": ("dft_calls", "dft_points"),
}
WORKLOAD = {
    "sx": 40.0,
    "sy": 40.0,
    "dpml": 1.0,
    "source_speed": 0.7,
    "source_frequency": 1e-10,
}


def expected_workload_contract(resolution):
    require(
        isinstance(resolution, int)
        and not isinstance(resolution, bool)
        and resolution > 0,
        "invalid moving-source resolution",
    )
    sx = Fraction(str(WORKLOAD["sx"]))
    sy = Fraction(str(WORKLOAD["sy"]))
    speed = Fraction(str(WORKLOAD["source_speed"]))
    courant = Fraction(1, 2)
    requested = sx / speed
    timestep = courant / resolution
    ratio = requested / timestep
    steps = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return {
        **WORKLOAD,
        "courant": float(courant),
        "requested_until": float(requested),
        "expected_timesteps": steps,
        "expected_meep_time": float(steps * timestep),
        "expected_source_updates": steps + 1,
        "expected_field_shape": [int(sx * resolution), int(sy * resolution)],
    }


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args():
    return argument_parser().parse_args()


def parse_output_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_known_args()[0].output


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_record(path):
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def invalidate_output(path):
    try:
        Path(path).resolve().unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot invalidate previous comparison output: {exc}") from exc


def atomic_json(path, value):
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(target)


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def load_record(path):
    with Path(path).open("rb") as source:
        return json.load(
            source,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def relative_difference(reference, candidate):
    require(finite_number(reference) and finite_number(candidate),
            "metric values must be finite")
    absolute = abs(float(reference) - float(candidate))
    scale = max(abs(float(reference)), abs(float(candidate)))
    return {"absolute": absolute, "relative": absolute / scale if scale else 0.0}


def timing_summary(samples):
    values = [float(value) for value in samples]
    require(values and all(math.isfinite(value) and value > 0 for value in values),
            "timing samples must be finite and positive")
    median = statistics.median(values)
    return {
        "samples_seconds": values,
        "median_seconds": median,
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "relative_span": (max(values) - min(values)) / median,
    }


def validate_file_record(record, name):
    require(isinstance(record, dict), f"missing {name} file record")
    require(isinstance(record.get("path"), str) and record["path"],
            f"invalid {name} path")
    require(isinstance(record.get("size_bytes"), int) and
            record["size_bytes"] > 0, f"invalid {name} size")
    digest = record.get("sha256")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(character in "0123456789abcdef" for character in digest),
            f"invalid {name} digest")


def validate_runtime(record, ranks):
    runtime = record.get("runtime")
    require(isinstance(runtime, dict), "missing runtime identity")
    for name in ("python", "extension", "mapped_libmeep"):
        validate_file_record(runtime.get(name), name)
    platform_identity = runtime.get("platform")
    require(isinstance(platform_identity, list) and len(platform_identity) == 6 and
            all(isinstance(value, str) for value in platform_identity),
            "invalid platform identity")
    cpu = runtime.get("cpu_identity")
    require(isinstance(cpu, dict) and cpu and
            all(value is None or isinstance(value, str) for value in cpu.values()),
            "invalid CPU identity")
    rank_runtime = record.get("runtime_by_rank")
    require(isinstance(rank_runtime, list) and len(rank_runtime) == ranks,
            "runtime rank count mismatch")
    require(all(item == runtime for item in rank_runtime),
            "rank runtime identities differ")
    return runtime


def validate_source_snapshot(snapshot):
    require(isinstance(snapshot, dict), "missing complete source snapshot")
    require(isinstance(snapshot.get("head"), str) and snapshot["head"],
            "invalid source HEAD")
    for name in ("index_sha256", "entries_sha256"):
        digest = snapshot.get(name)
        require(isinstance(digest, str) and len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                f"invalid source {name}")
    entries = snapshot.get("entries")
    require(isinstance(entries, list) and snapshot.get("entry_count") == len(entries),
            "source entry count mismatch")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    require(len(paths) == len(entries) and paths == sorted(paths) and
            len(set(paths)) == len(paths) and
            all(isinstance(path, str) and path for path in paths),
            "source entries are incomplete, duplicated, or unsorted")
    for entry in entries:
        kind = entry.get("kind")
        require(kind in ("file", "symlink", "missing"),
                "invalid source entry kind")
        if kind != "missing":
            require(isinstance(entry.get("mode"), int), "invalid source entry mode")
            digest = entry.get("sha256")
            require(isinstance(digest, str) and len(digest) == 64 and
                    all(character in "0123456789abcdef" for character in digest),
                    "invalid source entry digest")
        if kind == "file":
            require(isinstance(entry.get("size_bytes"), int) and
                    entry["size_bytes"] >= 0, "invalid source file size")
        if kind == "symlink":
            require(isinstance(entry.get("target"), str),
                    "invalid source symlink target")
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    require(snapshot["entries_sha256"] == sha256_bytes(payload),
            "source entry manifest digest mismatch")


def core_key(value):
    require(isinstance(value, dict), "invalid physical-core identity")
    host = value.get("host")
    package = value.get("package_id")
    core = value.get("core_id")
    require(isinstance(host, str) and host and isinstance(package, int) and
            isinstance(core, int), "invalid physical-core identity")
    return host, package, core


def validate_execution(record, expected_backend, ranks, threads):
    execution = record.get("execution_by_rank")
    require(isinstance(execution, list) and len(execution) == ranks,
            "execution rank count mismatch")
    require(sorted(item.get("rank") for item in execution) == list(range(ranks)),
            "execution ranks are incomplete or duplicated")
    occupied = set()
    identifiers = []
    allow_oversubscribe = str(
        record.get("environment", {}).get("allow_oversubscribe", "0")
    ).lower() in ("1", "true", "yes")
    for item in execution:
        affinity = item.get("physical_core_affinity")
        require(isinstance(affinity, list), "missing physical-core affinity")
        cores = {core_key(core) for core in affinity}
        require(len(cores) == threads,
                "rank physical-core affinity does not match OMP thread count")
        require(not occupied.intersection(cores),
                "MPI ranks overlap on a physical CPU core")
        occupied.update(cores)
        device = item.get("selected_device")
        identifier = item.get("selected_device_identifier")
        if expected_backend == "cpu":
            require(device == -1 and identifier == "",
                    "CPU record retained a CUDA device assignment")
        else:
            require(isinstance(device, int) and device >= 0 and
                    isinstance(identifier, str) and identifier,
                    "CUDA record has no physical device identity")
            identifiers.append(identifier)
    require(len(occupied) == record["total_host_thread_budget"],
            "physical-core budget differs from declared host budget")
    if expected_backend == "cuda" and not allow_oversubscribe:
        require(len(set(identifiers)) == len(identifiers),
                "CUDA ranks selected duplicate physical devices")
    return sorted(occupied)


def counter(statistics, group, name):
    value = statistics.get(group, {}).get(name)
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"missing or invalid native counter {group}.{name}")
    return value


def counter_total(statistics_by_rank, group, name):
    return sum(counter(statistics, group, name) for statistics in statistics_by_rank)


def validate_statistics(record, expected_backend, ranks):
    values = record.get("statistics_by_rank")
    require(isinstance(values, list) and len(values) == ranks,
            "statistics rank count mismatch")
    for statistics in values:
        require(isinstance(statistics, dict), "invalid native statistics record")
        for name in (
            "runtime_availability_probes",
            "runtime_device_enumerations",
            "runtime_device_selections",
        ):
            counter(statistics, "runtime", name)
        for group, suffixes in NATIVE_GROUPS.items():
            for backend in ("cpu", "cuda"):
                for suffix in suffixes:
                    counter(statistics, group, f"{backend}_{suffix}")
        for name in (
            "mpi_messages", "mpi_scalars", "cuda_aware_bytes",
            "pinned_staging_bytes", "pinned_device_to_host_bytes",
            "pinned_host_to_device_bytes",
        ):
            counter(statistics, "multi_gpu", name)

    opposite = "cuda" if expected_backend == "cpu" else "cpu"
    for group, suffixes in NATIVE_GROUPS.items():
        for suffix in suffixes:
            require(counter_total(values, group, f"{opposite}_{suffix}") == 0,
                    f"{expected_backend} run used {opposite} fallback in {group}")
    for group in ("dispatch", "field_updates", "sources", "boundaries"):
        call_suffix = NATIVE_GROUPS[group][0]
        point_suffix = NATIVE_GROUPS[group][1]
        require(counter_total(values, group, f"{expected_backend}_{call_suffix}") > 0 and
                counter_total(values, group, f"{expected_backend}_{point_suffix}") > 0,
                f"{expected_backend} run did not execute native {group}")

    runtime_totals = [
        counter_total(values, "runtime", name)
        for name in (
            "runtime_availability_probes",
            "runtime_device_enumerations",
            "runtime_device_selections",
        )
    ]
    if expected_backend == "cpu":
        require(runtime_totals == [0, 0, 0],
                "CPU baseline touched the CUDA runtime")
    else:
        require(all(value > 0 for value in runtime_totals),
                "CUDA run did not record runtime discovery and selection")
        if ranks > 1:
            messages = counter_total(values, "multi_gpu", "mpi_messages")
            scalars = counter_total(values, "multi_gpu", "mpi_scalars")
            cuda_bytes = counter_total(values, "multi_gpu", "cuda_aware_bytes")
            pinned_bytes = counter_total(values, "multi_gpu", "pinned_staging_bytes")
            require(messages > 0 and scalars > 0 and (cuda_bytes > 0) != (pinned_bytes > 0),
                    "CUDA MPI run did not record one real native transport")
            transport = record.get("environment", {}).get("gpu_mpi_transport", "auto")
            if transport == "cuda-aware":
                require(cuda_bytes > 0 and pinned_bytes == 0,
                        "CUDA-aware record contains the wrong transport bytes")
            elif transport == "pinned":
                require(pinned_bytes > 0 and cuda_bytes == 0,
                        "pinned record contains the wrong transport bytes")


def validate_repeat_statistics(record, expected_backend, ranks):
    previous = {group: 0 for group in ("dispatch", "field_updates", "sources", "boundaries")}
    last = None
    for repeat_samples in record["rank_samples"]:
        ordered = sorted(repeat_samples, key=lambda sample: sample["rank"])
        snapshot = [sample.get("statistics") for sample in ordered]
        snapshot_record = dict(record)
        snapshot_record["statistics_by_rank"] = snapshot
        validate_statistics(snapshot_record, expected_backend, ranks)
        for group in previous:
            suffix = NATIVE_GROUPS[group][0]
            current = counter_total(
                snapshot, group, f"{expected_backend}_{suffix}"
            )
            require(current > previous[group],
                    f"native {group} counters did not advance in every repeat")
            previous[group] = current
        last = snapshot
    def without_live_state(statistics_by_rank):
        result = []
        for statistics in statistics_by_rank:
            copied = {
                group: dict(values) if isinstance(values, dict) else values
                for group, values in statistics.items()
            }
            resident = copied.get("resident")
            if isinstance(resident, dict):
                resident.pop("live_resident_device_buffers", None)
            result.append(copied)
        return result

    # The worker records each repeat before Simulation.reset_meep(), then the
    # final snapshot after reset.  Reset must drop the instantaneous live
    # buffer gauge to zero while leaving every cumulative counter unchanged.
    # Comparing the gauge made a correct CUDA cleanup look like evidence
    # corruption; retain exact comparison for all non-live statistics.
    require(
        without_live_state(last)
        == without_live_state(record["statistics_by_rank"]),
        "final cumulative statistics disagree with the last repeat snapshot",
    )


def validate_metric(metric, contract):
    require(isinstance(metric, dict), "invalid numerical metric")
    shape = metric.get("shape")
    require(isinstance(shape, list) and shape and
            all(isinstance(value, int) and value > 0 for value in shape),
            "invalid field shape")
    require(shape == contract.get("expected_field_shape"),
            "unexpected moving-source field shape")
    require(metric.get("source_updates") == contract.get("expected_source_updates") and
            isinstance(metric.get("source_updates"), int) and
            metric["source_updates"] > 0,
            "unexpected moving-source update count")
    meep_time = metric.get("meep_time")
    expected_time = contract.get("expected_meep_time")
    requested_until = contract.get("requested_until")
    require(finite_number(meep_time) and finite_number(expected_time) and
            math.isclose(float(meep_time), float(expected_time), rel_tol=0.0, abs_tol=1e-9),
            "unexpected Meep end time")
    require(finite_number(requested_until) and requested_until > 0 and
            float(expected_time) >= float(requested_until),
            "invalid requested/expected Meep time contract")
    require(all(finite_number(metric.get(name)) for name in METRIC_SCALARS),
            "numerical metrics must be finite")


def validate_record(record, expected_backend):
    require(record.get("schema_version") == 2, "unsupported benchmark schema")
    require(record.get("development_only") is True,
            "input must be development-only evidence")
    require(record.get("authoritative_release_evidence") is False,
            "development input must not claim release authority")
    require(record.get("performance_claim_eligible") is False,
            "development input must not claim performance eligibility")
    require(record.get("backend") == expected_backend, "backend mismatch")
    require(isinstance(record.get("resolution"), int) and record["resolution"] > 0,
            "invalid resolution")
    require(isinstance(record.get("repeat"), int) and record["repeat"] > 0,
            "invalid repeat")
    ranks = record.get("mpi_ranks")
    threads = record.get("omp_threads_per_rank")
    require(isinstance(ranks, int) and ranks > 0 and
            isinstance(threads, int) and threads > 0, "invalid rank/thread count")
    require(record.get("total_host_thread_budget") == ranks * threads,
            "host thread budget mismatch")
    environment = record.get("environment")
    require(isinstance(environment, dict) and
            environment.get("omp_num_threads") == str(threads),
            "OMP_NUM_THREADS evidence disagrees with the thread contract")
    require(environment.get("gpu_mpi_transport", "auto") in
            ("auto", "pinned", "cuda-aware"),
            "invalid recorded GPU MPI transport")
    require(str(environment.get("allow_oversubscribe", "0")).lower() in
            ("0", "false", "no", "1", "true", "yes"),
            "invalid recorded oversubscription policy")
    if expected_backend == "cpu":
        require(ranks > 1, "CPU baseline must use multiple MPI stencil workers")
        require(threads == 1, "CPU baseline must use one thread per MPI rank")

    source = record.get("source_state")
    require(isinstance(source, dict) and source.get("start") == source.get("end") and
            isinstance(source.get("start"), dict),
            "source changed during benchmark execution")
    validate_source_snapshot(source["start"])
    contract = record.get("workload_contract")
    require(isinstance(contract, dict), "missing workload contract")
    require(
        contract == expected_workload_contract(record["resolution"]),
        "moving-source workload contract is not independently derived",
    )
    repeat = record["repeat"]
    solver = record.get("solver_seconds")
    metrics = record.get("metrics")
    global_work_totals = record.get("global_work_totals")
    owners = record.get("owners")
    samples = record.get("rank_samples")
    require(isinstance(solver, list) and len(solver) == repeat, "timing repeat mismatch")
    require(isinstance(metrics, list) and len(metrics) == repeat, "metric repeat mismatch")
    require(isinstance(global_work_totals, list) and
            len(global_work_totals) == repeat, "global work total repeat mismatch")
    require(isinstance(owners, list) and len(owners) == repeat, "owner repeat mismatch")
    require(isinstance(samples, list) and len(samples) == repeat,
            "rank sample repeat mismatch")
    timing = timing_summary(solver)
    runtime = validate_runtime(record, ranks)
    physical_cores = validate_execution(record, expected_backend, ranks, threads)
    execution_by_rank = {
        item["rank"]: item for item in record["execution_by_rank"]
    }

    reference_metric = None
    for repeat_index, (metric, owner, rank_samples) in enumerate(
        zip(metrics, owners, samples)
    ):
        validate_metric(metric, contract)
        require(
            global_work_totals[repeat_index] == {
                "field_sample_count": math.prod(metric["shape"]),
                "source_updates": metric["source_updates"],
                "meep_time": metric["meep_time"],
            },
            "global work totals disagree with numerical metrics",
        )
        if reference_metric is None:
            reference_metric = metric
        else:
            require(metric == reference_metric,
                    "repeat changed deterministic moving-source metrics")
        require(isinstance(owner, dict) and
                owner.get("selected_backend") == expected_backend and
                owner.get("process_backend") == expected_backend,
                "record selected the wrong backend")
        require(isinstance(rank_samples, list) and len(rank_samples) == ranks,
                "rank sample count mismatch")
        require(sorted(sample.get("rank") for sample in rank_samples) == list(range(ranks)),
                "rank samples are incomplete or duplicated")
        for sample in rank_samples:
            local_seconds = sample.get("local_seconds")
            require(finite_number(local_seconds) and 0 < local_seconds <= solver[repeat_index],
                    "invalid rank-local timing")
            require(sample.get("metric") == metric,
                    "one rank observed a different global workload or metric")
            sample_owner = sample.get("owner", {})
            require(sample_owner.get("selected_backend") == expected_backend and
                    sample_owner.get("process_backend") == expected_backend,
                    "one rank selected the wrong backend")
            affinity = sample.get("physical_core_affinity")
            require(isinstance(affinity, list) and
                    {core_key(core) for core in affinity} ==
                    {core_key(core) for core in execution_by_rank[sample["rank"]]["physical_core_affinity"]},
                    "rank affinity changed during benchmark repeats")

    validate_statistics(record, expected_backend, ranks)
    validate_repeat_statistics(record, expected_backend, ranks)
    return {
        "timing": timing,
        "runtime": runtime,
        "physical_cores": physical_cores,
    }


def main(arguments=None):
    if arguments is None:
        output_path = parse_output_path()
        invalidate_output(output_path)
        args = parse_args()
        if args.output.resolve() != output_path.resolve():
            raise RuntimeError("parsed output path changed during argument validation")
    else:
        args = arguments
        invalidate_output(args.output)
    cpu = load_record(args.cpu)
    cuda = load_record(args.cuda)
    cpu_validation = validate_record(cpu, "cpu")
    cuda_validation = validate_record(cuda, "cuda")
    require(cpu["source_state"]["start"] == cuda["source_state"]["start"],
            "CPU and CUDA runs used different complete source snapshots")
    require(cpu["runtime"] == cuda["runtime"],
            "CPU and CUDA runs used different Python/native runtime artifacts")
    require(cpu["resolution"] == cuda["resolution"], "resolution mismatch")
    require(cpu["repeat"] == cuda["repeat"], "repeat mismatch")
    require(cpu["workload_contract"] == cuda["workload_contract"],
            "CPU and CUDA workload contracts differ")
    require(cpu["global_work_totals"] == cuda["global_work_totals"],
            "CPU and CUDA global work totals differ")
    require(cpu["total_host_thread_budget"] == cuda["total_host_thread_budget"],
            "total host thread budgets differ")
    require(cpu_validation["physical_cores"] == cuda_validation["physical_cores"],
            "CPU and CUDA runs used different physical-core budgets")

    metric_differences = []
    for cpu_metric, cuda_metric in zip(cpu["metrics"], cuda["metrics"]):
        for key in ("shape", "source_updates", "meep_time"):
            require(cpu_metric.get(key) == cuda_metric.get(key),
                    "CPU and CUDA global workloads differ")
        metric_differences.append(
            {
                key: relative_difference(cpu_metric[key], cuda_metric[key])
                for key in METRIC_SCALARS
            }
        )

    cpu_timing = cpu_validation["timing"]
    cuda_timing = cuda_validation["timing"]
    output = {
        "schema_version": 2,
        "development_only": True,
        "authoritative_release_evidence": False,
        "performance_claim_eligible": False,
        "reason": (
            "paired uncommitted-source diagnostic; scalar summaries do not prove pointwise numerical agreement"
        ),
        "producer": file_record(__file__),
        "inputs": {"cpu": file_record(args.cpu), "cuda": file_record(args.cuda)},
        "source_state": cpu["source_state"],
        "runtime": cpu["runtime"],
        "physical_core_budget": cpu_validation["physical_cores"],
        "workload": {
            "resolution": cpu["resolution"],
            "repeat": cpu["repeat"],
            "contract": cpu["workload_contract"],
            "global_totals": cpu["global_work_totals"],
            "shape": cpu["metrics"][0]["shape"],
            "total_host_thread_budget": cpu["total_host_thread_budget"],
            "cpu_mpi_ranks": cpu["mpi_ranks"],
            "cpu_omp_threads_per_rank": cpu["omp_threads_per_rank"],
            "cuda_mpi_ranks": cuda["mpi_ranks"],
            "cuda_omp_threads_per_rank": cuda["omp_threads_per_rank"],
        },
        "timing": {"cpu": cpu_timing, "cuda": cuda_timing},
        "speedup_cpu_over_cuda": (
            cpu_timing["median_seconds"] / cuda_timing["median_seconds"]
        ),
        "scalar_metric_differences_by_repeat": metric_differences,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
