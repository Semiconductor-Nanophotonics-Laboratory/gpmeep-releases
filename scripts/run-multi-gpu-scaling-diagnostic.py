#!/usr/bin/env python3
"""Receipt-bound development diagnostic for C++ 1/2-GPU scaling.

This is deliberately not a release benchmark or qualification gate.  Its
fixed matrix screens phase-batched and per-operation curl/update/source kernels
across workload size, then runs the remaining launch-coalescing controls at a
causal anchor size.  Repeats reverse size, condition, and rank order so a
crossover recommendation cannot be manufactured by monotonic thermal drift.
The separately defined multi-GPU release criteria remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import stat
import statistics
import sys
import tempfile
import traceback
import types
from datetime import datetime, timezone
from typing import Any


SCRIPT = pathlib.Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
REPO = SCRIPT_DIR.parent


def load_exact_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    """Compile and execute exactly the observed source bytes, never a .pyc."""

    resolved = path.resolve()
    payload = resolved.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    module.__gpmeep_loaded_source_sha256__ = hashlib.sha256(payload).hexdigest()
    sys.modules[name] = module
    exec(compile(payload, str(resolved), "exec"), module.__dict__)
    return module


PROVENANCE = load_exact_source_module(
    "gpmeep_provenance", SCRIPT_DIR / "gpmeep_provenance.py"
)
EVIDENCE = load_exact_source_module(
    "gpmeep_benchmark_evidence", SCRIPT_DIR / "gpmeep_benchmark_evidence.py"
)
QUALIFICATION = load_exact_source_module(
    "gpmeep_qualification_contract",
    SCRIPT_DIR / "gpmeep_qualification_contract.py",
)
MGB = load_exact_source_module(
    "multi_gpu_benchmark", SCRIPT_DIR / "multi_gpu_benchmark.py"
)


def _loaded_exact_qualification_contract() -> types.ModuleType:
    return QUALIFICATION


# gpmeep_provenance normally re-loads the adjacent qualification source on
# every verification.  This authoritative runner has already compiled and
# hashed that exact archived source, so receipt checks must reuse that proven
# module rather than re-open a mutable pathname later in the run.
PROVENANCE._load_qualification_contract_from_source = (
    _loaded_exact_qualification_contract
)
atomic_write_json = EVIDENCE.atomic_write_json
atomic_write_text = EVIDENCE.atomic_write_text
ProvenanceError = PROVENANCE.ProvenanceError
file_record = PROVENANCE.file_record
sha256_file = PROVENANCE.sha256_file
verify_build_receipt = PROVENANCE.verify_build_receipt


# M16 made the per-operation kernels the production default after large-grid
# evidence contradicted the earlier launch-count assumption.  M19 measures the
# complete 1/2-GPU crossover instead of choosing one global policy from a few
# large points.
PROFILE_ID = "m19-cpp-phase-batch-crossover-v4"
CONDITION_ORDER = (
    "default",
    "disable-phase-batched-curl",
    "disable-phase-batched-update-eh",
    "disable-both-phase-batches",
    "disable-phase-batched-source",
    "disable-tile-coalescing",
    "disable-all-three",
    "no-tile-reference",
    "disable-boundary-descriptor-fast-replay",
)
PHASE_SCREEN_CONDITION_ORDER = (
    "default",
    "disable-phase-batched-curl",
    "disable-phase-batched-update-eh",
    "disable-both-phase-batches",
    "disable-phase-batched-source",
)
PHASE_EFFECT_SPECS = (
    {
        "effect": "curl-with-update-batched",
        "batched_condition": "default",
        "unbatched_condition": "disable-phase-batched-curl",
    },
    {
        "effect": "curl-with-update-unbatched",
        "batched_condition": "disable-phase-batched-update-eh",
        "unbatched_condition": "disable-both-phase-batches",
    },
    {
        "effect": "update-eh-with-curl-batched",
        "batched_condition": "default",
        "unbatched_condition": "disable-phase-batched-update-eh",
    },
    {
        "effect": "update-eh-with-curl-unbatched",
        "batched_condition": "disable-phase-batched-curl",
        "unbatched_condition": "disable-both-phase-batches",
    },
    {
        "effect": "curl-and-update-eh-combined",
        "batched_condition": "default",
        "unbatched_condition": "disable-both-phase-batches",
    },
    {
        "effect": "source-batching",
        "batched_condition": "default",
        "unbatched_condition": "disable-phase-batched-source",
    },
)
CONDITIONS = {
    "default": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
    "disable-phase-batched-curl": {
        "disable_phase_batched_curl": True,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
    "disable-phase-batched-update-eh": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": True,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
    "disable-both-phase-batches": {
        "disable_phase_batched_curl": True,
        "disable_phase_batched_update_eh": True,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
    "disable-phase-batched-source": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": True,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
    "disable-tile-coalescing": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": True,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "not-coalesced",
    },
    "disable-all-three": {
        "disable_phase_batched_curl": True,
        "disable_phase_batched_update_eh": True,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": True,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 128,
        "tile_expectation": "not-coalesced",
    },
    "no-tile-reference": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": False,
        "loop_tile_base_db": 0,
        "tile_expectation": "not-coalesced",
    },
    "disable-boundary-descriptor-fast-replay": {
        "disable_phase_batched_curl": False,
        "disable_phase_batched_update_eh": False,
        "disable_phase_batched_source": False,
        "disable_tile_coalescing": False,
        "disable_boundary_descriptor_fast_replay": True,
        "loop_tile_base_db": 128,
        "tile_expectation": "coalesced",
    },
}
PROFILE = {
    "profile_id": PROFILE_ID,
    "evidence_class": "development-diagnostic-only",
    "release_qualification": False,
    "ranks": [1, 2],
    "size_screen_pixels": [64, 96, 128, 160, 192, 224, 256],
    "phase_screen_conditions": list(PHASE_SCREEN_CONDITION_ORDER),
    "causal_pixels": 192,
    "repeats": 3,
    "warmup_steps": 12,
    "measured_steps": 80,
    "transport": "pinned",
    "completion_policy": "waitsome",
    "initial_condition": "affine-v1",
    "source_enabled": True,
    "source_profile": "dual-electric-v1",
    "overlap_material": False,
    "bfast": False,
    "timeout_seconds": 900,
    "observable_tolerance": 0.002,
    "phase_effect_margin": 0.03,
    "minimum_policy_support_points": 2,
    "conditions": CONDITIONS,
}
# Updated only when an intentional profile change is reviewed with its tests.
PROFILE_SHA256 = "7084108386244419eacf30b3e773500b7b1d1134bb869a8b4975b51f6a39206d"
DIAGNOSTIC_SCHEMA_VERSION = 6
CAUSAL_CONTRACT_VERSION = 2

# The outer controller invokes this immutable in-memory bootstrap with ``-c``.
# It opens the archived runner without following symlinks, verifies the exact
# bytes, and compile-executes only that captured payload.  The runner can then
# attest the bytes that were actually supplied to Python's compiler rather
# than merely hashing its path after the interpreter has already loaded it.
INNER_BOOTSTRAP = r"""
import hashlib, os, stat, sys
path = sys.argv[1]
expected = sys.argv[2]
bootstrap_sha256 = sys.argv[3]
forwarded = sys.argv[4:]
flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("archived runner is not an exclusive regular file")
    blocks = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        blocks.append(block)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
    before.st_ctime_ns, before.st_mode) != (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    after.st_ctime_ns, after.st_mode):
    raise RuntimeError("archived runner changed while being captured")
payload = b"".join(blocks)
observed = hashlib.sha256(payload).hexdigest()
if observed != expected:
    raise RuntimeError("archived runner hash differs from invocation")
sys.argv = [path, *forwarded]
scope = {
    "__name__": "__main__",
    "__file__": path,
    "__package__": None,
    "__cached__": None,
    "__gpmeep_runner_loaded_source_sha256__": observed,
    "__gpmeep_bootstrap_sha256__": bootstrap_sha256,
}
exec(compile(payload, path, "exec"), scope)
"""
INNER_BOOTSTRAP_SHA256 = hashlib.sha256(INNER_BOOTSTRAP.encode()).hexdigest()

TILE_PREFIX = "gpmeep-tile-coalescing-v1:"
PHASE_BATCH_POLICY_PREFIX = "gpmeep-phase-batch-policy-v1:"
CURL_PHASE_REPLAY_PREFIX = "gpmeep-curl-phase-replay-v1:"
GRAPH_PREFIX = "gpmeep-boundary-phase-graph-v1:"
EAGER_PREFIX = "gpmeep-eager-mpi-v1:"
PINGPONG_PREFIX = "gpmeep-receive-pingpong-v1:"
BOUNDARY_EH_PREFIX = "gpmeep-boundary-eh-overlap-v1:"
HALO_CURL_PREFIX = "gpmeep-halo-curl-overlap-v1:"
DFT_PREFIX = "gpmeep-dft-phase-sharing-v1:"
DFT_MULTI_PREFIX = "gpmeep-dft-multi-monitor-batch-v1:"
DFT_WARMUP_PLAN_PREFIX = "gpmeep-dft-warmup-plan-v1:"
BOUNDARY_DESCRIPTOR_PREFIX = "gpmeep-boundary-descriptor-replay-v1:"
INITIAL_CONDITION_PREFIX = "gpmeep-initial-condition-v1:"
SOURCE_PROFILE_PREFIX = "gpmeep-source-profile-v1:"
INITIALIZATION_TIMING_PREFIX = "gpmeep-initialize-field-timing-v1:"

FEATURE_SCHEMAS = {
    "tile_coalescing": (
        TILE_PREFIX,
        {
            "curl_chunk_phases",
            "curl_input_tiles",
            "update_eh_chunk_phases",
            "update_eh_input_tiles",
        },
    ),
    "phase_batch_policy": (
        PHASE_BATCH_POLICY_PREFIX,
        {
            "curl_automatic_checks",
            "curl_automatic_selected",
            "curl_automatic_rejected",
            "curl_forced_batches",
            "curl_batched_operations",
            "curl_unbatched_operations",
            "update_eh_automatic_checks",
            "update_eh_automatic_selected",
            "update_eh_automatic_rejected",
            "update_eh_forced_batches",
            "update_eh_batched_operations",
            "update_eh_unbatched_operations",
        },
    ),
    "curl_phase_replay": (
        CURL_PHASE_REPLAY_PREFIX,
        {
            "disabled_ranks",
            "checks",
            "hits",
            "unready",
            "generation_misses",
            "mirror_misses",
        },
    ),
    "boundary_phase_graph": (
        GRAPH_PREFIX,
        {"creations", "launches"},
    ),
    "eager_mpi": (
        EAGER_PREFIX,
        {
            "receive_start_calls",
            "send_start_calls",
            "receive_requests",
            "send_requests",
        },
    ),
    "receive_pingpong": (
        PINGPONG_PREFIX,
        {
            "warmup_secondary_allocations",
            "selections",
            "secondary_selections",
        },
    ),
    "boundary_eh_overlap": (
        BOUNDARY_EH_PREFIX,
        {
            "checks",
            "warmup_skipped_cold_topology",
            "warmup_launched_h",
            "warmup_launched_e",
            "eligible_ranks",
            "eligible",
            "launched_h",
            "launched_e",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected",
        },
    ),
    "halo_curl_overlap": (
        HALO_CURL_PREFIX,
        {
            "checks",
            "warmup_skipped_cold_topology",
            "warmup_launches",
            "eligible",
            "launches",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected_feature",
            "rejected_small",
            "full_points",
            "interior_points",
            "shell_points",
        },
    ),
    "dft_phase_sharing": (
        DFT_PREFIX,
        {
            "batch_calls",
            "submitted_updates",
            "phase_preparation_launches",
            "phase_reuses",
            "update_kernel_launches",
            "maximum_batch_size",
        },
    ),
    "dft_multi_monitor_batch": (
        DFT_MULTI_PREFIX,
        {
            "automatic_checks",
            "automatic_selected",
            "automatic_rejected",
            "forced_batches",
            "batched_updates",
            "unbatched_updates",
            "plan_uploads",
            "plan_reuses",
            "metadata_host_to_device_bytes",
        },
    ),
    "dft_warmup_plan": (
        DFT_WARMUP_PLAN_PREFIX,
        {
            "metadata_host_to_device_bytes",
            "plan_reuses",
            "plan_uploads",
        },
    ),
    "boundary_descriptor_replay": (
        BOUNDARY_DESCRIPTOR_PREFIX,
        {
            "disabled",
            "disabled_ranks",
            "gather_fast_replays",
            "gather_full_validations",
            "scatter_fast_replays",
            "scatter_full_validations",
        },
    ),
}
KNOWN_GPMEEP_PREFIXES = (
    MGB.RECORD_PREFIX,
    MGB.DEVICE_RECORD_PREFIX,
    *(prefix for prefix, _keys in FEATURE_SCHEMAS.values()),
    SOURCE_PROFILE_PREFIX,
    INITIAL_CONDITION_PREFIX,
    INITIALIZATION_TIMING_PREFIX,
)

TIMING_LABELS = (
    "time stepping",
    "connecting chunks",
    "copying boundaries",
    "all-all communication",
    "1-1 communication",
    "outputting fields",
    "Fourier transforming",
    "MPB mode solver",
    "far-field transform",
    "updating B field",
    "updating H field",
    "updating D field",
    "updating E field",
    "boundary stepping B",
    "boundary stepping WH",
    "boundary stepping PH",
    "boundary stepping H",
    "boundary stepping D",
    "boundary stepping WE",
    "boundary stepping PE",
    "boundary stepping E",
    "everything else",
)
TIMING_LABELS_BY_RANK = {
    1: (
        "time stepping",
        "copying boundaries",
        "Fourier transforming",
        "everything else",
        "updating B field",
        "updating H field",
        "updating D field",
        "updating E field",
        "boundary stepping B",
        "boundary stepping WH",
        "boundary stepping PH",
        "boundary stepping H",
        "boundary stepping D",
        "boundary stepping WE",
        "boundary stepping PE",
        "boundary stepping E",
    ),
    2: (
        "time stepping",
        "copying boundaries",
        "1-1 communication",
        "Fourier transforming",
        "everything else",
        "updating B field",
        "updating H field",
        "updating D field",
        "updating E field",
        "boundary stepping B",
        "boundary stepping WH",
        "boundary stepping PH",
        "boundary stepping H",
        "boundary stepping D",
        "boundary stepping WE",
        "boundary stepping PE",
        "boundary stepping E",
    ),
}
FLOAT_PATTERN = r"(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?|0?\.[0-9]+(?:[eE][+-]?[0-9]+)?"
TIMING_ROW = re.compile(
    rf"^\s+(?P<label>[^:]+):\s+(?P<mean>{FLOAT_PATTERN}) s"
    rf"(?: \+/- (?P<stddev>{FLOAT_PATTERN}) s)?$"
)

SOURCE_DEPENDENCIES = {
    "runner": SCRIPT,
    "multi_gpu_benchmark": pathlib.Path(MGB.__file__).resolve(),
    "benchmark_evidence": SCRIPT_DIR / "gpmeep_benchmark_evidence.py",
    "provenance": SCRIPT_DIR / "gpmeep_provenance.py",
    "qualification_contract": SCRIPT_DIR / "gpmeep_qualification_contract.py",
}
ARCHIVED_SOURCE_NAMES = {
    "runner": "run-multi-gpu-scaling-diagnostic.py",
    "multi_gpu_benchmark": "multi_gpu_benchmark.py",
    "benchmark_evidence": "gpmeep_benchmark_evidence.py",
    "provenance": "gpmeep_provenance.py",
    "qualification_contract": "gpmeep_qualification_contract.py",
}


class DiagnosticError(RuntimeError):
    """A diagnostic artifact cannot be treated as authoritative evidence."""


class SourceWindowError(DiagnosticError):
    """A source or receipt changed while its observed bytes were bound."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_strict_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=MGB._unique_object,
            parse_constant=MGB._reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, MGB.EvidenceError) as error:
        raise DiagnosticError(f"invalid JSON artifact {path}: {error}") from error
    if type(value) is not dict:
        raise DiagnosticError(f"JSON artifact is not an object: {path}")
    return value


def runtime_source_identity() -> dict[str, Any]:
    modules = {
        "multi_gpu_benchmark": MGB,
        "benchmark_evidence": EVIDENCE,
        "provenance": PROVENANCE,
        "qualification_contract": QUALIFICATION,
    }
    runner_loaded_sha256 = globals().get(
        "__gpmeep_runner_loaded_source_sha256__"
    )
    bootstrap_sha256 = globals().get("__gpmeep_bootstrap_sha256__")
    identity = {
        "runner": {
            "path": str(SCRIPT),
            "loaded_source_sha256": (
                runner_loaded_sha256
                if isinstance(runner_loaded_sha256, str)
                else sha256_file(SCRIPT)
            ),
            "execution": (
                "bootstrap-compile-exec-exact-source-bytes"
                if isinstance(runner_loaded_sha256, str)
                else "direct-python-source"
            ),
            "bootstrap_sha256": (
                bootstrap_sha256
                if isinstance(bootstrap_sha256, str)
                else None
            ),
        }
    }
    for name, module in modules.items():
        identity[name] = {
            "path": str(pathlib.Path(module.__file__).resolve()),
            "loaded_source_sha256": getattr(
                module, "__gpmeep_loaded_source_sha256__", None
            ),
            "execution": "compile-exec-exact-source-bytes",
        }
        if name == "provenance":
            identity[name]["qualification_contract_binding_sha256"] = getattr(
                QUALIFICATION, "__gpmeep_loaded_source_sha256__", None
            )
    return identity


def assert_fixed_profile() -> None:
    if tuple(CONDITIONS) != CONDITION_ORDER:
        raise DiagnosticError("causal condition order changed")
    if (
        tuple(PROFILE["phase_screen_conditions"])
        != PHASE_SCREEN_CONDITION_ORDER
        or not set(PHASE_SCREEN_CONDITION_ORDER).issubset(CONDITIONS)
    ):
        raise DiagnosticError("phase crossover condition order changed")
    effect_names = [item["effect"] for item in PHASE_EFFECT_SPECS]
    if len(effect_names) != len(set(effect_names)):
        raise DiagnosticError("phase crossover effect names are not unique")
    for item in PHASE_EFFECT_SPECS:
        if (
            set(item) != {"effect", "batched_condition", "unbatched_condition"}
            or item["batched_condition"] not in PHASE_SCREEN_CONDITION_ORDER
            or item["unbatched_condition"] not in PHASE_SCREEN_CONDITION_ORDER
            or item["batched_condition"] == item["unbatched_condition"]
        ):
            raise DiagnosticError("phase crossover effect specification changed")
    if canonical_sha256(semantic_profile_contract()) != PROFILE_SHA256:
        raise DiagnosticError("fixed diagnostic profile changed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-receipt", type=pathlib.Path)
    parser.add_argument("--expected-receipt-id")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--mpiexec", type=pathlib.Path)
    parser.add_argument("--internal-authoritative", action="store_true")
    parser.add_argument("--invocation", type=pathlib.Path)
    parser.add_argument("--invocation-sha256")
    parser.add_argument("--capability")
    return parser.parse_args(argv)


def stable_file_record(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"evidence file is absent: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_mode != after.st_mode
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise SourceWindowError(f"evidence file changed while hashing: {path}")
    try:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "size_bytes": after.st_size, "sha256": digest}


def stable_open_identity(
    path: pathlib.Path, *, process_executable: bool = False
) -> dict[str, Any]:
    """Hash one stable open file description, including inode and mode.

    ``/proc/self/exe`` is intentionally followed only for the kernel-owned
    process-executable link; all ordinary inputs reject symlinks and use
    ``O_NOFOLLOW`` where available.
    """

    if process_executable:
        if path != pathlib.Path("/proc/self/exe"):
            raise DiagnosticError("process executable identity requires /proc/self/exe")
    elif path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"identity input is not a regular file: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not process_executable and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DiagnosticError(f"cannot open identity input {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DiagnosticError(f"identity input is not regular: {path}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_mode",
    )
    if any(getattr(before, key) != getattr(after, key) for key in identity_fields):
        raise SourceWindowError(f"identity input changed while hashing: {path}")
    resolved = pathlib.Path("/proc/self/exe").resolve() if process_executable else path.resolve()
    return {
        "path": str(resolved),
        "device": after.st_dev,
        "inode": after.st_ino,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "mode": after.st_mode & 0o7777,
        "sha256": digest.hexdigest(),
    }


def verify_control_python_identity(expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise DiagnosticError("control Python binding is malformed")
    actual = stable_open_identity(
        pathlib.Path("/proc/self/exe"), process_executable=True
    )
    if actual != expected:
        raise DiagnosticError("actual process executable differs from control Python")
    return actual


def atomic_write_bytes(path: pathlib.Path, payload: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def stable_binary_snapshot(
    source: pathlib.Path,
    destination: pathlib.Path,
    root: pathlib.Path,
    *,
    executable_required: bool,
) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise SourceWindowError(f"binary snapshot source is invalid: {source}")
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    mode = before.st_mode & 0o7777
    stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(payload)
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and before.st_mode == after.st_mode
    )
    if not stable:
        raise SourceWindowError(
            f"binary source changed while bytes were observed: {source}"
        )
    if executable_required and not mode & 0o111:
        raise SourceWindowError(f"binary source is not executable: {source}")
    atomic_write_bytes(destination, payload, mode)
    destination_stat = destination.stat()
    if destination_stat.st_mode & 0o7777 != mode:
        raise SourceWindowError(f"binary snapshot mode differs: {destination}")
    return {
        "source_path": str(source.resolve()),
        "source_file": stable_file_record(source, root),
        "source_mode": mode,
        "snapshot_file": stable_file_record(destination, root),
        "snapshot_mode": destination_stat.st_mode & 0o7777,
        "executable_required": executable_required,
    }


def verify_binary_snapshot(
    binding: dict[str, Any], snapshot: pathlib.Path, root: pathlib.Path
) -> None:
    record = stable_file_record(snapshot, root)
    mode = snapshot.stat().st_mode & 0o7777
    if record != binding["snapshot_file"] or mode != binding["snapshot_mode"]:
        raise SourceWindowError(f"binary snapshot changed: {snapshot}")


def create_stable_execution_handle(
    snapshot: pathlib.Path, root: pathlib.Path, label: str
) -> tuple[int, dict[str, Any]]:
    """Hold one O_NOFOLLOW inode open for every subsequent launch."""

    if snapshot.is_symlink() or not snapshot.is_file():
        raise DiagnosticError(f"execution snapshot is invalid: {snapshot}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(snapshot, flags)
    try:
        identity = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < identity.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, identity.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        if (
            identity.st_dev != after.st_dev
            or identity.st_ino != after.st_ino
            or identity.st_size != after.st_size
            or identity.st_mtime_ns != after.st_mtime_ns
            or identity.st_ctime_ns != after.st_ctime_ns
            or identity.st_mode != after.st_mode
            or offset != after.st_size
        ):
            raise SourceWindowError(
                f"execution handle changed while hashing: {snapshot}"
            )
        execution_path = pathlib.Path(f"/proc/{os.getpid()}/fd/{descriptor}")
        source_record = stable_file_record(snapshot, root)
        if (
            source_record["size_bytes"] != identity.st_size
            or source_record["sha256"] != digest.hexdigest()
            or snapshot.stat().st_mode & 0o7777 != identity.st_mode & 0o7777
        ):
            raise SourceWindowError(
                f"open execution inode differs from snapshot binding: {snapshot}"
            )
        binding = {
            "label": label,
            "source_snapshot": source_record,
            "source_mode": identity.st_mode & 0o7777,
            "execution_model": "persistent-o-rdonly-o-nofollow-proc-fd",
            "execution_path": str(execution_path),
            "device": identity.st_dev,
            "inode": identity.st_ino,
            "link_count": identity.st_nlink,
            "size_bytes": identity.st_size,
            "mode": identity.st_mode & 0o7777,
            "mtime_ns": identity.st_mtime_ns,
            "ctime_ns": identity.st_ctime_ns,
            "sha256": digest.hexdigest(),
        }
        verify_stable_execution_handle(binding, descriptor)
        return descriptor, binding
    except BaseException:
        os.close(descriptor)
        raise


def verify_stable_execution_handle(binding: dict[str, Any], descriptor: int) -> None:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not block:
            break
        digest.update(block)
        offset += len(block)
    metadata = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_size",
        "st_mode",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(metadata, key) for key in stable_fields):
        raise SourceWindowError(
            f"stable execution handle changed while hashing: {binding.get('label')}"
        )
    actual = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "link_count": metadata.st_nlink,
        "size_bytes": metadata.st_size,
        "mode": metadata.st_mode & 0o7777,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": digest.hexdigest(),
        "execution_path": f"/proc/{os.getpid()}/fd/{descriptor}",
    }
    for key, value in actual.items():
        if binding.get(key) != value:
            raise SourceWindowError(
                f"stable execution handle changed: {binding.get('label')}"
            )


def verify_stable_execution_handles(
    bindings: dict[str, dict[str, Any]], descriptors: dict[str, int]
) -> None:
    if set(bindings) != set(descriptors):
        raise DiagnosticError("stable execution handle set is incomplete")
    for name in sorted(bindings):
        verify_stable_execution_handle(bindings[name], descriptors[name])


def validate_loader_trace(
    loader_output: str,
    library_execution: pathlib.Path,
    original_library: pathlib.Path,
    soname: str,
) -> dict[str, Any]:
    soname_lines = [line for line in loader_output.splitlines() if soname in line]
    preload_lines = [
        line
        for line in loader_output.splitlines()
        if str(library_execution) in line
    ]
    if (
        len(preload_lines) != 1
        or any(str(library_execution) not in line for line in soname_lines)
        or str(original_library) in loader_output
        or any(
            str(original_library.parent) in line
            for line in soname_lines
        )
    ):
        raise DiagnosticError(
            "worker loader did not resolve libmeep to the fixed open image"
        )
    return {
        "resolved_preload_line": preload_lines[0],
        "additional_soname_lines": soname_lines,
    }


def exclusive_publish_json(path: pathlib.Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"authority marker already exists: {path}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise DiagnosticError(f"authority marker raced: {path}") from error
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_terminal_marker(
    output: pathlib.Path, state: str, marker: dict[str, Any]
) -> None:
    if state not in {"COMPLETE", "FAILED"} or marker.get("state") != state:
        raise DiagnosticError("terminal publication state is invalid")
    opposite = output / ("FAILED.json" if state == "COMPLETE" else "COMPLETE.json")
    if opposite.exists() or opposite.is_symlink():
        raise DiagnosticError(f"opposite terminal marker already exists: {opposite}")
    terminal = output / "TERMINAL.json"
    exclusive_publish_json(
        terminal,
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "state": state,
            "run_id": marker.get("run_id"),
            "expected_receipt_id": marker.get("expected_receipt_id"),
            "verified_receipt_id": marker.get("verified_receipt_id"),
        },
    )
    enriched = dict(marker)
    enriched["terminal_claim"] = stable_file_record(terminal, output)
    exclusive_publish_json(output / f"{state}.json", enriched)
    if opposite.exists() or opposite.is_symlink():
        raise DiagnosticError("opposite terminal marker appeared during publication")


def claim_output_directory(path: pathlib.Path) -> None:
    if not path.is_absolute():
        raise DiagnosticError("output path must be absolute after normalization")
    parent = path.parent
    if (
        not parent.exists()
        or parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve() != parent
    ):
        raise DiagnosticError(
            "output parent must be an existing canonical non-symlink directory"
        )
    ancestor = parent
    while True:
        if ancestor.is_symlink():
            raise DiagnosticError(f"output ancestor is a symlink: {ancestor}")
        if ancestor.exists():
            if not ancestor.is_dir():
                raise DiagnosticError(f"output ancestor is not a directory: {ancestor}")
            break
        if ancestor == ancestor.parent:
            raise DiagnosticError("output has no existing directory ancestor")
        ancestor = ancestor.parent
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as error:
        raise DiagnosticError("output directory must be absent") from error
    if path.is_symlink() or not path.is_dir():
        raise DiagnosticError("exclusive output claim did not create a real directory")
    if path.stat().st_mode & 0o777 != 0o700:
        raise DiagnosticError("exclusive output claim does not have mode 0700")


def stable_snapshot(
    source: pathlib.Path, destination: pathlib.Path, root: pathlib.Path
) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise SourceWindowError(f"snapshot source is absent: {source}")
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    atomic_write_text(destination, payload.decode("utf-8"))
    observed = stable_file_record(destination, root)
    stable = (
        before.st_size == after.st_size == len(payload)
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and before.st_mode == after.st_mode
        and before.st_ino == after.st_ino
        and before.st_dev == after.st_dev
    )
    binding = {
        "source_path": str(source.resolve()),
        "source_identity": {
            "device": before.st_dev,
            "inode": before.st_ino,
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
            "mode": before.st_mode,
        },
        "observed_snapshot": observed,
        "stable_read": stable,
    }
    if not stable:
        raise SourceWindowError(
            f"snapshot source changed while bytes were observed: {source}"
        )
    return binding


def verify_recorded_file(
    record: dict[str, Any], root: pathlib.Path, label: str
) -> pathlib.Path:
    try:
        return MGB.verify_file_record(record, root, label)
    except MGB.EvidenceError as error:
        raise DiagnosticError(str(error)) from error


def verify_snapshot_window(
    binding: dict[str, Any], snapshot: pathlib.Path, root: pathlib.Path
) -> None:
    source = pathlib.Path(binding["source_path"])
    identity = binding["source_identity"]
    snapshot_record = stable_file_record(snapshot, root)
    if snapshot_record != binding["observed_snapshot"]:
        raise SourceWindowError(f"archived snapshot changed: {snapshot}")
    if not source.is_file():
        raise SourceWindowError(f"snapshot source disappeared: {source}")
    before = source.stat()
    digest = sha256_file(source)
    after = source.stat()
    current_identity = {
        "device": after.st_dev,
        "inode": after.st_ino,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "mode": after.st_mode,
    }
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_mode != after.st_mode
        or current_identity != identity
        or digest != snapshot_record["sha256"]
    ):
        raise SourceWindowError(f"snapshot source changed during run: {source}")


def expected_schedule() -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    position = 0
    sizes = [
        pixels
        for pixels in PROFILE["size_screen_pixels"]
        if pixels != PROFILE["causal_pixels"]
    ]
    for repeat in range(1, PROFILE["repeats"] + 1):
        size_order = sizes if repeat % 2 else list(reversed(sizes))
        rank_order = (
            PROFILE["ranks"]
            if repeat % 2
            else list(reversed(PROFILE["ranks"]))
        )
        condition_order = (
            list(PHASE_SCREEN_CONDITION_ORDER)
            if repeat % 2
            else list(reversed(PHASE_SCREEN_CONDITION_ORDER))
        )
        for pixels in size_order:
            for condition in condition_order:
                for ranks in rank_order:
                    position += 1
                    schedule.append(
                        {
                            "position": position,
                            "matrix": "size-screen",
                            "pixels": pixels,
                            "condition": condition,
                            "repeat": repeat,
                            "ranks": ranks,
                        }
                    )
    boundary_control = "disable-boundary-descriptor-fast-replay"
    other_controls = [
        condition
        for condition in CONDITION_ORDER
        if condition not in {"default", boundary_control}
    ]
    for repeat in range(1, PROFILE["repeats"] + 1):
        condition_order = (
            ["default", boundary_control, *other_controls]
            if repeat == 1
            else [boundary_control, "default", *reversed(other_controls)]
        )
        rank_order = (
            PROFILE["ranks"]
            if repeat % 2
            else list(reversed(PROFILE["ranks"]))
        )
        for condition in condition_order:
            for ranks in rank_order:
                position += 1
                schedule.append(
                    {
                        "position": position,
                        "matrix": "causal-controls",
                        "pixels": PROFILE["causal_pixels"],
                        "condition": condition,
                        "repeat": repeat,
                        "ranks": ranks,
                    }
                )
    return schedule


def semantic_profile_contract() -> dict[str, Any]:
    """Bind every parser/schedule semantic that changes evidence meaning."""

    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "worker_record_schema_version": MGB.SCHEMA_VERSION,
        "authoritative_source_execution": {
            "runner": "bootstrap-compile-exec-exact-source-bytes",
            "dependencies": "compile-exec-exact-source-bytes",
            "qualification_for_receipt_verification": (
                "reuse-exact-loaded-module-no-path-reread"
            ),
            "bootstrap_sha256": INNER_BOOTSTRAP_SHA256,
        },
        "authoritative_binary_execution": {
            "worker": "persistent-open-snapshot-fd",
            "mpiexec": "persistent-open-snapshot-fd",
            "libmeep": "persistent-open-fd-ld-preload-with-loader-trace",
            "mca": "persistent-open-snapshot-fd",
            "topology": "same-node-controller-proc-fd",
            "per_launch_identity_gate": (
                "dev-inode-nlink-size-mode-mtime-ctime-sha-before-and-after"
            ),
        },
        "profile": PROFILE,
        "condition_order": list(CONDITION_ORDER),
        "phase_screen_condition_order": list(PHASE_SCREEN_CONDITION_ORDER),
        "phase_effect_specs": [dict(item) for item in PHASE_EFFECT_SPECS],
        "device_identity_contract": {
            "fields": [
                "compatible",
                "compute_major",
                "compute_minor",
                "name",
                "ordinal",
                "rank",
                "uuid",
            ],
            "stable_across_all_samples": True,
            "one_gpu_mapping_is_two_gpu_prefix": True,
            "policy_reuse": "exact-rank-device-identity-only",
        },
        "schedule": expected_schedule(),
        "feature_schemas": {
            name: {
                "prefix": prefix,
                "keys": sorted(keys),
                "value_types": {
                    key: (
                        "boolean"
                        if name == "boundary_descriptor_replay"
                        and key == "disabled"
                        else "nonnegative-integer"
                    )
                    for key in sorted(keys)
                },
            }
            for name, (prefix, keys) in sorted(FEATURE_SCHEMAS.items())
        },
        "known_gpmeep_prefixes": list(KNOWN_GPMEEP_PREFIXES),
        "gpmeep_marker_streams": ["stdout", "stderr"],
        "initial_condition_marker": {
            "prefix": INITIAL_CONDITION_PREFIX,
            "keys": ["profile"],
            "profile": PROFILE["initial_condition"],
            "record_match": "exact",
        },
        "source_profile_marker": {
            "prefix": SOURCE_PROFILE_PREFIX,
            "keys": ["profile"],
            "profile": PROFILE["source_profile"],
            "record_match": "exact",
        },
        "initialization_timing_marker": {
            "prefix": INITIALIZATION_TIMING_PREFIX,
            "keys": ["applications_per_rank", "max_seconds"],
            "applications_per_rank": 1,
            "max_seconds": "finite-positive-rank-maximum",
            "timing_domain": "untimed-initialize-field-setup",
        },
        "timing_labels_by_rank": {
            str(ranks): list(labels)
            for ranks, labels in sorted(TIMING_LABELS_BY_RANK.items())
        },
        "timing_contract": {
            "blocks": 1,
            "label_order": "exact-by-rank",
            "mean_values": "finite-positive",
            "standard_deviation_values": "finite-nonnegative-when-present",
            "configuration_sample_count": "equals-repeats",
            "initializer_setup_timing": "separate-from-fdtd-seconds",
        },
        "causal_contract_version": CAUSAL_CONTRACT_VERSION,
        "causal_contract": {
            "phase_batch_disable_call_counts": (
                "strict-curl-update-eh-and-exact-two-to-one-source-per-repeat-and-rank"
            ),
            "boundary_descriptor_ab": "alternating-default-disabled",
            "boundary_descriptor_default_gate": "scatter-exclusive-fast-replay",
            "boundary_descriptor_gather_gate": "follow-up-phase-graph-disabled-only",
            "remote_rank_no_remote": "zero-required",
            "configuration_sample_count": "equals-repeats",
        },
    }


def condition_environment(
    base: dict[str, str], condition: str, pixels: int, ranks: int
) -> dict[str, str]:
    if condition not in CONDITIONS:
        raise DiagnosticError(f"unknown diagnostic condition: {condition}")
    specification = CONDITIONS[condition]
    environment = dict(base)
    environment.update(
        {
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE": "1",
            "MEEP_GPU_MPI_TRANSPORT": PROFILE["transport"],
            "MEEP_GPU_MPI_COMPLETION": PROFILE["completion_policy"],
            "MEEP_GPU_MULTI_PIXELS": str(pixels),
            "MEEP_GPU_MULTI_WARMUP_STEPS": str(PROFILE["warmup_steps"]),
            "MEEP_GPU_MULTI_STEPS": str(PROFILE["measured_steps"]),
            "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB": str(
                specification["loop_tile_base_db"]
            ),
            "MEEP_GPU_MULTI_BFAST": "0",
            "MEEP_GPU_MULTI_DISABLE_SOURCE": "0",
            "MEEP_GPU_MULTI_SOURCE_PROFILE": PROFILE["source_profile"],
            "MEEP_GPU_MULTI_OVERLAP_MATERIAL": "0",
            "MEEP_GPU_MULTI_INITIAL_CONDITION": PROFILE["initial_condition"],
            "MEEP_GPU_MULTI_PRINT_TIMES": "1",
        }
    )
    if specification["disable_phase_batched_curl"]:
        environment["MEEP_GPU_DISABLE_PHASE_BATCHED_CURL"] = "1"
    if specification["disable_phase_batched_update_eh"]:
        environment["MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH"] = "1"
    if specification["disable_phase_batched_source"]:
        environment["MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE"] = "1"
    if specification["disable_tile_coalescing"]:
        environment["MEEP_GPU_DISABLE_TILE_COALESCING"] = "1"
    if specification["disable_boundary_descriptor_fast_replay"]:
        environment["MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY"] = "1"
        if ranks == 2:
            environment["MEEP_GPU_EXPECT_NO_BOUNDARY_DESCRIPTOR_FAST_REPLAY"] = "1"
    elif condition == "default" and ranks == 2:
        environment["MEEP_GPU_EXPECT_BOUNDARY_DESCRIPTOR_FAST_REPLAY"] = "1"
    expectation = specification["tile_expectation"]
    environment[
        "MEEP_GPU_EXPECT_TILE_COALESCING"
        if expectation == "coalesced"
        else "MEEP_GPU_EXPECT_NO_TILE_COALESCING"
    ] = "1"
    return environment


def isolated_command(
    environment: dict[str, str], mpiexec: pathlib.Path,
    executable: pathlib.Path, preload_library: pathlib.Path, ranks: int,
) -> list[str]:
    assignments = [f"{key}={environment[key]}" for key in sorted(environment)]
    return [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{PROFILE['timeout_seconds']}s",
        "/usr/bin/env",
        "-i",
        *assignments,
        str(mpiexec),
        "--host",
        f"localhost:{ranks}",
        "--bind-to",
        "none",
        "-np",
        str(ranks),
        "/usr/bin/env",
        f"LD_PRELOAD={preload_library}",
        str(executable),
    ]


def exactly_one_marker_object(
    stdout: str, prefix: str, label: str, exact_keys: set[str]
) -> dict[str, Any]:
    payloads = [
        line[len(prefix) :]
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        raise DiagnosticError(
            f"expected exactly one {label} marker, found {len(payloads)}"
        )
    try:
        value = json.loads(
            payloads[0],
            object_pairs_hook=MGB._unique_object,
            parse_constant=MGB._reject_json_constant,
        )
    except (MGB.EvidenceError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise DiagnosticError(f"malformed {label} marker: {error}") from error
    if type(value) is not dict or set(value) != exact_keys:
        raise DiagnosticError(f"{label} marker has an invalid exact schema")
    return value


def exactly_one_feature(
    stdout: str, prefix: str, label: str, exact_keys: set[str]
) -> dict[str, int]:
    value = exactly_one_marker_object(stdout, prefix, label, exact_keys)
    for key, item in value.items():
        if label == "boundary_descriptor_replay" and key == "disabled":
            if type(item) is not bool:
                raise DiagnosticError(f"{label}.{key} must be boolean")
            continue
        if type(item) is not int or item < 0:
            raise DiagnosticError(f"{label}.{key} must be a nonnegative integer")
    return value


def parse_initial_condition_marker(stdout: str) -> dict[str, str]:
    value = exactly_one_marker_object(
        stdout, INITIAL_CONDITION_PREFIX, "initial condition", {"profile"}
    )
    profile = value["profile"]
    if type(profile) is not str or profile not in {
        "trigonometric-v1",
        "affine-v1",
    }:
        raise DiagnosticError("initial condition profile is invalid")
    return value


def parse_source_profile_marker(stdout: str) -> dict[str, str]:
    value = exactly_one_marker_object(
        stdout, SOURCE_PROFILE_PREFIX, "source profile", {"profile"}
    )
    profile = value["profile"]
    if type(profile) is not str or profile not in {
        "single-ez-v1",
        "dual-electric-v1",
    }:
        raise DiagnosticError("source profile is invalid")
    return value


def parse_initialization_timing_marker(stdout: str) -> dict[str, int | float]:
    value = exactly_one_marker_object(
        stdout,
        INITIALIZATION_TIMING_PREFIX,
        "initialize-field timing",
        {"applications_per_rank", "max_seconds"},
    )
    applications = value["applications_per_rank"]
    seconds = value["max_seconds"]
    if type(applications) is not int or applications != 1:
        raise DiagnosticError(
            "initialize-field timing must report one application per rank"
        )
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(float(seconds))
        or seconds <= 0
    ):
        raise DiagnosticError(
            "initialize-field timing max_seconds must be finite and positive"
        )
    return {"applications_per_rank": applications, "max_seconds": float(seconds)}


def reject_unknown_gpmeep_markers(stdout: str) -> None:
    for line in stdout.splitlines():
        if line.startswith("gpmeep-") and not any(
            line.startswith(prefix) for prefix in KNOWN_GPMEEP_PREFIXES
        ):
            marker = line.split(":", 1)[0]
            raise DiagnosticError(f"unknown gpmeep marker: {marker}")


def reject_unknown_gpmeep_streams(stdout: str, stderr: str) -> None:
    reject_unknown_gpmeep_markers(stdout)
    reject_unknown_gpmeep_markers(stderr)


def parse_print_times(
    stdout: str, ranks: int
) -> dict[str, dict[str, float | None]]:
    lines = stdout.replace("\r", "\n").splitlines()
    headings = [index for index, line in enumerate(lines) if line == "Field time usage:"]
    if len(headings) != 1:
        raise DiagnosticError(
            f"expected exactly one print_times block, found {len(headings)}"
        )
    records: dict[str, dict[str, float | None]] = {}
    for line in lines[headings[0] + 1 :]:
        if not line.strip():
            if records:
                break
            continue
        match = TIMING_ROW.fullmatch(line)
        if not match:
            raise DiagnosticError(f"malformed print_times row: {line!r}")
        label = match.group("label").strip()
        if label not in TIMING_LABELS or label in records:
            raise DiagnosticError(f"unknown or duplicate print_times label: {label!r}")
        mean = float(match.group("mean"))
        stddev_text = match.group("stddev")
        stddev = float(stddev_text) if stddev_text is not None else None
        if not math.isfinite(mean) or mean <= 0 or (
            stddev is not None and (not math.isfinite(stddev) or stddev < 0)
        ):
            raise DiagnosticError(
                "print_times means must be finite/positive and deviations "
                "finite/nonnegative"
            )
        records[label] = {"mean_seconds": mean, "stddev_seconds": stddev}
    expected_labels = TIMING_LABELS_BY_RANK.get(ranks)
    if expected_labels is None or tuple(records) != expected_labels:
        raise DiagnosticError(
            "print_times labels/order differ from the exact rank profile: "
            f"expected={expected_labels!r}, actual={tuple(records)!r}"
        )
    return records


def validate_tile_record(condition: str, record: dict[str, int]) -> None:
    specification = CONDITIONS[condition]
    pairs = (
        (record["curl_chunk_phases"], record["curl_input_tiles"]),
        (
            record["update_eh_chunk_phases"],
            record["update_eh_input_tiles"],
        ),
    )
    if specification["tile_expectation"] == "coalesced":
        if pairs[0][0] <= 0 or any(
            (phases == 0) != (tiles == 0)
            or (phases > 0 and tiles <= phases)
            for phases, tiles in pairs
        ):
            raise DiagnosticError("tile-coalescing condition did not execute")
    elif any(item != 0 for pair in pairs for item in pair):
        raise DiagnosticError("non-coalesced condition reported coalescing")


def validate_feature_contract(
    item: dict[str, Any],
    benchmark: dict[str, Any],
    features: dict[str, dict[str, int]],
) -> None:
    ranks = item["ranks"]
    steps = PROFILE["measured_steps"]
    replay = features["curl_phase_replay"]
    replay_outcomes = (
        replay["hits"]
        + replay["unready"]
        + replay["generation_misses"]
        + replay["mirror_misses"]
    )
    replay_values = tuple(
        replay[key]
        for key in (
            "checks",
            "hits",
            "unready",
            "generation_misses",
            "mirror_misses",
        )
    )
    if replay["disabled_ranks"] != 0:
        raise DiagnosticError("curl phase replay was unexpectedly opted out")
    if CONDITIONS[item["condition"]]["disable_phase_batched_curl"]:
        if any(replay_values):
            raise DiagnosticError(
                "disabled curl batching reported curl phase replay activity"
            )
    elif (
        replay["checks"] != 2 * steps * ranks
        or replay["hits"] != replay["checks"]
        or replay["checks"] != replay_outcomes
        or replay["unready"] != 0
        or replay["generation_misses"] != 0
        or replay["mirror_misses"] != 0
    ):
        raise DiagnosticError(
            "enabled curl batching did not prove conservative phase replay"
        )
    boundary = features["boundary_eh_overlap"]
    boundary_classified = sum(
        boundary[key]
        for key in (
            "eligible",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected",
        )
    )
    if (
        boundary["checks"] != 2 * steps * ranks
        or boundary_classified != boundary["checks"]
        or boundary["launched_h"] + boundary["launched_e"]
        != boundary["eligible"]
        or boundary["eligible_ranks"] > ranks
    ):
        raise DiagnosticError("boundary E/H marker is not an exact phase partition")
    if ranks == 2 and boundary["skipped_no_remote"] != 0:
        raise DiagnosticError("two-rank boundary marker reported no-remote work")

    halo = features["halo_curl_overlap"]
    halo_classified = sum(
        halo[key]
        for key in (
            "eligible",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected_feature",
            "rejected_small",
        )
    )
    if (
        halo["checks"] != steps * ranks
        or halo_classified != halo["checks"]
        or halo["interior_points"] + halo["shell_points"]
        != halo["full_points"]
    ):
        raise DiagnosticError("halo/curl marker is not an exact phase partition")
    if ranks == 2 and halo["skipped_no_remote"] != 0:
        raise DiagnosticError("two-rank halo marker reported no-remote work")

    dft = features["dft_phase_sharing"]
    dft_multi = features["dft_multi_monitor_batch"]
    if (
        dft["batch_calls"] <= 0
        or dft["submitted_updates"]
        != benchmark["phase_calls"]["dft"]["cuda_calls"]
        or dft["phase_preparation_launches"] + dft["phase_reuses"]
        != dft["submitted_updates"]
        or dft["maximum_batch_size"] <= 0
        or dft_multi["automatic_selected"]
        + dft_multi["automatic_rejected"]
        != dft_multi["automatic_checks"]
        or dft_multi["batched_updates"]
        + dft_multi["unbatched_updates"]
        != dft["submitted_updates"]
        or dft["update_kernel_launches"]
        != dft_multi["automatic_selected"]
        + dft_multi["forced_batches"]
        + dft_multi["unbatched_updates"]
        or dft_multi["plan_uploads"] + dft_multi["plan_reuses"]
        != dft_multi["automatic_selected"]
        + dft_multi["forced_batches"]
        or (dft_multi["plan_uploads"] == 0)
        != (dft_multi["metadata_host_to_device_bytes"] == 0)
    ):
        raise DiagnosticError("DFT marker is internally inconsistent")

    eager = features["eager_mpi"]
    if (
        eager["receive_start_calls"] > eager["receive_requests"]
        or eager["send_start_calls"] > eager["send_requests"]
    ):
        raise DiagnosticError("eager MPI marker has impossible request counts")
    pingpong = features["receive_pingpong"]
    if pingpong["secondary_selections"] > pingpong["selections"]:
        raise DiagnosticError("receive ping-pong marker has impossible selections")

    descriptor = features["boundary_descriptor_replay"]
    disabled = descriptor["disabled"]
    disabled_ranks = descriptor["disabled_ranks"]
    descriptor_values = tuple(
        descriptor[key]
        for key in (
            "gather_fast_replays",
            "gather_full_validations",
            "scatter_fast_replays",
            "scatter_full_validations",
        )
    )
    if ranks == 1:
        expected_disabled = item["condition"] == (
            "disable-boundary-descriptor-fast-replay"
        )
        if (
            disabled is not expected_disabled
            or disabled_ranks != int(expected_disabled)
            or any(descriptor_values)
        ):
            raise DiagnosticError(
                "one-rank boundary descriptor marker has invalid exact semantics"
            )
    elif item["condition"] == "disable-boundary-descriptor-fast-replay":
        if (
            disabled is not True
            or disabled_ranks != ranks
            or descriptor["gather_fast_replays"] != 0
            or descriptor["scatter_fast_replays"] != 0
            or descriptor["scatter_full_validations"] <= 0
        ):
            raise DiagnosticError(
                "disabled boundary descriptor replay did not prove full validation"
            )
    elif item["condition"] == "default":
        if (
            disabled is not False
            or disabled_ranks != 0
            or descriptor["scatter_fast_replays"] <= 0
            or descriptor["scatter_full_validations"] != 0
        ):
            raise DiagnosticError(
                "default two-rank workload did not prove exclusive scatter "
                "descriptor fast replay"
            )
    elif disabled is not False or disabled_ranks != 0:
        raise DiagnosticError(
            "non-optout workload unexpectedly disabled boundary descriptor replay"
        )


def expected_worker_profile(condition: str) -> dict[str, Any]:
    return {
        "loop_tile_base_db": CONDITIONS[condition]["loop_tile_base_db"],
        "bfast": PROFILE["bfast"],
        "source_enabled": PROFILE["source_enabled"],
        "overlap_material": PROFILE["overlap_material"],
        "completion_policy": PROFILE["completion_policy"],
        "initial_condition": PROFILE["initial_condition"],
    }


def parse_sample(stdout: str, item: dict[str, Any]) -> dict[str, Any]:
    reject_unknown_gpmeep_markers(stdout)
    try:
        benchmark = MGB.extract_record(
            stdout,
            expected_ranks=item["ranks"],
            expected_pixels=item["pixels"],
            expected_warmup_steps=PROFILE["warmup_steps"],
            expected_steps=PROFILE["measured_steps"],
            expected_transport=PROFILE["transport"],
            expected_worker_profile=expected_worker_profile(item["condition"]),
        )
        devices = MGB.extract_device_records(stdout, item["ranks"])
    except MGB.EvidenceError as error:
        raise DiagnosticError(str(error)) from error
    features = {
        name: exactly_one_feature(stdout, prefix, name, keys)
        for name, (prefix, keys) in FEATURE_SCHEMAS.items()
    }
    initial_condition = parse_initial_condition_marker(stdout)
    source_profile = parse_source_profile_marker(stdout)
    initialization_timing = parse_initialization_timing_marker(stdout)
    if (
        initial_condition["profile"] != PROFILE["initial_condition"]
        or initial_condition["profile"] != benchmark["initial_condition"]
    ):
        raise DiagnosticError(
            "initial-condition marker, worker record, and fixed profile differ"
        )
    if source_profile["profile"] != PROFILE["source_profile"]:
        raise DiagnosticError(
            "source-profile marker and fixed diagnostic profile differ"
        )
    validate_tile_record(item["condition"], features["tile_coalescing"])
    validate_feature_contract(item, benchmark, features)
    return {
        "benchmark": benchmark,
        "devices": devices,
        "feature_markers": features,
        "initial_condition": initial_condition,
        "source_profile": source_profile,
        "initialization_timing": initialization_timing,
        "print_times": parse_print_times(stdout, item["ranks"]),
    }


def validate_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected = expected_schedule()
    actual = [sample["schedule"] for sample in samples]
    if actual != expected:
        raise DiagnosticError("sample matrix is incomplete or reordered")
    mappings: dict[int, tuple[str, ...]] = {}
    identity_mappings: dict[int, tuple[tuple[tuple[str, Any], ...], ...]] = {}
    for sample in samples:
        ranks = sample["schedule"]["ranks"]
        devices = sample["records"]["devices"]
        mapping = tuple(item["uuid"] for item in devices)
        identity_mapping = tuple(
            tuple(sorted(item.items())) for item in devices
        )
        if ranks in mappings and mappings[ranks] != mapping:
            raise DiagnosticError("physical rank-to-GPU UUID mapping changed")
        if (
            ranks in identity_mappings
            and identity_mappings[ranks] != identity_mapping
        ):
            raise DiagnosticError(
                "physical rank-to-GPU identity mapping changed"
            )
        mappings[ranks] = mapping
        identity_mappings[ranks] = identity_mapping
    if set(mappings) != {1, 2} or mappings[1] != mappings[2][:1]:
        raise DiagnosticError("1/2-GPU mappings are not one canonical UUID prefix")
    if identity_mappings[1] != identity_mappings[2][:1]:
        raise DiagnosticError(
            "1/2-GPU identity mappings are not one canonical prefix"
        )

    for repeat in range(1, PROFILE["repeats"] + 1):
        for ranks in PROFILE["ranks"]:
            records = {
                sample["schedule"]["condition"]: sample["records"]["benchmark"]
                for sample in samples
                if sample["schedule"]["pixels"] == PROFILE["causal_pixels"]
                and sample["schedule"]["repeat"] == repeat
                and sample["schedule"]["ranks"] == ranks
            }
            if set(records) != set(CONDITION_ORDER):
                raise DiagnosticError("causal phase matrix is incomplete")
            calls = {
                condition: {
                    phase: record["phase_calls"][phase]["cuda_calls"]
                    for phase in ("curl", "update_eh", "source")
                }
                for condition, record in records.items()
            }
            default = calls["default"]
            curl_off = calls["disable-phase-batched-curl"]
            update_off = calls["disable-phase-batched-update-eh"]
            both_off = calls["disable-both-phase-batches"]
            source_off = calls["disable-phase-batched-source"]
            if not (
                curl_off["curl"] > default["curl"]
                and curl_off["update_eh"] == default["update_eh"]
                and curl_off["source"] == default["source"]
                and update_off["update_eh"] > default["update_eh"]
                and update_off["curl"] == default["curl"]
                and update_off["source"] == default["source"]
                and both_off["curl"] == curl_off["curl"]
                and both_off["update_eh"] == update_off["update_eh"]
                and both_off["source"] == default["source"]
                and default["source"] > 0
                and source_off["source"] == 2 * default["source"]
                and source_off["curl"] == default["curl"]
                and source_off["update_eh"] == default["update_eh"]
            ):
                raise DiagnosticError(
                    "phase-batch disable controls lack the required causal call counts"
                )
            for condition, specification in CONDITIONS.items():
                expected_curl = (
                    curl_off["curl"]
                    if specification["disable_phase_batched_curl"]
                    else default["curl"]
                )
                expected_update = (
                    update_off["update_eh"]
                    if specification["disable_phase_batched_update_eh"]
                    else default["update_eh"]
                )
                expected_source = (
                    source_off["source"]
                    if specification["disable_phase_batched_source"]
                    else default["source"]
                )
                if calls[condition] != {
                    "curl": expected_curl,
                    "update_eh": expected_update,
                    "source": expected_source,
                }:
                    raise DiagnosticError(
                        "phase-batch control call counts differ from their exact "
                        f"causal class: {condition}"
                    )

    comparisons = []
    for pixels in PROFILE["size_screen_pixels"]:
        candidates = [
            sample
            for sample in samples
            if sample["schedule"]["pixels"] == pixels
        ]
        reference = next(
            sample
            for sample in candidates
            if sample["schedule"]["condition"] == "default"
            and sample["schedule"]["ranks"] == 1
            and sample["schedule"]["repeat"] == 1
        )["records"]["benchmark"]
        for sample in candidates:
            comparison = MGB.compare_record(
                sample["records"]["benchmark"],
                reference,
                PROFILE["observable_tolerance"],
            )
            if not comparison["observable_gate"]:
                raise DiagnosticError(
                    "rank/condition changed a fixed-work physical observable"
                )
            comparisons.append(
                {
                    "position": sample["schedule"]["position"],
                    "reference_pixels": pixels,
                    **comparison,
                }
            )
    return {
        "rank_device_uuid_mapping": {
            str(ranks): list(mapping) for ranks, mapping in sorted(mappings.items())
        },
        "rank_device_identity_mapping": {
            str(ranks): [dict(item) for item in identity_mapping]
            for ranks, identity_mapping in sorted(identity_mappings.items())
        },
        "observable_comparisons": comparisons,
    }


def median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise DiagnosticError("timing summary contains invalid samples")
    return statistics.median(values)


def classify_phase_effect(speedup_samples: list[float]) -> str:
    """Classify only repeat-consistent effects outside the fixed noise band.

    A speedup is unbatched seconds divided by phase-batched seconds, so values
    above one favor batching.  Every paired repeat must clear the same side of
    the margin; a favorable median alone is deliberately insufficient.
    """

    if len(speedup_samples) != PROFILE["repeats"]:
        raise DiagnosticError("phase effect lacks one paired sample per repeat")
    if any(
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or value <= 0
        for value in speedup_samples
    ):
        raise DiagnosticError("phase effect contains an invalid speedup")
    margin = PROFILE["phase_effect_margin"]
    if type(margin) not in (int, float) or not 0 < margin < 1:
        raise DiagnosticError("phase effect margin is invalid")
    batched_floor = 1.0 + float(margin)
    unbatched_ceiling = 1.0 / batched_floor
    if min(speedup_samples) >= batched_floor:
        return "phase-batched-faster"
    if max(speedup_samples) <= unbatched_ceiling:
        return "per-operation-faster"
    return "indeterminate"


def summarize_phase_batch_crossover(
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = {
        (item["pixels"], item["condition"]): item
        for item in configurations
    }
    if len(lookup) != len(configurations):
        raise DiagnosticError("phase crossover configurations are duplicated")
    effects: list[dict[str, Any]] = []
    expected_repeats = {
        str(repeat) for repeat in range(1, PROFILE["repeats"] + 1)
    }
    for pixels in PROFILE["size_screen_pixels"]:
        for specification in PHASE_EFFECT_SPECS:
            batched = lookup.get((pixels, specification["batched_condition"]))
            unbatched = lookup.get(
                (pixels, specification["unbatched_condition"])
            )
            if batched is None or unbatched is None:
                raise DiagnosticError("phase crossover matrix is incomplete")
            for ranks in PROFILE["ranks"]:
                rank_key = str(ranks)
                batched_by_repeat = batched["rank_summaries"][rank_key][
                    "seconds_by_repeat"
                ]
                unbatched_by_repeat = unbatched["rank_summaries"][rank_key][
                    "seconds_by_repeat"
                ]
                if (
                    set(batched_by_repeat) != expected_repeats
                    or set(unbatched_by_repeat) != expected_repeats
                ):
                    raise DiagnosticError(
                        "phase crossover timing pairs are incomplete"
                    )
                paired = {
                    repeat: (
                        float(unbatched_by_repeat[repeat])
                        / float(batched_by_repeat[repeat])
                    )
                    for repeat in sorted(expected_repeats, key=int)
                }
                samples = list(paired.values())
                batched_seconds = float(
                    batched["rank_summaries"][rank_key]["median_seconds"]
                )
                unbatched_seconds = float(
                    unbatched["rank_summaries"][rank_key]["median_seconds"]
                )
                cells = pixels**3
                if cells % ranks:
                    raise DiagnosticError(
                        "screen cells do not divide evenly across ranks"
                    )
                effects.append(
                    {
                        "effect": specification["effect"],
                        "pixels": pixels,
                        "global_cells": cells,
                        "ranks": ranks,
                        "nominal_local_cells": cells // ranks,
                        "batched_condition": specification[
                            "batched_condition"
                        ],
                        "unbatched_condition": specification[
                            "unbatched_condition"
                        ],
                        "batched_median_seconds": batched_seconds,
                        "unbatched_median_seconds": unbatched_seconds,
                        "paired_speedup_by_repeat": paired,
                        "median_paired_speedup": median(samples),
                        "ratio_of_medians": (
                            unbatched_seconds / batched_seconds
                        ),
                        "time_reduction_fraction": (
                            1.0 - batched_seconds / unbatched_seconds
                        ),
                        "classification": classify_phase_effect(samples),
                    }
                )

    policy_candidates: list[dict[str, Any]] = []
    support_required = PROFILE["minimum_policy_support_points"]
    if type(support_required) is not int or support_required < 2:
        raise DiagnosticError("minimum policy support must be at least two")
    for specification in PHASE_EFFECT_SPECS:
        for ranks in PROFILE["ranks"]:
            rows = sorted(
                (
                    item
                    for item in effects
                    if item["effect"] == specification["effect"]
                    and item["ranks"] == ranks
                ),
                key=lambda item: item["nominal_local_cells"],
            )
            threshold_index = None
            for index in range(len(rows)):
                suffix = rows[index:]
                if len(suffix) >= support_required and all(
                    item["classification"] == "phase-batched-faster"
                    for item in suffix
                ):
                    threshold_index = index
                    break
            counts = {
                label: sum(item["classification"] == label for item in rows)
                for label in (
                    "phase-batched-faster",
                    "per-operation-faster",
                    "indeterminate",
                )
            }
            if threshold_index is None:
                policy_candidates.append(
                    {
                        "effect": specification["effect"],
                        "ranks": ranks,
                        "status": "NO_STABLE_BATCHED_SUFFIX",
                        "recommendation": "leave-phase-batching-disabled",
                        "minimum_nominal_local_cells": None,
                        "minimum_pixels": None,
                        "supporting_points": [],
                        "classification_counts": counts,
                    }
                )
            else:
                suffix = rows[threshold_index:]
                policy_candidates.append(
                    {
                        "effect": specification["effect"],
                        "ranks": ranks,
                        "status": "OBSERVED_THRESHOLD_CANDIDATE",
                        "recommendation": (
                            "consider-phase-batching-at-or-above-threshold"
                        ),
                        "minimum_nominal_local_cells": suffix[0][
                            "nominal_local_cells"
                        ],
                        "minimum_pixels": suffix[0]["pixels"],
                        "supporting_points": [
                            {
                                "pixels": item["pixels"],
                                "nominal_local_cells": item[
                                    "nominal_local_cells"
                                ],
                                "median_paired_speedup": item[
                                    "median_paired_speedup"
                                ],
                            }
                            for item in suffix
                        ],
                        "classification_counts": counts,
                    }
                )
    return {
        "speedup_definition": (
            "per-operation seconds divided by phase-batched seconds"
        ),
        "classification_contract": {
            "margin_fraction": PROFILE["phase_effect_margin"],
            "phase_batched_faster": (
                "every paired repeat speedup >= 1 + margin"
            ),
            "per_operation_faster": (
                "every paired repeat speedup <= 1 / (1 + margin)"
            ),
            "indeterminate": "all remaining outcomes",
        },
        "policy_scope": (
            "development candidate only; requires production-workload "
            "confirmation before backend policy changes"
        ),
        "effects": effects,
        "policy_candidates": policy_candidates,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    validation = validate_matrix(samples)
    configurations: list[dict[str, Any]] = []
    for pixels in PROFILE["size_screen_pixels"]:
        conditions = (
            CONDITION_ORDER
            if pixels == PROFILE["causal_pixels"]
            else PHASE_SCREEN_CONDITION_ORDER
        )
        for condition in conditions:
            by_rank: dict[int, dict[str, Any]] = {}
            for ranks in PROFILE["ranks"]:
                selected = [
                    sample
                    for sample in samples
                    if sample["schedule"]["pixels"] == pixels
                    and sample["schedule"]["condition"] == condition
                    and sample["schedule"]["ranks"] == ranks
                ]
                if len(selected) != PROFILE["repeats"]:
                    raise DiagnosticError(
                        "configuration sample_count must equal fixed repeats"
                    )
                seconds = [
                    float(sample["records"]["benchmark"]["seconds"])
                    for sample in selected
                ]
                process_seconds = [
                    float(sample["process_wall_seconds"]) for sample in selected
                ]
                initialization_seconds = [
                    float(
                        sample["records"]["initialization_timing"]["max_seconds"]
                    )
                    for sample in selected
                ]
                if any(
                    sample["records"]["initial_condition"]["profile"]
                    != PROFILE["initial_condition"]
                    or sample["records"]["initialization_timing"][
                        "applications_per_rank"
                    ]
                    != 1
                    for sample in selected
                ):
                    raise DiagnosticError(
                        "configuration changed the fixed initialization profile"
                    )
                phase_calls = {
                    phase: statistics.median(
                        sample["records"]["benchmark"]["phase_calls"][phase][
                            "cuda_calls"
                        ]
                        for sample in selected
                    )
                    for phase in MGB.ALL_PHASES
                }
                timing_labels = sorted(
                    {
                        label
                        for sample in selected
                        for label in sample["records"]["print_times"]
                    }
                )
                print_times = {}
                for label in timing_labels:
                    values = [
                        float(
                            sample["records"]["print_times"][label][
                                "mean_seconds"
                            ]
                        )
                        for sample in selected
                        if label in sample["records"]["print_times"]
                    ]
                    if len(values) != PROFILE["repeats"]:
                        raise DiagnosticError(
                            "print_times sample_count must equal fixed repeats"
                        )
                    print_times[label] = {
                        "sample_count": len(values),
                        "mean_seconds_samples": values,
                        "median_mean_seconds": median(values),
                    }
                tile_counters = {
                    key: statistics.median(
                        sample["records"]["feature_markers"][
                            "tile_coalescing"
                        ][key]
                        for sample in selected
                    )
                    for key in (
                        "curl_chunk_phases",
                        "curl_input_tiles",
                        "update_eh_chunk_phases",
                        "update_eh_input_tiles",
                    )
                }
                by_rank[ranks] = {
                    "sample_count": len(selected),
                    "seconds_samples": seconds,
                    "seconds_by_repeat": {
                        str(sample["schedule"]["repeat"]): float(
                            sample["records"]["benchmark"]["seconds"]
                        )
                        for sample in selected
                    },
                    "median_seconds": median(seconds),
                    "process_wall_samples": process_seconds,
                    "median_process_wall_seconds": median(process_seconds),
                    "initial_condition": PROFILE["initial_condition"],
                    "source_profile": PROFILE["source_profile"],
                    "initialize_field_setup_seconds_samples": (
                        initialization_seconds
                    ),
                    "median_initialize_field_setup_seconds": median(
                        initialization_seconds
                    ),
                    "initialization_timing_domain": (
                        "untimed setup; excluded from worker FDTD seconds"
                    ),
                    "median_cuda_phase_calls": phase_calls,
                    "print_times": print_times,
                    "median_tile_coalescing_counters": tile_counters,
                    "median_total_cuda_calls": statistics.median(
                        sample["records"]["benchmark"]["cuda_calls"]
                        for sample in selected
                    ),
                    "median_h2d_bytes": statistics.median(
                        sample["records"]["benchmark"]["h2d_bytes"]
                        for sample in selected
                    ),
                    "median_d2h_bytes": statistics.median(
                        sample["records"]["benchmark"]["d2h_bytes"]
                        for sample in selected
                    ),
                    "median_mpi_messages": statistics.median(
                        sample["records"]["benchmark"]["mpi_messages"]
                        for sample in selected
                    ),
                    "median_mpi_scalars": statistics.median(
                        sample["records"]["benchmark"]["mpi_scalars"]
                        for sample in selected
                    ),
                    "median_cuda_aware_bytes": statistics.median(
                        sample["records"]["benchmark"]["cuda_aware_bytes"]
                        for sample in selected
                    ),
                    "median_pinned_bytes": statistics.median(
                        sample["records"]["benchmark"]["pinned_bytes"]
                        for sample in selected
                    ),
                }
            speedup = by_rank[1]["median_seconds"] / by_rank[2]["median_seconds"]
            process_speedup = (
                by_rank[1]["median_process_wall_seconds"]
                / by_rank[2]["median_process_wall_seconds"]
            )
            configurations.append(
                {
                    "pixels": pixels,
                    "cells": pixels**3,
                    "condition": condition,
                    "rank_summaries": {str(key): value for key, value in by_rank.items()},
                    "steady_state_1_to_2_gpu_speedup": speedup,
                    "steady_state_2gpu_parallel_efficiency": speedup / 2,
                    "process_wall_1_to_2_gpu_speedup": process_speedup,
                }
            )
    causal = [
        item for item in configurations if item["pixels"] == PROFILE["causal_pixels"]
    ]
    default = next(item for item in causal if item["condition"] == "default")
    causal_effects = []
    for item in causal:
        effects = {"condition": item["condition"], "by_rank": {}}
        for ranks in PROFILE["ranks"]:
            key = str(ranks)
            baseline = default["rank_summaries"][key]
            candidate = item["rank_summaries"][key]
            effects["by_rank"][key] = {
                "time_ratio_vs_default": (
                    candidate["median_seconds"] / baseline["median_seconds"]
                ),
                "curl_call_ratio_vs_default": (
                    candidate["median_cuda_phase_calls"]["curl"]
                    / baseline["median_cuda_phase_calls"]["curl"]
                ),
                "update_eh_call_ratio_vs_default": (
                    candidate["median_cuda_phase_calls"]["update_eh"]
                    / baseline["median_cuda_phase_calls"]["update_eh"]
                ),
                "source_call_ratio_vs_default": (
                    candidate["median_cuda_phase_calls"]["source"]
                    / baseline["median_cuda_phase_calls"]["source"]
                ),
            }
        causal_effects.append(effects)
    phase_batch_crossover = summarize_phase_batch_crossover(configurations)
    hardware_binding = {
        "rank_device_identity_mapping": validation[
            "rank_device_identity_mapping"
        ],
        "reuse_contract": (
            "policy candidates are valid only for this exact GPU/rank "
            "identity; a different GPU model, capability, UUID mapping, or "
            "rank count requires a fresh crossover run"
        ),
    }
    hardware_binding["binding_sha256"] = canonical_sha256(hardware_binding)
    phase_batch_crossover["hardware_binding"] = hardware_binding
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "state": "VALIDATED_AWAITING_SEAL",
        "profile_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "evidence_class": "development-diagnostic-only",
        "release_qualification": {
            "evaluated": False,
            "pass": False,
            "reason": "diagnostic matrices cannot serve as release evidence",
        },
        "sample_count": len(samples),
        "timing_domains": {
            "worker_seconds": "warmup-excluded steady-state FDTD",
            "initialize_field_setup_seconds": (
                "untimed setup; never included in FDTD speedup"
            ),
            "process_wall_seconds": "end-to-end controller-observed process time",
        },
        "source_profile": PROFILE["source_profile"],
        **validation,
        "configurations": configurations,
        "causal_effects": causal_effects,
        "phase_batch_crossover": phase_batch_crossover,
    }


def resolve_recorded_path(record: dict[str, Any], label: str) -> pathlib.Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise DiagnosticError(f"receipt {label} record is malformed")
    path = pathlib.Path(record["path"])
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    actual = stable_file_record(path, REPO)
    if (
        actual["size_bytes"] != record.get("size_bytes")
        or actual["sha256"] != record.get("sha256")
    ):
        raise DiagnosticError(f"receipt-bound {label} changed: {path}")
    return path


def resolve_build_binding(
    receipt_snapshot: pathlib.Path,
    expected_receipt_id: str,
    requested_mpiexec: pathlib.Path,
) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path, pathlib.Path]:
    try:
        receipt = verify_build_receipt(receipt_snapshot, REPO)
    except ProvenanceError as error:
        raise DiagnosticError(f"invalid snapshotted build receipt: {error}") from error
    if receipt.get("receipt_id") != expected_receipt_id:
        raise DiagnosticError("snapshotted receipt differs from expected receipt ID")
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise DiagnosticError("snapshotted receipt has the wrong build kind")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DiagnosticError("snapshotted receipt has no artifact map")
    executable = resolve_recorded_path(
        artifacts.get("gpu_mpi_performance"), "gpu_mpi_performance"
    )
    library = resolve_recorded_path(artifacts.get("libmeep"), "libmeep")
    toolchain = receipt.get("toolchain")
    mpiexec_record = toolchain.get("mpiexec") if isinstance(toolchain, dict) else None
    mpiexec = resolve_recorded_path(mpiexec_record, "mpiexec")
    invoked = mpiexec_record.get("invoked_path") if isinstance(mpiexec_record, dict) else None
    if (
        not isinstance(invoked, str)
        or pathlib.Path(invoked).resolve() != mpiexec
        or requested_mpiexec.resolve() != mpiexec
    ):
        raise DiagnosticError("requested mpiexec is not the receipt-recorded tool")
    return receipt, executable, library, mpiexec


def build_manifest(output: pathlib.Path) -> dict[str, Any]:
    excluded = {
        output / "COMPLETE.json",
        output / "FAILED.json",
        output / "TERMINAL.json",
        output / "artifacts.sha256.json",
    }
    files = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if path.is_symlink():
            raise DiagnosticError(f"manifest path is a symlink: {relative}")
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise DiagnosticError(f"manifest path is not a regular file: {relative}")
        if metadata.st_nlink != 1:
            raise DiagnosticError(f"manifest path has external hard links: {relative}")
        try:
            path.resolve().relative_to(output.resolve())
        except ValueError as error:
            raise DiagnosticError(f"manifest path escapes output: {relative}") from error
        if path not in excluded:
            files.append(path)
    records = []
    unbound = []
    for path in files:
        try:
            records.append(stable_file_record(path, output))
        except Exception as error:
            unbound.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "schema_version": 1,
        "complete": not unbound,
        "files": records,
        "unbound_artifacts": unbound,
    }


def seal_failed(
    output: pathlib.Path,
    error: BaseException,
    *,
    run_id: str,
    expected_receipt_id: str | None,
    verified_receipt_id: str | None,
) -> None:
    complete = output / "COMPLETE.json"
    failed = output / "FAILED.json"
    terminal = output / "TERMINAL.json"
    if complete.exists() or complete.is_symlink():
        raise DiagnosticError("refusing to overwrite a COMPLETE diagnostic")
    if failed.exists() or failed.is_symlink():
        raise DiagnosticError("refusing to overwrite a FAILED diagnostic")
    if terminal.exists() or terminal.is_symlink():
        raise DiagnosticError("refusing to overwrite a terminal diagnostic claim")
    failure_report = output / "failure-report.json"
    atomic_write_json(
        failure_report,
        {
            "schema_version": 1,
            "state": "FAILED",
            "run_id": run_id,
            "finished_at_utc": utc_now(),
            "profile_id": PROFILE_ID,
            "evidence_class": "development-diagnostic-only",
            "release_qualification": False,
            "expected_receipt_id": expected_receipt_id,
            "verified_receipt_id": verified_receipt_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "source_identity_proven": False,
            "source_identity_note": (
                "FAILED evidence binds only the observed error and completed "
                "artifacts; it never self-attests mutable executed-source bytes"
            ),
        },
    )
    manifest_path = output / "artifacts.sha256.json"
    atomic_write_json(manifest_path, build_manifest(output))
    publish_terminal_marker(
        output,
        "FAILED",
        {
            "schema_version": 1,
            "state": "FAILED",
            "run_id": run_id,
            "expected_receipt_id": expected_receipt_id,
            "verified_receipt_id": verified_receipt_id,
            "failure_report": stable_file_record(failure_report, output),
            "manifest": stable_file_record(manifest_path, output),
            "complete_marker_absent": True,
        },
    )


def sample_stem(item: dict[str, Any]) -> str:
    return (
        f"sample-{item['position']:03d}-{item['matrix']}-p{item['pixels']}-"
        f"r{item['ranks']}-repeat{item['repeat']}-{item['condition']}"
    )


def run_sample(
    output: pathlib.Path,
    item: dict[str, Any],
    base_environment: dict[str, str],
    mpiexec: pathlib.Path,
    executable: pathlib.Path,
    preload_library: pathlib.Path,
    execution_bindings: dict[str, dict[str, Any]],
    execution_descriptors: dict[str, int],
) -> dict[str, Any]:
    stem = sample_stem(item)
    environment = condition_environment(
        base_environment,
        item["condition"],
        item["pixels"],
        item["ranks"],
    )
    command = isolated_command(
        environment,
        mpiexec,
        executable,
        preload_library,
        item["ranks"],
    )
    verify_stable_execution_handles(execution_bindings, execution_descriptors)
    run = MGB.run_logged(
        command,
        output / f"{stem}.log",
        outer_timeout=PROFILE["timeout_seconds"] + 30,
    )
    verify_stable_execution_handles(execution_bindings, execution_descriptors)
    attempt = {
        "schema_version": 1,
        "schedule": item,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "environment": dict(sorted(environment.items())),
        "environment_sha256": canonical_sha256(dict(sorted(environment.items()))),
        "returncode": run["returncode"],
        "process_wall_seconds": run["elapsed_seconds"],
        "stdout_sha256": hashlib.sha256(run["stdout"].encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(run["stderr"].encode()).hexdigest(),
        "log": stable_file_record(pathlib.Path(run["log"]), output),
    }
    atomic_write_json(output / f"{stem}-attempt.json", attempt)
    reject_unknown_gpmeep_streams(run["stdout"], run["stderr"])
    if run["returncode"] != 0:
        raise DiagnosticError(
            f"sample position {item['position']} exited with {run['returncode']}"
        )
    records = parse_sample(run["stdout"], item)
    sample = {
        **attempt,
        "records": records,
    }
    atomic_write_json(output / f"{stem}.json", sample)
    return sample


def markdown_report(summary: dict[str, Any]) -> str:
    crossover = summary["phase_batch_crossover"]
    hardware = crossover["hardware_binding"]
    lines = [
        "# C++ multi-GPU phase-batch crossover diagnostic",
        "",
        "This report is development diagnostic evidence, not release qualification.",
        "",
        f"- Hardware binding SHA-256: `{hardware['binding_sha256']}`",
        f"- Policy reuse contract: {hardware['reuse_contract']}",
        f"- Paired speedup definition: {crossover['speedup_definition']}",
        "",
        "## Bound GPU identity",
        "",
        "| MPI rank | GPU | compute capability | UUID |",
        "|---:|---|---:|---|",
    ]
    canonical_devices = hardware["rank_device_identity_mapping"]["2"]
    for device in canonical_devices:
        name = str(device["name"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {device['rank']} | {name} | "
            f"{device['compute_major']}.{device['compute_minor']} | "
            f"`{device['uuid']}` |"
        )
    lines.extend(
        [
            "",
            "## 1→2 GPU scaling",
            "",
        "| pixels | condition | steady 1→2 GPU | process 1→2 GPU |",
        "|---:|---|---:|---:|",
        ]
    )
    for item in summary["configurations"]:
        lines.append(
            f"| {item['pixels']} | `{item['condition']}` | "
            f"{item['steady_state_1_to_2_gpu_speedup']:.6f}x | "
            f"{item['process_wall_1_to_2_gpu_speedup']:.6f}x |"
        )
    lines.extend(
        [
            "",
            "## Phase-batch crossover",
            "",
            "A classification is favorable only when every paired repeat "
            "clears the fixed 3% margin in the same direction.",
            "",
            "| pixels | ranks | effect | batched s | per-operation s | "
            "paired speedup | classification |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in crossover["effects"]:
        lines.append(
            f"| {item['pixels']} | {item['ranks']} | `{item['effect']}` | "
            f"{item['batched_median_seconds']:.9f} | "
            f"{item['unbatched_median_seconds']:.9f} | "
            f"{item['median_paired_speedup']:.6f}x | "
            f"`{item['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Hardware-bound policy candidates",
            "",
            "These are development candidates, not portable defaults. A "
            "different device identity requires a fresh run.",
            "",
            "| ranks | effect | status | recommendation | minimum pixels | "
            "minimum nominal local cells |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for item in crossover["policy_candidates"]:
        pixels = (
            "—" if item["minimum_pixels"] is None
            else str(item["minimum_pixels"])
        )
        cells = (
            "—" if item["minimum_nominal_local_cells"] is None
            else str(item["minimum_nominal_local_cells"])
        )
        lines.append(
            f"| {item['ranks']} | `{item['effect']}` | "
            f"`{item['status']}` | `{item['recommendation']}` | "
            f"{pixels} | {cells} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_authoritative_inner(
    args: argparse.Namespace, run_id: str, invocation: dict[str, Any]
) -> int:
    archive = args.output / "archive"
    receipt_snapshot = args.build_receipt
    receipt, executable, library, mpiexec = resolve_build_binding(
        receipt_snapshot, args.expected_receipt_id, args.mpiexec
    )
    verified_receipt_id = receipt["receipt_id"]
    atomic_write_json(
        args.output / "verified-receipt-id.json",
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "expected_receipt_id": args.expected_receipt_id,
            "verified_receipt_id": verified_receipt_id,
        },
    )
    runtime = archive / "runtime"
    runtime_bin = runtime / "bin"
    runtime_lib = runtime / "lib"
    runtime_config = runtime / "config"
    runtime_bin.mkdir(parents=True)
    runtime_lib.mkdir()
    runtime_config.mkdir()
    executable_snapshot = runtime_bin / "gpu-mpi-performance"
    mpiexec_snapshot = runtime_bin / "mpiexec"
    soname_match = re.fullmatch(r"(libmeep\.so\.[0-9]+)(?:\..*)?", library.name)
    if soname_match is None:
        raise DiagnosticError(f"cannot derive receipt libmeep SONAME: {library.name}")
    library_snapshot = runtime_lib / soname_match.group(1)
    binary_bindings = {
        "worker_elf": stable_binary_snapshot(
            executable,
            executable_snapshot,
            REPO,
            executable_required=True,
        ),
        "libmeep_soname": stable_binary_snapshot(
            library,
            library_snapshot,
            REPO,
            executable_required=False,
        ),
        "mpiexec": stable_binary_snapshot(
            mpiexec,
            mpiexec_snapshot,
            REPO,
            executable_required=True,
        ),
    }
    mca_source = REPO / "environment/openmpi-qualification-mca-params.conf"
    mca_snapshot = runtime_config / "openmpi-qualification-mca-params.conf"
    mca_binding = stable_binary_snapshot(
        mca_source,
        mca_snapshot,
        REPO,
        executable_required=False,
    )
    execution_descriptors: dict[str, int] = {}
    stable_execution: dict[str, dict[str, Any]] = {}
    for name, snapshot in (
        ("worker_elf", executable_snapshot),
        ("libmeep_soname", library_snapshot),
        ("mpiexec", mpiexec_snapshot),
        ("mca", mca_snapshot),
    ):
        descriptor, binding = create_stable_execution_handle(
            snapshot, args.output, name
        )
        execution_descriptors[name] = descriptor
        stable_execution[name] = binding
    executable_execution = pathlib.Path(
        stable_execution["worker_elf"]["execution_path"]
    )
    library_execution = pathlib.Path(
        stable_execution["libmeep_soname"]["execution_path"]
    )
    mpiexec_execution = pathlib.Path(
        stable_execution["mpiexec"]["execution_path"]
    )
    mca_execution = pathlib.Path(stable_execution["mca"]["execution_path"])
    prefix = mpiexec.parent.parent
    home = args.output / "home"
    home.mkdir()
    base_environment = {
        "HOME": str(home),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{runtime_lib}:{prefix / 'lib'}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "OMPI_MCA_mca_base_param_files": str(mca_execution),
        "OMPI_MCA_mca_base_component_path": str(prefix / "lib/openmpi"),
        "PMIX_MCA_mca_base_param_files": str(mca_execution),
        "PMIX_MCA_mca_base_component_path": str(prefix / "lib/pmix"),
        "PRTE_MCA_mca_base_param_files": str(mca_execution),
        "OPAL_PREFIX": str(prefix),
    }
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if key in os.environ:
            base_environment[key] = os.environ[key]

    build_binding = {
        "expected_receipt_id": args.expected_receipt_id,
        "verified_receipt_id": verified_receipt_id,
        "build_input_id": receipt.get("build_input_id"),
        "artifact_set_id": receipt.get("artifact_set_id"),
        "receipt_payload_snapshot": stable_file_record(receipt_snapshot, args.output),
        "receipt_verified_originals": {
            "executable": stable_file_record(executable, REPO),
            "library": stable_file_record(library, REPO),
            "mpiexec": stable_file_record(mpiexec, REPO),
        },
        "executed_snapshots": binary_bindings,
        "stable_execution_images": stable_execution,
        "mca_snapshot": mca_binding,
    }
    protocol = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "profile": PROFILE,
        "semantic_profile_contract": semantic_profile_contract(),
        "profile_sha256": PROFILE_SHA256,
        "schedule": expected_schedule(),
        "evidence_class": "development-diagnostic-only",
        "release_qualification": False,
        "build_binding": build_binding,
        "authoritative_inner_stage": invocation["authoritative_inner_stage"],
        "runtime_source_identity": runtime_source_identity(),
        "control_python_process_executable": invocation["control_python"],
        "base_environment": dict(sorted(base_environment.items())),
    }
    atomic_write_json(args.output / "protocol.json", protocol)

    openmpi_preflight = []
    for option, label in (("--version", "version"), ("--help", "help")):
        command = [
            "/usr/bin/env",
            "-i",
            *[f"{key}={base_environment[key]}" for key in sorted(base_environment)],
            str(mpiexec_execution),
            option,
        ]
        verify_stable_execution_handles(stable_execution, execution_descriptors)
        run = MGB.run_logged(
            command,
            args.output / f"archived-mpiexec-{label}.log",
            outer_timeout=60,
        )
        verify_stable_execution_handles(stable_execution, execution_descriptors)
        combined = run["stdout"] + run["stderr"]
        if run["returncode"] != 0 or not combined.strip() or (
            option == "--version" and "Open MPI" not in combined
        ):
            raise DiagnosticError(
                f"archived mpiexec {label} preflight failed"
            )
        openmpi_preflight.append(
            {
                "option": option,
                "command": command,
                "command_sha256": canonical_sha256(command),
                "returncode": run["returncode"],
                "stdout_sha256": hashlib.sha256(run["stdout"].encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(run["stderr"].encode()).hexdigest(),
                "log": stable_file_record(pathlib.Path(run["log"]), args.output),
            }
        )
    atomic_write_json(
        args.output / "archived-mpiexec-preflight.json",
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "executed_mpiexec": stable_file_record(mpiexec_snapshot, args.output),
            "stable_mpiexec_execution": stable_execution["mpiexec"],
            "checks": openmpi_preflight,
        },
    )

    loader_environment = dict(base_environment)
    loader_environment["LD_PRELOAD"] = str(library_execution)
    loader_environment["LD_TRACE_LOADED_OBJECTS"] = "1"
    loader_command = [
        "/usr/bin/env",
        "-i",
        *[
            f"{key}={loader_environment[key]}"
            for key in sorted(loader_environment)
        ],
        str(executable_execution),
    ]
    verify_stable_execution_handles(stable_execution, execution_descriptors)
    loader_run = MGB.run_logged(
        loader_command,
        args.output / "stable-worker-loader-preflight.log",
        outer_timeout=60,
    )
    verify_stable_execution_handles(stable_execution, execution_descriptors)
    loader_output = loader_run["stdout"] + loader_run["stderr"]
    soname = library_snapshot.name
    if loader_run["returncode"] != 0:
        raise DiagnosticError(
            "stable worker loader did not resolve libmeep to the fixed open image"
        )
    loader_resolution = validate_loader_trace(
        loader_output, library_execution, library, soname
    )
    loader_preflight = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "command": loader_command,
        "command_sha256": canonical_sha256(loader_command),
        "returncode": loader_run["returncode"],
        **loader_resolution,
        "expected_library_execution_path": str(library_execution),
        "stdout_sha256": hashlib.sha256(loader_run["stdout"].encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(loader_run["stderr"].encode()).hexdigest(),
        "log": stable_file_record(pathlib.Path(loader_run["log"]), args.output),
    }
    atomic_write_json(
        args.output / "stable-worker-loader-preflight.json", loader_preflight
    )

    query_environment = dict(base_environment)
    query_environment["MEEP_GPU_MULTI_QUERY_DEVICE_COUNT"] = "1"
    query_command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "60s",
        "/usr/bin/env",
        "-i",
        *[f"{key}={query_environment[key]}" for key in sorted(query_environment)],
        f"LD_PRELOAD={library_execution}",
        str(executable_execution),
    ]
    verify_stable_execution_handles(stable_execution, execution_descriptors)
    query = MGB.run_logged(
        query_command, args.output / "device-query.log", outer_timeout=90
    )
    verify_stable_execution_handles(stable_execution, execution_descriptors)
    query_attempt = {
        "schema_version": 1,
        "command": query_command,
        "command_sha256": canonical_sha256(query_command),
        "returncode": query["returncode"],
        "stdout_sha256": hashlib.sha256(query["stdout"].encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(query["stderr"].encode()).hexdigest(),
        "log": stable_file_record(pathlib.Path(query["log"]), args.output),
    }
    atomic_write_json(args.output / "device-query.json", query_attempt)
    if query["returncode"] != 0:
        raise DiagnosticError("CUDA device query failed")
    try:
        available_gpus = MGB._device_count(query["stdout"])
    except MGB.EvidenceError as error:
        raise DiagnosticError(str(error)) from error
    if available_gpus < 2:
        raise DiagnosticError("fixed diagnostic requires two visible CUDA GPUs")

    samples = [
        run_sample(
            args.output,
            item,
            base_environment,
            mpiexec_execution,
            executable_execution,
            library_execution,
            stable_execution,
            execution_descriptors,
        )
        for item in expected_schedule()
    ]
    for sample in samples:
        verify_recorded_file(
            sample["log"],
            args.output,
            f"sample position {sample['schedule']['position']} log",
        )
    summary = summarize(samples)
    summary["sample_artifacts"] = [
        {
            "schedule": item,
            "sample": stable_file_record(
                args.output / f"{sample_stem(item)}.json", args.output
            ),
            "attempt": stable_file_record(
                args.output / f"{sample_stem(item)}-attempt.json", args.output
            ),
            "log": stable_file_record(
                args.output / f"{sample_stem(item)}.log", args.output
            ),
        }
        for item in expected_schedule()
    ]

    verify_binary_snapshot(
        binary_bindings["worker_elf"], executable_snapshot, REPO
    )
    verify_binary_snapshot(
        binary_bindings["libmeep_soname"], library_snapshot, REPO
    )
    verify_binary_snapshot(binary_bindings["mpiexec"], mpiexec_snapshot, REPO)
    verify_binary_snapshot(mca_binding, mca_snapshot, REPO)
    verify_stable_execution_handles(stable_execution, execution_descriptors)
    receipt_after, executable_after, library_after, mpiexec_after = (
        resolve_build_binding(
            receipt_snapshot, args.expected_receipt_id, args.mpiexec
        )
    )
    control_python_after = verify_control_python_identity(
        invocation.get("control_python")
    )
    end_binding = {
        "expected_receipt_id": args.expected_receipt_id,
        "verified_receipt_id": receipt_after["receipt_id"],
        "verified_originals": {
            "executable": stable_file_record(executable_after, REPO),
            "library": stable_file_record(library_after, REPO),
            "mpiexec": stable_file_record(mpiexec_after, REPO),
        },
        "executed_snapshots": {
            "worker_elf": stable_file_record(executable_snapshot, args.output),
            "libmeep_soname": stable_file_record(library_snapshot, args.output),
            "mpiexec": stable_file_record(mpiexec_snapshot, args.output),
            "mca": stable_file_record(mca_snapshot, args.output),
        },
        "stable_execution_images": stable_execution,
        "control_python_process_executable": control_python_after,
    }
    if (
        end_binding["verified_receipt_id"] != verified_receipt_id
        or end_binding["verified_originals"]
        != build_binding["receipt_verified_originals"]
    ):
        raise DiagnosticError("receipt-bound build closure changed during run")

    summary.update(
        run_id=run_id,
        finished_at_utc=utc_now(),
        available_compatible_gpus=available_gpus,
        build_binding=build_binding,
        end_build_binding=end_binding,
        protocol=stable_file_record(args.output / "protocol.json", args.output),
        device_query=query_attempt,
        archived_mpiexec_preflight=openmpi_preflight,
        sealed_worker_loader_preflight=loader_preflight,
    )
    report_path = args.output / "report.json"
    markdown_path = args.output / "report.md"
    atomic_write_json(report_path, summary)
    atomic_write_text(markdown_path, markdown_report(summary))
    manifest_path = args.output / "artifacts.sha256.json"
    manifest = build_manifest(args.output)
    if not manifest["complete"]:
        raise DiagnosticError("one or more diagnostic artifacts could not be bound")
    atomic_write_json(manifest_path, manifest)
    for record in manifest["files"]:
        verify_recorded_file(record, args.output, f"manifest file {record['path']}")
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "expected_receipt_id": args.expected_receipt_id,
        "verified_receipt_id": verified_receipt_id,
        "evidence_class": "development-diagnostic-only",
        "release_qualification": False,
        "report": stable_file_record(report_path, args.output),
        "manifest": stable_file_record(manifest_path, args.output),
    }
    if (args.output / "FAILED.json").exists():
        raise DiagnosticError("FAILED marker exists before COMPLETE publication")
    publish_terminal_marker(args.output, "COMPLETE", marker)
    try:
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
    except BrokenPipeError:
        # COMPLETE is already durable; a closed presentation pipe must not
        # retroactively relabel valid evidence as FAILED.
        pass
    return 0


INVOCATION_KEYS = {
    "schema_version",
    "run_id",
    "outer_pid",
    "output",
    "repo",
    "expected_receipt_id",
    "requested_mpiexec",
    "receipt_snapshot",
    "source_bindings",
    "runtime_source_identity",
    "capability_sha256",
    "bootstrap_sha256",
    "control_python",
    "authoritative_inner_stage",
}
RUNNING_KEYS = {
    "schema_version",
    "state",
    "run_id",
    "created_at_utc",
    "outer_pid",
    "capability_sha256",
}


def _normal_required_arguments(args: argparse.Namespace) -> None:
    missing = [
        option
        for option, value in (
            ("--build-receipt", args.build_receipt),
            ("--expected-receipt-id", args.expected_receipt_id),
            ("--output", args.output),
            ("--mpiexec", args.mpiexec),
        )
        if value is None
    ]
    if missing:
        raise DiagnosticError(
            "normal invocation requires " + ", ".join(missing)
        )
    if any(
        value is not None
        for value in (args.invocation, args.invocation_sha256, args.capability)
    ):
        raise DiagnosticError("internal authority arguments require internal mode")


def _verified_receipt_id_from_output(output: pathlib.Path) -> str | None:
    path = output / "verified-receipt-id.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        record = load_strict_json(path)
    except DiagnosticError:
        return None
    value = record.get("verified_receipt_id")
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _source_snapshots(
    output: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], pathlib.Path]:
    archive_python = output / "archive/python"
    archive_python.mkdir(parents=True)
    bindings: dict[str, Any] = {}
    runtime_identity: dict[str, Any] = {}
    for name, source in SOURCE_DEPENDENCIES.items():
        destination = archive_python / ARCHIVED_SOURCE_NAMES[name]
        binding = stable_snapshot(source, destination, output)
        bindings[name] = binding
        runtime_identity[name] = {
            "path": str(destination.resolve()),
            "loaded_source_sha256": binding["observed_snapshot"]["sha256"],
            "execution": (
                "bootstrap-compile-exec-exact-source-bytes"
                if name == "runner"
                else "compile-exec-exact-source-bytes"
            ),
        }
        if name == "runner":
            runtime_identity[name]["bootstrap_sha256"] = INNER_BOOTSTRAP_SHA256
    runtime_identity["provenance"][
        "qualification_contract_binding_sha256"
    ] = bindings["qualification_contract"]["observed_snapshot"]["sha256"]
    return bindings, runtime_identity, archive_python / ARCHIVED_SOURCE_NAMES["runner"]


def run_outer_controller(
    args: argparse.Namespace, run_id: str, capability: str
) -> int:
    assert_fixed_profile()
    source_bindings, expected_runtime_identity, archived_runner = (
        _source_snapshots(args.output)
    )
    archive_input = args.output / "archive/input"
    archive_control = args.output / "archive/control"
    archive_input.mkdir()
    archive_control.mkdir()
    receipt_snapshot = archive_input / "build-provenance.json"
    receipt_binding = stable_snapshot(args.build_receipt, receipt_snapshot, args.output)
    control_python_path = pathlib.Path(sys.executable).resolve()
    control_python = stable_open_identity(control_python_path)
    authoritative_inner_stage = {
        "mechanism": "isolated-bootstrap-compile-exec-archived-source",
        "bootstrap_sha256": INNER_BOOTSTRAP_SHA256,
        "runner": source_bindings["runner"]["observed_snapshot"],
        "transitive_dependencies": {
            name: source_bindings[name]["observed_snapshot"]
            for name in sorted(source_bindings)
            if name != "runner"
        },
    }
    invocation = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": run_id,
        "outer_pid": os.getpid(),
        "output": str(args.output),
        "repo": str(REPO.resolve()),
        "expected_receipt_id": args.expected_receipt_id,
        "requested_mpiexec": str(args.mpiexec),
        "receipt_snapshot": receipt_binding,
        "source_bindings": source_bindings,
        "runtime_source_identity": expected_runtime_identity,
        "capability_sha256": hashlib.sha256(capability.encode()).hexdigest(),
        "bootstrap_sha256": INNER_BOOTSTRAP_SHA256,
        "control_python": control_python,
        "authoritative_inner_stage": authoritative_inner_stage,
    }
    invocation_path = archive_control / "inner-invocation.json"
    atomic_write_json(invocation_path, invocation)
    invocation_sha256 = sha256_file(invocation_path)
    command = [
        str(control_python_path),
        "-I",
        "-B",
        "-c",
        INNER_BOOTSTRAP,
        str(archived_runner),
        source_bindings["runner"]["observed_snapshot"]["sha256"],
        INNER_BOOTSTRAP_SHA256,
        "--internal-authoritative",
        "--invocation",
        str(invocation_path),
        "--invocation-sha256",
        invocation_sha256,
        "--capability",
        capability,
    ]
    descriptor, transport_log_name = tempfile.mkstemp(
        prefix="gpmeep-authoritative-inner-", suffix=".log"
    )
    os.close(descriptor)
    transport_log = pathlib.Path(transport_log_name)
    try:
        result = MGB.run_logged(command, transport_log, outer_timeout=40000)
    finally:
        transport_log.unlink(missing_ok=True)
    if result["returncode"] != 0:
        raise DiagnosticError(
            "archived authoritative inner stage failed with return code "
            f"{result['returncode']}"
        )
    complete_path = args.output / "COMPLETE.json"
    failed_path = args.output / "FAILED.json"
    if failed_path.exists() or failed_path.is_symlink():
        raise DiagnosticError("authoritative inner stage sealed FAILED")
    complete = load_strict_json(complete_path)
    if (
        complete.get("state") != "COMPLETE"
        or complete.get("run_id") != run_id
        or complete.get("expected_receipt_id") != args.expected_receipt_id
        or complete.get("verified_receipt_id") != args.expected_receipt_id
    ):
        raise DiagnosticError("authoritative COMPLETE marker is inconsistent")
    verify_recorded_file(complete.get("report"), args.output, "complete report")
    verify_recorded_file(complete.get("manifest"), args.output, "complete manifest")
    terminal_path = verify_recorded_file(
        complete.get("terminal_claim"), args.output, "complete terminal claim"
    )
    terminal = load_strict_json(terminal_path)
    if (
        terminal.get("state") != "COMPLETE"
        or terminal.get("run_id") != run_id
        or terminal.get("expected_receipt_id") != args.expected_receipt_id
        or terminal.get("verified_receipt_id") != args.expected_receipt_id
    ):
        raise DiagnosticError("authoritative terminal claim is inconsistent")
    return 0


def _validate_internal_invocation(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, dict[str, Any]]:
    if any(
        value is not None
        for value in (
            args.build_receipt,
            args.expected_receipt_id,
            args.output,
            args.mpiexec,
        )
    ):
        raise DiagnosticError("internal invocation rejects normal-mode arguments")
    if args.invocation is None or args.invocation_sha256 is None or args.capability is None:
        raise DiagnosticError("internal invocation lacks authority arguments")
    invocation_path = pathlib.Path(os.path.abspath(args.invocation))
    if invocation_path.is_symlink() or not invocation_path.is_file():
        raise DiagnosticError("internal invocation payload is not a regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", args.invocation_sha256):
        raise DiagnosticError("internal invocation hash is malformed")
    if sha256_file(invocation_path) != args.invocation_sha256:
        raise DiagnosticError("internal invocation payload hash differs")
    invocation = load_strict_json(invocation_path)
    if set(invocation) != INVOCATION_KEYS:
        raise DiagnosticError("internal invocation has an invalid exact schema")
    if invocation.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise DiagnosticError("internal invocation schema version differs")
    if (
        invocation.get("bootstrap_sha256") != INNER_BOOTSTRAP_SHA256
        or globals().get("__gpmeep_bootstrap_sha256__")
        != INNER_BOOTSTRAP_SHA256
    ):
        raise DiagnosticError("internal bootstrap identity differs")
    if not isinstance(invocation.get("run_id"), str) or not invocation["run_id"]:
        raise DiagnosticError("internal invocation run ID is invalid")
    if type(invocation.get("outer_pid")) is not int or invocation["outer_pid"] <= 1:
        raise DiagnosticError("internal invocation outer PID is invalid")
    if os.getppid() != invocation["outer_pid"]:
        raise DiagnosticError("internal invocation parent process is not the claimant")
    if not re.fullmatch(r"[0-9a-f]{64}", args.capability):
        raise DiagnosticError("internal invocation capability is malformed")
    if hashlib.sha256(args.capability.encode()).hexdigest() != invocation.get(
        "capability_sha256"
    ):
        raise DiagnosticError("internal invocation capability differs")
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise DiagnosticError("authoritative runner requires isolated -I -B Python")

    output = pathlib.Path(invocation.get("output", ""))
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not output.is_dir()
        or output.resolve() != output
        or output.parent.resolve() != output.parent
        or invocation_path != output / "archive/control/inner-invocation.json"
    ):
        raise DiagnosticError("internal invocation output claim is invalid")
    running = load_strict_json(output / "RUNNING.json")
    if (
        set(running) != RUNNING_KEYS
        or stable_file_record(output / "RUNNING.json", output)["path"]
        != "RUNNING.json"
        or running.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION
        or running.get("state") != "CLAIMED"
        or running.get("run_id") != invocation["run_id"]
        or running.get("capability_sha256") != invocation["capability_sha256"]
    ):
        raise DiagnosticError("internal invocation has no matching exclusive claim")
    expected_receipt_id = invocation.get("expected_receipt_id")
    if not isinstance(expected_receipt_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_receipt_id
    ):
        raise DiagnosticError("internal expected receipt ID is malformed")

    repo = pathlib.Path(invocation.get("repo", ""))
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise DiagnosticError("internal repository binding is invalid")
    global REPO
    REPO = repo
    expected_bindings = invocation.get("source_bindings")
    if not isinstance(expected_bindings, dict) or set(expected_bindings) != set(
        SOURCE_DEPENDENCIES
    ):
        raise DiagnosticError("internal source binding map is incomplete")
    for name, binding in expected_bindings.items():
        if not isinstance(binding, dict):
            raise DiagnosticError(f"internal source binding is malformed: {name}")
        snapshot = output / binding.get("observed_snapshot", {}).get("path", "")
        verify_recorded_file(
            binding.get("observed_snapshot"), output, f"archived source {name}"
        )
        expected_path = pathlib.Path(SOURCE_DEPENDENCIES[name]).resolve()
        if snapshot.resolve() != expected_path:
            raise DiagnosticError(f"archived source path differs: {name}")
    stage = invocation.get("authoritative_inner_stage")
    expected_stage = {
        "mechanism": "isolated-bootstrap-compile-exec-archived-source",
        "bootstrap_sha256": INNER_BOOTSTRAP_SHA256,
        "runner": expected_bindings["runner"]["observed_snapshot"],
        "transitive_dependencies": {
            name: expected_bindings[name]["observed_snapshot"]
            for name in sorted(expected_bindings)
            if name != "runner"
        },
    }
    if stage != expected_stage:
        raise DiagnosticError("authoritative inner-stage source binding differs")
    actual_runtime_identity = runtime_source_identity()
    if actual_runtime_identity != invocation.get("runtime_source_identity"):
        raise DiagnosticError(
            "executed Python source bytes differ from the archived source closure"
        )
    receipt_binding = invocation.get("receipt_snapshot")
    if not isinstance(receipt_binding, dict):
        raise DiagnosticError("receipt snapshot binding is malformed")
    receipt_snapshot = verify_recorded_file(
        receipt_binding.get("observed_snapshot"), output, "receipt snapshot"
    )
    verify_control_python_identity(invocation.get("control_python"))
    requested_mpiexec = pathlib.Path(invocation.get("requested_mpiexec", ""))
    if not requested_mpiexec.is_absolute():
        raise DiagnosticError("requested mpiexec binding is not absolute")
    inner_args = argparse.Namespace(
        output=output,
        build_receipt=receipt_snapshot,
        expected_receipt_id=expected_receipt_id,
        mpiexec=requested_mpiexec,
    )
    return inner_args, invocation["run_id"], invocation


def run_internal_controller(args: argparse.Namespace) -> int:
    inner_args, run_id, invocation = _validate_internal_invocation(args)
    assert_fixed_profile()
    try:
        return run_authoritative_inner(inner_args, run_id, invocation)
    except BaseException as error:
        if not (inner_args.output / "COMPLETE.json").exists() and not (
            inner_args.output / "FAILED.json"
        ).exists():
            seal_failed(
                inner_args.output,
                error,
                run_id=run_id,
                expected_receipt_id=inner_args.expected_receipt_id,
                verified_receipt_id=_verified_receipt_id_from_output(
                    inner_args.output
                ),
            )
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal_authoritative:
        return run_internal_controller(args)
    _normal_required_arguments(args)
    args.output = pathlib.Path(os.path.abspath(args.output))
    args.build_receipt = args.build_receipt.resolve()
    args.mpiexec = args.mpiexec.resolve()
    claim_output_directory(args.output)
    run_id = "multi-gpu-scaling-diagnostic-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    capability = secrets.token_hex(32)
    capability_sha256 = hashlib.sha256(capability.encode()).hexdigest()
    exclusive_publish_json(
        args.output / "RUNNING.json",
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "state": "CLAIMED",
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "outer_pid": os.getpid(),
            "capability_sha256": capability_sha256,
        },
    )
    expected_receipt_id = (
        args.expected_receipt_id
        if isinstance(args.expected_receipt_id, str)
        and re.fullmatch(r"[0-9a-f]{64}", args.expected_receipt_id)
        else None
    )
    try:
        if expected_receipt_id is None:
            raise DiagnosticError(
                "expected receipt ID must be a lowercase SHA-256"
            )
        return run_outer_controller(args, run_id, capability)
    except BaseException as error:
        if not (args.output / "COMPLETE.json").exists() and not (
            args.output / "FAILED.json"
        ).exists():
            try:
                seal_failed(
                    args.output,
                    error,
                    run_id=run_id,
                    expected_receipt_id=expected_receipt_id,
                    verified_receipt_id=_verified_receipt_id_from_output(args.output),
                )
            except BaseException as sealing_error:
                print(
                    f"error: failed to seal diagnostic failure: {sealing_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    try:
        exit_status = main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(exit_status)
