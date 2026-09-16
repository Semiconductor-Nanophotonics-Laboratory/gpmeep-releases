#!/usr/bin/env python3
"""Shared, fail-closed helpers for the user-supplied workload gates.

The original inputs live outside the gpmeep repository.  Release evidence is
valid only for the exact byte streams named below; adapters must never quietly
run a similarly named, edited copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
import sys
import tarfile
import tempfile
from contextlib import AbstractContextManager
from typing import Any

_SCRIPTS_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIRECTORY))
from gpmeep_release_manifest import (  # noqa: E402
    DISTRIBUTION_VERSION as _DISTRIBUTION_VERSION,
    RELEASE_MANIFEST_FIELDS as _RELEASE_MANIFEST_FIELDS,
    RELEASE_MANIFEST_SCHEMA_VERSION as _RELEASE_MANIFEST_SCHEMA_VERSION,
    SEALED_M3_ANCHOR as _SEALED_M3_ANCHOR,
    ReleaseManifestError,
    validate_release_manifest,
)

TERS_SHA256 = "4a14fb7a6770802aeefa116768dd28ea3d92900f36fb06e1f01b9de5706f8bef"
TERS_PERFORMANCE_MEASUREMENT_MODE = "fixed-meep-time-performance-window"
TERS_PERFORMANCE_ADAPTATION = (
    "each original decay stop replaced by a fixed Meep-time interval"
)
# The sealed TERS monitor uses Meep's automatic DFT decimation. GaussianSource
# converts the input fwidth to width=1/fwidth; its fixed
# sqrt(-2*ln(1e-7))/(pi*width) effective bandwidth yields a decimation factor
# of 2,902 at resolution 7000. A 0.05-time (700-step) sample can therefore
# exercise curl/update work while producing no DFT work. A 0.25-time window
# contains 3,500 timesteps and crosses the first decimated update with a
# deterministic 598-step margin on both precision builds.
MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME = 0.25
MINIMUM_TERS_PERFORMANCE_TIMESTEPS_PER_PHASE = 3500
AUNP_ARCHIVE_SHA256 = "448e01dcf433435fc5e00ef40ef0cb260870b002a113f3340eacee84f205ff96"
# User-selected AuNP qualification profile.  The archive remains byte-exact;
# adapters apply these explicit physical stop-condition overrides in memory.
# Keeping the profile here gives exact, performance, comparison, controller,
# and replay code one fail-closed definition.
AUNP_ENGINE_POLARIZATIONS = ("physical_ex",)
AUNP_OUTPUT_POLARIZATIONS = ("TM_Ex",)
AUNP_DFT_MINIMUM_RUN_TIME = 20.0
AUNP_DFT_DECAY_TOLERANCE = 5e-8
GPMEEP_DISTRIBUTION_VERSION = _DISTRIBUTION_VERSION
GPMEEP_RELEASE_MANIFEST_SCHEMA_VERSION = _RELEASE_MANIFEST_SCHEMA_VERSION
GPMEEP_SEALED_M3_ANCHOR = _SEALED_M3_ANCHOR
GPMEEP_RELEASE_MANIFEST_FIELDS = _RELEASE_MANIFEST_FIELDS
AUNP_PHYSICAL_OVERRIDES = {
    "dft_minimum_run_time": AUNP_DFT_MINIMUM_RUN_TIME,
    "dft_decay_tolerance": AUNP_DFT_DECAY_TOLERANCE,
}


def typed_json_equal(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON while treating JSON booleans and numbers distinctly."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            typed_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            typed_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WorkloadError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def reject_json_constant(value: str) -> None:
    raise WorkloadError(f"non-finite JSON constant {value!r}")


def aunp_qualification_profile() -> dict[str, Any]:
    """Return a fresh JSON-safe copy of the user-selected AuNP profile."""

    return {
        "mode": "TM-only",
        "engine_polarizations": list(AUNP_ENGINE_POLARIZATIONS),
        "output_polarizations": list(AUNP_OUTPUT_POLARIZATIONS),
        "dft_minimum_run_time": AUNP_DFT_MINIMUM_RUN_TIME,
        "dft_decay_tolerance": AUNP_DFT_DECAY_TOLERANCE,
        "origin": "user-requested",
    }


AUNP_MEMBER_SHA256 = {
    "aunp_r4000_repro/REPRODUCE.md": "39439d2cbdd24528952abb83a59ab5eb58e9b11dedad203c0a04912cd4da849b",
    "aunp_r4000_repro/environment_meep-1.34-mpi.yml": "b8b7929154db79888926781af8cd07bc73a8d88fcbf9a323fd2e8de00f6b3c57",
    "aunp_r4000_repro/expected/A_resonance_fit.json": "e641b9d6063bc8e33e76851e1ddcd9dbec6bba143d24ff0c53551033239aeaf9",
    "aunp_r4000_repro/expected/TE_Ey_spectra.npz": "a9b18307745229d3d2222265951653e70642c0ee4c5c5cfc7b1a00fbeb1447b8",
    "aunp_r4000_repro/expected/TM_Ex_spectra.npz": "63b0d08e5fe1e69a12ff6bacfced29ab1d42272d18570938dc0f7126b6a17700",
    "aunp_r4000_repro/fdtd/sim_config.py": "f514896f51a5368ffe3f79d389425f4a271fc79bd1d8396278b6f8468c85b0a1",
    "aunp_r4000_repro/fdtd/meep/run_production.py": "7ff087b40a6b4e96f2a2c72fdfa0536a666dd49999096fd60816b77a3988146a",
    "aunp_r4000_repro/fdtd/meep/aunp_periodic_fdtd.py": "a56ade815e2689f037da547ca077fc41505c761364c27e58382b13da5f349cec",
}

# Exact serializable AppConfig() from the byte-bound AuNP package.  The
# engine omits run_mode from its manifest.  The byte-exact package defaults
# are retained here even though the qualification profile explicitly changes
# its two DFT stop-condition fields.
AUNP_EXPECTED_DEFAULT_CONFIG = {
    "period": 0.300,
    "particle_width": 0.050,
    "particle_height": 0.030,
    "particle_corner_radius": 0.002,
    "gap_thickness": 0.002,
    "gap_material_model": "air",
    "gap_epsilon": 1.0,
    "output_dir": "outputs",
    "simulation_name": "baseline_r4000",
    "required_mpi_ranks": 64,
    "reference_memory_gib": 16.0,
    "structure_memory_gib": 96.0,
    "postprocess_memory_gib": 32.0,
    "memory_poll_seconds": 30.0,
    "resolution": 4000,
    "geometry_resolution": 1000,
    "corner_arc_points": 33,
    "substrate_above_pml": 0.200,
    "particle_to_monitor1": 0.200,
    "monitor1_to_source": 0.100,
    "source_to_monitor2": 0.050,
    "monitor2_to_pml": 0.050,
    "pml_thickness_top": 0.500,
    "pml_thickness_bottom": 0.250,
    "wavelength_min_nm": 400,
    "wavelength_max_nm": 1000,
    "spectrum_step_nm": 2,
    "field_step_nm": 25,
    "dense_probe_step_nm": 2,
    "dense_line_offset": 0.001,
    "bottom_flux_above_lower_pml": 0.025,
    "dft_decay_tolerance": 1e-5,
    "dft_minimum_run_time": 150.0,
    "field_padding_bottom": 0.100,
    "field_padding_top": 0.150,
}


# Public ``mp.gpu.statistics()`` counters for every FDTD phase exercised by
# the sealed workloads.  Keep the mapping here so the adapters and independent
# verifier use one exact, auditable contract rather than a lossy sum of calls.
FDTD_PHASE_COUNTERS = {
    "curl": ("dispatch", "curl"),
    "update_eh": ("field_updates", "update_eh"),
    "polarization": ("polarizations", "polarization"),
    "source": ("sources", "source"),
    "boundary": ("boundaries", "boundary"),
    "dft": ("dfts", "dft"),
}
MULTI_GPU_COUNTERS = (
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_staging_bytes",
    "pinned_device_to_host_bytes",
    "pinned_host_to_device_bytes",
)
MPI_COMPLETION_COUNTERS = ("mpi_waitsome_executions", "mpi_waitall_executions")

MAX_ARCHIVE_MEMBERS = 64
MAX_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024


class WorkloadError(RuntimeError):
    """Raised when an input or retained workload artifact is not trustworthy."""


def validate_dft_coordinate_metadata(values: dict[str, Any], isfinite: Any) -> None:
    """Validate Meep coordinate axes and their cubature-weight array.

    ``get_array_metadata`` returns three one-dimensional coordinate axes and a
    weight array whose shape contains one extent for every non-collapsed axis.
    Thus ``w`` is normally two-dimensional for this 2D workload; it is not a
    fourth coordinate axis.  NumPy's ``isfinite`` is injected by callers so
    this control module remains usable with the stdlib-only Python verifier.
    """

    if not isinstance(values, dict) or set(values) != {"x", "y", "z", "w"}:
        raise WorkloadError("DFT coordinate metadata inventory is not exact")
    axes = []
    for name in ("x", "y", "z"):
        value = values[name]
        if (
            getattr(value, "ndim", None) != 1
            or getattr(value, "size", 0) == 0
            or not bool(isfinite(value).all())
        ):
            raise WorkloadError(f"DFT coordinate axis {name} is invalid")
        axes.append(value)
    weights = values["w"]
    expected_shape = tuple(int(axis.size) for axis in axes if axis.size > 1)
    if (
        tuple(getattr(weights, "shape", ())) != expected_shape
        or getattr(weights, "size", 0) == 0
        or not bool(isfinite(weights).all())
    ):
        raise WorkloadError(
            "DFT cubature weights are invalid: "
            f"expected shape {expected_shape}, got {getattr(weights, 'shape', None)}"
        )


class StableFile(AbstractContextManager["StableFile"]):
    """An O_NOFOLLOW, inode-bound file kept stable through scientific reads."""

    def __init__(
        self,
        path: pathlib.Path,
        label: str,
        expected: dict[str, Any] | None = None,
    ) -> None:
        self.path = pathlib.Path(os.path.abspath(os.fspath(path)))
        self.label = label
        self.closed = False
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.fd = os.open(self.path, flags)
        except OSError as exc:
            raise WorkloadError(
                f"could not stably open {label}: {self.path}: {exc}"
            ) from exc
        try:
            self.initial_stat = os.fstat(self.fd)
            if not stat.S_ISREG(self.initial_stat.st_mode):
                raise WorkloadError(f"{label} is not a regular file: {self.path}")
            self.parent_stat = self.path.parent.stat()
            self.sha256 = self._digest()
            if expected is not None:
                if self.initial_stat.st_size != int(expected.get("size_bytes", -1)):
                    raise WorkloadError(f"{label} size mismatch: {self.path}")
                if self.sha256 != expected.get("sha256"):
                    raise WorkloadError(f"{label} digest mismatch: {self.path}")
            self._verify_name()
        except Exception:
            os.close(self.fd)
            raise

    def _digest(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(self.fd, 1024 * 1024, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    def _verify_name(self) -> None:
        try:
            named_stat = self.path.lstat()
        except OSError as exc:
            raise WorkloadError(
                f"{self.label} name changed: {self.path}: {exc}"
            ) from exc
        if stat.S_ISLNK(named_stat.st_mode) or (
            named_stat.st_dev,
            named_stat.st_ino,
        ) != (self.initial_stat.st_dev, self.initial_stat.st_ino):
            raise WorkloadError(f"{self.label} name was replaced: {self.path}")

    @property
    def proc_path(self) -> pathlib.Path:
        return pathlib.Path(f"/proc/self/fd/{self.fd}")

    def file_object(self):
        return os.fdopen(os.dup(self.fd), "rb")

    def verify_unchanged(self) -> None:
        if self.closed:
            raise WorkloadError(f"{self.label} stable handle is already closed")
        current = os.fstat(self.fd)
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        initial = (
            self.initial_stat.st_dev,
            self.initial_stat.st_ino,
            self.initial_stat.st_size,
            self.initial_stat.st_mtime_ns,
            self.initial_stat.st_ctime_ns,
        )
        if identity != initial or self._digest() != self.sha256:
            raise WorkloadError(f"{self.label} changed while it was being read")
        parent_current = self.path.parent.stat()
        parent_identity = (
            parent_current.st_dev,
            parent_current.st_ino,
            parent_current.st_mtime_ns,
            parent_current.st_ctime_ns,
        )
        parent_initial = (
            self.parent_stat.st_dev,
            self.parent_stat.st_ino,
            self.parent_stat.st_mtime_ns,
            self.parent_stat.st_ctime_ns,
        )
        if parent_identity != parent_initial:
            raise WorkloadError(
                f"{self.label} parent directory changed while it was being read"
            )
        self._verify_name()

    def close(self, verify: bool = True) -> None:
        if self.closed:
            return
        try:
            if verify:
                self.verify_unchanged()
        finally:
            os.close(self.fd)
            self.closed = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(verify=exc_type is None)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def physical_core_affinity() -> list[dict[str, Any]]:
    cores: set[tuple[str, int, int]] = set()
    hostname = os.uname().nodename
    try:
        logical_cpus = os.sched_getaffinity(0)
    except AttributeError as exc:
        raise WorkloadError("physical CPU affinity is unavailable") from exc
    for logical_cpu in logical_cpus:
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{logical_cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError) as exc:
            raise WorkloadError(
                f"could not resolve physical topology for CPU {logical_cpu}: {exc}"
            ) from exc
        cores.add((hostname, package, core))
    return [
        {"hostname": host, "package_id": package, "core_id": core}
        for host, package, core in sorted(cores)
    ]


def regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    # Do not call resolve() before lstat(): that would follow a terminal
    # symlink and defeat the advertised non-symlink check.  abspath-style
    # normalization keeps the directory spelling canonical without following
    # the file itself.
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WorkloadError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise WorkloadError(f"{label} must be a regular non-symlink file: {path}")
    return path


def verify_file(path: pathlib.Path, expected_sha256: str, label: str) -> pathlib.Path:
    path = regular_file(path, label)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise WorkloadError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _safe_member_path(name: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise WorkloadError(f"unsafe archive member path: {name!r}")
    if "\\" in name or name.startswith("/"):
        raise WorkloadError(f"non-portable archive member path: {name!r}")
    return path


def extract_verified_aunp(
    archive: pathlib.Path, destination: pathlib.Path
) -> pathlib.Path:
    """Extract the exact AuNP package without trusting tar metadata or links."""

    archive = verify_file(archive, AUNP_ARCHIVE_SHA256, "AuNP archive")
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise WorkloadError(f"AuNP extraction directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    extracted_files: set[str] = set()
    total_size = 0
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise WorkloadError("AuNP archive contains too many members")
        for member in members:
            relative = _safe_member_path(member.name)
            if member.name in seen:
                raise WorkloadError(f"duplicate AuNP archive member: {member.name}")
            seen.add(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise WorkloadError(
                    f"AuNP archive contains a link or special member: {member.name}"
                )
            extracted_files.add(member.name)
            if member.size < 0 or member.size > MAX_ARCHIVE_FILE_BYTES:
                raise WorkloadError(
                    f"AuNP archive member has invalid size: {member.name}"
                )
            total_size += member.size
            if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                raise WorkloadError("AuNP archive uncompressed payload is too large")
            source = bundle.extractfile(member)
            if source is None:
                raise WorkloadError(
                    f"could not read AuNP archive member: {member.name}"
                )
            payload = source.read(MAX_ARCHIVE_FILE_BYTES + 1)
            if len(payload) != member.size:
                raise WorkloadError(f"AuNP archive member size changed: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    expected_members = set(AUNP_MEMBER_SHA256)
    if extracted_files != expected_members:
        raise WorkloadError(
            "AuNP archive file inventory mismatch: missing="
            f"{sorted(expected_members - extracted_files)}, extra="
            f"{sorted(extracted_files - expected_members)}"
        )
    for relative, expected in AUNP_MEMBER_SHA256.items():
        verify_file(destination / relative, expected, f"AuNP member {relative}")
    return destination / "aunp_r4000_repro"


def _counter_value(group: Any, key: str, label: str) -> int:
    if not isinstance(group, dict):
        raise WorkloadError(f"statistics group is missing for {label}")
    value = group.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadError(f"invalid statistics counter {label}.{key}")
    return value


def statistics_delta(before: Any, after: Any, label: str = "statistics") -> Any:
    """Return an exact non-negative recursive counter delta.

    The complete before/after snapshots are retained by each adapter.  Requiring
    identical schemas prevents a newly added counter from silently escaping the
    release evidence contract.
    """

    if not isinstance(before, dict) or not isinstance(after, dict):
        raise WorkloadError(f"{label} snapshots are not objects")
    if set(before) != set(after):
        raise WorkloadError(f"{label} counter schema changed during Simulation.run")
    result: dict[str, Any] = {}
    for key in sorted(before):
        left = before[key]
        right = after[key]
        child_label = f"{label}.{key}"
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise WorkloadError(f"{child_label} counter type changed")
            result[key] = statistics_delta(left, right, child_label)
            continue
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left < 0
            or right < left
        ):
            raise WorkloadError(f"invalid or decreasing counter {child_label}")
        result[key] = right - left
    return result


def phase_counter_view(statistics: Any) -> dict[str, dict[str, dict[str, int]]]:
    """Extract calls and points for every declared CPU/CUDA FDTD phase."""

    if not isinstance(statistics, dict):
        raise WorkloadError("GPU statistics are not an object")
    view = {"cpu": {}, "cuda": {}}
    for phase, (group_name, stem) in FDTD_PHASE_COUNTERS.items():
        group = statistics.get(group_name)
        for backend in view:
            view[backend][phase] = {
                field: _counter_value(
                    group,
                    f"{backend}_{stem}_{field}",
                    f"{group_name}.{phase}",
                )
                for field in ("calls", "points")
            }
    return view


def phase_call_totals(statistics: Any) -> dict[str, int]:
    """Compatibility summary derived only from the explicit FDTD phases."""

    view = phase_counter_view(statistics)
    return {
        backend: sum(values["calls"] for values in view[backend].values())
        for backend in view
    }


def validate_phase_contract(
    rank_records: list[dict[str, Any]],
    expected_backend: str,
    phase_contract: list[dict[str, Any]],
) -> None:
    """Validate complete per-phase and multi-GPU evidence for a lane.

    Sparse material/source/boundary/DFT ownership is expressed as aggregate
    requirements; curl/update and local-domain work remain rank-local gates.
    """

    if expected_backend not in ("cpu", "cuda"):
        raise WorkloadError("invalid expected backend for phase contract")
    if not isinstance(rank_records, list) or not rank_records:
        raise WorkloadError("phase contract has no rank records")
    mpi_size = len(rank_records)
    if sorted(int(item.get("rank", -1)) for item in rank_records) != list(
        range(mpi_size)
    ):
        raise WorkloadError("phase contract rank inventory is not canonical")
    if not isinstance(phase_contract, list) or not phase_contract:
        raise WorkloadError("phase contract is absent")
    other_backend = "cuda" if expected_backend == "cpu" else "cpu"
    all_phases = set(FDTD_PHASE_COUNTERS)

    for run_index, contract in enumerate(phase_contract):
        if not isinstance(contract, dict) or set(contract) != {
            "stage",
            "required_per_rank",
            "required_aggregate",
            "forbidden",
        }:
            raise WorkloadError(f"phase contract run {run_index} has an invalid schema")
        if not isinstance(contract["stage"], str) or not contract["stage"]:
            raise WorkloadError(f"phase contract run {run_index} has no stage name")
        sets = []
        for name in ("required_per_rank", "required_aggregate", "forbidden"):
            values = contract[name]
            if (
                not isinstance(values, list)
                or len(values) != len(set(values))
                or any(value not in all_phases for value in values)
            ):
                raise WorkloadError(f"phase contract run {run_index} {name} is invalid")
            sets.append(set(values))
        required_per_rank, required_aggregate, forbidden = sets
        if (
            required_per_rank & required_aggregate
            or required_per_rank & forbidden
            or required_aggregate & forbidden
            or required_per_rank | required_aggregate | forbidden != all_phases
        ):
            raise WorkloadError(f"phase contract run {run_index} is not exhaustive")

        aggregate = {
            backend: {phase: {"calls": 0, "points": 0} for phase in FDTD_PHASE_COUNTERS}
            for backend in ("cpu", "cuda")
        }
        for rank_record in rank_records:
            records = rank_record.get("records")
            if not isinstance(records, list) or len(records) != len(phase_contract):
                raise WorkloadError("rank record count differs from phase contract")
            record = records[run_index]
            before = record.get("statistics_before")
            after = record.get("statistics_after")
            published_delta = record.get("statistics_delta")
            derived_delta = statistics_delta(before, after)
            if published_delta != derived_delta:
                raise WorkloadError(
                    f"rank {rank_record['rank']} run {run_index} statistics delta was not derived"
                )
            view = phase_counter_view(derived_delta)
            if record.get("phase_counters") != view:
                raise WorkloadError(
                    f"rank {rank_record['rank']} run {run_index} phase view was not derived"
                )
            totals = {
                backend: sum(value["calls"] for value in view[backend].values())
                for backend in view
            }
            if record.get("phase_calls") != totals:
                raise WorkloadError(
                    f"rank {rank_record['rank']} run {run_index} phase totals were not derived"
                )
            for phase in all_phases:
                expected = view[expected_backend][phase]
                fallback = view[other_backend][phase]
                if fallback != {"calls": 0, "points": 0}:
                    raise WorkloadError(
                        f"rank {rank_record['rank']} run {run_index} used {other_backend} "
                        f"fallback for {phase}"
                    )
                if (expected["calls"] == 0) != (expected["points"] == 0):
                    raise WorkloadError(
                        f"rank {rank_record['rank']} run {run_index} {phase} calls/points disagree"
                    )
                if phase in required_per_rank and (
                    expected["calls"] <= 0 or expected["points"] <= 0
                ):
                    raise WorkloadError(
                        f"rank {rank_record['rank']} run {run_index} lacks "
                        f"{expected_backend} {phase} work"
                    )
                if phase in forbidden and expected != {"calls": 0, "points": 0}:
                    raise WorkloadError(
                        f"rank {rank_record['rank']} run {run_index} unexpectedly used {phase}"
                    )
                for backend in aggregate:
                    for field in ("calls", "points"):
                        aggregate[backend][phase][field] += view[backend][phase][field]

            multi_gpu = derived_delta.get("multi_gpu")
            completion = derived_delta.get("mpi_completion")
            traffic = {
                name: _counter_value(multi_gpu, name, "multi_gpu")
                for name in MULTI_GPU_COUNTERS
            }
            completions = {
                name: _counter_value(completion, name, "mpi_completion")
                for name in MPI_COMPLETION_COUNTERS
            }
            if expected_backend == "cuda" and mpi_size == 2:
                if (
                    traffic["mpi_messages"] <= 0
                    or traffic["mpi_scalars"] <= 0
                    or traffic["cuda_aware_bytes"] != 0
                    or traffic["pinned_staging_bytes"] <= 0
                    or traffic["pinned_device_to_host_bytes"] <= 0
                    or traffic["pinned_host_to_device_bytes"] <= 0
                    or traffic["pinned_device_to_host_bytes"]
                    + traffic["pinned_host_to_device_bytes"]
                    != traffic["pinned_staging_bytes"]
                    or traffic["pinned_staging_bytes"] != traffic["mpi_scalars"] * 4
                    or sum(completions.values()) <= 0
                ):
                    raise WorkloadError(
                        f"rank {rank_record['rank']} run {run_index} has no exact pinned "
                        "two-GPU MPI traffic/completion evidence"
                    )
            elif any(traffic.values()) or any(completions.values()):
                # These counters describe the CUDA-resident halo transport only.
                # CPU lanes and a singleton CUDA lane must never advance them.
                raise WorkloadError(
                    f"rank {rank_record['rank']} run {run_index} has unexpected multi-GPU traffic"
                )

        for phase in required_aggregate:
            values = aggregate[expected_backend][phase]
            if values["calls"] <= 0 or values["points"] <= 0:
                raise WorkloadError(
                    f"run {run_index} lacks aggregate {expected_backend} {phase} work"
                )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def atomic_write_text(path: pathlib.Path, payload: str) -> None:
    """Atomically publish text and make both file and directory entry durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    payload = (
        json.dumps(json_safe(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    atomic_write_text(path, payload)


def file_record(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    path = regular_file(path, "retained workload artifact")
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise WorkloadError(f"artifact escapes workload directory: {path}") from exc
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def absolute_file_record(path: pathlib.Path, label: str) -> dict[str, Any]:
    path = regular_file(path, label)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def mapped_libmeep_record() -> dict[str, Any]:
    paths: set[pathlib.Path] = set()
    maps = regular_file(pathlib.Path("/proc/self/maps"), "process memory map")
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = pathlib.Path(columns[5])
            if path.name.startswith("libmeep.so") and path.is_file():
                paths.add(path.resolve(strict=True))
    if len(paths) != 1:
        raise WorkloadError(
            "expected exactly one mapped libmeep, found "
            + ", ".join(str(path) for path in sorted(paths))
        )
    return absolute_file_record(next(iter(paths)), "mapped libmeep")


def runtime_artifact_records(mp: Any) -> dict[str, dict[str, Any]]:
    try:
        import meep._meep as extension
    except ImportError as exc:
        raise WorkloadError("could not locate the loaded Meep extension") from exc
    # Extension modules and shared libraries are normally imported through
    # ABI symlinks.  Record the canonical loaded inode; the receipt binds both
    # build-tree and installed real-file targets.
    extension_path = pathlib.Path(extension.__file__).resolve(strict=True)
    return {
        "python": absolute_file_record(
            pathlib.Path(sys.executable), "Python executable"
        ),
        "extension": absolute_file_record(extension_path, "loaded Meep extension"),
        "libmeep": mapped_libmeep_record(),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _verified_absolute_record(
    record: Any,
    label: str,
    *,
    prefix: pathlib.Path | None = None,
) -> pathlib.Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise WorkloadError(f"installed package {label} record schema differs")
    raw = record.get("path")
    if not isinstance(raw, str) or not pathlib.Path(raw).is_absolute():
        raise WorkloadError(f"installed package {label} path is not absolute")
    path = regular_file(pathlib.Path(raw), f"installed package {label}")
    if path.resolve(strict=True) != path:
        raise WorkloadError(f"installed package {label} path is not canonical")
    if prefix is not None:
        try:
            path.relative_to(prefix)
        except ValueError as exc:
            raise WorkloadError(
                f"installed package {label} escaped its prefix"
            ) from exc
    with StableFile(path, f"installed package {label}", expected=record):
        pass
    return path


def validate_package_provenance_attestation(
    attestation_path: pathlib.Path,
    expected_source_commit: str,
    expected_package_sha256: str,
) -> dict[str, Any]:
    """Replay a fresh installed-package provenance report fail closed."""

    if not _lower_hex(expected_source_commit, 40):
        raise WorkloadError("expected installed package source commit is invalid")
    if not _lower_hex(expected_package_sha256, 64):
        raise WorkloadError("expected installed package SHA-256 is invalid")
    attestation_path = regular_file(
        attestation_path, "installed package provenance attestation"
    )
    if attestation_path.resolve(strict=True) != attestation_path:
        raise WorkloadError(
            "installed package provenance attestation path is not canonical"
        )
    with StableFile(
        attestation_path, "installed package provenance attestation"
    ) as stable:
        if stable.initial_stat.st_size > 4 * 1024 * 1024:
            raise WorkloadError("installed package provenance attestation is too large")
        with stable.file_object() as handle:
            try:
                report = json.load(
                    handle,
                    object_pairs_hook=unique_json_object,
                    parse_constant=reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise WorkloadError(
                    "installed package provenance attestation is invalid JSON"
                ) from exc
    required = {
        "schema_version",
        "state",
        "prefix",
        "conda_record",
        "package_url",
        "package_sha256",
        "ownership",
        "manifest",
        "runtime",
        "report_id",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise WorkloadError("installed package provenance schema differs")
    if report.get("schema_version") != 1 or report.get("state") != "PASS":
        raise WorkloadError("installed package provenance is not PASS")
    unsigned = dict(report)
    report_id = unsigned.pop("report_id", None)
    if not _lower_hex(report_id, 64) or report_id != _canonical_sha256(unsigned):
        raise WorkloadError("installed package provenance report ID differs")
    if report.get("package_sha256") != expected_package_sha256:
        raise WorkloadError("installed package provenance package SHA-256 differs")
    raw_prefix = report.get("prefix")
    if not isinstance(raw_prefix, str) or not pathlib.Path(raw_prefix).is_absolute():
        raise WorkloadError("installed package provenance prefix is invalid")
    prefix = pathlib.Path(raw_prefix)
    if prefix.resolve(strict=True) != prefix or not prefix.is_dir():
        raise WorkloadError("installed package provenance prefix is not canonical")

    conda_record_path = _verified_absolute_record(
        report.get("conda_record"), "Conda record", prefix=prefix
    )
    try:
        with StableFile(
            conda_record_path,
            "installed package Conda record",
            expected=report["conda_record"],
        ) as stable:
            with stable.file_object() as handle:
                conda_record = json.load(
                    handle,
                    object_pairs_hook=unique_json_object,
                    parse_constant=reject_json_constant,
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("installed package Conda record is unreadable") from exc
    if (
        conda_record.get("name") != "gpmeep"
        or conda_record.get("version") != GPMEEP_DISTRIBUTION_VERSION
        or conda_record.get("sha256") != expected_package_sha256
    ):
        raise WorkloadError("installed package Conda identity differs")

    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("source_commit") != (
        expected_source_commit
    ):
        raise WorkloadError("installed package source commit differs")
    manifest_path = _verified_absolute_record(
        manifest.get("file"), "release manifest", prefix=prefix
    )
    try:
        with StableFile(
            manifest_path,
            "installed package release manifest",
            expected=manifest["file"],
        ) as stable:
            with stable.file_object() as handle:
                manifest_value = json.load(
                    handle,
                    object_pairs_hook=unique_json_object,
                    parse_constant=reject_json_constant,
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("installed package release manifest is unreadable") from exc
    if not typed_json_equal(
        manifest_value,
        {key: value for key, value in manifest.items() if key != "file"},
    ):
        raise WorkloadError("installed package release manifest content differs")
    try:
        validate_release_manifest(
            manifest_value, expected_source_commit=expected_source_commit
        )
    except ReleaseManifestError as exc:
        raise WorkloadError(f"installed package release contract differs: {exc}") from exc

    ownership = report.get("ownership")
    if (
        not isinstance(ownership, dict)
        or set(ownership)
        != {
            "owned_paths",
            "regular_files",
            "symlinks",
            "content_hashes_verified",
            "content_sizes_verified",
            "prefix_rewritten_sizes",
        }
        or any(type(value) is not int or value < 0 for value in ownership.values())
        or ownership.get("owned_paths")
        != ownership.get("regular_files", 0) + ownership.get("symlinks", 0)
        or ownership.get("owned_paths") != ownership.get("content_hashes_verified")
        or ownership.get("owned_paths")
        != ownership.get("content_sizes_verified", 0)
        + ownership.get("prefix_rewritten_sizes", 0)
        or ownership["owned_paths"] <= 0
    ):
        raise WorkloadError("installed package ownership attestation differs")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "module",
        "extension",
        "libmeep",
        "meep_version",
        "cuda_compiled",
        "mpi_enabled",
        "single_precision",
    }:
        raise WorkloadError("installed package runtime schema differs")
    if (
        runtime.get("cuda_compiled") is not True
        or runtime.get("mpi_enabled") is not True
        or runtime.get("single_precision") is not True
        or runtime.get("meep_version") != manifest.get("upstream_source_version")
    ):
        raise WorkloadError("installed package runtime contract differs")
    for name in ("python", "module", "extension", "libmeep"):
        _verified_absolute_record(runtime.get(name), name, prefix=prefix)
    return report


def verify_package_attestation_collective(
    comm: Any,
    rank: int,
    attestation_path: pathlib.Path,
    expected_source_commit: str,
    expected_package_sha256: str,
    mp: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every MPI rank's loaded runtime to one installed package report."""

    attestation_state = None
    if rank == 0:
        try:
            report = validate_package_provenance_attestation(
                attestation_path,
                expected_source_commit,
                expected_package_sha256,
            )
            attestation_state = {
                "report": report,
                "record": {
                    **absolute_file_record(
                        attestation_path, "installed package provenance attestation"
                    ),
                    "receipt_id": report["report_id"],
                    "package_sha256": report["package_sha256"],
                    "source_commit": report["manifest"]["source_commit"],
                    "prefix": report["prefix"],
                },
            }
        except Exception as exc:
            attestation_state = {"error": f"{type(exc).__name__}: {exc}"}
    attestation_state = comm.bcast(attestation_state, root=0)
    if "error" in attestation_state:
        raise WorkloadError(attestation_state["error"])
    local_runtime = None
    runtime_error = None
    try:
        local_runtime = runtime_artifact_records(mp)
        recorded = attestation_state["report"]["runtime"]
        for name in ("python", "extension", "libmeep"):
            if local_runtime[name] != recorded[name]:
                raise WorkloadError(
                    f"rank {rank} loaded {name} differs from installed package"
                )
    except Exception as exc:
        runtime_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(runtime_error)
    if any(error is not None for error in errors):
        raise WorkloadError(
            "runtime/package attestation mismatch: "
            + "; ".join(error for error in errors if error is not None)
        )
    assert local_runtime is not None
    return attestation_state["record"], local_runtime


def _receipt_path(
    record: dict[str, Any], repo: pathlib.Path, label: str
) -> pathlib.Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise WorkloadError(f"build receipt has no {label} path")
    path = pathlib.Path(raw)
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def validate_runtime_against_receipt(
    runtime: dict[str, dict[str, Any]], receipt: dict[str, Any], repo: pathlib.Path
) -> None:
    if set(runtime) != {"python", "extension", "libmeep"}:
        raise WorkloadError("runtime artifact inventory is not exact")
    python_record = receipt.get("toolchain", {}).get("python")
    if not isinstance(python_record, dict):
        raise WorkloadError("build receipt has no Python artifact")

    def matches(name: str, actual: dict[str, Any], recorded: Any) -> bool:
        return (
            isinstance(recorded, dict)
            and pathlib.Path(actual["path"]).resolve()
            == _receipt_path(recorded, repo, name)
            and actual["size_bytes"] == recorded.get("size_bytes")
            and actual["sha256"] == recorded.get("sha256")
        )

    if not matches("python", runtime["python"], python_record):
        raise WorkloadError("loaded runtime Python does not match build receipt")
    artifacts = receipt.get("artifacts", {})
    coherent_pairs = (
        (artifacts.get("python_extension"), artifacts.get("libmeep")),
        (
            artifacts.get("installed_python_extension"),
            artifacts.get("installed_libmeep"),
        ),
    )
    if not any(
        matches("extension", runtime["extension"], extension)
        and matches("libmeep", runtime["libmeep"], libmeep)
        for extension, libmeep in coherent_pairs
    ):
        raise WorkloadError(
            "loaded extension/libmeep are not one coherent build-tree or installed pair"
        )


def verify_build_receipt_collective(
    comm: Any, rank: int, receipt_path: pathlib.Path, repo: pathlib.Path, mp: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify source-to-binary provenance and each rank's loaded runtime."""

    receipt_state = None
    if rank == 0:
        try:
            scripts_dir = repo / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from gpmeep_provenance import ProvenanceError, verify_build_receipt

            try:
                receipt = verify_build_receipt(receipt_path, repo)
            except ProvenanceError as exc:
                raise WorkloadError(f"invalid build receipt: {exc}") from exc
            receipt_state = {
                "receipt": receipt,
                "record": {
                    **absolute_file_record(receipt_path, "build receipt"),
                    "receipt_id": receipt.get("receipt_id"),
                    "build_input_id": receipt.get("build_input_id"),
                    "artifact_set_id": receipt.get("artifact_set_id"),
                    "source_start": receipt.get("source_start"),
                },
            }
        except Exception as exc:
            receipt_state = {"error": f"{type(exc).__name__}: {exc}"}
    receipt_state = comm.bcast(receipt_state, root=0)
    if "error" in receipt_state:
        raise WorkloadError(receipt_state["error"])
    receipt = receipt_state["receipt"]
    local_runtime = None
    runtime_error = None
    try:
        local_runtime = runtime_artifact_records(mp)
        validate_runtime_against_receipt(local_runtime, receipt, repo)
    except Exception as exc:
        runtime_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(runtime_error)
    if any(error is not None for error in errors):
        raise WorkloadError(
            "runtime/build receipt mismatch: "
            + "; ".join(error for error in errors if error is not None)
        )
    assert local_runtime is not None
    return receipt_state["record"], local_runtime
