#!/usr/bin/env python3
"""Run the sealed CPU-FP64/CPU-FP32/1-GPU/2-GPU user-workload matrix."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import pathlib
import selectors
import shutil
import signal
import statistics
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from common import (  # noqa: E402
    AUNP_ARCHIVE_SHA256,
    StableFile,
    TERS_SHA256,
    WorkloadError,
    atomic_write_json,
    atomic_write_text,
    file_record,
    regular_file,
    verify_file,
)
from gpmeep_provenance import ProvenanceError, verify_build_receipt  # noqa: E402


SCHEMA = "gpmeep-user-workload-matrix-v2"
COMPLETE_SCHEMA = "gpmeep-user-workload-matrix-complete-v2"
CHECKPOINT_SCHEMA = "gpmeep-user-workload-matrix-checkpoint-v1"
# The byte-exact TERS input has a 5.435-Meep-time Gaussian source in each of
# two simulations.  On the qualified eight-core FP64 host, the source interval
# alone is longer than 48 hours.  Keep the per-lane watchdog useful for stale
# jobs without making the exact CPU reference impossible by construction.
DEFAULT_TIMEOUT = {"ters": 7 * 24 * 3600.0, "aunp": 72 * 3600.0}
MAX_REPEATS = 5
RELEASE_REPEATS = 3
RELEASE_SPEED_GATES = {
    "minimum_one_gpu_speedup": 1.5,
    "minimum_two_gpu_speedup": 2.0,
    "minimum_multi_gpu_scaling": 1.1,
}
BUILD_QUALIFICATION_CONTRACTS = {
    "cpu-mpi-python-fp64": "gpmeep-cpu-mpi-python-fp64-v1",
    "cuda-mpi-python-fp32": "gpmeep-cuda-mpi-python-fp32-v2",
}


def _load_comparator():
    path = SCRIPT_DIR / "compare_user_workloads.py"
    spec = importlib.util.spec_from_file_location("gpmeep_user_matrix_comparator", path)
    if spec is None or spec.loader is None:
        raise WorkloadError("could not load the user-workload comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = _load_comparator()


@dataclass(frozen=True)
class Build:
    name: str
    python: pathlib.Path
    mpiexec: pathlib.Path
    receipt_path: pathlib.Path
    receipt: dict[str, Any]
    pythonpath: pathlib.Path
    lib_directory: pathlib.Path
    single_precision: bool


@dataclass(frozen=True)
class Lane:
    label: str
    build: Build
    backend: str
    ranks: int
    devices: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workload", required=True, choices=("ters", "aunp"))
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    for prefix in ("cpu-fp64", "fp32"):
        parser.add_argument(f"--{prefix}-python", required=True, type=pathlib.Path)
        parser.add_argument(f"--{prefix}-mpiexec", required=True, type=pathlib.Path)
        parser.add_argument(f"--{prefix}-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--cpu-ranks", type=int, default=None)
    parser.add_argument(
        "--gpu-devices",
        required=True,
        help="two distinct physical GPU UUIDs, comma separated",
    )
    parser.add_argument("--repeats", type=int, default=RELEASE_REPEATS)
    parser.add_argument(
        "--development-allow-fewer-repeats",
        "--development-allow-one-repeat",
        dest="development_allow_fewer_repeats",
        action="store_true",
        help="allow one or two diagnostic repeats without release COMPLETE",
    )
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--comparison-timeout-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--stdout-limit-mib", type=int, default=128)
    parser.add_argument("--minimum-one-gpu-speedup", type=float, default=1.5)
    parser.add_argument("--minimum-two-gpu-speedup", type=float, default=2.0)
    parser.add_argument("--minimum-multi-gpu-scaling", type=float, default=1.1)
    parser.add_argument("--maximum-timing-cv", type=float, default=0.15)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume an interrupted matrix from its last atomically validated "
            "lane/comparison checkpoint"
        ),
    )
    return parser.parse_args(argv)


def _normalized_uuid(value: str) -> str:
    stripped = value.strip()
    normalized = stripped.lower().removeprefix("gpu-").replace("-", "")
    if len(normalized) != 32 or any(c not in "0123456789abcdef" for c in normalized):
        raise WorkloadError(f"invalid physical GPU UUID: {value!r}")
    return stripped


def physical_core_count() -> int:
    cores = set()
    for logical_cpu in os.sched_getaffinity(0):
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{logical_cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError) as exc:
            raise WorkloadError(f"could not resolve CPU topology: {exc}") from exc
        cores.add((package, core))
    if not cores:
        raise WorkloadError("no physical CPU cores are available")
    return len(cores)


def hardware_snapshot(selected_devices: tuple[str, ...]) -> dict[str, Any]:
    """Capture a fail-closed host/GPU state used to interpret timings."""

    cpuinfo = pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8")
    model_names = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name") and ":" in line
        }
    )
    if len(model_names) != 1:
        raise WorkloadError("CPU model inventory is not exact")
    governors = sorted(
        {
            path.read_text(encoding="utf-8").strip()
            for cpu in os.sched_getaffinity(0)
            if (
                path := pathlib.Path(
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                )
            ).is_file()
        }
    )
    meminfo = {}
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
            name, value, *_unit = line.replace(":", "").split()
            meminfo[name] = int(value) * 1024
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if query.returncode != 0:
        raise WorkloadError(
            "nvidia-smi hardware inventory failed: " + query.stderr.strip()
        )
    gpu_rows = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise WorkloadError("nvidia-smi GPU inventory row is malformed")
        gpu_rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "compute_capability": parts[4],
                "memory_total_mib": int(parts[5]),
            }
        )
    selected_normalized = {
        value.lower().removeprefix("gpu-").replace("-", "") for value in selected_devices
    }
    inventory_normalized = {
        row["uuid"].lower().removeprefix("gpu-").replace("-", "") for row in gpu_rows
    }
    if not selected_normalized <= inventory_normalized:
        raise WorkloadError("selected GPU UUIDs are absent from nvidia-smi inventory")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if processes.returncode != 0:
        raise WorkloadError(
            "nvidia-smi compute-process inventory failed: " + processes.stderr.strip()
        )
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    selected_process_rows = [
        line
        for line in process_rows
        if line.split(",", 1)[0].strip().lower().removeprefix("gpu-").replace("-", "")
        in selected_normalized
    ]
    if selected_process_rows:
        raise WorkloadError(
            "selected GPUs are not exclusively idle at qualification boundary: "
            + "; ".join(selected_process_rows)
        )
    return {
        "captured_unix_seconds": time.time(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "cpu_model": model_names[0],
        "visible_logical_cpus": sorted(os.sched_getaffinity(0)),
        "visible_physical_cores": physical_core_count(),
        "cpu_governors": governors,
        "load_average": list(os.getloadavg()),
        "memory_bytes": meminfo,
        "gpus": gpu_rows,
        "selected_gpu_compute_processes": selected_process_rows,
    }


def validate_hardware_evidence(
    evidence: Any, selected_devices: list[str]
) -> None:
    if not isinstance(evidence, dict) or set(evidence) != {"before", "after"}:
        raise WorkloadError("matrix hardware evidence is absent")
    selected = {
        value.lower().removeprefix("gpu-").replace("-", "") for value in selected_devices
    }
    stable_fields = (
        "hostname",
        "kernel",
        "cpu_model",
        "visible_logical_cpus",
        "visible_physical_cores",
        "cpu_governors",
    )
    for boundary in ("before", "after"):
        snapshot = evidence[boundary]
        if not isinstance(snapshot, dict) or snapshot.get("selected_gpu_compute_processes") != []:
            raise WorkloadError(f"matrix hardware {boundary} boundary is not idle")
        rows = snapshot.get("gpus")
        if not isinstance(rows, list):
            raise WorkloadError("matrix GPU hardware inventory is absent")
        inventory = {
            str(row.get("uuid", "")).lower().removeprefix("gpu-").replace("-", "")
            for row in rows
            if isinstance(row, dict)
        }
        if not selected <= inventory:
            raise WorkloadError("matrix selected GPUs are absent from hardware evidence")
        loads = snapshot.get("load_average")
        if (
            not isinstance(loads, list)
            or len(loads) != 3
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in loads)
        ):
            raise WorkloadError("matrix load-average evidence is invalid")
    if any(evidence["before"].get(name) != evidence["after"].get(name) for name in stable_fields):
        raise WorkloadError("matrix stable hardware identity changed during qualification")
    before_gpus = [
        {key: value for key, value in row.items() if key != "index"}
        for row in evidence["before"]["gpus"]
    ]
    after_gpus = [
        {key: value for key, value in row.items() if key != "index"}
        for row in evidence["after"]["gpus"]
    ]
    if before_gpus != after_gpus:
        raise WorkloadError("matrix GPU model/driver/memory identity changed")


def stable_file_record(
    path: pathlib.Path, root: pathlib.Path, label: str
) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        try:
            relative = stable.path.relative_to(root).as_posix()
        except ValueError as exc:
            raise WorkloadError(f"{label} is outside its evidence root") from exc
        return {
            "path": relative,
            "size_bytes": stable.initial_stat.st_size,
            "sha256": stable.sha256,
        }


def stable_identity_record(stable: StableFile, root: pathlib.Path) -> dict[str, Any]:
    try:
        relative = stable.path.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise WorkloadError(f"{stable.label} is outside its identity root") from exc
    return {
        "path": relative,
        "size_bytes": stable.initial_stat.st_size,
        "sha256": stable.sha256,
    }


def collect_partial_evidence(output: pathlib.Path) -> list[dict[str, Any]]:
    """Seal all useful small closure roots from an interrupted/failed matrix."""

    names = {
        "COMPLETE",
        "summary.json",
        "report.json",
        "report.md",
        "output_manifest.json",
        "simulation_config.json",
        "postprocess_complete.json",
        "FAILED.json",
        "CHECKPOINT.json",
    }
    paths = set()
    for path in output.rglob("*"):
        if path == output / "FAILED.json":
            continue
        if (
            not path.is_symlink()
            and path.is_file()
            and (path.name in names or path.suffix == ".log")
        ):
            paths.add(path)
    return [
        stable_file_record(path, output, f"partial evidence {path.name}")
        for path in sorted(paths)
    ]


def stable_record(
    root: pathlib.Path, record: dict[str, Any], label: str
) -> StableFile:
    raw = record.get("path")
    if not isinstance(raw, str):
        raise WorkloadError(f"{label} record has no path")
    relative = pathlib.PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
        or "\\" in raw
    ):
        raise WorkloadError(f"{label} record has an unsafe path")
    root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise WorkloadError(f"{label} record is unavailable: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkloadError(f"{label} record traverses a symlink")
    return StableFile(current, label, expected=record)


def load_stable_json_record(
    root: pathlib.Path, record: dict[str, Any], label: str
) -> dict[str, Any]:
    with stable_record(root, record, label) as stable:
        with stable.file_object() as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _receipt_path(record: dict[str, Any], label: str) -> pathlib.Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise WorkloadError(f"receipt has no {label} path")
    path = pathlib.Path(raw)
    return (REPO / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def validate_build_contract(
    name: str,
    receipt: dict[str, Any],
    expected_build_kind: str,
    expected_single_precision: bool,
    expected_cuda_capable: bool,
) -> tuple[list[str], bool]:
    if receipt.get("build_kind") != expected_build_kind:
        raise WorkloadError(
            f"{name} build kind is not {expected_build_kind!r}"
        )
    configuration_record = receipt.get("configuration")
    if not isinstance(configuration_record, dict):
        raise WorkloadError(f"{name} receipt has no configuration record")
    configuration = configuration_record.get("configure_argv")
    if not isinstance(configuration, list) or not all(
        isinstance(item, str) for item in configuration
    ):
        raise WorkloadError(f"{name} receipt has no configure arguments")
    expected_contract = BUILD_QUALIFICATION_CONTRACTS.get(expected_build_kind)
    if (
        expected_contract is None
        or configuration_record.get("qualification_contract")
        != expected_contract
    ):
        raise WorkloadError(
            f"{name} receipt has the wrong qualification contract"
        )

    def exact_feature(enable: str, disable: str, expected: bool, label: str) -> bool:
        aliases = [
            item
            for item in configuration
            if item.startswith(enable + "=") or item.startswith(disable + "=")
        ]
        if aliases:
            raise WorkloadError(
                f"{name} receipt contains noncanonical {label} override: {aliases}"
            )
        enable_count = configuration.count(enable)
        disable_count = configuration.count(disable)
        if enable_count + disable_count != 1:
            raise WorkloadError(
                f"{name} receipt does not select exactly one explicit {label} mode"
            )
        actual = enable_count == 1
        if actual is not expected:
            required = "enabled" if expected else "disabled"
            raise WorkloadError(f"{name} requires {label} to be {required}")
        return actual

    single_precision = exact_feature(
        "--enable-single", "--disable-single", expected_single_precision, "FP32"
    )
    exact_feature(
        "--enable-cuda", "--disable-cuda", expected_cuda_capable, "CUDA"
    )
    exact_feature("--with-mpi", "--without-mpi", True, "MPI")
    exact_feature("--with-python", "--without-python", True, "Python")
    exact_feature("--with-scheme", "--without-scheme", False, "Scheme")
    return configuration, single_precision


def load_build(
    name: str,
    python: pathlib.Path,
    mpiexec: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_build_kind: str,
    expected_single_precision: bool,
    expected_cuda_capable: bool,
) -> Build:
    python = python.resolve(strict=True)
    mpiexec = mpiexec.resolve(strict=True)
    receipt_path = regular_file(receipt_path, f"{name} build receipt")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise WorkloadError(f"{name} Python is not executable: {python}")
    if not mpiexec.is_file() or not os.access(mpiexec, os.X_OK):
        raise WorkloadError(f"{name} mpiexec is not executable: {mpiexec}")
    try:
        receipt = verify_build_receipt(receipt_path, REPO)
    except ProvenanceError as exc:
        raise WorkloadError(f"{name} build receipt is invalid: {exc}") from exc
    _configuration, single_precision = validate_build_contract(
        name,
        receipt,
        expected_build_kind,
        expected_single_precision,
        expected_cuda_capable,
    )
    extension = receipt.get("artifacts", {}).get("python_extension")
    libmeep = receipt.get("artifacts", {}).get("libmeep")
    if not isinstance(extension, dict) or not isinstance(libmeep, dict):
        raise WorkloadError(f"{name} receipt has no build-tree Python/libmeep pair")
    extension_path = _receipt_path(extension, f"{name} Python extension")
    libmeep_path = _receipt_path(libmeep, f"{name} libmeep")
    pythonpath = extension_path.parent.parent
    if pythonpath.name != "python" or extension_path.parent.name != "meep":
        raise WorkloadError(f"{name} build-tree Python package layout is unexpected")
    recorded_python = receipt.get("toolchain", {}).get("python")
    if not isinstance(recorded_python, dict):
        raise WorkloadError(f"{name} receipt has no Python toolchain record")
    if python != _receipt_path(recorded_python, f"{name} Python"):
        raise WorkloadError(f"{name} Python does not match its receipt")
    expected_mpiexec = python.parent / "mpiexec"
    if mpiexec != expected_mpiexec.resolve(strict=True):
        raise WorkloadError(f"{name} mpiexec is not from the Python environment")
    return Build(
        name=name,
        python=python,
        mpiexec=mpiexec,
        receipt_path=receipt_path,
        receipt=receipt,
        pythonpath=pythonpath,
        lib_directory=libmeep_path.parent,
        single_precision=single_precision,
    )


def _source_identity(build: Build) -> str:
    value = build.receipt.get("source_start", {}).get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadError(f"{build.name} receipt has no source snapshot identity")
    return value


def validate_shared_fp32_receipts(
    receipt_ids: dict[str, set[str]], labels: tuple[str, str, str, str]
) -> None:
    if (
        set(receipt_ids) != set(labels)
        or any(len(receipt_ids[label]) != 1 for label in labels)
        or len(
            receipt_ids[labels[1]]
            | receipt_ids[labels[2]]
            | receipt_ids[labels[3]]
        )
        != 1
        or receipt_ids[labels[0]] == receipt_ids[labels[1]]
    ):
        raise WorkloadError(
            "matrix CPU/GPU FP32 lanes are not the same exact build receipt"
        )


def clean_environment(lane: Lane, runtime_root: pathlib.Path) -> dict[str, str]:
    prefix = lane.build.python.parent.parent
    runtime_root.mkdir(parents=True, exist_ok=False)
    for directory in ("home", "cache", "tmp", "matplotlib"):
        (runtime_root / directory).mkdir()
    (runtime_root / "cache" / "fontconfig").mkdir()
    # Conda's default fonts.conf contains an absolute cachedir inside the
    # package prefix.  Matplotlib/fontconfig can chmod and populate that cache
    # even when the builder sealed it read-only, invalidating the next lane's
    # build receipt.  Use the same minimal, lock-font-only configuration as the
    # authoritative builder and direct its only cache through XDG_CACHE_HOME.
    fontconfig_file = runtime_root / "fontconfig.conf"
    escaped_prefix = escape(str(prefix))
    atomic_write_text(
        fontconfig_file,
        "\n".join(
            (
                '<?xml version="1.0"?>',
                '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">',
                "<fontconfig>",
                f"  <dir>{escaped_prefix}/fonts</dir>",
                f'  <include ignore_missing="yes">{escaped_prefix}/etc/fonts/conf.d</include>',
                '  <cachedir prefix="xdg">fontconfig</cachedir>',
                "</fontconfig>",
                "",
            )
        ),
    )
    environment = {
        "PATH": f"{lane.build.python.parent}:/usr/bin:/bin",
        # The supplied AuNP workload records the exact package environment in
        # its scientific provenance and therefore requires CONDA_PREFIX.  Do
        # not inherit a caller value: bind it to the same receipt-qualified
        # prefix as Python, mpiexec, and the runtime libraries.
        "CONDA_PREFIX": str(prefix),
        "HOME": str(runtime_root / "home"),
        "XDG_CACHE_HOME": str(runtime_root / "cache"),
        "TMPDIR": str(runtime_root / "tmp"),
        "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
        "MPLBACKEND": "Agg",
        "FONTCONFIG_FILE": str(fontconfig_file),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONPATH": str(lane.build.pythonpath),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "LD_LIBRARY_PATH": f"{lane.build.lib_directory}:{prefix / 'lib'}",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MEEP_GPU_BACKEND": lane.backend,
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
        "MEEP_GPU_MPI_TRANSPORT": "pinned",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(lane.devices),
        "OMPI_MCA_mca_base_component_path": str(prefix / "lib" / "openmpi"),
        "PMIX_MCA_mca_base_component_path": str(prefix / "lib" / "pmix"),
    }
    mca_file = REPO / "environment" / "openmpi-qualification-mca-params.conf"
    if mca_file.is_file():
        for variable in (
            "OMPI_MCA_mca_base_param_files",
            "PMIX_MCA_mca_base_param_files",
            "PRTE_MCA_mca_base_param_files",
        ):
            environment[variable] = str(mca_file)
    return environment


def process_group_rss_bytes(process_group: int) -> int:
    total = 0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
            if int(fields[2]) != process_group:
                continue
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def process_group_alive(process_group: int) -> bool:
    """Return true while a non-zombie member of a process group exists."""

    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
            state = fields[0]
            if int(fields[2]) == process_group and state not in {"Z", "X"}:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    for selected_signal, wait_seconds in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, selected_signal)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=1.0)
            return
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not process_group_alive(process.pid):
                if process.poll() is None:
                    process.wait(timeout=1.0)
                return
            time.sleep(0.05)
    raise WorkloadError(f"could not terminate process group {process.pid}")


def run_bounded(
    command: list[str],
    environment: dict[str, str],
    log_path: pathlib.Path,
    timeout_seconds: float,
    output_limit_bytes: int,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or output_limit_bytes <= 0:
        raise WorkloadError("process limits must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started = time.monotonic()
    peak_rss = 0
    output_bytes = 0
    timed_out = False
    output_limited = False
    interrupted_signal: signal.Signals | None = None
    previous_handlers: dict[signal.Signals, Any] = {}

    def interrupt_controller(selected_signal: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signal.Signals(selected_signal)

    for selected_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[selected_signal] = signal.getsignal(selected_signal)
        signal.signal(selected_signal, interrupt_controller)
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)
        raise
    if process.stdout is None:
        terminate_group(process)
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)
        raise WorkloadError("could not capture workload output")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        with log_path.open("xb") as log:
            while True:
                if interrupted_signal is not None:
                    terminate_group(process)
                    raise WorkloadError(
                        f"controller received {interrupted_signal.name}"
                    )
                elapsed = time.monotonic() - started
                peak_rss = max(peak_rss, process_group_rss_bytes(process.pid))
                if elapsed > timeout_seconds:
                    timed_out = True
                    terminate_group(process)
                events = selector.select(timeout=0.25)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = output_limit_bytes - output_bytes
                    captured = chunk[:remaining]
                    log.write(captured)
                    output_bytes += len(captured)
                    if len(chunk) > remaining:
                        output_limited = True
                        terminate_group(process)
                        break
                if timed_out or output_limited:
                    break
                if process.poll() is not None and not selector.get_map():
                    break
            if interrupted_signal is not None:
                terminate_group(process)
                raise WorkloadError(
                    f"controller received {interrupted_signal.name}"
                )
            if process.poll() is None:
                process.wait(timeout=5.0)
            log.flush()
            os.fsync(log.fileno())
    finally:
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            terminate_group(process)
    ended_wall = time.time()
    return {
        "command": command,
        "command_pid": process.pid,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "output_limited": output_limited,
        "output_bytes": output_bytes,
        "peak_process_group_rss_bytes": peak_rss,
        "started_unix_seconds": started_wall,
        "ended_unix_seconds": ended_wall,
        "wall_seconds": time.monotonic() - started,
        "log": file_record(log_path, log_path.parents[2]),
    }


def lane_command(lane: Lane, workload: str, source: pathlib.Path, output: pathlib.Path) -> list[str]:
    wrapper = SCRIPT_DIR / f"run_{workload}_workload.py"
    input_option = "--source" if workload == "ters" else "--archive"
    return [
        str(lane.build.mpiexec),
        "--bind-to",
        "core",
        "--map-by",
        "core:PE=1",
        "--report-bindings",
        "-n",
        str(lane.ranks),
        str(lane.build.python),
        "-u",
        str(wrapper),
        input_option,
        str(source),
        "--output",
        str(output),
        "--build-receipt",
        str(lane.build.receipt_path),
        "--expected-backend",
        lane.backend,
    ]


def validate_lane_output(lane: Lane, output: pathlib.Path, workload: str) -> dict[str, Any]:
    schema = COMPARATOR.TERS_SCHEMA if workload == "ters" else COMPARATOR.AUNP_SCHEMA
    loaded = COMPARATOR.load_lane(output, schema)
    provenance = COMPARATOR.validate_lane_provenance(loaded)
    summary = loaded["summary"]
    if summary["expected_backend"] != lane.backend:
        raise WorkloadError(f"{lane.label} published the wrong backend")
    if summary["single_precision"] is not lane.build.single_precision:
        raise WorkloadError(f"{lane.label} published the wrong precision")
    if summary["mpi_size"] != lane.ranks:
        raise WorkloadError(f"{lane.label} published the wrong MPI size")
    if lane.backend == "cuda":
        expected = {
            value.lower().removeprefix("gpu-").replace("-", "")
            for value in lane.devices
        }
        if set(provenance["gpu_devices"].values()) != expected:
            raise WorkloadError(f"{lane.label} used unexpected physical GPUs")
    metrics = derive_lane_metrics(summary, workload, lane.label)
    return {"summary": summary, "provenance": provenance, **metrics}


def derive_lane_metrics(
    summary: dict[str, Any], workload: str, label: str
) -> dict[str, Any]:
    if workload not in ("ters", "aunp"):
        raise WorkloadError(f"{label} has an unknown workload")
    rank_records = summary["rank_records"]
    phase_contract = summary.get("phase_contract")
    if not isinstance(phase_contract, list) or not phase_contract:
        raise WorkloadError(f"{label} has no phase contract")
    if any(
        not isinstance(contract, dict)
        or not isinstance(contract.get("stage"), str)
        or not contract["stage"]
        for contract in phase_contract
    ):
        raise WorkloadError(f"{label} has an invalid phase contract")
    run_count = len(phase_contract)
    for rank_record in rank_records:
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != run_count:
            raise WorkloadError(f"{label} rank record count differs from phase contract")
    phase_walls = []
    timestep_contract = []
    for run_index in range(run_count):
        records = [rank["records"][run_index] for rank in rank_records]
        if any(record.get("run_index") != run_index for record in records):
            raise WorkloadError(f"{label} has a non-canonical run record")
        timesteps = {record["timestep_delta"] for record in records}
        meep_times = {float(record["meep_time"]) for record in records}
        if len(timesteps) != 1 or len(meep_times) != 1:
            raise WorkloadError(f"{label} ranks performed different work")
        phase_walls.append(max(float(record["wall_seconds"]) for record in records))
        timestep_contract.append(
            {
                "timestep_delta": next(iter(timesteps)),
                "meep_time": next(iter(meep_times)),
            }
        )
    timing_field = (
        "exact_end_to_end_seconds" if workload == "ters" else "process_wall_seconds"
    )
    workload_end_to_end = max(
        float(rank[timing_field]) for rank in rank_records
    )
    if not math.isfinite(workload_end_to_end) or workload_end_to_end <= 0:
        raise WorkloadError(f"{label} has no positive workload end-to-end time")
    return {
        "phase_wall_seconds": phase_walls,
        "fdtd_wall_seconds": sum(phase_walls),
        "workload_end_to_end_seconds": workload_end_to_end,
        "timestep_contract": timestep_contract,
    }


def comparison_command(
    python: pathlib.Path,
    workload: str,
    output: pathlib.Path,
    reference: pathlib.Path,
    candidates: list[tuple[str, str, pathlib.Path]],
    source: pathlib.Path,
) -> list[str]:
    command = [
        str(python),
        str(SCRIPT_DIR / "compare_user_workloads.py"),
        "--output",
        str(output),
        workload,
        "--reference",
        str(reference),
    ]
    for label, comparison_class, path in candidates:
        command.extend(("--candidate", f"{label}:{comparison_class}:{path}"))
    if workload == "aunp":
        command.extend(("--expected-archive", str(source)))
    return command


def comparison_specs(
    labels: tuple[str, str, str, str], repeat: int
) -> list[tuple[str, str, int, tuple[tuple[str, str], ...], pathlib.PurePosixPath]]:
    specs = [
        (
            "fp64",
            labels[0],
            repeat,
            (
                (labels[1], "cpu-fp64-fp32"),
                (labels[2], "cuda-fp64-fp32"),
                (labels[3], "cuda-fp64-fp32"),
            ),
            pathlib.PurePosixPath("comparisons") / "fp64" / f"repeat-{repeat:02d}",
        ),
        (
            "fp32",
            labels[1],
            repeat,
            (
                (labels[2], "cuda-same-fp32"),
                (labels[3], "cuda-same-fp32"),
            ),
            pathlib.PurePosixPath("comparisons") / "fp32" / f"repeat-{repeat:02d}",
        ),
    ]
    if repeat > 0:
        repeat_classes = (
            "repeat-cpu-fp64",
            "repeat-cpu-fp32",
            "repeat-cuda-fp32",
            "repeat-cuda-fp32",
        )
        for label, comparison_class in zip(labels, repeat_classes):
            specs.append(
                (
                    f"repeatability-{label}",
                    label,
                    0,
                    ((label, comparison_class),),
                    pathlib.PurePosixPath("comparisons")
                    / "repeatability"
                    / label
                    / f"repeat-{repeat:02d}",
                )
            )
    return specs


def conservative_speedup(reference: list[float], candidate: list[float]) -> dict[str, float]:
    if not reference or len(reference) != len(candidate):
        raise WorkloadError("performance samples are incomplete")
    if any(not math.isfinite(value) or value <= 0 for value in reference + candidate):
        raise WorkloadError("performance samples are not positive and finite")
    paired = [left / right for left, right in zip(reference, candidate)]
    return {
        "reference_median_seconds": statistics.median(reference),
        "candidate_median_seconds": statistics.median(candidate),
        "median_speedup": statistics.median(reference) / statistics.median(candidate),
        "conservative_speedup": min(paired),
        "paired_speedups_min": min(paired),
        "paired_speedups_max": max(paired),
    }


def timing_stability(samples: list[float], maximum_cv: float) -> dict[str, Any]:
    if len(samples) < RELEASE_REPEATS:
        raise WorkloadError("release timing stability requires at least three samples")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise WorkloadError("timing samples are not positive and finite")
    mean = statistics.mean(samples)
    cv = statistics.stdev(samples) / mean
    return {
        "mean_seconds": mean,
        "median_seconds": statistics.median(samples),
        "coefficient_of_variation": cv,
        "maximum": maximum_cv,
        "outcome": "PASS" if cv <= maximum_cv else "FAIL",
    }


def classify_outcome(
    release_eligible: bool,
    performance: dict[str, dict[str, Any]],
    timing_variability: dict[str, dict[str, Any]],
) -> str:
    gates_pass = all(
        metric.get("outcome") == "PASS" for metric in performance.values()
    ) and all(
        metric.get("outcome") == "PASS" for metric in timing_variability.values()
    )
    if not gates_pass:
        return "FAIL"
    return "PASS" if release_eligible else "DEVELOPMENT_ONLY"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# gpmeep user-workload matrix",
        "",
        f"Outcome: **{report['outcome']}**",
        "",
        "| Metric | Median | Conservative | Gate | Outcome |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, metric in report["performance"].items():
        lines.append(
            f"| {name} | {metric['median_speedup']:.4f}x | "
            f"{metric['conservative_speedup']:.4f}x | {metric['minimum']:.4f}x | "
            f"**{metric['outcome']}** |"
        )
    lines.extend(
        (
            "",
            "## Diagnostic workload end-to-end speedups",
            "",
            "These metrics are re-derived and retained but are not release gates. TERS "
            "runs the original plot/CSV/NPY work once per private MPI-rank directory, so "
            "different rank counts create presentation/I/O contention unrelated to solver speed.",
            "",
            "| Metric | Median | Conservative |",
            "| --- | ---: | ---: |",
        )
    )
    for name, metric in report["diagnostic_performance"].items():
        lines.append(
            f"| {name} | {metric['median_speedup']:.4f}x | "
            f"{metric['conservative_speedup']:.4f}x |"
        )
    lines.extend(
        (
            "",
            "## Raw lane timings",
            "",
            "| Lane | Repeat | FDTD-only (s) | Workload end-to-end (s) | Adapter process (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for label, runs in report["lanes"].items():
        for run in runs:
            lines.append(
                f"| `{label}` | {run['repeat']} | {run['fdtd_wall_seconds']:.6f} | "
                f"{run['workload_end_to_end_seconds']:.6f} | "
                f"{run['process']['wall_seconds']:.6f} |"
            )
    before = report["hardware"]["before"]
    lines.extend(
        (
            "",
            "## Hardware boundary",
            "",
            f"- CPU: {before['cpu_model']} ({before['visible_physical_cores']} physical cores visible)",
            f"- CPU governors: {', '.join(before['cpu_governors']) or 'unreported'}",
            f"- Initial load average: {', '.join(f'{value:.3f}' for value in before['load_average'])}",
        )
    )
    for gpu in before["gpus"]:
        lines.append(
            f"- GPU {gpu['index']}: {gpu['name']}, {gpu['memory_total_mib']} MiB, "
            f"driver {gpu['driver_version']}, CC {gpu['compute_capability']}, `{gpu['uuid']}`"
        )
    lines.extend(
        (
            "",
            "TERS diagnostic workload end-to-end is the interval around the unmodified original "
            "script and includes its plots, CSV/NPY, backup, and natural simulation lifetime. "
            "AuNP diagnostic workload end-to-end includes only its explicitly reported execution-policy adaptations.",
            "",
            f"Validated resume boundaries used: {len(report['resume_history'])}.",
            "",
            "All lane, comparison, provenance, coverage, and timing records are in `report.json`.",
            "",
        )
    )
    return "\n".join(lines)


def _verify_process_record(
    output: pathlib.Path,
    process: dict[str, Any],
    expected_log: str,
    label: str,
) -> None:
    expected_keys = {
        "command",
        "command_pid",
        "returncode",
        "timed_out",
        "output_limited",
        "output_bytes",
        "peak_process_group_rss_bytes",
        "started_unix_seconds",
        "ended_unix_seconds",
        "wall_seconds",
        "log",
    }
    if not isinstance(process, dict) or set(process) != expected_keys:
        raise WorkloadError(f"{label} process record schema is not exact")
    if (
        process["returncode"] != 0
        or process["timed_out"] is not False
        or process["output_limited"] is not False
        or type(process["command_pid"]) is not int
        or process["command_pid"] <= 0
        or type(process["output_bytes"]) is not int
        or process["output_bytes"] < 0
        or type(process["peak_process_group_rss_bytes"]) is not int
        or process["peak_process_group_rss_bytes"] < 0
        or not isinstance(process["command"], list)
        or not process["command"]
    ):
        raise WorkloadError(f"{label} process did not complete cleanly")
    started = float(process["started_unix_seconds"])
    ended = float(process["ended_unix_seconds"])
    wall = float(process["wall_seconds"])
    if not all(math.isfinite(value) for value in (started, ended, wall)) or (
        ended < started or wall <= 0
    ):
        raise WorkloadError(f"{label} process timing is invalid")
    log_record = process["log"]
    if log_record.get("path") != expected_log:
        raise WorkloadError(f"{label} process log path is not canonical")
    if log_record.get("size_bytes") != process["output_bytes"]:
        raise WorkloadError(f"{label} process output count differs from its log")
    stable_record(output, log_record, f"{label} process log").close()


def verify_matrix_complete(output: pathlib.Path) -> dict[str, Any]:
    """Recompute matrix semantics plus cryptographic/scientific closure."""

    output = output.resolve(strict=True)
    with StableFile(output / "COMPLETE", "matrix COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if set(complete) != {"schema", "report", "markdown", "journal", "outcome"} or (
        complete["schema"] != COMPLETE_SCHEMA or complete["outcome"] != "PASS"
    ):
        raise WorkloadError("matrix COMPLETE is not an exact release PASS")
    report = load_stable_json_record(output, complete["report"], "matrix report")
    stable_record(output, complete["markdown"], "matrix Markdown").close()
    journal = load_stable_json_record(output, complete["journal"], "matrix journal")
    repeats = report.get("repeats")
    if (
        report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or report.get("release_eligible") is not True
        or type(repeats) is not int
        or not RELEASE_REPEATS <= repeats <= MAX_REPEATS
        or report.get("workload") not in ("ters", "aunp")
    ):
        raise WorkloadError("matrix report is not release eligible")
    checkpoint = load_stable_json_record(
        output,
        report.get("recovery_checkpoint", {}),
        "matrix recovery checkpoint",
    )
    if (
        set(checkpoint)
        != {
            "schema",
            "contract",
            "hardware_before",
            "completed_lanes",
            "completed_comparisons",
        }
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or not isinstance(checkpoint.get("contract"), dict)
        or not isinstance(checkpoint.get("completed_lanes"), list)
        or not isinstance(checkpoint.get("completed_comparisons"), list)
    ):
        raise WorkloadError("matrix recovery checkpoint is not exact")
    actual_resume_history = resume_history_records(output)
    if report.get("resume_history") != actual_resume_history:
        raise WorkloadError("matrix resume-history inventory differs")
    for record in actual_resume_history:
        archive = load_stable_json_record(output, record, "matrix resume archive")
        if archive.get("schema") != "gpmeep-user-workload-resume-archive-v1":
            raise WorkloadError("matrix resume archive schema differs")
    evidence_code = report.get("evidence_code")
    if not isinstance(evidence_code, dict) or set(evidence_code) != {
        "controller",
        "comparator",
        "common",
        "ters_adapter",
        "aunp_adapter",
        "aunp_exact_resume",
    }:
        raise WorkloadError("matrix evidence-code inventory is not exact")
    for name, expected_path in (
        ("controller", pathlib.Path(__file__)),
        ("comparator", SCRIPT_DIR / "compare_user_workloads.py"),
        ("common", SCRIPT_DIR / "common.py"),
        ("ters_adapter", SCRIPT_DIR / "run_ters_workload.py"),
        ("aunp_adapter", SCRIPT_DIR / "run_aunp_workload.py"),
        ("aunp_exact_resume", SCRIPT_DIR / "aunp_exact_resume.py"),
    ):
        stable = stable_record(REPO, evidence_code[name], f"matrix {name}")
        if stable.path != expected_path.resolve(strict=True):
            stable.close(verify=False)
            raise WorkloadError(f"matrix {name} path is not canonical")
        stable.close()
    input_record = report.get("input")
    if not isinstance(input_record, dict) or not pathlib.Path(
        str(input_record.get("path", ""))
    ).is_absolute():
        raise WorkloadError("matrix input record is not absolute")
    with StableFile(
        pathlib.Path(input_record["path"]), "matrix input", expected=input_record
    ):
        pass
    expected_input_hash = (
        TERS_SHA256 if report["workload"] == "ters" else AUNP_ARCHIVE_SHA256
    )
    if input_record.get("sha256") != expected_input_hash:
        raise WorkloadError("matrix input is not the exact required workload")
    if (
        journal.get("schema") != "gpmeep-user-workload-matrix-journal-v2"
        or journal.get("input") != input_record
        or journal.get("controller") != evidence_code["controller"]
        or journal.get("comparator") != evidence_code["comparator"]
        or journal.get("common") != evidence_code["common"]
        or journal.get("ters_adapter") != evidence_code["ters_adapter"]
        or journal.get("aunp_adapter") != evidence_code["aunp_adapter"]
        or journal.get("aunp_exact_resume")
        != evidence_code["aunp_exact_resume"]
        or not isinstance(journal.get("events"), list)
    ):
        raise WorkloadError("matrix journal identity is inconsistent")

    cpu_ranks = report.get("cpu_physical_cores")
    if type(cpu_ranks) is not int or cpu_ranks < 2:
        raise WorkloadError("matrix CPU physical-core count is invalid")
    labels = (
        f"cpu-fp64-{cpu_ranks}r",
        f"cpu-fp32-{cpu_ranks}r",
        "cuda-fp32-1g",
        "cuda-fp32-2g",
    )
    roles = {
        labels[0]: ("cpu", False, cpu_ranks),
        labels[1]: ("cpu", True, cpu_ranks),
        labels[2]: ("cuda", True, 1),
        labels[3]: ("cuda", True, 2),
    }
    devices = report.get("gpu_devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise WorkloadError("matrix GPU inventory is not exact")
    normalized_devices = [
        value.lower().removeprefix("gpu-").replace("-", "")
        if isinstance(value, str)
        else ""
        for value in devices
    ]
    if len(set(normalized_devices)) != 2 or any(
        len(value) != 32 or any(character not in "0123456789abcdef" for character in value)
        for value in normalized_devices
    ):
        raise WorkloadError("matrix GPU UUID inventory is invalid")
    validate_hardware_evidence(report.get("hardware"), devices)
    checkpoint_contract_record = checkpoint["contract"]
    if (
        checkpoint_contract_record.get("workload") != report["workload"]
        or checkpoint_contract_record.get("input") != input_record
        or checkpoint_contract_record.get("evidence_code") != evidence_code
        or checkpoint_contract_record.get("cpu_ranks") != cpu_ranks
        or checkpoint_contract_record.get("gpu_devices") != devices
        or checkpoint_contract_record.get("repeats") != repeats
        or checkpoint_contract_record.get("release_eligible") is not True
    ):
        raise WorkloadError("matrix checkpoint/report contract differs")
    lanes = report.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(labels):
        raise WorkloadError("matrix must contain the exact four release lanes")
    expected_schedule = []
    for repeat in range(repeats):
        order = labels if repeat % 2 == 0 else tuple(reversed(labels))
        expected_schedule.extend({"repeat": repeat, "lane": label} for label in order)
    if report.get("schedule") != expected_schedule:
        raise WorkloadError("matrix lane schedule is not the fixed alternating schedule")

    schema = (
        COMPARATOR.TERS_SCHEMA
        if report["workload"] == "ters"
        else COMPARATOR.AUNP_SCHEMA
    )
    lane_paths: dict[tuple[str, int], pathlib.Path] = {}
    fdtd_timings: dict[str, list[float]] = {label: [] for label in labels}
    process_timings: dict[str, list[float]] = {label: [] for label in labels}
    derived_contract = None
    source_snapshots = set()
    receipt_ids: dict[str, set[str]] = {label: set() for label in labels}
    journal_events = journal["events"]
    expected_events: list[dict[str, Any]] = []
    if (
        not journal_events
        or not isinstance(journal_events[0], dict)
        or set(journal_events[0]) != {"state", "unix_seconds"}
        or journal_events[0].get("state") != "controller-started"
        or not math.isfinite(float(journal_events[0].get("unix_seconds", math.nan)))
    ):
        raise WorkloadError("matrix journal has no controller-started event")
    expected_events.append(journal_events[0])
    for repeat in range(repeats):
        order = labels if repeat % 2 == 0 else tuple(reversed(labels))
        for label in order:
            runs = lanes[label]
            if not isinstance(runs, list) or len(runs) != repeats:
                raise WorkloadError(f"matrix lane {label} has the wrong repeat count")
            run = runs[repeat]
            if run.get("repeat") != repeat:
                raise WorkloadError(f"matrix lane {label} repeat index is not canonical")
            relative = pathlib.PurePosixPath("lanes") / label / f"repeat-{repeat:02d}"
            if run.get("output") != relative.as_posix() or run.get("complete", {}).get(
                "path"
            ) != (relative / "COMPLETE").as_posix():
                raise WorkloadError(f"matrix lane {label} output path is not canonical")
            expected_log = f"logs/{label}/repeat-{repeat:02d}.log"
            _verify_process_record(output, run.get("process"), expected_log, label)
            validate_hardware_evidence(run.get("hardware"), devices)
            stable_record(output, run["complete"], f"matrix lane {label} COMPLETE").close()
            lane_path = output.joinpath(*relative.parts)
            lane_paths[(label, repeat)] = lane_path
            lane = COMPARATOR.load_lane(lane_path, schema)
            summary = lane["summary"]
            backend, single_precision, mpi_size = roles[label]
            if (
                summary.get("expected_backend") != backend
                or summary.get("single_precision") is not single_precision
                or summary.get("mpi_size") != mpi_size
            ):
                raise WorkloadError(f"matrix lane {label} has the wrong role")
            provenance = COMPARATOR.validate_lane_provenance(lane)
            published_provenance = json.loads(
                json.dumps(provenance, sort_keys=True, separators=(",", ":"))
            )
            if run.get("provenance") != published_provenance:
                raise WorkloadError(f"matrix lane {label} provenance was not re-derived")
            source_start = provenance.get("source_start", {})
            if not isinstance(source_start, dict):
                raise WorkloadError(f"matrix lane {label} source identity is invalid")
            source_snapshots.add(source_start.get("sha256"))
            receipt_id = provenance.get("receipt_id")
            if not isinstance(receipt_id, str) or len(receipt_id) != 64:
                raise WorkloadError(f"matrix lane {label} receipt identity is invalid")
            receipt_ids[label].add(receipt_id)
            gpu_values = set(provenance.get("gpu_devices", {}).values())
            expected_gpu_values = (
                set()
                if backend == "cpu"
                else set(normalized_devices[:1] if mpi_size == 1 else normalized_devices)
            )
            if gpu_values != expected_gpu_values:
                raise WorkloadError(f"matrix lane {label} used the wrong GPU UUIDs")
            metrics = derive_lane_metrics(summary, report["workload"], label)
            for key in (
                "phase_wall_seconds",
                "fdtd_wall_seconds",
                "workload_end_to_end_seconds",
                "timestep_contract",
            ):
                if run.get(key) != metrics[key]:
                    raise WorkloadError(f"matrix lane {label} {key} was not re-derived")
            if derived_contract is None:
                derived_contract = metrics["timestep_contract"]
            elif metrics["timestep_contract"] != derived_contract:
                raise WorkloadError("matrix lanes performed different FDTD work")
            fdtd_timings[label].append(metrics["fdtd_wall_seconds"])
            process_timings[label].append(float(run["workload_end_to_end_seconds"]))
            if schema == COMPARATOR.TERS_SCHEMA:
                COMPARATOR._ters_files(lane)
            else:
                artifacts = COMPARATOR.load_aunp_output_manifest(lane)
                COMPARATOR.validate_aunp_simulation_config(lane, artifacts)
            expected_events.extend(
                (
                    {"state": "lane-started", "repeat": repeat, "lane": label},
                    {
                        "state": "lane-process-ended",
                        "repeat": repeat,
                        "lane": label,
                        "process": run["process"],
                    },
                    {
                        "state": "lane-validated",
                        "repeat": repeat,
                        "lane": label,
                        "complete": run["complete"],
                    },
                )
            )
    source_snapshot = report.get("source_snapshot_sha256")
    if (
        not isinstance(source_snapshot, str)
        or len(source_snapshot) != 64
        or any(character not in "0123456789abcdef" for character in source_snapshot)
        or source_snapshots != {source_snapshot}
    ) or (
        report.get("workload_contract") != derived_contract
    ):
        raise WorkloadError("matrix source/work contract was not re-derived")
    validate_shared_fp32_receipts(receipt_ids, labels)

    comparisons = report.get("comparisons")
    expected_comparison_count = 2 * repeats + 4 * (repeats - 1)
    if not isinstance(comparisons, list) or len(comparisons) != expected_comparison_count:
        raise WorkloadError("matrix comparison inventory is incomplete")
    comparison_index = 0
    for repeat in range(repeats):
        for (
            precision,
            reference_label,
            reference_repeat,
            candidate_specs,
            relative,
        ) in comparison_specs(labels, repeat):
            comparison = comparisons[comparison_index]
            comparison_index += 1
            if (
                comparison.get("precision") != precision
                or comparison.get("repeat") != repeat
                or comparison.get("reference_repeat") != reference_repeat
                or comparison.get("output") != relative.as_posix()
                or comparison.get("complete", {}).get("path")
                != (relative / "COMPLETE").as_posix()
            ):
                raise WorkloadError("matrix comparison role/path is not canonical")
            expected_log = f"logs/comparisons/{precision}-repeat-{repeat:02d}.log"
            _verify_process_record(
                output, comparison.get("process"), expected_log, f"{precision} comparison"
            )
            stable_record(
                output, comparison["complete"], "matrix comparison COMPLETE"
            ).close()
            comparison_path = output.joinpath(*relative.parts)
            with StableFile(
                comparison_path / "COMPLETE",
                "comparison COMPLETE",
                expected=comparison["complete"],
            ) as stable:
                with stable.file_object() as handle:
                    nested_complete = json.load(handle)
            if (
                nested_complete.get("schema")
                != "gpmeep-user-workload-comparison-complete-v2"
                or nested_complete.get("outcome") != "PASS"
            ):
                raise WorkloadError("nested comparison is not an exact PASS")
            nested_report = load_stable_json_record(
                comparison_path, nested_complete.get("report", {}), "comparison report"
            )
            stable_record(
                comparison_path,
                nested_complete.get("markdown", {}),
                "comparison Markdown",
            ).close()
            candidate_paths = [
                (label, comparison_class, lane_paths[(label, repeat)])
                for label, comparison_class in candidate_specs
            ]
            comparator_args = argparse.Namespace(
                workload=report["workload"],
                reference=lane_paths[(reference_label, reference_repeat)],
                candidate=candidate_paths,
                expected_archive=pathlib.Path(input_record["path"]),
            )
            recomputed = (
                COMPARATOR.compare_ters(comparator_args)
                if report["workload"] == "ters"
                else COMPARATOR.compare_aunp(comparator_args)
            )
            published_recomputed = json.loads(
                json.dumps(recomputed, sort_keys=True, separators=(",", ":"))
            )
            if (
                nested_report != published_recomputed
                or recomputed.get("outcome") != "PASS"
            ):
                raise WorkloadError("nested comparison report was not semantically re-derived")
            expected_events.extend(
                (
                    {
                        "state": "comparison-process-ended",
                        "precision": precision,
                        "repeat": repeat,
                        "process": comparison["process"],
                    },
                    {
                        "state": "comparison-validated",
                        "precision": precision,
                        "repeat": repeat,
                        "complete": comparison["complete"],
                    },
                )
            )
    expected_checkpoint_lanes = [
        {
            "lane": scheduled["lane"],
            "run": lanes[scheduled["lane"]][scheduled["repeat"]],
        }
        for scheduled in expected_schedule
    ]
    if (
        checkpoint.get("hardware_before") != report["hardware"]["before"]
        or checkpoint.get("completed_lanes") != expected_checkpoint_lanes
        or checkpoint.get("completed_comparisons") != comparisons
    ):
        raise WorkloadError("matrix checkpoint progress differs from the final report")
    if journal_events != expected_events:
        raise WorkloadError("matrix journal event sequence was not re-derived")

    performance_specs = {
        "fdtd-cpu-fp32-to-1gpu": (fdtd_timings[labels[1]], fdtd_timings[labels[2]], 1.5),
        "fdtd-cpu-fp32-to-2gpu": (fdtd_timings[labels[1]], fdtd_timings[labels[3]], 2.0),
        "fdtd-one-gpu-to-two-gpu": (fdtd_timings[labels[2]], fdtd_timings[labels[3]], 1.1),
    }
    performance = report.get("performance")
    if not isinstance(performance, dict) or set(performance) != set(performance_specs):
        raise WorkloadError("matrix performance inventory is not exact")
    for name, (reference, candidate, release_floor) in performance_specs.items():
        minimum = float(performance[name].get("minimum", math.nan))
        if not math.isfinite(minimum) or minimum < release_floor:
            raise WorkloadError(f"matrix {name} weakens its release floor")
        expected = {**conservative_speedup(reference, candidate), "minimum": minimum}
        expected["outcome"] = (
            "PASS" if expected["conservative_speedup"] >= minimum else "FAIL"
        )
        if performance[name] != expected:
            raise WorkloadError(f"matrix {name} was not re-derived from raw timings")
    diagnostic_specs = {
        "workload-end-to-end-cpu-fp32-to-1gpu": (
            process_timings[labels[1]], process_timings[labels[2]]
        ),
        "workload-end-to-end-cpu-fp32-to-2gpu": (
            process_timings[labels[1]], process_timings[labels[3]]
        ),
        "workload-end-to-end-one-gpu-to-two-gpu": (
            process_timings[labels[2]], process_timings[labels[3]]
        ),
    }
    diagnostic_performance = report.get("diagnostic_performance")
    if not isinstance(diagnostic_performance, dict) or set(
        diagnostic_performance
    ) != set(diagnostic_specs):
        raise WorkloadError("matrix diagnostic-performance inventory is not exact")
    for name, (reference, candidate) in diagnostic_specs.items():
        if diagnostic_performance[name] != conservative_speedup(reference, candidate):
            raise WorkloadError(
                f"matrix diagnostic {name} was not re-derived from raw timings"
            )
    variability = report.get("timing_variability")
    expected_variability_keys = {
        f"{label}:fdtd" for label in labels
    }
    if not isinstance(variability, dict) or set(variability) != expected_variability_keys:
        raise WorkloadError("matrix timing-variability inventory is not exact")
    for label in labels:
        name = f"{label}:fdtd"
        maximum = float(variability[name].get("maximum", math.nan))
        if not math.isfinite(maximum) or not 0 < maximum <= 0.15:
            raise WorkloadError(f"matrix {name} weakens the release CV gate")
        if variability[name] != timing_stability(fdtd_timings[label], maximum):
            raise WorkloadError(f"matrix {name} CV was not re-derived")
    diagnostic_variability = report.get("diagnostic_timing_variability")
    if not isinstance(diagnostic_variability, dict) or set(diagnostic_variability) != {
        f"{label}:workload-end-to-end" for label in labels
    }:
        raise WorkloadError("matrix diagnostic timing-variability inventory is not exact")
    for label in labels:
        name = f"{label}:workload-end-to-end"
        if diagnostic_variability[name] != timing_stability(
            process_timings[label], 0.15
        ):
            raise WorkloadError(f"matrix diagnostic {name} CV was not re-derived")
    if classify_outcome(True, performance, variability) != "PASS":
        raise WorkloadError("matrix release gates do not re-derive PASS")
    return report


def _safe_output_relative(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise WorkloadError(f"{label} path is missing")
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
        or "\\" in value
    ):
        raise WorkloadError(f"{label} path is unsafe")
    return relative


def checkpoint_contract(
    args: argparse.Namespace,
    source: pathlib.Path,
    source_sha256: str,
    evidence_code: dict[str, Any],
    cpu_fp64: Build,
    fp32: Build,
    cpu_ranks: int,
    devices: tuple[str, ...],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return the exact immutable invocation contract for safe resumption."""

    return {
        "workload": args.workload,
        "input": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "evidence_code": evidence_code,
        "builds": {
            "cpu_fp64": {
                "receipt_id": cpu_fp64.receipt["receipt_id"],
                "build_input_id": cpu_fp64.receipt["build_input_id"],
                "artifact_set_id": cpu_fp64.receipt["artifact_set_id"],
                "source_sha256": _source_identity(cpu_fp64),
            },
            "shared_fp32": {
                "receipt_id": fp32.receipt["receipt_id"],
                "build_input_id": fp32.receipt["build_input_id"],
                "artifact_set_id": fp32.receipt["artifact_set_id"],
                "source_sha256": _source_identity(fp32),
            },
        },
        "cpu_ranks": cpu_ranks,
        "gpu_devices": list(devices),
        "repeats": args.repeats,
        "limits": {
            "lane_timeout_seconds": timeout_seconds,
            "comparison_timeout_seconds": args.comparison_timeout_seconds,
            "stdout_limit_bytes": args.stdout_limit_mib * 1024 * 1024,
        },
        "gates": {
            "minimum_one_gpu_speedup": args.minimum_one_gpu_speedup,
            "minimum_two_gpu_speedup": args.minimum_two_gpu_speedup,
            "minimum_multi_gpu_scaling": args.minimum_multi_gpu_scaling,
            "maximum_timing_cv": args.maximum_timing_cv,
        },
        "release_eligible": args.repeats >= RELEASE_REPEATS,
    }


def write_checkpoint(
    output: pathlib.Path,
    contract: dict[str, Any],
    hardware_before: dict[str, Any],
    completed_lanes: list[dict[str, Any]],
    completed_comparisons: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        output / "CHECKPOINT.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract": contract,
            "hardware_before": hardware_before,
            "completed_lanes": completed_lanes,
            "completed_comparisons": completed_comparisons,
        },
    )


def load_checkpoint(
    output: pathlib.Path, expected_contract: dict[str, Any]
) -> dict[str, Any]:
    with StableFile(output / "CHECKPOINT.json", "matrix recovery checkpoint") as stable:
        with stable.file_object() as handle:
            checkpoint = json.load(handle)
    expected_keys = {
        "schema",
        "contract",
        "hardware_before",
        "completed_lanes",
        "completed_comparisons",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
        raise WorkloadError("matrix recovery checkpoint schema is not exact")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise WorkloadError("matrix recovery checkpoint version differs")
    if checkpoint["contract"] != expected_contract:
        raise WorkloadError("resume invocation differs from the sealed checkpoint contract")
    if not isinstance(checkpoint["hardware_before"], dict):
        raise WorkloadError("matrix recovery checkpoint lacks initial hardware")
    if not isinstance(checkpoint["completed_lanes"], list) or not isinstance(
        checkpoint["completed_comparisons"], list
    ):
        raise WorkloadError("matrix recovery checkpoint progress is malformed")
    return checkpoint


def validate_resumed_lane(
    output: pathlib.Path,
    lane: Lane,
    workload: str,
    repeat: int,
    entry: dict[str, Any],
    selected_devices: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(entry, dict) or set(entry) != {"lane", "run"}:
        raise WorkloadError("checkpoint lane entry schema is not exact")
    if entry["lane"] != lane.label or not isinstance(entry["run"], dict):
        raise WorkloadError("checkpoint lane order differs from the matrix schedule")
    run = entry["run"]
    lane_relative = pathlib.PurePosixPath(
        "lanes", lane.label, f"repeat-{repeat:02d}"
    )
    expected_log = f"logs/{lane.label}/repeat-{repeat:02d}.log"
    if run.get("repeat") != repeat or run.get("output") != lane_relative.as_posix():
        raise WorkloadError("checkpoint lane role/path differs from the matrix schedule")
    _verify_process_record(
        output, run.get("process"), expected_log, f"resumed {lane.label}"
    )
    validate_hardware_evidence(run.get("hardware"), list(selected_devices))
    lane_output = output.joinpath(*lane_relative.parts)
    validated = validate_lane_output(lane, lane_output, workload)
    complete = stable_file_record(
        lane_output / "COMPLETE", output, f"resumed {lane.label} COMPLETE"
    )
    recomputed = {
        "repeat": repeat,
        "output": lane_relative.as_posix(),
        "process": run["process"],
        "hardware": run["hardware"],
        "fdtd_wall_seconds": validated["fdtd_wall_seconds"],
        "workload_end_to_end_seconds": validated["workload_end_to_end_seconds"],
        "phase_wall_seconds": validated["phase_wall_seconds"],
        "timestep_contract": validated["timestep_contract"],
        "provenance": validated["provenance"],
        "complete": complete,
    }
    if run != recomputed:
        raise WorkloadError(f"checkpoint {lane.label} record was not re-derived")
    return run


def validate_resumed_comparison(
    output: pathlib.Path,
    expected: tuple[
        str, str, int, tuple[tuple[str, str], ...], pathlib.PurePosixPath
    ],
    repeat: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    precision, _reference_label, reference_repeat, _candidate_specs, relative = expected
    expected_log = f"logs/comparisons/{precision}-repeat-{repeat:02d}.log"
    if (
        not isinstance(record, dict)
        or record.get("precision") != precision
        or record.get("repeat") != repeat
        or record.get("reference_repeat") != reference_repeat
        or record.get("output") != relative.as_posix()
    ):
        raise WorkloadError("checkpoint comparison order/path differs")
    _verify_process_record(
        output, record.get("process"), expected_log, f"resumed {precision} comparison"
    )
    comparison_output = output.joinpath(*relative.parts)
    complete_record = stable_file_record(
        comparison_output / "COMPLETE",
        output,
        f"resumed {precision} comparison COMPLETE",
    )
    if record.get("complete") != complete_record:
        raise WorkloadError("checkpoint comparison COMPLETE record differs")
    with StableFile(
        comparison_output / "COMPLETE", "resumed comparison COMPLETE"
    ) as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if complete.get("outcome") != "PASS":
        raise WorkloadError("checkpoint comparison is not a PASS")
    return record


def validate_resume_journal_prefix(
    journal: dict[str, Any],
    input_record: dict[str, Any],
    evidence_code: dict[str, Any],
    completed_lanes: list[dict[str, Any]],
    completed_comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_keys = {"schema", "input", *evidence_code, "events"}
    if not isinstance(journal, dict) or set(journal) != expected_keys:
        raise WorkloadError("resume journal schema is not exact")
    if (
        journal["schema"] != "gpmeep-user-workload-matrix-journal-v2"
        or journal["input"] != input_record
        or any(journal[name] != record for name, record in evidence_code.items())
        or not isinstance(journal["events"], list)
        or not journal["events"]
    ):
        raise WorkloadError("resume journal identity differs")
    first = journal["events"][0]
    if (
        not isinstance(first, dict)
        or set(first) != {"state", "unix_seconds"}
        or first["state"] != "controller-started"
        or not math.isfinite(float(first["unix_seconds"]))
    ):
        raise WorkloadError("resume journal has no valid controller start")
    expected_events = [first]
    for entry in completed_lanes:
        lane = entry["lane"]
        run = entry["run"]
        repeat = run["repeat"]
        expected_events.extend(
            (
                {"state": "lane-started", "repeat": repeat, "lane": lane},
                {
                    "state": "lane-process-ended",
                    "repeat": repeat,
                    "lane": lane,
                    "process": run["process"],
                },
                {
                    "state": "lane-validated",
                    "repeat": repeat,
                    "lane": lane,
                    "complete": run["complete"],
                },
            )
        )
    for record in completed_comparisons:
        expected_events.extend(
            (
                {
                    "state": "comparison-process-ended",
                    "precision": record["precision"],
                    "repeat": record["repeat"],
                    "process": record["process"],
                },
                {
                    "state": "comparison-validated",
                    "precision": record["precision"],
                    "repeat": record["repeat"],
                    "complete": record["complete"],
                },
            )
        )
    if journal["events"][: len(expected_events)] != expected_events:
        raise WorkloadError("resume journal disagrees with the validated checkpoint")
    return expected_events


def archive_interrupted_attempt(
    output: pathlib.Path, partial_paths: list[pathlib.PurePosixPath]
) -> pathlib.Path:
    history = output / "resume-history"
    history.mkdir(exist_ok=True)
    attempt_index = 0
    while (history / f"attempt-{attempt_index:04d}").exists():
        attempt_index += 1
    attempt = history / f"attempt-{attempt_index:04d}"
    attempt.mkdir()
    shutil.copy2(output / "JOURNAL.json", attempt / "JOURNAL.json")
    if (output / "CHECKPOINT.json").is_file():
        shutil.copy2(output / "CHECKPOINT.json", attempt / "CHECKPOINT.json")
    candidates = [
        pathlib.PurePosixPath(name)
        for name in (
            "FAILED.json",
            "RUNNING.json",
            "report.json",
            "report.md",
            "DEVELOPMENT.json",
        )
    ]
    candidates.extend(partial_paths)
    seen: set[pathlib.PurePosixPath] = set()
    for relative in candidates:
        if relative in seen:
            continue
        seen.add(relative)
        source = output.joinpath(*relative.parts)
        if not source.exists():
            continue
        if source.is_symlink():
            raise WorkloadError("resume candidate traverses a symlink")
        destination = attempt.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    atomic_write_json(
        attempt / "ARCHIVED.json",
        {
            "schema": "gpmeep-user-workload-resume-archive-v1",
            "archived_unix_seconds": time.time(),
            "partial_paths": [value.as_posix() for value in partial_paths],
        },
    )
    return attempt


def resume_history_records(output: pathlib.Path) -> list[dict[str, Any]]:
    history = output / "resume-history"
    if not history.exists():
        return []
    if history.is_symlink() or not history.is_dir():
        raise WorkloadError("matrix resume history is not a directory")
    records = []
    for attempt in sorted(history.iterdir()):
        if (
            attempt.is_symlink()
            or not attempt.is_dir()
            or not attempt.name.startswith("attempt-")
            or not attempt.name.removeprefix("attempt-").isdigit()
        ):
            raise WorkloadError("matrix resume history contains a noncanonical entry")
        records.append(
            stable_file_record(
                attempt / "ARCHIVED.json",
                output,
                f"matrix resume archive {attempt.name}",
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1 or args.repeats > MAX_REPEATS:
        raise WorkloadError(f"repeats must be in [1,{MAX_REPEATS}]")
    if args.repeats < RELEASE_REPEATS and not args.development_allow_fewer_repeats:
        raise WorkloadError("release evidence requires at least three measured repeats")
    timeout_seconds = args.timeout_seconds or DEFAULT_TIMEOUT[args.workload]
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WorkloadError("timeout must be positive and finite")
    if args.stdout_limit_mib < 1 or args.stdout_limit_mib > 1024:
        raise WorkloadError("stdout limit must be in [1,1024] MiB")
    for name, release_minimum in RELEASE_SPEED_GATES.items():
        value = getattr(args, name)
        if not math.isfinite(value) or value < release_minimum:
            raise WorkloadError(
                f"{name} must be at least the release floor {release_minimum}"
            )
    if not math.isfinite(args.maximum_timing_cv) or not 0 < args.maximum_timing_cv <= 0.15:
        raise WorkloadError("maximum timing CV must be in (0,0.15]")

    expected_hash = TERS_SHA256 if args.workload == "ters" else AUNP_ARCHIVE_SHA256
    source = verify_file(args.input, expected_hash, f"{args.workload} workload input")
    if args.resume:
        output = args.output.resolve(strict=True)
        if (output / "COMPLETE").exists() or (output / "DEVELOPMENT.json").exists():
            raise WorkloadError("a terminal matrix cannot be resumed")
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume:
        lock_metadata = lock_path.lstat()
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise WorkloadError("matrix resume lock is not a regular file")
    lock_handle = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_handle.close()
        raise WorkloadError(f"could not acquire workload matrix lock: {exc}") from exc

    try:
        source_stable = StableFile(
            source,
            f"{args.workload} workload input",
            expected={"size_bytes": source.stat().st_size, "sha256": expected_hash},
        )
    except Exception:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise
    code_stables: dict[str, StableFile] = {}
    try:
        for name, path in (
            ("controller", pathlib.Path(__file__)),
            ("comparator", SCRIPT_DIR / "compare_user_workloads.py"),
            ("common", SCRIPT_DIR / "common.py"),
            ("ters_adapter", SCRIPT_DIR / "run_ters_workload.py"),
            ("aunp_adapter", SCRIPT_DIR / "run_aunp_workload.py"),
            ("aunp_exact_resume", SCRIPT_DIR / "aunp_exact_resume.py"),
        ):
            code_stables[name] = StableFile(path, f"matrix {name}")
        evidence_code = {
            name: stable_identity_record(stable, REPO)
            for name, stable in code_stables.items()
        }
        input_record = {
            "path": str(source),
            "size_bytes": source_stable.initial_stat.st_size,
            "sha256": source_stable.sha256,
        }
        if args.resume:
            with StableFile(output / "JOURNAL.json", "matrix resume journal") as stable:
                with stable.file_object() as handle:
                    journal = json.load(handle)
        else:
            journal = {
                "schema": "gpmeep-user-workload-matrix-journal-v2",
                "input": input_record,
                **evidence_code,
                "events": [],
            }

        def append_journal(event: dict[str, Any]) -> None:
            journal["events"].append(event)
            atomic_write_json(output / "JOURNAL.json", journal)

        if not args.resume:
            atomic_write_json(
                output / "RUNNING.json", {"schema": SCHEMA, "started": time.time()}
            )
            append_journal({"state": "controller-started", "unix_seconds": time.time()})
    except Exception:
        for stable in code_stables.values():
            stable.close(verify=False)
        source_stable.close(verify=False)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise
    try:
        cpu_fp64 = load_build(
            "cpu-fp64",
            args.cpu_fp64_python,
            args.cpu_fp64_mpiexec,
            args.cpu_fp64_receipt,
            "cpu-mpi-python-fp64",
            False,
            False,
        )
        fp32 = load_build(
            "shared-fp32",
            args.fp32_python,
            args.fp32_mpiexec,
            args.fp32_receipt,
            "cuda-mpi-python-fp32",
            True,
            True,
        )
        identities = {_source_identity(build) for build in (cpu_fp64, fp32)}
        if len(identities) != 1:
            raise WorkloadError("the FP64 and shared FP32 builds do not share one source snapshot")
        available_cores = physical_core_count()
        cpu_ranks = args.cpu_ranks or available_cores
        if cpu_ranks < 2 or cpu_ranks > available_cores:
            raise WorkloadError(
                f"cpu-ranks must be in [2,{available_cores}] for this host"
            )
        devices = tuple(_normalized_uuid(value) for value in args.gpu_devices.split(","))
        normalized_devices = {
            value.lower().removeprefix("gpu-").replace("-", "") for value in devices
        }
        if len(devices) != 2 or len(normalized_devices) != 2:
            raise WorkloadError("release matrix requires exactly two distinct GPU UUIDs")
        current_hardware = hardware_snapshot(devices)

        lanes = (
            Lane(f"cpu-fp64-{cpu_ranks}r", cpu_fp64, "cpu", cpu_ranks, ()),
            Lane(f"cpu-fp32-{cpu_ranks}r", fp32, "cpu", cpu_ranks, ()),
            Lane("cuda-fp32-1g", fp32, "cuda", 1, devices[:1]),
            Lane("cuda-fp32-2g", fp32, "cuda", 2, devices),
        )
        lane_runs: dict[str, list[dict[str, Any]]] = {lane.label: [] for lane in lanes}
        schedule_entries: list[tuple[int, Lane]] = []
        for repeat in range(args.repeats):
            ordered = lanes if repeat % 2 == 0 else tuple(reversed(lanes))
            for lane in ordered:
                schedule_entries.append((repeat, lane))
        schedule = [
            {"repeat": repeat, "lane": lane.label}
            for repeat, lane in schedule_entries
        ]
        lane_labels = tuple(lane.label for lane in lanes)
        comparison_schedule = [
            (repeat, spec)
            for repeat in range(args.repeats)
            for spec in comparison_specs(lane_labels, repeat)
        ]
        contract = checkpoint_contract(
            args,
            source,
            source_stable.sha256,
            evidence_code,
            cpu_fp64,
            fp32,
            cpu_ranks,
            devices,
            timeout_seconds,
        )
        completed_lanes: list[dict[str, Any]] = []
        comparison_records: list[dict[str, Any]] = []
        if args.resume:
            checkpoint = load_checkpoint(output, contract)
            saved_hardware = checkpoint["hardware_before"]
            validate_hardware_evidence(
                {"before": saved_hardware, "after": current_hardware}, list(devices)
            )
            hardware_before = saved_hardware
            checkpoint_lanes = checkpoint["completed_lanes"]
            checkpoint_comparisons = checkpoint["completed_comparisons"]
            if len(checkpoint_lanes) > len(schedule_entries):
                raise WorkloadError("checkpoint contains too many completed lanes")
            if checkpoint_comparisons and len(checkpoint_lanes) != len(schedule_entries):
                raise WorkloadError("checkpoint comparisons precede unfinished lanes")
            if len(checkpoint_comparisons) > len(comparison_schedule):
                raise WorkloadError("checkpoint contains too many comparisons")
            for index, entry in enumerate(checkpoint_lanes):
                repeat, lane = schedule_entries[index]
                run = validate_resumed_lane(
                    output, lane, args.workload, repeat, entry, devices
                )
                completed_lanes.append(entry)
                lane_runs[lane.label].append(run)
            for index, record in enumerate(checkpoint_comparisons):
                repeat, expected = comparison_schedule[index]
                comparison_records.append(
                    validate_resumed_comparison(
                        output, expected, repeat, record
                    )
                )
            expected_events = validate_resume_journal_prefix(
                journal,
                input_record,
                evidence_code,
                completed_lanes,
                comparison_records,
            )
            partial_paths: list[pathlib.PurePosixPath] = []
            if len(completed_lanes) < len(schedule_entries):
                repeat, lane = schedule_entries[len(completed_lanes)]
                partial_paths.extend(
                    (
                        pathlib.PurePosixPath(
                            "lanes", lane.label, f"repeat-{repeat:02d}"
                        ),
                        pathlib.PurePosixPath(
                            "runtime", lane.label, f"repeat-{repeat:02d}"
                        ),
                        pathlib.PurePosixPath(
                            "logs", lane.label, f"repeat-{repeat:02d}.log"
                        ),
                    )
                )
            elif len(comparison_records) < len(comparison_schedule):
                repeat, expected = comparison_schedule[len(comparison_records)]
                precision, _reference, _reference_repeat, _candidates, relative = expected
                partial_paths.extend(
                    (
                        relative,
                        pathlib.PurePosixPath(
                            "logs", "comparisons", f"{precision}-repeat-{repeat:02d}.log"
                        ),
                    )
                )
            archive_interrupted_attempt(output, partial_paths)
            journal["events"] = expected_events
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(
                output / "RUNNING.json",
                {
                    "schema": SCHEMA,
                    "resumed": time.time(),
                    "completed_lanes": len(completed_lanes),
                    "completed_comparisons": len(comparison_records),
                },
            )
        else:
            hardware_before = current_hardware
            write_checkpoint(
                output, contract, hardware_before, completed_lanes, comparison_records
            )

        for repeat, lane in schedule_entries[len(completed_lanes) :]:
            lane_output = output / "lanes" / lane.label / f"repeat-{repeat:02d}"
            lane_output.parent.mkdir(parents=True, exist_ok=True)
            runtime_root = output / "runtime" / lane.label / f"repeat-{repeat:02d}"
            environment = clean_environment(lane, runtime_root)
            log_path = output / "logs" / lane.label / f"repeat-{repeat:02d}.log"
            lane_hardware_before = hardware_snapshot(devices)
            append_journal(
                {"state": "lane-started", "repeat": repeat, "lane": lane.label}
            )
            process_record = run_bounded(
                lane_command(lane, args.workload, source, lane_output),
                environment,
                log_path,
                timeout_seconds,
                args.stdout_limit_mib * 1024 * 1024,
            )
            append_journal(
                {
                    "state": "lane-process-ended",
                    "repeat": repeat,
                    "lane": lane.label,
                    "process": process_record,
                }
            )
            if (
                process_record["returncode"] != 0
                or process_record["timed_out"]
                or process_record["output_limited"]
            ):
                raise WorkloadError(
                    f"{lane.label} repeat {repeat} failed; see {log_path}"
                )
            lane_hardware_after = hardware_snapshot(devices)
            lane_hardware = {
                "before": lane_hardware_before,
                "after": lane_hardware_after,
            }
            validate_hardware_evidence(lane_hardware, list(devices))
            validated = validate_lane_output(lane, lane_output, args.workload)
            lane_complete = stable_file_record(
                lane_output / "COMPLETE", output, f"{lane.label} COMPLETE"
            )
            run_record = {
                "repeat": repeat,
                "output": str(lane_output.relative_to(output)),
                "process": process_record,
                "hardware": lane_hardware,
                "fdtd_wall_seconds": validated["fdtd_wall_seconds"],
                "workload_end_to_end_seconds": validated[
                    "workload_end_to_end_seconds"
                ],
                "phase_wall_seconds": validated["phase_wall_seconds"],
                "timestep_contract": validated["timestep_contract"],
                "provenance": validated["provenance"],
                "complete": lane_complete,
            }
            lane_runs[lane.label].append(run_record)
            completed_lanes.append({"lane": lane.label, "run": run_record})
            append_journal(
                {
                    "state": "lane-validated",
                    "repeat": repeat,
                    "lane": lane.label,
                    "complete": lane_complete,
                }
            )
            write_checkpoint(
                output, contract, hardware_before, completed_lanes, comparison_records
            )

        # Every lane must have performed exactly the same per-stage timesteps
        # and Meep times.  This is checked independently of numerical outputs.
        workload_contract = lane_runs[lanes[0].label][0]["timestep_contract"]
        for runs in lane_runs.values():
            for run in runs:
                if run["timestep_contract"] != workload_contract:
                    raise WorkloadError("CPU/GPU lanes did not perform identical FDTD work")

        comparison_environment = {
            "PATH": f"{fp32.python.parent}:/usr/bin:/bin",
            "HOME": str(output / "runtime"),
            "XDG_CACHE_HOME": str(output / "runtime"),
            "TMPDIR": "/tmp",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        for repeat, spec in comparison_schedule[len(comparison_records) :]:
            (
                precision,
                reference_label,
                reference_repeat,
                candidate_specs,
                relative,
            ) = spec
            reference = output / lane_runs[reference_label][reference_repeat]["output"]
            candidates = [
                (
                    lane_label,
                    comparison_class,
                    output / lane_runs[lane_label][repeat]["output"],
                )
                for lane_label, comparison_class in candidate_specs
            ]
            comparison_output = output.joinpath(*relative.parts)
            log_path = output / "logs" / "comparisons" / f"{precision}-repeat-{repeat:02d}.log"
            process_record = run_bounded(
                comparison_command(
                    fp32.python,
                    args.workload,
                    comparison_output,
                    reference,
                    candidates,
                    source,
                ),
                comparison_environment,
                log_path,
                args.comparison_timeout_seconds,
                args.stdout_limit_mib * 1024 * 1024,
            )
            append_journal(
                {
                    "state": "comparison-process-ended",
                    "precision": precision,
                    "repeat": repeat,
                    "process": process_record,
                }
            )
            if (
                process_record["returncode"] != 0
                or process_record["timed_out"]
                or process_record["output_limited"]
            ):
                raise WorkloadError(
                    f"{precision} comparison repeat {repeat} failed; see {log_path}"
                )
            with StableFile(
                comparison_output / "COMPLETE", "comparison COMPLETE marker"
            ) as stable_complete:
                with stable_complete.file_object() as handle:
                    complete = json.load(handle)
            if complete.get("outcome") != "PASS":
                raise WorkloadError(f"{precision} comparison did not pass")
            comparison_complete = stable_file_record(
                comparison_output / "COMPLETE",
                output,
                f"{precision} comparison COMPLETE",
            )
            comparison_record = {
                "precision": precision,
                "repeat": repeat,
                "reference_repeat": reference_repeat,
                "output": str(comparison_output.relative_to(output)),
                "process": process_record,
                "complete": comparison_complete,
            }
            comparison_records.append(comparison_record)
            append_journal(
                {
                    "state": "comparison-validated",
                    "precision": precision,
                    "repeat": repeat,
                    "complete": comparison_complete,
                }
            )
            write_checkpoint(
                output, contract, hardware_before, completed_lanes, comparison_records
            )

        timings = {
            label: [record["fdtd_wall_seconds"] for record in records]
            for label, records in lane_runs.items()
        }
        workload_end_to_end_timings = {
            label: [record["workload_end_to_end_seconds"] for record in records]
            for label, records in lane_runs.items()
        }
        performance = {
            "fdtd-cpu-fp32-to-1gpu": {
                **conservative_speedup(timings[lanes[1].label], timings[lanes[2].label]),
                "minimum": args.minimum_one_gpu_speedup,
            },
            "fdtd-cpu-fp32-to-2gpu": {
                **conservative_speedup(timings[lanes[1].label], timings[lanes[3].label]),
                "minimum": args.minimum_two_gpu_speedup,
            },
            "fdtd-one-gpu-to-two-gpu": {
                **conservative_speedup(timings[lanes[2].label], timings[lanes[3].label]),
                "minimum": args.minimum_multi_gpu_scaling,
            },
        }
        for metric in performance.values():
            metric["outcome"] = (
                "PASS"
                if metric["conservative_speedup"] >= metric["minimum"]
                else "FAIL"
            )
        diagnostic_performance = {
            "workload-end-to-end-cpu-fp32-to-1gpu": conservative_speedup(
                workload_end_to_end_timings[lanes[1].label],
                workload_end_to_end_timings[lanes[2].label],
            ),
            "workload-end-to-end-cpu-fp32-to-2gpu": conservative_speedup(
                workload_end_to_end_timings[lanes[1].label],
                workload_end_to_end_timings[lanes[3].label],
            ),
            "workload-end-to-end-one-gpu-to-two-gpu": conservative_speedup(
                workload_end_to_end_timings[lanes[2].label],
                workload_end_to_end_timings[lanes[3].label],
            ),
        }
        timing_variability = {}
        diagnostic_timing_variability = {}
        if args.repeats >= RELEASE_REPEATS:
            for lane in lanes:
                timing_variability[f"{lane.label}:fdtd"] = timing_stability(
                    timings[lane.label], args.maximum_timing_cv
                )
                diagnostic_timing_variability[
                    f"{lane.label}:workload-end-to-end"
                ] = timing_stability(
                    workload_end_to_end_timings[lane.label], 0.15
                )
        release_eligible = args.repeats >= RELEASE_REPEATS
        outcome = classify_outcome(
            release_eligible, performance, timing_variability
        )
        hardware_after = hardware_snapshot(devices)
        hardware_evidence = {"before": hardware_before, "after": hardware_after}
        validate_hardware_evidence(hardware_evidence, list(devices))
        source_stable.verify_unchanged()
        recovery_checkpoint = stable_file_record(
            output / "CHECKPOINT.json", output, "matrix recovery checkpoint"
        )
        resume_history = resume_history_records(output)
        report = {
            "schema": SCHEMA,
            "outcome": outcome,
            "workload": args.workload,
            "input": {
                "path": str(source),
                "size_bytes": source_stable.initial_stat.st_size,
                "sha256": source_stable.sha256,
            },
            "release_eligible": release_eligible,
            "evidence_code": evidence_code,
            "source_snapshot_sha256": next(iter(identities)),
            "cpu_physical_cores": cpu_ranks,
            "gpu_devices": list(devices),
            "hardware": hardware_evidence,
            "repeats": args.repeats,
            "schedule": schedule,
            "workload_contract": workload_contract,
            "lanes": lane_runs,
            "comparisons": comparison_records,
            "recovery_checkpoint": recovery_checkpoint,
            "resume_history": resume_history,
            "performance": performance,
            "diagnostic_performance": diagnostic_performance,
            "timing_variability": timing_variability,
            "diagnostic_timing_variability": diagnostic_timing_variability,
            "diagnostic_timing_warning": (
                "Workload end-to-end timing is not a release performance gate. "
                "For TERS it includes one original presentation/I/O workload per "
                "private MPI-rank directory, so lane rank counts contaminate comparisons."
            ),
        }
        atomic_write_json(output / "report.json", report)
        markdown = output / "report.md"
        atomic_write_text(markdown, markdown_report(report))
        if outcome == "DEVELOPMENT_ONLY":
            (output / "RUNNING.json").unlink()
            atomic_write_json(
                output / "DEVELOPMENT.json",
                {
                    "schema": "gpmeep-user-workload-matrix-development-v2",
                    "report": file_record(output / "report.json", output),
                    "markdown": file_record(markdown, output),
                    "journal": file_record(output / "JOURNAL.json", output),
                    "outcome": outcome,
                },
            )
            for stable in code_stables.values():
                stable.close()
            source_stable.close()
            return 0
        if outcome != "PASS":
            (output / "RUNNING.json").unlink()
            atomic_write_json(
                output / "FAILED.json",
                {
                    "schema": "gpmeep-user-workload-matrix-failed-v2",
                    "report": file_record(output / "report.json", output),
                    "journal": file_record(output / "JOURNAL.json", output),
                },
            )
            for stable in code_stables.values():
                stable.close()
            source_stable.close()
            return 3
        (output / "RUNNING.json").unlink()
        atomic_write_json(
            output / "COMPLETE",
            {
                "schema": COMPLETE_SCHEMA,
                "report": file_record(output / "report.json", output),
                "markdown": file_record(markdown, output),
                "journal": file_record(output / "JOURNAL.json", output),
                "outcome": "PASS",
            },
        )
        verify_matrix_complete(output)
        for stable in code_stables.values():
            stable.close()
        source_stable.close()
        return 0
    except Exception as exc:
        running = output / "RUNNING.json"
        if running.exists():
            running.unlink()
        for terminal_name in ("COMPLETE", "DEVELOPMENT.json"):
            terminal = output / terminal_name
            if terminal.exists():
                terminal.unlink()
        partial_artifacts = collect_partial_evidence(output)
        failure = {
            "schema": "gpmeep-user-workload-matrix-failed-v2",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_artifacts": partial_artifacts,
        }
        if (output / "JOURNAL.json").is_file():
            failure["journal"] = file_record(output / "JOURNAL.json", output)
        atomic_write_json(
            output / "FAILED.json",
            failure,
        )
        raise
    finally:
        for stable in code_stables.values():
            stable.close(verify=False)
        source_stable.close(verify=False)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkloadError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"user workload matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
