"""Adversarial tests for the relocatable validation archive sealer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SEALER_PATH = pathlib.Path(__file__).resolve().parents[1] / "archive_validation.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_archive_validation", SEALER_PATH)
assert SPEC and SPEC.loader
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)

from gpmeep_provenance import (  # noqa: E402
    canonical_sha256,
    file_record,
    immutable_directory_record,
    sha256_file,
    source_snapshot,
    tree_manifest,
)
from gpmeep_qualification_contract import (  # noqa: E402
    AUDIT_LOG_NAME,
    DIRECT_ELF_LOG_BINDINGS,
    INSTALLED_CUDA_QUALIFICATION_SPECS,
    INSTALLED_LAZY_API_TEST_SPECS,
    PYTHON_RUNTIME_LOG_BINDINGS,
    REQUIRED_ARTIFACT_BINDINGS,
    REQUIRED_CONFIGURATION_BINDINGS,
    REQUIRED_QUALIFICATION_LOGS,
    SENTINEL_LOG_NAME,
    STEP_FINITE_FAILURE_DIAGNOSTIC,
    STEP_FINITE_FAILURE_LOG_NAME,
    create_identity_snapshot,
    marker_for,
    seal_contract,
)
from qualification_fixture import (  # noqa: E402
    write_synthetic_auxiliary_logs,
    write_synthetic_release_cuda_math_configuration,
)


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ArchiveFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.output = root / "runner-output"
        self.receipt = root / "build-provenance.json"
        self.case_directory = sealer.runner_contract.case_directory_id(
            "cases/example.py"
        )
        self.repo.mkdir()
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo, check=True
        )
        fixture_manifest = {
            "schema_version": sealer.runner_contract.SCHEMA_VERSION,
            "inventory": [
                {
                    "glob": "cases/*.py",
                    "kind": "example",
                    "disposition": "run",
                    "reason": "archive fixture",
                    "tier": ["full"],
                    "timeout_seconds": 30,
                    "comparison": {
                        "mode": "json_metrics",
                        "prefix": "gpmeep-metrics:",
                        "tolerances": {
                            "value": {"rtol": 0.0, "atol": 0.0}
                        },
                    },
                    "compute_scope": "fdtd_cuda",
                    "gpu_contract": "cuda_dispatch",
                }
            ],
            "overrides": [],
        }
        files = {
            ".gitignore": (
                "install/\n"
                "build/meep-cuda-mpi-python-fp32/qualification-logs/\n"
                "build/meep-cuda-mpi-python-fp32/qualification-identity-*.json\n"
                "build/meep-cuda-mpi-python-fp32/qualification-contract-v2.json\n"
                "build/meep-cuda-mpi-python-fp32/receipt-cache/\n"
                "build/meep-cuda-mpi-python-fp32/canonical-build-environment.json\n"
                "build/meep-cuda-mpi-python-fp32/config.status\n"
                "build/meep-cuda-mpi-python-fp32/src/cuda-runtime-flags.stamp\n"
                "build/meep-cuda-mpi-python-fp32/cuda-runtime-qualification/\n"
            ),
            "scripts/python-validation/manifest.json": json.dumps(
                fixture_manifest, sort_keys=True,
            )
            + "\n",
            "scripts/python-validation/run_validation.py": "# runner fixture\n",
            "scripts/python-validation/run_example_oracle.py": "# oracle fixture\n",
            "scripts/python-validation/archive_validation.py": "# archive fixture\n",
            "scripts/python-validation/sitecustomize.py": "# hook fixture\n",
            "scripts/python-validation/absorber_branch_matrix.py": "# absorber fixture\n",
            "scripts/python-validation/perturbation_branch_matrix.py": "# perturbation fixture\n",
            "scripts/python-validation/run_point_dipole_cyl_validation.py": "# point-dipole fixture\n",
            "scripts/python-validation/stochastic_branch_matrix.py": "# stochastic fixture\n",
            "scripts/python-validation/stochastic_line_basis_matrix.py": "# stochastic line-basis fixture\n",
            "scripts/python-validation/stochastic_reciprocity_matrix.py": "# stochastic reciprocity fixture\n",
            "scripts/gpmeep_provenance.py": "# provenance fixture\n",
            "scripts/gpmeep_qualification_contract.py": "# qualification fixture\n",
            "scripts/gpmeep-control-python.py": "# control-python fixture\n",
            "scripts/build-meep-cuda-mpi-python.sh": "#!/bin/sh\nexit 0\n",
            "python/tests/test_gpu_backend.py": "# GPU backend fixture\n",
            "python/tests/test_adjoint_default_material_grid.py": (
                "# adjoint fixture\n"
            ),
            "cases/example.py": "print('fixture')\n",
            ".envs/meep-gpu-cuda-mpi/bin/python": "fixture-python\n",
            ".envs/meep-gpu-cuda-mpi/bin/nvcc": "synthetic nvcc\n",
            ".envs/meep-gpu-cuda-mpi/bin/x86_64-conda-linux-gnu-c++": (
                "synthetic host compiler\n"
            ),
            "build/meep-cuda-mpi-python-fp32/python/meep/__init__.py": "# meep\n",
            "build/meep-cuda-mpi-python-fp32/python/meep/_meep.so": "extension\n",
            "build/meep-cuda-mpi-python-fp32/src/.libs/libmeep.so": "libmeep\n",
            "build/meep-cuda-mpi-python-fp32/qualification-fontconfig.conf": "fontconfig\n",
            ".micromamba/cache/build-home-fixture/state": "build-home\n",
            "build/meep-cuda-mpi-python-fp32/qualification-home/state": "qualification-home\n",
        }
        for relative, _selectors, _tests_run in INSTALLED_LAZY_API_TEST_SPECS:
            files.setdefault(relative, "# installed lazy API fixture\n")
        for relative, contents in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        os.chmod(self.repo / "scripts/build-meep-cuda-mpi-python.sh", 0o755)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Archive Fixture",
                "-c", "user.email=archive@example.invalid",
                "commit", "-qm", "fixture",
            ],
            cwd=self.repo,
            check=True,
        )
        self.snapshot = source_snapshot(self.repo)
        self._write_receipt()
        self._write_runner_output()

    def _write_qualification_evidence(
        self, build_root: pathlib.Path
    ) -> tuple[dict[str, pathlib.Path], pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
        extension = build_root / "python/meep/_meep.so"
        libmeep = build_root / "src/.libs/libmeep.so"
        install_prefix = self.repo / "install/meep-cuda-mpi-python-fp32"
        installed_extension = (
            install_prefix / "lib/python3.11/site-packages/meep/_meep.so.38.0.0"
        )
        installed_mpb_extension = (
            install_prefix
            / "lib/python3.11/site-packages/meep/mpb/_mpb.so.38.0.0"
        )
        installed_libmeep = install_prefix / "lib/libmeep.so.38.0.0"
        installed_libpympb = install_prefix / "lib/libpympb.so.38.0.0"
        for path, contents in (
            (installed_extension, b"installed extension\n"),
            (installed_mpb_extension, b"installed MPB extension\n"),
            (installed_libmeep, b"installed libmeep\n"),
            (installed_libpympb, b"installed libpympb\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        artifact_paths = {
            "cuda_architecture_test": extension,
            "cuda_formula_test": extension,
            "cuda_near2far_runtime_validation": extension,
            "cuda_runtime_validation": extension,
            "cuda_smoke": extension,
            "fd_allocation_shim": extension,
            "python_extension": extension,
            "libmeep": libmeep,
            "installed_python_extension": installed_extension,
            "installed_libmeep": installed_libmeep,
            "installed_mpb_extension": installed_mpb_extension,
            "installed_libpympb": installed_libpympb,
            "gpu_backend_test": extension,
            "gpu_step_db_test": extension,
            "gpu_mpi_performance": extension,
        }
        self.assert_artifact_names = tuple(sorted(artifact_paths))
        if self.assert_artifact_names != REQUIRED_ARTIFACT_BINDINGS:
            raise AssertionError("archive fixture artifact set is stale")
        log_root = build_root / "qualification-logs"
        log_root.mkdir(parents=True, exist_ok=True)
        (log_root / AUDIT_LOG_NAME).unlink(missing_ok=True)
        for name in REQUIRED_QUALIFICATION_LOGS:
            if name == STEP_FINITE_FAILURE_LOG_NAME:
                lines = [
                    "gpmeep-finite-abort-probe:rank=0,stage=before-failing-step",
                    "gpmeep-finite-abort-probe:rank=1,stage=before-failing-step",
                    f"meep: {STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    f"meep: {STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    f"gpmeep-expected-mpi-failure:{name}:status=1:"
                    f"diagnostic={STEP_FINITE_FAILURE_DIAGNOSTIC}",
                    marker_for(name).decode("ascii"),
                ]
                (log_root / name).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            else:
                (log_root / name).write_bytes(marker_for(name) + b"\n")
        for name, (extension_name, libmeep_name) in (
            PYTHON_RUNTIME_LOG_BINDINGS.items()
        ):
            if name in INSTALLED_CUDA_QUALIFICATION_SPECS:
                continue
            value = {
                "python_extension": file_record(
                    artifact_paths[extension_name], self.repo
                ),
                "libmeep": file_record(
                    artifact_paths[libmeep_name], self.repo
                ),
            }
            (log_root / name).write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                + "\n"
                + marker_for(name).decode("ascii")
                + "\n",
                encoding="utf-8",
            )
        write_synthetic_auxiliary_logs(log_root, self.repo, artifact_paths)
        for name, artifact_name in DIRECT_ELF_LOG_BINDINGS.items():
            executable = artifact_paths[artifact_name].resolve()
            library = artifact_paths["libmeep"].resolve()
            executable_record = file_record(executable, self.repo)
            library_record = file_record(library, self.repo)
            (log_root / name).write_text(
                f"executable={executable}\n"
                f"loaded_libmeep={library}\n"
                f"{executable_record['sha256']}  {executable}\n"
                f"{library_record['sha256']}  {library}\n"
                "synthetic ldd output\n"
                + marker_for(name).decode("ascii")
                + "\n",
                encoding="utf-8",
            )
        pycache = build_root / "qualification-pycache"
        pycache.mkdir(mode=0o700, exist_ok=True)
        pycache.chmod(0o700)
        identity = create_identity_snapshot(
            self.repo,
            self.repo / ".envs/meep-gpu-cuda-mpi",
            artifact_paths,
            installed_prefix=install_prefix,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        (log_root / SENTINEL_LOG_NAME).write_text(
            json.dumps(
                {
                    "identity_sha256": canonical_sha256(identity),
                    "watch_count": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            + marker_for(SENTINEL_LOG_NAME).decode("ascii")
            + "\n",
            encoding="utf-8",
        )
        before = build_root / "qualification-identity-before.json"
        after = build_root / "qualification-identity-after.json"
        seal = build_root / "qualification-contract-v2.json"
        write_json(before, identity)
        write_json(after, identity)
        seal_contract(self.repo, log_root, before, after, seal)
        return artifact_paths, log_root, before, after, seal

    def _write_receipt(self) -> None:
        tool = pathlib.Path("/bin/sh").resolve()
        python = self.repo / ".envs/meep-gpu-cuda-mpi/bin/python"
        build_root = self.repo / "build/meep-cuda-mpi-python-fp32"
        install_prefix = self.repo / "install/meep-cuda-mpi-python-fp32"
        cuda_math = write_synthetic_release_cuda_math_configuration(
            self.repo, build_root, install_prefix
        )
        configuration = {
            "builder": file_record(
                self.repo / "scripts/build-meep-cuda-mpi-python.sh", self.repo
            ),
            "lockfiles": {},
            "qualification_contract": "gpmeep-cuda-mpi-python-fp32-v2",
            "configure_argv": cuda_math["configure_argv"],
            "environment": {"MEEP_GPU_FAST_MATH": "OFF"},
        }
        artifact_paths, log_root, identity_before, identity_after, seal = (
            self._write_qualification_evidence(build_root)
        )
        receipt_cache = build_root / "receipt-cache"
        receipt_cache.mkdir(mode=0o755, exist_ok=True)
        receipt_cache.chmod(0o755)
        cache_sentinel = receipt_cache / ".leave"
        if cache_sentinel.exists():
            cache_sentinel.chmod(0o644)
        cache_sentinel.write_bytes(b"")
        cache_sentinel.chmod(0o444)
        receipt_cache.chmod(0o555)
        configuration_files = {
            name: file_record(path, self.repo)
            for name, path in cuda_math["configuration_files"].items()
        }
        configuration_files.update(
            {
                "qualification_fontconfig": file_record(
                    build_root / "qualification-fontconfig.conf", self.repo
                ),
                "qualification_identity_before": file_record(
                    identity_before, self.repo
                ),
                "qualification_identity_after": file_record(
                    identity_after, self.repo
                ),
                "qualification_contract_v2": file_record(seal, self.repo),
            }
        )
        if not set(REQUIRED_CONFIGURATION_BINDINGS).issubset(
            configuration_files
        ):
            raise AssertionError("archive fixture configuration set is stale")
        receipt = {
            "schema_version": 1,
            "state": "complete",
            "build_kind": "cuda-mpi-python-fp32",
            "started_at_utc": "2000-01-01T00:00:00Z",
            "completed_at_utc": "2000-01-01T00:00:01Z",
            "repo": str(self.repo.resolve()),
            "build_dir": str(build_root.resolve()),
            "source_start": self.snapshot,
            "source_end": self.snapshot,
            "source_unchanged": True,
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip(),
            "git_status_porcelain": subprocess.run(
                [
                    "git", "status", "--porcelain=v1",
                    "--untracked-files=all",
                ],
                cwd=self.repo,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.splitlines(),
            "configuration": configuration,
            "configuration_files": configuration_files,
            "toolchain": {
                "nvcc": file_record(cuda_math["nvcc"], self.repo),
                "sh": file_record(tool, self.repo),
                "python": file_record(python, self.repo),
            },
            "artifacts": {
                name: file_record(path, self.repo)
                for name, path in artifact_paths.items()
            },
            "manifests": {
                "in_place_python": tree_manifest(
                    build_root / "python", self.repo
                ),
                "build_home": tree_manifest(
                    self.repo / ".micromamba/cache/build-home-fixture", self.repo
                ),
                "qualification_home": tree_manifest(
                    build_root / "qualification-home", self.repo
                ),
                "installed_environment": tree_manifest(
                    self.repo / ".envs/meep-gpu-cuda-mpi", self.repo,
                    excluded_suffixes=(".pyc",),
                ),
                "installed_prefix": tree_manifest(install_prefix, self.repo),
                "qualification_logs": tree_manifest(log_root, self.repo),
            },
            "immutable_directories": {
                "receipt_cache": immutable_directory_record(
                    receipt_cache, self.repo
                )
            },
        }
        receipt["build_input_id"] = canonical_sha256(
            {
                "schema_version": receipt["schema_version"],
                "build_kind": receipt["build_kind"],
                "source_start": receipt["source_start"],
                "configuration": receipt["configuration"],
            }
        )
        receipt["artifact_set_id"] = canonical_sha256(
            {
                "configuration_files": receipt["configuration_files"],
                "toolchain": receipt["toolchain"],
                "artifacts": receipt["artifacts"],
                "manifests": receipt["manifests"],
                "immutable_directories": receipt["immutable_directories"],
            }
        )
        receipt["receipt_id"] = canonical_sha256(receipt)
        write_json(self.receipt, receipt)
        self.receipt_value = receipt

    def _runtime_contract(self) -> dict:
        def absolute_record(record: dict) -> dict:
            value = copy.deepcopy(record)
            value["path"] = str((self.repo / value["path"]).resolve())
            return value

        receipt = self.receipt_value
        in_place = receipt["manifests"]["in_place_python"]
        module = next(
            record
            for record in in_place["files"]
            if record["path"] == "meep/__init__.py"
        )
        module = {
            "path": str(
                (self.repo / in_place["root"] / module["path"]).resolve()
            ),
            "size_bytes": module["size_bytes"],
            "sha256": module["sha256"],
        }
        return {
            "python_executable": absolute_record(receipt["toolchain"]["python"]),
            "meep_module": module,
            "extension": absolute_record(receipt["artifacts"]["python_extension"]),
            "libmeep": absolute_record(receipt["artifacts"]["libmeep"]),
            "fontconfig_file": absolute_record(
                receipt["configuration_files"]["qualification_fontconfig"]
            ),
            "build_home": str(
                (self.repo / receipt["manifests"]["build_home"]["root"]).resolve()
            ),
            "qualification_home": str(
                (
                    self.repo
                    / receipt["manifests"]["qualification_home"]["root"]
                ).resolve()
            ),
            "installed_environment": str(
                (
                    self.repo
                    / receipt["manifests"]["installed_environment"]["root"]
                ).resolve()
            ),
            "receipt_id": self.receipt_value["receipt_id"],
        }

    def _run_record(self, backend: str) -> dict:
        directory = self.output / "runs" / self.case_directory / backend
        directory.mkdir(parents=True)
        stdout = directory / "stdout.log"
        stderr = directory / "stderr.log"
        statistics = directory / "gpu-statistics.json"
        stdout.write_text(
            "gpmeep-metrics:{\"value\":1.0}\n", encoding="utf-8"
        )
        stderr.write_text("", encoding="utf-8")
        nonce = ("c" if backend == "cpu" else "d") * 32
        runtime_contract = self._runtime_contract()
        dispatch = {
            "cpu_curl_calls": 1 if backend == "cpu" else 0,
            "cuda_curl_calls": 1 if backend == "cuda" else 0,
        }
        stats_value = {
            "capture_status": "captured",
            "gpu_api_available": True,
            "run_nonce": nonce,
            "build_receipt_id": self.receipt_value["receipt_id"],
            "single_precision": True,
            "with_mpi": True,
            "gpu_compiled": True,
            "strict_cuda_marker": backend == "cuda",
            "python_executable": runtime_contract["python_executable"],
            "meep_module": runtime_contract["meep_module"],
            "extension": runtime_contract["extension"],
            "libmeep": runtime_contract["libmeep"],
            "statistics": {"dispatch": dispatch},
            "requested_backend": backend,
            "active_backend": backend,
            "runtime_available": True,
            "selected_device": 0 if backend == "cuda" else None,
        }
        write_json(statistics, stats_value)
        runtime = directory / "runtime" / nonce
        (runtime / "cache" / "fontconfig").mkdir(parents=True)
        (runtime / "cache" / "fontconfig" / "cache-9").write_text(
            "derived cache\n", encoding="utf-8"
        )
        (runtime / "matplotlib").mkdir()
        (runtime / "matplotlib" / "fontlist.json").write_text(
            "derived font list\n", encoding="utf-8"
        )
        (runtime / "home").mkdir()
        (runtime / "config").mkdir()
        (directory / "work").mkdir()
        evidence = {
            "stdout": {"available": True, **file_record(stdout, self.repo)},
            "stderr": {"available": True, **file_record(stderr, self.repo)},
            "statistics": {
                "available": True,
                **file_record(statistics, self.repo),
            },
        }
        runtime_root = runtime.resolve()
        case_root = directory.resolve()
        build_python = (
            self.repo / "build/meep-cuda-mpi-python-fp32/python"
        ).resolve()
        environment = {
            "PYTHONPATH": os.pathsep.join(
                (
                    str((self.repo / "scripts/python-validation").resolve()),
                    str(build_python),
                )
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(runtime_root / "home"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
            "XDG_CACHE_HOME": str(runtime_root / "cache"),
            "XDG_CONFIG_HOME": str(runtime_root / "config"),
            "FONTCONFIG_FILE": runtime_contract["fontconfig_file"]["path"],
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "MEEP_GPU_BACKEND": backend,
            "GPMEEP_VALIDATION_STATS_FILE": str(case_root / "gpu-statistics.json"),
            "GPMEEP_VALIDATION_EXPECTED_BACKEND": backend,
            "GPMEEP_VALIDATION_RUN_NONCE": nonce,
            "GPMEEP_VALIDATION_BUILD_RECEIPT_ID": self.receipt_value["receipt_id"],
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    str(pathlib.Path(runtime_contract["libmeep"]["path"]).parent),
                    str(pathlib.Path(runtime_contract["installed_environment"]) / "lib"),
                )
            ),
        }
        if backend == "cuda":
            environment["GPMEEP_VALIDATION_STRICT_CUDA"] = "1"
        environment_contract = {
            "MEEP_GPU_BACKEND": backend,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "HOME": environment["HOME"],
            "MPLCONFIGDIR": environment["MPLCONFIGDIR"],
            "XDG_CACHE_HOME": environment["XDG_CACHE_HOME"],
            "XDG_CONFIG_HOME": environment["XDG_CONFIG_HOME"],
            "FONTCONFIG_FILE": environment["FONTCONFIG_FILE"],
            "strict_cuda": backend == "cuda",
            "keys": sorted(environment),
            "sha256": sealer.runner_contract.environment_sha256(environment),
            "environment": dict(sorted(environment.items())),
        }
        return {
            "backend": backend,
            "command": [
                self._runtime_contract()["python_executable"]["path"],
                str((self.repo / "cases/example.py").resolve()),
            ],
            "cwd": f"runs/{self.case_directory}/{backend}/work",
            "evidence_case_directory": self.case_directory,
            "environment_contract": environment_contract,
            "started_at_utc": "2000-01-01T00:00:00Z",
            "run_nonce": nonce,
            "stdout_log": str(stdout.resolve()),
            "stderr_log": str(stderr.resolve()),
            "statistics_file": str(statistics.resolve()),
            "evidence_files": evidence,
            "statistics": json.loads(statistics.read_text()),
            "backend_contract_ok": True,
            "backend_contract_problems": [],
            "duration_seconds": 1.0,
            "exit_code": 0,
            "timeout": False,
            "output_limit": None,
            "output_sizes": {
                "stdout": stdout.stat().st_size,
                "stderr": stderr.stat().st_size,
                "combined": stdout.stat().st_size + stderr.stat().st_size,
            },
            "output_limits": {
                "stdout": sealer.runner_contract.MAX_BACKEND_STDOUT_BYTES,
                "stderr": sealer.runner_contract.MAX_BACKEND_STDERR_BYTES,
                "combined": sealer.runner_contract.MAX_BACKEND_OUTPUT_BYTES,
            },
            "unittest_skips": 0,
            "unittest_skip_details": [],
            "allowed_unittest_skips": 0,
            "unittest_skip_policy_problems": [],
            "unittest_reported_test_counts": [],
            "unittest_test_count": None,
            "unittest_test_identities": [],
            "expected_unittest_test_count": None,
            "expected_unittest_test_identities": None,
            "unittest_test_contract_ok": True,
            "unittest_test_contract_problems": [],
            "unittest_terminal_summaries": [],
            "outcome": "PASS",
        }

    def _write_runner_output(self) -> None:
        self.output.mkdir()
        cpu = self._run_record("cpu")
        cuda = self._run_record("cuda")
        compact_snapshot = {
            "available": True,
            "algorithm": self.snapshot["algorithm"],
            "file_count": self.snapshot["file_count"],
            "sha256": self.snapshot["sha256"],
        }
        receipt_record = {
            "available": True,
            "path": str(self.receipt.resolve()),
            "sha256": sha256_file(self.receipt),
            "receipt_id": self.receipt_value["receipt_id"],
            "build_input_id": self.receipt_value["build_input_id"],
            "artifact_set_id": self.receipt_value["artifact_set_id"],
        }
        endpoint = {
            "captured_at_utc": "2000-01-01T00:00:00Z",
            "available": True,
            "source_snapshot": compact_snapshot,
            "build_extension": {
                "available": True,
                **self._runtime_contract()["extension"],
            },
            "build_receipt": receipt_record,
            "runtime_contract": self._runtime_contract(),
            "problems": [],
        }
        comparison_policy = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"value": {"rtol": 0.0, "atol": 0.0}},
        }
        comparison = {
            "mode": "json_metrics",
            "cpu_seconds": 1.0,
            "cuda_seconds": 1.0,
            "speedup_cpu_over_cuda": 1.0,
            **sealer.runner_contract.compare_json_metrics(
                "gpmeep-metrics:{\"value\":1.0}\n",
                "gpmeep-metrics:{\"value\":1.0}\n",
                comparison_policy,
            ),
            "raw_metric_evidence": {
                "cpu_stdout": cpu["evidence_files"]["stdout"],
                "cuda_stdout": cuda["evidence_files"]["stdout"],
            },
        }
        result = {
            "id": "cases/example.py",
            "path": "cases/example.py",
            "kind": "example",
            "tier": ["full"],
            "disposition": "run",
            "compute_scope": "fdtd_cuda",
            "reason": "archive fixture",
            "milestone": None,
            "selected": True,
            "outcome": "PASS",
            "runs": {"cpu": cpu, "cuda": cuda},
            "comparison": comparison,
        }
        self.report = {
            "schema_version": sealer.RUNNER_REPORT_SCHEMA_VERSION,
            "generated_at_utc": "2000-01-01T00:00:01Z",
            "exit_code": 0,
            "exit_code_meaning": {
                str(code): f"fixture exit {code}"
                for code in sorted(sealer.RUNNER_EXIT_CODES)
            },
            "configuration": {
                "tiers": ["full"],
                "case_ids": [],
                "backends": ["cpu", "cuda"],
                "performance_evidence": {
                    "valid_for_speed_gate": False,
                    "concurrency_detected": False,
                    "invalid_reasons": [
                        "whole-process single-sample timing includes Python startup",
                        "no benchmark warm-up or repeated-sample statistics",
                    ],
                },
            },
            "inventory": {
                "total": 1,
                "selected": 1,
                "scheme_scope": "excluded by project requirement",
            },
            "summary": {"PASS": 1},
            "provenance": {
                "captured_at_utc": "2000-01-01T00:00:00Z",
                "hostname": "fixture",
                "platform": "fixture",
                "machine": "fixture",
                "repository": str(self.repo.resolve()),
                "python_executable": self._runtime_contract()[
                    "python_executable"
                ]["path"],
                "build_python": str(
                    (
                        self.repo / "build/meep-cuda-mpi-python-fp32/python"
                    ).resolve()
                ),
                "manifest": str(
                    (
                        self.repo
                        / "scripts/python-validation/manifest.json"
                    ).resolve()
                ),
                "manifest_sha256": sha256_file(
                    self.repo / "scripts/python-validation/manifest.json"
                ),
                "python_version": None,
                "git_head": None,
                "git_status": None,
                "source_snapshot": compact_snapshot,
                "build_python_exists": True,
                "build_extensions": [],
                "build_package_probe": None,
                "install_prefix": str(
                    self.repo / "install/meep-cuda-mpi-python-fp32"
                ),
                "install_extensions": [],
                "install_package_probe": None,
                "source_files": [],
                "environment_selection": {
                    "CONDA_PREFIX": None,
                    "CUDA_DEVICE_ORDER": None,
                    "CUDA_VISIBLE_DEVICES": None,
                    "MEEP_GPU_DEVICE": None,
                },
                "validation_window": {
                    "start": copy.deepcopy(endpoint),
                    "end": copy.deepcopy(endpoint),
                    "unchanged": True,
                    "problems": [],
                },
            },
            "results": [result],
        }
        (self.output / "report.md").write_text("# fixture report\n", encoding="utf-8")
        self.reseal_report()

    def reseal_report(self) -> None:
        write_json(self.output / "report.json", self.report)
        markdown = self.output / "report.md"
        write_json(
            self.output / "COMPLETE",
            {
                "schema_version": self.report["schema_version"],
                "completed_at_utc": "2000-01-01T00:00:02Z",
                "exit_code": self.report["exit_code"],
                "report_sha256": sha256_file(self.output / "report.json"),
                "report_size_bytes": (self.output / "report.json").stat().st_size,
                "markdown_sha256": sha256_file(markdown),
                "markdown_size_bytes": markdown.stat().st_size,
            },
        )

    def refresh_source_and_receipt_bindings(self) -> None:
        self.snapshot = source_snapshot(self.repo)
        self._write_receipt()
        for backend, run in self.report["results"][0]["runs"].items():
            stats_path = pathlib.Path(run["statistics_file"])
            stats = json.loads(stats_path.read_text())
            stats["build_receipt_id"] = self.receipt_value["receipt_id"]
            write_json(stats_path, stats)
            run["statistics"] = stats
            run["evidence_files"]["statistics"] = {
                "available": True,
                **file_record(stats_path, self.repo),
            }
            environment = run["environment_contract"]["environment"]
            environment["GPMEEP_VALIDATION_BUILD_RECEIPT_ID"] = (
                self.receipt_value["receipt_id"]
            )
            run["environment_contract"]["keys"] = sorted(environment)
            run["environment_contract"]["sha256"] = (
                sealer.runner_contract.environment_sha256(environment)
            )
        compact_snapshot = {
            "available": True,
            "algorithm": self.snapshot["algorithm"],
            "file_count": self.snapshot["file_count"],
            "sha256": self.snapshot["sha256"],
        }
        receipt_record = {
            "available": True,
            "path": str(self.receipt.resolve()),
            "sha256": sha256_file(self.receipt),
            "receipt_id": self.receipt_value["receipt_id"],
            "build_input_id": self.receipt_value["build_input_id"],
            "artifact_set_id": self.receipt_value["artifact_set_id"],
        }
        endpoint = {
            "captured_at_utc": "2000-01-01T00:00:00Z",
            "available": True,
            "source_snapshot": compact_snapshot,
            "build_extension": {
                "available": True,
                **self._runtime_contract()["extension"],
            },
            "build_receipt": receipt_record,
            "runtime_contract": self._runtime_contract(),
            "problems": [],
        }
        provenance = self.report["provenance"]
        provenance["source_snapshot"] = compact_snapshot
        provenance["validation_window"] = {
            "start": copy.deepcopy(endpoint),
            "end": copy.deepcopy(endpoint),
            "unchanged": True,
            "problems": [],
        }
        self.reseal_report()

    def add_covered_case(self, *, select_target: bool) -> None:
        """Extend the fixture with one coverage-only case and reseal inputs."""

        covered_path = "cases/zcovered.py"
        covered_reason = "coverage replay fixture"
        source = self.repo / covered_path
        source.write_text("# covered by cases/example.py\n", encoding="utf-8")
        manifest_path = (
            self.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["overrides"].append(
            {
                "glob": covered_path,
                "disposition": "covered_by_test",
                "reason": covered_reason,
                "compute_scope": "coverage_only",
                "covered_by": ["cases/example.py"],
                "coverage_relation": "strict_superset",
            }
        )
        write_json(manifest_path, manifest)
        subprocess.run(
            ["git", "add", covered_path, "scripts/python-validation/manifest.json"],
            cwd=self.repo,
            check=True,
        )

        target = self.report["results"][0]
        if not select_target:
            target["selected"] = False
            target["outcome"] = "NOT_SELECTED"
            target["runs"] = {}
            target["comparison"] = None
            shutil.rmtree(self.output / "runs" / self.case_directory)
            self.report["configuration"]["case_ids"] = [covered_path]
        covered_outcome = (
            "PASS_COVERED" if select_target else "COVERAGE_UNRESOLVED"
        )
        self.report["results"].append(
            {
                "id": covered_path,
                "path": covered_path,
                "kind": "example",
                "tier": ["full"],
                "disposition": "covered_by_test",
                "compute_scope": "coverage_only",
                "reason": covered_reason,
                "milestone": None,
                "selected": True,
                "outcome": covered_outcome,
                "runs": {},
                "comparison": None,
            }
        )
        self.report["results"].sort(key=lambda value: value["path"])
        self.report["inventory"] = {
            "total": 2,
            "selected": 2 if select_target else 1,
            "scheme_scope": "excluded by project requirement",
        }
        self.report["summary"] = (
            {"PASS": 1, "PASS_COVERED": 1}
            if select_target
            else {"COVERAGE_UNRESOLVED": 1, "NOT_SELECTED": 1}
        )
        self.report["exit_code"] = (
            sealer.runner_contract.EXIT_OK
            if select_target
            else sealer.runner_contract.EXIT_BLOCKED
        )
        self.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        self.refresh_source_and_receipt_bindings()

    def fail_covered_target(self) -> None:
        target = next(
            result
            for result in self.report["results"]
            if result["path"] == "cases/example.py"
        )
        target["runs"]["cuda"]["exit_code"] = 7
        target["runs"]["cuda"]["outcome"] = "PROCESS_FAILED"
        target["comparison"] = {
            "mode": "json_metrics",
            "outcome": "NOT_COMPARABLE",
            "reason": "backend runs did not pass: cuda",
        }
        target["outcome"] = "PROCESS_FAILED"
        covered = next(
            result
            for result in self.report["results"]
            if result["path"] == "cases/zcovered.py"
        )
        covered["outcome"] = "COVERAGE_FAILED"
        self.report["summary"] = {
            "COVERAGE_FAILED": 1,
            "PROCESS_FAILED": 1,
        }
        self.report["exit_code"] = sealer.runner_contract.EXIT_BLOCKED
        self.reseal_report()


class ArchiveValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.fixture = ArchiveFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def seal(self, archive: pathlib.Path | None = None) -> pathlib.Path:
        archive = archive or self.root / "archive"
        sealer.seal_archive(
            self.fixture.output,
            archive,
            self.fixture.repo,
            self.fixture.receipt,
        )
        return archive

    def test_checked_manifest_command_wrappers_are_fixed_archive_inputs(self):
        repo = SEALER_PATH.parents[2]
        manifest = json.loads(
            (repo / "scripts/python-validation/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        required = {
            pathlib.PurePosixPath(token.removeprefix("{repo}/"))
            for override in manifest["overrides"]
            for token in override.get("command", [])
            if token.startswith("{repo}/scripts/")
        }
        archived = set(sealer.FIXED_REPO_INPUTS.values())
        self.assertTrue(required)
        self.assertSetEqual(required - archived, set())

    def rewrite_manifest_and_complete(
        self, archive: pathlib.Path, mutate
    ) -> None:
        manifest_path = archive / sealer.MANIFEST_NAME
        complete_path = archive / sealer.COMPLETE_NAME
        manifest = json.loads(manifest_path.read_text())
        complete = json.loads(complete_path.read_text())
        mutate(manifest)
        write_json(manifest_path, manifest)
        complete["archive_manifest_sha256"] = sha256_file(manifest_path)
        write_json(complete_path, complete)

    def rebind_archive_payload(self, archive: pathlib.Path) -> None:
        manifest_path = archive / sealer.MANIFEST_NAME
        complete_path = archive / sealer.COMPLETE_NAME
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            payload = archive / record["path"]
            record["size_bytes"] = payload.stat().st_size
            record["sha256"] = sha256_file(payload)
        manifest["payload_sha256"] = canonical_sha256(manifest["files"])
        manifest["file_count"] = len(manifest["files"])
        write_json(manifest_path, manifest)
        complete = json.loads(complete_path.read_text())
        complete["archive_manifest_sha256"] = sha256_file(manifest_path)
        complete["payload_sha256"] = manifest["payload_sha256"]
        complete["payload_file_count"] = len(manifest["files"])
        complete["report_sha256"] = sha256_file(
            archive / "runner-output/report.json"
        )
        complete["runner_complete_sha256"] = sha256_file(
            archive / "runner-output/COMPLETE"
        )
        write_json(complete_path, complete)

    def rebind_report_attack(self, archive: pathlib.Path, mutate) -> None:
        report_path = archive / "runner-output/report.json"
        runner_complete_path = archive / "runner-output/COMPLETE"
        report = json.loads(report_path.read_text())
        mutate(report)
        write_json(report_path, report)
        runner_complete = json.loads(runner_complete_path.read_text())
        runner_complete["schema_version"] = report["schema_version"]
        runner_complete["exit_code"] = report["exit_code"]
        runner_complete["report_sha256"] = sha256_file(report_path)
        runner_complete["report_size_bytes"] = report_path.stat().st_size
        write_json(runner_complete_path, runner_complete)
        manifest_path = archive / sealer.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        source = manifest["source_contract"]
        source["runner_report_schema_version"] = report["schema_version"]
        source["runner_exit_code"] = report["exit_code"]
        source["report_sha256"] = sha256_file(report_path)
        source["runner_complete_sha256"] = sha256_file(runner_complete_path)
        write_json(manifest_path, manifest)
        self.rebind_archive_payload(archive)

    def test_seals_relocatable_complete_archive_and_excludes_runtime_cache(self):
        archive = self.seal()
        complete = sealer.verify_archive(archive)
        self.assertEqual(complete["state"], "COMPLETE")
        manifest = json.loads((archive / sealer.MANIFEST_NAME).read_text())
        self.assertEqual(
            manifest["source_contract"]["raw_evidence_file_count"], 6
        )
        self.assertEqual(
            manifest["runtime_cache_exclusion"]["file_count"], 4
        )
        self.assertFalse(any("/runtime/" in item["path"] for item in manifest["files"]))
        expected = {
            "runner-output/report.json",
            "runner-output/report.md",
            "runner-output/COMPLETE",
            "inputs/manifest.json",
            "inputs/run_validation.py",
            "inputs/run_example_oracle.py",
            "inputs/archive_validation.py",
            "inputs/gpmeep_provenance.py",
            "inputs/gpmeep_qualification_contract.py",
            "inputs/sitecustomize.py",
            "inputs/absorber_branch_matrix.py",
            "inputs/perturbation_branch_matrix.py",
            "inputs/run_point_dipole_cyl_validation.py",
            "inputs/stochastic_branch_matrix.py",
            "inputs/stochastic_line_basis_matrix.py",
            "inputs/stochastic_reciprocity_matrix.py",
            "inputs/build-meep-cuda-mpi-python.sh",
            "inputs/build-provenance.json",
            "sources/cases/example.py",
        }
        paths = {item["path"] for item in manifest["files"]}
        self.assertTrue(expected.issubset(paths))
        tar_metadata = manifest["deterministic_tar"]
        self.assertIn("--sort=name", tar_metadata["command_argv_template"])
        self.assertIn("--mtime=@0", tar_metadata["command_argv_template"])
        self.assertNotIn(str(self.root), json.dumps(tar_metadata))

        relocated_archive = self.root / "relocated-archive"
        shutil.copytree(archive, relocated_archive)
        self.assertEqual(
            sealer.verify_archive(relocated_archive)["report_sha256"],
            complete["report_sha256"],
        )

    def test_replays_pass_covered_from_terminal_target(self):
        self.fixture.add_covered_case(select_target=True)
        archive = self.seal()
        self.assertEqual(sealer.verify_archive(archive)["state"], "COMPLETE")

    def test_rejects_rebound_pass_covered_with_unselected_target(self):
        self.fixture.add_covered_case(select_target=False)
        archive = self.seal()
        self.assertEqual(sealer.verify_archive(archive)["state"], "COMPLETE")

        def forge_coverage(report):
            covered = next(
                result
                for result in report["results"]
                if result["path"] == "cases/zcovered.py"
            )
            covered["outcome"] = "PASS_COVERED"
            report["summary"] = {"NOT_SELECTED": 1, "PASS_COVERED": 1}
            report["exit_code"] = sealer.runner_contract.EXIT_OK

        self.rebind_report_attack(archive, forge_coverage)
        with self.assertRaisesRegex(
            sealer.ArchiveError, "covered case .* outcome differs"
        ):
            sealer.verify_archive(archive)

    def test_rejects_rebound_pass_covered_with_failed_target(self):
        self.fixture.add_covered_case(select_target=True)
        self.fixture.fail_covered_target()
        archive = self.seal()
        self.assertEqual(sealer.verify_archive(archive)["state"], "COMPLETE")

        def forge_coverage(report):
            covered = next(
                result
                for result in report["results"]
                if result["path"] == "cases/zcovered.py"
            )
            covered["outcome"] = "PASS_COVERED"
            report["summary"] = {"PASS_COVERED": 1, "PROCESS_FAILED": 1}
            report["exit_code"] = sealer.runner_contract.EXIT_EXECUTION_FAILED

        self.rebind_report_attack(archive, forge_coverage)
        with self.assertRaisesRegex(
            sealer.ArchiveError, "covered case .* outcome differs"
        ):
            sealer.verify_archive(archive)

    def test_rejects_rebound_metric_count_and_summary_attacks(self):
        attacks = {
            "metric-count": (
                lambda report: report["results"][0]["comparison"][
                    "metric_evidence"
                ].__setitem__("count", 7),
                "comparison replay",
            ),
            "summary": (
                lambda report: report.__setitem__("summary", {"PASS": 99}),
                "summary",
            ),
            "embedded-statistics": (
                lambda report: report["results"][0]["runs"]["cuda"][
                    "statistics"
                ].__setitem__("active_backend", "cpu"),
                "statistics",
            ),
        }
        for label, (mutate, message) in attacks.items():
            with self.subTest(label=label):
                archive = self.seal(self.root / f"archive-{label}")
                self.rebind_report_attack(archive, mutate)
                with self.assertRaisesRegex(sealer.ArchiveError, message):
                    sealer.verify_archive(archive)

    def test_seals_declared_backend_release_after_proven_cuda_dispatch(self):
        manifest_path = (
            self.fixture.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["inventory"][0]["require_active_backend_at_exit"] = False
        write_json(manifest_path, manifest)
        self.fixture.refresh_source_and_receipt_bindings()
        self.fixture.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )

        run = self.fixture.report["results"][0]["runs"]["cuda"]
        statistics_path = pathlib.Path(run["statistics_file"])
        statistics = json.loads(statistics_path.read_text())
        statistics["requested_backend"] = "cpu"
        statistics["active_backend"] = "cpu"
        statistics["selected_device"] = -1
        write_json(statistics_path, statistics)
        run["statistics"] = statistics
        run["evidence_files"]["statistics"] = {
            "available": True,
            **file_record(statistics_path, self.fixture.repo),
        }
        self.fixture.reseal_report()

        archive = self.seal()
        complete = sealer.verify_archive(archive)
        self.assertEqual(complete["state"], "COMPLETE")

    def test_rejects_nonboolean_backend_release_waiver_during_seal(self):
        manifest_path = (
            self.fixture.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["inventory"][0]["require_active_backend_at_exit"] = 0
        write_json(manifest_path, manifest)
        self.fixture.refresh_source_and_receipt_bindings()
        self.fixture.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        self.fixture.reseal_report()

        with self.assertRaisesRegex(
            sealer.ArchiveError, "must be a boolean"
        ):
            self.seal()

    def test_rejects_unpinned_no_dispatch_backend_release_during_seal(self):
        manifest_path = (
            self.fixture.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        case = manifest["inventory"][0]
        case["require_active_backend_at_exit"] = False
        case["gpu_contract"] = "none"
        write_json(manifest_path, manifest)
        self.fixture.refresh_source_and_receipt_bindings()
        self.fixture.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        self.fixture.reseal_report()

        with self.assertRaisesRegex(
            sealer.ArchiveError, "requires pinned expected_unittest identities"
        ):
            self.seal()

    def test_seals_pinned_unittest_backend_release_without_dispatch_gate(self):
        manifest_path = (
            self.fixture.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        case = manifest["inventory"][0]
        case["kind"] = "unittest"
        case["require_active_backend_at_exit"] = False
        case["gpu_contract"] = "none"
        identity = "cases.Example.test_expected"
        case["expected_unittest"] = {
            "count": 1,
            "identities": [identity],
        }
        write_json(manifest_path, manifest)

        result = self.fixture.report["results"][0]
        result["kind"] = "unittest"
        for backend, run in result["runs"].items():
            run["command"] = [
                self.fixture._runtime_contract()["python_executable"]["path"],
                "-m",
                "unittest",
                "discover",
                "-v",
                "-s",
                str((self.fixture.repo / "cases").resolve()),
                "-p",
                "example.py",
            ]
            stderr = pathlib.Path(run["stderr_log"])
            stderr.write_text(
                "test_expected (cases.Example.test_expected) ... ok\n\n"
                "----------------------------------------------------------------------\n"
                "Ran 1 test in 0.001s\n\nOK\n",
                encoding="utf-8",
            )
            run["evidence_files"]["stderr"] = {
                "available": True,
                **file_record(stderr, self.fixture.repo),
            }
            run["output_sizes"]["stderr"] = stderr.stat().st_size
            run["output_sizes"]["combined"] = (
                run["output_sizes"]["stdout"] + stderr.stat().st_size
            )
            run["unittest_reported_test_counts"] = [1]
            run["unittest_test_count"] = 1
            run["unittest_test_identities"] = [identity]
            run["expected_unittest_test_count"] = 1
            run["expected_unittest_test_identities"] = [identity]
            run["unittest_terminal_summaries"] = ["OK"]

        cuda_run = result["runs"]["cuda"]
        statistics_path = pathlib.Path(cuda_run["statistics_file"])
        statistics = json.loads(statistics_path.read_text())
        statistics["requested_backend"] = "cpu"
        statistics["active_backend"] = "cpu"
        statistics["selected_device"] = -1
        write_json(statistics_path, statistics)
        cuda_run["statistics"] = statistics
        cuda_run["evidence_files"]["statistics"] = {
            "available": True,
            **file_record(statistics_path, self.fixture.repo),
        }

        self.fixture.refresh_source_and_receipt_bindings()
        self.fixture.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        self.fixture.reseal_report()

        archive = self.seal()
        complete = sealer.verify_archive(archive)
        self.assertEqual(complete["state"], "COMPLETE")

    def test_successful_json_metrics_requires_raw_metric_binding(self):
        del self.fixture.report["results"][0]["comparison"][
            "raw_metric_evidence"
        ]
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "mandatory raw"):
            self.seal()

    def test_json_metric_archive_replay_never_materializes_raw_stdout(self):
        result = self.fixture.report["results"][0]
        protected = {
            pathlib.Path(result["runs"][backend]["stdout_log"]).resolve()
            for backend in ("cpu", "cuda")
        }
        original_read_text = pathlib.Path.read_text

        def guarded_read_text(path, *args, **kwargs):
            if pathlib.Path(path).resolve() in protected:
                raise AssertionError(
                    "json_metrics archive replay materialized raw stdout"
                )
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(
            pathlib.Path, "read_text", guarded_read_text
        ), mock.patch.object(
            sealer.runner_contract,
            "compare_json_metrics",
            side_effect=AssertionError(
                "archive replay used the unbounded in-memory comparator"
            ),
        ):
            self.seal()

    def test_invalid_json_metric_archive_replays_raw_bound_mismatch(self):
        result = self.fixture.report["results"][0]
        cpu = result["runs"]["cpu"]
        cpu_stdout = pathlib.Path(cpu["stdout_log"])
        cpu_stdout.write_text(
            'gpmeep-metrics:{"value":1e999}\n', encoding="utf-8"
        )
        cpu_record = {
            "available": True,
            **file_record(cpu_stdout, self.fixture.repo),
        }
        cpu["evidence_files"]["stdout"] = cpu_record
        cpu["output_sizes"]["stdout"] = cpu_stdout.stat().st_size
        cpu["output_sizes"]["combined"] = (
            cpu["output_sizes"]["stdout"] + cpu["output_sizes"]["stderr"]
        )
        comparison = {
            "mode": "json_metrics",
            "cpu_seconds": 1.0,
            "cuda_seconds": 1.0,
            "speedup_cpu_over_cuda": 1.0,
            "outcome": "MISMATCH",
            "reason": "json_metrics value 'value' must be finite",
            "raw_metric_evidence": {
                "cpu_stdout": cpu_record,
                "cuda_stdout": result["runs"]["cuda"]["evidence_files"][
                    "stdout"
                ],
            },
        }
        result["comparison"] = comparison
        result["outcome"] = "MISMATCH"
        self.fixture.report["summary"] = {"MISMATCH": 1}
        self.fixture.report["exit_code"] = 3
        self.fixture.reseal_report()

        self.seal()

    def test_runner_schema_downgrade_upgrade_and_extra_fields_fail_closed(self):
        complete = json.loads((self.fixture.output / "COMPLETE").read_text())
        current = sealer.RUNNER_REPORT_SCHEMA_VERSION
        for schema in (0, 1, current - 1, current + 1, str(current), True):
            with self.subTest(schema=schema):
                report = copy.deepcopy(self.fixture.report)
                candidate_complete = copy.deepcopy(complete)
                report["schema_version"] = schema
                candidate_complete["schema_version"] = schema
                with self.assertRaisesRegex(sealer.ArchiveError, "schema"):
                    sealer._validate_runner_control(report, candidate_complete)
        for target in ("report", "complete"):
            with self.subTest(extra=target):
                report = copy.deepcopy(self.fixture.report)
                candidate_complete = copy.deepcopy(complete)
                (report if target == "report" else candidate_complete)[
                    "forged"
                ] = True
                with self.assertRaisesRegex(
                    sealer.ArchiveError, "unexpected or missing"
                ):
                    sealer._validate_runner_control(report, candidate_complete)
        nested_attacks = {
            "configuration": lambda report: report["configuration"].__setitem__(
                "forged", True
            ),
            "performance": lambda report: report["configuration"][
                "performance_evidence"
            ].__setitem__("forged", True),
            "provenance": lambda report: report["provenance"].__setitem__(
                "forged", True
            ),
            "result": lambda report: report["results"][0].__setitem__(
                "forged", True
            ),
            "endpoint": lambda report: report["provenance"][
                "validation_window"
            ]["start"].__setitem__("forged", True),
        }
        for label, mutate in nested_attacks.items():
            with self.subTest(nested=label):
                report = copy.deepcopy(self.fixture.report)
                candidate_complete = copy.deepcopy(complete)
                mutate(report)
                with self.assertRaisesRegex(sealer.ArchiveError, "fields differ"):
                    sealer._validate_runner_control(report, candidate_complete)

    def test_rejects_tampered_raw_input_before_publication(self):
        raw = (
            self.fixture.output
            / "runs"
            / self.fixture.case_directory
            / "cuda/stdout.log"
        )
        raw.write_text(raw.read_text() + "tampered\n", encoding="utf-8")
        archive = self.root / "archive"
        with self.assertRaisesRegex(sealer.ArchiveError, "raw evidence differs"):
            self.seal(archive)
        self.assertFalse((archive / sealer.COMPLETE_NAME).exists())

    def test_archives_json_metric_failure_sidecar(self):
        comparison_dir = (
            self.fixture.output
            / "runs"
            / self.fixture.case_directory
            / "comparison"
        )
        comparison_dir.mkdir()
        sidecar = comparison_dir / "json-metric-failures.ndjson"
        result = self.fixture.report["results"][0]
        cuda = result["runs"]["cuda"]
        cuda_stdout = pathlib.Path(cuda["stdout_log"])
        cuda_stdout.write_text(
            'gpmeep-metrics:{"value":2.0}\n', encoding="utf-8"
        )
        cuda_stdout_record = {
            "available": True,
            **file_record(cuda_stdout, self.fixture.repo),
        }
        cuda["evidence_files"]["stdout"] = cuda_stdout_record
        policy = {
            "mode": "json_metrics",
            "prefix": "gpmeep-metrics:",
            "tolerances": {"value": {"rtol": 0.0, "atol": 0.0}},
        }
        comparison = {
            "mode": "json_metrics",
            "cpu_seconds": 1.0,
            "cuda_seconds": 1.0,
            "speedup_cpu_over_cuda": 1.0,
            **sealer.runner_contract.compare_json_metrics(
                'gpmeep-metrics:{"value":1.0}\n',
                'gpmeep-metrics:{"value":2.0}\n',
                policy,
                failure_sidecar=sidecar,
            ),
            "raw_metric_evidence": {
                "cpu_stdout": result["runs"]["cpu"]["evidence_files"]["stdout"],
                "cuda_stdout": cuda_stdout_record,
            },
        }
        comparison["failure_evidence"] = {
            "available": True,
            **file_record(sidecar, self.fixture.repo),
        }
        result["comparison"] = comparison
        result["outcome"] = "MISMATCH"
        self.fixture.report["summary"] = {"MISMATCH": 1}
        self.fixture.report["exit_code"] = 3
        self.fixture.reseal_report()

        archive = self.seal()
        manifest = json.loads((archive / sealer.MANIFEST_NAME).read_text())
        paths = {record["path"] for record in manifest["files"]}
        self.assertIn(
            f"runner-output/runs/{self.fixture.case_directory}/comparison/"
            "json-metric-failures.ndjson",
            paths,
        )

    def test_rejects_colliding_case_directory_evidence(self):
        duplicate = copy.deepcopy(self.fixture.report["results"][0])
        duplicate["id"] = "different/result.py"
        self.fixture.report["results"].append(duplicate)
        self.fixture.report["inventory"]["selected"] = 2
        self.fixture.report["summary"]["PASS"] = 2
        self.fixture.reseal_report()
        with self.assertRaisesRegex(
            sealer.ArchiveError, "duplicate/colliding raw evidence path"
        ):
            self.seal()

    def test_rejects_symlinked_evidence_even_when_target_bytes_match(self):
        raw = (
            self.fixture.output
            / "runs"
            / self.fixture.case_directory
            / "cpu/stdout.log"
        )
        external = self.root / "external-stdout.log"
        external.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(external)
        with self.assertRaisesRegex(sealer.ArchiveError, "symbolic links"):
            self.seal()

    def test_rejects_existing_destination_without_touching_it(self):
        archive = self.root / "archive"
        archive.mkdir()
        sentinel = archive / "sentinel"
        sentinel.write_text("keep\n", encoding="utf-8")
        with self.assertRaisesRegex(sealer.ArchiveError, "already exists"):
            self.seal(archive)
        self.assertEqual(sentinel.read_text(), "keep\n")
        self.assertFalse((archive / sealer.COMPLETE_NAME).exists())

    def test_fsyncs_payload_directories_before_writing_archive_complete(self):
        events: list[str] = []
        original_write = sealer.atomic_write_json
        original_fsync = sealer._fsync_directories

        def recording_write(path, value):
            events.append(f"write:{path.name}")
            return original_write(path, value)

        def recording_fsync(path):
            events.append("fsync:payload-directories")
            return original_fsync(path)

        with mock.patch.object(
            sealer, "atomic_write_json", side_effect=recording_write
        ), mock.patch.object(
            sealer, "_fsync_directories", side_effect=recording_fsync
        ):
            self.seal()

        manifest_index = events.index(f"write:{sealer.MANIFEST_NAME}")
        complete_index = events.index(f"write:{sealer.COMPLETE_NAME}")
        final_fsync_index = len(events) - 1 - events[::-1].index(
            "fsync:payload-directories"
        )
        self.assertLess(manifest_index, complete_index)
        self.assertLess(complete_index, final_fsync_index)

    def test_cli_seal_and_verify(self):
        archive = self.root / "cli-archive"
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        seal = subprocess.run(
            [
                sys.executable,
                str(SEALER_PATH),
                "seal",
                "--input",
                str(self.fixture.output),
                "--archive",
                str(archive),
                "--repo",
                str(self.fixture.repo),
                "--build-receipt",
                str(self.fixture.receipt),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        self.assertEqual(seal.returncode, 0, seal.stderr)
        self.assertIn("deterministic tar template", seal.stdout)
        verify = subprocess.run(
            [
                sys.executable,
                str(SEALER_PATH),
                "verify",
                "--archive",
                str(archive),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("verified archive", verify.stdout)

    def test_archived_verifier_runs_standalone_with_empty_pythonpath(self):
        real_inputs = {
            SEALER_PATH: (
                self.fixture.repo
                / "scripts/python-validation/archive_validation.py"
            ),
            SEALER_PATH.parent / "run_validation.py": (
                self.fixture.repo
                / "scripts/python-validation/run_validation.py"
            ),
            SEALER_PATH.parents[1] / "gpmeep_provenance.py": (
                self.fixture.repo / "scripts/gpmeep_provenance.py"
            ),
            SEALER_PATH.parents[1] / "gpmeep_qualification_contract.py": (
                self.fixture.repo
                / "scripts/gpmeep_qualification_contract.py"
            ),
        }
        for source, destination in real_inputs.items():
            shutil.copy2(source, destination)
        subprocess.run(["git", "add", "."], cwd=self.fixture.repo, check=True)
        self.fixture.refresh_source_and_receipt_bindings()
        archive = self.seal()
        empty_cwd = self.root / "empty-cwd"
        empty_cwd.mkdir()
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
        }
        before = {
            path.relative_to(archive).as_posix(): (
                path.stat().st_mode,
                path.stat().st_size,
                sha256_file(path) if path.is_file() else None,
            )
            for path in archive.rglob("*")
        }
        verify = subprocess.run(
            [
                sys.executable,
                str(archive / "inputs/archive_validation.py"),
                "verify",
                "--archive",
                str(archive),
            ],
            cwd=empty_cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("verified archive", verify.stdout)
        after = {
            path.relative_to(archive).as_posix(): (
                path.stat().st_mode,
                path.stat().st_size,
                sha256_file(path) if path.is_file() else None,
            )
            for path in archive.rglob("*")
        }
        self.assertEqual(after, before)
        self.assertFalse((archive / "inputs/__pycache__").exists())

    def test_rejects_unreferenced_non_runtime_file(self):
        stale = self.fixture.output / "runs/stale/cpu/stdout.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale\n", encoding="utf-8")
        with self.assertRaisesRegex(sealer.ArchiveError, "unreferenced"):
            self.seal()

    def test_verifier_detects_post_seal_payload_tampering(self):
        archive = self.seal()
        raw = (
            archive
            / "runner-output/runs"
            / self.fixture.case_directory
            / "cpu/stdout.log"
        )
        raw.write_text(raw.read_text() + "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(sealer.ArchiveError, "changed"):
            sealer.verify_archive(archive)

    def test_receipt_inventory_rejects_omitted_glob_match(self):
        omitted = self.fixture.repo / "cases/omitted.py"
        omitted.write_text("print('omitted')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.fixture.repo, check=True)
        self.fixture.refresh_source_and_receipt_bindings()
        with self.assertRaisesRegex(
            sealer.ArchiveError, "inventory differs.*missing"
        ):
            self.seal()

    def test_rejects_unknown_case_selection_even_when_self_consistent(self):
        self.fixture.report["configuration"]["case_ids"] = ["ghost-case-id"]
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "unknown case"):
            self.seal()

    def test_rejects_unknown_tier_empty_success_claim(self):
        self.fixture.report["configuration"]["tiers"] = ["ghost-tier"]
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "unknown tier"):
            self.seal()

    def test_rejects_repository_and_command_rebound_to_report_claim(self):
        forged = pathlib.Path("/attacker/rebound")
        self.fixture.report["provenance"]["repository"] = str(forged)
        for run in self.fixture.report["results"][0]["runs"].values():
            run["command"] = ["/fixture/python", str(forged / "cases/example.py")]
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "repository differs"):
            self.seal()

    def test_runtime_contract_is_reconstructed_from_build_receipt(self):
        for endpoint in ("start", "end"):
            self.fixture.report["provenance"]["validation_window"][endpoint][
                "runtime_contract"
            ]["libmeep"]["path"] = "/hostile/lib/libmeep.so"
        self.fixture.reseal_report()
        with self.assertRaisesRegex(
            sealer.ArchiveError, "runtime contract differs from build receipt"
        ):
            self.seal()

    def test_run_environment_root_is_bound_to_raw_evidence_root(self):
        run = self.fixture.report["results"][0]["runs"]["cuda"]
        environment = run["environment_contract"]["environment"]
        old_root = pathlib.Path(environment["HOME"]).parents[5]
        hostile_root = pathlib.Path("/hostile-output-root")
        for key in (
            "HOME",
            "MPLCONFIGDIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "GPMEEP_VALIDATION_STATS_FILE",
        ):
            value = pathlib.Path(environment[key])
            environment[key] = str(hostile_root / value.relative_to(old_root))
        for key in ("HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            run["environment_contract"][key] = environment[key]
        run["environment_contract"]["sha256"] = (
            sealer.runner_contract.environment_sha256(environment)
        )
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "evidence root"):
            self.seal()

    def test_build_python_claim_is_bound_to_receipt_package_root(self):
        self.fixture.report["provenance"]["build_python"] = "/hostile/build-python"
        for run in self.fixture.report["results"][0]["runs"].values():
            environment = run["environment_contract"]["environment"]
            first, _ = environment["PYTHONPATH"].split(os.pathsep, 1)
            environment["PYTHONPATH"] = os.pathsep.join(
                (first, "/hostile/build-python")
            )
            run["environment_contract"]["sha256"] = (
                sealer.runner_contract.environment_sha256(environment)
            )
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "build-Python"):
            self.seal()

    def test_rejects_missing_or_hostile_exact_environment(self):
        attacks = ("missing", "poison")
        for label in attacks:
            with self.subTest(label=label):
                fixture_root = self.root / label
                fixture_root.mkdir()
                fixture = ArchiveFixture(fixture_root)
                run = fixture.report["results"][0]["runs"]["cuda"]
                if label == "missing":
                    del run["environment_contract"]
                else:
                    environment = run["environment_contract"]["environment"]
                    environment["LD_PRELOAD"] = "/poison.so"
                    run["environment_contract"]["keys"] = sorted(environment)
                    run["environment_contract"]["sha256"] = (
                        sealer.runner_contract.environment_sha256(environment)
                    )
                fixture.reseal_report()
                with self.assertRaisesRegex(
                    sealer.ArchiveError, "run record|allowlist"
                ):
                    sealer.seal_archive(
                        fixture.output,
                        self.root / f"archive-{label}",
                        fixture.repo,
                        fixture.receipt,
                    )

    def test_rejects_failed_run_outcome_contradicting_raw_evidence(self):
        result = self.fixture.report["results"][0]
        result["runs"]["cpu"]["outcome"] = "TIMEOUT"
        result["comparison"] = {
            "mode": "json_metrics",
            "outcome": "NOT_COMPARABLE",
            "reason": "backend runs did not pass: cpu",
        }
        result["outcome"] = "TIMEOUT"
        self.fixture.report["summary"] = {"TIMEOUT": 1}
        self.fixture.report["exit_code"] = 1
        self.fixture.reseal_report()
        with self.assertRaisesRegex(sealer.ArchiveError, "outcome differs"):
            self.seal()

    def test_replays_unittest_identity_and_positive_count_from_raw_stderr(self):
        manifest_path = (
            self.fixture.repo / "scripts/python-validation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        rule = manifest["inventory"][0]
        rule["kind"] = "unittest"
        rule["expected_unittest"] = {
            "count": 1,
            "identities": ["cases.Example.test_expected"],
        }
        write_json(manifest_path, manifest)
        result = self.fixture.report["results"][0]
        result["kind"] = "unittest"
        for backend, run in result["runs"].items():
            run["command"] = [
                self.fixture._runtime_contract()["python_executable"]["path"],
                "-m",
                "unittest",
                "discover",
                "-v",
                "-s",
                str((self.fixture.repo / "cases").resolve()),
                "-p",
                "example.py",
            ]
            stderr = pathlib.Path(run["stderr_log"])
            stderr.write_text("Ran 0 tests in 0.000s\n\nOK\n", encoding="utf-8")
            run["evidence_files"]["stderr"] = {
                "available": True,
                **file_record(stderr, self.fixture.repo),
            }
            run["output_sizes"]["stderr"] = stderr.stat().st_size
            run["output_sizes"]["combined"] = (
                run["output_sizes"]["stdout"] + stderr.stat().st_size
            )
            # Forge a self-consistent PASS claim; raw replay must override it.
            run["unittest_reported_test_counts"] = [0]
            run["unittest_test_count"] = 0
            run["expected_unittest_test_count"] = 1
            run["expected_unittest_test_identities"] = [
                "cases.Example.test_expected"
            ]
        subprocess.run(["git", "add", "."], cwd=self.fixture.repo, check=True)
        self.fixture.report["provenance"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        self.fixture.refresh_source_and_receipt_bindings()
        with self.assertRaisesRegex(sealer.ArchiveError, "unittest/backend replay"):
            self.seal()

    def test_raw_unittest_failed_summary_cannot_be_reported_as_pass(self):
        stdout = self.root / "unit.stdout"
        stderr = self.root / "unit.stderr"
        stdout.write_text("", encoding="utf-8")
        stderr.write_text(
            "test_expected (cases.Example.test_expected) ... FAIL\n\n"
            "Ran 1 test in 0.001s\n\nFAILED (failures=1)\n",
            encoding="utf-8",
        )
        case = {
            "kind": "unittest",
            "allowed_unittest_skips": 0,
            "allowed_unittest_skip_details": [],
            "expected_unittest": {
                "count": 1,
                "identities": ["cases.Example.test_expected"],
            },
        }
        run = {
            "output_limit": None,
            "timeout": False,
            "exit_code": 0,
            "unittest_skips": 0,
            "unittest_skip_details": [],
            "allowed_unittest_skips": 0,
            "unittest_skip_policy_problems": [],
            "unittest_reported_test_counts": [1],
            "unittest_test_count": 1,
            "unittest_test_identities": ["cases.Example.test_expected"],
            "expected_unittest_test_count": 1,
            "expected_unittest_test_identities": [
                "cases.Example.test_expected"
            ],
            "unittest_test_contract_ok": True,
            "unittest_test_contract_problems": [],
            "unittest_terminal_summaries": ["FAILED (failures=1)"],
            "backend_contract_ok": True,
            "backend_contract_problems": [],
        }
        with self.assertRaisesRegex(sealer.ArchiveError, "unittest/backend replay"):
            sealer._replay_run_outcome(
                case=case,
                backend="cpu",
                run=run,
                stdout_path=stdout,
                stderr_path=stderr,
                contract_ok=True,
                contract_problems=[],
            )

    def test_verifier_rejects_permission_only_payload_tampering(self):
        archive = self.seal()
        source = archive / "sources/cases/example.py"
        os.chmod(source, 0o755)
        with self.assertRaisesRegex(sealer.ArchiveError, "mode differs"):
            sealer.verify_archive(archive)

    def test_verifier_rejects_absolute_and_parent_archive_members(self):
        for malicious in ("/absolute/member", "../parent/member"):
            with self.subTest(malicious=malicious):
                suffix = hashlib.sha256(malicious.encode()).hexdigest()[:8]
                archive = self.seal(self.root / f"archive-{suffix}")
                manifest_path = archive / sealer.MANIFEST_NAME
                complete_path = archive / sealer.COMPLETE_NAME
                manifest = json.loads(manifest_path.read_text())
                complete = json.loads(complete_path.read_text())
                manifest["files"][0]["path"] = malicious
                manifest["files"] = sorted(
                    manifest["files"], key=lambda item: item["path"]
                )
                manifest["payload_sha256"] = canonical_sha256(manifest["files"])
                write_json(manifest_path, manifest)
                complete["archive_manifest_sha256"] = sha256_file(manifest_path)
                complete["payload_sha256"] = manifest["payload_sha256"]
                write_json(complete_path, complete)
                with self.assertRaisesRegex(
                    sealer.ArchiveError, "normalized relative path"
                ):
                    sealer.verify_archive(archive)

    def test_verifier_rejects_forged_source_and_exclusion_counts(self):
        attacks = {
            "source": lambda manifest: manifest["source_contract"].__setitem__(
                "raw_evidence_file_count", 999
            ),
            "runtime": lambda manifest: manifest[
                "runtime_cache_exclusion"
            ].__setitem__("file_count", 999),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                archive = self.seal(self.root / f"archive-{name}")
                self.rewrite_manifest_and_complete(archive, mutate)
                with self.assertRaisesRegex(sealer.ArchiveError, "contract|count"):
                    sealer.verify_archive(archive)

    def test_verifier_rejects_semantically_forged_report_chain(self):
        archive = self.seal()
        report_path = archive / "runner-output/report.json"
        runner_complete_path = archive / "runner-output/COMPLETE"
        report = json.loads(report_path.read_text())
        for location in (
            report["provenance"]["source_snapshot"],
            report["provenance"]["validation_window"]["start"][
                "source_snapshot"
            ],
            report["provenance"]["validation_window"]["end"][
                "source_snapshot"
            ],
        ):
            location["sha256"] = "0" * 64
        write_json(report_path, report)
        runner_complete = json.loads(runner_complete_path.read_text())
        runner_complete["report_sha256"] = sha256_file(report_path)
        write_json(runner_complete_path, runner_complete)
        manifest_path = archive / sealer.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["source_contract"]["report_sha256"] = sha256_file(report_path)
        manifest["source_contract"]["runner_complete_sha256"] = sha256_file(
            runner_complete_path
        )
        manifest["source_contract"]["source_snapshot_sha256"] = "0" * 64
        write_json(manifest_path, manifest)
        self.rebind_archive_payload(archive)
        with self.assertRaisesRegex(
            sealer.ArchiveError, "source snapshot differs"
        ):
            sealer.verify_archive(archive)

    def test_verifier_rejects_extra_source_contract_fields(self):
        archive = self.seal()
        self.rewrite_manifest_and_complete(
            archive,
            lambda manifest: manifest["source_contract"].__setitem__(
                "forged", True
            ),
        )
        with self.assertRaisesRegex(sealer.ArchiveError, "source contract"):
            sealer.verify_archive(archive)

    def test_verifier_rejects_rebound_fixed_report_path(self):
        archive = self.seal()
        complete_path = archive / sealer.COMPLETE_NAME
        complete = json.loads(complete_path.read_text())
        complete["report_path"] = "runner-output/report.md"
        complete["report_sha256"] = sha256_file(
            archive / "runner-output/report.md"
        )
        write_json(complete_path, complete)
        with self.assertRaisesRegex(sealer.ArchiveError, "path is not fixed"):
            sealer.verify_archive(archive)

    def test_rejects_unowned_runtime_exclusion_source(self):
        stale = (
            self.fixture.output
            / "runs/stale/cpu/runtime"
            / ("e" * 32)
            / "cache/secret"
        )
        stale.parent.mkdir(parents=True)
        stale.write_text("not owned by a report run\n", encoding="utf-8")
        with self.assertRaisesRegex(sealer.ArchiveError, "unreferenced"):
            self.seal()

    def test_verifier_rejects_forged_unowned_runtime_exclusion(self):
        archive = self.seal()

        def mutate(manifest):
            exclusion = manifest["runtime_cache_exclusion"]
            exclusion["files"][0]["path"] = (
                "runs/stale/cpu/runtime/"
                + "e" * 32
                + "/cache/forged"
            )
            exclusion["records_sha256"] = canonical_sha256(
                exclusion["files"]
            )

        self.rewrite_manifest_and_complete(archive, mutate)
        with self.assertRaisesRegex(sealer.ArchiveError, "path is invalid"):
            sealer.verify_archive(archive)

    def test_rejects_archive_destination_inside_repository(self):
        archive = self.fixture.repo / "evidence-archive"
        with self.assertRaisesRegex(sealer.ArchiveError, "repository"):
            self.seal(archive)
        self.assertFalse(archive.exists())

    def test_publication_failure_leaves_no_final_or_staging_directory(self):
        archive = self.root / "archive"
        with mock.patch.object(
            sealer,
            "_verify_copy_sources",
            side_effect=sealer.ArchiveError("fault after staged COMPLETE"),
        ):
            with self.assertRaisesRegex(sealer.ArchiveError, "fault after"):
                self.seal(archive)
        self.assertFalse(archive.exists())
        self.assertEqual(list(self.root.glob(".archive.staging-*")), [])

    def test_concurrent_archive_destination_is_never_replaced(self):
        archive = self.root / "archive-race"
        original_publish = sealer._publish_noreplace

        def inject_destination(source, destination):
            if destination == archive and not destination.exists():
                destination.mkdir()
                (destination / "sentinel").write_text(
                    "concurrent owner\n", encoding="utf-8"
                )
            return original_publish(source, destination)

        with mock.patch.object(
            sealer, "_publish_noreplace", side_effect=inject_destination
        ):
            with self.assertRaisesRegex(sealer.ArchiveError, "already exists"):
                self.seal(archive)
        self.assertEqual(
            (archive / "sentinel").read_text(), "concurrent owner\n"
        )
        self.assertEqual(list(self.root.glob(".archive-race.staging-*")), [])

    def test_concurrent_tar_destination_is_never_replaced(self):
        archive = self.seal()
        destination = self.root / "race.tar"
        destination.write_bytes(b"concurrent owner")
        with self.assertRaisesRegex(sealer.ArchiveError, "already exists"):
            sealer.create_deterministic_tar(archive, destination)
        self.assertEqual(destination.read_bytes(), b"concurrent owner")

    def test_verifier_rejects_hardlinked_payload_members(self):
        archive = self.seal()
        cpu = (
            archive
            / "runner-output/runs"
            / self.fixture.case_directory
            / "cpu/stderr.log"
        )
        cuda = (
            archive
            / "runner-output/runs"
            / self.fixture.case_directory
            / "cuda/stderr.log"
        )
        cuda.unlink()
        os.link(cpu, cuda)
        with self.assertRaisesRegex(sealer.ArchiveError, "hard-linked"):
            sealer.verify_archive(archive)

    def test_deterministic_tar_is_reproducible_and_ignores_tar_options(self):
        archive = self.seal()
        first_path = self.root / "first.tar"
        second_path = self.root / "second.tar"
        with mock.patch.dict(
            os.environ, {"TAR_OPTIONS": "--exclude=*"}, clear=False
        ):
            first = sealer.create_deterministic_tar(archive, first_path)
        for path in archive.rglob("*"):
            os.utime(path, (123456789, 123456789), follow_symlinks=False)
        with mock.patch.dict(
            os.environ, {"TAR_OPTIONS": "--exclude=*"}, clear=False
        ):
            second = sealer.create_deterministic_tar(archive, second_path)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_deterministic_tar_scales_bounded_output_for_large_evidence(self):
        archive = self.root / "large-evidence"
        archive.mkdir()
        payload = archive / "payload.bin"
        with payload.open("xb") as target:
            target.truncate(sealer.MINIMUM_TAR_OUTPUT_LIMIT_BYTES + 1)
        destination = self.root / "large-evidence.tar"
        result = sealer.create_deterministic_tar(archive, destination)
        self.assertGreater(
            result["size_bytes"], sealer.MINIMUM_TAR_OUTPUT_LIMIT_BYTES
        )
        self.assertGreater(result["output_limit_bytes"], result["size_bytes"])
        self.assertEqual(result["size_bytes"], destination.stat().st_size)

    def test_deterministic_tar_ignores_path_injected_fake_tar(self):
        archive = self.seal()
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        fake_tar = fake_bin / "tar"
        fake_tar.write_text(
            "#!/bin/sh\nprintf 'ATTACKER-NOT-A-TAR' > \"$2\"\n",
            encoding="utf-8",
        )
        os.chmod(fake_tar, 0o755)
        destination = self.root / "path-safe.tar"
        with mock.patch.dict(
            os.environ,
            {"PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "")},
            clear=False,
        ):
            result = sealer.create_deterministic_tar(archive, destination)
        self.assertGreater(result["size_bytes"], len(b"ATTACKER-NOT-A-TAR"))
        self.assertNotEqual(destination.read_bytes(), b"ATTACKER-NOT-A-TAR")

    def test_deterministic_tar_uses_no_raceable_named_staging_output(self):
        archive = self.seal()
        victim = self.root / "victim.bin"
        victim.write_bytes(b"SAFE")
        legacy_stage = self.root / ".result.tar.staging-predicted"
        legacy_stage.symlink_to(victim)
        destination = self.root / "result.tar"
        sealer.create_deterministic_tar(archive, destination)
        self.assertEqual(victim.read_bytes(), b"SAFE")
        self.assertFalse(destination.is_symlink())

    def test_verifier_accepts_and_enforces_external_trust_anchors(self):
        archive = self.seal()
        complete = json.loads((archive / sealer.COMPLETE_NAME).read_text())
        self.assertEqual(
            sealer.verify_archive(
                archive,
                expected_manifest_sha256=complete[
                    "archive_manifest_sha256"
                ],
                expected_receipt_id=self.fixture.receipt_value["receipt_id"],
            )["state"],
            "COMPLETE",
        )
        with self.assertRaisesRegex(sealer.ArchiveError, "trusted expected"):
            sealer.verify_archive(
                archive, expected_manifest_sha256="0" * 64
            )
        with self.assertRaisesRegex(
            sealer.ArchiveError, "authenticates only the build"
        ):
            sealer.verify_archive(
                archive,
                expected_receipt_id=self.fixture.receipt_value["receipt_id"],
            )


if __name__ == "__main__":
    unittest.main()
