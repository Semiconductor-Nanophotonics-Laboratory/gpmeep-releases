"""Self-tests for the isolated CPU FP64 MPI+Python reference build."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
BUILDER = SCRIPTS / "build-meep-cpu-mpi-python-fp64.sh"
QUALIFIER_PATH = SCRIPTS / "qualify-cpu-mpi-fp64.py"
VERIFIER_PATH = SCRIPTS / "verify-build-receipt.py"
MATRIX_PATH = SCRIPTS / "user-workloads" / "run_user_workload_matrix.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QUALIFIER = load_module("gpmeep_test_cpu_mpi_fp64_qualifier", QUALIFIER_PATH)
VERIFIER = load_module("gpmeep_test_build_receipt_verifier", VERIFIER_PATH)
QUALIFICATION_CONTRACT = load_module(
    "gpmeep_test_cpu_reference_contract",
    SCRIPTS / "gpmeep_qualification_contract.py",
)
PROVENANCE = load_module(
    "gpmeep_test_cpu_reference_provenance",
    SCRIPTS / "gpmeep_provenance.py",
)


def fake_meep(
    *, mpi: bool = True, single: bool = False, compiled: bool = False,
    requested: str = "cpu", active: str = "cpu", processors: object = 2
):
    return types.SimpleNamespace(
        with_mpi=lambda: mpi,
        is_single_precision=lambda: single,
        count_processors=lambda: processors,
        gpu=types.SimpleNamespace(
            compiled=compiled,
            requested_backend=requested,
            active_backend=active,
        ),
    )


def cpu_statistics(*, boundary_calls: int = 1):
    return {
        "dispatch": {
            "cpu_curl_calls": 2,
            "cpu_curl_points": 20,
            "cuda_curl_calls": 0,
        },
        "field_updates": {
            "cpu_update_eh_calls": 2,
            "cpu_update_eh_points": 20,
            "cuda_update_eh_calls": 0,
        },
        "sources": {
            "cpu_source_calls": 1,
            "cpu_source_points": 10,
            "cuda_source_calls": 0,
        },
        "boundaries": {
            "cpu_boundary_calls": boundary_calls,
            "cpu_boundary_points": boundary_calls * 2,
            "cuda_boundary_calls": 0,
        },
        "dfts": {
            "cpu_dft_calls": 1,
            "cpu_dft_points": 10,
            "cuda_dft_calls": 0,
        },
        "runtime": {"runtime_availability_probes": 0},
    }


class CpuMpiFp64QualifierTests(unittest.TestCase):
    def test_file_identity_binds_canonical_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "runtime.so"
            path.write_bytes(b"runtime image")
            identity = QUALIFIER.file_identity(path)
            self.assertEqual(identity["path"], str(path))
            self.assertEqual(identity["size_bytes"], len(b"runtime image"))
            self.assertEqual(
                identity["sha256"],
                "ab903792fc3c58a8d45877b3796ad3b917c55a8db5d357995c774e843d478ab0",
            )
            link = path.with_name("runtime-link.so")
            link.symlink_to(path)
            with self.assertRaisesRegex(RuntimeError, "canonical regular file"):
                QUALIFIER.file_identity(link)

    def test_loaded_libmeep_mapping_is_unique_and_live(self):
        root = pathlib.Path("/tmp/gpmeep/libmeep.so.1")
        with mock.patch.object(pathlib.Path, "resolve", return_value=root):
            observed = QUALIFIER.loaded_libmeep_path(
                "1000-2000 r-xp 0 00:00 1 /tmp/gpmeep/libmeep.so.1\n"
                "2000-3000 r--p 0 00:00 1 /tmp/gpmeep/libmeep.so.1\n"
            )
        self.assertEqual(observed, root)

        with self.assertRaisesRegex(RuntimeError, "has been deleted"):
            QUALIFIER.loaded_libmeep_path(
                "1000-2000 r-xp 0 00:00 1 /tmp/gpmeep/libmeep.so.1 (deleted)\n"
            )
        with self.assertRaisesRegex(RuntimeError, "observed 0"):
            QUALIFIER.loaded_libmeep_path(
                "1000-2000 r-xp 0 00:00 1 /tmp/gpmeep/not-meep.so\n"
            )

    def test_runtime_identity_accepts_only_cpu_fp64_mpi(self):
        QUALIFIER.validate_runtime_identity(fake_meep(), 2, 2)
        cases = (
            (fake_meep(), 1, 2, "mpi4py communicator size mismatch"),
            (fake_meep(mpi=False), 2, 2, "not compiled with MPI"),
            (fake_meep(processors=1), 2, 2, "Meep communicator size mismatch"),
            (fake_meep(processors=True), 2, 2, "Meep communicator size mismatch"),
            (fake_meep(single=True), 2, 2, "not FP64"),
            (fake_meep(compiled=True), 2, 2, "contains CUDA"),
            (fake_meep(requested="auto"), 2, 2, "CPU backend"),
            (fake_meep(active="cuda"), 2, 2, "CPU backend"),
        )
        for meep, actual, expected, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(
                RuntimeError, diagnostic
            ):
                QUALIFIER.validate_runtime_identity(meep, actual, expected)

    def test_statistics_require_real_cpu_work_and_zero_cuda_work(self):
        selected = QUALIFIER.validate_cpu_statistics(
            cpu_statistics(), require_boundary=True
        )
        self.assertEqual(
            set(selected),
            {
                "curl_calls",
                "curl_points",
                "update_eh_calls",
                "update_eh_points",
                "source_calls",
                "source_points",
                "boundary_calls",
                "boundary_points",
                "dft_calls",
                "dft_points",
            },
        )
        QUALIFIER.validate_cpu_statistics(
            cpu_statistics(boundary_calls=0), require_boundary=False
        )
        with self.assertRaisesRegex(RuntimeError, "boundary dispatch"):
            QUALIFIER.validate_cpu_statistics(
                cpu_statistics(boundary_calls=0), require_boundary=True
            )

        for group, key in (
            ("dispatch", "cpu_curl_calls"),
            ("dispatch", "cpu_curl_points"),
            ("field_updates", "cpu_update_eh_calls"),
            ("field_updates", "cpu_update_eh_points"),
            ("sources", "cpu_source_calls"),
            ("sources", "cpu_source_points"),
            ("dfts", "cpu_dft_calls"),
            ("dfts", "cpu_dft_points"),
        ):
            invalid = cpu_statistics()
            invalid[group][key] = 0
            with self.subTest(group=group, key=key), self.assertRaisesRegex(
                RuntimeError, f"{group}.{key}"
            ):
                QUALIFIER.validate_cpu_statistics(invalid, require_boundary=True)

        invalid = cpu_statistics()
        invalid["dfts"]["cuda_dft_calls"] = 1
        with self.assertRaisesRegex(RuntimeError, "recorded CUDA work"):
            QUALIFIER.validate_cpu_statistics(invalid, require_boundary=True)

        invalid = cpu_statistics()
        invalid["runtime"] = []
        with self.assertRaisesRegex(RuntimeError, "is not an object"):
            QUALIFIER.validate_cpu_statistics(invalid, require_boundary=True)

        invalid = cpu_statistics()
        invalid["runtime"]["runtime_availability_probes"] = True
        with self.assertRaisesRegex(RuntimeError, "is invalid"):
            QUALIFIER.validate_cpu_statistics(invalid, require_boundary=True)

    def test_collective_call_reports_every_rank_failure_without_stranding(self):
        class FakeComm:
            def __init__(self, remote):
                self.remote = remote

            def allgather(self, local):
                return [local, self.remote]

        self.assertEqual(
            QUALIFIER.collective_call(FakeComm(None), "fixture", lambda: 7), 7
        )
        with self.assertRaisesRegex(
            RuntimeError, "rank 0: ValueError: local; rank 1: remote"
        ):
            QUALIFIER.collective_call(
                FakeComm("remote"),
                "fixture",
                lambda: (_ for _ in ()).throw(ValueError("local")),
            )


class CpuMpiFp64ReceiptVerifierTests(unittest.TestCase):
    @staticmethod
    def runtime_identity(name: str) -> dict[str, object]:
        return {"path": f"/runtime/{name}", "size_bytes": 10, "sha256": "a" * 64}

    def qualification_record(self, mpi_size: int = 1) -> dict[str, object]:
        runtime = {
            name: self.runtime_identity(name)
            for name in ("python", "python_extension", "libmeep")
        }
        ranks = []
        for rank in range(mpi_size):
            ranks.append(
                {
                    "rank": rank,
                    "energy": 1.0,
                    "dft_norm": 2.0,
                    "meep_time": 8.0,
                    "statistics": {
                        "curl_calls": 1,
                        "curl_points": 10,
                        "update_eh_calls": 1,
                        "update_eh_points": 10,
                        "source_calls": 1,
                        "source_points": 10,
                        "boundary_calls": 1 if mpi_size > 1 else 0,
                        "boundary_points": 10 if mpi_size > 1 else 0,
                        "dft_calls": 1,
                        "dft_points": 10,
                    },
                    "runtime_artifacts": copy.deepcopy(runtime),
                }
            )
        return {
            "schema": "gpmeep-cpu-mpi-fp64-qualification-v1",
            "mpi_size": mpi_size,
            "runtime_kind": "in-place",
            "single_precision": False,
            "cuda_compiled": False,
            "ranks": ranks,
        }

    @staticmethod
    def _write(path: pathlib.Path, payload: bytes | str = b"fixture") -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_bytes(payload)
        return path.resolve(strict=True)

    @staticmethod
    def _runtime_identity(path: pathlib.Path, repo: pathlib.Path) -> dict[str, object]:
        record = PROVENANCE.file_record(path, repo)
        return {
            "path": str(path.resolve(strict=True)),
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }

    def _full_receipt_fixture(
        self, repo: pathlib.Path
    ) -> tuple[dict[str, object], dict[str, object]]:
        repo = repo.resolve(strict=True)
        build = repo / "build" / "meep-cpu-mpi-python-fp64"
        install = repo / "install" / "meep-cpu-mpi-python-fp64"
        environment = repo / ".envs" / "gpmeep-cpu-mpi-fp64"
        build_home = build / "build-home"
        for directory in (
            build_home / ".cache",
            build_home / ".config",
            build_home / ".matplotlib",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        builder = self._write(
            repo / "scripts" / "build-meep-cpu-mpi-python-fp64.sh",
            "#!/bin/bash -p\n",
        )
        lock_digest = hashlib.sha256(b"locked-package").hexdigest()
        lock_url = "https://example.invalid/pkgs/dependency-1.0-0.conda"
        lock = self._write(
            repo / "environment" / "locks" / "cuda-mpi-linux-64.lock",
            f"@EXPLICIT\n{lock_url}#{lock_digest}\n",
        )
        openmpi_params = self._write(
            repo / "environment" / "openmpi-qualification-mca-params.conf",
            "pml=ob1\n",
        )

        tool_names = set(QUALIFICATION_CONTRACT.CPU_REFERENCE_TOOLS)
        tool_names.add("gfortran")
        tool_paths = {
            name: self._write(environment / "bin" / name, f"tool:{name}\n")
            for name in tool_names
        }
        build_extension = self._write(
            build / "python" / "meep" / "_meep.so.38.0.0", b"build extension"
        )
        build_libmeep = self._write(
            build / "src" / ".libs" / "libmeep.so.38.0.0", b"build libmeep"
        )
        installed_extension = self._write(
            install
            / "lib"
            / "python3.14"
            / "site-packages"
            / "meep"
            / "_meep.so.38.0.0",
            b"installed extension",
        )
        installed_libmeep = self._write(
            install / "lib" / "libmeep.so.38.0.0", b"installed libmeep"
        )
        config_h = self._write(build / "config.h", "#define HAVE_MPI 1\n")
        config_status = self._write(build / "config.status", "cpu fixture\n")
        environment_explicit = self._write(
            build / "environment-explicit.lock",
            f"@EXPLICIT\n{lock_url}#{lock_digest}\n",
        )
        package_cache = repo / ".micromamba" / "pkgs"
        package_cache.mkdir(parents=True)

        lock_record = PROVENANCE.file_record(lock, repo)
        audit = {
            "schema_version": 3,
            "environment_prefix": str(environment),
            "package_cache": {
                "path": str(package_cache),
                "archive_count": 1,
            },
            "archive_snapshot": {
                "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
                "hash_and_parse_same_private_snapshot": True,
                "descriptor_retained_until_terminal_verification": True,
                "pathname_fingerprint_rechecked": True,
            },
            "archive_decoder": {
                "path": "/usr/bin/zstd",
                "sha256": QUALIFICATION_CONTRACT._sha256_file(
                    pathlib.Path("/usr/bin/zstd")
                ),
                "version": "fixture-zstd",
                "timeout_seconds": 120,
            },
            "lock": {
                "path": str(lock),
                "package_count": 1,
                "sha256": lock_record["sha256"],
            },
            "metadata_package_count": 1,
            "owned_path_count": len(tool_paths),
            "hashed_installed_path_count": len(tool_paths),
            "archive_bound_hardlink_count": len(tool_paths),
            "relocated_hardlink_count": 0,
            "unhashed_bytecode_path_count": 0,
            "generated_entry_points": [],
            "softlink_count": 0,
            "softlinks": [],
            "unowned_allowed_path_count": 0,
            "package_records": [{"url": lock_url, "sha256": lock_digest}],
            "pass": True,
        }
        audit_path = self._write(
            build / "conda-prefix-content-audit.json",
            json.dumps(audit, sort_keys=True) + "\n",
        )

        runtime_identities = {
            "python": self._runtime_identity(tool_paths["python"], repo),
            "build_extension": self._runtime_identity(build_extension, repo),
            "build_libmeep": self._runtime_identity(build_libmeep, repo),
            "installed_extension": self._runtime_identity(
                installed_extension, repo
            ),
            "installed_libmeep": self._runtime_identity(installed_libmeep, repo),
        }
        qualification_dir = build / "cpu-mpi-fp64-qualification"
        qualification_dir.mkdir(parents=True, exist_ok=True)
        lane_records: dict[str, dict[str, object]] = {}
        for name, (
            runtime_kind,
            mpi_size,
            extension_name,
            libmeep_name,
        ) in QUALIFICATION_CONTRACT.CPU_REFERENCE_LOG_SPECS.items():
            ranks = []
            for rank in range(mpi_size):
                ranks.append(
                    {
                        "rank": rank,
                        "energy": 1.25,
                        "dft_norm": 2.5,
                        "meep_time": 8.0,
                        "statistics": {
                            "curl_calls": 1,
                            "curl_points": 10,
                            "update_eh_calls": 1,
                            "update_eh_points": 10,
                            "source_calls": 1,
                            "source_points": 10,
                            "boundary_calls": 1 if mpi_size > 1 else 0,
                            "boundary_points": 10 if mpi_size > 1 else 0,
                            "dft_calls": 1,
                            "dft_points": 10,
                        },
                        "runtime_artifacts": {
                            "python": copy.deepcopy(runtime_identities["python"]),
                            "python_extension": copy.deepcopy(
                                runtime_identities[
                                    "build_extension"
                                    if extension_name == "python_extension"
                                    else "installed_extension"
                                ]
                            ),
                            "libmeep": copy.deepcopy(
                                runtime_identities[
                                    "build_libmeep"
                                    if libmeep_name == "libmeep"
                                    else "installed_libmeep"
                                ]
                            ),
                        },
                    }
                )
            record = {
                "schema": QUALIFICATION_CONTRACT.CPU_REFERENCE_SCHEMA,
                "mpi_size": mpi_size,
                "runtime_kind": runtime_kind,
                "single_precision": False,
                "cuda_compiled": False,
                "ranks": ranks,
            }
            lane_records[name] = record
            self._write(
                qualification_dir / name,
                "Meep fixture\n"
                + QUALIFICATION_CONTRACT.CPU_REFERENCE_MARKER
                + json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n",
            )

        host_logs = build / "host-build-logs"
        for label in ("autoreconf", "configure", "build", "check", "install"):
            self._write(
                host_logs / f"{label}.log",
                f"fixture output\ngpmeep-host-build-step-pass:{label}\n",
            )

        artifacts = {
            "python_extension": PROVENANCE.file_record(build_extension, repo),
            "libmeep": PROVENANCE.file_record(build_libmeep, repo),
            "installed_python_extension": PROVENANCE.file_record(
                installed_extension, repo
            ),
            "installed_libmeep": PROVENANCE.file_record(installed_libmeep, repo),
        }
        configuration_files = {
            "config_h": PROVENANCE.file_record(config_h, repo),
            "config_status": PROVENANCE.file_record(config_status, repo),
            "environment_explicit": PROVENANCE.file_record(
                environment_explicit, repo
            ),
            "conda_prefix_content_audit": PROVENANCE.file_record(audit_path, repo),
            "openmpi_qualification_params": PROVENANCE.file_record(
                openmpi_params, repo
            ),
        }
        toolchain = {
            name: {
                **PROVENANCE.file_record(tool_paths[name], repo),
                "invoked_path": str(tool_paths[name]),
                "version": {"available": True},
            }
            for name in QUALIFICATION_CONTRACT.CPU_REFERENCE_TOOLS
        }
        manifests = {
            "in_place_python": PROVENANCE.tree_manifest(
                build / "python" / "meep", repo
            ),
            "installed_python": PROVENANCE.tree_manifest(
                installed_extension.parent, repo
            ),
            "installed_prefix": PROVENANCE.tree_manifest(install, repo),
            "installed_environment": PROVENANCE.tree_manifest(environment, repo),
            "qualification_logs": PROVENANCE.tree_manifest(
                qualification_dir, repo
            ),
            "host_build_logs": PROVENANCE.tree_manifest(host_logs, repo),
        }
        receipt = {
            "build_kind": QUALIFICATION_CONTRACT.CPU_REFERENCE_BUILD_KIND,
            "build_dir": str(build),
            "git_status_porcelain": [],
            "configuration": {
                "builder": PROVENANCE.file_record(builder, repo),
                "qualification_contract": (
                    QUALIFICATION_CONTRACT.CPU_REFERENCE_CONTRACT_NAME
                ),
                "configure_argv": [
                    "--enable-maintainer-mode",
                    "--enable-shared",
                    "--disable-single",
                    "--disable-cuda",
                    "--disable-cuda-fast-math",
                    "--with-openmp",
                    "--with-mpi",
                    "--with-python",
                    "--without-scheme",
                    f"--prefix={install}",
                ],
                "environment": {
                    "MEEP_GPU_MAKE_JOBS": "2",
                    "MEEP_GPU_BACKEND": "cpu",
                    "CC": str(tool_paths["cc"]),
                    "CXX": str(tool_paths["c++"]),
                    "FC": str(tool_paths["gfortran"]),
                    "F77": str(tool_paths["gfortran"]),
                    "MPICXX": str(tool_paths["mpicxx"]),
                    "HOME": str(build_home),
                    "XDG_CACHE_HOME": str(build_home / ".cache"),
                    "XDG_CONFIG_HOME": str(build_home / ".config"),
                    "MPLCONFIGDIR": str(build_home / ".matplotlib"),
                },
                "lockfiles": {"environment_lock": lock_record},
            },
            "configuration_files": configuration_files,
            "artifacts": artifacts,
            "manifests": manifests,
            "toolchain": toolchain,
        }
        context = {
            "audit": audit,
            "audit_path": audit_path,
            "lane_records": lane_records,
            "qualification_dir": qualification_dir,
            "repo": repo,
        }
        return receipt, context

    def test_full_cpu_receipt_contract_accepts_only_bound_four_lane_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary).resolve()
            receipt, context = self._full_receipt_fixture(repo)
            QUALIFICATION_CONTRACT.validate_v2_receipt(receipt, repo)

            invalid = copy.deepcopy(receipt)
            invalid["artifacts"]["libmeep"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "not receipt-bound",
            ):
                QUALIFICATION_CONTRACT.validate_v2_receipt(invalid, repo)

            nonstandard_extension = self._write(
                repo
                / "install"
                / "meep-cpu-mpi-python-fp64"
                / "nonstandard"
                / "_meep.so.38.0.0",
                b"wrong layout",
            )
            invalid = copy.deepcopy(receipt)
            invalid["artifacts"]["installed_python_extension"] = (
                PROVENANCE.file_record(nonstandard_extension, repo)
            )
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "artifact paths are not fixed",
            ):
                QUALIFICATION_CONTRACT.validate_v2_receipt(invalid, repo)

            alternate_lock = self._write(
                repo / "environment" / "locks" / "alternate.lock",
                "@EXPLICIT\n",
            )
            invalid = copy.deepcopy(receipt)
            invalid["configuration"]["lockfiles"]["environment_lock"] = (
                PROVENANCE.file_record(alternate_lock, repo)
            )
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "lock path is not authoritative",
            ):
                QUALIFICATION_CONTRACT.validate_v2_receipt(invalid, repo)

            audit_path = context["audit_path"]
            invalid_audit = copy.deepcopy(context["audit"])
            invalid_audit["pass"] = False
            audit_path.write_text(json.dumps(invalid_audit), encoding="utf-8")
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "content audit is not exact",
            ):
                QUALIFICATION_CONTRACT.validate_v2_receipt(receipt, repo)
            audit_path.write_text(
                json.dumps(context["audit"], sort_keys=True) + "\n",
                encoding="utf-8",
            )

            lane_name = "installed-two-rank.log"
            lane_record = copy.deepcopy(context["lane_records"][lane_name])
            for rank in lane_record["ranks"]:
                rank["energy"] = 1.5
            lane_path = context["qualification_dir"] / lane_name
            lane_path.write_text(
                QUALIFICATION_CONTRACT.CPU_REFERENCE_MARKER
                + json.dumps(lane_record, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "qualification lanes disagree",
            ):
                QUALIFICATION_CONTRACT.validate_v2_receipt(receipt, repo)

    def test_cpu_qualification_log_contract_replays_semantics_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "in-place-one-rank.log"
            record = self.qualification_record()

            def write(value):
                log.write_text(
                    "Meep output\n"
                    + QUALIFICATION_CONTRACT.CPU_REFERENCE_MARKER
                    + __import__("json").dumps(value, sort_keys=True)
                    + "\npost-run output\n",
                    encoding="utf-8",
                )

            expected = record["ranks"][0]["runtime_artifacts"]
            write(record)
            QUALIFICATION_CONTRACT._validate_cpu_reference_log(
                log,
                runtime_kind="in-place",
                mpi_size=1,
                expected_python=expected["python"],
                expected_extension=expected["python_extension"],
                expected_libmeep=expected["libmeep"],
            )

            invalid = copy.deepcopy(record)
            invalid["ranks"][0]["runtime_artifacts"]["libmeep"]["sha256"] = "b" * 64
            write(invalid)
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "not receipt-bound",
            ):
                QUALIFICATION_CONTRACT._validate_cpu_reference_log(
                    log,
                    runtime_kind="in-place",
                    mpi_size=1,
                    expected_python=expected["python"],
                    expected_extension=expected["python_extension"],
                    expected_libmeep=expected["libmeep"],
                )

            invalid = copy.deepcopy(record)
            invalid["ranks"][0]["statistics"]["source_points"] = 0
            write(invalid)
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "no positive work",
            ):
                QUALIFICATION_CONTRACT._validate_cpu_reference_log(
                    log,
                    runtime_kind="in-place",
                    mpi_size=1,
                    expected_python=expected["python"],
                    expected_extension=expected["python_extension"],
                    expected_libmeep=expected["libmeep"],
                )

            invalid = copy.deepcopy(record)
            invalid["ranks"][0]["meep_time"] = 8.000001
            write(invalid)
            with self.assertRaisesRegex(
                QUALIFICATION_CONTRACT.QualificationContractError,
                "physical observables are invalid",
            ):
                QUALIFICATION_CONTRACT._validate_cpu_reference_log(
                    log,
                    runtime_kind="in-place",
                    mpi_size=1,
                    expected_python=expected["python"],
                    expected_extension=expected["python_extension"],
                    expected_libmeep=expected["libmeep"],
                )

    def test_cpu_build_kind_requires_the_exact_cpu_contract(self):
        with self.assertRaisesRegex(
            QUALIFICATION_CONTRACT.QualificationContractError,
            "wrong qualification contract",
        ):
            QUALIFICATION_CONTRACT.validate_v2_receipt(
                {
                    "build_kind": "cpu-mpi-python-fp64",
                    "configuration": {"qualification_contract": "generic"},
                },
                ROOT,
            )

    def test_verifier_emits_stable_identity_and_checks_kind(self):
        receipt = {
            "build_kind": "cpu-mpi-python-fp64",
            "build_input_id": "a" * 64,
            "artifact_set_id": "b" * 64,
            "receipt_id": "c" * 64,
            "source_end": {"sha256": "d" * 64},
        }
        output = io.StringIO()
        with mock.patch.object(
            VERIFIER, "verify_build_receipt", return_value=receipt
        ) as verify, mock.patch.object(
            VERIFIER, "verify_conda_prefix_against_archives"
        ) as archive_verify, contextlib.redirect_stdout(output):
            status = VERIFIER.main(
                [
                    "--repo",
                    str(ROOT),
                    "--receipt",
                    str(ROOT / "synthetic.json"),
                    "--expected-build-kind",
                    "cpu-mpi-python-fp64",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn('"receipt_id":"' + "c" * 64 + '"', output.getvalue())
        self.assertTrue(verify.call_args.kwargs["verify_source"])
        archive_verify.assert_called_once()

        error = io.StringIO()
        with mock.patch.object(
            VERIFIER, "verify_build_receipt", return_value=receipt
        ), mock.patch.object(
            VERIFIER, "verify_conda_prefix_against_archives"
        ), contextlib.redirect_stderr(error):
            status = VERIFIER.main(
                [
                    "--repo",
                    str(ROOT),
                    "--receipt",
                    str(ROOT / "synthetic.json"),
                    "--expected-build-kind",
                    "cuda-mpi-python-fp32",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("build kind mismatch", error.getvalue())

    def test_verifier_replays_archive_audit_from_recorded_external_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary).resolve()
            build = repo / "build" / "meep-cpu-mpi-python-fp64"
            build.mkdir(parents=True)
            cache = repo / "external-cache" / "pkgs"
            cache.mkdir(parents=True)
            prefix = repo / ".envs" / "cpu"
            lock = repo / "environment.lock"
            retained = {
                "schema_version": 3,
                "pass": True,
                "environment_prefix": str(prefix),
                "lock": {"path": str(lock)},
                "package_cache": {"path": str(cache)},
                "archive_snapshot": {
                    "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
                    "hash_and_parse_same_private_snapshot": True,
                    "descriptor_retained_until_terminal_verification": True,
                    "pathname_fingerprint_rechecked": True,
                },
            }
            audit_path = build / "conda-prefix-content-audit.json"
            audit_path.write_text(json.dumps(retained), encoding="utf-8")
            receipt = {
                "build_kind": "cpu-mpi-python-fp64",
                "configuration_files": {
                    "conda_prefix_content_audit": {
                        "path": str(audit_path.relative_to(repo))
                    }
                },
            }

            def reproduce(command, **_kwargs):
                output = pathlib.Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(retained), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                VERIFIER.subprocess, "run", side_effect=reproduce
            ) as run:
                VERIFIER.verify_conda_prefix_against_archives(receipt, repo)
            command = run.call_args.args[0]
            self.assertEqual(
                command[command.index("--package-cache") + 1], str(cache)
            )

            def drift(command, **_kwargs):
                output = pathlib.Path(command[command.index("--output") + 1])
                changed = dict(retained)
                changed["owned_path_count"] = 1
                output.write_text(json.dumps(changed), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(VERIFIER.subprocess, "run", side_effect=drift):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    VERIFIER.verify_conda_prefix_against_archives(receipt, repo)

    def test_cuda_verifier_requires_normalized_source_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary).resolve()
            build = repo / "build/meep-cuda-mpi-python-fp32"
            build.mkdir(parents=True)
            cache = repo / "cache"
            cache.mkdir()
            retained = {
                "schema_version": 4,
                "pass": True,
                "environment_prefix": str(repo / "prefix"),
                "lock": {"path": str(repo / "lock")},
                "package_cache": {"path": str(cache)},
                "archive_snapshot": {
                    "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
                    "hash_and_parse_same_private_snapshot": True,
                    "descriptor_retained_until_terminal_verification": True,
                    "pathname_fingerprint_rechecked": True,
                },
                "relocated_source_bytecode_count": 1,
                "relocated_source_bytecode": [
                    {
                        "current_state": "relocated-source-compiled",
                        "current_sha256": "1" * 64,
                        "derived_sha256": "1" * 64,
                    }
                ],
                "generated_source_bytecode_count": 1,
                "generated_source_bytecode": [
                    {
                        "current_state": "source-compiled",
                        "current_sha256": "2" * 64,
                        "derived_sha256": "2" * 64,
                    }
                ],
            }
            audit = build / "conda-prefix-content-audit.json"
            audit.write_text(json.dumps(retained), encoding="utf-8")
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "configuration_files": {
                    "conda_prefix_content_audit": {
                        "path": str(audit.relative_to(repo))
                    }
                },
            }

            def reproduce(command, **_kwargs):
                pathlib.Path(command[command.index("--output") + 1]).write_text(
                    json.dumps(retained), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                VERIFIER.subprocess, "run", side_effect=reproduce
            ):
                VERIFIER.verify_conda_prefix_against_archives(receipt, repo)

            retained["relocated_source_bytecode"][0]["current_state"] = (
                "archive-exact"
            )
            audit.write_text(json.dumps(retained), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                VERIFIER.verify_conda_prefix_against_archives(receipt, repo)


class CpuMpiFp64BuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BUILDER.read_text(encoding="utf-8")

    def test_builder_has_valid_protected_shell_and_executable_scripts(self):
        subprocess.run(["/bin/bash", "-n", str(BUILDER)], check=True)
        self.assertTrue(self.source.startswith("#!/bin/bash -p\n"))
        for path in (BUILDER, QUALIFIER_PATH, VERIFIER_PATH):
            with self.subTest(path=path.name):
                self.assertTrue(path.stat().st_mode & 0o111)

    def test_builder_is_isolated_exact_lock_cpu_fp64_mpi_python(self):
        for required in (
            'ENV_PREFIX="${REPO_ROOT}/.envs/gpmeep-cpu-mpi-fp64"',
            'BUILD_DIR="${REPO_ROOT}/build/meep-cpu-mpi-python-fp64"',
            'INSTALL_DIR="${REPO_ROOT}/install/meep-cpu-mpi-python-fp64"',
            "--disable-single",
            "--disable-cuda",
            "--disable-cuda-fast-math",
            "--with-mpi",
            "--with-python",
            "--without-scheme",
            "--extra-safety-checks",
            "/usr/bin/timeout --signal=TERM --kill-after=10s",
            "cpu-mpi-python-fp64",
            "python_extension=${BUILD_EXTENSION}",
            "libmeep=${BUILD_LIBMEEP}",
            '$(/usr/bin/nproc)',
            "run_bounded_build_step check 7200 make check",
            "assert_no_runtime_bytecode \"${BUILD_DIR}/python/meep\"",
            "assert_no_runtime_bytecode \"${INSTALL_DIR}\"",
            '--manifest "installed_environment=${ENV_PREFIX}"',
            '--manifest "host_build_logs=${HOST_BUILD_LOGS}"',
            'BUILD_HOME="${BUILD_DIR}/build-home"',
            'export HOME="${BUILD_HOME}"',
            'export XDG_CACHE_HOME="${BUILD_CACHE_HOME}"',
            'export XDG_CONFIG_HOME="${BUILD_CONFIG_HOME}"',
            'export MPLCONFIGDIR="${BUILD_MPLCONFIG}"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)
        self.assertNotIn("--enable-single", self.source)
        self.assertNotIn("--enable-cuda\n", self.source)
        self.assertNotIn("rm -rf", self.source)
        self.assertIn('/usr/bin/mv -- "${old_directory}"', self.source)
        self.assertIn("MAMBA_SHA256=", self.source)
        self.assertNotIn("${ENV_PREFIX}/bin/mktemp", self.source)
        self.assertNotIn("run_bounded_build_step check 7200 make -j", self.source)
        find_contract = self.source[
            self.source.index("assert_no_runtime_bytecode()") :
            self.source.index('assert_no_runtime_bytecode "${BUILD_DIR}/python/meep"')
        ]
        self.assertNotIn("-type", find_contract)

    def test_builder_runs_in_place_and_installed_one_and_two_rank_gates(self):
        for label in (
            "in-place-one-rank",
            "in-place-two-rank",
            "installed-one-rank",
            "installed-two-rank",
        ):
            with self.subTest(label=label):
                self.assertEqual(self.source.count(label), 1)
        self.assertEqual(self.source.count("run_qualification \""), 4)
        self.assertIn("--expected-build-kind \"${BUILD_KIND}\"", self.source)

    def test_runtime_bytecode_gate_rejects_symlink_names(self):
        function_source = self.source[
            self.source.index("assert_no_runtime_bytecode()") :
            self.source.index('assert_no_runtime_bytecode "${BUILD_DIR}/python/meep"')
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "runtime"
            root.mkdir()
            target = pathlib.Path(temporary) / "target"
            target.write_bytes(b"not bytecode")
            for name in ("hidden.pyc", "__pycache__"):
                link = root / name
                link.symlink_to(target)
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "-p",
                        "-c",
                        "set -euo pipefail\n"
                        + function_source
                        + '\nassert_no_runtime_bytecode "$1"',
                        "bytecode-gate",
                        str(root),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                with self.subTest(name=name):
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(name, result.stderr)
                link.unlink()

    def test_user_matrix_requires_cpu_fp64_receipt_to_be_cpu_only(self):
        source = MATRIX_PATH.read_text(encoding="utf-8")
        call = source[source.index('cpu_fp64 = load_build('):]
        call = call[: call.index("fp32 = load_build(")]
        self.assertIn(
            'args.cpu_fp64_receipt,\n            "cpu-mpi-python-fp64",\n'
            "            False,\n            False,",
            call,
        )

if __name__ == "__main__":
    unittest.main()
