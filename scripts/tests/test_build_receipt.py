"""Self-tests for the fail-closed gpmeep build receipt."""

from __future__ import annotations

import importlib.util
import io
import copy
import json
import os
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gpmeep_provenance
import gpmeep_qualification_contract
from gpmeep_provenance import (
    ProvenanceError,
    StatHashCache,
    bounded_command,
    canonical_sha256,
    source_snapshot,
    tree_manifest,
    verify_build_receipt,
)


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "write-build-receipt.py"
QUALIFICATION_SCRIPT = SCRIPT.with_name("gpmeep_qualification_contract.py")
CONTROL_PYTHON_RUNNER = SCRIPT.with_name("gpmeep-control-python.py")
BUILDER = SCRIPT.with_name("build-meep-cuda-mpi-python.sh")
GPU_STEP_DB = BUILDER.parents[1] / "tests" / "gpu-step-db.cpp"
INSTALLED_CUDA_RUNNER = BUILDER.with_name(
    "run-installed-cuda-qualification.py"
)


class BuildReceiptTest(unittest.TestCase):
    def test_cuda_release_kind_cannot_claim_a_generic_contract(self):
        for configuration in ({}, {"qualification_contract": "generic"}):
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "configuration": configuration,
            }
            with self.subTest(configuration=configuration), self.assertRaisesRegex(
                RuntimeError, "wrong qualification contract"
            ):
                gpmeep_qualification_contract.validate_v2_receipt(
                    receipt, self.repo
                )

        # Generic receipts remain outside the qualification-v2 family.  A
        # release consumer must separately require the exact CUDA build kind.
        gpmeep_qualification_contract.validate_v2_receipt(
            {
                "build_kind": "generic",
                "configuration": {"qualification_contract": "generic"},
            },
            self.repo,
        )

    def test_release_builder_forces_and_binds_cuda_fast_math_off(self):
        builder = BUILDER.read_text(encoding="utf-8")
        self.assertIn(
            "authoritative release builds require CUDA fast-math OFF", builder
        )
        self.assertEqual(builder.count("--env MEEP_GPU_FAST_MATH=OFF"), 2)
        self.assertNotIn(
            "for variable in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH", builder
        )
        self.assertIn(
            "for receipt_environment in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH",
            builder,
        )
        self.assertIn("--disable-cuda-fast-math", builder)
        self.assertNotIn("--enable-cuda-fast-math", builder)
        self.assertIn("-DMEEP_GPU_FAST_MATH=OFF", builder)
        self.assertNotIn('-DMEEP_GPU_FAST_MATH="${FAST_MATH}"', builder)
        for binding in (
            'cuda_runtime_flags_stamp=${BUILD_DIR}/src/cuda-runtime-flags.stamp',
            'cuda_runtime_cmake_cache=${CUDA_RUNTIME_BUILD_DIR}/CMakeCache.txt',
        ):
            self.assertEqual(builder.count(binding), 1)

    def test_release_cuda_math_policy_rejects_every_fast_math_injection(self):
        self.build = self.repo / "build" / "meep-cuda-mpi-python-fp32"
        self.build.mkdir(parents=True)
        canonical_path = self.build / "canonical.json"
        status_path = self.build / "config.status"
        stamp_path = self.build / "src" / "cuda-runtime-flags.stamp"
        cache_path = (
            self.build / "cuda-runtime-qualification" / "CMakeCache.txt"
        )
        stamp_path.parent.mkdir()
        cache_path.parent.mkdir()
        canonical_path = self.build / "canonical-build-environment.json"
        compiler_dir = self.repo / ".envs" / "meep-gpu-cuda-mpi" / "bin"
        compiler_dir.mkdir(parents=True)
        nvcc_path = compiler_dir / "nvcc"
        host_path = compiler_dir / "x86_64-conda-linux-gnu-c++"
        nvcc_path.write_text("synthetic nvcc\n", encoding="utf-8")
        host_path.write_text("synthetic host compiler\n", encoding="utf-8")
        canonical_off = {
            "schema_version": 2,
            "environment": {
                "MEEP_GPU_FAST_MATH": "OFF",
                "NVCC_PREPEND_FLAGS": f"-ccbin={host_path}",
                "CXX": str(host_path),
            },
            "normalized_shell_keys": ["SHLVL", "_"],
        }
        status_off = (
            "  set X /bin/bash '../../configure' "
            "'--disable-cuda-fast-math' 'CC=/synthetic'\n"
            'S["NVCCFLAGS"]="-O3"\n'
        )
        stamp_off = (
            f"NVCC={nvcc_path}\n"
            f"CUDAHOSTCXX={host_path}\n"
            "NVCCFLAGS=-O3\n"
            "CUDA_ARCH_FLAGS=-gencode=arch=compute_80,code=sm_80\n"
        )
        cache_off = (
            "//Compile CUDA kernels with --use_fast_math\n"
            "MEEP_GPU_FAST_MATH:BOOL=OFF\n"
            "CMAKE_CUDA_FLAGS:STRING=\n"
            "CMAKE_CUDA_FLAGS_DEBUG:STRING=-g\n"
            "CMAKE_CUDA_FLAGS_MINSIZEREL:STRING=-O1 -DNDEBUG\n"
            "CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
            "CMAKE_CUDA_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG\n"
            f"CMAKE_CUDA_COMPILER:UNINITIALIZED={nvcc_path}\n"
            f"CMAKE_CUDA_HOST_COMPILER:UNINITIALIZED={host_path}\n"
        )

        def restore_files() -> None:
            canonical_path.write_text(json.dumps(canonical_off), encoding="utf-8")
            status_path.write_text(status_off, encoding="utf-8")
            stamp_path.write_text(stamp_off, encoding="utf-8")
            cache_path.write_text(cache_off, encoding="utf-8")

        restore_files()
        receipt = {
            "build_kind": "cuda-mpi-python-fp32",
            "build_dir": str(self.build),
            "configuration": {
                "builder": {
                    "path": str(
                        self.repo / "scripts" / "build-meep-cuda-mpi-python.sh"
                    )
                },
                "configure_argv": ["--disable-cuda-fast-math"],
                "environment": {"MEEP_GPU_FAST_MATH": "OFF"},
            },
            "configuration_files": {
                "canonical_build_environment": {"path": str(canonical_path)},
                "config_status": {"path": str(status_path)},
                "cuda_runtime_flags_stamp": {"path": str(stamp_path)},
                "cuda_runtime_cmake_cache": {"path": str(cache_path)},
            },
            "toolchain": {"nvcc": {"path": str(nvcc_path)}},
        }
        validate = gpmeep_qualification_contract.validate_release_cuda_math_policy
        validate(receipt, self.repo)

        for argv in (["--enable-cuda-fast-math"], []):
            bad = copy.deepcopy(receipt)
            bad["configuration"]["configure_argv"] = argv
            with self.subTest(argv=argv), self.assertRaisesRegex(
                RuntimeError, "fast-math disable"
            ):
                validate(bad, self.repo)
        for value in ("ON", None):
            bad = copy.deepcopy(receipt)
            bad["configuration"]["environment"]["MEEP_GPU_FAST_MATH"] = value
            with self.subTest(receipt_environment=value), self.assertRaisesRegex(
                RuntimeError, "MEEP_GPU_FAST_MATH=OFF"
            ):
                validate(bad, self.repo)

        canonical_on = copy.deepcopy(canonical_off)
        canonical_on["environment"]["MEEP_GPU_FAST_MATH"] = "ON"
        canonical_path.write_text(json.dumps(canonical_on), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "canonical.*OFF"):
            validate(receipt, self.repo)
        restore_files()
        canonical_unsafe = copy.deepcopy(canonical_off)
        canonical_unsafe["environment"]["NVCC_PREPEND_FLAGS"] += " --use_fast_math"
        canonical_path.write_text(json.dumps(canonical_unsafe), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unsafe NVCC"):
            validate(receipt, self.repo)
        restore_files()
        self.assertFalse(
            gpmeep_qualification_contract._contains_unsafe_cuda_math_token(
                ["-ccbin=/workspace/fast_math_disabled/bin/c++"]
            )
        )
        self.assertFalse(
            gpmeep_qualification_contract._contains_unsafe_cuda_math_token(
                ["-I/workspace/fast-math-disabled/include"]
            )
        )

        for payload, diagnostic in (
            (stamp_off.replace("NVCCFLAGS=-O3", "NVCCFLAGS=-O3 --use_fast_math"), "strict -O3"),
            (stamp_off.replace("CUDA_ARCH_FLAGS=-gencode=arch=compute_80,code=sm_80", "CUDA_ARCH_FLAGS=--options-file=/tmp/unsafe.rsp"), "strict -O3"),
            (stamp_off + "NVCCFLAGS=-O3\n", "duplicate or malformed"),
            (stamp_off.replace("NVCCFLAGS=-O3\n", ""), "incomplete"),
        ):
            stamp_path.write_text(payload, encoding="utf-8")
            with self.subTest(stamp=diagnostic), self.assertRaisesRegex(
                RuntimeError, diagnostic
            ):
                validate(receipt, self.repo)
            restore_files()

        alternate_compiler = compiler_dir / "alternate-compiler"
        alternate_compiler.write_text("alternate\n", encoding="utf-8")
        stamp_path.write_text(
            stamp_off.replace(f"NVCC={nvcc_path}", f"NVCC={alternate_compiler}"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "compiler identity is unbound"):
            validate(receipt, self.repo)
        restore_files()
        cache_path.write_text(
            cache_off.replace(
                f"CMAKE_CUDA_COMPILER:UNINITIALIZED={nvcc_path}",
                f"CMAKE_CUDA_COMPILER:UNINITIALIZED={alternate_compiler}",
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "compiler identity is unbound"):
            validate(receipt, self.repo)
        restore_files()

        for payload, diagnostic in (
            (cache_off.replace("MEEP_GPU_FAST_MATH:BOOL=OFF", "MEEP_GPU_FAST_MATH:BOOL=ON"), "does not disable"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--use_fast_math"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--ftz=true"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--prec-div=false"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--prec-sqrt=false"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--ftz true"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=-prec-div false"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--prec-sqrt false"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=--options-file=/tmp/unsafe.rsp"), "unsafe fast-math"),
            (cache_off.replace("CMAKE_CUDA_FLAGS:STRING=", "CMAKE_CUDA_FLAGS:STRING=-Xcompiler=-ffast-math"), "unsafe fast-math"),
        ):
            cache_path.write_text(payload, encoding="utf-8")
            with self.subTest(cache=diagnostic), self.assertRaisesRegex(
                RuntimeError, diagnostic
            ):
                validate(receipt, self.repo)
            restore_files()

        for name in tuple(receipt["configuration_files"]):
            bad = copy.deepcopy(receipt)
            del bad["configuration_files"][name]
            with self.subTest(missing=name), self.assertRaisesRegex(
                RuntimeError, "binding is invalid"
            ):
                validate(bad, self.repo)
        for name in tuple(receipt["configuration_files"]):
            source = pathlib.Path(receipt["configuration_files"][name]["path"])
            alternate = self.build / f"alternate-{name}.txt"
            alternate.write_bytes(source.read_bytes())
            bad = copy.deepcopy(receipt)
            bad["configuration_files"][name]["path"] = str(alternate)
            with self.subTest(decoy=name), self.assertRaisesRegex(
                RuntimeError, "not build-bound"
            ):
                validate(bad, self.repo)
        alternate_root = self.repo / "alternate-release-tree"
        alternate_paths = {
            "canonical_build_environment": "canonical-build-environment.json",
            "config_status": "config.status",
            "cuda_runtime_flags_stamp": "src/cuda-runtime-flags.stamp",
            "cuda_runtime_cmake_cache": "cuda-runtime-qualification/CMakeCache.txt",
        }
        joint_decoy = copy.deepcopy(receipt)
        joint_decoy["build_dir"] = str(alternate_root)
        for name, relative in alternate_paths.items():
            source = pathlib.Path(receipt["configuration_files"][name]["path"])
            destination = alternate_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            joint_decoy["configuration_files"][name]["path"] = str(destination)
        with self.assertRaisesRegex(RuntimeError, "fixed authoritative path"):
            validate(joint_decoy, self.repo)

    def test_expected_failure_status_rejects_timeout_exec_and_all_signals(self):
        acceptable = gpmeep_qualification_contract.expected_failure_status_is_acceptable
        self.assertTrue(acceptable(1))
        self.assertTrue(acceptable(123))
        for status in (0, 124, 125, 126, 127, 128, 129, 137, 143, 255):
            with self.subTest(status=status):
                self.assertFalse(acceptable(status))
        builder = BUILDER.read_text(encoding="utf-8")
        self.assertEqual(
            builder.count("if ! expected_failure_status_is_acceptable"), 2
        )

    def test_step_finite_failure_payload_requires_each_rank_and_attestation(self):
        name = gpmeep_qualification_contract.STEP_FINITE_FAILURE_LOG_NAME
        diagnostic = (
            gpmeep_qualification_contract.STEP_FINITE_FAILURE_DIAGNOSTIC
        )

        def payload(markers=(0, 1), status=1, returned=False):
            lines = [
                f"gpmeep-finite-abort-probe:rank={rank},"
                "stage=before-failing-step"
                for rank in markers
            ]
            lines.extend([f"meep: {diagnostic}", f"meep: {diagnostic}"])
            if returned:
                lines.append(
                    "FAIL: fields::step returned after rank-local device NaN "
                    "on rank 0"
                )
            lines.extend(
                [
                    f"gpmeep-expected-mpi-failure:{name}:status={status}:"
                    f"diagnostic={diagnostic}",
                    gpmeep_qualification_contract.marker_for(name).decode(
                        "ascii"
                    ),
                ]
            )
            return ("\n".join(lines) + "\n").encode("ascii")

        gpmeep_qualification_contract.validate_step_finite_failure_payload(
            payload()
        )
        for bad_payload in (
            payload(markers=(0, 0)),
            payload(markers=(1, 1)),
            payload(status=124),
            payload(returned=True),
        ):
            with self.subTest(payload=bad_payload):
                with self.assertRaises(
                    gpmeep_qualification_contract.QualificationContractError
                ):
                    gpmeep_qualification_contract.validate_step_finite_failure_payload(
                        bad_payload
                    )

    def test_builder_checks_finite_failure_markers_per_rank(self):
        builder = BUILDER.read_text(encoding="utf-8")
        self.assertIn("for finite_rank in 0 1; do", builder)
        self.assertIn(
            '"rank=${finite_rank},stage=before-failing-step"', builder
        )
        self.assertIn(
            "'meep: simulation fields are NaN or Inf'", builder
        )
        source = GPU_STEP_DB.read_text(encoding="utf-8")
        probe = source.split(
            "void run_distributed_step_finite_abort_probe()", 1
        )[1].split("void run_distributed_near2far_failure_abort_probe()", 1)[0]
        self.assertEqual(probe.count("write_stderr_line_atomically("), 7)
        self.assertNotIn(
            'std::cerr << "gpmeep-finite-abort-probe:', probe
        )
        self.assertRegex(
            source,
            r"write\(STDERR_FILENO, payload\.data\(\), payload\.size\(\)\)",
        )
        helper = source.split(
            "void write_stderr_line_atomically", 1
        )[1].split("struct diagonal_bicgstab_data", 1)[0]
        self.assertNotIn("require(", helper)

    def test_builder_pass_producers_exactly_match_central_v2_log_set(self):
        builder = BUILDER.read_text(encoding="utf-8")
        literal = set(
            re.findall(
                r"gpmeep-qualification:([A-Za-z0-9_.-]+\.log):PASS",
                builder,
            )
        )
        expected_failures = set(
            re.findall(
                r"run_expected_(?:mpi|preflight)_failure \\\n"
                r"  ([A-Za-z0-9_.-]+)",
                builder,
            )
        )
        produced = (
            literal
            | {f"{name}.log" for name in expected_failures}
            | set(gpmeep_qualification_contract.DIRECT_ELF_LOG_BINDINGS)
            | set(gpmeep_qualification_contract.PYTHON_RUNTIME_LOG_BINDINGS)
            | {
                gpmeep_qualification_contract.AUDIT_LOG_NAME,
                gpmeep_qualification_contract.INSTALLED_MPB_LOG_NAME,
                gpmeep_qualification_contract.SENTINEL_LOG_NAME,
            }
        )
        self.assertEqual(
            produced,
            set(gpmeep_qualification_contract.ALL_REQUIRED_QUALIFICATION_LOGS),
        )

    def test_cw_breakdown_qualification_covers_singleton_and_two_rank_consensus(self):
        builder = BUILDER.read_text(encoding="utf-8")
        for name, ranks in (
            ("gpu-step-db-cw-breakdown-one-rank", 1),
            ("gpu-step-db-cw-breakdown-two-rank", 2),
        ):
            with self.subTest(name=name):
                redirect = builder.index(
                    f'>"${{TEST_LOG_DIR}}/{name}.log" 2>&1'
                )
                start = builder.rfind("env HOME=", 0, redirect)
                self.assertNotEqual(start, -1)
                invocation = builder[start:redirect]
                self.assertIn("MEEP_GPU_TEST_CW_BREAKDOWN_ONLY=1", invocation)
                self.assertIn(
                    f'"${{PREFIX}}/bin/mpiexec" -n {ranks} \\', invocation
                )
                self.assertIn("MEEP_GPU_BACKEND=cuda", invocation)
                self.assertIn("MEEP_GPU_STRICT=1", invocation)

        source = GPU_STEP_DB.read_text(encoding="utf-8")
        lane = source.split(
            'if (std::getenv("MEEP_GPU_TEST_CW_BREAKDOWN_ONLY"))', 1
        )[1].split('if (std::getenv("MEEP_GPU_TEST_CW_SOLVER_ONLY"))', 1)[0]
        self.assertIn("if (count_processors() > 1)", lane)
        self.assertIn("rank-asymmetric resident CW numerical breakdown", lane)
        self.assertIn("singleton resident CW numerical breakdown", lane)

    def test_near2far2d_qualification_requires_two_rank_strict_cuda(self):
        builder = BUILDER.read_text(encoding="utf-8")
        name = "gpu-step-db-near2far2d-two-gpu"
        redirect = builder.index(f'>"${{TEST_LOG_DIR}}/{name}.log" 2>&1')
        start = builder.rfind("env HOME=", 0, redirect)
        self.assertNotEqual(start, -1)
        invocation = builder[start:redirect]
        self.assertIn("MEEP_GPU_TEST_NEAR2FAR_2D_MPI_ONLY=1", invocation)
        self.assertIn('"${PREFIX}/bin/mpiexec" -n 2 \\', invocation)
        self.assertIn("MEEP_GPU_BACKEND=cuda", invocation)
        self.assertIn("MEEP_GPU_STRICT=1", invocation)
        self.assertIn("MEEP_GPU_MPI_TRANSPORT=pinned", invocation)

        source = GPU_STEP_DB.read_text(encoding="utf-8")
        lane = source.split(
            "void require_near2far2d_two_gpu_mpi_contract()", 1
        )[1].split("void require_near2far_cross_rank_collective_cancellation()", 1)[0]
        self.assertIn("count_processors() == 2", lane)
        self.assertIn("gpu::selected_device_identifier()", lane)
        self.assertIn("near.F = nullptr", lane)
        self.assertIn("13 * targets.size()", lane)
        self.assertIn("require_near2far_complex_agreement", lane)
        self.assertIn("require_near2far2d_two_gpu_mpi_contract();", source)

    def test_near2farcyl_qualification_requires_two_rank_strict_cuda(self):
        builder = BUILDER.read_text(encoding="utf-8")
        name = "gpu-step-db-near2farcyl-two-gpu"
        redirect = builder.index(f'>"${{TEST_LOG_DIR}}/{name}.log" 2>&1')
        start = builder.rfind("env HOME=", 0, redirect)
        self.assertNotEqual(start, -1)
        invocation = builder[start:redirect]
        self.assertIn("MEEP_GPU_TEST_NEAR2FAR_CYL_MPI_ONLY=1", invocation)
        self.assertIn('"${PREFIX}/bin/mpiexec" -n 2 \\', invocation)
        self.assertIn("MEEP_GPU_BACKEND=cuda", invocation)
        self.assertIn("MEEP_GPU_STRICT=1", invocation)
        self.assertIn("MEEP_GPU_MPI_TRANSPORT=pinned", invocation)

        source = GPU_STEP_DB.read_text(encoding="utf-8")
        lane = source.split(
            "void require_near2far_cylindrical_two_gpu_mpi_contract()", 1
        )[1].split("void require_near2far_channel_cancellation_retry()", 1)[0]
        self.assertIn("count_processors() == 2", lane)
        self.assertIn("gpu::selected_device_identifier()", lane)
        self.assertIn("chunk->fc->m == azimuthal_mode", lane)
        self.assertIn("near.F = nullptr", lane)
        self.assertIn("13 * targets.size()", lane)
        self.assertIn("require_near2far_complex_agreement", lane)
        self.assertIn(
            "require_near2far_cylindrical_two_gpu_mpi_contract();", source
        )

    def test_ldos_migration_qualification_requires_one_rank_and_two_devices(self):
        builder = BUILDER.read_text(encoding="utf-8")
        name = "gpu-step-db-ldos-migration-one-rank"
        redirect = builder.index(
            f'>"${{TEST_LOG_DIR}}/{name}.log" 2>&1'
        )
        start = builder.rfind("env HOME=", 0, redirect)
        self.assertNotEqual(start, -1)
        invocation = builder[start:redirect]
        self.assertIn("MEEP_GPU_TEST_LDOS_MIGRATION_ONLY=1", invocation)
        self.assertIn('"${PREFIX}/bin/mpiexec" -n 1 \\', invocation)
        self.assertIn("MEEP_GPU_BACKEND=cuda", invocation)
        self.assertIn("MEEP_GPU_STRICT=1", invocation)
        self.assertNotIn("SKIP", invocation)
        self.assertIn(
            "gpmeep-qualification:" + name + ".log:PASS", builder
        )

        source = GPU_STEP_DB.read_text(encoding="utf-8")
        self.assertIn(
            "require_resident_ldos_two_device_migration_contract();", source
        )

    def test_rank_local_backend_injections_execute_the_preflight_lane(self):
        builder = BUILDER.read_text(encoding="utf-8")
        cases = (
            (
                "automatic-backend-request-mismatch-two-rank",
                "MEEP_GPU_TEST_RANK_BACKEND_MISMATCH=1",
            ),
            (
                "automatic-threshold-mismatch-two-rank",
                "MEEP_GPU_TEST_RANK_AUTO_THRESHOLD_MISMATCH=1",
            ),
            (
                "automatic-invalid-device-two-rank",
                "MEEP_GPU_TEST_RANK_INVALID_DEVICE=1",
            ),
        )
        for name, injection in cases:
            with self.subTest(name=name):
                call_start = builder.index(
                    "run_expected_preflight_failure \\\n" + f"  {name} 2 \\\n"
                )
                next_call = builder.find("\nrun_expected_", call_start + 1)
                self.assertNotEqual(next_call, -1)
                call = builder[call_start:next_call]
                self.assertIn("MEEP_GPU_TEST_PREFLIGHT_ONLY=1", call)
                self.assertIn(injection, call)
                self.assertIn("MEEP_GPU_BACKEND=cuda", call)

    def test_qualification_log_requires_unique_exact_terminal_marker(self):
        name = "example.log"
        marker = gpmeep_qualification_contract.marker_for(name)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / name
            path.write_bytes(b"output\n" + marker + b"\n")
            record = gpmeep_qualification_contract.validate_log(path, name)
            self.assertEqual(record["terminal_marker"], marker.decode("ascii"))
            path.write_bytes(b"output\n" + marker + b"\ntrailing\n")
            with self.assertRaisesRegex(RuntimeError, "terminal PASS"):
                gpmeep_qualification_contract.validate_log(path, name)
            path.write_bytes(marker + b"\n" + marker + b"\n")
            with self.assertRaisesRegex(RuntimeError, "terminal PASS"):
                gpmeep_qualification_contract.validate_log(path, name)

    def test_python_runtime_provenance_is_bound_by_path_size_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary) / "repo"
            log_dir = pathlib.Path(temporary) / "logs"
            repo.mkdir()
            log_dir.mkdir()
            paths = {
                "python_extension": repo / "build/_meep.so",
                "libmeep": repo / "build/libmeep.so",
                "installed_python_extension": repo / "install/_meep.so",
                "installed_libmeep": repo / "install/libmeep.so",
            }
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + "\n").encode("ascii"))
            artifacts = {
                name: gpmeep_provenance.file_record(path, repo)
                for name, path in paths.items()
            }
            for log_name, (extension_name, library_name) in (
                gpmeep_qualification_contract.PYTHON_RUNTIME_LOG_BINDINGS.items()
            ):
                attestation = {
                    "python_extension": artifacts[extension_name],
                    "libmeep": artifacts[library_name],
                }
                (log_dir / log_name).write_text(
                    json.dumps(attestation, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    + gpmeep_qualification_contract.marker_for(log_name).decode(
                        "ascii"
                    )
                    + "\n",
                    encoding="utf-8",
                )

            gpmeep_qualification_contract.validate_python_runtime_logs(
                log_dir, artifacts, repo
            )
            forged_log = next(
                iter(gpmeep_qualification_contract.PYTHON_RUNTIME_LOG_BINDINGS)
            )
            lines = (log_dir / forged_log).read_text(encoding="utf-8").splitlines()
            forged = json.loads(lines[-2])
            forged["libmeep"]["sha256"] = "0" * 64
            (log_dir / forged_log).write_text(
                json.dumps(forged, sort_keys=True, separators=(",", ":"))
                + "\n"
                + lines[-1]
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not receipt-bound"):
                gpmeep_qualification_contract.validate_python_runtime_logs(
                    log_dir, artifacts, repo
                )

    def test_qualification_directory_requires_exact_116_log_set(self):
        self.assertEqual(
            len(
                gpmeep_qualification_contract.ALL_QUALIFICATION_DIRECTORY_LOGS
            ),
            116,
        )
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = pathlib.Path(temporary)
            materialization = set(
                gpmeep_qualification_contract.MATERIALIZATION_LOGS
            )
            for name in (
                gpmeep_qualification_contract.ALL_QUALIFICATION_DIRECTORY_LOGS
            ):
                if name in materialization:
                    marker = f"gpmeep-libtool-materialization:{name}:PASS"
                    payload = f"materialized\n{marker}\n"
                else:
                    payload = "placeholder\n"
                (log_dir / name).write_text(payload, encoding="utf-8")

            records = (
                gpmeep_qualification_contract.validate_no_uncontracted_pass_logs(
                    log_dir
                )
            )
            self.assertEqual(set(records), materialization)

            extra = log_dir / "unexpected-markerless.log"
            extra.write_text("not a PASS marker\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact contracted set"):
                gpmeep_qualification_contract.validate_no_uncontracted_pass_logs(
                    log_dir
                )
            extra.unlink()

            audit = log_dir / gpmeep_qualification_contract.AUDIT_LOG_NAME
            audit.unlink()
            gpmeep_qualification_contract.validate_no_uncontracted_pass_logs(
                log_dir, include_audit=False
            )
            missing = log_dir / next(iter(materialization))
            missing.unlink()
            with self.assertRaisesRegex(RuntimeError, "exact contracted set"):
                gpmeep_qualification_contract.validate_no_uncontracted_pass_logs(
                    log_dir, include_audit=False
                )

    def test_installed_cuda_qualification_is_same_process_rank_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary) / "repo"
            log_dir = pathlib.Path(temporary) / "logs"
            repo.mkdir()
            log_dir.mkdir()
            extension = repo / "install/meep/_meep.so"
            library = repo / "install/lib/libmeep.so"
            extension.parent.mkdir(parents=True)
            library.parent.mkdir(parents=True)
            extension.write_bytes(b"installed extension\n")
            library.write_bytes(b"installed libmeep\n")
            artifacts = {
                "installed_python_extension": gpmeep_provenance.file_record(
                    extension, repo
                ),
                "installed_libmeep": gpmeep_provenance.file_record(
                    library, repo
                ),
            }

            def artifact_value(record):
                return {
                    "path": record["path"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }

            def value_for(log_name):
                spec = gpmeep_qualification_contract.INSTALLED_CUDA_QUALIFICATION_SPECS[
                    log_name
                ]
                suite_path = repo / spec["test_file"]
                suite_path.parent.mkdir(parents=True, exist_ok=True)
                suite_path.write_text(
                    f"# source-bound suite: {spec['test_file']}\n",
                    encoding="utf-8",
                )
                suite_record = gpmeep_provenance.file_record(suite_path, repo)
                rank_records = []
                for rank in range(spec["world_size"]):
                    rank_records.append(
                        {
                            "world_rank": rank,
                            "world_size": spec["world_size"],
                            "suite": {
                                "tests_run": spec["tests_run"],
                                "failures": 0,
                                "errors": 0,
                                "skipped": 0,
                                "expected_failures": 0,
                                "unexpected_successes": 0,
                                "successful": True,
                            },
                            "python_extension": artifact_value(
                                artifacts["installed_python_extension"]
                            ),
                            "libmeep": artifact_value(
                                artifacts["installed_libmeep"]
                            ),
                            "probe": {
                                "requested_backend": "cuda",
                                "active_backend": "cuda",
                                "selected_device": rank,
                                "selected_device_identifier": f"GPU-{rank}",
                                "cuda_execution_selected": True,
                                "execution_diagnostic": "selected CUDA",
                                "statistics": {
                                    "dispatch": {
                                        "cuda_curl_calls": 2,
                                        "cpu_curl_calls": 0,
                                    },
                                    "field_updates": {
                                        "cuda_update_eh_calls": 2,
                                        "cpu_update_eh_calls": 0,
                                    },
                                    "sources": {
                                        "cuda_source_calls": 1 if rank == 0 else 0,
                                        "cpu_source_calls": 0,
                                    },
                                },
                            },
                        }
                    )
                return {
                    "schema_version": 1,
                    "qualification": (
                        gpmeep_qualification_contract.INSTALLED_CUDA_QUALIFICATION_NAME
                    ),
                    "log_name": log_name,
                    "mode": spec["mode"],
                    "expected_world_size": spec["world_size"],
                    "environment": spec["environment"],
                    "suite_file": artifact_value(suite_record),
                    "python_extension": artifact_value(
                        artifacts["installed_python_extension"]
                    ),
                    "libmeep": artifact_value(
                        artifacts["installed_libmeep"]
                    ),
                    "rank_records": rank_records,
                }

            values = {
                name: value_for(name)
                for name in (
                    gpmeep_qualification_contract.INSTALLED_CUDA_QUALIFICATION_SPECS
                )
            }

            def write_logs(overrides=None):
                current = values if overrides is None else overrides
                for name, value in current.items():
                    prefix = ""
                    if value["mode"] == "adjoint":
                        directions = (
                            "cosine",
                            "quasiperiodic-sine",
                            "component-6",
                            "component-12",
                            "component-18",
                        )
                        prefix = "".join(
                            "gpmeep-fd-direction:"
                            + json.dumps({"name": direction, "pass": True})
                            + "\n"
                            for direction in directions
                        )
                    (log_dir / name).write_text(
                        prefix
                        + json.dumps(value, sort_keys=True, separators=(",", ":"))
                        + "\n"
                        + gpmeep_qualification_contract.marker_for(name).decode(
                            "ascii"
                        )
                        + "\n",
                        encoding="utf-8",
                    )

            write_logs()
            gpmeep_qualification_contract.validate_installed_cuda_qualification_logs(
                log_dir, artifacts, repo
            )

            two_rank_name = "installed-python-gpu-backend-two-rank.log"
            mutations = []
            cpu_fallback = copy.deepcopy(values)
            cpu_fallback[two_rank_name]["rank_records"][0]["probe"][
                "statistics"
            ]["dispatch"]["cpu_curl_calls"] = 1
            mutations.append(cpu_fallback)
            duplicate_device = copy.deepcopy(values)
            duplicate_device[two_rank_name]["rank_records"][1]["probe"][
                "selected_device_identifier"
            ] = "GPU-0"
            mutations.append(duplicate_device)
            missing_rank = copy.deepcopy(values)
            missing_rank[two_rank_name]["rank_records"].pop()
            mutations.append(missing_rank)
            failed_suite = copy.deepcopy(values)
            failed_suite[two_rank_name]["rank_records"][0]["suite"][
                "successful"
            ] = False
            mutations.append(failed_suite)
            wrong_extension = copy.deepcopy(values)
            wrong_extension[two_rank_name]["rank_records"][0][
                "python_extension"
            ]["sha256"] = "0" * 64
            mutations.append(wrong_extension)

            for mutated in mutations:
                with self.subTest(mutation=mutations.index(mutated)):
                    write_logs(mutated)
                    with self.assertRaises(RuntimeError):
                        gpmeep_qualification_contract.validate_installed_cuda_qualification_logs(
                            log_dir, artifacts, repo
                        )

    def test_bounded_command_stops_at_stdout_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = bounded_command(
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, b'x' * 1048576)",
                ],
                cwd=pathlib.Path(temporary),
                timeout_seconds=5,
                stdout_limit=4096,
                stderr_limit=4096,
            )
        self.assertEqual(result["output_limit"], "stdout")
        self.assertFalse(result["timeout"])
        self.assertEqual(result["stdout"], b"x" * 4096)

    def test_bounded_command_times_out_after_child_closes_pipes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = bounded_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,time; os.close(1); os.close(2); "
                        "time.sleep(60)"
                    ),
                ],
                cwd=pathlib.Path(temporary),
                timeout_seconds=0.2,
                stdout_limit=4096,
                stderr_limit=4096,
            )
        self.assertTrue(result["timeout"])
        self.assertIsNone(result["output_limit"])
        self.assertIsNotNone(result["exit_code"])

    def test_bounded_command_cleans_same_group_helper_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sentinel = root / "leftover-process.txt"
            helper = (
                "import pathlib,signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(1); "
                f"pathlib.Path({str(sentinel)!r}).write_text('leftover')"
            )
            leader = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {helper!r}]); "
                "time.sleep(0.1)"
            )
            result = bounded_command(
                [sys.executable, "-c", leader],
                cwd=root,
                timeout_seconds=0.2,
                stdout_limit=4096,
                stderr_limit=4096,
            )
            time.sleep(1.0)
            helper_survived = sentinel.exists()
        self.assertTrue(result["timeout"])
        self.assertFalse(helper_survived)

    def test_bounded_command_can_stream_stdout_to_open_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            destination = root / "payload.bin"
            with destination.open("xb") as target:
                result = bounded_command(
                    [
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'payload')",
                    ],
                    cwd=root,
                    timeout_seconds=5,
                    stderr_limit=4096,
                    stdout_fd=target.fileno(),
                )
            payload = destination.read_bytes()
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timeout"])
        self.assertIsNone(result["output_limit"])
        self.assertEqual(result["stdout"], b"")
        self.assertEqual(payload, b"payload")

    def test_bounded_command_caps_stdout_streamed_to_open_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            destination = root / "bounded-payload.bin"
            with destination.open("xb") as target:
                result = bounded_command(
                    [
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'x' * 8192)",
                    ],
                    cwd=root,
                    timeout_seconds=5,
                    stdout_limit=4096,
                    stderr_limit=4096,
                    stdout_fd=target.fileno(),
                )
            payload = destination.read_bytes()
        self.assertEqual(result["output_limit"], "stdout")
        self.assertFalse(result["timeout"])
        self.assertEqual(result["stdout"], b"")
        self.assertEqual(payload, b"x" * 4096)

    def test_stat_hash_cache_reuses_unchanged_file_and_rehashes_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "payload.bin"
            path.write_bytes(b"first")
            cache = StatHashCache()
            original = gpmeep_provenance.sha256_file
            with mock.patch.object(
                gpmeep_provenance, "sha256_file", wraps=original
            ) as hashing:
                first = cache.digest(path)
                self.assertEqual(cache.digest(path), first)
                self.assertEqual(hashing.call_count, 1)
                path.write_bytes(b"other")
                second = cache.digest(path)
                self.assertNotEqual(second, first)
                self.assertEqual(hashing.call_count, 2)

    def test_stat_hash_cache_rejects_path_swap_while_open_inode_is_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "payload.bin"
            replacement = root / "replacement.bin"
            path.write_bytes(b"original")
            replacement.write_bytes(b"replaced")
            original_hash = gpmeep_provenance.sha256_file

            def swap_then_hash(open_descriptor_path):
                os.replace(replacement, path)
                return original_hash(open_descriptor_path)

            with mock.patch.object(
                gpmeep_provenance,
                "sha256_file",
                side_effect=swap_then_hash,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while hashing"):
                    StatHashCache().digest(path)

    def test_writer_loads_provenance_source_not_valid_adjacent_pyc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(SCRIPT, scripts / SCRIPT.name)
            shutil.copy2(QUALIFICATION_SCRIPT, scripts / QUALIFICATION_SCRIPT.name)
            provenance = scripts / "gpmeep_provenance.py"
            template = (
                "import pathlib\n"
                "pathlib.Path(__file__).with_name('sentinel.txt').write_text(%r)\n"
                "atomic_write_json = canonical_sha256 = command_probe = file_record = "
                "git_output = source_snapshot = tree_manifest = lambda *a, **k: None\n"
            )
            source_text = template % "SOURCE"
            cached_text = template % "CACHE!"
            self.assertEqual(len(source_text.encode()), len(cached_text.encode()))
            fixed_timestamp = 1_700_000_000
            provenance.write_text(cached_text, encoding="utf-8")
            os.utime(provenance, (fixed_timestamp, fixed_timestamp))
            # This regression deliberately creates an adjacent cache even
            # when the outer test runner itself uses a private cache prefix.
            with mock.patch.object(sys, "pycache_prefix", None):
                cache_path = pathlib.Path(
                    importlib.util.cache_from_source(str(provenance))
                )
            cache_path.parent.mkdir()
            py_compile.compile(str(provenance), cfile=str(cache_path), doraise=True)
            provenance.write_text(source_text, encoding="utf-8")
            os.utime(provenance, (fixed_timestamp, fixed_timestamp))
            process = subprocess.run(
                [sys.executable, str(scripts / SCRIPT.name), "--help"],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(
                (scripts / "sentinel.txt").read_text(encoding="utf-8"), "SOURCE"
            )

    def test_tree_manifest_includes_executable_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            cache = package / "__pycache__"
            cache.mkdir(parents=True)
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (cache / "module.cpython-311.pyc").write_bytes(b"executable-bytecode")
            manifest = tree_manifest(package, root)
            paths = {record["path"] for record in manifest["files"]}
            self.assertIn("__pycache__/module.cpython-311.pyc", paths)

    def test_tree_manifest_can_explicitly_exclude_volatile_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            cache = package / "__pycache__"
            cache.mkdir(parents=True)
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            bytecode = cache / "module.cpython-311.pyc"
            bytecode.write_bytes(b"first-bytecode")
            manifest = tree_manifest(
                package, root, excluded_suffixes=(".pyc",)
            )
            self.assertEqual(manifest["excluded_suffixes"], [".pyc"])
            self.assertEqual(
                [record["path"] for record in manifest["files"]],
                ["module.py"],
            )
            bytecode.write_bytes(b"changed-bytecode")
            self.assertEqual(
                tree_manifest(package, root, excluded_suffixes=(".pyc",)),
                manifest,
            )

    def test_tree_manifest_binds_internal_symlink_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "target-a").write_bytes(b"identical")
            (package / "target-b").write_bytes(b"identical")
            alias = package / "alias"
            alias.symlink_to("target-a")
            before = tree_manifest(package, root)
            self.assertEqual(before["tree_manifest_schema_version"], 2)
            self.assertEqual(before["symlink_count"], 1)
            self.assertEqual(before["symlinks"][0]["link_target"], "target-a")
            alias.unlink()
            alias.symlink_to("target-b")
            self.assertNotEqual(tree_manifest(package, root), before)

    def test_tree_manifest_rejects_external_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir()
            external = root / "external"
            external.write_bytes(b"identical")
            (package / "alias").symlink_to("../external")
            with self.assertRaisesRegex(RuntimeError, "escapes its root"):
                tree_manifest(package, root)

    def test_tree_manifest_does_not_hide_receipt_named_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "build-provenance.json").write_text(
                "untrusted payload\n", encoding="utf-8"
            )
            manifest = tree_manifest(package, root)
            self.assertEqual(
                [record["path"] for record in manifest["files"]],
                ["build-provenance.json"],
            )

    def test_tree_manifest_binds_root_directory_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir(mode=0o755)
            (package / "payload").write_bytes(b"payload")
            before = tree_manifest(package, root)
            self.assertEqual(before["root_mode_octal"], "0755")
            package.chmod(0o777)
            after = tree_manifest(package, root)
            self.assertEqual(after["root_mode_octal"], "0777")
            self.assertNotEqual(after, before)

    def test_default_source_snapshot_rejects_hash_time_symlink_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            source = repo / "tracked.txt"
            source.write_bytes(b"identical bytes\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            external = root / "external.txt"
            external.write_bytes(source.read_bytes())
            original_hash = gpmeep_provenance.sha256_file
            swapped = False

            def swap_during_descriptor_hash(path):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    source.unlink()
                    source.symlink_to(external)
                return original_hash(path)

            with mock.patch.object(
                gpmeep_provenance,
                "sha256_file",
                side_effect=swap_during_descriptor_hash,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while hashing"):
                    source_snapshot(repo)
            self.assertTrue(swapped)

    def test_default_tree_manifest_rejects_hash_time_symlink_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir()
            payload = package / "payload.bin"
            payload.write_bytes(b"identical bytes\n")
            external = root / "external.bin"
            external.write_bytes(payload.read_bytes())
            original_hash = gpmeep_provenance.sha256_file
            swapped = False

            def swap_during_descriptor_hash(path):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    payload.unlink()
                    payload.symlink_to(external)
                return original_hash(path)

            with mock.patch.object(
                gpmeep_provenance,
                "sha256_file",
                side_effect=swap_during_descriptor_hash,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while hashing"):
                    tree_manifest(package, root)
            self.assertTrue(swapped)

    def test_tree_manifest_rechecks_complete_topology_after_hashing(self):
        mutations = ("root-mode", "new-file", "symlink-retarget")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                package = root / "package"
                package.mkdir(mode=0o755)
                payload = package / "payload.bin"
                payload.write_bytes(b"payload\n")
                (package / "target-a").write_bytes(b"target\n")
                (package / "target-b").write_bytes(b"target\n")
                alias = package / "alias"
                alias.symlink_to("target-a")
                original_hash = gpmeep_provenance.sha256_file
                changed = False

                def mutate_during_hash(path):
                    nonlocal changed
                    if not changed:
                        changed = True
                        if mutation == "root-mode":
                            package.chmod(0o777)
                        elif mutation == "new-file":
                            (package / "unseen.py").write_text(
                                "VALUE = 1\n", encoding="utf-8"
                            )
                        else:
                            alias.unlink()
                            alias.symlink_to("target-b")
                    return original_hash(path)

                with mock.patch.object(
                    gpmeep_provenance,
                    "sha256_file",
                    side_effect=mutate_during_hash,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "topology changed while hashing"
                    ):
                        tree_manifest(package, root)
                self.assertTrue(changed)

    def test_source_snapshot_contains_a_hash_bound_per_file_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
            (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
            (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "tracked.txt", ".gitignore"], cwd=root, check=True
            )
            snapshot = source_snapshot(root)
            paths = [record["path"] for record in snapshot["files"]]
            self.assertEqual(paths, [".gitignore", "tracked.txt", "untracked.txt"])
            self.assertEqual(snapshot["file_count"], len(snapshot["files"]))
            self.assertEqual(snapshot["source_manifest_schema_version"], 2)
            self.assertEqual(
                snapshot["source_manifest_sha256"],
                canonical_sha256(snapshot["files"]),
            )
            self.assertNotIn("ignored.txt", paths)

    def test_source_snapshot_rejects_tracked_external_symlink_with_same_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            tracked = repo / "tracked.txt"
            tracked.write_bytes(b"identical")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            external = root / "external.txt"
            external.write_bytes(tracked.read_bytes())
            tracked.unlink()
            tracked.symlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "canonical regular file"):
                source_snapshot(repo)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = pathlib.Path(self.temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "receipt@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Receipt Test"],
            cwd=self.repo,
            check=True,
        )
        (self.repo / ".gitignore").write_text("build/\ninstall/\n")
        (self.repo / "source.cpp").write_text("int value = 1;\n")
        (self.repo / "environment.lock").write_text("locked\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True
        )
        self.build = self.repo / "build" / "fixture"
        self.package = self.build / "package"

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=self.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def begin(self) -> subprocess.CompletedProcess[str]:
        return self.invoke(
            "begin",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--build-kind", "synthetic",
            "--builder", str(self.repo / "source.cpp"),
            "--qualification-contract", "synthetic-v1",
            "--configure-arg=--enable-single",
            "--lockfile", f"environment={self.repo / 'environment.lock'}",
            "--record-env", "PATH",
        )

    def create_outputs(self) -> None:
        self.package.mkdir(parents=True, exist_ok=True)
        (self.build / "config.h").write_text("#define SINGLE 1\n")
        (self.build / "artifact.so").write_bytes(b"synthetic binary")
        (self.package / "__init__.py").write_text("VALUE = 1\n")

    def finalize(self) -> subprocess.CompletedProcess[str]:
        return self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--tool", "git",
        )

    def start_immutable_empty_directory_sentinel(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifacts,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        identity_path = self.build / "qualification-identity.json"
        ready_path = self.build / "sentinel.ready"
        gpmeep_provenance.atomic_write_json(identity_path, identity)
        process = subprocess.Popen(
            [
                sys.executable,
                str(QUALIFICATION_SCRIPT),
                "sentinel",
                "--identity", str(identity_path),
                "--repo", str(self.repo),
                "--prefix", str(prefix),
                "--ready", str(ready_path),
            ],
            cwd=self.repo,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(pycache),
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(
            lambda: process.kill() if process.poll() is None else None
        )
        for _ in range(500):
            if ready_path.is_file() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(
            ready_path.is_file(), f"sentinel returncode={process.poll()}"
        )
        return process, pycache, identity

    @staticmethod
    def finish_sentinel(process: subprocess.Popen[str]) -> str:
        for _ in range(200):
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            process.terminate()
        output, _ = process.communicate(timeout=10)
        return output

    def test_complete_receipt_binds_source_configuration_and_artifacts(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        completed = self.finalize()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt_path = self.build / "build-provenance.json"
        value = json.loads(receipt_path.read_text())
        self.assertEqual(value["state"], "complete")
        self.assertTrue(value["source_unchanged"])
        self.assertEqual(value["source_start"], value["source_end"])
        self.assertEqual(
            value["source_start"]["file_count"],
            len(value["source_start"]["files"]),
        )
        self.assertEqual(
            value["source_start"]["source_manifest_sha256"],
            canonical_sha256(value["source_start"]["files"]),
        )
        self.assertEqual(value["artifacts"]["extension"]["size_bytes"], 16)
        self.assertEqual(value["manifests"]["package"]["file_count"], 1)
        self.assertEqual(
            value["configuration"]["qualification_contract"], "synthetic-v1"
        )
        self.assertEqual(len(value["toolchain"]["git"]["sha256"]), 64)
        self.assertEqual(len(value["build_input_id"]), 64)
        self.assertEqual(len(value["artifact_set_id"]), 64)
        self.assertEqual(len(value["receipt_id"]), 64)
        self.assertFalse((self.build / "build-provenance.pending.json").exists())
        verified = verify_build_receipt(receipt_path, self.repo)
        self.assertEqual(verified["receipt_id"], value["receipt_id"])
        relative_repo = pathlib.Path(os.path.relpath(self.repo, pathlib.Path.cwd()))
        relative_verified = verify_build_receipt(receipt_path, relative_repo)
        self.assertEqual(relative_verified["receipt_id"], value["receipt_id"])
        previous_directory = pathlib.Path.cwd()
        try:
            os.chdir(self.repo)
            dot_verified = verify_build_receipt(receipt_path, pathlib.Path("."))
        finally:
            os.chdir(previous_directory)
        self.assertEqual(dot_verified["receipt_id"], value["receipt_id"])

    def test_verifier_replays_recorded_git_status(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        completed = self.finalize()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt_path = self.build / "build-provenance.json"
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        value["git_status_porcelain"] = [" M source.cpp"]
        unsigned = dict(value)
        unsigned.pop("receipt_id")
        value["receipt_id"] = canonical_sha256(unsigned)
        receipt_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ProvenanceError, "Git status differs"):
            verify_build_receipt(receipt_path, self.repo)

    def test_verifier_terminally_rechecks_source_topology(self):
        untracked = self.repo / "untracked.txt"
        untracked.write_bytes(b"identical bytes\n")
        external = self.repo.parent / f"{self.repo.name}-external.txt"
        self.addCleanup(external.unlink, missing_ok=True)
        external.write_bytes(untracked.read_bytes())
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        completed = self.finalize()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt_path = self.build / "build-provenance.json"
        swapped = False

        class SwapAfterInitialSourceValidation:
            @staticmethod
            def validate_v2_receipt(_value, _repo):
                nonlocal swapped
                untracked.unlink()
                untracked.symlink_to(external)
                swapped = True

        with mock.patch.object(
            gpmeep_provenance,
            "_load_qualification_contract_from_source",
            return_value=SwapAfterInitialSourceValidation,
        ):
            with self.assertRaisesRegex(
                ProvenanceError, "source path|source snapshot changed"
            ):
                verify_build_receipt(receipt_path, self.repo)
        self.assertTrue(swapped)

    def test_receipt_records_and_replays_manifest_exclusions(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        cache = self.package / "__pycache__"
        cache.mkdir()
        bytecode = cache / "module.pyc"
        bytecode.write_bytes(b"volatile-one")
        completed = self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--manifest-exclude-suffix", "package=.pyc",
            "--tool", "git",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt_path = self.build / "build-provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            receipt["manifests"]["package"]["excluded_suffixes"], [".pyc"]
        )
        bytecode.write_bytes(b"volatile-two")
        verify_build_receipt(receipt_path, self.repo)
        (self.package / "__init__.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ProvenanceError, "manifests.package changed"):
            verify_build_receipt(receipt_path, self.repo)

    def test_receipt_records_and_replays_immutable_directory_modes(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        prefix = self.build / "prefix"
        cache = prefix / "var" / "cache" / "fontconfig"
        cache.mkdir(parents=True)
        marker = cache / ".leave"
        marker.write_bytes(b"")
        marker.chmod(0o444)
        cache.chmod(0o555)
        completed = self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--manifest", f"installed_environment={prefix}",
            "--immutable-directory", f"prefix_fontconfig_cache={cache}",
            "--tool", "git",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt_path = self.build / "build-provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        immutable = receipt["immutable_directories"][
            "prefix_fontconfig_cache"
        ]
        self.assertEqual(immutable["mode_octal"], "0555")
        self.assertEqual(immutable["entry_count"], 1)
        self.assertEqual(immutable["entries"][0]["mode_octal"], "0444")
        self.assertEqual(
            receipt["artifact_set_id"],
            canonical_sha256(
                {
                    "configuration_files": receipt["configuration_files"],
                    "toolchain": receipt["toolchain"],
                    "artifacts": receipt["artifacts"],
                    "manifests": receipt["manifests"],
                    "immutable_directories": receipt[
                        "immutable_directories"
                    ],
                }
            ),
        )
        verify_build_receipt(receipt_path, self.repo)

        cache.chmod(0o755)
        with self.assertRaisesRegex(
            ProvenanceError, "immutable provenance directory has write bits"
        ):
            verify_build_receipt(receipt_path, self.repo)
        cache.chmod(0o555)
        marker.chmod(0o644)
        with self.assertRaisesRegex(
            ProvenanceError, "immutable provenance entry has write bits"
        ):
            verify_build_receipt(receipt_path, self.repo)

    def test_receipt_refuses_writable_immutable_directory(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        cache = self.build / "writable-cache"
        cache.mkdir()
        (cache / ".leave").write_bytes(b"")
        completed = self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--immutable-directory", f"cache={cache}",
            "--tool", "git",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("immutable provenance directory has write bits", completed.stderr)
        self.assertFalse((self.build / "build-provenance.json").exists())

    def test_immutable_directory_record_refuses_symlinked_entry(self):
        cache = self.build / "symlinked-cache"
        cache.mkdir(parents=True)
        marker = cache / ".leave"
        marker.write_bytes(b"")
        marker.chmod(0o444)
        (cache / "alias").symlink_to(marker)
        cache.chmod(0o555)
        with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
            gpmeep_provenance.immutable_directory_record(cache, self.repo)

    def test_builder_seals_prefix_fontconfig_cache_in_receipt(self):
        builder = BUILDER.read_text(encoding="utf-8")
        self.assertIn('PREFIX_FONTCONFIG_CACHE="${PREFIX}/var/cache/fontconfig"', builder)
        self.assertIn('/usr/bin/chmod 0444 "${PREFIX_FONTCONFIG_CACHE}/.leave"', builder)
        self.assertIn('/usr/bin/chmod 0555 "${PREFIX_FONTCONFIG_CACHE}"', builder)
        self.assertIn(
            '"prefix_fontconfig_cache=${PREFIX_FONTCONFIG_CACHE}"', builder
        )
        chmod_index = builder.index(
            '/usr/bin/chmod 0555 "${PREFIX_FONTCONFIG_CACHE}"'
        )
        snapshot_index = builder.index(
            '"${SCRIPT_DIR}/gpmeep_qualification_contract.py" snapshot'
        )
        audit_index = builder.index(
            '"${SCRIPT_DIR}/audit-conda-prefix.py"'
        )
        normalize_bytecode_index = builder.index(
            '"${SCRIPT_DIR}/normalize-conda-generated-bytecode.py"'
        )
        attestation_index = builder.index(
            '"${FRESH_ATTESTATION_SCRIPT}" create'
        )
        self.assertLess(normalize_bytecode_index, audit_index)
        self.assertLess(chmod_index, audit_index)
        self.assertLess(chmod_index, attestation_index)
        self.assertLess(chmod_index, snapshot_index)
        self.assertIn(
            '--configuration-file '
            '"conda_generated_bytecode_normalization=${PREFIX_BYTECODE_NORMALIZATION}"',
            builder,
        )

    def test_inotify_sentinel_rejects_transient_installed_bytecode(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        installed = self.build / "installed-prefix"
        installed.mkdir()
        (installed / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifacts,
            installed_prefix=installed,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        identity_path = self.build / "qualification-identity.json"
        ready_path = self.build / "sentinel.ready"
        gpmeep_provenance.atomic_write_json(identity_path, identity)
        process = subprocess.Popen(
            [
                sys.executable,
                str(QUALIFICATION_SCRIPT),
                "sentinel",
                "--identity", str(identity_path),
                "--repo", str(self.repo),
                "--prefix", str(prefix),
                "--ready", str(ready_path),
            ],
            cwd=self.repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(
            lambda: process.kill() if process.poll() is None else None
        )
        for _ in range(500):
            if ready_path.is_file() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(
            ready_path.is_file(), f"sentinel returncode={process.poll()}"
        )
        transient_directory = installed / "__pycache__"
        transient_directory.mkdir()
        transient = transient_directory / "transient.pyc"
        transient.write_bytes(b"transient bytecode")
        transient.unlink()
        transient_directory.rmdir()
        for _ in range(200):
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            process.terminate()
        output, _ = process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("installed-prefix mutation", output)

    def test_unknown_manifest_exclusion_is_rejected(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        completed = self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--manifest-exclude-suffix", "missing=.pyc",
            "--tool", "git",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unknown manifests: missing", completed.stderr)

    def test_v2_label_cannot_finalize_without_central_qualification_evidence(self):
        started = self.invoke(
            "begin",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--build-kind", "cuda-mpi-python-fp32",
            "--builder", str(self.repo / "source.cpp"),
            "--qualification-contract",
            gpmeep_qualification_contract.CONTRACT_NAME,
            "--lockfile", f"environment={self.repo / 'environment.lock'}",
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        self.create_outputs()
        completed = self.finalize()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("qualification-v2", completed.stderr)
        self.assertFalse((self.build / "build-provenance.json").exists())

    def test_immutable_empty_directory_snapshot_rejects_preexisting_pyc(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        attacker_source = self.build / "attacker.py"
        attacker_source.write_text("ATTACKER = True\n", encoding="utf-8")
        py_compile.compile(
            str(attacker_source),
            cfile=str(pycache / "injected.cpython-311.pyc"),
            doraise=True,
        )
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        with self.assertRaisesRegex(RuntimeError, "is not empty"):
            gpmeep_qualification_contract.create_identity_snapshot(
                self.repo,
                prefix,
                artifacts,
                immutable_empty_directories={
                    "qualification_pycache": pycache
                },
            )

    def test_immutable_empty_directory_sentinel_rejects_transient_pyc(self):
        process, pycache, _identity = (
            self.start_immutable_empty_directory_sentinel()
        )
        injected = pycache / "injected.cpython-311.pyc"
        attacker_source = self.build / "transient-attacker.py"
        attacker_source.write_text("ATTACKER = True\n", encoding="utf-8")
        py_compile.compile(
            str(attacker_source), cfile=str(injected), doraise=True
        )
        injected.unlink()
        output = self.finish_sentinel(process)
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("identity mutation", output)
        marker = gpmeep_qualification_contract.marker_for(
            gpmeep_qualification_contract.SENTINEL_LOG_NAME
        ).decode("ascii")
        self.assertNotIn(marker, output)

    def test_immutable_empty_directory_sentinel_rejects_replacement(self):
        process, pycache, _identity = (
            self.start_immutable_empty_directory_sentinel()
        )
        displaced = pycache.with_name("qualification-pycache-displaced")
        pycache.rename(displaced)
        pycache.mkdir(mode=0o700)
        output = self.finish_sentinel(process)
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("identity mutation", output)
        marker = gpmeep_qualification_contract.marker_for(
            gpmeep_qualification_contract.SENTINEL_LOG_NAME
        ).decode("ascii")
        self.assertNotIn(marker, output)

    def test_immutable_empty_directory_clean_epoch_is_identity_bound(self):
        process, pycache, identity = (
            self.start_immutable_empty_directory_sentinel()
        )
        unrelated = pycache.parent / "unrelated-sibling.tmp"
        unrelated.write_bytes(b"not an identity path")
        unrelated.unlink()
        process.terminate()
        output, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, output)
        self.assertEqual(list(pycache.iterdir()), [])
        binding = identity[
            gpmeep_qualification_contract.IMMUTABLE_EMPTY_DIRECTORIES_KEY
        ]["qualification_pycache"]
        self.assertEqual(binding["kind"], "immutable-empty-directory")
        self.assertEqual(binding["mode"], "0700")
        lines = output.splitlines()
        marker = gpmeep_qualification_contract.marker_for(
            gpmeep_qualification_contract.SENTINEL_LOG_NAME
        ).decode("ascii")
        self.assertEqual(lines[-1], marker)
        self.assertEqual(
            json.loads(lines[-2])["identity_sha256"],
            canonical_sha256(identity),
        )

    def test_builder_binds_fresh_empty_qualification_pycache_to_epoch(self):
        builder = BUILDER.read_text(encoding="utf-8")
        binding = (
            '--immutable-empty-directory '
            '"qualification_pycache=${QUALIFICATION_PYCACHE}"'
        )
        self.assertEqual(builder.count(binding), 1)
        self.assertIn(
            '/usr/bin/install -d -m 700 "${QUALIFICATION_PYCACHE}"',
            builder,
        )
        self.assertIn(
            '"${#QUALIFICATION_PYCACHE_ENTRIES[@]}" -ne 0',
            builder,
        )
        self.assertEqual(builder.count('"${QUALIFICATION_IDENTITY_ARGS[@]}"'), 2)
        self.assertLess(
            builder.index(binding),
            builder.index("QUALIFICATION_IDENTITY_BEFORE}"),
        )
        self.assertLess(
            builder.index('wait "${QUALIFICATION_SENTINEL_PID}"'),
            builder.index("QUALIFICATION_PYCACHE_ENTRIES="),
        )
        target_export = (
            'export PYTHONPYCACHEPREFIX="${QUALIFICATION_PYCACHE}"'
        )
        self.assertEqual(builder.count(target_export), 1)
        target_export_index = builder.index(target_export)
        self.assertGreater(
            target_export_index,
            builder.index(
                'qualification mutation sentinel startup timed out'
            ),
        )
        before_snapshot = builder.index(
            '--output "${QUALIFICATION_IDENTITY_BEFORE}"'
        )
        self.assertIn(
            '"${PREFIX_CONTROL_PYTHON[@]}"',
            builder[before_snapshot - 350 : before_snapshot],
        )
        sentinel_launch = builder.index(
            '"${SCRIPT_DIR}/gpmeep_qualification_contract.py" sentinel'
        )
        self.assertIn(
            '"${PREFIX_CONTROL_PYTHON[@]}"',
            builder[sentinel_launch - 200 : sentinel_launch],
        )
        after_snapshot = builder.index(
            '--output "${QUALIFICATION_IDENTITY_AFTER}"'
        )
        self.assertIn(
            '"${PREFIX_CONTROL_PYTHON[@]}"',
            builder[after_snapshot - 350 : after_snapshot],
        )
        sentinel_wait = builder.index(
            'wait "${QUALIFICATION_SENTINEL_PID}"', after_snapshot
        )
        post_epoch_null = builder.index(
            "export PYTHONPYCACHEPREFIX=/dev/null", sentinel_wait
        )
        seal_invocation = builder.index(
            '"${SCRIPT_DIR}/gpmeep_qualification_contract.py" seal'
        )
        finalize_invocation = builder.index(
            '"${SCRIPT_DIR}/write-build-receipt.py" finalize'
        )
        self.assertLess(sentinel_wait, post_epoch_null)
        self.assertLess(post_epoch_null, seal_invocation)
        self.assertIn(
            '"${PREFIX_CONTROL_PYTHON[@]}"',
            builder[seal_invocation - 100 : seal_invocation],
        )
        self.assertIn(
            '"${PREFIX_CONTROL_PYTHON[@]}"',
            builder[finalize_invocation - 100 : finalize_invocation],
        )
        self.assertNotRegex(builder, r"(?:^|[ ])-(?:I|E)(?:[ ]|$)")
        self.assertIn(
            '/usr/bin/python3 -S -P -B "${CONTROL_PYTHON_RUNNER}"',
            builder,
        )
        self.assertIn(
            '"${PREFIX}/bin/python" -S -P -B '
            '"${CONTROL_PYTHON_RUNNER}"',
            builder,
        )
        self.assertIn("PYTHONPYCACHEPREFIX=/dev/null", builder)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", builder)
        self.assertIn("-u PYTHONPATH", builder)
        self.assertIn("-u PYTHONHOME", builder)

    def test_builder_binds_every_qualification_artifact_to_final_receipt(self):
        builder = BUILDER.read_text(encoding="utf-8")
        identity_start = builder.index("QUALIFICATION_IDENTITY_ARGS=(")
        identity_end = builder.index("\n)", identity_start)
        identity_block = builder[identity_start:identity_end]
        finalize_start = builder.index(
            '"${SCRIPT_DIR}/write-build-receipt.py" finalize'
        )
        finalize_block = builder[finalize_start:]

        shell_variables = {
            "cuda_architecture_test": "CUDA_ARCHITECTURE_TEST_ELF",
            "cuda_formula_test": "CUDA_FORMULA_TEST_ELF",
            "cuda_near2far_runtime_validation": (
                "CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF"
            ),
            "cuda_runtime_validation": "CUDA_RUNTIME_VALIDATION_ELF",
            "cuda_smoke": "CUDA_SMOKE_ELF",
            "fd_allocation_shim": "FD_ALLOCATION_SHIM",
            "gpu_backend_test": "GPU_BACKEND_ELF",
            "gpu_mpi_performance": "GPU_MPI_PERFORMANCE_ELF",
            "gpu_step_db_test": "GPU_STEP_DB_ELF",
            "installed_libmeep": "INSTALLED_LIBMEEP",
            "installed_libpympb": "INSTALLED_LIBPYMPB",
            "installed_mpb_extension": "INSTALLED_MPB_EXTENSION",
            "installed_python_extension": "INSTALLED_EXTENSION",
            "libmeep": "BUILD_LIBMEEP",
            "python_extension": "BUILD_EXTENSION",
        }
        self.assertEqual(
            tuple(sorted(shell_variables)),
            gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS,
        )
        for name, variable in shell_variables.items():
            binding = f'--artifact "{name}=${{{variable}}}"'
            self.assertEqual(identity_block.count(binding), 1, binding)
            self.assertEqual(finalize_block.count(binding), 1, binding)

        for tool in ("cc", "cmake", "ninja"):
            self.assertIn(f"--tool {tool}", finalize_block)

    def test_builder_runs_release_near2far_qualification_after_receipt(self):
        builder = BUILDER.read_text(encoding="utf-8")
        finalize = builder.index(
            '"${SCRIPT_DIR}/write-build-receipt.py" finalize'
        )
        qualification = builder.index(
            '"${SCRIPT_DIR}/run-near2far-mpi-qualification.py"'
        )
        self.assertLess(finalize, qualification)
        block = builder[qualification - 700 : qualification + 700]
        for fragment in (
            '--qualification-tier release',
            '--build-receipt "${BUILD_DIR}/build-provenance.json"',
            '--executable "${GPU_STEP_DB_ELF}"',
            '--output "${NEAR2FAR_RELEASE_QUALIFICATION}"',
            '--lane-timeout-seconds 300',
            '--stdout-limit-bytes 8388608',
            '--stderr-limit-bytes 8388608',
            '"${TIMEOUT}" --kill-after=30s 1800s',
        ):
            self.assertIn(fragment, block)
        receipt_block = builder[finalize:qualification]
        self.assertNotIn(
            '--manifest "near2far_release_qualification=', receipt_block
        )

    def test_builder_seals_installed_cuda_python_and_adjoint_qualification(self):
        builder = BUILDER.read_text(encoding="utf-8")
        logs = {
            "installed-python-gpu-backend-singleton.log": (
                "GPMEEP_REQUIRE_CUDA_TEST=1",
                'CUDA_VISIBLE_DEVICES=0',
                "MEEP_GPU_BACKEND=cuda",
                "MEEP_GPU_STRICT=1",
                '"${PREFIX}/bin/python"',
                '"${SCRIPT_DIR}/run-installed-cuda-qualification.py"',
            ),
            "installed-python-gpu-backend-two-rank.log": (
                "GPMEEP_REQUIRE_CUDA_TEST=1",
                'CUDA_VISIBLE_DEVICES=0,1',
                "MEEP_GPU_BACKEND=cuda",
                "MEEP_GPU_STRICT=1",
                '"${PREFIX}/bin/mpiexec" -n 2',
                '"${SCRIPT_DIR}/run-installed-cuda-qualification.py"',
            ),
            "installed-adjoint-default-material-grid-cuda.log": (
                "MEEP_GPU_BACKEND=cuda",
                "MEEP_GPU_STRICT=1",
                "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1",
                "GPMEEP_REQUIRE_CUDA_TEST=1",
                '"${SCRIPT_DIR}/run-installed-cuda-qualification.py"',
            ),
        }
        self.assertTrue(
            set(logs).issubset(
                gpmeep_qualification_contract.REQUIRED_QUALIFICATION_LOGS
            )
        )
        self.assertTrue(
            set(logs).issubset(
                gpmeep_qualification_contract.PYTHON_RUNTIME_LOG_BINDINGS
            )
        )
        for log_name, required_fragments in logs.items():
            with self.subTest(log_name=log_name):
                variable = {
                    "installed-python-gpu-backend-singleton.log": (
                        "INSTALLED_GPU_SINGLETON_LOG"
                    ),
                    "installed-python-gpu-backend-two-rank.log": (
                        "INSTALLED_GPU_TWO_RANK_LOG"
                    ),
                    "installed-adjoint-default-material-grid-cuda.log": (
                        "INSTALLED_ADJOINT_CUDA_LOG"
                    ),
                }[log_name]
                redirect = builder.index(f'>"${{{variable}}}" 2>&1')
                command_start = builder.rfind("\nenv ", 0, redirect)
                command = builder[command_start:redirect]
                self.assertEqual(
                    command.count('PYTHONPATH="${INSTALLED_PYTHON}"'), 1
                )
                self.assertEqual(command.count("PYTHONPATH="), 1)
                self.assertEqual(command.count("MEEP_GPU_BACKEND=cuda"), 1)
                self.assertEqual(command.count("MEEP_GPU_BACKEND="), 1)
                self.assertEqual(command.count("MEEP_GPU_STRICT=1"), 1)
                self.assertEqual(command.count("MEEP_GPU_STRICT="), 1)
                self.assertEqual(command.count("-u MEEP_GPU_DEVICE"), 1)
                self.assertNotRegex(command, r"\bMEEP_GPU_DEVICE=")
                self.assertEqual(
                    command.count(
                        '"${SCRIPT_DIR}/run-installed-cuda-qualification.py"'
                    ),
                    1,
                )
                self.assertEqual(command.count(f"--log-name {log_name}"), 1)
                self.assertEqual(
                    command.count(
                        '--expected-extension "${INSTALLED_EXTENSION}"'
                    ),
                    1,
                )
                self.assertEqual(
                    command.count('--expected-libmeep "${INSTALLED_LIBMEEP}"'),
                    1,
                )
                self.assertNotIn("attest-python-runtime", command)
                for fragment in required_fragments:
                    self.assertIn(fragment, command)
                next_block = builder.find("\n\n", redirect)
                self.assertNotEqual(next_block, -1)
                self.assertNotIn(
                    '>>"${' + variable + '}"', builder[redirect:next_block]
                )

        runner = INSTALLED_CUDA_RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("from gpmeep_qualification_contract import", runner)
        self.assertIn(
            '"gpmeep_qualification_contract_for_installed_cuda", path',
            runner,
        )
        self.assertIn("unittest.TextTestRunner", runner)
        self.assertIn("python_runtime_attestation(", runner)
        self.assertIn("_cuda_execution_probe()", runner)
        self.assertIn("communicator.gather(rank_record, root=0)", runner)
        cleanup = runner.index("mp._gpu_finalize_distributed_runtime()")
        finalize = runner.index("MPI.Finalize()")
        terminal_print = runner.index(
            "print(marker_for(args.log_name).decode(\"ascii\"))"
        )
        self.assertLess(cleanup, finalize)
        self.assertLess(finalize, terminal_print)
        self.assertIn("MPI.COMM_WORLD.Abort(1)", runner)
        self.assertIn("os._exit(0 if complete else 1)", runner)

    def test_builder_scopes_install_pycache_away_from_qualification(self):
        builder = BUILDER.read_text(encoding="utf-8")
        install_cache_definition = (
            'INSTALL_PYCACHE="${BUILD_DIR}/install-pycache"'
        )
        scoped_install = (
            '/usr/bin/env PYTHONPYCACHEPREFIX="${INSTALL_PYCACHE}" \\\n'
            '  "${TIMEOUT}" --kill-after=60s 1800s make install'
        )
        self.assertEqual(builder.count(install_cache_definition), 1)
        self.assertEqual(builder.count(scoped_install), 1)
        self.assertNotIn(
            'export PYTHONPYCACHEPREFIX="${INSTALL_PYCACHE}"', builder
        )
        self.assertIn(
            '[[ -e "${INSTALL_PYCACHE}" || -L "${INSTALL_PYCACHE}" ]]',
            builder,
        )
        self.assertIn(
            '/usr/bin/install -d -m 700 "${INSTALL_PYCACHE}"', builder
        )
        self.assertIn(
            'INSTALL_PYCACHE_IDENTITY="$(/usr/bin/stat -c \'%d:%i\' '
            '"${INSTALL_PYCACHE}")"',
            builder,
        )
        self.assertEqual(builder.count("validate_install_pycache"), 3)
        self.assertEqual(builder.count("assert_no_installed_bytecode"), 3)
        self.assertIn(
            '-type f ! -path "${INSTALL_PYCACHE_PAYLOAD_ROOT}/*.pyc"',
            builder,
        )
        self.assertIn(
            "\\( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \\)",
            builder,
        )
        self.assertIn(
            '--configuration-file '
            '"install_pycache_policy=${INSTALL_PYCACHE_POLICY}"',
            builder,
        )
        self.assertIn(
            '--manifest "install_pycache=${INSTALL_PYCACHE}"', builder
        )

        make_all = builder.index(
            '"${TIMEOUT}" --kill-after=60s 3600s make -j"${MAKE_JOBS}"'
        )
        install_cache = builder.index(install_cache_definition)
        install = builder.index(scoped_install)
        sentinel_ready = builder.index(
            'QUALIFICATION_SENTINEL_READY="${BUILD_DIR}/'
            'qualification-mutation-sentinel.ready"'
        )
        ready_confirmed = builder.index(
            'if [[ ! -f "${QUALIFICATION_SENTINEL_READY}" ]]'
        )
        qualification_cache = builder.index(
            'export PYTHONPYCACHEPREFIX="${QUALIFICATION_PYCACHE}"'
        )
        identity_after = builder.index(
            '--output "${QUALIFICATION_IDENTITY_AFTER}"'
        )
        sentinel_stop = builder.index(
            'wait "${QUALIFICATION_SENTINEL_PID}"', identity_after
        )
        restored_control = builder.index(
            'export PYTHONPYCACHEPREFIX=/dev/null', sentinel_stop
        )
        self.assertLess(make_all, install_cache)
        self.assertLess(install_cache, install)
        self.assertLess(install, sentinel_ready)
        self.assertLess(sentinel_ready, ready_confirmed)
        self.assertLess(ready_confirmed, qualification_cache)
        self.assertLess(qualification_cache, identity_after)
        self.assertLess(identity_after, sentinel_stop)
        self.assertLess(sentinel_stop, restored_control)
        self.assertEqual(
            builder.count(
                'export PYTHONPYCACHEPREFIX="${QUALIFICATION_PYCACHE}"'
            ),
            1,
        )

    def test_builder_normalizes_and_replays_conda_source_bytecode(self):
        builder = BUILDER.read_text(encoding="utf-8")
        pre_audit = builder.index("PREFIX_PRE_NORMALIZATION_AUDIT_TEMP=")
        normalize = builder.index(
            '"${SCRIPT_DIR}/normalize-conda-relocated-bytecode.py"'
        )
        final_audit = builder.index("PREFIX_CONTENT_AUDIT_TEMP=")
        probe_loop = builder.index("for probe_number in one two; do")
        attestation = builder.index("mapfile -t FRESH_ATTESTATION_OUTPUT")
        self.assertLess(pre_audit, normalize)
        self.assertLess(normalize, final_audit)
        self.assertLess(final_audit, probe_loop)
        self.assertLess(probe_loop, attestation)
        self.assertIn(
            '"${PREFIX}/bin/python"',
            builder,
        )
        self.assertIn(
            '/usr/bin/cmp --silent "${PREFIX_CONTENT_AUDIT_TEMP}" '
            '"${!audit_variable}"',
            builder,
        )
        self.assertIn(
            '"${PREFIX_IMPORT_PROBE_ONE_TEMP}" '
            '"${PREFIX_IMPORT_PROBE_TWO_TEMP}"',
            builder,
        )
        for name in (
            "conda_prefix_pre_normalization_audit",
            "conda_prefix_post_import_audit_one",
            "conda_prefix_post_import_audit_two",
            "conda_source_bytecode_normalization",
            "conda_source_bytecode_import_probe_one",
            "conda_source_bytecode_import_probe_two",
        ):
            self.assertIn(f'--configuration-file "{name}=', builder)

    def test_explicit_pycompile_uses_isolated_cache_not_installed_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            install_root = root / "install"
            install_root.mkdir()
            source = install_root / "gpmeep_install_cache_probe.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            source_stat = source.stat()
            install_cache = root / "build" / "install-pycache"
            install_cache.mkdir(parents=True, mode=0o700)
            compile_program = (
                "import importlib.util,py_compile,sys; "
                "source=sys.argv[1]; "
                "py_compile.compile(source, "
                "importlib.util.cache_from_source(source), source, "
                "doraise=True)"
            )
            base_environment = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            broken = subprocess.run(
                [sys.executable, "-c", compile_program, str(source)],
                env={
                    **base_environment,
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn("/dev/null", broken.stdout + broken.stderr)

            compiled = subprocess.run(
                [sys.executable, "-c", compile_program, str(source)],
                env={
                    **base_environment,
                    "PYTHONPYCACHEPREFIX": str(install_cache),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            cache_files = sorted(install_cache.rglob("*.pyc"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(list(install_root.rglob("__pycache__")), [])
            self.assertEqual(list(install_root.rglob("*.pyc")), [])
            cache_snapshot = {
                path.relative_to(install_cache): path.read_bytes()
                for path in cache_files
            }

            # Keep timestamp and size equal so this bytecode is demonstrably
            # loadable when its private prefix is selected, then prove that
            # the later /dev/null control path reads the installed source.
            source.write_text("VALUE = 2\n", encoding="utf-8")
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            import_program = (
                "import sys,gpmeep_install_cache_probe as probe; "
                "print(sys.pycache_prefix); print(probe.VALUE)"
            )
            private_cache_import = subprocess.run(
                [sys.executable, "-c", import_program],
                env={
                    **base_environment,
                    "PYTHONPATH": str(install_root),
                    "PYTHONPYCACHEPREFIX": str(install_cache),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                private_cache_import.returncode, 0, private_cache_import.stderr
            )
            self.assertEqual(
                private_cache_import.stdout.splitlines(),
                [str(install_cache), "1"],
            )

            controlled_import = subprocess.run(
                [sys.executable, "-c", import_program],
                env={
                    **base_environment,
                    "PYTHONPATH": str(install_root),
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(controlled_import.returncode, 0, controlled_import.stderr)
            self.assertEqual(controlled_import.stdout.splitlines(), ["/dev/null", "2"])
            self.assertEqual(
                {
                    path.relative_to(install_cache): path.read_bytes()
                    for path in cache_files
                },
                cache_snapshot,
            )

    def test_isolated_control_python_cannot_load_injected_target_prefix_pyc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "injected_dependency.py"
            target_pycache = root / "qualification-pycache"
            target_pycache.mkdir(mode=0o700)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            original = source.stat()
            compile_code = (
                "import importlib.util,py_compile; "
                f"source={str(source)!r}; "
                "py_compile.compile(source, "
                "cfile=importlib.util.cache_from_source(source), doraise=True)"
            )
            environment = {
                **os.environ,
                "PYTHONPATH": str(root),
                "PYTHONPYCACHEPREFIX": str(target_pycache),
            }
            compiled = subprocess.run(
                [sys.executable, "-c", compile_code],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            self.assertTrue(any(target_pycache.rglob("*.pyc")))
            source.write_text("VALUE = 2\n", encoding="utf-8")
            os.utime(
                source,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
            import_code = (
                "import injected_dependency; "
                "print(injected_dependency.VALUE)"
            )
            hostile = subprocess.run(
                [sys.executable, "-c", import_code],
                env={
                    **environment,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(hostile.returncode, 0, hostile.stderr)
            self.assertEqual(hostile.stdout.strip(), "1")
            isolated_code = (
                "import importlib.util; "
                f"path={str(source)!r}; "
                "spec=importlib.util.spec_from_file_location('isolated', path); "
                "module=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); print(module.VALUE)"
            )
            protected_environment = dict(environment)
            for name in (
                "PYTHONPATH",
                "PYTHONHOME",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
                "PYTHONINSPECT",
                "PYTHONWARNINGS",
                "PYTHONBREAKPOINT",
                "PYTHONPROFILEIMPORTTIME",
            ):
                protected_environment.pop(name, None)
            protected_environment.update(
                {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                    "PYTHONSAFEPATH": "1",
                }
            )
            protected = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-P",
                    "-B",
                    "-c",
                    (
                        "import sys; assert sys.pycache_prefix == '/dev/null'; "
                        "assert sys.dont_write_bytecode and sys.flags.no_site; "
                        "assert sys.flags.safe_path and not "
                        "sys.flags.ignore_environment; "
                        + isolated_code
                    ),
                ],
                env=protected_environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(protected.returncode, 0, protected.stderr)
            self.assertEqual(protected.stdout.strip(), "2")

    def test_control_python_enforces_flags_environment_and_allowlist(self):
        specification = importlib.util.spec_from_file_location(
            "gpmeep_control_python_allowlist_test", CONTROL_PYTHON_RUNNER
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(
            specification.loader if specification is not None else None
        )
        module = importlib.util.module_from_spec(specification)
        assert specification is not None and specification.loader is not None
        specification.loader.exec_module(module)
        self.assertIn(
            "normalize-conda-relocated-bytecode.py",
            module.ALLOWED_CONTROL_SCRIPTS,
        )
        environment = dict(os.environ)
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "PYTHONINSPECT",
            "PYTHONWARNINGS",
            "PYTHONBREAKPOINT",
            "PYTHONPROFILEIMPORTTIME",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": "/dev/null",
                "PYTHONSAFEPATH": "1",
            }
        )
        command = [
            sys.executable,
            "-S",
            "-P",
            "-B",
            str(CONTROL_PYTHON_RUNNER),
            str(QUALIFICATION_SCRIPT),
            "check-expected-failure-status",
            "1",
        ]
        valid = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        ignored_environment = subprocess.run(
            [sys.executable, "-I", *command[4:]],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(ignored_environment.returncode, 0)
        self.assertIn("requires non--E/-I Python", ignored_environment.stderr)
        unallowlisted = subprocess.run(
            [*command[:4], str(CONTROL_PYTHON_RUNNER), str(BUILDER)],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(unallowlisted.returncode, 0)
        self.assertIn("not allowlisted", unallowlisted.stderr)

    def test_control_python_rejects_transient_source_write_and_restore(self):
        spec = importlib.util.spec_from_file_location(
            "gpmeep_control_python_for_test", CONTROL_PYTHON_RUNNER
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec is not None else None)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "control.py"
            original = b"VALUE = 1\n"
            path.write_bytes(original)
            descriptor = os.open(path, os.O_RDONLY)
            self.addCleanup(lambda: os.close(descriptor))
            initial = os.fstat(descriptor)
            payload = os.read(descriptor, initial.st_size)
            path.write_bytes(b"VALUE = 2\n")
            path.write_bytes(original)
            os.utime(
                path,
                ns=(initial.st_atime_ns, initial.st_mtime_ns),
            )
            with (
                mock.patch.object(module.sys, "stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                module.assert_stable_open_source(
                    descriptor, path, initial, payload, "read"
                )
            scripts = root / "scripts"
            scripts.mkdir()
            runner = scripts / CONTROL_PYTHON_RUNNER.name
            shutil.copy2(CONTROL_PYTHON_RUNNER, runner)
            target = scripts / "capture-build-environment.py"
            executed_source = (
                b"import pathlib,sys,time\n"
                b"pathlib.Path(sys.argv[1]).write_text('READY')\n"
                b"time.sleep(0.3)\n"
            )
            target.write_bytes(executed_source)
            marker = root / "target-ready"
            environment = dict(os.environ)
            for name in (
                "PYTHONPATH",
                "PYTHONHOME",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
                "PYTHONINSPECT",
                "PYTHONWARNINGS",
                "PYTHONBREAKPOINT",
                "PYTHONPROFILEIMPORTTIME",
            ):
                environment.pop(name, None)
            environment.update(
                {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                    "PYTHONSAFEPATH": "1",
                }
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-S",
                    "-P",
                    "-B",
                    str(runner),
                    str(target),
                    str(marker),
                ],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.addCleanup(
                lambda: process.kill() if process.poll() is None else None
            )
            for _ in range(200):
                if marker.is_file() or process.poll() is not None:
                    break
                time.sleep(0.005)
            self.assertTrue(marker.is_file(), f"returncode={process.poll()}")
            target_info = target.stat()
            target.write_bytes(executed_source.replace(b"0.3", b"0.4"))
            target.write_bytes(executed_source)
            os.utime(
                target,
                ns=(target_info.st_atime_ns, target_info.st_mtime_ns),
            )
            stdout, stderr = process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0, stdout + stderr)
            self.assertIn("identity changed during execution", stderr)
            second_marker = root / "second-target-ready"
            second = subprocess.Popen(
                [
                    sys.executable,
                    "-S",
                    "-P",
                    "-B",
                    str(runner),
                    str(target),
                    str(second_marker),
                ],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.addCleanup(
                lambda: second.kill() if second.poll() is None else None
            )
            for _ in range(200):
                if second_marker.is_file() or second.poll() is not None:
                    break
                time.sleep(0.005)
            self.assertTrue(
                second_marker.is_file(), f"returncode={second.poll()}"
            )
            runner_bytes = runner.read_bytes()
            runner_info = runner.stat()
            runner.write_bytes(runner_bytes.replace(b"16 * 1024", b"15 * 1024", 1))
            runner.write_bytes(runner_bytes)
            os.utime(
                runner,
                ns=(runner_info.st_atime_ns, runner_info.st_mtime_ns),
            )
            second_stdout, second_stderr = second.communicate(timeout=10)
            self.assertNotEqual(
                second.returncode, 0, second_stdout + second_stderr
            )
            self.assertIn(
                "identity changed during target execution", second_stderr
            )

    def test_authoritative_identity_requires_exact_pycache_name_mode_and_path(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        artifact = self.build / "artifact.so"
        artifact_paths = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        installed_lib = prefix / "lib"
        installed_extension_dir = prefix / "python" / "meep" / "mpb"
        installed_lib.mkdir()
        installed_extension_dir.mkdir(parents=True)
        installed_files = {
            "installed_libmeep": installed_lib / "libmeep.so.38.0.0",
            "installed_libpympb": installed_lib / "libpympb.so.38.0.0",
            "installed_python_extension": (
                installed_extension_dir.parent / "_meep.so.38.0.0"
            ),
            "installed_mpb_extension": installed_extension_dir / "_mpb.so.38.0.0",
        }
        for name, path in installed_files.items():
            path.write_bytes(name.encode("ascii"))
            artifact_paths[name] = path
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        with self.assertRaisesRegex(RuntimeError, "wrong immutable.*binding set"):
            gpmeep_qualification_contract.create_identity_snapshot(
                self.repo,
                prefix,
                artifact_paths,
                require_contract_bindings=True,
            )
        extra = self.build / "extra-empty"
        extra.mkdir(mode=0o700)
        with self.assertRaisesRegex(RuntimeError, "wrong immutable.*binding set"):
            gpmeep_qualification_contract.create_identity_snapshot(
                self.repo,
                prefix,
                artifact_paths,
                immutable_empty_directories={
                    "qualification_pycache": pycache,
                    "uncontracted_empty": extra,
                },
                require_contract_bindings=True,
            )
        pycache.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "wrong required mode"):
            gpmeep_qualification_contract.create_identity_snapshot(
                self.repo,
                prefix,
                artifact_paths,
                immutable_empty_directories={
                    "qualification_pycache": pycache
                },
                require_contract_bindings=True,
            )
        pycache.chmod(0o700)
        wrong_path = self.build / "wrong-empty-path"
        wrong_path.mkdir(mode=0o700)
        wrong_path_identity = (
            gpmeep_qualification_contract.create_identity_snapshot(
                self.repo,
                prefix,
                artifact_paths,
                immutable_empty_directories={
                    "qualification_pycache": wrong_path
                },
                require_contract_bindings=True,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "wrong required path"):
            gpmeep_qualification_contract._identity_immutable_empty_directory_paths(
                wrong_path_identity,
                self.repo,
                require_contract_bindings=True,
                expected_directory_parent=self.build,
            )

    def test_seal_and_receipt_bind_immutable_empty_directory_semantics(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        artifact = self.build / "artifact.so"
        artifact_paths = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        installed_lib = prefix / "lib"
        installed_extension_dir = prefix / "python" / "meep" / "mpb"
        installed_lib.mkdir()
        installed_extension_dir.mkdir(parents=True)
        installed_files = {
            "installed_libmeep": installed_lib / "libmeep.so.38.0.0",
            "installed_libpympb": installed_lib / "libpympb.so.38.0.0",
            "installed_python_extension": (
                installed_extension_dir.parent / "_meep.so.38.0.0"
            ),
            "installed_mpb_extension": installed_extension_dir / "_mpb.so.38.0.0",
        }
        for name, path in installed_files.items():
            path.write_bytes(name.encode("ascii"))
            artifact_paths[name] = path
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifact_paths,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        before = self.build / "qualification-identity-before.json"
        after = self.build / "qualification-identity-after.json"
        seal_path = self.build / "qualification-contract-v2.json"
        log_dir = self.build / "qualification-logs"
        log_dir.mkdir()
        gpmeep_provenance.atomic_write_json(before, identity)
        gpmeep_provenance.atomic_write_json(after, identity)
        with (
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_logs",
                return_value={},
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_no_uncontracted_pass_logs",
                return_value={
                    name: {
                        "path": name,
                        "size_bytes": 1,
                        "sha256": "2" * 64,
                        "terminal_marker": "materialized",
                    }
                    for name in gpmeep_qualification_contract.MATERIALIZATION_LOGS
                },
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_mutation_sentinel_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_python_runtime_logs",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_lazy_api_qualification_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_mpb_qualification_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_cuda_qualification_logs",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_direct_elf_logs",
            ),
        ):
            gpmeep_qualification_contract.seal_contract(
                self.repo, log_dir, before, after, seal_path
            )
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        key = gpmeep_qualification_contract.IMMUTABLE_EMPTY_DIRECTORIES_KEY
        self.assertEqual(seal[key], identity[key])

        fake_logs = {
            name: {
                "path": name,
                "size_bytes": 1,
                "sha256": "1" * 64,
                "terminal_marker": gpmeep_qualification_contract.marker_for(
                    name
                ).decode("ascii"),
            }
            for name in gpmeep_qualification_contract.ALL_REQUIRED_QUALIFICATION_LOGS
        }
        seal["logs"] = fake_logs
        fake_materialization_logs = seal["materialization_logs"]
        gpmeep_provenance.atomic_write_json(seal_path, seal)
        release_build = self.repo / "build" / "meep-cuda-mpi-python-fp32"
        release_build.mkdir()
        compiler_dir = self.repo / ".envs" / "meep-gpu-cuda-mpi" / "bin"
        compiler_dir.mkdir(parents=True)
        nvcc_path = compiler_dir / "nvcc"
        host_path = compiler_dir / "x86_64-conda-linux-gnu-c++"
        nvcc_path.write_text("synthetic nvcc\n", encoding="utf-8")
        host_path.write_text("synthetic host compiler\n", encoding="utf-8")
        canonical_build_environment = (
            release_build / "canonical-build-environment.json"
        )
        canonical_build_environment.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "environment": {
                        "MEEP_GPU_FAST_MATH": "OFF",
                        "NVCC_PREPEND_FLAGS": f"-ccbin={host_path}",
                        "CXX": str(host_path),
                    },
                    "normalized_shell_keys": ["SHLVL", "_"],
                }
            ),
            encoding="utf-8",
        )
        config_status = release_build / "config.status"
        config_status.write_text(
            "  set X /bin/bash '../../configure' "
            "'--disable-cuda-fast-math' 'CC=/synthetic'\n"
            'S["NVCCFLAGS"]="-O3"\n',
            encoding="utf-8",
        )
        cuda_runtime_flags_stamp = (
            release_build / "src" / "cuda-runtime-flags.stamp"
        )
        cuda_runtime_flags_stamp.parent.mkdir(exist_ok=True)
        cuda_runtime_flags_stamp.write_text(
            f"NVCC={nvcc_path}\n"
            f"CUDAHOSTCXX={host_path}\n"
            "NVCCFLAGS=-O3\n"
            "CUDA_ARCH_FLAGS=-gencode=arch=compute_80,code=sm_80\n",
            encoding="utf-8",
        )
        cuda_runtime_cmake_cache = (
            release_build / "cuda-runtime-qualification" / "CMakeCache.txt"
        )
        cuda_runtime_cmake_cache.parent.mkdir(exist_ok=True)
        cuda_runtime_cmake_cache.write_text(
            "MEEP_GPU_FAST_MATH:BOOL=OFF\n"
            "CMAKE_CUDA_FLAGS:STRING=\n"
            "CMAKE_CUDA_FLAGS_DEBUG:STRING=-g\n"
            "CMAKE_CUDA_FLAGS_MINSIZEREL:STRING=-O1 -DNDEBUG\n"
            "CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
            "CMAKE_CUDA_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG\n"
            f"CMAKE_CUDA_COMPILER:UNINITIALIZED={nvcc_path}\n"
            f"CMAKE_CUDA_HOST_COMPILER:UNINITIALIZED={host_path}\n",
            encoding="utf-8",
        )
        receipt = {
            "build_kind": "cuda-mpi-python-fp32",
            "build_dir": str(release_build),
            "configuration": {
                "builder": {
                    "path": str(
                        self.repo / "scripts" / "build-meep-cuda-mpi-python.sh"
                    )
                },
                "qualification_contract": (
                    gpmeep_qualification_contract.CONTRACT_NAME
                ),
                "configure_argv": ["--disable-cuda-fast-math"],
                "environment": {"MEEP_GPU_FAST_MATH": "OFF"},
            },
            "configuration_files": {
                "canonical_build_environment": gpmeep_provenance.file_record(
                    canonical_build_environment, self.repo
                ),
                "config_status": gpmeep_provenance.file_record(
                    config_status, self.repo
                ),
                "cuda_runtime_cmake_cache": gpmeep_provenance.file_record(
                    cuda_runtime_cmake_cache, self.repo
                ),
                "cuda_runtime_flags_stamp": gpmeep_provenance.file_record(
                    cuda_runtime_flags_stamp, self.repo
                ),
                "qualification_contract_v2": gpmeep_provenance.file_record(
                    seal_path, self.repo
                ),
                "qualification_identity_before": (
                    gpmeep_provenance.file_record(before, self.repo)
                ),
                "qualification_identity_after": (
                    gpmeep_provenance.file_record(after, self.repo)
                ),
            },
            "artifacts": identity["artifacts"],
            "toolchain": {"nvcc": {"path": str(nvcc_path)}},
            "manifests": {
                "installed_environment": identity["installed_environment"],
                "installed_prefix": identity["installed_prefix"],
                "qualification_logs": {
                    "root": gpmeep_provenance.display_path(
                        log_dir, self.repo
                    ),
                    "files": [
                        {
                            key_name: record[key_name]
                            for key_name in ("path", "size_bytes", "sha256")
                        }
                        for record in (
                            *fake_logs.values(),
                            *fake_materialization_logs.values(),
                        )
                    ],
                },
            },
            "source_start": identity["source"],
            "source_end": identity["source"],
        }

        validation_patches = (
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_logs",
                return_value=fake_logs,
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_no_uncontracted_pass_logs",
                return_value={
                    name: {
                        "path": name,
                        "size_bytes": 1,
                        "sha256": "2" * 64,
                        "terminal_marker": "materialized",
                    }
                    for name in gpmeep_qualification_contract.MATERIALIZATION_LOGS
                },
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_mutation_sentinel_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_python_runtime_logs",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_lazy_api_qualification_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_mpb_qualification_log",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_installed_cuda_qualification_logs",
            ),
            mock.patch.object(
                gpmeep_qualification_contract,
                "validate_direct_elf_logs",
            ),
        )
        for patcher in validation_patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        gpmeep_qualification_contract.validate_v2_receipt(receipt, self.repo)
        forged_identity = dict(identity)
        forged_identity[key] = {}
        gpmeep_provenance.atomic_write_json(before, forged_identity)
        gpmeep_provenance.atomic_write_json(after, forged_identity)
        seal[key] = {}
        seal["identity_sha256"] = canonical_sha256(forged_identity)
        gpmeep_provenance.atomic_write_json(seal_path, seal)
        with self.assertRaisesRegex(RuntimeError, "wrong immutable.*binding set"):
            gpmeep_qualification_contract.validate_v2_receipt(receipt, self.repo)

    def test_inotify_sentinel_rejects_transient_source_swap_and_restore(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifacts,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        identity_path = self.build / "qualification-identity.json"
        ready_path = self.build / "sentinel.ready"
        gpmeep_provenance.atomic_write_json(identity_path, identity)
        process = subprocess.Popen(
            [
                sys.executable,
                str(QUALIFICATION_SCRIPT),
                "sentinel",
                "--identity", str(identity_path),
                "--repo", str(self.repo),
                "--prefix", str(prefix),
                "--ready", str(ready_path),
            ],
            cwd=self.repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(
            lambda: process.kill() if process.poll() is None else None
        )
        for _ in range(500):
            if ready_path.is_file() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(
            ready_path.is_file(), f"sentinel returncode={process.poll()}"
        )
        source = self.repo / "source.cpp"
        original = source.read_bytes()
        source.write_bytes(b"int transient_attack = 1;\n")
        source.write_bytes(original)
        for _ in range(200):
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            process.terminate()
        output, _ = process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("mutation", output)
        sentinel_marker = gpmeep_qualification_contract.marker_for(
            gpmeep_qualification_contract.SENTINEL_LOG_NAME
        ).decode("ascii")
        self.assertNotIn(
            sentinel_marker, output, sentinel_marker
        )

    def test_inotify_sentinel_rejects_transient_new_source_create_delete(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifacts,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        identity_path = self.build / "qualification-identity.json"
        ready_path = self.build / "sentinel.ready"
        gpmeep_provenance.atomic_write_json(identity_path, identity)
        process = subprocess.Popen(
            [
                sys.executable,
                str(QUALIFICATION_SCRIPT),
                "sentinel",
                "--identity", str(identity_path),
                "--repo", str(self.repo),
                "--prefix", str(prefix),
                "--ready", str(ready_path),
            ],
            cwd=self.repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(
            lambda: process.kill() if process.poll() is None else None
        )
        for _ in range(500):
            if ready_path.is_file() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(
            ready_path.is_file(), f"sentinel returncode={process.poll()}"
        )
        transient = self.repo / "transient-new-source.cpp"
        transient.write_text("int transient = 1;\n", encoding="utf-8")
        transient.unlink()
        for _ in range(200):
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            process.terminate()
        output, _ = process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("source-dir mutation", output)

    def test_inotify_sentinel_clean_epoch_emits_bound_terminal_pass(self):
        self.create_outputs()
        prefix = self.build / "prefix"
        prefix.mkdir()
        (prefix / "package.txt").write_text("locked\n", encoding="utf-8")
        artifact = self.build / "artifact.so"
        artifacts = {
            name: artifact
            for name in gpmeep_qualification_contract.REQUIRED_ARTIFACT_BINDINGS
        }
        pycache = self.build / "qualification-pycache"
        pycache.mkdir(mode=0o700)
        identity = gpmeep_qualification_contract.create_identity_snapshot(
            self.repo,
            prefix,
            artifacts,
            immutable_empty_directories={"qualification_pycache": pycache},
        )
        identity_path = self.build / "qualification-identity.json"
        ready_path = self.build / "sentinel.ready"
        gpmeep_provenance.atomic_write_json(identity_path, identity)
        process = subprocess.Popen(
            [
                sys.executable,
                str(QUALIFICATION_SCRIPT),
                "sentinel",
                "--identity", str(identity_path),
                "--repo", str(self.repo),
                "--prefix", str(prefix),
                "--ready", str(ready_path),
            ],
            cwd=self.repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(
            lambda: process.kill() if process.poll() is None else None
        )
        for _ in range(500):
            if ready_path.is_file() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(
            ready_path.is_file(), f"sentinel returncode={process.poll()}"
        )
        process.terminate()
        output, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, output)
        lines = output.splitlines()
        marker = gpmeep_qualification_contract.marker_for(
            gpmeep_qualification_contract.SENTINEL_LOG_NAME
        ).decode("ascii")
        self.assertEqual(lines[-1], marker)
        attestation = json.loads(lines[-2])
        self.assertEqual(
            attestation["identity_sha256"], canonical_sha256(identity)
        )
        self.assertGreater(attestation["watch_count"], 0)

    def test_source_mutation_rejects_final_receipt(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        (self.repo / "source.cpp").write_text("int value = 2;\n")
        completed = self.finalize()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("source tree changed", completed.stderr)
        self.assertFalse((self.build / "build-provenance.json").exists())

    def test_source_mutation_after_finalization_is_rejected_on_replay(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        self.assertEqual(self.finalize().returncode, 0)
        receipt_path = self.build / "build-provenance.json"
        (self.repo / "source.cpp").write_text("int value = 3;\n")
        with self.assertRaisesRegex(
            ProvenanceError, "changed|source snapshot differs"
        ):
            verify_build_receipt(receipt_path, self.repo, verify_source=True)

    def test_begin_removes_stale_authoritative_receipt(self):
        self.build.mkdir(parents=True)
        stale = self.build / "build-provenance.json"
        stale.write_text('{"state":"complete"}\n')
        self.assertEqual(self.begin().returncode, 0)
        self.assertFalse(stale.exists())
        self.assertTrue((self.build / "build-provenance.pending.json").is_file())

    def test_tampered_artifact_and_receipt_payload_are_rejected(self):
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        self.assertEqual(self.finalize().returncode, 0)
        receipt_path = self.build / "build-provenance.json"
        (self.build / "artifact.so").write_bytes(b"changed")
        with self.assertRaisesRegex(ProvenanceError, "artifacts.extension changed"):
            verify_build_receipt(receipt_path, self.repo)

        value = json.loads(receipt_path.read_text())
        value["build_kind"] = "tampered"
        receipt_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ProvenanceError, "receipt ID"):
            verify_build_receipt(receipt_path, self.repo, verify_source=False)

    def test_tampered_toolchain_binary_is_rejected(self):
        tool = self.repo / "synthetic-tool"
        tool.write_text("#!/bin/sh\necho synthetic\n", encoding="utf-8")
        tool.chmod(0o755)
        self.assertEqual(self.begin().returncode, 0)
        self.create_outputs()
        completed = self.invoke(
            "finalize",
            "--repo", str(self.repo),
            "--build-dir", str(self.build),
            "--configuration-file", f"config_h={self.build / 'config.h'}",
            "--artifact", f"extension={self.build / 'artifact.so'}",
            "--manifest", f"package={self.package}",
            "--tool", str(tool),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        tool.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ProvenanceError, "toolchain.*changed"):
            verify_build_receipt(
                self.build / "build-provenance.json", self.repo,
                verify_source=False,
            )


if __name__ == "__main__":
    unittest.main()
