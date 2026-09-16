from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def load_comparator():
    path = SCRIPTS / "compare-dense-long-horizon.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_dense_long_horizon_comparator_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dense long-horizon comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = load_comparator()


def load_runner():
    path = SCRIPTS / "run-dense-long-horizon-oracle.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_dense_long_horizon_runner_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dense long-horizon runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def write_evidence(
    path: pathlib.Path,
    lane: str,
    *,
    field_delta: float = 0.0,
    energy: float = 1.25,
) -> None:
    precision, backend = {
        "cpu-fp64": (64, "cpu"),
        "cpu-fp32": (32, "cpu"),
        "cuda-fp32": (32, "cuda"),
    }[lane]
    cpu = backend == "cpu"
    executable_artifact = path.parent / f"{lane}.runtime-elf"
    libmeep_artifact = path.parent / f"{lane}.runtime-libmeep"
    executable_artifact.write_bytes((lane + " executable\n").encode("ascii"))
    libmeep_artifact.write_bytes((lane + " libmeep\n").encode("ascii"))
    lines = [
        COMPARATOR.MAGIC,
        f"lane={lane}",
        f"precision_bits={precision}",
        f"backend={backend}",
        f"executable_path={executable_artifact.resolve()}",
        f"libmeep_path={libmeep_artifact.resolve()}",
        f"timesteps={COMPARATOR.EXPECTED_TIMESTEPS}",
        f"grid_sx={float(COMPARATOR.EXPECTED_GRID[0]).hex()}",
        f"grid_sy={float(COMPARATOR.EXPECTED_GRID[1]).hex()}",
        f"resolution={float(COMPARATOR.EXPECTED_GRID[2]).hex()}",
        f"source_count={len(COMPARATOR.EXPECTED_SOURCES)}",
        f"source_frequency={COMPARATOR.EXPECTED_SOURCE_FREQUENCY.hex()}",
    ]
    for source_index, source_parameters in enumerate(COMPARATOR.EXPECTED_SOURCES):
        component, x, y, real, imag = source_parameters
        lines.extend(
            (
                f"source{source_index}_component={component}",
                f"source{source_index}_x={x.hex()}",
                f"source{source_index}_y={y.hex()}",
                f"source{source_index}_real={real.hex()}",
                f"source{source_index}_imag={imag.hex()}",
            )
        )
    lines.extend(
        [
        f"array_count={COMPARATOR.EXPECTED_ARRAY_COUNT}",
        f"scalar_count={COMPARATOR.EXPECTED_SCALAR_COUNT}",
        f"cpu_curl_calls={344064 if cpu else 0}",
        f"cuda_curl_calls={0 if cpu else 2048}",
        f"cpu_update_eh_calls={79872 if cpu else 0}",
        f"cuda_update_eh_calls={0 if cpu else 2048}",
        f"energy={energy.hex()}",
        ]
    )
    global_index = 0
    for key in COMPARATOR.EXPECTED_KEYS:
        value_count = COMPARATOR.EXPECTED_VALUES_PER_CHUNK[key[0]]
        lines.append(
            "array="
            + ",".join(
                str(item)
                for item in (*key, value_count)
            )
        )
        for local_index in range(value_count):
            value = (global_index + 1) * 1e-7 + (local_index % 13) * 1e-8
            if global_index == 0:
                value += field_delta
            lines.append(f"value={value.hex()}")
            global_index += 1
    lines.append("end=1")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise AssertionError(f"expected one occurrence of {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


class DenseLongHorizonComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.paths = {}
        for lane in ("cpu-fp64", "cpu-fp32", "cuda-fp32"):
            path = self.root / f"{lane}.evidence"
            write_evidence(path, lane)
            self.paths[lane] = path

    def tearDown(self):
        self.temporary.cleanup()

    def compare(self):
        return COMPARATOR.compare_evidence(
            self.paths["cpu-fp64"],
            self.paths["cpu-fp32"],
            self.paths["cuda-fp32"],
        )

    def test_accepts_complete_three_lane_full_array_evidence(self):
        report = self.compare()
        self.assertTrue(report["gate"]["pass"])
        self.assertEqual(
            report["workload"]["scalar_count"],
            COMPARATOR.EXPECTED_SCALAR_COUNT,
        )
        self.assertFalse(report["claims"]["cuda_fp64"])
        self.assertEqual(
            set(report["comparisons"]),
            {
                "cpu_fp32_vs_cuda_fp32",
                "cpu_fp64_vs_cpu_fp32",
                "cpu_fp64_vs_cuda_fp32",
            },
        )

    def test_rejects_nonfinite_field_value(self):
        path = self.paths["cuda-fp32"]
        first = next(
            line for line in path.read_text(encoding="ascii").splitlines()
            if line.startswith("value=")
        )
        replace_once(path, first, "value=nan")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "hexadecimal float"):
            self.compare()

    def test_rejects_nonfinite_energy(self):
        path = self.paths["cpu-fp64"]
        replace_once(path, "energy=0x1.4000000000000p+0", "energy=inf")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "hexadecimal float"):
            self.compare()

    def test_rejects_shortened_physical_timestep_count(self):
        path = self.paths["cpu-fp64"]
        replace_once(
            path,
            f"timesteps={COMPARATOR.EXPECTED_TIMESTEPS}",
            "timesteps=1",
        )
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "workload identity"):
            self.compare()

    def test_rejects_changed_source_identity(self):
        path = self.paths["cpu-fp32"]
        replace_once(path, "source0_component=4", "source0_component=0")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "source identity"):
            self.compare()

    def test_rejects_short_cpu_dispatch_count(self):
        path = self.paths["cpu-fp32"]
        replace_once(path, "cpu_curl_calls=344064", "cpu_curl_calls=336")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "CPU lane dispatch"):
            self.compare()

    def test_rejects_duplicate_array_key(self):
        path = self.paths["cpu-fp32"]
        first, second = COMPARATOR.EXPECTED_KEYS[:2]
        replace_once(
            path,
            "array="
            + ",".join(
                str(item)
                for item in (
                    *second,
                    COMPARATOR.EXPECTED_VALUES_PER_CHUNK[second[0]],
                )
            ),
            "array="
            + ",".join(
                str(item)
                for item in (
                    *first,
                    COMPARATOR.EXPECTED_VALUES_PER_CHUNK[first[0]],
                )
            ),
        )
        with self.assertRaisesRegex(
            COMPARATOR.EvidenceError, "duplicate or noncanonical"
        ):
            self.compare()

    def test_rejects_missing_scalar_by_declared_count(self):
        path = self.paths["cpu-fp64"]
        replace_once(
            path,
            f"scalar_count={COMPARATOR.EXPECTED_SCALAR_COUNT}",
            f"scalar_count={COMPARATOR.EXPECTED_SCALAR_COUNT - 1}",
        )
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "count is incomplete"):
            self.compare()

    def test_rejects_wrong_per_array_count(self):
        path = self.paths["cuda-fp32"]
        key = COMPARATOR.EXPECTED_KEYS[0]
        value_count = COMPARATOR.EXPECTED_VALUES_PER_CHUNK[key[0]]
        replace_once(
            path,
            "array="
            + ",".join(
                str(item)
                for item in (*key, value_count)
            ),
            "array="
            + ",".join(
                str(item)
                for item in (*key, value_count - 1)
            ),
        )
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "key or count"):
            self.compare()

    def test_rejects_zero_energy(self):
        path = self.paths["cpu-fp32"]
        replace_once(path, "energy=0x1.4000000000000p+0", "energy=0x0.0p+0")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "must be positive"):
            self.compare()

    def test_rejects_cuda_lane_without_cuda_dispatch(self):
        path = self.paths["cuda-fp32"]
        replace_once(path, "cuda_curl_calls=2048", "cuda_curl_calls=0")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "CUDA lane dispatch"):
            self.compare()

    def test_rejects_lane_substitution(self):
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "expected cpu-fp64"):
            COMPARATOR.compare_evidence(
                self.paths["cpu-fp32"],
                self.paths["cpu-fp64"],
                self.paths["cuda-fp32"],
            )

    def test_rejects_shared_runtime_artifact_between_lanes(self):
        fp64 = self.paths["cpu-fp64"]
        fp32 = self.paths["cpu-fp32"]
        fp64_executable = next(
            line for line in fp64.read_text(encoding="ascii").splitlines()
            if line.startswith("executable_path=")
        )
        fp32_executable = next(
            line for line in fp32.read_text(encoding="ascii").splitlines()
            if line.startswith("executable_path=")
        )
        replace_once(fp32, fp32_executable, fp64_executable)
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "distinct executable"):
            self.compare()

    def test_rejects_field_norm_drift(self):
        write_evidence(self.paths["cuda-fp32"], "cuda-fp32", field_delta=0.01)
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "numerical gate"):
            self.compare()

    def test_rejects_energy_norm_drift(self):
        write_evidence(self.paths["cuda-fp32"], "cuda-fp32", energy=1.3)
        with self.assertRaisesRegex(
            COMPARATOR.EvidenceError, "energy_relative=.*numerical gate|numerical gate"
        ):
            self.compare()

    def test_rejects_trailing_data(self):
        path = self.paths["cpu-fp64"]
        with path.open("a", encoding="ascii") as stream:
            stream.write("forged=1\n")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "trailing data"):
            self.compare()

    def test_rejects_symlink_evidence(self):
        link = self.root / "linked.evidence"
        link.symlink_to(self.paths["cpu-fp64"])
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "non-symlink"):
            COMPARATOR.parse_evidence(link)

    def test_rejects_truncated_evidence(self):
        path = self.paths["cpu-fp32"]
        lines = path.read_text(encoding="ascii").splitlines()
        path.write_text("\n".join(lines[:-3]) + "\n", encoding="ascii")
        with self.assertRaisesRegex(COMPARATOR.EvidenceError, "unexpected EOF"):
            self.compare()


class DenseLongHorizonRunnerTests(unittest.TestCase):
    def test_environment_is_allowlisted_and_drops_ambient_injection(self):
        ambient = {
            "PATH": "/malicious/path",
            "MEEP_GPU_BACKEND": "cpu",
            "MEEP_GPU_TEST_DENSE_EVIDENCE_BACKEND": "forged",
            "MEEP_UNRELATED_INJECTION": "1",
            "OMPI_MCA_pml": "forged",
            "PMI_RANK": "7",
            "LD_PRELOAD": "/tmp/forged.so",
            "PYTHONPATH": "/tmp/forged",
        }
        with mock.patch.dict("os.environ", ambient, clear=True):
            environment = RUNNER._clean_environment()
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(
            environment["HOME"], "/tmp/gpmeep-dense-oracle-empty-home"
        )
        self.assertEqual(environment["MEEP_GPU_ALLOW_OVERSUBSCRIBE"], "0")
        self.assertFalse(
            any(
                name.startswith(("LD_", "OMPI_", "PMI_", "PYTHON"))
                for name in environment
            )
        )
        self.assertEqual(
            [name for name in environment if name.startswith("MEEP_")],
            ["MEEP_GPU_ALLOW_OVERSUBSCRIBE"],
        )

    def test_runtime_identity_binds_actual_elf_and_loaded_libmeep(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary) / "build"
            tests = build / "tests"
            libraries = build / "src" / ".libs"
            (tests / ".libs").mkdir(parents=True)
            libraries.mkdir(parents=True)
            wrapper = tests / "gpu-step-db"
            elf = tests / ".libs" / "lt-gpu-step-db"
            libmeep = libraries / "libmeep.so.38.0.0"
            wrapper.write_text("wrapper", encoding="ascii")
            elf.write_text("elf", encoding="ascii")
            libmeep.write_text("library", encoding="ascii")
            (libraries / "libmeep.so").symlink_to(libmeep.name)
            parsed = {
                "runtime": {
                    "executable": {
                        "path": str(elf.resolve()),
                        "sha256": COMPARATOR.sha256_file(elf),
                    },
                    "libmeep": {
                        "path": str(libmeep.resolve()),
                        "sha256": COMPARATOR.sha256_file(libmeep),
                    },
                }
            }
            expected = RUNNER.expected_runtime_identity(wrapper.resolve())
            self.assertEqual(
                RUNNER.validate_runtime_identity(
                    wrapper.resolve(), parsed, before_execution=expected
                ),
                expected,
            )
            parsed["runtime"]["libmeep"]["path"] = str(elf.resolve())
            with self.assertRaisesRegex(RuntimeError, "loaded libmeep mismatch"):
                RUNNER.validate_runtime_identity(wrapper.resolve(), parsed)

    def test_runtime_identity_rejects_forged_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary) / "build"
            tests = build / "tests"
            libraries = build / "src" / ".libs"
            (tests / ".libs").mkdir(parents=True)
            libraries.mkdir(parents=True)
            wrapper = tests / "gpu-step-db"
            elf = tests / ".libs" / "lt-gpu-step-db"
            libmeep = libraries / "libmeep.so.38.0.0"
            wrapper.write_text("wrapper", encoding="ascii")
            elf.write_text("elf", encoding="ascii")
            libmeep.write_text("library", encoding="ascii")
            (libraries / "libmeep.so").symlink_to(libmeep.name)
            parsed = {"runtime": RUNNER.expected_runtime_identity(wrapper)}
            parsed["runtime"]["executable"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "executable SHA-256 mismatch"):
                RUNNER.validate_runtime_identity(wrapper, parsed)

    def test_runtime_identity_rejects_artifact_changed_during_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary) / "build"
            tests = build / "tests"
            libraries = build / "src" / ".libs"
            (tests / ".libs").mkdir(parents=True)
            libraries.mkdir(parents=True)
            wrapper = tests / "gpu-step-db"
            elf = tests / ".libs" / "lt-gpu-step-db"
            libmeep = libraries / "libmeep.so.38.0.0"
            wrapper.write_text("wrapper", encoding="ascii")
            elf.write_text("elf-before", encoding="ascii")
            libmeep.write_text("library", encoding="ascii")
            (libraries / "libmeep.so").symlink_to(libmeep.name)
            before = RUNNER.expected_runtime_identity(wrapper)
            elf.write_text("elf-after", encoding="ascii")
            parsed = {"runtime": RUNNER.expected_runtime_identity(wrapper)}
            with self.assertRaisesRegex(RuntimeError, "changed while"):
                RUNNER.validate_runtime_identity(
                    wrapper, parsed, before_execution=before
                )


if __name__ == "__main__":
    unittest.main()
