import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = load(
    "gpmeep_release_manifest_v103", ROOT / "scripts/gpmeep_release_manifest.py"
)
GLIBC = load(
    "check_glibc_compatibility_v103",
    ROOT / "scripts/check-glibc-compatibility.py",
)
CUDA = load(
    "check_cuda_architecture_inventory_v103",
    ROOT / "scripts/check-cuda-architecture-inventory.py",
)


def manifest_value(specification="AUTO", commit="a" * 40):
    value = MANIFEST.default_release_manifest_fields()
    real, virtual = MANIFEST.architectures_for_specification(specification)
    value.update(
        cuda_architectures=specification,
        cuda_real_architectures=real,
        cuda_virtual_architectures=virtual,
        runtime_validated_architectures=MANIFEST.runtime_validated_architectures(
            real
        ),
        source_commit=commit,
    )
    return value


class V103FieldFixTests(unittest.TestCase):
    def test_release_identity_and_default_auto_policy(self):
        value = manifest_value()
        self.assertEqual(MANIFEST.DISTRIBUTION_VERSION, "1.0.3")
        self.assertEqual(MANIFEST.RELEASE_MANIFEST_SCHEMA_VERSION, 3)
        self.assertEqual(value["glibc_minimum"], "2.17")
        self.assertEqual(value["runtime_validated_architectures"], ["sm86"])
        self.assertIs(MANIFEST.validate_release_manifest(value), value)

    def test_explicit_sm86_policy_is_accepted_and_bound(self):
        value = manifest_value("86-real;86-virtual")
        self.assertEqual(value["cuda_real_architectures"], [86])
        self.assertEqual(value["cuda_virtual_architectures"], [86])
        self.assertEqual(value["runtime_validated_architectures"], ["sm86"])
        MANIFEST.validate_release_manifest(value)

    def test_list_without_native_sm86_makes_no_sm86_runtime_claim(self):
        value = manifest_value("90-real;90-virtual")
        self.assertEqual(value["runtime_validated_architectures"], [])
        MANIFEST.validate_release_manifest(value)
        value["runtime_validated_architectures"] = ["sm86"]
        with self.assertRaisesRegex(
            MANIFEST.ReleaseManifestError, "runtime-validated"
        ):
            MANIFEST.validate_release_manifest(value)

    def test_architecture_inventory_mismatch_and_malformed_specs_fail(self):
        value = manifest_value("86-real;86-virtual")
        value["cuda_virtual_architectures"] = [90]
        with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "inventories"):
            MANIFEST.validate_release_manifest(value)
        for specification in (
            "",
            "59-real",
            "086-real",
            "86-real;",
            "90-virtual",
        ):
            with self.subTest(specification=specification):
                with self.assertRaises(MANIFEST.ReleaseManifestError):
                    MANIFEST.architectures_for_specification(specification)

    def test_cuda_binary_listing_parser_and_build_audit(self):
        elf_listing = (
            "ELF file    1: gpmeep.1.sm_86.cubin\n"
            "ELF file    2: gpmeep.2.sm_86.cubin\n"
        )
        ptx_listing = "PTX file    1: gpmeep.1.sm_86.ptx\n"
        self.assertEqual(CUDA.parse_listing(elf_listing, "elf"), ([86], 2))
        self.assertEqual(CUDA.parse_listing(ptx_listing, "ptx"), ([86], 1))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "lib/libmeep.so.38.0.0"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFcuda")
            cuobjdump = root / "cuobjdump"
            cuobjdump.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cuobjdump.chmod(0o755)
            outputs = [
                subprocess.CompletedProcess([], 0, stdout=elf_listing, stderr=""),
                subprocess.CompletedProcess([], 0, stdout=ptx_listing, stderr=""),
                subprocess.CompletedProcess(
                    [], 0, stdout="CUDA cuobjdump release 12.4\n", stderr=""
                ),
            ]
            with mock.patch.object(CUDA.subprocess, "run", side_effect=outputs):
                report = CUDA.audit(
                    root=root,
                    binary=binary,
                    cuobjdump=cuobjdump,
                    expected_real=[86],
                    expected_virtual=[86],
                )
            self.assertEqual(report["real_architectures"], [86])
            self.assertEqual(report["virtual_architectures"], [86])
            self.assertEqual(report["elf_record_count"], 2)

    def test_installed_cuda_audit_is_hash_inventory_and_binary_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            binary = prefix / "lib/libmeep.so.38.0.0"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFbound-cuda")
            report = {
                "schema": MANIFEST.CUDA_AUDIT_SCHEMA,
                "state": "PASS",
                "cuda_binary": {
                    "path": "lib/libmeep.so.38.0.0",
                    "size_bytes": binary.stat().st_size,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                },
                "cuobjdump_version": "CUDA cuobjdump release 12.4",
                "real_architectures": [86],
                "virtual_architectures": [86],
                "elf_record_count": 2,
                "ptx_record_count": 1,
                "elf_listing_sha256": "c" * 64,
                "ptx_listing_sha256": "d" * 64,
            }
            path = prefix / MANIFEST.CUDA_AUDIT_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            MANIFEST.validate_cuda_architecture_audit(
                prefix, digest, [86], [86]
            )
            with self.assertRaisesRegex(
                MANIFEST.ReleaseManifestError, "inventories differ"
            ):
                MANIFEST.validate_cuda_architecture_audit(
                    prefix, digest, [86, 90], [86]
                )
            binary.write_bytes(b"\x7fELFtampered-cuda")
            with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "identity"):
                MANIFEST.validate_cuda_architecture_audit(
                    prefix, digest, [86], [86]
                )
            bad_binary = prefix / "lib/libmeep.so-evil"
            bad_binary.write_bytes(b"\x7fELFbad-name")
            report["cuda_binary"] = {
                "path": "lib/libmeep.so-evil",
                "size_bytes": bad_binary.stat().st_size,
                "sha256": hashlib.sha256(bad_binary.read_bytes()).hexdigest(),
            }
            path.write_text(
                json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            bad_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "unsafe"):
                MANIFEST.validate_cuda_architecture_audit(
                    prefix, bad_digest, [86], [86]
                )

    def test_glibc_version_parser_and_floor(self):
        output = """
          0x0010:   Name: GLIBC_2.2.5  Flags: none
          0x0020:   Name: GLIBC_2.17  Flags: none
        """
        self.assertEqual(GLIBC.parse_required_versions(output), ["2.2.5", "2.17"])
        self.assertGreater(GLIBC.version_key("2.38"), GLIBC.version_key("2.17"))

    def test_glibc_builder_audit_passes_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            elf = root / "lib/libmeep.so.38.0.0"
            elf.parent.mkdir(parents=True)
            elf.write_bytes(b"\x7fELFfixture")
            readelf = root / "readelf"
            readelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            readelf.chmod(0o755)
            passed = subprocess.CompletedProcess(
                [],
                0,
                stdout="Name: GLIBC_2.2.5\nName: GLIBC_2.17\n",
                stderr="",
            )
            with mock.patch.object(GLIBC.subprocess, "run", return_value=passed):
                report = GLIBC.audit(
                    root=root,
                    paths=[elf],
                    maximum="2.17",
                    readelf=readelf,
                )
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(report["overall_maximum_required"], "2.17")
            failed = subprocess.CompletedProcess(
                [], 0, stdout="Name: GLIBC_2.38\n", stderr=""
            )
            with mock.patch.object(GLIBC.subprocess, "run", return_value=failed):
                with self.assertRaisesRegex(GLIBC.CompatibilityError, "above 2.17"):
                    GLIBC.audit(
                        root=root,
                        paths=[elf],
                        maximum="2.17",
                        readelf=readelf,
                    )

    def test_installed_glibc_audit_is_hash_and_elf_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            elf = prefix / "lib/libmeep.so.38.0.0"
            elf.parent.mkdir(parents=True)
            elf.write_bytes(b"\x7fELFbound")
            report = {
                "schema": MANIFEST.GLIBC_AUDIT_SCHEMA,
                "state": "PASS",
                "maximum_allowed": "2.17",
                "overall_maximum_required": "2.17",
                "elf_count": 1,
                "elf_files": [
                    {
                        "path": "lib/libmeep.so.38.0.0",
                        "size_bytes": elf.stat().st_size,
                        "sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
                        "required_versions": ["2.2.5", "2.17"],
                        "maximum_required": "2.17",
                    }
                ],
            }
            path = prefix / MANIFEST.GLIBC_AUDIT_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            MANIFEST.validate_glibc_compatibility_audit(prefix, digest)
            elf.write_bytes(b"\x7fELFtampered")
            with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "identity"):
                MANIFEST.validate_glibc_compatibility_audit(prefix, digest)

    def test_audits_reject_intermediate_parent_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            prefix = root / "prefix"
            outside = root / "outside"
            binary = prefix / "lib/libmeep.so.38.0.0"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFinside")
            outside.mkdir()
            cuda_report = {
                "schema": MANIFEST.CUDA_AUDIT_SCHEMA,
                "state": "PASS",
                "cuda_binary": {
                    "path": "lib/libmeep.so.38.0.0",
                    "size_bytes": binary.stat().st_size,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                },
                "cuobjdump_version": "CUDA cuobjdump release 12.4",
                "real_architectures": [86],
                "virtual_architectures": [86],
                "elf_record_count": 1,
                "ptx_record_count": 1,
                "elf_listing_sha256": "c" * 64,
                "ptx_listing_sha256": "d" * 64,
            }
            glibc_report = {
                "schema": MANIFEST.GLIBC_AUDIT_SCHEMA,
                "state": "PASS",
                "maximum_allowed": "2.17",
                "overall_maximum_required": "2.17",
                "elf_count": 1,
                "elf_files": [
                    {
                        "path": "lib/libmeep.so.38.0.0",
                        "size_bytes": binary.stat().st_size,
                        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "required_versions": ["2.2.5", "2.17"],
                        "maximum_required": "2.17",
                    }
                ],
            }
            cuda_path = outside / "cuda-architecture-inventory.json"
            glibc_path = outside / "glibc-compatibility.json"
            for path, value in (
                (cuda_path, cuda_report),
                (glibc_path, glibc_report),
            ):
                path.write_text(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            (prefix / "share").mkdir()
            (prefix / "share/gpmeep").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "escapes"):
                MANIFEST.validate_cuda_architecture_audit(
                    prefix,
                    hashlib.sha256(cuda_path.read_bytes()).hexdigest(),
                    [86],
                    [86],
                )
            with self.assertRaisesRegex(MANIFEST.ReleaseManifestError, "escapes"):
                MANIFEST.validate_glibc_compatibility_audit(
                    prefix,
                    hashlib.sha256(glibc_path.read_bytes()).hexdigest(),
                )

    def test_build_audits_reject_parent_symlink_escape(self):
        with tempfile.TemporaryDirectory() as root_raw, tempfile.TemporaryDirectory() as outside_raw:
            root = pathlib.Path(root_raw)
            outside = pathlib.Path(outside_raw)
            binary = outside / "libmeep.so.38.0.0"
            binary.write_bytes(b"\x7fELFoutside")
            (root / "lib").symlink_to(outside, target_is_directory=True)
            tool = root / "tool"
            tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tool.chmod(0o755)
            with self.assertRaisesRegex(GLIBC.CompatibilityError, "escapes"):
                GLIBC.audit(
                    root=root,
                    paths=[root / "lib/libmeep.so.38.0.0"],
                    maximum="2.17",
                    readelf=tool,
                )
            with self.assertRaisesRegex(CUDA.InventoryError, "escapes"):
                CUDA.audit(
                    root=root,
                    binary=root / "lib/libmeep.so.38.0.0",
                    cuobjdump=tool,
                    expected_real=[86],
                    expected_virtual=[86],
                )

    def test_builder_canonicalizes_output_and_propagates_failures(self):
        builder = (ROOT / "scripts/build-conda-package.sh").read_text()
        self.assertIn("realpath --canonicalize-missing", builder)
        self.assertIn("canonical output directory is not absolute", builder)
        self.assertIn("CONDA_BUILD_RC=$?", builder)
        self.assertIn('exit "${CONDA_BUILD_RC}"', builder)
        self.assertIn('"${OUTPUT_DIRECTORY}/broken"', builder)
        self.assertIn("artifact in broken/", builder)
        self.assertIn("--output-dir /absolute/path", builder)

    def test_builder_execution_preserves_rc_rejects_broken_and_resolves_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "repo"
            caller = pathlib.Path(directory) / "caller"
            for path in (
                root / "scripts",
                root / "environment/locks",
                root / "packaging/conda",
                root / ".tools",
                root / ".envs/gpmeep-package-builder/bin",
                caller,
            ):
                path.mkdir(parents=True, exist_ok=True)
            (root / ".gitignore").write_text(
                ".tools/\n.envs/\n.micromamba/\n", encoding="utf-8"
            )
            builder = root / "scripts/build-conda-package.sh"
            builder.write_text(
                (ROOT / "scripts/build-conda-package.sh").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            builder.chmod(0o755)
            create_env = root / "scripts/create-env.sh"
            create_env.write_text("#!/bin/bash -p\nexit 0\n", encoding="utf-8")
            create_env.chmod(0o755)
            (root / "environment/locks/package-builder-linux-64.lock").write_text(
                "fixture\n", encoding="utf-8"
            )
            fake_mamba = root / ".tools/micromamba"
            fake_mamba.write_text(
                """#!/bin/bash -p
set -euo pipefail
output=
for argument in "$@"; do
  case "${argument}" in
    CONDA_BLD_PATH=*) output="${argument#CONDA_BLD_PATH=}" ;;
  esac
done
[[ -n "${output}" ]]
case "${output}" in
  *fail-rc*) exit 37 ;;
  *broken*)
    /usr/bin/mkdir -p "${output}/broken"
    printf fixture >"${output}/broken/gpmeep-1.0.3-fixture.conda"
    ;;
  *)
    /usr/bin/mkdir -p "${output}/linux-64"
    printf fixture >"${output}/linux-64/gpmeep-1.0.3-fixture.conda"
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_mamba.chmod(0o755)
            fake_conda = root / ".envs/gpmeep-package-builder/bin/conda"
            fake_conda.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_conda.chmod(0o755)
            subprocess.run(
                ["/usr/bin/git", "init", "-q", str(root)], check=True
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(root), "add", "."], check=True
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=gpmeep-test",
                    "-c",
                    "user.email=gpmeep-test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )

            def run(output: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "/bin/bash",
                        "-p",
                        str(builder),
                        "--output-dir",
                        output,
                    ],
                    cwd=caller,
                    check=False,
                    capture_output=True,
                    text=True,
                )

            failed = run("fail-rc")
            self.assertEqual(failed.returncode, 37)
            self.assertIn("conda build failed with status 37", failed.stderr)
            broken = run("broken")
            self.assertEqual(broken.returncode, 1)
            self.assertIn("artifact in broken/", broken.stderr)
            passed = run("relative-success")
            self.assertEqual(passed.returncode, 0, passed.stderr)
            expected = (
                caller / "relative-success/linux-64/gpmeep-1.0.3-fixture.conda"
            ).resolve()
            self.assertTrue(expected.is_file())
            self.assertIn(f"gpmeep-package={expected}", passed.stdout)

    def test_recipe_and_docs_bind_portable_glibc_and_separate_downloads(self):
        recipe = (ROOT / "packaging/conda/meta.yaml").read_text()
        variants = (ROOT / "packaging/conda/conda_build_config.yaml").read_text()
        readme = (ROOT / "README.md").read_text()
        install_doc = (ROOT / "doc/docs/GPMEEP_INSTALLATION.md").read_text()
        self.assertIn("{{ stdlib('c') }}", recipe)
        self.assertIn("binary_relocation: false", recipe)
        self.assertIn('c_stdlib_version:\n  - "2.17"', variants)
        self.assertNotIn("--package dist/linux-64/gpmeep-1.0.3", readme)
        self.assertIn("downloads/gpmeep-1.0.3", readme)
        self.assertIn("Ubuntu 22.04/glibc 2.35", install_doc)
        self.assertIn(
            "at least one native `-real` target", install_doc.replace("\n", " ")
        )
        ignored = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(ROOT),
                "check-ignore",
                "downloads/gpmeep-1.0.3-fixture.conda",
                "downloads/gpmeep-1.0.3-fixture.conda.sha256",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)
        self.assertEqual(len(ignored.stdout.splitlines()), 2)

    def test_build_installs_shared_validator_and_binds_binary_audits(self):
        build = (ROOT / "packaging/conda/build.sh").read_text()
        template = (ROOT / "packaging/conda/release.json.in").read_text()
        for expected in (
            "gpmeep_release_manifest.py",
            "check-glibc-compatibility.py",
            "check-cuda-architecture-inventory.py",
            "CUOBJDUMP",
            "CUDA_ARCHITECTURE_AUDIT_SHA256",
            "GPMEEP_ELF_FILES",
            "GLIBC_AUDIT_SHA256",
            "RUNTIME_VALIDATED_ARCHITECTURES",
            "must include at least one native -real target",
        ):
            self.assertIn(expected, build)
        self.assertIn('"glibc_minimum": "2.17"', template)
        self.assertIn("@CUDA_ARCHITECTURE_AUDIT_SHA256@", template)
        self.assertIn("@GLIBC_AUDIT_SHA256@", template)
        self.assertLess(
            build.index("--set-rpath"), build.index("check-glibc-compatibility.py")
        )
        self.assertIn(
            "audited package ELF retains replaceable build-prefix bytes", build
        )


if __name__ == "__main__":
    unittest.main()
