import json
import marshal
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class PackagingContractTests(unittest.TestCase):
    def test_installed_launchers_select_prefix_python_with_scrubbed_path(self):
        launchers = {
            "gpmeep-self-check": (
                "gpmeep-self-check-launcher.sh",
                "gpmeep-self-check.py",
            ),
            "gpmeep-verify-provenance": (
                "gpmeep-verify-provenance-launcher.sh",
                "gpmeep-verify-provenance.py",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "deep" / "relocated" / "prefix"
            binary = prefix / "bin"
            libexec = prefix / "share" / "gpmeep" / "libexec"
            binary.mkdir(parents=True)
            libexec.mkdir(parents=True)
            (binary / "python3.11").symlink_to(pathlib.Path(sys.executable).resolve())
            for command, (launcher, payload_name) in launchers.items():
                installed_launcher = binary / command
                shutil.copy2(ROOT / "scripts" / launcher, installed_launcher)
                installed_launcher.chmod(0o755)
                payload = libexec / payload_name
                payload.write_text(
                    "import json, os, sys\n"
                    "print(json.dumps({'executable': sys.executable, "
                    "'argv0': sys.argv[0], 'args': sys.argv[1:], "
                    "'path': os.environ['PATH']}))\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [str(installed_launcher), "probe"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={"PATH": "/usr/bin:/bin"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                record = json.loads(completed.stdout)
                self.assertEqual(
                    pathlib.Path(record["executable"]).resolve(),
                    pathlib.Path(sys.executable).resolve(),
                )
                self.assertEqual(pathlib.Path(record["argv0"]).resolve(), payload)
                self.assertEqual(record["args"], ["probe"])
                self.assertEqual(record["path"], f"{binary}:/usr/bin:/bin")

    def test_bootstrap_chain_is_protected_against_hostile_path_bash(self):
        protected = (
            ROOT / "scripts" / "bootstrap-micromamba.sh",
            ROOT / "scripts" / "create-env.sh",
        )
        for script in protected:
            source = script.read_text(encoding="utf-8")
            self.assertTrue(source.startswith("#!/bin/bash -p\n"))
            self.assertIn("requires protected Bash mode", source)
            subprocess.run(["/bin/bash", "-p", "-n", str(script)], check=True)
        with tempfile.TemporaryDirectory() as directory:
            hostile = pathlib.Path(directory)
            marker = hostile / "ambient-bash-ran"
            for name in ("bash", "dirname"):
                executable = hostile / name
                executable.write_text(
                    '#!/bin/sh\nprintf hostile >"$GPMEEP_HOSTILE_MARKER"\nexit 93\n',
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            environment = {
                "PATH": str(hostile),
                "GPMEEP_HOSTILE_MARKER": str(marker),
            }
            for script in protected:
                completed = subprocess.run(
                    ["/bin/bash", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 1)
                self.assertIn("protected Bash mode", completed.stderr)
                self.assertFalse(marker.exists())
        installer = (ROOT / "scripts" / "install-gpmeep.sh").read_text()
        builder = (ROOT / "scripts" / "build-conda-package.sh").read_text()
        self.assertIn('/bin/bash -p "${SCRIPT_DIR}/bootstrap-micromamba.sh"', installer)
        self.assertIn('/bin/bash -p "${SCRIPT_DIR}/create-env.sh"', builder)

    def test_release_sidecars_and_secret_ignore_are_portable(self):
        builder = (ROOT / "scripts" / "build-conda-package.sh").read_text()
        self.assertIn('$(/usr/bin/basename "${SOURCE_ARCHIVE}")', builder)
        self.assertIn('$(/usr/bin/basename "${PACKAGE}")', builder)
        self.assertNotIn(
            '/usr/bin/sha256sum "${SOURCE_ARCHIVE}" >"${SOURCE_ARCHIVE_SHA256}"',
            builder,
        )
        self.assertNotIn(
            '/usr/bin/sha256sum "${PACKAGE}" >"${PACKAGE}.sha256"', builder
        )
        ignores = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("*.env", ignores)

    def test_package_metadata_suppresses_build_path_residue(self):
        pkg_config = (ROOT / "packaging" / "conda" / "meep.pc").read_text()
        recipe = (ROOT / "packaging" / "conda" / "meta.yaml").read_text()
        build = (ROOT / "packaging" / "conda" / "build.sh").read_text()
        self.assertIn("prefix=${pcfiledir}/../..", pkg_config)
        self.assertNotIn("/scratch/", pkg_config)
        self.assertNotIn("-Wl,-rpath", pkg_config)
        self.assertIn("skip_compile_pyc:", recipe)
        self.assertIn("-ffile-prefix-map=${SRC_DIR}=gpmeep-source", build)
        self.assertIn('"${RECIPE_DIR}/meep.pc"', build)

    def test_package_owns_prefix_independent_checked_hash_bytecode(self):
        build = (ROOT / "packaging" / "conda" / "build.sh").read_text()
        self.assertIn("--invalidation-mode checked-hash", build)
        self.assertIn('-s "${PREFIX}" -p \'\'', build)
        self.assertIn('BYTECODE_COUNT="$(find', build)
        self.assertIn("package-owned Python bytecode inventory is incomplete", build)

        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "private-build-prefix"
            package = prefix / "lib/python3.11/site-packages/meep"
            package.mkdir(parents=True)
            source = package / "probe.py"
            source.write_text("VALUE = 7\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "compileall",
                    "--invalidation-mode",
                    "checked-hash",
                    "-q",
                    "-f",
                    "-s",
                    str(prefix),
                    "-p",
                    "",
                    str(package),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            bytecode = next(package.rglob("*.pyc")).read_bytes()
            self.assertEqual(int.from_bytes(bytecode[4:8], "little"), 3)
            code = marshal.loads(bytecode[16:])
            self.assertEqual(
                code.co_filename,
                "lib/python3.11/site-packages/meep/probe.py",
            )
            self.assertNotIn(str(prefix), bytecode.decode("latin-1"))

    def test_exact_locks_are_canonical_explicit_files(self):
        for name in (
            "gpmeep-runtime-linux-64.lock",
            "package-builder-linux-64.lock",
        ):
            lines = (ROOT / "environment/locks" / name).read_text().splitlines()
            self.assertEqual(lines[2], "# platform: linux-64")
            self.assertEqual(lines[3], "@EXPLICIT")
            self.assertGreater(len(lines), 100)
            for line in lines[4:]:
                self.assertRegex(
                    line,
                    r"^https://conda\.anaconda\.org/conda-forge/.+#[0-9a-f]{64}$",
                )

    def test_recipe_is_cuda_mpi_fp32_and_dedicated(self):
        recipe = (ROOT / "packaging/conda/meta.yaml").read_text()
        build = (ROOT / "packaging/conda/build.sh").read_text()
        for expected in (
            "name: gpmeep",
            'distribution_version = "1.0.3"',
            "build_number = 0",
            "cuda-cudart =12.4",
            "cuda-cuobjdump =12.4",
            "openmpi =5.0.10",
            "libstdcxx >=12",
            "libgcc >=12",
            "libgomp >=12",
            "__cuda >=12.4",
            "pymeep <0a0",
        ):
            self.assertIn(expected, recipe)
        for expected in (
            "--enable-single",
            "--enable-cuda",
            "--with-mpi",
            "--with-python",
            "--without-scheme",
            "--disable-cuda-fast-math",
        ):
            self.assertIn(expected, build)
        self.assertIn("values#0, }", build)

    def test_release_manifest_has_portability_disclosure(self):
        manifest = (ROOT / "packaging/conda/release.json.in").read_text()
        self.assertIn('"schema_version": 3', manifest)
        self.assertIn('"distribution_version": "1.0.3"', manifest)
        self.assertIn(
            '"runtime_validated_architectures": [@RUNTIME_VALIDATED_ARCHITECTURES@]',
            manifest,
        )
        self.assertIn('"cuda_architectures": "@CUDA_ARCHITECTURES@"', manifest)
        self.assertIn('"glibc_minimum": "2.17"', manifest)
        self.assertIn(
            '"cuda_architecture_audit_sha256": "@CUDA_ARCHITECTURE_AUDIT_SHA256@"',
            manifest,
        )
        self.assertIn('"glibc_compatibility_audit_sha256": "@GLIBC_AUDIT_SHA256@"', manifest)
        self.assertIn('"package_policy": "dedicated-environment-only"', manifest)
        self.assertIn(
            '"source_commit": "adc67c3466774bbdb1875d996e4e0e50aeb7a845"',
            manifest,
        )
        self.assertIn(
            '"report_sha256": '
            '"50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc"',
            manifest,
        )

    def test_installer_help_is_non_mutating(self):
        result = subprocess.run(
            ["/bin/bash", "-p", str(ROOT / "scripts/install-gpmeep.sh"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--package-sha256", result.stderr)
        self.assertIn("--expected-source-commit", result.stderr)

    def test_installer_requires_absent_absolute_prefix_and_checksum(self):
        installer = (ROOT / "scripts/install-gpmeep.sh").read_text()
        for expected in (
            "installation prefix must be absent",
            "a 64-hex package checksum",
            "materialized runtime differs from the exact lock",
            "--no-deps",
            "--safety-checks warn",
            "HOME=${RUNTIME_HOME}",
            "--expected-source-commit",
            "--expected-package-sha256",
            "gpmeep-self-check",
        ):
            self.assertIn(expected, installer)

    def test_package_builder_requires_clean_committed_source(self):
        builder = (ROOT / "scripts/build-conda-package.sh").read_text()
        recipe = (ROOT / "packaging/conda/meta.yaml").read_text()
        self.assertIn("package source tree must be clean and committed", builder)
        self.assertIn("GPMEEP_SOURCE_COMMIT", builder)
        self.assertIn('git -C "${REPO_ROOT}" archive', builder)
        self.assertIn("GPMEEP_SOURCE_TREE", recipe)
        self.assertIn("package-builder-linux-64.lock", builder)
        self.assertIn("CPU_COUNT=${BUILD_JOBS}", builder)
        self.assertIn(
            'MAMBA_PROCESS_CACHE="${MAMBA_ROOT}/package-builder-xdg-cache"', builder
        )
        self.assertIn('/usr/bin/install -d -m 0700 "${MAMBA_PROCESS_CACHE}"', builder)
        self.assertIn('XDG_CACHE_HOME="${MAMBA_PROCESS_CACHE}"', builder)


if __name__ == "__main__":
    unittest.main()
