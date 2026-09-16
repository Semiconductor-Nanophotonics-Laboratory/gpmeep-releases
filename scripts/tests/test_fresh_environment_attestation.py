from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
ATTESTATION = SCRIPTS / "fresh-environment-attestation.py"
CREATE_ENV = SCRIPTS / "create-env.sh"
BOOTSTRAP_MICROMAMBA = SCRIPTS / "bootstrap-micromamba.sh"


class FreshEnvironmentAttestationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = pathlib.Path(self.temporary.name).resolve()
        self.prefix = self.repo / "prefix"
        self.prefix.mkdir()
        (self.prefix / "library.so").write_bytes(b"trusted environment bytes\n")
        self.lock = self.repo / "environment.lock"
        self.lock.write_text(
            "@EXPLICIT\nhttps://example.invalid/pkg.conda#" + "1" * 64 + "\n",
            encoding="utf-8",
        )
        self.attestations = self.repo / "attestations"

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/python3", str(ATTESTATION), *arguments],
            cwd=self.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    def create(self) -> tuple[str, pathlib.Path]:
        completed = self.invoke(
            "create",
            "--repo",
            str(self.repo),
            "--prefix",
            str(self.prefix),
            "--lock",
            str(self.lock),
            "--output-directory",
            str(self.attestations),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        return lines[0], pathlib.Path(lines[1])

    def verify(self, nonce: str, attestation: pathlib.Path):
        return self.invoke(
            "verify",
            "--repo",
            str(self.repo),
            "--prefix",
            str(self.prefix),
            "--lock",
            str(self.lock),
            "--attestation",
            str(attestation),
            "--attestation-directory",
            str(self.attestations),
            "--nonce",
            nonce,
        )

    def test_fresh_prefix_and_nonce_binding_passes(self):
        nonce, attestation = self.create()
        self.assertEqual(attestation.stat().st_mode & 0o777, 0o600)
        completed = self.verify(nonce, attestation)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_wrong_nonce_and_prefix_tamper_fail_closed(self):
        nonce, attestation = self.create()
        wrong = self.verify("2" * 64, attestation)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("binding is invalid", wrong.stderr)
        (self.prefix / "library.so").write_bytes(b"tampered\n")
        changed = self.verify(nonce, attestation)
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("changed after recreation", changed.stderr)

    def test_volatile_bytecode_is_excluded_but_source_is_bound(self):
        cache = self.prefix / "__pycache__"
        cache.mkdir()
        bytecode = cache / "module.pyc"
        bytecode.write_bytes(b"first-bytecode")
        source = self.prefix / "module.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        nonce, attestation = self.create()
        bytecode.write_bytes(b"second-bytecode")
        passed = self.verify(nonce, attestation)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        source.write_text("VALUE = 2\n", encoding="utf-8")
        rejected = self.verify(nonce, attestation)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("changed after recreation", rejected.stderr)

    def test_micromamba_recreation_enables_extra_safety_checks(self):
        source = CREATE_ENV.read_text(encoding="utf-8")
        self.assertIn("--no-pyc", source)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("--safety-checks enabled", source)
        self.assertIn("--extra-safety-checks", source)

    def test_micromamba_bootstrap_pins_build_qualified_release_and_two_hashes(self):
        source = BOOTSTRAP_MICROMAMBA.read_text(encoding="utf-8")
        self.assertIn('MAMBA_VERSION="2.8.1"', source)
        self.assertIn('MAMBA_RELEASE_BUILD="0"', source)
        self.assertIn("github.com/mamba-org/micromamba-releases/releases/download/", source)
        self.assertIn("${MAMBA_VERSION}-${MAMBA_RELEASE_BUILD}", source)
        self.assertIn("micromamba-linux-64.tar.bz2", source)
        self.assertNotIn("micro.mamba.pm/api/micromamba", source)
        self.assertIn(
            'ARCHIVE_SHA256="a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515"',
            source,
        )
        self.assertIn(
            'BINARY_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"',
            source,
        )
        self.assertIn('[[ "${ACTUAL_ARCHIVE_SHA256}" != "${ARCHIVE_SHA256}" ]]', source)
        self.assertIn('[[ "${ACTUAL_BINARY_SHA256}" != "${BINARY_SHA256}" ]]', source)


if __name__ == "__main__":
    unittest.main()
