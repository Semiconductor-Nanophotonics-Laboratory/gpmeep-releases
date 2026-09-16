from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "normalize-conda-generated-bytecode.py"


class CondaBytecodeNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.prefix = self.root / "prefix"
        self.metadata = self.prefix / "conda-meta"
        self.metadata.mkdir(parents=True)
        self.source = self.prefix / "lib/python3.11/site-packages/probe.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("VALUE = 1\n", encoding="utf-8")
        self.bytecode = self.source.parent / "__pycache__/probe.cpython-311.pyc"
        self.bytecode.parent.mkdir()
        self.output = self.root / "normalization.json"

    def write_metadata(self, generated: str = "lib/python3.11/site-packages/probe/__pycache__/bad.cpython-311.pyc") -> None:
        paths = [
            {
                "_path": "lib/python3.11/site-packages/probe.py",
                "path_type": "hardlink",
            },
            {
                "_path": generated,
                "path_type": "pyc_file",
            },
        ]
        (self.metadata / "probe-1.0-0.json").write_text(
            json.dumps({"paths_data": {"paths": paths}}), encoding="utf-8"
        )

    def invoke(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-S",
                "-P",
                "-B",
                str(SCRIPT),
                "--prefix",
                str(self.prefix),
                "--output",
                str(self.output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    def test_removes_only_declared_generated_bytecode_and_records_digest(self):
        self.write_metadata("lib/python3.11/site-packages/__pycache__/probe.cpython-311.pyc")
        self.bytecode.write_bytes(b"generated-bytecode")
        unrelated = self.bytecode.parent / "unrelated.cpython-311.pyc"
        unrelated.write_bytes(b"not-declared")
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.bytecode.exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(self.source.read_text(encoding="utf-8"), "VALUE = 1\n")
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(report["pass"])
        self.assertEqual(report["removed_path_count"], 1)
        self.assertEqual(report["already_absent_path_count"], 0)
        self.assertEqual(report["removed"][0]["size_bytes"], 18)
        self.assertRegex(report["removed"][0]["sha256"], r"^[0-9a-f]{64}$")

    def test_absent_declared_bytecode_is_accepted(self):
        self.write_metadata("lib/python3.11/site-packages/__pycache__/probe.cpython-311.pyc")
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["removed_path_count"], 0)
        self.assertEqual(report["already_absent_path_count"], 1)

    def test_optimized_declared_bytecode_is_removed(self):
        self.bytecode = self.bytecode.with_name("probe.cpython-311.opt-1.pyc")
        self.write_metadata(
            "lib/python3.11/site-packages/__pycache__/probe.cpython-311.opt-1.pyc"
        )
        self.bytecode.write_bytes(b"optimized-generated-bytecode")
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.bytecode.exists())
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["removed_path_count"], 1)

    def test_symlinked_declared_bytecode_fails_closed(self):
        self.write_metadata("lib/python3.11/site-packages/__pycache__/probe.cpython-311.pyc")
        outside = self.root / "outside.pyc"
        outside.write_bytes(b"outside")
        self.bytecode.symlink_to(outside)
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a canonical file", completed.stderr)
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_non_source_derived_metadata_fails_closed(self):
        self.write_metadata()
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("source is not package-owned", completed.stderr)


if __name__ == "__main__":
    unittest.main()
