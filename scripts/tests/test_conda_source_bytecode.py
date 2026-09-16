from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
COMPILER = SCRIPTS / "compile-conda-relocated-bytecode.py"
NORMALIZER_PATH = SCRIPTS / "normalize-conda-relocated-bytecode.py"
PROBE_PATH = SCRIPTS / "probe-conda-relocated-bytecode-imports.py"


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


NORMALIZER = load_module("gpmeep_conda_source_bytecode_normalizer", NORMALIZER_PATH)
PROBE = load_module("gpmeep_conda_source_bytecode_probe", PROBE_PATH)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CondaSourceBytecodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.prefix = self.root / "prefix"
        runner = self.prefix / "bin/python3.11"
        runner.parent.mkdir(parents=True)
        runner.write_text(
            "#!/bin/sh\nexec " + str(pathlib.Path(sys.executable).resolve()) + ' "$@"\n',
            encoding="utf-8",
        )
        runner.chmod(0o755)
        package = self.prefix / "lib/python3.11/site-packages/pkg"
        package.mkdir(parents=True)
        self.relocated_source = package / "relocated.py"
        generated_package = self.prefix / "lib/python3.11/site-packages/pkg2"
        generated_package.mkdir(parents=True)
        self.generated_source = generated_package / "generated.py"
        self.relocated_source.write_text("VALUE = 'installed-prefix'\n", encoding="utf-8")
        self.generated_source.write_text("VALUE = 7\n", encoding="utf-8")
        cache = package / "__pycache__"
        cache.mkdir()
        self.relocated_pyc = cache / "relocated.cpython-311.pyc"
        self.generated_pyc = (
            generated_package / "__pycache__/generated.cpython-311.pyc"
        )
        self.relocated_pyc.write_bytes(b"archive-checked-hash-pyc")
        self.python_config = (
            self.prefix
            / "lib/python3.11/config-3.11-test-linux-gnu/python-config.py"
        )
        self.python_config.parent.mkdir(parents=True)
        self.python_config.write_text("VALUE = 1\n", encoding="utf-8")
        self.pre_audit = self.root / "pre-audit.json"
        self.normalization = self.root / "normalization.json"

    def compiler_records(self):
        request = self.root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "path": str(self.relocated_pyc.relative_to(self.prefix)),
                            "source_path": str(
                                self.relocated_source.relative_to(self.prefix)
                            ),
                        },
                        {
                            "path": str(self.generated_pyc.relative_to(self.prefix)),
                            "source_path": str(
                                self.generated_source.relative_to(self.prefix)
                            ),
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "compiled"
        output.mkdir()
        report = self.root / "compiler.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(COMPILER),
                "--prefix",
                str(self.prefix),
                "--request",
                str(request),
                "--output-directory",
                str(output),
                "--report",
                str(report),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(report.read_text(encoding="utf-8"))["records"]

    def write_pre_audit(self, compiled_records):
        by_path = {record["path"]: record for record in compiled_records}
        relocated_path = str(self.relocated_pyc.relative_to(self.prefix))
        generated_path = str(self.generated_pyc.relative_to(self.prefix))
        value = {
            "schema_version": 4,
            "environment_prefix": str(self.prefix),
            "python_version": "3.11",
            "relocated_source_bytecode": [
                {
                    "archive_sha256": sha256(self.relocated_pyc),
                    "current_sha256": sha256(self.relocated_pyc),
                    "current_state": "archive-exact",
                    "derived_sha256": by_path[relocated_path]["pyc_sha256"],
                    "installed_source_hash_hex": by_path[relocated_path][
                        "source_hash_hex"
                    ],
                    "installed_source_sha256": sha256(self.relocated_source),
                    "path": relocated_path,
                    "source_path": str(self.relocated_source.relative_to(self.prefix)),
                }
            ],
            "generated_source_bytecode": [
                {
                    "current_sha256": None,
                    "current_state": "absent",
                    "derived_sha256": by_path[generated_path]["pyc_sha256"],
                    "installed_source_hash_hex": by_path[generated_path][
                        "source_hash_hex"
                    ],
                    "installed_source_sha256": sha256(self.generated_source),
                    "path": generated_path,
                    "source_path": str(self.generated_source.relative_to(self.prefix)),
                }
            ],
            "pass": True,
        }
        self.pre_audit.write_text(json.dumps(value), encoding="utf-8")

    def test_normalizer_installs_exact_checked_hash_bytecode(self):
        compiled = self.compiler_records()
        self.write_pre_audit(compiled)
        report = NORMALIZER.normalize(self.prefix, self.pre_audit, COMPILER)
        self.assertTrue(report["pass"])
        self.assertEqual(report["normalized_count"], 2)
        self.assertEqual(report["created_cache_directory_count"], 1)
        expected = {record["path"]: record["pyc_sha256"] for record in compiled}
        self.assertEqual(
            sha256(self.relocated_pyc),
            expected[str(self.relocated_pyc.relative_to(self.prefix))],
        )
        self.assertEqual(
            sha256(self.generated_pyc),
            expected[str(self.generated_pyc.relative_to(self.prefix))],
        )

        self.normalization.write_text(json.dumps(report), encoding="utf-8")
        probe = PROBE.probe(self.prefix, self.normalization)
        self.assertTrue(probe["pass"])
        self.assertEqual(len(probe["bytecode"]), 2)

    def test_normalizer_rejects_source_and_derived_digest_mutations(self):
        compiled = self.compiler_records()
        self.write_pre_audit(compiled)
        value = json.loads(self.pre_audit.read_text(encoding="utf-8"))
        value["generated_source_bytecode"][0]["derived_sha256"] = "0" * 64
        self.pre_audit.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "audited derivation"):
            NORMALIZER.normalize(self.prefix, self.pre_audit, COMPILER)

        self.write_pre_audit(compiled)
        self.generated_source.write_text("VALUE = 8\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            NORMALIZER.normalize(self.prefix, self.pre_audit, COMPILER)

    def test_compiler_honors_optimized_bytecode_target(self):
        optimized_pyc = self.relocated_pyc.with_name(
            "relocated.cpython-311.opt-1.pyc"
        )
        self.relocated_pyc = optimized_pyc
        compiled = self.compiler_records()
        relocated = next(
            record for record in compiled if record["path"].endswith("opt-1.pyc")
        )
        self.assertEqual(relocated["optimization"], 1)
        self.assertEqual(compiled[1]["optimization"], -1)


if __name__ == "__main__":
    unittest.main()
