import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify-gpmeep-package.py"
SPEC = importlib.util.spec_from_file_location("verify_gpmeep_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def release_manifest(commit="a" * 40):
    return copy.deepcopy({**MODULE.RELEASE_MANIFEST_FIELDS, "source_commit": commit})


class PackageProvenanceTests(unittest.TestCase):
    def test_conda_record_requires_exact_v103_build_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            metadata = prefix / "conda-meta"
            metadata.mkdir()
            path = metadata / "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.json"
            valid = {
                "name": "gpmeep",
                "version": "1.0.3",
                "build_number": 0,
                "build": "cuda124_mpi_openmpi_py311_0",
                "subdir": "linux-64",
            }
            path.write_text(json.dumps(valid), encoding="utf-8")
            selected, record = MODULE._one_conda_record(prefix)
            self.assertEqual(selected, path)
            self.assertEqual(record, valid)

            mutations = {
                "stale_build_number": {"build_number": 1},
                "boolean_build_number": {"build_number": True},
                "stale_build_string": {"build": "cuda124_mpi_openmpi_py311_1"},
                "wrong_subdir": {"subdir": "noarch"},
                "missing_build": {"build": None},
            }
            for name, mutation in mutations.items():
                with self.subTest(name=name):
                    value = {**valid, **mutation}
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(
                        MODULE.VerificationError, "conda build identity"
                    ):
                        MODULE._one_conda_record(prefix)

    def test_package_record_sha_is_valid_and_expected(self):
        digest = "a" * 64
        self.assertEqual(
            MODULE._validate_package_identity({"sha256": digest}, digest), digest
        )
        with self.assertRaisesRegex(MODULE.VerificationError, "differs"):
            MODULE._validate_package_identity({"sha256": digest}, "b" * 64)
        with self.assertRaises(MODULE.VerificationError):
            MODULE._validate_package_identity({}, None)

    def test_owned_path_audit_rejects_escape_missing_and_external_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "prefix"
            prefix.mkdir()
            (prefix / "file").write_text("ok")
            self.assertEqual(
                MODULE._audit_owned_paths(prefix, ["file"])["regular_files"], 1
            )
            for path in ("../escape", "/absolute", "missing"):
                with self.assertRaises(MODULE.VerificationError):
                    MODULE._audit_owned_paths(prefix, [path])
            (prefix / "external").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(MODULE.VerificationError, "escapes"):
                MODULE._audit_owned_paths(prefix, ["external"])
            outside = pathlib.Path(directory) / "outside"
            outside.mkdir()
            (outside / "payload").write_text("outside", encoding="utf-8")
            (prefix / "parent").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(MODULE.VerificationError, "escapes"):
                MODULE._audit_owned_paths(prefix, ["parent/payload"])

    def test_owned_paths_are_unique_and_nonempty(self):
        self.assertEqual(MODULE._owned_paths({"files": ["a", "b"]}), ["a", "b"])
        for record in ({}, {"files": []}, {"files": ["a", "a"]}):
            with self.assertRaises(MODULE.VerificationError):
                MODULE._owned_paths(record)

    def test_owned_path_metadata_hashes_every_installed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / "payload"
            path.write_bytes(b"verified")
            digest = MODULE._sha256_file(path)
            record = {
                "files": ["payload"],
                "paths_data": {
                    "paths": [
                        {
                            "_path": "payload",
                            "path_type": "hardlink",
                            "sha256_in_prefix": digest,
                            "size_in_bytes": len(b"verified"),
                        }
                    ]
                },
            }
            paths = MODULE._owned_paths(record)
            metadata = MODULE._owned_path_metadata(record, paths)
            audit = MODULE._audit_owned_paths(prefix, paths, metadata)
            self.assertEqual(audit["content_hashes_verified"], 1)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(MODULE.VerificationError, "hash differs"):
                MODULE._audit_owned_paths(prefix, paths, metadata)

    def test_manifest_binds_commit_and_configuration(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / "share/gpmeep/release.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(release_manifest(commit)))
            with mock.patch.object(
                MODULE, "validate_cuda_architecture_audit"
            ), mock.patch.object(MODULE, "validate_glibc_compatibility_audit"):
                self.assertEqual(
                    MODULE._manifest(prefix, commit)["source_commit"], commit
                )
                self.assertEqual(
                    MODULE._manifest(prefix, commit)["performance_anchor"],
                    MODULE.SEALED_M3_ANCHOR,
                )
                with self.assertRaisesRegex(MODULE.VerificationError, "differs"):
                    MODULE._manifest(prefix, "b" * 40)

    def test_manifest_rejects_schema_policy_and_anchor_mutations(self):
        mutations = {
            "missing_python_import": lambda value: value.pop("python_import"),
            "unexpected_key": lambda value: value.update(unexpected_claim=True),
            "wrong_cuda_toolkit": lambda value: value.update(cuda_toolkit="0.0"),
            "wrong_cuda_policy": lambda value: value.update(
                cuda_architectures="1-real"
            ),
            "empty_validated_architectures": lambda value: value.update(
                runtime_validated_architectures=[]
            ),
            "boolean_as_integer": lambda value: value["performance_anchor"].update(
                qualification_eligible=1
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / "share/gpmeep/release.json"
            path.parent.mkdir(parents=True)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = release_manifest()
                    mutate(value)
                    path.write_text(json.dumps(value))
                    with self.assertRaises(MODULE.VerificationError):
                        MODULE._manifest(prefix, "a" * 40)

    def test_manifest_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / "share/gpmeep/release.json"
            path.parent.mkdir(parents=True)
            raw = json.dumps(release_manifest()).replace(
                '"qualification_eligible": true',
                '"qualification_eligible": 1, "qualification_eligible": true',
            )
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(MODULE.VerificationError, "duplicate JSON"):
                MODULE._manifest(prefix, "a" * 40)

    def test_manifest_rejects_intermediate_parent_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            prefix = root / "prefix"
            outside = root / "outside"
            (prefix / "share").mkdir(parents=True)
            outside.mkdir()
            (outside / "release.json").write_text(
                json.dumps(release_manifest()), encoding="utf-8"
            )
            (prefix / "share/gpmeep").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(MODULE.VerificationError, "unreadable"):
                MODULE._manifest(prefix, "a" * 40)


if __name__ == "__main__":
    unittest.main()
