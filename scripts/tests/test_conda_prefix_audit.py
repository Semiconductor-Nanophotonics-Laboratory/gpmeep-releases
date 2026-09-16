from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
import pathlib
import py_compile
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
AUDITOR = SCRIPTS / "audit-conda-prefix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_conda_prefix_audit", AUDITOR)
assert SPEC and SPEC.loader
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT_MODULE
SPEC.loader.exec_module(AUDIT_MODULE)


class CondaPrefixAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.prefix = self.root / "prefix"
        self.metadata = self.prefix / "conda-meta"
        self.library = self.prefix / "lib" / "libtrusted.so"
        self.metadata.mkdir(parents=True)
        self.library.parent.mkdir()
        self.library.write_bytes(b"trusted package bytes\n")
        self.package_cache = self.root / "package-cache"
        self.archive = (
            self.package_cache
            / "https"
            / "example.invalid"
            / "pkg-1.0-0.tar.bz2"
        )
        self.archive.parent.mkdir(parents=True)
        self.url = "https://example.invalid/pkg-1.0-0.tar.bz2"
        self.package_digest = self.write_package_archive()
        installed_digest = hashlib.sha256(self.library.read_bytes()).hexdigest()
        (self.metadata / "pkg-1.0-0.json").write_text(
            json.dumps(
                {
                    "name": "python",
                    "version": "3.11.9",
                    "build": "0",
                    "url": self.url,
                    "sha256": self.package_digest,
                    "package_tarball_full_path": str(self.archive),
                    "paths_data": {
                        "paths": [
                            {
                                "_path": "lib/libtrusted.so",
                                "path_type": "hardlink",
                                "sha256": installed_digest,
                                "sha256_in_prefix": installed_digest,
                                "size_in_bytes": self.library.stat().st_size,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        self.lock = self.root / "environment.lock"
        self.lock.write_text(
            f"@EXPLICIT\n{self.url}#{self.package_digest}\n", encoding="utf-8"
        )
        self.output = self.root / "audit.json"

    def add_prefix_python_runner(self):
        runner = self.prefix / "bin/python3.11"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text(
            "#!/bin/sh\nexec " + str(pathlib.Path(sys.executable).resolve()) + ' "$@"\n',
            encoding="utf-8",
        )
        runner.chmod(0o755)
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(runner.read_bytes()).hexdigest()
        metadata["paths_data"]["paths"].append(
            {
                "_path": str(runner.relative_to(self.prefix)),
                "path_type": "hardlink",
                "sha256": digest,
                "sha256_in_prefix": digest,
                "size_in_bytes": runner.stat().st_size,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def write_package_archive(
        self,
        softlinks: dict[str, str] | None = None,
        entry_points: dict[str, tuple[str, str]] | None = None,
    ) -> str:
        metadata_path = self.metadata / "pkg-1.0-0.json"
        payload_files = {"lib/libtrusted.so": self.library.read_bytes()}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for record in metadata["paths_data"]["paths"]:
                if record.get("path_type") != "hardlink":
                    continue
                installed = self.prefix / record["_path"]
                if installed.is_file() and not installed.is_symlink():
                    payload_files[record["_path"]] = installed.read_bytes()
        path_records = []
        for path, payload in sorted(payload_files.items()):
            path_records.append(
                {
                    "_path": path,
                    "path_type": "hardlink",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_in_bytes": len(payload),
                }
            )
        for path in sorted((softlinks or {})):
            path_records.append({"_path": path, "path_type": "softlink"})
        index = {"name": "python", "version": "3.11.9", "build": "0"}
        link = None
        if entry_points is not None:
            link = {
                "noarch": {
                    "type": "python",
                    "entry_points": [
                        f"{name} = {module}:{function}"
                        for name, (module, function) in sorted(entry_points.items())
                    ],
                },
                "package_metadata_version": 1,
            }
        with tarfile.open(self.archive, mode="w:bz2") as package:
            for path, payload in sorted(payload_files.items()):
                item = tarfile.TarInfo(path)
                item.size = len(payload)
                item.mode = 0o755
                package.addfile(item, io.BytesIO(payload))
            for path, target in sorted((softlinks or {}).items()):
                alias = tarfile.TarInfo(path)
                alias.type = tarfile.SYMTYPE
                alias.linkname = target
                package.addfile(alias)
            for path, value in (
                ("info/paths.json", {"paths": path_records}),
                ("info/index.json", index),
            ):
                payload = json.dumps(value).encode("utf-8")
                item = tarfile.TarInfo(path)
                item.size = len(payload)
                package.addfile(item, io.BytesIO(payload))
            if link is not None:
                payload = json.dumps(link).encode("utf-8")
                item = tarfile.TarInfo("info/link.json")
                item.size = len(payload)
                package.addfile(item, io.BytesIO(payload))
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(index)
            metadata["sha256"] = digest
            for record in metadata["paths_data"]["paths"]:
                if record.get("path_type") != "hardlink":
                    continue
                payload = payload_files[record["_path"]]
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["sha256_in_prefix"] = record["sha256"]
                record["size_in_bytes"] = len(payload)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        lock = self.root / "environment.lock"
        if lock.exists():
            lock.write_text(f"@EXPLICIT\n{self.url}#{digest}\n", encoding="utf-8")
        self.package_digest = digest
        return digest

    def write_conda_package_archive(
        self, softlinks: dict[str, str], trailing_zeros: int
    ) -> str:
        package_tar = io.BytesIO()
        with tarfile.open(fileobj=package_tar, mode="w") as package:
            payload = self.library.read_bytes()
            library = tarfile.TarInfo("lib/libtrusted.so")
            library.size = len(payload)
            library.mode = 0o755
            package.addfile(library, io.BytesIO(payload))
            for path, target in sorted(softlinks.items()):
                alias = tarfile.TarInfo(path)
                alias.type = tarfile.SYMTYPE
                alias.linkname = target
                package.addfile(alias)
        path_records = [
            {
                "_path": "lib/libtrusted.so",
                "path_type": "hardlink",
                "sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
                "size_in_bytes": self.library.stat().st_size,
            }
        ] + [
            {"_path": path, "path_type": "softlink"}
            for path in sorted(softlinks)
        ]
        info_tar = io.BytesIO()
        with tarfile.open(fileobj=info_tar, mode="w") as info:
            for path, value in (
                ("info/paths.json", {"paths": path_records}),
                (
                    "info/index.json",
                    {"name": "python", "version": "3.11.9", "build": "0"},
                ),
            ):
                payload = json.dumps(value).encode("utf-8")
                item = tarfile.TarInfo(path)
                item.size = len(payload)
                info.addfile(item, io.BytesIO(payload))
        compressed = subprocess.run(
            ["/usr/bin/zstd", "-q", "-c"],
            input=package_tar.getvalue() + b"\0" * trailing_zeros,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        compressed_info = subprocess.run(
            ["/usr/bin/zstd", "-q", "-c"],
            input=info_tar.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        self.archive = self.archive.parent / "pkg-1.0-0.conda"
        self.url = "https://example.invalid/pkg-1.0-0.conda"
        with zipfile.ZipFile(self.archive, mode="w") as container:
            container.writestr("metadata.json", b'{}\n')
            container.writestr("pkg-pkg-1.0-0.tar.zst", compressed)
            container.writestr("info-pkg-1.0-0.tar.zst", compressed_info)
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["url"] = self.url
        metadata["sha256"] = digest
        metadata["package_tarball_full_path"] = str(self.archive)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.lock.write_text(
            f"@EXPLICIT\n{self.url}#{digest}\n", encoding="utf-8"
        )
        self.package_digest = digest
        return digest

    def run_audit(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "/usr/bin/python3",
                str(AUDITOR),
                "--prefix",
                str(self.prefix),
                "--lock",
                str(self.lock),
                "--package-cache",
                str(self.package_cache),
                "--output",
                str(self.output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    def test_exact_package_owned_bytes_pass(self):
        completed = self.run_audit()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(result["pass"])
        self.assertEqual(result["hashed_installed_path_count"], 1)
        self.assertEqual(
            result["archive_snapshot"]["open_flags"],
            ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
        )
        self.assertTrue(
            result["archive_snapshot"][
                "hash_and_parse_same_private_snapshot"
            ]
        )

    def test_archive_swap_between_hash_and_parse_cannot_forge_metadata(self):
        trusted_bytes = self.archive.read_bytes()
        trusted_digest = self.package_digest

        self.library.write_bytes(b"attacker-controlled package bytes\n")
        forged_installed_digest = hashlib.sha256(
            self.library.read_bytes()
        ).hexdigest()
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = metadata["paths_data"]["paths"][0]
        record["sha256"] = forged_installed_digest
        record["sha256_in_prefix"] = forged_installed_digest
        record["size_in_bytes"] = self.library.stat().st_size
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive()
        forged_archive = self.root / "forged-package.tar.bz2"
        forged_archive.write_bytes(self.archive.read_bytes())

        self.archive.write_bytes(trusted_bytes)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["sha256"] = trusted_digest
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.lock.write_text(
            f"@EXPLICIT\n{self.url}#{trusted_digest}\n", encoding="utf-8"
        )

        original_parse = AUDIT_MODULE._package_archive_info
        swapped = False

        def swap_path_only_during_parse(snapshot, archive_name):
            nonlocal swapped
            trusted_path = self.root / "trusted-open-inode.tar.bz2"
            os.replace(self.archive, trusted_path)
            os.replace(forged_archive, self.archive)
            try:
                result = original_parse(snapshot, archive_name)
                swapped = True
                return result
            finally:
                os.replace(self.archive, forged_archive)
                os.replace(trusted_path, self.archive)

        with mock.patch.object(
            AUDIT_MODULE,
            "_package_archive_info",
            side_effect=swap_path_only_during_parse,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "differs from exact archive"
            ):
                AUDIT_MODULE.audit(self.prefix, self.lock, self.package_cache)
        self.assertTrue(swapped)

    def test_package_owned_byte_tamper_fails(self):
        self.library.write_bytes(b"tampered\n")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("package-owned prefix file changed", completed.stderr)

    def test_hardlink_and_mutable_metadata_tamper_cannot_bypass_archive(self):
        self.library.write_bytes(b"attacker-controlled bytes\n")
        forged = hashlib.sha256(self.library.read_bytes()).hexdigest()
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = metadata["paths_data"]["paths"][0]
        record["sha256"] = forged
        record["sha256_in_prefix"] = forged
        record["size_in_bytes"] = self.library.stat().st_size
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("differs from exact archive", completed.stderr)

    def test_hardlink_replaced_by_external_symlink_fails(self):
        external = self.root / "external-identical.so"
        external.write_bytes(self.library.read_bytes())
        self.library.unlink()
        self.library.symlink_to(external)
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a canonical regular file", completed.stderr)

    def test_hash_time_hardlink_to_external_symlink_swap_fails(self):
        external = self.root / "external-identical.so"
        external.write_bytes(self.library.read_bytes())
        original_hash = AUDIT_MODULE._PROVENANCE.sha256_file
        swapped = False
        library_identity = (self.library.stat().st_dev, self.library.stat().st_ino)

        def swap_during_descriptor_hash(path):
            nonlocal swapped
            hashed = pathlib.Path(path).stat()
            if (hashed.st_dev, hashed.st_ino) == library_identity and not swapped:
                swapped = True
                self.library.unlink()
                self.library.symlink_to(external)
            return original_hash(path)

        with mock.patch.object(
            AUDIT_MODULE._PROVENANCE,
            "sha256_file",
            side_effect=swap_during_descriptor_hash,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed while hashing"):
                AUDIT_MODULE.audit(self.prefix, self.lock, self.package_cache)
        self.assertTrue(swapped)

    def test_generated_entry_point_replaced_by_symlink_fails(self):
        entry = self.prefix / "bin" / "tool"
        entry.parent.mkdir()
        entry.write_bytes(
            AUDIT_MODULE._entry_point_bytes(
                self.prefix, "3.11", "pkg.cli", "main"
            )
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {"_path": "bin/tool", "path_type": "unix_python_entry_point"}
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive(entry_points={"tool": ("pkg.cli", "main")})
        external = self.root / "external-tool"
        external.write_bytes(entry.read_bytes())
        entry.unlink()
        entry.symlink_to(external)
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a canonical regular file", completed.stderr)

    def test_generated_entry_point_is_rechecked_at_audit_end(self):
        entry = self.prefix / "bin" / "tool"
        entry.parent.mkdir()
        entry.write_bytes(
            AUDIT_MODULE._entry_point_bytes(
                self.prefix, "3.11", "pkg.cli", "main"
            )
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {"_path": "bin/tool", "path_type": "unix_python_entry_point"}
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive(entry_points={"tool": ("pkg.cli", "main")})
        external = self.root / "external-tool"
        external.write_bytes(entry.read_bytes())
        original_verify = AUDIT_MODULE.StatHashCache.verify
        swapped = False

        def swap_before_terminal_verify(cache, path, expected):
            nonlocal swapped
            if pathlib.Path(path) == entry and not swapped:
                swapped = True
                entry.unlink()
                entry.symlink_to(external)
            return original_verify(cache, path, expected)

        with mock.patch.object(
            AUDIT_MODULE.StatHashCache,
            "verify",
            new=swap_before_terminal_verify,
        ):
            with self.assertRaisesRegex(RuntimeError, "canonical regular"):
                AUDIT_MODULE.audit(self.prefix, self.lock, self.package_cache)
        self.assertTrue(swapped)

    def test_internal_softlink_target_and_topology_are_attested(self):
        alias = self.prefix / "lib" / "libtrusted-alias.so"
        alias.symlink_to("libtrusted.so")
        self.write_package_archive(
            {"lib/libtrusted-alias.so": "libtrusted.so"}
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": "lib/libtrusted-alias.so",
                "path_type": "softlink",
                "sha256_in_prefix": AUDIT_MODULE.EMPTY_SHA256,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = self.run_audit()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["schema_version"], 4)
        self.assertEqual(result["softlink_count"], 1)
        self.assertEqual(result["softlinks"][0]["link_target"], "libtrusted.so")
        self.assertEqual(
            result["softlinks"][0]["target_sha256"],
            hashlib.sha256(self.library.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            result["softlinks"][0]["package_archive_sha256"],
            self.package_digest,
        )

    def test_softlink_topology_must_match_exact_package_archive(self):
        alias = self.prefix / "lib" / "libtrusted-alias.so"
        alias.symlink_to("../lib/libtrusted.so")
        self.write_package_archive(
            {"lib/libtrusted-alias.so": "libtrusted.so"}
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": "lib/libtrusted-alias.so",
                "path_type": "softlink",
                "sha256_in_prefix": AUDIT_MODULE.EMPTY_SHA256,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("differs from its exact archive", completed.stderr)

    def test_softlink_package_archive_tamper_fails(self):
        alias = self.prefix / "lib" / "libtrusted-alias.so"
        alias.symlink_to("libtrusted.so")
        self.write_package_archive(
            {"lib/libtrusted-alias.so": "libtrusted.so"}
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": "lib/libtrusted-alias.so",
                "path_type": "softlink",
                "sha256_in_prefix": AUDIT_MODULE.EMPTY_SHA256,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.archive.open("ab") as package:
            package.write(b"tampered")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exact-lock package archive changed", completed.stderr)

    def test_conda_payload_tail_is_drained_before_zstd_wait(self):
        alias = self.prefix / "lib" / "libtrusted-alias.so"
        alias.symlink_to("libtrusted.so")
        self.write_conda_package_archive(
            {"lib/libtrusted-alias.so": "libtrusted.so"},
            trailing_zeros=4 * 1024 * 1024,
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": "lib/libtrusted-alias.so",
                "path_type": "softlink",
                "sha256_in_prefix": AUDIT_MODULE.EMPTY_SHA256,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = self.run_audit()
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_conda_decoder_timeout_is_bounded_and_cleans_temporaries(self):
        self.write_conda_package_archive({}, trailing_zeros=0)
        original_run = AUDIT_MODULE.subprocess.run

        def timeout_decoder(command, *args, **kwargs):
            if command[0] == "/usr/bin/zstd" and "-d" in command:
                raise subprocess.TimeoutExpired(command, 120)
            return original_run(command, *args, **kwargs)

        with mock.patch.object(
            AUDIT_MODULE.subprocess, "run", side_effect=timeout_decoder
        ):
            with AUDIT_MODULE._open_exact_archive(self.archive) as opened:
                with opened.trusted_snapshot(self.package_digest) as snapshot:
                    with self.assertRaisesRegex(
                        RuntimeError, "exceeded 120 seconds"
                    ):
                        AUDIT_MODULE._package_archive_info(
                            snapshot, self.archive.name
                        )

    def test_binary_prefix_relocation_pads_after_the_complete_c_string(self):
        placeholder = b"/opt/build/" + b"placeholder_" * 12
        prefix_bytes = str(self.prefix).encode("utf-8")
        self.assertLess(len(prefix_bytes), len(placeholder))
        raw = (
            b"ELF\0"
            + placeholder
            + b"/etc/config\0tail\0"
            + placeholder
            + b"/lib\0"
        )
        record = {
            "prefix_placeholder": placeholder.decode("ascii"),
            "file_mode": "binary",
        }
        raw_digest, installed_digest, size = AUDIT_MODULE._hash_relocated_payload(
            io.BytesIO(raw), record, self.prefix
        )
        padding = b"\0" * (len(placeholder) - len(prefix_bytes))
        expected = (
            b"ELF\0"
            + prefix_bytes
            + b"/etc/config"
            + padding
            + b"\0tail\0"
            + prefix_bytes
            + b"/lib"
            + padding
            + b"\0"
        )
        self.assertEqual(raw_digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(installed_digest, hashlib.sha256(expected).hexdigest())
        self.assertEqual(size, len(raw))

    def test_absolute_softlink_is_rejected_even_when_target_is_inside_prefix(self):
        alias = self.prefix / "lib" / "libtrusted-alias.so"
        alias.symlink_to(self.library)
        self.write_package_archive(
            {"lib/libtrusted-alias.so": "libtrusted.so"}
        )
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": "lib/libtrusted-alias.so",
                "path_type": "softlink",
                "sha256_in_prefix": hashlib.sha256(
                    self.library.read_bytes()
                ).hexdigest(),
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("package-owned symlink is invalid", completed.stderr)

    def test_unowned_runtime_file_fails(self):
        (self.prefix / "etc").mkdir()
        (self.prefix / "etc" / "ucx.conf").write_text("UCX_TLS=tcp\n")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unowned files appeared", completed.stderr)

    def test_unowned_bytecode_fails(self):
        bytecode = (
            self.prefix
            / "lib"
            / "python3.11"
            / "site-packages"
            / "rogue"
            / "__pycache__"
            / "payload.cpython-311.pyc"
        )
        bytecode.parent.mkdir(parents=True)
        bytecode.write_bytes(b"untrusted bytecode")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unowned files appeared", completed.stderr)

    def test_absent_generated_bytecode_is_safe_and_counted(self):
        self.add_prefix_python_runner()
        source = self.prefix / "lib/python3.11/site-packages/pkg/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="utf-8")
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["paths_data"]["paths"].append(
            {
                "_path": str(source.relative_to(self.prefix)),
                "path_type": "hardlink",
                "sha256": source_digest,
                "sha256_in_prefix": source_digest,
                "size_in_bytes": source.stat().st_size,
            }
        )
        metadata["paths_data"]["paths"].append(
            {
                "_path": (
                    "lib/python3.11/site-packages/pkg/__pycache__/"
                    "module.cpython-311.pyc"
                ),
                "path_type": "pyc_file",
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive()
        completed = self.run_audit()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["declared_generated_bytecode_path_count"], 1)
        self.assertEqual(result["generated_source_bytecode_count"], 1)
        self.assertEqual(
            result["generated_source_bytecode"][0]["current_state"], "absent"
        )

    def test_present_unhashed_generated_bytecode_fails(self):
        self.add_prefix_python_runner()
        source = self.prefix / "lib/python3.11/site-packages/pkg/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="utf-8")
        bytecode = (
            self.prefix
            / "lib"
            / "python3.11"
            / "site-packages"
            / "pkg"
            / "__pycache__"
            / "module.cpython-311.pyc"
        )
        bytecode.parent.mkdir(parents=True)
        bytecode.write_bytes(b"unverifiable bytecode")
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["paths_data"]["paths"].append(
            {
                "_path": str(source.relative_to(self.prefix)),
                "path_type": "hardlink",
                "sha256": source_digest,
                "sha256_in_prefix": source_digest,
                "size_in_bytes": source.stat().st_size,
            }
        )
        metadata["paths_data"]["paths"].append(
            {"_path": str(bytecode.relative_to(self.prefix)), "path_type": "pyc_file"}
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive()
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("generated source bytecode is unverifiable", completed.stderr)

    def test_checked_hash_generated_bytecode_is_source_derived(self):
        self.add_prefix_python_runner()
        source = self.prefix / "lib/python3.11/site-packages/pkg/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="utf-8")
        bytecode = source.parent / "__pycache__/module.cpython-311.pyc"
        bytecode.parent.mkdir()
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["paths_data"]["paths"].extend(
            [
                {
                    "_path": str(source.relative_to(self.prefix)),
                    "path_type": "hardlink",
                    "sha256": source_digest,
                    "sha256_in_prefix": source_digest,
                    "size_in_bytes": source.stat().st_size,
                },
                {
                    "_path": str(bytecode.relative_to(self.prefix)),
                    "path_type": "pyc_file",
                },
            ]
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive()
        py_compile.compile(
            str(source),
            cfile=str(bytecode),
            dfile=str(source),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )
        completed = self.run_audit()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(
            result["generated_source_bytecode"][0]["current_state"],
            "source-compiled",
        )

    def test_hashed_package_bytecode_is_verified(self):
        bytecode = (
            self.prefix
            / "lib"
            / "python3.11"
            / "site-packages"
            / "pkg"
            / "__pycache__"
            / "module.cpython-311.pyc"
        )
        bytecode.parent.mkdir(parents=True)
        bytecode.write_bytes(b"package bytecode")
        metadata_path = self.metadata / "pkg-1.0-0.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["paths_data"]["paths"].append(
            {
                "_path": str(bytecode.relative_to(self.prefix)),
                "path_type": "hardlink",
                "sha256": hashlib.sha256(bytecode.read_bytes()).hexdigest(),
                "sha256_in_prefix": hashlib.sha256(bytecode.read_bytes()).hexdigest(),
                "size_in_bytes": bytecode.stat().st_size,
            }
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_package_archive()
        self.assertEqual(self.run_audit().returncode, 0)
        bytecode.write_bytes(b"tampered bytecode")
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("package-owned prefix file changed", completed.stderr)

    def test_relocated_source_hashed_bytecode_accepts_only_archive_or_derivation(self):
        source = self.prefix / "lib/python3.11/site-packages/pkg/config.py"
        bytecode = (
            self.prefix
            / "lib/python3.11/site-packages/pkg/__pycache__/config.cpython-311.pyc"
        )
        bytecode.parent.mkdir(parents=True)
        placeholder = "/opt/gpmeep-build-prefix-" + "x" * 120
        raw_source = f'PREFIX = "{placeholder}"\n'.encode()
        installed_source = raw_source.replace(
            placeholder.encode(), str(self.prefix).encode()
        )
        source.write_bytes(installed_source)
        archive_pyc = b"archive checked-hash bytecode"
        derived_pyc = b"installed-source checked-hash bytecode"
        bytecode.write_bytes(archive_pyc)
        payloads = {
            "lib/libtrusted.so": self.library.read_bytes(),
            str(source.relative_to(self.prefix)): raw_source,
            str(bytecode.relative_to(self.prefix)): archive_pyc,
        }
        path_records = []
        for name, payload in payloads.items():
            record = {
                "_path": name,
                "path_type": "hardlink",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_in_bytes": len(payload),
            }
            if name == str(source.relative_to(self.prefix)):
                record.update(
                    {
                        "prefix_placeholder": placeholder,
                        "file_mode": "text",
                    }
                )
            path_records.append(record)
        with tarfile.open(self.archive, mode="w:bz2") as package:
            for name, payload in payloads.items():
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                package.addfile(item, io.BytesIO(payload))
            for name, value in (
                ("info/paths.json", {"paths": path_records}),
                (
                    "info/index.json",
                    {"name": "python", "version": "3.11.9", "build": "0"},
                ),
            ):
                payload = json.dumps(value).encode()
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                package.addfile(item, io.BytesIO(payload))
        package_digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        metadata = {
            "name": "python",
            "version": "3.11.9",
            "build": "0",
            "url": self.url,
            "sha256": package_digest,
            "paths_data": {
                "paths": [
                    {
                        "_path": "lib/libtrusted.so",
                        "path_type": "hardlink",
                        "sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
                        "sha256_in_prefix": hashlib.sha256(
                            self.library.read_bytes()
                        ).hexdigest(),
                        "size_in_bytes": self.library.stat().st_size,
                    },
                    {
                        "_path": str(source.relative_to(self.prefix)),
                        "path_type": "hardlink",
                        "sha256": hashlib.sha256(raw_source).hexdigest(),
                        "sha256_in_prefix": hashlib.sha256(installed_source).hexdigest(),
                        "size_in_bytes": len(raw_source),
                    },
                    {
                        "_path": str(bytecode.relative_to(self.prefix)),
                        "path_type": "hardlink",
                        "sha256": hashlib.sha256(archive_pyc).hexdigest(),
                        "sha256_in_prefix": hashlib.sha256(archive_pyc).hexdigest(),
                        "size_in_bytes": len(archive_pyc),
                    },
                ]
            },
        }
        (self.metadata / "pkg-1.0-0.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        self.lock.write_text(
            f"@EXPLICIT\n{self.url}#{package_digest}\n", encoding="utf-8"
        )

        relative = str(bytecode.relative_to(self.prefix))
        compiled = {
            relative: {
                "cache_tag": "cpython-311",
                "path": relative,
                "pyc_sha256": hashlib.sha256(derived_pyc).hexdigest(),
                "pyc_size_bytes": len(derived_pyc),
                "source_hash_hex": "12" * 8,
                "source_path": str(source.relative_to(self.prefix)),
                "source_sha256": hashlib.sha256(installed_source).hexdigest(),
            }
        }
        with mock.patch.object(
            AUDIT_MODULE,
            "_compile_relocated_source_bytecode",
            return_value=compiled,
        ):
            archive_result = AUDIT_MODULE.audit(
                self.prefix, self.lock, self.package_cache
            )
            self.assertEqual(
                archive_result["relocated_source_bytecode"][0]["current_state"],
                "archive-exact",
            )
            bytecode.write_bytes(derived_pyc)
            derived_result = AUDIT_MODULE.audit(
                self.prefix, self.lock, self.package_cache
            )
            self.assertEqual(
                derived_result["relocated_source_bytecode"][0]["current_state"],
                "relocated-source-compiled",
            )
            bytecode.write_bytes(b"forged")
            with self.assertRaisesRegex(RuntimeError, "bytecode changed"):
                AUDIT_MODULE.audit(self.prefix, self.lock, self.package_cache)

    def test_micromamba_state_environment_injection_fails(self):
        (self.metadata / "state").write_text(
            json.dumps({"env_vars": {"CFLAGS": "-DGPMEEP_POISON=1"}}),
            encoding="utf-8",
        )
        completed = self.run_audit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("conda-meta/state", completed.stderr)


if __name__ == "__main__":
    unittest.main()
