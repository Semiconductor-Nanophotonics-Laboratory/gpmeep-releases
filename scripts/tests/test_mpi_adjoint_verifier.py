from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import platform
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def load_verifier():
    path = SCRIPTS / "verify-mpi-adjoint-evidence.py"
    spec = importlib.util.spec_from_file_location("gpmeep_mpi_adjoint_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import MPI adjoint evidence verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MpiAdjointVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()

    def test_help_is_a_successful_argparse_exit(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                self.verifier.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("Replay a COMPLETE gpmeep", stdout.getvalue())
        self.assertNotIn("verification failed", stderr.getvalue())

    def test_ordinary_verification_failure_is_reported(self):
        stderr = io.StringIO()
        with mock.patch.object(
            self.verifier, "_parse_args", return_value=object()
        ), mock.patch.object(
            self.verifier, "verify", side_effect=RuntimeError("synthetic failure")
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(self.verifier.main([]), 1)
        self.assertIn("synthetic failure", stderr.getvalue())

    def test_internal_system_exit_cannot_report_success(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            self.verifier, "_parse_args", return_value=object()
        ), mock.patch.object(
            self.verifier, "verify", side_effect=SystemExit(0)
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(self.verifier.main([]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("verification failed", stderr.getvalue())

    def test_postprocessor_platform_is_bound_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            interpreter = pathlib.Path(sys.executable).resolve(strict=True)
            runner = types.SimpleNamespace(
                RUNNER_SOURCES=(),
                git_output=lambda _repo, *args: (
                    "synthetic-head\n" if args == ("rev-parse", "HEAD") else ""
                ),
            )
            postprocessor = {
                "git_head": "synthetic-head",
                "git_status_porcelain": [],
                "source_snapshot": {},
                "source_sha256": {},
                "python": {
                    "path": str(interpreter),
                    "sha256": self.verifier._sha256(interpreter),
                    "version": sys.version,
                },
                "platform": platform.platform() + "-forged",
                "nvidia_driver": {},
                "nvidia_topology": {},
                "driver_device_binding": {},
                "command": [],
            }
            with self.assertRaisesRegex(RuntimeError, "source binding changed"):
                self.verifier._validate_postprocessor(
                    runner,
                    postprocessor,
                    repo=root,
                    receipt_path=root / "receipt.json",
                    receipt={"source_end": {}},
                    output=root,
                    interpreter=interpreter,
                    comparison={},
                )

    def test_recorded_command_help_cannot_escape_with_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            interpreter = pathlib.Path(sys.executable).resolve(strict=True)

            def escaped_parse_args(_argv):
                raise SystemExit(0)

            runner = types.SimpleNamespace(
                RUNNER_SOURCES=(),
                git_output=lambda _repo, *args: (
                    "synthetic-head\n" if args == ("rev-parse", "HEAD") else ""
                ),
                parse_args=escaped_parse_args,
            )
            postprocessor = {
                "git_head": "synthetic-head",
                "git_status_porcelain": [],
                "source_snapshot": {},
                "source_sha256": {},
                "python": {
                    "path": str(interpreter),
                    "sha256": self.verifier._sha256(interpreter),
                    "version": sys.version,
                },
                "platform": platform.platform(),
                "nvidia_driver": {},
                "nvidia_topology": {},
                "driver_device_binding": {},
                "command": [
                    str(interpreter),
                    str(root / "scripts" / "run-mpi-adjoint-benchmark.py"),
                    "--help",
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "escaped verification"):
                self.verifier._validate_postprocessor(
                    runner,
                    postprocessor,
                    repo=root,
                    receipt_path=root / "receipt.json",
                    receipt={"source_end": {}},
                    output=root,
                    interpreter=interpreter,
                    comparison={},
                )


if __name__ == "__main__":
    unittest.main()
