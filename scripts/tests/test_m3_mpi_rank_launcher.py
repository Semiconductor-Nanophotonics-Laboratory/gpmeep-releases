#!/usr/bin/env python3
"""Tests for the collision-free M3 MPI rank launcher."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "m3_mpi_rank_launcher.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_mpi_rank_launcher", SOURCE)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


def mpi_environment(rank: str = "1", size: str = "2", local: str = "1") -> dict[str, str]:
    return {
        "OMPI_COMM_WORLD_RANK": rank,
        "OMPI_COMM_WORLD_SIZE": size,
        "OMPI_COMM_WORLD_LOCAL_RANK": local,
    }


class M3MpiRankLauncherTests(unittest.TestCase):
    def make_root(self, parent: pathlib.Path) -> pathlib.Path:
        root = parent / "evidence"
        (root / "statistics").mkdir(parents=True)
        (root / "identity").mkdir()
        (root / "work").mkdir()
        return root

    def test_assigns_private_rank_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.make_root(pathlib.Path(raw))
            value = LAUNCHER.derive_rank_paths(root, 2, mpi_environment())
            self.assertEqual(value.rank, 1)
            self.assertEqual(value.world_size, 2)
            self.assertEqual(value.local_rank, 1)
            self.assertEqual(value.statistics.name, "rank-00001.json")
            self.assertEqual(value.identity.name, "rank-00001.json")
            self.assertEqual(value.work.name, "rank-00001")
            self.assertTrue(value.work.is_dir())
            self.assertFalse(value.statistics.exists())
            self.assertEqual(value.home.parent, value.work)
            self.assertTrue(value.temporary.is_dir())

    def test_rejects_missing_noncanonical_and_wrong_world_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.make_root(pathlib.Path(raw))
            for environment, pattern in (
                ({}, "MPI rank"),
                (mpi_environment(rank="01"), "canonical"),
                (mpi_environment(size="3"), "topology differs"),
                (mpi_environment(rank="2"), "topology differs"),
            ):
                with self.subTest(environment=environment):
                    with self.assertRaisesRegex(Exception, pattern):
                        LAUNCHER.derive_rank_paths(root, 2, environment)

    def test_rejects_symlink_roots_and_preexisting_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = pathlib.Path(raw)
            root = self.make_root(parent)
            (root / "statistics" / "rank-00000.json").write_text("stale")
            with self.assertRaisesRegex(Exception, "already exists"):
                LAUNCHER.derive_rank_paths(root, 1, mpi_environment("0", "1", "0"))

            target = parent / "target"
            (target / "statistics").mkdir(parents=True)
            (target / "identity").mkdir()
            (target / "work").mkdir()
            link = parent / "linked"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Exception, "non-symlink directory"):
                LAUNCHER.derive_rank_paths(link, 1, mpi_environment("0", "1", "0"))

    def test_rejects_second_claim_for_same_rank(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.make_root(pathlib.Path(raw))
            LAUNCHER.derive_rank_paths(root, 1, mpi_environment("0", "1", "0"))
            with self.assertRaisesRegex(Exception, "work target already exists"):
                LAUNCHER.derive_rank_paths(root, 1, mpi_environment("0", "1", "0"))

    def test_command_delimiter_is_removed(self) -> None:
        args = LAUNCHER.parse_args(
            ["--evidence-root", "/tmp/e", "--expected-ranks", "2", "--", "/bin/true"]
        )
        self.assertEqual(args.command, ["/bin/true"])

    def test_main_publishes_pid_bound_identity_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.make_root(pathlib.Path(raw))
            environment = {
                **os.environ,
                **mpi_environment("0", "1", "0"),
                "GPMEEP_VALIDATION_RUN_NONCE": "abc123",
                "GPMEEP_VALIDATION_EXPECTED_BACKEND": "cuda",
                "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "auto",
                "CUDA_VISIBLE_DEVICES": "GPU-example",
            }
            captured: dict[str, object] = {}

            def fake_exec(executable: str, command: list[str], env: dict[str, str]) -> None:
                captured.update(executable=executable, command=command, environment=env)
                raise RuntimeError("exec intercepted")

            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                LAUNCHER.os, "execve", side_effect=fake_exec
            ), mock.patch.object(LAUNCHER.os, "chdir"):
                with self.assertRaisesRegex(RuntimeError, "exec intercepted"):
                    LAUNCHER.main(
                        [
                            "--evidence-root",
                            str(root),
                            "--expected-ranks",
                            "1",
                            "--",
                            "/usr/bin/bash",
                        ]
                    )
            identity = json.loads(
                (root / "identity" / "rank-00000.json").read_text()
            )
            self.assertEqual(identity["pid"], os.getpid())
            self.assertEqual(identity["run_nonce"], "abc123")
            self.assertEqual(identity["requested_backend"], "auto")
            self.assertEqual(identity["visible_devices"], "GPU-example")
            self.assertEqual(captured["command"], ["/usr/bin/bash"])
            self.assertEqual(
                captured["environment"]["GPMEEP_VALIDATION_STATS_FILE"],
                str(root / "statistics" / "rank-00000.json"),
            )


if __name__ == "__main__":
    unittest.main()
