#!/usr/bin/env python3
"""Focused process-lifecycle tests for the observed bounded runner."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD_DIR = ROOT / "scripts" / "user-workloads"
sys.path.insert(0, str(WORKLOAD_DIR))


def load_module(name: str):
    path = WORKLOAD_DIR / name
    spec = importlib.util.spec_from_file_location(f"gpmeep_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("run_observed_process.py")


OBSERVER = """
import pathlib
import sys
import time

pid = int(sys.argv[1])
mode = sys.argv[2]
if mode == "fail":
    raise SystemExit(7)
if mode == "early":
    raise SystemExit(0)
deadline = time.monotonic() + 10
while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
    time.sleep(0.02)
print("observer-finished", flush=True)
"""


class ObservedRunnerTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        observer = root / "observer.py"
        observer.write_text(OBSERVER, encoding="utf-8")
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return root, observer, environment

    def execute(self, root, observer, environment, command, mode="wait", **limits):
        observed_pid = []

        def factory(pid):
            observed_pid.append(pid)
            return [
                RUNNER.ObserverSpec(
                    "fixture",
                    (sys.executable, str(observer), str(pid), mode),
                    root / "observer.log",
                )
            ]

        result = RUNNER.run_bounded_observed(
            command,
            environment,
            root,
            root / "workload.log",
            limits.get("timeout_seconds", 5.0),
            limits.get("output_limit_bytes", 1024 * 1024),
            factory,
            observer_exit_timeout_seconds=2.0,
        )
        return result, observed_pid

    def test_success_retains_canonical_process_and_observer_records(self) -> None:
        root, observer, environment = self.fixture()
        (process, observers), observed_pid = self.execute(
            root,
            observer,
            environment,
            [sys.executable, "-c", "import time; print('ok'); time.sleep(.2)"],
        )
        self.assertEqual(process["returncode"], 0)
        self.assertEqual(process["command_pid"], observed_pid[0])
        self.assertEqual(process["log"]["path"], "workload.log")
        self.assertEqual(observers[0]["returncode"], 0)
        self.assertEqual(observers[0]["log"]["path"], "observer.log")
        self.assertIn("observer-finished", (root / "observer.log").read_text())
        expected = RUNNER.ObserverSpec(
            "fixture",
            (
                sys.executable,
                str(observer),
                str(observed_pid[0]),
                "wait",
            ),
            root / "observer.log",
        )
        self.assertEqual(
            RUNNER.verify_observer_record(root, observers[0], expected),
            observers[0],
        )
        mutated = dict(observers[0])
        mutated["returncode"] = 1
        with self.assertRaisesRegex(Exception, "outcome"):
            RUNNER.verify_observer_record(root, mutated, expected)

    def test_timeout_kills_workload_and_observer_finishes(self) -> None:
        root, observer, environment = self.fixture()
        (process, observers), _ = self.execute(
            root,
            observer,
            environment,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=0.1,
        )
        self.assertTrue(process["timed_out"])
        self.assertNotEqual(process["returncode"], 0)
        self.assertEqual(observers[0]["returncode"], 0)

    def test_output_limit_kills_workload(self) -> None:
        root, observer, environment = self.fixture()
        (process, _observers), _ = self.execute(
            root,
            observer,
            environment,
            [sys.executable, "-c", "print('x' * 10000)"],
            output_limit_bytes=100,
        )
        self.assertTrue(process["output_limited"])
        self.assertEqual(process["output_bytes"], 100)

    def test_observer_output_limit_kills_workload(self) -> None:
        root, observer, environment = self.fixture()
        noisy = root / "noisy.py"
        noisy.write_text(
            "import sys,time\nprint('x'*10000, flush=True)\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        captured_pid = []

        def factory(pid):
            captured_pid.append(pid)
            return [
                RUNNER.ObserverSpec(
                    "noisy",
                    (sys.executable, str(noisy)),
                    root / "observer.log",
                )
            ]

        with self.assertRaisesRegex(Exception, "output limit"):
            RUNNER.run_bounded_observed(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                environment,
                root,
                root / "workload.log",
                5.0,
                1024,
                factory,
                observer_exit_timeout_seconds=2.0,
                observer_output_limit_bytes=100,
            )
        self.assertFalse(RUNNER._proc_group_alive(captured_pid[0]))

    def test_failed_or_early_observer_kills_workload(self) -> None:
        for mode in ("fail", "early"):
            with self.subTest(mode=mode):
                root, observer, environment = self.fixture()
                captured_pid = []

                def factory(pid):
                    captured_pid.append(pid)
                    return [
                        RUNNER.ObserverSpec(
                            "fixture",
                            (sys.executable, str(observer), str(pid), mode),
                            root / "observer.log",
                        )
                    ]

                with self.assertRaisesRegex(
                    Exception, "observer fixture exited early"
                ):
                    RUNNER.run_bounded_observed(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        environment,
                        root,
                        root / "workload.log",
                        5.0,
                        1024,
                        factory,
                        observer_exit_timeout_seconds=2.0,
                    )
                self.assertFalse(RUNNER._proc_group_alive(captured_pid[0]))

    def test_log_escape_and_duplicate_observer_are_rejected(self) -> None:
        root, observer, environment = self.fixture()
        with self.assertRaisesRegex(Exception, "escapes"):
            RUNNER.run_bounded_observed(
                [sys.executable, "-c", "print('unused')"],
                environment,
                root,
                root.parent / "escape.log",
                1.0,
                1024,
                lambda _pid: [],
            )

        def duplicates(pid):
            return [
                RUNNER.ObserverSpec(
                    "same",
                    (sys.executable, str(observer), str(pid), "wait"),
                    root / "one.log",
                ),
                RUNNER.ObserverSpec(
                    "same",
                    (sys.executable, str(observer), str(pid), "wait"),
                    root / "two.log",
                ),
            ]

        with self.assertRaisesRegex(Exception, "duplicate"):
            RUNNER.run_bounded_observed(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                environment,
                root,
                root / "workload.log",
                1.0,
                1024,
                duplicates,
            )


if __name__ == "__main__":
    unittest.main()
