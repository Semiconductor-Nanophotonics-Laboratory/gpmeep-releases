from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "test-dft-workload-preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_dft_workload_preflight", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class MarkerTests(unittest.TestCase):
    def test_exact_marker_is_accepted(self) -> None:
        expected = {"components": 2, "frequencies": 32, "monitors": 8}
        output = PREFLIGHT.PREFIX + (
            '{"components":2,"frequencies":32,"monitors":8}\n'
        )
        self.assertEqual(PREFLIGHT.parse_success_marker(output, expected), expected)

    def test_duplicate_key_is_rejected(self) -> None:
        output = PREFLIGHT.PREFIX + (
            '{"components":2,"frequencies":32,"monitors":8,"monitors":8}\n'
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            PREFLIGHT.parse_success_marker(
                output, {"components": 2, "frequencies": 32, "monitors": 8}
            )

    def test_extra_gpmeep_marker_is_rejected(self) -> None:
        output = (
            PREFLIGHT.PREFIX
            + '{"components":2,"frequencies":2,"monitors":64}\n'
            + "gpmeep-unexpected-v1:{}\n"
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected gpmeep"):
            PREFLIGHT.parse_success_marker(
                output, {"components": 2, "frequencies": 2, "monitors": 64}
            )

    def test_controlled_environment_has_no_inherited_gpu_controls(self) -> None:
        environment = PREFLIGHT.controlled_environment(
            pathlib.Path("/tmp/isolated-home"), pathlib.Path("/opt/prefix")
        )
        self.assertFalse(any(key.startswith("MEEP_GPU") for key in environment))


if __name__ == "__main__":
    unittest.main()
