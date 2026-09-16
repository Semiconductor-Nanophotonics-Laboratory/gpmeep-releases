#!/usr/bin/env python3
"""Capture the complete exported environment used for build qualification.

The build is intentionally sensitive to more variables than Autoconf documents
in one place (for example ``NVCC_PREPEND_FLAGS``, ``MAKEFILES``, and
``ACLOCAL_PATH``).  Recording a selected list would therefore create a bypass.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import types


def _load_provenance_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name("gpmeep_provenance.py")
    module = types.ModuleType("gpmeep_provenance")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


_PROVENANCE = _load_provenance_from_source()
NORMALIZED_SHELL_KEYS = ("SHLVL", "_")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        environment = dict(sorted(os.environ.items()))
        for name in NORMALIZED_SHELL_KEYS:
            environment[name] = "<shell-managed>"
        _PROVENANCE.atomic_write_json(
            args.output.resolve(),
            {
                "schema_version": 2,
                "environment": dict(sorted(environment.items())),
                "normalized_shell_keys": list(NORMALIZED_SHELL_KEYS),
            },
        )
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
