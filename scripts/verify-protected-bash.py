#!/usr/bin/env python3
"""Verify that the parent Bash process started in protected mode."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=pathlib.Path, required=True)
    parser.add_argument("--stage", choices=("bootstrap", "audited-worker"), required=True)
    args = parser.parse_args()
    parent = os.getppid()
    proc = pathlib.Path("/proc") / str(parent)
    try:
        executable = (proc / "exe").resolve(strict=True)
        raw = (proc / "cmdline").read_bytes()
    except OSError as error:
        raise RuntimeError("cannot inspect the authoritative Bash parent") from error
    trusted_bash = {
        path.resolve()
        for path in (pathlib.Path("/bin/bash"), pathlib.Path("/usr/bin/bash"))
        if path.is_file()
    }
    argv = [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]
    script = args.script.resolve(strict=True)
    expected_tail = [] if args.stage == "bootstrap" else ["audited-worker"]
    if (
        executable not in trusted_bash
        or len(argv) != 3 + len(expected_tail)
        or argv[1] != "-p"
        or pathlib.Path(argv[2]).resolve() != script
        or argv[3:] != expected_tail
    ):
        raise RuntimeError("authoritative Bash was not started with protected argv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
