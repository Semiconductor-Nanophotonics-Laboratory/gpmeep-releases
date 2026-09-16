#!/usr/bin/env python3
"""Exec one M3 MPI Python rank with collision-free evidence paths.

Open MPI starts the same command on every rank.  The normal Python validation
hook accepts one statistics pathname, so passing a shared pathname would let
the last exiting rank replace every other rank's evidence.  This launcher is
the deliberately small boundary between ``mpiexec`` and the real case command:
it validates the MPI identity supplied by the launcher, assigns a private work
directory and statistics file to that rank, and then ``execve`` replaces itself
with the receipt-bound Python command.

The parent controller owns creation and sealing of the evidence root.  This
program never creates the root and refuses pre-existing per-rank outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys
from dataclasses import dataclass


SCHEMA = "gpmeep-m3-mpi-rank-launcher-v1"
MAX_RANKS = 4096


class LauncherError(RuntimeError):
    """Raised when the MPI rank launch contract is not exact."""


@dataclass(frozen=True)
class RankPaths:
    rank: int
    world_size: int
    local_rank: int
    statistics: pathlib.Path
    identity: pathlib.Path
    work: pathlib.Path
    home: pathlib.Path
    cache: pathlib.Path
    config: pathlib.Path
    matplotlib: pathlib.Path
    temporary: pathlib.Path


def _canonical_nonnegative(value: str | None, name: str) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdigit()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise LauncherError(f"{name} is not a canonical nonnegative integer")
    return int(value)


def _require_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(path))
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LauncherError(f"{label} is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise LauncherError(f"{label} is not a canonical non-symlink directory")
    return path


def derive_rank_paths(
    evidence_root: pathlib.Path,
    expected_ranks: int,
    environment: dict[str, str],
) -> RankPaths:
    if (
        isinstance(expected_ranks, bool)
        or not isinstance(expected_ranks, int)
        or not 1 <= expected_ranks <= MAX_RANKS
    ):
        raise LauncherError(f"expected ranks must be in [1,{MAX_RANKS}]")
    rank = _canonical_nonnegative(environment.get("OMPI_COMM_WORLD_RANK"), "MPI rank")
    world = _canonical_nonnegative(
        environment.get("OMPI_COMM_WORLD_SIZE"), "MPI world size"
    )
    local_rank = _canonical_nonnegative(
        environment.get("OMPI_COMM_WORLD_LOCAL_RANK"), "MPI local rank"
    )
    if world != expected_ranks or rank >= world or local_rank >= world:
        raise LauncherError("MPI rank topology differs from the sealed request")

    root = _require_directory(evidence_root, "MPI evidence root")
    stats_root = _require_directory(root / "statistics", "MPI statistics root")
    identity_root = _require_directory(root / "identity", "MPI identity root")
    work_root = _require_directory(root / "work", "MPI work root")
    rank_name = f"rank-{rank:05d}"
    statistics = stats_root / f"{rank_name}.json"
    identity = identity_root / f"{rank_name}.json"
    work = work_root / rank_name
    if statistics.exists() or statistics.is_symlink():
        raise LauncherError(f"rank statistics target already exists: {statistics}")
    if identity.exists() or identity.is_symlink():
        raise LauncherError(f"rank identity target already exists: {identity}")
    if work.exists() or work.is_symlink():
        raise LauncherError(f"rank work target already exists: {work}")
    work.mkdir(mode=0o700)
    if work.resolve(strict=True).parent != work_root:
        raise LauncherError("rank work directory escaped its evidence root")
    runtime = {}
    for name in ("home", "cache", "config", "matplotlib", "tmp"):
        path = work / name
        path.mkdir(mode=0o700)
        runtime[name] = path
    return RankPaths(
        rank,
        world,
        local_rank,
        statistics,
        identity,
        work,
        runtime["home"],
        runtime["cache"],
        runtime["config"],
        runtime["matplotlib"],
        runtime["tmp"],
    )


def _absolute_executable(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise LauncherError("rank command executable must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LauncherError(f"rank command executable is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.resolve(strict=True) != path
        or not os.access(path, os.X_OK)
    ):
        raise LauncherError("rank command executable is not a canonical executable file")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    executable = _absolute_executable(args.command[0])
    paths = derive_rank_paths(args.evidence_root, args.expected_ranks, dict(os.environ))
    environment = dict(os.environ)
    environment.update(
        {
            "GPMEEP_VALIDATION_STATS_FILE": str(paths.statistics),
            "GPMEEP_M3_MPI_LAUNCHER_SCHEMA": SCHEMA,
            "GPMEEP_M3_MPI_RANK": str(paths.rank),
            "GPMEEP_M3_MPI_WORLD_SIZE": str(paths.world_size),
            "GPMEEP_M3_MPI_LOCAL_RANK": str(paths.local_rank),
            "HOME": str(paths.home),
            "XDG_CACHE_HOME": str(paths.cache),
            "XDG_CONFIG_HOME": str(paths.config),
            "MPLCONFIGDIR": str(paths.matplotlib),
            "TMPDIR": str(paths.temporary),
        }
    )
    identity = {
        "schema": SCHEMA,
        "rank": paths.rank,
        "world_size": paths.world_size,
        "local_rank": paths.local_rank,
        "pid": os.getpid(),
        "executable": str(executable),
        "command": list(args.command),
        "run_nonce": environment.get("GPMEEP_VALIDATION_RUN_NONCE"),
        "requested_backend": environment.get(
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND",
            environment.get("GPMEEP_VALIDATION_EXPECTED_BACKEND"),
        ),
        "visible_devices": environment.get("CUDA_VISIBLE_DEVICES", ""),
        "statistics": str(paths.statistics),
        "work": str(paths.work),
    }
    payload = (
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        paths.identity,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise LauncherError("rank identity write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chdir(paths.work)
    os.execve(str(executable), args.command, environment)
    raise AssertionError("os.execve returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, OSError, ValueError) as error:
        print(f"M3 MPI rank launcher error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
