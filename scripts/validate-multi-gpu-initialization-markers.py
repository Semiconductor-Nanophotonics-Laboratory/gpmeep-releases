#!/usr/bin/env python3
"""Strictly validate one record-emitting gpu-mpi-performance log."""

from __future__ import annotations

import argparse
import pathlib
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import multi_gpu_benchmark as MGB  # noqa: E402


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument(
        "--expected-profile",
        choices=("trigonometric-v1", "affine-v1"),
        required=True,
    )
    parser.add_argument(
        "--expected-source-profile",
        choices=("single-ez-v1", "dual-electric-v1"),
        required=True,
    )
    parser.add_argument(
        "--expected-applications-per-rank", type=positive_int, required=True
    )
    parser.add_argument("--expected-ranks", type=positive_int, required=True)
    parser.add_argument("--expected-pixels", type=positive_int, required=True)
    parser.add_argument(
        "--expected-warmup-steps", type=positive_int, required=True
    )
    parser.add_argument("--expected-steps", type=positive_int, required=True)
    parser.add_argument(
        "--expected-transport",
        choices=("pinned", "cuda-aware"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.log.is_symlink() or not args.log.is_file():
        raise RuntimeError("worker log must be a regular non-symlink file")
    output = args.log.read_text(encoding="utf-8", errors="strict")
    MGB.reject_unknown_worker_markers(output, "")
    expected_worker_profile = {
        **MGB.FIXED_WORKER_PROFILE,
        "initial_condition": args.expected_profile,
    }
    benchmark = MGB.extract_record(
        output,
        expected_ranks=args.expected_ranks,
        expected_pixels=args.expected_pixels,
        expected_warmup_steps=args.expected_warmup_steps,
        expected_steps=args.expected_steps,
        expected_transport=args.expected_transport,
        expected_worker_profile=expected_worker_profile,
    )
    MGB.extract_device_records(output, args.expected_ranks)
    initialization = MGB.extract_initialization_records(
        output,
        expected_profile=args.expected_profile,
        expected_applications_per_rank=(
            args.expected_applications_per_rank
        ),
    )
    MGB.extract_source_profile_record(
        output, expected_profile=args.expected_source_profile
    )
    if (
        initialization["initial_condition"]["profile"]
        != benchmark["initial_condition"]
    ):
        raise MGB.EvidenceError(
            "initial-condition marker differs from the benchmark record"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
