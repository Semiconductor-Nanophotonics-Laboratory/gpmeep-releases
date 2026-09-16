# Public release policy

This repository publishes accepted gpmeep source and user documentation.
The original development and validation history is retained privately by the
project maintainer, Dong-Joon Yi.

## What is public

- Numerical solver, CUDA kernels, Python API, and package build source.
- Upstream license, copyright, contributor attribution, and citation.
- Package/source identities, checksums, installation instructions, and limits.
- Reviewed summaries of hardware specifications, numerical comparisons,
  elapsed times, and plots when a report is published.

## What is private

- Raw validation/build logs, environment dumps, and execution transcripts.
- Personal filesystem paths, hostnames, device UUIDs, and local process data.
- Development checkpoints, raw type-trace databases, private evidence anchors,
  and unfinished branches.

Inherited notebook output paths are replaced with generic placeholders;
notebook source cells remain unchanged.
Actual hardware UUID literals in inherited non-runtime test fixtures are
replaced consistently with synthetic UUIDs; no runtime solver code is changed.

Public Git history starts from a privacy-cleaned source snapshot. No original
development or validation commits are imported. An existing release package
is not rebuilt or relabelled merely to change publication documentation.
`PUBLIC_RELEASE.json` records its original build commit and package SHA-256.
The README installer example passes that original commit explicitly so a
different public snapshot commit does not weaken provenance verification.

Before each publication, scan every public Git ref and release asset for
private metadata and verify that retained solver/build source matches the
accepted private source. Never import old private refs or raw logs into this
repository.
