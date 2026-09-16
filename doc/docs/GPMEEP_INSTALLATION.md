# gpmeep v1 installation and runtime guide

gpmeep v1 is distributed as a Linux x86-64 Conda package backed by an exact
Micromamba runtime lock. It keeps the standard Python interface:

```python
import meep as mp
```

Install gpmeep in its own environment. Installing official `pymeep` into the
same prefix is unsupported because both distributions provide the `meep`
module and native `libmeep` ABI.

## Public download availability

This repository currently publishes v1.0.3 source and its tag, not a prebuilt
Conda asset. Existing binary recipe metadata is awaiting a privacy review.
`PUBLIC_RELEASE.json` records the accepted original binary's identity, not a
download promise. The binary installation examples below apply only if you
already have that exact package and its matching checksum sidecar.

## Building from the public source

On a supported Linux x86-64 build host, use a clean committed clone and a new
output directory. The script creates an isolated Micromamba package-builder
environment from the included lock; no system Meep environment is modified.

```sh
/bin/bash -p scripts/build-conda-package.sh \
  --jobs 4 --output-dir /absolute/new/path/gpmeep-build
```

The output package and `.sha256` sidecar are under that directory's
`linux-64/` subdirectory. For a local build, pass `git rev-parse HEAD` as
`--expected-source-commit` to the installer, not the original binary commit
used in the examples below. A local build must still pass the installed
provenance and CPU/CUDA self-checks; it is not the original checksum-pinned
release binary.

## Requirements

- Linux x86-64
- glibc 2.17 or newer. The v1.0.3 release is built against the 2.17 sysroot
  and its gpmeep-owned ELF symbol requirements are audited during packaging;
  Ubuntu 22.04/glibc 2.35 is an explicit clean-install qualification target.
- NVIDIA driver reporting CUDA 12.4 or newer through Conda's `__cuda` virtual
  package
- one NVIDIA GPU for CUDA execution; two GPUs for the optional GPU2 check
- network access to the exact conda-forge package URLs in the runtime lock
- adequate local storage for the dedicated environment and Micromamba cache

The v1 build policy compiles every CUDA architecture exposed by CUDA 12.4 from
sm60 upward and adds PTX for the newest supported architecture. Packaging runs
CUDA 12.4 `cuobjdump` on the final `libmeep`, requires its native cubin and PTX
sets to equal the requested build policy, and binds that inventory to the
binary and release manifest by SHA-256. Actual runtime release qualification
was performed on RTX 3090 Ti (sm86). Other generations are portable-build
targets, not yet validated runtime claims.

An explicit `--cuda-architectures LIST` must include at least one native
`-real` target; PTX-only (`-virtual`) packages are rejected before release
provenance is generated.

Maintainers build the binary from a clean commit with
`scripts/build-conda-package.sh --jobs 4`; the job cap is explicit so the
native CUDA/C++ build does not unexpectedly consume every logical CPU.

## One-command installation from a clone

After downloading the release package and its `.sha256` sidecar into a
`downloads/` directory, separate from the builder's default `dist/` output:

```sh
/bin/bash -p scripts/install-gpmeep.sh \
  --prefix /absolute/path/to/gpmeep-v1 \
  --package downloads/gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda \
  --expected-source-commit b48c9b678f2316cefa95e48a661f7a877547a000 \
  --mode gpu1 \
  --gpu-devices 0
```

For two GPUs, use `--mode gpu2 --gpu-devices 0,1`. For installation on a
machine without a visible GPU, use `--mode cpu`; CUDA must still have been
compiled into the package. The installer refuses an existing prefix, verifies
the package SHA-256, materializes the runtime from the explicit lock, compares
the installed archives with the lock, installs the package without dependency
resolution, verifies provenance, and runs the installed self-check.
Micromamba's local-package repodata check runs in warning mode only for that
one `--no-deps` transaction because a direct file has no channel repodata;
the installer first requires and verifies the package's external SHA-256.
For the published v1.0.3 package, pass the original binary build commit shown
above even from a Git clone: a privacy-cleaned public snapshot has a different
Git commit. Without an explicit override the installer uses clone HEAD, which
is appropriate only for a package built from that exact commit. Exported
source trees also require an explicit `--expected-source-commit`.

It creates only these caller-selected paths:

- the requested environment prefix;
- `<prefix>.micromamba-root`, its dedicated package cache;
- `<prefix>.micromamba-root/home`, a private HOME used only by clean-environment
  Open MPI verification;
- `<prefix>.self-check.json`, unless `--report` selects another absent path.

On failure, these paths are retained for diagnosis. The installer does not
silently remove or overwrite an existing environment.

## Installed verification

```sh
/absolute/path/to/gpmeep-v1/bin/gpmeep-verify-provenance \
  --prefix /absolute/path/to/gpmeep-v1

/absolute/path/to/gpmeep-v1/bin/gpmeep-self-check \
  --mode gpu1 --gpu-devices 0 \
  --report /absolute/new/path/gpmeep-self-check.json
```

These relocatable launchers always execute the `python3.11` and MPI helpers
from their own installed prefix. They are safe to invoke by absolute path
without activating the environment and do not resolve Python or MPI through
the caller's `PATH`.

`gpu1` runs fresh CPU and CUDA processes and compares field/flux observables.
`gpu2` additionally launches two MPI ranks, requires distinct device UUIDs,
pinned MPI traffic, waitsome completion, CUDA work on each rank, and zero CPU
fallback. These are small correctness checks and are not performance results.
Release maintainers additionally run the two-sample installed-package AuNP
replay documented in [the performance summary](GPMEEP_PERFORMANCE.md#installed-package-replay-gate).
That replay is a packaging-regression gate, not a new benchmark claim.

## Backend selection

Strict selection fails instead of silently falling back:

```python
mp.gpu.set_backend("cpu")
mp.gpu.set_backend("cuda")
```

The environment alternative must be set before Python starts:

```sh
MEEP_GPU_BACKEND=cuda CUDA_VISIBLE_DEVICES=0 python simulation.py
```

For production two-GPU pinned transport:

```sh
CUDA_VISIBLE_DEVICES=0,1 \
MEEP_GPU_BACKEND=cuda \
MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
MEEP_GPU_MPI_TRANSPORT=pinned \
MEEP_GPU_MPI_COMPLETION=waitsome \
mpiexec -n 2 python simulation.py
```

The installed examples are under `<prefix>/share/gpmeep/examples`.

## Removal and reinstall

Package-level removal is performed with the same Micromamba root used during
installation:

```sh
<clone>/.tools/micromamba --no-rc remove --yes \
  --root-prefix <prefix>.micromamba-root --prefix <prefix> gpmeep
```

Reinstall the same checksum-pinned package with `micromamba install --no-deps`
and rerun both installed verifiers. The v1 release qualification also tests a
fully relocated prefix and a clean uninstall/reinstall cycle.
