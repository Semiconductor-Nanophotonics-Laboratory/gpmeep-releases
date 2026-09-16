# Meep GPU Development

This branch develops a CUDA backend without modifying any system-wide Meep,
Python, Conda, or CUDA installation.

## Repository baseline

- Upstream snapshot: `NanoComp/meep@3be5b944ebf2358a1c3433773e477453d1e36ae5`
- Imported version: `1.35.0-beta`
- Local immutable reference: branch `upstream-baseline`, tag
  `upstream-3be5b944`
- Development branch: `cuda-mvp`

All generated state stays in ignored directories inside this repository:

| Purpose | Path |
| --- | --- |
| Portable Micromamba | `.tools/micromamba` |
| Micromamba cache, process locks, and root | `.micromamba/` |
| CPU and CUDA environments | `.envs/` |
| Out-of-source builds | `build/` |
| Private install prefixes | `install/` |
| Benchmark output | `benchmark-results/` |

## Create an environment

The CPU environment is sufficient for reference builds:

```sh
scripts/create-env.sh cpu
scripts/check-dev-env.sh cpu
```

The CUDA environment adds a self-contained CUDA 12.4 compiler and runtime
development files:

```sh
scripts/create-env.sh cuda
scripts/check-dev-env.sh cuda
```

The CUDA MPI environment additionally provides an isolated CUDA-capable
Open MPI toolchain, parallel HDF5, and `mpi4py`:

```sh
scripts/create-env.sh cuda-mpi
scripts/check-dev-env.sh cuda-mpi
```

No shell activation is required. Build scripts enter the selected environment
using `micromamba run --clean-env`. Caller `PYTHONPATH`, `LD_LIBRARY_PATH`,
`PKG_CONFIG_PATH`, and similar build-affecting paths are not inherited. The
scripts pass through only their documented `MEEP_GPU_*` controls and
repository-local cache locations. The CUDA script also preserves
`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` so scheduler and operator GPU
visibility choices remain authoritative.

On Linux x86_64, environment creation uses the SHA-256-pinned explicit files
under `environment/locks/`. The YAML files document the intended dependency
set and are used only on platforms without a committed lock. An existing
environment is checked for exact package equality; if it has extra or changed
packages, preserve it and create a clean replacement with:

```sh
scripts/create-env.sh cpu --recreate
scripts/create-env.sh cuda --recreate
```

The Micromamba bootstrap is likewise content-pinned. It downloads the official
`mamba-org/micromamba-releases` asset for the exact `2.8.1-0` build and verifies
both the transport archive and the extracted executable by SHA-256. It does not
use the rolling version endpoint, because that endpoint can advance to a new
Conda build without changing Micromamba's upstream version string.

### Release-package builder paths and architecture policy

The release-package builder is separate from the in-place development builds:

```sh
/bin/bash -p scripts/build-conda-package.sh \
  --output-dir /absolute/path/to/gpmeep-build-output \
  --cuda-architectures AUTO \
  --jobs 4
```

`--output-dir` may be absolute or relative to the caller's current directory;
the script canonicalizes it before both source staging and Conda rendering.
Release downloads belong in `downloads/`, not in the builder's default
`dist/`, so an install attempt does not collide with a later local build.
Existing same-version packages and any same-version artifact under `broken/`
remain fail-closed.

`AUTO` is the portable release policy. A local targeted package may instead
use a list such as `86-real;86-virtual`. v1.0.3 records the requested policy
and uses CUDA 12.4 `cuobjdump --list-elf`/`--list-ptx` against the installed
`libmeep` to prove its exact native/PTX inventories. The canonical inventory
is bound to both the `libmeep` bytes and `release.json`; installed provenance
validation replays those hashes and checks the inventory against AUTO or LIST.
Such a LIST package is not the portable AUTO release and must pass self-check
and native-code inventory qualification on its target host before production
use.

The CUDA compiler and user-space libraries can be isolated, but the NVIDIA
kernel driver cannot. `nvidia-smi` must work on the host before CUDA programs
can run.

## Establish CPU references

Meep already supports both precisions. These builds provide the correctness
references for the future CUDA backend:

```sh
scripts/build-cpu-baseline.sh fp64
scripts/build-cpu-baseline.sh fp32
```

Each precision has an independent build and install directory.

## CUDA portability policy

The first backend targets NVIDIA CUDA and uses FP32 fields by default. It must:

1. Detect the CUDA device and compute capability at runtime.
2. Avoid hard-coding a GPU ordinal, memory size, or SM count.
3. Compile real code for a conservative set of deployed architectures and
   include PTX for the newest selected virtual architecture.
4. Allow the architecture list to be overridden at configure time.
5. Keep CUDA-specific storage and kernels behind a backend interface so the
   CPU implementation remains the reference and a future HIP backend is
   possible.
6. Compare CUDA FP32 against both CPU FP32 and CPU FP64. Tolerances must be
   defined per observable rather than by requiring bitwise equality.

The initial architecture policy will cover Pascal and newer devices where
supported by the selected CUDA compiler. CI and release builds should publish
the exact architecture list and CUDA toolkit version.

By default, the prototype asks the selected `nvcc` which of compute
capabilities 6.0, 6.1, 6.2, 7.0, 7.2, 7.5, 8.0, 8.6, 8.7, 8.9, and 9.0 it
supports, builds those native cubins, and retains PTX for the newest selected
architecture. The resulting device code covers desktop, data-center, and some
Jetson GPU architectures, but the checked-in bootstrap and exact environments
currently validate only an x86_64 host toolchain. Override the list for a
deployment or faster local build:

```sh
MEEP_GPU_CUDA_ARCHS="86-real;90-virtual" scripts/build-cuda-prototype.sh
```

The compiled native and PTX architecture sets are embedded into the runtime.
Device discovery and selection reject a GPU that cannot load either an exact
native cubin or PTX compiled for its compute capability or an older one. Thus,
for example, the override above intentionally rejects an `sm_80` GPU instead
of allowing a later kernel launch to fail.

Compile and run the first FP32 curl-update smoke test with:

```sh
scripts/build-cuda-prototype.sh
```

Compilation does not require a working GPU driver. CTest reports the runtime
test as skipped with return code 77 when no CUDA device is available.

## Build the CUDA-enabled Meep library

The integrated library build is separate from the standalone prototype:

```sh
scripts/build-meep-cuda.sh
```

It configures an FP32 Meep build with `--enable-cuda`, runs the complete C++
test suite, and installs into `install/meep-cuda-fp32/`. The corresponding
out-of-source build is `build/meep-cuda-fp32/`. Both remain repository-local
and ignored by Git.

The build accepts these explicit controls:

| Variable | Meaning | Default |
| --- | --- | --- |
| `MEEP_GPU_CUDA_ARCHS` | CMake-style architecture list such as `86-real;90-virtual` | `AUTO` |
| `MEEP_GPU_FAST_MATH` | Enable CUDA `--use_fast_math` with `ON`; keep IEEE-oriented operations with `OFF` | `OFF` |
| `MEEP_GPU_MAKE_JOBS` | Parallel build job count | host CPU count |
| `CUDA_VISIBLE_DEVICES` | Restrict or reorder devices visible at runtime | inherited if set |
| `CUDA_DEVICE_ORDER` | CUDA device enumeration order | inherited if set |

The installed public header `meep/gpu.hpp` provides backend discovery, device
enumeration (including theoretical memory bandwidth), compatibility reporting,
explicit device selection, backend selection, CUDA-runtime touch counters,
curl counters, and constitutive E/H-update counters. `fields` exposes the
owner-specific execution decision and diagnostic, which is the authoritative
answer when different simulations use the process-wide `auto` request. The
same API is present in CPU-only builds;
there it reports that CUDA was not compiled and never loads a CUDA library. A
CUDA-enabled library can also load safely on a host without an NVIDIA driver
and returns a diagnostic instead of failing during process startup.

## M3 staged FDTD dispatch

The first physical integration dispatches the common FP32 `step_db` Yee curl
to CUDA. M3 intentionally copies complete operand arrays and the exact
non-contiguous Meep loop indices to the device for each curl invocation, then
copies the destination field back. This establishes an auditable correctness
path, but it is not expected to be faster than the CPU. Persistent device
storage and transfer elimination belong to the next milestone.

The CPU backend is the default. Select a policy before stepping fields:

```cpp
#include <meep/gpu.hpp>

meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
meep::gpu::set_backend(meep::gpu::backend_mode::automatic);
meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
```

The equivalent process controls are `MEEP_GPU_BACKEND=cpu|auto|cuda` and the
optional CUDA runtime ordinal `MEEP_GPU_DEVICE=N`. The M3 checkpoint used a
fixed 65,536-cell automatic threshold; the current workload/device policy is
documented in M7 below. Override it with `MEEP_GPU_AUTO_MIN_CELLS=N` (`0`
forces automatic CUDA candidacy) after calibrating a new GPU architecture.
`cuda` is strict and ignores this performance threshold: a missing runtime or
unsupported configuration throws a diagnostic. Both B and D curl phases are
preflighted before either modifies a field, so strict mode cannot silently
leave a half-updated time step. An automatic CPU choice is local to the
current `fields` owner: it preserves the requested `auto` policy, and a later
simulation re-evaluates its own rank-local size instead of inheriting a
permanent process-wide CPU selection.

`get_dispatch_statistics()` reports CPU/CUDA curl call and point counts plus
the successful staging transfer bytes. Call `reset_dispatch_statistics()`
before a measured run. These counters distinguish actual CUDA dispatch from
an automatic CPU fallback.

The current CUDA curl accepts:

- an FP32 CUDA build;
- real or complex fields, including Bloch-periodic fields;
- Cartesian 1D, 2D, or 3D grids;
- every split-field PML/material-conductivity combination used by
  `step_curl`;
- zero `beta` and no BFAST wavevector.

At the M3 checkpoint, other Meep phases remained host-resident. `auto`
executes an unsupported curl on the reference CPU path; strict `cuda` rejects
it before the time-step curl begins. The CUDA correctness test compares real
1D, tiled 2D, 3D,
PML/conductive, and complex Bloch simulations under CPU and CUDA modes and
asserts that strict mode records no CPU curl:

```sh
build/meep-cuda-fp32/tests/gpu-step-db
```

It exits with Automake skip status 77 after validating CPU/automatic fallback
when no physical CUDA device is available.

## M4 resident mirrors

M4 replaces the per-curl temporary allocations with a device mirror cache
associated with each `fields_chunk` by an ABI-private registry. A cache has
the same lifetime as its chunk and is never shared with a copied simulation.
At the start of each B or D curl
phase, host arrays are authoritative. The first use of each field uploads it;
later components and loop tiles reuse the same device allocation and device
contents. Dirty destination arrays are copied back once when the phase
finishes, before any host-resident Meep phase can observe them.

This phase boundary is deliberate: sources, boundary exchange, constitutive
updates, and monitors still modify or read host arrays at this milestone.
Keeping a stale device value across one of those operations would be fast but
incorrect. Later kernels can extend the residence interval as those operations
move to CUDA.

The per-chunk resident phase object provides a strong host-commit guarantee. A
normal exit synchronizes kernels, stages every dirty destination in temporary
host buffers, and commits the fields only after all copies succeed. An
allocation, synchronization, or copy error before that commit discards the
current chunk's device-dirty state and leaves that chunk's pre-phase host
arrays authoritative. This is not a transaction over all chunks or the whole
time step: chunks already completed before a later chunk's runtime failure
remain advanced. Configuration preflight is still all-chunk/all-phase atomic
for unsupported strict-CUDA configurations. Device selection is reasserted
for the calling host thread, and changing `MEEP_GPU_DEVICE` through the public
selection API causes an existing cache to be rebuilt on that device.

Checkpoint loading invalidates the affected chunks' caches before any host
field pointer is deleted or replaced. This bounds resident allocation lifetime
and guarantees that the next CUDA phase uploads the newly loaded fields.

The new `resident_statistics` result from `get_resident_statistics()` reports:

- host-to-device and device-to-host bytes actually transferred;
- bytes not transferred because a resident mirror was already current;
- device buffer allocations and reuse hits.

The separate additive `get_live_resident_device_buffers()` API reports the
current process-wide live resident-buffer count. It is a state gauge and is
intentionally not cleared by `reset_dispatch_statistics()`. Keeping it
separate preserves the original by-value `resident_statistics` ABI.

The original `dispatch_statistics` layout and `get_dispatch_statistics()`
symbol remain unchanged for binary compatibility with existing
`libmeep.so.38` consumers. `reset_dispatch_statistics()` resets both groups of
counters.

The integrated correctness case enables Meep loop tiling and requires
non-zero avoided-transfer and buffer-reuse counters on a physical GPU. This
guards against regressing to a result-correct but per-tile staging path.

## M5 material and complex curl coverage

The resident curl now mirrors the complete coefficient and auxiliary-field
set for the Cartesian `step_curl` update:

- `sig`, `kap`, and `siginv` for the first PML direction;
- `f_u`, `sigu`, `kapu`, and `siginvu` for the split auxiliary direction;
- conductivity, conductivity inverse, and the simultaneous
  PML/conductivity auxiliary field;
- both real and imaginary field components for complex and Bloch-periodic
  simulations.

The CUDA kernel implements all eight combinations of first-direction PML,
auxiliary-direction PML, and material conductivity. The scalar update is a
single host/device function tested without a GPU for every combination. The
physical-GPU smoke test additionally checks the device pointer wiring,
per-point sigma indices, negative/positive field strides, and all dirty
auxiliary outputs. The integrated Meep test uses PML and nonzero conductivity
together in real and complex Bloch simulations and requires strict CUDA mode
to report zero CPU curl calls.

PML coefficient arrays remain read-only resident mirrors. `f`, `f_u`, and
the conductivity auxiliary are dirty outputs and participate in the same
stage-all-then-commit guarantee as the basic curl field. Nonzero `beta`,
BFAST, and cylindrical-coordinate correction terms are still explicit
unsupported boundaries for this sub-milestone.

## M5 constitutive E/H update coverage

The Cartesian FP32 backend also dispatches Meep's `step_update_EDHB`
constitutive update. Its CUDA kernel covers the CPU specialization matrix:

- identity or spatially varying diagonal inverse susceptibility;
- one or two off-diagonal tensor rows using Meep's stable Yee-grid average;
- simultaneous chi2/chi3 nonlinear response;
- the split-field PML `W` auxiliary and its sigma/kappa update;
- real and imaginary components, tiled domains, and negative H-field
  strides.

The per-point formula and off-diagonal operand normalization are shared
host/device functions. Device-free tests cover the specialization matrix and
all runtime pointer invariants. The physical-GPU smoke matrix covers PML
on/off, zero/one/two off-diagonal rows, nonlinear on/off, and explicit versus
identity diagonal response. The integrated real/complex test combines a
dielectric, chi2/chi3, PML, and conductivity and requires strict CUDA mode to
record both CUDA curl calls and CUDA E/H-update calls with no CPU fallback.
It also repeats integrated-source add/remove cycles and requires the live
resident-buffer count to remain bounded, guarding `f_minus_p` cache
invalidation when its host allocation is released.

`get_field_update_statistics()` reports CPU/CUDA E/H call and point counts.
It is a separate additive API so the pre-existing `dispatch_statistics`
layout and symbol ABI remain unchanged. `reset_dispatch_statistics()` resets
curl, E/H, transfer, and resident-cache counters together.

## M5 full-step residence, Lorentz/Drude, sources, boundaries, and DFT

Strict CUDA mode now opens one resident phase per local Cartesian chunk around
the complete time step. Curl, constitutive E/H, standard Lorentzian/Drude
polarization, volume-source addition, local boundary operations, and DFT
updates borrow that phase. A normal step keeps dirty field, polarization, and
DFT mirrors device-authoritative for the next step rather than downloading
them at each internal phase boundary. The per-step finite-value guard runs on
the device and returns one integer instead of copying a field array.

The hot curl, constitutive E/H, and Lorentz/Drude launches use a compact
three-dimensional index-space descriptor. The GPU reconstructs Meep's
`LOOP_OVER_IVECS` field and PML indices from a start, three extents, and three
strides. This removes both the CPU loop that previously rebuilt one descriptor
per Yee point and the corresponding O(number of cells) PCIe upload on every
time step; index metadata is now O(1) per launch.

Volume-source indices and interleaved FP32 complex amplitudes are likewise
immutable resident profiles. Only the source-time complex scalar is passed on
each launch, and bounds are checked once per cache lifetime. Extended line,
plane, and volume sources therefore do not rebuild or upload an
O(number of source points) update list every step. Source add/combine/remove
and boundary-source repair invalidate the affected owner cache.

Host-reading APIs synchronize the affected chunk before integrations, array
slices, output, checkpoints, DFT processing, or scalar field access.
Host-mutating lifecycle paths invalidate the cache before replacing or
zeroing arrays. Material replacement and structure phasing synchronize and
rebuild mirrors before reusing changed coefficients. This makes persistent
residence an explicit coherence protocol rather than an assumption that no
host code touched a public field pointer.

DFT coherence includes flux/complex-flux, electric and magnetic energy,
stress/force, near-to-far, LDOS, HDF5 save/load, Python monitor-data
get/load, and adjoint material-gradient readers. Monitor destruction releases
its output, index, and weight mirrors without discarding unrelated fields;
same-sized shifted monitor replacement is covered by a lifecycle regression.
The constant-wave solver synchronizes before packing fields into its host
Krylov vector and invalidates stale mirrors before unpacking a trial vector.

The new CUDA phases cover:

- the exact standard `lorentzian_susceptibility` implementation, including
  its Drude specialization, real/complex polarization state, update, and
  subtraction;
- real and complex volume sources;
- local Cartesian metal-zero, periodic copy, negate, and complex Bloch-phase
  boundary operations with a gather-then-scatter snapshot;
- real and complex DFT accumulation with one or two centered-grid averaging
  offsets, per-point weights, arbitrary frequencies, and decimation.

At this M5 checkpoint, custom susceptibility subclasses, noisy polarization,
and cylindrical grids remained explicit unsupported cases in strict mode.
Automatic mode uses the CPU reference path for an unsupported phase.

Cartesian multi-process runs keep local fields resident and exchange remote
Yee boundaries through persistent GPU gather/scatter descriptors. CUDA-aware
MPI passes device staging buffers directly. The portable path uses persistent
pinned-host buffers and aggregates each peer's noncontiguous blocks into one
MPI derived-datatype request. Both paths cache the immutable connected-boundary
topology and replay it after validating resident-cache generations.

The additive `get_polarization_statistics()`,
`get_source_statistics()`, `get_boundary_statistics()`, and
`get_dft_statistics()` APIs report CPU/CUDA calls and scalar point counts.
They let correctness tests reject a numerically matching result whose
important work silently ran on the CPU.

### Production performance gate

Build and run the physical-GPU scale sweep with:

```sh
scripts/build-meep-cuda.sh
scripts/benchmark-gpu-core.sh
```

The benchmark compares CPU and strict CUDA for 2D line-source vacuum+DFT, 2D
Lorentz+DFT, and 3D vacuum+DFT workloads. It verifies field and DFT numerical
agreement, rejects any measured CPU fallback, and writes raw CSV under
`benchmark-results/`. Small sizes expose launch-overhead crossover; only the
largest size in each family is a release gate.

By default every production row must reach at least `2.0x` wall-time speedup
and transfer no more than 64 host/device bytes per logical cell-step. Override
the thresholds only for an explicitly documented hardware qualification:

```sh
MEEP_GPU_MIN_SPEEDUP=2.5 \
MEEP_GPU_MAX_TRANSFER_BYTES_PER_CELL_STEP=32 \
scripts/benchmark-gpu-core.sh
```

`MEEP_GPU_BENCH_QUICK=1` reduces sizes and steps for smoke testing but does
not replace the default production run. If no physical CUDA device is
visible, the benchmark exits with skip status 77; a skipped benchmark never
counts as performance evidence.

## Broad feature endpoint

The staged and resident curl milestones are infrastructure, not the final
support boundary. The project acceptance target is broad acceleration of the
normal Meep workflow, including PML/absorbers, conductivity, complex and Bloch
fields, anisotropic and dispersive media, cylindrical coordinates, sources,
boundary handling, DFT/flux/near-to-far monitors, and multichunk execution
where CUDA implementations are technically viable.

Example validation must check numerical agreement and CUDA coverage. An
example that produces the right answer while all important work silently
falls back to the CPU is reported as a coverage failure, not a GPU pass.
For representative production-sized workloads, no material speedup or a
slowdown is a release blocker. Benchmark reports must separate kernel time,
host/device transfer, synchronization, and CPU fallback coverage so those
failures drive residency, fusion, and batching work instead of being hidden
inside an aggregate wall time. Tiny launch-overhead-dominated cases are
reported honestly with a scale sweep and CPU/GPU crossover; they do not
substitute for the production-size performance gate.
The final matrix classifies the repository's C++ regression tests and Python
examples by feature, removes only documented near-duplicates, and treats the
Python adjoint/inverse-design examples as a mandatory objective/gradient
gate. Scheme/CTL examples are outside the requested release validation scope.

## Mandatory multi-GPU endpoint

Multi-GPU is a release requirement, not an optional follow-up. The supported
execution model is Meep's existing MPI spatial decomposition with one MPI rank
per GPU by default. Each rank selects a device from its node-local rank and
the visible CUDA device list; an explicit per-rank device override remains
available for schedulers with nonstandard placement.

Remote Yee-boundary exchange must preserve device-authoritative fields.
CUDA-aware MPI uses device buffers directly when the MPI implementation and
runtime support that path. A pinned-host staging path is required as the
portable fallback; pageable full-field staging and silent CPU stepping are
not acceptable. The final qualification matrix includes:

- one- versus two-GPU numerical agreement for real and complex/Bloch fields,
  PML, sources, Lorentz/Drude media, and DFT monitors;
- two- and four-GPU rank/device mapping, including multiple ranks on a node
  and explicit device overrides;
- strict checks that every participating rank performs CUDA work and that
  boundary exchange does not invalidate unrelated resident arrays;
- production-size strong-scaling measurements for one, two, and four GPUs,
  with communication, synchronization, transfer, and compute time reported
  separately.

Physical two- and four-GPU measurements are required before release. A
single-GPU host, a device-free compile test, or successful MPI CPU fallback
cannot satisfy this gate.

### Current multi-GPU implementation and qualification

Build the isolated FP32 CUDA MPI configuration with:

```sh
scripts/build-meep-cuda-mpi.sh
scripts/build-meep-cuda-mpi-python.sh
```

The first command builds the C++ MPI library/test surface. The second verifies
the installed environment against the exact package lock, preserves any old
build/install trees, starts from fresh directories, builds the MPI-enabled
Python module, and runs the C++ MPI suite plus focused singleton/two-rank
Python lifecycle and DFT-decimation checks under timeouts. It then tests both
the in-place and installed Python packages and writes a source-stable receipt
containing the exact environment manifest, both Python trees, test binaries,
and qualification logs. Generic Python tests which assume singleton callbacks
or local files belong to the separate broad validation matrix; this build
qualification is not described as the complete Python suite. Both build
commands use the same locked `cuda-mpi` environment; the Python build is not a
hand-maintained hybrid of the single-GPU and MPI prefixes.

After the Python build, run the receipt-bound public-binding correctness
matrix with:

```sh
.envs/meep-gpu-cuda-mpi/bin/python \
  scripts/run-mpi-python-validation.py \
  --build-receipt build/meep-cuda-mpi-python-fp32/build-provenance.json \
  --devices 0,1 \
  --output /path/outside/the/source/tree/c7-mpi-python
```

This controller obtains a nonblocking host GPU lock, requires idle NVIDIA
compute state before and after the matrix, and monitors every CUDA rank PID.
It executes the same fixed two-rank Python FDTD/DFT workload in GPU-hidden CPU,
pinned/waitsome CUDA, and CUDA-aware/waitall CUDA lanes. Every rank is bound to
one non-overlapping physical CPU core. The gate compares all 6,400 field values
and all flux values, requires zero opposite-backend calls and points, proves
that import and CPU execution never touch the CUDA runtime, binds every rank's
CUDA UUID to its physical NVIDIA ordinal, and requires mutually exclusive
transport bytes plus observed `MPI_Waitsome`/`MPI_Waitall` execution counters.
It runs `test_divide_mpi_processes.py` under both CPU and CUDA with per-rank
backend/device/dispatch telemetry. Runtime Python, extension, and libmeep
hashes must match the build receipt directly. The final
`COMPLETE` marker binds the current source/receipt, raw logs/records, NVIDIA
inventory/process monitor, report, and artifact manifest. Its one-sample tiny
workload is correctness-only and is never eligible for a speed claim.

One rank is mapped to one visible GPU using its physical node-local MPI rank.
`MEEP_GPU_DEVICE` remains an explicit per-process override, but it does not
opt in to sharing one physical GPU. CUDA/MIG UUIDs are reserved in an
MPI-world RMA claim table, so ranks in different
`divide_parallel_processes` subcommunicators cannot silently oversubscribe
the same device. Set `MEEP_GPU_ALLOW_OVERSUBSCRIBE=1` on every sharing rank
only when that behavior is intentional. A valid collective device
permutation releases the old claim set before atomically claiming the new
set.

In Python, `mpi4py` owns MPI initialization. Importing the MPI build creates
the world-wide GPU claim window before Meep subcommunicators can be formed and
registers its teardown ahead of `mpi4py` finalization. The lifecycle is
idempotent and supports finalize/re-initialize tests, including an already
finalized MPI runtime. Because `MPI_Win_free` is a world collective, every
rank in `MPI_COMM_WORLD` must import Meep and leave the process symmetrically;
an application which imports or exits on only a subset of world ranks violates
this distributed lifecycle contract.

`gpu::set_backend` is deliberately process-local and contains no hidden MPI
collective. Distributed applications with potentially rank-specific
configuration should set `MEEP_GPU_BACKEND` and let the first `fields::step`
perform the collective no-throw preflight, or explicitly agree on
`set_backend` success before continuing. A rank-local failure in automatic
mode makes every rank fall back to CPU; required CUDA mode fails every rank
before boundary traffic. Once validated, the backend generation is cached;
steady steps perform only one cache-consensus reduction instead of repeating
the complete configuration handshake.

Select the
transport with `MEEP_GPU_MPI_TRANSPORT=auto|pinned|cuda-aware`. `auto` uses
direct device buffers only when every active rank reports CUDA support;
otherwise all automatic ranks safely intersect on pinned staging. An
explicit pinned/device conflict, or forced `cuda-aware` without support on
every rank, fails collectively before exchange. Managers created before a
CUDA preflight always use the legacy pinned wire protocol, so CPU execution
never parses or depends on a CUDA transport setting.

`MEEP_GPU_MPI_COMPLETION=waitsome|waitall` selects the CUDA-aware
boundary-request completion experiment. The current release-compatible
default is `waitsome`; `waitall` is opt-in until same-binary correctness and
repeated timing evidence show that it improves the target multi-GPU workloads.
Pinned staging retains its original completion path. The value must agree on
every active rank and is validated collectively before boundary traffic.

The authoritative comparison entry point is
`scripts/run-mpi-completion-ab.py`. Its immutable
`m19-mpi-completion-ab-v3` profile runs two ranks on two physical GPUs, one
warm-up and six measured samples per policy in alternating order. Each policy
therefore occupies the first and second position three times. A fresh
nonce binds every raw child to the receipt ID, full source snapshot, exact
producer and command. Objective, every gradient value, exact CUDA counter
signatures, physical-device mapping, runtime closure, workload wall and fresh
process wall are independently recomputed. Valid negative evidence is still
published as `COMPLETE`, but `waitall` is promoted only when both wall-time
medians exceed the source-bound `1.02x` floor and a majority of pairs are
faster. Fresh-process promotion uses first-position and second-position
medians separately, requires both strata to clear the floor, and reports their
geometric-mean balanced estimator; the unstratified process median is
diagnostic-only. A symmetric first/second position-effect ratio above the
source-bound `1.05x` threshold is emitted as a warning. Cross-sample
environment comparison removes only an exact list of
per-`mpiexec` Open MPI/PMIx session identifiers, paths, and rendezvous URIs;
the presence of those keys must remain stable. `OMPI_ARGV` is separately
required to equal the already validated producer command, while any
unclassified environment change fails closed. The raw file set is an exact
allowlist, and the sole authoritative
`COMPLETE` marker binds the report plus an artifact manifest containing all
raw JSON/logs, the copied receipt and report Markdown.
The controller ledger is reconstructed from every raw child after sealing.
Stored validation, receipt/runtime, process-environment, lazy-import, material-
gradient, parent-timing, and stdout-pointer records must exactly equal values
freshly derived from the raw files. The complete matrix is reverified once
before report construction and again inside the capability-gated publisher;
the publisher then rebuilds the comparison, raw artifact allowlist, source and
driver bindings, and Markdown before it can create `COMPLETE`.
After `COMPLETE` is written, it reopens the marker, report, manifest, every
manifest record, and all raw semantics for a final terminal replay; any byte,
size, or derived-value change removes the success marker.

Run the correctness matrix and strong-scaling harness with:

```sh
mpirun --bind-to none -np 2 \
  build/meep-cuda-mpi-fp32/tests/gpu-step-db
scripts/benchmark-multi-gpu.sh
```

The release harness requires every requested rank count by default
(`1,2,4`). On a smaller development host, explicitly set
`MEEP_GPU_MULTI_REQUIRE_ALL=0` to collect partial smoke evidence; skipped
physical GPU counts never satisfy the release gate. The harness obtains the
compatible visible-device count from the CUDA runtime (including correct MIG
enumeration), validates its efficiency and require-all controls, and rejects
an unavailable requested rank count rather than relying on `nvidia-smi`
display-line counting.

The version-3 evidence contract runs each executable rank count at least three
times in repeat-outer/rank-inner order and uses the median worker time. Every
sample is compared with the first one-GPU observable vector, and the physical
rank-to-UUID mapping must remain stable across repeats and form the same
canonical prefix across the one-, two-, and four-rank runs. Curl, E/H, source,
boundary, and DFT calls are reported and gated separately with zero CPU calls;
polarization is explicitly `not_applicable` for this vacuum workload. The
release profile is fixed at `192^3` cells, 12 warm-up steps, 80 measured steps,
pinned transport, at least 55% parallel efficiency, and at most `0.002`
relative observable error. Quick or relaxed development settings cannot
produce a release-qualified report. Individual logs, CSVs, the combined log,
the build receipt, and all benchmark ELFs are re-hashed immediately before the
hash-bound `COMPLETE` marker is published.

The correctness matrix covers real and complex/Bloch 1D/2D/3D fields,
PML/conductivity, Lorentzian media, nonlinearity, sources, local and remote
boundaries, DFT monitors, constant-wave solves, and lifecycle transitions.
It requires CUDA work on every rank and distinguishes direct CUDA-aware bytes
from pinned-staging bytes.

MPI adjoint monitors also compute automatic DFT decimation from globally
reduced source bandwidth and safety flags. `IndexedSource` may exist only on
its owning rank; an empty rank-local source list therefore no longer forces
unit decimation on the entire job. A focused two-rank regression requires a
band-limited source present on rank zero only to select the same factor greater
than one on both ranks, and a continuous source present on rank zero only to
select factor one globally. The audited production case reduced adjoint DFT
calls from 19,840 to 211 and brought one-/two-rank gradient disagreement down
to `1.12e-7` maximum absolute and `8.63e-8` relative L2.

`scripts/run-mpi-adjoint-benchmark.py` is the only entry point allowed to
publish an MPI-adjoint `COMPLETE` marker. It accepts a build receipt and output
directory, not arbitrary raw result files or adjustable release tolerances.
The fixed `m19-cpu-gpu-adjoint-release-v3` profile uses one identical
4,194,304-cell 2D MaterialGrid problem for the performance lanes: CPU
`2 MPI x 4 OpenMP`, one GPU, and two GPUs. It runs one warm-up per lane and
four pairwise-position-balanced measured triplets (`ABC`, `CBA`, `BCA`,
`ACB`). The release headline is the barrier-delimited maximum-rank workload
wall time, with separately recomputed CPU-to-one-GPU, CPU-to-two-GPU, and
one-to-two-GPU median/conservative speedup gates plus a fixed timing-CV gate.
This qualification profile is deliberately host-specific: it requires the
recorded 8 physical cores to be the complete CPU affinity available to the
runner and requires the CPU baseline to cover every one of them. CPU and CUDA
rank zero must report the same hostname. The CPU model, normalized topology,
governor/driver/EPP/frequency limits, start/end load averages, and thermal
throttle counters are sealed into both positive and negative reports; changed
power configuration, any throttle increment, or excessive load fails closed.
Linux CPU pressure totals are sampled immediately before and after every raw
child, so a competitor that starts only during the CPU lane is still visible;
the fixed CPU-performance lane rejects a `some` scheduler-stall fraction above
`0.01` for each warm-up or measured process, and each pressure interval must
enclose the controller-measured child duration with at most `0.25` s overhead.
A different CPU topology requires a new source-reviewed qualification profile
after a fresh topology-derived CPU sweep, so an 8-core denominator cannot be
silently reused to exaggerate speedup on a larger host.
A separate one-rank FP32 CPU legacy full-gradient oracle retains actual
legacy-kernel dispatch evidence and is never used as the performance baseline.
The earlier 1,048,576-cell profile remains negative crossover evidence: its
communication-dominated two-GPU run was approximately `0.98x` the one-GPU
rate. A receipt-bound, exploratory one-pair 4,194,304-cell probe reached
`1.433x` with identical objectives and no failed entries among all 1,681
gradient tolerances. That probe is crossover-selection evidence rather than a
release result; the larger fixed profile must still pass its CPU oracle,
finite-difference checks, four position-balanced pairs, and unchanged median
and conservative performance thresholds before publication.
This is an explicit workload revision, not merely zero-padding: the source
definition remains tied to the left PML and therefore moves from `x=-6.5` in
the 16-unit cell to `x=-14.5` in the 32-unit cell. The design region,
resolution, objective, 1,681-variable gradient, and release thresholds remain
fixed, and the exploratory profile retained a nonzero gradient norm.
If any release gate fails after all samples finish, the runner atomically
writes `UNQUALIFIED_REPORT.json` with the recomputed comparison, fresh-process
times, FD oracle, and reverified raw/log hashes; `FAILED.json` binds that file
by SHA-256 so negative performance evidence remains independently auditable.
A separate installed-package test compares legacy and analytic full gradients
and checks two dense directions plus three separated interior components by
central finite differences. Every child report is launched by the runner with
a fresh nonce and is bound to the receipt ID, full source snapshot, producer
hash, exact command and a clean allow-listed environment. The installed Python
package manifest, extension, `libmeep`, Python executable, mapped
MPI/CUDA/HDF5 libraries, CUDA/MIG identifiers and semantic rank environment
must stay receipt-bound and stable across repeats. Rank-local visible-device
inventories may differ for scheduler-isolated or multi-node jobs, but each
selected UUID must remain in that rank's stable inventory. Every rank binds
its UUID to a protected, hashed local `nvidia-smi` physical/MIG inventory;
the launch-host probe separately binds rank zero without assuming that remote
node GPUs are locally visible.

`scripts/compare-mpi-adjoint-benchmarks.py` supplies the raw-record validators
but its standalone arbitrary-input CLI is diagnostic-only and always writes a
failure state. The validators independently recompute the full-gradient
digest and timing, require phase-local CUDA calls *and points* in forward and
adjoint with zero CPU fallback, require exactly one measured MPI transport per
phase and across all repeats, require FP32 byte/scalar accounting and
bidirectional pinned staging, and enforce fixed absolute forward/adjoint DFT
calls and singleton point counts plus a cross-rank DFT-point bound. Both the
barrier-delimited phase time and fresh-process wall time must achieve at least
`1.25x` median and `1.15x` conservative two-GPU speedup; these thresholds are
constants and are not command-line options.

The build receipt must name the fixed builder and qualification contract,
match the exact Micromamba lock, bind hashed tool binaries and actual
`config.h`/`config.status` feature state, and contain hash-bound PASS-marked
build, MPI, DFT, rank-failure, installed-import, and installed-adjoint logs.
An authoritative build preserves the previous prefix, recreates the dedicated
prefix from the exact lock, force-reinstalls with Micromamba's safety and extra
safety checks, verifies package-owned bytes and rejects unowned influential
files, and then seals a complete prefix manifest. The actual configure/build
stage is launched only through `env -i` plus `micromamba run --clean-env`; its
entire influence-bearing exported environment is independently captured twice
and compared byte for byte, so Autoconf, make, NVCC, Python, or pkg-config
control variables not present in a finite denylist cannot enter silently.
Bash-maintained `_` and `SHLVL` are the only explicitly named normalized
values. The receipt binds that full environment record and the pinned
Micromamba binary, and the qualification runner independently recreates the
same clean activation and requires exact equality. The authoritative Bash
runs in protected mode, and the build itself uses a fresh nonce-scoped empty
`HOME` whose final tree is receipt-bound, preventing user ccache, Autotools,
PMIx, or PRRTE configuration from influencing compilation or qualification.
Authoritative builds disable optional ccache entirely so a persistent cache or
`prefix_command` cannot substitute compiler output.

Every receipt, runtime artifact, package/environment manifest, raw JSON,
stdout and stderr is reverified again immediately before publication. Runtime
records hash every mapped regular file from the isolated prefix, including
plugins loaded lazily by MPI, PMIx, UCX, HDF5, and CUDA. Python bytecode is
either manifest-bound or explicitly excluded from manifests only when
execution is redirected to a fresh, verified-empty `PYTHONPYCACHEPREFIX`;
Python and native sources remain fully hashed. Local qualification helpers are
loaded directly from source, and Open MPI/PMIx/PRRTE use source-bound parameter
files rather than user configuration. Within a verification epoch, a digest
is reused only after device/inode/mode/size/mtime/ctime validation; hashing
uses one open file descriptor with before/after `fstat` and pathname checks.
The profile and evidence contract are covered by the full `751/751` script
tests and a successful current-package five-direction/component FD run
(`58.933` s), but are not yet qualified against the next fresh receipt. The
latest fixed-runner and CPU-baseline adversarial re-audits are complete.

CPU/GPU speedup claims use a separate fixed parallel-CPU tuning contract, not
the single-rank legacy correctness oracle. On the current 8-core/16-thread
host, `scripts/run-cpu-parallel-benchmark.py` measures `1x1`, `2x1`, `4x1`,
`8x1`, `16x1`, `1x8`, `2x4`, and `4x2` MPI-rank/OpenMP-thread layouts. It
records the actual affinity of every live thread, excludes `1x1` from winner
selection, rejects overlap, oversubscription, hidden runtime controls, timing
CV above the fixed limit, changed power policy, or thermal-throttle increments,
and derives the final selection again from receipt-bound raw JSON immediately
before publishing `COMPLETE`. Objective and all 1,681 gradient values must
match the reference. The latest completed authoritative sweep retained at
`benchmark-results/cpu-parallel/20260805T220634Z/` selected
`2 MPI x 4 OpenMP` (eight physical workers) with a `32.350858` s workload-wall
median and `1.077%` CV. The topology-canonical `1 MPI x 8 OpenMP` cross-check
was `33.973227` s and `1x1` remained diagnostic-only. Because CPU/GPU speedup
depends on both processor models, a new host repeats the topology-derived
rank/thread sweep instead of reusing this host's denominator.

The CPU runner now also commits an immutable run anchor and per-sample
hash-chain journal with trusted same-boot `--resume`. Exact-next interrupted
artifacts and partial publication files are preserved in private,
intent-bound orphan records before the fixed schedule continues. Publication
uses Linux `O_TMPFILE` plus `linkat(AT_EMPTY_PATH)` with no unsafe fallback;
this requirement applies only to authoritative benchmark evidence tooling,
not the gpmeep runtime package. The implementation passed focused `49/49` and
full `168/168` tests, an actual `/scratch` capability probe, and a final
end-to-end independent audit with zero Critical, High, or Medium defects.
Because these evidence-source changes invalidate the prior source-bound build
receipt, a fresh build and final repeated sweep are still required before
GPU/CPU publication from the new journaled contract.

The first full sweep failed closed at `8x1` after preserving the valid raw
record: domain decomposition legitimately leaves ranks that do not intersect a
source or DFT monitor with exact zero source/DFT counters. The corrected gate
requires curl/update/boundary work on every rank and source/DFT work in each
forward/adjoint phase only after aggregation over all ranks. It additionally
requires exact final-to-phase counter accounting, zero GPU transfer/residency
activity on the CPU lane, top/rank-zero counter equality, and invariant
aggregate counter signatures between samples that use the same MPI world size.
The failed evidence remains under
`benchmark-results/m8.7-cpu-parallel-20260804T115050Z/` and is not a performance
publication. An independent adversarial audit replayed every preserved raw
record and rejected 92 resealed counter/accounting/signature mutations; it
reported zero Critical, High, or Medium findings.

Four subsequent checkpoint-hardening audit snapshots intentionally failed and
are retained under
`benchmark-results/m8.7-cpu-checkpoint-hardening-20260804/`; they exposed and
closed shallow COMPLETE verification, realistic interruption recovery,
path-alias, post-publication fallback, lock-inode, and JSON round-trip defects
before the final PASS snapshot. The final runner/test SHA-256 values are
`801cca93e92ab11b9bed43bbd469a0286a697d611956d91ef85d432d22ef15bc`
and `d1f18f66309063f0cd1f764127241070a3f35687dcecdbf67b02f908fbe0b15a`.

Historical context remains important: on the earlier 262,144-cell 2D adjoint
workload, two GPUs took about `2.14x` as long as one GPU (`0.468x` speedup).
That result remains a performance failure. The `192^3` C++ workload still
scales about `1.79x`; the smaller 2D workload is dominated by synchronous
small halo exchanges. The next optimization batches asynchronous pinned
D2H/H2D work per boundary phase and then replaces repeated MPI setup with
persistent peer halo plans. A size sweep must establish the automatic
one-/multi-GPU crossover before release.

The initial full Python-under-two-ranks build attempt reported failures in
generic unit tests. An archived, unmodified upstream Meep tree rebuilt with
the same FP32/MPI dependencies reproduces the MaterialGrid, MPB, Simulation,
and user-material failures; their source files are byte-identical. The
binary-grating failure comes from a singleton expected-error test whose C++
abort path correctly invokes `MPI_Abort` in a distributed process. Raw logs
and hashes are retained in
`benchmark-results/m8.7-upstream-mpi-python-baseline-20260804/`. Generic Python
units therefore run as singleton tests, while dedicated rank-safe lifecycle,
DFT, agreement, coverage, transport, and scaling tests qualify MPI behavior.

Before the version-3 repeated-median contract was introduced, the current
two-RTX-3090-Ti development host produced these single-sample 192-cubed,
80-step baselines. They are retained only as optimization context, not as
release evidence:

| Transport | 1 GPU | 2 GPUs | Speedup | Efficiency | Observable error |
| --- | ---: | ---: | ---: | ---: | ---: |
| pinned | 7.192 s | 4.035 s | 1.782x | 89.1% | 0 |
| CUDA-aware | 7.125 s | 4.075 s | 1.748x | 87.4% | 0 |

Raw logs are generated under the ignored `benchmark-results/` directory and
will be copied into the final release evidence bundle. This host exposes only
two physical GPUs, so four-GPU physical correctness and scaling remain an
unsatisfied release gate; the harness reports that configuration as skipped
instead of treating it as a pass.

## M6 nonzero-beta and cylindrical coordinates

The FP32 resident step now supports Meep's nonzero `beta` coupling for 2D
`exp(i beta z)` simulations. A structured CUDA correction kernel reproduces
the CPU sign convention for B/D, X/Y, real storage, and both complex
components. The correction updates the destination, split-PML auxiliary, and
simultaneous PML/conductivity auxiliary without ending the resident phase.
Positive and negative beta are compared directly against CPU FP32 for real
and complex fields with PML and conductivity.

Cylindrical R/P/Z fields are also supported in strict CUDA mode:

- R and P use the ordinary structured curl after Meep's cylindrical operand
  selection.
- Z uses a parallel local finite-difference form of
  `(1/r) d(r F_phi)/dr`. It is algebraically equivalent to adjacent
  differences of Meep's serial radial-prefix scratch array and removes that
  host prepass and its synchronization.
- nonzero azimuthal `m` uses a structured `i*m/r` correction with the same
  inverse PML/conductivity propagation as the beta kernel;
- dedicated axis kernels implement the `m=0` Dz recurrence and the
  `|m|=1` Dp/Br recurrence;
- resident zero spans implement the axis and high-`|m|` regularity policy,
  including `zero_fields_near_cylorigin=false`.

The cylindrical enablement includes resident constitutive E/H updates with
the axis off-diagonal rule, standard Lorentzian/Drude polarization, sources,
local and remote boundaries, and DFT/flux consumers. The host/device formula
suite checks both radial-difference signs, shifted radial origins, axis and
`i*m/r` drives, and all eight additive PML/conductivity combinations.

Qualification evidence for this checkpoint:

- standalone CUDA formula/runtime tests: 3 pass plus one expected physical
  smoke skip when run inside the restricted build sandbox;
- strict physical-GPU C++ suite:
  `25 PASS / 0 FAIL / 0 SKIP`;
- direct CPU/CUDA beta matrix: beta `+/-0.17`, real/complex, PML, and
  conductivity, with zero CPU curl fallback;
- direct CPU/CUDA cylindrical matrix: `m=0, +/-1, +/-2`, real/complex,
  axis and split annular chunks, PML, conductivity, high-m regularity on/off,
  chi2/chi3 nonlinearity, DFT accumulation, and standard Lorentzian/Drude
  dispersion, with zero CPU curl, E/H, source, boundary, DFT, or polarization
  fallback. Dedicated no-PML conductivity cases cover both axis recurrences;
- CPU-only FP32 and FP64 library builds pass;
- `make dist` succeeds and includes the new CUDA cylindrical and shared curl
  formula headers in the source archive;
- two physical GPUs pass the complete direct comparison under both pinned
  and CUDA-aware MPI. The transport counters report 163000 pinned bytes and
  zero direct bytes for pinned mode, and 163000 direct bytes and zero pinned
  bytes for CUDA-aware mode.

## Cartesian BFAST fixed-angle broadband updates

The resident FP32 CUDA B/D phase now implements Cartesian BFAST for 1D, 2D,
and 3D grids. The ordinary curl and persistent BFAST `F` recurrence execute
in the same resident session, and the destination, split-PML auxiliary,
simultaneous PML/conductivity auxiliary, and BFAST auxiliary remain
device-authoritative. Tiled grids use two batched launches per field
component—ordinary curl followed by BFAST—rather than one launch per CPU
cache tile.

The CUDA formula preserves two easily missed CPU compatibility details:

- a missing first operand swaps the operands, strides, and cross-product
  wavevector coefficients before evaluation;
- the bare, single-operand, no-PML, no-conductivity branch stores
  `F=drive`, while every other specialization stores `F=drive-F_previous`.

Host formula tests cover positive and negative strides, both operand
normalizations, two consecutive recurrences, and all eight combinations of
PML-f, PML-u, and conductivity. The physical CUDA smoke matrix runs three
consecutive recurrences for 11 targeted operand/material cases on both the
single and batched launch paths. The direct integrated matrix compares CPU
and CUDA across real and complex storage, 1D/2D/3D, PML on/off,
conductivity on/off, and 18 time steps. It requires BFAST CUDA point
accounting in addition to ordinary curl work and zero CPU curl fallback.
Both the batched and forced non-batched tile paths pass. CPU-only FP32 and
FP64 builds also remain valid.

Cylindrical BFAST is deliberately not inferred from the Cartesian formula.
Strict CUDA preflight rejects that combination atomically with an explicit
`cylindrical BFAST` diagnostic until its coordinate semantics are proven and
covered independently.

## M7 special polarizations and automatic crossover policy

Standard gyrotropic Lorentzian, Drude, and saturated/LLG susceptibilities
remain resident on CUDA. The physical Faraday-rotation regression covers all
three models, and the production 512-squared LLG case is 5.30x faster than
the FP32 CPU reference.

Standard multilevel atoms now execute their population relaxation,
field/polarization interaction, and transition oscillators in two fused CUDA
launches per susceptibility. Population scratch is point-local
(`ntot*levels`), so concurrent GPU threads cannot race through the historical
single shared scratch vector. Material matrices, population state, all
transition-polarization pairs, and descriptor plans are resident and reused;
steady measured steps transfer no field-sized host/device arrays.

The qualification matrix includes:

- a raw three-level/two-transition/two-channel CUDA oracle with structured
  spaces, shifted neighbors, canaries, and `5.96e-8` maximum absolute error;
- integrated real 1D, complex Bloch 2D, and real 3D CPU/CUDA comparisons with
  PML, sources, DFT consumers, and local/remote polarization boundaries;
- adversarial rollback checks that inject allocation failure after CUDA
  descriptor buffers exist, require zero leaked resident buffers, and then
  require a successful CUDA retry;
- resident-range topology checks that reject ambiguous nested or overlapping
  mirrors before allocating or mutating device state;
- an FP32 Python observable oracle in
  `python/tests/test_gpu_multilevel_atom.py`, which requires strict CUDA
  polarization calls and zero CPU polarization fallback;
- the upstream FP64 CPU `test_multilevel_atom.py` physical oracle, retained as
  a reference-precision compatibility gate while the FP32 test owns the CUDA
  dispatch contract;
- two-rank/two-GPU agreement with one RTX 3090 Ti per rank and pinned
  boundary transport (`1156` messages, `40750` scalars, `163000` bytes);
- a post-audit production 512-squared multilevel/DFT case running 7.74x
  faster than its CPU reference. The same production sweep reports 2.74x
  vacuum, 2.90x Lorentz, 5.30x gyrotropic LLG, and 2.22x 3D-vacuum
  speedups.

The size sweep also records the launch-latency crossover honestly: 64-squared
strict-CUDA cases run at 0.38--0.53x CPU speed, while 256-squared cases reach
1.84--4.16x depending on material complexity. The current `auto` policy is a
two-stage, whole-`fields` decision:

1. It counts one real CPU FDTD worker per MPI rank. The Yee curl,
   constitutive-update, and polarization loops in this build are serial inside
   a rank; OpenMP threads used by auxiliary routines are therefore not
   misreported as stencil workers. MPI decomposition is already reflected in
   each rank's smaller local-cell count. The host-only floor is 524,288
   rank-local cells per FDTD worker. If any participating rank is below its
   floor, every rank selects CPU without calling the CUDA runtime.
2. Only surviving owners discover a device. The floor may increase for a GPU
   below the 1.008-TB/s RTX 3090 Ti reference bandwidth (84-SM scaling is the
   fallback when bandwidth is unavailable); it never decreases below the
   host-only floor. Every rank must still select CUDA, or the distributed
   owner uses CPU as one atomic decision.

The floor was reduced from 1,048,576 after the resolution-40 triangular
diffraction workload exposed a clear false negative: 754,810 local cells were
rejected before probing CUDA even though one RTX 3090 Ti completed the Meep
interval 7.66x faster than eight physical-core MPI CPU ranks. The largest
forced-CUDA-slow fixture in the same batch has only 148,877 local cells, so it
and the 39,032-cell polarization and 13,932-cell zone-plate fixtures remain on
CPU with substantial margin. Exact boundary and device-bandwidth scaling tests
pin the revised policy.

`MEEP_GPU_AUTO_MIN_CELLS` is an exact explicit override of both stages; `0`
forces CUDA candidacy, while malformed or out-of-range values fail before
availability fallback. An explicit `MEEP_GPU_DEVICE` is also fail-closed and
intentionally probes the requested device even for a small owner. Strict
`cuda` mode deliberately bypasses the performance policy for coverage tests
and explicit user control.

The default policy diagnostic records `cpu_fdtd_workers_per_rank=1` so it
cannot be mistaken for an OpenMP speedup claim. Fair CPU/GPU performance
reports use an independently tuned multi-rank CPU baseline.
`scripts/benchmark-moving-source.py` is consequently restricted to one MPI
rank and marks every CPU result `serial-single-rank-diagnostic` with
`performance_claim_eligible=false`; it cannot seal a release CPU-speed row.
CUDA rows may use the same total physical-core budget as a multi-rank CPU
baseline for auxiliary host work; those threads are not counted as CPU FDTD
workers by the automatic crossover policy.

Single- and multi-rank regressions verify CPU selection with zero runtime
availability probes, enumerations, selections, allocations, and transfers;
later CUDA selection without resetting the requested `auto` policy; CUDA
selection at zero; canonical MPI diagnostics; duplicate physical-device
rejection; and collective rejection of invalid or rank-mismatched values.
`initialize_field` resolves the same owner policy before its first boundary or
E/H update. `fields::gpu_cuda_execution_selected()` and
`fields::gpu_execution_diagnostic()` report the cached owner decision;
`mp.gpu.statistics()["runtime"]` reports CUDA-runtime touches.

## M7 FP32 eigensolver and Python example oracles

The upstream `solve_eigfreq` test no longer skips FP32. The FP32 branch uses
three shift-and-invert iterations with `tol=1e-3`, `cwtol=5e-3`, and
BiCGSTAB-L order 2. Resident CUDA preserves that requested order and verifies
each apparent convergence with an independently recomputed `b-Ax` residual.
A failed check restarts from the current device solution using only the
remaining public iteration budget. Both the CPU and strict CUDA paths
reproduce the reference mode
`0.2344541314-0.0003147776i` within `5e-5`; the FP64 CPU branch retains its
original `tol=1e-6` oracle and precision.

Every Krylov vector publication must invalidate host-address-bearing CUDA
descriptors. The CW reset therefore still rebuilds boundary, structured-curl,
multilevel, and finite-check plans, but moves plain device mirrors into an
exact-size allocation pool. Reuse is exception-safe and keeps allocation/live
buffer accounting physical: the strict eigfreq regression performs 9,695,348
mirror reuses and 150,378 physical allocations, rather than over one million
plain-mirror allocations, and releases every live buffer at exit. The
integrated `gpu-step-db` CW gate also requires physical allocations to remain
below CUDA curl calls and reuse to exceed allocation.

The tiny eigfreq fixture is deliberately a correctness/coverage test, not a
release speed result. Shift-and-invert eigfrequency iteration now uses the
resident CUDA Krylov path when CUDA is selected; its FP32 true-residual policy
is described above. Automatic backend selection may still keep a small problem
on CPU, while strict CUDA validates nonzero CUDA work and zero CPU field
fallback.

The validation harness now supports recursively flattened, finite JSON
metrics with explicit per-metric or wildcard absolute/relative tolerances.
Duplicate manifest/metric keys, flattened-path collisions, unused tolerance
rules, missing metrics, and non-finite values are hard failures.
`faraday-rotation.py` emits four Ex/Ey samples and two field norms in
validation mode; its CPU/CUDA maximum sample difference is `6.44e-6`.
`multilevel-atom.py` preserves the published long-run defaults while a reduced
validation mode emits five cavity-field samples whose maximum difference is
`6.44e-10`. The BFAST notebook is classified as a duplicate of
`test_refl_angular.py`, which exercises BFAST and fixed-k reflectance at
multiple angles/frequencies against Fresnel theory. All four selected M7
Python cases pass with no unit-test skips and strict CUDA phase counters with
zero CPU fallback.

## M8 adjoint and inverse-design qualification

Adjoint workflows now have a dedicated FP32 CPU/strict-CUDA validation and
performance gate. JAX and the CPU-only `jaxlib` build are part of the exact
CUDA environment so automatic differentiation on the host does not introduce
a second CUDA toolchain.

`fields::dft_norm` no longer downloads each device-authoritative DFT array to
decide whether a monitor has decayed. A CUDA reduction accumulates contiguous
or persistent/indexed complex-FP32 monitor values into one FP64 scalar per
local DFT chunk. The persistent point map and scalar scratch allocation are
cached. Repeated norm checks therefore download only eight bytes per monitor
chunk and do not re-upload static metadata. The focused upstream DFT adjoint
case reduced device-to-host traffic from 510,888,848 to 3,759,736 bytes.
Earlier wall-clock observations of this decay-controlled fixture are excluded
from performance evidence: the pre-fix CPU and CUDA norm paths made slightly
different FP32 decay decisions and therefore executed different timestep
counts. Release evidence requires reduction-on/off runs to agree on decay work
and observables, while the speed gate below uses a fixed post-source duration.

The collective norm path also guards rank-local exceptions before MPI
reduction. A deterministic two-rank regression injects a rank-zero failure
and requires the communicator to abort promptly instead of leaving another
rank blocked in `sum_to_all`. Contiguous and persistent norms are compared
directly with CPU FP32 on one and two GPUs. The CUDA runtime uses native FP64
atomics, and architecture selection rejects targets below the supported
sm_60/compute_60 floor.

The numerical qualification includes:

- all 13 Cartesian upstream adjoint-solver tests on CPU FP32 and strict CUDA,
  with only the upstream FP64-specific `unfilter_design` test skipped;
- six JAX wrapper/custom-VJP adjoint tests plus two JAX utility tests, and all
  four cylindrical Near2Far adjoint tests, on CPU and strict CUDA;
- adjoint utility, wrapper, DFT, eigenmode, LDOS, damping, anisotropy,
  complex-field, multi-objective/frequency, periodic-design, and mapped
  gradient coverage;
- six deterministic inverse-design examples covering binary-grating and
  multilayer shape VJPs, Cartesian Near2Far epigraph objectives,
  solid/void connectivity and length constraints, a six-wavelength mode
  converter, and low/extreme-beta smoothed/tanh waveguide-crossing
  projections;
- an explicit coverage mapping for all ten checked-in adjoint notebooks using
  fixed-design objective/gradient oracles and identified subset or duplicate
  cases, without repeating long and trajectory-sensitive optimization loops.
  The connectivity validator deliberately corrects the notebook cell's
  endpoint-grid and fixed-boundary masking mistakes, so it is an
  intended-semantics regression rather than a byte-for-byte notebook replay.

The FP32 solver keeps the upstream finite-difference perturbation for the
general matrix. Only damping and mapped-gradient tests use a ten-times-larger
FP32 perturbation because their original objective differences sit at the
stopping/noise floor; LDOS retains the smaller step because a global increase
would make that difference nonlinear. The original per-test error tolerances
remain unchanged.

The release performance workload uses a fixed post-source duration so CPU and
CUDA execute the same physical work. The latest sealed pre-M8.5 baseline on
the current RTX 3090 Ti qualification host used three fresh processes per
backend for a complete 262,144-cell MaterialGrid forward-plus-adjoint
evaluation:

| Backend | Measured times (s) | Median (s) |
| --- | --- | ---: |
| CPU FP32 | sealed three-run sample | 17.2833 |
| strict CUDA FP32 | sealed three-run sample | 8.9862 |

The resulting speedup is 1.9233x, above the 1.25x adjoint release threshold
but still below the M8.5 optimization target.
Objective, gradient norm, gradient sum, and a fixed gradient projection all
pass their CPU/CUDA tolerances; every measured CUDA run reports zero CPU
phase calls. Warm-up runs are excluded from the median. Tiny forced-CUDA
correctness examples remain explicitly invalid as speed evidence because
Python startup, MPB work, host-side mappings, and launch latency can dominate;
normal automatic policy retains small rank-local simulations on the CPU.

The final release still requires the remaining general Python/example,
portability, and four-physical-GPU gates described elsewhere. Application
ports are outside the current build scope and require a separate user request.

## M8.5/M8.6 adjoint stepping optimization

The fixed adjoint performance workload now batches compatible curl and
constitutive-update commands without changing their field dependencies.  Its
sealed M8.5 evidence uses three fresh processes per backend, equal timestep
work, strict CUDA coverage, and all 1,681 gradient values.  On the two-RTX
3090 Ti development host it measured a 17.3144-second CPU median and a
7.1756-second CUDA median, or 2.4130x speedup.  The complete report is
`benchmark-results/m8.5-phase-fusion-sealed-20260804/report.json`; generated
evidence is intentionally outside the source distribution until release
bundling.

M8.6 reduces fixed per-step boundary and validation overhead while preserving
the existing host-visible and MPI semantics:

- cached CUDA graphs replay the ordered gather-then-scatter snapshot for
  aliasing local boundary plans;
- all-zero plans use one direct kernel, and source/destination-disjoint local
  plans use one direct non-aliasing kernel;
- remote MPI pack and unpack operate directly between disjoint field and
  exchange buffers, reducing each operation from two kernels to one;
- local direct execution is enabled only after an exact scalar-address set
  comparison proves that no destination is also a source anywhere in the
  plan. Periodic chains and permutations continue to use the snapshot graph;
- the per-step finite guard uses a 32-bit generation token.  Normal steps no
  longer launch a separate result-reset kernel, deferred `finish(false)`
  sessions retain their generation, and wraparound explicitly clears the
  device token. Exact duplicate finite spans are scanned only once.

The diagnostic switches
`MEEP_GPU_DISABLE_BOUNDARY_GRAPH`,
`MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO`,
`MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY`, and
`MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY` force the corresponding reference
path for same-binary A/B qualification.  They are diagnostics rather than
public compatibility promises.

The stored same-binary boundary-graph A/B sample reduced the fixed workload
median from 7.1114 to 6.8676 seconds (3.43%).  The independently repeated
two-rank pinned-transport direct/legacy comparison produced bit-for-bit equal
observables, phase counters, MPI traffic, and transfer bytes.  Its short
single timing sample is retained as functional evidence, not promoted to a
release speed claim.

M8.6 qualification rebuilt the isolated Python and MPI packages from one
source snapshot, passed the single-GPU default/graph/staged boundary paths,
passed both CPU-only precision suites (24 pass and one CUDA-only skip each),
and passed all 22 MPI tests on two RTX 3090 Ti GPUs.  Independent adversarial
audits found and closed duplicate finite-descriptor sizing, deferred-verdict
reset/migration, nested-session, graph-availability, and transactional remote
scatter defects.  The post-audit pre-documentation performance run measured
approximately 17.3 seconds on CPU and 6.7 seconds on strict CUDA: about 2.6x
for the forward-plus-adjoint-plus-gradient evaluation timer and about 2.35x
for fresh-process wall time.  All 1,681 gradient components passed.  The exact
documentation-snapshot measurements and hashes are authoritative in
`benchmark-results/m8.6-boundary-finite-audit-sealed-20260804/report.json`.

The remaining adjoint gradient phase is approximately 1.1 seconds on both
backends because it is mostly serial host geometry and material
differentiation.  It is the M8.7 optimization target rather than being hidden
as CUDA work. The final release still requires broader examples, portability,
and four-physical-GPU qualification; the present host's MPI uses the pinned
transport and correctly fails closed when CUDA-aware transport is explicitly
required but unavailable. Application-specific ports are deferred until the
user requests them separately.

## M8.7 scalar MaterialGrid gradient optimization

M8.7 removes the dominant host-geometry loop for one common inverse-design
case.  The optimized operation is an analytic evaluation of Meep's existing
central-difference inverse-permittivity expression; it is not a CUDA gradient
kernel and must not be reported as one.  CUDA still accelerates the forward and
adjoint FDTD phases, while this common host post-processing phase falls from
about 1.1 seconds to roughly 8 milliseconds.

The automatic path is deliberately narrow.  It requires a 2D Cartesian
electric/D-field Z-to-Z contraction, `U_DEFAULT`, `beta=0`, finite `u_p`, and
two positive isotropic scalar dielectric endpoints.  The endpoints must have
no dispersion, damping, conductivity, magnetic response, nonlinearity, or
off-diagonal tensor entries.  Both averaging-enabled and center-value
MaterialGrids are supported because the D2 scalar Z row has the same inverse
permittivity expression.  Geometry-interface voxels that can require the
Kottke derivative retain the legacy path.  Every other dimension, component,
grid combination, projection, tensor, or material model falls back pointwise
to the existing implementation.

`MEEP_MATERIAL_GRADIENT_PATH=legacy|auto|analytic-required` controls
qualification.  `auto` is the normal default, `legacy` forces the reference
implementation for same-binary A/B tests, and `analytic-required` aborts on
the first unsupported or singular point.  Invalid values also abort.
`MEEP_MATERIAL_GRADIENT_STATS=1` emits exact request accounting; under MPI the
printed record is now the sum across all ranks.  These variables are
diagnostic qualification controls rather than a permanent public API promise.

The implementation also corrects two pre-existing default-MaterialGrid
gradient defects: the grid-combination kind was uninitialized when no geometry
tree existed, and normalized lattice coordinates were incorrectly reused as
physical coordinates during the legacy material lookup.  A translated-cell
regression now compares legacy and required paths and also checks the adjoint
projection against an independent objective finite difference.  Subprocess
regressions require `analytic-required` to reject invalid environment values,
projection, 3D, non-Z fields, dispersion, conductivity, off-diagonal tensors,
and `U_MEAN`/`U_MIN`/`U_PROD`.

The seven-repeat pre-audit A/B sample measured a 155.7x CPU and 138.9x
CUDA-run host gradient-phase improvement.  End-to-end evaluation improved
from 17.3264 to 16.2386 seconds on CPU and from 6.6308 to 5.5096 seconds on
strict CUDA.  The optimized evaluation speedup was 2.9473x and fresh-process
wall speedup was 2.5970x.  All 1,681 gradient values passed, with maximum
cross-path absolute errors of 2.94e-8 on CPU and 1.49e-7 on CUDA.  This sample
motivated the retained optimization but is superseded for release sealing by
the post-audit, source-snapshot-bound reports under
`benchmark-results/m8.7-gradient-*-audit-sealed-20260804`.

The A/B postprocessor independently revalidates input COMPLETE markers,
source/build/runtime provenance, run counts, aggregate timings, objectives,
and every gradient element.  Its release gates use conservative worst-sample
ratios for the claimed gradient, CUDA evaluation, CUDA wall, and optimized
CPU/CUDA improvements so separate path campaigns cannot pass merely because
of median drift.  A failed standalone or combined gate leaves `FAILED.json`
and never publishes `COMPLETE`.

This optimization does not accelerate 1D/3D/cylindrical, TE/cross-component,
projected or overlapping grids, anisotropic/dispersive/conductive materials,
or general MPI gradient assembly.  Those cases retain correct legacy behavior
and remain explicit breadth targets; they are not counted as completed GPU
coverage.  In particular, upstream `U_MIN`/`U_PROD` overlap and distinct
overlapping design arrays have legacy semantic limitations that require a
separate implementation milestone.

## M8.9/M8.10 multi-GPU boundary scheduling

M8.9 groups the resident boundary zero, remote gather, and local-copy kernels
in a cached CUDA phase graph. The graph owns its send-ready and tail events,
is invalidated by allocation/plan generation changes, and is destroyed only
after its tail event completes. It is correctness-safe and retained, but the
receipt-bound `192^3` two-GPU A/B improved by only 0.334% at the paired
median. That experiment therefore failed its performance gate and is not a
material-speedup claim.

M8.10 addresses the actual CUDA-aware MPI scheduling gap. The communications
manager now prepares every callback and request slot transactionally, posts
all receives before the resident gather/local work, posts sends only after the
CUDA send-ready event has completed, and drains completions with
`MPI_Waitsome` by default. An opt-in same-binary `MPI_Waitall` experiment
defers all callbacks until every request completes. Completion callbacks
remain host-gated inside `comms_finish`,
so remote scatter is enqueued after the local graph/direct work. A two-slot
receive registry alternates independent device buffers; each slot retains its
own immutable scatter plan and completion event, and a slot is synchronized
before MPI can reuse it. The secondary slot is allocated lazily only when
CUDA-aware receive ping-pong is actually enabled; pinned staging and the
same-binary disabled reference path do not pay the extra device-memory cost.
Boundary phases with no physical messages skip the eager split-start state
machine entirely, and eager start counters advance only when at least one
request is posted. Pinned staging keeps its original ordered path.

The state machine rejects enqueue-after-post, send-before-receive, duplicate
posting, and reentrant finish. Qualification injects callback copies that
throw both while enqueueing and while preparing the physical plan, a callback
body failure, and an external unwind with a posted receive. It requires all
other requests/callbacks to drain and the manager to remain reusable; the
unwind case must terminate promptly through the explicit distributed abort
rather than hang or access destroyed callback state. Both CUDA-aware and
pinned transports run the failure/reuse suite.

The final receipt-bound six-pair completion-policy experiment is retained at
`benchmark-results/mpi-completion-ab/20260806T030701Z/`. Waitsome and Waitall
had exactly equal objectives, all 1,681 gradient values agreed to
`4.313e-8` maximum absolute and `1.004e-7` relative L2 error, and every CUDA,
MPI, runtime, and physical-device signature matched. Waitsome took
`9.925514` s at the workload median versus `9.941726` s for Waitall. The
position-balanced process-wall estimate was only `1.004099x` in Waitall's
favor, its two position strata disagreed materially, and Waitall won only two
of six pairs. It therefore failed every source-bound `1.02x` promotion gate;
Waitsome remains the default. An independent artifact audit approved the
complete evidence with C0/H0/M0 and only the documented lack of an external
cryptographic signature as L1.

`MEEP_GPU_DISABLE_EAGER_MPI` and
`MEEP_GPU_DISABLE_RECEIVE_PINGPONG` force the old same-binary reference paths.
`MEEP_GPU_EXPECT_EAGER_MPI` and
`MEEP_GPU_EXPECT_RECEIVE_PINGPONG` are qualification assertions backed by
machine-readable request, lazy-allocation, and buffer-selection counters.
`MEEP_GPU_EXPECT_NO_EAGER_MPI` and
`MEEP_GPU_EXPECT_NO_RECEIVE_SECONDARY_ALLOCATION` prove that fallback paths
retain zero eager requests and zero secondary allocations. The receipt-bound
qualification executes the actual libtool `.libs/lt-*` ELF files directly,
records their hashes and the hash/path of the loaded build-tree `libmeep`, and
stores all of those artifacts in the build receipt. These environment
variables are diagnostic controls, not compatibility promises.

On the two-RTX-3090-Ti development host, an alternating-order six-pair
`16^3`, 1000-step sample improved by 9.96% at the paired median; a three-pair
`192^3`, 80-step sample was effectively neutral at +0.068%. Every A/B
observable was exactly equal. These development numbers justify retaining the
default CUDA-aware path, but release reporting must use the later
receipt-bound direct ELF evidence. More than two ranks already use the same
correct receive/send state machine; peer-specific send-ready posting and
interior/halo compute overlap remain explicit performance work for a host
with more than two physical GPUs.

## M8.11 CUDA-aware boundary/E-H overlap

With the explicit experimental opt-in described below, a warmed, cached
CUDA-aware boundary topology can post B receives and sends before the
pointwise H constitutive update is submitted; D traffic can similarly overlap
with E. The receive callback scatters only into
non-owned halo locations after MPI completion, while E/H updates touch owned
locations. Boundary gather, E/H update, and callback scatter all use the same
CUDA default stream, so their device order remains gather, update, scatter
without requiring stream-aware MPI. Eligibility is rank-local: a rank which
cannot overlap performs its ordinary E/H update after communication, and the
next unchanged boundary collective provides the normal distributed ordering.

The first implementation is deliberately fail-closed. It requires strict
FP32 CUDA, a persistent resident phase, an unchanged material/chunk topology,
cached remote messages, eager CUDA-aware request posting, Cartesian fields,
initialized nonempty diagonal E/H updates, and no fluxes, solve-CW state,
polarization, integrated source, off-diagonal coefficient, nonlinearity,
previous-field auxiliary, D/B-minus-polarization scratch, or lazy E/H/PML
allocation. Pointwise diagonal PML is supported after allocation. Pinned MPI
and `MEEP_GPU_DISABLE_EAGER_MPI=1` do not post requests early enough to
overlap; they retain the original ordering and increment
`skipped_unsupported_schedule` rather than claiming a launch.

The optimization is **off by default**. Only the exact environment value
`MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1` opts in; unset, empty, `0`, or any
other value remains off. `MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP` is an
absolute override and wins if both variables are present. Expectation
variables used by qualification never enable the feature.
`MEEP_GPU_DEBUG_BOUNDARY_EH_OVERLAP=1` prints fail-closed reasons.
The additive `gpu::get_boundary_eh_overlap_statistics()` API and
`mp.gpu.statistics()["boundary_eh_overlap"]` expose checks, H/E launches,
disabled/unsupported/no-remote/cold skips, and predicate rejections without
changing the existing multi-GPU statistics ABI. The C++ benchmark also emits
the number of ranks which launched overlap in its machine-readable marker.

Fresh-build qualification compares every field probe, full-grid norm and
weighted checksum, energy, DFT norm, remote-cut norm, phase counter, transfer
byte, and message count exactly between enabled and disabled runs. It covers
complete source-free diagonal E/H overlap, a source-driven rank-asymmetric
eligible/rejected split, cold-topology fallback, pinned and eager-disabled
zero-launch schedules, and both Waitsome and Waitall completion policies.
The authoritative performance entry point is
`scripts/benchmark-boundary-eh-overlap.py`: one warm process per condition and
six `192^3`, 80-step pairs alternate enabled-first and disabled-first order on
two physical GPUs. Valid negative evidence still publishes `COMPLETE`; the
default is retained only if the median paired speedup is at least `1.01x`, at
least four pairs are faster, both position strata improve, and the 90% upper
enabled/disabled time-ratio bound is at most one. The report seals the exact
receipt, direct ELF, loaded `libmeep`, launcher, helper, raw logs, full child
environment, device mapping, continuous process monitor, report, and artifact
manifest.

The authoritative run in
`benchmark-results/boundary-eh-overlap/20260806T045719Z` produced exact
physical observables on all 14 samples and a median paired speedup of only
`1.001862x`. Four of six pairs improved, but the 90% enabled/disabled time
ratio interval was `[0.996631, 1.001305]`; the median and upper-bound gates
failed. Its sealed decision is therefore
`disable-boundary-eh-overlap-default`. Qualification keeps the code path for
experimentation, but directly proves default-off, strict opt-in parsing,
disable-overrides-enable, opt-in CUDA-aware execution, pinned/eager-disabled
fallback, rank-asymmetric eligibility, cold topology, and Waitall behavior.

## M8.12 final-H-halo / D-curl overlap

The experimental halo/curl path overlaps the final remote H boundary exchange
with the next D curl. It launches a conservative D interior while eager
CUDA-aware MPI requests are live, lets the receive callbacks enqueue the H
scatter, and then launches a disjoint D shell on the same CUDA stream. The
stream order is therefore interior, received-H scatter, shell. The ordinary D
source-time calculation remains in its historical location because the curl
does not consume source state; source application still follows the completed
curl exactly as before.

The implementation is **off by default**. Only the exact value
`MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1` opts in, and the presence of
`MEEP_GPU_DISABLE_HALO_CURL_OVERLAP` always wins. Unset, empty, `0`, and
invalid enable values remain off. Eligibility currently requires FP32 CUDA,
3D Cartesian D fields, eager CUDA-aware remote traffic, a warm resident phase,
no BFAST wavevector, warm PML/conductivity auxiliaries, a sufficiently large
interior, and the ordinary non-tiled D-curl layout. Explicit CPU loop tiling
falls back because splitting every small historical tile into face fragments
was measured to make descriptor traversal dominate the stencil. Pinned MPI,
an explicitly disabled eager schedule, cold topology, small domains, and
unsupported features all retain the original full-curl order.

`MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS` sets the nonnegative rank-local
interior threshold; the default is 65536. The additive
`gpu::get_halo_curl_overlap_statistics()` and
`mp.gpu.statistics()["halo_curl_overlap"]` APIs report checks, eligibility,
launches, every fallback classification, and full/interior/shell point counts.
Every completed launch must satisfy
`interior_points + shell_points == full_points`. Qualification assertion and
debug variables do not enable the feature.

The receipt-bound entry point is
`scripts/benchmark-halo-curl-overlap.py`. It uses two physical GPUs, eager
CUDA-aware Waitsome, a non-tiled `192^3` problem, 400 measured steps, and six
alternating-order pairs. Physical probes, norms, energy, DFT, remote-cut
values, and MPI traffic must be exact. Default promotion requires at least a
`1.03x` paired-median speedup, four of six faster pairs, improvement in both
position strata, and a 90% upper enabled/disabled time-ratio bound no larger
than one. A valid negative run is still sealed as `COMPLETE` and retains the
strict opt-in policy.

The authoritative negative result is sealed at
`benchmark-results/halo-curl-overlap/20260806T070440Z/` against build receipt
`c2732f4a6e777083de153869210a713e5658a170d80ecf1287949857e657d384`.
All physical observables and MPI traffic were exact in all 14 samples, every
measured phase remained CUDA-only, and the continuous two-device process
monitor passed.  The enabled path nevertheless lost all six pairs: its
paired-median speedup was only `0.979643x`, with enabled-first and
enabled-second stratum medians of `0.983279x` and `0.976008x`.  The 90% interval
for the enabled/disabled calculation-time ratio was `[1.013675, 1.026609]`.
The sealed decision is therefore `retain-halo-curl-overlap-opt-in`; this path
does not contribute to release speedup claims and remains off by default.

## M8.13 CUDA loop-tile coalescing

Historical CPU loop tiling invokes `step_db` and the anisotropic E/H update
once per small tile.  A resident CUDA dispatch must not repeat the complete
rank-local grid operation for every one of those CPU scheduling tiles.
M8.13 recognizes this CUDA-only case and coalesces the tile sequence back to
one full rank-local launch per field phase.  It preserves the historical loop
tile value in the workload identity and does not change CPU execution.
`MEEP_GPU_DISABLE_TILE_COALESCING=1` restores the legacy GPU behavior for
qualification and diagnostics.

The additive `gpu::get_tile_coalescing_statistics()` API and
`mp.gpu.statistics()["tile_coalescing"]` report curl and E/H coalesced chunk
phases plus their input-tile counts.  An optimized phase must consume more
than one input tile; the disabled and no-tile controls must report all-zero
counters.  Each phase/input pair must be jointly zero or consume strictly
more input tiles than emitted chunk phases.  C++ qualification compares
coalesced and legacy fields, energy,
DFT observables, and field-update counts exactly, proves both curl and E/H
paths in the broad `gpu-step-db` fixture, and proves physical observables and
MPI traffic exactly in the direct two-GPU worker.  The build also retains the
M8.12 default-off, invalid/empty/zero/conflict, tiled, pinned, eager-disabled,
small-domain, and BFAST negative cases.

The authoritative entry point is `scripts/benchmark-tile-coalescing.py`.
Its fixed profile uses two physical GPUs, a `192^3` workload, 12 warmup and 80
measured steps, six alternating enabled/legacy pairs, and six interleaved
no-tile references.  Promotion requires an `8x` legacy-recovery median, all
six enabled runs faster, both position strata positive, a 90% upper
enabled/legacy time-ratio bound no larger than `0.20`, and no-tile parity at
both the median and 90% upper bound.  Every sample must remain CUDA-only and
must exactly match all physical observables.  The evidence runner also fixes
one rank-to-GPU UUID mapping across the complete 21-process schedule.

The sealed run at
`benchmark-results/tile-coalescing/20260806T083100Z/` used receipt
`bfd1e031911dcf1ecdecd6879724b9265163dbaf9336f88aa9b27d12eadcf7db`.
That receipt records identical source-start and source-end aggregate hashes
`ea785f957a6583d5794f2adf1239edba5d55e969cda61aeafd2229146455a4a1`,
a per-file source-manifest hash
`0bdd00d127e00b1c69c3fbb252e332f767764787a65b5992b592a3098835789d`
over 833 files, 61 qualification logs, and a 22/22 C++ MPI suite.

All 21 benchmark samples had exact physical observables and passed the
continuous two-device process monitor.  Coalesced execution took `0.406679`
s at the six-run median versus `5.391226` s for the forced legacy path, a
`13.287107x` paired-median recovery with six of six wins.  The 90% interval
for the enabled/legacy calculation-time ratio was
`[0.074619, 0.075550]`.  The no-tile reference median was `0.404978` s; the
coalesced/reference median ratio was `1.000750`, and its 90% interval was
`[0.994903, 1.004997]`.  The sealed decision is therefore
`retain-gpu-tile-coalescing-default`.  These values quantify removal of a
pathological GPU scheduling regression; they are not an overall GPU-versus-
CPU speedup claim.

The post-evidence adversarial audit approved the milestone with C0/H0/M0/L2.
Both low hardening findings were closed in the runner: impossible E/H counter
pairs are rejected and cross-sample rank-to-UUID remapping is a hard failure.
Replaying all 21 sealed JSON records through the strengthened validators kept
the exact report values and decision and proved the fixed mapping from rank 0
to UUID `a352234bb1f55d7c88d2b2d34e2af850` and rank 1 to UUID
`cdd6d9212c7ae30732247d74c478a84f`.

## M9.1/M9.2 direct Python-example oracle

`scripts/python-validation/run_example_oracle.py` executes a checked-in
example while wrapping `Simulation.run`.  Most examples remain byte-for-byte
upstream; narrowly documented FP32 compatibility or result-retention edits are
allowed when their original physical work and interactive output are preserved.
Each registration pins the expected number of runs and at least one exact or
per-run bounded timestep/work contract or minimum physical signal. Dynamic
decay examples use one explicit range per run and cannot silently drop a
reference or structure trajectory. For every run the driver snapshots the timestep and
all CPU/CUDA phase-call counters before execution and immediately after it,
then separately records dispatch caused by the whole-cell energy and DFT-norm
probes.  A zero-work run, selected-backend run with no phase dispatch, or any
backend fallback fails before CPU/CUDA observables can be compared.

The emitted comparison record contains both aggregate checksums and direct
per-run vectors for timestep, timestep delta, Meep time, field energy, and DFT
norm.  Multi-run registrations require every applicable run, rather than only
the maximum signal, to exceed its physical-signal floor.  CPU/strict-CUDA
comparison therefore detects a missing or zeroed reference/structure run even
when a different run remains nonzero.

For examples whose defining result is a spectrum, coefficient, radiation
pattern, analytic error, or integration result, the driver captures selected
top-level numerical scalars/arrays from the completed `runpy` namespace. Every
result has an independent exact shape contract. NumPy-like shape and size are
checked before `tolist()`, preventing an oversized result from allocating a
large Python object graph. The driver rejects missing, empty, ragged,
nonnumeric, nonfinite, oversized, or shape-inconsistent data and
emits direct real/imaginary vectors plus shape/rank and redundant checksums.
Per-result minimum-L2 and maximum-absolute contracts prevent a mutually zero or
mutually wrong CPU/CUDA result from passing solely because both backends agree.

The first registered example was unchanged `straight-waveguide.py`: one run,
4000 final and incremental timesteps, and a minimum whole-cell field energy of
`1e-6`, and now directly compares every value of its 160-by-80 `Ez` field.
The registered set now also includes `bent-waveguide.py`,
`oblique-planewave.py`, `refl-quartz.py`, `cylinder_cross_section.py`,
`finite_grating.py`, and `binary_grating_oblique.py`, each with exact run/work
contracts and direct per-run CPU/CUDA observables.  The current development
binary produced straight-waveguide CPU energy
`2.8763389638078363` and strict-CUDA energy `2.8763389509291155`, an absolute
difference of `1.287872075650398e-08` and relative difference
`4.477469769228627e-09`.  Its run-only phase-call audit recorded 228000 CPU
calls or 51999 CUDA calls, while the post-run probes were independently
recorded as 32 CPU or 34 CUDA calls.  The tiny example is a correctness and
coverage case, not performance evidence.

The expanded direct-result set adds `binary_grating.py` (all 210 MPB angles and
transmittances plus a bounded/recorded direction-cosine overshoot),
`binary_grating_oblique.py` (all 59 reflected and 39 transmitted mode results
plus independent modal and Poynting closure), `refl-angular-kz2d.py` (three
kz-storage reflectances, independent Fresnel errors, and representation spread),
`mode_coeff_phase.py` (S/P complex coefficients and independent Fresnel error
bounds), `ring-mode-overlap.py` (independent `ω`/`2ω` trajectories, Harminv
frequencies, field energies, and cross-fields `integrate2` result), and
`antenna_pec_ground_plane.py` (raw 50-by-6 complex near2far E/H amplitudes, raw
radial flux, normalized radiation, and analytic error). `finite_grating.py`
pins the full 501-by-352 result and compares all 176,352 complex values in
addition to 257 deterministic samples and whole-array summaries;
`oblique-planewave.py` checks the complete
502-value line and an independent adjacent-cell Bloch phase relation. The
binary-grating compatibility helper rejects nonfinite values and overshoots
larger than 32 FP32 epsilons before clamping; its direct CPU hardening probe
recorded only `3.8125e-13` overshoot. The mode-phase edit only retains values
already printed by the tutorial. Dynamic DFT-decay
work is allowed a bounded CPU/CUDA timestep difference while its physical
arrays remain tightly compared.

The next feature-expansion registration adds `solve-cw.py`. NumPy 2 removed
the example's `np.complex_` alias, and the upstream `1e-8..1e-12` residual
sequence becomes solver-noise dominated in FP32. The compatibility path uses
`np.complex128` storage and four representable FP32 residuals `1e-2..1e-5`,
while preserving the original FP64 sequence. It requires monotonic convergence,
final field error below `0.02`, and compares every value in the phase-aligned
242-by-242 CW field and 242-by-242 time-domain DFT field. Direct CPU/strict-CUDA
correctness passes with zero fallback. The small strict-CUDA probe is slower
(`84.0827 s` CPU versus `130.6156 s` forced CUDA), and therefore is not speed
evidence; the actual first-timestep automatic policy selects CPU for this case.
The development record is in
`artifacts/python-validation/m9-feature-expansion-20260806/`.

The next feature batch adds six further upstream workflows. `3rd-harm-1d.py`
runs all 35 original simulations, compares all four 400-frequency spectra and
the 31-point harmonic sweep, and rejects loss of the perturbative
`P(3ω) ∝ chi3²` law. `metal-cavity-ldos.py` retains its original three-aperture
default and adds a bounded finite-Q validation profile which compares Harminv
frequency/Q, the dynamic second-run duration, direct `dft_ldos`, and Q/V
closure. `dipole_in_vacuum_cyl_on_axis.py both` executes cylindrical
`m=+1,-1,0`, compares all complex near-to-far E/H values for x and z dipoles,
and limits both analytic radiation-pattern errors below five percent.

`absorbed_power_density.py` compares every point of the 201-by-201 dispersive
`Dz`, `Ez`, and absorbed-power-density arrays and requires volume/surface
absorption closure below two percent. Its final hardened-source development
probes were `47.22 s` CPU and `13.21 s` strict CUDA (`3.575x`), with full-field relative-L2
errors below `6.5e-7` and power-density relative-L2 error `8.72e-4`.
`parallel-wvgs-force.py --validation` runs both MPB-derived mode parities and
requires opposite normalized Maxwell-stress force signs. `stochastic_emitter.py`
adds an optional deterministic per-dipole seed, retains the default Ag geometry
and resolution, compares all 64-by-4 trial spectra plus ensemble statistics,
and proves identical callback counts. It measured `81.23 s` CPU versus
`58.45 s` strict CUDA (`1.390x`) with spectrum relative-L2 error `2.68e-6`.

All six CPU/strict-CUDA comparisons pass with nonzero selected-backend dispatch
and zero opposite-backend fallback. The raw logs, `/usr/bin/time` records,
checksums, source/binary hashes, and retained failed stochastic resolution-30
attempt are under
`artifacts/python-validation/m9-feature-expansion-b2-20260806/`. The timings
are development diagnostics, not release evidence: they are single,
non-interleaved probes without a fresh receipt. Forced CUDA is deliberately
slower for several small fixtures; normal automatic policy keeps them on CPU.

## CUDA-resident FP32 CW Krylov solver

The CW bottleneck identified by the earlier audit is now removed for the
supported FP32 path. BiCGSTAB-L vectors, elementwise updates, field packing,
and operator publication stay on the selected GPU. Dot products and scaled
norms accumulate FP32 products into FP64 device reductions; only reduction
scalars and the final solution return to the host. Exact-size field mirrors,
curl sessions, and packed-vector plans remain resident across operator calls.

`MEEP_GPU_CW_SOLVER=auto|host|resident` controls this path. The default
`auto` selects the resident solver when the build is FP32, the fields owner
has selected CUDA, and the solve has no legacy flux monitor. This includes
shift-and-invert eigfrequency iteration. `host` is the compatibility/control
path. `resident` is fail-closed if those requirements are not met. Legacy flux
monitors remain on the host-vector compatibility path in `auto`, and an
explicit `resident` request for them is rejected rather than partially
accelerated.

FP32 recursive residuals were observed to understate the true `b-Ax` residual,
so the default uses a true-residual reliable restart every 100 BiCGSTAB-L
iterations on both CPU and CUDA. A 20-iteration interval passed the original
small fixtures but repeatedly discarded the `L=10` recurrence in the resonant
ring example; the 100-iteration policy reached independently recomputed
`9.72e-6` CPU and `9.24e-6` CUDA true residuals for a requested `1e-5` while
retaining the drift bound. FP64 retains the legacy no-restart default.
`MEEP_GPU_CW_RELIABLE_RESTART=auto|off|0|N` selects the policy, where positive
`N` is the interval. Solver selection, restart interval, and the optional
`MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL=0|1|off|on|false|true` setting must
agree across every MPI rank;
rank mismatch fails closed before the ranks can enter different collective
schedules. Resident setup also reaches a rank-wide success consensus before
the first Krylov collective. Resident shift-and-invert eigfrequency solves
instead use the whole remaining iteration budget as one batch: this preserves
their resonant recurrence while still verifying every apparent convergence
with a true residual and retrying false convergence within the original
budget. Host eigfrequency behavior is unchanged.

The post-audit 256-by-256 converged gate recorded true residuals of
`8.52e-6` CPU, `9.79e-6` host-vector CUDA, and `8.69e-6` resident CUDA.
All raw and phase-aligned full-vector relative-L2 errors were below
`1.39e-4`; raw physical probes and PML auxiliary values passed their pointwise
FP32 contracts. Timings were `18.3223 s`, `25.0018 s`, and `1.36657 s`,
respectively: resident CUDA was `18.30x` faster than host-vector CUDA and
`13.41x` faster than this serial CPU diagnostic. This is not yet the requested
formal multi-core CPU release comparison.

The revised strong-scaling diagnostic uses `L=2`, 100 fixed iterations,
reliable restart interval 10, approximately 1280-by-1280 cells, identical 848
CUDA curl calls per GPU, and the maximum wall time across ranks. Two repeats
gave median times `1.57637 s` on one GPU and `0.932343 s` on two pinned-staging
GPUs: `1.6908x` strong scaling and 84.5% parallel efficiency. This deliberately
unconverged fixed-work probe is throughput evidence only; correctness comes
from the converged gates above. The earlier unconverged `L=10` probe sometimes
produced a nonfinite dot product because high-order FP32 recurrence depended on
atomic reduction order; its failed logs are retained and it is not used for a
speed claim. The public requested `L` is preserved. Collectively detected
nonfinite Krylov arithmetic now returns the documented numerical-breakdown
status (`solve_cw` returns false), skips publication of the partial device
solution, restores the finite pre-solve field, and recommends changing the
requested order/reliable-restart interval or using the host Krylov fallback.
A rank-zero-only injected breakdown verifies serial and two-rank
failure/recovery on the same fields owner without an exception, `MPI_ABORT`,
or stale resident state.

After the rank-local work gate was hardened, a final identical fixed-work
closure recorded `1.57873 s` on one GPU and `0.929401 s` on two GPUs, or
`1.6987x` strong scaling and 84.9% parallel efficiency. Independent rank-min
and rank-max records were both exactly 848 in the two-GPU run.

Serial and two-rank full `gpu-step-db` runs now cover default `auto`, explicit
host/resident agreement, Cartesian and cylindrical `r=0` packing, global raw
and phase-aligned error reductions under unequal rank-local vector lengths,
rank-maximum timing, and persistent adjoint DFT publication through fields
destruction both directly and after `clear_dft_monitors()`. The fixed-work
scaling gate also requires the exact expected CUDA operator-dispatch count on
every rank using independent rank-minimum and rank-maximum reductions, so a
balanced global sum cannot hide missing work on one rank and an early FP32
Krylov breakdown cannot masquerade as a speedup.
A rank-asymmetric diagnostic failure injection exits via
`MPI_ABORT` without timeout. Direct rank-mismatch fixtures likewise prove
fail-closed consensus for solver mode, reliable-restart interval, diagnostic
mode, and requested BiCGSTAB order. The main evidence-tool suite passes
374/374.

`test_generic_example_oracle_state.py` compares two real Meep trajectories
with and without an intermediate energy/DFT probe.  Subsequent fields, DFT
accumulators, energy, and timestep agree under both CPU and strict CUDA.  The
driver tests additionally reject measurement-only dispatch and missing run
dispatch and cover restoration after normal exit, `SystemExit`, run failure,
and metric failure. At the end of that batch, the regression passed 171/171
Python-validation tests and 374/374 main evidence-tool tests. The original
oracle implementation re-audit closed all code findings at C0/H0/L0; the
expanded result-vector batch receives a separate adversarial audit before this
milestone is sealed.  Authority remains deliberately
pending until the broad M9 manifest stabilizes: a fresh isolated receipt and
official CPU/strict-CUDA harness `report.json` plus `COMPLETE` are still
required and older receipts must fail their source/install closure checks.

The third and fourth M9 batches continue that expansion without treating
small forced-CUDA fixtures as speed gates. Batch 3 covers all four stochastic
emitter branches, periodic-unit-cell versus explicit-supercell near-to-far,
two-pole material dispersion, and both cylindrical and 2D perturbation-theory
polarizations. Its 6,910 metrics and three adversarial audits are retained in
`artifacts/python-validation/m9-feature-expansion-b3-20260806/`.

Batch 4 adds complete material-phasing epsilon and Ez/Dz grids plus a
static-material negative control, densely sampled dispersive Al ringdown
through both 1D Absorber and PML, Ex/Ey/Ez closed-surface antenna near-to-far,
all four checked-in taper lengths, and all 20 propagating normal/oblique
`DiffractedPlanewave` orders. The material oracle verifies inverse-permittivity
interpolation and proves that changing coefficients alter the CUDA field
trajectory. Antenna power must be outward and radial. Taper reflectances are
bound to an independent FP64 CPU vector and only gauge-invariant MPB
magnitudes are compared. Each diffraction order is independently closed to
its analytic transverse/longitudinal wavevector and vacuum group velocity.
All 182,556 metrics pass in final-source diagnostics, every strict-CUDA run has
positive CUDA dispatch and zero CPU fallback, and that batch passed 171/171
plus 374/374 across the two test suites. Raw recaptured evidence is under
`artifacts/python-validation/m9-feature-expansion-b4-20260807/`.

This batch makes no performance claim: its CPU-first, cold-process elapsed
values are execution receipts and produced contradictory startup-sensitive
ratios for small workflows. At the time of this batch, CUDA covered the FDTD
and DFT accumulation while the Near2Far Green-function transform and MPB
eigensolves remained host-side. M20 subsequently moved the 3D Cartesian
Near2Far transform to CUDA, and M21 extends that implementation to 2D as
described below. The M21 cylindrical follow-on now implements the cylindrical
Green quadrature on CUDA; its source and non-hardware validation status is
described below. MPB remains explicit breadth and performance work. Small 1D
and symmetry-reduced workflows remain launch
dominated and normally stay on CPU under the automatic policy.

The sixth general-example batch adds four distinct physics workflows.
`grating2d_triangular_lattice.py` retains all four propagating diffraction
orders, complex mode coefficients, group velocities, dominant wavevectors,
vacuum-dispersion error, and the even/odd selection-rule ratio.
`polarization_grating.py` executes both uniaxial and twisted anisotropic
branches at three thicknesses and compares input flux, all five mode angles,
gauge-invariant mode-coefficient magnitudes, transmitted powers, the analytic
uniaxial efficiencies, and a seventh matched-thickness geometry that isolates
director twist from thickness. Its 14 FDTD/DFT trajectories therefore include
an independent twist-only metamorphic oracle. `zone_plate.py` retains every
complex E/H component from both radial and axial Near2Far scans, requires
nonzero electric and magnetic group norms, and constrains the two H/E norm
ratios to within 5% of the vacuum-radiation value. It independently checks
focal position and contrast with a 2-um axial-error limit.
`planar_cavity_ldos.py` executes cylindrical m=-1 and mirror-reduced 3D bulk
references plus three lossless-metal cavity thicknesses, comparing all LDOS
and Purcell factors with each other and with the analytic theory under an 18%
maximum relative-error gate. Together these fixtures exercise anisotropic
media, MPB mode decomposition,
cylindrical coordinates, Near2Far, dynamic stopping, symmetries, and
`dft_ldos`; all four direct CPU/strict-CUDA metric comparisons pass, with
positive CUDA dispatch and zero opposite-backend fallback.

The bounded fixtures are correctness tests, not uniformly speed tests. Forced
CUDA is slower for the tiny zone-plate and eight symmetry-reduced LDOS runs,
and the polarization workflow is dominated by host MPB/geometry work; normal
automatic dispatch should keep such launch-dominated trajectories on CPU.
The larger triangular-lattice diagnostic at resolution 40 completed the same
14,409 timesteps in `88.3454 s` on one CPU process, `67.2165 s` on eight MPI
processes pinned to the i9-11900K's eight physical cores, and `8.7733 s` on
one RTX 3090 Ti. That is `7.66x` faster than the physical-core CPU baseline in
Meep's reported in-process elapsed interval (`5.50x` for process wall time,
72.54 versus 13.19 seconds), while preserving identical stopping work and
diffraction orders. The revised automatic policy sends this 754,810-local-cell
case to CUDA while self-contained receipts with the automatic-policy overrides
removed show CPU dispatch and zero CUDA calls for the 13,932-cell zone plate,
39,032-cell polarization grating, and at-most-148,877-cell LDOS runs. These are
still single-run development diagnostics; repeated,
interleaved release benchmarks and a fresh sealed receipt remain required.
Raw direct and oracle logs are retained under
`gpmeep-evidence-notes/dev-tools/m9-b6-*` outside the source repository.

The seventh general-example batch adds the two Brillouin-zone-integration
radiation workflows. `antenna_pec_ground_plane_1D.py` executes eight complex-k
Ey trajectories with metallic ground-plane boundaries and compares seven
resolved polar samples with the independent two-element-array law; its bounded
resolution-40 fixture has a 5.75% analytic pattern error and retains positive
raw and peak radial fluxes. The Ex `dipole_in_vacuum_1D.py` fixture executes
33 unique complex-k trajectories on a seam-free 7x8 angular grid plus one
off-axis Ey rotational-symmetry control. It retains every raw and normalized
flux sample. Its duplicate-free analytic error against the independent dipole
law is 1.57%, and the Ey
rotational control residual is zero on CPU and 2.72e-8 on CUDA. Their direct
development CPU/strict-CUDA comparisons pass 203 and 731 flattened metrics
respectively with zero opposite-backend dispatch.

Two solver examples already covered by bounded tests are not redundantly
rerun. The existing `test_holey_wvg_bands.py` covers both the 19-interpolation
`run_k_points`
branch and the fixed-k Harminv branch with complex modal references, while
`test_ring_cyl.py` checks the cylindrical m=3 ring frequency, decay, Q, and
complex amplitude. All three underlying tests pass under both requested CPU
and strict CUDA; the retained backend statistics show only the requested
phase calls. `plot_radiation_pattern_dipole.py` is classified as host-only
compatibility work because it contains NumPy/Matplotlib postprocessing and no
Meep solver path. At the end of batch 7, nine non-notebook Python examples
remained oracle-pending.

These 1D trajectories are deliberately not GPU performance claims. Meep's
in-process elapsed interval for forced CUDA was 13.98 versus 3.52 seconds for
the PEC antenna and 42.24 versus 9.63 seconds for the revised dipole sweep.
Self-contained, override-free automatic-policy receipts instead select CPU for
every 1,768-cell and 972-cell trajectory,
record zero CUDA runtime probes and zero CUDA phase calls. Their reported
times are Meep in-process elapsed intervals rather than whole-process wall
times; automatic CPU completes in 3.56 and 9.65 seconds respectively. This is
the intended production behavior for launch-dominated work; larger workloads
remain the source of GPU speedup claims.

The eighth general-example batch covers phase-map generation and its use in a
complete metasurface-lens workflow. `binary_grating_phasemap.py` runs both
even and odd z polarizations at three duty cycles, preserving seven-frequency
transmission vectors and unit-complex phase factors. Its bounded fixture
executes 12 FDTD/DFT trajectories and requires passive finite transmission,
nontrivial duty-cycle variation, polarization separation, unit phasors, and
nonzero circular phase spread along the duty-cycle, frequency, and
polarization axes. The direct CPU/strict-CUDA comparison passes all 502
flattened metrics. The CPU circular-spread L2 norms are 3.0513 across duty
cycle and 2.4301 across frequency; the polarization phasor distance is 1.9058.
Automatic dispatch keeps all twelve 3,123-local-cell trajectories on CPU,
performs no CUDA runtime probe, and completes in 5.64 seconds; forced CUDA's
13.55 seconds is retained only as a correctness receipt.

`metasurface_lens.py` now has a bounded 20-trajectory oracle that derives a
nine-point transmission/phase map, constructs 121-cell and 301-cell lens
profiles, and performs two 60-point axial Near2Far focus scans. Phase samples
are unwrapped before interpolation, and each lens cell is selected by circular
phasor distance rather than branch-dependent scalar distance. The oracle
retains all 422 target and achieved phase phasors, duty cycles, and circular
residuals. The CPU phase range is 6.3656 radians and the maximum profile phase
error is 0.01372 radians. The two focus positions are 19.492 and 19.695 for a
target of 20, with peak-to-background contrasts 16.70 and 25.81. CPU and
strict CUDA pass all 6,029 flattened metrics; their focus positions are
identical and their peak intensities differ by less than 1.8e-6 absolute. The
automatic policy leaves the eighteen 2,520-local-cell unit solves and the
229,320-cell lens on CPU, then sends only the 569,520-cell lens to CUDA. It
records exactly one runtime availability probe and completes in 11.76 seconds.

A separate production-scale `--benchmark` mode supplies the performance gate.
It uses the same nine-point phase-map-to-lens workflow and one 801-cell lens,
with identical resolution, simulation duration, and physical outputs on an
i9-11900K's eight physical cores and one RTX 3090 Ti (GPU ordinal 0, UUID
`GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee`). The eight-rank CPU run is bound
one rank per core with one thread per rank. Meep's reported in-process elapsed
intervals are 100.5220 seconds for CPU and 25.7925 seconds for CUDA, or a
3.8973x speedup. The focus location is identical; the relative differences in
contrast, peak intensity, and phase range are respectively 2.73e-8, 2.11e-8,
and 5.78e-9. The maximum profile phase error is 0.01447 radians on both
backends. These are development measurements, not yet repeated release
benchmark statistics. Raw logs, hardware attestation, and hashes are retained
under `gpmeep-evidence-notes/dev-tools/m9-b8-*` outside the source repository.
The complete current hardening regressions pass 172/172 Python-validation
tests and 374/374 main evidence-tool tests. Seven non-notebook Python examples
remain oracle-pending after this batch.

The ninth general-example batch covers the two remaining stochastic
extended-source workflows. `stochastic_emitter_line.py` executes all 15
orthonormal cosine line-source modes and both 45-degree rotations of the first
two modes on flat and textured Ag-backed geometries. Every actual `amp_func`
callback is counted. The per-frequency quadratic-flux trace closes to
`2.34e-7` relative L2 or better on CPU and `2.14e-7` or better on strict CUDA;
the M=12-to-M=15 normalized ensemble changes by only 0.109%, and the texture
changes the complete modal matrix by 82.64%. The direct comparison passes all
3,099 flattened metrics. One-process cold observations are 208.8904 seconds
on CPU and 92.7503 seconds on one RTX 3090 Ti, a diagnostic 2.252x ratio.

`stochastic_emitter_reciprocity.py` retains the unique directional ODD_Z
forward eigenmode and backward plane-wave DFT-overlap paths. Grid-aligned
quadrature represents all 55 source positions with 28 unique textured
positions, directly checks an additional translated flat source and reflected
textured partner, and verifies the backward N+3 metadata topology including
zero-weight ghosts and half-weight periodic endpoints. Four representative
forward/backward spectra are also rerun at twice the runtime; their worst
relative change is 8.879%. Forward/backward normalized texture response closes
to 3.6933% relative L2 with a 4.1255% worst point on CPU and 3.6927%/4.1170%
on strict CUDA. All 5,046 flattened metrics pass with 107,004,381 CUDA phase
calls and no CPU fallback. Its 37-trajectory CPU and strict-CUDA observations
are 1,497.5526 and 804.5531 seconds, a diagnostic 1.861x ratio. These direct
cold-process observations are correctness diagnostics, not release speed
gates. The conservative `auto` policy still classifies these small rank-local
domains as CPU; explicit CUDA is beneficial for their unusually long time
horizons, and workload-aware automatic promotion remains a policy-tuning item.
`extraction_eff_ldos.py` is not redundantly swept because `test_ldos_ext_eff`
already exercises its cylindrical m=-1, mirror-reduced 3D, signed-flux, F/J
LDOS, and cylindrical/3D agreement paths plus an m=+1 control. Raw comparison
and dispatch evidence is retained under
`gpmeep-evidence-notes/dev-tools/m9-b9-*`. The updated hardening regression is
182/182 Python-validation tests; three non-notebook Python examples remain
oracle-pending after this batch: `disc_extraction_efficiency.py`,
`disc_radiation_pattern.py`, and `edge_emitter_2D.py`. The earlier count of
four became stale when `point_dipole_cyl.py` received a deterministic
validation oracle.

The final general-example batch closes those three entries. The materialized
Python inventory now contains 202 items: 131 direct runs, 24 examples covered
by an existing numerical test, 15 host compatibility cases, 31 notebook
duplicates, and one MPI-only case. No item remains `oracle_pending`.
`disc_radiation_pattern.py` retains 48 angles, all complex E/H components,
Cartesian and signed/magnitude radial Poynting flux, near/far power, and the
full stopping work. Its bounded cylindrical m=-1 run closes near/far power to
7.3653%. The direct CPU/strict-CUDA comparison passes all 1,171 flattened
metrics with 7,205 identical timesteps and no opposite-backend phase calls.

`edge_emitter_2D.py` executes x, y, and z dipoles at 14 fixed kz samples using
the real/imag kz formulation. Its 42 trajectories preserve the combined and
three independent face flux monitors, F/J LDOS state, final fields, complete
stopping work, and integrated spectra. Monitor additivity, terminal source
tail, air light-cone tail, positive extraction, and polarization separation
are independent gates. CPU and strict CUDA pass all 1,761 flattened metrics.
Their efficiencies are respectively 6.7400%, 15.8745%, and 19.7284%; the
largest terminal source and air-tail ratios are 1.34e-8 and 3.913e-4.

`disc_extraction_efficiency.py` executes the displaced center m=-1 source and
four off-axis radii at every m=0..14, for 61 real cylindrical FDTD/DFT
trajectories and exactly 780,800 timesteps. It retains every 40-angle complex
E/H far field, signed and magnitude flux, source flux, radiation flux, F/J
LDOS value, radial quadrature contribution, modal tail, and work counter. A
fixed 120-unit post-source interval bounds the validation path; the upstream
no-argument example retains its dynamic DFT-decay behavior. All requested
m=0..14 raw vectors remain in the comparison, while position totals reproduce
the published loop's inclusive 1% radiation-flux stop. CPU and strict CUDA
both select stop modes [6, 9, 11, 10] for the four radii; their integrated
extraction efficiencies are 0.89122574 and 0.89122568. An adversarial audit
rejected the earlier untruncated diagnostic efficiency of about 0.6798 because
unsupported high-m source self-terms inflated its integrated source flux.
The largest final-three-mode tail ratio is 5.531e-5. Negligible high-m fields
are excluded from H/E and direction tests with separately retained 0/1
activity masks; all raw fields remain part of the CPU/CUDA comparison. The
direction gate exposes only the nonnegative inward fraction, so a harmless
positive signed-radial minimum cannot be misclassified as inward power. The
modal source-flux comparison retains a 1% relative bound and a 2e-5 absolute
floor. The floor covers cancellation in physically negligible high-m tails:
the source-flux outlier that established it is the third off-axis radius at
m=14 (5.8933e-4 on CPU and 5.7414e-4 on strict CUDA), whose absolute
difference is 1.5195e-5 and whose contribution is below 9e-9 of the summed
modal source flux. Significant modal source fluxes remain governed by the 1%
relative bound.

These validation domains are intentionally small and are correctness, not
performance, fixtures. One-process development elapsed times for CPU versus
forced CUDA were 3.6261 versus 6.5861 seconds for disc radiation, 38.1412
versus 308.3026 seconds for the 2D edge sweep, and 179.2201 versus 1023.2969
seconds for disc extraction. The last CUDA observation overlapped the
validation-tool regression and is timing-invalid even as correctness
evidence. Automatic dispatch keeps these launch-dominated domains on CPU;
production speed claims continue to use the separate large-domain repeated
benchmark gates. The current infrastructure regressions pass 268/268
Python-validation tests and 775/775 main evidence-tool tests. Development raw
logs and hashes are retained under
`benchmark-results/m17-development-strict-cuda`; a fresh source-bound build
receipt and archived CPU/strict-CUDA comparison remain the authoritative
milestone gate. A failed first source-bound attempt is retained as explicitly
non-authoritative evidence rather than being silently deleted.

## Cached boundary-descriptor replay

Distributed CUDA boundary exchange keeps gather/scatter device plans across
time steps. The ordinary internal gather/scatter APIs still compare every
host descriptor on every call. This is required because a caller may mutate
an element of an existing descriptor array without changing the array's
address.

The `fields::step` cached-topology path can additionally provide an immutable
topology identity and a monotonic topology generation. The token has a private
constructor: `fields` and its private topology are the only production code
that can mint one. The integrated regression defines the one explicitly named
test-only friend instead of exposing a generic token factory. Once a full
validation has bound that token to a device plan, later calls skip the
O(boundary scalar) descriptor comparison. They still validate the unique
resident-cache allocation generations and active device. A changed token,
topology replacement, allocation-generation change, cold call, invalid token,
or generic null-token call returns to full validation and rebuilds stale
device addresses. Thus the optimization is a fields-owned capability, not a
pointer-equality heuristic available to arbitrary callers.

Topology generations use a compare/exchange saturating allocator. It issues
`UINT64_MAX` once and then permanently returns zero while leaving its counter
saturated. Since generation zero is ineligible for replay, even theoretical
counter exhaustion degrades all later topologies to full validation instead
of wrapping to an earlier capability value.

`MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY=1` disables this path and
forces full descriptor validation on every call. The additive
`gpu::boundary_descriptor_replay_statistics` getter/reset API separately
counts gather/scatter fast replays and full validations. The distributed
performance executable emits the same data as
`gpmeep-boundary-descriptor-replay-v1`; setting
`MEEP_GPU_EXPECT_BOUNDARY_DESCRIPTOR_FAST_REPLAY=1` requires every MPI rank to
prove exclusive cached scatter replay in the measured window. Pinned transport
can additionally set
`MEEP_GPU_EXPECT_BOUNDARY_GATHER_DESCRIPTOR_FAST_REPLAY=1` to require exclusive
gather replay. An opt-out A/B run sets both the disable variable and
`MEEP_GPU_EXPECT_NO_BOUNDARY_DESCRIPTOR_FAST_REPLAY=1`; every rank must report
zero gather/scatter fast hits and at least one full scatter validation. The
marker includes the effective `disabled` boolean and `disabled_ranks`; mixed
rank-local modes are rejected, so evidence cannot infer configuration from
counters alone. Gather replay can legitimately be zero when the
already-established CUDA boundary phase graph owns the complete send phase.
This change removes host validation work only; production speedup remains to
be established by a fresh receipt-bound multi-GPU A/B benchmark.

## Versioned large-grid scaling initial condition

The C++ multi-GPU worker has two explicit initial-condition profiles selected
by `MEEP_GPU_MULTI_INITIAL_CONDITION`. An unset variable preserves the release
and historical `trigonometric-v1` profile. Release controllers pass that value
explicitly. `affine-v1` is reserved for the development scaling diagnostic and
uses

```
0.375 + x / 128 + y / 256 + z / 512
```

to populate a positive, spatially varying, decomposition-independent Ez field
without evaluating transcendental functions at every Yee point. An empty or
unknown environment value is an error, and parsing occurs before grid,
structure, or field allocation. A startup regression guard checks two frozen
outputs from the historical trigonometric formula and two exact binary-fraction
outputs from the affine formula, rather than recomputing expected values with
the implementation under test.

For every normal record-emitting MPI workload, each rank locally parses the
grid size, warmup/measured step counts, tile size, BFAST, source, material,
boundary-lifecycle, preflight-control, and initial-condition identity. All
ranks then enter the same validity and min/max-unanimity collectives before
any grid, structure, fields, or profile-dependent vector is allocated. A
rank-local invalid value or a valid-but-different physical profile therefore
fails collectively instead of leaving peers in a later FDTD collective.

Changing the initial condition changes the physical observables. Therefore a
rank or optimization comparison is valid only when every sample has the same
profile. The development diagnostic uses `affine-v1` consistently for
192-, 256-, 320-, and 384-cubed screens; affine results must not be compared
numerically with older trigonometric records. The release benchmark remains
`trigonometric-v1`.

Worker output is fail-closed and self-describing. Schema/prefix v4 adds the
exact `initial_condition` field. The worker also emits exactly one
`gpmeep-initial-condition-v1` marker and one
`gpmeep-initialize-field-timing-v1` marker. The latter reports the number of
applications on every rank and the rank-maximum time spent in
`initialize_field`. Normal source-on and source-off runs apply the profile
once per rank; the boundary-graph lifecycle regression applies it twice. The
diagnostic requires one application and binds the marker, v4 record, and fixed
profile to the same `affine-v1` value.

The receipt-bound same-profile boundary-graph lifecycle qualification executes
a short two-rank workload and strictly requires `trigonometric-v1`, the exact
schema-v4 fixed-workload record, 16-cubed pixels, two warmup and two measured
steps, exclusive CUDA dispatch, non-vacuous remote-cut and MPI activity, and
CUDA-aware transport. It also binds ranks 0 and 1 to two distinct compatible
physical GPU UUIDs. The initialization markers must report two applications
per rank and a finite positive rank-maximum time. Normal source-on/source-off
release and A/B lanes continue to require one application per rank.

Three intentionally recordless lanes do not represent a normal workload
sample. The device-count query and distributed completion-policy preflight
exit before fixed-workload parsing. The test-only initial-condition preflight
participates in fixed-workload consensus, then exits before allocation after
emitting only its profile marker. None emits a v4 benchmark plus
initialization-timing pair. These exceptions do not weaken the rule that every
normal record-emitting lane must fail before allocation on invalid or mixed
rank-local workload identity.

Release and direct A/B consumers reserve the `gpmeep-` namespace: stdout (or
the merged direct-run stream) accepts only the exact current worker-prefix
allowlist, while the release controller rejects every such marker on stderr.
Unknown versions and mixed v3/v4 streams fail closed. Build-log comparators
also require one exact terminal
`gpmeep-qualification:<input-basename>:PASS` annotation before accepting the
strict initialization markers and physical records.

Initialization is outside the steady-state FDTD timer. Reports consequently
keep three distinct timing domains: worker `seconds` for measured FDTD steps,
initializer setup time, and end-to-end process wall time. Reducing initializer
setup time makes the 384-cubed diagnostic practical but is not a gpmeep kernel
speedup and must never be reported as one. The 384-cubed memory fit and setup
improvement require a new receipt-bound GPU run; source inspection alone is
not release evidence.

## M16 phase-batch production policy

This section records the historical M16 decision. It is superseded for curl
and constitutive E/H work by the M25 automatic policy below. Indexed source
batching remains under the M16 opt-in rule.

At M16, the cross-operation curl and constitutive E/H phase kernels remained
available only as experimental opt-in paths. Production CUDA stepping used
the structured per-operation kernels by default. The corresponding
experimental paths were exercised by setting
`MEEP_GPU_ENABLE_PHASE_BATCHED_CURL=1` and/or
`MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH=1`. Indexed source batching is
likewise opt-in through
`MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE=1`; only the exact value `1` opts in.
The legacy
`MEEP_GPU_DISABLE_PHASE_BATCHED_CURL` and
`MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH` controls, plus
`MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE`, take precedence over the matching
opt-in variable, which keeps causal and failure-injection tests unambiguous.

This policy is based on a receipt-bound two-repeat size screen using the same
M15 FP32 CUDA/MPI worker, affine initial condition, 12 warmup steps, and 80
measured steps. Disabling both cross-operation phase kernels improved every
tested absolute runtime: `1.314x/1.224x` for one/two GPUs at `256^3`,
`1.399x/1.311x` at `320^3`, and `1.450x/1.357x` at `384^3`. Every required
phase remained CUDA-only, every two-rank sample used two distinct physical
GPU UUIDs, and the maximum observable relative error was
`1.0531441879755109e-14`. The sealed development evidence is under
`benchmark-results/m16-phase-batch-size-screen-bd08f2d-20260810T194536KST/`.

The multi-GPU root-cause runner explicitly opts all three phase kernels in
before applying its disable controls. Its `default` condition therefore means
the historical batched reference within that diagnostic, not the M16
production default.

## M19 indexed source and deterministic LDOS integration

The M19 development diagnostic extends the phase crossover to indexed
sources. Every condition explicitly opts curl, E/H, and source batching in,
then disables exactly the feature named by the condition. The fixed v3 matrix
contains 234 samples: seven cubic sizes, one/two GPU ranks, three paired
repeats, five phase-screen conditions, and nine causal-anchor conditions. It
produces 84 paired effects and 12 hardware-bound policy candidates. Disabling
source batching must increase only source CUDA calls; curl and E/H calls must
remain identical. All other controls must retain the default source call
count, and every paired sample must preserve the fixed physical observables.
Source batching remains experimental until this repeated matrix establishes
a stable, hardware-bound crossover. No default policy is inferred merely from
lower launch counts.

The current v4 parser adds the exact
`gpmeep-phase-batch-policy-v1` counter schema to the worker stream. It does
not reinterpret the historical v3 evidence or change its explicitly forced
conditions.

The v3 diagnostic also freezes the source topology through
`MEEP_GPU_MULTI_SOURCE_PROFILE`. An unset variable preserves the historical
and release `single-ez-v1` workload. `dual-electric-v1` is diagnostic-only:
it retains the same Ez source and adds one Ey source with the same continuous
source time, plane, and unit amplitude. Empty or unknown values are errors.
The profile is parsed locally and agreed by every rank before any grid,
structure, fields, or profile-dependent allocation. Rank-local invalid or
different values therefore fail collectively during the fixed-workload
preflight.

Every normal worker run emits exactly one
`gpmeep-source-profile-v1:{"profile":...}` marker. Release evidence requires
`single-ez-v1`; the M19 v3 controller explicitly requests and binds
`dual-electric-v1` in every sample. This marker is separate from the v4
benchmark record so the established physical-record schema remains unchanged.
For every rank and repeat, disabling source batching must produce exactly
twice the default source CUDA calls while curl and E/H calls remain identical.
The complete matrix is rejected if that strict 2:1 causal relation fails.

Repeated indices in one public source profile are legal. The phase-batched
kernel therefore uses atomic subtraction for every contribution. Stable
source topology caches the descriptor and block-to-operation map, but source
profile replacement invalidates the host identity snapshots of every source
or cross-cache LDOS plan that captured the old device pointers. The retained
allocation is rewritten on the next launch; unrelated curl, E/H, finite,
DFT, and boundary plans remain reusable.

CUDA `dft_ldos` updates consume device-authoritative FP32 fields and source
profiles without publishing full fields to the host. Each nonempty reduction
uses two deterministic kernels. The first maps logical source blocks through
a cached `uint32_t` operation map, accumulates four FP64 channels into at most
1,024 fixed partial slots, and contains no result atomic. A fixed 256-thread
second stage reduces those slots in a stable order and either replaces or
adds Re/Im electric and magnetic results. Stable topology performs no H2D
copy and returns exactly four doubles (32 bytes) per update. Literal launch
statistics therefore report two kernels per successful reduction.

Each immutable source profile caches its amplitude L1 norm and updates that
cache on mutation or combination, so LDOS normalization no longer rescans
every source point on every timestep. Rank-local resident sessions are stored
directly in a reserved vector using a no-throw move constructor, removing one
heap allocation per local chunk and update. The explicit diagnostic/fallback
control `MEEP_GPU_DISABLE_RESIDENT_LDOS=1` forces the former full-field host
path; it is used for paired performance evidence and as a troubleshooting
escape hatch, not as the CUDA default.

The whole batch is shape- and bounds-validated before CUDA-side preparation.
Observable publication is failure-atomic: replacement descriptors are
uploaded to an inactive plan slot before an allocation-free metadata commit,
the caller result is staged until the D2H succeeds, and overlapping or
reentrant LDOS reductions are rejected. Safe validation-cache entries and
resident mirrors prepared before a later allocation failure may remain for
reuse, but the active plan and caller result do not change.
Two high-water plan slots bound retained descriptor storage and eliminate
allocation/free churn after both slots are warm; their combined capacity may
be up to twice the largest retained plan. Source/mirror replacement, reset,
and cross-cache destruction invalidate every LDOS pointer snapshot while
retaining the two result-cache plan allocations. Same-device allocation-
reusing reset also retains the address-independent reduction workspace.
Device migration and full cache destruction free both device-specific plan
slots and the workspace with exact live-buffer accounting.

The deterministic regression injects reentry before resident mutation and a
failure after inactive-slot upload but before publication. It requires the
caller result and active plan to survive, warms both slots, then checks 16
source-profile replacements with zero plan allocations. It also destroys and
recreates a non-result cache to reject stale cross-cache pointers. The
dedicated `MEEP_GPU_TEST_LDOS_MIGRATION_ONLY=1` lane requires two compatible
physical GPUs, is mandatory in the authoritative two-GPU qualification, and
migrates both result and non-result caches in both directions. Fixed-device
repeats are bitwise; cross-device values use the numerical contract because
bitwise portability across GPU architectures is not claimed. This migration
lane is lifecycle coverage, not multi-GPU scaling evidence. The process-long
LDOS mutex serializes reductions within one MPI rank; fatal CUDA runtime/device
failures are outside the recovery claim.

The focused duplicate-index regressions submit the repeated-index operation
alongside at least one independent destination operation. They therefore
exercise atomics in an actual multi-operation phase launch rather than the
singleton legacy source kernel.

The focused qualification modes are
`MEEP_GPU_TEST_PHASE_SOURCE_ONLY=1` and
`MEEP_GPU_TEST_LDOS_REDUCTION_ONLY=1`. The latter combines the low-level
32-repeat bitwise contract with four independent real Meep simulations using
full-volume electric and magnetic sources. Both one-rank and two-rank lanes
require CPU-FP32/CUDA-FP32 F, J, and LDOS agreement, real CUDA work on every
rank, exact two-kernel and 32-byte-result accounting, descriptor replay, full
field transfer avoidance, bitwise repeatability, and leak-free teardown. The
central qualification contract requires the four focused one/two-rank source
and LDOS logs plus the one-rank, two-device migration log before a build
receipt can be sealed. The migration qualification has no SKIP path.

`MEEP_GPU_TEST_LDOS_BENCHMARK_ONLY=1` runs five order-alternating paired
measurements on a fixed 64-cubed electric-plus-magnetic source workload. Each
pair compares the same CUDA FDTD state with resident reduction and forced
full-field host fallback, checks the complete F/J/LDOS payload, and requires
the resident path to beat the fallback by at least 5% in every repetition.
This focused reduction/transfer speedup is not an end-to-end FDTD speedup and
must be reported separately from the release benchmark matrix.

### M19 MPI-adjoint publication freshness

The release publisher treats current-run ownership as part of correctness,
not merely as a reporting convention. Every raw result, stdout/stderr stream,
and parent-timing record must be a regular non-symlink file directly inside
the canonical output directory with the exact run-ID-derived basename. The
raw qualification record now carries the controller run ID in addition to its
unique nonce, receipt, source snapshot, producer digest, lane, kind, and
iteration. The final closure independently reconstructs the fixed launch
command and clean environment, rechecks the stdout result pointer, and
recomputes validation, receipt/runtime, process-environment, lazy-import, and
material-gradient gates from the raw bytes. The execution controller also
captures every returned parent-wall duration, command/environment digest, and
artifact identity in an immutable run-instance ledger before caller-owned
summaries can be published. That ledger is sealed and its capability can
authorize only one publication attempt. Samples use an exact top-level schema,
and unreferenced top-level evidence files make the final closure fail closed.
The trust boundary is the audited CLI controller process: its source and
in-process memory are trusted. These contracts catch stale inputs, coordinated
caller-data mutation, accidental direct-publisher misuse, and publication I/O
failures; they are not a sandbox against arbitrary hostile Python already
executing inside that same interpreter or an OS account that can rewrite both
the program and its evidence. Source/receipt hashes and independent replay are
required to detect changes outside that boundary.

The installed finite-difference oracle has the same ownership contract. Its
test output is reparsed for the exact three-method inventory, terminal OK, and
five direction records, while a separate parent-timing record binds the run
ID, command, environment, timestamps, and duration. Publication begins only
from the matching RUNNING state. A validation or write failure removes every
success artifact and leaves a non-COMPLETE state. The success transition is
`RUNNING -> FINALIZING -> COMPLETE`: the authoritative `COMPLETE` file is
written while `state.json` is still FINALIZING, then `state.json` is replaced
with the exact same COMPLETE marker. Consumers require both markers to exist
and match, and both terminal markers plus the final report/Markdown hashes are
reread before the publisher returns.

The terminal marker uses schema v2 and contains two explicitly named epochs.
The report's two integrity records retain the hash of the exact
`prepublication-running` closure, including the then-current RUNNING-state
digest. After `report.json` and `report.md` are written, the publisher requires
an exact FINALIZING-state record bound to their digests and constructs a
`terminal-complete` manifest. That manifest hashes and sizes every final
regular file except the two self-referential marker files, declares the exact
top-level file and allowed runtime-directory sets, and requires `COMPLETE` and
`state.json` to be byte-identical. Missing, extra, reordered, duplicated,
symlinked, special, or modified entries fail closed. This avoids pretending a
prepublication RUNNING-state hash describes the later terminal directory.

`scripts/verify-mpi-adjoint-evidence.py` is the read-only consumer entry point.
It first verifies the build receipt and re-executes itself with the exact
receipt-bound Python interpreter. It then replays the terminal closure, every
raw sample and finite-difference artifact, launch command/environment,
controller ledger, both integrity epochs, numerical/performance comparison,
postprocessor source/driver binding, report, and Markdown from the immutable
bytes. The verifier is itself included in the runner source set and therefore
in the build receipt and postprocessor source hashes. Relative repository
arguments are canonicalized at the receipt trust boundary so `Path('.')` and
an already-resolved repository root have identical verification semantics.

### M20 Near2Far release-evidence authority

The M20 Near2Far performance bundle is a host-local, receipt-bound
qualification artifact. Its report deliberately binds the absolute measured
ELF, loaded libraries, build receipt, source tree, and evidence directory.
Consequently, copying the evidence directory to a different path or machine is
not a supported verification operation. This does not limit the portability of
the gpmeep source or package; a different machine must build its own receipt
and produce its own hardware qualification bundle.

Release verification is caller-authoritative. A consumer must request the
`release` tier and supply both the externally published 64-hex bundle ID and
the external release-attestation file. The verifier never decides whether to
apply those requirements from a mutable report field. The attestation declares
the `host-local-receipt-bound-v1` scope and binds the receipt, source snapshot,
report, raw manifest, and bundle ID. It is content-addressed but not a digital
signature: the final GitHub release/tag workflow must publish or sign the
expected bundle ID through its independent release channel.

All control JSON and Markdown files and all 64 raw evidence files are required
to be direct regular files with explicit individual and aggregate size bounds
before hashing or parsing. Release report, manifest, provenance, inventory,
lane, probe, and rank-record schemas are exact. The verifier reparses raw
rank data and recomputes timings, numerical errors, device ownership, and
performance gates rather than accepting the report summaries.

### M21 2D Cartesian Near2Far implementation

The M21 source implementation generalizes the retained M20 Cartesian plan to
carry an exact two- or three-dimensional cache key. In FP32 builds, the public
single-point, batched-point, and output-grid APIs now select CUDA for positive-
frequency 2D transforms as well as 3D transforms. They retain the same
rank-local DFT ownership, deterministic source-parallel partials, FP64 final
reduction, 13-double MPI cancellation record, fast-to-mixed retry, workspace
tiling, and one-/multi-GPU process mapping. A 2D periodic direction is limited
to X or Y and is represented by the same precomputed displacement/Bloch-phase
descriptor used by 3D.

The fast kernel evaluates the outgoing Hankel tensor with CUDA FP32 J0/J1/Y0/Y1
functions and forms H2 from `2*H1/(kr)-H0`, avoiding two additional special-
function calls per source/target/frequency interaction. Its error-evidence
channel combines a host-derived FP64-to-FP32 coordinate/material/kr bound with
kernel arithmetic error and NVIDIA's documented 9-ULP/small-argument and
2.2e-6 large-argument absolute bounds. H0/H1 argument perturbation is bounded
from their derivative identities with a Gronwall envelope, then propagated
through the H2 recurrence including reciprocal-argument error. An interval
that approaches the kr singularity selects mixed CUDA before launching fast.
The mixed retry retains FP32 DFT storage but evaluates coordinates, material
arithmetic, Hankel functions, partials, and final assembly in FP64. CUDA fast
math remains forbidden for release builds.

Current non-hardware validation includes successful CUDA 12.4 compilation and
linking of the standalone runtime, the CUDA+MPI FP32 libmeep and integrated
`gpu-step-db`, plus a separately configured `--disable-cuda` FP32 libmeep.
The device-independent malformed-plan regression passes, the CPU fallback
integration run reaches its expected CUDA-unavailable skip after completing
its CPU reference, and all 820 evidence-tool tests pass. The CUDA smoke test
contains independent FP64 Hankel references for all six electric/magnetic
source orientations. The integrated regression covers 2D scalar, batch, grid,
and public periodic/Bloch transforms. It includes an actual-error-versus-
published-bound sweep across small kr, the 8.0 implementation boundary, H2
cancellation-sensitive points, and direct-mixed 2048/extreme-material cases.
A 256-byte test ceiling forces target, frequency, and operation tiling while
requiring the ordinary fixture to remain on fast CUDA, and a same-owner 2D/3D
sequence verifies the retained-plan dimension key. Fresh-build qualification
also contains a dedicated two-rank test which requires two distinct device
identifiers, rank-local 2D CUDA work, periodic/Bloch agreement, exact 13-double
collective accounting, and correct participation by a rank with no local DFT
chunks.

M21 is not accepted from compile evidence alone. A restored NVIDIA runtime
must execute those regressions, sweep Hankel accuracy over the supported kr
range, measure single-point and production-sized batches on one and two GPUs,
and publish fresh receipt-bound evidence.

### M21 cylindrical Near2Far follow-on

The cylindrical follow-on extends the retained Near2Far plan with an explicit
cylindrical dimension key and moves `greencyl`'s nested azimuthal trapezoidal
quadrature to CUDA. Each source/copy/target/frequency interaction remains
independent across CUDA threads while preserving Meep's original quadrature
reuse order. Radial and azimuthal equivalent currents are rotated into the
Cartesian Green tensor, axial currents remain unrotated, and the integrand
retains the exact `exp(i*m*phi)` mode phase. The public scalar, batched-point,
and output-grid APIs accept cylindrical targets, all six electric/magnetic
R/P/Z source orientations, nonzero positive or negative frequencies,
periodic-Z/Bloch copies, and
rank-local MPI execution.

The fast path evaluates geometry, phase, Green arithmetic, and quadrature in
FP32, publishes an inflated per-interaction error envelope, and requests the
existing mixed-CUDA retry if quadrature does not converge or collective
cancellation exhausts the error budget. The mixed path retains resident FP32
DFT storage while evaluating geometry, Green arithmetic, quadrature, partials,
and final assembly in FP64. Mode and tolerance are kernel arguments rather
than static descriptors, so changing `m` does not cause a descriptor upload;
changing between 2D, 3D, and cylindrical does invalidate the exact plan key.
Distributed execution derives `m` from the first rank with monitor work and
requires all other owning ranks to agree before the result collective, which
keeps a chunkless rank on the same physical transform.

The compact fast evidence stores a maximum-component Green envelope while the
`greencyl` stopping rule sums refinement deltas over all six complex field
components. Its host-side truncation allowance therefore carries an explicit
factor of six; because the device quadrature uses one quarter of the public
tolerance, the published term is `6*(tol/4) = 1.5*tol`. Nonconvergence makes
the evidence infinite and forces the mixed retry. Regression references use
an independent `1e-10` FP64 quadrature tolerance while sweeping both `1e-6`
and Meep's default `1e-3` device tolerances, and a cancellation fixture at the
default tolerance must execute one fast attempt followed by one mixed attempt.

The partial kernels are compile-time-specialized by dimension rather than
carrying the cylindrical branch through the established Cartesian binaries.
CUDA 12.4 `sm_86` resource diagnostics report no local-memory spills: fast
2D/3D/cylindrical specializations use 104/100/143 registers and mixed
specializations use 120/120/238 registers. Cartesian launches therefore keep
their 256-thread shape without inheriting cylindrical register pressure; the
cylindrical path uses 128 threads so its independent quadratures expose more
resident blocks on register-limited devices. These are static resource facts,
not a substitute for restored-device timings.

Validation includes successful CUDA-runtime, CUDA+MPI-integrated, and CPU-only
compilation, device-independent malformed input tests, independent FP64
cylindrical references for m=-2,0,3 and all six source orientations, and
public scalar/batch/grid fixtures. A 256-byte ceiling
forces cylindrical target/frequency/operation tiling while checking descriptor
reuse. The fresh-build contract now includes a dedicated two-rank lane which
requires distinct physical GPU identifiers, periodic-Z/Bloch agreement,
rank-local CUDA work, exact 13-double accounting, and a chunkless-rank CPU
oracle. The cylindrical standalone smoke deliberately launches both fast and
mixed transforms with the production 128-thread shape, covering the dynamic
shared-memory reduction layout rather than relying on the Cartesian
256-thread launch. Twelve Python Near2Far examples, including on-axis and
off-axis cylindrical dipoles and the 61-trajectory disc-extraction case, now
require a nonzero CUDA transform counter so a host Green-transform fallback
cannot pass strict validation. Physical qualification on two distinct RTX
3090 Ti devices now passes the D2 and cylindrical rank-local CUDA, CPU-oracle,
periodic/Bloch, and chunkless-rank contracts. The receipt is preserved under
`artifacts/near2far/m22-adjoint-vjp-qualification-20260814`. These short
correctness-test timings are not performance speedup measurements; one- and
two-GPU performance acceptance remains a separate benchmark milestone.

### M22 Near2Far adjoint VJP

`dft_near2far::near_sourcedata`, which constructs adjoint source profiles for
Near2Far objectives, now has a strict CUDA implementation for 2D Cartesian,
3D Cartesian, and cylindrical monitors. Surface quadrature, stored monitor
weight, symmetry multiplicity, electric-current sign, and the historical
cylindrical radial correction are folded into source descriptors once. One
cooperative block then contracts all six complex `dJ` components and every
periodic/Bloch image for one source/frequency output. No full far-field tensor
or DFT array is published to the host and strict CUDA validation rejects a
CPU VJP fallback.

The normal path uses FP32 descriptors, Green arithmetic, phase factors, and
contractions with deterministic block reduction. Its L1 envelope includes
input conversion, geometry/direction, material, phase, Green-function, and
cylindrical-quadrature error. Destructive cancellation or non-finite evidence
replays the complete VJP with FP64 geometry/arithmetic on CUDA. An exact
retained-problem match remembers that mixed decision, avoiding a redundant
FP32 probe on an identical subsequent call.

All source, target, frequency, periodic-copy, gradient, output, and condition
buffers share the 64 MiB Near2Far workspace ceiling. If a full launch does not
fit, the backend automatically tiles all four descriptor axes and accumulates
result and L1 tiles in FP64 device storage. The retained plan is keyed by the
monitor's DFT-list owner, reuses allocation capacities, and skips unchanged
source/target/frequency/copy and identical-gradient uploads. Monitor removal,
device migration, backend reset, and a reduced workspace ceiling invalidate or
shrink that state before any stale pointer can be launched.

Standalone CUDA smoke coverage compares fast and mixed D2/D3/cylindrical VJPs
against independent FP64 references and explicitly splits target tiles to
exercise device accumulation. The integrated 3D test checks CPU agreement,
forces a 512-byte multi-launch workspace, proves identical-call zero-H2D
descriptor replay, and forces cancellation-to-mixed retry. The existing
two-rank D2 and cylindrical qualification lanes now also require rank-local
adjoint CUDA calls, distinct physical GPUs, periodic/Bloch term accounting,
and per-rank CPU agreement. These paths compile with CUDA 12.4, CUDA+MPI FP32,
and CPU-only configurations. On two RTX 3090 Ti devices, the D2 lane passes in
0.277512 s and the cylindrical lane passes in 0.76464 s, including forward,
adjoint, CPU-oracle, and chunkless-rank coverage with no strict CUDA fallback.
The final adversarial M22 audit reports no remaining Critical, High, or Medium
finding. Performance acceptance remains pending the dedicated benchmark suite;
the qualification-lane elapsed times must not be reported as speedups.

M23 closes the remaining Near2Far breadth item by supporting cylindrical
monitors evaluated at arbitrary 3D Cartesian observation points.

### M23 cylindrical-monitor Cartesian observations

The Near2Far implementation now carries two dimensions explicitly: the
monitor dimension selects cylindrical Green functions, source directions,
periodic copies, mode number, and error tolerance, while the observation
dimension selects the target coordinate system and output field layout. A
cylindrical monitor may therefore target any finite `(x, y, z)` point in D3.
The CUDA selector uses `hypot(x, y)` only for conservative radial-domain and
error-bound decisions; the kernel receives and evaluates the original signed
Cartesian coordinates. Other cross-dimensional combinations fail before a
CUDA launch.

Python retains the historical phi=0 projected behavior by default. Passing
`cartesian=True` to `get_farfield`, `get_farfields_points`, `get_farfields`,
or `output_farfields` preserves D3 target coordinates and returns global
Cartesian field components. `Near2FarFields(..., cartesian=True)` applies the
same convention to both its forward objective and adjoint VJP.

MPI batch and grid calls now include the monitor dimension in their metadata
consensus. A rank-local disagreement fails closed before any result or
azimuth collective. Standalone and integrated tests cover negative x,
nonzero y, m=-2/0/3, all E/H source directions, scalar/batch/grid transforms,
rotation covariance, adjoint VJP, negative-R periodic images, FP32-to-mixed
retry, workspace tiling, and retained-plan lifecycle.

On one RTX 3090 Ti, the standalone cylindrical fast-FP32 relative errors were
1.44612e-06, 1.63123e-06, and 1.52114e-06 for m=-2, 0, and 3; the cylindrical
adjoint error was 1.20385e-06. The focused integrated regression passed in
3.55662 s. A two-rank/two-GPU periodic-Z/Bloch contract passed in 0.785092 s
with distinct physical GPUs, rank-local CUDA, a chunkless rank, and CPU
agreement. The Python public API passed 7/7 tests on one GPU and 7/7 on each
of two MPI ranks. The independent adversarial audit reports no remaining
Critical, High, or Medium finding. These short qualification timings prove
dispatch and correctness only; performance acceptance remains a separate
production-scale benchmark milestone. The reproducible receipt is preserved
under `artifacts/near2far/m23-cyl-cartesian-qualification-20260814`.

### M25 automatic curl/E/H phase batching

Unset curl and constitutive E/H phase controls now select an automatic,
topology-aware policy. The exact value `1` for
`MEEP_GPU_ENABLE_PHASE_BATCHED_CURL` or
`MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH` still forces the corresponding
cross-operation kernel. Any other present enable value disables that path,
and the matching `MEEP_GPU_DISABLE_*` variable takes precedence whenever it
is present. Indexed source batching is deliberately unchanged and remains
exact-opt-in only.

The automatic decision is made independently for every collected phase from
the work that will actually launch. It requires at least four operations,
sums `ceil(point_count / 256)` over those operations, and batches when that
logical-block count is no greater than
`operation_count * multiprocessor_count * 24`. The selected CUDA device's SM
count supplies the portable scale. The largest individual operation must also
be no greater than `multiprocessor_count * 24` blocks, preventing a few tiny
operations from hiding one pathologically large chunk behind a favorable
average. There is no GPU UUID, model name, pixel count, or rank-count table.
Invalid inputs and arithmetic overflow reject batching. A rejected phase
invokes the already established structured CUDA kernels sequentially rather
than dispatching any CPU field work.

`meep::gpu::get_phase_batch_policy_statistics()` and
`mp.gpu.statistics()["phase_batch_policy"]` publish automatic checks,
selections, rejections, forced batches, batched operations, and unbatched
operations separately for curl and E/H. The fixed MPI performance worker
reduces the same 12 counters into the exact
`gpmeep-phase-batch-policy-v1` marker. This makes a performance result prove
which policy executed instead of inferring it from a launch-count reduction.

The dedicated `scripts/benchmark-phase-batch-policy.py` controller compares
automatic, disabled, and forced modes at 128-cubed and 256-cubed on one and
two GPUs. It rotates condition order across six repeats so every condition
occupies each execution position exactly twice, holds indexed source
batching and every other worker setting fixed, requires exact equality of all
recorded physical observables, rejects any CPU field dispatch, binds each MPI
rank to a distinct compatible GPU UUID, archives the receipt-bound worker,
library, MPI launcher, fixed Open MPI policy, controller dependencies, and
raw sample streams. The archived runtime files are opened once with
read-only/no-follow descriptors and every launch uses their fixed
`/proc/<controller-pid>/fd/<fd>` images; descriptor identity and content are
rechecked before and after every sample, and a loader preflight proves that
the fixed library image was preloaded. Rank/PID markers bind continuous GPU
process-monitor rows to the exact workers and reject even an unrelated
same-name process. The recursive SHA-256 manifest excludes terminal markers,
and `COMPLETE.json` is published last while binding the manifest, summary,
and report hashes, so COMPLETE and FAILED cannot coexist. Its predeclared performance gates
require at least a `1.15x` median paired disabled/automatic time ratio at
128-cubed and no more than a `1.02x` median paired automatic/disabled or
automatic/forced time ratio at 256-cubed.

A preliminary three-repeat RTX 3090 Ti development run, before final receipt
sealing, measured disabled/automatic median ratios of `1.43783x` on one GPU
and `1.31588x` on two GPUs at 128-cubed. The corresponding 256-cubed ratios
were `1.00307x` and `1.00111x`; automatic mode avoided the one-GPU forced-mode
time increase of `1.01799x`. Every compared checksum and DFT norm was exactly
identical. These values are implementation-development evidence only until
the receipt-bound controller completes and its evidence anchor is committed.

### M26 resident multi-monitor DFT batching

`fields::update_dfts` already collected every rank-local monitor update before
entering the CUDA backend, but it historically launched one update kernel per
DFT chunk. M26 retains that global collection and executes independent monitor
arrays in one physical CUDA launch. Each logical block is mapped to exactly one
monitor descriptor and one point/frequency tile; output arrays must be pairwise
disjoint and must not alias any field, index, weight, or frequency input. No
atomics or cross-monitor reduction are used, so every complex output receives
the same single FP32 multiply/add update as the unbatched kernel.

Per-timestep host preparation is subquadratic in the monitor count. Output
and input alias intervals and repeated mirror extents are sorted and swept in
`O(N log N)`, rather than compared pairwise. Phase keys hash the exact scalar
bits and every frequency byte once, then re-run the full bitwise equality
check within a matching hash bucket; hash collisions therefore cannot share a
phase vector incorrectly, while ordinary expected grouping cost is
`O(total frequencies)`.

The phase vectors for distinct `(omega,time,scale)` keys, the operation
descriptors, and the exact logical-block map occupy one dedicated resident
plan. Only phase values are refreshed each timestep. An unchanged topology
reuses descriptor and map bytes without a host-to-device copy; a changed
topology is uploaded into a replacement allocation and becomes active only
after both metadata copies succeed. The previous plan remains intact on an
allocation or copy failure. Monitor/cache destruction releases the retained
allocation and live-buffer accounting includes it.

With both controls unset, the automatic policy currently records a check and
rejects the fused path. Request count alone is not a defensible crossover
model for heterogeneous monitor shapes, descriptor pressure, and 64-bit tile
decoding, so default selection remains fail-closed until the sealed
cross-size, one/two-GPU matrix establishes a no-regression boundary. Exact
`MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH=1` forces the fused kernel. Any other
present enable value disables it, and the presence of
`MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH` always wins. Disabling this feature
still uses the existing resident CUDA DFT kernels and phase sharing; it never
selects CPU work.

`meep::gpu::get_dft_batch_statistics()` and
`mp.gpu.statistics()["dft_batches"]` distinguish logical updates, physical
update launches, phase preparations/reuses, automatic decisions, forced
batches, fused/unfused updates, resident-plan uploads/reuses, and metadata
upload bytes. The MPI worker additionally emits
`gpmeep-dft-multi-monitor-batch-v1`; its accounting requires physical launches
to equal selected batches plus individually launched updates.

Device-independent policy and malformed-input checks, CUDA 12.4 compilation,
the integrated Meep CUDA+MPI build, and sealed-runner parser tests pass without
a working GPU driver. The standalone smoke covers five heterogeneous monitors,
including 513 frequencies, 257 points, real/complex fields, and every averaging
shape, with guarded outputs and independent FP32 references. The integrated
regression compares automatic, forced, and disabled modes bit-for-bit across
complete real and complex monitor arrays, phase-key mutation, multiple
resident caches, and representative flux/energy/force/Near2Far consumers.
Those CUDA executions and the performance/no-regression gate remain pending
driver restoration; compilation alone is not accepted as runtime evidence.

The isolated M26 host gate uses a relative-source out-of-tree FP32 build so
the upstream HDF5 fixtures remain addressable. It reports 24/24 CPU tests
passing and one expected CUDA-only skip. A separate CUDA+MPI+Python FP32 build
successfully compiles and links the current CUDA runtime, Meep library, both
GPU integration workers, and the SWIG extension with compute capability 8.6
embedded without querying the unavailable driver. The focused Python GPU API
suite reports 7/7 tests successful (three runtime-dependent skips), the
validation-harness suite reports 270/270, and the evidence/benchmark-tool
suite reports 842/842. A dedicated no-driver CUDA host test also validates the
exact fused-DFT descriptor image, fixed thread/tile geometry, contiguous block
prefixes, required pointers, and every packed-output size overflow boundary.
These results prove host behavior, build integration, and parser contracts
only; they do not replace the pending device executions.

### M27 sealed multi-monitor DFT qualification

M27 adds a dedicated production worker workload and an authoritative A/B
controller for the fused M26 path. `MEEP_GPU_MULTI_DFT_MONITORS` and
`MEEP_GPU_MULTI_DFT_FREQUENCIES` select positive counts capped at 256 and
1024, respectively. Every MPI rank must agree on both values before any field
allocation, and `gpmeep-dft-workload-v1` records the exact two-component
monitor identity used by the timed run. The test-only
`MEEP_GPU_TEST_DFT_WORKLOAD_PREFLIGHT_ONLY=1` lane returns before field or CUDA
allocation; it exists only to validate parsing, limits, marker schema, and
distributed unanimity when a CUDA driver is unavailable.

`scripts/benchmark-dft-multi-monitor.py` predeclares two workloads on both one
and two GPUs. The 32-cubed launch-dominated case uses 64 monitors, two
frequencies, and eight alternating forced/disabled pairs. It requires at
least 10% median paired speedup and a 90% upper confidence bound of at most
0.95 for the forced/disabled time ratio. The 96-cubed frequency-heavy case
uses eight monitors, 32 frequencies, and six alternating pairs; it requires
at least 3% median speedup and a 90% upper time-ratio bound of at most 1.02.
One fail-closed automatic-mode control follows each topology/workload matrix,
for 60 total process launches. Every synchronized measured interval must last
at least two seconds so the one-second continuous GPU process monitor can bind
each emitted worker PID. Two-rank samples require exclusive CUDA-aware MPI;
all phase counters require CUDA and no CPU dispatch.

The forced lane records the cold resident-plan upload and metadata transfer in
the untimed warmup marker. After the worker resets dispatch statistics, every
timed forced batch must be a warm-plan reuse with zero plan uploads and zero
metadata transfer. Disabled and fail-closed automatic lanes must report zero
warmup and measured plan activity. This separates one-time setup cost from the
declared steady-state speedup without losing evidence that the plan was really
created.

Every physical probe vector, norm, weighted checksum, energy, DFT norm, and
remote-cut observable must be exactly identical across all forced, disabled,
and automatic samples within one topology/workload. The controller rejects
unknown markers and duplicate required DFT, workload, initialization,
device/process, and communication markers, non-finite values, partial schemas,
counter inconsistencies, missing resident-plan reuse, PID reuse, and any
continuous process-monitor row not bound to an exact worker-emitted PID and
GPU ordinal. It loads its own source through an immutable byte snapshot,
archives the exact loaded helper and provenance sources, and executes archived
worker, libmeep, MPI launcher, and Open MPI policy images through persistent
read-only file descriptors verified before and after every sample. A terminal
`COMPLETE.json` is written only after the manifest, summary, and report exist;
success and failure markers cannot coexist.

With the NVIDIA driver unavailable, the current CUDA+MPI+Python FP32 tree
compiles and links the updated worker. Twenty-three focused Python contract
tests, the complete evidence/benchmark-tool suite at 865/865, and the Python
validation-harness suite at 270/270 pass in the isolated CUDA/MPI environment.
The allocation-free MPI suite passes all seven one/two-rank cases:
both declared workload identities, monitor and frequency upper bounds, a zero
monitor count, mutually exclusive preflight modes, and a distributed
preflight mismatch. These checks do not claim CUDA execution or performance.
The 60-sample forced/disabled/automatic GPU receipt remains explicitly pending
driver restoration.

### M28 isolated CPU FP64 MPI+Python reference

The release workload matrix requires a CPU-only FP64 MPI+Python package built
from the exact same source snapshot as the shared CUDA-capable FP32 package.
The older `build-cpu-baseline.sh` intentionally omits MPI and Python and cannot
serve as that reference. M28 adds
`scripts/build-meep-cpu-mpi-python-fp64.sh`, which creates a separate
`.envs/gpmeep-cpu-mpi-fp64` prefix from the exact pinned MPI/Python dependency
lock and configures Meep with explicit `--disable-single`, `--disable-cuda`,
`--with-mpi`, `--with-python`, and `--without-scheme`. Using the shared
dependency lock removes MPI, HDF5, Python, and numerical-package drift; the
resulting Meep binary and receipt are nevertheless required to be CPU-only.

The builder preserves any previous environment, build tree, and install tree
under timestamped names. It checks the pinned Micromamba digest, exact package
lock, isolated compilers, parallel HDF5, mpi4py, and MPI-enabled h5py before
building. Autoreconf, configure, build, serial regression, and install are
individually time-bounded and write terminally marked logs. Build and test
processes use a private 0700 `HOME`, XDG cache/config roots, and Matplotlib
configuration directory under the isolated build tree; they do not write to
the account home. A dedicated qualifier then exercises real curl, constitutive,
source, boundary, and DFT work in four lanes: in-place and installed Python at
one and two MPI ranks. It requires FP64, MPI, a CPU-only binary, the CPU
backend, agreement between the mpi4py and Meep communicator sizes, positive
CPU phase counters, zero CUDA counters, finite field/DFT observables, and rank
agreement. The CPU-specific semantic receipt contract binds the source,
authoritative dependency-lock path and bytes, exact toolchain, build-tree and
installed Python/libmeep runtime images, conda-prefix content audit, all four
qualification logs, host-build logs, and complete installed environment and
runtime manifests. Runtime manifests admit no bytecode exclusions, and the
builder rejects bytecode files, directories, and same-named symlinks.

The user-workload controller now accepts only the exact
`cpu-mpi-python-fp64` and `cuda-mpi-python-fp32` build kinds. Precision and
CUDA state must each have one explicit enable/disable configure flag, MPI and
Python must be enabled, and Scheme must be explicitly excluded. This closes a
former provenance gap where an unspecified or CUDA-capable FP64 receipt could
be accepted as the CPU reference. The actual isolated build and two-rank
receipt are a host gate; they do not replace the still-pending GPU workload
matrix. The M28 builder/qualifier/controller contracts bring the complete
evidence/benchmark-tool regression to 880/880 tests.

### M35 stable complete-curl phase replay

The persistent FP32 CUDA time step repeatedly presents the same cross-chunk
curl descriptor topology. M35 retains the already-uploaded phase descriptor
and, after two identical warm reuses, validates its resident-cache allocation
generations and every input/output mirror before launching it directly. This
removes repeated host descriptor construction, pointer lookup, and mirror-map
traversal while preserving the same CUDA kernel and logical curl counters.

Replay is fail-closed. A plan is eligible only when the prior phase completed
with exactly one batched flush and no direct beta, BFAST, cylindrical, or
zero-span operation. Every invocation fingerprints the current Cartesian
field/material pointers, grid/tile topology, beta/BFAST/m state, and relevant
tile-policy environment before the chunk loop may be skipped. Cache
destruction or migration invalidates dependent plans. A topology change,
host-write publication, changed mirror size, missing mirror,
allocation-generation change, device change, batching-policy change, or
partial phase returns to ordinary collection. A replay launch exception
invalidates the plan and propagates; it is counted as `unready` so the outcome
conservation contract remains exact. Output mirrors become device-authoritative
only after all validation succeeds and the kernel is launched. The presence
of `MEEP_GPU_DISABLE_CURL_PHASE_REPLAY` (normally set to `1`) is the
same-binary opt-out.

The additive `curl_phase_replay_statistics` API and Python
`_gpu_statistics()` expose replay checks, hits, unready decisions,
allocation-generation misses, and mirror misses without changing the ABI of
the existing `phase_batch_policy_statistics` structure. Their conservation contract is
`checks = hits + unready + generation_misses + mirror_misses`; every replay
counter remains zero when the opt-out is present. Native regressions require
zero hits for a fixed direct beta phase and for the first stable-plan-to-beta
and stable-plan-to-BFAST transition, while comparing both transitions with
the CPU result.

The MPI performance worker reduces the five outcome counters plus the number
of opted-out ranks into the separate
`gpmeep-curl-phase-replay-v1` marker, leaving the established 12-field
`gpmeep-phase-batch-policy-v1` schema unchanged. The one/two-GPU scaling
diagnostic requires zero opted-out ranks, exact counter conservation, positive
replay of both B/D curl phases on every measured rank-step whenever curl
batching is enabled, zero activity whenever curl batching is disabled, and no
unready, allocation-generation-miss, or mirror-miss outcome in the warmed
measurement interval.

On the qualified RTX 3090 Ti, the final cross-ordered two-pair
adjoint-gradient test-body median fell from 87.5320 seconds with replay
disabled to 81.2125 seconds with replay enabled (7.22% less time, 1.078x).
Both enabled runs recorded 386,370 hits out of 386,510 checks, no mirror
misses, and passed the gradient oracle after the topology-fingerprint and ABI
fixes. These measurements establish a local steady-state improvement; the
later CPU-8/GPU-1/GPU-2 workload matrix remains the release performance gate.

## Historical first implementation boundary

The original MVP targeted the nondispersive Yee-grid electric and magnetic
field updates for a single process:

- real-valued fields;
- 1D/2D/3D Cartesian grids;
- FP32 field and coefficient arrays;
- no MPI, cylindrical coordinates, dispersive polarization, DFT monitors, or
  adjoint solver in the first kernel.

That boundary has since expanded as documented above. Unsupported
configurations still must fall back to the existing CPU path or fail with a
clear diagnostic. They must never silently produce partial results.
