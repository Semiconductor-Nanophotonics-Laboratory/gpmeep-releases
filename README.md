# gpmeep

**gpmeep is an open-source, GPU-accelerated finite-difference time-domain
(FDTD) package built on Meep.** It uses NVIDIA CUDA, persistent device-resident
field data, and MPI multi-GPU execution to accelerate Meep FDTD simulations
while preserving the familiar Python API:

```python
import meep as mp
```

**Project keywords:** `GPU-accelerated FDTD`, `GPU FDTD`, `CUDA FDTD`,
`Meep GPU`.

Validated source snapshots, packages, checksums, release notes, and installation
instructions are published in this repository. Active development and
candidate qualification occur separately so the public release history
contains accepted versions rather than unfinished development branches.

Raw validation logs and private execution metadata are not published. See
[the public release policy](PUBLIC_RELEASE_POLICY.md) and
[v1.0.3 package identity](PUBLIC_RELEASE.json). The public snapshot removes
non-runtime evidence files; its Git commit is not the original binary's build
commit.

## Project creator and gpmeep contribution

**gpmeep was conceived, created, and is led by
[Dong-Joon Yi](https://iverydj.github.io/) ([GitHub](https://github.com/iverydj)).**
The gpmeep-specific contribution is the architecture and implementation that
extends upstream Meep with a persistent FP32 CUDA execution layer, explicit
CPU/CUDA control, MPI-based multi-GPU execution, device-resident DFT and
scientific reductions, release provenance, and correctness/performance gates.
This work includes the production-driven feature requirements, numerical
acceptance criteria, failure policy, portability design, benchmark program,
and release engineering needed to make the accelerator usable as a Meep
distribution rather than as an isolated demonstration kernel.

gpmeep is a derivative of the
[NanoComp Meep project](https://github.com/NanoComp/meep). Meep's FDTD
formulation, public APIs, and upstream history remain the work of the upstream
authors and contributors. The credit above describes the original gpmeep GPU,
integration, validation, and distribution work; it does not replace upstream
attribution. See [`AUTHORS`](AUTHORS), [`CITATION.cff`](CITATION.cff), and the
[license and provenance notes](doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md).

## Compatibility with upstream Meep

For Python users, gpmeep is a distribution of Meep rather than a new Python
package namespace. Scripts continue to import `meep`, and the usual
`mp.Simulation`, geometry, sources, materials, DFT monitors, run callbacks,
`meep.adjoint`, and `meep.mpb` interfaces keep their upstream spelling. This is
source/API compatibility within the validated v1 feature contract, not a claim
that every upstream build option or execution path is present. Do not change a
script to `import gpmeep`: there is no `gpmeep` Python module.

v1.0.3 retains the accepted v1.0.2 CUDA and numerical implementation while
repairing production-reported Linux/Conda portability, build-path, error
propagation, and custom CUDA-architecture provenance defects. A worktree or
version string alone is not a release: use the annotated tag, package checksum,
installed GLIBC attestation, build-produced SASS/PTX inventory, and provenance
checks to identify an accepted artifact.

The user-visible differences are:

| Area | Upstream Meep script | gpmeep |
|---|---|---|
| Conda distribution | usually installed as `pymeep` | installed as `gpmeep` in a separate prefix |
| Python import | `import meep as mp` | exactly the same |
| Backend | CPU | CPU by default, with opt-in `auto` or NVIDIA CUDA |
| Extra runtime API | no CUDA controller | `mp.gpu` discovery, selection, diagnostics, and counters |
| Precision | depends on the upstream build; commonly FP64 | the v1 CPU and CUDA lanes are both FP32 |
| Adjoint / MPB execution | build dependent | upstream imports and spelling; FDTD portions may use the selected backend, while packaged JAX and MPB remain CPU-side and MPB is non-MPI |
| Multi-GPU | MPI CPU decomposition | one MPI rank per selected physical/MIG GPU; rank-local selection from the visible-device pool |
| Scheme | available in some upstream builds | intentionally not packaged |
| Native GDSII helpers | build dependent | native support is disabled; wrapper names may remain but fail when called, so use the included Python `gdstk` |
| Platform in the binary release | upstream supports a wider platform set | v1 package is Linux x86-64, Python 3.11, NVIDIA CUDA 12.4, Open MPI |

A Python program within the validated v1 feature contract normally runs
unchanged on gpmeep's default CPU backend. Scripts requiring Scheme, native
GDSII, FP64, or an unvalidated feature combination require migration or
separate qualification.

```python
import meep as mp

sim = mp.Simulation(...)  # existing Meep code
sim.run(...)
```

To request a backend, add one gpmeep-specific line before constructing or
running simulations:

```python
import meep as mp

mp.gpu.set_backend("cuda")  # "cpu", "auto", or "cuda"
print(mp.gpu.runtime_available, mp.gpu.backend_diagnostic)

sim = mp.Simulation(...)
sim.run(...)
```

The equivalent process-level selection must be set before Python starts:

```sh
MEEP_GPU_BACKEND=cuda CUDA_VISIBLE_DEVICES=0 python simulation.py
```

`cuda` is an explicit, all-or-nothing request for gpmeep's FDTD device backend:
configuration or rank/device-selection failure raises an error instead of
silently selecting the CPU backend. This does not imply that Python control
flow, MPB, JAX/NLopt, or other host-side work executes on the GPU. `auto` is
opt-in and may select CPU for a small or unsuitable workload; it is not a
promise that GPU is faster for every simulation. Inspect
`mp.gpu.active_backend`, `mp.gpu.backend_diagnostic`, and
`mp.gpu.statistics()` when validating which path actually ran.

For two GPUs, gpmeep uses MPI rank decomposition rather than making one Python
process transparently drive both devices:

```sh
CUDA_VISIBLE_DEVICES=0,1 \
MEEP_GPU_BACKEND=cuda \
MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
MEEP_GPU_MPI_TRANSPORT=pinned \
MEEP_GPU_MPI_COMPLETION=waitsome \
mpiexec -n 2 python simulation.py
```

This example exposes the same two-device pool to both ranks. On a two-rank
single-node run, gpmeep selects compatible ordinals by node-local rank and
validates that the ranks selected distinct physical/MIG device identifiers. A
scheduler may instead expose one distinct GPU to each rank, in which case each
rank sees its own rank-local ordinal. Oversubscription is an explicit exception,
not the recommended performance configuration. More GPUs do not guarantee a
faster run: rank-local work must be large enough to repay MPI and CUDA
synchronization costs.

## How GPU acceleration works

gpmeep keeps Meep's Python control plane, object model, Yee update order, and
MPI domain decomposition. The acceleration layer sits below the existing
`fields::step()` pipeline: a fields owner is admitted to CUDA before a time
step starts, each owned `fields_chunk` acquires a persistent device cache, and
the repeated numerical phases operate on those device-resident arrays. This is
different from copying the complete field state to and from a GPU around every
kernel.

### Design rationale

The central performance observation is that an FDTD run applies the same local
stencil and material updates to large field arrays for many time steps. A GPU
is useful only if those arrays remain on the device long enough to amortize
allocation, kernel-launch, PCIe, and MPI costs. gpmeep therefore accelerates
the repeated data path rather than merely wrapping each upstream operation in
an independent host-to-device/device-to-host copy:

1. **Preserve the numerical contract.** The existing Yee leapfrog order,
   material equations, boundary phases, and MPI domain decomposition define
   the result. CUDA kernels implement those phases in the same dependency
   order instead of introducing a different solver.
2. **Keep evolving state resident.** Each `fields_chunk` has a persistent,
   generation-tracked device mirror. Ordinary time steps reuse allocations
   and leave the newest field state on the GPU.
3. **Move consumers to the data.** Sources, dispersive updates, DFT monitors,
   Near2Far/LDOS/eigenmode work, and compact scientific reductions operate on
   resident arrays where qualified. Only requested output or genuinely
   host-side work crosses the coherence boundary.
4. **Scale through Meep's decomposition.** One MPI rank owns one GPU or MIG
   instance. Rank-local subdomains advance on their devices while halo data
   uses qualified CUDA-aware MPI or retained pinned-host staging.
5. **Fail closed at boundaries.** Collective preflight rejects unsupported
   physics, inconsistent rank/device mappings, or insufficient memory before
   solver mutation. Generation and transaction checks prevent stale or
   partially published host/device state.
6. **Treat speed as a measured crossover.** Small, short, callback-heavy, or
   over-decomposed jobs may not amortize GPU overhead. Released v1.0.x keeps
   explicit `cpu` and `cuda` choices and offers `auto` only as an opt-in policy;
   no backend mode is presented as a universal speed guarantee.

The resulting speedup comes primarily from eliminating repeated full-state
transfers and executing the dominant update/monitor pipeline at GPU memory
bandwidth. Kernel batching, cached descriptors, asynchronous staging, overlap,
and optional replay then reduce launch and communication overhead without
changing the physical model.

```text
Python mp.Simulation / the existing Meep API
                     |
     collective backend and feature preflight
              /                      \
        CPU owner                 CUDA owner
                                      |
             persistent mirror cache per fields_chunk
                                      |
       B curl -> H/material -> boundary/MPI halo
       D curl -> E/material -> boundary/MPI halo
                                      |
          resident DFT updates and scientific reductions
                                      |
       selective host publication only when it is required

multi-GPU: MPI rank 0 -> GPU 0, rank 1 -> GPU 1, ...
halo data: validated CUDA-aware MPI or pinned-host staging
```

### Backend selection is owner-wide and state-atomic

CPU is the default. `mp.gpu.set_backend("cuda")` is a required-CUDA request:
runtime, device assignment, or supported-time-step preflight failure raises
before the step modifies any field. `auto` is an opt-in admission policy. All
MPI ranks first agree on the requested mode, workload decision, feature set,
transport, and device assignment. A validated time step is therefore entirely
CPU or entirely CUDA for that fields owner; gpmeep does not silently execute an
unsupported chunk on the CPU after partially advancing other chunks on CUDA.

Automatic selection has two stages. The host-only stage can reject a small
rank-local domain without initializing the CUDA runtime. In the current build,
the reference floor is 524,288 local cells for one serial FDTD worker per MPI
rank; `MEEP_GPU_AUTO_MIN_CELLS` provides an explicit override. Owners that pass
the first stage are checked against GPU memory bandwidth and a conservative
memory budget: 1,024 bytes per local cell plus 64 MiB, while reserving the
larger of 512 MiB or 10% of total device memory. Every rank must pass. This
policy is designed to reject the bounded launch-dominated cases used to
calibrate it, but it cannot know the future run length, monitor count, output
cost, Python/optimizer share, or MPI topology, so it is not a guarantee that
the selected backend is faster.

### Persistent field residency and coherence

Meep's host arrays remain the API-visible storage identity. gpmeep associates
each local fields chunk with a CUDA cache keyed by those host pointers. A cache
records the device allocation, byte size, upload epoch, allocation/content
generations, and whether the device mirror is the authoritative dirty copy. It
also retains reusable allocations and per-chunk descriptors for curl,
constitutive updates, sources, DFT monitors, and LDOS. The resident backend
keeps dependent boundary, eigenmode-overlap, and Near2Far plans in separate
owner-level registries and invalidates them when their cache dependencies
change.

The first device use allocates and uploads a mirror. Later phases and time
steps reuse it. CUDA writes mark the device copy authoritative, and the outer
step session ends with `finish(false)` when the persistent path is eligible,
so a normal step does not publish all fields back to the host. A device-side
finite scan preserves Meep's per-step NaN/Inf failure check while returning
only its compact verdict.

A selective point read can copy one requested scalar without publishing its
complete host mirror. Generic host operations or callbacks that directly read
or mutate host field arrays first synchronize the affected owner; selected
output/checkpoint work, material or structure changes, backend/device changes,
and object destruction establish related coherence boundaries. When a complete
synchronization is required, every dirty D2H copy is first placed in staging;
host fields are updated only after all copies succeed. This transactional rule
prevents a failed transfer from leaving a mixture of old and new host arrays.
Generation checks invalidate cached pointer plans after allocation, topology,
material, or device changes.

### CUDA FDTD pipeline and supported physics

The CUDA path preserves Meep's leapfrog/Yee sequence rather than replacing the
solver algorithm. One step performs B curl, B sources, and B boundaries; H
constitutive and polarization updates plus their boundary phases; D curl, D
sources, and D boundaries; E constitutive and polarization updates plus their
boundary phases; and then DFT accumulation. The main device work is:

| Phase | CUDA implementation |
|---|---|
| B/D curl | FP32 Cartesian 1D/2D/3D and cylindrical stencil kernels, including supported PML curl auxiliaries, electric/magnetic conductivity, 2D beta, and BFAST forms |
| H/E constitutive update | diagonal/off-diagonal inverse susceptibility, PML H/E auxiliaries, integrated-source terms, and supported `chi2`/`chi3` nonlinear terms |
| Dispersive polarization | exact standard Lorentzian/Drude, eligible gyrotropic, and eligible multilevel/gain state updates and polarization subtraction |
| Sources | scalar time amplitudes are prepared by host control code; supported indexed spatial profiles are applied to resident fields on CUDA |
| Boundaries | resident metal zeroing, local copy/negation/Bloch phase, retained gather/scatter plans, and eligible CUDA Graph replay |
| Validation | device finite checks retain the NaN/Inf contract without a full field readback |

Cylindrical coordinates and BFAST each have supported CUDA paths, but their
combination is not a supported curl configuration.

Tile coalescing, topology-aware curl/H/E phase batching, stable phase replay,
cached boundary descriptors, and eligible communication/compute overlap reduce
launch and scheduling overhead. These are conditional fast paths, not extra
physics restrictions: an ordinary supported CUDA kernel can still run when a
particular overlap or replay optimization is ineligible. Indexed-source
batching remains experimental/forced-only, and automatic multi-monitor DFT
batching is fail-closed in v1.0.2; neither is presented as a default speedup.
An arbitrary custom susceptibility subclass is not implicitly CUDA-capable:
required CUDA rejects it during preflight, while `auto` selects CPU for the
whole owner before the step.

### DFT monitors and scientific reductions

DFT samples are interleaved-complex FP32 arrays with retained device
descriptors. Monitor updates accumulate on the GPU, and consumers avoid
publishing the full point-by-frequency data when a smaller result is enough.
CUDA implementations cover resident norm and scale operations, flux and
complex-flux spectra, energy, force, LDOS field/source contraction, eigenmode
overlap, DFT checkpoint staging, requested-array materialization, and supported
forward and adjoint Near2Far contractions. For scalar/spectral reductions such
as flux, energy, force, and LDOS, the kernels return rank-local compact results
that are combined by the required MPI collective.
Requested-array materialization and checkpoint/output instead use bounded
staging for the requested payload and are not compact scalar reductions.

There are deliberate host boundaries. HDF5 file I/O is host work, though its
CUDA path stages only the required DFT ranges. MPB samples eigenmode profiles
on the host and uploads them for the device overlap reduction. Geometry-derived
Dielectric/Permeability arrays are synthetic host queries rather than DFT
fallback. An unsupported postprocessing shape may use its independently
eligible CPU path after publishing the needed resident data; required CUDA's
fail-closed statement applies to the validated FDTD time step, not to every
output API.

### MPI and multi-GPU execution

Multi-GPU execution extends Meep's existing spatial MPI decomposition. Each
MPI rank advances its rank-local chunks on one selected physical GPU or MIG
instance. Absent an explicit `MEEP_GPU_DEVICE` or
`mp.gpu.select_device(...)` selection, node-local rank chooses among ordinals
exposed by `CUDA_VISIBLE_DEVICES`; the resulting stable physical/MIG identities
are then validated collectively across ranks and independent subcommunicators.
Distinct devices are required unless oversubscription is explicitly enabled;
one Python rank does not transparently divide one domain over every GPU.

`MEEP_GPU_MPI_TRANSPORT=auto` negotiates a safe common transport on all ranks.
Direct device-buffer `MPI_Isend`/`MPI_Irecv` is used only when every rank and
the Open MPI runtime report CUDA-aware support. Otherwise gpmeep uses retained
page-locked host buffers and asynchronous D2H/H2D staging. The default
completion policy is `waitsome`; `waitall` is available for qualified
CUDA-aware runs. Eligible schedules can overlap halo traffic with an H/E
update or an interior D curl and execute the shell after the receive, while
the general retained boundary path remains available when overlap is unsafe.

### FP32, reductions, and numerical accuracy

The v1 package builds both its CPU and CUDA lanes with Meep single precision;
the comparison is not FP64 CPU versus FP32 GPU. Fields, material coefficients,
time-step kernels, and complex DFT storage are FP32, and CUDA fast math is
disabled. Selected numerically sensitive consumers use deterministic FP64
accumulation or finalization, including DFT pair reductions and LDOS, while
Near2Far can retry a cancellation-sensitive contraction with a higher-precision
CUDA path. This improves reduction stability without turning the stored FDTD
state into FP64. CPU/CUDA acceptance is based on scientific tolerances, not
bitwise identity or equivalence to a separate upstream FP64 build.

### Adjoint, CW, and host-side work

Forward and adjoint FDTD simulations enter the same CUDA time-step and monitor
paths. Supported eigenmode and Near2Far adjoint contractions also have device
implementations. The complete inverse-design program is not GPU-resident:
Python orchestration, JAX/Autograd/NLopt, geometry/design mutation, and MPB
profile generation remain host-side, and a design mutation establishes a
coherence boundary. The continuous-wave solver has a qualified resident FP32
CUDA BiCGSTAB-L path, but legacy flux-monitor restrictions and FP32 convergence
diagnostics still apply; its host solver remains available.

Likewise, selecting CUDA does not move Python callbacks, geometry construction,
file I/O, or every possible third-party/custom extension onto the GPU. Small,
short, monitor-heavy, host-heavy, or excessively decomposed jobs can remain
slower than CPU. Use `mp.gpu.active_backend`, `mp.gpu.backend_diagnostic`, the
selected device identifier, and `mp.gpu.statistics()` to prove which kernels,
transfers, reductions, and MPI transport actually ran, then measure complete
end-to-end time for the real workload.

### Implementation source map

| Layer | Primary source |
|---|---|
| Python controller and telemetry | [`python/meep.i`](python/meep.i), [`src/meep/gpu.hpp`](src/meep/gpu.hpp) |
| backend selection, device mapping, resident cache, and coherence | [`src/gpu_backend.cpp`](src/gpu_backend.cpp), [`src/gpu_backend_internal.hpp`](src/gpu_backend_internal.hpp) |
| time-step order and boundary/MPI scheduling | [`src/step.cpp`](src/step.cpp) |
| B/D curl and automatic crossover policy | [`src/step_db.cpp`](src/step_db.cpp) |
| H/E and polarization updates | [`src/update_eh.cpp`](src/update_eh.cpp), [`src/update_pols.cpp`](src/update_pols.cpp), [`src/susceptibility.cpp`](src/susceptibility.cpp), [`src/multilevel-atom.cpp`](src/multilevel-atom.cpp) |
| DFT, LDOS, eigenmode, and Near2Far consumers | [`src/dft.cpp`](src/dft.cpp), [`src/dft_ldos.cpp`](src/dft_ldos.cpp), [`src/near2far.cpp`](src/near2far.cpp) |
| MPI transport negotiation | [`src/mympi.cpp`](src/mympi.cpp) |
| CUDA API and kernels | [`cuda/include/meep_cuda/runtime.hpp`](cuda/include/meep_cuda/runtime.hpp), [`cuda/src/runtime.cu`](cuda/src/runtime.cu) |

The exact validated coverage and known limits remain normative in the
[feature matrix](doc/docs/GPMEEP_FEATURE_MATRIX.md); crossover behavior and
measurements are in the [CPU/automatic/CUDA guide](doc/docs/GPMEEP_GPU_CROSSOVER.md).

## Numerical and environment migration notes

- Check `mp.is_single_precision()` and use tolerances appropriate for FP32.
  Results should satisfy the workload's scientific tolerance, but they need
  not be bit-for-bit identical to a separate FP64 Meep build.
- Do not install official `pymeep` and `gpmeep` into the same environment.
  Both provide the `meep` module and native `libmeep` ABI. Keep separate Conda
  or Micromamba prefixes and compare outputs across processes/files.
- The Conda package version (`gpmeep 1.0.3`, etc.) is the distribution version.
  `mp.__version__` remains the embedded upstream Meep source version, so use
  the Conda record and gpmeep provenance tools when identifying a release.
- `meep.mpb` and `meep.adjoint` retain their Python imports. In the v1 package,
  the MPB solver and packaged JAX are CPU-side (and MPB is non-MPI); only the
  supported FDTD portions follow the selected gpmeep backend.
- `mp.GDSII_vol`, `mp.GDSII_prisms`, and `mp.GDSII_layers` names may exist as
  disabled native-library stubs when `mp.with_libGDSII()` is false. Calling
  them is not supported; parse GDSII with the included `gdstk` instead.
- v1.0.3 retains the sm86 runtime qualification. The default AUTO package
  contains CUDA 12.4's native sm60+ targets plus newest-architecture PTX.
  During packaging, CUDA 12.4 `cuobjdump` enumerates the actual native cubins
  and embedded PTX in `libmeep`; the resulting hash- and binary-bound inventory
  must equal the requested AUTO or LIST policy. Custom LIST builds therefore
  require at least one native `-real` target, record and verify only what they
  compile, and claim sm86 runtime
  qualification only when native sm86 code is present. Run the installed
  self-check on each new machine before production use.
- Most validated Python functionality is CUDA accelerated, including broad
  FDTD, DFT, MPI, and adjoint families, but the 42-feature release contract is
  not a proof of every upstream callback or parameter combination. See the
  feature matrix for exact claims and known limits.

The v1.0.3 line was opened for confirmed binary compatibility and packaging
defects observed in a production deployment. It does not introduce a
speculative kernel or MPI transport rewrite. Structural GPU changes remain
separate explicitly approved v1.1.0 candidates.

The v1 release target is a checksum-locked Linux x86-64 Conda package and a
one-command installer into a dedicated environment. Runtime release validation
currently covers RTX 3090 Ti (sm86); the CUDA 12.4 portable build policy emits
native sm60+ targets exposed by nvcc and newest-architecture PTX. Scheme and
co-installation with official `pymeep` are intentionally unsupported.

- [Installation and runtime checks](doc/docs/GPMEEP_INSTALLATION.md)
- [Validated feature matrix and limits](doc/docs/GPMEEP_FEATURE_MATRIX.md)
- [Correctness and performance summary](doc/docs/GPMEEP_PERFORMANCE.md)
- [Choosing CPU, automatic, or CUDA execution](doc/docs/GPMEEP_GPU_CROSSOVER.md)
- [v1.0.3 field-feedback compatibility release notes](doc/docs/GPMEEP_V1_0_3_RELEASE_NOTES.md)
- [v1.0.2 crossover-guidance release notes](doc/docs/GPMEEP_V1_0_2_RELEASE_NOTES.md)
- [v1.0.1 provenance correction](doc/docs/GPMEEP_V1_0_1_RELEASE_NOTES.md)
- [License, attribution, and provenance](doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md)
- [v1 release gates](doc/docs/GPMEEP_V1_RELEASE_PLAN_KO.md)

After downloading the v1 Conda package and its `.sha256` sidecar into a
`downloads/` directory that is separate from the builder's `dist/` output:

```sh
/bin/bash -p scripts/install-gpmeep.sh \
  --prefix /absolute/path/to/gpmeep-v1 \
  --package downloads/gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda \
  --expected-source-commit b48c9b678f2316cefa95e48a661f7a877547a000 \
  --mode gpu1 --gpu-devices 0
```

The installer verifies the package checksum, exact dependency lock,
package/source provenance, installed CPU/CUDA numerical equivalence, strict
CUDA work, and zero CPU fallback. `--mode gpu2 --gpu-devices 0,1` adds the
two-rank/two-device transport check.

The explicit source commit above identifies the original v1.0.3 binary, not
this privacy-cleaned publication snapshot. Keep it when installing that
package; a package rebuilt locally uses its own clean build commit instead.

gpmeep is a GPL-2.0-or-later derivative of the upstream Meep project. The
upstream overview and citation information follow.

![Meep banner](doc/docs/images/Meep-banner.png)

[![CI](https://github.com/NanoComp/meep/actions/workflows/build-ci.yml/badge.svg)](https://github.com/NanoComp/meep/actions/workflows/build-ci.yml)
[![Sanitizers](https://github.com/NanoComp/meep/actions/workflows/build-san.yml/badge.svg)](https://github.com/NanoComp/meep/actions/workflows/build-san.yml)
[![Latest Docs](https://readthedocs.org/projects/meep/badge/?version=latest)](http://meep.readthedocs.io/en/latest/)
![Python versions](https://img.shields.io/badge/python-3.9%2C%203.11-brightgreen.svg)

**Meep** is a free and open-source software package for simulating [electromagnetics](https://en.wikipedia.org/wiki/Electromagnetism) across a broad range of applications using the [finite-difference time-domain](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) (FDTD) method.

## Key Features

-   **Free and open-source software** under the [GNU GPL](https://en.wikipedia.org/wiki/GNU_General_Public_License).
-   Complete **scriptability** via [Python](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/), [Scheme](https://meep.readthedocs.io/en/latest/Scheme_Tutorials/Basics/), or [C++](https://meep.readthedocs.io/en/master/C++_Tutorial/) APIs.
-   Simulation in **1d, 2d, 3d**, and **[cylindrical](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/#cylindrical-symmetry)** coordinates.
-   Distributed memory [parallelism](https://meep.readthedocs.io/en/latest/Parallel_Meep/) on any system supporting [MPI](https://en.wikipedia.org/wiki/MPI).
-   Portable to any Unix-like operating system such as [Linux](https://en.wikipedia.org/wiki/Linux), [macOS](https://en.wikipedia.org/wiki/macOS), and [FreeBSD](https://en.wikipedia.org/wiki/FreeBSD).
-   **Precompiled binary packages** of official releases via [Conda](https://meep.readthedocs.io/en/latest/Installation/#conda-packages).
-   Variety of arbitrary [material](https://meep.readthedocs.io/en/latest/Materials/) types: **anisotropic** electric permittivity ε and magnetic permeability μ, along with **dispersive** ε(ω) and μ(ω) including loss/gain, **nonlinear** (Kerr & Pockels) dielectric and magnetic materials, electric/magnetic **conductivities** σ, **saturable** gain/absorption, and **gyrotropic** media (magneto-optical effects).
-   [Materials library](https://meep.readthedocs.io/en/latest/Materials/#materials-library) containing predefined broadband, complex refractive indices.
-   [Perfectly matched layer](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/) (**PML**) absorbing boundaries as well as **Bloch-periodic** and perfect-conductor boundary conditions.
-   Exploitation of [symmetries](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/) to reduce the computation size, including even/odd mirror planes and 90°/180° rotations.
-   [Subpixel smoothing](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/) for improving accuracy and shape optimization.
-   [Custom current sources](https://meep.readthedocs.io/en/latest/Python_Tutorials/Custom_Source/) with arbitrary time and spatial profile as well as a [mode launcher](https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/) for waveguides and planewaves, and [Gaussian beams](https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussianbeam3dsource).
-   [Frequency-domain solver](https://meep.readthedocs.io/en/latest/Python_User_Interface/#frequency-domain-solver) for finding the response to a [continuous-wave](https://en.wikipedia.org/wiki/Continuous_wave) (CW) source as well as a [frequency-domain eigensolver](https://meep.readthedocs.io/en/latest/Python_User_Interface/#frequency-domain-eigensolver) for finding resonant modes.
-   ε/μ and field import/export in the [HDF5](https://en.wikipedia.org/wiki/HDF5) data format.
-   [GDS](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDS_Import/) file import for planar geometries.
-   Field analyses including [discrete-time Fourier transform (DTFT)](https://meep.readthedocs.io/en/latest/Python_User_Interface/#field-computations), [Poynting flux](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#transmittance-spectrum-of-a-waveguide-bend), [mode decomposition](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/) (for [S-parameters](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDS_Import/#s-parameters-of-a-directional-coupler)), [energy density](https://meep.readthedocs.io/en/latest/Python_User_Interface/#energy-density-spectra), [near to far transformation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/), [frequency extraction](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#modes-of-a-ring-resonator), [local density of states](https://meep.readthedocs.io/en/latest/Python_Tutorials/Local_Density_of_States/) (LDOS), [modal volume](https://meep.readthedocs.io/en/latest/Python_User_Interface/#field-computations), [scattering cross section](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#mie-scattering-of-a-lossless-dielectric-sphere), [Maxwell stress tensor](https://meep.readthedocs.io/en/latest/Python_Tutorials/Optical_Forces/), [absorbed power density](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#absorbed-power-density-map-of-a-lossy-cylinder), [arbitrary functions](https://meep.readthedocs.io/en/latest/Field_Functions/); completely programmable.
-   [Adjoint solver](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/) for **inverse design** and **topology optimization**.
-   [Visualization routines](https://meep.readthedocs.io/en/latest/Python_User_Interface/#data-visualization) for the simulation domain involving geometries, fields, boundary layers, sources, and monitors.

## Citing Meep

We kindly request that you cite the following paper in any published work for which you used Meep:

- A. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J.D. Joannopoulos, and S.G. Johnson, [MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method](http://dx.doi.org/doi:10.1016/j.cpc.2009.11.008), Computer Physics Communications, Vol. 181, pp. 687-702 (2010) ([pdf](http://ab-initio.mit.edu/~oskooi/papers/Oskooi10.pdf)).


## Documentation

See the [manual on readthedocs](https://meep.readthedocs.io/en/latest) for the latest documentation.
