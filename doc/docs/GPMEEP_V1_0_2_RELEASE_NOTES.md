# gpmeep v1.0.2 release notes

v1.0.2 is a compatibility-preserving crossover-guidance patch for the CUDA
12.4, Open MPI, FP32 v1 package line. Its first Conda build identity is:

```text
gpmeep-1.0.2-cuda124_mpi_openmpi_py311_0.conda
```

The installed release manifest remains schema 2. CUDA 12.4, Python 3.11,
Open MPI, FP32 CPU/CUDA execution, `import meep as mp`, the dedicated-prefix
policy, compiled CUDA architecture set, Python API, native ABI, and FDTD
numerical algorithm are unchanged from v1.0.1. There is no CUDA-kernel change
in this patch.

## GPU slower than CPU: bounded classification

v1.0.2 investigated two representative cases where forced CUDA was much slower
than CPU. Automatic backend selection already existed; this release documents
and qualifies its conservative rank-local cell-floor behavior instead of
introducing a new backend API or forcing every simulation onto a GPU.

For `3rd-harm-1d.py`, exact v1.0.1 package probes observed CPU 29.473 seconds,
forced CUDA 139.674 seconds, forced DFT batching 135.161 seconds, forced
field-phase batching 122.020 seconds, and automatic 29.172 seconds. Each CUDA,
batching, and automatic comparison covered 4,591 numerical metrics with zero
failures. Automatic selected CPU for 2,503 rank-local cells before CUDA runtime
discovery. These are cold whole-process single samples on an RTX 3090 Ti host,
with a CPU1 reference; they are diagnostic observations rather than a release
speed gate or repeatable speedup claim.

For `parallel-wvgs-force.py`, a new exact-package automatic probe selected CPU
for 14,729 rank-local cells and took 108.021 seconds. The contextual CPU
107.543-second and forced-CUDA 247.051-second observations came from an earlier
implementation-equivalent M4 correctness window, not a paired exact-package
benchmark. The release therefore does not claim the difference between those
windows as a new speed result.

The existing DFT and field-phase batching controls reduced launch counts for
the bounded `3rd-harm-1d.py` probe but did not close its CPU/GPU gap. They are
not enabled globally. CUDA Graphs, persistent kernels, deeper phase fusion, and
a dynamic crossover model remain structural v1.1.0 candidates requiring a
separately approved repeated performance matrix.

See [Choosing CPU, automatic, or CUDA execution](GPMEEP_GPU_CROSSOVER.md) for
selection instructions, exact telemetry, limitations, and multi-GPU guidance.
Forced `cuda` remains fail-closed for the FDTD device backend; it is not a
promise that every workload will run faster than CPU.

## Performance and provenance continuity

The official repeated v1 performance anchor is unchanged: AuNP GPU1 and GPU2
FDTD were 9.8214x and 14.9595x faster than CPU8 on the qualified host. v1.0.2
retains the exact accepted M3 source commit, report hash, COMPLETE hash, audit
hash, means, and manifest schema established by v1.0.1. Relabeling that
historical measurement as a new v1.0.2 benchmark would be incorrect.

The v1.0.1 provenance-correction notes remain packaged alongside these notes
so the retained anchor can be audited offline. A v1.0.2 artifact is accepted
only after an exact clean build, fresh-prefix CPU/GPU1/GPU2/MPI/adjoint checks,
installed AuNP replay, uninstall/reinstall lifecycle checks, regression tests,
and independent adversarial audit all pass.

## Version-line policy

After v1.0.2, v1.0.3 is opened only if a concrete stability, bug, or
compatibility issue is confirmed. If none is found, v1.0.x ends at v1.0.2.
Structural GPU work is evaluated separately as an explicitly approved v1.1.0
candidate.
