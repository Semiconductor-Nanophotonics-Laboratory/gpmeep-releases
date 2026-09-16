# gpmeep v1.0.3 release notes

v1.0.3 is a field-feedback compatibility and packaging patch. It does not
change the accepted CUDA FDTD numerical kernels or their FP32 policy. The
release was opened after a production deployment of v1.0.1 on Ubuntu 22.04
identified four concrete install/build defects that remained present in
v1.0.2.

The release package name is:

```text
gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda
```

## Compatibility and build fixes

- The Conda recipe explicitly targets the conda-forge glibc 2.17 C sysroot.
  Every gpmeep-owned ELF is audited during the build; an ELF requiring a
  newer GLIBC symbol fails the package build. The audit is installed and its
  SHA-256 is bound into `share/gpmeep/release.json`.
- Release downloads are documented under `downloads/`, separate from the
  builder's default `dist/` output. Existing same-version artifacts still
  fail closed, and the diagnostic directs maintainers to `--output-dir`.
- `scripts/build-conda-package.sh --output-dir PATH` canonicalizes PATH before
  source staging and before passing it to Conda. Relative paths therefore
  have one unambiguous absolute interpretation.
- A nonzero Conda build status is reported and preserved explicitly. A
  same-version artifact under `broken/` is rejected even if Conda itself
  returned success. The zero-status behavior was reported in production but
  was not independently reproduced from the supplied snapshot; the new gates
  are regression-tested fail-closed safeguards.
- Explicit CUDA architecture lists such as `86-real;86-virtual` are supported
  by provenance verification and must include at least one native `-real`
  target. PTX-only lists fail before packaging. The requested policy must exactly match the
  recorded native and PTX inventories. CUDA 12.4 `cuobjdump` derives those
  inventories from the final `libmeep`; the installed report is SHA-bound to
  both that binary and the release manifest. `runtime_validated_architectures`
  contains `sm86` only when native sm86 code is present.

## Field performance observation

The report that triggered this patch used a 2D AuNP-on-mirror r3000 workload
on four RTX A6000 GPUs. It observed an unchanged 862.0 nm resonance, identical
GPU1/GPU2/GPU3/GPU4 spectra, about 1.7x CPU60-to-GPU1 and 2.8x
CPU60-to-GPU4 wall speedups, and 41% four-GPU strong-scaling efficiency. It
also observed gpmeep CPU30 about 18% slower than a separate pymeep 1.34 FP64
CPU30 environment. These measurements are production observations, not
portable v1.0.3 performance promises. v1.0.3 includes no speculative kernel
or MPI transport rewrite; only a localized, independently regression-tested
performance fix may be added to this patch line.

## Required qualification

The release is not accepted until all of the following pass on the exact
package: clean Ubuntu 22.04/glibc 2.35 installation and import, installed
provenance, CPU and CUDA numerical self-checks, native CUDA-code inventory,
one- and two-GPU execution, multi-GPU transport, representative general
examples, adjoint/MPB coverage, and the sealed AuNP regression workload.
