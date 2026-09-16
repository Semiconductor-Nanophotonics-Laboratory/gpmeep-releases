# gpmeep v1.0.1 release notes

v1.0.1 is a provenance-correction patch release for the v1.0.0 CUDA 12.4,
Open MPI, FP32 package line. It does not move or rewrite the immutable v1.0.0
tag and does not claim a new performance measurement.

## Corrected performance binding

The v1.0.0 documentation and installed-package replay validator accidentally
embedded means from the superseded `3ca2476` AuNP terminal. That terminal was
already marked `qualification_eligible=false`. The implementation and the
accepted measurements remain valid; the published evidence binding was wrong.

v1.0.1 binds the documentation, package manifest, self-check, provenance
verifier, and installed AuNP replay validator to the accepted terminal:

- source commit: `adc67c3466774bbdb1875d996e4e0e50aeb7a845`
- report SHA-256:
  `50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc`
- COMPLETE SHA-256:
  `7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f`
- final adversarial-audit SHA-256:
  `4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639`

The accepted two-repeat means and ratios are:

| topology | FDTD mean | end-to-end mean | FDTD vs CPU8 | E2E vs CPU8 |
|---|---:|---:|---:|---:|
| CPU8 physical ranks | 163.823542 s | 178.197563 s | 1.0000x | 1.0000x |
| GPU1 | 16.680281 s | 41.041740 s | 9.8214x | 4.3419x |
| GPU2 | 10.951151 s | 30.025477 s | 14.9595x | 5.9349x |

GPU1-to-GPU2 scaling is 1.5232x for FDTD and 1.3669x end to end. These values
are workload- and host-specific, not universal hardware ratios.

## Contract changes

- Conda distribution version advances from 1.0.0 to 1.0.1.
- Conda build 1 owns deterministic, prefix-relative checked-hash bytecode for
  every installed `meep` Python source. Normal imports therefore do not leave
  an unowned `meep/__pycache__` namespace behind after package removal.
- The installed release manifest advances to schema 2 and embeds the complete
  accepted M3 anchor identity and means.
- The installed AuNP matrix evidence advances to schema 2 and records that
  same anchor in every new report.
- The package provenance verifier and self-check fail closed if the version or
  accepted anchor differs.
- The installed provenance verifier also requires the exact Linux build number
  and build string, so an obsolete v1.0.1 candidate cannot be accepted merely
  by supplying that candidate's matching package hash.

There are no CUDA-kernel, numerical algorithm, Python API, or ABI changes from
v1.0.0. The patch is deliberately limited to evidence binding, fail-closed
validation, documentation, and package lifecycle integrity.

The v1.0.1 candidate must still pass a clean package build, fresh-prefix
installation, CPU/GPU1/GPU2 self-checks, installed AuNP replay, uninstall and
reinstall lifecycle checks, regression suites, and an independent adversarial
audit before publication.

## Version policy

Compatible corrections and improvements continue as v1.x.y. A measured
performance improvement below 10% does not prevent a release when correctness,
regression, packaging, and adversarial gates all pass. v2 is reserved for a
genuinely incompatible API or ABI change.
