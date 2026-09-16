# gpmeep v1 correctness and performance summary

Only the sealed, repeated M3 AuNP workload is an official v1 speed anchor.
M4 feature-coverage timings are single cold correctness executions and are
explicitly invalid for speed claims.

## Official current-host anchor

The workload is AuNP TM-only with resolution 4000, minimum run window 20, and
DFT-field decay threshold `5e-8`. Each official topology is the mean of two
accepted executions on an Intel i9-11900K host and two RTX 3090 Ti GPUs.

| topology | FDTD mean | end-to-end mean | FDTD vs CPU8 | E2E vs CPU8 |
|---|---:|---:|---:|---:|
| CPU8, one physical core per MPI rank | 163.823542 s | 178.197563 s | 1.0000x | 1.0000x |
| GPU1 | 16.680281 s | 41.041740 s | 9.8214x | 4.3419x |
| GPU2 | 10.951151 s | 30.025477 s | 14.9595x | 5.9349x |

GPU2/GPU1 FDTD scaling is 1.5232x and end-to-end scaling is 1.3669x. These figures are tied to this host,
workload, device model, driver/toolkit, topology, and acceptance evidence; they
are not universal hardware ratios.

The authoritative terminal is source commit
`adc67c3466774bbdb1875d996e4e0e50aeb7a845`. Its `report.json`, `COMPLETE`,
and final adversarial-audit SHA-256 values are respectively
`50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc`,
`7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f`,
and `4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639`.
The earlier `3ca2476` terminal was explicitly marked
`qualification_eligible=false`; v1.0.0 documentation and its shipped replay
validator accidentally used that superseded terminal's means. v1.0.1 corrects
the provenance binding without changing the measured implementation.

## Installed-package replay gate

Each release candidate is also replayed from a fresh, relocated Conda prefix.
This replay uses the installed prefix's Python and MPI launcher while the
source-clone controller binds every rank's loaded Python executable, Meep
extension, and `libmeep` to a freshly generated package-provenance report.
Exactly two samples are run for GPU2, GPU1, and CPU8 in that fixed order per
repeat. There is no repeat-count or physics override.

```sh
/absolute/installed/prefix/bin/python3.11 \
  scripts/user-workloads/run_installed_aunp_performance_matrix.py run \
  --archive /absolute/path/aunp_r4000_repro_20260807code.tar.gz \
  --output /absolute/new/evidence/directory \
  --prefix /absolute/installed/prefix \
  --package-provenance /absolute/new/package-provenance.json \
  --source-commit FULL_40_CHARACTER_COMMIT \
  --package-sha256 FULL_64_CHARACTER_PACKAGE_SHA256 \
  --gpu-devices 0,1

/absolute/installed/prefix/bin/python3.11 \
  scripts/user-workloads/run_installed_aunp_performance_matrix.py validate \
  --output /absolute/new/evidence/directory
```

The terminal validator re-hashes the controller, adapter, archive, provenance
report, logs, and every nested lane output; independently replays the installed
package and rank runtime attestations; re-derives CPU-core/GPU-device topology,
fixed timestep work, timings, means, dispersion, and speedups; and rejects
extra or linked output files. The mandatory gates are:

- exactly two finite positive samples per topology with max/min at most 1.20;
- CPU8-to-GPU1 speedup at least 1.5x for FDTD and end-to-end time;
- CPU8-to-GPU2 speedup at least 2.0x for both timings;
- GPU1-to-GPU2 scaling at least 1.1x for both timings;
- no installed mean more than 1.25x slower than the sealed M3 anchor.

This installed replay deliberately has `valid_for_new_speed_claim=false`.
It detects packaging regressions but has no fresh thermal-telemetry seal, so it
cannot replace or update the official M3 figures above.

## CPU topology baseline

| topology | FDTD mean | E2E mean |
|---|---:|---:|
| CPU8 physical MPI ranks | 163.823542 s | 178.197563 s |
| CPU16 SMT MPI ranks | 172.129242 s | 184.620092 s |

CPU16 was 5.070% slower in FDTD and 3.604% slower end to end. The CPU8
physical-rank topology is therefore the official baseline on this host.

## Coverage timings are not benchmarks

M4-A compared 5,003,547 numerical metrics across 25 generic cases with zero
failures, plus five CPU8/GPU1/GPU2 MPI routes and a specialized two-GPU
transport route. Some small CUDA correctness lanes were slower than CPU due to
fixed launch/transfer overhead. They are disclosed rather than hidden. The
existing automatic policy can select CPU from its rank-local cell floor; the
two exact-package v1.0.2 probes below confirmed that decision for two observed
launch-dominated cases, not for every M4 route or workload shape.

Use a representative fixed-work simulation, repeated accepted samples,
thermal/throttle checks, and an appropriate physical-core CPU baseline before
making workload-specific performance claims.

## Forced-GPU crossover guidance

v1.0.2 separately investigated two large `GPU << CPU` observations under a
bounded diagnostic policy. It found that forced DFT batching and field-phase
batching reduced launch counts but did not make the small-domain
`3rd-harm-1d.py` workload competitive with CPU. The existing `auto` backend
selected CPU before gpmeep performed CUDA availability or device-discovery
calls for both `3rd-harm-1d.py` and `parallel-wvgs-force.py`. The forced
`parallel-wvgs-force.py` comparison came from a separate implementation-
equivalent M4 validation window, so it is classification context rather than a
paired exact-package timing claim.

These crossover probes are single cold diagnostic samples, not replacements
for the repeated AuNP speed anchor above. Their actual times, numerical checks,
telemetry, interpretation limits, and backend-selection instructions are in
[Choosing CPU, automatic, or CUDA execution](GPMEEP_GPU_CROSSOVER.md).
