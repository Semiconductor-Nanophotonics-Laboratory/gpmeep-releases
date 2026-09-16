# Sealed AuNP user-workload qualification

The terminal user-workload scope qualifies the supplied AuNP archive under
`workspace/meep_gpu/test_example`:

- `aunp_r4000_repro_20260807code.tar.gz`

The user-selected scientific profile is fail-closed and fixed in code:

- physical Ex / `TM_Ex` only;
- minimum DFT run window `20 µm/c`;
- all-monitor DFT decay tolerance `5e-8`;
- one air-reference and one Au-structure simulation per exact lane.

The archive itself remains byte-exact. The two stop-condition changes are
applied in memory and recorded as physical overrides in every exact,
performance, comparison, checkpoint, and terminal report. A TE result or the
archived `150 µm/c` / `1e-5` defaults cannot satisfy the current schemas.

`Hor_TM_60deg_saw_v6.py` was explicitly user-deferred after its partial exact
CPU run was sealed as `NOT_EVALUATED`; it is not a terminal prerequisite. The
AuNP matrix uses one identical CUDA-capable FP32 receipt for multicore CPU,
one-GPU, and two-GPU execution and differs only in backend/rank/device
selection. It retains every raw numerical, provenance, affinity, device,
memory, log, telemetry, and timing record. Performance uses exactly two
measured runs per topology and publishes arithmetic-mean times and ratios.
Dispersion is diagnostic only and cannot change the release outcome.

Run the controller with one fresh, receipt-bound CUDA-capable FP32 MPI+Python
build. CPU8, GPU1, and GPU2 all use that identical build; only backend, MPI
rank count, and visible GPU devices differ. This isolates backend performance
from precision, source, and dependency differences.

```sh
.envs/meep-gpu-cuda-mpi/bin/python \
  scripts/user-workloads/run_hybrid_aunp_matrix.py \
  --archive ../meep_gpu/test_example/aunp_r4000_repro_20260807code.tar.gz \
  --output evidence/user-workloads/aunp-hybrid \
  --fp32-python PATH --fp32-mpiexec PATH --fp32-receipt PATH \
  --gpu-devices GPU-UUID-0,GPU-UUID-1
```

For M4 release packaging, replay the already sealed M3 performance anchor from
the fresh installed prefix with:

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
```

That controller requires exactly two GPU2/GPU1/CPU8 samples. It accepts only a
fresh `gpmeep-verify-provenance` report for the named package and source commit,
then proves every MPI rank actually loaded that installed Python, extension,
and `libmeep`. Its speed and slowdown thresholds are package-regression gates.
Because it does not repeat M3's thermal-telemetry protocol, its report is
explicitly invalid for a new speed claim and cannot replace the sealed M3
numbers.

Output paths must be absent. The default release gates are all mandatory:

- exactly two measured executions per CPU8/GPU1/GPU2 topology;
- release speedups computed only from ratios of arithmetic-mean times;
- at least 1.5x CPU-FP32 to one-GPU speedup;
- at least 2.0x CPU-FP32 to two-GPU speedup;
- at least 1.1x one-GPU to two-GPU scaling;
- timed `Simulation.run` work must pass the release performance gates;
- full retained TM scientific outputs must pass the packaged CPU-FP64 oracle
  and bidirectional GPU1/GPU2 FP32 comparisons with strict build/backend
  provenance.

AuNP retains its explicitly declared rank/output/memory-admission policy and
user-requested DFT stop-condition adaptations, so its workload end-to-end
value is labelled as adapted rather than byte-exact launcher timing. Every
lane retains complete before/after statistics. Release validation requires
phase-specific calls and points,
zero fallback, and for two GPUs nonempty rank-local work plus measured pinned
MPI traffic, D2H/H2D staging, and completion.

`--development-allow-fewer-repeats` permits a one- or two-repeat diagnostic
run but can only publish `DEVELOPMENT.json`; it can never publish the release
`COMPLETE` seal. The former `--development-allow-one-repeat` spelling remains
as a compatibility alias.
A release failure publishes `FAILED.json` with the journal and retained partial
logs. Timeout, output overflow, SIGINT, or SIGTERM terminates the entire active
MPI process group.

Run the adapter/comparator/controller self-tests with:

```sh
.envs/meep-gpu-cuda-mpi/bin/python -m unittest -v \
  scripts.tests.test_user_workloads
```

The focused regression count is reported by the current test run. Actual CPU/GPU numerical and
performance evidence must be generated after `nvidia-smi` reports a healthy
driver and after the two builds have fresh receipts for the current source.
The controller also records CPU/GPU identity, driver, memory, governors, load,
and refuses release runs when either selected GPU has an existing compute
process at the initial or final qualification boundary.

Every validated lane and numerical comparison is committed atomically to
`CHECKPOINT.json`.  If the host or controller stops, rerun the identical command
with `--resume`.  The controller re-hashes and semantically revalidates every
completed result, rejects any changed input/build/device/limit/gate setting,
moves the incomplete attempt into `resume-history/`, and continues at the first
unfinished lane or comparison.  A live controller still owns `LOCK`, so a
second resume cannot run concurrently.  Final reports bind both the completed
checkpoint and every resume archive marker.

## Historical deferred Hor_TM controller

The exact TERS source contains two long, high-resolution simulations. Its
default per-lane watchdog is seven days so the measured eight-core FP64
reference is not guaranteed to time out before the Gaussian sources have even
finished. Every lane receives a private fontconfig configuration and cache;
plotting by the unmodified source therefore cannot mutate the receipt-bound
Conda prefix between repeats.

The controller below is retained for reproducibility but is not part of the
current terminal scope unless the user explicitly restores it:

```sh
.envs/meep-gpu-cuda-mpi/bin/python3.11 \
  scripts/user-workloads/run_hybrid_ters_matrix.py \
  --input ../meep_gpu/test_example/Hor_TM_60deg_saw_v6.py \
  --output evidence/user-workloads/ters-hybrid \
  --cpu-fp64-python PATH --cpu-fp64-mpiexec PATH --cpu-fp64-receipt PATH \
  --fp32-python PATH --fp32-mpiexec PATH --fp32-receipt PATH \
  --gpu-devices GPU-UUID-0,GPU-UUID-1
```

This runs the byte-exact, natural-lifetime workload once on CPU FP64, one GPU,
and two GPUs, then compares every retained DFT array and gap-enhancement metric.
The retained historical controller used five alternating FP32 CPU/one-GPU/
two-GPU samples and a disclosed fixed-window timing protocol. That obsolete
protocol is not executed or used by the current AuNP-only release scope.
Fixed-window output can never satisfy the exact-correctness role: the adapter,
checkpoint, report, and
independent final verifier all bind the measurement mode, exact task schedule,
command, build receipt, rank-to-GPU topology, and physics-override disclosure.
Every exact and fixed-window lane also closes the full DFT/CSV/PNG/coordinate
artifact inventory by size and SHA-256. The hybrid controller supports the same
`--resume` recovery behavior at every validated lane and comparison boundary.
