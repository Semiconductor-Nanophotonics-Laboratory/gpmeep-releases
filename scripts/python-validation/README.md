# Python CPU/CUDA validation harness

This harness inventories every checked-in Python test, example, and notebook,
while deliberately excluding Scheme. It runs selected cases in fresh working
directories against the same FP32 gpmeep build with `MEEP_GPU_BACKEND=cpu` and
`MEEP_GPU_BACKEND=cuda`.

The current terminal scope uses 70 deduplicated paired CPU/CUDA units, four
host-compatibility units, and five specialized correctness/topology units.
The specialized set includes MPI Python, one- and two-GPU examples, and
Near2Far-adjoint validation, but no repeated example-speed benchmark. Official
example timings are diagnostic only. Release performance is measured only by
the AuNP controller, with exactly two CPU8/GPU1/GPU2 executions and ratios of
arithmetic-mean times. `run_current_scope_release.py` combines that AuNP
terminal and its adversarial audit with the 79 physical example units and the
explicit Hor_TM user-deferral receipt. The older M1--M3 aggregation and
repeated-performance scripts remain solely for historical evidence replay.

The CUDA selection is strict in gpmeep: unsupported stepping must fail. A
process-local `sitecustomize.py` hook captures the imported package path,
precision, runtime/device inventory, selected backend, and all `mp.gpu`
dispatch counters at interpreter shutdown. A CUDA case that reports any CPU
phase calls fails the backend contract.

Run the initial representative gate:

```sh
scripts/run-python-validation.sh --tier smoke
```

Run one case:

```sh
scripts/run-python-validation.sh \
  --case python/tests/test_pml_cyl.py
```

List the complete classified inventory without running it:

```sh
scripts/run-python-validation.sh --list
```

Evidence is written by default to
`artifacts/python-validation/<UTC timestamp>/report.json` and `report.md`.
Per-process stdout, stderr, working files, and GPU statistics are kept below
the same directory. The JSON records source/build/install provenance and the
manifest hash.

Manifest schema 2 assigns every materialized item a `compute_scope`:
`fdtd_cuda` for runnable FDTD work, `host_only` for deterministic host
algorithms, `mpi_fdtd_cuda` for the separate MPI gate, and `coverage_only` for
non-runnable coverage relationships. Report schema 4 retains that scope on
every result. Host-only cases may be checked on both named runner backends but
are never counted as GPU coverage or performance evidence.

Runnable examples can emit one prefixed JSON object and select the
`json_metrics` comparison mode in `manifest.json`. The runner recursively
flattens numeric objects/arrays, rejects duplicate manifest or metric keys,
ambiguous flattened paths, missing, non-finite, or non-numeric metrics,
requires identical CPU/CUDA metric keys, and applies the declared
absolute/relative tolerance to every metric. This avoids treating a plotting
script's unstable progress text as its physical oracle while keeping the
normal interactive behavior unchanged.

Examples can instead run through `run_example_oracle.py`. The driver wraps
`Simulation.run` without changing
the timestep loop and, immediately after each completed run, records the
exact timestep and Meep time together with whole-cell field energy and the
global DFT norm. It emits the single
`gpmeep-generic-example-metrics:` JSON record used by the same tolerance
engine. Every registration must pin the expected run count and at least one
exact timestep/work contract, one explicit timestep range for every run, or a
minimum physical-signal threshold. Exact and range forms cannot be mixed. The driver
snapshots phase counters before and immediately after `Simulation.run`, so a
zero-step run or CUDA work caused only by the post-run energy probe cannot
masquerade as FDTD coverage. Raw per-run timestep, selected-backend dispatch,
and measurement-only dispatch deltas are retained in a separate
`gpmeep-generic-example-audit:` record. A real two-segment CPU/strict-CUDA
regression also proves that the intermediate energy and DFT probes do not
change the later field trajectory.

Registrations may also request named top-level numerical results from the
completed example namespace. Every requested result must have exactly one
independent expected-shape contract. Scalars use `rank: 0`; non-scalar arrays
have a nonempty exact shape. NumPy-like shape and size are checked before
`tolist()`, so a million-plus-value result fails before Python-list
materialization. Direct real/imaginary values and redundant L2, maximum, and
weighted checksums are compared. Missing, private, duplicate, empty, ragged,
nonnumeric, nonfinite, oversized, or shape-inconsistent results fail closed.
Optional minimum-L2 and maximum-absolute contracts enforce nonzero published
signals and analytic-error bounds independently of CPU/CUDA agreement.

Examples which bypass `Simulation.run`, deliberately end with fully
decayed zero fields, use stochastic sources, or need a richer published
observable retain a dedicated oracle or an explicit corresponding-test
classification.

Non-run classifications such as `compatibility_only`, `deferred_milestone`,
`expected_feature_gap`, `external_dependency`, `notebook_duplicate`, and
`oracle_pending` remain visible in every report. They are not counted as
successful execution. Selecting an `expected_feature_gap`,
`external_dependency`, or `oracle_pending` case returns blocker code 4, so a
premature full gate cannot appear successful. A missing dependency on a case
classified as runnable is also a blocker rather than a skip.

Exit codes are stable:

| Code | Meaning |
| ---: | --- |
| 0 | Selected runnable cases passed |
| 1 | Process, timeout, or unittest-skip policy failed |
| 2 | Invalid manifest or invocation |
| 3 | CPU/CUDA observable mismatch |
| 4 | Selected case is blocked, unsupported, or has unresolved/failed coverage |
| 5 | Backend provenance or strict-CUDA dispatch contract failed |
| 6 | Source, build, extension, or retained evidence integrity failed |

The generic timer includes Python startup and is useful for evidence triage,
not as the release speed gate. Every report therefore sets
`performance_evidence.valid_for_speed_gate` to false. If another GPU workload
overlaps a correctness run, record that explicitly:

```sh
scripts/run-python-validation.sh --concurrent-gpu-work --tier smoke
```

This adds `concurrency_detected` to the invalidity reasons. Release performance
measurements use dedicated warm-up, isolation, and repeated-run benchmarks.
Strict CUDA is intentionally used here to prove coverage even for tiny
launch-latency-dominated fixtures. Their process times must not be reported as
speedups; normal `auto` policy keeps rank-local domains below the calibrated
cell threshold on the CPU.

Run the harness self-tests without Meep or a GPU:

```sh
.envs/meep-gpu-cuda-mpi/bin/python -m unittest discover \
  -s scripts/python-validation/tests -p 'test_*.py' -v
```

The release-candidate hardening regression is 291/291 tests. The separate
source, provenance, packaging, and benchmark-tool regression is 1173/1173
under `scripts/tests`. Both counts are tied to the exact candidate commit and
must match the sealed release evidence.

`aggregate_validation.py` can seal exact, disjoint `(case_id, backend)` batch
assignments and later aggregate their completed runner directories. It
replays raw runner semantics, re-derives manifest-owned assignment policy,
binds build/source/manifest/native-artifact identities, and writes `COMPLETE`
only after a second stable reread. Covered-by relationships, MPI semantics,
and external gates remain explicit unresolved inputs; this aggregator is an
evidence combiner, not yet the final release qualification gate.

The second M9 feature-expansion batch raises the generic direct-example set to
19 scripts. It adds complete or representative workflows for Kerr `chi3`,
Harminv plus direct LDOS, cylindrical `m=+1,-1,0` near-to-far fields,
dispersive absorbed-power density, MPB-derived optical force, and deterministic
stochastic `CustomSource` ensembles. Validation profiles are explicit command
line modes which retain the scientific operators while bounding repeated
parameter sweeps; each script's published defaults remain available. The raw
CPU/strict-CUDA logs, timings, hashes, retained fail-closed attempts, and
compact measurements are stored in
private raw-evidence storage and are not included in public source snapshots.

These generic process times remain diagnostics. In this batch the 320k-cell
absorbed-power case was 3.575x faster and the stochastic case 1.390x faster in
single probes, while the small 1D, LDOS, cylindrical, and symmetry-reduced
force profiles were slower or near parity under forced CUDA. Those small cases
are strict-CUDA coverage tests; the normal automatic policy is expected to
keep sub-threshold rank-local domains on CPU. Only the later receipt-bound,
interleaved benchmark can publish performance claims.

The fourth M9 feature-expansion batch adds material phasing plus a static
negative control, densely sampled Absorber/PML ringdown, all three antenna
polarizations with signed-radial/tangential closure, the checked-in four-length
taper sweep against an independent FP64 reference, and analytic signed-order
dispersion closure for normal/oblique diffraction. It compares 182,556 metrics
and keeps complete material/field grids or complete order/field/coefficient
arrays wherever the observable has a fixed physical gauge. Raw development
evidence is retained privately and is not included in public source snapshots.

The Absorber/PML oracle keeps both complete 4096-sample complex ringdown traces
verdict-bearing. Its signed weighted trace checksum is diagnostic-only: that
synthetic scalar is cancellation-sensitive and can move by about 2% even when
the complete CPU/CUDA trace has relative L2 error below `1e-5`, while the trace
L2/max summaries, decay ratios, stopping times, and branch-equivalence gate all
remain verdict-bearing.

These cases prove strict-CUDA FDTD and DFT accumulation plus CPU/CUDA
compatibility of the complete workflows. M20--M23 subsequently moved the
Cartesian and cylindrical Near2Far Green transforms, Cartesian observation,
and the adjoint Near2Far VJP to CUDA. MPB eigensolves and the final
flux/energy/force spectral contractions remain host-side. Accordingly, the
validation harness's zero-fallback assertion is a statement about its declared
FDTD/DFT/LDOS/adjoint-Near2Far phase counters, not every Python postprocessing
operation. CPU-first, cold-process elapsed observations are execution receipts
only and must not be converted into speedups. Small forced-CUDA workflows are
not release speed evidence.

The ninth general-example batch adds the two extended-source workflows.
`stochastic_emitter_line.py` retains all 15 orthonormal cosine line-source
modes plus 45-degree unitary rotations of the first two on flat and textured
Ag-backed geometries. It counts every real `amp_func` callback, closes the
quadratic-flux trace to `2.34e-7` relative L2 or better on CPU and `2.14e-7` or
better on strict CUDA, and verifies 0.109% M=12-to-M=15 ensemble convergence.
The texture changes the complete modal response by 82.64%. All 3,099 flattened
CPU/CUDA metrics pass. The 34-run one-process development observations are
208.89 seconds on CPU and 92.75 seconds on one RTX 3090 Ti, a diagnostic
2.252x ratio.

`stochastic_emitter_reciprocity.py` retains directional ODD_Z eigenmode power
from forward dipoles and the cubature-weighted backward plane-wave DFT overlap.
Its 37 trajectories represent all 55 dipole positions through 28 unique
textured positions, directly check translated/reflected partners, validate the
N+3 DFT metadata topology, and rerun representative forward/backward spectra
at twice the runtime. The worst runtime-convergence change is 8.879%. The
normalized forward/backward reciprocity residual is 3.693% relative L2 with a
4.126% worst point on CPU and nearly identical strict-CUDA values. All 5,046
flattened metrics pass with no CPU fallback. Direct observations are 1,497.55
seconds on CPU and 804.55 seconds on one RTX 3090 Ti, a diagnostic 1.861x
ratio. These cold-process correctness observations are not release benchmark
claims. The conservative `auto` policy still selects CPU for these small
rank-local domains; explicit CUDA is faster for their long time horizons, so
workload-aware automatic promotion remains a policy-tuning item.
`extraction_eff_ldos.py` is classified as covered by the stricter existing
`test_ldos_ext_eff` cylindrical/3D equivalence test. Raw logs and their
comparison receipt are retained under `gpmeep-evidence-notes/dev-tools/m9-b9-*`
outside the source repository.
