# Choosing CPU, automatic, or CUDA execution

gpmeep does not promise that a forced GPU run is faster than CPU for every
simulation. CUDA can win when each rank has enough parallel work per launch to
repay device dispatch, synchronization, and MPI costs. A small spatial domain
can remain launch dominated even when it executes millions of timesteps.

The default backend remains CPU. To ask gpmeep to make its existing bounded
owner-level choice, start with the opt-in automatic backend and verify the
result for the workload:

```python
import meep as mp

mp.gpu.set_backend("auto")
sim = mp.Simulation(...)
sim.run(...)

print(mp.gpu.requested_backend)
print(mp.gpu.active_backend)
print(mp.gpu.backend_diagnostic)
```

The process-level equivalent must be set before Python starts:

```sh
MEEP_GPU_BACKEND=auto python simulation.py
```

Inspect the backend after a simulation has initialized or run. Automatic
selection is made for the fields owner using rank-local work, so the value
before simulation initialization is not a workload decision. The diagnostic
records the policy and the local-cell comparison that selected CPU or CUDA.
The exact crossover threshold is an implementation detail and may evolve; do
not copy it into application logic.

The current policy uses rank-local cell count and conservative host/device
facts. It does not know a future stop condition or timestep count, model DFT
monitor/frequency launch count, measure Python/JAX/NLopt host-side share, or
change the MPI rank/GPU layout chosen by the launcher. Those factors still
require an end-to-end CPU/CUDA comparison for important production workloads.

Use `cuda` instead of `auto` when the purpose is to qualify the CUDA path,
measure a known GPU-suitable fixed workload, or require fail-closed GPU
execution. Forced CUDA prevents silent CPU backend selection, but it is not a
performance guarantee.

## Workload classification

| Workload shape | Recommended starting backend | Reason |
|---|---|---|
| Large rank-local grid with substantial repeated FDTD work | `auto`, then verify CUDA | Enough work can amortize launch and synchronization overhead |
| Small 1D/2D grid or short run | `auto`; also time CPU if the run is short | The cell floor can catch a small grid, but automatic does not predict run length |
| Small grid with a very long run | `auto` | Timestep count alone does not make tiny per-launch work GPU-suitable |
| Many tiny DFT monitors or frequencies | `auto`, then compare CPU | Monitor-update launches can dominate, and monitor count is not an automatic-policy input |
| Multi-GPU with small rank-local subdomains | choose fewer ranks/GPUs before launch; then use `auto` or an explicit backend | The library cannot resize a running MPI/GPU layout |
| Adjoint/optimization with large FDTD phases | `auto`, profile end to end | FDTD may accelerate, while packaged MPB, JAX/NLopt, and Python control remain CPU-side and are not policy inputs |

For multi-GPU runs, use one MPI rank per selected physical or MIG GPU and
confirm distinct device identifiers in `mp.gpu.statistics()`. Adding GPUs is
useful only while the rank-local domain stays above the crossover region.

## v1.0.2 bounded crossover evidence

The v1.0.2 investigation deliberately examined only two large inversion cases
and stopped before structural kernel work. The exact v1.0.1 package was used
for the `3rd-harm-1d.py` CPU/CUDA/ablation/automatic probes and the
`parallel-wvgs-force.py` automatic probe. The latter case's CPU/forced-CUDA
context comes from the earlier implementation-equivalent M4 window. All cited
probe measurements are cold, whole-process, single samples on one host with an
RTX 3090 Ti. The new CPU lane is one process/one FDTD worker. They are
diagnostic observations, not release speed gates or universal hardware ratios.

### `3rd-harm-1d.py`

| Lane | Time | Relative to CPU | Result |
|---|---:|---:|---|
| CPU | 29.473 s | 1.000x | reference |
| forced CUDA baseline | 139.674 s | 4.739x slower | 4,591 numerical metrics pass |
| forced CUDA + DFT monitor batching | 135.161 s | 4.586x slower | 4,591 metrics pass |
| forced CUDA + field-phase batching | 122.020 s | 4.140x slower | 4,591 metrics pass |
| `auto` | 29.172 s | 0.990x of CPU sample | selected CPU; 4,591 metrics pass |

Forced DFT batching reduced DFT update-kernel launches from 4,621,848 to
1,225,490; its single sample was 4.513 seconds (3.23%) shorter. Forced
field-phase batching reduced CUDA curl calls from 7,353,045 to 2,451,085; its
sample was 17.654 seconds (12.64%) shorter, yet remained more than four times
slower than CPU. This single cold evidence is insufficient to activate either
path globally and neither sample closed the crossover. Automatic mode selected
CPU for 2,503 rank-local cells. Its runtime availability-probe,
device-enumeration, and device-selection counters were all zero, and its
whole-process sample was 110.502 seconds shorter than the independent forced-
CUDA sample; this is an observed difference, not a repeatable speedup claim.

### `parallel-wvgs-force.py`

The prior correctness run observed CPU 107.543 seconds and forced CUDA 247.051
seconds. Those figures are a single cold implementation-equivalent comparison,
not a speed gate. The exact v1.0.1 package was then probed with `auto`: it
selected CPU for 14,729 rank-local cells and completed in 108.021 seconds,
with all three gpmeep runtime availability/device-discovery counters equal to
zero. The observations were made in separate validation windows, so neither
the small CPU/auto difference nor the prior forced-CUDA gap is a paired exact-
package performance claim.

## v1.0.2 decision

The safe remedy for these two cases is the existing automatic host-floor
policy plus explicit user-facing diagnostics, not mandatory GPU execution.
v1.0.2 therefore documents this behavior and adds bounded evidence for these
two cases without changing the FDTD numerical algorithm, Python API, ABI, or
default CPU backend. Global activation of the experimental batching paths is
rejected for v1.0.2 because one bounded sample did not close the CPU/GPU gap.
CUDA Graphs, persistent kernels, deeper phase fusion, or a new dynamic
crossover model are structural v1.1.0 candidates and require a separately
approved performance matrix.

The official repeated v1 speed anchor remains the large AuNP workload, where
GPU1 and GPU2 accelerate FDTD by 9.821x and 14.959x versus CPU8 on the qualified
host. See [the performance summary](GPMEEP_PERFORMANCE.md) for its exact scope.
