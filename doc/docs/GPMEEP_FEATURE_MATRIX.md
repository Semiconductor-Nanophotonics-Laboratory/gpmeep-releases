# gpmeep v1 feature matrix and known limits

The release contract is feature based, not a claim that every similar example
was repeated. The frozen source of truth is
`scripts/python-validation/v1_feature_coverage.json`: 42 features are covered
by 32 deduplicated release routes. M4-A passed all 32 routes, 13 controller
task units, and every required CPU/CUDA/MPI comparison at source commit
`3b750a83fe9ec056793af05c0402d6473a244170`.

## Validated feature families

| family | validated contract features |
|---|---|
| Solver and coordinates | real Cartesian, complex Bloch, 1D/2D/3D/cylindrical dimensions, PML conductivity, Absorber, PEC, symmetry |
| Sources and materials | standard, custom, stochastic, dispersive, Kerr, gain/multilevel, anisotropic, gyrotropic, time-varying material, MaterialGrid |
| Solvers and stopping | dynamic stopping/Harminv, continuous-wave solver |
| DFT and observables | DFT accumulation, reduction, materialization, checkpoint/load-minus, LDOS, eigenmode overlap |
| Near2Far | 2D, 3D, cylindrical, cylindrical-to-Cartesian, Near2Far adjoint |
| Adjoint/inverse design | Cartesian and cylindrical adjoint, full gradients, independent directional derivatives, MaterialGrid |
| Distributed runtime | device ownership, pinned MPI, capability-gated CUDA-aware MPI, multi-rank, multi-GPU correctness and strong scaling |
| Runtime policy/evidence | small/production automatic backend policy, long-horizon small-domain policy, durable resume replay |

The five non-duplicate adjoint representatives are mode conversion,
connectivity constraint, Near2Far epigraph, multilayer optimization, and
waveguide crossing. All five MPI routes passed CPU8, GPU1, GPU2, and all three
pairwise comparisons. The specialized two-GPU route passed pinned/waitsome and
CUDA-aware/waitall transports with distinct physical GPUs and no CUDA-rank CPU
fallback.

## Packaging matrix

| property | v1 package |
|---|---|
| Python import | `import meep as mp` |
| Precision | FP32 for both CPU and CUDA lanes |
| CUDA fast math | disabled |
| MPI | Open MPI, multi-rank and multi-GPU |
| Python/adjoint/MPB | included |
| Scheme | excluded by project scope |
| Packaged platform | Linux x86-64 |
| Runtime architecture validated | sm86 (RTX 3090 Ti) |
| Portable build policy | CUDA 12.4 native sm60+ targets exposed by nvcc plus newest PTX |
| Co-install with `pymeep` | unsupported and dependency-constrained |

## Known limits and non-claims

- Other NVIDIA architectures have not yet received actual release-runtime
  qualification. The binary policy is portable; validation is currently sm86.
- Windows, macOS, Linux aarch64, AMD GPU, and Intel GPU packages are not v1
  deliverables.
- Scheme is intentionally disabled. Python is the primary interface.
- Native libGDSII-backed helpers are not enabled in the frozen v1 build.
  Python-side `gdstk` is present, but that is not a substitute for claiming the
  native Meep GDSII API as validated.
- The package is single precision. Scientific workloads that require a strict
  FP64 reference should compare against a separate official CPU Meep
  environment.
- The 42-feature set is the agreed broad, deduplicated contract. It is not a
  proof over every user callback, every parameter combination, or every
  upstream example/notebook.
- CUDA launch and transfer overhead can make tiny problems slower. Production
  evidence and automatic crossover policy, not a small self-check, determine
  whether GPU execution is appropriate.
- Universal pip wheels are not a v1 release gate because CUDA, MPI, parallel
  HDF5, MPB, and native ABI dependencies are installed as one Conda runtime.
