# gpmeep v1 release plan

상태: v1 개발 기준 문서  
기준 커밋: `adc67c3466774bbdb1875d996e4e0e50aeb7a845`  
기준 태그: `m3-final-adc67c3`  
원격 저장소: `git@github.com:iverydj/gpmeep.git` (private)

## 확정된 release 순서

1. A: 중복 제거한 Python/adjoint/MPI 기능 coverage를 닫는다.
2. B: clean environment에서 설치 가능한 portable package를 만든다.
3. A와 B가 모두 적대적 감사를 통과하면 `v1.0.0`을 배포한다.
4. C: 성능 고도화는 v1 이후 별도 실험 branch에서 수행한다.
5. C가 정량 promotion gate를 통과할 때만 `v2`를 배포한다.
6. 개선이 noise 수준이거나 회귀가 있으면 v2 tag를 만들지 않는다.

Scheme example, Hor_TM, TERS 및 기존 응용 코드 포팅은 이 v1 범위에 포함하지
않는다.

## Git 작업 정책

- `main`은 검증된 checkpoint만 가진다.
- M3의 immutable 기준점은 `m3-final-adc67c3` 태그로 보존한다.
- A는 `feature/v1-coverage`, B는 `feature/v1-packaging`에서 작업한다.
- source 또는 build identity가 바뀌면 종전 receipt를 새 결과에 재사용하지 않는다.
- `.envs`, `build`, raw telemetry와 대형 evidence는 commit하지 않는다.
- 공개 가능한 요약, 실행 manifest와 verifier만 source tree에 넣는다.
- 인증 token은 source, `.env`, remote URL, log에 저장하지 않는다.
- Git transport는 SSH를 사용한다.

## A: 42-feature coverage

v1은 예제 개수가 아니라 frozen 42-feature contract의 빈칸이 0인 것을 목표로 한다.
종료 코드나 GPU visibility만으로 기능을 닫지 않고, 다음 evidence를 요구한다.

- CPU/GPU complete observable 또는 analytic oracle
- strict CUDA work counter와 CPU fallback 0
- multi-rank 기능이면 participating rank 전체의 work
- GPU2 결과의 직접 numerical comparison
- adjoint이면 gradient 전체와 independent directional difference
- input/source/build/output/telemetry identity의 replay

### Tier 0: 매 build 필수

- source/tool full suite
- C++/MPI suite
- installed Python/API qualification
- MaterialGrid forward/adjoint qualification
- one-rank/two-rank strict CUDA smoke
- build receipt와 relocated-install replay

### Tier 1: fail-fast canary

| route | 주요 기능 |
|---|---|
| `straight-waveguide.py` | Yee/PML/source/DFT |
| `antenna-radiation.py` | 2D Near2Far |
| `dipole_in_vacuum_cyl_on_axis.py` | cylindrical axis |
| `near2far_3d_validation.py` | 3D Near2Far adjoint |
| `edge_emitter_3D.py` | 3D/DFT/LDOS/MPI |
| `3rd-harm-1d.py` | Kerr/DFT reduction |
| `parallel-wvgs-force.py` | stress/force reduction |
| `differential_cross_section.py` | 3D scattering/analytic oracle |

### Tier 2: deduplicated feature closure

| route | 추가 핵심 기능 |
|---|---|
| `binary_grating_oblique.py` | complex/Bloch/symmetry |
| `absorber-1d.py` | Absorber/PML branch |
| `antenna_pec_ground_plane_1D.py` | PEC/small auto policy |
| `cherenkov-radiation.py` | moving/custom source |
| `stochastic_emitter_line.py` | extended stochastic source |
| `multilevel-atom.py` | gain/multilevel polarization |
| `polarization_grating.py` | rotated anisotropy |
| `faraday-rotation.py` | gyrotropy |
| `phase_in_material.py` | time-varying material |
| `material-dispersion.py` | dispersion/Harminv stop |
| `solve-cw.py` | CW solver |
| `finite_grating.py` | DFT accumulation/materialization |
| `refl-quartz.py` | DFT checkpoint/load-minus |
| `mode-decomposition.py` | eigenmode overlap |
| `dipole_in_vacuum_cyl_off_axis.py` | cylindrical N2F/cartesian transform |
| specialized `zone_plate.py` | two-GPU cylindrical N2F |
| `mpi-python-validation-probe.py` | MPI/pinned/CUDA-aware capability |
| `test_adjoint_cyl.py` | cylindrical adjoint derivative |

### Tier 3: adjoint/inverse-design 대표군

다음은 gradient와 constraint surface가 다르므로 중복으로 보지 않는다.

- `mode_converter.py`
- `connectivity_validation.py`
- `near2far_epigraph_validation.py`
- `multilayer_opt.py`
- `waveguide_crossing.py`

launch 전 set-cover verifier가 42개 feature 모두에 release-blocking route를 최소
하나 배정했는지 확인한다. 현재 동결 계약은 32개 route다. 구성은 M3 AuNP anchor
1개, Tier 1 fail-fast 8개, Tier 2 기능 closure 18개, Tier 3 adjoint 5개이며, 88개
script와 42개 notebook 전체를 무조건 실행하지 않는다. 32개 중 실제 새 실행은
30개 case와 MPI Python qualification 1개이고, 나머지 1개는 해시와 최종 감사까지
검증하는 M3 AuNP 봉인 anchor다.

계약과 기존 runner registration은 다음 명령으로 실행 전에 fail-closed 검증한다.

```sh
python scripts/python-validation/v1_feature_coverage.py \
  --repo "$PWD" \
  --evidence-root /path/to/gpmeep-evidence
```

## B: portable package

repository와 distribution 이름은 `gpmeep`를 사용한다. 기존 Python 코드 호환성을
위해 import는 `import meep as mp`를 유지한다. 공식 `pymeep`와 같은 environment에
동시 설치하는 것은 v1에서 지원하지 않고 전용 environment를 기본으로 한다.

v1 primary 설치 방식은 conda/micromamba다.

1. locked dependency environment 생성
2. driver와 GPU architecture probe
3. source 또는 conda package 설치
4. installed CPU/CUDA smoke
5. 선택적 two-GPU qualification

v1 배포물:

- GitHub source와 tagged source archive
- Micromamba lock과 one-command installer
- conda recipe/package
- installed-runtime self-check
- CPU/GPU1/GPU2 사용 예제
- feature matrix와 known limitations
- M3 correctness/performance 요약
- source/build provenance verifier

CUDA, MPI, parallel HDF5와 MPB ABI를 안정적으로 묶기 전까지 universal pip wheel은
v1 release blocker로 두지 않는다.

## CPU baseline

현재 host는 Intel i9-11900K 8 physical cores / 16 logical CPUs다. 동일 AuNP
fixed-work workload를 두 번씩 실행한 결과:

| topology | FDTD mean | E2E mean |
|---|---:|---:|
| CPU8, rank당 physical core 1개 | 163.823542 s | 178.197563 s |
| CPU16, rank당 SMT thread 1개 | 172.129242 s | 184.620092 s |

CPU16은 CPU8보다 FDTD 5.070%, E2E 3.604% 느렸다. CPU16 FDTD CV가 0.012%로
안정적이므로 v1의 공식 current-host CPU baseline은 CPU8 physical MPI다.

## Portability 표현

현재 실제 runtime 검증은 Linux x86_64 + RTX 3090 Ti(sm86)다. release 문서는
실제로 시험한 architecture만 `validated`로 표기한다.

- P0: 현재 sm86 clean install/reinstall
- P1: 다른 NVIDIA architecture 1종 실제 설치/시험
- P2: 다른 architecture 2종 이상과 multi-GPU 1종

v1 final 전 최소 P1을 권장한다. 장비가 없으면 v1을 막지는 않지만 “portable build
policy, runtime validated on sm86 only”라고 제한을 명시한다.

## v1 release gate

- clean clone install/build PASS
- relocated import PASS
- Tier 0 전체 PASS
- 42-feature coverage 빈칸 0
- 대표 CPU/GPU numerical comparison 전체 PASS
- strict single/two-GPU work, UUID, traffic PASS
- M3 AuNP terminal replay PASS
- GPL/upstream attribution와 source 제공 조건 충족
- fresh uninstall/reinstall PASS
- 최종 적대적 감사 unresolved Critical/High/Medium 0

## MPI adjoint gradient 수정의 upstream 출처

M4-A의 `near2far_3d_validation.py` CPU8 검증은 objective와 finite
difference는 유지되지만 MaterialGrid adjoint gradient 벡터가 MPI rank
수에 따라 달라지는 upstream Meep 결함을 발견하고 fail-closed했다.
진단 rank sweep은 1/2/4/8 rank에서 같은 objective를 확인했지만 gradient
벡터의 상대 차이가 유의미했다.

gpmeep은 아직 upstream `master`에 merge되지 않은 NanoComp/meep
[PR #3274](https://github.com/NanoComp/meep/pull/3274), commit
`897dbe06b0ed6442d34f2a940f21acbff9375523`의 수정을 출처와 commit
identity를 보존한 manual backport로 적용한다. 핵심은 component Yee
grid에 맞춘 persistent DFT halo clamp, linked-list 순서 대신 공간
identity로 forward/adjoint DFT chunk matching, 좌표 기반 DFT indexing,
차원별 bounds check이다. 추가 communication이나 monitor gather는 없다.

v1 수용 조건은 단순 backport와 compile 성공이 아니라, 추가된
`python/tests/test_adjoint_chunks.py`의 CPU/CUDA 통과, 1/2/4/8 MPI rank
gradient invariant, 독립 finite-difference correctness, 기존 adjoint/DFT
회귀검증 통과이다. upstream PR이 merge되면 commit 차이를 다시
비교해 backport 유지/제거를 결정한다.

## C와 v2 promotion gate

C는 `perf/v2-lab`에서만 수행한다. 아래를 모두 통과해야 v2로 승격한다.

1. v1 correctness/coverage 전체 유지
2. 대표 production FDTD geometric mean 최소 10% 개선
3. 실제 대형 workload E2E 최소 한 개에서 10% 개선
4. release-blocking workload 회귀 3% 이하
5. GPU2/GPU1 scaling 1.1배 이상 유지
6. 근거 없는 peak device-memory 10% 초과 증가 없음
7. throttle-clean, 반복 변동 gate 통과

통과하지 못하면 benchmark와 실험 branch만 보존하고 `v1.0.0`을 최신 release로
유지한다.
