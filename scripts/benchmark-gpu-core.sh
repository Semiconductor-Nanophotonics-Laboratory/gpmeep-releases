#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda"
RESULT_DIR="${REPO_ROOT}/benchmark-results"
BENCHMARK="${REPO_ROOT}/build/meep-cuda-fp32/tests/gpu-performance"

if [[ ! -x "${BENCHMARK}" ]]; then
  echo "error: build ${BENCHMARK} with scripts/build-meep-cuda.sh first" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${MAMBA_CACHE}" "${MAMBA_PACKAGES}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT="${RESULT_DIR}/core-${TIMESTAMP}.csv"
LOG="${RESULT_DIR}/core-${TIMESTAMP}.log"
export XDG_CACHE_HOME="${MAMBA_CACHE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"

RUN_ENVIRONMENT=(
  --clean-env
  --env "CCACHE_DIR=${MAMBA_CACHE}/ccache"
)
for variable in MEEP_GPU_DEVICE CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \
  MEEP_GPU_BENCH_QUICK MEEP_GPU_BENCH_2D_STEPS MEEP_GPU_BENCH_3D_STEPS \
  MEEP_GPU_MIN_SPEEDUP MEEP_GPU_MAX_TRANSFER_BYTES_PER_CELL_STEP; do
  if [[ -v "${variable}" ]]; then
    RUN_ENVIRONMENT+=(--env "${variable}=${!variable}")
  fi
done

set +e
"${MAMBA}" --no-rc run "${RUN_ENVIRONMENT[@]}" \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
  "${BENCHMARK}" 2>&1 |
  tee "${LOG}" |
  awk -F, '
    /^case,linear_pixels,cells,steps,/ { print; next }
    /^(2d-vacuum-dft|2d-lorentz-dft|2d-gyrotropic-llg-dft|2d-multilevel-dft|3d-vacuum-dft),/ { print }
  ' >"${RESULT}"
status="${PIPESTATUS[0]}"
set -e

echo "GPU core benchmark CSV written to ${RESULT}"
echo "GPU core benchmark log written to ${LOG}"
exit "${status}"
