#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda"

if [[ "${MEEP_GPU_ENV_ACTIVE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
  if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
    echo "error: CUDA environment is missing; run scripts/create-env.sh cuda" >&2
    exit 1
  fi
  mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}"
  export XDG_CACHE_HOME="${MAMBA_CACHE}"
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
  export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"
  RUN_ENVIRONMENT=(
    --clean-env
    --env MEEP_GPU_ENV_ACTIVE=1
    --env "XDG_CACHE_HOME=${MAMBA_CACHE}"
    --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
    --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}"
    --env "CCACHE_DIR=${MAMBA_CACHE}/ccache"
  )
  for variable in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH MEEP_GPU_CUDA_ARCHS \
    MEEP_GPU_DEVICE CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER; do
    if [[ -v "${variable}" ]]; then
      RUN_ENVIRONMENT+=(--env "${variable}=${!variable}")
    fi
  done
  exec "${MAMBA}" --no-rc run "${RUN_ENVIRONMENT[@]}" \
    --root-prefix "${MAMBA_ROOT}" \
    --prefix "${PREFIX}" \
    "${BASH_SOURCE[0]}" "$@"
fi

if [[ "${CONDA_PREFIX:-}" != "${PREFIX}" ]]; then
  echo "error: expected isolated environment ${PREFIX}, got ${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi
for tool in cmake c++ nvcc ninja; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the isolated environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done
for compiler_variable in CC CXX; do
  COMPILER_PATH="${!compiler_variable:-}"
  if [[ ! -x "${COMPILER_PATH}" || "${COMPILER_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${compiler_variable} resolved outside the isolated environment: ${COMPILER_PATH:-<unset>}" >&2
    exit 1
  fi
done

BUILD_DIR="${REPO_ROOT}/build/cuda-prototype"
FAST_MATH="${MEEP_GPU_FAST_MATH:-OFF}"
CUDA_ARCH_ARGUMENT=()
if [[ -n "${MEEP_GPU_CUDA_ARCHS:-}" ]]; then
  CUDA_ARCH_ARGUMENT=(-DMEEP_GPU_CUDA_ARCHITECTURES="${MEEP_GPU_CUDA_ARCHS}")
else
  CUDA_ARCH_ARGUMENT=(-DMEEP_GPU_CUDA_ARCHITECTURES=AUTO)
fi

cmake \
  -S "${REPO_ROOT}/cuda" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CONDA_PREFIX}/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX}" \
  -DMEEP_GPU_FAST_MATH="${FAST_MATH}" \
  "${CUDA_ARCH_ARGUMENT[@]}"

cmake --build "${BUILD_DIR}" --parallel "${MEEP_GPU_MAKE_JOBS:-$(nproc)}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "CUDA prototype built in ${BUILD_DIR}"
