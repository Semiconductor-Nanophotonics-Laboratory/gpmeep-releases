#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRECISION="${1:-fp64}"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cpu"

case "${PRECISION}" in
  fp64)
    PRECISION_CONFIG=()
    ;;
  fp32)
    PRECISION_CONFIG=(--enable-single)
    ;;
  *)
    echo "usage: $0 [fp64|fp32]" >&2
    exit 2
    ;;
esac

if [[ "${MEEP_GPU_ENV_ACTIVE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
  if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
    echo "error: CPU environment is missing; run scripts/create-env.sh cpu" >&2
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
  if [[ -n "${MEEP_GPU_MAKE_JOBS:-}" ]]; then
    RUN_ENVIRONMENT+=(--env "MEEP_GPU_MAKE_JOBS=${MEEP_GPU_MAKE_JOBS}")
  fi
  exec "${MAMBA}" --no-rc run "${RUN_ENVIRONMENT[@]}" \
    --root-prefix "${MAMBA_ROOT}" \
    --prefix "${PREFIX}" \
    "${BASH_SOURCE[0]}" "${PRECISION}"
fi

if [[ "${CONDA_PREFIX:-}" != "${PREFIX}" ]]; then
  echo "error: expected isolated environment ${PREFIX}, got ${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi
for tool in autoreconf c++ make; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the isolated environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done
for compiler_variable in CC CXX FC F77; do
  COMPILER_PATH="${!compiler_variable:-}"
  if [[ ! -x "${COMPILER_PATH}" || "${COMPILER_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${compiler_variable} resolved outside the isolated environment: ${COMPILER_PATH:-<unset>}" >&2
    exit 1
  fi
done

BUILD_DIR="${REPO_ROOT}/build/cpu-${PRECISION}"
INSTALL_DIR="${REPO_ROOT}/install/cpu-${PRECISION}"
MAKE_JOBS="${MEEP_GPU_MAKE_JOBS:-$(nproc)}"

cd "${REPO_ROOT}"
autoreconf --verbose --install --symlink --force
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
cd "${BUILD_DIR}"

"../../configure" \
  --enable-maintainer-mode \
  --enable-ccache \
  --enable-shared \
  --with-openmp \
  --without-mpi \
  --without-python \
  --without-scheme \
  --prefix="${INSTALL_DIR}" \
  "${PRECISION_CONFIG[@]}"

make -j"${MAKE_JOBS}"
make -j"${MAKE_JOBS}" check
make install

echo "CPU ${PRECISION} baseline installed in ${INSTALL_DIR}"
