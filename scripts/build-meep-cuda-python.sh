#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
MPL_CONFIG="${MAMBA_CACHE}/matplotlib"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda"

if [[ "${MEEP_GPU_ENV_ACTIVE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
  if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
    echo "error: CUDA environment is missing; run scripts/create-env.sh cuda" >&2
    exit 1
  fi
  mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}" "${MPL_CONFIG}"
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
    --env "MPLCONFIGDIR=${MPL_CONFIG}"
    --env MPLBACKEND=Agg
    --env PYTHONNOUSERSITE=1
  )
  for variable in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH \
    MEEP_GPU_CUDA_ARCHS MEEP_GPU_DEVICE CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER; do
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
for tool in autoreconf c++ make nvcc python swig; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the isolated environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done
for compiler_variable in CC CXX FC F77; do
  COMPILER_PATH="${!compiler_variable:-}"
  if [[ ! -x "${COMPILER_PATH}" ||
        "${COMPILER_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${compiler_variable} resolved outside the isolated environment: ${COMPILER_PATH:-<unset>}" >&2
    exit 1
  fi
done

BUILD_DIR="${REPO_ROOT}/build/meep-cuda-python-fp32"
INSTALL_DIR="${REPO_ROOT}/install/meep-cuda-python-fp32"
MAKE_JOBS="${MEEP_GPU_MAKE_JOBS:-$(nproc)}"
CUDA_ARCHITECTURES="${MEEP_GPU_CUDA_ARCHS:-AUTO}"
FAST_MATH="${MEEP_GPU_FAST_MATH:-OFF}"
FAST_MATH_CONFIG=()
case "${FAST_MATH}" in
  ON|on|1|yes|YES|true|TRUE)
    FAST_MATH_CONFIG=(--enable-cuda-fast-math)
    ;;
  OFF|off|0|no|NO|false|FALSE)
    ;;
  *)
    echo "error: MEEP_GPU_FAST_MATH must be ON or OFF" >&2
    exit 2
    ;;
esac

CONFIGURE_ARGS=(
  --enable-maintainer-mode
  --enable-ccache
  --enable-shared
  --enable-single
  --enable-cuda
  "--with-cuda-arch=${CUDA_ARCHITECTURES}"
  --with-openmp
  --without-mpi
  --with-python
  --without-scheme
  "--prefix=${INSTALL_DIR}"
  "${FAST_MATH_CONFIG[@]}"
)

RECEIPT_BEGIN=(
  "${PREFIX}/bin/python" "${SCRIPT_DIR}/write-build-receipt.py" begin
  --repo "${REPO_ROOT}"
  --build-dir "${BUILD_DIR}"
  --build-kind cuda-python-fp32
  --builder "${BASH_SOURCE[0]}"
  --qualification-contract gpmeep-cuda-python-fp32-v1
  --lockfile "environment_lock=${REPO_ROOT}/environment/locks/cuda-linux-64.lock"
)
for configure_argument in "${CONFIGURE_ARGS[@]}"; do
  RECEIPT_BEGIN+=("--configure-arg=${configure_argument}")
done
for receipt_environment in MEEP_GPU_FAST_MATH MEEP_GPU_CUDA_ARCHS \
  CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CC CXX FC F77 NVCC CUDAHOSTCXX; do
  RECEIPT_BEGIN+=(--record-env "${receipt_environment}")
done
"${RECEIPT_BEGIN[@]}"

cd "${REPO_ROOT}"
autoreconf --verbose --install --symlink --force
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
cd "${BUILD_DIR}"

NVCC="${PREFIX}/bin/nvcc" CUDAHOSTCXX="${CXX}" "../../configure" \
  "${CONFIGURE_ARGS[@]}"

make -j"${MAKE_JOBS}"
env PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.gpu.compiled; print(mp.__version__)'
make install

"${PREFIX}/bin/python" "${SCRIPT_DIR}/write-build-receipt.py" finalize \
  --repo "${REPO_ROOT}" \
  --build-dir "${BUILD_DIR}" \
  --configuration-file "config_h=${BUILD_DIR}/config.h" \
  --configuration-file "config_status=${BUILD_DIR}/config.status" \
  --artifact "python_extension=${BUILD_DIR}/python/meep/_meep.so" \
  --artifact "libmeep=${BUILD_DIR}/src/.libs/libmeep.so.38.0.0" \
  --artifact "installed_python_extension=${INSTALL_DIR}/lib/python3.11/site-packages/meep/_meep.so.38.0.0" \
  --artifact "installed_libmeep=${INSTALL_DIR}/lib/libmeep.so.38.0.0" \
  --manifest "in_place_python=${BUILD_DIR}/python/meep" \
  --manifest "installed_python=${INSTALL_DIR}/lib/python3.11/site-packages/meep" \
  --tool autoreconf --tool c++ --tool make --tool nvcc --tool python --tool swig

echo "CUDA-enabled FP32 Meep Python package installed in ${INSTALL_DIR}"
echo "In-place package for tests: PYTHONPATH=${BUILD_DIR}/python"
