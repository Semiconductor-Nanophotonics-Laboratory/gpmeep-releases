#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda-mpi"

if [[ "${MEEP_GPU_ENV_ACTIVE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
  if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
    echo "error: CUDA MPI environment is missing; run scripts/create-env.sh cuda-mpi" >&2
    exit 1
  fi
  mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}"
  if [[ -z "${HOME:-}" || ! -d "${HOME}" ]]; then
    echo "error: Open MPI requires the launch user's real home directory" >&2
    exit 1
  fi
  export XDG_CACHE_HOME="${MAMBA_CACHE}"
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
  export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"
  RUN_ENVIRONMENT=(
    --clean-env
    --env MEEP_GPU_ENV_ACTIVE=1
    --env "HOME=${HOME}"
    --env "XDG_CACHE_HOME=${MAMBA_CACHE}"
    --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
    --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}"
    --env "CCACHE_DIR=${MAMBA_CACHE}/ccache"
  )
  for variable in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH MEEP_GPU_CUDA_ARCHS \
    MEEP_GPU_DEVICE MEEP_GPU_ALLOW_OVERSUBSCRIBE MEEP_GPU_MPI_TRANSPORT \
    CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER OMPI_MCA_opal_cuda_support \
    UCX_MEMTYPE_CACHE; do
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
for tool in autoreconf c++ make nvcc mpicxx mpiexec h5pcc; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the isolated environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done
HDF5_CONFIGURATION="$("${PREFIX}/bin/h5pcc" -showconfig)"
if [[ "${HDF5_CONFIGURATION}" != *"Parallel HDF5: yes"* ]]; then
  echo "error: the CUDA/MPI build requires a parallel HDF5 package" >&2
  echo "error: recreate the cuda-mpi environment from its exact lock" >&2
  exit 1
fi
for compiler_variable in CC CXX FC F77; do
  COMPILER_PATH="${!compiler_variable:-}"
  if [[ ! -x "${COMPILER_PATH}" || "${COMPILER_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${compiler_variable} resolved outside the isolated environment: ${COMPILER_PATH:-<unset>}" >&2
    exit 1
  fi
done

BUILD_DIR="${REPO_ROOT}/build/meep-cuda-mpi-fp32"
INSTALL_DIR="${REPO_ROOT}/install/meep-cuda-mpi-fp32"
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
  --with-mpi
  --without-python
  --without-scheme
  "--prefix=${INSTALL_DIR}"
  "${FAST_MATH_CONFIG[@]}"
)

RECEIPT_BEGIN=(
  "${PREFIX}/bin/python" "${SCRIPT_DIR}/write-build-receipt.py" begin
  --repo "${REPO_ROOT}"
  --build-dir "${BUILD_DIR}"
  --build-kind cuda-mpi-fp32
  --builder "${BASH_SOURCE[0]}"
  --qualification-contract gpmeep-cuda-mpi-fp32-v1
  --lockfile "environment_lock=${REPO_ROOT}/environment/locks/cuda-mpi-linux-64.lock"
)
for configure_argument in "${CONFIGURE_ARGS[@]}"; do
  RECEIPT_BEGIN+=("--configure-arg=${configure_argument}")
done
for receipt_environment in MEEP_GPU_FAST_MATH MEEP_GPU_CUDA_ARCHS \
  MEEP_GPU_MPI_TRANSPORT CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CC CXX FC F77 \
  NVCC CUDAHOSTCXX MPICXX OMPI_MCA_opal_cuda_support UCX_MEMTYPE_CACHE; do
  RECEIPT_BEGIN+=(--record-env "${receipt_environment}")
done
"${RECEIPT_BEGIN[@]}"

cd "${REPO_ROOT}"
autoreconf --verbose --install --symlink --force
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
cd "${BUILD_DIR}"

NVCC="${PREFIX}/bin/nvcc" CUDAHOSTCXX="${CXX}" MPICXX="${PREFIX}/bin/mpicxx" \
  "../../configure" "${CONFIGURE_ARGS[@]}"

make -j"${MAKE_JOBS}"
# Each test already launches two MPI ranks with two OpenMP threads. Running
# many such tests concurrently oversubscribes a development node and can make
# the suite several times slower, so keep the MPI test harness serial.
# Keep the generic upstream regressions on their CPU reference path. Dedicated
# gpu-backend/gpu-step-db tests select strict CUDA explicitly and validate both
# numerical agreement and distributed device coverage. Letting every tiny
# upstream case choose automatic CUDA turns kernel-launch latency into minutes
# of noise and is not a meaningful acceleration test.
MEEP_GPU_BACKEND=cpu make check
make install

"${PREFIX}/bin/python" "${SCRIPT_DIR}/write-build-receipt.py" finalize \
  --repo "${REPO_ROOT}" \
  --build-dir "${BUILD_DIR}" \
  --configuration-file "config_h=${BUILD_DIR}/config.h" \
  --configuration-file "config_status=${BUILD_DIR}/config.status" \
  --artifact "libmeep=${BUILD_DIR}/src/.libs/libmeep.so.38.0.0" \
  --artifact "installed_libmeep=${INSTALL_DIR}/lib/libmeep.so.38.0.0" \
  --artifact "gpu_backend_test=${BUILD_DIR}/tests/.libs/gpu-backend" \
  --artifact "gpu_step_db_test=${BUILD_DIR}/tests/.libs/gpu-step-db" \
  --artifact "gpu_mpi_performance=${BUILD_DIR}/tests/.libs/gpu-mpi-performance" \
  --artifact "gpu_performance=${BUILD_DIR}/tests/.libs/gpu-performance" \
  --tool autoreconf --tool c++ --tool make --tool nvcc --tool mpicxx \
  --tool mpiexec --tool h5pcc

echo "CUDA/MPI-enabled Meep FP32 installed in ${INSTALL_DIR}"
