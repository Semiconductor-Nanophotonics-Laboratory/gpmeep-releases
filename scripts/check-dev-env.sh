#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${1:-cpu}"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-${PROFILE}"

if [[ ! "${PROFILE}" =~ ^(cpu|cuda|cuda-mpi)$ ]]; then
  echo "usage: $0 [cpu|cuda|cuda-mpi]" >&2
  exit 2
fi

"${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
  echo "error: ${PROFILE} environment is missing; run scripts/create-env.sh ${PROFILE}" >&2
  exit 1
fi

mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}"
export XDG_CACHE_HOME="${MAMBA_CACHE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"

"${MAMBA}" --no-rc run --clean-env \
  --env "XDG_CACHE_HOME=${MAMBA_CACHE}" \
  --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
  --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}" \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" bash -c '
set -euo pipefail
for tool in cc c++ gfortran autoreconf automake libtool make pkg-config swig python; do
  printf "%-12s %s\n" "${tool}" "$(command -v "${tool}")"
done
if [[ "${CONDA_PREFIX}" == *cuda-mpi ]]; then
  for tool in mpicc mpicxx mpiexec h5pcc; do
    printf "%-12s %s\n" "${tool}" "$(command -v "${tool}")"
  done
  hdf5_configuration="$(h5pcc -showconfig)"
  if [[ "${hdf5_configuration}" != *"Parallel HDF5: yes"* ]]; then
    echo "error: cuda-mpi profile resolved a non-parallel HDF5 build" >&2
    exit 1
  fi
  hdf5_wrapper=h5pcc
  python -c "import mpi4py; print(\"mpi4py      \" + mpi4py.__version__)"
else
  hdf5_wrapper=h5cc
fi
for package in harminv; do
  printf "%-12s %s\n" "${package}" "$(pkg-config --modversion "${package}")"
done
printf "%-12s %s\n" "hdf5" "$("${hdf5_wrapper}" -showconfig | sed -n "s/^[[:space:]]*HDF5 Version: //p")"
for header in hdf5.h ctl.h ctlgeom.h mpb.h; do
  test -f "${CONDA_PREFIX}/include/${header}"
  printf "%-12s %s\n" "${header}" "${CONDA_PREFIX}/include/${header}"
done
python -c "import numpy; print(\"numpy       \" + numpy.__version__)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc         not installed in this profile"
fi
'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "Host NVIDIA driver:"
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
else
  echo "Host NVIDIA driver: nvidia-smi not found"
fi
