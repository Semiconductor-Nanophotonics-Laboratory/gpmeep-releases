#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
MPL_CONFIG="${MAMBA_CACHE}/matplotlib"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda-mpi"
RUNNER="${SCRIPT_DIR}/python-validation/run_validation.py"
MANIFEST="${SCRIPT_DIR}/python-validation/manifest.json"
BUILD_PYTHON="${REPO_ROOT}/build/meep-cuda-mpi-python-fp32/python"
INSTALL_PREFIX="${REPO_ROOT}/install/meep-cuda-mpi-python-fp32"
QUALIFICATION_FONTCONFIG="${REPO_ROOT}/build/meep-cuda-mpi-python-fp32/qualification-fontconfig.conf"
QUALIFICATION_XDG_CONFIG="${REPO_ROOT}/build/meep-cuda-mpi-python-fp32/qualification-home/.config"

for argument in "$@"; do
  case "${argument}" in
    --repo|--repo=*|--manifest|--manifest=*|--build-python|--build-python=*|\
    --install-prefix|--install-prefix=*|--python|--python=*)
      echo "error: ${argument%%=*} is fixed by the authoritative validation wrapper" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
  echo "error: create the isolated CUDA environment first" >&2
  echo "run: scripts/create-env.sh cuda-mpi" >&2
  exit 2
fi
if [[ ! -d "${BUILD_PYTHON}" ]]; then
  echo "error: build the Python package first" >&2
  echo "run: scripts/build-meep-cuda-mpi-python.sh" >&2
  exit 2
fi
if [[ ! -f "${QUALIFICATION_FONTCONFIG}" ]]; then
  echo "error: authoritative qualification Fontconfig is missing" >&2
  echo "run: scripts/build-meep-cuda-mpi-python.sh" >&2
  exit 2
fi
USER_HOME="$(/usr/bin/getent passwd "$(/usr/bin/id -u)" | /usr/bin/cut -d: -f6)"
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
  echo "error: Open MPI requires the launch user's account home directory" >&2
  exit 2
fi

mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}" "${MPL_CONFIG}"
export XDG_CACHE_HOME="${MAMBA_CACHE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"

RUN_ENVIRONMENT=(
  --clean-env
  --env "HOME=${USER_HOME}"
  --env "XDG_CACHE_HOME=${MAMBA_CACHE}"
  --env "XDG_CONFIG_HOME=${QUALIFICATION_XDG_CONFIG}"
  --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
  --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}"
  --env "MPLCONFIGDIR=${MPL_CONFIG}"
  --env MPLBACKEND=Agg
  --env PYTHONNOUSERSITE=1
  --env PYTHONDONTWRITEBYTECODE=1
  --env "FONTCONFIG_FILE=${QUALIFICATION_FONTCONFIG}"
  --env CUDA_CACHE_DISABLE=1
  --env JAX_PLATFORMS=cpu
)
for variable in CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER MEEP_GPU_DEVICE; do
  if [[ -v "${variable}" ]]; then
    RUN_ENVIRONMENT+=(--env "${variable}=${!variable}")
  fi
done

exec "${MAMBA}" --no-rc run "${RUN_ENVIRONMENT[@]}" \
  --root-prefix "${MAMBA_ROOT}" \
  --prefix "${PREFIX}" \
  "${PREFIX}/bin/python" "${RUNNER}" \
  "$@" \
  --repo "${REPO_ROOT}" \
  --manifest "${MANIFEST}" \
  --build-python "${BUILD_PYTHON}" \
  --install-prefix "${INSTALL_PREFIX}" \
  --python "${PREFIX}/bin/python"
