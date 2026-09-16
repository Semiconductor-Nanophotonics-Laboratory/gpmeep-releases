#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: environment creation requires protected Bash mode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${1:-cpu}"
RECREATE="${2:-}"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"

case "${PROFILE}" in
  cpu|cuda|cuda-mpi)
    PREFIX="${REPO_ROOT}/.envs/meep-gpu-${PROFILE}"
    ;;
  package-builder)
    PREFIX="${REPO_ROOT}/.envs/gpmeep-package-builder"
    ;;
  *)
    echo "usage: $0 [cpu|cuda|cuda-mpi|package-builder] [--recreate]" >&2
    exit 2
    ;;
esac

if [[ -n "${RECREATE}" && "${RECREATE}" != "--recreate" ]]; then
  echo "usage: $0 [cpu|cuda|cuda-mpi|package-builder] [--recreate]" >&2
  exit 2
fi

/bin/bash -p "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null

/usr/bin/mkdir -p "${MAMBA_ROOT}" "${MAMBA_CACHE}" "${MAMBA_PACKAGES}" "${REPO_ROOT}/.envs"
export XDG_CACHE_HOME="${MAMBA_CACHE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"
export PYTHONDONTWRITEBYTECODE=1

LOCK="${REPO_ROOT}/environment/locks/${PROFILE}-linux-64.lock"
YAML_SPEC="${REPO_ROOT}/environment/${PROFILE}.yml"
if [[ "$(/usr/bin/uname -s)" == "Linux" && "$(/usr/bin/uname -m)" == "x86_64" && -f "${LOCK}" ]]; then
  SPEC="${LOCK}"
  EXACT_LOCK=1
else
  SPEC="${YAML_SPEC}"
  EXACT_LOCK=0
  echo "warning: no exact lock is available for $(/usr/bin/uname -s)/$(/usr/bin/uname -m); resolving ${YAML_SPEC}" >&2
fi

if [[ "${RECREATE}" == "--recreate" && -f "${PREFIX}/conda-meta/history" ]]; then
  BACKUP_PREFIX="${PREFIX}.backup.$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)"
  /usr/bin/mv "${PREFIX}" "${BACKUP_PREFIX}"
  echo "Previous environment preserved at ${BACKUP_PREFIX}"
fi

verify_exact_environment() {
  local installed_packages
  local locked_packages
  installed_packages="$(/usr/bin/mktemp "${MAMBA_ROOT}/installed-${PROFILE}.XXXXXX")"
  locked_packages="$(/usr/bin/mktemp "${MAMBA_ROOT}/locked-${PROFILE}.XXXXXX")"

  if ! "${MAMBA}" --no-rc list \
      --root-prefix "${MAMBA_ROOT}" \
      --prefix "${PREFIX}" \
      --explicit \
      --sha256 |
    /usr/bin/sed -n '/:\/\//p' |
    /usr/bin/sort >"${installed_packages}"; then
    /usr/bin/rm -f "${installed_packages}" "${locked_packages}"
    return 2
  fi
  /usr/bin/sed -n '/:\/\//p' "${LOCK}" | /usr/bin/sort >"${locked_packages}"
  local status=0
  /usr/bin/diff -u "${locked_packages}" "${installed_packages}" || status=$?
  /usr/bin/rm -f "${installed_packages}" "${locked_packages}"
  return "${status}"
}

if [[ "${EXACT_LOCK}" == "1" && -f "${PREFIX}/conda-meta/history" ]]; then
  if verify_exact_environment; then
    echo "Exact environment verified against ${LOCK}"
    echo "Environment ready: ${PREFIX}"
    "${MAMBA}" --no-rc run --clean-env \
      --env "XDG_CACHE_HOME=${MAMBA_CACHE}" \
      --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
      --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}" \
      --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
      python -c "import sys; print(sys.version)"
    exit 0
  fi
  echo "error: existing environment differs from ${LOCK}; it was not modified" >&2
  echo "rerun with --recreate to preserve it and create an exact replacement" >&2
  exit 1
fi

if [[ -f "${PREFIX}/conda-meta/history" ]]; then
  ACTION=install
else
  ACTION=create
fi

MAMBA_ARGUMENTS=(
  "${ACTION}"
  --yes
  --no-pyc
  --safety-checks enabled
  --extra-safety-checks
  --root-prefix "${MAMBA_ROOT}"
  --prefix "${PREFIX}"
  --file "${SPEC}"
)
if [[ "${EXACT_LOCK}" == "0" ]]; then
  MAMBA_ARGUMENTS+=(--strict-channel-priority)
fi
"${MAMBA}" --no-rc "${MAMBA_ARGUMENTS[@]}"

if [[ "${EXACT_LOCK}" == "1" ]]; then
  if ! verify_exact_environment; then
    echo "error: environment differs from ${LOCK}" >&2
    echo "rerun with --recreate to preserve the old prefix and create an exact environment" >&2
    exit 1
  fi
  echo "Exact environment verified against ${LOCK}"
fi

echo "Environment ready: ${PREFIX}"
"${MAMBA}" --no-rc run --clean-env \
  --env "XDG_CACHE_HOME=${MAMBA_CACHE}" \
  --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
  --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}" \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
  python -c "import sys; print(sys.version)"
