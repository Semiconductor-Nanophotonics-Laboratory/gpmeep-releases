#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: package builder requires protected Bash mode" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$('/usr/bin/dirname' "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_PROCESS_CACHE="${MAMBA_ROOT}/package-builder-xdg-cache"
PREFIX="${REPO_ROOT}/.envs/gpmeep-package-builder"
LOCK="${REPO_ROOT}/environment/locks/package-builder-linux-64.lock"
OUTPUT_DIRECTORY="${REPO_ROOT}/dist"
CUDA_ARCHITECTURES="AUTO"
BUILD_JOBS=4
DISTRIBUTION_VERSION="1.0.3"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --output-dir requires a nonempty path" >&2
        exit 2
      fi
      OUTPUT_DIRECTORY="$2"
      shift 2
      ;;
    --cuda-architectures)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --cuda-architectures requires AUTO or a nonempty LIST" >&2
        exit 2
      fi
      CUDA_ARCHITECTURES="$2"
      shift 2
      ;;
    --jobs)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --jobs requires a positive integer" >&2
        exit 2
      fi
      BUILD_JOBS="$2"
      shift 2
      ;;
    --help|-h)
      echo "usage: $0 [--output-dir PATH] [--cuda-architectures AUTO|LIST] [--jobs N]" >&2
      exit 0
      ;;
    *)
      echo "usage: $0 [--output-dir PATH] [--cuda-architectures AUTO|LIST] [--jobs N]" >&2
      exit 2
      ;;
  esac
done
OUTPUT_DIRECTORY="$(/usr/bin/realpath --canonicalize-missing -- "${OUTPUT_DIRECTORY}")"
if [[ "${OUTPUT_DIRECTORY}" != /* ]]; then
  echo "error: canonical output directory is not absolute" >&2
  exit 2
fi
if [[ "$('/usr/bin/uname' -s)" != Linux || "$('/usr/bin/uname' -m)" != x86_64 ]]; then
  echo "error: gpmeep v1 package builds support Linux x86_64" >&2
  exit 2
fi
if [[ ! -f "${LOCK}" ]]; then
  echo "error: exact package-builder lock is absent: ${LOCK}" >&2
  exit 1
fi
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --jobs must be a positive integer" >&2
  exit 2
fi
if [[ -n "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "error: package source tree must be clean and committed" >&2
  exit 1
fi
SOURCE_COMMIT="$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: package source identity is invalid" >&2
  exit 1
fi

/bin/bash -p "${SCRIPT_DIR}/create-env.sh" package-builder
if [[ ! -x "${MAMBA}" || ! -x "${PREFIX}/bin/conda" ]]; then
  echo "error: exact package builder environment is incomplete" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIRECTORY}" ]]; then
  if [[ ! -d "${OUTPUT_DIRECTORY}" || -L "${OUTPUT_DIRECTORY}" ]]; then
    echo "error: output path is not a safe directory" >&2
    exit 1
  fi
else
  /usr/bin/install -d -m 0755 "${OUTPUT_DIRECTORY}"
fi
if /usr/bin/find "${OUTPUT_DIRECTORY}" -mindepth 1 -maxdepth 2 \
    \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
       -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
    -print -quit | /usr/bin/grep -q .; then
  echo "error: output directory already contains a gpmeep ${DISTRIBUTION_VERSION} package" >&2
  echo "hint: choose a clean build directory with --output-dir /absolute/path" >&2
  exit 1
fi

# Conda's path source copies ignored files as well as Git-tracked content.
# Create and retain a checksum-bound archive of the exact commit, extract it
# into a fresh tree, and give only that tree to the recipe.
SOURCE_ARCHIVE="${OUTPUT_DIRECTORY}/gpmeep-source-${SOURCE_COMMIT}.tar.gz"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE}.sha256"
SOURCE_TREE="${OUTPUT_DIRECTORY}/source-${SOURCE_COMMIT}"
for path in "${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE_SHA256}" "${SOURCE_TREE}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "error: source staging path already exists: ${path}" >&2
    exit 1
  fi
done
/usr/bin/install -d -m 0755 "${SOURCE_TREE}"
/usr/bin/git -C "${REPO_ROOT}" archive --format=tar.gz \
  --prefix=gpmeep-source/ --output="${SOURCE_ARCHIVE}" "${SOURCE_COMMIT}"
/usr/bin/tar -xzf "${SOURCE_ARCHIVE}" --strip-components=1 -C "${SOURCE_TREE}"
SOURCE_ARCHIVE_DIGEST="$(/usr/bin/sha256sum "${SOURCE_ARCHIVE}" | /usr/bin/awk '{print $1}')"
/usr/bin/printf '%s  %s\n' "${SOURCE_ARCHIVE_DIGEST}" \
  "$(/usr/bin/basename "${SOURCE_ARCHIVE}")" >"${SOURCE_ARCHIVE_SHA256}"
if /usr/bin/find "${SOURCE_TREE}" -mindepth 1 \
    \( -name .git -o -name .envs -o -name benchmark-results \) \
    -print -quit | /usr/bin/grep -q .; then
  echo "error: exact source snapshot contains a forbidden local-state path" >&2
  exit 1
fi

export GPMEEP_SOURCE_COMMIT="${SOURCE_COMMIT}"
export GPMEEP_CUDA_ARCHS="${CUDA_ARCHITECTURES}"
export GPMEEP_SOURCE_TREE="${SOURCE_TREE}"
export CONDA_BLD_PATH="${OUTPUT_DIRECTORY}"
export PYTHONNOUSERSITE=1
/usr/bin/install -d -m 0700 "${MAMBA_PROCESS_CACHE}"
if /usr/bin/env XDG_CACHE_HOME="${MAMBA_PROCESS_CACHE}" \
    "${MAMBA}" --no-rc run --clean-env \
    --env "GPMEEP_SOURCE_COMMIT=${GPMEEP_SOURCE_COMMIT}" \
    --env "GPMEEP_CUDA_ARCHS=${GPMEEP_CUDA_ARCHS}" \
    --env "GPMEEP_SOURCE_TREE=${GPMEEP_SOURCE_TREE}" \
    --env "CONDA_BLD_PATH=${CONDA_BLD_PATH}" \
    --env "CPU_COUNT=${BUILD_JOBS}" \
    --env "XDG_CACHE_HOME=${MAMBA_PROCESS_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
    conda build "${REPO_ROOT}/packaging/conda" \
      --override-channels --channel conda-forge --no-anaconda-upload; then
  :
else
  CONDA_BUILD_RC=$?
  echo "error: conda build failed with status ${CONDA_BUILD_RC}" >&2
  exit "${CONDA_BUILD_RC}"
fi

if [[ -d "${OUTPUT_DIRECTORY}/broken" ]] && \
   /usr/bin/find "${OUTPUT_DIRECTORY}/broken" -type f \
     \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
        -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
     -print -quit | /usr/bin/grep -q .; then
  echo "error: conda placed a gpmeep ${DISTRIBUTION_VERSION} artifact in broken/" >&2
  exit 1
fi

mapfile -t PACKAGES < <(
  /usr/bin/find "${OUTPUT_DIRECTORY}/linux-64" -maxdepth 1 -type f \
    \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
       -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
    -print
)
if [[ "${#PACKAGES[@]}" -ne 1 ]]; then
  echo "error: package build produced ${#PACKAGES[@]} gpmeep artifacts" >&2
  exit 1
fi
PACKAGE="${PACKAGES[0]}"
PACKAGE_DIGEST="$(/usr/bin/sha256sum "${PACKAGE}" | /usr/bin/awk '{print $1}')"
/usr/bin/printf '%s  %s\n' "${PACKAGE_DIGEST}" \
  "$(/usr/bin/basename "${PACKAGE}")" >"${PACKAGE}.sha256"
printf 'gpmeep-package=%s\n' "${PACKAGE}"
printf 'source-commit=%s\n' "${SOURCE_COMMIT}"
/usr/bin/sha256sum "${PACKAGE}" "${PACKAGE}.sha256" \
  "${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE_SHA256}"
