#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: gpmeep installer requires protected Bash mode" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$('/usr/bin/dirname' "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
LOCK="${REPO_ROOT}/environment/locks/gpmeep-runtime-linux-64.lock"
PREFIX=""
PACKAGE=""
EXPECTED_PACKAGE_SHA256=""
EXPECTED_SOURCE_COMMIT=""
MODE="gpu1"
GPU_DEVICES=""
REPORT=""
GPMEEP_GLIBC_MINIMUM="2.17"

usage() {
  echo "usage: $0 --prefix ABSOLUTE_PATH --package FILE [--package-sha256 HEX] [--expected-source-commit HEX] [--mode cpu|gpu1|gpu2] [--gpu-devices LIST] [--report FILE]" >&2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --package)
      PACKAGE="$2"
      shift 2
      ;;
    --package-sha256)
      EXPECTED_PACKAGE_SHA256="$2"
      shift 2
      ;;
    --expected-source-commit)
      EXPECTED_SOURCE_COMMIT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpu-devices)
      GPU_DEVICES="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
if [[ "$('/usr/bin/uname' -s)" != Linux || "$('/usr/bin/uname' -m)" != x86_64 ]]; then
  echo "error: gpmeep v1 binary installation supports Linux x86_64" >&2
  exit 2
fi
GLIBC_VERSION="$('/usr/bin/getconf' GNU_LIBC_VERSION 2>/dev/null | \
  /usr/bin/awk '$1 == "glibc" {print $2}')"
if [[ ! "${GLIBC_VERSION}" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
  echo "error: could not determine the host glibc version" >&2
  exit 2
fi
if [[ "$(/usr/bin/printf '%s\n%s\n' "${GPMEEP_GLIBC_MINIMUM}" "${GLIBC_VERSION}" | \
  /usr/bin/sort -V | /usr/bin/head -n 1)" != "${GPMEEP_GLIBC_MINIMUM}" ]]; then
  echo "error: gpmeep v1.0.3 requires glibc ${GPMEEP_GLIBC_MINIMUM} or newer; host has ${GLIBC_VERSION}" >&2
  exit 2
fi
if [[ -z "${PREFIX}" || -z "${PACKAGE}" ]]; then
  usage
  exit 2
fi
if [[ "${PREFIX}" != /* || "${PREFIX}" == / || "${PREFIX}" == /home || "${PREFIX}" == /scratch ]]; then
  echo "error: --prefix must be a non-broad absolute path" >&2
  exit 2
fi
if [[ -e "${PREFIX}" || -L "${PREFIX}" ]]; then
  echo "error: installation prefix must be absent: ${PREFIX}" >&2
  exit 1
fi
if [[ ! -f "${PACKAGE}" || -L "${PACKAGE}" ]]; then
  echo "error: package must be a regular non-symlink file" >&2
  exit 1
fi
PACKAGE="$(cd "$('/usr/bin/dirname' "${PACKAGE}")" && pwd)/$(/usr/bin/basename "${PACKAGE}")"
if [[ ! "${PACKAGE}" =~ \.(conda|tar\.bz2)$ ]]; then
  echo "error: --package is not a conda package" >&2
  exit 2
fi
if [[ ! -f "${LOCK}" ]]; then
  echo "error: exact gpmeep runtime lock is absent" >&2
  exit 1
fi
if [[ "${MODE}" != cpu && "${MODE}" != gpu1 && "${MODE}" != gpu2 ]]; then
  echo "error: --mode must be cpu, gpu1, or gpu2" >&2
  exit 2
fi
if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]] && \
    /usr/bin/git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  EXPECTED_SOURCE_COMMIT="$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)"
fi
if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: a full lowercase --expected-source-commit is required outside a Git clone" >&2
  exit 2
fi
if [[ -z "${EXPECTED_PACKAGE_SHA256}" && -f "${PACKAGE}.sha256" ]]; then
  EXPECTED_PACKAGE_SHA256="$(/usr/bin/awk 'NR==1 {print $1}' "${PACKAGE}.sha256")"
fi
if [[ ! "${EXPECTED_PACKAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "error: a 64-hex package checksum or valid .sha256 sidecar is required" >&2
  exit 2
fi
ACTUAL_PACKAGE_SHA256="$(/usr/bin/sha256sum "${PACKAGE}" | /usr/bin/awk '{print $1}')"
if [[ "${ACTUAL_PACKAGE_SHA256}" != "${EXPECTED_PACKAGE_SHA256}" ]]; then
  echo "error: gpmeep package checksum mismatch" >&2
  exit 1
fi
if [[ -z "${REPORT}" ]]; then
  REPORT="${PREFIX}.self-check.json"
fi
if [[ "${REPORT}" != /* ]]; then
  echo "error: --report must be an absolute path" >&2
  exit 2
fi
if [[ -e "${REPORT}" || -L "${REPORT}" ]]; then
  echo "error: self-check report path must be absent" >&2
  exit 1
fi

/bin/bash -p "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
MAMBA_ROOT="${PREFIX}.micromamba-root"
if [[ -e "${MAMBA_ROOT}" || -L "${MAMBA_ROOT}" ]]; then
  echo "error: dedicated Micromamba root must be absent: ${MAMBA_ROOT}" >&2
  exit 1
fi
/usr/bin/install -d -m 0700 "${MAMBA_ROOT}"
RUNTIME_HOME="${MAMBA_ROOT}/home"
/usr/bin/install -d -m 0700 "${RUNTIME_HOME}"
"${MAMBA}" --no-rc create --yes --no-pyc \
  --safety-checks enabled --extra-safety-checks \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" --file "${LOCK}"

INSTALLED_LOCK="$(/usr/bin/mktemp "${MAMBA_ROOT}/installed-lock.XXXXXX")"
EXPECTED_LOCK="$(/usr/bin/mktemp "${MAMBA_ROOT}/expected-lock.XXXXXX")"
cleanup_lock_files() {
  /usr/bin/rm -f "${INSTALLED_LOCK}" "${EXPECTED_LOCK}"
}
trap cleanup_lock_files EXIT
"${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
  --prefix "${PREFIX}" --explicit --sha256 | \
  /usr/bin/sed -n '/:\/\//p' | /usr/bin/sort >"${INSTALLED_LOCK}"
/usr/bin/sed -n '/:\/\//p' "${LOCK}" | /usr/bin/sort >"${EXPECTED_LOCK}"
if ! /usr/bin/cmp -s "${INSTALLED_LOCK}" "${EXPECTED_LOCK}"; then
  echo "error: materialized runtime differs from the exact lock" >&2
  exit 1
fi
cleanup_lock_files
trap - EXIT

"${MAMBA}" --no-rc install --yes --no-deps \
  --safety-checks warn \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" "${PACKAGE}"
"${MAMBA}" --no-rc run --clean-env \
  --env "HOME=${RUNTIME_HOME}" \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
  gpmeep-verify-provenance --prefix "${PREFIX}" \
    --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
    --expected-package-sha256 "${ACTUAL_PACKAGE_SHA256}"
SELF_CHECK_ARGUMENTS=(--mode "${MODE}" --report "${REPORT}")
if [[ -n "${GPU_DEVICES}" ]]; then
  SELF_CHECK_ARGUMENTS+=(--gpu-devices "${GPU_DEVICES}")
fi
"${MAMBA}" --no-rc run --clean-env \
  --env "HOME=${RUNTIME_HOME}" \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
  gpmeep-self-check "${SELF_CHECK_ARGUMENTS[@]}"
printf 'gpmeep installation PASS\n'
printf 'prefix=%s\n' "${PREFIX}"
printf 'package-sha256=%s\n' "${ACTUAL_PACKAGE_SHA256}"
printf 'source-commit=%s\n' "${EXPECTED_SOURCE_COMMIT}"
printf 'host-glibc=%s\n' "${GLIBC_VERSION}"
printf 'self-check-report=%s\n' "${REPORT}"
