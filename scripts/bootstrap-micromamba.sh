#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: Micromamba bootstrap requires protected Bash mode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/.tools"
MAMBA="${TOOLS_DIR}/micromamba"
MAMBA_VERSION="2.8.1"
MAMBA_RELEASE_BUILD="0"
# Pin the immutable build-qualified upstream release asset.  The rolling
# micro.mamba.pm version endpoint may advance to a newer conda build while
# retaining the same upstream version string.
DOWNLOAD_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${MAMBA_VERSION}-${MAMBA_RELEASE_BUILD}/micromamba-linux-64.tar.bz2"
ARCHIVE_SHA256="a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515"
BINARY_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"

if [[ "$(/usr/bin/uname -s)" != "Linux" || "$(/usr/bin/uname -m)" != "x86_64" ]]; then
  echo "error: this bootstrap script currently supports Linux x86_64 only" >&2
  exit 2
fi

for tool in curl tar install sha256sum awk mktemp mkdir rm rmdir; do
  if [[ ! -x "/usr/bin/${tool}" ]]; then
    echo "error: required protected bootstrap tool is missing: /usr/bin/${tool}" >&2
    exit 1
  fi
done

if [[ -x "${MAMBA}" ]]; then
  INSTALLED_SHA256="$(/usr/bin/sha256sum "${MAMBA}" | /usr/bin/awk '{print $1}')"
  if [[ "${INSTALLED_SHA256}" != "${BINARY_SHA256}" ]]; then
    echo "error: existing Micromamba checksum does not match pinned ${MAMBA_VERSION}" >&2
    exit 1
  fi
  "${MAMBA}" --version
  exit 0
fi

/usr/bin/mkdir -p "${TOOLS_DIR}"
TEMP_DIR="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/meep-gpu-micromamba.XXXXXX")"
ARCHIVE="${TEMP_DIR}/micromamba.tar.bz2"

cleanup() {
  /usr/bin/rm -f "${ARCHIVE}" "${TEMP_DIR}/bin/micromamba"
  /usr/bin/rmdir "${TEMP_DIR}/bin" "${TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Downloading portable Micromamba ${MAMBA_VERSION} into ${TOOLS_DIR}"
/usr/bin/curl --disable --fail --location --silent --show-error "${DOWNLOAD_URL}" --output "${ARCHIVE}"

ACTUAL_ARCHIVE_SHA256="$(/usr/bin/sha256sum "${ARCHIVE}" | /usr/bin/awk '{print $1}')"
if [[ "${ACTUAL_ARCHIVE_SHA256}" != "${ARCHIVE_SHA256}" ]]; then
  echo "error: Micromamba archive checksum verification failed" >&2
  exit 1
fi

/usr/bin/tar -xjf "${ARCHIVE}" -C "${TEMP_DIR}" bin/micromamba

ACTUAL_BINARY_SHA256="$(/usr/bin/sha256sum "${TEMP_DIR}/bin/micromamba" | /usr/bin/awk '{print $1}')"
if [[ "${ACTUAL_BINARY_SHA256}" != "${BINARY_SHA256}" ]]; then
  echo "error: extracted Micromamba binary checksum verification failed" >&2
  exit 1
fi

/usr/bin/install -m 755 "${TEMP_DIR}/bin/micromamba" "${MAMBA}"

"${MAMBA}" --version
echo "${BINARY_SHA256}  ${MAMBA}"
