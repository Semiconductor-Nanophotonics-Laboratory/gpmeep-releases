#!/bin/bash
set -euo pipefail

if [[ ! "${GPMEEP_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: GPMEEP_SOURCE_COMMIT must be a full Git commit" >&2
  exit 1
fi
CUDA_ARCHITECTURES="${GPMEEP_CUDA_ARCHS:-AUTO}"
GPMEEP_GLIBC_MINIMUM="2.17"
if [[ ! "${CUDA_ARCHITECTURES}" =~ ^(AUTO|[0-9]+([-](real|virtual))?([,;][0-9]+([-](real|virtual))?)*)$ ]]; then
  echo "error: invalid GPMEEP_CUDA_ARCHS=${CUDA_ARCHITECTURES}" >&2
  exit 1
fi
if [[ -d "${SRC_DIR}/.git" ]]; then
  ACTUAL_COMMIT="$(git -C "${SRC_DIR}" rev-parse HEAD)"
  if [[ "${ACTUAL_COMMIT}" != "${GPMEEP_SOURCE_COMMIT}" ]]; then
    echo "error: package source commit differs from requested identity" >&2
    exit 1
  fi
fi

NVCC="$(command -v nvcc)"
if [[ ! -x "${NVCC}" ]]; then
  echo "error: the conda build prefix has no nvcc" >&2
  exit 1
fi
REAL_ARCHITECTURES="$(
  "${SRC_DIR}/scripts/cuda-architectures.sh" \
    "${NVCC}" "${CUDA_ARCHITECTURES}" real-values
)"
VIRTUAL_ARCHITECTURES="$(
  "${SRC_DIR}/scripts/cuda-architectures.sh" \
    "${NVCC}" "${CUDA_ARCHITECTURES}" virtual-values
)"
# The architecture helper emits a leading zero as a C++ array sentinel.  It is
# implementation metadata, not a compiled architecture, so never publish it
# in the release manifest.
manifest_architectures() {
  local values="$1"
  if [[ "${values}" == "0" ]]; then
    printf ''
  else
    printf '%s' "${values#0, }"
  fi
}
REAL_MANIFEST_ARCHITECTURES="$(manifest_architectures "${REAL_ARCHITECTURES}")"
VIRTUAL_MANIFEST_ARCHITECTURES="$(manifest_architectures "${VIRTUAL_ARCHITECTURES}")"
REAL_MANIFEST_COMPACT="${REAL_MANIFEST_ARCHITECTURES// /}"
VIRTUAL_MANIFEST_COMPACT="${VIRTUAL_MANIFEST_ARCHITECTURES// /}"
if [[ -z "${REAL_MANIFEST_COMPACT}" ]]; then
  echo "error: CUDA architecture LIST must include at least one native -real target" >&2
  exit 1
fi
if [[ ",${REAL_MANIFEST_COMPACT}," == *",86,"* ]]; then
  RUNTIME_VALIDATED_ARCHITECTURES='"sm86"'
else
  RUNTIME_VALIDATED_ARCHITECTURES=''
fi

export MPICXX="${PREFIX}/bin/mpicxx"
export NVCC
export CUDAHOSTCXX="${CXX}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# Keep compiler diagnostics and __FILE__ strings independent of the Conda
# build directory. This is not required for loading, but avoids publishing
# internal build paths in libmeep diagnostics.
export CFLAGS="${CFLAGS:-} -ffile-prefix-map=${SRC_DIR}=gpmeep-source"
export CXXFLAGS="${CXXFLAGS:-} -ffile-prefix-map=${SRC_DIR}=gpmeep-source"

cd "${SRC_DIR}"
autoreconf --verbose --install --symlink --force
BUILD_DIRECTORY="${SRC_DIR}/_gpmeep_conda_build"
if [[ -e "${BUILD_DIRECTORY}" ]]; then
  echo "error: conda package build directory is not fresh" >&2
  exit 1
fi
mkdir -p "${BUILD_DIRECTORY}"
cd "${BUILD_DIRECTORY}"
"${SRC_DIR}/configure" \
  --enable-maintainer-mode \
  --enable-shared \
  --enable-single \
  --enable-cuda \
  "--with-cuda-arch=${CUDA_ARCHITECTURES}" \
  --with-openmp \
  --with-mpi \
  --with-python \
  --without-scheme \
  --disable-cuda-fast-math \
  "--prefix=${PREFIX}"
make -j"${CPU_COUNT:-2}"
make install

# Static and libtool archives are not part of the supported v1 runtime and
# can retain build-prefix paths. ELF RPATHs are finalized below before audits.
find "${PREFIX}/lib" -type f \( -name '*.a' -o -name '*.la' \) -delete
# make/install can retain build-qualified Python filenames. Replace them with
# deterministic checked-hash bytecode whose code-object filenames are relative
# to the eventual install prefix.  Keeping these files package-owned prevents
# normal imports from creating unowned caches that survive package removal.
MEEP_SITE_PACKAGES="${PREFIX}/lib/python3.11/site-packages/meep"
find "${MEEP_SITE_PACKAGES}" -type f -name '*.pyc' -delete
"${PREFIX}/bin/python3.11" -I -m compileall \
  --invalidation-mode checked-hash \
  -q -f -s "${PREFIX}" -p '' \
  "${MEEP_SITE_PACKAGES}"
SOURCE_COUNT="$(find "${MEEP_SITE_PACKAGES}" -type f -name '*.py' | wc -l)"
BYTECODE_COUNT="$(find "${MEEP_SITE_PACKAGES}" -type f -name '*.pyc' | wc -l)"
if [[ "${SOURCE_COUNT}" -eq 0 || "${BYTECODE_COUNT}" -ne "${SOURCE_COUNT}" ]]; then
  echo "error: package-owned Python bytecode inventory is incomplete" >&2
  exit 1
fi

# Upstream dependency link flags can contain feedstock and local build paths.
# The supported shared-library interface needs only libmeep itself.
install -m 0644 "${RECIPE_DIR}/meep.pc" \
  "${PREFIX}/lib/pkgconfig/meep.pc"

install -d "${PREFIX}/bin" "${PREFIX}/share/gpmeep/examples" \
  "${PREFIX}/share/gpmeep/docs" "${PREFIX}/share/gpmeep/libexec"
install -m 0644 "${SRC_DIR}/scripts/gpmeep-self-check.py" \
  "${PREFIX}/share/gpmeep/libexec/gpmeep-self-check.py"
install -m 0644 "${SRC_DIR}/scripts/verify-gpmeep-package.py" \
  "${PREFIX}/share/gpmeep/libexec/gpmeep-verify-provenance.py"
install -m 0644 "${SRC_DIR}/scripts/gpmeep_release_manifest.py" \
  "${PREFIX}/share/gpmeep/libexec/gpmeep_release_manifest.py"
install -m 0644 "${SRC_DIR}/scripts/check-cuda-architecture-inventory.py" \
  "${PREFIX}/share/gpmeep/libexec/check-cuda-architecture-inventory.py"
install -m 0755 "${SRC_DIR}/scripts/gpmeep-self-check-launcher.sh" \
  "${PREFIX}/bin/gpmeep-self-check"
install -m 0755 "${SRC_DIR}/scripts/gpmeep-verify-provenance-launcher.sh" \
  "${PREFIX}/bin/gpmeep-verify-provenance"
install -m 0644 "${SRC_DIR}/LICENSE" "${PREFIX}/share/gpmeep/LICENSE"
install -m 0644 \
  "${SRC_DIR}/python/examples/gpmeep/installed_cpu.py" \
  "${SRC_DIR}/python/examples/gpmeep/installed_gpu1.py" \
  "${SRC_DIR}/python/examples/gpmeep/installed_gpu2.py" \
  "${PREFIX}/share/gpmeep/examples/"
install -m 0644 \
  "${SRC_DIR}/doc/docs/GPMEEP_INSTALLATION.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_FEATURE_MATRIX.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_PERFORMANCE.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_GPU_CROSSOVER.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_V1_0_1_RELEASE_NOTES.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_V1_0_2_RELEASE_NOTES.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_V1_0_3_RELEASE_NOTES.md" \
  "${SRC_DIR}/doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md" \
  "${PREFIX}/share/gpmeep/docs/"

GLIBC_AUDIT="${PREFIX}/share/gpmeep/glibc-compatibility.json"
mapfile -t GPMEEP_ELF_FILES < <(
  {
    find "${PREFIX}/lib" -maxdepth 1 -type f \
      \( -name 'libmeep.so*' -o -name 'libpympb.so*' \) -print
    find "${MEEP_SITE_PACKAGES}" -type f -name '*.so*' -print
  } | sort -u
)
if [[ "${#GPMEEP_ELF_FILES[@]}" -eq 0 ]]; then
  echo "error: no gpmeep-owned ELF files were found for GLIBC audit" >&2
  exit 1
fi
PATCHELF="$(command -v patchelf)"
if [[ ! -x "${PATCHELF}" ]]; then
  echo "error: the conda build prefix has no patchelf" >&2
  exit 1
fi
# Conda normally rewrites RPATHs after build.sh. The installed provenance
# contract binds the exact ELF bytes, so finalize the same package-relative
# layout here and disable Conda's later binary relocation in meta.yaml.
for ELF in "${GPMEEP_ELF_FILES[@]}"; do
  ELF_DIRECTORY="$(dirname -- "${ELF}")"
  LIB_RELATIVE="$(realpath --relative-to="${ELF_DIRECTORY}" -- "${PREFIX}/lib")"
  PACKAGE_RPATH="\$ORIGIN/${LIB_RELATIVE}"
  "${PATCHELF}" --force-rpath --set-rpath "${PACKAGE_RPATH}" "${ELF}"
  if "${PATCHELF}" --print-rpath "${ELF}" | grep -F -- "${PREFIX}" >/dev/null; then
    echo "error: package ELF retains an absolute build-prefix RPATH: ${ELF}" >&2
    exit 1
  fi
  if LC_ALL=C grep -a -F -q -- "${PREFIX}" "${ELF}"; then
    echo "error: audited package ELF retains replaceable build-prefix bytes: ${ELF}" >&2
    exit 1
  fi
done
"${PREFIX}/bin/python3.11" -I \
  "${SRC_DIR}/scripts/check-glibc-compatibility.py" \
  --root "${PREFIX}" \
  --maximum "${GPMEEP_GLIBC_MINIMUM}" \
  --report "${GLIBC_AUDIT}" \
  "${GPMEEP_ELF_FILES[@]}"
GLIBC_AUDIT_SHA256="$(sha256sum "${GLIBC_AUDIT}" | awk '{print $1}')"

CUOBJDUMP="$(command -v cuobjdump)"
if [[ ! -x "${CUOBJDUMP}" ]]; then
  echo "error: the conda build prefix has no cuobjdump" >&2
  exit 1
fi
LIBMEEP_CUDA_ELF="$(readlink -f -- "${PREFIX}/lib/libmeep.so")"
if [[ ! -f "${LIBMEEP_CUDA_ELF}" || -L "${LIBMEEP_CUDA_ELF}" ]]; then
  echo "error: installed libmeep CUDA ELF is unavailable" >&2
  exit 1
fi
CUDA_ARCHITECTURE_AUDIT="${PREFIX}/share/gpmeep/cuda-architecture-inventory.json"
"${PREFIX}/bin/python3.11" -I \
  "${SRC_DIR}/scripts/check-cuda-architecture-inventory.py" \
  --root "${PREFIX}" \
  --binary "${LIBMEEP_CUDA_ELF}" \
  --cuobjdump "${CUOBJDUMP}" \
  --expected-real "${REAL_MANIFEST_COMPACT}" \
  --expected-virtual "${VIRTUAL_MANIFEST_COMPACT}" \
  --report "${CUDA_ARCHITECTURE_AUDIT}"
CUDA_ARCHITECTURE_AUDIT_SHA256="$(
  sha256sum "${CUDA_ARCHITECTURE_AUDIT}" | awk '{print $1}'
)"

MANIFEST="${PREFIX}/share/gpmeep/release.json"
sed \
  -e "s/@SOURCE_COMMIT@/${GPMEEP_SOURCE_COMMIT}/g" \
  -e "s/@CUDA_ARCHITECTURES@/${CUDA_ARCHITECTURES}/g" \
  -e "s/@CUDA_REAL_ARCHITECTURES@/${REAL_MANIFEST_ARCHITECTURES}/g" \
  -e "s/@CUDA_VIRTUAL_ARCHITECTURES@/${VIRTUAL_MANIFEST_ARCHITECTURES}/g" \
  -e "s/@RUNTIME_VALIDATED_ARCHITECTURES@/${RUNTIME_VALIDATED_ARCHITECTURES}/g" \
  -e "s/@CUDA_ARCHITECTURE_AUDIT_SHA256@/${CUDA_ARCHITECTURE_AUDIT_SHA256}/g" \
  -e "s/@GLIBC_AUDIT_SHA256@/${GLIBC_AUDIT_SHA256}/g" \
  "${RECIPE_DIR}/release.json.in" >"${MANIFEST}"
python -m json.tool "${MANIFEST}" >/dev/null
