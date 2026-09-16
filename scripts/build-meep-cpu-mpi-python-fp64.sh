#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: CPU FP64 builder requires protected Bash mode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_KIND="cpu-mpi-python-fp64"
QUALIFICATION_CONTRACT="gpmeep-cpu-mpi-python-fp64-v1"
LOCK="${REPO_ROOT}/environment/locks/cuda-mpi-linux-64.lock"
OPENMPI_PARAMS="${REPO_ROOT}/environment/openmpi-qualification-mca-params.conf"
MAMBA_ROOT="${GPMEEP_MAMBA_ROOT:-${REPO_ROOT}/.micromamba}"
MAMBA="${GPMEEP_MICROMAMBA:-${REPO_ROOT}/.tools/micromamba}"
MAMBA_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
PACKAGE_CACHE="${MAMBA_ROOT}/pkgs"
ENV_PREFIX="${REPO_ROOT}/.envs/gpmeep-cpu-mpi-fp64"
BUILD_DIR="${REPO_ROOT}/build/meep-cpu-mpi-python-fp64"
INSTALL_DIR="${REPO_ROOT}/install/meep-cpu-mpi-python-fp64"
BUILD_HOME="${BUILD_DIR}/build-home"
BUILD_CACHE_HOME="${BUILD_HOME}/.cache"
BUILD_CONFIG_HOME="${BUILD_HOME}/.config"
BUILD_MPLCONFIG="${BUILD_HOME}/.matplotlib"
MAKE_JOBS="${MEEP_GPU_MAKE_JOBS:-$(/usr/bin/nproc)}"
QUALIFICATION_TIMEOUT_SECONDS=300
STAGE="${1:-bootstrap}"

if [[ ! "${MAKE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MEEP_GPU_MAKE_JOBS must be a positive decimal integer" >&2
  exit 2
fi

usage() {
  echo "usage: $0" >&2
  exit 2
}

if [[ "$#" -gt 1 ]]; then
  usage
fi

if [[ "${STAGE}" == "bootstrap" ]]; then
  if [[ "$#" -ne 0 ]]; then
    usage
  fi
  if [[ ! -x "${MAMBA}" ]]; then
    if [[ -n "${GPMEEP_MICROMAMBA:-}" ]]; then
      echo "error: GPMEEP_MICROMAMBA is not executable: ${MAMBA}" >&2
      exit 1
    fi
    /bin/bash -p "${SCRIPT_DIR}/bootstrap-micromamba.sh" >/dev/null
  fi
  if [[ ! -x "${MAMBA}" || ! -f "${LOCK}" ]]; then
    echo "error: pinned Micromamba or MPI lock is unavailable" >&2
    exit 1
  fi
  if [[ "$(/usr/bin/sha256sum "${MAMBA}" | /usr/bin/awk '{print $1}')" != \
        "${MAMBA_SHA256}" ]]; then
    echo "error: Micromamba does not match the pinned protected binary" >&2
    exit 1
  fi

  USER_HOME="$(/usr/bin/getent passwd "$(/usr/bin/id -u)" | /usr/bin/cut -d: -f6)"
  if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
    echo "error: Open MPI requires a valid account home" >&2
    exit 1
  fi
  MAMBA_PROCESS_CACHE="${MAMBA_ROOT}/cache"
  /usr/bin/mkdir -p "${MAMBA_ROOT}" "${PACKAGE_CACHE}" \
    "${MAMBA_PROCESS_CACHE}" "${REPO_ROOT}/.envs"
  CLEAN_BOOTSTRAP_ENV=(
    /usr/bin/env -i
    "HOME=${USER_HOME}"
    "XDG_CACHE_HOME=${MAMBA_PROCESS_CACHE}"
    "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
    "CONDA_PKGS_DIRS=${PACKAGE_CACHE}"
    PATH=/usr/bin:/bin
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONPYCACHEPREFIX=/dev/null
  )
  for variable in HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY \
    http_proxy https_proxy no_proxy all_proxy SSL_CERT_FILE SSL_CERT_DIR; do
    if [[ -v "${variable}" ]]; then
      CLEAN_BOOTSTRAP_ENV+=("${variable}=${!variable}")
    fi
  done
  if [[ -e "${ENV_PREFIX}" ]]; then
    BACKUP="${ENV_PREFIX}.previous.$(/usr/bin/date -u +%Y%m%dT%H%M%SZ).$$"
    /usr/bin/mv -- "${ENV_PREFIX}" "${BACKUP}"
    echo "Previous isolated CPU MPI environment preserved at ${BACKUP}"
  fi
  "${CLEAN_BOOTSTRAP_ENV[@]}" "${MAMBA}" \
    --no-rc create --yes --safety-checks enabled \
    --extra-safety-checks --root-prefix "${MAMBA_ROOT}" \
    --prefix "${ENV_PREFIX}" --file "${LOCK}"

  exec "${CLEAN_BOOTSTRAP_ENV[@]}" \
    "GPMEEP_MICROMAMBA=${MAMBA}" \
    "GPMEEP_MAMBA_ROOT=${MAMBA_ROOT}" \
    "MEEP_GPU_MAKE_JOBS=${MAKE_JOBS}" \
    "${MAMBA}" --no-rc run --clean-env \
    --env "HOME=${USER_HOME}" \
    --env "GPMEEP_MICROMAMBA=${MAMBA}" \
    --env "GPMEEP_MAMBA_ROOT=${MAMBA_ROOT}" \
    --env "MEEP_GPU_MAKE_JOBS=${MAKE_JOBS}" \
    --env "XDG_CACHE_HOME=${MAMBA_PROCESS_CACHE}" \
    --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
    --env "CONDA_PKGS_DIRS=${PACKAGE_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPYCACHEPREFIX=/dev/null \
    --env MEEP_GPU_BACKEND=cpu \
    --root-prefix "${MAMBA_ROOT}" --prefix "${ENV_PREFIX}" \
    /bin/bash -p "${BASH_SOURCE[0]}" audited-worker
fi

if [[ "${STAGE}" != "audited-worker" || "$#" -ne 1 ]]; then
  usage
fi
if [[ "${CONDA_PREFIX:-}" != "${ENV_PREFIX}" ]]; then
  echo "error: expected isolated prefix ${ENV_PREFIX}, got ${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi
if [[ "${MEEP_GPU_BACKEND:-}" != "cpu" ]]; then
  echo "error: CPU FP64 builder requires MEEP_GPU_BACKEND=cpu" >&2
  exit 1
fi
if [[ ! -x "${MAMBA}" || ! -f "${OPENMPI_PARAMS}" ]]; then
  echo "error: builder provenance inputs are absent" >&2
  exit 1
fi
if [[ "$(/usr/bin/sha256sum "${MAMBA}" | /usr/bin/awk '{print $1}')" != \
      "${MAMBA_SHA256}" ]]; then
  echo "error: audited worker received an unpinned Micromamba binary" >&2
  exit 1
fi

for tool in autoreconf cc c++ make mpicxx mpiexec h5pcc python swig; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${ENV_PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside isolated prefix: ${TOOL_PATH}" >&2
    exit 1
  fi
done
for compiler_variable in CC CXX FC F77; do
  COMPILER_PATH="${!compiler_variable:-}"
  if [[ ! -x "${COMPILER_PATH}" || "${COMPILER_PATH}" != "${ENV_PREFIX}/bin/"* ]]; then
    echo "error: ${compiler_variable} is not an isolated compiler: ${COMPILER_PATH:-<unset>}" >&2
    exit 1
  fi
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/dev/null
export MEEP_GPU_BACKEND=cpu
export MPICXX="${ENV_PREFIX}/bin/mpicxx"
export OMPI_MCA_mca_base_param_files="${OPENMPI_PARAMS}"
export OMPI_MCA_mca_base_component_path="${ENV_PREFIX}/lib/openmpi"
export PMIX_MCA_mca_base_param_files="${OPENMPI_PARAMS}"
export PMIX_MCA_mca_base_component_path="${ENV_PREFIX}/lib/pmix"
export PRTE_MCA_mca_base_param_files="${OPENMPI_PARAMS}"

INSTALLED_LOCK="$(/usr/bin/mktemp "${MAMBA_ROOT}/cpu-mpi-installed.XXXXXX")"
EXPECTED_LOCK="$(/usr/bin/mktemp "${MAMBA_ROOT}/cpu-mpi-expected.XXXXXX")"
cleanup_lock_audit() {
  /usr/bin/rm -f -- "${INSTALLED_LOCK}" "${EXPECTED_LOCK}"
}
trap cleanup_lock_audit EXIT
"${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
  --prefix "${ENV_PREFIX}" --explicit --sha256 | \
  /usr/bin/sed -n '/:\/\//p' | /usr/bin/sort >"${INSTALLED_LOCK}"
/usr/bin/sed -n '/:\/\//p' "${LOCK}" | /usr/bin/sort >"${EXPECTED_LOCK}"
if ! /usr/bin/diff -u "${EXPECTED_LOCK}" "${INSTALLED_LOCK}"; then
  echo "error: CPU MPI prefix differs from the exact lock" >&2
  exit 1
fi
cleanup_lock_audit
trap - EXIT

HDF5_CONFIGURATION="$(h5pcc -showconfig)"
if [[ "${HDF5_CONFIGURATION}" != *"Parallel HDF5: yes"* ]]; then
  echo "error: CPU MPI baseline requires parallel HDF5" >&2
  exit 1
fi
python -c 'import mpi4py, h5py; assert h5py.get_config().mpi'

CONFIGURE_ARGS=(
  --enable-maintainer-mode
  --enable-shared
  --disable-single
  --disable-cuda
  --disable-cuda-fast-math
  --with-openmp
  --with-mpi
  --with-python
  --without-scheme
  "--prefix=${INSTALL_DIR}"
)

TIMESTAMP="$(/usr/bin/date -u +%Y%m%dT%H%M%SZ).$$"
for old_directory in "${BUILD_DIR}" "${INSTALL_DIR}"; do
  if [[ -e "${old_directory}" ]]; then
    /usr/bin/mv -- "${old_directory}" "${old_directory}.previous.${TIMESTAMP}"
  fi
done
/usr/bin/mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
HOST_BUILD_LOGS="${BUILD_DIR}/host-build-logs"
/usr/bin/install -d -m 700 "${HOST_BUILD_LOGS}" "${BUILD_HOME}" \
  "${BUILD_CACHE_HOME}" "${BUILD_CONFIG_HOME}" "${BUILD_MPLCONFIG}"
export HOME="${BUILD_HOME}"
export XDG_CACHE_HOME="${BUILD_CACHE_HOME}"
export XDG_CONFIG_HOME="${BUILD_CONFIG_HOME}"
export MPLCONFIGDIR="${BUILD_MPLCONFIG}"

run_bounded_build_step() {
  local label="$1"
  local timeout_seconds="$2"
  shift 2
  local log="${HOST_BUILD_LOGS}/${label}.log"
  if /usr/bin/timeout --signal=TERM --kill-after=30s \
    "${timeout_seconds}s" "$@" >"${log}" 2>&1; then
    printf 'gpmeep-host-build-step-pass:%s\n' "${label}" >>"${log}"
  else
    local status=$?
    echo "error: ${label} failed with status ${status}" >&2
    /usr/bin/cat "${log}" >&2
    exit "${status}"
  fi
}

RECEIPT_BEGIN=(
  python "${SCRIPT_DIR}/write-build-receipt.py" begin
  --repo "${REPO_ROOT}"
  --build-dir "${BUILD_DIR}"
  --build-kind "${BUILD_KIND}"
  --builder "${BASH_SOURCE[0]}"
  --qualification-contract "${QUALIFICATION_CONTRACT}"
  --lockfile "environment_lock=${LOCK}"
)
for argument in "${CONFIGURE_ARGS[@]}"; do
  RECEIPT_BEGIN+=("--configure-arg=${argument}")
done
for variable in MEEP_GPU_MAKE_JOBS MEEP_GPU_BACKEND CC CXX FC F77 MPICXX \
  HOME XDG_CACHE_HOME XDG_CONFIG_HOME MPLCONFIGDIR; do
  RECEIPT_BEGIN+=(--record-env "${variable}")
done
"${RECEIPT_BEGIN[@]}"

cd "${REPO_ROOT}"
run_bounded_build_step autoreconf 1800 autoreconf --verbose --install --symlink --force
cd "${BUILD_DIR}"
run_bounded_build_step configure 1800 "../../configure" "${CONFIGURE_ARGS[@]}"
run_bounded_build_step build 7200 make -j"${MAKE_JOBS}"
run_bounded_build_step check 7200 make check

INSTALL_PYCACHE="${BUILD_DIR}/install-pycache"
/usr/bin/install -d -m 700 "${INSTALL_PYCACHE}"
run_bounded_build_step install 1800 /usr/bin/env \
  "PYTHONPYCACHEPREFIX=${INSTALL_PYCACHE}" make install

assert_no_runtime_bytecode() {
  local root="$1"
  local unexpected
  unexpected="$(/usr/bin/find "${root}" \
    \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) \
    -print -quit)"
  if [[ -n "${unexpected}" ]]; then
    echo "error: executable Python bytecode is not allowed in ${root}: ${unexpected}" >&2
    exit 1
  fi
}
assert_no_runtime_bytecode "${BUILD_DIR}/python/meep"
assert_no_runtime_bytecode "${INSTALL_DIR}"

BUILD_EXTENSION="$(/usr/bin/readlink -f "${BUILD_DIR}/python/meep/_meep.so")"
BUILD_LIBMEEP="$(/usr/bin/readlink -f "${BUILD_DIR}/src/.libs/libmeep.so")"
shopt -s nullglob
INSTALLED_SITE_DIRS=("${INSTALL_DIR}"/lib/python*/site-packages)
if [[ "${#INSTALLED_SITE_DIRS[@]}" -ne 1 ]]; then
  echo "error: installed Python site-packages layout is ambiguous" >&2
  exit 1
fi
INSTALLED_SITE="${INSTALLED_SITE_DIRS[0]}"
INSTALLED_EXTENSION="$(/usr/bin/readlink -f "${INSTALLED_SITE}/meep/_meep.so")"
INSTALLED_LIBMEEP="$(/usr/bin/readlink -f "${INSTALL_DIR}/lib/libmeep.so")"
shopt -u nullglob
for artifact in "${BUILD_EXTENSION}" "${BUILD_LIBMEEP}" \
  "${INSTALLED_EXTENSION}" "${INSTALLED_LIBMEEP}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "error: expected build artifact is absent: ${artifact}" >&2
    exit 1
  fi
done

QUALIFICATION_DIR="${BUILD_DIR}/cpu-mpi-fp64-qualification"
QUALIFICATION_HOME="${BUILD_DIR}/qualification-home"
/usr/bin/install -d -m 700 "${QUALIFICATION_DIR}" "${QUALIFICATION_HOME}"
printf '%s\n' 'gpmeep isolated CPU MPI qualification home' > \
  "${QUALIFICATION_HOME}/.gpmeep-empty-home"

run_qualification() {
  local package_root="$1"
  local library_root="$2"
  local ranks="$3"
  local label="$4"
  local runtime_kind="$5"
  local expected_extension="$6"
  local expected_libmeep="$7"
  if /usr/bin/timeout --signal=TERM --kill-after=10s \
    "${QUALIFICATION_TIMEOUT_SECONDS}s" /usr/bin/env -i \
    "HOME=${QUALIFICATION_HOME}" PATH="${ENV_PREFIX}/bin:/usr/bin:/bin" \
    "PYTHONPATH=${package_root}" "LD_LIBRARY_PATH=${library_root}:${ENV_PREFIX}/lib" \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
    MEEP_GPU_BACKEND=cpu \
    "OMPI_MCA_mca_base_param_files=${OPENMPI_PARAMS}" \
    "OMPI_MCA_mca_base_component_path=${ENV_PREFIX}/lib/openmpi" \
    "PMIX_MCA_mca_base_param_files=${OPENMPI_PARAMS}" \
    "PMIX_MCA_mca_base_component_path=${ENV_PREFIX}/lib/pmix" \
    "PRTE_MCA_mca_base_param_files=${OPENMPI_PARAMS}" \
    "${ENV_PREFIX}/bin/mpiexec" --bind-to core -n "${ranks}" \
    "${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/qualify-cpu-mpi-fp64.py" \
    --expected-size "${ranks}" --runtime-kind "${runtime_kind}" \
    --expected-extension "${expected_extension}" \
    --expected-libmeep "${expected_libmeep}" \
    >"${QUALIFICATION_DIR}/${label}.log" 2>&1; then
    :
  else
    local status=$?
    echo "error: ${label} qualification failed with status ${status}" >&2
    /usr/bin/cat "${QUALIFICATION_DIR}/${label}.log" >&2
    exit "${status}"
  fi
  if [[ "$(/usr/bin/grep -c '^gpmeep-cpu-mpi-fp64-qualification:' \
      "${QUALIFICATION_DIR}/${label}.log")" -ne 1 ]]; then
    echo "error: ${label} qualification emitted no exact success marker" >&2
    /usr/bin/cat "${QUALIFICATION_DIR}/${label}.log" >&2
    exit 1
  fi
}

run_qualification "${BUILD_DIR}/python" "${BUILD_DIR}/src/.libs" 1 \
  in-place-one-rank in-place "${BUILD_EXTENSION}" "${BUILD_LIBMEEP}"
run_qualification "${BUILD_DIR}/python" "${BUILD_DIR}/src/.libs" 2 \
  in-place-two-rank in-place "${BUILD_EXTENSION}" "${BUILD_LIBMEEP}"
run_qualification "${INSTALLED_SITE}" "${INSTALL_DIR}/lib" 1 \
  installed-one-rank installed "${INSTALLED_EXTENSION}" "${INSTALLED_LIBMEEP}"
run_qualification "${INSTALLED_SITE}" "${INSTALL_DIR}/lib" 2 \
  installed-two-rank installed "${INSTALLED_EXTENSION}" "${INSTALLED_LIBMEEP}"

PREFIX_CONTENT_AUDIT="${BUILD_DIR}/conda-prefix-content-audit.json"
/usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  /usr/bin/python3 -S -P -B "${SCRIPT_DIR}/audit-conda-prefix.py" \
  --prefix "${ENV_PREFIX}" --lock "${LOCK}" \
  --package-cache "${PACKAGE_CACHE}" \
  --output "${PREFIX_CONTENT_AUDIT}"

ENVIRONMENT_EXPLICIT="${BUILD_DIR}/environment-explicit.lock"
"${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
  --prefix "${ENV_PREFIX}" --explicit --sha256 >"${ENVIRONMENT_EXPLICIT}"

python "${SCRIPT_DIR}/write-build-receipt.py" finalize \
  --repo "${REPO_ROOT}" \
  --build-dir "${BUILD_DIR}" \
  --configuration-file "config_h=${BUILD_DIR}/config.h" \
  --configuration-file "config_status=${BUILD_DIR}/config.status" \
  --configuration-file "environment_explicit=${ENVIRONMENT_EXPLICIT}" \
  --configuration-file "conda_prefix_content_audit=${PREFIX_CONTENT_AUDIT}" \
  --configuration-file "openmpi_qualification_params=${OPENMPI_PARAMS}" \
  --artifact "python_extension=${BUILD_EXTENSION}" \
  --artifact "libmeep=${BUILD_LIBMEEP}" \
  --artifact "installed_python_extension=${INSTALLED_EXTENSION}" \
  --artifact "installed_libmeep=${INSTALLED_LIBMEEP}" \
  --manifest "in_place_python=${BUILD_DIR}/python/meep" \
  --manifest "installed_python=${INSTALLED_SITE}/meep" \
  --manifest "installed_prefix=${INSTALL_DIR}" \
  --manifest "installed_environment=${ENV_PREFIX}" \
  --manifest "qualification_logs=${QUALIFICATION_DIR}" \
  --manifest "host_build_logs=${HOST_BUILD_LOGS}" \
  --tool autoreconf --tool cc --tool c++ --tool make --tool python \
  --tool swig --tool mpicxx --tool mpiexec --tool h5pcc

python "${SCRIPT_DIR}/verify-build-receipt.py" \
  --repo "${REPO_ROOT}" \
  --receipt "${BUILD_DIR}/build-provenance.json" \
  --expected-build-kind "${BUILD_KIND}"

echo "CPU-only FP64 MPI+Python gpmeep build installed in ${INSTALL_DIR}"
echo "Build receipt: ${BUILD_DIR}/build-provenance.json"
