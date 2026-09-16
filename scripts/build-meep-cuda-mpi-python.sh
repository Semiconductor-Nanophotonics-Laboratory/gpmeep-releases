#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: authoritative builder requires protected Bash mode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
MAMBA_PACKAGES="${MAMBA_ROOT}/pkgs"
MPL_CONFIG="${MAMBA_CACHE}/matplotlib"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda-mpi"
LOCK="${REPO_ROOT}/environment/locks/cuda-mpi-linux-64.lock"
OPENMPI_QUALIFICATION_PARAMS="${REPO_ROOT}/environment/openmpi-qualification-mca-params.conf"
QUALIFICATION_CONTRACT="gpmeep-cuda-mpi-python-fp32-v2"
RELEASE_INITIAL_CONDITION="trigonometric-v1"
RELEASE_SOURCE_PROFILE="single-ez-v1"
DFT_PHASE_SHARING_QUALIFICATION_PIXELS=16
BUILD_DIR="${REPO_ROOT}/build/meep-cuda-mpi-python-fp32"
INSTALL_DIR="${REPO_ROOT}/install/meep-cuda-mpi-python-fp32"
FRESH_ATTESTATION_DIRECTORY="${MAMBA_ROOT}/fresh-environment-attestations"
FRESH_ATTESTATION_SCRIPT="${SCRIPT_DIR}/fresh-environment-attestation.py"
CONTROL_PYTHON_RUNNER="${SCRIPT_DIR}/gpmeep-control-python.py"
CONTROL_PYTHON_ENV=(
  /usr/bin/env
  -u PYTHONPATH
  -u PYTHONHOME
  -u PYTHONSTARTUP
  -u PYTHONUSERBASE
  -u PYTHONINSPECT
  -u PYTHONWARNINGS
  -u PYTHONBREAKPOINT
  -u PYTHONPROFILEIMPORTTIME
  PYTHONNOUSERSITE=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONPYCACHEPREFIX=/dev/null
  PYTHONSAFEPATH=1
)
SYSTEM_CONTROL_PYTHON=(
  "${CONTROL_PYTHON_ENV[@]}"
  /usr/bin/python3 -S -P -B "${CONTROL_PYTHON_RUNNER}"
)
PREFIX_CONTROL_PYTHON=(
  "${CONTROL_PYTHON_ENV[@]}"
  "${PREFIX}/bin/python" -S -P -B "${CONTROL_PYTHON_RUNNER}"
)

expected_failure_status_is_acceptable() {
  "${PREFIX_CONTROL_PYTHON[@]}" \
    "${SCRIPT_DIR}/gpmeep_qualification_contract.py" \
    check-expected-failure-status "$1"
}

BUILD_STAGE="${1:-bootstrap}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
"${SYSTEM_CONTROL_PYTHON[@]}" "${SCRIPT_DIR}/verify-protected-bash.py" \
  --script "${BASH_SOURCE[0]}" --stage "${BUILD_STAGE}"

if [[ "${BUILD_STAGE}" == "bootstrap" ]]; then
  if [[ "$#" -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
  fi
  USER_HOME="$(/usr/bin/getent passwd "$(/usr/bin/id -u)" | /usr/bin/cut -d: -f6)"
  if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
    echo "error: Open MPI requires the launch user's account home directory" >&2
    exit 1
  fi
  CLEAN_SYSTEM_ENV=(/usr/bin/env -i "HOME=${USER_HOME}" PATH=/usr/bin:/bin)
  for variable in HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY \
    http_proxy https_proxy no_proxy all_proxy SSL_CERT_FILE SSL_CERT_DIR; do
    if [[ -v "${variable}" ]]; then
      CLEAN_SYSTEM_ENV+=("${variable}=${!variable}")
    fi
  done
  case "${MEEP_GPU_FAST_MATH:-OFF}" in
    OFF|off|0|no|NO|false|FALSE)
      ;;
    ON|on|1|yes|YES|true|TRUE)
      echo "error: authoritative release builds require CUDA fast-math OFF" >&2
      exit 2
      ;;
    *)
      echo "error: MEEP_GPU_FAST_MATH must be OFF for authoritative release builds" >&2
      exit 2
      ;;
  esac
  "${CLEAN_SYSTEM_ENV[@]}" /bin/bash -p \
    "${SCRIPT_DIR}/bootstrap-micromamba.sh"
  # An authoritative build starts from a newly materialized exact-lock prefix.
  # The previous dedicated gpmeep environment is preserved by create-env, so
  # pre-existing binary/metadata tampering cannot be sealed into a new receipt.
  "${CLEAN_SYSTEM_ENV[@]}" /bin/bash -p \
    "${SCRIPT_DIR}/create-env.sh" cuda-mpi --recreate
  if [[ ! -x "${MAMBA}" || ! -d "${PREFIX}" ]]; then
    echo "error: CUDA MPI environment is missing; run scripts/create-env.sh cuda-mpi" >&2
    exit 1
  fi
  /usr/bin/mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}" "${MPL_CONFIG}"
  RUN_ENVIRONMENT=(
    --clean-env
    --env "HOME=${USER_HOME}"
    --env "XDG_CACHE_HOME=${MAMBA_CACHE}"
    --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
    --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}"
    --env "MPLCONFIGDIR=${MPL_CONFIG}"
    --env MPLBACKEND=Agg
    --env PYTHONNOUSERSITE=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONPYCACHEPREFIX=/dev/null
    --env PYTHONSAFEPATH=1
    --env MEEP_GPU_FAST_MATH=OFF
    --env "MEEP_GPU_MULTI_INITIAL_CONDITION=${RELEASE_INITIAL_CONDITION}"
    --env "MEEP_GPU_MULTI_SOURCE_PROFILE=${RELEASE_SOURCE_PROFILE}"
  )
  for variable in MEEP_GPU_MAKE_JOBS \
    MEEP_GPU_CUDA_ARCHS MEEP_GPU_DEVICE MEEP_GPU_ALLOW_OVERSUBSCRIBE \
    MEEP_GPU_MPI_TRANSPORT CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER OMPI_MCA_opal_cuda_support UCX_MEMTYPE_CACHE; do
    if [[ -v "${variable}" ]]; then
      RUN_ENVIRONMENT+=(--env "${variable}=${!variable}")
    fi
  done
  cd "${REPO_ROOT}"
  exec /usr/bin/env -i "HOME=${USER_HOME}" PATH=/usr/bin:/bin \
    "XDG_CACHE_HOME=${MAMBA_CACHE}" "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
    "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}" \
    "${MAMBA}" --no-rc run "${RUN_ENVIRONMENT[@]}" \
    --root-prefix "${MAMBA_ROOT}" \
    --prefix "${PREFIX}" \
    /bin/bash -p "${BASH_SOURCE[0]}" audited-worker
fi

if [[ "${BUILD_STAGE}" != "audited-worker" || "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
if [[ "${MEEP_GPU_FAST_MATH:-}" != "OFF" ]]; then
  echo "error: audited release worker requires MEEP_GPU_FAST_MATH=OFF" >&2
  exit 2
fi
if [[ "${CONDA_PREFIX:-}" != "${PREFIX}" ]]; then
  echo "error: expected isolated environment ${PREFIX}, got ${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi
if [[ ! -x /usr/bin/python3 ]]; then
  echo "error: protected system Python is required for prefix verification" >&2
  exit 1
fi
if [[ ! -x "${MAMBA}" || \
      "$(/usr/bin/sha256sum "${MAMBA}" | /usr/bin/awk '{print $1}')" != \
      "${MAMBA_SHA256}" ]]; then
  echo "error: Micromamba does not match the pinned protected binary" >&2
  exit 1
fi

# The only worker entry point repairs and audits the prefix itself. Positional
# audit, attestation, or canonical-environment inputs are never accepted, so a
# direct worker invocation cannot self-seal fabricated evidence.
USER_HOME="$(/usr/bin/getent passwd "$(/usr/bin/id -u)" | /usr/bin/cut -d: -f6)"
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
  echo "error: Open MPI requires the launch user's account home directory" >&2
  exit 1
fi
/usr/bin/mkdir -p "${MAMBA_CACHE}" "${MAMBA_PACKAGES}" \
  "${FRESH_ATTESTATION_DIRECTORY}"
/usr/bin/chmod 700 "${FRESH_ATTESTATION_DIRECTORY}"
"${MAMBA}" --no-rc install --yes --force-reinstall \
  --no-pyc \
  --safety-checks enabled --extra-safety-checks \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" --file "${LOCK}"
PREFIX_BYTECODE_NORMALIZATION_TEMP="${MAMBA_CACHE}/cuda-mpi-bytecode-normalization.$$.json"
"${SYSTEM_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/normalize-conda-generated-bytecode.py" \
  --prefix "${PREFIX}" --output "${PREFIX_BYTECODE_NORMALIZATION_TEMP}"
# The pinned Fontconfig package installs an empty writable cache directory.
# Seal it before the byte audit and fresh-environment attestation: directory
# modes are part of every later identity and receipt tree manifest.
PREFIX_FONTCONFIG_CACHE="${PREFIX}/var/cache/fontconfig"
if [[ ! -d "${PREFIX_FONTCONFIG_CACHE}" || \
      -L "${PREFIX_FONTCONFIG_CACHE}" || \
      ! -f "${PREFIX_FONTCONFIG_CACHE}/.leave" || \
      -L "${PREFIX_FONTCONFIG_CACHE}/.leave" || \
      "$(/usr/bin/stat -c '%s' "${PREFIX_FONTCONFIG_CACHE}/.leave")" -ne 0 || \
      "$(/usr/bin/sha256sum "${PREFIX_FONTCONFIG_CACHE}/.leave" | \
           /usr/bin/awk '{print $1}')" != \
        e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]]; then
  echo "error: conda prefix Fontconfig cache baseline is invalid" >&2
  exit 1
fi
PREFIX_FONTCONFIG_UNEXPECTED="$(
  /usr/bin/find "${PREFIX_FONTCONFIG_CACHE}" -mindepth 1 -maxdepth 1 \
    ! -name .leave -print -quit
)"
if [[ -n "${PREFIX_FONTCONFIG_UNEXPECTED}" ]]; then
  echo "error: conda prefix Fontconfig cache was populated: ${PREFIX_FONTCONFIG_UNEXPECTED}" >&2
  exit 1
fi
/usr/bin/chmod 0444 "${PREFIX_FONTCONFIG_CACHE}/.leave"
/usr/bin/chmod 0555 "${PREFIX_FONTCONFIG_CACHE}"
if [[ "$(/usr/bin/stat -c '%a' "${PREFIX_FONTCONFIG_CACHE}")" != 555 || \
      "$(/usr/bin/stat -c '%a' "${PREFIX_FONTCONFIG_CACHE}/.leave")" != 444 ]]; then
  echo "error: conda prefix Fontconfig cache could not be sealed" >&2
  exit 1
fi
PREFIX_PRE_NORMALIZATION_AUDIT_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-pre-normalization-audit.$$.json"
"${SYSTEM_CONTROL_PYTHON[@]}" "${SCRIPT_DIR}/audit-conda-prefix.py" \
  --prefix "${PREFIX}" --lock "${LOCK}" \
  --package-cache "${MAMBA_PACKAGES}" \
  --output "${PREFIX_PRE_NORMALIZATION_AUDIT_TEMP}"
PREFIX_SOURCE_BYTECODE_NORMALIZATION_TEMP="${MAMBA_CACHE}/cuda-mpi-source-bytecode-normalization.$$.json"
"${SYSTEM_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/normalize-conda-relocated-bytecode.py" \
  --prefix "${PREFIX}" \
  --pre-audit "${PREFIX_PRE_NORMALIZATION_AUDIT_TEMP}" \
  --compiler "${SCRIPT_DIR}/compile-conda-relocated-bytecode.py" \
  --output "${PREFIX_SOURCE_BYTECODE_NORMALIZATION_TEMP}"
PREFIX_CONTENT_AUDIT_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-content-audit.$$.json"
"${SYSTEM_CONTROL_PYTHON[@]}" "${SCRIPT_DIR}/audit-conda-prefix.py" \
  --prefix "${PREFIX}" --lock "${LOCK}" \
  --package-cache "${MAMBA_PACKAGES}" \
  --output "${PREFIX_CONTENT_AUDIT_TEMP}"

# Exercise the normalized cache from two ordinary interpreter processes.  No
# -B flag or bytecode environment override is allowed here: this is the exact
# behavior that previously rewrote a relocated NumPy CHECKED_HASH pyc.
PREFIX_IMPORT_PROBE_ONE_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-import-probe-one.$$.json"
PREFIX_IMPORT_PROBE_TWO_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-import-probe-two.$$.json"
PREFIX_POST_IMPORT_AUDIT_ONE_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-post-import-audit-one.$$.json"
PREFIX_POST_IMPORT_AUDIT_TWO_TEMP="${MAMBA_CACHE}/cuda-mpi-prefix-post-import-audit-two.$$.json"
for probe_number in one two; do
  probe_variable="PREFIX_IMPORT_PROBE_${probe_number^^}_TEMP"
  audit_variable="PREFIX_POST_IMPORT_AUDIT_${probe_number^^}_TEMP"
  /usr/bin/env -i HOME="${USER_HOME}" \
    PATH="${PREFIX}/bin:/usr/bin:/bin" PYTHONNOUSERSITE=1 \
    "${PREFIX}/bin/python" \
    "${SCRIPT_DIR}/probe-conda-relocated-bytecode-imports.py" \
    --prefix "${PREFIX}" \
    --normalization "${PREFIX_SOURCE_BYTECODE_NORMALIZATION_TEMP}" \
    --output "${!probe_variable}"
  "${SYSTEM_CONTROL_PYTHON[@]}" "${SCRIPT_DIR}/audit-conda-prefix.py" \
    --prefix "${PREFIX}" --lock "${LOCK}" \
    --package-cache "${MAMBA_PACKAGES}" \
    --output "${!audit_variable}"
  /usr/bin/cmp --silent "${PREFIX_CONTENT_AUDIT_TEMP}" "${!audit_variable}"
done
/usr/bin/cmp --silent \
  "${PREFIX_IMPORT_PROBE_ONE_TEMP}" "${PREFIX_IMPORT_PROBE_TWO_TEMP}"
mapfile -t FRESH_ATTESTATION_OUTPUT < <(
  "${SYSTEM_CONTROL_PYTHON[@]}" "${FRESH_ATTESTATION_SCRIPT}" create \
    --repo "${REPO_ROOT}" --prefix "${PREFIX}" --lock "${LOCK}" \
    --output-directory "${FRESH_ATTESTATION_DIRECTORY}"
)
if [[ "${#FRESH_ATTESTATION_OUTPUT[@]}" -ne 2 ]]; then
  echo "error: audited exact-lock environment attestation was not created" >&2
  exit 1
fi
export GPMEEP_FRESH_ENV_NONCE="${FRESH_ATTESTATION_OUTPUT[0]}"
GPMEEP_FRESH_ENV_ATTESTATION="${FRESH_ATTESTATION_OUTPUT[1]}"
BUILD_HOME="${MAMBA_CACHE}/build-home-${GPMEEP_FRESH_ENV_NONCE}"
if [[ -e "${BUILD_HOME}" ]]; then
  echo "error: nonce-scoped isolated build HOME already exists" >&2
  exit 1
fi
/usr/bin/install -d -m 700 "${BUILD_HOME}"
printf '%s\n' 'gpmeep isolated build home' > \
  "${BUILD_HOME}/.gpmeep-empty-build-home"
/usr/bin/mkdir -p "${BUILD_HOME}/cache" "${BUILD_HOME}/matplotlib"
MPL_CONFIG="${BUILD_HOME}/matplotlib"
export HOME="${BUILD_HOME}"
export XDG_CACHE_HOME="${BUILD_HOME}/cache"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export CONDA_PKGS_DIRS="${MAMBA_PACKAGES}"
export MPLCONFIGDIR="${MPL_CONFIG}"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/dev/null
export PYTHONSAFEPATH=1
export OMPI_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
export OMPI_MCA_mca_base_component_path="${PREFIX}/lib/openmpi"
export PMIX_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
export PMIX_MCA_mca_base_component_path="${PREFIX}/lib/pmix"
export PRTE_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
# Every direct gpu-mpi-performance qualification inherits these explicit
# release profiles. No record-emitting lane relies on either worker unset-env
# compatibility default.
export MEEP_GPU_MULTI_INITIAL_CONDITION="${RELEASE_INITIAL_CONDITION}"
export MEEP_GPU_MULTI_SOURCE_PROFILE="${RELEASE_SOURCE_PROFILE}"
WORKER_ENVIRONMENT=(
  --clean-env
  --env "HOME=${HOME}"
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  --env "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}"
  --env "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}"
  --env "MPLCONFIGDIR=${MPL_CONFIG}"
  --env MPLBACKEND=Agg
  --env PYTHONNOUSERSITE=1
  --env PYTHONDONTWRITEBYTECODE=1
  --env "PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}"
  --env PYTHONSAFEPATH=1
  --env MEEP_GPU_FAST_MATH=OFF
  --env "MEEP_GPU_MULTI_INITIAL_CONDITION=${RELEASE_INITIAL_CONDITION}"
  --env "MEEP_GPU_MULTI_SOURCE_PROFILE=${RELEASE_SOURCE_PROFILE}"
  --env "GPMEEP_FRESH_ENV_NONCE=${GPMEEP_FRESH_ENV_NONCE}"
  --env "OMPI_MCA_mca_base_param_files=${OMPI_MCA_mca_base_param_files}"
  --env "OMPI_MCA_mca_base_component_path=${OMPI_MCA_mca_base_component_path}"
  --env "PMIX_MCA_mca_base_param_files=${PMIX_MCA_mca_base_param_files}"
  --env "PMIX_MCA_mca_base_component_path=${PMIX_MCA_mca_base_component_path}"
  --env "PRTE_MCA_mca_base_param_files=${PRTE_MCA_mca_base_param_files}"
)
for variable in MEEP_GPU_MAKE_JOBS \
  MEEP_GPU_CUDA_ARCHS MEEP_GPU_DEVICE MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  MEEP_GPU_MPI_TRANSPORT CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER OMPI_MCA_opal_cuda_support UCX_MEMTYPE_CACHE; do
  if [[ -v "${variable}" ]]; then
    WORKER_ENVIRONMENT+=(--env "${variable}=${!variable}")
  fi
done
if [[ "${PWD}" != "${REPO_ROOT}" ]]; then
  echo "error: audited worker must start in repository root ${REPO_ROOT}" >&2
  exit 1
fi
CANONICAL_BUILD_ENVIRONMENT_TEMP="${MAMBA_CACHE}/canonical-build-environment.$$.json"
/usr/bin/env -i "HOME=${HOME}" PATH=/usr/bin:/bin \
  "XDG_CACHE_HOME=${MAMBA_CACHE}" "MAMBA_ROOT_PREFIX=${MAMBA_ROOT}" \
  "CONDA_PKGS_DIRS=${MAMBA_PACKAGES}" \
  "${MAMBA}" --no-rc run "${WORKER_ENVIRONMENT[@]}" \
  --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
  "${SYSTEM_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/capture-build-environment.py" --output \
  "${CANONICAL_BUILD_ENVIRONMENT_TEMP}"
ACTUAL_BUILD_ENVIRONMENT_TEMP="${MAMBA_CACHE}/actual-build-environment.$$.json"
"${SYSTEM_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/capture-build-environment.py" \
  --output "${ACTUAL_BUILD_ENVIRONMENT_TEMP}"
if ! /usr/bin/cmp -s "${CANONICAL_BUILD_ENVIRONMENT_TEMP}" \
    "${ACTUAL_BUILD_ENVIRONMENT_TEMP}"; then
  /usr/bin/diff -u "${CANONICAL_BUILD_ENVIRONMENT_TEMP}" \
    "${ACTUAL_BUILD_ENVIRONMENT_TEMP}" >&2 || true
  echo "error: audited worker differs from the complete clean environment" >&2
  exit 1
fi
/usr/bin/rm -f "${ACTUAL_BUILD_ENVIRONMENT_TEMP}"
"${SYSTEM_CONTROL_PYTHON[@]}" "${FRESH_ATTESTATION_SCRIPT}" verify \
  --repo "${REPO_ROOT}" --prefix "${PREFIX}" --lock "${LOCK}" \
  --attestation "${GPMEEP_FRESH_ENV_ATTESTATION}" \
  --attestation-directory "${FRESH_ATTESTATION_DIRECTORY}" \
  --nonce "${GPMEEP_FRESH_ENV_NONCE}" --maximum-age-seconds 900
CLAIMED_FRESH_ENV_ATTESTATION="${GPMEEP_FRESH_ENV_ATTESTATION}.claimed.$$"
/usr/bin/mv -- "${GPMEEP_FRESH_ENV_ATTESTATION}" \
  "${CLAIMED_FRESH_ENV_ATTESTATION}"

# This verification is deliberately repeated inside the activated process.
# The build worker itself was launched by env -i plus micromamba --clean-env.
INSTALLED_LOCK_AUDIT="$(mktemp "${MAMBA_ROOT}/active-cuda-mpi.XXXXXX")"
EXPECTED_LOCK_AUDIT="$(mktemp "${MAMBA_ROOT}/expected-cuda-mpi.XXXXXX")"
cleanup_lock_audit() {
  rm -f "${INSTALLED_LOCK_AUDIT}" "${EXPECTED_LOCK_AUDIT}"
}
trap cleanup_lock_audit EXIT
if ! "${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
    --prefix "${PREFIX}" --explicit --sha256 | sed -n '/:\/\//p' | \
    sort >"${INSTALLED_LOCK_AUDIT}"; then
  echo "error: cannot audit the active CUDA/MPI environment" >&2
  exit 1
fi
sed -n '/:\/\//p' "${LOCK}" | sort >"${EXPECTED_LOCK_AUDIT}"
if ! diff -u "${EXPECTED_LOCK_AUDIT}" "${INSTALLED_LOCK_AUDIT}"; then
  echo "error: active CUDA/MPI environment differs from the exact lock" >&2
  exit 1
fi
cleanup_lock_audit
trap - EXIT

for tool in autoreconf cc c++ cmake make ninja nvcc mpicxx mpiexec h5pcc python swig; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the isolated environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done
TIMEOUT=""
for timeout_candidate in /usr/bin/timeout /bin/timeout; do
  if [[ -x "${timeout_candidate}" ]]; then
    TIMEOUT="${timeout_candidate}"
    break
  fi
done
if [[ -z "${TIMEOUT}" ]]; then
  echo "error: a protected system timeout is required for MPI build qualification" >&2
  exit 1
fi
if [[ ! -f "${OPENMPI_QUALIFICATION_PARAMS}" ]]; then
  echo "error: fixed Open MPI qualification parameter file is absent" >&2
  exit 1
fi
for mpi_override in openmpi-mca-params-override.conf \
  pmix-mca-params-override.conf prte-mca-params-override.conf; do
  if [[ -e "${PREFIX}/etc/${mpi_override}" ]]; then
    echo "error: MPI override parameter file must be absent: ${mpi_override}" >&2
    exit 1
  fi
done
export OMPI_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
export OMPI_MCA_mca_base_component_path="${PREFIX}/lib/openmpi"
export PMIX_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
export PMIX_MCA_mca_base_component_path="${PREFIX}/lib/pmix"
export PRTE_MCA_mca_base_param_files="${OPENMPI_QUALIFICATION_PARAMS}"
"${PREFIX}/bin/python" -c \
  'import mpi4py; print("mpi4py", mpi4py.__version__)' >/dev/null
HDF5_CONFIGURATION="$("${PREFIX}/bin/h5pcc" -showconfig)"
if [[ "${HDF5_CONFIGURATION}" != *"Parallel HDF5: yes"* ]]; then
  echo "error: the CUDA/MPI/Python build requires parallel HDF5" >&2
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

MAKE_JOBS="${MEEP_GPU_MAKE_JOBS:-$(nproc)}"
CUDA_ARCHITECTURES="${MEEP_GPU_CUDA_ARCHS:-AUTO}"
CONFIGURE_ARGS=(
  --enable-maintainer-mode
  --enable-shared
  --enable-single
  --enable-cuda
  "--with-cuda-arch=${CUDA_ARCHITECTURES}"
  --with-openmp
  --with-mpi
  --with-python
  --without-scheme
  "--prefix=${INSTALL_DIR}"
  --disable-cuda-fast-math
)

RECEIPT_BEGIN=(
  "${PREFIX_CONTROL_PYTHON[@]}"
  "${SCRIPT_DIR}/write-build-receipt.py" begin
  --repo "${REPO_ROOT}"
  --build-dir "${BUILD_DIR}"
  --build-kind cuda-mpi-python-fp32
  --builder "${BASH_SOURCE[0]}"
  --qualification-contract "${QUALIFICATION_CONTRACT}"
  --lockfile "environment_lock=${LOCK}"
)
for configure_argument in "${CONFIGURE_ARGS[@]}"; do
  RECEIPT_BEGIN+=("--configure-arg=${configure_argument}")
done
for receipt_environment in MEEP_GPU_MAKE_JOBS MEEP_GPU_FAST_MATH \
  MEEP_GPU_CUDA_ARCHS MEEP_GPU_DEVICE MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  MEEP_GPU_MPI_TRANSPORT CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER CC CXX FC F77 NVCC CUDAHOSTCXX MPICXX \
  OMPI_MCA_opal_cuda_support OMPI_MCA_mca_base_param_files \
  OMPI_MCA_mca_base_component_path PMIX_MCA_mca_base_param_files \
  PMIX_MCA_mca_base_component_path PRTE_MCA_mca_base_param_files \
  UCX_MEMTYPE_CACHE GPMEEP_FRESH_ENV_NONCE; do
  RECEIPT_BEGIN+=(--record-env "${receipt_environment}")
done
BUILD_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ).$$"
for existing_directory in "${BUILD_DIR}" "${INSTALL_DIR}"; do
  if [[ -e "${existing_directory}" ]]; then
    mv "${existing_directory}" "${existing_directory}.previous.${BUILD_TIMESTAMP}"
  fi
done
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
FRESH_ENV_ATTESTATION_RECEIPT="${BUILD_DIR}/cuda-mpi-${GPMEEP_FRESH_ENV_NONCE}.json"
/usr/bin/mv -- "${CLAIMED_FRESH_ENV_ATTESTATION}" \
  "${FRESH_ENV_ATTESTATION_RECEIPT}"
chmod 600 "${FRESH_ENV_ATTESTATION_RECEIPT}"
PREFIX_CONTENT_AUDIT="${BUILD_DIR}/conda-prefix-content-audit.json"
/usr/bin/mv -- "${PREFIX_CONTENT_AUDIT_TEMP}" "${PREFIX_CONTENT_AUDIT}"
PREFIX_PRE_NORMALIZATION_AUDIT="${BUILD_DIR}/conda-prefix-pre-normalization-audit.json"
/usr/bin/mv -- "${PREFIX_PRE_NORMALIZATION_AUDIT_TEMP}" \
  "${PREFIX_PRE_NORMALIZATION_AUDIT}"
PREFIX_BYTECODE_NORMALIZATION="${BUILD_DIR}/conda-generated-bytecode-normalization.json"
/usr/bin/mv -- "${PREFIX_BYTECODE_NORMALIZATION_TEMP}" \
  "${PREFIX_BYTECODE_NORMALIZATION}"
PREFIX_SOURCE_BYTECODE_NORMALIZATION="${BUILD_DIR}/conda-source-bytecode-normalization.json"
/usr/bin/mv -- "${PREFIX_SOURCE_BYTECODE_NORMALIZATION_TEMP}" \
  "${PREFIX_SOURCE_BYTECODE_NORMALIZATION}"
PREFIX_IMPORT_PROBE_ONE="${BUILD_DIR}/conda-source-bytecode-import-probe-one.json"
PREFIX_IMPORT_PROBE_TWO="${BUILD_DIR}/conda-source-bytecode-import-probe-two.json"
PREFIX_POST_IMPORT_AUDIT_ONE="${BUILD_DIR}/conda-prefix-post-import-audit-one.json"
PREFIX_POST_IMPORT_AUDIT_TWO="${BUILD_DIR}/conda-prefix-post-import-audit-two.json"
/usr/bin/mv -- "${PREFIX_IMPORT_PROBE_ONE_TEMP}" "${PREFIX_IMPORT_PROBE_ONE}"
/usr/bin/mv -- "${PREFIX_IMPORT_PROBE_TWO_TEMP}" "${PREFIX_IMPORT_PROBE_TWO}"
/usr/bin/mv -- "${PREFIX_POST_IMPORT_AUDIT_ONE_TEMP}" \
  "${PREFIX_POST_IMPORT_AUDIT_ONE}"
/usr/bin/mv -- "${PREFIX_POST_IMPORT_AUDIT_TWO_TEMP}" \
  "${PREFIX_POST_IMPORT_AUDIT_TWO}"
QUALIFICATION_HOME="${BUILD_DIR}/qualification-home"
mkdir -p "${QUALIFICATION_HOME}"
printf '%s\n' 'gpmeep isolated MPI qualification home' > \
  "${QUALIFICATION_HOME}/.gpmeep-empty-home"
QUALIFICATION_PYCACHE="${BUILD_DIR}/qualification-pycache"
if [[ -e "${QUALIFICATION_PYCACHE}" || -L "${QUALIFICATION_PYCACHE}" ]]; then
  echo "error: qualification pycache path was not fresh" >&2
  exit 1
fi
/usr/bin/install -d -m 700 "${QUALIFICATION_PYCACHE}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/dev/null
"${SYSTEM_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/capture-build-environment.py" \
  --output "${CANONICAL_BUILD_ENVIRONMENT_TEMP}"
CANONICAL_BUILD_ENVIRONMENT="${BUILD_DIR}/canonical-build-environment.json"
/usr/bin/mv -- "${CANONICAL_BUILD_ENVIRONMENT_TEMP}" \
  "${CANONICAL_BUILD_ENVIRONMENT}"
"${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
  --prefix "${PREFIX}" --explicit --sha256 > \
  "${BUILD_DIR}/environment-explicit.lock"

"${TIMEOUT}" --kill-after=30s 120s "${RECEIPT_BEGIN[@]}"

cd "${REPO_ROOT}"
"${TIMEOUT}" --kill-after=30s 600s \
  autoreconf --verbose --install --symlink --force
cd "${BUILD_DIR}"

NVCC="${PREFIX}/bin/nvcc" CUDAHOSTCXX="${CXX}" \
  MPICXX="${PREFIX}/bin/mpicxx" "${TIMEOUT}" --kill-after=30s 600s \
  "../../configure" "${CONFIGURE_ARGS[@]}"

"${TIMEOUT}" --kill-after=60s 3600s make -j"${MAKE_JOBS}"

# Build the tiny controller-only FD allocation shim with caller-owned result
# storage.  Its bytes are qualification-sentinel monitored and receipt-bound
# below; the evidence controller never compiles native code at runtime.
FD_ALLOCATION_SHIM_DIR="${BUILD_DIR}/controller-support"
FD_ALLOCATION_SHIM="${FD_ALLOCATION_SHIM_DIR}/libgpmeep-fd-allocation-shim.so.1"
/usr/bin/install -d -m 755 "${FD_ALLOCATION_SHIM_DIR}"
"${TIMEOUT}" --kill-after=30s 120s "${CC}" \
  -std=c11 -O2 -fPIC -fvisibility=hidden -Wall -Wextra -Werror \
  -shared -Wl,-z,relro -Wl,-z,now \
  -Wl,-soname,libgpmeep-fd-allocation-shim.so.1 \
  "${REPO_ROOT}/scripts/gpmeep-fd-allocation-shim.c" \
  -o "${FD_ALLOCATION_SHIM}"
/usr/bin/chmod 0555 "${FD_ALLOCATION_SHIM}"
if [[ ! -x "${FD_ALLOCATION_SHIM}" ]]; then
  echo "error: FD allocation shim was not built" >&2
  exit 1
fi
if [[ "${PYTHONPYCACHEPREFIX}" != /dev/null ]]; then
  echo "error: global build Python cache control is not /dev/null" >&2
  exit 1
fi
INSTALL_PYCACHE="${BUILD_DIR}/install-pycache"
if [[ -e "${INSTALL_PYCACHE}" || -L "${INSTALL_PYCACHE}" ]]; then
  echo "error: install-only pycache path was not fresh" >&2
  exit 1
fi
/usr/bin/install -d -m 700 "${INSTALL_PYCACHE}"
INSTALL_PYCACHE_IDENTITY="$(/usr/bin/stat -c '%d:%i' "${INSTALL_PYCACHE}")"
INSTALL_PYCACHE_PAYLOAD_ROOT="${INSTALL_PYCACHE}${INSTALL_DIR}"

# Automake's py-compile supplies an explicit cache path.  With the global
# /dev/null cache control this becomes /dev/null/<absolute path> and install
# fails.  Override the prefix for this one make process only; it is neither
# exported nor used by any qualification/runtime import.
/usr/bin/env PYTHONPYCACHEPREFIX="${INSTALL_PYCACHE}" \
  "${TIMEOUT}" --kill-after=60s 1800s make install
if [[ "${PYTHONPYCACHEPREFIX}" != /dev/null ]]; then
  echo "error: make install changed global Python cache control" >&2
  exit 1
fi

validate_install_pycache() {
  local actual_identity unexpected_entry pyc_sample
  if [[ ! -d "${INSTALL_PYCACHE}" || -L "${INSTALL_PYCACHE}" || \
        "$(/usr/bin/stat -c '%a' "${INSTALL_PYCACHE}")" != 700 ]]; then
    echo "error: install-only pycache root changed kind or mode" >&2
    return 1
  fi
  actual_identity="$(/usr/bin/stat -c '%d:%i' "${INSTALL_PYCACHE}")"
  if [[ "${actual_identity}" != "${INSTALL_PYCACHE_IDENTITY}" ]]; then
    echo "error: install-only pycache root identity changed" >&2
    return 1
  fi
  unexpected_entry="$(
    /usr/bin/find "${INSTALL_PYCACHE}" \
      \( -type l -o \( ! -type d ! -type f \) -o \
         \( -type f ! -path "${INSTALL_PYCACHE_PAYLOAD_ROOT}/*.pyc" \) \) \
      -print -quit
  )" || return 1
  if [[ -n "${unexpected_entry}" ]]; then
    echo "error: unexpected install-only pycache entry: ${unexpected_entry}" >&2
    return 1
  fi
  pyc_sample="$(
    /usr/bin/find "${INSTALL_PYCACHE_PAYLOAD_ROOT}" \
      -type f -name '*.pyc' -print -quit
  )" || return 1
  if [[ -z "${pyc_sample}" ]]; then
    echo "error: make install did not create isolated bytecode" >&2
    return 1
  fi
}

assert_no_installed_bytecode() {
  local adjacent_entry
  adjacent_entry="$(
    /usr/bin/find "${INSTALL_DIR}" \
      \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) \
      -print -quit
  )" || return 1
  if [[ -n "${adjacent_entry}" ]]; then
    echo "error: installed tree contains adjacent bytecode: ${adjacent_entry}" >&2
    return 1
  fi
}

validate_install_pycache
assert_no_installed_bytecode
INSTALL_PYCACHE_POLICY="${BUILD_DIR}/install-pycache-policy.txt"
{
  printf '%s\n' 'schema=gpmeep-install-pycache-policy-v1'
  printf 'path=%s\n' "${INSTALL_PYCACHE}"
  printf 'payload_root=%s\n' "${INSTALL_PYCACHE_PAYLOAD_ROOT}"
  printf 'device_inode=%s\n' "${INSTALL_PYCACHE_IDENTITY}"
  printf '%s\n' 'mode=0700'
  printf '%s\n' 'scope=make-install-only'
  printf '%s\n' 'global_before_after=/dev/null'
  printf '%s\n' 'installed_adjacent_bytecode=0'
  printf '%s\n' 'qualification_execution_exposure=none'
} >"${INSTALL_PYCACHE_POLICY}"
TEST_LOG_DIR="${BUILD_DIR}/qualification-logs"
mkdir -p "${TEST_LOG_DIR}"
CUDA_RUNTIME_BUILD_DIR="${BUILD_DIR}/cuda-runtime-qualification"
NVCC="${PREFIX}/bin/nvcc" CUDAHOSTCXX="${CXX}" \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/cmake" \
  -S "${REPO_ROOT}/cuda" -B "${CUDA_RUNTIME_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${PREFIX}/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX}" \
  -DMEEP_GPU_FAST_MATH=OFF \
  -DMEEP_GPU_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
"${TIMEOUT}" --kill-after=60s 1800s "${PREFIX}/bin/cmake" \
  --build "${CUDA_RUNTIME_BUILD_DIR}" --parallel "${MAKE_JOBS}"
# CUDA otherwise creates $HOME/.nv/ComputeCache even for a CPU-mode import of
# a CUDA-linked extension. Qualification does not benchmark JIT latency, so
# disable that cache and keep the receipt-bound HOME genuinely empty.
export CUDA_CACHE_DISABLE=1
# Fontconfig's conda configuration caches host fonts inside the otherwise
# exact prefix. Use only the lock-provided fonts and put the cache in the
# receipt-bound isolated build HOME during qualification. Runtime validation
# supplies a separate XDG cache root so the sealed build HOME stays immutable.
if [[ "${PREFIX}${BUILD_HOME}" == *['<>&']* ]]; then
  echo "error: qualification paths contain XML-sensitive characters" >&2
  exit 1
fi
QUALIFICATION_FONTCONFIG="${BUILD_DIR}/qualification-fontconfig.conf"
/usr/bin/mkdir -p "${BUILD_HOME}/cache/fontconfig"
{
  printf '%s\n' '<?xml version="1.0"?>'
  printf '%s\n' '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">'
  printf '%s\n' '<fontconfig>'
  printf '  <dir>%s/fonts</dir>\n' "${PREFIX}"
  printf '  <include ignore_missing="yes">%s/etc/fonts/conf.d</include>\n' "${PREFIX}"
  printf '%s\n' '  <cachedir prefix="xdg">fontconfig</cachedir>'
  printf '%s\n' '</fontconfig>'
} >"${QUALIFICATION_FONTCONFIG}"
export FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}"

# These three direct qualification executables are check_PROGRAMS rather than
# ordinary all/install targets.  Build them explicitly before invoking their
# libtool wrappers and before opening the immutable qualification epoch.
"${TIMEOUT}" --kill-after=30s 1800s \
  make -C tests -j"${MAKE_JOBS}" \
  gpu-backend gpu-step-db gpu-mpi-performance

# Libtool wrappers execute .libs/lt-* ELFs against the build-tree libmeep.
# Automake's check harness may execute a check-program ELF without invoking
# its libtool wrapper, while gpu-mpi-performance is check-built but is not in
# TESTS. Run one explicitly non-authoritative, short singleton smoke through
# each wrapper solely to materialize a current relinked lt-* ELF. Every
# receipt-critical qualification then executes the actual ELF directly; both
# it and the loaded build library are hashed as receipt artifacts/provenance.
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s \
  "${BUILD_DIR}/tests/gpu-backend" \
  >"${TEST_LOG_DIR}/gpu-backend-libtool-materialization.log" 2>&1
printf '%s\n' \
  'gpmeep-libtool-materialization:gpu-backend-libtool-materialization.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-backend-libtool-materialization.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cpu \
  MEEP_GPU_TEST_PREFLIGHT_ONLY=1 MEEP_GPU_TEST_EXPECT_CPU=1 \
  "${TIMEOUT}" --kill-after=10s 120s \
  "${BUILD_DIR}/tests/gpu-step-db" \
  >"${TEST_LOG_DIR}/gpu-step-db-libtool-materialization.log" 2>&1
printf '%s\n' \
  'gpmeep-libtool-materialization:gpu-step-db-libtool-materialization.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-libtool-materialization.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=1 \
  MEEP_GPU_MULTI_STEPS=1 \
  "${TIMEOUT}" --kill-after=10s 120s \
  "${BUILD_DIR}/tests/gpu-mpi-performance" \
  >"${TEST_LOG_DIR}/gpu-mpi-performance-libtool-materialization.log" 2>&1
printf '%s\n' \
  'gpmeep-libtool-materialization:gpu-mpi-performance-libtool-materialization.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-mpi-performance-libtool-materialization.log"

GPU_MPI_PERFORMANCE_ELF="${BUILD_DIR}/tests/.libs/lt-gpu-mpi-performance"
GPU_STEP_DB_ELF="${BUILD_DIR}/tests/.libs/lt-gpu-step-db"
GPU_BACKEND_ELF="${BUILD_DIR}/tests/.libs/lt-gpu-backend"
CUDA_ARCHITECTURE_TEST_ELF="${CUDA_RUNTIME_BUILD_DIR}/meep_cuda_architecture_test"
CUDA_FORMULA_TEST_ELF="${CUDA_RUNTIME_BUILD_DIR}/meep_cuda_formula_test"
CUDA_RUNTIME_VALIDATION_ELF="${CUDA_RUNTIME_BUILD_DIR}/meep_cuda_runtime_validation_test"
CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF="${CUDA_RUNTIME_BUILD_DIR}/meep_cuda_near2far_runtime_validation_test"
CUDA_SMOKE_ELF="${CUDA_RUNTIME_BUILD_DIR}/meep_cuda_smoke"
for executable in "${GPU_MPI_PERFORMANCE_ELF}" "${GPU_STEP_DB_ELF}" \
                  "${GPU_BACKEND_ELF}" "${CUDA_ARCHITECTURE_TEST_ELF}" \
                  "${CUDA_FORMULA_TEST_ELF}" \
                  "${CUDA_RUNTIME_VALIDATION_ELF}" \
                  "${CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF}" \
                  "${CUDA_SMOKE_ELF}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "error: libtool qualification ELF is missing: ${executable}" >&2
    exit 1
  fi
done
record_direct_elf_provenance() {
  local name="$1"
  local executable="$2"
  local log="${TEST_LOG_DIR}/${name}-direct-elf-provenance.log"
  local loaded_libmeep
  loaded_libmeep="$(/usr/bin/ldd "${executable}" | \
    /usr/bin/awk '$1 ~ /^libmeep\.so/ {print $3; exit}')"
  if [[ -z "${loaded_libmeep}" || ! -f "${loaded_libmeep}" ]]; then
    echo "error: direct qualification ELF does not resolve libmeep" >&2
    exit 1
  fi
  loaded_libmeep="$(readlink -f "${loaded_libmeep}")"
  case "${loaded_libmeep}" in
    "${BUILD_DIR}/src/.libs/"*) ;;
    *)
      echo "error: direct qualification ELF resolves unexpected libmeep: ${loaded_libmeep}" >&2
      exit 1
      ;;
  esac
  {
    printf 'executable=%s\n' "$(readlink -f "${executable}")"
    printf 'loaded_libmeep=%s\n' "${loaded_libmeep}"
    /usr/bin/sha256sum "${executable}" "${loaded_libmeep}"
    /usr/bin/ldd "${executable}"
    printf 'gpmeep-qualification:%s-direct-elf-provenance.log:PASS\n' "${name}"
  } >"${log}"
}
PYTHON_ABI="$("${PREFIX}/bin/python" -c \
  'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
INSTALLED_PYTHON="${INSTALL_DIR}/lib/${PYTHON_ABI}/site-packages"
BUILD_EXTENSION="$(readlink -f "${BUILD_DIR}/python/meep/_meep.so")"
BUILD_LIBMEEP="$(readlink -f "${BUILD_DIR}/src/.libs/libmeep.so")"
INSTALLED_EXTENSION="$(readlink -f "${INSTALLED_PYTHON}/meep/_meep.so")"
INSTALLED_LIBMEEP="$(readlink -f "${INSTALL_DIR}/lib/libmeep.so")"
INSTALLED_MPB_EXTENSION="$(readlink -f "${INSTALLED_PYTHON}/meep/mpb/_mpb.so")"
INSTALLED_LIBPYMPB="$(readlink -f "${INSTALL_DIR}/lib/libpympb.so")"
QUALIFICATION_IDENTITY_BEFORE="${BUILD_DIR}/qualification-identity-before.json"
QUALIFICATION_IDENTITY_AFTER="${BUILD_DIR}/qualification-identity-after.json"
QUALIFICATION_CONTRACT_V2="${BUILD_DIR}/qualification-contract-v2.json"
QUALIFICATION_IDENTITY_ARGS=(
  --immutable-empty-directory "qualification_pycache=${QUALIFICATION_PYCACHE}"
  --artifact "python_extension=${BUILD_EXTENSION}"
  --artifact "libmeep=${BUILD_LIBMEEP}"
  --artifact "installed_python_extension=${INSTALLED_EXTENSION}"
  --artifact "installed_libmeep=${INSTALLED_LIBMEEP}"
  --artifact "installed_mpb_extension=${INSTALLED_MPB_EXTENSION}"
  --artifact "installed_libpympb=${INSTALLED_LIBPYMPB}"
  --artifact "fd_allocation_shim=${FD_ALLOCATION_SHIM}"
  --artifact "gpu_backend_test=${GPU_BACKEND_ELF}"
  --artifact "gpu_step_db_test=${GPU_STEP_DB_ELF}"
  --artifact "gpu_mpi_performance=${GPU_MPI_PERFORMANCE_ELF}"
  --artifact "cuda_architecture_test=${CUDA_ARCHITECTURE_TEST_ELF}"
  --artifact "cuda_formula_test=${CUDA_FORMULA_TEST_ELF}"
  --artifact "cuda_runtime_validation=${CUDA_RUNTIME_VALIDATION_ELF}"
  --artifact "cuda_near2far_runtime_validation=${CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF}"
  --artifact "cuda_smoke=${CUDA_SMOKE_ELF}"
)
"${PREFIX_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" snapshot \
  --repo "${REPO_ROOT}" --prefix "${PREFIX}" \
  --installed-prefix "${INSTALL_DIR}" \
  --output "${QUALIFICATION_IDENTITY_BEFORE}" \
  "${QUALIFICATION_IDENTITY_ARGS[@]}"
QUALIFICATION_SENTINEL_READY="${BUILD_DIR}/qualification-mutation-sentinel.ready"
QUALIFICATION_SENTINEL_LOG="${TEST_LOG_DIR}/qualification-mutation-sentinel.log"
QUALIFICATION_SENTINEL_PID=""
stop_qualification_sentinel_on_exit() {
  if [[ -n "${QUALIFICATION_SENTINEL_PID}" ]] && \
     /usr/bin/kill -0 "${QUALIFICATION_SENTINEL_PID}" 2>/dev/null; then
    /usr/bin/kill -TERM "${QUALIFICATION_SENTINEL_PID}" 2>/dev/null || true
    wait "${QUALIFICATION_SENTINEL_PID}" 2>/dev/null || true
  fi
}
trap stop_qualification_sentinel_on_exit EXIT
"${PREFIX_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" sentinel \
  --identity "${QUALIFICATION_IDENTITY_BEFORE}" \
  --repo "${REPO_ROOT}" --prefix "${PREFIX}" \
  --ready "${QUALIFICATION_SENTINEL_READY}" \
  >"${QUALIFICATION_SENTINEL_LOG}" 2>&1 &
QUALIFICATION_SENTINEL_PID=$!
for ((_sentinel_wait = 0; _sentinel_wait < 6000; _sentinel_wait++)); do
  if [[ -f "${QUALIFICATION_SENTINEL_READY}" ]]; then
    break
  fi
  if ! /usr/bin/kill -0 "${QUALIFICATION_SENTINEL_PID}" 2>/dev/null; then
    wait "${QUALIFICATION_SENTINEL_PID}" || true
    echo "error: qualification mutation sentinel failed during startup" >&2
    /usr/bin/cat "${QUALIFICATION_SENTINEL_LOG}" >&2
    exit 1
  fi
  /usr/bin/sleep 0.1
done
if [[ ! -f "${QUALIFICATION_SENTINEL_READY}" ]]; then
  echo "error: qualification mutation sentinel startup timed out" >&2
  exit 1
fi
export PYTHONPYCACHEPREFIX="${QUALIFICATION_PYCACHE}"

record_direct_elf_provenance gpu-mpi-performance "${GPU_MPI_PERFORMANCE_ELF}"
record_direct_elf_provenance gpu-step-db "${GPU_STEP_DB_ELF}"
record_direct_elf_provenance gpu-backend "${GPU_BACKEND_ELF}"

env HOME="${QUALIFICATION_HOME}" \
  "${TIMEOUT}" --kill-after=10s 120s "${CUDA_FORMULA_TEST_ELF}" \
  >"${TEST_LOG_DIR}/cuda-host-formula.log" 2>&1
printf '%s\n' 'gpmeep-qualification:cuda-host-formula.log:PASS' >> \
  "${TEST_LOG_DIR}/cuda-host-formula.log"
env HOME="${QUALIFICATION_HOME}" \
  "${TIMEOUT}" --kill-after=10s 120s "${CUDA_ARCHITECTURE_TEST_ELF}" \
  >"${TEST_LOG_DIR}/cuda-architecture-metadata.log" 2>&1
printf '%s\n' 'gpmeep-qualification:cuda-architecture-metadata.log:PASS' >> \
  "${TEST_LOG_DIR}/cuda-architecture-metadata.log"
env HOME="${QUALIFICATION_HOME}" \
  "${TIMEOUT}" --kill-after=10s 120s "${CUDA_RUNTIME_VALIDATION_ELF}" \
  >"${TEST_LOG_DIR}/cuda-runtime-validation.log" 2>&1
printf '%s\n' 'gpmeep-qualification:cuda-runtime-validation.log:PASS' >> \
  "${TEST_LOG_DIR}/cuda-runtime-validation.log"
env HOME="${QUALIFICATION_HOME}" \
  "${TIMEOUT}" --kill-after=10s 120s \
  "${CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF}" \
  >"${TEST_LOG_DIR}/cuda-near2far-runtime-validation.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:cuda-near2far-runtime-validation.log:PASS' >> \
  "${TEST_LOG_DIR}/cuda-near2far-runtime-validation.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_DEVICE="${MEEP_GPU_DEVICE:-0}" \
  "${TIMEOUT}" --kill-after=10s 240s "${CUDA_SMOKE_ELF}" \
  >"${TEST_LOG_DIR}/cuda-smoke.log" 2>&1
printf '%s\n' 'gpmeep-qualification:cuda-smoke.log:PASS' >> \
  "${TEST_LOG_DIR}/cuda-smoke.log"

env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" attest-python-runtime \
  --expected-extension "${BUILD_EXTENSION}" \
  --expected-libmeep "${BUILD_LIBMEEP}" \
  --log-name in-place-python-runtime-provenance.log \
  >"${TEST_LOG_DIR}/in-place-python-runtime-provenance.log" 2>&1
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" attest-python-runtime \
  --expected-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  --log-name installed-python-runtime-provenance.log \
  >"${TEST_LOG_DIR}/installed-python-runtime-provenance.log" 2>&1

env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.with_mpi() and mp.gpu.compiled; print(mp.__version__)' \
  >"${TEST_LOG_DIR}/in-place-singleton-import.log" 2>&1
printf '%s\n' 'gpmeep-qualification:in-place-singleton-import.log:PASS' >> \
  "${TEST_LOG_DIR}/in-place-singleton-import.log"
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.count_processors() == 2 and mp.gpu.compiled' \
  >"${TEST_LOG_DIR}/in-place-two-rank-import.log" 2>&1
printf '%s\n' 'gpmeep-qualification:in-place-two-rank-import.log:PASS' >> \
  "${TEST_LOG_DIR}/in-place-two-rank-import.log"

# The C++ MPI suite is the broad distributed build qualification. Generic
# Python unit tests are not all rank-safe (several intentionally use local
# callbacks/files); run them in the separate broad validation matrix. The
# focused tests below cover the Python binding, physical device mapping, and
# the rank-local source/DFT correctness defect fixed in M8.7.
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=30s 1800s make -C tests check \
  >"${TEST_LOG_DIR}/cpp-mpi-test-suite.log" 2>&1
printf '%s\n' 'gpmeep-qualification:cpp-mpi-test-suite.log:PASS' >> \
  "${TEST_LOG_DIR}/cpp-mpi-test-suite.log"

env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-full-one-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:gpu-step-db-full-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-full-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PHASE_SOURCE_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-phase-source-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-phase-source-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-phase-source-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_LDOS_REDUCTION_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 240s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-ldos-deterministic-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-ldos-deterministic-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-ldos-deterministic-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_LDOS_MIGRATION_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 240s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-ldos-migration-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-ldos-migration-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-ldos-migration-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_TEST_CW_VECTOR_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-cw-vector-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-cw-vector-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-cw-vector-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_TEST_CW_SOLVER_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-cw-solver-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-cw-solver-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-cw-solver-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_TEST_CW_BREAKDOWN_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-cw-breakdown-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-cw-breakdown-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-cw-breakdown-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_TEST_CW_BREAKDOWN_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-cw-breakdown-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-cw-breakdown-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-cw-breakdown-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_TEST_BOUNDARY_LIFETIME_LEAK_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/boundary-lifetime-double-failure-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-lifetime-double-failure-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-lifetime-double-failure-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-full-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:gpu-step-db-full-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-full-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_NEAR2FAR_2D_MPI_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-near2far2d-two-gpu.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-near2far2d-two-gpu.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-near2far2d-two-gpu.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_NEAR2FAR_CYL_MPI_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 900s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-near2farcyl-two-gpu.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-near2farcyl-two-gpu.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-near2farcyl-two-gpu.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PHASE_SOURCE_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-phase-source-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-phase-source-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-phase-source-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_LDOS_REDUCTION_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 240s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-ldos-deterministic-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-ldos-deterministic-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-ldos-deterministic-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-full-pinned-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-full-pinned-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-full-pinned-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-full-waitall-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-full-waitall-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-full-waitall-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cuda-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cuda-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cuda-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_CPU=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cpu-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cpu-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cpu-one-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_PINNED_TRANSPORT=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cuda-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cuda-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cuda-two-rank.log"
env HOME="${QUALIFICATION_HOME}" CUDA_VISIBLE_DEVICES=0,1 \
  MEEP_GPU_BACKEND=auto MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_MPI_COMPLETION=waitsome \
  MEEP_GPU_TEST_DYNAMIC_CLAIM_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-dynamic-claim-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-dynamic-claim-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-dynamic-claim-two-rank.log"
env HOME="${QUALIFICATION_HOME}" CUDA_VISIBLE_DEVICES=0 \
  MEEP_GPU_BACKEND=auto MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_ALLOW_OVERSUBSCRIBE=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_PINNED_TRANSPORT=1 \
  MEEP_GPU_TEST_EXPECT_POSITIVE_OVERSUBSCRIPTION=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/automatic-explicit-oversubscription-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:automatic-explicit-oversubscription-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/automatic-explicit-oversubscription-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_CUDA_AWARE_TRANSPORT=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cuda-aware-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cuda-aware-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cuda-aware-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_AUTO_MIN_CELLS=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_CUDA_AWARE_TRANSPORT=1 \
  MEEP_GPU_TEST_EXPECT_WAITALL_COMPLETION=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cuda-aware-waitall-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cuda-aware-waitall-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cuda-aware-waitall-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=auto \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_EXPECT_CPU=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/gpu-step-db-auto-cpu-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:gpu-step-db-auto-cpu-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/gpu-step-db-auto-cpu-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_TEST_DFT_PHASE_SHARING_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/dft-phase-sharing-exact-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:dft-phase-sharing-exact-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/dft-phase-sharing-exact-two-rank.log"

# The generic material suite catches numerical mismatches, but these focused
# gates bind the FP32 ADE representation contract to the production
# fields_chunk dispatch.  The singleton covers a million-step Drude neutral
# mode plus seeded noisy Lorentz/Drude semantics; the distributed lanes force
# the mixed anisotropic Lorentz+Drude material across both supported MPI
# transports and two distinct physical GPUs.
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_TEST_ADE_STATE_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 1 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/ade-state-contract-one-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:ade-state-contract-one-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/ade-state-contract-one-rank.log"
env HOME="${QUALIFICATION_HOME}" CUDA_VISIBLE_DEVICES=0,1 \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_TEST_MIXED_DISPERSIVE_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/mixed-dispersive-cuda-aware-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:mixed-dispersive-cuda-aware-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/mixed-dispersive-cuda-aware-two-rank.log"
env HOME="${QUALIFICATION_HOME}" CUDA_VISIBLE_DEVICES=0,1 \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned \
  MEEP_GPU_TEST_MIXED_DISPERSIVE_ONLY=1 \
  "${TIMEOUT}" --kill-after=30s 600s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/mixed-dispersive-pinned-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:mixed-dispersive-pinned-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/mixed-dispersive-pinned-two-rank.log"

env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  CUDA_VISIBLE_DEVICES=0 GPMEEP_REQUIRE_CUDA_TEST=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/python" \
  "${REPO_ROOT}/python/tests/test_gpu_backend.py" \
  >"${TEST_LOG_DIR}/python-gpu-backend-singleton.log" 2>&1
printf '%s\n' 'gpmeep-qualification:python-gpu-backend-singleton.log:PASS' >> \
  "${TEST_LOG_DIR}/python-gpu-backend-singleton.log"
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${BUILD_DIR}/python" MEEP_GPU_BACKEND=cpu \
  CUDA_VISIBLE_DEVICES=0,1 GPMEEP_REQUIRE_CUDA_TEST=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${PREFIX}/bin/python" "${REPO_ROOT}/python/tests/test_gpu_backend.py" \
  >"${TEST_LOG_DIR}/python-gpu-backend-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:python-gpu-backend-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/python-gpu-backend-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_PIXELS=16 \
  MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_TEST_COMMS_MANAGER_FAILURE_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/comms-manager-eager-failure-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:comms-manager-eager-failure-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/comms-manager-eager-failure-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_TEST_COMMS_MANAGER_FAILURE_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/comms-manager-waitall-failure-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:comms-manager-waitall-failure-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/comms-manager-waitall-failure-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_MULTI_PIXELS=16 \
  MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_TEST_COMMS_MANAGER_FAILURE_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/comms-manager-pinned-failure-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:comms-manager-pinned-failure-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/comms-manager-pinned-failure-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH=1 \
  MEEP_GPU_EXPECT_EAGER_MPI=1 MEEP_GPU_EXPECT_RECEIVE_PINGPONG=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_MIXED=1 \
  MEEP_GPU_EXPECT_DFT_PHASE_SHARING=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/eager-pingpong-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:eager-pingpong-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/eager-pingpong-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH=1 \
  MEEP_GPU_EXPECT_EAGER_MPI=1 MEEP_GPU_EXPECT_RECEIVE_PINGPONG=1 \
  MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_DFT_PHASE_SHARING=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-mixed-disabled-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-mixed-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-mixed-disabled-two-rank.log"
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/compare-boundary-eh-overlap-observables.py" \
  --profile mixed \
  "${TEST_LOG_DIR}/eager-pingpong-two-rank.log" \
  "${TEST_LOG_DIR}/boundary-eh-overlap-mixed-disabled-two-rank.log" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-mixed-exact-comparison.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-mixed-exact-comparison.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-mixed-exact-comparison.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH=1 \
  MEEP_GPU_EXPECT_EAGER_MPI=1 MEEP_GPU_EXPECT_RECEIVE_PINGPONG=1 \
  MEEP_GPU_EXPECT_DFT_PHASE_SHARING=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/eager-pingpong-waitall-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:eager-pingpong-waitall-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/eager-pingpong-waitall-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_COLD_WARMUP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-default-off-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-default-off-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-default-off-two-rank.log"
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/compare-boundary-eh-overlap-observables.py" \
  "${TEST_LOG_DIR}/boundary-eh-overlap-two-rank.log" \
  "${TEST_LOG_DIR}/boundary-eh-overlap-default-off-two-rank.log" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-exact-comparison.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-exact-comparison.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-exact-comparison.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=0 \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-zero-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-opt-in-zero-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-zero-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP= \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-empty-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-opt-in-empty-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-empty-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=invalid \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-invalid-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-opt-in-invalid-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-invalid-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-conflict-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-opt-in-conflict-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-opt-in-conflict-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_UNSUPPORTED_SCHEDULE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-pinned-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-pinned-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-pinned-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_DISABLE_EAGER_MPI=1 \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_UNSUPPORTED_SCHEDULE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-eager-disabled-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-eager-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-eager-disabled-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-waitall-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-waitall-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-waitall-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MPI_COMPLETION=waitall \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-waitall-default-off-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-waitall-default-off-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-waitall-default-off-two-rank.log"
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/compare-boundary-eh-overlap-observables.py" \
  "${TEST_LOG_DIR}/boundary-eh-overlap-waitall-two-rank.log" \
  "${TEST_LOG_DIR}/boundary-eh-overlap-waitall-default-off-two-rank.log" \
  >"${TEST_LOG_DIR}/boundary-eh-overlap-waitall-exact-comparison.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:boundary-eh-overlap-waitall-exact-comparison.log:PASS' >> \
  "${TEST_LOG_DIR}/boundary-eh-overlap-waitall-exact-comparison.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_OVERLAP_MATERIAL=1 MEEP_GPU_MULTI_DISABLE_SOURCE=1 \
  MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-default-off-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-default-off-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-default-off-two-rank.log"
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/compare-halo-curl-overlap-observables.py" \
  "${TEST_LOG_DIR}/halo-curl-overlap-two-rank.log" \
  "${TEST_LOG_DIR}/halo-curl-overlap-default-off-two-rank.log" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-exact-comparison.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-exact-comparison.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-exact-comparison.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_DISABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-opt-in-conflict-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-opt-in-conflict-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-opt-in-conflict-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_FEATURE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-tiled-rejected-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-tiled-rejected-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-tiled-rejected-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_UNSUPPORTED_SCHEDULE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-pinned-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-pinned-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-pinned-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=0 \
  MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-opt-in-zero-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-opt-in-zero-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-opt-in-zero-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP= \
  MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-opt-in-empty-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-opt-in-empty-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-opt-in-empty-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=invalid \
  MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-opt-in-invalid-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-opt-in-invalid-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-opt-in-invalid-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_DISABLE_EAGER_MPI=1 \
  MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_UNSUPPORTED_SCHEDULE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-eager-disabled-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-eager-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-eager-disabled-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=18446744073709551615 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_SMALL=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-small-rejected-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-small-rejected-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-small-rejected-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=0 \
  MEEP_GPU_MULTI_BFAST=1 MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS=0 \
  MEEP_GPU_ENABLE_HALO_CURL_OVERLAP=1 \
  MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_FEATURE=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=4 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/halo-curl-overlap-bfast-rejected-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:halo-curl-overlap-bfast-rejected-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/halo-curl-overlap-bfast-rejected-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=128 \
  MEEP_GPU_EXPECT_TILE_COALESCING=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/tile-coalescing-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:tile-coalescing-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/tile-coalescing-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_LOOP_TILE_BASE_DB=128 \
  MEEP_GPU_DISABLE_TILE_COALESCING=1 \
  MEEP_GPU_EXPECT_NO_TILE_COALESCING=1 \
  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \
  MEEP_GPU_MULTI_STEPS=10 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/tile-coalescing-disabled-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:tile-coalescing-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/tile-coalescing-disabled-two-rank.log"
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/compare-tile-coalescing-observables.py" \
  --profile mixed \
  "${TEST_LOG_DIR}/tile-coalescing-two-rank.log" \
  "${TEST_LOG_DIR}/tile-coalescing-disabled-two-rank.log" \
  >"${TEST_LOG_DIR}/tile-coalescing-exact-comparison.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:tile-coalescing-exact-comparison.log:PASS' >> \
  "${TEST_LOG_DIR}/tile-coalescing-exact-comparison.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=pinned \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_EXPECT_NO_RECEIVE_SECONDARY_ALLOCATION=1 \
  MEEP_GPU_EXPECT_NO_EAGER_MPI=1 \
  MEEP_GPU_EXPECT_DFT_PHASE_SHARING=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/pinned-boundary-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:pinned-boundary-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/pinned-boundary-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_DISABLE_RECEIVE_PINGPONG=1 \
  MEEP_GPU_EXPECT_NO_RECEIVE_SECONDARY_ALLOCATION=1 \
  MEEP_GPU_EXPECT_EAGER_MPI=1 \
  MEEP_GPU_EXPECT_DFT_PHASE_SHARING=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/pingpong-disabled-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:pingpong-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/pingpong-disabled-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware \
  MEEP_GPU_MULTI_PIXELS="${DFT_PHASE_SHARING_QUALIFICATION_PIXELS}" \
  MEEP_GPU_MULTI_WARMUP_STEPS=4 MEEP_GPU_MULTI_STEPS=10 \
  MEEP_GPU_DISABLE_DFT_PHASE_SHARING=1 \
  MEEP_GPU_EXPECT_NO_DFT_PHASE_SHARING=1 \
  MEEP_GPU_EXPECT_EAGER_MPI=1 MEEP_GPU_EXPECT_RECEIVE_PINGPONG=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" \
  >"${TEST_LOG_DIR}/dft-phase-sharing-disabled-two-rank.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:dft-phase-sharing-disabled-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/dft-phase-sharing-disabled-two-rank.log"
BOUNDARY_GRAPH_LIFECYCLE_LOG="${TEST_LOG_DIR}/boundary-graph-lifecycle-two-rank.log"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
  MEEP_GPU_MPI_TRANSPORT=cuda-aware MEEP_GPU_MULTI_PIXELS=16 \
  MEEP_GPU_MULTI_WARMUP_STEPS=2 MEEP_GPU_MULTI_STEPS=2 \
  MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_MPI_PERFORMANCE_ELF}" >"${BOUNDARY_GRAPH_LIFECYCLE_LOG}" 2>&1
"${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/validate-multi-gpu-initialization-markers.py" \
  --log "${BOUNDARY_GRAPH_LIFECYCLE_LOG}" \
  --expected-profile trigonometric-v1 \
  --expected-source-profile single-ez-v1 \
  --expected-applications-per-rank 2 \
  --expected-ranks 2 \
  --expected-pixels 16 \
  --expected-warmup-steps 2 \
  --expected-steps 2 \
  --expected-transport cuda-aware
printf '%s\n' \
  'gpmeep-qualification:boundary-graph-lifecycle-two-rank.log:PASS' >> \
  "${BOUNDARY_GRAPH_LIFECYCLE_LOG}"
env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cpu MEEP_GPU_TEST_DFT_DECIMATION_ONLY=1 \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 \
  "${GPU_STEP_DB_ELF}" \
  >"${TEST_LOG_DIR}/dft-decimation-two-rank.log" 2>&1
printf '%s\n' 'gpmeep-qualification:dft-decimation-two-rank.log:PASS' >> \
  "${TEST_LOG_DIR}/dft-decimation-two-rank.log"

run_expected_mpi_failure() {
  local name="$1"
  local injection_variable="$2"
  local required_diagnostic="$3"
  local executable="${4:-${GPU_STEP_DB_ELF}}"
  local transport="${5:-auto}"
  local completion="${6:-waitsome}"
  local log="${TEST_LOG_DIR}/${name}.log"
  local status
  set +e
  env HOME="${QUALIFICATION_HOME}" MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
    MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 \
    MEEP_GPU_MPI_TRANSPORT="${transport}" \
    MEEP_GPU_MPI_COMPLETION="${completion}" \
    MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=2 \
    "${injection_variable}=1" \
    "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
    "${executable}" >"${log}" 2>&1
  status=$?
  set -e
  if ! expected_failure_status_is_acceptable "${status}"; then
    echo "error: ${name} did not fail promptly through MPI" >&2
    exit 1
  fi
  if ! /usr/bin/grep -aFq -- "${required_diagnostic}" "${log}"; then
    echo "error: ${name} failed for the wrong reason" >&2
    exit 1
  fi
  # Open MPI may terminate its diagnostic stream with a NUL rather than a
  # newline. Keep those raw bytes and force the machine-readable attestation
  # to begin on its own text line.
  printf '\n' >>"${log}"
  printf '%s\n' \
    "gpmeep-expected-mpi-failure:${name}.log:status=${status}:diagnostic=${required_diagnostic}" \
    >>"${log}"
  printf '%s\n' "gpmeep-qualification:${name}.log:PASS" >>"${log}"
}

run_expected_preflight_failure() {
  local name="$1"
  local ranks="$2"
  local required_diagnostic="$3"
  shift 3
  local log="${TEST_LOG_DIR}/${name}.log"
  local status
  set +e
  env HOME="${QUALIFICATION_HOME}" \
    MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 "$@" \
    "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" \
    -n "${ranks}" "${GPU_STEP_DB_ELF}" >"${log}" 2>&1
  status=$?
  set -e
  if ! expected_failure_status_is_acceptable "${status}"; then
    echo "error: ${name} did not fail promptly through preflight" >&2
    exit 1
  fi
  if ! /usr/bin/grep -aFq -- "${required_diagnostic}" "${log}"; then
    echo "error: ${name} failed for the wrong reason" >&2
    exit 1
  fi
  printf '\n' >>"${log}"
  printf '%s\n' \
    "gpmeep-expected-preflight-failure:${name}.log:status=${status}:diagnostic=${required_diagnostic}" \
    >>"${log}"
  printf '%s\n' "gpmeep-qualification:${name}.log:PASS" >>"${log}"
}

run_expected_mpi_failure \
  comms-manager-destructor-unwind-two-rank \
  MEEP_GPU_TEST_COMMS_MANAGER_DESTRUCTOR_ABORT_ONLY \
  "communications manager destroyed during exceptional MPI boundary exchange" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  comms-manager-completion-policy-mismatch-two-rank \
  MEEP_GPU_TEST_COMPLETION_POLICY_MISMATCH \
  "distributed CUDA ranks selected incompatible MPI completion policies" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware waitsome
run_expected_mpi_failure \
  comms-manager-invalid-completion-policy-two-rank \
  MEEP_GPU_TEST_COMPLETION_POLICY_PREFLIGHT_ONLY \
  "invalid MEEP_GPU_MPI_COMPLETION='invalid'" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware invalid
run_expected_mpi_failure \
  initial-condition-profile-mismatch-two-rank \
  MEEP_GPU_TEST_INITIAL_CONDITION_RANK_MISMATCH \
  "fixed workload profile differs across MPI ranks before allocation" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  initial-condition-invalid-rank-two-rank \
  MEEP_GPU_TEST_INITIAL_CONDITION_RANK_INVALID \
  "invalid MEEP_GPU_MULTI_INITIAL_CONDITION='invalid-rank-profile'" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  initial-condition-preflight-mismatch-two-rank \
  MEEP_GPU_TEST_INITIAL_CONDITION_PREFLIGHT_RANK_MISMATCH \
  "fixed workload profile differs across MPI ranks before allocation" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  source-profile-mismatch-two-rank \
  MEEP_GPU_TEST_SOURCE_PROFILE_RANK_MISMATCH \
  "fixed workload profile differs across MPI ranks before allocation" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  source-profile-invalid-rank-two-rank \
  MEEP_GPU_TEST_SOURCE_PROFILE_RANK_INVALID \
  "invalid MEEP_GPU_MULTI_SOURCE_PROFILE='invalid-rank-profile'" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  source-enabled-mismatch-two-rank \
  MEEP_GPU_TEST_SOURCE_ENABLED_RANK_MISMATCH \
  "fixed workload profile differs across MPI ranks before allocation" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  boundary-lifecycle-profile-mismatch-two-rank \
  MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE_RANK_MISMATCH \
  "fixed workload profile differs across MPI ranks before allocation" \
  "${GPU_MPI_PERFORMANCE_ELF}" cuda-aware
run_expected_mpi_failure \
  dft-norm-rank-failure-two-rank MEEP_GPU_TEST_DFT_NORM_RANK_FAILURE \
  "rank-local failure during a distributed DFT norm reduction"
run_expected_mpi_failure \
  integrate2-rank-failure-two-rank MEEP_GPU_TEST_INTEGRATE2_RANK_FAILURE \
  "rank-local failure during distributed integrate2"
run_expected_mpi_failure \
  step-finite-rank-failure-two-rank MEEP_GPU_TEST_STEP_FINITE_ABORT \
  "simulation fields are NaN or Inf"
STEP_FINITE_FAILURE_LOG="${TEST_LOG_DIR}/step-finite-rank-failure-two-rank.log"
for finite_rank in 0 1; do
  if [[ "$(/usr/bin/grep -aFc -- \
            "rank=${finite_rank},stage=before-failing-step" \
            "${STEP_FINITE_FAILURE_LOG}")" -ne 1 ]]; then
    echo "error: rank ${finite_rank} did not enter the finite-failure fields::step exactly once" >&2
    exit 1
  fi
done
if [[ "$(/usr/bin/grep -aFc -- \
          'meep: simulation fields are NaN or Inf' \
          "${STEP_FINITE_FAILURE_LOG}")" -ne 2 ]]; then
  echo "error: finite failure was not diagnosed exactly once on every rank" >&2
  exit 1
fi
if /usr/bin/grep -aFq -- \
    'FAIL: fields::step returned after rank-local device NaN' \
    "${STEP_FINITE_FAILURE_LOG}"; then
  echo "error: a rank returned successfully from the finite-failure step" >&2
  exit 1
fi
run_expected_preflight_failure \
  automatic-backend-request-mismatch-two-rank 2 \
  "distributed Meep backend requests differ across MPI ranks" \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_MPI_TRANSPORT=auto MEEP_GPU_MPI_COMPLETION=waitsome \
  MEEP_GPU_TEST_PREFLIGHT_ONLY=1 MEEP_GPU_TEST_RANK_BACKEND_MISMATCH=1
run_expected_preflight_failure \
  automatic-threshold-mismatch-two-rank 2 \
  "MEEP_GPU_AUTO_MIN_CELLS differs across MPI ranks" \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_MPI_TRANSPORT=auto MEEP_GPU_MPI_COMPLETION=waitsome \
  MEEP_GPU_TEST_PREFLIGHT_ONLY=1 \
  MEEP_GPU_TEST_RANK_AUTO_THRESHOLD_MISMATCH=1
run_expected_preflight_failure \
  automatic-invalid-device-two-rank 2 \
  "invalid MEEP_GPU_DEVICE='invalid'" \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_MPI_TRANSPORT=auto MEEP_GPU_MPI_COMPLETION=waitsome \
  MEEP_GPU_TEST_PREFLIGHT_ONLY=1 MEEP_GPU_TEST_RANK_INVALID_DEVICE=1
run_expected_preflight_failure \
  automatic-invalid-backend-hidden-one-rank 1 \
  "invalid MEEP_GPU_BACKEND='definitely-invalid'" \
  CUDA_VISIBLE_DEVICES= MEEP_GPU_BACKEND=definitely-invalid \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1
run_expected_preflight_failure \
  automatic-overflow-threshold-hidden-one-rank 1 \
  "MEEP_GPU_AUTO_MIN_CELLS must be a non-negative integer" \
  CUDA_VISIBLE_DEVICES= MEEP_GPU_BACKEND=auto \
  MEEP_GPU_AUTO_MIN_CELLS=18446744073709551616 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1
run_expected_preflight_failure \
  automatic-explicit-duplicate-device-two-rank 2 \
  "MPI ranks did not select distinct physical CUDA/MIG devices" \
  CUDA_VISIBLE_DEVICES=0,1 MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
  MEEP_GPU_BACKEND=auto MEEP_GPU_AUTO_MIN_CELLS=0 MEEP_GPU_DEVICE=0 \
  MEEP_GPU_MPI_TRANSPORT=pinned MEEP_GPU_TEST_PREFLIGHT_ONLY=1
run_expected_preflight_failure \
  ade-state-contract-cuda-required-hidden-one-rank 1 \
  "production ADE state contract requires CUDA on every rank" \
  CUDA_VISIBLE_DEVICES= MEEP_GPU_BACKEND=cpu \
  MEEP_GPU_TEST_ADE_STATE_ONLY=1
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.with_mpi() and mp.gpu.compiled; print(mp.__version__)' \
  >"${TEST_LOG_DIR}/installed-singleton-import.log" 2>&1
printf '%s\n' 'gpmeep-qualification:installed-singleton-import.log:PASS' >> \
  "${TEST_LOG_DIR}/installed-singleton-import.log"
env -u MEEP_GPU_DEVICE -u MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  CUDA_VISIBLE_DEVICES= HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 \
  MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/run-installed-mpb-qualification.py" \
  --expected-extension "${INSTALLED_MPB_EXTENSION}" \
  --expected-libpympb "${INSTALLED_LIBPYMPB}" \
  --expected-python-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  >"${TEST_LOG_DIR}/installed-mpb-runtime.log" 2>&1
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/mpiexec" -n 2 "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.count_processors() == 2 and mp.gpu.compiled' \
  >"${TEST_LOG_DIR}/installed-two-rank-import.log" 2>&1
printf '%s\n' 'gpmeep-qualification:installed-two-rank-import.log:PASS' >> \
  "${TEST_LOG_DIR}/installed-two-rank-import.log"
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=30s 1800s "${PREFIX}/bin/python" \
  "${REPO_ROOT}/python/tests/test_adjoint_default_material_grid.py" -v \
  >"${TEST_LOG_DIR}/installed-adjoint-default-material-grid.log" 2>&1
printf '%s\n' \
  'gpmeep-qualification:installed-adjoint-default-material-grid.log:PASS' >> \
  "${TEST_LOG_DIR}/installed-adjoint-default-material-grid.log"

# Receipt-bind the optimized startup contract to real installed consumers:
# legacy module/wildcard aliases, Matplotlib type introspection, Pade,
# visualization, SciPy filters, an executed JAX wrapper gradient/finite-
# difference oracle, and multifrequency adjoint paths.
env -u MEEP_GPU_DEVICE -u MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  CUDA_VISIBLE_DEVICES= HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 \
  MPLBACKEND=Agg MPLCONFIGDIR="${MPL_CONFIG}" \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=30s 1800s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/run-installed-lazy-api-qualification.py" \
  --repo "${REPO_ROOT}" --expected-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  >"${TEST_LOG_DIR}/installed-lazy-api-compatibility.log" 2>&1

# The installed-tree import gates above prove packaging and CPU availability.
# Seal separate CUDA executions as well so the receipt itself, rather than a
# later live audit, proves that the installed extension exercises one GPU,
# two distinct MPI GPUs, and the strict-CUDA adjoint/finite-difference path.
INSTALLED_GPU_SINGLETON_LOG="${TEST_LOG_DIR}/installed-python-gpu-backend-singleton.log"
env -u MEEP_GPU_DEVICE -u MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  -u MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS \
  HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="${MPL_CONFIG}" PYTHONPATH="${INSTALLED_PYTHON}" \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  CUDA_VISIBLE_DEVICES=0 GPMEEP_REQUIRE_CUDA_TEST=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/run-installed-cuda-qualification.py" \
  --repo "${REPO_ROOT}" --expected-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  --log-name installed-python-gpu-backend-singleton.log \
  >"${INSTALLED_GPU_SINGLETON_LOG}" 2>&1

INSTALLED_GPU_TWO_RANK_LOG="${TEST_LOG_DIR}/installed-python-gpu-backend-two-rank.log"
env -u MEEP_GPU_DEVICE -u MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS \
  HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="${MPL_CONFIG}" PYTHONPATH="${INSTALLED_PYTHON}" \
  MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 CUDA_VISIBLE_DEVICES=0,1 \
  MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 GPMEEP_REQUIRE_CUDA_TEST=1 \
  "${TIMEOUT}" --kill-after=10s 180s "${PREFIX}/bin/mpiexec" -n 2 \
  "${PREFIX}/bin/python" "${SCRIPT_DIR}/run-installed-cuda-qualification.py" \
  --repo "${REPO_ROOT}" --expected-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  --log-name installed-python-gpu-backend-two-rank.log \
  >"${INSTALLED_GPU_TWO_RANK_LOG}" 2>&1

INSTALLED_ADJOINT_CUDA_LOG="${TEST_LOG_DIR}/installed-adjoint-default-material-grid-cuda.log"
env -u MEEP_GPU_DEVICE -u MEEP_GPU_ALLOW_OVERSUBSCRIBE \
  HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="${MPL_CONFIG}" PYTHONPATH="${INSTALLED_PYTHON}" \
  CUDA_VISIBLE_DEVICES=0 MEEP_GPU_BACKEND=cuda MEEP_GPU_STRICT=1 \
  MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS=1 GPMEEP_REQUIRE_CUDA_TEST=1 \
  "${TIMEOUT}" --kill-after=30s 1800s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/run-installed-cuda-qualification.py" \
  --repo "${REPO_ROOT}" --expected-extension "${INSTALLED_EXTENSION}" \
  --expected-libmeep "${INSTALLED_LIBMEEP}" \
  --log-name installed-adjoint-default-material-grid-cuda.log \
  >"${INSTALLED_ADJOINT_CUDA_LOG}" 2>&1

RUNTIME_DEPENDENCY_CLOSURE="${BUILD_DIR}/runtime-dependency-closure.json"
env HOME="${QUALIFICATION_HOME}" PYTHONNOUSERSITE=1 \
  PYTHONPATH="${INSTALLED_PYTHON}" MEEP_GPU_BACKEND=cpu \
  "${TIMEOUT}" --kill-after=10s 120s "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/capture-runtime-dependency-closure.py" \
  --output "${RUNTIME_DEPENDENCY_CLOSURE}" \
  --environment-prefix "${PREFIX}" \
  --installed-prefix "${INSTALL_DIR}" \
  >"${TEST_LOG_DIR}/runtime-dependency-closure.log" 2>&1
printf '%s\n' 'gpmeep-qualification:runtime-dependency-closure.log:PASS' >> \
  "${TEST_LOG_DIR}/runtime-dependency-closure.log"

# Close the qualification epoch before sealing PASS evidence.  The two
# snapshots bind source, exact-lock prefix, both Python extensions/libraries,
# and every directly executed test ELF to the final receipt artifact set.
"${PREFIX_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" snapshot \
  --repo "${REPO_ROOT}" --prefix "${PREFIX}" \
  --installed-prefix "${INSTALL_DIR}" \
  --output "${QUALIFICATION_IDENTITY_AFTER}" \
  "${QUALIFICATION_IDENTITY_ARGS[@]}"
/usr/bin/kill -TERM "${QUALIFICATION_SENTINEL_PID}"
if ! wait "${QUALIFICATION_SENTINEL_PID}"; then
  echo "error: qualification mutation sentinel observed an identity change" >&2
  /usr/bin/cat "${QUALIFICATION_SENTINEL_LOG}" >&2
  exit 1
fi
QUALIFICATION_SENTINEL_PID=""
trap - EXIT
export PYTHONPYCACHEPREFIX=/dev/null
shopt -s dotglob nullglob
QUALIFICATION_PYCACHE_ENTRIES=("${QUALIFICATION_PYCACHE}"/*)
shopt -u dotglob nullglob
if [[ ! -d "${QUALIFICATION_PYCACHE}" || -L "${QUALIFICATION_PYCACHE}" || \
      "$(/usr/bin/stat -c '%a' "${QUALIFICATION_PYCACHE}")" != 700 || \
      "${#QUALIFICATION_PYCACHE_ENTRIES[@]}" -ne 0 ]]; then
  echo "error: immutable qualification pycache changed during qualification" >&2
  exit 1
fi
"${PREFIX_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/gpmeep_qualification_contract.py" seal \
  --repo "${REPO_ROOT}" --log-dir "${TEST_LOG_DIR}" \
  --before "${QUALIFICATION_IDENTITY_BEFORE}" \
  --after "${QUALIFICATION_IDENTITY_AFTER}" \
  --output "${QUALIFICATION_CONTRACT_V2}"

shopt -s dotglob nullglob
QUALIFICATION_HOME_ENTRIES=("${QUALIFICATION_HOME}"/*)
shopt -u dotglob nullglob
if [[ "${#QUALIFICATION_HOME_ENTRIES[@]}" -ne 1 || \
      "${QUALIFICATION_HOME_ENTRIES[0]}" != "${QUALIFICATION_HOME}/.gpmeep-empty-home" ]]; then
  echo "error: isolated MPI qualification HOME was modified" >&2
  exit 1
fi
if [[ "${PYTHONPYCACHEPREFIX}" != /dev/null ]]; then
  echo "error: qualification did not restore global Python cache control" >&2
  exit 1
fi
validate_install_pycache
assert_no_installed_bytecode

"${SYSTEM_CONTROL_PYTHON[@]}" "${FRESH_ATTESTATION_SCRIPT}" verify \
  --repo "${REPO_ROOT}" --prefix "${PREFIX}" --lock "${LOCK}" \
  --attestation "${FRESH_ENV_ATTESTATION_RECEIPT}" \
  --attestation-directory "${BUILD_DIR}" \
  --nonce "${GPMEEP_FRESH_ENV_NONCE}" --maximum-age-seconds 86400

"${TIMEOUT}" --kill-after=10s 180s \
  "${PREFIX_CONTROL_PYTHON[@]}" \
  "${SCRIPT_DIR}/write-build-receipt.py" finalize \
  --repo "${REPO_ROOT}" \
  --build-dir "${BUILD_DIR}" \
  --configuration-file "config_h=${BUILD_DIR}/config.h" \
  --configuration-file "config_status=${BUILD_DIR}/config.status" \
  --configuration-file "cuda_runtime_flags_stamp=${BUILD_DIR}/src/cuda-runtime-flags.stamp" \
  --configuration-file "cuda_runtime_cmake_cache=${CUDA_RUNTIME_BUILD_DIR}/CMakeCache.txt" \
  --configuration-file "environment_explicit=${BUILD_DIR}/environment-explicit.lock" \
  --configuration-file "openmpi_qualification_params=${OPENMPI_QUALIFICATION_PARAMS}" \
  --configuration-file "prte_mca_params=${PREFIX}/etc/prte-mca-params.conf" \
  --configuration-file "prte_default_hostfile=${PREFIX}/etc/prte-default-hostfile" \
  --configuration-file "runtime_dependency_closure=${RUNTIME_DEPENDENCY_CLOSURE}" \
  --configuration-file "fresh_environment_attestation=${FRESH_ENV_ATTESTATION_RECEIPT}" \
  --configuration-file "conda_prefix_content_audit=${PREFIX_CONTENT_AUDIT}" \
  --configuration-file "conda_prefix_pre_normalization_audit=${PREFIX_PRE_NORMALIZATION_AUDIT}" \
  --configuration-file "conda_prefix_post_import_audit_one=${PREFIX_POST_IMPORT_AUDIT_ONE}" \
  --configuration-file "conda_prefix_post_import_audit_two=${PREFIX_POST_IMPORT_AUDIT_TWO}" \
  --configuration-file "conda_generated_bytecode_normalization=${PREFIX_BYTECODE_NORMALIZATION}" \
  --configuration-file "conda_source_bytecode_normalization=${PREFIX_SOURCE_BYTECODE_NORMALIZATION}" \
  --configuration-file "conda_source_bytecode_import_probe_one=${PREFIX_IMPORT_PROBE_ONE}" \
  --configuration-file "conda_source_bytecode_import_probe_two=${PREFIX_IMPORT_PROBE_TWO}" \
  --configuration-file "micromamba=${MAMBA}" \
  --configuration-file "canonical_build_environment=${CANONICAL_BUILD_ENVIRONMENT}" \
  --configuration-file "qualification_fontconfig=${QUALIFICATION_FONTCONFIG}" \
  --configuration-file "install_pycache_policy=${INSTALL_PYCACHE_POLICY}" \
  --configuration-file "qualification_identity_before=${QUALIFICATION_IDENTITY_BEFORE}" \
  --configuration-file "qualification_identity_after=${QUALIFICATION_IDENTITY_AFTER}" \
  --configuration-file "qualification_contract_v2=${QUALIFICATION_CONTRACT_V2}" \
  --artifact "python_extension=${BUILD_EXTENSION}" \
  --artifact "libmeep=${BUILD_LIBMEEP}" \
  --artifact "installed_python_extension=${INSTALLED_EXTENSION}" \
  --artifact "installed_libmeep=${INSTALLED_LIBMEEP}" \
  --artifact "installed_mpb_extension=${INSTALLED_MPB_EXTENSION}" \
  --artifact "installed_libpympb=${INSTALLED_LIBPYMPB}" \
  --artifact "fd_allocation_shim=${FD_ALLOCATION_SHIM}" \
  --artifact "gpu_backend_test=${GPU_BACKEND_ELF}" \
  --artifact "gpu_step_db_test=${GPU_STEP_DB_ELF}" \
  --artifact "gpu_mpi_performance=${GPU_MPI_PERFORMANCE_ELF}" \
  --artifact "cuda_architecture_test=${CUDA_ARCHITECTURE_TEST_ELF}" \
  --artifact "cuda_formula_test=${CUDA_FORMULA_TEST_ELF}" \
  --artifact "cuda_runtime_validation=${CUDA_RUNTIME_VALIDATION_ELF}" \
  --artifact "cuda_near2far_runtime_validation=${CUDA_NEAR2FAR_RUNTIME_VALIDATION_ELF}" \
  --artifact "cuda_smoke=${CUDA_SMOKE_ELF}" \
  --manifest "in_place_python=${BUILD_DIR}/python/meep" \
  --manifest "installed_python=${INSTALLED_PYTHON}/meep" \
  --manifest "installed_environment=${PREFIX}" \
  --manifest "installed_prefix=${INSTALL_DIR}" \
  --manifest "install_pycache=${INSTALL_PYCACHE}" \
  --manifest "qualification_logs=${TEST_LOG_DIR}" \
  --manifest "qualification_home=${QUALIFICATION_HOME}" \
  --manifest "build_home=${BUILD_HOME}" \
  --immutable-directory \
    "prefix_fontconfig_cache=${PREFIX_FONTCONFIG_CACHE}" \
  --manifest-exclude-suffix "in_place_python=.pyc" \
  --manifest-exclude-suffix "installed_python=.pyc" \
  --manifest-exclude-suffix "installed_environment=.pyc" \
  --tool autoreconf --tool cc --tool c++ --tool cmake --tool make --tool ninja \
  --tool nvcc --tool python --tool swig --tool mpicxx --tool mpiexec \
  --tool h5pcc --tool timeout

# The receipt is now immutable. Bind the production Near2Far CPU/1-GPU/
# 2-GPU timing and numerical evidence back to that receipt, the directly
# executed receipt-bound ELF, its loaded libmeep/CUDA/MPI/HDF5 libraries, and
# the exact source snapshot. This output is deliberately outside the receipt
# manifests: the report consumes the receipt, so including it would create a
# circular artifact identity. The runner instead publishes a detached outer
# COMPLETE seal plus an external release attestation binding the receipt,
# bundle ID, report, source snapshot, and complete raw tree. Release consumers
# must supply the published bundle ID and this attestation to the verifier.
NEAR2FAR_RELEASE_QUALIFICATION="${BUILD_DIR}/near2far-release-qualification"
NEAR2FAR_RELEASE_HOME="${BUILD_DIR}/near2far-release-home"
mkdir -m 700 "${NEAR2FAR_RELEASE_HOME}"
env HOME="${NEAR2FAR_RELEASE_HOME}" \
  "${TIMEOUT}" --kill-after=30s 1800s \
  "${PREFIX}/bin/python" \
  "${SCRIPT_DIR}/run-near2far-mpi-qualification.py" \
  --qualification-tier release \
  --build-receipt "${BUILD_DIR}/build-provenance.json" \
  --write-release-attestation \
  "${BUILD_DIR}/near2far-release-attestation.json" \
  --executable "${GPU_STEP_DB_ELF}" \
  --output "${NEAR2FAR_RELEASE_QUALIFICATION}" \
  --cpu-threads "${MEEP_GPU_NEAR2FAR_CPU_THREADS:-8}" \
  --lane-timeout-seconds 300 \
  --stdout-limit-bytes 8388608 \
  --stderr-limit-bytes 8388608

echo "CUDA/MPI-enabled FP32 Meep Python package installed in ${INSTALL_DIR}"
echo "In-place package for tests: PYTHONPATH=${BUILD_DIR}/python"
