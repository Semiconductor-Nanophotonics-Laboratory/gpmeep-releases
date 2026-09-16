#!/usr/bin/env bash
set -euo pipefail

NVCC="${1:-}"
SPECIFICATION="${2:-AUTO}"
OUTPUT_MODE="${3:-cmake}"

if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
  echo "error: first argument must be an executable nvcc path" >&2
  exit 2
fi

SUPPORTED_CODES="$("${NVCC}" --list-gpu-code)"
SUPPORTED_ARCHITECTURES="$("${NVCC}" --list-gpu-arch)"
REAL_ARCHITECTURES=()
VIRTUAL_ARCHITECTURES=()
CMAKE_ARCHITECTURES=()

contains_line() {
  local lines="$1"
  local expected="$2"
  while IFS= read -r line; do
    if [[ "${line}" == "${expected}" ]]; then
      return 0
    fi
  done <<<"${lines}"
  return 1
}

append_unique() {
  local -n target_array="$1"
  local value="$2"
  local existing
  for existing in "${target_array[@]}"; do
    if [[ "${existing}" == "${value}" ]]; then
      return
    fi
  done
  target_array+=("${value}")
}

add_real() {
  local architecture="$1"
  if (( 10#${architecture} < 60 )); then
    echo "error: CUDA architecture sm_${architecture} is below gpmeep's sm_60 minimum" >&2
    exit 1
  fi
  if ! contains_line "${SUPPORTED_CODES}" "sm_${architecture}"; then
    echo "error: ${NVCC} does not support native sm_${architecture}" >&2
    exit 1
  fi
  append_unique REAL_ARCHITECTURES "${architecture}"
  append_unique CMAKE_ARCHITECTURES "${architecture}-real"
}

add_virtual() {
  local architecture="$1"
  if (( 10#${architecture} < 60 )); then
    echo "error: CUDA architecture compute_${architecture} is below gpmeep's compute_60 minimum" >&2
    exit 1
  fi
  if ! contains_line "${SUPPORTED_ARCHITECTURES}" "compute_${architecture}"; then
    echo "error: ${NVCC} does not support virtual compute_${architecture}" >&2
    exit 1
  fi
  append_unique VIRTUAL_ARCHITECTURES "${architecture}"
  append_unique CMAKE_ARCHITECTURES "${architecture}-virtual"
}

if [[ "${SPECIFICATION}" == "AUTO" ]]; then
  # Select every numeric native architecture exposed by this nvcc.  Keeping a
  # hard-coded list silently omitted new GPU generations (for example sm_100
  # or sm_120) when users built gpmeep with a newer CUDA toolkit.
  mapfile -t REQUESTED_ARCHITECTURES < <(
    while IFS= read -r code; do
      if [[ "${code}" =~ ^sm_([0-9]+)$ ]] &&
         (( 10#${BASH_REMATCH[1]} >= 60 )); then
        printf '%s\n' "${BASH_REMATCH[1]}"
      fi
    done <<<"${SUPPORTED_CODES}" | sort -n -u
  )
  NEWEST_ARCHITECTURE=""
  for architecture in "${REQUESTED_ARCHITECTURES[@]}"; do
    if contains_line "${SUPPORTED_CODES}" "sm_${architecture}"; then
      add_real "${architecture}"
      NEWEST_ARCHITECTURE="${architecture}"
    fi
  done
  if [[ -z "${NEWEST_ARCHITECTURE}" ]]; then
    echo "error: ${NVCC} supports none of the portable default CUDA architectures" >&2
    exit 1
  fi
  if contains_line "${SUPPORTED_ARCHITECTURES}" "compute_${NEWEST_ARCHITECTURE}"; then
    add_virtual "${NEWEST_ARCHITECTURE}"
  fi
else
  NORMALIZED_SPECIFICATION="${SPECIFICATION//,/ }"
  NORMALIZED_SPECIFICATION="${NORMALIZED_SPECIFICATION//;/ }"
  read -r -a REQUESTED_ENTRIES <<<"${NORMALIZED_SPECIFICATION}"
  if [[ "${#REQUESTED_ENTRIES[@]}" -eq 0 ]]; then
    echo "error: CUDA architecture specification is empty" >&2
    exit 1
  fi
  for entry in "${REQUESTED_ENTRIES[@]}"; do
    if [[ "${entry}" =~ ^([0-9]+)-real$ ]]; then
      add_real "${BASH_REMATCH[1]}"
    elif [[ "${entry}" =~ ^([0-9]+)-virtual$ ]]; then
      add_virtual "${BASH_REMATCH[1]}"
    elif [[ "${entry}" =~ ^[0-9]+$ ]]; then
      add_real "${entry}"
      add_virtual "${entry}"
    else
      echo "error: invalid CUDA architecture '${entry}'" >&2
      exit 1
    fi
  done
fi

if [[ "${#CMAKE_ARCHITECTURES[@]}" -eq 0 ]]; then
  echo "error: no CUDA architectures were selected" >&2
  exit 1
fi

join_by() {
  local delimiter="$1"
  shift
  local first=1
  local value
  for value in "$@"; do
    if [[ "${first}" == "0" ]]; then
      printf '%s' "${delimiter}"
    fi
    printf '%s' "${value}"
    first=0
  done
}

case "${OUTPUT_MODE}" in
  cmake)
    join_by ';' "${CMAKE_ARCHITECTURES[@]}"
    ;;
  nvcc)
    for architecture in "${REAL_ARCHITECTURES[@]}"; do
      printf '%s ' "-gencode=arch=compute_${architecture},code=sm_${architecture}"
    done
    for architecture in "${VIRTUAL_ARCHITECTURES[@]}"; do
      printf '%s ' "-gencode=arch=compute_${architecture},code=compute_${architecture}"
    done
    ;;
  real-values)
    printf '0'
    for architecture in "${REAL_ARCHITECTURES[@]}"; do
      printf ', %s' "${architecture}"
    done
    ;;
  virtual-values)
    printf '0'
    for architecture in "${VIRTUAL_ARCHITECTURES[@]}"; do
      printf ', %s' "${architecture}"
    done
    ;;
  *)
    echo "usage: $0 NVCC [AUTO|ARCH-LIST] [cmake|nvcc|real-values|virtual-values]" >&2
    exit 2
    ;;
esac

printf '\n'
