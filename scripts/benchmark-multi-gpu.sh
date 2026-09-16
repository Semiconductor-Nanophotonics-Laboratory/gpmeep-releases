#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point.  The Python controller owns strict JSON parsing,
# timeout handling for every mpirun, requested/executed-rank accounting, and
# the generated CSV/JSON evidence.  Keep this shell name for existing docs and
# automation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.envs/meep-gpu-cuda-mpi/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: isolated CUDA/MPI Python is missing: ${PYTHON}" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/multi_gpu_benchmark.py" "$@"
