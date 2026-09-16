#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: gpmeep self-check launcher requires protected Bash mode" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PATH="${SCRIPT_DIR}:/usr/bin:/bin"
exec "${SCRIPT_DIR}/python3.11" \
  "${SCRIPT_DIR}/../share/gpmeep/libexec/gpmeep-self-check.py" "$@"
