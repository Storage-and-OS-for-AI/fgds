#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") GPU_ID NUM_ROUNDS" >&2
  echo "Example: $(basename "$0") 0 3" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

GPU_ID="$1"
NUM_ROUNDS="$2"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

FAIL=0

run_one() {
  local name="$1"
  echo "=== Running ${name} (gpu_id=${GPU_ID}, num_rounds=${NUM_ROUNDS}) ==="
  set +e
  python3 "${SCRIPT_DIR}/${name}" "${GPU_ID}" "${NUM_ROUNDS}"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "=== ${name} FAILED (exit_code=${rc}) ===" >&2
    FAIL=1
  else
    echo "=== ${name} OK ==="
  fi
}

run_one "bench_fgds_pytorch.py"
echo
sleep 3

run_one "bench_gds_pytorch.py"
echo
sleep 3

run_one "bench_posix_pytorch.py"
echo

echo "=== Done ==="
exit $FAIL
