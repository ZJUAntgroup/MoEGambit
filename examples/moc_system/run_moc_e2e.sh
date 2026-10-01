#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export BASE_DIR="${BASE_DIR:-/shared/moegambit/baseline/seed_1234}"
export LOG_DIR="${LOG_DIR:-/personal/moegambit/moc_e2e}"
export RUN_ID="${RUN_ID:-moc_e2e_01}"
export MOC_CKPT_ROOT="${MOC_CKPT_ROOT:-${FSE_CKPT_ROOT:-/shared/moegambit/moc_e2e_ckpts/${RUN_ID}}}"
export MOC_RESULT_DIR="${LOG_DIR}/${RUN_ID}"
export MOC_START_STEP="${MOC_START_STEP:-4000}"
export MOC_REPEATS="${MOC_REPEATS:-1}"
export MOC_CKPT_INTERVAL="${MOC_CKPT_INTERVAL:-200}"
export MOC_FAILURE_GAP="${MOC_FAILURE_GAP:-5}"
export MOC_POST_FAILURE_STEPS="${MOC_POST_FAILURE_STEPS:-10}"
export MOC_FAILURE_COUNTS="${MOC_FAILURE_COUNTS:-1}"
export MOC_SNAPSHOT_K="${MOC_SNAPSHOT_K:-32}"
export MOC_PERSIST_K="${MOC_PERSIST_K:-16}"
export MOC_FSYNC=1
export MASTER_PORT="${MASTER_PORT:-20140}"
export PYTHONUNBUFFERED=1
while IFS= read -r name; do
  case "$name" in
    FSE_QUALITY_*|FSE_BRANCH_*|FSE_EP_LOAD_*|BSR_FAULT_*|BSR_MOC_*|MOEGAMBIT_FAULT_*|MOEGAMBIT_MOC_*|BSR_FORCE_CHECKPOINT_RESTART|MOEGAMBIT_FORCE_CHECKPOINT_RESTART) unset "$name" ;;
  esac
done < <(compgen -e)
source "$SCRIPT_DIR/common.sh"
export PP_SIZE EP_SIZE NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR SEED NUM_EXPERTS MOC_MODEL_PROFILE
moc_model_args
moc_torchrun
TORCHRUN_CMD[${#TORCHRUN_CMD[@]}-1]="$SCRIPT_DIR/moc_e2e_pretrain.py"
CMD=(torchrun --rdzv-conf timeout=300 "${TORCHRUN_CMD[@]:1}" "${MODEL_ARGS[@]}"
  --data-cache-path "$MOC_RESULT_DIR/data_cache"
  --eval-iters 0 --eval-interval 10001 --distributed-timeout-minutes 5)
if [[ "${DRY_RUN:-0}" == 1 || "${1:-}" == --plan-only ]]; then
  python3 "$SCRIPT_DIR/moc_e2e_plan.py"
  printf '\n[moc-e2e] worker template:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
(( $# == 0 )) || { echo "only --plan-only is accepted" >&2; exit 2; }
moc_require_inputs
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
exec python3 -u "$SCRIPT_DIR/moc_e2e_launch.py" -- "${CMD[@]}"
