#!/usr/bin/env bash
# =============================================================================
# bench_moe16.sh
#
# Drives the same fault-injection micro-benchmark as bench_moe128.sh, but on
# the 16-GPU MoE configuration:
#     NNODES=2, TP=1, PP=8, EP=2, DP=1, GBS=16
#
# Phase 1 (MODE=moegambit): N in-process hybrid recoveries
# Phase 2 (MODE=baseline) : N crash + ckpt-restart events
#
# Both phases share an injection cadence (first at BENCH_FIRST_INJECT_STEP,
# every BENCH_INJECT_INTERVAL steps thereafter).
#
# Overrides (all optional):
#   BENCH_PHASES               default "moegambit baseline"
#   BENCH_NUM_INJECTIONS       default 10
#   BENCH_INJECT_INTERVAL      default 40
#   BENCH_FIRST_INJECT_STEP    default 70
#   BENCH_TAIL_STEPS           default 30
#   BENCH_BASE_DIR             default /mnt/ais-c1/dataset/zds
#   BENCH_TAG                  default 16gpu_bench
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT  (passed through)
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_INNER="${SCRIPT_DIR}/run_moe128.sh"
if [ ! -f "${ORIGINAL_INNER}" ]; then
  echo "[bench_moe16] missing ${ORIGINAL_INNER}"
  exit 2
fi

# 16-GPU layout-specific overrides.
PHASE_TP_SIZE=1
PHASE_PP_SIZE=8
PHASE_EP_SIZE=2
PHASE_GBS=16

# Build a one-off inner script that hard-codes the GBS this benchmark needs.
# (run_moe128.sh accepts TP/PP/EP via env vars, but --global-batch-size is
# hard-coded; so we patch a temp copy rather than mutating the original.)
INNER_SCRIPT="$(mktemp -t run_moe16_XXXXXX.sh)"
trap 'rm -f "${INNER_SCRIPT}"' EXIT
sed -E "s|--global-batch-size[[:space:]]+[0-9]+|--global-batch-size ${PHASE_GBS}|" \
    "${ORIGINAL_INNER}" > "${INNER_SCRIPT}"
chmod +x "${INNER_SCRIPT}"

BENCH_PHASES="${BENCH_PHASES:-moegambit baseline}"
BENCH_NUM_INJECTIONS="${BENCH_NUM_INJECTIONS:-10}"
BENCH_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL:-40}"
BENCH_FIRST_INJECT_STEP="${BENCH_FIRST_INJECT_STEP:-70}"
BENCH_TAIL_STEPS="${BENCH_TAIL_STEPS:-30}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-/mnt/ais-c1/dataset/zds}"
BENCH_TAG="${BENCH_TAG:-16gpu_bench}"

PHASE_TOTAL_STEPS=$(( BENCH_FIRST_INJECT_STEP \
                     + (BENCH_NUM_INJECTIONS - 1) * BENCH_INJECT_INTERVAL \
                     + BENCH_TAIL_STEPS ))

echo "[bench_moe16] config:"
echo "  LAYOUT            = NNODES=2 TP=${PHASE_TP_SIZE} PP=${PHASE_PP_SIZE} EP=${PHASE_EP_SIZE} DP=1 GBS=${PHASE_GBS}"
echo "  PHASES            = ${BENCH_PHASES}"
echo "  NUM_INJECTIONS    = ${BENCH_NUM_INJECTIONS}"
echo "  FIRST_INJECT_STEP = ${BENCH_FIRST_INJECT_STEP}"
echo "  INJECT_INTERVAL   = ${BENCH_INJECT_INTERVAL}"
echo "  TAIL_STEPS        = ${BENCH_TAIL_STEPS}"
echo "  PHASE_TOTAL_STEPS = ${PHASE_TOTAL_STEPS}"
echo "  BASE_DIR          = ${BENCH_BASE_DIR}"
echo "  TAG               = ${BENCH_TAG}"

run_phase() {
  local mode="$1"
  local ckpt_dir="${BENCH_BASE_DIR}/${BENCH_TAG}/${mode}"
  local log_dir="${BENCH_BASE_DIR}/log/${BENCH_TAG}/${mode}"

  echo "============================================================"
  echo "[bench_moe16] starting phase mode=${mode}"
  echo "  CKPT_DIR = ${ckpt_dir}"
  echo "  LOG_DIR  = ${log_dir}"
  echo "============================================================"

  mkdir -p "${ckpt_dir}" "${log_dir}"

  if [ -n "${ckpt_dir}" ] && [ "${ckpt_dir}" != "/" ]; then
    rm -f  "${ckpt_dir}/latest_checkpointed_iteration.txt"
    rm -rf "${ckpt_dir}"/iter_*
  fi

  # 16-GPU parallelism layout.
  export NNODES="${NNODES:-2}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-20116}"
  export TP_SIZE="${PHASE_TP_SIZE}"
  export PP_SIZE="${PHASE_PP_SIZE}"
  export EP_SIZE="${PHASE_EP_SIZE}"
  export TRAIN_ITERS="${PHASE_TOTAL_STEPS}"
  export SAVE_INTERVAL="${BENCH_INJECT_INTERVAL}"
  export CKPT_DIR="${ckpt_dir}"
  export TRAIN_LOG_DIR="${log_dir}"

  if [ "${mode}" = "moegambit" ]; then
    export MODE=moegambit
    export BSR_FAULT_INJECT_TYPE=restart_in_place
    export BSR_FAULT_INJECT_RANK=-1
    export BSR_FAULT_INJECT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export BSR_FAULT_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export BSR_FAULT_INJECT_SEED=42
    export BSR_FAULT_REPLACEMENT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export BSR_FAULT_REPLACEMENT_RANK=-1
    export BSR_FAULT_ZERO_MEMORY=1
    export BSR_FAULT_MEMORY_FILL=zero
    export MAX_RETRIES=1
  else
    export MODE=baseline
    export CRASH_AT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export CRASH_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export CRASH_RANK=-1
    export CRASH_SEED=42
    export MAX_RETRIES="${BENCH_NUM_INJECTIONS}"
    export RETRY_DELAY="${RETRY_DELAY:-15}"
  fi

  bash "${INNER_SCRIPT}"
  local rc=$?
  echo "[bench_moe16] phase mode=${mode} finished with rc=${rc}"
  return ${rc}
}

overall_rc=0
for ph in ${BENCH_PHASES}; do
  case "${ph}" in
    moegambit|baseline) ;;
    *)
      echo "[bench_moe16] invalid phase '${ph}', must be 'moegambit' or 'baseline'"
      exit 2
      ;;
  esac
  run_phase "${ph}"
  rc=$?
  if [ "${ph}" = "baseline" ] && [ ${rc} -ne 0 ]; then
    echo "[bench_moe16] baseline phase ended with rc=${rc} after ${BENCH_NUM_INJECTIONS} planned crashes (expected); treating as success"
    rc=0
  fi
  if [ ${rc} -ne 0 ]; then
    echo "[bench_moe16] phase ${ph} exited with rc=${rc}"
    overall_rc=${rc}
  fi
done

echo "[bench_moe16] all phases done, overall_rc=${overall_rc}"
exit ${overall_rc}
