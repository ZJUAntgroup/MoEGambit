#!/usr/bin/env bash
# =============================================================================
# bench_moe64_tp1pp8ep4.sh
#
# Parallelism sensitivity experiment #3:
#   TP=1, PP=8, EP=4, EDP=2  (64 GPUs = 8 nodes x 8 GPUs)
#
# EDP=2 means each expert shard has a DP peer.  Recovery uses the
# FULL_PEER_RECOVERY path: ALL parameters (dense + expert weights +
# optimizer state) are pulled from a healthy DP peer, avoiding
# checkpoint I/O entirely.
#
# Drives a fault-injection micro-benchmark:
#   Phase 1 (MODE=moegambit): N in-process hybrid recoveries
#   Phase 2 (MODE=baseline) : N crash + ckpt-restart events
#
# Both phases use identical injection cadence so per-event recovery cost
# can be measured under matched conditions.
#
# Overrides (all optional):
#   BENCH_PHASES               default "moegambit baseline"
#   BENCH_NUM_INJECTIONS       default 10
#   BENCH_INJECT_INTERVAL      default 40
#   BENCH_FIRST_INJECT_STEP    default 70
#   BENCH_TAIL_STEPS           default 30
#   BENCH_BASE_DIR             default /mnt/ais-c1/dataset/zds
#   BENCH_TAG                  default 64gpu_tp1pp8ep4_bench
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT  (passed through)
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/run_moe64_par.sh"
if [ ! -f "${INNER_SCRIPT}" ]; then
  echo "[bench] missing ${INNER_SCRIPT}"
  exit 2
fi

# ---- Parallelism for this experiment ----
export TP_SIZE=1
export PP_SIZE=8
export EP_SIZE=4
# EDP = 64 / (1*8*4) = 2  -> DP peer available, full peer recovery
export MOEGAMBIT_FULL_PEER_RECOVERY=1

BENCH_PHASES="${BENCH_PHASES:-moegambit baseline}"
BENCH_NUM_INJECTIONS="${BENCH_NUM_INJECTIONS:-10}"
BENCH_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL:-40}"
BENCH_FIRST_INJECT_STEP="${BENCH_FIRST_INJECT_STEP:-70}"
BENCH_TAIL_STEPS="${BENCH_TAIL_STEPS:-30}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-/mnt/ais-c1/dataset/zds}"
BENCH_TAG="${BENCH_TAG:-64gpu_tp1pp8ep4_bench}"

PHASE_TOTAL_STEPS=$(( BENCH_FIRST_INJECT_STEP \
                     + (BENCH_NUM_INJECTIONS - 1) * BENCH_INJECT_INTERVAL \
                     + BENCH_TAIL_STEPS ))

echo "[bench_tp1pp8ep4] config:"
echo "  PARALLELISM       = TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, EDP=2"
echo "  RECOVERY          = FULL_PEER_RECOVERY (all params from DP peer)"
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
  echo "[bench_tp1pp8ep4] starting phase mode=${mode}"
  echo "  CKPT_DIR = ${ckpt_dir}"
  echo "  LOG_DIR  = ${log_dir}"
  echo "============================================================"

  mkdir -p "${ckpt_dir}" "${log_dir}"

  if [ -n "${ckpt_dir}" ] && [ "${ckpt_dir}" != "/" ]; then
    rm -f  "${ckpt_dir}/latest_checkpointed_iteration.txt"
    rm -rf "${ckpt_dir}"/iter_*
  fi

  export NNODES="${NNODES:-8}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-20115}"
  export TRAIN_ITERS="${PHASE_TOTAL_STEPS}"
  export SAVE_INTERVAL="${BENCH_INJECT_INTERVAL}"
  export CKPT_DIR="${ckpt_dir}"
  export TRAIN_LOG_DIR="${log_dir}"

  if [ "${mode}" = "moegambit" ]; then
    export MODE=moegambit
    export MOEGAMBIT_FAULT_INJECT_TYPE=restart_in_place
    export MOEGAMBIT_FAULT_INJECT_RANK=-1
    export MOEGAMBIT_FAULT_INJECT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export MOEGAMBIT_FAULT_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export MOEGAMBIT_FAULT_INJECT_SEED=42
    export MOEGAMBIT_FAULT_REPLACEMENT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export MOEGAMBIT_FAULT_REPLACEMENT_RANK=-1
    export MOEGAMBIT_FAULT_ZERO_MEMORY=1
    export MOEGAMBIT_FAULT_MEMORY_FILL=zero
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
  echo "[bench_tp1pp8ep4] phase mode=${mode} finished with rc=${rc}"
  return ${rc}
}

overall_rc=0
for ph in ${BENCH_PHASES}; do
  case "${ph}" in
    moegambit|baseline) ;;
    *)
      echo "[bench_tp1pp8ep4] invalid phase '${ph}', must be 'moegambit' or 'baseline'"
      exit 2
      ;;
  esac
  run_phase "${ph}"
  rc=$?
  if [ "${ph}" = "baseline" ] && [ ${rc} -ne 0 ]; then
    echo "[bench_tp1pp8ep4] baseline phase ended with rc=${rc} after ${BENCH_NUM_INJECTIONS} planned crashes (expected); treating as success"
    rc=0
  fi
  if [ ${rc} -ne 0 ]; then
    echo "[bench_tp1pp8ep4] phase ${ph} exited with rc=${rc}"
    overall_rc=${rc}
  fi
done

echo "[bench_tp1pp8ep4] all phases done, overall_rc=${overall_rc}"
exit ${overall_rc}
