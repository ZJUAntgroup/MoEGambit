#!/usr/bin/env bash
# =============================================================================
# bench_moe128.sh
#
# Drives a fault-injection micro-benchmark on the 128-GPU MoE configuration:
#   Phase 1 (MODE=moegambit): N in-process hybrid recoveries
#                             (MOEGAMBIT_FAULT_INJECT_STEP, MOEGAMBIT_FAULT_INJECT_INTERVAL)
#   Phase 2 (MODE=baseline) : N crash + ckpt-restart events
#                             (CRASH_AT_STEP, CRASH_INTERVAL)
#
# Both phases use identical injection cadence (first at step BENCH_FIRST_INJECT_STEP,
# every BENCH_INJECT_INTERVAL steps thereafter) so the per-event recovery cost can
# be measured under matched conditions.  Each phase trains from step 0 with an
# isolated CKPT_DIR / TRAIN_LOG_DIR and auto-stops once N+tail steps have elapsed.
#
# Overrides (all optional):
#   BENCH_PHASES               default "moegambit baseline"
#   BENCH_NUM_INJECTIONS       default 10
#   BENCH_INJECT_INTERVAL      default 40
#   BENCH_FIRST_INJECT_STEP    default 70
#   BENCH_TAIL_STEPS           default 30
#   BENCH_BASE_DIR             default artifact runs directory
#   BENCH_TAG                  default 128gpu61_bench
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT  (passed through)
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ARTIFACT_RUN_ROOT="${ARTIFACT_RUN_ROOT:-${ARTIFACT_ROOT}/runs}"
export DATA_PATH="${DATA_PATH:-./sample_data_text_document}"
INNER_SCRIPT="${SCRIPT_DIR}/run_moe128.sh"
if [ ! -f "${INNER_SCRIPT}" ]; then
  echo "[bench_moe128] missing ${INNER_SCRIPT}"
  exit 2
fi

BENCH_PHASES="${BENCH_PHASES:-moegambit baseline}"
BENCH_NUM_INJECTIONS="${BENCH_NUM_INJECTIONS:-10}"
BENCH_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL:-40}"
BENCH_FIRST_INJECT_STEP="${BENCH_FIRST_INJECT_STEP:-70}"
BENCH_TAIL_STEPS="${BENCH_TAIL_STEPS:-30}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-${ARTIFACT_RUN_ROOT}}"
BENCH_TAG="${BENCH_TAG:-128gpu61_bench}"

# Step budget for one phase: first injection + (N-1) intervals + tail.
PHASE_TOTAL_STEPS=$(( BENCH_FIRST_INJECT_STEP \
                     + (BENCH_NUM_INJECTIONS - 1) * BENCH_INJECT_INTERVAL \
                     + BENCH_TAIL_STEPS ))

echo "[bench_moe128] config:"
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
  echo "[bench_moe128] starting phase mode=${mode}"
  echo "  CKPT_DIR = ${ckpt_dir}"
  echo "  LOG_DIR  = ${log_dir}"
  echo "============================================================"

  mkdir -p "${ckpt_dir}" "${log_dir}"

  # Start each phase from step 0: wipe prior ckpt artefacts in this benchmark
  # directory (does NOT touch the long-running training CKPT_DIRs).
  if [ -n "${ckpt_dir}" ] && [ "${ckpt_dir}" != "/" ]; then
    rm -f  "${ckpt_dir}/latest_checkpointed_iteration.txt"
    rm -rf "${ckpt_dir}"/iter_*
  fi

  # Common variables for both modes.
  export NNODES="${NNODES:-16}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-20115}"
  export TRAIN_ITERS="${PHASE_TOTAL_STEPS}"
  export SAVE_INTERVAL="${BENCH_INJECT_INTERVAL}"
  export CKPT_DIR="${ckpt_dir}"
  export TRAIN_LOG_DIR="${log_dir}"

  if [ "${mode}" = "moegambit" ]; then
    # In-process hybrid recovery: the python process never exits, so a single
    # torchrun absorbs all N injections driven by MOEGAMBIT_FAULT_INJECT_INTERVAL.
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
    # MAX_RETRIES=1 -> outer retry loop in run_moe128.sh does not relaunch.
    export MAX_RETRIES=1
  else
    # Checkpoint-restart baseline: each crash kills torchrun; the outer
    # while-loop in run_moe128.sh restarts via --load.  N crashes => N+1 sub-runs.
    export MODE=baseline
    export CRASH_AT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export CRASH_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export CRASH_RANK=-1
    export CRASH_SEED=42
    # Allow exactly N restarts: first launch + N crash-relaunches.
    export MAX_RETRIES="${BENCH_NUM_INJECTIONS}"
    # Faster turnaround between restarts (default 30s is fine; keep override path).
    export RETRY_DELAY="${RETRY_DELAY:-15}"
  fi

  # Hand off to the inner runner.  We invoke it via `bash` (not source) so the
  # exit code propagates cleanly and any per-mode `unset` it does cannot leak
  # back into this driver between phases.
  bash "${INNER_SCRIPT}"
  local rc=$?
  echo "[bench_moe128] phase mode=${mode} finished with rc=${rc}"
  return ${rc}
}

overall_rc=0
for ph in ${BENCH_PHASES}; do
  case "${ph}" in
    moegambit|baseline) ;;
    *)
      echo "[bench_moe128] invalid phase '${ph}', must be 'moegambit' or 'baseline'"
      exit 2
      ;;
  esac
  run_phase "${ph}"
  rc=$?
  # baseline phase is *expected* to end non-zero: after the Nth scheduled
  # crash the outer retry loop in run_moe128.sh hits MAX_RETRIES and exits
  # with the crash's rc.  Treat that as success for the benchmark.
  if [ "${ph}" = "baseline" ] && [ ${rc} -ne 0 ]; then
    echo "[bench_moe128] baseline phase ended with rc=${rc} after ${BENCH_NUM_INJECTIONS} planned crashes (expected); treating as success"
    rc=0
  fi
  if [ ${rc} -ne 0 ]; then
    echo "[bench_moe128] phase ${ph} exited with rc=${rc}"
    overall_rc=${rc}
    # Continue to next phase; do not abort the benchmark just because one
    # phase reported a non-zero rc.
  fi
done

echo "[bench_moe128] all phases done, overall_rc=${overall_rc}"
exit ${overall_rc}
