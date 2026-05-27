#!/usr/bin/env bash
set -uo pipefail
set -x

# ============================================================
# Main experiment: Megatron-LM full-ckpt restart BASELINE
# ============================================================
# Native Megatron-LM behavior: on any rank failure, training is aborted,
# the launcher restarts from the latest distributed checkpoint, and all
# ranks replay every iteration since that checkpoint. No BSR runtime,
# no hybrid recovery, no PEC.
#
# To exercise the same 10-fault trace as the other two systems while
# still letting Megatron run end-to-end, we keep the BSR fault-injection
# subsystem enabled (it is the trace driver) but explicitly select
# CHECKPOINT_RESTART for every event by:
#   * disabling --moe-bsr-stale-expert-restore
#   * disabling --moe-bsr-hybrid-expert-restore
#   * disabling --moe-bsr-dense-param-sync
#   * disabling --moe-bsr-weights-first-recovery
#   * disabling --moe-bsr-defer-optimizer-load
#   * setting BSR_REQUIRE_OLD_PARAM_RESTORE=0 so the controller is allowed
#     to fall back to CHECKPOINT_RESTART when no hybrid path is available
# This gives a realistic Megatron-LM full-restart wall-clock and accuracy
# while reusing the same fault trace generator and DP-safety constraints
# as the MoEGuard / MoC-System runs.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/main_exp_common.sh"
export_common_runtime

BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/main_exp/5.27/baseline}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/ckpt}"
export CKPT_DIR
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${BASE_DIR}/log}"
mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# Explicitly disable MoC-PEC emulation in this run.
export BSR_MOC_PEC_EMULATE=0

echo "============================================================"
echo "[main_exp_baseline] generating 10-fault plan (seed=${PLAN_SEED})"
echo "============================================================"
if ! MAIN_FAULT_PLAN="$(build_main_plan 2>/tmp/main_plan_debug.$$)"; then
  echo "[main_exp_baseline] FAILED to build plan" >&2
  cat /tmp/main_plan_debug.$$ >&2 || true
  rm -f /tmp/main_plan_debug.$$
  exit 2
fi
cat /tmp/main_plan_debug.$$
rm -f /tmp/main_plan_debug.$$
echo "[main_exp_baseline] resolved plan: ${MAIN_FAULT_PLAN}"
echo "============================================================"

export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
export BSR_FAULT_INJECT_PLAN="${MAIN_FAULT_PLAN}"
export BSR_FAULT_INJECT_PLAN_MODE="${BSR_FAULT_INJECT_PLAN_MODE:-burst_all}"
export BSR_FAULT_INJECT_STEP="0"
export BSR_FAULT_INJECT_INTERVAL="0"
export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:-0}"
export BSR_FAULT_REPLACEMENT_STEP="0"
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"
# Baseline must fall back to CHECKPOINT_RESTART; do not enforce hybrid restore.
export BSR_REQUIRE_OLD_PARAM_RESTORE=0

export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

run_training() {
  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-20115}" \
    ./Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    --expert-model-parallel-size "${EP_SIZE}" \
    --sequence-parallel \
    --legacy-tokenizer \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ./tokenizer \
    --vocab-file "./tokenizer/vocab.json" \
    --merge-file "./tokenizer/merges.txt" \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --qk-layernorm \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --train-iters 10000 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --normalization RMSNorm \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-rope-fusion \
    --swiglu \
    --no-bias-swiglu-fusion \
    --untie-embeddings-and-output-weights \
    --bf16 \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-bsr-enable \
    --moe-bsr-fault-injection \
    --moe-bsr-restart-in-place \
    --moe-bsr-recovery-controller \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 200 \
    --eval-interval 500 \
    --eval-iters 20 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}
# Deliberately OMITTED (compared to MoEGuard) to force CHECKPOINT_RESTART:
#   --moe-bsr-dense-param-sync          (no peer-pull)
#   --moe-bsr-stale-expert-restore      (no selective expert load)
#   --moe-bsr-hybrid-expert-restore     (no hybrid path)
#   --moe-bsr-expert-opt-restore        (no selective expert opt restore)
#   --moe-bsr-weights-first-recovery    (no two-phase weights-first)
#   --moe-bsr-defer-optimizer-load      (no two-phase opt-later)
#   --moe-bsr-deferred-optimizer-load   (no two-phase coordinator)
#   --moe-bsr-degraded-mode-policy      (no degraded-mode logic)

MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY="${RETRY_DELAY:-30}"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"
retry=0

while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
    rc=$?
  else
    echo "[main_exp_baseline] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[main_exp_baseline] Megatron baseline finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[main_exp_baseline] failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[main_exp_baseline] reached max retries, aborting"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done

echo "============================================================"
echo "[main_exp_baseline] all done"
echo "Logs:  ${TRAIN_LOG_DIR}"
echo "Ckpts: ${CKPT_DIR}"
echo "Plan:  ${MAIN_FAULT_PLAN}"
echo "Seed:  ${PLAN_SEED}"
echo "============================================================"
