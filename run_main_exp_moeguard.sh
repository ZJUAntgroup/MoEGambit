#!/usr/bin/env bash
set -uo pipefail
set -x

# ============================================================
# Main experiment: MoEGuard on the 10-fault canonical trace
# ============================================================
# Trace design (10 events over 10000 iter, one event per 1000-iter window):
#
#   #   window         step  Δ    |F|   category
#   1   [501..1000]    723   123  1     single-GPU
#   2   [1001..2000]   1188  188  1     single-GPU (HBM)
#   3   [2001..3000]   2461  61   1     single-GPU  (small Δ, Δ_min test)
#   4   [3001..4000]   3517  117  1     single-GPU (SSD/PCIe)
#   5   [4001..5000]   4309  109  1     repeat — SAME rank as #1, tests Φ' accumulation
#   6   [5001..6000]   5640  40   1     single-GPU
#   7   [6001..7000]   6855  55   1     single-GPU
#   8   [7001..8000]   7912  112  8     single-host-class burst (8-card)
#   9   [8001..9000]   8689  89   8     single-host-class burst (other PP region)
#   10  [9001..10000]  9304  104  8     8-card burst (downgraded from 16; NCCL stability)
# Distribution = 7 single-GPU (70%) + 3 eight-card (30%) — note: original event
# #10 was a 16-card rack-outage burst, downgraded to 8-card after the previous
# run hit NCCL collective timeout / hang; the 8-card budget keeps the recovery
# path exercised end-to-end (hybrid restore + reintegration) without crossing
# the all-gather stall regime we observed at |F|=16 on this cluster. Resume
# point: latest ckpt at iter 9200, so only event #10 actually fires.
#
# ----- Randomness -----
# Failed ranks are drawn by a seeded Python RNG (PLAN_SEED, default 42) so the
# trace is reproducible. The step list, Δ list, and |F| list are fixed; only
# the rank identities are randomized within the topology constraints below.
# Event #5 deliberately reuses event #1's rank to exercise Φ'(t) accumulation
# under repeated-failure on the same expert slot.
#
# ----- DP safety constraint -----
# Dense / router parameters must be peer-pulled from a healthy DP sibling.
# With PP=8, EP=8, world=64 the DP group size = 64/(PP*TP) = 8, and ranks in
# the same PP stage share one DP group (stage_width = 8). Therefore the plan
# generator enforces:
#     for every PP stage,  #failed_in_stage <= stage_width - 1
# i.e. at least one healthy rank survives in each stage so dense_param_sync
# never starves. The 8-card and 16-card bursts are spread across two PP
# stages (7 in one stage + 1 in another, or 7+7+1+1) rather than collapsing
# a whole stage; this preserves hybrid recovery feasibility while staying
# faithful to the production-burst |F| budget.
# ============================================================

# ---- Shared plan generator + runtime env ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/main_exp_common.sh"
export_common_runtime

# ---- Output dirs ----
BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard}"
export CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${BASE_DIR}/log}"
mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# Explicitly disable MoC-PEC emulation in the MoEGuard run.
export BSR_MOC_PEC_EMULATE=0

# ---- Auto-detect resume point so already-fired faults are not back-fired ----
# When CKPT_DIR has a latest_checkpointed_iteration.txt, read it as
# RESUME_FROM_ITER and pass to build_main_plan, which will drop every
# scheduled fault with step <= RESUME_FROM_ITER. This is critical: without
# the filter the injector treats every plan event whose scheduled_step is
# already in the past as "due now" and back-fires them all in the first
# 10-20 iterations after resume (observed in 5281.log on the 9200 resume).
# Override with RESUME_FROM_ITER=0 if you really want to replay everything.
if [ -z "${RESUME_FROM_ITER:-}" ]; then
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
    RESUME_FROM_ITER="$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")"
    if ! [[ "${RESUME_FROM_ITER}" =~ ^[0-9]+$ ]]; then
      echo "[main_exp_moeguard] latest_checkpointed_iteration.txt is not an int (${RESUME_FROM_ITER}); defaulting to 0" >&2
      RESUME_FROM_ITER=0
    fi
  else
    RESUME_FROM_ITER=0
  fi
fi
export RESUME_FROM_ITER
echo "[main_exp_moeguard] RESUME_FROM_ITER=${RESUME_FROM_ITER} (events with step <= this are filtered out)"

echo "============================================================"
echo "[main_exp_moeguard] generating 10-fault plan (seed=${PLAN_SEED}, resume_from=${RESUME_FROM_ITER})"
echo "============================================================"
if ! MAIN_FAULT_PLAN="$(build_main_plan 2>/tmp/main_plan_debug.$$)"; then
  echo "[main_exp_moeguard] FAILED to build plan" >&2
  cat /tmp/main_plan_debug.$$ >&2 || true
  rm -f /tmp/main_plan_debug.$$
  exit 2
fi
cat /tmp/main_plan_debug.$$
rm -f /tmp/main_plan_debug.$$
echo "[main_exp_moeguard] resolved plan: ${MAIN_FAULT_PLAN}"
echo "============================================================"

# ---- BSR fault injection knobs (plan-driven, burst_all) ----
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
export BSR_REQUIRE_OLD_PARAM_RESTORE="${BSR_REQUIRE_OLD_PARAM_RESTORE:-1}"

# ---- Log analysis hooks ----
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
    --moe-bsr-health-mask \
    --moe-bsr-rank-quarantine \
    --moe-bsr-dispatch-quarantine-assert \
    --moe-bsr-dispatch-sanitize \
    --moe-bsr-expert-directory \
    --moe-bsr-replacement-protocol \
    --moe-bsr-group-rebuild \
    --moe-bsr-dispatch-topology-refresh \
    --moe-bsr-dense-param-sync \
    --moe-bsr-stale-expert-restore \
    --moe-bsr-recovery-controller \
    --moe-bsr-deferred-optimizer-load \
    --moe-bsr-degraded-mode-policy \
    --moe-bsr-reintegration-barrier \
    --moe-bsr-fault-injection \
    --moe-bsr-restart-in-place \
    --moe-bsr-hybrid-expert-restore \
    --moe-bsr-expert-opt-restore \
    --moe-bsr-weights-first-recovery \
    --moe-bsr-defer-optimizer-load \
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 1000 \
    --moe-bsr-degraded-s-max 500 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 200 \
    --eval-interval 1000 \
    --eval-iters 50 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

# ============================================================
# Main loop with retry + log analysis
# ============================================================
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY="${RETRY_DELAY:-30}"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"
retry=0

while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
    rc=$?
  else
    echo "[main_exp_moeguard] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[main_exp_moeguard] MoEGuard main experiment finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[main_exp_moeguard] failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[main_exp_moeguard] reached max retries, aborting"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done

echo "============================================================"
echo "[main_exp_moeguard] all done"
echo "Logs:    ${TRAIN_LOG_DIR}"
echo "Ckpts:   ${CKPT_DIR}"
echo "Plan:    ${MAIN_FAULT_PLAN}"
echo "Seed:    ${PLAN_SEED}"
echo "============================================================"
