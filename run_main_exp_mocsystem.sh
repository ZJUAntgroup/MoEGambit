#!/usr/bin/env bash
set -uo pipefail
set -x

# ============================================================
# Main experiment: MoC-System EMULATION on the 10-fault canonical trace
# ============================================================
# Accuracy-equivalent emulation of MoC-System's Partial Experts
# Checkpointing (PEC), implemented as a non-invasive overlay on
# MoEGuard's runtime. Activated by the single env var
# BSR_MOC_PEC_EMULATE=1. See:
#   Megatron-LM/megatron/core/transformer/moe/moc_pec_emulation.py
#
# Key emulation properties:
#   * Save path is UNCHANGED — MoEGuard still writes all N experts to
#     disk every save-interval, so ckpt is bit-perfect. A sidecar
#     moc_pec_metadata.json records which K_pec experts are "fresh"
#     for this PEC round and, for each other expert, the iter_*
#     directory holding its last-fresh version.
#   * Restore path is unchanged in MoEGuard, but each plan entry's
#     checkpoint_dir is rewritten to the directory chosen by PEC's
#     round-robin schedule before the expert load runs. Therefore the
#     bytes loaded into the model are exactly what a faithful
#     MoC-System would have loaded — accuracy is byte-identical.
#   * Hybrid restore (peer-pull dense/router) is DISABLED here because
#     MoC-System has no such mechanism. dense/router fall back to
#     CHECKPOINT_RESTART, matching MoC-System's actual recovery path.
#   * Two-phase (weights-first / opt-later) is DISABLED — MoC-System
#     does not use it.
#
# Recovery WALL-CLOCK from this script is NOT directly comparable to a
# real MoC-System because the save path writes more data than PEC
# would. The paper uses MoC-System's published recovery latency as a
# conservative upper bound on its advantage; see §Threats to Validity.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/main_exp_common.sh"
export_common_runtime

# ---- Output dirs ----
BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/main_exp/5.27/mocsystem}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/ckpt}"
export CKPT_DIR
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${BASE_DIR}/log}"
mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# ---- MoC-System emulation knobs ----
# THE ONE SWITCH that turns this script into a MoC-System emulation.
export BSR_MOC_PEC_EMULATE="${BSR_MOC_PEC_EMULATE:-1}"
# K_pec: number of "fresh" experts MoC-System writes per save round.
# 16 = paper's recommended setting for PLT≈3.75% on 128-expert models.
export BSR_MOC_PEC_K="${BSR_MOC_PEC_K:-16}"
export BSR_MOC_PEC_N_EXPERT="${BSR_MOC_PEC_N_EXPERT:-128}"
export BSR_MOC_PEC_SCHEDULE="${BSR_MOC_PEC_SCHEDULE:-round_robin}"

# ---- Build the simplified 10-fault plan for MoC-System ----
# MoC-System recovers every fault with a full CHECKPOINT_RESTART (the PEC
# overlay only changes WHICH expert file gets loaded, not the fact that
# every rank reloads). Therefore the rank identities in the plan have no
# bearing on accuracy or wall-clock cost; we collapse them to [0] and
# keep only the 10 step indices, which remain bit-identical with
# build_main_plan so loss-trajectory comparison stays step-aligned.
echo "============================================================"
echo "[main_exp_mocsystem] generating simplified 10-fault plan (steps match MoEGuard)"
echo "============================================================"
if ! MAIN_FAULT_PLAN="$(build_mocsystem_plan 2>/tmp/main_plan_debug.$$)"; then
  echo "[main_exp_mocsystem] FAILED to build plan" >&2
  cat /tmp/main_plan_debug.$$ >&2 || true
  rm -f /tmp/main_plan_debug.$$
  exit 2
fi
cat /tmp/main_plan_debug.$$
rm -f /tmp/main_plan_debug.$$
echo "[main_exp_mocsystem] resolved plan: ${MAIN_FAULT_PLAN}"
echo "[main_exp_mocsystem] MoC-PEC emulation enabled: "
echo "    BSR_MOC_PEC_EMULATE=${BSR_MOC_PEC_EMULATE}"
echo "    BSR_MOC_PEC_K=${BSR_MOC_PEC_K}"
echo "    BSR_MOC_PEC_N_EXPERT=${BSR_MOC_PEC_N_EXPERT}"
echo "    BSR_MOC_PEC_SCHEDULE=${BSR_MOC_PEC_SCHEDULE}"
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

  # MoC-System emulation: enable BSR infrastructure (needed for fault
  # injection + plan parsing + manifest sidecar) but DISABLE hybrid
  # restore, two-phase recovery, and stale-expert peer-pull. The
  # recovery_controller therefore falls back to CHECKPOINT_RESTART for
  # every fault, exactly matching MoC-System's restore behavior. The
  # PEC emulation overlay (apply_pec_to_plan) then redirects expert
  # entries to historical ckpts so accuracy matches MoC-System.
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
    --moe-bsr-recovery-controller \
    --moe-bsr-reintegration-barrier \
    --moe-bsr-fault-injection \
    --moe-bsr-restart-in-place \
    --moe-bsr-stale-expert-restore \
    --moe-bsr-degraded-mode-policy \
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
# NOTE: deliberately OMITTED switches (compared to MoEGuard) to faithfully
# emulate MoC-System's restore semantics:
#   --moe-bsr-dense-param-sync         (peer-pull dense — MoC-System has no equivalent)
#   --moe-bsr-hybrid-expert-restore    (selective expert load — disabled; PEC overlay handles)
#   --moe-bsr-expert-opt-restore       (expert opt selective restore — disabled)
#   --moe-bsr-weights-first-recovery   (two-phase weights-first — disabled)
#   --moe-bsr-defer-optimizer-load     (two-phase opt-later — disabled)
#   --moe-bsr-deferred-optimizer-load  (two-phase coordinator — disabled)

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
    echo "[main_exp_mocsystem] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[main_exp_mocsystem] MoC-System emulation finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[main_exp_mocsystem] failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[main_exp_mocsystem] reached max retries, aborting"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done

echo "============================================================"
echo "[main_exp_mocsystem] all done"
echo "Logs:    ${TRAIN_LOG_DIR}"
echo "Ckpts:   ${CKPT_DIR}"
echo "Plan:    ${MAIN_FAULT_PLAN}"
echo "Seed:    ${PLAN_SEED}"
echo "PEC:     K=${BSR_MOC_PEC_K}/N=${BSR_MOC_PEC_N_EXPERT}, schedule=${BSR_MOC_PEC_SCHEDULE}"
echo "============================================================"
