#!/usr/bin/env bash
set -uo pipefail
set -x

export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

# ============================================================
# findmax_baseline_long.sh: 1 run, 10000 iterations
# ============================================================
# Pure-baseline long run for noise-floor / loss-curve characterization.
# Settings:
#   - only one run (no 10x sweep)
#   - 10000 train iterations
#   - save every 200 iterations  (--save-interval 200)
#   - eval every 1000 iterations (--eval-interval 1000)
#   - no fault injection / no restart-in-place
#   - ALL --moe-bsr-* switches are disabled (vanilla Megatron-LM training)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/find_max_baseline_long/5.27}"
TORCHRUN="${TORCHRUN:-torchrun}"
if ! command -v "${TORCHRUN}" >/dev/null 2>&1 && [ -x "/Users/zds/miniconda3/bin/torchrun" ]; then
  TORCHRUN="/Users/zds/miniconda3/bin/torchrun"
fi
export SCRIPT_DIR BASE_DIR TORCHRUN

RUN_ID="${RUN_ID:-1}"

run_training() {
  local RUN_ID="${1:?RUN_ID required}"

  local RUN_DIR="${BASE_DIR}/run_${RUN_ID}"
  local CKPT_DIR="${RUN_DIR}/ckpt"
  mkdir -p "${CKPT_DIR}"

  export TRAIN_LOG_DIR="${RUN_DIR}/log"
  export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
  export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
  export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

  # Guard against inherited fault-injection environment from previous runs.
  unset BSR_FAULT_INJECT_TYPE
  unset BSR_FAULT_INJECT_RANK
  unset BSR_FAULT_INJECT_STEP
  unset BSR_FAULT_INJECT_INTERVAL
  unset BSR_FAULT_INJECT_SEED
  unset BSR_FAULT_INJECT_PLAN
  unset BSR_FAULT_INJECT_PLAN_MODE
  unset BSR_FAULT_REPLACEMENT_STEP
  unset BSR_FAULT_REPLACEMENT_RANK
  unset BSR_FAULT_ZERO_MEMORY
  unset BSR_FAULT_MEMORY_FILL
  unset BSR_REQUIRE_OLD_PARAM_RESTORE

  echo "============================================================"
  echo "[findmax_baseline_long] Starting long baseline run=${RUN_ID}"
  echo "[findmax_baseline_long] CKPT_DIR=${CKPT_DIR}"
  echo "[findmax_baseline_long] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
  echo "[findmax_baseline_long] train-iters=10000  save-interval=200  eval-interval=1000"
  echo "[findmax_baseline_long] BSR runtime: DISABLED (vanilla baseline)"
  echo "============================================================"

  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  "${TORCHRUN}" \
    --nproc_per_node=8 \
    --nnodes="${NNODES:-8}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-20115}" \
    ./Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
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

MAX_RETRIES=1
RETRY_DELAY="${RETRY_DELAY:-30}"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"

# ============================================================
# Single run
# ============================================================
echo ""
echo "############################################################"
echo "# findmax_baseline_long: single run=${RUN_ID} (10000 iters)"
echo "############################################################"

RUN_DIR="${BASE_DIR}/run_${RUN_ID}"
export TRAIN_LOG_DIR="${RUN_DIR}/log"
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

retry=0
while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training ${RUN_ID}"
    rc=$?
  else
    echo "[findmax_baseline_long] save_train_log.sh not found, running without log analysis"
    run_training "${RUN_ID}"
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[findmax_baseline_long] run=${RUN_ID} finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[findmax_baseline_long] run=${RUN_ID} failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[findmax_baseline_long] run=${RUN_ID} reached max retries, aborting"
    break
  fi

  sleep "${RETRY_DELAY}"
done

echo ""
echo "============================================================"
echo "findmax_baseline_long: run completed"
echo "Result in: ${BASE_DIR}/run_${RUN_ID}/log/"
echo "============================================================"
