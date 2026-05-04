set -uo pipefail
set -x

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONPATH=$PYTHONPATH:./Megatron-LM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

# ============================================================
# Crash Injection Configuration (checkpoint-restart baseline)
# ============================================================
# This script does NOT use BSR-MoE recovery.  Instead, it uses a
# simple crash injection mechanism: at the configured step the
# process exits, and the outer retry loop restarts training from
# the latest checkpoint.
#
# First step to crash
export CRASH_AT_STEP="${CRASH_AT_STEP:-70}"
# Interval between crashes (0 = single crash only)
export CRASH_INTERVAL="${CRASH_INTERVAL:-40}"
# Which rank to crash (-1 = random rank per crash, seeded; default: -1)
export CRASH_RANK="${CRASH_RANK:--1}"
# Random seed for crash rank selection (ensures reproducible fault sequence)
export CRASH_SEED="${CRASH_SEED:-42}"

# ============================================================
# Log & Analysis Configuration
# ============================================================
# Log directory
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/5.4/baseline/log}"
# Run incremental analysis every N iterations (0 = only at end)
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-100}"
# Run analysis when training ends (1 = yes)
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-1}"
# Path to analysis script (auto-detected from this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

export CKPT_DIR="/mnt/ais-c1/dataset/zds/5.4/baseline"
mkdir -p "${CKPT_DIR}"

MAX_RETRIES=100
RETRY_DELAY=30
retry=0

# ============================================================
# Training command (wrapped in a function for save_train_log.sh)
# ============================================================
run_training() {
  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  torchrun \
    --nproc_per_node=8 \
    --nnodes=${NNODES:-8} \
    --node_rank=${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-6000} \
    ./Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --sequence-parallel \
    --legacy-tokenizer \
    --tokenizer-type GPT2BPETokenizer \
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
    --train-iters 20000 \
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
    --data-path "/mnt/ais-c1/dataset/zds/datasets/64/my_gpt_data_text_document" \
    --split 100,0,0 \
    --save "${CKPT_DIR}" \
    --save-interval 40 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

# ============================================================
# Main loop with retry + log analysis
# ============================================================
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"

while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    # Use save_train_log.sh wrapper for logging + incremental analysis
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
    rc=$?
  else
    # Fallback: direct execution without log wrapper
    echo "[run_moe_baseline] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "training finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "training crashed with exit code ${rc}, restarting from checkpoint... retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "reach max retries, exit"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done
