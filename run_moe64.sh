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
# MOEGAMBIT-MoE Fault Injection Configuration (via environment variables)
# ============================================================
# Fault injection type: "quarantine", "hard_failure", or "restart_in_place"
#   - quarantine:         soft fault (rank still alive but isolated)
#   - hard_failure:       hard fault (rank cannot participate in collectives)
#   - restart_in_place:   in-place restart simulation (replacement=self);
#                         with dense-param-sync + stale-expert-restore below,
#                         recovery follows the HYBRID_RECOVERY path.
export MOEGAMBIT_FAULT_INJECT_TYPE="${MOEGAMBIT_FAULT_INJECT_TYPE:-restart_in_place}"
# Which rank to inject the fault on (0-based global rank)
# Set to -1 to enable random rank selection per fault (seeded)
export MOEGAMBIT_FAULT_INJECT_RANK="${MOEGAMBIT_FAULT_INJECT_RANK:--1}"
# First training step to inject the fault
export MOEGAMBIT_FAULT_INJECT_STEP="${MOEGAMBIT_FAULT_INJECT_STEP:-70}"
# Interval between repeated fault injections (0 = single injection only)
export MOEGAMBIT_FAULT_INJECT_INTERVAL="${MOEGAMBIT_FAULT_INJECT_INTERVAL:-40}"
# Random seed for fault rank selection (ensures reproducible fault sequence)
export MOEGAMBIT_FAULT_INJECT_SEED="${MOEGAMBIT_FAULT_INJECT_SEED:-42}"
# At which training step the replacement rank becomes ready
# (ignored for restart_in_place mode — replacement is immediate)
export MOEGAMBIT_FAULT_REPLACEMENT_STEP="${MOEGAMBIT_FAULT_REPLACEMENT_STEP:-70}"
# Replacement rank ID (-1 = auto-assign; for restart_in_place, always = failed_rank)
export MOEGAMBIT_FAULT_REPLACEMENT_RANK="${MOEGAMBIT_FAULT_REPLACEMENT_RANK:--1}"
# Simulate a device whose model/optimizer memory comes back zeroed.
export MOEGAMBIT_FAULT_ZERO_MEMORY="${MOEGAMBIT_FAULT_ZERO_MEMORY:-1}"
export MOEGAMBIT_FAULT_MEMORY_FILL="${MOEGAMBIT_FAULT_MEMORY_FILL:-zero}"

# ============================================================
# Log & Analysis Configuration
# ============================================================
# Log directory
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/5.13/2}"
# Run incremental analysis every N iterations (0 = disabled; final analysis still runs)
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
# Run analysis when training ends (1 = yes; 0 = disabled)
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
# Path to analysis script (auto-detected from this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/5.13/bsr/2}"


mkdir -p "${CKPT_DIR}"

MAX_RETRIES=1
RETRY_DELAY="${RETRY_DELAY:-30}"
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
    --master_port=${MASTER_PORT:-20115} \
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
    --moe-moegambit-enable \
    --moe-moegambit-health-mask \
    --moe-moegambit-rank-quarantine \
    --moe-moegambit-dispatch-quarantine-assert \
    --moe-moegambit-dispatch-sanitize \
    --moe-moegambit-expert-directory \
    --moe-moegambit-replacement-protocol \
    --moe-moegambit-group-rebuild \
    --moe-moegambit-dispatch-topology-refresh \
    --moe-moegambit-dense-param-sync \
    --moe-moegambit-stale-expert-restore \
    --moe-moegambit-recovery-controller \
    --moe-moegambit-deferred-optimizer-load \
    --moe-moegambit-degraded-mode-policy \
    --moe-moegambit-reintegration-barrier \
    --moe-moegambit-fault-injection \
    --moe-moegambit-restart-in-place \
    --moe-moegambit-degraded-tau-c 0.5 \
    --moe-moegambit-degraded-t-max 1000 \
    --moe-moegambit-degraded-s-max 500 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 100,0,0 \
    --ckpt-format torch \
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
    echo "[run_moe] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "training finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "training failed with exit code ${rc}, retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "reach max retries, exit"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done
