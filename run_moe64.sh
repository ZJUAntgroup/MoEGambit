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
# BSR-MoE Fault Injection Configuration (via environment variables)
# ============================================================
# Fault injection type: "quarantine", "hard_failure", or "restart_in_place"
#   - quarantine:         soft fault (rank still alive but isolated)
#   - hard_failure:       hard fault (rank cannot participate in collectives)
#   - restart_in_place:   NaN-sentinel restart simulation (replacement=self)
export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
# Which rank to inject the fault on (0-based global rank)
# Set to -1 to enable random rank selection per fault (seeded)
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:--1}"
# At which training step to inject the fault
export BSR_FAULT_INJECT_STEP="${BSR_FAULT_INJECT_STEP:-70}"
# Interval between repeated fault injections (0 = single injection only)
export BSR_FAULT_INJECT_INTERVAL="${BSR_FAULT_INJECT_INTERVAL:-40}"
# Random seed for fault rank selection (ensures reproducible fault sequence)
export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
# At which training step the replacement rank becomes ready
# (ignored for restart_in_place mode — replacement is immediate)
export BSR_FAULT_REPLACEMENT_STEP="${BSR_FAULT_REPLACEMENT_STEP:-70}"
# Replacement rank ID (-1 = auto-assign; for restart_in_place, always = failed_rank)
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"

# ============================================================
# Log & Analysis Configuration
# ============================================================
# Log directory
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/5.9}"
# Run incremental analysis every N iterations (0 = only at end)
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-100}"
# Run analysis when training ends (1 = yes)
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-1}"
# Path to analysis script (auto-detected from this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

export CKPT_DIR="/mnt/ais-c1/dataset/zds/5.9/bsr"


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
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 1000 \
    --moe-bsr-degraded-s-max 500 \
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
