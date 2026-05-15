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
# Experiment: sweep fault injection step in [150, 199]
# ============================================================
# Each run: 600 iters total, first 200 iters normal training,
# then inject a single hard_failure at a specific step and
# test hybrid recovery (stale expert restore + dense sync from peer).
#
# Fault injection points: step 150, 151, ..., 199 (50 runs)
#
# Each run gets its own checkpoint & log directory:
#   /mnt/ais-c1/dataset/zds/find_max/step_150/
#   /mnt/ais-c1/dataset/zds/find_max/step_151/
#   ...
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/mnt/ais-c1/dataset/zds/find_max/5.15"

# Fault injection steps to sweep: 150 to 199
FAULT_STEPS=($(seq 150 199))

# ============================================================
# Training function (parameterized by FAULT_STEP)
# ============================================================
run_training() {
  local FAULT_STEP="${1:?FAULT_STEP required}"

  local RUN_DIR="${BASE_DIR}/step_${FAULT_STEP}"
  local CKPT_DIR="${RUN_DIR}/ckpt"
  mkdir -p "${CKPT_DIR}"

  export TRAIN_LOG_DIR="${RUN_DIR}/log"
  export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
  export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
  export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

  # Fault injection config
  export BSR_FAULT_INJECT_TYPE="hard_failure"
  export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:--1}"
  export BSR_FAULT_INJECT_STEP="${FAULT_STEP}"
  export BSR_FAULT_INJECT_INTERVAL="0"
  export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
  export BSR_FAULT_REPLACEMENT_STEP="${FAULT_STEP}"
  export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"

  echo "============================================================"
  echo "[find_max] Starting run: fault_inject_step=${FAULT_STEP}"
  echo "[find_max] CKPT_DIR=${CKPT_DIR}"
  echo "[find_max] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
  echo "============================================================"

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
    --train-iters 600 \
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
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 10000 \
    --moe-bsr-degraded-s-max 5000 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 50 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

# ============================================================
# Baseline: no fault injection, 600 iters
# ============================================================
run_baseline() {
  local RUN_DIR="${BASE_DIR}/baseline"
  local CKPT_DIR="${RUN_DIR}/ckpt"
  mkdir -p "${CKPT_DIR}"

  export TRAIN_LOG_DIR="${RUN_DIR}/log"
  export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
  export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
  export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

  # Disable fault injection
  unset BSR_FAULT_INJECT_TYPE
  unset BSR_FAULT_INJECT_STEP
  unset BSR_FAULT_INJECT_INTERVAL
  unset BSR_FAULT_REPLACEMENT_STEP

  echo "============================================================"
  echo "[find_max] Starting baseline run (no fault injection)"
  echo "[find_max] CKPT_DIR=${CKPT_DIR}"
  echo "[find_max] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
  echo "============================================================"

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
    --train-iters 600 \
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
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 10000 \
    --moe-bsr-degraded-s-max 5000 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 50 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

MAX_RETRIES=1
RETRY_DELAY=30
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"

# ============================================================
# Run baseline first
# ============================================================
echo ""
echo "############################################################"
echo "# find_max: baseline (no fault injection)"
echo "############################################################"

retry=0
while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_baseline); run_baseline"
    rc=$?
  else
    echo "[find_max] save_train_log.sh not found, running without log analysis"
    run_baseline
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[find_max] baseline finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[find_max] baseline failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[find_max] baseline reached max retries, aborting"
    break
  fi

  sleep "${RETRY_DELAY}"
done

echo "[find_max] baseline done, starting fault injection sweep"

# ============================================================
# Sweep loop
# ============================================================
for STEP in "${FAULT_STEPS[@]}"; do
  echo ""
  echo "############################################################"
  echo "# find_max: fault_inject_step=${STEP}"
  echo "############################################################"

  retry=0
  while true; do
    if [ -f "${SAVE_LOG_SCRIPT}" ]; then
      bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training ${STEP}"
      rc=$?
    else
      echo "[find_max] save_train_log.sh not found, running without log analysis"
      run_training "${STEP}"
      rc=$?
    fi

    if [ $rc -eq 0 ]; then
      echo "[find_max] step=${STEP} finished normally"
      break
    fi

    retry=$((retry + 1))
    echo "[find_max] step=${STEP} failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

    if [ $retry -ge $MAX_RETRIES ]; then
      echo "[find_max] step=${STEP} reached max retries, aborting this run"
      break
    fi

    sleep "${RETRY_DELAY}"
  done

  echo "[find_max] step=${STEP} done (or aborted), moving to next experiment"
done

echo ""
echo "============================================================"
echo "find_max: all experiments completed"
echo "Fault injection steps tested: ${FAULT_STEPS[*]}"
echo "Results in: ${BASE_DIR}/step_*/log/"
echo "============================================================"
