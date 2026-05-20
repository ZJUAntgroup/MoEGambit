#!/usr/bin/env bash
# BSR-MoE recovery ablation runner.
#
# Goal:
#   Separate the recovery-time gains from:
#     1. selective stale-expert restore instead of full checkpoint load;
#     2. weights-first recovery with optimizer state loaded after weights;
#     3. the combined effect.
#
# Default run order:
#   selective_deferred   : dense from peer + stale experts from checkpoint,
#                          expert weights first, optimizer state deferred.
#   selective_sync_opt   : dense from peer + stale experts from checkpoint,
#                          and expert optimizer state loaded synchronously in
#                          the safe-point critical path.
#   full_checkpoint      : baseline; BSR safe-point repair, but force full
#                          checkpoint load for model + optimizer in the
#                          critical path.  Run last by default.
#
# Attribution from mean recovery/fault-window times:
#   selective_restore_gain = full_checkpoint - selective_sync_opt
#   optimizer_defer_gain   = selective_sync_opt - selective_deferred
#   total_gain             = full_checkpoint - selective_deferred

set -uo pipefail
set -x

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Keep ablation logs parseable. Use ABLATION_NCCL_DEBUG=INFO only when
# debugging NCCL bring-up.
export NCCL_DEBUG="${ABLATION_NCCL_DEBUG:-WARN}"
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"
TIMING_REPORT_SCRIPT="${SCRIPT_DIR}/log_analysis/bsr_timing_report.py"

# Keep all ablation outputs isolated from the main/baseline runs.
export ABLATION_ROOT="${ABLATION_ROOT:-/mnt/ais-c1/dataset/zds/5.18/ablation}"
export ABLATION_CKPT_ROOT="${ABLATION_CKPT_ROOT:-${ABLATION_ROOT}/ckpt}"
export ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-/mnt/ais-c1/dataset/zds/log/5.18_ablation}"
export ABLATION_MODES="${ABLATION_MODES:-selective_deferred selective_sync_opt full_checkpoint}"

# Default: 50 faults per mode.
# Faults at 70 + k*40 for k=0..49, so the last fault is at iteration 2030.
# Train to 2060 to leave a short post-recovery window after the last fault.
export ABLATION_TRAIN_ITERS="${ABLATION_TRAIN_ITERS:-2060}"
export ABLATION_SAVE_INTERVAL="${ABLATION_SAVE_INTERVAL:-40}"
export ABLATION_FAULT_STEP="${ABLATION_FAULT_STEP:-70}"
export ABLATION_FAULT_INTERVAL="${ABLATION_FAULT_INTERVAL:-40}"
export ABLATION_FAULT_RANK="${ABLATION_FAULT_RANK:--1}"
export ABLATION_FAULT_SEED="${ABLATION_FAULT_SEED:-42}"
export ABLATION_FAULT_TYPE="${ABLATION_FAULT_TYPE:-restart_in_place}"
export ABLATION_MAX_RETRIES="${ABLATION_MAX_RETRIES:-1}"
export RETRY_DELAY="${RETRY_DELAY:-30}"

# Log analysis is run once at mode end.  The old every-N-iteration analysis
# would perturb timings, so keep it disabled during the experiment.
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

case "${ABLATION_TRAIN_ITERS}" in
  ''|*[!0-9]*)
    echo "[ablation] invalid ABLATION_TRAIN_ITERS=${ABLATION_TRAIN_ITERS}" >&2
    exit 2
    ;;
esac

case "${ABLATION_SAVE_INTERVAL}" in
  ''|*[!0-9]*)
    echo "[ablation] invalid ABLATION_SAVE_INTERVAL=${ABLATION_SAVE_INTERVAL}" >&2
    exit 2
    ;;
esac

mode_args() {
  local mode="$1"
  case "${mode}" in
    full_checkpoint)
      # Full model+optimizer checkpoint load in the BSR critical path.
      printf '%s\n' \
        --moe-bsr-force-checkpoint-restart \
        --moe-bsr-hybrid-expert-restore \
        --moe-bsr-expert-opt-restore \
        --moe-bsr-weights-first-recovery \
        --no-moe-bsr-defer-optimizer-load
      ;;
    selective_sync_opt)
      # Selective stale-expert restore, but optimizer state is NOT deferred.
      printf '%s\n' \
        --moe-bsr-hybrid-expert-restore \
        --moe-bsr-expert-opt-restore \
        --moe-bsr-weights-first-recovery \
        --no-moe-bsr-defer-optimizer-load
      ;;
    selective_deferred)
      # Production BSR path: weights first, optimizer state second/deferred.
      printf '%s\n' \
        --moe-bsr-hybrid-expert-restore \
        --moe-bsr-expert-opt-restore \
        --moe-bsr-weights-first-recovery \
        --moe-bsr-defer-optimizer-load
      ;;
    selective_weights_only)
      # Optional diagnostic: measures expert-weight restore without optimizer
      # state restoration.  Do not use as the main correctness result.
      printf '%s\n' \
        --moe-bsr-hybrid-expert-restore \
        --no-moe-bsr-expert-opt-restore \
        --moe-bsr-weights-first-recovery \
        --moe-bsr-defer-optimizer-load
      ;;
    *)
      echo "[ablation] unknown mode: ${mode}" >&2
      return 1
      ;;
  esac
}

run_training() {
  local mode="$1"
  shift
  local mode_specific_args=("$@")

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
    --train-iters "${ABLATION_TRAIN_ITERS}" \
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
    "${mode_specific_args[@]}" \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 100,0,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval "${ABLATION_SAVE_INTERVAL}" \
    --eval-interval 1000 \
    --eval-iters 0 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

run_one_mode() {
  local mode="$1"
  local -a MODE_ARGS
  mapfile -t MODE_ARGS < <(mode_args "${mode}") || return 2

  export CKPT_DIR="${ABLATION_CKPT_ROOT}/${mode}"
  export TRAIN_LOG_DIR="${ABLATION_LOG_ROOT}/${mode}"
  export TRAIN_RUN_ID="${mode}_$(date +%Y%m%d_%H%M%S)_${NODE_RANK:-0}"

  export BSR_FAULT_INJECT_TYPE="${ABLATION_FAULT_TYPE}"
  export BSR_FAULT_INJECT_RANK="${ABLATION_FAULT_RANK}"
  export BSR_FAULT_INJECT_STEP="${ABLATION_FAULT_STEP}"
  export BSR_FAULT_INJECT_INTERVAL="${ABLATION_FAULT_INTERVAL}"
  export BSR_FAULT_INJECT_SEED="${ABLATION_FAULT_SEED}"
  export BSR_FAULT_REPLACEMENT_STEP="${ABLATION_FAULT_STEP}"
  export BSR_FAULT_REPLACEMENT_RANK="-1"
  export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
  export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"

  mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"
  rm -f "${TRAIN_LOG_DIR}/train_latest.log" "${TRAIN_LOG_DIR}/train_full.log" \
        "${TRAIN_LOG_DIR}/train_latest.status" 2>/dev/null || true

  echo "[ablation] mode=${mode}"
  echo "[ablation] ckpt=${CKPT_DIR}"
  echo "[ablation] log=${TRAIN_LOG_DIR}"
  echo "[ablation] args=${MODE_ARGS[*]}"
  export TRAIN_LAUNCH_DESC="run_moe64_ablation mode=${mode} args=${MODE_ARGS[*]}"

  local retry=0
  while true; do
    if [ -f "${SAVE_LOG_SCRIPT}" ]; then
      bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training \"\$@\"" bash "${mode}" "${MODE_ARGS[@]}"
      rc=$?
    else
      run_training "${mode}" "${MODE_ARGS[@]}"
      rc=$?
    fi

    if [ "${rc}" -eq 0 ]; then
      echo "[ablation] mode=${mode} finished normally"
      break
    fi

    retry=$((retry + 1))
    echo "[ablation] mode=${mode} failed with exit code ${rc}, retry=${retry}/${ABLATION_MAX_RETRIES}"

    if [ "${retry}" -ge "${ABLATION_MAX_RETRIES}" ]; then
      echo "[ablation] mode=${mode} reached max retries"
      return "${rc}"
    fi

    sleep "${RETRY_DELAY}"
  done

  if [ "${NODE_RANK:-0}" = "0" ] && [ -f "${TIMING_REPORT_SCRIPT}" ]; then
    python3 "${TIMING_REPORT_SCRIPT}" "${TRAIN_LOG_DIR}/train_full.log" \
      --top 5 \
      --summary-csv "${TRAIN_LOG_DIR}/timing_summary.csv" \
      --fault-csv "${TRAIN_LOG_DIR}/fault_windows.csv" \
      --json "${TRAIN_LOG_DIR}/timing_report.json" || true
  fi
}

for mode in ${ABLATION_MODES}; do
  run_one_mode "${mode}" || exit $?
done

if [ "${NODE_RANK:-0}" = "0" ] && [ -f "${SCRIPT_DIR}/log_analysis/bsr_ablation_report.py" ]; then
  python3 "${SCRIPT_DIR}/log_analysis/bsr_ablation_report.py" "${ABLATION_LOG_ROOT}" \
    --json "${ABLATION_LOG_ROOT}/ablation_report.json" \
    --csv "${ABLATION_LOG_ROOT}/ablation_report.csv" || true
fi

echo "[ablation] all modes finished: ${ABLATION_MODES}"
