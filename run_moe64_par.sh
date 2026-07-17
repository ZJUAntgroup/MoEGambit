set -uo pipefail
set -x

# =============================================================================
# 64-GPU parallelism-sensitivity runner: unifies MoEGambit (MOEGAMBIT-MoE recovery)
# and Restart (baseline) into a single script with configurable parallelism.
#
# Select mode via the MODE environment variable:
#   MODE=moegambit (default)  -> MOEGAMBIT-MoE hybrid recovery stack
#   MODE=baseline             -> plain checkpoint-restart loop
#
# Parallelism is configured via environment variables:
#   TP_SIZE  (default 1)
#   PP_SIZE  (default 8)
#   EP_SIZE  (default 8)
#   EDP is derived: EDP = world_size / (TP_SIZE * PP_SIZE * EP_SIZE)
#
# When EDP > 1, expert parameters can be pulled from a healthy DP peer
# instead of loading from checkpoint.  Set MOEGAMBIT_FULL_PEER_RECOVERY=1 to
# enable the FULL_PEER_RECOVERY path (all params from peer, no checkpoint).
#
# For the parallelism-sensitivity benchmark, see bench_moe64_tp*.sh which
# invoke this script with different TP/PP/EP settings.
#
# 64-GPU layout: 8 nodes x 8 GPUs.
# Set NNODES=8 (default) and NODE_RANK / MASTER_ADDR / MASTER_PORT on launch.
# =============================================================================


export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

# Mode switch
MODE="${MODE:-moegambit}"
case "${MODE}" in
  moegambit|baseline) ;;
  *)
    echo "[run_moe64_par] invalid MODE=${MODE}, must be 'moegambit' or 'baseline'"
    exit 2
    ;;
esac

# 64-GPU defaults (override on launch if topology differs)
export NNODES="${NNODES:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"
export NODE_RANK="${NODE_RANK:-0}"

# Parallelism: configurable via environment variables.
[ -z "${TP_SIZE:-}" ] && unset TP_SIZE
[ -z "${PP_SIZE:-}" ] && unset PP_SIZE
[ -z "${EP_SIZE:-}" ] && unset EP_SIZE
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-40}"

# Compute EDP for logging (EDP = world_size / (TP * PP * EP))
WORLD_SIZE=$((NNODES * 8))
EDP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE * EP_SIZE)))
echo "[run_moe64_par] parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, EDP=${EDP_SIZE}, world_size=${WORLD_SIZE}"

# Global batch size: scale with DP degree.
# DP = world_size / (TP * PP) = EP * EDP.
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
export GLOBAL_BATCH_SIZE

# Full peer recovery: when EDP > 1, all params (dense + expert) can be
# pulled from a healthy DP peer, avoiding checkpoint I/O entirely.
MOEGAMBIT_FULL_PEER_RECOVERY="${MOEGAMBIT_FULL_PEER_RECOVERY:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Mode-specific configuration
# =============================================================================
if [ "${MODE}" = "moegambit" ]; then
  # ---------- MoEGambit (MOEGAMBIT-MoE hybrid recovery) ----------
  export MOEGAMBIT_FAULT_INJECT_TYPE="${MOEGAMBIT_FAULT_INJECT_TYPE:-restart_in_place}"
  export MOEGAMBIT_FAULT_INJECT_RANK="${MOEGAMBIT_FAULT_INJECT_RANK:--1}"
  export MOEGAMBIT_FAULT_INJECT_STEP="${MOEGAMBIT_FAULT_INJECT_STEP:-70}"
  export MOEGAMBIT_FAULT_INJECT_INTERVAL="${MOEGAMBIT_FAULT_INJECT_INTERVAL:-40}"
  export MOEGAMBIT_FAULT_INJECT_SEED="${MOEGAMBIT_FAULT_INJECT_SEED:-42}"
  export MOEGAMBIT_FAULT_REPLACEMENT_STEP="${MOEGAMBIT_FAULT_REPLACEMENT_STEP:-70}"
  export MOEGAMBIT_FAULT_REPLACEMENT_RANK="${MOEGAMBIT_FAULT_REPLACEMENT_RANK:--1}"
  export MOEGAMBIT_FAULT_ZERO_MEMORY="${MOEGAMBIT_FAULT_ZERO_MEMORY:-1}"
  export MOEGAMBIT_FAULT_MEMORY_FILL="${MOEGAMBIT_FAULT_MEMORY_FILL:-zero}"

  export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/64gpu_par/moegambit}"
  export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/64gpu_par/moegambit}"
  MAX_RETRIES="${MAX_RETRIES:-1}"
else
  # ---------- Baseline (checkpoint-restart loop) ----------
  export CRASH_AT_STEP="${CRASH_AT_STEP:-70}"
  export CRASH_INTERVAL="${CRASH_INTERVAL:-80}"
  export CRASH_RANK="${CRASH_RANK:--1}"
  export CRASH_SEED="${CRASH_SEED:-42}"
  BASE_CRASH_RANK="${CRASH_RANK}"
  NEXT_CRASH_STEP="${CRASH_AT_STEP}"
  CRASH_INJECT_INDEX=0

  # Isolate baseline from MOEGAMBIT-MoE recovery stack
  unset MOEGAMBIT_FAULT_INJECT_TYPE
  unset MOEGAMBIT_FAULT_INJECT_RANK
  unset MOEGAMBIT_FAULT_INJECT_STEP
  unset MOEGAMBIT_FAULT_INJECT_INTERVAL
  unset MOEGAMBIT_FAULT_INJECT_SEED
  unset MOEGAMBIT_FAULT_REPLACEMENT_STEP
  unset MOEGAMBIT_FAULT_REPLACEMENT_RANK
  unset MOEGAMBIT_FAULT_ZERO_MEMORY
  unset MOEGAMBIT_FAULT_MEMORY_FILL

  export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/64gpu_par/baseline}"
  export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/64gpu_par/baseline}"
  MAX_RETRIES="${MAX_RETRIES:-300}"
fi

# Common log/analysis settings
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"
RETRY_DELAY="${RETRY_DELAY:-30}"
retry=0

mkdir -p "${CKPT_DIR}"

# =============================================================================
# Baseline-only helper functions (no-op for moegambit mode)
# =============================================================================
if [ "${MODE}" = "baseline" ]; then
  case "${CRASH_INTERVAL}" in
    ''|*[!0-9]*)
      echo "[run_moe64_par] invalid CRASH_INTERVAL=${CRASH_INTERVAL}, fallback to 80"
      export CRASH_INTERVAL=80
      ;;
  esac
  case "${NEXT_CRASH_STEP}" in
    -1|''|*[!0-9-]*)
      if [ "${NEXT_CRASH_STEP}" != "-1" ]; then
        echo "[run_moe64_par] invalid CRASH_AT_STEP=${NEXT_CRASH_STEP}, fallback to 70"
        NEXT_CRASH_STEP=70
      fi
      ;;
  esac
  case "${MAX_RETRIES}" in
    ''|*[!0-9]*)
      echo "[run_moe64_par] invalid MAX_RETRIES=${MAX_RETRIES}, fallback to 300"
      MAX_RETRIES=300
      ;;
  esac
  case "${RETRY_DELAY}" in
    ''|*[!0-9]*)
      echo "[run_moe64_par] invalid RETRY_DELAY=${RETRY_DELAY}, fallback to 30"
      RETRY_DELAY=30
      ;;
  esac
  case "${BASE_CRASH_RANK}" in
    -1|''|*[!0-9-]*)
      if [ "${BASE_CRASH_RANK}" != "-1" ]; then
        echo "[run_moe64_par] invalid CRASH_RANK=${BASE_CRASH_RANK}, fallback to random"
        BASE_CRASH_RANK=-1
      fi
      ;;
  esac

  select_random_crash_rank() {
    local index="$1"
    local nnodes="${NNODES:-8}"
    local world_size="${CRASH_WORLD_SIZE:-$((8 * nnodes))}"

    python3 - "${CRASH_SEED}" "${index}" "${world_size}" <<'PY'
import random
import sys

seed = int(sys.argv[1])
index = int(sys.argv[2])
world_size = int(sys.argv[3])
if world_size <= 0:
    raise SystemExit("world_size must be positive")

rng = random.Random(seed)
rank = 0
for _ in range(index + 1):
    rank = rng.choice(range(world_size))
print(rank)
PY
  }

  prepare_next_crash_injection() {
    export CRASH_AT_STEP="${NEXT_CRASH_STEP}"
    if [ "${BASE_CRASH_RANK}" -lt 0 ]; then
      export CRASH_RANK="$(select_random_crash_rank "${CRASH_INJECT_INDEX}")"
      echo "[run_moe64_par] next checkpoint-restart fault: step=${CRASH_AT_STEP}, random_rank=${CRASH_RANK}, seed=${CRASH_SEED}, index=${CRASH_INJECT_INDEX}, interval=${CRASH_INTERVAL}"
    else
      export CRASH_RANK="${BASE_CRASH_RANK}"
      echo "[run_moe64_par] next checkpoint-restart fault: step=${CRASH_AT_STEP}, rank=${CRASH_RANK}, interval=${CRASH_INTERVAL}"
    fi
  }

  advance_next_crash_injection() {
    if [ "${CRASH_INTERVAL}" -gt 0 ] && [ "${NEXT_CRASH_STEP}" -ge 0 ]; then
      NEXT_CRASH_STEP=$((NEXT_CRASH_STEP + CRASH_INTERVAL))
      CRASH_INJECT_INDEX=$((CRASH_INJECT_INDEX + 1))
    else
      NEXT_CRASH_STEP=-1
    fi
  }
fi

# =============================================================================
# Training command
# =============================================================================
run_training() {
  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  MOEGAMBIT_ARGS=()
  if [ "${MODE}" = "moegambit" ]; then
    MOEGAMBIT_ARGS=(
      --moe-moegambit-enable
      --moe-moegambit-health-mask
      --moe-moegambit-rank-quarantine
      --moe-moegambit-dispatch-quarantine-assert
      --moe-moegambit-dispatch-sanitize
      --moe-moegambit-expert-directory
      --moe-moegambit-replacement-protocol
      --moe-moegambit-group-rebuild
      --moe-moegambit-dispatch-topology-refresh
      --moe-moegambit-dense-param-sync
      --moe-moegambit-stale-expert-restore
      --moe-moegambit-recovery-controller
      --moe-moegambit-deferred-optimizer-load
      --moe-moegambit-degraded-mode-policy
      --moe-moegambit-reintegration-barrier
      --moe-moegambit-fault-injection
      --moe-moegambit-restart-in-place
      --moe-moegambit-degraded-tau-c 0.5
      --moe-moegambit-degraded-t-max 1000
      --moe-moegambit-degraded-s-max 500
    )

    # When EDP > 1 and full peer recovery is requested, all params
    # (dense + expert weights + optimizer state) are pulled from a
    # healthy DP peer instead of loading from checkpoint.
    if [ "${MOEGAMBIT_FULL_PEER_RECOVERY}" = "1" ]; then
      MOEGAMBIT_ARGS+=(
        --moe-moegambit-full-peer-recovery
      )
    fi

    # Gap-aware hybrid recovery policy: uses Φ'(t) staleness guard
    # to choose between hybrid recovery and checkpoint restart.
    if [ "${MOEGAMBIT_GAP_AWARE_RECOVERY:-0}" = "1" ]; then
      MOEGAMBIT_ARGS+=(
        --moe-moegambit-gap-aware-recovery
        --moe-moegambit-recovery-policy-type "${MOEGAMBIT_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
        --moe-moegambit-gap-threshold "${MOEGAMBIT_GAP_THRESHOLD:-100}"
      )
    fi

    # Hot-spare node pool: pre-launched spare GPU ranks for instant
    # fault replacement. Spares do NOT join training NCCL groups
    # until activated at a safe-point.
    if [ "${MOEGAMBIT_HOT_SPARE_POOL:-0}" = "1" ]; then
      MOEGAMBIT_ARGS+=(
        --moe-moegambit-hot-spare-pool
        --moe-moegambit-num-hot-spares "${MOEGAMBIT_NUM_HOT_SPARES:-8}"
      )
    fi
  fi

  torchrun \
    --nproc_per_node=8 \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ./Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size "${TP_SIZE}" \
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
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" \
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
    "${MOEGAMBIT_ARGS[@]}" \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 100,0,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
    --eval-interval 1000 \
    --eval-iters 0 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

# =============================================================================
# Main loop
# =============================================================================
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"

while true; do
  if [ "${MODE}" = "baseline" ]; then
    prepare_next_crash_injection
  fi

  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
    rc=$?
  else
    echo "[run_moe64_par] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "training finished normally"
    break
  fi

  retry=$((retry + 1))
  if [ "${MODE}" = "baseline" ]; then
    echo "training crashed with exit code ${rc}, restarting from checkpoint... retry=${retry}/${MAX_RETRIES}"
    advance_next_crash_injection
  else
    echo "training failed with exit code ${rc}, retry=${retry}/${MAX_RETRIES}"
  fi

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "reach max retries, exit"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done
