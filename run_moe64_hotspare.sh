#!/usr/bin/env bash
# =============================================================================
# run_moe64_hotspare.sh
#
# Hot-spare node replacement for MoEGambit training (FlashRecovery style).
#
# Architecture:
#   - 8 training nodes (64 GPUs): custom launcher with world_size=64
#   - 1 spare node (8 GPUs): runs elastic_watcher.py (NOT in torch.distributed)
#   - On fault: surviving ranks pause at safe point → destroy_process_group →
#     spare node replaces faulty node → all re-init_process_group(world_size=64)
#     → new rank recovers params from DP peer → training continues (RPO=0)
#
# Why NOT torchrun:
#   torchrun's elastic agent kills ALL workers on a node when ANY worker exits.
#   This prevents the "hot rebuild" approach where surviving processes stay alive
#   and rebuild communication groups. Our custom elastic_launcher.py does NOT
#   auto-kill workers on peer failure.
#
# Deployment:
#   Training nodes: NODE_RANK=0..7 bash run_moe64_hotspare.sh
#   Spare node:     NODE_RANK=8   bash run_moe64_hotspare.sh
#
# The spare node does NOT run training. It runs elastic_watcher.py which:
#   - Monitors training ranks via heartbeat (TCP)
#   - On fault detection: signals surviving ranks to pause
#   - Launches elastic_launcher.py on itself with the faulty node's NODE_RANK
#   - Coordinates the group rebuild
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================

# Parallelism
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export MODE=moegambit

# Topology
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"

# Watcher coordination port (spare node listens, training ranks connect)
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"

# Determine role: NODE_RANK 0-7 = training, NODE_RANK 8 = spare
TRAINING_NNODES=8
if [ "${NODE_RANK}" -ge "${TRAINING_NNODES}" ]; then
  IS_SPARE=1
else
  IS_SPARE=0
fi

# For training nodes: standard 8-node setup
export NNODES="${TRAINING_NNODES}"

# Elastic recovery env vars (consumed by elastic_client.py in training code)
# ELASTIC_WATCHER_ADDR: the spare node's IP where watcher listens
# In typical deployment, spare node IP is passed via this env var.
# If spare is on a different machine, set ELASTIC_WATCHER_ADDR to its IP.
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"
rm -rf "${ELASTIC_FAULT_DIR}"
mkdir -p "${ELASTIC_FAULT_DIR}"

# Recovery policy
export MOEGAMBIT_HOT_SPARE_POOL=1
export MOEGAMBIT_NUM_HOT_SPARES="${NPROC_PER_NODE}"
export MOEGAMBIT_GAP_AWARE_RECOVERY=1
export MOEGAMBIT_RECOVERY_POLICY_TYPE="${MOEGAMBIT_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export MOEGAMBIT_GAP_THRESHOLD="${MOEGAMBIT_GAP_THRESHOLD:-100}"

# Fault injection
export MOEGAMBIT_FAULT_INJECT_TYPE="${MOEGAMBIT_FAULT_INJECT_TYPE:-restart_in_place}"
export MOEGAMBIT_FAULT_INJECT_RANK="${MOEGAMBIT_FAULT_INJECT_RANK:--1}"
export MOEGAMBIT_FAULT_INJECT_STEP="${MOEGAMBIT_FAULT_INJECT_STEP:-70}"
export MOEGAMBIT_FAULT_INJECT_INTERVAL="${MOEGAMBIT_FAULT_INJECT_INTERVAL:-40}"
export MOEGAMBIT_FAULT_INJECT_SEED="${MOEGAMBIT_FAULT_INJECT_SEED:-42}"
export MOEGAMBIT_FAULT_REPLACEMENT_STEP="${MOEGAMBIT_FAULT_REPLACEMENT_STEP:-70}"
export MOEGAMBIT_FAULT_REPLACEMENT_RANK="${MOEGAMBIT_FAULT_REPLACEMENT_RANK:--1}"
export MOEGAMBIT_FAULT_ZERO_MEMORY="${MOEGAMBIT_FAULT_ZERO_MEMORY:-1}"
export MOEGAMBIT_FAULT_MEMORY_FILL="${MOEGAMBIT_FAULT_MEMORY_FILL:-zero}"

# Checkpoint & logging
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/hotspare/615ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/hotspare}"

# Environment
export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH:-}:./src:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-40}"

mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# =============================================================================
# Compute derived values
# =============================================================================

TRAINING_WORLD_SIZE=$((TRAINING_NNODES * NPROC_PER_NODE))  # 64
DP_SIZE=$((TRAINING_WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
export GLOBAL_BATCH_SIZE

echo "[hotspare] =============================================="
echo "[hotspare] MoEGambit Hot-Spare (FlashRecovery style)"
echo "[hotspare] =============================================="
echo "[hotspare] NODE_RANK:       ${NODE_RANK} $([ "${IS_SPARE}" = "1" ] && echo "(SPARE WATCHER)" || echo "(TRAINING)")"
echo "[hotspare] Training nodes:  ${TRAINING_NNODES} (${TRAINING_WORLD_SIZE} GPUs)"
echo "[hotspare] Spare node:      1 (${NPROC_PER_NODE} GPUs, standby)"
echo "[hotspare] Parallelism:     TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, DP=${DP_SIZE}"
echo "[hotspare] Global batch:    ${GLOBAL_BATCH_SIZE}"
echo "[hotspare] Launcher:        elastic_launcher.py (no auto-kill on peer failure)"
echo "[hotspare] CKPT_DIR:        ${CKPT_DIR}"
echo "[hotspare] =============================================="

# =============================================================================
# SPARE NODE: run watcher (does NOT join torch.distributed)
# =============================================================================

if [ "${IS_SPARE}" = "1" ]; then
  echo "[hotspare] Starting elastic watcher on spare node..."
  python3 "${SCRIPT_DIR}/elastic_watcher.py" \
    --port "${ELASTIC_WATCHER_PORT}" \
    --training-nnodes "${TRAINING_NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --fault-dir "${ELASTIC_FAULT_DIR}"
  exit $?
fi

# =============================================================================
# TRAINING NODE: custom launcher (no torchrun — workers survive peer failures)
# =============================================================================

LOAD_ARGS=()
if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
  LOAD_ARGS=(--load "${CKPT_DIR}")
fi

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
  --moe-moegambit-hot-spare-pool
  --moe-moegambit-num-hot-spares "${NPROC_PER_NODE}"
)

if [ "${MOEGAMBIT_GAP_AWARE_RECOVERY:-0}" = "1" ]; then
  MOEGAMBIT_ARGS+=(
    --moe-moegambit-gap-aware-recovery
    --moe-moegambit-recovery-policy-type "${MOEGAMBIT_RECOVERY_POLICY_TYPE}"
    --moe-moegambit-gap-threshold "${MOEGAMBIT_GAP_THRESHOLD}"
  )
fi

python3 "${SCRIPT_DIR}/elastic_launcher.py" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  -- \
  python3 ./Megatron-LM/pretrain_gpt.py \
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
