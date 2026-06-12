#!/usr/bin/env bash
# =============================================================================
# run_moe64_hotspare.sh
#
# Hot-spare node replacement for MoEGambit training.
#
# Architecture:
#   - 9 nodes total: 8 training nodes (64 GPUs) + 1 spare node (8 GPUs)
#   - All 9 nodes run torchrun with NNODES=9, world_size=72
#   - Spare ranks (64-71) participate in init_process_group and
#     initialize_model_parallel (required for collective new_group() calls)
#     then enter standby loop
#   - ELASTIC_TRAINING_WORLD_SIZE=64 tells Megatron to compute
#     data_parallel_size using 64 (not 72)
#   - On rank failure: spare rank is activated, NCCL groups are rebuilt
#     at safe point, new rank walks hybrid recovery (dense from peer +
#     expert from checkpoint)
#   - Training continues WITHOUT restarting all ranks
#
# Key advantage over pure torchrun restart:
#   torchrun --max-restarts kills ALL ranks → must restart from checkpoint
#   This approach keeps surviving ranks alive → hybrid recovery possible
#
# Deployment:
#   Run this script on EACH of the 9 nodes with appropriate NODE_RANK:
#     Node 0-7 (training): NODE_RANK=0..7 bash run_moe64_hotspare.sh
#     Node 8 (spare):      NODE_RANK=8   bash run_moe64_hotspare.sh
#
# For single-node testing (SINGLE_NODE_MODE=1):
#   All 72 ranks run on one machine (requires 8 GPUs, multiple ranks per GPU)
#   nohup bash run_moe64_hotspare.sh > spare.log 2>&1 &
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================

# Mode
SINGLE_NODE_MODE="${SINGLE_NODE_MODE:-0}"

# Parallelism
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export MODE=moegambit

# Topology: 9 nodes (8 training + 1 spare)
export NNODES="${NNODES:-9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_SPARES="${NUM_SPARES:-8}"  # one spare node = 8 GPUs
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"

# Elastic launcher env vars (consumed by Megatron internally)
TRAINING_WORLD_SIZE=$(( (NNODES - 1) * NPROC_PER_NODE ))  # 8*8=64 (exclude spare node)
export ELASTIC_TRAINING_WORLD_SIZE="${TRAINING_WORLD_SIZE}"
export ELASTIC_LAUNCHER=1
export ELASTIC_IS_SPARE="$( [ "${NODE_RANK}" -ge $((NNODES - 1)) ] && echo 1 || echo 0 )"
export ELASTIC_NUM_SPARES="${NUM_SPARES}"
export ELASTIC_SPARE_RANK_START="${TRAINING_WORLD_SIZE}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"

# Recovery policy
export BSR_HOT_SPARE_POOL=1
export BSR_NUM_HOT_SPARES="${NUM_SPARES}"
export BSR_GAP_AWARE_RECOVERY=1
export BSR_RECOVERY_POLICY_TYPE="${BSR_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export BSR_GAP_THRESHOLD="${BSR_GAP_THRESHOLD:-100}"

# Fault injection (restart_in_place for testing hybrid recovery path)
export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:--1}"
export BSR_FAULT_INJECT_STEP="${BSR_FAULT_INJECT_STEP:-70}"
export BSR_FAULT_INJECT_INTERVAL="${BSR_FAULT_INJECT_INTERVAL:-40}"
export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
export BSR_FAULT_REPLACEMENT_STEP="${BSR_FAULT_REPLACEMENT_STEP:-70}"
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"

# Checkpoint & logging
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/hotspare/ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/hotspare}"

# Environment
export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
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

TOTAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))  # 72
DP_SIZE=$((TRAINING_WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
export GLOBAL_BATCH_SIZE

echo "[hotspare] =============================================="
echo "[hotspare] MoEGambit Hot-Spare Node Launcher"
echo "[hotspare] =============================================="
echo "[hotspare] NODE_RANK:       ${NODE_RANK} $([ "${ELASTIC_IS_SPARE}" = "1" ] && echo "(SPARE)" || echo "(TRAINING)")"
echo "[hotspare] Training ranks:  ${TRAINING_WORLD_SIZE} (${NNODES}-1 nodes x ${NPROC_PER_NODE} GPUs)"
echo "[hotspare] Spare ranks:     ${NUM_SPARES} (node ${NNODES}-1)"
echo "[hotspare] Total world:     ${TOTAL_WORLD_SIZE}"
echo "[hotspare] Parallelism:     TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, DP=${DP_SIZE}"
echo "[hotspare] Global batch:    ${GLOBAL_BATCH_SIZE}"
echo "[hotspare] CKPT_DIR:        ${CKPT_DIR}"
echo "[hotspare] Fault dir:       ${ELASTIC_FAULT_DIR}"
echo "[hotspare] =============================================="

# =============================================================================
# Build training arguments
# =============================================================================

LOAD_ARGS=()
if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
  LOAD_ARGS=(--load "${CKPT_DIR}")
fi

BSR_ARGS=(
  --moe-bsr-enable
  --moe-bsr-health-mask
  --moe-bsr-rank-quarantine
  --moe-bsr-dispatch-quarantine-assert
  --moe-bsr-dispatch-sanitize
  --moe-bsr-expert-directory
  --moe-bsr-replacement-protocol
  --moe-bsr-group-rebuild
  --moe-bsr-dispatch-topology-refresh
  --moe-bsr-dense-param-sync
  --moe-bsr-stale-expert-restore
  --moe-bsr-recovery-controller
  --moe-bsr-deferred-optimizer-load
  --moe-bsr-degraded-mode-policy
  --moe-bsr-reintegration-barrier
  --moe-bsr-fault-injection
  --moe-bsr-restart-in-place
  --moe-bsr-degraded-tau-c 0.5
  --moe-bsr-degraded-t-max 1000
  --moe-bsr-degraded-s-max 500
  --moe-bsr-hot-spare-pool
  --moe-bsr-num-hot-spares "${NUM_SPARES}"
)

if [ "${BSR_GAP_AWARE_RECOVERY:-0}" = "1" ]; then
  BSR_ARGS+=(
    --moe-bsr-gap-aware-recovery
    --moe-bsr-recovery-policy-type "${BSR_RECOVERY_POLICY_TYPE}"
    --moe-bsr-gap-threshold "${BSR_GAP_THRESHOLD}"
  )
fi

# =============================================================================
# Launch via torchrun (each node runs this independently)
# =============================================================================

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
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
  "${BSR_ARGS[@]}" \
  --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
  --split 100,0,0 \
  --ckpt-format torch \
  --save "${CKPT_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval 1000 \
  --eval-iters 0 \
  --log-interval 1 \
  "${LOAD_ARGS[@]}"
