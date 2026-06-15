#!/usr/bin/env bash
# =============================================================================
# test_hotspare_replace.sh
#
# 测试备用节点替换功能的脚本。
#
# 与 run_moe64_hotspare.sh 的区别：
#   - 不使用 restart_in_place（BSR_FAULT_INJECT_TYPE=kill_rank）
#   - save interval = 10（更频繁保存 checkpoint）
#   - 故障注入在 step 19（确保有 checkpoint 可用）
#   - 目的：验证当一个 rank 被杀死后，备用节点能否接管
#
# 使用方式：
#   训练节点 (NODE_RANK=0-7):
#     export NODE_RANK=<0-7>
#     export MASTER_ADDR=<rank0 IP>
#     export MASTER_PORT=20117
#     export ELASTIC_WATCHER_ADDR=<备用节点 IP>
#     bash test_hotspare_replace.sh
#
#   备用节点 (NODE_RANK=8):
#     export NODE_RANK=8
#     export MASTER_ADDR=<rank0 IP>
#     export MASTER_PORT=20117
#     bash test_hotspare_replace.sh
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
export MASTER_PORT="${MASTER_PORT:-20117}"

# Watcher coordination port
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"

# Determine role
TRAINING_NNODES=8
if [ "${NODE_RANK}" -ge "${TRAINING_NNODES}" ]; then
  IS_SPARE=1
else
  IS_SPARE=0
fi

export NNODES="${TRAINING_NNODES}"

# Elastic watcher address (备用节点的 IP)
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"
rm -rf "${ELASTIC_FAULT_DIR}"
mkdir -p "${ELASTIC_FAULT_DIR}"

# Recovery policy
export BSR_HOT_SPARE_POOL=1
export BSR_NUM_HOT_SPARES="${NPROC_PER_NODE}"
export BSR_GAP_AWARE_RECOVERY=1
export BSR_RECOVERY_POLICY_TYPE="${BSR_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export BSR_GAP_THRESHOLD="${BSR_GAP_THRESHOLD:-100}"

# ============================================================================
# 测试配置：硬故障注入，触发备用节点替换
# ============================================================================
# hard_failure: 清零故障 rank 的参数，触发 recovery controller 的硬故障恢复路径
# 这不会杀死进程，但会模拟参数丢失并触发从 checkpoint 恢复
export BSR_FAULT_INJECT_TYPE="hard_failure"
# 在 step 19 注入故障（确保 step 10 已保存 checkpoint）
export BSR_FAULT_INJECT_STEP=19
# 不设置周期性注入
export BSR_FAULT_INJECT_INTERVAL=0
export BSR_FAULT_INJECT_SEED=42
# -1 表示随机选择一个 rank
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:--1}"
export BSR_FAULT_REPLACEMENT_STEP=19
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
export BSR_FAULT_ZERO_MEMORY=1
export BSR_FAULT_MEMORY_FILL=zero

# Checkpoint: 每 10 步保存一次（确保故障时有近期 checkpoint）
export SAVE_INTERVAL=10
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/hotspare/test_replace_ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/test_replace}"

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

export TRAIN_ITERS="${TRAIN_ITERS:-100}"

mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# =============================================================================
# Compute derived values
# =============================================================================

TRAINING_WORLD_SIZE=$((TRAINING_NNODES * NPROC_PER_NODE))  # 64
DP_SIZE=$((TRAINING_WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
export GLOBAL_BATCH_SIZE

echo "[test-replace] =============================================="
echo "[test-replace] Hot-Spare REPLACEMENT TEST"
echo "[test-replace] =============================================="
echo "[test-replace] NODE_RANK:       ${NODE_RANK} $([ "${IS_SPARE}" = "1" ] && echo "(SPARE WATCHER)" || echo "(TRAINING)")"
echo "[test-replace] Training nodes:  ${TRAINING_NNODES} (${TRAINING_WORLD_SIZE} GPUs)"
echo "[test-replace] Fault inject:    kill_rank at step ${BSR_FAULT_INJECT_STEP}"
echo "[test-replace] Save interval:   ${SAVE_INTERVAL}"
echo "[test-replace] Train iters:     ${TRAIN_ITERS}"
echo "[test-replace] CKPT_DIR:        ${CKPT_DIR}"
echo "[test-replace] =============================================="

# =============================================================================
# SPARE NODE: run watcher
# =============================================================================

if [ "${IS_SPARE}" = "1" ]; then
  echo "[test-replace] Starting elastic watcher on spare node..."
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
# TRAINING NODE: custom launcher
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
  --moe-bsr-hot-spare-pool
  --moe-bsr-num-hot-spares "${NPROC_PER_NODE}"
  --moe-bsr-degraded-tau-c 0.5
  --moe-bsr-degraded-t-max 1000
  --moe-bsr-degraded-s-max 500
)

if [ "${BSR_GAP_AWARE_RECOVERY:-0}" = "1" ]; then
  BSR_ARGS+=(
    --moe-bsr-gap-aware-recovery
    --moe-bsr-recovery-policy-type "${BSR_RECOVERY_POLICY_TYPE}"
    --moe-bsr-gap-threshold "${BSR_GAP_THRESHOLD}"
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
