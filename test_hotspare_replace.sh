#!/usr/bin/env bash
# =============================================================================
# test_hotspare_replace.sh
#
# 测试备用节点替换功能的脚本。
#
# 工作原理：
#   1. 训练正常启动在 NODE_RANK=0-7 (64 GPU)
#   2. 备用节点 (NODE_RANK=8) 运行 elastic_watcher
#   3. 到达指定步数时，watcher 通过 TCP 发送 kill_node 命令杀死目标节点
#   4. 目标节点所有 worker 进程被杀死 (SIGKILL)
#   5. 存活节点检测到 NCCL 超时 → 捕获异常 → 进入 elastic 恢复路径
#   6. 存活节点通知 watcher "ready_to_rebuild"
#   7. watcher 在备用节点上用故障节点的 NODE_RANK 启动新 worker
#   8. 所有节点重新 init_process_group → sync params → 恢复训练
#
# 使用方式：
#   训练节点 (NODE_RANK=0-7):
#     export NODE_RANK=<0-7>
#     export MASTER_ADDR=<rank0 IP>
#     export ELASTIC_WATCHER_ADDR=<备用节点 IP>
#     bash test_hotspare_replace.sh
#
#   备用节点 (NODE_RANK=8):
#     export NODE_RANK=8
#     export MASTER_ADDR=<rank0 IP>
#     export ELASTIC_WATCHER_ADDR=<本机 IP>
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

# ============================================================================
# Fault injection: watcher 在 step N 时杀死目标节点的指定 local_rank
# ============================================================================
FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-17}"
FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
# NOTE: Must NOT be 0 — local_rank=0 holds the TCP connection to watcher.
# If local_rank=0 is killed, other ranks on that node can't receive rebuild signal.
FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-1}"

# ============================================================================
# BSR recovery settings (for surviving nodes during recovery)
# ============================================================================
export BSR_HOT_SPARE_POOL=1
export BSR_NUM_HOT_SPARES="${NPROC_PER_NODE}"
export BSR_GAP_AWARE_RECOVERY=1
export BSR_RECOVERY_POLICY_TYPE="${BSR_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export BSR_GAP_THRESHOLD="${BSR_GAP_THRESHOLD:-100}"

# NO BSR fault injection — we do real kills via the watcher
# (BSR's hard_failure doesn't actually kill processes)
unset BSR_FAULT_INJECT_TYPE 2>/dev/null || true
unset BSR_FAULT_INJECT_STEP 2>/dev/null || true

# Checkpoint: 每 10 步保存一次（确保故障时有近期 checkpoint）
export SAVE_INTERVAL=10
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/621hotspare/test_replace_ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/test_replace}"

# ============================================================================
# NCCL configuration: short timeout for fast failure detection
# ============================================================================
export NCCL_DEBUG=WARN
# CRITICAL: Set to 0 so that NCCL timeout raises a catchable Python RuntimeError
# instead of calling std::abort() (SIGABRT).  With =1, the C++ watchdog kills
# the process immediately on timeout — Python never gets a chance to handle it.
# With =0, the Python thread blocked on the NCCL op gets a RuntimeError after
# the timeout, which our except handler catches and routes to elastic recovery.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
# Disable the HEARTBEAT MONITOR — this is the mechanism that causes SIGABRT
# when the watchdog itself gets stuck (e.g., during error handling).
export TORCH_NCCL_ENABLE_MONITORING=0
# Also increase the heartbeat timeout to be safe
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# Other environment
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-10}"
PHASE_TIMEOUT_DEFAULT=$((DISTRIBUTED_TIMEOUT_MINUTES * 60 + 120))
export ELASTIC_PHASE_TIMEOUT_SECONDS="${ELASTIC_PHASE_TIMEOUT_SECONDS:-${ELASTIC_REBUILD_PHASE_TIMEOUT:-${PHASE_TIMEOUT_DEFAULT}}}"
export ELASTIC_REBUILD_PHASE_TIMEOUT="${ELASTIC_REBUILD_PHASE_TIMEOUT:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS="${ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_BARRIER="${ELASTIC_MPU_GROUP_ORDINAL_BARRIER:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS="${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_INIT_PG_DEVICE_ID="${ELASTIC_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_REBUILD_INIT_PG_DEVICE_ID="${ELASTIC_REBUILD_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_POST_REBUILD_COMM_WARMUP="${ELASTIC_POST_REBUILD_COMM_WARMUP:-0}"

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
echo "[test-replace] Fault inject:    kill node ${FAULT_INJECT_NODE} local_rank ${FAULT_INJECT_LOCAL_RANK} at step ${FAULT_INJECT_STEP}"
echo "[test-replace] Save interval:   ${SAVE_INTERVAL}"
echo "[test-replace] Train iters:     ${TRAIN_ITERS}"
echo "[test-replace] Dist timeout:    ${DISTRIBUTED_TIMEOUT_MINUTES}min (60s after init)"
echo "[test-replace] Phase timeout:   ${ELASTIC_PHASE_TIMEOUT_SECONDS}s"
echo "[test-replace] Group barrier:   ${ELASTIC_MPU_GROUP_ORDINAL_BARRIER} (${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS}s)"
echo "[test-replace] PG device_id:    init=${ELASTIC_INIT_PG_DEVICE_ID}, rebuild=${ELASTIC_REBUILD_INIT_PG_DEVICE_ID}"
echo "[test-replace] Post warmup:     ${ELASTIC_POST_REBUILD_COMM_WARMUP}"
echo "[test-replace] CKPT_DIR:        ${CKPT_DIR}"
echo "[test-replace] =============================================="

# =============================================================================
# SPARE NODE: run watcher (with fault injection)
# =============================================================================

if [ "${IS_SPARE}" = "1" ]; then
  echo "[test-replace] Starting elastic watcher on spare node..."
  echo "[test-replace] Fault injection: kill node ${FAULT_INJECT_NODE} local_rank ${FAULT_INJECT_LOCAL_RANK} at step ${FAULT_INJECT_STEP}"
  python3 "${SCRIPT_DIR}/elastic_watcher.py" \
    --port "${ELASTIC_WATCHER_PORT}" \
    --training-nnodes "${TRAINING_NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --fault-dir "${ELASTIC_FAULT_DIR}" \
    --fault-inject-step "${FAULT_INJECT_STEP}" \
    --fault-inject-node "${FAULT_INJECT_NODE}" \
    --fault-inject-local-rank "${FAULT_INJECT_LOCAL_RANK}"
  exit $?
fi

# =============================================================================
# TRAINING NODE: custom launcher (no torchrun)
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
  --no-moe-bsr-weights-first-recovery
  --moe-bsr-degraded-mode-policy
  --moe-bsr-reintegration-barrier
  # NOTE: no --moe-bsr-fault-injection — we do real kills via watcher
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
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
  --distributed-timeout-seconds-after-init 60 \
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
