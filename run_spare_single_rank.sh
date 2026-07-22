#!/usr/bin/env bash
# =============================================================================
# run_spare_single_rank.sh
#
# Launched by elastic_watcher to start a SINGLE replacement worker process.
# Environment variables are set by the watcher:
#   NODE_RANK, LOCAL_RANK, RANK, WORLD_SIZE, MASTER_PORT,
#   ELASTIC_REBUILD_MODE=1, CUDA_VISIBLE_DEVICES
#
# This script runs the same training command as the original workers,
# but as a single process (not via elastic_launcher).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"

if [ "${ELASTIC_STANDBY_MODE:-0}" = "1" ]; then
  ASSIGNMENT_FILE="${ELASTIC_SPARE_ASSIGNMENT_FILE:-/tmp/elastic_faults/spare_assignment.json}"
  echo "[spare-rank] Warm standby pid=$$ waiting for assignment: ${ASSIGNMENT_FILE}"
  if [ "${ELASTIC_STANDBY_PRELOAD:-1}" = "1" ]; then
    python3 - <<'PY'
import importlib

for module in ("torch", "megatron.training", "megatron.core"):
    importlib.import_module(module)
print("[spare-rank] Warm standby preload complete: torch/megatron imported", flush=True)
PY
  fi
  while [ ! -s "${ASSIGNMENT_FILE}" ]; do
    sleep 1
  done
  eval "$(
    python3 - "${ASSIGNMENT_FILE}" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

for key, value in data.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
  )"
  unset ELASTIC_STANDBY_MODE
  echo "[spare-rank] Warm standby activated: RANK=${RANK}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Inherit most settings from environment (set by watcher)
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
export GROUP_RANK="${GROUP_RANK:-${NODE_RANK}}"

# Parallelism (must match training config)
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-1}"
export MEGATRON_PARALLEL_ORDER="${MEGATRON_PARALLEL_ORDER:-tp-cp-ep-dp-pp}"

# NCCL config
export ELASTIC_RECOVERY_NCCL_SOCKET_ONLY="${ELASTIC_RECOVERY_NCCL_SOCKET_ONLY:-1}"
export ELASTIC_RECOVERY_NCCL_DEBUG="${ELASTIC_RECOVERY_NCCL_DEBUG:-WARN}"
export NCCL_DEBUG="${NCCL_DEBUG:-${ELASTIC_RECOVERY_NCCL_DEBUG}}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
if [ "${ELASTIC_RECOVERY_NCCL_SOCKET_ONLY}" = "1" ]; then
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_FAMILY=AF_INET
  if [ -n "${ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME:-}" ]; then
    export NCCL_SOCKET_IFNAME="${ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME}"
  elif command -v ip >/dev/null 2>&1; then
    NCCL_ROUTE_IFNAME="$(ip -o route get "${MASTER_ADDR}" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}')"
    if [ -n "${NCCL_ROUTE_IFNAME}" ]; then
      export NCCL_SOCKET_IFNAME="=${NCCL_ROUTE_IFNAME}"
      export ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    fi
  fi
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# Other environment
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

NPROC_PER_NODE="${ELASTIC_TRAINING_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
TRAINING_NNODES="${NNODES:-8}"
TRAINING_WORLD_SIZE=$((TRAINING_NNODES * NPROC_PER_NODE))
DP_SIZE=$((TRAINING_WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-10}"
PHASE_TIMEOUT_DEFAULT=$((DISTRIBUTED_TIMEOUT_MINUTES * 60 + 120))

CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/77hotspare/test_replace_ckpt}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
ELASTIC_REBUILD_TIMEOUT_MINUTES="${ELASTIC_REBUILD_TIMEOUT_MINUTES:-${DISTRIBUTED_TIMEOUT_MINUTES}}"
export ELASTIC_PHASE_TIMEOUT_SECONDS="${ELASTIC_PHASE_TIMEOUT_SECONDS:-${ELASTIC_REBUILD_PHASE_TIMEOUT:-${PHASE_TIMEOUT_DEFAULT}}}"
export ELASTIC_REBUILD_PHASE_TIMEOUT="${ELASTIC_REBUILD_PHASE_TIMEOUT:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS="${ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_BARRIER="${ELASTIC_MPU_GROUP_ORDINAL_BARRIER:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS="${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_SELECTIVE_GROUP_REBUILD="${ELASTIC_SELECTIVE_GROUP_REBUILD:-1}"
export ELASTIC_INIT_PG_DEVICE_ID="${ELASTIC_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_REBUILD_INIT_PG_DEVICE_ID="${ELASTIC_REBUILD_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER="${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER:-1}"
export ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST="${ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST:-0}"
export ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT="${ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT:-70}"
export MOEGAMBIT_RECOVERY_POLICY_TYPE="${MOEGAMBIT_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export MOEGAMBIT_DELTA_TIME_MIN_GAP="${MOEGAMBIT_DELTA_TIME_MIN_GAP:-0}"
export MOEGAMBIT_MAX_SINGLE_GAP="${MOEGAMBIT_MAX_SINGLE_GAP:-192}"
export MOEGAMBIT_EXPOSURE_WINDOW_STEPS="${MOEGAMBIT_EXPOSURE_WINDOW_STEPS:-20000}"
export MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY="${MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY:-0.1}"
export MOEGAMBIT_NUM_EXPERTS="${MOEGAMBIT_NUM_EXPERTS:-128}"
export ELASTIC_TWO_PHASE_RECOVERY="${ELASTIC_TWO_PHASE_RECOVERY:-0}"
export ELASTIC_ZERO2_MEMORY_REPLICATION="${ELASTIC_ZERO2_MEMORY_REPLICATION:-0}"
export ELASTIC_ZERO2_RESTORE_SCOPE="${ELASTIC_ZERO2_RESTORE_SCOPE:-non_expert}"
export ELASTIC_ZERO2_REPLICATION_TIMEOUT="${ELASTIC_ZERO2_REPLICATION_TIMEOUT:-300}"
export ELASTIC_ZERO2_MAX_HOST_GB_PER_RANK="${ELASTIC_ZERO2_MAX_HOST_GB_PER_RANK:-0}"
export ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER="${ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER:-0}"
if [ "${ELASTIC_ZERO2_MEMORY_REPLICATION}" = "1" ] && \
   [ "${ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER}" != "1" ]; then
  echo "[spare-rank] ERROR: optimizer memory replication requires distributed optimizer" >&2
  exit 64
fi
# Replacement is launched directly by the watcher, not by elastic_launcher.
unset ELASTIC_LAUNCHER_CONTROL_SOCKET 2>/dev/null || true

# Elastic watcher connection
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"

echo "[spare-rank] Starting replacement: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, NODE_RANK=${NODE_RANK}"
echo "[spare-rank] logical_node=${ELASTIC_LOGICAL_NODE_RANK:-${NODE_RANK}}, physical_node=${ELASTIC_PHYSICAL_NODE_RANK:-${GROUP_RANK}}, GROUP_RANK=${GROUP_RANK}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}, training_nproc=${NPROC_PER_NODE}"
echo "[spare-rank] MASTER=${MASTER_ADDR}:${MASTER_PORT}, WORLD_SIZE=${WORLD_SIZE}"
echo "[spare-rank] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[spare-rank] ELASTIC_REBUILD_MODE=${ELASTIC_REBUILD_MODE}"
echo "[spare-rank] recovery epoch=${ELASTIC_RECOVERY_EPOCH:-unset}, descriptor=${ELASTIC_RECOVERY_DESCRIPTOR:-unset}"
echo "[spare-rank] distributed timeout=${DISTRIBUTED_TIMEOUT_MINUTES}min"
echo "[spare-rank] recovery NCCL socket_only=${ELASTIC_RECOVERY_NCCL_SOCKET_ONLY}, debug=${ELASTIC_RECOVERY_NCCL_DEBUG}"
echo "[spare-rank] NCCL transport ib_disable=${NCCL_IB_DISABLE:-0}, ifname=${NCCL_SOCKET_IFNAME:-auto}, family=${NCCL_SOCKET_FAMILY:-auto}"
echo "[spare-rank] phase timeout=${ELASTIC_PHASE_TIMEOUT_SECONDS}s"
echo "[spare-rank] group ordinal barrier=${ELASTIC_MPU_GROUP_ORDINAL_BARRIER} (${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS}s)"
echo "[spare-rank] prearmed=${ELASTIC_PREARMED_STANDBY:-0}, selective_group_rebuild=${ELASTIC_SELECTIVE_GROUP_REBUILD}"
echo "[spare-rank] pg device_id init=${ELASTIC_INIT_PG_DEVICE_ID}, rebuild=${ELASTIC_REBUILD_INIT_PG_DEVICE_ID}"
echo "[spare-rank] moe first collective barrier=${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER}"
if [ "${ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST}" = "0" ]; then
    echo "[spare-rank] moe first collective fail-fast=disabled"
else
    echo "[spare-rank] moe first collective fail-fast=enabled, timeout=${ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT}s"
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
  --no-moe-moegambit-weights-first-recovery
  --moe-moegambit-degraded-mode-policy
  --moe-moegambit-reintegration-barrier
  --moe-moegambit-hot-spare-pool
  --moe-moegambit-num-hot-spares "${NPROC_PER_NODE}"
  --moe-moegambit-degraded-tau-c 0.5
  --moe-moegambit-degraded-t-max 1000
  --moe-moegambit-degraded-s-max 500
  --moe-moegambit-gap-aware-recovery
  --moe-moegambit-recovery-policy-type "${MOEGAMBIT_RECOVERY_POLICY_TYPE}"
  --moe-moegambit-gap-threshold 100
  --moe-moegambit-delta-time-min-gap "${MOEGAMBIT_DELTA_TIME_MIN_GAP}"
  --moe-moegambit-max-single-gap "${MOEGAMBIT_MAX_SINGLE_GAP}"
  --moe-moegambit-exposure-window-steps "${MOEGAMBIT_EXPOSURE_WINDOW_STEPS}"
  --moe-moegambit-max-expert-staleness-density "${MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY}"
)

ZERO2_ARGS=()
if [ "${ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER}" = "1" ]; then
  ZERO2_ARGS+=(--use-distributed-optimizer)
fi

LOAD_ARGS=()
if [ "${ELASTIC_PREARMED_STANDBY:-0}" = "1" ] || \
   [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || \
   ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
  LOAD_ARGS=(--load "${CKPT_DIR}")
fi

exec python3 ./Megatron-LM/pretrain_gpt.py \
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
  --num-experts "${MOEGAMBIT_NUM_EXPERTS}" \
  --moe-ffn-hidden-size 768 \
  --moe-router-topk 8 \
  --moe-router-dtype fp32 \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 1e-3 \
  --moe-token-dispatcher-type alltoall \
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
  --distributed-timeout-seconds-after-init 60 \
  "${ZERO2_ARGS[@]}" \
  "${MOEGAMBIT_ARGS[@]}" \
  --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
  --split 100,0,0 \
  --ckpt-format torch \
  --save "${CKPT_DIR}" \
  --save-interval 10 \
  --eval-interval 1000 \
  --eval-iters 0 \
  --log-interval 1 \
  "${LOAD_ARGS[@]}"
