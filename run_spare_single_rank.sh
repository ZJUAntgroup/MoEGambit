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

# NCCL config
export NCCL_DEBUG=WARN
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

CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/hotspare/test_replace_ckpt}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
ELASTIC_REBUILD_TIMEOUT_MINUTES="${ELASTIC_REBUILD_TIMEOUT_MINUTES:-${DISTRIBUTED_TIMEOUT_MINUTES}}"
export ELASTIC_PHASE_TIMEOUT_SECONDS="${ELASTIC_PHASE_TIMEOUT_SECONDS:-${ELASTIC_REBUILD_PHASE_TIMEOUT:-${PHASE_TIMEOUT_DEFAULT}}}"
export ELASTIC_REBUILD_PHASE_TIMEOUT="${ELASTIC_REBUILD_PHASE_TIMEOUT:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS="${ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_BARRIER="${ELASTIC_MPU_GROUP_ORDINAL_BARRIER:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS="${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_INIT_PG_DEVICE_ID="${ELASTIC_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_REBUILD_INIT_PG_DEVICE_ID="${ELASTIC_REBUILD_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_POST_REBUILD_COMM_WARMUP="${ELASTIC_POST_REBUILD_COMM_WARMUP:-0}"
export ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER="${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER:-1}"
export ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP="${ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP:-0}"

# Elastic watcher connection
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"

echo "[spare-rank] Starting replacement: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, NODE_RANK=${NODE_RANK}"
echo "[spare-rank] logical_node=${ELASTIC_LOGICAL_NODE_RANK:-${NODE_RANK}}, physical_node=${ELASTIC_PHYSICAL_NODE_RANK:-${GROUP_RANK}}, GROUP_RANK=${GROUP_RANK}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}, training_nproc=${NPROC_PER_NODE}"
echo "[spare-rank] MASTER=${MASTER_ADDR}:${MASTER_PORT}, WORLD_SIZE=${WORLD_SIZE}"
echo "[spare-rank] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[spare-rank] ELASTIC_REBUILD_MODE=${ELASTIC_REBUILD_MODE}"
echo "[spare-rank] distributed timeout=${DISTRIBUTED_TIMEOUT_MINUTES}min"
echo "[spare-rank] phase timeout=${ELASTIC_PHASE_TIMEOUT_SECONDS}s"
echo "[spare-rank] group ordinal barrier=${ELASTIC_MPU_GROUP_ORDINAL_BARRIER} (${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS}s)"
echo "[spare-rank] pg device_id init=${ELASTIC_INIT_PG_DEVICE_ID}, rebuild=${ELASTIC_REBUILD_INIT_PG_DEVICE_ID}"
echo "[spare-rank] post-rebuild comm warmup=${ELASTIC_POST_REBUILD_COMM_WARMUP}"
echo "[spare-rank] moe first collective barrier=${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER}"
echo "[spare-rank] moe first collective warmup=${ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP}"

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
  --moe-bsr-hot-spare-pool
  --moe-bsr-num-hot-spares "${NPROC_PER_NODE}"
  --moe-bsr-degraded-tau-c 0.5
  --moe-bsr-degraded-t-max 1000
  --moe-bsr-degraded-s-max 500
  --moe-bsr-gap-aware-recovery
  --moe-bsr-recovery-policy-type rank_exposure_guarded_hybrid
  --moe-bsr-gap-threshold 100
)

LOAD_ARGS=()
if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
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
  --save-interval 10 \
  --eval-interval 1000 \
  --eval-iters 0 \
  --log-interval 1 \
  "${LOAD_ARGS[@]}"
