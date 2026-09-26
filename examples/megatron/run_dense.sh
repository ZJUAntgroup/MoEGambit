#!/usr/bin/env bash
# Small dense GPT fault-recovery example: two training nodes and one spare.
# One GPU per node by default. Run this script on all three nodes; set
# NODE_RANK=0, 1, and 2 respectively. RUN_ROOT must be shared by all nodes.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE_RANK="${NODE_RANK:-0}"
TRAINING_NNODES="${TRAINING_NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-20117}"
ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-127.0.0.1}"
ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-5}"
FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-0}"
DRY_RUN="${DRY_RUN:-0}"

die() { printf '[dense-megatron] %s\n' "$*" >&2; exit 64; }
for value in "$NODE_RANK" "$TRAINING_NNODES" "$NPROC_PER_NODE" \
  "$MASTER_PORT" "$ELASTIC_WATCHER_PORT" "$TRAIN_ITERS" \
  "$SAVE_INTERVAL" "$FAULT_INJECT_STEP" "$FAULT_INJECT_NODE" \
  "$FAULT_INJECT_LOCAL_RANK"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "topology, port, and step values must be integers"
done
(( TRAINING_NNODES >= 2 && NPROC_PER_NODE >= 1 )) || die "at least two data-parallel training ranks are required"
(( NODE_RANK <= TRAINING_NNODES )) || die "NODE_RANK must be 0..${TRAINING_NNODES}"
(( FAULT_INJECT_NODE < TRAINING_NNODES && FAULT_INJECT_LOCAL_RANK < NPROC_PER_NODE )) || die "fault rank is outside the training world"
(( TRAIN_ITERS > FAULT_INJECT_STEP && FAULT_INJECT_STEP > SAVE_INTERVAL )) || die "fault step must follow a checkpoint and precede completion"
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -n "${RUN_ROOT:-}" ]] || die "set RUN_ROOT to a shared, writable directory"
  [[ "$RUN_ROOT" = /* ]] || die "RUN_ROOT must be an absolute path shared by all nodes"
  [[ "$MASTER_ADDR" != 127.0.0.1 && "$ELASTIC_WATCHER_ADDR" != 127.0.0.1 ]] || die "set routable MASTER_ADDR and ELASTIC_WATCHER_ADDR"
  [[ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ]] || die "Megatron-LM source is missing"
fi
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/dense-megatron-dry-run}"
CKPT_DIR="${RUN_ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export MOEGAMBIT_MODEL_KIND=dense
export ELASTIC_EXPERT_SIDECAR=0
export ELASTIC_FALLBACK_RELAUNCH=0
export ELASTIC_WATCHER_ADDR ELASTIC_WATCHER_PORT
export ELASTIC_FAULT_DIR="${RUN_ROOT}/faults/node_${NODE_RANK}"
export ELASTIC_PRESTART_SPARE=1
export ELASTIC_PREARM_PLANNED_SPARE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ENABLE_MONITORING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

if (( NODE_RANK == TRAINING_NNODES )); then
  command=(python3 "$ROOT/elastic_watcher.py"
    --port "$ELASTIC_WATCHER_PORT"
    --training-nnodes "$TRAINING_NNODES"
    --nproc-per-node "$NPROC_PER_NODE"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
    --fault-dir "$ELASTIC_FAULT_DIR"
    --fault-inject-step "$FAULT_INJECT_STEP"
    --fault-inject-node "$FAULT_INJECT_NODE"
    --fault-inject-local-rank "$FAULT_INJECT_LOCAL_RANK")
else
  world_size=$((TRAINING_NNODES * NPROC_PER_NODE))
  command=(python3 "$ROOT/elastic_launcher.py"
    --nproc-per-node "$NPROC_PER_NODE" --nnodes "$TRAINING_NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
    -- python3 "$ROOT/Megatron-LM/pretrain_gpt.py"
    --use-mcore-models --transformer-impl local
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --num-layers 2 --hidden-size 128 --ffn-hidden-size 512
    --num-attention-heads 4 --seq-length 128 --max-position-embeddings 128
    --tokenizer-type NullTokenizer --vocab-size 1024 --mock-data
    --micro-batch-size 1 --global-batch-size "$world_size"
    --train-iters "$TRAIN_ITERS" --lr 0.0001 --min-lr 0.00001
    --lr-decay-style cosine --bf16
    --moe-moegambit-enable --moe-moegambit-rank-quarantine
    --moe-moegambit-replacement-protocol --moe-moegambit-group-rebuild
    --moe-moegambit-dense-param-sync --moe-moegambit-recovery-controller
    --moe-moegambit-reintegration-barrier --moe-moegambit-hot-spare-pool
    --moe-moegambit-num-hot-spares "$NPROC_PER_NODE"
    --ckpt-format torch --save "$CKPT_DIR" --save-interval "$SAVE_INTERVAL"
    --eval-iters 0 --log-interval 1)
  if [[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
    command+=(--load "$CKPT_DIR")
  fi
fi

printf '[dense-megatron] node=%s command:' "$NODE_RANK"
printf ' %q' "${command[@]}"
printf '\n'
[[ "$DRY_RUN" == 1 ]] && exit 0
mkdir -p "$CKPT_DIR" "$ELASTIC_FAULT_DIR"
cd "$ROOT"
exec "${command[@]}"
