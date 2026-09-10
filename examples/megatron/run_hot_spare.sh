#!/usr/bin/env bash
# Nine-node Megatron validation: 8 active nodes + 1 hot-spare node.
#
# Run this file on every physical node. Set NODE_RANK and routable addresses.
# Paths below are public examples; override them with your shared mount paths.

set -Eeuo pipefail

REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
)"

export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20117}"
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"

export TRAINING_NNODES="${TRAINING_NNODES:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"

export DATA_PATH="${DATA_PATH:-/shared/moegambit/data/train_text_document}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${REPOSITORY_ROOT}/tokenizer}"
export CKPT_DIR="${CKPT_DIR:-/shared/moegambit/checkpoints/megatron}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/shared/moegambit/logs/megatron}"

export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-17}"
export FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
export FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-1}"
export DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPOSITORY_ROOT}/src:${REPOSITORY_ROOT}/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

cat <<EOF
[moegambit-megatron-example]
  repository=${REPOSITORY_ROOT}
  node_rank=${NODE_RANK}
  master=${MASTER_ADDR}:${MASTER_PORT}
  watcher=${ELASTIC_WATCHER_ADDR}:${ELASTIC_WATCHER_PORT}
  topology=${TRAINING_NNODES}x${NPROC_PER_NODE}
  data=${DATA_PATH}
  checkpoint=${CKPT_DIR}
  dry_run=${DRY_RUN}
EOF

cd "${REPOSITORY_ROOT}"
exec bash "${REPOSITORY_ROOT}/test_hotspare_replace.sh"
