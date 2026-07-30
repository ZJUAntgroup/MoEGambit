#!/usr/bin/env bash
# Nine-node DeepSpeed validation: 8 active nodes + 1 hot-spare node.
#
# Run this file on every physical node. Only NODE_RANK changes. The paths below
# intentionally match the currently validated cluster and have not been
# anonymized.

set -Eeuo pipefail

REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
)"

export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20121}"
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR="${MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR:-${ELASTIC_WATCHER_ADDR}}"

export TRAINING_NNODES="${TRAINING_NNODES:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export SPARE_NODE_RANK="${SPARE_NODE_RANK:-8}"
export TEST_MODE="${TEST_MODE:-hot_swap}"

export MODEL_CONFIG="${MODEL_CONFIG:-${REPOSITORY_ROOT}/tokenizer}"
export DATA_PATH="${DATA_PATH:-/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document}"
export RUN_ROOT="${RUN_ROOT:-/mnt/ais-c1/dataset/zds/89hotspare/deepspeed_real}"

export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-17}"
export FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
export FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-1}"
export DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPOSITORY_ROOT}/src:${REPOSITORY_ROOT}/DeepSpeed:${REPOSITORY_ROOT}/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

cat <<EOF
[moegambit-deepspeed-example]
  repository=${REPOSITORY_ROOT}
  mode=${TEST_MODE}
  node_rank=${NODE_RANK}
  master=${MASTER_ADDR}:${MASTER_PORT}
  coordinator=${MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR}
  topology=${TRAINING_NNODES}x${NPROC_PER_NODE}+spare-${SPARE_NODE_RANK}
  data=${DATA_PATH}
  run_root=${RUN_ROOT}
  dry_run=${DRY_RUN}
EOF

cd "${REPOSITORY_ROOT}"
exec bash "${REPOSITORY_ROOT}/test_deepspeed_hotspare_replace.sh"
