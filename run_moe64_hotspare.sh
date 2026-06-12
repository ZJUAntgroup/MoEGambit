#!/usr/bin/env bash
# =============================================================================
# run_moe64_hotspare.sh
#
# Hot-Spare Node Pool experiment (daemon mode):
#   Training: 64 GPUs = 8 nodes × 8 GPUs/node (standard torchrun world)
#   Hot spares: 8 GPUs on a 9th node, running as independent daemons
#               (NOT part of the torchrun world)
#
# Architecture:
#   - torchrun launches 64 ranks (NNODES=8) as normal training
#   - Hot-spare daemons are started separately on the spare node via
#     hot_spare_daemon.py (see below)
#   - On fault, RecoveryController signals a daemon via TCPStore
#   - The daemon joins a rebuilt NCCL group at the next safe-point
#
# Recovery pipeline:
#   fault injection → hot-spare allocation → gap-aware policy decision
#   → NCCL group rebuild (spare joins) → hybrid recovery
#   (Path P: non-expert from DP peer + Path C: expert from checkpoint)
#   → reintegration into training
#
# This script launches the TRAINING side only.  For the spare node, run:
#   python hot_spare_daemon.py --master-addr $MASTER_ADDR \
#       --master-port $MASTER_PORT --num-gpus 8
#
# Overrides (all optional):
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT  (passed through)
#   CKPT_DIR / TRAIN_LOG_DIR
#   BSR_NUM_HOT_SPARES      default 8
#   BSR_GAP_THRESHOLD        default 100
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/run_moe64_par.sh"
if [ ! -f "${INNER_SCRIPT}" ]; then
  echo "[run_moe64_hotspare] missing ${INNER_SCRIPT}"
  exit 2
fi

# ---- 8-node training topology (64 GPUs, standard torchrun world) ----
export NNODES="${NNODES:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"

# ---- Parallelism (same as default run_moe64) ----
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"

# ---- Mode: moegambit (BSR-MoE hybrid recovery) ----
export MODE=moegambit

# ---- Hot-spare pool (daemon mode: does NOT affect torchrun world size) ----
export BSR_HOT_SPARE_POOL=1
export BSR_NUM_HOT_SPARES="${BSR_NUM_HOT_SPARES:-8}"

# ---- Gap-aware hybrid recovery policy ----
export BSR_GAP_AWARE_RECOVERY=1
export BSR_RECOVERY_POLICY_TYPE="${BSR_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export BSR_GAP_THRESHOLD="${BSR_GAP_THRESHOLD:-100}"

# ---- Fault injection ----
export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:--1}"
export BSR_FAULT_INJECT_STEP="${BSR_FAULT_INJECT_STEP:-70}"
export BSR_FAULT_INJECT_INTERVAL="${BSR_FAULT_INJECT_INTERVAL:-40}"
export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
export BSR_FAULT_REPLACEMENT_STEP="${BSR_FAULT_REPLACEMENT_STEP:-70}"
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"

# ---- Checkpoint & logging ----
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/hotspare/ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/hotspare}"
export MAX_RETRIES="${MAX_RETRIES:-1}"

echo "[run_moe64_hotspare] config:"
echo "  TRAINING   = ${NNODES} nodes × 8 GPUs (torchrun world_size=$((NNODES * 8)))"
echo "  HOT SPARES = ${BSR_NUM_HOT_SPARES} GPUs (daemon mode, separate node)"
echo "  PARALLEL   = TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}"
echo "  RECOVERY   = gap-aware hybrid (threshold=${BSR_GAP_THRESHOLD})"
echo "  FAULT      = ${BSR_FAULT_INJECT_TYPE}, step=${BSR_FAULT_INJECT_STEP}, interval=${BSR_FAULT_INJECT_INTERVAL}"
echo "  CKPT_DIR   = ${CKPT_DIR}"
echo "  LOG_DIR    = ${TRAIN_LOG_DIR}"

# ---- Delegate to inner script (NNODES=8, 64 GPUs only) ----
bash "${INNER_SCRIPT}"
