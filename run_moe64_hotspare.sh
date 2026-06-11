#!/usr/bin/env bash
# =============================================================================
# run_moe64_hotspare.sh
#
# Hot-Spare Node Pool experiment:
#   72 GPUs = 9 nodes × 8 GPUs/node
#   Ranks 0-63 : training ranks (64 GPUs, 8 nodes)
#   Ranks 64-71: hot-spare ranks (8 GPUs, 1 node, STANDBY)
#
# NCCL safety: spare ranks do NOT join any training NCCL process group.
# They communicate with the coordinator only via Gloo/TCPStore and are
# integrated into NCCL groups only at safe-points after activation.
#
# Recovery pipeline:
#   fault injection → hot-spare allocation → gap-aware policy decision
#   → hybrid recovery (Path P: non-expert from DP peer + Path C: expert
#     from checkpoint) → reintegration into training
#
# This script is a thin wrapper around run_moe64_par.sh (same pattern
# as bench_moe64_tp*.sh).  It exports the necessary environment variables
# and then delegates to run_moe64_par.sh for the actual training loop.
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

# ---- 9-node topology (64 training + 8 spare) ----
export NNODES="${NNODES:-9}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"

# ---- Parallelism (same as default run_moe64) ----
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"

# ---- Mode: moegambit (BSR-MoE hybrid recovery) ----
export MODE=moegambit

# ---- Hot-spare pool ----
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
echo "  TOPOLOGY   = ${NNODES} nodes × 8 GPUs (${BSR_NUM_HOT_SPARES} hot spares)"
echo "  PARALLEL   = TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}"
echo "  RECOVERY   = gap-aware hybrid (threshold=${BSR_GAP_THRESHOLD})"
echo "  FAULT      = ${BSR_FAULT_INJECT_TYPE}, step=${BSR_FAULT_INJECT_STEP}, interval=${BSR_FAULT_INJECT_INTERVAL}"
echo "  CKPT_DIR   = ${CKPT_DIR}"
echo "  LOG_DIR    = ${TRAIN_LOG_DIR}"

# ---- Delegate to inner script ----
bash "${INNER_SCRIPT}"
