#!/usr/bin/env bash
# =============================================================================
# run_moe64_hotspare.sh
#
# Scheduler-level hot-spare node replacement for MoEGambit training.
#
# Architecture:
#   - Maintains a pool of training nodes (8) and spare nodes (N)
#   - Launches torchrun on 8 training nodes (64 GPUs)
#   - On training crash, probes all nodes via SSH to find the failed one
#   - Replaces the failed node with a spare from the pool
#   - Restarts torchrun with the updated node list
#   - Training resumes from checkpoint via hybrid recovery path
#
# The spare nodes should be pre-warmed (Python env, CUDA, dependencies ready)
# so restart is faster than waiting for a cold node to come back online.
#
# Node list format (environment variable or file):
#   TRAIN_NODES="node0,node1,node2,node3,node4,node5,node6,node7"
#   SPARE_NODES="spare0,spare1"
#
# This script is the MASTER launcher — run it on the control node.
# Each training node is launched via SSH (pdsh/parallel-ssh style).
#
# For single-node testing (simulated), set SINGLE_NODE_MODE=1 and it
# behaves like run_moe64_par.sh with retry logic.
#
# Overrides:
#   TRAIN_NODES / SPARE_NODES   comma-separated hostnames/IPs
#   NNODES / MASTER_ADDR / MASTER_PORT
#   CKPT_DIR / TRAIN_LOG_DIR
#   MAX_RETRIES                 max fault-recovery cycles (default 10)
#   NODE_PROBE_TIMEOUT          SSH probe timeout in seconds (default 5)
#   SINGLE_NODE_MODE            set to 1 for local testing without SSH
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/run_moe64_par.sh"
if [ ! -f "${INNER_SCRIPT}" ]; then
  echo "[hotspare] missing ${INNER_SCRIPT}"
  exit 2
fi

# =============================================================================
# Configuration
# =============================================================================

# Node lists (comma-separated IPs or hostnames)
TRAIN_NODES="${TRAIN_NODES:-}"
SPARE_NODES="${SPARE_NODES:-}"

# Single-node mode: skip SSH probing, just retry like run_moe64_par.sh
SINGLE_NODE_MODE="${SINGLE_NODE_MODE:-1}"

# Parallelism
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export MODE=moegambit

# Recovery policy
export BSR_HOT_SPARE_POOL=1
export BSR_NUM_HOT_SPARES="${BSR_NUM_HOT_SPARES:-8}"
export BSR_GAP_AWARE_RECOVERY=1
export BSR_RECOVERY_POLICY_TYPE="${BSR_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export BSR_GAP_THRESHOLD="${BSR_GAP_THRESHOLD:-100}"

# Fault injection
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

# Retry settings
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY="${RETRY_DELAY:-15}"
NODE_PROBE_TIMEOUT="${NODE_PROBE_TIMEOUT:-5}"

export NNODES="${NNODES:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20115}"

mkdir -p "${CKPT_DIR}"

# =============================================================================
# Node management functions
# =============================================================================

# Convert comma-separated string to bash array
IFS=',' read -ra _TRAIN_ARRAY <<< "${TRAIN_NODES}"
IFS=',' read -ra _SPARE_ARRAY <<< "${SPARE_NODES}"

# Track which spares have been used
_SPARE_INDEX=0
_REPLACEMENTS_MADE=0

probe_node() {
  # Check if a node is reachable via SSH (timeout-based)
  local node="$1"
  ssh -o ConnectTimeout="${NODE_PROBE_TIMEOUT}" \
      -o StrictHostKeyChecking=no \
      -o BatchMode=yes \
      "${node}" "echo ok" >/dev/null 2>&1
}

find_failed_node() {
  # Probe all training nodes, return the index of the first unreachable one
  # Returns -1 if all nodes are healthy
  local i=0
  for node in "${_TRAIN_ARRAY[@]}"; do
    if ! probe_node "${node}"; then
      echo "${i}"
      return 0
    fi
    i=$((i + 1))
  done
  echo "-1"
  return 0
}

replace_failed_node() {
  # Replace the failed node at index $1 with the next available spare
  local failed_idx="$1"
  local failed_node="${_TRAIN_ARRAY[$failed_idx]}"

  if [ "${_SPARE_INDEX}" -ge "${#_SPARE_ARRAY[@]}" ]; then
    echo "[hotspare] ERROR: no spare nodes left! Cannot replace ${failed_node}"
    return 1
  fi

  local spare_node="${_SPARE_ARRAY[$_SPARE_INDEX]}"
  _SPARE_INDEX=$((_SPARE_INDEX + 1))
  _REPLACEMENTS_MADE=$((_REPLACEMENTS_MADE + 1))

  echo "[hotspare] REPLACING node ${failed_idx} (${failed_node}) with spare (${spare_node})"
  _TRAIN_ARRAY[$failed_idx]="${spare_node}"

  # Update MASTER_ADDR if the failed node was the master (node 0)
  if [ "${failed_idx}" -eq 0 ]; then
    export MASTER_ADDR="${spare_node}"
    echo "[hotspare] master node failed — new MASTER_ADDR=${MASTER_ADDR}"
  fi

  return 0
}

# =============================================================================
# Main loop
# =============================================================================

echo "[hotspare] =============================================="
echo "[hotspare] MoEGambit Hot-Spare Node Pool Launcher"
echo "[hotspare] =============================================="
echo "[hotspare] Training nodes: ${TRAIN_NODES:-<single-node-mode>}"
echo "[hotspare] Spare nodes:    ${SPARE_NODES:-<none>}"
echo "[hotspare] Single-node:    ${SINGLE_NODE_MODE}"
echo "[hotspare] Max retries:    ${MAX_RETRIES}"
echo "[hotspare] CKPT_DIR:       ${CKPT_DIR}"
echo "[hotspare] =============================================="

retry=0

while [ "${retry}" -lt "${MAX_RETRIES}" ]; do
  echo "[hotspare] === Attempt $((retry + 1))/${MAX_RETRIES} ==="

  if [ "${SINGLE_NODE_MODE}" = "1" ]; then
    # ---- Single-node mode: just run inner script directly ----
    export NODE_RANK="${NODE_RANK:-0}"
    bash "${INNER_SCRIPT}"
    rc=$?
  else
    # ---- Multi-node mode: launch via SSH on each node ----
    # Update MASTER_ADDR to first training node if not explicitly set
    if [ "${MASTER_ADDR}" = "127.0.0.1" ] && [ "${#_TRAIN_ARRAY[@]}" -gt 0 ]; then
      export MASTER_ADDR="${_TRAIN_ARRAY[0]}"
    fi

    # Launch on all training nodes in parallel
    pids=()
    for i in $(seq 0 $((NNODES - 1))); do
      node="${_TRAIN_ARRAY[$i]}"
      echo "[hotspare] launching on node ${i} (${node})..."
      ssh -o StrictHostKeyChecking=no "${node}" \
        "cd $(pwd) && NODE_RANK=${i} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} bash ${INNER_SCRIPT}" \
        >> "${TRAIN_LOG_DIR}/node${i}_${node}.log" 2>&1 &
      pids+=($!)
    done

    # Wait for any process to exit (indicates crash or completion)
    rc=0
    for pid in "${pids[@]}"; do
      wait "${pid}" || rc=$?
    done
  fi

  # ---- Check result ----
  if [ "${rc}" -eq 0 ]; then
    echo "[hotspare] training finished normally"
    echo "[hotspare] total replacements made: ${_REPLACEMENTS_MADE}"
    exit 0
  fi

  echo "[hotspare] training failed with rc=${rc} (attempt $((retry + 1))/${MAX_RETRIES})"

  # ---- Try to identify and replace the failed node ----
  if [ "${SINGLE_NODE_MODE}" != "1" ] && [ "${#_SPARE_ARRAY[@]}" -gt 0 ]; then
    failed_idx=$(find_failed_node)
    if [ "${failed_idx}" -ge 0 ]; then
      if ! replace_failed_node "${failed_idx}"; then
        echo "[hotspare] cannot replace — no spares left, exiting"
        exit "${rc}"
      fi
      echo "[hotspare] node replaced, restarting training..."
    else
      echo "[hotspare] all nodes reachable — crash may be software-level, retrying..."
    fi
  fi

  retry=$((retry + 1))
  if [ "${retry}" -ge "${MAX_RETRIES}" ]; then
    echo "[hotspare] max retries reached, exiting"
    exit "${rc}"
  fi

  sleep "${RETRY_DELAY}"
done
