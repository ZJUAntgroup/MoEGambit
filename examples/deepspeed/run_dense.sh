#!/usr/bin/env bash
# Small dense DeepSpeed fault-recovery example: two training nodes, one spare.
# Run on every node with NODE_RANK=0, 1, and 2 respectively. RUN_ROOT must
# refer to the same shared filesystem path on all nodes.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE_RANK="${NODE_RANK:-0}"
TRAINING_NNODES="${TRAINING_NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SPARE_NODE_RANK="$TRAINING_NNODES"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-20121}"
SPARE_ADDR="${ELASTIC_WATCHER_ADDR:-127.0.0.1}"
COORDINATOR_PORT="${MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT:-$((MASTER_PORT + 100))}"
RUN_ID="${RUN_ID:-dense-example}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-5}"
FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-0}"
DRY_RUN="${DRY_RUN:-0}"

die() { printf '[dense-deepspeed] %s\n' "$*" >&2; exit 64; }
for value in "$NODE_RANK" "$TRAINING_NNODES" "$NPROC_PER_NODE" \
  "$MASTER_PORT" "$COORDINATOR_PORT" "$TRAIN_ITERS" \
  "$SAVE_INTERVAL" "$FAULT_INJECT_STEP" "$FAULT_INJECT_NODE" \
  "$FAULT_INJECT_LOCAL_RANK"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "topology, port, and step values must be integers"
done
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "RUN_ID may contain only letters, numbers, dots, underscores, and hyphens"
(( TRAINING_NNODES >= 2 && NPROC_PER_NODE >= 1 )) || die "at least two data-parallel training ranks are required"
(( NODE_RANK <= SPARE_NODE_RANK )) || die "NODE_RANK must be 0..${SPARE_NODE_RANK}"
(( FAULT_INJECT_NODE < TRAINING_NNODES && FAULT_INJECT_LOCAL_RANK < NPROC_PER_NODE )) || die "fault rank is outside the training world"
(( TRAIN_ITERS > FAULT_INJECT_STEP && FAULT_INJECT_STEP > SAVE_INTERVAL )) || die "fault step must follow a checkpoint and precede completion"
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -n "${RUN_ROOT:-}" ]] || die "set RUN_ROOT to a shared, writable directory"
  [[ "$RUN_ROOT" = /* ]] || die "RUN_ROOT must be an absolute path shared by all nodes"
  [[ "$MASTER_ADDR" != 127.0.0.1 && "$SPARE_ADDR" != 127.0.0.1 ]] || die "set routable MASTER_ADDR and ELASTIC_WATCHER_ADDR"
  [[ -d "$ROOT/DeepSpeed/deepspeed" ]] || die "DeepSpeed source is missing"
fi
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/dense-deepspeed-dry-run}"
CASE_ROOT="${RUN_ROOT%/}/${RUN_ID}"
CHECKPOINT_DIR="$CASE_ROOT/checkpoints"
STATE_DIR="$CASE_ROOT/state"
HOSTFILE="$CASE_ROOT/hosts.node_${NODE_RANK}"
FAULT_RANK=$((FAULT_INJECT_NODE * NPROC_PER_NODE + FAULT_INJECT_LOCAL_RANK))

export PYTHONPATH="${ROOT}/src:${ROOT}/DeepSpeed${PYTHONPATH:+:${PYTHONPATH}}"
export MOEGAMBIT_MODEL_KIND=dense
export MOEGAMBIT_HOT_SWAP=1
export MOEGAMBIT_ZERO2=0
export MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE=1
export MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY=1
export MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY=rank_in_process_hybrid
export MOEGAMBIT_ZERO2_REPLICA_SCOPE=all
export MOEGAMBIT_REPLICA_FAILURE_DOMAIN_SIZE="$NPROC_PER_NODE"
export MOEGAMBIT_STANDBY_RESIDENT=0
export DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT=0
export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=0
export MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR="$SPARE_ADDR"
export MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT="$COORDINATOR_PORT"
export MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL="$SAVE_INTERVAL"
export MOEGAMBIT_DEEPSPEED_EXTERNAL_COORDINATOR=1
export MASTER_ADDR MASTER_PORT

launch_node_rank='{logical_node}'
launch_master_addr='{master_addr}'
launch_master_port='{master_port}'
runner=(python3 -u -m deepspeed.launcher.runner
  --hostfile "$HOSTFILE" --no_ssh
  --node_rank "$launch_node_rank" --num_nodes "$TRAINING_NNODES"
  --num_gpus "$NPROC_PER_NODE"
  --enable_each_rank_log "$STATE_DIR/rank_logs/epoch_{recovery_epoch}/node_{physical_node}"
  --master_addr "$launch_master_addr" --master_port "$launch_master_port"
  "$ROOT/examples/deepspeed/dense_workload.py"
  --checkpoint-dir "$CHECKPOINT_DIR" --state-dir "$STATE_DIR"
  --train-iters "$TRAIN_ITERS" --fault-step "$FAULT_INJECT_STEP"
  --fault-rank "$FAULT_RANK")
supervisor="$ROOT/elastic_launcher.py"
if (( NODE_RANK == SPARE_NODE_RANK )); then
  supervisor="$ROOT/elastic_watcher.py"
fi
command=(python3 -u "$supervisor" --adapter deepspeed
  --coordinator-host "$SPARE_ADDR" --coordinator-port "$COORDINATOR_PORT"
  --run-id "$RUN_ID" --training-nodes "$TRAINING_NNODES"
  --spare-node "$SPARE_NODE_RANK" --physical-node "$NODE_RANK"
  --local-world-size "$NPROC_PER_NODE" --base-master-port "$MASTER_PORT"
  --state-path "$STATE_DIR/coordinator.json" --rank-hot-swap)
if (( NODE_RANK == SPARE_NODE_RANK )); then
  command+=(--listen-host 0.0.0.0 --advertise-addr "$SPARE_ADDR")
elif (( NODE_RANK == 0 )); then
  command+=(--advertise-addr "$MASTER_ADDR")
fi
command+=(-- "${runner[@]}")

printf '[dense-deepspeed] node=%s command:' "$NODE_RANK"
printf ' %q' "${command[@]}"
printf '\n'
[[ "$DRY_RUN" == 1 ]] && exit 0
[[ ! -e "$STATE_DIR/completed.json" ]] || die "run already completed; choose a new RUN_ID"
mkdir -p "$CHECKPOINT_DIR" "$STATE_DIR/rank_logs"
for ((node = 0; node < TRAINING_NNODES; node++)); do
  printf 'training-node-%d slots=%d\n' "$node" "$NPROC_PER_NODE"
done > "$HOSTFILE"
cd "$ROOT"
"${command[@]}"
if (( NODE_RANK == 0 )); then
  python3 - "$STATE_DIR/completed.json" "$TRAIN_ITERS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing completion artifact: {path}")
result = json.loads(path.read_text(encoding="utf-8"))
assert result["global_step"] == int(sys.argv[2]), result
assert result["restart_count"] >= 1, result
assert result["recovery_contract"]["mode"] == "rank_in_process_peer", result
assert result["recovery_contract"]["expert_staleness"] == 0, result
print(f"[dense-deepspeed] PASS step={result['global_step']}")
PY
fi
