#!/usr/bin/env bash
# Real DeepSpeed validation for Qwen3-MoE and the Megatron mmap dataset.
#
# Start this script independently on NODE_RANK=0..8.  Physical nodes 0..7
# host the healthy 64-rank training world.  Physical node 8 runs the recovery
# coordinator and remains outside torch.distributed until a failure.  It then
# takes over the failed logical node while the other seven nodes enter the same
# mixed-version recovery epoch without rolling healthy state back.
#
# Cases:
#   TEST_MODE=hot_swap  PP=8, EP=8, ZeRO-1, node-8 hot replacement
#   TEST_MODE=zero2     PP=1, EP=8, ZeRO-2 optimizer D2H/H2H replication
#   TEST_MODE=combined  PP=1, EP=8, ZeRO-2 plus node-8 hot replacement
#   TEST_MODE=all       run hot_swap and zero2 sequentially

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKLOAD="${SCRIPT_DIR}/deepspeed_qwen3_moe_pretrain.py"
DEEPSPEED_ROOT="${SCRIPT_DIR}/DeepSpeed"
ADAPTER_ROOT="${SCRIPT_DIR}/deepspeed_adapter"

TEST_MODE="${TEST_MODE:-hot_swap}"
TRAINING_NNODES="${TRAINING_NNODES:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
SPARE_NODE_RANK="${SPARE_NODE_RANK:-${TRAINING_NNODES}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-20121}"
BASE_MASTER_PORT="${MASTER_PORT}"
HOT_SPARE_ADDR="${MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR:-${ELASTIC_WATCHER_ADDR:-}}"
HOT_SPARE_PORT="${MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT:-$((BASE_MASTER_PORT + 100))}"
HOT_SPARE_HEARTBEAT_TIMEOUT="${HOT_SPARE_HEARTBEAT_TIMEOUT:-30}"
HOT_SPARE_RECOVERY_TIMEOUT="${HOT_SPARE_RECOVERY_TIMEOUT:-300}"
HOT_SPARE_STARTUP_TIMEOUT="${HOT_SPARE_STARTUP_TIMEOUT:-600}"
RUN_ID="${RUN_ID:-ds-real-${MASTER_PORT}}"
RESET_RUN="${RESET_RUN:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-17}"
FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
EP_SIZE="${EP_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-14400}"
DRY_RUN="${DRY_RUN:-0}"
PACKED_EXPERT_CHECKPOINT="${DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT:-1}"
PACKED_EXPERT_CACHE="${MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE:-1}"
HANDOFF_ROOT_BASE="${MOEGAMBIT_RECOVERY_HANDOFF_DIR:-/tmp/moegambit-deepspeed-handoff}"

MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/tokenizer}"
DATA_PATH="${DATA_PATH:-/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document}"
RUN_ROOT_BASE="${RUN_ROOT:-/mnt/ais-c1/dataset/zds/86hotspare/deepspeed_real}"
RUN_ROOT="${RUN_ROOT_BASE%/}/${RUN_ID}"
HOSTFILE="${DEEPSPEED_HOSTFILE:-/tmp/moegambit-deepspeed-hosts-${MASTER_PORT}}"

fail() {
  echo "[deepspeed-real-launch] ERROR: $*" >&2
  exit 64
}

acquire_launcher_lock() {
  local lock_root="${MOEGAMBIT_RUN_LOCK_DIR:-/tmp}"
  local lock_token
  lock_token="$(
    printf '%s' "${RUN_ID}-${TEST_MODE}-${MASTER_PORT}-node${NODE_RANK}" \
      | tr -c '[:alnum:]_.-' '_'
  )"
  LAUNCHER_LOCK_PATH="${lock_root%/}/moegambit-${lock_token}.lock"
  mkdir -p "${lock_root}"

  if command -v flock >/dev/null 2>&1; then
    exec 9>>"${LAUNCHER_LOCK_PATH}"
    if ! flock -n 9; then
      fail "another launcher already owns ${LAUNCHER_LOCK_PATH}; do not start the same RUN_ID twice"
    fi
    printf 'pid=%s started=%s\n' "$$" "$(date -Is)" >&9
    return
  fi

  LAUNCHER_LOCK_PATH="${LAUNCHER_LOCK_PATH}.d"
  if ! mkdir "${LAUNCHER_LOCK_PATH}" 2>/dev/null; then
    fail "another launcher already owns ${LAUNCHER_LOCK_PATH}; remove a stale lock only after confirming no run is active"
  fi
  trap 'rm -rf "${LAUNCHER_LOCK_PATH}"' EXIT
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer"
}

for pair in \
  "TRAINING_NNODES:${TRAINING_NNODES}" \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "NODE_RANK:${NODE_RANK}" \
  "SPARE_NODE_RANK:${SPARE_NODE_RANK}" \
  "MASTER_PORT:${MASTER_PORT}" \
  "HOT_SPARE_PORT:${HOT_SPARE_PORT}" \
  "HOT_SPARE_RECOVERY_TIMEOUT:${HOT_SPARE_RECOVERY_TIMEOUT}" \
  "HOT_SPARE_STARTUP_TIMEOUT:${HOT_SPARE_STARTUP_TIMEOUT}" \
  "TRAIN_ITERS:${TRAIN_ITERS}" \
  "FAULT_INJECT_STEP:${FAULT_INJECT_STEP}" \
  "FAULT_INJECT_NODE:${FAULT_INJECT_NODE}" \
  "FAULT_INJECT_LOCAL_RANK:${FAULT_INJECT_LOCAL_RANK}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "EP_SIZE:${EP_SIZE}" \
  "SEQ_LENGTH:${SEQ_LENGTH}" \
  "MICRO_BATCH_SIZE:${MICRO_BATCH_SIZE}" \
  "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done

if (( HOT_SPARE_RECOVERY_TIMEOUT < HOT_SPARE_STARTUP_TIMEOUT )); then
  echo "[deepspeed-real-launch] recovery timeout " \
    "${HOT_SPARE_RECOVERY_TIMEOUT}s is shorter than cold-start timeout " \
    "${HOT_SPARE_STARTUP_TIMEOUT}s; using ${HOT_SPARE_STARTUP_TIMEOUT}s"
  HOT_SPARE_RECOVERY_TIMEOUT="${HOT_SPARE_STARTUP_TIMEOUT}"
fi

case "${TEST_MODE}" in
  hot_swap|zero2|combined|all) ;;
  *) fail "TEST_MODE must be hot_swap, zero2, combined, or all" ;;
esac
(( TRAINING_NNODES > 0 )) || fail "TRAINING_NNODES must be positive"
(( NPROC_PER_NODE > 0 )) || fail "NPROC_PER_NODE must be positive"
(( SPARE_NODE_RANK == TRAINING_NNODES )) || fail \
  "SPARE_NODE_RANK must equal TRAINING_NNODES (${TRAINING_NNODES})"
(( NODE_RANK <= SPARE_NODE_RANK )) || fail \
  "NODE_RANK must be in 0..${SPARE_NODE_RANK}"
(( EP_SIZE > 0 )) || fail "EP_SIZE must be positive"
(( SAVE_INTERVAL > 0 )) || fail "SAVE_INTERVAL must be positive"
if [[ "${TEST_MODE}" != "zero2" ]]; then
  (( TRAIN_ITERS > FAULT_INJECT_STEP )) || fail \
    "TRAIN_ITERS must be greater than FAULT_INJECT_STEP"
  (( FAULT_INJECT_STEP > SAVE_INTERVAL )) || fail \
    "FAULT_INJECT_STEP must be after the first checkpoint"
fi
(( FAULT_INJECT_NODE < TRAINING_NNODES )) || fail \
  "FAULT_INJECT_NODE is outside the training worker set"
(( FAULT_INJECT_LOCAL_RANK < NPROC_PER_NODE )) || fail \
  "FAULT_INJECT_LOCAL_RANK is outside the local worker set"

[[ -f "${WORKLOAD}" ]] || fail "missing workload: ${WORKLOAD}"
[[ -d "${DEEPSPEED_ROOT}/deepspeed" ]] || fail "missing DeepSpeed source: ${DEEPSPEED_ROOT}"
[[ -d "${ADAPTER_ROOT}/moegambit_deepspeed" ]] || fail "missing DeepSpeed adapter: ${ADAPTER_ROOT}"
[[ -f "${MODEL_CONFIG}/config.json" ]] || fail "missing Qwen config: ${MODEL_CONFIG}/config.json"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${DATA_PATH}.idx" ]] || fail "missing dataset index: ${DATA_PATH}.idx"
  [[ -f "${DATA_PATH}.bin" ]] || fail "missing dataset data: ${DATA_PATH}.bin"
fi
[[ "${MASTER_ADDR}" != "127.0.0.1" ]] || {
  (( TRAINING_NNODES == 1 )) || fail "set MASTER_ADDR to the routable rank-0 address"
}
if [[ "${TEST_MODE}" != "zero2" ]]; then
  [[ -n "${HOT_SPARE_ADDR}" ]] || fail \
    "set MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR or ELASTIC_WATCHER_ADDR to the routable spare-node address"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export LOCAL_WORLD_SIZE="${NPROC_PER_NODE}"
export PYTHONPATH="${ADAPTER_ROOT}:${DEEPSPEED_ROOT}:${SCRIPT_DIR}/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
export MOEGAMBIT_ZERO2_REPLICATION_TIMEOUT="${MOEGAMBIT_ZERO2_REPLICATION_TIMEOUT:-1800}"
# One local staging slot plus one peer slot. Two slots would use roughly four
# optimizer shards of host memory per rank for this 30B-class model.
export MOEGAMBIT_ZERO2_BUFFER_SLOTS="${MOEGAMBIT_ZERO2_BUFFER_SLOTS:-1}"
export MOEGAMBIT_DEEPSPEED_APPLICATION_CHECKPOINT=0
export MOEGAMBIT_RELAY_RANK_LOG="${MOEGAMBIT_RELAY_RANK_LOG:-key}"
export MOEGAMBIT_DEEPSPEED_LOG_LEVEL="${MOEGAMBIT_DEEPSPEED_LOG_LEVEL:-info}"
export MOEGAMBIT_LAUNCHER_LOG_LEVEL="${MOEGAMBIT_LAUNCHER_LOG_LEVEL:-warning}"
export MOEGAMBIT_STANDBY_RESIDENT="${MOEGAMBIT_STANDBY_RESIDENT:-1}"
export MOEGAMBIT_STANDBY_READY_TIMEOUT="${MOEGAMBIT_STANDBY_READY_TIMEOUT:-10}"
export MOEGAMBIT_RECOVERY_GPU_MODEL_BUILD="${MOEGAMBIT_RECOVERY_GPU_MODEL_BUILD:-1}"
export MOEGAMBIT_RECOVERY_FORCE_PREEMPT="${MOEGAMBIT_RECOVERY_FORCE_PREEMPT:-1}"
export MOEGAMBIT_STANDBY_PREFETCH="${MOEGAMBIT_STANDBY_PREFETCH:-1}"
export MOEGAMBIT_STANDBY_PREFETCH_LOGICAL_NODE="$FAULT_INJECT_NODE"
export MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB="${MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB:-128}"
export MOEGAMBIT_STANDBY_PACKED_EXPERT_PIN_MEMORY="${MOEGAMBIT_STANDBY_PACKED_EXPERT_PIN_MEMORY:-1}"
export MOEGAMBIT_STANDBY_PACKED_EXPERT_MAX_GIB_PER_RANK="${MOEGAMBIT_STANDBY_PACKED_EXPERT_MAX_GIB_PER_RANK:-16}"
export MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR="${HOT_SPARE_ADDR}"
export MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT="${HOT_SPARE_PORT}"

if [[ -n "${MOEGAMBIT_REPLICA_ADVERTISE_ADDR:-}" ]]; then
  export MOEGAMBIT_REPLICA_ADVERTISE_ADDR
fi

generate_hostfile() {
  : > "${HOSTFILE}"
  local node
  for ((node = 0; node < TRAINING_NNODES; node++)); do
    printf 'deepspeed-node-%d slots=%d\n' \
      "${node}" "${NPROC_PER_NODE}" >> "${HOSTFILE}"
  done
}

preflight() {
  "${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import torch
import transformers
import deepspeed
import moegambit_deepspeed

assert torch.cuda.is_available(), "CUDA is not available"
assert Version(transformers.__version__) >= Version("5.0.0"), (
    "DeepSpeed Qwen3 AutoEP requires transformers>=5.0.0; "
    f"found {transformers.__version__}"
)
print(
    "[deepspeed-real-launch] preflight "
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"deepspeed={deepspeed.__version__} "
    f"grouped_mm={callable(getattr(torch, '_grouped_mm', None))}",
    flush=True,
)
PY
}

run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=60 \
      "${TEST_TIMEOUT_SECONDS}" "$@"
  else
    "$@"
  fi
}

validate_case() {
  local case_name="$1"
  local expect_fault="$2"
  local expect_zero2="$3"
  local state_dir="${RUN_ROOT}/${case_name}/state"
  local completion="${state_dir}/completed.json"

  [[ -f "${completion}" ]] || fail \
    "${case_name} launcher exited without ${completion}"
  "${PYTHON_BIN}" - \
    "${completion}" "${state_dir}/fault_injected.json" \
    "${expect_fault}" "${expect_zero2}" "${TRAIN_ITERS}" \
    "${FAULT_INJECT_STEP}" <<'PY'
import json
import pathlib
import sys

completion = pathlib.Path(sys.argv[1])
fault_marker = pathlib.Path(sys.argv[2])
expect_fault = sys.argv[3] == "1"
expect_zero2 = sys.argv[4] == "1"
train_iters = int(sys.argv[5])
fault_step = int(sys.argv[6])
state = json.loads(completion.read_text(encoding="utf-8"))

assert state["global_step"] == train_iters, state
if expect_fault:
    assert fault_marker.is_file(), "fault marker was not written"
    assert state["restart_count"] >= 1, state
    contract = state.get("recovery_contract")
    assert contract, "mixed-version recovery contract is missing"
    assert contract["mode"] == "mixed_version", contract
    assert contract["resume_step"] == fault_step, contract
    assert contract["rollback_steps"] == 0, contract
    assert contract["checkpoint_step"] < fault_step, contract
    assert contract["survivor_state"] == "current_step_handoff", contract
    assert contract["replacement_non_expert_model"] == "current_step_peer", contract
    assert contract["replacement_non_expert_optimizer"] == "current_step_peer_replica", contract
    assert contract["replacement_expert_model"] == "checkpoint", contract
    assert contract["replacement_expert_optimizer"] == "checkpoint", contract
else:
    assert state["restart_count"] == 0, state

if expect_zero2:
    replicas = state.get("zero2_replication")
    assert replicas, "ZeRO-2 replication summary is missing"
    for namespace, replica in replicas.items():
        assert replica["local_replicated_step"] >= train_iters, (
            namespace,
            replica,
        )
        assert replica["peer_committed_step"] >= train_iters, (
            namespace,
            replica,
        )
else:
    assert state.get("zero2_replication") is None, state

print(
    "[deepspeed-real-launch] VALIDATED "
    f"case={state['case']} step={state['global_step']} "
    f"restarts={state['restart_count']} zero={state['zero_stage']}",
    flush=True,
)
PY
}

run_case() {
  local case_name="$1"
  local pp_size="$2"
  local zero_stage="$3"
  local hot_swap="$4"
  local zero2="$5"
  local port_offset="$6"
  local case_root="${RUN_ROOT}/${case_name}"
  local checkpoint_dir="${case_root}/checkpoint"
  local state_dir="${case_root}/state"
  local case_port=$((BASE_MASTER_PORT + port_offset))
  local coordinator_port=$((HOT_SPARE_PORT + port_offset))
  local recovery_run_id="${RUN_ID}-${case_name}"
  local handoff_dir="${HANDOFF_ROOT_BASE%/}/${recovery_run_id}/physical_${NODE_RANK}"
  local fault_rank=$((FAULT_INJECT_NODE * NPROC_PER_NODE + FAULT_INJECT_LOCAL_RANK))
  local fault_step=-1

  if [[ "${hot_swap}" == "1" ]]; then
    fault_step="${FAULT_INJECT_STEP}"
  elif (( NODE_RANK == SPARE_NODE_RANK )); then
    echo "[deepspeed-real-launch] node ${NODE_RANK} is standby-only; skipping ${case_name}"
    return 0
  fi

  local reset_owner=0
  if [[ "${hot_swap}" == "1" ]]; then
    # The spare owns the coordinator, so it must finish cleanup before it
    # starts accepting active-node registrations.
    reset_owner="${SPARE_NODE_RANK}"
  fi
  if (( NODE_RANK == reset_owner )) && [[ "${RESET_RUN}" == "1" ]]; then
    rm -rf "${case_root}"
  fi
  if [[ "${RESET_RUN}" == "1" ]]; then
    rm -rf "${handoff_dir}"
  fi
  mkdir -p "${checkpoint_dir}" "${state_dir}"
  mkdir -p "${handoff_dir}"

  export MASTER_ADDR
  export MASTER_PORT="${case_port}"
  export ELASTIC_RUN_ID="${recovery_run_id}"
  export MOEGAMBIT_HOT_SWAP="${hot_swap}"
  export MOEGAMBIT_ZERO2="${zero2}"
  export MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE="${hot_swap}"
  export MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF="${hot_swap}"
  export MOEGAMBIT_RECOVERY_HANDOFF_DIR="${handoff_dir}"
  export MOEGAMBIT_RECOVERY_HANDOFF_TIMEOUT="${MOEGAMBIT_RECOVERY_HANDOFF_TIMEOUT:-180}"
  export MOEGAMBIT_ZERO2_REPLICA_SCOPE="$(
    if [[ "${hot_swap}" == "1" ]]; then
      printf 'non_expert'
    else
      printf 'all'
    fi
  )"
  export MOEGAMBIT_REPLICA_FAILURE_DOMAIN_SIZE="$(
    if [[ "${hot_swap}" == "1" ]]; then
      printf '%s' "${NPROC_PER_NODE}"
    else
      printf '1'
    fi
  )"
  export DEEPSPEED_MOEGAMBIT_HOT_SWAP="${hot_swap}"
  export DEEPSPEED_MOEGAMBIT_ZERO2="${zero2}"
  if [[ "${hot_swap}" == "1" ]]; then
    export DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT="${PACKED_EXPERT_CHECKPOINT}"
    export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE="${PACKED_EXPERT_CACHE}"
  else
    export DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT=0
    export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=0
  fi
  export MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT="${coordinator_port}"
  export MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR="${checkpoint_dir}"
  export MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL="${SAVE_INTERVAL}"
  export MOEGAMBIT_DEEPSPEED_MIN_NODES="${TRAINING_NNODES}"
  export MOEGAMBIT_DEEPSPEED_MAX_NODES="${TRAINING_NNODES}"
  export MOEGAMBIT_DEEPSPEED_EXTERNAL_ELASTIC="${hot_swap}"
  export MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY="$(
    if [[ "${hot_swap}" == "1" ]]; then
      printf 'mixed_version_survivor_handoff'
    else
      printf 'disabled'
    fi
  )"
  unset TORCHELASTIC_RESTART_COUNT 2>/dev/null || true
  unset MOEGAMBIT_RECOVERY_EPOCH 2>/dev/null || true
  unset MOEGAMBIT_LOGICAL_NODE_RANK 2>/dev/null || true
  unset MOEGAMBIT_PHYSICAL_NODE_RANK 2>/dev/null || true

  echo "[deepspeed-real-launch] case=${case_name} node=${NODE_RANK} "\
"PP=${pp_size} EP=${EP_SIZE} ZeRO=${zero_stage} hot_swap=${hot_swap} zero2=${zero2} "\
"hybrid_restore=${MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE} "\
"survivor_handoff=${MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF} "\
"optimizer_peer_replica=${hot_swap} replica_scope=${MOEGAMBIT_ZERO2_REPLICA_SCOPE} "\
"replica_failure_domain=${MOEGAMBIT_REPLICA_FAILURE_DOMAIN_SIZE} "\
"resident_standby=${MOEGAMBIT_STANDBY_RESIDENT} "\
"packed_experts=${DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT} "\
"packed_cache=${MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE} "\
"recovery_gpu_build=${MOEGAMBIT_RECOVERY_GPU_MODEL_BUILD} "\
"force_preempt=${MOEGAMBIT_RECOVERY_FORCE_PREEMPT} "\
"handoff_dir=${MOEGAMBIT_RECOVERY_HANDOFF_DIR} "\
"spare=${SPARE_NODE_RANK} coordinator=${HOT_SPARE_ADDR}:${coordinator_port} "\
"recovery_timeout=${HOT_SPARE_RECOVERY_TIMEOUT}s"
  local launch_node_rank="${NODE_RANK}"
  local launch_master_addr="${MASTER_ADDR}"
  local launch_master_port="${case_port}"
  local rank_log_dir="${state_dir}/rank_logs/epoch_0/node_${NODE_RANK}"
  if [[ "${hot_swap}" == "1" ]]; then
    launch_node_rank="{logical_node}"
    launch_master_addr="{master_addr}"
    launch_master_port="{master_port}"
    rank_log_dir="${state_dir}/rank_logs/epoch_{recovery_epoch}/node_{physical_node}"
  fi
  mkdir -p "${state_dir}/rank_logs"
  echo "[deepspeed-real-launch] full_rank_logs=${rank_log_dir}"
  local -a launch_command=(
    "${PYTHON_BIN}" -u -m deepspeed.launcher.runner \
    --hostfile "${HOSTFILE}" \
    --no_ssh \
    --node_rank "${launch_node_rank}" \
    --num_nodes "${TRAINING_NNODES}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --enable_each_rank_log "${rank_log_dir}" \
    --log_level "${MOEGAMBIT_LAUNCHER_LOG_LEVEL}" \
    --master_addr "${launch_master_addr}" \
    --master_port "${launch_master_port}"
  )
  launch_command+=(
    "${WORKLOAD}" \
    --case-name "${case_name}" \
    --model-config "${MODEL_CONFIG}" \
    --data-path "${DATA_PATH}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --state-dir "${state_dir}" \
    --pipeline-parallel-size "${pp_size}" \
    --expert-parallel-size "${EP_SIZE}" \
    --zero-stage "${zero_stage}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --sequence-length "${SEQ_LENGTH}" \
    --train-iters "${TRAIN_ITERS}" \
    --fault-step "${fault_step}" \
    --fault-rank "${fault_rank}" \
    --activation-checkpointing 1
  )
  local -a supervised_command=("${launch_command[@]}")
  if [[ "${hot_swap}" == "1" ]]; then
    local supervisor_mode="agent"
    if (( NODE_RANK == SPARE_NODE_RANK )); then
      supervisor_mode="coordinator-agent"
    fi
    supervised_command=(
      "${PYTHON_BIN}" -u -m moegambit.runtime.hot_spare \
      --mode "${supervisor_mode}" \
      --coordinator-host "${HOT_SPARE_ADDR}" \
      --coordinator-port "${coordinator_port}" \
      --run-id "${recovery_run_id}" \
      --training-nodes "${TRAINING_NNODES}" \
      --spare-node "${SPARE_NODE_RANK}" \
      --physical-node "${NODE_RANK}" \
      --base-master-port "${case_port}" \
      --heartbeat-timeout "${HOT_SPARE_HEARTBEAT_TIMEOUT}" \
      --recovery-timeout "${HOT_SPARE_RECOVERY_TIMEOUT}" \
      --startup-timeout "${HOT_SPARE_STARTUP_TIMEOUT}" \
      --state-path "${state_dir}/hot_spare_coordinator.json"
    )
    local advertise_addr="${MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR:-}"
    if [[ -z "${advertise_addr}" ]] && (( NODE_RANK == 0 )); then
      advertise_addr="${MASTER_ADDR}"
    elif [[ -z "${advertise_addr}" ]] && (( NODE_RANK == SPARE_NODE_RANK )); then
      advertise_addr="${HOT_SPARE_ADDR}"
    fi
    if [[ -n "${advertise_addr}" ]]; then
      supervised_command+=(--advertise-addr "${advertise_addr}")
    fi
    if [[ "${supervisor_mode}" == "coordinator-agent" ]]; then
      supervised_command+=(--listen-host "0.0.0.0")
    fi
    supervised_command+=(-- "${launch_command[@]}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[deepspeed-real-launch] DRY_RUN'
    printf ' %q' "${supervised_command[@]}"
    printf '\n'
    return 0
  fi
  run_with_timeout "${supervised_command[@]}"

  validate_case "${case_name}" "${hot_swap}" "${zero2}"
}

acquire_launcher_lock
generate_hostfile
if [[ "${DRY_RUN}" != "1" ]]; then
  preflight
fi
mkdir -p "${RUN_ROOT}"

case "${TEST_MODE}" in
  hot_swap)
    run_case hot_swap_pp8_ep8 8 1 1 0 0
    ;;
  zero2)
    run_case zero2_pp1_ep8 1 2 0 1 20
    ;;
  combined)
    run_case combined_pp1_ep8 1 2 1 1 40
    ;;
  all)
    run_case hot_swap_pp8_ep8 8 1 1 0 0
    run_case zero2_pp1_ep8 1 2 0 1 20
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[deepspeed-real-launch] dry run complete"
else
  echo "[deepspeed-real-launch] all requested cases passed"
fi
