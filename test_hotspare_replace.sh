#!/usr/bin/env bash
# =============================================================================
# test_hotspare_replace.sh
#
# 测试备用节点替换功能的脚本。
#
# 工作原理：
#   1. 训练正常启动在 NODE_RANK=0-7 (64 GPU)
#   2. 备用节点 (NODE_RANK=8) 运行 elastic_watcher
#   3. 到达指定步数时，watcher 让各节点在迭代边界暂停
#   4. launcher 收齐本节点 survivor 状态后杀死目标 local_rank
#   5. 每个 launcher 向 watcher 提交全 rank 状态版本证明
#   6. watcher 验证 63 个 survivor 位于同一已提交版本
#   7. watcher 在备用节点上用故障 rank 的逻辑身份启动新 worker
#   8. 所有节点重新 init_process_group → sync params → 恢复训练
#
# 使用方式：
#   训练节点 (NODE_RANK=0-7):
#     export NODE_RANK=<0-7>
#     export MASTER_ADDR=<rank0 IP>
#     export ELASTIC_WATCHER_ADDR=<备用节点 IP>
#     bash test_hotspare_replace.sh
#
#   备用节点 (NODE_RANK=8):
#     export NODE_RANK=8
#     export MASTER_ADDR=<rank0 IP>
#     export ELASTIC_WATCHER_ADDR=<本机 IP>
#     bash test_hotspare_replace.sh
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================

# Parallelism
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-1}"
export MEGATRON_PARALLEL_ORDER="${MEGATRON_PARALLEL_ORDER:-tp-cp-ep-dp-pp}"
export MODE=moegambit

# Topology
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-20117}"

# Watcher coordination port
export ELASTIC_WATCHER_PORT="${ELASTIC_WATCHER_PORT:-20200}"

# Determine role
TRAINING_NNODES=8
if [ "${NODE_RANK}" -ge "${TRAINING_NNODES}" ]; then
  IS_SPARE=1
else
  IS_SPARE=0
fi

export NNODES="${TRAINING_NNODES}"

# Elastic watcher address (备用节点的 IP)
export ELASTIC_WATCHER_ADDR="${ELASTIC_WATCHER_ADDR:-${MASTER_ADDR}}"
export ELASTIC_FAULT_DIR="${ELASTIC_FAULT_DIR:-/tmp/elastic_faults}"
if [ "${ELASTIC_RESET_FAULT_DIR:-1}" = "1" ]; then
  rm -rf "${ELASTIC_FAULT_DIR}"
fi
mkdir -p "${ELASTIC_FAULT_DIR}"

# ============================================================================
# Fault injection: watcher 在 step N 时杀死目标节点的指定 local_rank
# ============================================================================
FAULT_INJECT_STEP="${FAULT_INJECT_STEP:-17}"
FAULT_INJECT_NODE="${FAULT_INJECT_NODE:-0}"
# The launcher owns watcher control, so local_rank=0 is also replaceable.
FAULT_INJECT_LOCAL_RANK="${FAULT_INJECT_LOCAL_RANK:-1}"

# ============================================================================
# MOEGAMBIT recovery settings (for surviving nodes during recovery)
# ============================================================================
export MOEGAMBIT_HOT_SPARE_POOL=1
export MOEGAMBIT_NUM_HOT_SPARES="${NPROC_PER_NODE}"
export MOEGAMBIT_GAP_AWARE_RECOVERY=1
export MOEGAMBIT_RECOVERY_POLICY_TYPE="${MOEGAMBIT_RECOVERY_POLICY_TYPE:-rank_exposure_guarded_hybrid}"
export MOEGAMBIT_GAP_THRESHOLD="${MOEGAMBIT_GAP_THRESHOLD:-100}"
export MOEGAMBIT_DELTA_TIME_MIN_GAP="${MOEGAMBIT_DELTA_TIME_MIN_GAP:-0}"
export MOEGAMBIT_MAX_SINGLE_GAP="${MOEGAMBIT_MAX_SINGLE_GAP:-192}"
export MOEGAMBIT_EXPOSURE_WINDOW_STEPS="${MOEGAMBIT_EXPOSURE_WINDOW_STEPS:-20000}"
export MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY="${MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY:-0.1}"
export MOEGAMBIT_NUM_EXPERTS="${MOEGAMBIT_NUM_EXPERTS:-128}"
export ELASTIC_TWO_PHASE_RECOVERY=0

# NO MOEGAMBIT fault injection — we do real kills via the watcher
# (MOEGAMBIT's hard_failure doesn't actually kill processes)
unset MOEGAMBIT_FAULT_INJECT_TYPE 2>/dev/null || true
unset MOEGAMBIT_FAULT_INJECT_STEP 2>/dev/null || true

# Checkpoint: 每 10 步保存一次（确保故障时有近期 checkpoint）
export SAVE_INTERVAL=10
export CKPT_DIR="${CKPT_DIR:-/mnt/ais-c1/dataset/zds/77hotspare/test_replace_ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/log/test_replace}"

# ============================================================================
# NCCL configuration: short timeout for fast failure detection
# ============================================================================
export NCCL_DEBUG=WARN
# CRITICAL: Set to 0 so that NCCL timeout raises a catchable Python RuntimeError
# instead of calling std::abort() (SIGABRT).  With =1, the C++ watchdog kills
# the process immediately on timeout — Python never gets a chance to handle it.
# With =0, the Python thread blocked on the NCCL op gets a RuntimeError after
# the timeout, which our except handler catches and routes to elastic recovery.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
# Disable the HEARTBEAT MONITOR — this is the mechanism that causes SIGABRT
# when the watchdog itself gets stuck (e.g., during error handling).
export TORCH_NCCL_ENABLE_MONITORING=0
# Also increase the heartbeat timeout to be safe
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# Other environment
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-10}"
PHASE_TIMEOUT_DEFAULT=$((DISTRIBUTED_TIMEOUT_MINUTES * 60 + 120))
export ELASTIC_PHASE_TIMEOUT_SECONDS="${ELASTIC_PHASE_TIMEOUT_SECONDS:-${ELASTIC_REBUILD_PHASE_TIMEOUT:-${PHASE_TIMEOUT_DEFAULT}}}"
export ELASTIC_REBUILD_PHASE_TIMEOUT="${ELASTIC_REBUILD_PHASE_TIMEOUT:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS="${ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_BARRIER="${ELASTIC_MPU_GROUP_ORDINAL_BARRIER:-1}"
export ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS="${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS:-${ELASTIC_PHASE_TIMEOUT_SECONDS}}"
export ELASTIC_INIT_PG_DEVICE_ID="${ELASTIC_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_REBUILD_INIT_PG_DEVICE_ID="${ELASTIC_REBUILD_INIT_PG_DEVICE_ID:-0}"
export ELASTIC_REBUILD_EAGER_NCCL_GROUPS="${ELASTIC_REBUILD_EAGER_NCCL_GROUPS:-EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP}"
export ELASTIC_POST_REBUILD_COMM_WARMUP="${ELASTIC_POST_REBUILD_COMM_WARMUP:-0}"
export ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER="${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER:-1}"
export ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP="${ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP:-0}"
export ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST="${ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST:-1}"
export ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT="${ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT:-180}"
export ELASTIC_FALLBACK_RELAUNCH="${ELASTIC_FALLBACK_RELAUNCH:-1}"
export ELASTIC_FALLBACK_EXIT_CODE="${ELASTIC_FALLBACK_EXIT_CODE:-75}"
export ELASTIC_FALLBACK_RESTART_STANDBY="${ELASTIC_FALLBACK_RESTART_STANDBY:-1}"
export ELASTIC_LAUNCHER_CONTROL_PLANE="${ELASTIC_LAUNCHER_CONTROL_PLANE:-1}"
export ELASTIC_LAUNCHER_HEARTBEAT_INTERVAL="${ELASTIC_LAUNCHER_HEARTBEAT_INTERVAL:-1.0}"
export ELASTIC_WATCHER_STARTUP_TIMEOUT_SECONDS="${ELASTIC_WATCHER_STARTUP_TIMEOUT_SECONDS:-600}"
export ELASTIC_WATCHER_CONNECT_INTERVAL_SECONDS="${ELASTIC_WATCHER_CONNECT_INTERVAL_SECONDS:-2}"
export ELASTIC_QUIESCENCE_TIMEOUT_SECONDS="${ELASTIC_QUIESCENCE_TIMEOUT_SECONDS:-300}"
export ELASTIC_NCCL_CLASSIFICATION_GRACE_SECONDS="${ELASTIC_NCCL_CLASSIFICATION_GRACE_SECONDS:-5}"
HOTSPARE_MAX_RETRIES="${HOTSPARE_MAX_RETRIES:-2}"
HOTSPARE_RETRY_DELAY="${HOTSPARE_RETRY_DELAY:-30}"

mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# =============================================================================
# Compute derived values
# =============================================================================

TRAINING_WORLD_SIZE=$((TRAINING_NNODES * NPROC_PER_NODE))  # 64
DP_SIZE=$((TRAINING_WORLD_SIZE / (TP_SIZE * PP_SIZE)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((8 * DP_SIZE))}"
export GLOBAL_BATCH_SIZE

echo "[test-replace] =============================================="
echo "[test-replace] Hot-Spare REPLACEMENT TEST"
echo "[test-replace] =============================================="
echo "[test-replace] NODE_RANK:       ${NODE_RANK} $([ "${IS_SPARE}" = "1" ] && echo "(SPARE WATCHER)" || echo "(TRAINING)")"
echo "[test-replace] Training nodes:  ${TRAINING_NNODES} (${TRAINING_WORLD_SIZE} GPUs)"
echo "[test-replace] Fault inject:    kill node ${FAULT_INJECT_NODE} local_rank ${FAULT_INJECT_LOCAL_RANK} at step ${FAULT_INJECT_STEP}"
echo "[test-replace] Save interval:   ${SAVE_INTERVAL}"
echo "[test-replace] Train iters:     ${TRAIN_ITERS}"
echo "[test-replace] Dist timeout:    ${DISTRIBUTED_TIMEOUT_MINUTES}min (60s after init)"
echo "[test-replace] Phase timeout:   ${ELASTIC_PHASE_TIMEOUT_SECONDS}s"
echo "[test-replace] Group barrier:   ${ELASTIC_MPU_GROUP_ORDINAL_BARRIER} (${ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS}s)"
echo "[test-replace] PG device_id:    init=${ELASTIC_INIT_PG_DEVICE_ID}, rebuild=${ELASTIC_REBUILD_INIT_PG_DEVICE_ID}"
echo "[test-replace] Eager rebuild PG:${ELASTIC_REBUILD_EAGER_NCCL_GROUPS}"
echo "[test-replace] Post warmup:     ${ELASTIC_POST_REBUILD_COMM_WARMUP}"
echo "[test-replace] MoE first:       barrier=${ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER}, warmup=${ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP}"
echo "[test-replace] MoE fail-fast:   enabled=${ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST}, timeout=${ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT}s"
echo "[test-replace] Fallback:        relaunch=${ELASTIC_FALLBACK_RELAUNCH}, exit=${ELASTIC_FALLBACK_EXIT_CODE}, retries=${HOTSPARE_MAX_RETRIES} (fallback exit only)"
echo "[test-replace] Control plane:   launcher=${ELASTIC_LAUNCHER_CONTROL_PLANE}, heartbeat=${ELASTIC_LAUNCHER_HEARTBEAT_INTERVAL}s"
echo "[test-replace] Watcher startup: ${ELASTIC_WATCHER_ADDR}:${ELASTIC_WATCHER_PORT}, timeout=${ELASTIC_WATCHER_STARTUP_TIMEOUT_SECONDS}s"
echo "[test-replace] Quiescence:      timeout=${ELASTIC_QUIESCENCE_TIMEOUT_SECONDS}s"
echo "[test-replace] R2 contract:      gap=[${MOEGAMBIT_DELTA_TIME_MIN_GAP},${MOEGAMBIT_MAX_SINGLE_GAP}], window=${MOEGAMBIT_EXPOSURE_WINDOW_STEPS}, phi_max=${MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY}, experts=${MOEGAMBIT_NUM_EXPERTS}"
echo "[test-replace] CKPT_DIR:        ${CKPT_DIR}"
echo "[test-replace] =============================================="

# =============================================================================
# SPARE NODE: run watcher (with fault injection)
# =============================================================================

if [ "${IS_SPARE}" = "1" ]; then
  echo "[test-replace] Starting elastic watcher on spare node..."
  echo "[test-replace] Fault injection: kill node ${FAULT_INJECT_NODE} local_rank ${FAULT_INJECT_LOCAL_RANK} at step ${FAULT_INJECT_STEP}"
  python3 "${SCRIPT_DIR}/elastic_watcher.py" \
    --port "${ELASTIC_WATCHER_PORT}" \
    --training-nnodes "${TRAINING_NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --fault-dir "${ELASTIC_FAULT_DIR}" \
    --fallback-relaunch "${ELASTIC_FALLBACK_RELAUNCH}" \
    --fallback-exit-code "${ELASTIC_FALLBACK_EXIT_CODE}" \
    --fallback-restart-standby "${ELASTIC_FALLBACK_RESTART_STANDBY}" \
    --fault-inject-step "${FAULT_INJECT_STEP}" \
    --fault-inject-node "${FAULT_INJECT_NODE}" \
    --fault-inject-local-rank "${FAULT_INJECT_LOCAL_RANK}"
  exit $?
fi

# =============================================================================
# TRAINING NODE: custom launcher (no torchrun)
# =============================================================================

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
  # NOTE: no --moe-moegambit-fault-injection — we do real kills via watcher
  --moe-moegambit-hot-spare-pool
  --moe-moegambit-num-hot-spares "${NPROC_PER_NODE}"
  --moe-moegambit-degraded-tau-c 0.5
  --moe-moegambit-degraded-t-max 1000
  --moe-moegambit-degraded-s-max 500
)

if [ "${MOEGAMBIT_GAP_AWARE_RECOVERY:-0}" = "1" ]; then
  MOEGAMBIT_ARGS+=(
    --moe-moegambit-gap-aware-recovery
    --moe-moegambit-recovery-policy-type "${MOEGAMBIT_RECOVERY_POLICY_TYPE}"
    --moe-moegambit-gap-threshold "${MOEGAMBIT_GAP_THRESHOLD}"
    --moe-moegambit-delta-time-min-gap "${MOEGAMBIT_DELTA_TIME_MIN_GAP}"
    --moe-moegambit-max-single-gap "${MOEGAMBIT_MAX_SINGLE_GAP}"
    --moe-moegambit-exposure-window-steps "${MOEGAMBIT_EXPOSURE_WINDOW_STEPS}"
    --moe-moegambit-max-expert-staleness-density "${MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY}"
  )
fi

wait_for_watcher() {
  python3 - \
    "${ELASTIC_WATCHER_ADDR}" \
    "${ELASTIC_WATCHER_PORT}" \
    "${ELASTIC_WATCHER_STARTUP_TIMEOUT_SECONDS}" \
    "${ELASTIC_WATCHER_CONNECT_INTERVAL_SECONDS}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
interval = max(float(sys.argv[4]), 0.1)
deadline = time.monotonic() + timeout
last_error = None

print(f"[test-replace] waiting for watcher at {host}:{port}", flush=True)
while True:
    try:
        with socket.create_connection((host, port), timeout=min(5.0, interval)):
            pass
        print(f"[test-replace] watcher is ready at {host}:{port}", flush=True)
        raise SystemExit(0)
    except OSError as exc:
        last_error = exc

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        print(
            f"[test-replace] watcher readiness timed out after {timeout:.1f}s: "
            f"{host}:{port}: {last_error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(70)
    time.sleep(min(interval, remaining))
PY
}

cleanup_elastic_attempt_files() {
  rm -f "${ELASTIC_FAULT_DIR}/pause_signal" \
        "${ELASTIC_FAULT_DIR}/rebuild_signal.json" \
        "${ELASTIC_FAULT_DIR}/fallback_relaunch_signal.json"
  rm -f "${ELASTIC_FAULT_DIR}"/worker_pid_* 2>/dev/null || true
}

run_training() {
  if ! wait_for_watcher; then
    return 70
  fi
  cleanup_elastic_attempt_files

  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
    echo "[test-replace] Restart/relaunch attempt will load checkpoint from ${CKPT_DIR}"
  else
    echo "[test-replace] No checkpoint found; starting without --load"
  fi

  python3 "${SCRIPT_DIR}/elastic_launcher.py" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  -- \
  python3 ./Megatron-LM/pretrain_gpt.py \
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
  "${MOEGAMBIT_ARGS[@]}" \
  --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
  --split 100,0,0 \
  --ckpt-format torch \
  --save "${CKPT_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval 1000 \
  --eval-iters 0 \
  --log-interval 1 \
  "${LOAD_ARGS[@]}"
}

retry=0
while true; do
  export ELASTIC_LAUNCH_ATTEMPT="${retry}"
  echo "[test-replace] launch attempt=${ELASTIC_LAUNCH_ATTEMPT}"
  run_training
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "[test-replace] training finished normally"
    exit 0
  fi

  retry=$((retry + 1))
  echo "[test-replace] training launcher exited rc=${rc}; retry=${retry}/${HOTSPARE_MAX_RETRIES}"
  if [ "${rc}" -ne "${ELASTIC_FALLBACK_EXIT_CODE}" ]; then
    echo "[test-replace] rc=${rc} is not fallback exit code ${ELASTIC_FALLBACK_EXIT_CODE}; not relaunching"
    exit "${rc}"
  fi

  if [ "${retry}" -gt "${HOTSPARE_MAX_RETRIES}" ]; then
    echo "[test-replace] reached max relaunch retries; exiting rc=${rc}"
    exit "${rc}"
  fi

  if [ ! -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] \
      && ! ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    echo "[test-replace] fallback requested but no checkpoint exists; refusing fresh restart"
    exit "${rc}"
  fi

  echo "[test-replace] relaunching from checkpoint after ${HOTSPARE_RETRY_DELAY}s..."
  sleep "${HOTSPARE_RETRY_DELAY}"
done
