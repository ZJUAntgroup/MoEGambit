#!/usr/bin/env bash
set -uo pipefail
set -x

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

# ============================================================
# Multi-rank restart-in-place fault sweep
# ============================================================
# Experiments:
#   fault burst starts at iteration: 250, 300, 350, 399
#   failed cards per burst:          8, 16, 32
#
# The rank plan is balanced by PP stage.  With the default 64-card setup
# and PP=8, each PP stage owns 8 ranks:
#   stage 0:  0..7
#   stage 1:  8..15
#   ...
#   stage 7: 56..63
#
# For 8/16/32-card bursts we pick 1/2/4 ranks from each PP stage.  This
# guarantees every PP stage keeps healthy peers for dense-param sync.
#
# Multi-rank cases follow find_max.sh's restart_in_place path: replacement_rank
# stays equal to failed_rank, tensors are zeroed to simulate lost local state,
# and all ranks listed for the same scheduled step are injected as one burst.
# Recovery must reload stale expert weights from an older checkpoint before
# normal training resumes.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/find_multi_fault/5.21}"
export SCRIPT_DIR BASE_DIR

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-8}"
# Do not use the generic WORLD_SIZE environment variable here: torchrun and
# some cluster launchers set it for their own process context, which can
# shadow the intended 64-rank experiment topology.  PLAN_WORLD_SIZE is only
# for building the synthetic fault plan.
PLAN_WORLD_SIZE="${PLAN_WORLD_SIZE:-$((NPROC_PER_NODE * NNODES))}"
PP_SIZE="${PP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
export NPROC_PER_NODE NNODES PLAN_WORLD_SIZE PP_SIZE EP_SIZE

FAULT_STEPS=(${FAULT_STEPS:-250 300 350 399})
FAULT_COUNTS=(${FAULT_COUNTS:-8 16 32})
PLAN_SEED="${PLAN_SEED:-42}"

build_fault_plan() {
  local fault_step="${1:?fault step required}"
  local fault_count="${2:?fault count required}"

  python3 - "${fault_step}" "${fault_count}" "${PLAN_WORLD_SIZE}" "${PP_SIZE}" "${PLAN_SEED}" <<'PY'
import random
import sys

fault_step = int(sys.argv[1])
fault_count = int(sys.argv[2])
world_size = int(sys.argv[3])
pp_size = int(sys.argv[4])
seed = int(sys.argv[5])

if pp_size <= 0 or world_size <= 0 or world_size % pp_size != 0:
    raise SystemExit(f"invalid world_size={world_size}, pp_size={pp_size}")

stage_width = world_size // pp_size
if fault_count <= 0:
    raise SystemExit(f"invalid fault_count={fault_count}")
if fault_count >= world_size:
    raise SystemExit("fault_count must leave at least one healthy rank")

base = fault_count // pp_size
remainder = fault_count % pp_size
quotas = [base] * pp_size

rng = random.Random(seed + fault_step * 1009 + fault_count * 9176)
stage_order = list(range(pp_size))
rng.shuffle(stage_order)
for stage in stage_order[:remainder]:
    quotas[stage] += 1

if any(quota >= stage_width for quota in quotas):
    raise SystemExit(
        f"unsafe fault plan: a PP stage would be fully failed "
        f"(stage_width={stage_width}, quotas={quotas})"
    )

ranks = []
for stage, quota in enumerate(quotas):
    candidates = list(range(stage * stage_width, (stage + 1) * stage_width))
    ranks.extend(sorted(rng.sample(candidates, quota)))

print(f"{fault_step}:{','.join(str(rank) for rank in sorted(ranks))}")
PY
}

run_training() {
  local FAULT_STEP="${CURRENT_FAULT_STEP:?CURRENT_FAULT_STEP required}"
  local FAULT_COUNT="${CURRENT_FAULT_COUNT:?CURRENT_FAULT_COUNT required}"
  local FAULT_PLAN="${CURRENT_FAULT_PLAN:?CURRENT_FAULT_PLAN required}"

  local RUN_DIR="${BASE_DIR}/step_${FAULT_STEP}_faults_${FAULT_COUNT}"
  local CKPT_DIR="${RUN_DIR}/ckpt"
  mkdir -p "${CKPT_DIR}"

  export TRAIN_LOG_DIR="${RUN_DIR}/log"
  export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
  export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
  export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

  # Match find_max.sh: restart_in_place avoids PP=8 group rebuild deadlocks
  # while still corrupting local model/optimizer state before recovery.
  export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
  export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:-0}"
  export BSR_FAULT_INJECT_STEP="${FAULT_STEP}"
  export BSR_FAULT_INJECT_INTERVAL="0"
  export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
  export BSR_FAULT_INJECT_PLAN="${FAULT_PLAN}"
  export BSR_FAULT_INJECT_PLAN_MODE="${BSR_FAULT_INJECT_PLAN_MODE:-burst_all}"
  export BSR_FAULT_REPLACEMENT_STEP="${FAULT_STEP}"
  export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
  export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
  export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"
  export BSR_REQUIRE_OLD_PARAM_RESTORE="${BSR_REQUIRE_OLD_PARAM_RESTORE:-1}"

  echo "============================================================"
  echo "[find_multi_fault] Starting run: fault_step=${FAULT_STEP}, fault_count=${FAULT_COUNT}"
  echo "[find_multi_fault] fault_plan=${FAULT_PLAN}"
  echo "[find_multi_fault] fault_type=${BSR_FAULT_INJECT_TYPE}"
  echo "[find_multi_fault] fault_plan_mode=${BSR_FAULT_INJECT_PLAN_MODE}"
  echo "[find_multi_fault] zero_memory=${BSR_FAULT_ZERO_MEMORY}, memory_fill=${BSR_FAULT_MEMORY_FILL}"
  echo "[find_multi_fault] require_old_param_restore=${BSR_REQUIRE_OLD_PARAM_RESTORE}"
  echo "[find_multi_fault] topology: nproc_per_node=${NPROC_PER_NODE}, nnodes=${NNODES}, pp_size=${PP_SIZE}, ep_size=${EP_SIZE}, plan_world_size=${PLAN_WORLD_SIZE}"
  echo "[find_multi_fault] CKPT_DIR=${CKPT_DIR}"
  echo "[find_multi_fault] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
  echo "============================================================"

  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-20115}" \
    ./Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
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
    --global-batch-size 64 \
    --train-iters 600 \
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
    --moe-bsr-enable \
    --moe-bsr-health-mask \
    --moe-bsr-rank-quarantine \
    --moe-bsr-dispatch-quarantine-assert \
    --moe-bsr-dispatch-sanitize \
    --moe-bsr-expert-directory \
    --moe-bsr-replacement-protocol \
    --moe-bsr-group-rebuild \
    --moe-bsr-dispatch-topology-refresh \
    --moe-bsr-dense-param-sync \
    --moe-bsr-stale-expert-restore \
    --moe-bsr-recovery-controller \
    --moe-bsr-deferred-optimizer-load \
    --moe-bsr-degraded-mode-policy \
    --moe-bsr-reintegration-barrier \
    --moe-bsr-fault-injection \
    --moe-bsr-restart-in-place \
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 1000 \
    --moe-bsr-degraded-s-max 500 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 200 \
    --eval-interval 100 \
    --eval-iters 50 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

MAX_RETRIES=1
RETRY_DELAY="${RETRY_DELAY:-30}"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"

for STEP in "${FAULT_STEPS[@]}"; do
  for COUNT in "${FAULT_COUNTS[@]}"; do
    if ! PLAN="$(build_fault_plan "${STEP}" "${COUNT}")"; then
      echo "[find_multi_fault] failed to build fault plan: step=${STEP}, count=${COUNT}, plan_world_size=${PLAN_WORLD_SIZE}, pp_size=${PP_SIZE}" >&2
      continue
    fi
    if [ -z "${PLAN}" ]; then
      echo "[find_multi_fault] failed to build fault plan: step=${STEP}, count=${COUNT}, plan_world_size=${PLAN_WORLD_SIZE}, pp_size=${PP_SIZE}" >&2
      continue
    fi
    RUN_DIR="${BASE_DIR}/step_${STEP}_faults_${COUNT}"

    export CURRENT_FAULT_STEP="${STEP}"
    export CURRENT_FAULT_COUNT="${COUNT}"
    export CURRENT_FAULT_PLAN="${PLAN}"
    export TRAIN_LOG_DIR="${RUN_DIR}/log"
    export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
    export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
    export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

    echo ""
    echo "############################################################"
    echo "# find_multi_fault: fault_step=${STEP}, fault_count=${COUNT}"
    echo "# plan=${PLAN}"
    echo "############################################################"

    retry=0
    while true; do
      if [ -f "${SAVE_LOG_SCRIPT}" ]; then
        bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
        rc=$?
      else
        echo "[find_multi_fault] save_train_log.sh not found, running without log analysis"
        run_training
        rc=$?
      fi

      if [ $rc -eq 0 ]; then
        echo "[find_multi_fault] step=${STEP}, count=${COUNT} finished normally"
        break
      fi

      retry=$((retry + 1))
      echo "[find_multi_fault] step=${STEP}, count=${COUNT} failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

      if [ $retry -ge $MAX_RETRIES ]; then
        echo "[find_multi_fault] step=${STEP}, count=${COUNT} reached max retries, aborting this run"
        break
      fi

      sleep "${RETRY_DELAY}"
    done

    echo "[find_multi_fault] step=${STEP}, count=${COUNT} done (or aborted), moving to next experiment"
  done
done

echo ""
echo "============================================================"
echo "find_multi_fault: all experiments completed"
echo "Fault starts tested: ${FAULT_STEPS[*]}"
echo "Fault counts tested: ${FAULT_COUNTS[*]}"
echo "Results in: ${BASE_DIR}/step_*_faults_*/log/"
echo "============================================================"
