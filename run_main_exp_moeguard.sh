#!/usr/bin/env bash
set -uo pipefail
set -x

# ============================================================
# Main experiment: MoEGuard on the 10-fault canonical trace
# ============================================================
# Trace design (10 events over 10000 iter, one event per 1000-iter window):
#
#   #   window         step  Δ    |F|   category
#   1   [501..1000]    723   123  1     single-GPU
#   2   [1001..2000]   1188  188  1     single-GPU (HBM)
#   3   [2001..3000]   2461  61   1     single-GPU  (small Δ, Δ_min test)
#   4   [3001..4000]   3517  117  1     single-GPU (SSD/PCIe)
#   5   [4001..5000]   4309  109  1     repeat — SAME rank as #1, tests Φ' accumulation
#   6   [5001..6000]   5640  40   1     single-GPU
#   7   [6001..7000]   6855  55   1     single-GPU
#   8   [7001..8000]   7912  112  8     single-host-class burst (8-card)
#   9   [8001..9000]   8689  89   8     single-host-class burst (other PP region)
#   10  [9001..10000]  9304  104  16    rack-level outage (16-card cap)
#
# Distribution = 7 single-GPU (70%) + 2 eight-card (20%) + 1 sixteen-card (10%),
# matching production fault statistics (Llama-3 Dubey 2024 / MegaScale Jiang 2024).
#
# ----- Randomness -----
# Failed ranks are drawn by a seeded Python RNG (PLAN_SEED, default 42) so the
# trace is reproducible. The step list, Δ list, and |F| list are fixed; only
# the rank identities are randomized within the topology constraints below.
# Event #5 deliberately reuses event #1's rank to exercise Φ'(t) accumulation
# under repeated-failure on the same expert slot.
#
# ----- DP safety constraint -----
# Dense / router parameters must be peer-pulled from a healthy DP sibling.
# With PP=8, EP=8, world=64 the DP group size = 64/(PP*TP) = 8, and ranks in
# the same PP stage share one DP group (stage_width = 8). Therefore the plan
# generator enforces:
#     for every PP stage,  #failed_in_stage <= stage_width - 1
# i.e. at least one healthy rank survives in each stage so dense_param_sync
# never starves. The 8-card and 16-card bursts are spread across two PP
# stages (7 in one stage + 1 in another, or 7+7+1+1) rather than collapsing
# a whole stage; this preserves hybrid recovery feasibility while staying
# faithful to the production-burst |F| budget.
# ============================================================

# ---- Standard cluster env (identical to run_moe64.sh) ----
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Topology ----
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-8}"
PP_SIZE="${PP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
PLAN_WORLD_SIZE="${PLAN_WORLD_SIZE:-$((NPROC_PER_NODE * NNODES))}"
export NPROC_PER_NODE NNODES PP_SIZE EP_SIZE PLAN_WORLD_SIZE

# ---- Plan controls ----
PLAN_SEED="${PLAN_SEED:-42}"
export PLAN_SEED

# ---- Output dirs ----
BASE_DIR="${BASE_DIR:-/mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/ckpt}"
export TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${BASE_DIR}/log}"
mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"

# ============================================================
# Build the 10-event fault plan with seeded Python RNG
# ============================================================
# The Python helper enforces all three constraints:
#   (a) every failed rank lies in [0, PLAN_WORLD_SIZE)
#   (b) #failed_in_stage <= stage_width - 1  for every PP stage
#   (c) event #5 reuses event #1's rank exactly
# Output format matches bsr_integration._parse_fault_inject_plan:
#     "<step>:<rank[,rank...]>;<step>:<rank[,rank...]>;..."
build_main_plan() {
  python3 - "${PLAN_WORLD_SIZE}" "${PP_SIZE}" "${PLAN_SEED}" <<'PY'
import random
import sys

world_size = int(sys.argv[1])
pp_size    = int(sys.argv[2])
seed       = int(sys.argv[3])

if pp_size <= 0 or world_size <= 0 or world_size % pp_size != 0:
    raise SystemExit(f"invalid world_size={world_size}, pp_size={pp_size}")
stage_width = world_size // pp_size       # = 8 here
max_per_stage = stage_width - 1           # DP safety: leave 1 healthy rank per stage

# Fixed (step, |F|) schedule. Δ is implied by (step - latest_save_step) with
# save-interval=200; printed below for transparency.
events = [
    # (step, fault_count, category)
    ( 723,  1, "single-GPU"),
    (1188,  1, "single-GPU HBM"),
    (2461,  1, "single-GPU (small Δ, Δ_min test)"),
    (3517,  1, "single-GPU SSD/PCIe"),
    (4309,  1, "single-GPU REPEAT (same rank as #1)"),
    (5640,  1, "single-GPU"),
    (6855,  1, "single-GPU"),
    (7912,  8, "8-card burst (DP-safe spread)"),
    (8689,  8, "8-card burst (DP-safe spread, other PP region)"),
    (9304, 16, "16-card rack outage (DP-safe spread)"),
]

def sample_ranks(rng, fault_count):
    """Sample fault_count distinct ranks under the per-stage DP cap."""
    if fault_count >= world_size:
        raise SystemExit("fault_count must leave at least one healthy rank globally")
    if fault_count > pp_size * max_per_stage:
        raise SystemExit(f"fault_count {fault_count} exceeds DP-safe cap "
                         f"{pp_size}*{max_per_stage}={pp_size*max_per_stage}")
    # Allocate quotas per stage round-robin in random stage order, capped at max_per_stage
    quotas = [0] * pp_size
    stage_order = list(range(pp_size)); rng.shuffle(stage_order)
    remaining = fault_count
    # First pass: distribute as evenly as possible while respecting cap
    while remaining > 0:
        progressed = False
        for s in stage_order:
            if remaining == 0:
                break
            if quotas[s] < max_per_stage:
                quotas[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise SystemExit("could not assign DP-safe quotas")
    if any(q > max_per_stage for q in quotas):
        raise SystemExit(f"DP safety violated: quotas={quotas}")
    ranks = []
    for stage, quota in enumerate(quotas):
        if quota == 0: continue
        candidates = list(range(stage * stage_width, (stage + 1) * stage_width))
        ranks.extend(sorted(rng.sample(candidates, quota)))
    return sorted(ranks)

plan_parts = []
debug_lines = []
event_1_ranks = None
for idx, (step, fcnt, cat) in enumerate(events, start=1):
    if idx == 5:
        # Event #5: reuse event #1's exact rank set (Φ' accumulation test).
        if event_1_ranks is None:
            raise SystemExit("event #5 needs event #1 ranks but they are missing")
        ranks = list(event_1_ranks)
    else:
        # Per-event seeded RNG so that changing one event does not perturb others.
        rng = random.Random(seed + step * 1009 + fcnt * 9176 + idx * 31337)
        ranks = sample_ranks(rng, fcnt)
    if idx == 1:
        event_1_ranks = list(ranks)
    plan_parts.append(f"{step}:{','.join(str(r) for r in ranks)}")
    # Stage distribution for debug
    stage_hist = [0] * pp_size
    for r in ranks:
        stage_hist[r // stage_width] += 1
    debug_lines.append(
        f"  #{idx:>2}  step={step:>4}  |F|={fcnt:>2}  ranks={ranks}  "
        f"stage_hist={stage_hist}  cat={cat}"
    )

# Emit machine-readable plan on stdout (first line), debug on stderr.
print(";".join(plan_parts))
for ln in debug_lines:
    print(ln, file=sys.stderr)
print(f"  PLAN_SEED={seed}, world_size={world_size}, pp_size={pp_size}, "
      f"stage_width={stage_width}, max_per_stage={max_per_stage}", file=sys.stderr)
PY
}

# Generate and capture plan
echo "============================================================"
echo "[main_exp_moeguard] generating 10-fault plan (seed=${PLAN_SEED})"
echo "============================================================"
if ! MAIN_FAULT_PLAN="$(build_main_plan 2>/tmp/main_plan_debug.$$)"; then
  echo "[main_exp_moeguard] FAILED to build plan" >&2
  cat /tmp/main_plan_debug.$$ >&2 || true
  rm -f /tmp/main_plan_debug.$$
  exit 2
fi
cat /tmp/main_plan_debug.$$
rm -f /tmp/main_plan_debug.$$
echo "[main_exp_moeguard] resolved plan: ${MAIN_FAULT_PLAN}"
echo "============================================================"

# ---- BSR fault injection knobs (plan-driven, burst_all) ----
export BSR_FAULT_INJECT_TYPE="${BSR_FAULT_INJECT_TYPE:-restart_in_place}"
export BSR_FAULT_INJECT_PLAN="${MAIN_FAULT_PLAN}"
export BSR_FAULT_INJECT_PLAN_MODE="${BSR_FAULT_INJECT_PLAN_MODE:-burst_all}"
export BSR_FAULT_INJECT_STEP="0"
export BSR_FAULT_INJECT_INTERVAL="0"
export BSR_FAULT_INJECT_SEED="${BSR_FAULT_INJECT_SEED:-42}"
export BSR_FAULT_INJECT_RANK="${BSR_FAULT_INJECT_RANK:-0}"
export BSR_FAULT_REPLACEMENT_STEP="0"
export BSR_FAULT_REPLACEMENT_RANK="${BSR_FAULT_REPLACEMENT_RANK:--1}"
export BSR_FAULT_ZERO_MEMORY="${BSR_FAULT_ZERO_MEMORY:-1}"
export BSR_FAULT_MEMORY_FILL="${BSR_FAULT_MEMORY_FILL:-zero}"
export BSR_REQUIRE_OLD_PARAM_RESTORE="${BSR_REQUIRE_OLD_PARAM_RESTORE:-1}"

# ---- Log analysis hooks ----
export LOG_ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
export LOG_ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-0}"
export LOG_ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/log_analysis/analyze_train_log.py}"

run_training() {
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
    --train-iters 10000 \
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
    --moe-bsr-hybrid-expert-restore \
    --moe-bsr-expert-opt-restore \
    --moe-bsr-weights-first-recovery \
    --moe-bsr-defer-optimizer-load \
    --moe-bsr-degraded-tau-c 0.5 \
    --moe-bsr-degraded-t-max 1000 \
    --moe-bsr-degraded-s-max 500 \
    --data-path "/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document" \
    --split 99,1,0 \
    --ckpt-format torch \
    --save "${CKPT_DIR}" \
    --save-interval 200 \
    --eval-interval 1000 \
    --eval-iters 50 \
    --log-interval 1 \
    "${LOAD_ARGS[@]}"
}

# ============================================================
# Main loop with retry + log analysis
# ============================================================
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY="${RETRY_DELAY:-30}"
SAVE_LOG_SCRIPT="${SCRIPT_DIR}/log_analysis/save_train_log.sh"
retry=0

while true; do
  if [ -f "${SAVE_LOG_SCRIPT}" ]; then
    bash "${SAVE_LOG_SCRIPT}" bash -c "$(declare -f run_training); run_training"
    rc=$?
  else
    echo "[main_exp_moeguard] save_train_log.sh not found, running without log analysis"
    run_training
    rc=$?
  fi

  if [ $rc -eq 0 ]; then
    echo "[main_exp_moeguard] MoEGuard main experiment finished normally"
    break
  fi

  retry=$((retry + 1))
  echo "[main_exp_moeguard] failed (exit=${rc}), retry=${retry}/${MAX_RETRIES}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "[main_exp_moeguard] reached max retries, aborting"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done

echo "============================================================"
echo "[main_exp_moeguard] all done"
echo "Logs:    ${TRAIN_LOG_DIR}"
echo "Ckpts:   ${CKPT_DIR}"
echo "Plan:    ${MAIN_FAULT_PLAN}"
echo "Seed:    ${PLAN_SEED}"
echo "============================================================"
