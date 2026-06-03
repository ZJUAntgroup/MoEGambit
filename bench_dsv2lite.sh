#!/usr/bin/env bash
# =============================================================================
# bench_dsv2lite.sh
#
# Drives the same fault-injection micro-benchmark as bench_moe128.sh but on a
# DeepSeek-V2-Lite-style configuration to exercise MoEGambit's RQ8
# cross-architecture generalisation experiment:
#
#   - 32 layers (DeepSeek-V2-Lite-style); first layer is dense FFN, layers 2..32 are MoE
#   - 64 routed experts, 2 shared experts, top-6 routing
#   - expert FFN hidden = 1408 (vs. 768 in bench_moe128)
#   - attention is kept as standard GQA (32 heads / 4 KV groups, head dim 128)
#     instead of MLA, so MoE-structure is the only architectural variable wrt.
#     bench_moe128.sh.
#
# Cluster size is selectable via DSV2_GPUS in {16, 32, 64, 128} so the same
# script feeds the §RQ7 scaling slice of RQ8.
#
# The script does NOT modify run_moe128.sh.  It instead rewrites a temporary
# copy of run_moe128.sh in-place (sed) to override the hardcoded model and MoE
# arguments, then drives it via the same phase loop as bench_moe128.sh.
#
# Overrides (all optional):
#   DSV2_GPUS                  default 64        (16 / 32 / 64 / 128)
#   BENCH_PHASES               default "moegambit baseline"
#   BENCH_NUM_INJECTIONS       default 10
#   BENCH_INJECT_INTERVAL      default 40
#   BENCH_FIRST_INJECT_STEP    default 70
#   BENCH_TAIL_STEPS           default 30
#   BENCH_BASE_DIR             default /mnt/ais-c1/dataset/zds
#   BENCH_TAG                  default dsv2lite_${DSV2_GPUS}gpu_bench
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT  (passed through)
# =============================================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_INNER="${SCRIPT_DIR}/run_moe128.sh"
if [ ! -f "${ORIGINAL_INNER}" ]; then
  echo "[bench_dsv2lite] missing template runner ${ORIGINAL_INNER}"
  exit 2
fi

# -----------------------------------------------------------------------------
# Cluster-size profile
# -----------------------------------------------------------------------------
DSV2_GPUS="${DSV2_GPUS:-64}"
case "${DSV2_GPUS}" in
  16)  PHASE_NNODES=2;  PHASE_TP_SIZE=1; PHASE_PP_SIZE=4; PHASE_EP_SIZE=4;  PHASE_GBS=16  ;;
  32)  PHASE_NNODES=4;  PHASE_TP_SIZE=1; PHASE_PP_SIZE=4; PHASE_EP_SIZE=8;  PHASE_GBS=32  ;;
  64)  PHASE_NNODES=8;  PHASE_TP_SIZE=1; PHASE_PP_SIZE=8; PHASE_EP_SIZE=8;  PHASE_GBS=64  ;;
  128) PHASE_NNODES=16; PHASE_TP_SIZE=1; PHASE_PP_SIZE=8; PHASE_EP_SIZE=8;  PHASE_GBS=128 ;;
  *)
    echo "[bench_dsv2lite] DSV2_GPUS must be 16 / 32 / 64 / 128, got ${DSV2_GPUS}"
    exit 2
    ;;
esac

# Sanity: TP * PP * EP * DP = N_GPU; here DP is implicit and equals 1 for the
# 16/32/64/128 layouts above (PP carries the data-parallel dimension via the
# pipeline-stage count, exactly as in bench_moe16/32/128.sh).
PHASE_DP=$(( DSV2_GPUS / (PHASE_TP_SIZE * PHASE_PP_SIZE * PHASE_EP_SIZE) ))

# -----------------------------------------------------------------------------
# Bench knobs
# -----------------------------------------------------------------------------
BENCH_PHASES="${BENCH_PHASES:-moegambit baseline}"
BENCH_NUM_INJECTIONS="${BENCH_NUM_INJECTIONS:-10}"
BENCH_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL:-40}"
BENCH_FIRST_INJECT_STEP="${BENCH_FIRST_INJECT_STEP:-70}"
BENCH_TAIL_STEPS="${BENCH_TAIL_STEPS:-30}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-/mnt/ais-c1/dataset/zds}"
BENCH_TAG="${BENCH_TAG:-dsv2lite_${DSV2_GPUS}gpu_bench}"

PHASE_TOTAL_STEPS=$(( BENCH_FIRST_INJECT_STEP \
                     + (BENCH_NUM_INJECTIONS - 1) * BENCH_INJECT_INTERVAL \
                     + BENCH_TAIL_STEPS ))

echo "[bench_dsv2lite] config:"
echo "  DSV2_GPUS         = ${DSV2_GPUS}  (TP=${PHASE_TP_SIZE} PP=${PHASE_PP_SIZE} EP=${PHASE_EP_SIZE} DP=${PHASE_DP})"
echo "  PHASES            = ${BENCH_PHASES}"
echo "  NUM_INJECTIONS    = ${BENCH_NUM_INJECTIONS}"
echo "  FIRST_INJECT_STEP = ${BENCH_FIRST_INJECT_STEP}"
echo "  INJECT_INTERVAL   = ${BENCH_INJECT_INTERVAL}"
echo "  TAIL_STEPS        = ${BENCH_TAIL_STEPS}"
echo "  PHASE_TOTAL_STEPS = ${PHASE_TOTAL_STEPS}"
echo "  BASE_DIR          = ${BENCH_BASE_DIR}"
echo "  TAG               = ${BENCH_TAG}"
echo "  GBS               = ${PHASE_GBS}"

# -----------------------------------------------------------------------------
# Patch the inner runner: produce a one-shot temp copy of run_moe128.sh whose
# model / MoE arguments match DeepSeek-V2-Lite.  We do this with sed instead
# of editing run_moe128.sh in place so the 128-GPU benchmark is unaffected.
# -----------------------------------------------------------------------------
INNER_SCRIPT="$(mktemp -t run_dsv2lite_XXXXXX.sh)"
trap 'rm -f "${INNER_SCRIPT}"' EXIT

# Override scalar args via line-by-line regex replacement.
sed -E \
  -e "s|--num-layers[[:space:]]+[0-9]+|--num-layers 32|" \
  -e "s|--hidden-size[[:space:]]+[0-9]+|--hidden-size 2048|" \
  -e "s|--ffn-hidden-size[[:space:]]+[0-9]+|--ffn-hidden-size 10944|" \
  -e "s|--num-attention-heads[[:space:]]+[0-9]+|--num-attention-heads 32|" \
  -e "s|--num-experts[[:space:]]+[0-9]+|--num-experts 64|" \
  -e "s|--moe-ffn-hidden-size[[:space:]]+[0-9]+|--moe-ffn-hidden-size 1408|" \
  -e "s|--moe-router-topk[[:space:]]+[0-9]+|--moe-router-topk 6|" \
  -e "s|--global-batch-size[[:space:]]+[0-9]+|--global-batch-size ${PHASE_GBS}|" \
  "${ORIGINAL_INNER}" > "${INNER_SCRIPT}"

# Inject the shared-expert flag (Megatron-LM standard switch).  We splice it in
# right before the BSR_ARGS expansion so that MoEGambit's wrapper still sees the
# resulting argv.  The pattern matches the line that ends the MoE arg block.
python3 - "${INNER_SCRIPT}" <<'PYEOF'
import re, sys
p = sys.argv[1]
src = open(p).read()
inject = (
    "    --moe-shared-expert-intermediate-size 2816 \\\n"  # 2 * 1408
    "    --moe-shared-expert-overlap \\\n"
)
# Insert immediately before the line that expands BSR_ARGS so the new flags
# are visible to both Megatron and the MoEGambit wrapper.
new = re.sub(
    r"(\n)(\s*\"\$\{BSR_ARGS\[@\]\}\"[[:space:]]*\\\n)",
    lambda m: m.group(1) + inject + m.group(2),
    src,
    count=1,
)
if new == src:
    # Fallback: append before the data-path line if the BSR marker moved.
    new = re.sub(
        r"(\n)(\s*--data-path)",
        lambda m: m.group(1) + inject + m.group(2),
        src,
        count=1,
    )
if new == src:
    sys.stderr.write("[bench_dsv2lite] could not inject shared-expert flags; aborting\n")
    sys.exit(1)
open(p, "w").write(new)
PYEOF
chmod +x "${INNER_SCRIPT}"

echo "[bench_dsv2lite] patched inner runner: ${INNER_SCRIPT}"

# -----------------------------------------------------------------------------
# Phase runner (mirrors bench_moe128.sh)
# -----------------------------------------------------------------------------
run_phase() {
  local mode="$1"
  local ckpt_dir="${BENCH_BASE_DIR}/${BENCH_TAG}/${mode}"
  local log_dir="${BENCH_BASE_DIR}/log/${BENCH_TAG}/${mode}"

  echo "============================================================"
  echo "[bench_dsv2lite] starting phase mode=${mode}"
  echo "  CKPT_DIR = ${ckpt_dir}"
  echo "  LOG_DIR  = ${log_dir}"
  echo "============================================================"

  mkdir -p "${ckpt_dir}" "${log_dir}"

  if [ -n "${ckpt_dir}" ] && [ "${ckpt_dir}" != "/" ]; then
    rm -f  "${ckpt_dir}/latest_checkpointed_iteration.txt"
    rm -rf "${ckpt_dir}"/iter_*
  fi

  # Cluster + parallelism overrides honoured by run_moe128.sh.
  export NNODES="${NNODES:-${PHASE_NNODES}}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-20118}"
  export TP_SIZE="${PHASE_TP_SIZE}"
  export PP_SIZE="${PHASE_PP_SIZE}"
  export EP_SIZE="${PHASE_EP_SIZE}"
  export TRAIN_ITERS="${PHASE_TOTAL_STEPS}"
  export SAVE_INTERVAL="${BENCH_INJECT_INTERVAL}"
  export CKPT_DIR="${ckpt_dir}"
  export TRAIN_LOG_DIR="${log_dir}"

  if [ "${mode}" = "moegambit" ]; then
    export MODE=moegambit
    export BSR_FAULT_INJECT_TYPE=restart_in_place
    export BSR_FAULT_INJECT_RANK=-1
    export BSR_FAULT_INJECT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export BSR_FAULT_INJECT_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export BSR_FAULT_INJECT_SEED=42
    export BSR_FAULT_REPLACEMENT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export BSR_FAULT_REPLACEMENT_RANK=-1
    export BSR_FAULT_ZERO_MEMORY=1
    export BSR_FAULT_MEMORY_FILL=zero
    export MAX_RETRIES=1
  else
    export MODE=baseline
    export CRASH_AT_STEP="${BENCH_FIRST_INJECT_STEP}"
    export CRASH_INTERVAL="${BENCH_INJECT_INTERVAL}"
    export CRASH_RANK=-1
    export CRASH_SEED=42
    export MAX_RETRIES="${BENCH_NUM_INJECTIONS}"
    export RETRY_DELAY="${RETRY_DELAY:-15}"
  fi

  bash "${INNER_SCRIPT}"
  local rc=$?
  echo "[bench_dsv2lite] phase mode=${mode} finished with rc=${rc}"
  return ${rc}
}

overall_rc=0
for ph in ${BENCH_PHASES}; do
  case "${ph}" in
    moegambit|baseline) ;;
    *)
      echo "[bench_dsv2lite] invalid phase '${ph}', must be 'moegambit' or 'baseline'"
      exit 2
      ;;
  esac
  run_phase "${ph}"
  rc=$?
  if [ "${ph}" = "baseline" ] && [ ${rc} -ne 0 ]; then
    echo "[bench_dsv2lite] baseline phase ended with rc=${rc} after ${BENCH_NUM_INJECTIONS} planned crashes (expected); treating as success"
    rc=0
  fi
  if [ ${rc} -ne 0 ]; then
    echo "[bench_dsv2lite] phase ${ph} exited with rc=${rc}"
    overall_rc=${rc}
  fi
done

echo "[bench_dsv2lite] all phases done, overall_rc=${overall_rc}"
exit ${overall_rc}
