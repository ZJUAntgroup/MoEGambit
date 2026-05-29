#!/usr/bin/env bash
# ============================================================
# Convert a Megatron-LM MoE checkpoint (--ckpt-format torch, PP×EP shards
# under iter_XXXXXXX/mp_rank_00_{pp:03d}_{ep:03d}/model_optim_rng.pt) to
# a self-contained HuggingFace Qwen3MoeForCausalLM directory that can be
# loaded by lm-evaluation-harness / vLLM / transformers.
#
# This is a thin wrapper around our in-tree converter:
#   eval/scripts/megatron_moe_to_hf.py
#
# Megatron-LM's bundled tools/checkpoint/convert.py CANNOT do this — it
# ships a `--saver llama_mistral` that does not exist in this checkout
# and its dense savers do not understand MoE-grouped expert weights,
# fused QKV with GQA, or per-(PP,EP) shards. We use our own converter
# instead.
# ============================================================

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Inputs (override on the command line / via env) ----
CKPT_IN="${CKPT_IN:?must set CKPT_IN to the Megatron ckpt dir (the one containing iter_*/)}"
CKPT_OUT="${CKPT_OUT:?must set CKPT_OUT to the HF output dir}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${REPO_ROOT}/tokenizer}"

# Optional explicit iteration; default = latest_checkpointed_iteration.txt
CKPT_ITER="${CKPT_ITER:-}"
SHARD_SIZE_GB="${SHARD_SIZE_GB:-5}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "${CKPT_OUT}"

ITER_ARGS=()
if [ -n "${CKPT_ITER}" ]; then
    ITER_ARGS=(--iter "${CKPT_ITER}")
fi

python "${SCRIPT_DIR}/megatron_moe_to_hf.py" \
    --ckpt "${CKPT_IN}" \
    --out  "${CKPT_OUT}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --shard-size-gb "${SHARD_SIZE_GB}" \
    --dtype "${DTYPE}" \
    "${ITER_ARGS[@]}"

echo "============================================================"
echo "[convert] HF checkpoint written to ${CKPT_OUT}"
echo "[convert] now pass it to run_eval.sh:"
echo "  MODEL_PATH=${CKPT_OUT} bash $(dirname "${BASH_SOURCE[0]}")/run_eval.sh"
echo "============================================================"
