#!/usr/bin/env bash
# ============================================================
# Convert a Megatron-LM MoE checkpoint to HuggingFace format
# so it can be fed to lm-evaluation-harness (run_eval.sh).
#
# This is a thin wrapper around Megatron-LM's bundled converter:
#   tools/checkpoint/convert.py  (--model-type GPT)
# Run this on the training cluster where Megatron-LM lives.
# ============================================================

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Inputs (override on the command line / via env) ----
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
CKPT_IN="${CKPT_IN:?must set CKPT_IN to the Megatron ckpt dir (the one containing iter_*/)}"
CKPT_OUT="${CKPT_OUT:?must set CKPT_OUT to the HF output dir}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${REPO_ROOT}/tokenizer}"

# Match the training topology used in run_main_exp_moeguard.sh:
#   PP=8, EP=8, TP=1, 128 experts, MoE hidden = 768, ffn-hidden = 6144
TARGET_TP="${TARGET_TP:-1}"
TARGET_PP="${TARGET_PP:-1}"

mkdir -p "${CKPT_OUT}"

python "${MEGATRON_ROOT}/tools/checkpoint/convert.py" \
    --model-type GPT \
    --loader mcore \
    --saver llama_mistral \
    --load-dir "${CKPT_IN}" \
    --save-dir "${CKPT_OUT}" \
    --target-tensor-parallel-size "${TARGET_TP}" \
    --target-pipeline-parallel-size "${TARGET_PP}" \
    --hf-tokenizer-path "${TOKENIZER_DIR}" \
    --megatron-path "${MEGATRON_ROOT}"

# Megatron's saver dumps weights + config; copy tokenizer files so HF can
# load the directory standalone.
cp -v "${TOKENIZER_DIR}"/tokenizer* "${CKPT_OUT}/" 2>/dev/null || true
cp -v "${TOKENIZER_DIR}"/vocab.json "${TOKENIZER_DIR}"/merges.txt "${CKPT_OUT}/" 2>/dev/null || true

echo "============================================================"
echo "[convert] HF checkpoint written to ${CKPT_OUT}"
echo "[convert] now pass it to run_eval.sh:"
echo "  MODEL_PATH=${CKPT_OUT} bash $(dirname "${BASH_SOURCE[0]}")/run_eval.sh"
echo "============================================================"
