#!/usr/bin/env bash
# ============================================================
# Run lm-evaluation-harness against a MoEGuard checkpoint on 8 downstream
# tasks, fully offline on an air-gapped GPU host.
#
# Smart dispatch:
#   * MODEL_PATH contains config.json     -> treat as HF, eval directly
#   * MODEL_PATH contains iter_* /        -> auto-convert Megatron -> HF
#     latest_checkpointed_iteration.txt      into HF_OUT_DIR, then eval
#
# Usage:
#   # case A: Megatron ckpt straight from training
#   MODEL_PATH=/mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard/ckpt \
#   bash run_eval.sh
#
#   # case B: pre-converted HF checkpoint
#   MODEL_PATH=/mnt/ais-c1/dataset/zds/eval/models/moeguard-hf \
#   bash run_eval.sh
#
# All knobs are env-var driven so this script does NOT need editing
# between checkpoints.
# ============================================================

set -uo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================
# Force-offline HuggingFace stack (set EARLY, before any python import).
# The previous placement (right before lm_eval invocation) could be too
# late if lm_eval performs lazy HuggingFace lookups during import that
# block silently on socket recv (observed: process stays in sleeping
# state, 0% CPU, 0% GPU, holding on huggingface_hub network call).
# ============================================================
EVAL_DATA_ROOT_EARLY="${EVAL_DATA_ROOT:-/mnt/ais-c1/dataset/zds/evaldata}"
export HF_HOME="${HF_HOME:-${EVAL_DATA_ROOT_EARLY}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EVAL_DATA_ROOT_EARLY}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${EVAL_DATA_ROOT_EARLY}/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${EVAL_DATA_ROOT_EARLY}/hf_cache/transformers}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Avoid tokenizers forking deadlock spam.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Belt-and-suspenders: kill any inherited proxy so even if some lib
# tries to phone home it fails fast instead of hanging on connect.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"

# ---- Paths (match run_main_exp_moeguard.sh /mnt/ais-c1 convention) ----
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/mnt/ais-c1/dataset/zds/evaldata}"
MODEL_PATH="${MODEL_PATH:?must set MODEL_PATH (Megatron ckpt dir OR HF model dir)}"
HF_MODELS_ROOT="${HF_MODELS_ROOT:-/mnt/ais-c1/dataset/zds/eval/models}"

# ============================================================
# Step 0: detect format + auto-convert if needed
# ============================================================
is_hf_dir() {
    [ -f "$1/config.json" ] && {
        ls "$1"/*.safetensors >/dev/null 2>&1 || \
        ls "$1"/pytorch_model*.bin >/dev/null 2>&1 || \
        ls "$1"/model*.safetensors* >/dev/null 2>&1
    }
}

is_megatron_dir() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] || ls -d "$1"/iter_* >/dev/null 2>&1
}

HF_MODEL_PATH=""
if is_hf_dir "${MODEL_PATH}"; then
    echo "[eval] detected HF checkpoint at ${MODEL_PATH}"
    HF_MODEL_PATH="${MODEL_PATH}"
elif is_megatron_dir "${MODEL_PATH}"; then
    # Derive a deterministic HF output dir so a re-run reuses the conversion.
    iter="$(cat "${MODEL_PATH}/latest_checkpointed_iteration.txt" 2>/dev/null || echo unknown)"
    src_tag="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}")_iter${iter}_hf"
    HF_MODEL_PATH="${HF_MODEL_PATH_OVERRIDE:-${HF_MODELS_ROOT}/${src_tag}}"

    if is_hf_dir "${HF_MODEL_PATH}" && [ "${FORCE_RECONVERT:-0}" != "1" ]; then
        echo "[eval] reusing existing HF conversion at ${HF_MODEL_PATH}"
        echo "       (set FORCE_RECONVERT=1 to redo)"
    else
        echo "[eval] detected Megatron checkpoint at ${MODEL_PATH} (iter=${iter})"
        echo "[eval] converting -> ${HF_MODEL_PATH}"
        CKPT_IN="${MODEL_PATH}" \
        CKPT_OUT="${HF_MODEL_PATH}" \
        bash "${SCRIPT_DIR}/convert_megatron_to_hf.sh"
        rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "[eval] convert_megatron_to_hf.sh failed (rc=${rc})" >&2
            exit "${rc}"
        fi
        if ! is_hf_dir "${HF_MODEL_PATH}"; then
            echo "[eval] post-convert sanity check failed: ${HF_MODEL_PATH} is not a valid HF dir" >&2
            ls -la "${HF_MODEL_PATH}" >&2 || true
            exit 3
        fi
    fi
else
    echo "[eval] ERROR: ${MODEL_PATH} is neither an HF model dir nor a Megatron ckpt dir." >&2
    echo "             expected one of:" >&2
    echo "               - config.json + *.safetensors / pytorch_model*.bin   (HF)" >&2
    echo "               - iter_*/ or latest_checkpointed_iteration.txt        (Megatron)" >&2
    exit 2
fi

# ---- Output dir (named after the *HF* model so re-runs accumulate) ----
# Logs/results go to /personal (typically a faster/dedicated user home),
# while the converted model and dataset cache stay on /mnt/ais-c1.
RESULTS_ROOT="${RESULTS_ROOT:-/personal/eval_results}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/$(basename "${HF_MODEL_PATH}")_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

# ============================================================
# Step 1: offline HuggingFace stack (already exported at top of script;
# kept here as a no-op marker for readers).
# ============================================================
: "${HF_HUB_OFFLINE:?should have been set at top of script}"

# ---- Eval hyperparameters ----
TASKS="${TASKS:-boolq,winogrande,race,mathqa,swag,piqa,arc_easy,openbookqa}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

# Backend & parallelism. Default to vLLM with TP=8 because we trained on
# 8 H20-3e per node and a 30B-A3B MoE only fits on one card if the KV
# cache is starved; spreading it over 8 cards keeps batch_size=auto sane.
# Override with MODEL_BACKEND=hf to use the slower transformers backend.
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

if [ "${MODEL_BACKEND}" = "vllm" ]; then
    MODEL_ARGS="pretrained=${HF_MODEL_PATH},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},trust_remote_code=True,enforce_eager=False"
    # vLLM's multiproc executor forks 1 worker per TP rank. lm-eval touches
    # CUDA in the parent during model registration, so the default fork
    # start method dies with
    #   RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    # 'spawn' avoids inheriting the parent's CUDA context.
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
elif [ "${MODEL_BACKEND}" = "hf" ]; then
    # Spread the model across all visible GPUs using transformers' native
    # device_map=auto. We deliberately do NOT use accelerate's
    # parallelize=True by default: combined with k8s PyTorchJob env vars
    # it makes Accelerator() call init_process_group() and hang forever
    # on TCPStore rendezvous (other worker pods are not running lm_eval).
    # device_map=auto stays single-process and just shards layers.
    HF_PARALLEL_MODE="${HF_PARALLEL_MODE:-device_map}"   # device_map | parallelize | single
    case "${HF_PARALLEL_MODE}" in
        device_map)
            MODEL_ARGS="pretrained=${HF_MODEL_PATH},dtype=${DTYPE},trust_remote_code=True,device_map=auto"
            ;;
        parallelize)
            MODEL_ARGS="pretrained=${HF_MODEL_PATH},dtype=${DTYPE},trust_remote_code=True,parallelize=True"
            ;;
        single)
            MODEL_ARGS="pretrained=${HF_MODEL_PATH},dtype=${DTYPE},trust_remote_code=True"
            ;;
        *)
            echo "[eval] ERROR: unknown HF_PARALLEL_MODE=${HF_PARALLEL_MODE}" >&2
            exit 2
            ;;
    esac
else
    MODEL_ARGS="pretrained=${HF_MODEL_PATH},dtype=${DTYPE},trust_remote_code=True"
fi

LM_EVAL_BIN="$(command -v lm_eval || true)"
if [ -z "${LM_EVAL_BIN}" ]; then
    LM_EVAL_BIN="python -m lm_eval"
fi

echo "============================================================"
echo "[eval] MODEL_PATH    = ${MODEL_PATH}   (input)"
echo "[eval] HF_MODEL_PATH = ${HF_MODEL_PATH} (used by lm-eval)"
echo "[eval] MODEL_BACKEND = ${MODEL_BACKEND}"
echo "[eval] TASKS         = ${TASKS}"
echo "[eval] NUM_FEWSHOT   = ${NUM_FEWSHOT}"
echo "[eval] DATA_ROOT     = ${EVAL_DATA_ROOT}"
echo "[eval] RESULTS_DIR   = ${RESULTS_DIR}"
echo "============================================================"

# Sanity-check that the offline cache actually has the data.
if [ ! -d "${EVAL_DATA_ROOT}/hf_cache" ]; then
    echo "[eval] ERROR: ${EVAL_DATA_ROOT}/hf_cache not found." >&2
    echo "        Did you run download_datasets.py on a networked host" >&2
    echo "        and ship ${EVAL_DATA_ROOT} to this machine?" >&2
    exit 2
fi
if [ ! -d "${HF_HOME}" ]; then
    echo "[eval] WARN: HF_HOME=${HF_HOME} does not exist, creating it" >&2
    mkdir -p "${HF_HOME}"
fi

# Dump the offline env we are about to inherit into lm_eval, so that
# the next 'silent hang' is easy to diagnose from eval.log alone.
echo "[eval] offline env:"
env | grep -E '^(HF_|TRANSFORMERS_|TOKENIZERS_|NO_PROXY|no_proxy|HTTP_|HTTPS_|http_|https_)' | sort

# ============================================================
# Step 2: run lm-eval
# ============================================================
# vLLM owns its own device placement (tensor_parallel_size handles it),
# so --device / --max_batch_size are not passed. For the hf backend the
# original flags still apply.
LM_EVAL_ARGS=(
    --model "${MODEL_BACKEND}"
    --model_args "${MODEL_ARGS}"
    --tasks "${TASKS}"
    --num_fewshot "${NUM_FEWSHOT}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${RESULTS_DIR}"
    --log_samples
)
if [ "${MODEL_BACKEND}" != "vllm" ]; then
    LM_EVAL_ARGS+=(--max_batch_size "${MAX_BATCH_SIZE}" --device "${DEVICE}")
fi

${LM_EVAL_BIN} "${LM_EVAL_ARGS[@]}" 2>&1 | tee "${RESULTS_DIR}/eval.log"

rc=${PIPESTATUS[0]}
if [ "${rc}" -ne 0 ]; then
    echo "[eval] lm_eval exited with code ${rc}" >&2
    exit "${rc}"
fi

echo "============================================================"
echo "[eval] done. results at ${RESULTS_DIR}"
echo "[eval] summary:"
ls -1 "${RESULTS_DIR}"
echo "============================================================"
