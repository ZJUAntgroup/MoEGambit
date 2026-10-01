#!/usr/bin/env bash
# Native Megatron workload configuration for the independent MoC mechanism port.
set -euo pipefail
MOC_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$MOC_EXAMPLE_DIR/../.." && pwd)"
cd "$ROOT"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" NNODES="${NNODES:-8}"
export PP_SIZE="${PP_SIZE:-8}" EP_SIZE="${EP_SIZE:-8}"
export NODE_RANK="${NODE_RANK:-0}" SEED="${SEED:-1234}" NUM_EXPERTS="${NUM_EXPERTS:-128}"
export MOC_MODEL_PROFILE="${MOC_MODEL_PROFILE:-${FSE_MODEL_PROFILE:-qwen3_48x2048}}"
export DATA_PATH="${DATA_PATH:-/shared/moegambit/data/my_qwen3_data_text_document}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/tokenizer}"
export PYTHONPATH="$ROOT/src:$ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# This benchmark owns checkpoint/restart behavior; do not activate MoEGambit recovery.
export MOEGAMBIT_ENABLED=0 MOEGAMBIT_MOC_PEC_EMULATE=0
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR to the node-0 address on every node}"
else
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
for name in NNODES NPROC_PER_NODE PP_SIZE EP_SIZE NODE_RANK SEED NUM_EXPERTS MOC_START_STEP; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "invalid integer $name=${!name}" >&2; exit 2; }
done
(( NNODES * NPROC_PER_NODE == 64 && PP_SIZE == 8 && EP_SIZE == 8 && NODE_RANK < NNODES )) || {
  echo "the supplied baseline requires world=64, PP=8, EP=8" >&2; exit 2;
}
(( NUM_EXPERTS >= EP_SIZE && NUM_EXPERTS % EP_SIZE == 0 )) || {
  echo "NUM_EXPERTS must be a positive multiple of EP_SIZE" >&2; exit 2;
}
source "$MOC_EXAMPLE_DIR/model_profile.sh"

moc_require_inputs() {
  command -v torchrun >/dev/null || { echo "torchrun not found" >&2; exit 127; }
  [[ -f "${DATA_PATH}.idx" && -f "${DATA_PATH}.bin" ]] || {
    echo "indexed dataset missing: ${DATA_PATH}" >&2; exit 2;
  }
  [[ -f "$TOKENIZER_DIR/vocab.json" && -f "$TOKENIZER_DIR/merges.txt" ]] || {
    echo "tokenizer files missing: $TOKENIZER_DIR" >&2; exit 2;
  }
  local checkpoint="$BASE_DIR/ckpt/iter_$(printf '%07d' "$MOC_START_STEP")"
  [[ -d "$checkpoint" ]] || { echo "baseline checkpoint missing: $checkpoint" >&2; exit 2; }
}

moc_model_args() {
  moc_select_model_profile
  MODEL_ARGS=(
    --use-mcore-models --transformer-impl transformer_engine
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8 --expert-model-parallel-size 8
    --sequence-parallel
    --legacy-tokenizer --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_DIR" --vocab-file "$TOKENIZER_DIR/vocab.json"
    --merge-file "$TOKENIZER_DIR/merges.txt"
    --num-layers "${MOC_NUM_LAYERS}" --hidden-size "${MOC_HIDDEN_SIZE}" --ffn-hidden-size "${MOC_FFN_HIDDEN_SIZE}"
    --num-attention-heads "${MOC_NUM_ATTENTION_HEADS}"
    --kv-channels "${MOC_KV_CHANNELS}" --qk-layernorm --seq-length 4096
    --max-position-embeddings 40960 --rotary-base 1000000 --rotary-percent 1.0
    --micro-batch-size 1 --global-batch-size 64
    --train-iters 10000 --seed "${SEED}"
    --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 5
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95
    --init-method-std 0.02 --normalization RMSNorm --disable-bias-linear
    --position-embedding-type rope --no-rope-fusion --swiglu
    --no-bias-swiglu-fusion --untie-embeddings-and-output-weights --bf16
    --num-experts "${NUM_EXPERTS}" --moe-ffn-hidden-size "${MOC_MOE_FFN_HIDDEN_SIZE}" --moe-router-topk 8
    --moe-router-dtype fp32 --moe-router-score-function sigmoid
    --moe-router-load-balancing-type none --moe-aux-loss-coeff 0
    --moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall
    --ckpt-format torch
    --eval-interval 1000 --log-interval 1
  )
  if (( MOC_USE_MLA )); then
    MODEL_ARGS+=(--multi-latent-attention --kv-lora-rank "${MOC_KV_LORA_RANK}"
      --qk-head-dim "${MOC_QK_HEAD_DIM}"
      --qk-pos-emb-head-dim "${MOC_QK_POS_EMB_HEAD_DIM}"
      --v-head-dim "${MOC_V_HEAD_DIM}" --rope-type rope
      --no-masked-softmax-fusion --attention-softmax-in-fp32
      --moe-layer-freq "${MOC_MOE_LAYER_FREQ}")
  else
    MODEL_ARGS+=(--group-query-attention --num-query-groups "${MOC_NUM_QUERY_GROUPS}")
  fi
  if (( MOC_ROUTER_NUM_GROUPS > 0 )); then
    MODEL_ARGS+=(--moe-router-num-groups "${MOC_ROUTER_NUM_GROUPS}"
      --moe-router-group-topk "${MOC_ROUTER_GROUP_TOPK}")
  fi
  if (( MOC_SHARED_EXPERT_FFN_SIZE > 0 )); then
    MODEL_ARGS+=(--moe-shared-expert-intermediate-size "${MOC_SHARED_EXPERT_FFN_SIZE}")
  fi
  if [[ -n "${MOC_VALID_DATA_PATH:-}" ]]; then
    MODEL_ARGS+=(--train-data-path "${DATA_PATH}"
      --valid-data-path "${MOC_VALID_DATA_PATH}")
  else
    MODEL_ARGS+=(--data-path "${DATA_PATH}" --split 99,1,0)
  fi
}

moc_torchrun() {
  TORCHRUN_CMD=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}" ./Megatron-LM/pretrain_gpt.py
  )
}
