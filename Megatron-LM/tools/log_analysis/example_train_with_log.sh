#!/usr/bin/env bash
set -euo pipefail
set -x

export PYTHONPATH=$PYTHONPATH:/ossfs/workspace/Megatron-LM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST="9.0"

# ============================================================
# 日志保存设置 (新增 3 行)
# ============================================================
LOG_DIR="${TRAIN_LOG_DIR:-/mnt/ais-c1/dataset/zds/train_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: ${LOG_FILE}"

CKPT_DIR="/mnt/ais-c1/dataset/zds/4.4.3"
mkdir -p "${CKPT_DIR}"

MAX_RETRIES=100
RETRY_DELAY=30
retry=0

while true; do
  LOAD_ARGS=()
  if [ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ] || ls "${CKPT_DIR}"/iter_* >/dev/null 2>&1; then
    LOAD_ARGS=(--load "${CKPT_DIR}")
  fi

  # ============================================================
  # 唯一改动: 末尾加 2>&1 | tee -a "${LOG_FILE}"
  #   2>&1     — stderr 合并到 stdout
  #   tee -a   — 追加写入日志文件，同时终端实时显示
  # ============================================================
  torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=6000 \
    /ossfs/workspace/Megatron-LM/pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --legacy-tokenizer \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file "/ossfs/workspace/tokenizer/vocab.json" \
    --merge-file "/ossfs/workspace/tokenizer/merges.txt" \
    --num-layers 50 \
    --hidden-size 768 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 6 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
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
    --swiglu \
    --untie-embeddings-and-output-weights \
    --bf16 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --moe-token-dispatcher-type alltoall \
    --data-path "/ossfs/workspace/datasets/my_gpt_data_text_document" \
    --split 100,0,0 \
    --save "${CKPT_DIR}" \
    --save-interval 200 \
    --eval-interval 100 \
    --eval-iters 0 \
    --log-interval 1 \
    --timing-log-level 2 \
    "${LOAD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then
    echo "training finished normally" | tee -a "${LOG_FILE}"
    break
  fi

  retry=$((retry + 1))
  echo "training failed with exit code ${rc}, retry=${retry}/${MAX_RETRIES}" | tee -a "${LOG_FILE}"

  if [ $retry -ge $MAX_RETRIES ]; then
    echo "reach max retries, exit" | tee -a "${LOG_FILE}"
    exit $rc
  fi

  sleep "${RETRY_DELAY}"
done

# ============================================================
# 训练结束后自动分析日志 (可选，取消注释即可启用)
# ============================================================
# python /ossfs/workspace/Megatron-LM/tools/log_analysis/analyze_train_log.py "${LOG_FILE}"
