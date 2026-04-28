#!/usr/bin/env bash
# ============================================================
# save_train_log.sh — 保存 Megatron 训练日志
#
# 用法:
#   bash save_train_log.sh <训练启动命令>
#
# 示例:
#   bash save_train_log.sh python pretrain_gpt.py --num-layers 24 ...
#   bash save_train_log.sh torchrun --nproc_per_node=8 pretrain_gpt.py ...
#
# 日志保存到: ./train_logs/train_<YYYYMMDD_HHMMSS>.log
# stdout 和 stderr 合并保存，同时在终端实时显示。
# ============================================================

set -euo pipefail

# --- 日志目录 ---
LOG_DIR="${TRAIN_LOG_DIR:-./train_logs}"
mkdir -p "$LOG_DIR"

# --- 日志文件名 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "============================================================"
echo " Megatron 训练日志保存"
echo " 日志文件: ${LOG_FILE}"
echo " 启动命令: $*"
echo " 开始时间: $(date)"
echo "============================================================"

# --- 执行训练，stdout+stderr 同时写文件和终端 ---
# 使用 unbuffered 模式确保日志实时写入
"$@" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo " 训练结束"
echo " 退出码:   ${EXIT_CODE}"
echo " 结束时间: $(date)"
echo " 日志文件: ${LOG_FILE}"
echo "============================================================"

exit ${EXIT_CODE}
