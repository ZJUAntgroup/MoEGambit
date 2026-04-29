#!/usr/bin/env bash
# save_train_log.sh — Megatron + BSR-MoE 训练日志保存与增量分析
#
# 功能:
#   1. 将训练 stdout+stderr 实时写入日志文件
#   2. 每隔 N 个 iteration 自动运行一次日志分析 (可配置)
#   3. 训练结束后自动运行完整分析
#
# 用法:
#   bash save_train_log.sh <训练命令...>
#
# 环境变量:
#   TRAIN_LOG_DIR          日志保存目录 (默认 ./train_logs)
#   LOG_ANALYZE_INTERVAL   每隔多少个 iteration 分析一次 (默认 0=不做增量分析)
#   LOG_ANALYZE_SCRIPT     分析脚本路径 (默认 同目录下的 analyze_train_log.py)
#   LOG_ANALYZE_ON_EXIT    训练结束后是否自动分析 (默认 1=是)
#
# 示例:
#   LOG_ANALYZE_INTERVAL=100 bash save_train_log.sh torchrun ... pretrain_gpt.py ...
#   LOG_ANALYZE_INTERVAL=50 LOG_ANALYZE_ON_EXIT=1 bash save_train_log.sh ./run_moe.sh

set -euo pipefail

# ============================================================
# 配置
# ============================================================
LOG_DIR="${TRAIN_LOG_DIR:-./train_logs}"
ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-1}"

# 分析脚本路径: 默认与本脚本同目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/analyze_train_log.py}"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ANALYSIS_DIR="${LOG_DIR}/analysis_${TIMESTAMP}"

echo "============================================================"
echo " Megatron + BSR-MoE 训练日志保存"
echo " 日志文件:     ${LOG_FILE}"
echo " 分析间隔:     ${ANALYZE_INTERVAL} iterations (0=仅结束时分析)"
echo " 结束时分析:   ${ANALYZE_ON_EXIT}"
echo " 分析脚本:     ${ANALYZE_SCRIPT}"
echo " 启动命令:     $*"
echo " 开始时间:     $(date)"
echo "============================================================"

# ============================================================
# 增量分析函数
# ============================================================
_last_analyzed_iter=0

run_incremental_analysis() {
    local current_iter="$1"
    local label="$2"  # "incremental" 或 "final"

    if [ ! -f "${ANALYZE_SCRIPT}" ]; then
        echo "[save_train_log] 分析脚本不存在: ${ANALYZE_SCRIPT}, 跳过分析"
        return
    fi

    mkdir -p "${ANALYSIS_DIR}"
    local out_file="${ANALYSIS_DIR}/analysis_${label}_iter${current_iter}.txt"
    local csv_file="${ANALYSIS_DIR}/analysis_${label}_iter${current_iter}.csv"

    echo "[save_train_log] 运行${label}分析 (iteration ${current_iter})..."

    python3 "${ANALYZE_SCRIPT}" "${LOG_FILE}" \
        --csv "${csv_file}" \
        > "${out_file}" 2>&1 || true

    echo "[save_train_log] 分析完成: ${out_file}"
    _last_analyzed_iter="${current_iter}"
}

# ============================================================
# 主执行: 训练 + 实时日志 + 增量分析
# ============================================================

if [ "${ANALYZE_INTERVAL}" -gt 0 ]; then
    # 带增量分析模式: 用 pipe + 后台监控
    # 创建命名管道 (放在 /tmp 避免网络文件系统不支持 mkfifo)
    PIPE_FILE=$(mktemp -u "/tmp/train_pipe_XXXXXX")
    mkfifo "${PIPE_FILE}"

    # 后台: 从管道读取，写入日志文件 + 终端，同时监控 iteration
    (
        iter_count=0
        while IFS= read -r line; do
            # 写入日志文件
            echo "${line}" >> "${LOG_FILE}"
            # 写入终端
            echo "${line}"

            # 检测 iteration 行
            if echo "${line}" | grep -qP 'iteration\s+\d+\s*/'; then
                # 提取 iteration 数字
                current_iter=$(echo "${line}" | grep -oP 'iteration\s+\K\d+' | head -1)
                if [ -n "${current_iter}" ]; then
                    iter_count=$((iter_count + 1))
                    # 检查是否到达分析间隔
                    if [ $((current_iter % ANALYZE_INTERVAL)) -eq 0 ] && \
                       [ "${current_iter}" -gt "${_last_analyzed_iter}" ]; then
                        run_incremental_analysis "${current_iter}" "incremental" &
                    fi
                fi
            fi
        done < "${PIPE_FILE}"
    ) &
    READER_PID=$!

    # 前台: 执行训练命令，输出到管道
    "$@" > "${PIPE_FILE}" 2>&1
    EXIT_CODE=$?

    # 等待读取进程结束
    wait "${READER_PID}" 2>/dev/null || true
    rm -f "${PIPE_FILE}"
else
    # 简单模式: 直接 tee
    "$@" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
fi

echo ""
echo "============================================================"
echo " 训练结束"
echo " 退出码:   ${EXIT_CODE}"
echo " 结束时间: $(date)"
echo " 日志文件: ${LOG_FILE}"
echo "============================================================"

# ============================================================
# 训练结束后自动分析
# ============================================================
if [ "${ANALYZE_ON_EXIT}" = "1" ] && [ -f "${ANALYZE_SCRIPT}" ] && [ -s "${LOG_FILE}" ]; then
    echo ""
    echo "[save_train_log] 运行最终分析..."
    mkdir -p "${ANALYSIS_DIR}"

    FINAL_REPORT="${ANALYSIS_DIR}/analysis_final.txt"
    FINAL_CSV="${ANALYSIS_DIR}/analysis_final.csv"
    BSR_REPORT="${ANALYSIS_DIR}/analysis_bsr_only.txt"

    # 完整分析
    python3 "${ANALYZE_SCRIPT}" "${LOG_FILE}" \
        --csv "${FINAL_CSV}" \
        > "${FINAL_REPORT}" 2>&1 || true

    # BSR-only 分析
    python3 "${ANALYZE_SCRIPT}" "${LOG_FILE}" \
        --bsr-only \
        > "${BSR_REPORT}" 2>&1 || true

    echo "[save_train_log] 分析报告:"
    echo "  完整报告: ${FINAL_REPORT}"
    echo "  BSR 报告: ${BSR_REPORT}"
    echo "  CSV 数据: ${FINAL_CSV}"
    echo ""

    # 打印 BSR 摘要到终端
    if [ -s "${BSR_REPORT}" ]; then
        echo "============================================================"
        echo " BSR-MoE 故障恢复分析摘要"
        echo "============================================================"
        cat "${BSR_REPORT}"
    fi
fi

exit ${EXIT_CODE}
