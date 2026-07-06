#!/usr/bin/env bash
# save_train_log.sh — Megatron + MoEGambit 训练日志保存与增量分析
#
# 功能:
#   1. 只保留最后一次运行的日志 (固定文件名 train_latest.log)
#   2. 训练过程中实时写入 latest/full 日志，进程被 SIGTERM/INT 时尽量保留已输出内容
#   3. 每隔 N 个 iteration 自动运行一次日志分析 (可配置)
#   4. 训练结束后自动运行完整分析，报告采用 tmp + atomic mv，避免半截报告冒充完整报告
#
# 用法:
#   bash save_train_log.sh <训练命令...>
#
# 环境变量:
#   TRAIN_LOG_DIR          日志保存目录 (默认 ./train_logs)
#   LOG_ANALYZE_INTERVAL   每隔多少个 iteration 分析一次 (默认 0=不做增量分析)
#   LOG_ANALYZE_SCRIPT     分析脚本路径 (默认 同目录下的 analyze_train_log.py)
#   LOG_ANALYZE_ON_EXIT    训练结束后是否自动分析 (默认 1=是)
#   LOG_ANALYZE_ON_SIGNAL  收到 SIGTERM/INT 后是否仍尝试最终分析 (默认 1=是)
#   LOG_SYNC_EVERY_LINES   每写入多少行 fsync 一次日志文件 (默认 0=不强制 fsync)
#   TRAIN_RUN_ID           本次运行 id (默认 timestamp_pid)
#   TRAIN_LOG_FALLBACK_DIR 主日志目录不可写时的降级目录 (默认 ./train_logs_fallback)
#
# 示例:
#   LOG_ANALYZE_INTERVAL=100 bash save_train_log.sh torchrun ... pretrain_gpt.py ...
#   LOG_ANALYZE_INTERVAL=50 LOG_ANALYZE_ON_EXIT=1 bash save_train_log.sh ./run_moe.sh

set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "用法: bash save_train_log.sh <训练命令...>" >&2
    exit 2
fi

# ============================================================
# 配置
# ============================================================
LOG_DIR="${TRAIN_LOG_DIR:-./train_logs}"
PRIMARY_LOG_DIR="${LOG_DIR}"
FALLBACK_LOG_DIR="${TRAIN_LOG_FALLBACK_DIR:-./train_logs_fallback}"
ANALYZE_INTERVAL="${LOG_ANALYZE_INTERVAL:-0}"
ANALYZE_ON_EXIT="${LOG_ANALYZE_ON_EXIT:-1}"
ANALYZE_ON_SIGNAL="${LOG_ANALYZE_ON_SIGNAL:-1}"
SYNC_EVERY_LINES="${LOG_SYNC_EVERY_LINES:-0}"
RUN_ID="${TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

case "${ANALYZE_INTERVAL}" in
    ''|*[!0-9]*)
        echo "[save_train_log] LOG_ANALYZE_INTERVAL=${ANALYZE_INTERVAL} 非法，回退为 0" >&2
        ANALYZE_INTERVAL=0
        ;;
esac

case "${SYNC_EVERY_LINES}" in
    ''|*[!0-9]*)
        echo "[save_train_log] LOG_SYNC_EVERY_LINES=${SYNC_EVERY_LINES} 非法，回退为 0" >&2
        SYNC_EVERY_LINES=0
        ;;
esac

# 分析脚本路径: 默认与本脚本同目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_SCRIPT="${LOG_ANALYZE_SCRIPT:-${SCRIPT_DIR}/analyze_train_log.py}"

ensure_dir() {
    local dir="$1"
    [ -n "${dir}" ] || return 1
    mkdir -p "${dir}" 2>/dev/null
}

if ! ensure_dir "${LOG_DIR}"; then
    echo "[save_train_log] 日志目录不可写: ${LOG_DIR}，降级到 ${FALLBACK_LOG_DIR}" >&2
    LOG_DIR="${FALLBACK_LOG_DIR}"
    if ! ensure_dir "${LOG_DIR}"; then
        LOG_DIR="${TMPDIR:-/tmp}/train_logs_${USER:-unknown}"
        echo "[save_train_log] fallback 日志目录不可写，继续降级到 ${LOG_DIR}" >&2
        ensure_dir "${LOG_DIR}" || {
            echo "[save_train_log] 无法创建任何日志目录，退出" >&2
            exit 1
        }
    fi
fi

# 两个日志文件:
# 1. train_latest.log - 只保留最后一次运行（覆盖）
# 2. train_full.log - 从头累积所有运行（追加）
LOG_FILE_LATEST="${LOG_DIR}/train_latest.log"
LOG_FILE_FULL="${LOG_DIR}/train_full.log"
STATUS_FILE="${LOG_DIR}/train_latest.status"
ANALYSIS_DIR="${LOG_DIR}/analysis_latest"

# 日志写入目标可独立降级。比如 NFS 上 latest 失效时，至少保留终端输出和 full。
LATEST_LOG_DISABLED=0
FULL_LOG_DISABLED=0
STATUS_DISABLED=0

prepare_log_file() {
    local path="$1"
    local mode="$2"

    ensure_dir "$(dirname "${path}")" || return 1

    if [ "${mode}" = "truncate" ]; then
        # rm + noclobber-safe create: 避免外层 SHELLOPTS=noclobber 时 ": > file" 失败。
        rm -f "${path}" 2>/dev/null || true
        : 2>/dev/null >| "${path}" || return 1
    else
        : 2>/dev/null >> "${path}" || return 1
    fi
}

# 清空 latest 日志，开始新的记录；full 仅确保可追加。
if ! prepare_log_file "${LOG_FILE_LATEST}" "truncate"; then
    echo "[save_train_log] 无法初始化 latest 日志: ${LOG_FILE_LATEST}，将只输出到终端/full" >&2
    LATEST_LOG_DISABLED=1
fi
if ! prepare_log_file "${LOG_FILE_FULL}" "append"; then
    echo "[save_train_log] 无法初始化 full 日志: ${LOG_FILE_FULL}，将只输出到终端/latest" >&2
    FULL_LOG_DISABLED=1
fi
rm -rf "${ANALYSIS_DIR}" 2>/dev/null || true
ensure_dir "${ANALYSIS_DIR}" || true

PIPE_DIR=""
PIPE_FILE=""
READER_PID=""
TRAIN_PID=""
EXIT_CODE=0
TERMINATING=0
TERMINATING_SIGNAL=""
_last_analyzed_iter=0

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    local signal="${3:-}"
    local tmp_file="${STATUS_FILE}.tmp.$$"

    if [ "${STATUS_DISABLED}" -eq 1 ]; then
        return
    fi

    ensure_dir "$(dirname "${STATUS_FILE}")" || {
        STATUS_DISABLED=1
        return
    }

    if ! {
        echo "run_id=${RUN_ID}"
        echo "state=${state}"
        echo "exit_code=${exit_code}"
        echo "signal=${signal}"
        echo "updated_at=$(date)"
        echo "primary_log_dir=${PRIMARY_LOG_DIR}"
        echo "active_log_dir=${LOG_DIR}"
        echo "latest_log=${LOG_FILE_LATEST}"
        echo "full_log=${LOG_FILE_FULL}"
        echo "analysis_dir=${ANALYSIS_DIR}"
    } 2>/dev/null > "${tmp_file}"; then
        STATUS_DISABLED=1
        rm -f "${tmp_file}" 2>/dev/null || true
        return
    fi

    if ! mv -f "${tmp_file}" "${STATUS_FILE}" 2>/dev/null; then
        STATUS_DISABLED=1
        rm -f "${tmp_file}" 2>/dev/null || true
    fi
}

safe_append_log() {
    local path="$1"
    local label="$2"
    local disabled_var="$3"
    local line="$4"

    if [ "${!disabled_var}" -eq 1 ]; then
        return
    fi

    if { printf '%s\n' "${line}"; } 2>/dev/null >> "${path}"; then
        return
    fi

    # Slow path only: shared filesystems can briefly lose directories/handles.
    # Recreate the parent once and retry, but do not pay mkdir cost per line.
    ensure_dir "$(dirname "${path}")" || {
        printf '[save_train_log] 无法写入 %s 日志: %s，已禁用该目标\n' "${label}" "${path}" >&2
        printf -v "${disabled_var}" '1'
        return
    }

    if ! { printf '%s\n' "${line}"; } 2>/dev/null >> "${path}"; then
        printf '[save_train_log] 无法写入 %s 日志: %s，已禁用该目标\n' "${label}" "${path}" >&2
        printf -v "${disabled_var}" '1'
    fi
}

append_line_to_logs() {
    local line="$1"
    printf '%s\n' "${line}"
    safe_append_log "${LOG_FILE_LATEST}" "latest" "LATEST_LOG_DISABLED" "${line}"
    safe_append_log "${LOG_FILE_FULL}" "full" "FULL_LOG_DISABLED" "${line}"
}

emit_control() {
    append_line_to_logs "$*"
}

sync_log_files() {
    if [ "${SYNC_EVERY_LINES}" -le 0 ]; then
        return
    fi

    local paths=()
    if [ "${LATEST_LOG_DISABLED}" -eq 0 ]; then
        paths+=("${LOG_FILE_LATEST}")
    fi
    if [ "${FULL_LOG_DISABLED}" -eq 0 ]; then
        paths+=("${LOG_FILE_FULL}")
    fi
    if [ "${#paths[@]}" -eq 0 ]; then
        return
    fi

    python3 -c 'import os, sys
for path in sys.argv[1:]:
    if not os.path.exists(path):
        continue
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
' "${paths[@]}" 2>/dev/null || true
}

run_training_command() {
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

    if command -v python3 >/dev/null 2>&1; then
        if command -v stdbuf >/dev/null 2>&1; then
            exec python3 -c 'import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' stdbuf -oL -eL "$@"
        else
            exec python3 -c 'import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' "$@"
        fi
    elif command -v setsid >/dev/null 2>&1; then
        if command -v stdbuf >/dev/null 2>&1; then
            exec setsid stdbuf -oL -eL "$@"
        else
            exec setsid "$@"
        fi
    elif command -v stdbuf >/dev/null 2>&1; then
        exec stdbuf -oL -eL "$@"
    else
        exec "$@"
    fi
}

# ============================================================
# 分析函数：全部采用 tmp + atomic mv，避免生成半截正式报告
# ============================================================
run_atomic_analysis() {
    local label="$1"
    local input_file="$2"
    local out_file="$3"
    local csv_file="$4"
    shift 4

    if [ ! -f "${ANALYZE_SCRIPT}" ]; then
        emit_control "[save_train_log] 分析脚本不存在: ${ANALYZE_SCRIPT}, 跳过 ${label} 分析"
        return
    fi

    if [ ! -s "${input_file}" ]; then
        emit_control "[save_train_log] 日志为空: ${input_file}, 跳过 ${label} 分析"
        return
    fi

    if ! ensure_dir "$(dirname "${out_file}")"; then
        emit_control "[save_train_log] 无法创建分析目录: $(dirname "${out_file}")，跳过 ${label} 分析"
        return
    fi

    local tmp_out="${out_file}.tmp.$$"
    local tmp_csv=""
    local rc=0

    if [ -n "${csv_file}" ]; then
        tmp_csv="${csv_file}.tmp.$$"
        python3 "${ANALYZE_SCRIPT}" "${input_file}" --csv "${tmp_csv}" "$@" \
            > "${tmp_out}" 2>&1
        rc=$?
    else
        python3 "${ANALYZE_SCRIPT}" "${input_file}" "$@" \
            > "${tmp_out}" 2>&1
        rc=$?
    fi

    if [ "${rc}" -eq 0 ]; then
        mv -f "${tmp_out}" "${out_file}"
        if [ -n "${csv_file}" ] && [ -f "${tmp_csv}" ]; then
            mv -f "${tmp_csv}" "${csv_file}"
        fi
        emit_control "[save_train_log] ${label} 分析完成: ${out_file}"
    else
        local failed_out="${out_file}.failed.$$"
        if [ -f "${tmp_out}" ]; then
            mv -f "${tmp_out}" "${failed_out}"
        fi
        if [ -n "${tmp_csv}" ]; then
            rm -f "${tmp_csv}"
        fi
        emit_control "[save_train_log] ${label} 分析失败(rc=${rc})，未覆盖正式报告；失败输出: ${failed_out}"
    fi
}

run_incremental_analysis() {
    local current_iter="$1"
    local label="incremental"
    local out_file="${ANALYSIS_DIR}/analysis_${label}_iter${current_iter}.txt"
    local csv_file="${ANALYSIS_DIR}/analysis_${label}_iter${current_iter}.csv"

    emit_control "[save_train_log] 运行${label}分析 (iteration ${current_iter})..."
    run_atomic_analysis "${label}" "${LOG_FILE_LATEST}" "${out_file}" "${csv_file}"
}

run_final_analysis() {
    if [ "${ANALYZE_ON_EXIT}" != "1" ]; then
        return
    fi

    if [ "${TERMINATING}" -eq 1 ] && [ "${ANALYZE_ON_SIGNAL}" != "1" ]; then
        emit_control "[save_train_log] 收到 ${TERMINATING_SIGNAL}，按配置跳过最终分析"
        return
    fi

    if [ ! -f "${ANALYZE_SCRIPT}" ] || [ ! -s "${LOG_FILE_LATEST}" ]; then
        return
    fi

    emit_control ""
    emit_control "[save_train_log] 运行最终分析..."
    ensure_dir "${ANALYSIS_DIR}" || {
        emit_control "[save_train_log] 无法创建分析目录: ${ANALYSIS_DIR}，跳过最终分析"
        return
    }

    local final_report="${ANALYSIS_DIR}/analysis_final.txt"
    local final_csv="${ANALYSIS_DIR}/analysis_final.csv"
    local moegambit_report="${ANALYSIS_DIR}/analysis_moegambit_only.txt"

    run_atomic_analysis "final" "${LOG_FILE_LATEST}" "${final_report}" "${final_csv}"
    run_atomic_analysis "moegambit-only" "${LOG_FILE_LATEST}" "${moegambit_report}" "" --moegambit-only

    emit_control "[save_train_log] 分析报告:"
    emit_control "  完整报告: ${final_report}"
    emit_control "  moegambit 报告: ${moegambit_report}"
    emit_control "  CSV 数据: ${final_csv}"
    emit_control ""

    # 打印 moegambit 摘要到终端。正式报告只在 analyzer 完成后才会被 atomic mv 出来。
    if [ -s "${moegambit_report}" ]; then
        echo "============================================================"
        echo " MoEGambit 故障恢复分析摘要"
        echo "============================================================"
        cat "${moegambit_report}"
    fi
}

cleanup() {
    if [ -n "${PIPE_FILE}" ]; then
        rm -f "${PIPE_FILE}" 2>/dev/null || true
    fi
    if [ -n "${PIPE_DIR}" ]; then
        rmdir "${PIPE_DIR}" 2>/dev/null || true
    fi
}

kill_child_tree() {
    local sig="$1"
    local parent_pid="$2"
    local child_pid=""

    if ! command -v pgrep >/dev/null 2>&1; then
        return
    fi

    for child_pid in $(pgrep -P "${parent_pid}" 2>/dev/null); do
        kill_child_tree "${sig}" "${child_pid}"
        kill "-${sig}" "${child_pid}" 2>/dev/null || true
    done
}

terminate_training_processes() {
    local sig="$1"

    if [ -z "${TRAIN_PID}" ] || ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        return
    fi

    # run_training_command starts the training command in its own session
    # when python3/setsid is available, so this reaches torchrun children too.
    kill "-${sig}" "-${TRAIN_PID}" 2>/dev/null || true

    kill_child_tree "${sig}" "${TRAIN_PID}"
    kill "-${sig}" "${TRAIN_PID}" 2>/dev/null || true
}

on_signal() {
    local sig="$1"
    TERMINATING=1
    TERMINATING_SIGNAL="${sig}"
    write_status "terminating" "" "${sig}"
    emit_control "[save_train_log] 收到 ${sig}，正在转发给训练进程并保留已写入日志..."

    terminate_training_processes "${sig}"
}

trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap 'on_signal QUIT' QUIT

# ============================================================
# 运行头
# ============================================================
emit_control "============================================================"
emit_control " Megatron 训练日志保存"
emit_control " run_id:       ${RUN_ID}"
emit_control " 最新日志:     ${LOG_FILE_LATEST} (覆盖)"
emit_control " 完整日志:     ${LOG_FILE_FULL} (追加)"
emit_control " 状态文件:     ${STATUS_FILE}"
emit_control " 分析间隔:     ${ANALYZE_INTERVAL} iterations (0=仅结束时分析)"
emit_control " 结束时分析:   ${ANALYZE_ON_EXIT}"
emit_control " 信号后分析:   ${ANALYZE_ON_SIGNAL}"
emit_control " 分析脚本:     ${ANALYZE_SCRIPT}"
COMMAND_DESC="${TRAIN_LAUNCH_DESC:-$*}"
emit_control " 启动命令:     ${COMMAND_DESC}"
emit_control " 开始时间:     $(date)"
emit_control "============================================================"

write_status "running" "" ""

# ============================================================
# 主执行: 训练 + 实时日志 + 增量分析
# ============================================================
PIPE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/train_log_pipe.XXXXXX")"
PIPE_FILE="${PIPE_DIR}/stream"
mkfifo "${PIPE_FILE}"

(
    line_count=0
    while IFS= read -r line || [ -n "${line}" ]; do
        append_line_to_logs "${line}"
        line_count=$((line_count + 1))

        if [ "${SYNC_EVERY_LINES}" -gt 0 ] && \
           [ $((line_count % SYNC_EVERY_LINES)) -eq 0 ]; then
            sync_log_files
        fi

        if [ "${ANALYZE_INTERVAL}" -gt 0 ] && \
           echo "${line}" | grep -qE 'iteration[[:space:]]+[0-9]+[[:space:]]*/'; then
            current_iter="$(echo "${line}" | grep -oE 'iteration[[:space:]]+[0-9]+' | grep -oE '[0-9]+' | head -1)"
            if [ -n "${current_iter}" ] && \
               [ $((current_iter % ANALYZE_INTERVAL)) -eq 0 ] && \
               [ "${current_iter}" -gt "${_last_analyzed_iter}" ]; then
                _last_analyzed_iter="${current_iter}"
                run_incremental_analysis "${current_iter}" &
            fi
        fi
    done < "${PIPE_FILE}"

    sync_log_files
) &
READER_PID=$!

run_training_command "$@" > "${PIPE_FILE}" 2>&1 &
TRAIN_PID=$!

wait "${TRAIN_PID}"
EXIT_CODE=$?

wait "${READER_PID}" 2>/dev/null || true
sync_log_files

if [ "${TERMINATING}" -eq 1 ] && [ "${EXIT_CODE}" -lt 128 ]; then
    case "${TERMINATING_SIGNAL}" in
        INT) EXIT_CODE=130 ;;
        TERM) EXIT_CODE=143 ;;
        HUP) EXIT_CODE=129 ;;
        QUIT) EXIT_CODE=131 ;;
    esac
fi

emit_control ""
emit_control "============================================================"
emit_control " 训练结束"
emit_control " 退出码:   ${EXIT_CODE}"
emit_control " 结束时间: $(date)"
emit_control " 最新日志: ${LOG_FILE_LATEST}"
emit_control " 完整日志: ${LOG_FILE_FULL}"
emit_control "============================================================"

write_status "finished" "${EXIT_CODE}" "${TERMINATING_SIGNAL}"
run_final_analysis

exit "${EXIT_CODE}"
