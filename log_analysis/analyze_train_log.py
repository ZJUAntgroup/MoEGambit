#!/usr/bin/env python3
"""
analyze_train_log.py — Megatron + BSR-MoE 训练日志耗时分析工具

离线解析 Megatron 训练日志，统计各阶段耗时及占比，
并提取 BSR-MoE 故障恢复系统的完整事件时间线。

用法:
    python analyze_train_log.py <日志文件>
    python analyze_train_log.py train.log --top 20
    python analyze_train_log.py train.log --csv output.csv
    python analyze_train_log.py train.log --iter-range 10-100
    python analyze_train_log.py train.log --bsr-only

支持的日志格式:
    1. iteration 行:  [datetime] iteration N/M | ... | elapsed time per iteration (ms): XXX | ...
    2. timer 行:      (min, max) time across ranks (ms): / max time across ranks (ms):
    3. checkpoint 行: saving checkpoint at iteration N
    4. evaluate 行:   evaluating ...
    5. setup 行:      model-and-optimizer-setup / train/valid/test-data-iterators-setup
    6. BSR-MoE 事件:  BSR-MoE 前缀的所有日志（故障注入、隔离、状态迁移、恢复等）

分析输出:
    - 每个 timer 的总耗时、平均耗时、调用次数、占比
    - 按类别 (data/compute/comm/optimizer/checkpoint/eval/bsr/other) 汇总
    - iteration 耗时统计 (min/max/mean/p50/p95/p99)
    - BSR-MoE 事件时间线和恢复耗时分析
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. Timer 分类规则
# ============================================================

TIMER_CATEGORY_EXACT = {
    # compute
    'forward-backward':       'compute',
    'forward-compute':        'compute',
    'backward-compute':       'compute',
    # comm
    'layernorm-grads-all-reduce':   'comm',
    'embedding-grads-all-reduce':   'comm',
    'all-grads-sync':               'comm',
    'params-all-gather':            'comm',
    'forward-recv':                 'comm',
    'forward-send':                 'comm',
    'backward-recv':                'comm',
    'backward-send':                'comm',
    'forward-send-forward-recv':    'comm',
    'forward-send-backward-recv':   'comm',
    'backward-send-forward-recv':   'comm',
    'backward-send-backward-recv':  'comm',
    'forward-backward-send-forward-backward-recv': 'comm',
    # optimizer
    'optimizer':                          'optimizer',
    'optimizer-copy-to-main-grad':        'optimizer',
    'optimizer-unscale-and-check-inf':    'optimizer',
    'optimizer-clip-main-grad':           'optimizer',
    'optimizer-count-zeros':              'optimizer',
    'optimizer-inner-step':               'optimizer',
    'optimizer-copy-main-to-model-params':'optimizer',
    # data
    'batch-generator':                         'data',
    'train/valid/test-data-iterators-setup':    'data',
    # checkpoint
    'save-checkpoint':                'checkpoint',
    'save-checkpoint-non-persistent': 'checkpoint',
    'load-checkpoint':                'checkpoint',
    # eval
    'evaluate':   'eval',
    'eval-time':  'eval',
    # setup
    'model-and-optimizer-setup': 'setup',
}

TIMER_CATEGORY_KEYWORDS = [
    ('optimizer',   'optimizer'),
    ('checkpoint',  'checkpoint'),
    ('eval',        'eval'),
    ('valid',       'eval'),
    ('forward',     'compute'),
    ('backward',    'compute'),
    ('all-reduce',  'comm'),
    ('all-gather',  'comm'),
    ('reduce-scatter', 'comm'),
    ('send',        'comm'),
    ('recv',        'comm'),
    ('sync',        'comm'),
    ('batch',       'data'),
    ('data',        'data'),
]


def classify_timer(name: str) -> str:
    """将 timer 名称分类到 data/compute/comm/optimizer/checkpoint/eval/setup/other。"""
    low = name.lower().strip()
    if low in TIMER_CATEGORY_EXACT:
        return TIMER_CATEGORY_EXACT[low]
    for keyword, cat in TIMER_CATEGORY_KEYWORDS:
        if keyword in low:
            return cat
    return 'other'


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class TimerRecord:
    """单个 timer 的累计统计。"""
    name: str
    category: str
    total_ms: float = 0.0
    count: int = 0
    values: list = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class IterationRecord:
    """单次 iteration 的信息。"""
    iteration: int
    elapsed_ms: float
    timestamp: str = ''
    loss: float = 0.0
    lr: float = 0.0


# ============================================================
# 2b. BSR-MoE 事件数据结构
# ============================================================

@dataclass
class BSREvent:
    """单条 BSR-MoE 事件。"""
    timestamp: str          # 原始时间戳字符串
    datetime: Optional[datetime] = None  # 解析后的 datetime
    rank: int = -1          # 产生日志的 rank (-1 = unknown)
    level: str = 'INFO'     # WARNING / INFO / ERROR
    event_type: str = ''    # 事件分类 (见下方 BSR_EVENT_TYPES)
    step: int = -1          # 关联的 training step
    message: str = ''       # 原始消息
    details: Dict = field(default_factory=dict)  # 解析出的结构化字段


@dataclass
class BSRTimingRecord:
    """单条 BSR-MoE 操作的计时记录。"""
    timestamp: str = ''
    datetime: Optional[datetime] = None
    phase: str = ''           # 操作阶段 (group_repair, dense_sync, expert_restore, pipeline_repair, etc.)
    success: bool = True
    elapsed_seconds: float = 0.0
    step: int = -1
    details: Dict = field(default_factory=dict)
    message: str = ''


# BSR 计时阶段分类
BSR_TIMING_PHASES = {
    'group_repair':         '安全点组修复',
    'dense_sync':           '稠密参数同步',
    'expert_restore':       '专家参数恢复',
    'pipeline_repair':      'Pipeline 修复',
    'pipeline_stage_repair':'Pipeline 阶段修复',
    'pipeline_rollback':    'Pipeline 回滚',
    'async_worker':         '异步恢复 Worker',
    'deferred_optim':       '延迟优化器加载',
    'safe_point_total':     '安全点修复总耗时',
    'unknown_timed':        '其他计时操作',
}


# BSR 事件分类
BSR_EVENT_TYPES = {
    'init':             'BSR 初始化',
    'fault_inject':     '故障注入',
    'quarantine':       'Rank 隔离',
    'quarantine_lift':  '隔离解除',
    'state_transition': '专家状态迁移',
    'health_mask':      '健康掩码更新',
    'sanitize':         'Dispatch 清洗',
    'controller_phase': '控制器阶段迁移',
    'safe_point_repair':'安全点修复',
    'replacement':      '替换 Rank',
    'group_rebuild':    'NCCL 组重建',
    'topology_refresh': '拓扑刷新',
    'dense_sync':       '稠密参数同步',
    'expert_restore':   '专家参数恢复',
    'reintegration':    '重新集成',
    'degraded_policy':  '降级策略',
    'deferred_optim':   '延迟优化器加载',
    'callback_error':   '回调错误',
    'checkpoint_meta':  'Checkpoint 元数据',
    'unknown':          '其他 BSR 事件',
}


# ============================================================
# 3. 日志解析 — Megatron 原有部分
# ============================================================

# --- iteration 行 ---
RE_ITERATION = re.compile(
    r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?\s+'
    r'iteration\s+(\d+)\s*/\s*(\d+)\s*\|'
    r'.*?elapsed time per iteration \(ms\):\s*([\d.]+)'
)

RE_LOSS = re.compile(r'lm loss:\s*([\d.eE+\-]+)')
RE_LR = re.compile(r'learning rate:\s*([\d.eE+\-]+)')

# --- timer 块头 ---
RE_TIMER_HEADER = re.compile(r'(?:(?:\(min,\s*max\)|max)\s+)?time across ranks \(ms\):')

# --- timer 数据行 ---
RE_TIMER_LINE = re.compile(
    r'^\s{2,}([\w/\-]+)\s*\.{2,}:\s*'
    r'(?:\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)|([\d.]+))'
)

# --- checkpoint ---
RE_CKPT_SAVE = re.compile(r'saving checkpoint at iteration\s+(\d+)', re.IGNORECASE)
RE_CKPT_LOAD = re.compile(r'loading checkpoint', re.IGNORECASE)

# --- evaluate ---
RE_EVAL_START = re.compile(r'evaluating', re.IGNORECASE)


# ============================================================
# 3b. BSR-MoE 日志正则
# ============================================================

# 通用时间戳提取 (Megatron 日志格式: [2026-04-14 10:38:39] 或无括号)
RE_TIMESTAMP = re.compile(r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?')

# Rank 前缀 (torchrun 格式: [rank0]: 或 [W rank0]:)
RE_RANK = re.compile(r'\[(?:W\s+)?rank(\d+)\]')

# 日志级别
RE_LOG_LEVEL = re.compile(r'\b(WARNING|INFO|ERROR)\b')

# --- BSR-MoE 事件正则 ---

# 初始化
RE_BSR_INIT = re.compile(r'BSR-MoE:\s*initializ')
RE_BSR_INIT_COMPLETE = re.compile(r'BSR-MoE:\s*initialization complete on rank\s+(\d+)')

# 故障注入 (带时间戳: [ts] BSR-MoE FAULT INJECTION: type=X, rank=X, step=X, experts=[...])
RE_BSR_FAULT_INJECT = re.compile(
    r'BSR-MoE FAULT INJECTION:\s*type=(\w+),\s*rank=(\d+),\s*step=(\d+),\s*experts=(\[[\d,\s]*\])'
)
RE_BSR_FAULT_REPLACEMENT = re.compile(
    r'BSR-MoE FAULT INJECTION:\s*replacement ready.*?'
    r'failed_rank=(\d+),\s*replacement_rank=(\d+),\s*step=(\d+)'
)

# Rank 隔离 (新格式: [ts] BSR-MoE controller: rank N QUARANTINED (step=N, reason=X, experts=[...]))
RE_BSR_QUARANTINE = re.compile(
    r'BSR-MoE controller:\s*rank\s+(\d+)\s+QUARANTINED\s*\(step=(\d+),\s*reason=([^,)]+)'
)
RE_BSR_QUARANTINE_OLD = re.compile(
    r'BSR-MoE:\s*rank\s+(\d+)\s+QUARANTINED\s*\(step=(\d+),\s*reason=([^)]+)\)'
)
RE_BSR_QUARANTINE_LIFT = re.compile(
    r'BSR-MoE.*?rank\s+(\d+)\s+quarantine\s+LIFTED.*?step=(\d+)'
)

# Hard failure (新增: [ts] BSR-MoE controller: rank N HARD FAILURE (step=N, reason=X, ...))
RE_BSR_HARD_FAILURE = re.compile(
    r'BSR-MoE controller:\s*rank\s+(\d+)\s+HARD FAILURE\s*\(step=(\d+),\s*reason=([^,)]+)'
)

# 替换 rank (新增: [ts] BSR-MoE controller: replacement assigned/ready)
RE_BSR_REPLACEMENT_ASSIGNED = re.compile(
    r'BSR-MoE controller:\s*replacement assigned.*?'
    r'failed_rank=(\d+),\s*replacement_rank=(\d+).*?step=(\d+)'
)
RE_BSR_REPLACEMENT_READY = re.compile(
    r'BSR-MoE controller:\s*replacement ready.*?'
    r'failed_rank=(\d+).*?step=(\d+)'
)

# 专家状态迁移
RE_BSR_STATE_TRANSITION = re.compile(
    r'BSR-MoE\s+layer\s+(\d+):\s*experts?\s+(\[[\d,\s]*\])\s*→\s*(\w+)\s*\(step=(\d+)\)'
)

# 健康掩码
RE_BSR_HEALTH_UNHEALTHY = re.compile(
    r'BSR-MoE.*?marked experts?\s+(\[[\d,\s]*\])\s+as\s+UNHEALTHY'
)
RE_BSR_HEALTH_HEALTHY = re.compile(
    r'BSR-MoE.*?marked experts?\s+(\[[\d,\s]*\])\s+as\s+HEALTHY'
)

# Dispatch sanitize
RE_BSR_SANITIZE = re.compile(
    r'BSR-MoE sanitize_routing_map:\s*zeroing\s+(\d+)\s+token-expert assignments\s+'
    r'across\s+(\d+)\s+expert columns\s*\(quarantined EP ranks:\s*(\[[\d,\s]*\])\)'
)

# 控制器阶段迁移 (新格式: [ts] BSR-MoE controller: PHASE_A → PHASE_B (event=X, step=N))
RE_BSR_CONTROLLER_PHASE = re.compile(
    r'BSR-MoE controller:\s*(\w+)\s*→\s*(\w+)\s*\(event=(\w+),\s*step=(\d+)\)'
)

# 安全点修复开始
RE_BSR_SAFE_POINT = re.compile(
    r'BSR-MoE controller:\s*executing safe-point repair at step\s+(\d+)'
)

# 安全点修复完成 (新增: [ts] BSR-MoE controller: safe-point repair COMPLETED total_elapsed=X.XXXs)
RE_BSR_SAFE_POINT_COMPLETED = re.compile(
    r'BSR-MoE controller:\s*safe-point repair COMPLETED.*?'
    r'total_elapsed=([\d.]+)s.*?step=(\d+)'
)

# 重新集成完成
RE_BSR_REINTEGRATION = re.compile(
    r'BSR-MoE controller:\s*reintegration finalized at step\s+(\d+)'
)

# 回调错误
RE_BSR_CALLBACK_ERROR = re.compile(
    r'BSR-MoE\s+(\w+)\s+failed:\s*(.*)'
)

# Dispatch 一致性违规
RE_BSR_DISPATCH_VIOLATION = re.compile(
    r'BSR-MoE dispatch consistency violation'
)

# --- BSR-MoE 计时正则 ---

# 安全点组修复: "[ts] BSR-MoE: safe-point group repair SUCCEEDED — invalidated=X, rebuilt=X, rebound=X, verified=X, elapsed=X.XXs"
RE_BSR_GROUP_REPAIR_TIMING = re.compile(
    r'BSR-MoE.*?safe-point group repair\s+(\w+)\s*—\s*'
    r'invalidated=(\d+),\s*rebuilt=(\d+),\s*rebound=(\d+),\s*'
    r'verified=(\w+),\s*elapsed=([\d.]+)s'
)

# 稠密参数同步: "[ts] BSR-MoE dense param sync: SUCCESS — synced X params (X scalars) from rank X (attempt X, X.XXs, skipped X expert params)"
RE_BSR_DENSE_SYNC_TIMING = re.compile(
    r'BSR-MoE dense param sync:\s*(\w+)\s*—\s*synced\s+(\d+)\s+params\s+'
    r'\((\d+)\s+scalars\)\s+from rank\s+(\d+)\s+'
    r'\(attempt\s+(\d+),\s*([\d.]+)s,\s*skipped\s+(\d+)\s+expert params\)'
)

# 专家恢复 (stale_expert_restore.py): "BSR-MoE stale expert restore: SUCCESS — restored X/X experts (transitions=X, directory=X, barrier=X, X.XXs)"
RE_BSR_EXPERT_RESTORE_TIMING = re.compile(
    r'BSR-MoE stale expert restore:\s*(\w+)\s*—\s*restored\s+(\d+)/(\d+)\s+experts\s+'
    r'\(transitions=(\d+),\s*directory=(\d+),\s*barrier=(\d+),\s*([\d.]+)s\)'
)

# 专家恢复 (bsr_integration): "[ts] BSR-MoE expert_restore_fn: SUCCESS — restored X experts (X state transitions, X directory updates, X barrier params, X.XXs)"
RE_BSR_EXPERT_RESTORE_FN_TIMING = re.compile(
    r'BSR-MoE expert_restore_fn:\s*(\w+)\s*—\s*restored\s+(\d+)\s+experts\s+'
    r'\((\d+)\s+state transitions,\s*(\d+)\s+directory updates,\s*(\d+)\s+barrier params,\s*([\d.]+)s\)'
)

# Pipeline 修复: "BSR-MoE pipeline repair SUCCEEDED — stage=X, ..., elapsed=X.XXs"
RE_BSR_PIPELINE_REPAIR_TIMING = re.compile(
    r'BSR-MoE pipeline repair\s+(\w+)\s*—\s*stage=(\d+).*?elapsed=([\d.]+)s'
)

# Pipeline stage repair (bsr_integration): "[ts] BSR-MoE pipeline_stage_repair_fn: SUCCESS — pp_rebuilt=X, prev_next=X, p2p_rebound=X, elapsed=X.XXs"
RE_BSR_PIPELINE_STAGE_REPAIR_FN_TIMING = re.compile(
    r'BSR-MoE pipeline_stage_repair_fn:\s*(\w+)\s*—\s*'
    r'pp_rebuilt=(\w+),\s*prev_next=(\w+),\s*p2p_rebound=(\w+),\s*elapsed=([\d.]+)s'
)

# Pipeline rollback: "BSR-MoE pipeline: rollback COMPLETE — step=X, ..., elapsed=X.XXXs"
RE_BSR_PIPELINE_ROLLBACK_TIMING = re.compile(
    r'BSR-MoE pipeline:\s*rollback\s+COMPLETE\s*—\s*step=(\d+).*?elapsed=([\d.]+)s'
)

# 异步恢复 worker: "BSR-MoE async worker: completed X X (layer=X, expert=X, success=X, X.XXXs)"
RE_BSR_ASYNC_WORKER_TIMING = re.compile(
    r'BSR-MoE async worker:\s*completed\s+(\w+)\s+\S+\s+'
    r'\(layer=(\d+),\s*expert=(\d+),\s*success=(\w+),\s*([\d.]+)s\)'
)

# 延迟优化器加载: "BSR-MoE deferred loader: loaded optimizer state for expert (layer=X, id=X) at step X (X.XXs)"
RE_BSR_DEFERRED_OPTIM_TIMING = re.compile(
    r'BSR-MoE deferred loader:\s*(?:loaded|async)\s+optimizer\s+.*?'
    r'\(layer=(\d+),\s*(?:id|expert)=(\d+)\)\s+at step\s+(\d+)\s+\(([\d.]+)s\)'
)

# 控制器步骤计时 (新增): "[ts] BSR-MoE controller: stepN xxx elapsed=X.XXXs (step=N)"
# 也匹配 "step7 expert_restore (sync) elapsed=..." 和 "step8 reintegration barrier ..."
RE_BSR_CONTROLLER_STEP_TIMING = re.compile(
    r'BSR-MoE controller:\s*step(\d+)\s+(.+?)\s+elapsed=([\d.]+)s\s+\(step=(\d+)\)'
)

# 安全点修复总耗时 (新增): "[ts] BSR-MoE controller: safe-point repair COMPLETED total_elapsed=X.XXXs (step=N, ...)"
RE_BSR_SAFE_POINT_TOTAL_TIMING = re.compile(
    r'BSR-MoE controller:\s*safe-point repair COMPLETED.*?'
    r'total_elapsed=([\d.]+)s.*?step=(\d+)'
)

# 异步专家恢复提交 (新增): "[ts] BSR-MoE controller: step7 async expert restore submitted (N requests, elapsed=X.XXXs, step=N)"
RE_BSR_ASYNC_EXPERT_SUBMIT_TIMING = re.compile(
    r'BSR-MoE controller:\s*step7 async expert restore submitted\s+'
    r'\((\d+)\s+requests,\s*elapsed=([\d.]+)s,\s*step=(\d+)\)'
)

# 通用 elapsed 提取 (兜底): 匹配任何 BSR-MoE 行中的 elapsed=X.XXs 或 (X.XXs)
RE_BSR_ELAPSED_GENERIC = re.compile(r'(?:elapsed=|[\(\s])([\d.]+)s[)\s,]')

# 通用 BSR-MoE 行 (兜底)
RE_BSR_GENERIC = re.compile(r'BSR-MoE')


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """解析时间戳字符串为 datetime。"""
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_step_from_line(line: str) -> int:
    """尝试从日志行中提取 step 数字。"""
    m = re.search(r'step[=:]\s*(\d+)', line)
    if m:
        return int(m.group(1))
    return -1


def _classify_bsr_event(line: str) -> Tuple[str, Dict]:
    """对一行 BSR-MoE 日志进行事件分类，返回 (event_type, details)。"""

    m = RE_BSR_FAULT_INJECT.search(line)
    if m:
        return 'fault_inject', {
            'fault_type': m.group(1), 'target_rank': int(m.group(2)),
            'step': int(m.group(3)), 'experts': m.group(4),
        }

    m = RE_BSR_FAULT_REPLACEMENT.search(line)
    if m:
        return 'replacement', {
            'failed_rank': int(m.group(1)), 'replacement_rank': int(m.group(2)),
            'step': int(m.group(3)),
        }

    # 新格式: BSR-MoE controller: rank N QUARANTINED
    m = RE_BSR_QUARANTINE.search(line)
    if m:
        return 'quarantine', {
            'rank': int(m.group(1)), 'step': int(m.group(2)),
            'reason': m.group(3).strip("'\""),
        }

    # 旧格式: BSR-MoE: rank N QUARANTINED
    m = RE_BSR_QUARANTINE_OLD.search(line)
    if m:
        return 'quarantine', {
            'rank': int(m.group(1)), 'step': int(m.group(2)),
            'reason': m.group(3).strip("'\""),
        }

    m = RE_BSR_QUARANTINE_LIFT.search(line)
    if m:
        return 'quarantine_lift', {
            'rank': int(m.group(1)), 'step': int(m.group(2)),
        }

    # Hard failure
    m = RE_BSR_HARD_FAILURE.search(line)
    if m:
        return 'quarantine', {
            'rank': int(m.group(1)), 'step': int(m.group(2)),
            'reason': m.group(3).strip("'\""),
            'fault_type': 'hard',
        }

    # Replacement assigned (新增)
    m = RE_BSR_REPLACEMENT_ASSIGNED.search(line)
    if m:
        return 'replacement', {
            'failed_rank': int(m.group(1)), 'replacement_rank': int(m.group(2)),
            'step': int(m.group(3)),
        }

    # Replacement ready (新增)
    m = RE_BSR_REPLACEMENT_READY.search(line)
    if m:
        return 'replacement', {
            'failed_rank': int(m.group(1)), 'step': int(m.group(2)),
            'sub': 'ready',
        }

    m = RE_BSR_STATE_TRANSITION.search(line)
    if m:
        return 'state_transition', {
            'layer': int(m.group(1)), 'experts': m.group(2),
            'new_state': m.group(3), 'step': int(m.group(4)),
        }

    m = RE_BSR_HEALTH_UNHEALTHY.search(line)
    if m:
        return 'health_mask', {'action': 'UNHEALTHY', 'experts': m.group(1)}

    m = RE_BSR_HEALTH_HEALTHY.search(line)
    if m:
        return 'health_mask', {'action': 'HEALTHY', 'experts': m.group(1)}

    m = RE_BSR_SANITIZE.search(line)
    if m:
        return 'sanitize', {
            'zeroed_tokens': int(m.group(1)), 'expert_cols': int(m.group(2)),
            'quarantined_ep_ranks': m.group(3),
        }

    m = RE_BSR_CONTROLLER_PHASE.search(line)
    if m:
        return 'controller_phase', {
            'from_phase': m.group(1), 'to_phase': m.group(2),
            'event': m.group(3), 'step': int(m.group(4)),
        }

    # 安全点修复完成 (新增，带总耗时)
    m = RE_BSR_SAFE_POINT_COMPLETED.search(line)
    if m:
        return 'safe_point_repair', {
            'sub': 'completed',
            'total_elapsed': float(m.group(1)),
            'step': int(m.group(2)),
        }

    m = RE_BSR_SAFE_POINT.search(line)
    if m:
        return 'safe_point_repair', {'step': int(m.group(1))}

    m = RE_BSR_REINTEGRATION.search(line)
    if m:
        return 'reintegration', {'step': int(m.group(1))}

    m = RE_BSR_CALLBACK_ERROR.search(line)
    if m:
        return 'callback_error', {'callback': m.group(1), 'error': m.group(2)}

    if RE_BSR_INIT_COMPLETE.search(line):
        return 'init', {'sub': 'complete'}
    if RE_BSR_INIT.search(line):
        return 'init', {'sub': 'start'}

    if RE_BSR_DISPATCH_VIOLATION.search(line):
        return 'sanitize', {'sub': 'violation'}

    # 新增: replacement_registry 日志
    if 'replacement_registry' in line or 'replacement:' in line.lower():
        if 'replacement announced' in line:
            m = re.search(r'failed_rank=(\d+)', line)
            return 'replacement', {
                'sub': 'announced',
                'failed_rank': int(m.group(1)) if m else -1,
            }
        if 'replacement:' in line:
            m = re.search(r'failed_rank=(\d+)', line)
            return 'replacement', {
                'sub': 'registry',
                'failed_rank': int(m.group(1)) if m else -1,
            }

    # 新增: checkpoint metadata
    if 'checkpoint metadata injected' in line:
        m = re.search(r'iteration=(\d+).*?phase=(\w+)', line)
        if m:
            return 'checkpoint_meta', {
                'iteration': int(m.group(1)),
                'phase': m.group(2),
            }

    # 新增: manifest saved
    if 'manifest saved' in line:
        return 'checkpoint_meta', {'sub': 'manifest'}

    # 新增: zeroed expert parameter tensors
    if 'zeroed' in line and 'expert parameter tensors' in line:
        return 'checkpoint_meta', {'sub': 'zeroed_params'}

    # 新增: pipeline rollback coordinator
    if 'pipeline rollback coordinator' in line:
        return 'init', {'sub': 'pipeline_rollback'}

    # 新增: reintegration barrier wired
    if 'reintegration barrier wired' in line:
        return 'init', {'sub': 'barrier_wired'}

    # 新增: 各种初始化日志
    init_keywords = [
        'initialized', 'initializ', 'configured', 'wired',
        'num_experts=', 'health masks', 'quarantine registry',
        'expert directory', 'dispatch topology', 'recovery controller',
        'hard failure detector', 'rollback/replay', 'optimizer commit guard',
        'async recovery worker', 'fault injection configured',
        'AsyncRecoveryWorker', 'worker threads',
    ]
    for kw in init_keywords:
        if kw in line:
            return 'init', {'sub': 'setup'}

    return 'unknown', {}


def _parse_bsr_timing(line: str) -> Optional[BSRTimingRecord]:
    """尝试从 BSR-MoE 日志行中提取计时信息。

    Returns:
        BSRTimingRecord if the line contains timing info, else None.
    """
    # 提取时间戳
    ts_match = RE_TIMESTAMP.search(line)
    ts_str = ts_match.group(1) if ts_match else ''
    ts_dt = _parse_timestamp(ts_str) if ts_str else None
    step = _extract_step_from_line(line)

    # 安全点组修复
    m = RE_BSR_GROUP_REPAIR_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='group_repair',
            success=(m.group(1).upper() == 'SUCCEEDED'),
            elapsed_seconds=float(m.group(6)), step=step,
            details={
                'invalidated': int(m.group(2)), 'rebuilt': int(m.group(3)),
                'rebound': int(m.group(4)), 'verified': m.group(5),
            },
            message=line.strip(),
        )

    # 稠密参数同步
    m = RE_BSR_DENSE_SYNC_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='dense_sync',
            success=(m.group(1).upper() == 'SUCCESS'),
            elapsed_seconds=float(m.group(6)), step=step,
            details={
                'synced_params': int(m.group(2)), 'synced_scalars': int(m.group(3)),
                'source_rank': int(m.group(4)), 'attempt': int(m.group(5)),
                'skipped_expert_params': int(m.group(7)),
            },
            message=line.strip(),
        )

    # 专家恢复 (stale_expert_restore.py)
    m = RE_BSR_EXPERT_RESTORE_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='expert_restore',
            success=(m.group(1).upper() == 'SUCCESS'),
            elapsed_seconds=float(m.group(7)), step=step,
            details={
                'restored': int(m.group(2)), 'total': int(m.group(3)),
                'transitions': int(m.group(4)), 'directory': int(m.group(5)),
                'barrier': int(m.group(6)),
            },
            message=line.strip(),
        )

    # 专家恢复 (bsr_integration.py expert_restore_fn)
    m = RE_BSR_EXPERT_RESTORE_FN_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='expert_restore',
            success=(m.group(1).upper() == 'SUCCESS'),
            elapsed_seconds=float(m.group(6)), step=step,
            details={
                'restored': int(m.group(2)), 'state_transitions': int(m.group(3)),
                'directory_updates': int(m.group(4)), 'barrier_params': int(m.group(5)),
            },
            message=line.strip(),
        )

    # Pipeline 修复
    m = RE_BSR_PIPELINE_REPAIR_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='pipeline_repair',
            success=(m.group(1).upper() == 'SUCCEEDED'),
            elapsed_seconds=float(m.group(3)), step=step,
            details={'stage': int(m.group(2))},
            message=line.strip(),
        )

    # Pipeline stage repair (bsr_integration)
    m = RE_BSR_PIPELINE_STAGE_REPAIR_FN_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='pipeline_stage_repair',
            success=(m.group(1).upper() == 'SUCCESS'),
            elapsed_seconds=float(m.group(5)), step=step,
            details={
                'pp_rebuilt': m.group(2), 'prev_next': m.group(3),
                'p2p_rebound': m.group(4),
            },
            message=line.strip(),
        )

    # Pipeline rollback
    m = RE_BSR_PIPELINE_ROLLBACK_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='pipeline_rollback',
            success=True,
            elapsed_seconds=float(m.group(2)), step=int(m.group(1)),
            message=line.strip(),
        )

    # 异步恢复 worker
    m = RE_BSR_ASYNC_WORKER_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='async_worker',
            success=(m.group(4).lower() == 'true'),
            elapsed_seconds=float(m.group(5)), step=step,
            details={
                'request_type': m.group(1), 'layer': int(m.group(2)),
                'expert': int(m.group(3)),
            },
            message=line.strip(),
        )

    # 延迟优化器加载
    m = RE_BSR_DEFERRED_OPTIM_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='deferred_optim',
            success=True,
            elapsed_seconds=float(m.group(4)), step=int(m.group(3)),
            details={'layer': int(m.group(1)), 'expert': int(m.group(2))},
            message=line.strip(),
        )

    # 控制器步骤计时 (新增)
    m = RE_BSR_CONTROLLER_STEP_TIMING.search(line)
    if m:
        step_num = m.group(1)
        step_name = m.group(2).strip()
        # 映射步骤名到阶段 (支持带括号的格式如 "expert_restore (sync)")
        step_name_lower = step_name.lower()
        if 'replacement_integrate' in step_name_lower:
            phase = 'group_repair'
        elif 'group_rebuild' in step_name_lower:
            phase = 'group_repair'
        elif 'topology_refresh' in step_name_lower:
            phase = 'group_repair'
        elif 'dense_sync' in step_name_lower:
            phase = 'dense_sync'
        elif 'expert_restore' in step_name_lower or 'async expert' in step_name_lower:
            phase = 'expert_restore'
        elif 'pipeline' in step_name_lower:
            phase = 'pipeline_stage_repair'
        else:
            phase = 'unknown_timed'
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase=phase,
            success=True,
            elapsed_seconds=float(m.group(3)), step=int(m.group(4)),
            details={'controller_step': int(step_num), 'step_name': step_name},
            message=line.strip(),
        )

    # 安全点修复总耗时 (新增)
    m = RE_BSR_SAFE_POINT_TOTAL_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='safe_point_total',
            success=True,
            elapsed_seconds=float(m.group(1)), step=int(m.group(2)),
            message=line.strip(),
        )

    # 异步专家恢复提交 (新增)
    m = RE_BSR_ASYNC_EXPERT_SUBMIT_TIMING.search(line)
    if m:
        return BSRTimingRecord(
            timestamp=ts_str, datetime=ts_dt, phase='expert_restore',
            success=True,
            elapsed_seconds=float(m.group(2)), step=int(m.group(3)),
            details={'async': True, 'num_requests': int(m.group(1))},
            message=line.strip(),
        )

    return None


# ============================================================
# 4. 主解析函数
# ============================================================

def parse_log(filepath: str, iter_range: Optional[Tuple[int, int]] = None):
    """解析 Megatron + BSR-MoE 训练日志文件。

    Returns:
        timers:         Dict[str, TimerRecord]
        iterations:     List[IterationRecord]
        bsr_events:     List[BSREvent]
        meta:           dict
        bsr_timings:    List[BSRTimingRecord]
    """
    timers: Dict[str, TimerRecord] = {}
    iterations: List[IterationRecord] = []
    bsr_events: List[BSREvent] = []
    bsr_timings: List[BSRTimingRecord] = []
    meta = {
        'total_iterations': 0,
        'max_iterations': 0,
        'checkpoint_saves': 0,
        'checkpoint_loads': 0,
        'eval_count': 0,
        'bsr_enabled': False,
        'bsr_event_count': 0,
        'bsr_fault_count': 0,
        'bsr_error_count': 0,
        'bsr_sanitize_count': 0,
        'bsr_sanitize_total_tokens': 0,
    }

    in_timer_block = False
    # 维护最近的时间戳，用于没有时间戳的 BSR 行
    last_timestamp_str = ''
    last_timestamp_dt = None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')

            # --- 1. iteration 行 ---
            m = RE_ITERATION.search(line)
            if m:
                in_timer_block = False
                ts = m.group(1)
                it, total, elapsed = int(m.group(2)), int(m.group(3)), float(m.group(4))
                meta['max_iterations'] = max(meta['max_iterations'], total)
                meta['total_iterations'] = max(meta['total_iterations'], it)

                # 更新最近时间戳
                last_timestamp_str = ts
                last_timestamp_dt = _parse_timestamp(ts)

                if iter_range and not (iter_range[0] <= it <= iter_range[1]):
                    continue

                rec = IterationRecord(iteration=it, elapsed_ms=elapsed, timestamp=ts)
                m_loss = RE_LOSS.search(line)
                if m_loss:
                    rec.loss = float(m_loss.group(1))
                m_lr = RE_LR.search(line)
                if m_lr:
                    rec.lr = float(m_lr.group(1))
                iterations.append(rec)
                _add_timer(timers, 'iteration-elapsed', elapsed)
                continue

            # --- 2. timer 块头 ---
            if RE_TIMER_HEADER.search(line):
                in_timer_block = True
                continue

            # --- 3. timer 数据行 ---
            if in_timer_block:
                m = RE_TIMER_LINE.match(line)
                if m:
                    name = m.group(1)
                    if m.group(4):
                        val = float(m.group(4))
                    else:
                        val = float(m.group(3))
                    _add_timer(timers, name, val)
                    continue
                else:
                    if line.strip():
                        in_timer_block = False

            # --- 4. checkpoint ---
            if RE_CKPT_SAVE.search(line):
                meta['checkpoint_saves'] += 1
            if RE_CKPT_LOAD.search(line):
                meta['checkpoint_loads'] += 1

            # --- 5. evaluate ---
            if RE_EVAL_START.search(line):
                meta['eval_count'] += 1

            # --- 6. BSR-MoE 事件 ---
            if RE_BSR_GENERIC.search(line):
                meta['bsr_enabled'] = True
                meta['bsr_event_count'] += 1

                # 提取时间戳 (优先从行内提取，否则使用最近的时间戳)
                ts_match = RE_TIMESTAMP.search(line)
                ts_str = ts_match.group(1) if ts_match else ''
                ts_dt = _parse_timestamp(ts_str) if ts_str else None

                # 如果行内有时间戳，更新 last_timestamp
                if ts_dt:
                    last_timestamp_str = ts_str
                    last_timestamp_dt = ts_dt
                else:
                    # 使用最近的时间戳作为 fallback
                    ts_str = last_timestamp_str
                    ts_dt = last_timestamp_dt

                # 提取 rank
                rank_match = RE_RANK.search(line)
                rank = int(rank_match.group(1)) if rank_match else -1

                # 提取日志级别
                level_match = RE_LOG_LEVEL.search(line)
                level = level_match.group(1) if level_match else 'INFO'

                # 分类事件
                event_type, details = _classify_bsr_event(line)
                step = details.get('step', _extract_step_from_line(line))

                evt = BSREvent(
                    timestamp=ts_str,
                    datetime=ts_dt,
                    rank=rank,
                    level=level,
                    event_type=event_type,
                    step=step,
                    message=line.strip(),
                    details=details,
                )
                bsr_events.append(evt)

                # 统计
                if event_type == 'fault_inject':
                    meta['bsr_fault_count'] += 1
                if event_type == 'callback_error' or level == 'ERROR':
                    meta['bsr_error_count'] += 1
                if event_type == 'sanitize' and 'zeroed_tokens' in details:
                    meta['bsr_sanitize_count'] += 1
                    meta['bsr_sanitize_total_tokens'] += details['zeroed_tokens']

                # 尝试提取计时信息
                timing = _parse_bsr_timing(line)
                if timing:
                    bsr_timings.append(timing)

    return timers, iterations, bsr_events, meta, bsr_timings


def _add_timer(timers: Dict[str, TimerRecord], name: str, value_ms: float):
    """向 timers 字典中添加一条记录。"""
    if name not in timers:
        timers[name] = TimerRecord(name=name, category=classify_timer(name))
    rec = timers[name]
    rec.total_ms += value_ms
    rec.count += 1
    rec.values.append(value_ms)


# ============================================================
# 5. 统计计算
# ============================================================

def percentile(sorted_vals: List[float], p: float) -> float:
    """计算百分位数 (0-100)。"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def compute_iter_stats(iterations: List[IterationRecord]) -> dict:
    """计算 iteration 耗时统计。"""
    if not iterations:
        return {}
    vals = sorted([it.elapsed_ms for it in iterations])
    return {
        'count': len(vals),
        'min': vals[0],
        'max': vals[-1],
        'mean': sum(vals) / len(vals),
        'p50': percentile(vals, 50),
        'p95': percentile(vals, 95),
        'p99': percentile(vals, 99),
        'total_ms': sum(vals),
    }


def compute_category_summary(timers: Dict[str, TimerRecord]) -> Dict[str, float]:
    """按类别汇总总耗时 (ms)。"""
    cats: Dict[str, float] = defaultdict(float)
    for rec in timers.values():
        if rec.name == 'iteration-elapsed':
            continue
        cats[rec.category] += rec.total_ms
    return dict(cats)


def compute_bsr_recovery_timeline(bsr_events: List[BSREvent]) -> List[Dict]:
    """从 BSR 事件中提取故障→恢复的完整时间线。

    每次故障注入到重新集成完成为一个 recovery cycle。
    多 rank 日志去重：同一 (step, event_type, target_rank) 的事件只计一次。
    这样即使不同 rank 在同一 step 出故障也能正确区分。
    """
    # 第一步：按 (step, event_type, target_rank) 去重，保留第一个有时间戳的事件
    seen_keys = set()
    deduped_events = []
    for evt in bsr_events:
        # 对于恢复时间线关键事件，去重
        if evt.event_type in ('fault_inject', 'quarantine', 'replacement',
                              'controller_phase', 'safe_point_repair',
                              'reintegration', 'quarantine_lift'):
            step = evt.details.get('step', evt.step)
            sub = evt.details.get('sub', '')
            # 提取涉及的 rank（故障目标 rank）
            target_rank = (
                evt.details.get('target_rank',
                evt.details.get('failed_rank',
                evt.details.get('rank', -1)))
            )
            key = (step, evt.event_type, sub, target_rank)
            if key in seen_keys:
                continue
            seen_keys.add(key)
        deduped_events.append(evt)

    cycles = []
    current_cycle = None

    for evt in deduped_events:
        if evt.event_type == 'fault_inject':
            current_cycle = {
                'fault_time': evt.datetime,
                'fault_step': evt.details.get('step', evt.step),
                'fault_type': evt.details.get('fault_type', ''),
                'target_rank': evt.details.get('target_rank', -1),
                'quarantine_time': None,
                'quarantine_step': -1,
                'replacement_time': None,
                'replacement_step': -1,
                'safe_point_time': None,
                'safe_point_step': -1,
                'reintegration_time': None,
                'reintegration_step': -1,
                'phase_transitions': [],
                'sanitize_count': 0,
                'sanitize_tokens': 0,
                'errors': [],
            }
            cycles.append(current_cycle)

        if current_cycle is None:
            continue

        if evt.event_type == 'quarantine' and current_cycle['quarantine_time'] is None:
            current_cycle['quarantine_time'] = evt.datetime
            current_cycle['quarantine_step'] = evt.details.get('step', evt.step)

        elif evt.event_type == 'replacement' and current_cycle['replacement_time'] is None:
            current_cycle['replacement_time'] = evt.datetime
            current_cycle['replacement_step'] = evt.details.get('step', evt.step)

        elif evt.event_type == 'safe_point_repair' and current_cycle['safe_point_time'] is None:
            current_cycle['safe_point_time'] = evt.datetime
            current_cycle['safe_point_step'] = evt.details.get('step', evt.step)

        elif evt.event_type == 'reintegration':
            current_cycle['reintegration_time'] = evt.datetime
            current_cycle['reintegration_step'] = evt.details.get('step', evt.step)
            current_cycle = None  # cycle 结束

        elif evt.event_type == 'controller_phase':
            current_cycle['phase_transitions'].append({
                'from': evt.details.get('from_phase', ''),
                'to': evt.details.get('to_phase', ''),
                'event': evt.details.get('event', ''),
                'step': evt.details.get('step', evt.step),
                'time': evt.datetime,
            })

        elif evt.event_type == 'sanitize' and 'zeroed_tokens' in evt.details:
            current_cycle['sanitize_count'] += 1
            current_cycle['sanitize_tokens'] += evt.details['zeroed_tokens']

        elif evt.event_type == 'callback_error':
            current_cycle['errors'].append(evt.message)

    return cycles


def compute_bsr_event_summary(bsr_events: List[BSREvent]) -> Dict[str, int]:
    """按事件类型统计 BSR 事件数量。"""
    summary: Dict[str, int] = defaultdict(int)
    for evt in bsr_events:
        summary[evt.event_type] += 1
    return dict(summary)


def compute_bsr_event_summary_deduped(bsr_events: List[BSREvent]) -> Dict[str, int]:
    """按事件类型统计 BSR 事件数量（多 rank 去重）。

    对于关键事件（fault_inject, quarantine, controller_phase, replacement 等），
    同一 step 的同类事件只计一次。state_transition 按 (layer, step) 去重。
    """
    summary: Dict[str, int] = defaultdict(int)
    seen: Dict[str, set] = defaultdict(set)

    for evt in bsr_events:
        etype = evt.event_type
        step = evt.details.get('step', evt.step)

        if etype in ('fault_inject', 'quarantine', 'quarantine_lift',
                     'controller_phase', 'safe_point_repair', 'reintegration'):
            sub = evt.details.get('sub', '')
            key = (step, sub)
            if key in seen[etype]:
                continue
            seen[etype].add(key)
        elif etype == 'state_transition':
            layer = evt.details.get('layer', -1)
            new_state = evt.details.get('new_state', '')
            key = (step, layer, new_state)
            if key in seen[etype]:
                continue
            seen[etype].add(key)
        elif etype == 'replacement':
            sub = evt.details.get('sub', '')
            key = (step, sub)
            if key in seen[etype]:
                continue
            seen[etype].add(key)
        elif etype == 'checkpoint_meta':
            sub = evt.details.get('sub', '')
            iteration = evt.details.get('iteration', -1)
            key = (iteration, sub)
            if key in seen[etype]:
                continue
            seen[etype].add(key)
        elif etype == 'init':
            sub = evt.details.get('sub', '')
            key = sub
            if key in seen[etype]:
                continue
            seen[etype].add(key)

        summary[etype] += 1

    return dict(summary)


def compute_bsr_sanitize_per_iter(
    bsr_events: List[BSREvent], iterations: List[IterationRecord]
) -> Dict[int, int]:
    """统计每个 iteration 的 sanitize 次数（用于检测 router 是否真正隔离）。"""
    sanitize_by_step: Dict[int, int] = defaultdict(int)
    for evt in bsr_events:
        if evt.event_type == 'sanitize' and 'zeroed_tokens' in evt.details:
            if evt.step >= 0:
                sanitize_by_step[evt.step] += 1
    return dict(sanitize_by_step)


def compute_bsr_timing_stats(bsr_timings: List[BSRTimingRecord]) -> Dict[str, Dict]:
    """按阶段统计 BSR 操作耗时。

    Returns:
        Dict[phase, {count, total_s, avg_s, min_s, max_s, success_count, fail_count, values}]
    """
    phase_data: Dict[str, List[BSRTimingRecord]] = defaultdict(list)
    for t in bsr_timings:
        phase_data[t.phase].append(t)

    stats = {}
    for phase, records in phase_data.items():
        vals = [r.elapsed_seconds for r in records]
        sorted_vals = sorted(vals)
        success_count = sum(1 for r in records if r.success)
        fail_count = len(records) - success_count
        stats[phase] = {
            'count': len(records),
            'total_s': sum(vals),
            'avg_s': sum(vals) / len(vals) if vals else 0.0,
            'min_s': sorted_vals[0] if sorted_vals else 0.0,
            'max_s': sorted_vals[-1] if sorted_vals else 0.0,
            'p50_s': percentile(sorted_vals, 50) if sorted_vals else 0.0,
            'p95_s': percentile(sorted_vals, 95) if sorted_vals else 0.0,
            'success_count': success_count,
            'fail_count': fail_count,
            'values': sorted_vals,
        }
    return stats


def compute_bsr_total_repair_time(bsr_timings: List[BSRTimingRecord]) -> Dict:
    """计算单次安全点修复的总耗时（组修复 + 稠密同步 + 专家恢复 + pipeline 修复）。

    按时间窗口聚合：如果多个操作的时间戳在 5 秒内，认为属于同一次修复。
    """
    repair_phases = ['group_repair', 'dense_sync', 'expert_restore',
                     'pipeline_repair', 'pipeline_stage_repair']
    repair_timings = [t for t in bsr_timings if t.phase in repair_phases and t.datetime]

    if not repair_timings:
        return {'repairs': [], 'total_repair_time_s': 0.0}

    # 按时间排序
    repair_timings.sort(key=lambda t: t.datetime)

    repairs = []
    current_repair = {'start': repair_timings[0].datetime, 'phases': [], 'total_s': 0.0}

    for t in repair_timings:
        if current_repair['phases']:
            last_time = current_repair['phases'][-1].datetime
            gap = (t.datetime - last_time).total_seconds() if last_time else 0
            if gap > 30:  # 超过 30 秒认为是新的修复周期
                repairs.append(current_repair)
                current_repair = {'start': t.datetime, 'phases': [], 'total_s': 0.0}
        current_repair['phases'].append(t)
        current_repair['total_s'] += t.elapsed_seconds

    repairs.append(current_repair)

    total_time = sum(r['total_s'] for r in repairs)
    return {'repairs': repairs, 'total_repair_time_s': total_time}


# ============================================================
# 6. 输出格式化
# ============================================================

CATEGORY_ORDER = ['compute', 'comm', 'optimizer', 'data', 'checkpoint', 'eval', 'setup', 'other']
CATEGORY_LABELS = {
    'compute':    '计算 (fwd/bwd)',
    'comm':       '通信 (allreduce/p2p)',
    'optimizer':  '优化器',
    'data':       '数据加载',
    'checkpoint': 'Checkpoint',
    'eval':       '评估/验证',
    'setup':      '初始化',
    'other':      '其他',
}


def _fmt_duration(dt1: Optional[datetime], dt2: Optional[datetime]) -> str:
    """格式化两个时间点之间的耗时。"""
    if dt1 is None or dt2 is None:
        return 'N/A'
    delta = dt2 - dt1
    secs = delta.total_seconds()
    if secs < 0:
        return 'N/A'
    if secs < 60:
        return f'{secs:.1f}s'
    mins = secs / 60
    return f'{mins:.1f}min'


def format_report(timers, iterations, bsr_events, meta, top_n=30, bsr_timings=None):
    """生成文本报告。"""
    lines = []
    sep = '=' * 88

    # --- 标题 ---
    lines.append(sep)
    lines.append('  Megatron + BSR-MoE 训练日志分析报告')
    lines.append(sep)
    lines.append('')

    # --- 元信息 ---
    lines.append(f'  总 iteration 数:  {meta["total_iterations"]} / {meta["max_iterations"]}')
    lines.append(f'  Checkpoint 保存:  {meta["checkpoint_saves"]} 次')
    lines.append(f'  Checkpoint 加载:  {meta["checkpoint_loads"]} 次')
    lines.append(f'  评估次数:         {meta["eval_count"]} 次')
    lines.append(f'  解析到的 timer:   {len(timers)} 个')
    bsr_str = '已启用' if meta['bsr_enabled'] else '未检测到'
    lines.append(f'  BSR-MoE:          {bsr_str} ({meta["bsr_event_count"]} 条事件)')
    lines.append('')

    # --- Iteration 耗时统计 ---
    stats = compute_iter_stats(iterations)
    if stats:
        lines.append('-' * 88)
        lines.append('  Iteration 耗时统计 (ms)')
        lines.append('-' * 88)
        lines.append(f'    样本数:  {stats["count"]}')
        lines.append(f'    总耗时:  {stats["total_ms"]:,.1f} ms  ({stats["total_ms"]/1000:.1f} s)')
        lines.append(f'    最小值:  {stats["min"]:.1f}')
        lines.append(f'    最大值:  {stats["max"]:.1f}')
        lines.append(f'    平均值:  {stats["mean"]:.1f}')
        lines.append(f'    P50:     {stats["p50"]:.1f}')
        lines.append(f'    P95:     {stats["p95"]:.1f}')
        lines.append(f'    P99:     {stats["p99"]:.1f}')

        threshold = stats['p95'] * 2
        outliers = [it for it in iterations if it.elapsed_ms > threshold]
        if outliers:
            lines.append(f'    异常 iteration (>{threshold:.0f}ms):')
            for it in outliers[:10]:
                lines.append(f'      iter {it.iteration}: {it.elapsed_ms:.1f} ms  [{it.timestamp}]')
            if len(outliers) > 10:
                lines.append(f'      ... 共 {len(outliers)} 个')
        lines.append('')

    # --- 按类别汇总 ---
    cat_summary = compute_category_summary(timers)
    cat_total = sum(cat_summary.values()) or 1.0
    lines.append('-' * 88)
    lines.append('  按类别汇总 (基于 timer 累计值)')
    lines.append('-' * 88)
    lines.append(f'  {"类别":<28s} {"总耗时(ms)":>14s} {"占比":>8s}')
    lines.append(f'  {"─"*28} {"─"*14} {"─"*8}')
    for cat in CATEGORY_ORDER:
        if cat in cat_summary:
            ms = cat_summary[cat]
            pct = ms / cat_total * 100
            label = CATEGORY_LABELS.get(cat, cat)
            lines.append(f'  {label:<28s} {ms:>14,.1f} {pct:>7.1f}%')
    lines.append(f'  {"─"*28} {"─"*14} {"─"*8}')
    lines.append(f'  {"合计":<28s} {cat_total:>14,.1f} {"100.0%":>8s}')
    lines.append('')

    # --- 详细 timer 列表 ---
    sorted_timers = sorted(
        [r for r in timers.values() if r.name != 'iteration-elapsed'],
        key=lambda r: r.total_ms,
        reverse=True,
    )
    lines.append('-' * 88)
    lines.append(f'  详细 Timer 列表 (Top {min(top_n, len(sorted_timers))})')
    lines.append('-' * 88)
    lines.append(f'  {"Timer 名称":<50s} {"类别":<12s} {"次数":>5s} {"总耗时(ms)":>14s} {"平均(ms)":>10s} {"占比":>7s}')
    lines.append(f'  {"─"*50} {"─"*12} {"─"*5} {"─"*14} {"─"*10} {"─"*7}')
    for rec in sorted_timers[:top_n]:
        pct = rec.total_ms / cat_total * 100 if cat_total else 0
        lines.append(
            f'  {rec.name:<50s} {rec.category:<12s} {rec.count:>5d} '
            f'{rec.total_ms:>14,.1f} {rec.avg_ms:>10,.1f} {pct:>6.1f}%'
        )
    if len(sorted_timers) > top_n:
        lines.append(f'  ... 还有 {len(sorted_timers) - top_n} 个 timer 未显示')
    lines.append('')

    # --- Loss 趋势 ---
    if iterations:
        first = iterations[0]
        last = iterations[-1]
        lines.append('-' * 88)
        lines.append('  训练概览')
        lines.append('-' * 88)
        lines.append(f'    首次 iteration: {first.iteration}  loss={first.loss:.6f}  lr={first.lr:.2e}')
        lines.append(f'    末次 iteration: {last.iteration}  loss={last.loss:.6f}  lr={last.lr:.2e}')
        if first.loss > 0 and last.loss > 0:
            lines.append(f'    Loss 变化:      {first.loss:.4f} → {last.loss:.4f}  '
                         f'({"↓" if last.loss < first.loss else "↑"} '
                         f'{abs(last.loss - first.loss):.4f})')
        lines.append('')

    # ============================================================
    # BSR-MoE 分析报告
    # ============================================================
    if meta['bsr_enabled'] and bsr_events:
        lines.append(sep)
        lines.append('  BSR-MoE 故障恢复分析')
        lines.append(sep)
        lines.append('')

        # --- 事件统计 ---
        evt_summary = compute_bsr_event_summary_deduped(bsr_events)
        evt_summary_raw = compute_bsr_event_summary(bsr_events)
        lines.append('-' * 88)
        lines.append('  BSR-MoE 事件统计 (多 rank 去重后)')
        lines.append('-' * 88)
        lines.append(f'  {"事件类型":<24s} {"中文名称":<20s} {"去重后":>6s} {"原始":>6s}')
        lines.append(f'  {"─"*24} {"─"*20} {"─"*6} {"─"*6}')
        for etype in sorted(evt_summary.keys(), key=lambda x: evt_summary[x], reverse=True):
            label = BSR_EVENT_TYPES.get(etype, etype)
            raw = evt_summary_raw.get(etype, 0)
            lines.append(f'  {etype:<24s} {label:<20s} {evt_summary[etype]:>6d} {raw:>6d}')
        lines.append(f'  {"─"*24} {"─"*20} {"─"*6} {"─"*6}')
        lines.append(f'  {"合计":<24s} {"":20s} {sum(evt_summary.values()):>6d} {sum(evt_summary_raw.values()):>6d}')
        lines.append('')

        # --- Sanitize 分析 ---
        if meta['bsr_sanitize_count'] > 0:
            lines.append('-' * 88)
            lines.append('  Dispatch Sanitize 分析 (router 未隔离的 token 被 dispatcher 清零)')
            lines.append('-' * 88)
            lines.append(f'    触发次数:        {meta["bsr_sanitize_count"]}')
            lines.append(f'    清零 token 总数:  {meta["bsr_sanitize_total_tokens"]}')
            avg_tokens = meta['bsr_sanitize_total_tokens'] / meta['bsr_sanitize_count']
            lines.append(f'    平均每次清零:    {avg_tokens:.0f} tokens')

            sanitize_per_iter = compute_bsr_sanitize_per_iter(bsr_events, iterations)
            if sanitize_per_iter:
                first_iter = min(sanitize_per_iter.keys())
                last_iter = max(sanitize_per_iter.keys())
                lines.append(f'    影响 iteration:  {first_iter} ~ {last_iter} '
                             f'(共 {len(sanitize_per_iter)} 个 iteration)')
                if meta['bsr_sanitize_count'] > 10:
                    lines.append(f'    *** 注意: sanitize 频繁触发说明 router 层的 health mask 可能未正确生效 ***')
            lines.append('')

        # --- BSR 操作计时分析 ---
        if bsr_timings:
            timing_stats = compute_bsr_timing_stats(bsr_timings)
            if timing_stats:
                lines.append('-' * 88)
                lines.append('  BSR-MoE 操作计时分析')
                lines.append('-' * 88)
                lines.append(f'  {"操作阶段":<24s} {"中文名称":<18s} {"次数":>4s} {"成功":>4s} {"失败":>4s} '
                             f'{"总耗时(s)":>10s} {"平均(s)":>8s} {"最小(s)":>8s} {"最大(s)":>8s} {"P50(s)":>8s} {"P95(s)":>8s}')
                lines.append(f'  {"─"*24} {"─"*18} {"─"*4} {"─"*4} {"─"*4} '
                             f'{"─"*10} {"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*8}')

                total_time = 0.0
                total_count = 0
                for phase in ['group_repair', 'dense_sync', 'expert_restore',
                              'pipeline_repair', 'pipeline_stage_repair',
                              'pipeline_rollback', 'async_worker', 'deferred_optim',
                              'safe_point_total', 'unknown_timed']:
                    if phase not in timing_stats:
                        continue
                    s = timing_stats[phase]
                    label = BSR_TIMING_PHASES.get(phase, phase)
                    lines.append(
                        f'  {phase:<24s} {label:<18s} {s["count"]:>4d} {s["success_count"]:>4d} {s["fail_count"]:>4d} '
                        f'{s["total_s"]:>10.2f} {s["avg_s"]:>8.3f} {s["min_s"]:>8.3f} {s["max_s"]:>8.3f} '
                        f'{s["p50_s"]:>8.3f} {s["p95_s"]:>8.3f}'
                    )
                    total_time += s['total_s']
                    total_count += s['count']

                lines.append(f'  {"─"*24} {"─"*18} {"─"*4} {"─"*4} {"─"*4} '
                             f'{"─"*10} {"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*8}')
                lines.append(f'  {"合计":<24s} {"":18s} {total_count:>4d} {"":4s} {"":4s} '
                             f'{total_time:>10.2f}')
                lines.append('')

                # 修复周期耗时汇总
                repair_info = compute_bsr_total_repair_time(bsr_timings)
                if repair_info['repairs']:
                    lines.append(f'  ── 安全点修复周期耗时 ({len(repair_info["repairs"])} 次修复) ──')
                    for i, repair in enumerate(repair_info['repairs']):
                        phase_names = [BSR_TIMING_PHASES.get(p.phase, p.phase) for p in repair['phases']]
                        phase_times = [f'{p.elapsed_seconds:.2f}s' for p in repair['phases']]
                        lines.append(f'    修复 #{i+1}: 总耗时 {repair["total_s"]:.2f}s')
                        for p in repair['phases']:
                            label = BSR_TIMING_PHASES.get(p.phase, p.phase)
                            status = '✓' if p.success else '✗'
                            lines.append(f'      {status} {label:<18s} {p.elapsed_seconds:.3f}s')
                    lines.append(f'    ── 所有修复总耗时: {repair_info["total_repair_time_s"]:.2f}s ──')
                    lines.append('')

                # 异步恢复和延迟优化器加载详情
                async_timings = [t for t in bsr_timings if t.phase == 'async_worker']
                deferred_timings = [t for t in bsr_timings if t.phase == 'deferred_optim']

                if async_timings:
                    lines.append(f'  ── 异步恢复 Worker 详情 ({len(async_timings)} 次) ──')
                    for t in async_timings[:20]:
                        req_type = t.details.get('request_type', '?')
                        layer = t.details.get('layer', '?')
                        expert = t.details.get('expert', '?')
                        status = '✓' if t.success else '✗'
                        lines.append(f'    {status} {req_type} layer={layer} expert={expert} '
                                     f'{t.elapsed_seconds:.3f}s [{t.timestamp}]')
                    if len(async_timings) > 20:
                        lines.append(f'    ... 共 {len(async_timings)} 次 (仅显示前 20 次)')
                    lines.append('')

                if deferred_timings:
                    lines.append(f'  ── 延迟优化器加载详情 ({len(deferred_timings)} 次) ──')
                    for t in deferred_timings[:20]:
                        layer = t.details.get('layer', '?')
                        expert = t.details.get('expert', '?')
                        lines.append(f'    layer={layer} expert={expert} step={t.step} '
                                     f'{t.elapsed_seconds:.3f}s [{t.timestamp}]')
                    if len(deferred_timings) > 20:
                        lines.append(f'    ... 共 {len(deferred_timings)} 次 (仅显示前 20 次)')
                    lines.append('')

        # --- 错误汇总 ---
        if meta['bsr_error_count'] > 0:
            lines.append('-' * 88)
            lines.append(f'  BSR-MoE 错误 ({meta["bsr_error_count"]} 条)')
            lines.append('-' * 88)
            error_events = [e for e in bsr_events if e.event_type == 'callback_error' or e.level == 'ERROR']
            for evt in error_events[:20]:
                ts = evt.timestamp or '?'
                lines.append(f'    [{ts}] {evt.message[:120]}')
            if len(error_events) > 20:
                lines.append(f'    ... 共 {len(error_events)} 条错误')
            lines.append('')

        # --- 恢复时间线 ---
        cycles = compute_bsr_recovery_timeline(bsr_events)
        if cycles:
            lines.append('-' * 88)
            lines.append(f'  BSR-MoE 故障恢复时间线 ({len(cycles)} 次恢复)')
            lines.append('-' * 88)
            for i, cyc in enumerate(cycles):
                lines.append(f'')
                lines.append(f'  ── 恢复 #{i+1} ──')
                lines.append(f'    故障类型:     {cyc["fault_type"]}')
                lines.append(f'    目标 Rank:    {cyc["target_rank"]}')
                lines.append(f'    故障 Step:    {cyc["fault_step"]}')
                if cyc['quarantine_step'] >= 0:
                    lines.append(f'    隔离 Step:    {cyc["quarantine_step"]}')
                if cyc['replacement_step'] >= 0:
                    lines.append(f'    替换 Step:    {cyc["replacement_step"]}')
                if cyc['safe_point_step'] >= 0:
                    lines.append(f'    修复 Step:    {cyc["safe_point_step"]}')
                if cyc['reintegration_step'] >= 0:
                    lines.append(f'    集成 Step:    {cyc["reintegration_step"]}')

                # 耗时分析
                lines.append(f'    ── 耗时分析 ──')
                ft = cyc['fault_time']
                qt = cyc['quarantine_time']
                rt = cyc['replacement_time']
                st = cyc['safe_point_time']
                it = cyc['reintegration_time']

                lines.append(f'      故障→隔离:    {_fmt_duration(ft, qt)}')
                lines.append(f'      隔离→替换:    {_fmt_duration(qt, rt)}')
                lines.append(f'      替换→修复:    {_fmt_duration(rt, st)}')
                lines.append(f'      修复→集成:    {_fmt_duration(st, it)}')
                lines.append(f'      总恢复耗时:   {_fmt_duration(ft, it)}')

                # Step 跨度
                if cyc['fault_step'] >= 0 and cyc['reintegration_step'] >= 0:
                    span = cyc['reintegration_step'] - cyc['fault_step']
                    lines.append(f'      降级 iteration 数: {span}')

                if cyc['sanitize_count'] > 0:
                    lines.append(f'      Sanitize 触发:  {cyc["sanitize_count"]} 次 '
                                 f'({cyc["sanitize_tokens"]} tokens)')

                if cyc['errors']:
                    lines.append(f'      错误数:        {len(cyc["errors"])}')

                # 阶段迁移
                if cyc['phase_transitions']:
                    lines.append(f'    ── 控制器阶段迁移 ──')
                    for pt in cyc['phase_transitions']:
                        lines.append(f'      step {pt["step"]}: {pt["from"]} → {pt["to"]} ({pt["event"]})')

            lines.append('')

        # --- BSR 事件时间线 (最近 50 条) ---
        lines.append('-' * 88)
        lines.append(f'  BSR-MoE 事件时间线 (最近 {min(50, len(bsr_events))} 条)')
        lines.append('-' * 88)
        for evt in bsr_events[-50:]:
            ts = evt.timestamp or '?'
            rank_str = f'rank{evt.rank}' if evt.rank >= 0 else '     '
            step_str = f'step={evt.step:>5d}' if evt.step >= 0 else '          '
            label = BSR_EVENT_TYPES.get(evt.event_type, evt.event_type)
            lines.append(f'  [{ts}] {rank_str} {step_str} {label:<16s} | {evt.message[:90]}')
        if len(bsr_events) > 50:
            lines.append(f'  ... 共 {len(bsr_events)} 条事件 (仅显示最近 50 条)')
        lines.append('')

    lines.append(sep)
    return '\n'.join(lines)


def write_csv(filepath: str, timers, iterations, bsr_events, bsr_timings=None):
    """导出 CSV 文件 (含 BSR 事件和计时)。"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        # Sheet 1: timers
        w.writerow(['# Timer 统计'])
        w.writerow(['name', 'category', 'count', 'total_ms', 'avg_ms'])
        for rec in sorted(timers.values(), key=lambda r: r.total_ms, reverse=True):
            if rec.name == 'iteration-elapsed':
                continue
            w.writerow([rec.name, rec.category, rec.count,
                        f'{rec.total_ms:.2f}', f'{rec.avg_ms:.2f}'])
        w.writerow([])

        # Sheet 2: iterations
        w.writerow(['# Iteration 耗时'])
        w.writerow(['iteration', 'elapsed_ms', 'timestamp', 'loss', 'lr'])
        for it in iterations:
            w.writerow([it.iteration, f'{it.elapsed_ms:.1f}', it.timestamp,
                        f'{it.loss:.6f}', f'{it.lr:.2e}'])
        w.writerow([])

        # Sheet 3: BSR events
        if bsr_events:
            w.writerow(['# BSR-MoE 事件'])
            w.writerow(['timestamp', 'rank', 'level', 'event_type', 'step', 'message'])
            for evt in bsr_events:
                w.writerow([evt.timestamp, evt.rank, evt.level,
                            evt.event_type, evt.step, evt.message[:200]])
            w.writerow([])

        # Sheet 4: BSR timings
        if bsr_timings:
            w.writerow(['# BSR-MoE 操作计时'])
            w.writerow(['timestamp', 'phase', 'phase_cn', 'success', 'elapsed_seconds',
                         'step', 'details'])
            for t in bsr_timings:
                label = BSR_TIMING_PHASES.get(t.phase, t.phase)
                w.writerow([t.timestamp, t.phase, label, t.success,
                            f'{t.elapsed_seconds:.4f}', t.step,
                            str(t.details) if t.details else ''])

    print(f'CSV 已保存: {filepath}')


# ============================================================
# 7. 主入口
# ============================================================

def parse_iter_range(s: str) -> Optional[Tuple[int, int]]:
    """解析 '10-100' 格式的 iteration 范围。"""
    if not s:
        return None
    parts = s.split('-')
    if len(parts) != 2:
        print(f'错误: --iter-range 格式应为 START-END，例如 10-100', file=sys.stderr)
        sys.exit(1)
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='Megatron + BSR-MoE 训练日志分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_train_log.py train.log
  python analyze_train_log.py train.log --top 50
  python analyze_train_log.py train.log --csv result.csv
  python analyze_train_log.py train.log --iter-range 10-500
  python analyze_train_log.py train.log --bsr-only
  python analyze_train_log.py train.log --output report.txt
        """,
    )
    parser.add_argument('logfile', help='Megatron 训练日志文件路径')
    parser.add_argument('--top', type=int, default=30, help='显示 Top N 个 timer (默认 30)')
    parser.add_argument('--csv', type=str, default=None, help='导出 CSV 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 txt 报告文件路径 (不指定则自动生成)')
    parser.add_argument('--iter-range', type=str, default=None,
                        help='只分析指定范围的 iteration，格式: START-END')
    parser.add_argument('--bsr-only', action='store_true',
                        help='只输出 BSR-MoE 相关分析')
    parser.add_argument('--no-output-file', action='store_true',
                        help='不自动生成 txt 输出文件')
    args = parser.parse_args()

    iter_range = parse_iter_range(args.iter_range)

    print(f'正在解析: {args.logfile} ...')
    timers, iterations, bsr_events, meta, bsr_timings = parse_log(
        args.logfile, iter_range=iter_range
    )

    if not timers and not iterations and not bsr_events:
        print('未解析到任何数据。请检查日志格式。', file=sys.stderr)
        sys.exit(1)

    report = format_report(
        timers, iterations, bsr_events, meta,
        top_n=args.top, bsr_timings=bsr_timings,
    )

    if args.bsr_only:
        # 只输出 BSR 部分
        bsr_lines = []
        in_bsr = False
        for line in report.split('\n'):
            if 'BSR-MoE 故障恢复分析' in line:
                in_bsr = True
            if in_bsr:
                print(line)
                bsr_lines.append(line)
        report_to_save = '\n'.join(bsr_lines)
    else:
        print(report)
        report_to_save = report

    # --- 自动输出 txt 文件 ---
    if not args.no_output_file:
        import os
        if args.output:
            txt_path = args.output
        else:
            # 自动生成: 与日志文件同目录，后缀改为 _analysis.txt
            log_base = os.path.splitext(args.logfile)[0]
            suffix = '_bsr_analysis.txt' if args.bsr_only else '_analysis.txt'
            txt_path = log_base + suffix

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_to_save)
            f.write('\n')
        print(f'\n报告已保存: {txt_path}')

    # --- CSV 导出 ---
    if args.csv:
        write_csv(args.csv, timers, iterations, bsr_events, bsr_timings=bsr_timings)


if __name__ == '__main__':
    main()
