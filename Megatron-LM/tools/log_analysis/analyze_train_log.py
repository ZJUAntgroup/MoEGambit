#!/usr/bin/env python3
"""
analyze_train_log.py — Megatron 训练日志耗时分析工具

离线解析 Megatron 训练日志，统计各阶段耗时及占比。

用法:
    python analyze_train_log.py <日志文件>
    python analyze_train_log.py train.log --top 20
    python analyze_train_log.py train.log --csv output.csv
    python analyze_train_log.py train.log --iter-range 10-100

支持的日志格式:
    1. iteration 行:  [datetime] iteration N/M | ... | elapsed time per iteration (ms): XXX | ...
    2. timer 行:      (min, max) time across ranks (ms):
                          timer-name ......: (min, max)
                      或  max time across ranks (ms):
                          timer-name ......: max
    3. checkpoint 行: timers.log(['save-checkpoint']) 输出
    4. evaluate 行:   timers.log(['evaluate']) 输出
    5. setup 行:      model-and-optimizer-setup / train/valid/test-data-iterators-setup

分析输出:
    - 每个 timer 的总耗时、平均耗时、调用次数、占比
    - 按类别 (data/compute/comm/optimizer/checkpoint/eval/other) 汇总
    - iteration 耗时统计 (min/max/mean/p50/p95/p99)
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. Timer 分类规则
# ============================================================

# 每个 timer 名称 → 类别。优先精确匹配，再用前缀/关键字匹配。
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
    # setup (归入 other)
    'model-and-optimizer-setup': 'setup',
}

# 关键字 fallback 规则 (按优先级)
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
# 3. 日志解析
# ============================================================

# --- iteration 行 ---
# [2026-04-09 14:33:50] iteration 2/ 600 | ... | elapsed time per iteration (ms): 778.5 | ...
RE_ITERATION = re.compile(
    r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?\s+'
    r'iteration\s+(\d+)\s*/\s*(\d+)\s*\|'
    r'.*?elapsed time per iteration \(ms\):\s*([\d.]+)'
)

# --- loss ---
RE_LOSS = re.compile(r'lm loss:\s*([\d.eE+\-]+)')
# --- lr ---
RE_LR = re.compile(r'learning rate:\s*([\d.eE+\-]+)')

# --- timer 块头 ---
# "(min, max) time across ranks (ms):" 或 "max time across ranks (ms):"
RE_TIMER_HEADER = re.compile(r'(?:(?:\(min,\s*max\)|max)\s+)?time across ranks \(ms\):')

# --- timer 数据行 ---
# "    forward-backward ..............................: (123.45, 678.90)"
# "    forward-backward ..............................: 678.90"
RE_TIMER_LINE = re.compile(
    r'^\s{2,}([\w/\-]+)\s*\.{2,}:\s*'
    r'(?:\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)|([\d.]+))'
)

# --- checkpoint 保存 ---
RE_CKPT_SAVE = re.compile(r'saving checkpoint at iteration\s+(\d+)', re.IGNORECASE)
RE_CKPT_LOAD = re.compile(r'loading checkpoint', re.IGNORECASE)

# --- evaluate ---
RE_EVAL_START = re.compile(r'evaluating', re.IGNORECASE)


def parse_log(filepath: str, iter_range: Optional[Tuple[int, int]] = None):
    """解析 Megatron 训练日志文件。

    Returns:
        timers:     Dict[str, TimerRecord]  — 每个 timer 的统计
        iterations: List[IterationRecord]   — 每次 iteration 的信息
        meta:       dict                    — 元信息 (总 iteration 数等)
    """
    timers: Dict[str, TimerRecord] = {}
    iterations: List[IterationRecord] = []
    meta = {
        'total_iterations': 0,
        'max_iterations': 0,
        'checkpoint_saves': 0,
        'checkpoint_loads': 0,
        'eval_count': 0,
    }

    in_timer_block = False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')

            # --- 1. iteration 行 ---
            m = RE_ITERATION.search(line)
            if m:
                in_timer_block = False
                ts, it, total, elapsed = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
                meta['max_iterations'] = max(meta['max_iterations'], total)
                meta['total_iterations'] = max(meta['total_iterations'], it)

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

                # 把 elapsed time 也记入 timers (作为 "iteration-elapsed" 虚拟 timer)
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
                    if m.group(4):  # max-only 格式
                        val = float(m.group(4))
                    else:           # (min, max) 格式，取 max
                        val = float(m.group(3))
                    _add_timer(timers, name, val)
                    continue
                else:
                    # 非 timer 行，结束 timer 块
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

    return timers, iterations, meta


def _add_timer(timers: Dict[str, TimerRecord], name: str, value_ms: float):
    """向 timers 字典中添加一条记录。"""
    if name not in timers:
        timers[name] = TimerRecord(name=name, category=classify_timer(name))
    rec = timers[name]
    rec.total_ms += value_ms
    rec.count += 1
    rec.values.append(value_ms)


# ============================================================
# 4. 统计计算
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
            continue  # 虚拟 timer，不参与分类汇总
        cats[rec.category] += rec.total_ms
    return dict(cats)


# ============================================================
# 5. 输出格式化
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


def format_report(timers, iterations, meta, top_n=30):
    """生成文本报告。"""
    lines = []
    sep = '=' * 80

    # --- 标题 ---
    lines.append(sep)
    lines.append('  Megatron 训练日志耗时分析报告')
    lines.append(sep)
    lines.append('')

    # --- 元信息 ---
    lines.append(f'  总 iteration 数:  {meta["total_iterations"]} / {meta["max_iterations"]}')
    lines.append(f'  Checkpoint 保存:  {meta["checkpoint_saves"]} 次')
    lines.append(f'  Checkpoint 加载:  {meta["checkpoint_loads"]} 次')
    lines.append(f'  评估次数:         {meta["eval_count"]} 次')
    lines.append(f'  解析到的 timer:   {len(timers)} 个')
    lines.append('')

    # --- Iteration 耗时统计 ---
    stats = compute_iter_stats(iterations)
    if stats:
        lines.append('-' * 80)
        lines.append('  Iteration 耗时统计 (ms)')
        lines.append('-' * 80)
        lines.append(f'    样本数:  {stats["count"]}')
        lines.append(f'    总耗时:  {stats["total_ms"]:,.1f} ms  ({stats["total_ms"]/1000:.1f} s)')
        lines.append(f'    最小值:  {stats["min"]:.1f}')
        lines.append(f'    最大值:  {stats["max"]:.1f}')
        lines.append(f'    平均值:  {stats["mean"]:.1f}')
        lines.append(f'    P50:     {stats["p50"]:.1f}')
        lines.append(f'    P95:     {stats["p95"]:.1f}')
        lines.append(f'    P99:     {stats["p99"]:.1f}')

        # 标记异常 iteration (> P95 * 2)
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
    lines.append('-' * 80)
    lines.append('  按类别汇总 (基于 timer 累计值)')
    lines.append('-' * 80)
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
    lines.append('-' * 80)
    lines.append(f'  详细 Timer 列表 (Top {min(top_n, len(sorted_timers))})')
    lines.append('-' * 80)
    lines.append(f'  {"Timer 名称":<50s} {"类别":<12s} {"次数":>5s} {"总耗时(ms)":>14s} {"平均(ms)":>10s} {"占比":>7s}')
    lines.append(f'  {"─"*50} {"─"*12} {"─"*5} {"─"*14} {"─"*10} {"─"*7}')
    for rec in sorted_timers[:top_n]:
        pct = rec.total_ms / cat_total * 100 if cat_total else 0
        cat_label = rec.category
        lines.append(
            f'  {rec.name:<50s} {cat_label:<12s} {rec.count:>5d} '
            f'{rec.total_ms:>14,.1f} {rec.avg_ms:>10,.1f} {pct:>6.1f}%'
        )
    if len(sorted_timers) > top_n:
        lines.append(f'  ... 还有 {len(sorted_timers) - top_n} 个 timer 未显示')
    lines.append('')

    # --- Loss 趋势 (首尾对比) ---
    if iterations:
        first = iterations[0]
        last = iterations[-1]
        lines.append('-' * 80)
        lines.append('  训练概览')
        lines.append('-' * 80)
        lines.append(f'    首次 iteration: {first.iteration}  loss={first.loss:.6f}  lr={first.lr:.2e}')
        lines.append(f'    末次 iteration: {last.iteration}  loss={last.loss:.6f}  lr={last.lr:.2e}')
        if first.loss > 0 and last.loss > 0:
            lines.append(f'    Loss 变化:      {first.loss:.4f} → {last.loss:.4f}  '
                         f'({"↓" if last.loss < first.loss else "↑"} '
                         f'{abs(last.loss - first.loss):.4f})')
        lines.append('')

    lines.append(sep)
    return '\n'.join(lines)


def write_csv(filepath: str, timers: Dict[str, TimerRecord], iterations: List[IterationRecord]):
    """导出 CSV 文件。"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Sheet 1: timers
        w = csv.writer(f)
        w.writerow(['# Timer 统计'])
        w.writerow(['name', 'category', 'count', 'total_ms', 'avg_ms'])
        for rec in sorted(timers.values(), key=lambda r: r.total_ms, reverse=True):
            if rec.name == 'iteration-elapsed':
                continue
            w.writerow([rec.name, rec.category, rec.count, f'{rec.total_ms:.2f}', f'{rec.avg_ms:.2f}'])
        w.writerow([])
        w.writerow(['# Iteration 耗时'])
        w.writerow(['iteration', 'elapsed_ms', 'timestamp', 'loss', 'lr'])
        for it in iterations:
            w.writerow([it.iteration, f'{it.elapsed_ms:.1f}', it.timestamp,
                        f'{it.loss:.6f}', f'{it.lr:.2e}'])
    print(f'CSV 已保存: {filepath}')


# ============================================================
# 6. 主入口
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
        description='Megatron 训练日志耗时分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_train_log.py train.log
  python analyze_train_log.py train.log --top 50
  python analyze_train_log.py train.log --csv result.csv
  python analyze_train_log.py train.log --iter-range 10-500
        """,
    )
    parser.add_argument('logfile', help='Megatron 训练日志文件路径')
    parser.add_argument('--top', type=int, default=30, help='显示 Top N 个 timer (默认 30)')
    parser.add_argument('--csv', type=str, default=None, help='导出 CSV 文件路径')
    parser.add_argument('--iter-range', type=str, default=None,
                        help='只分析指定范围的 iteration，格式: START-END')
    args = parser.parse_args()

    iter_range = parse_iter_range(args.iter_range)

    print(f'正在解析: {args.logfile} ...')
    timers, iterations, meta = parse_log(args.logfile, iter_range=iter_range)

    if not timers and not iterations:
        print('未解析到任何 timer 或 iteration 数据。请检查日志格式。', file=sys.stderr)
        sys.exit(1)

    report = format_report(timers, iterations, meta, top_n=args.top)
    print(report)

    if args.csv:
        write_csv(args.csv, timers, iterations)


if __name__ == '__main__':
    main()
