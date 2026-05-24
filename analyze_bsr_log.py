#!/usr/bin/env python3
"""
Analyze BSR-MoE training log to extract timing statistics for each recovery phase.

Usage:
    python3 analyze_bsr_log.py <log_file>
    python3 analyze_bsr_log.py bsr_per_70.log
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class SafePointRepair:
    """A single safe-point repair cycle (one rank recovery)."""
    step: int
    failed_rank: int
    replacement_rank: int
    path: str  # CHECKPOINT_RESTART or HYBRID_RECOVERY
    total_s: float
    phase_a_s: float
    phase_b_s: float
    phase_c_s: float
    ckpt_restart_fn_s: Optional[float] = None
    convergence_time_s: Optional[float] = None
    timestamp: Optional[str] = None  # when completed


@dataclass
class FaultInjection:
    """A single fault injection event."""
    inject_num: int
    step: int
    fault_type: str
    rank: int
    experts: List[int] = field(default_factory=list)
    timestamp: Optional[str] = None


@dataclass
class IterationRecord:
    """A single training iteration record."""
    step: int
    elapsed_ms: float
    lm_loss: float
    timestamp: Optional[str] = None


def parse_log(filepath: str):
    """Parse the BSR training log and extract structured data."""
    repairs: List[SafePointRepair] = []
    faults: List[FaultInjection] = []
    iterations: List[IterationRecord] = []

    # Regex patterns
    # Safe-point repair completion: Total=171.917s | Breakdown: PhaseA=0.000s, PhaseB=171.916s, PhaseC=0.000s
    RE_REPAIR_COMPLETE = re.compile(
        r'SAFE-POINT REPAIR COMPLETED\s*\(\s*path=(\w+),\s*step=(\d+),\s*failed=(\d+),\s*replacement=(\d+)\s*\)'
        r'\s*\|\s*Total=([\d.]+)s\s*\|\s*Breakdown:\s*PhaseA=([\d.]+)s,\s*PhaseB=([\d.]+)s,\s*PhaseC=([\d.]+)s'
    )

    # checkpoint_restart_fn elapsed
    RE_CKPT_RESTART = re.compile(
        r'checkpoint_restart_fn elapsed=([\d.]+)s\s*\(\s*step=(\d+)'
    )

    # Fault injection
    RE_FAULT_INJECT = re.compile(
        r'FAULT INJECTION #(\d+):\s*type=(\w+),\s*rank=(\d+),\s*step=(\d+),\s*experts=\[([^\]]*)\]'
    )

    # Training iteration
    RE_ITERATION = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+iteration\s+(\d+)/\s*(\d+)\s*\|\s*consumed samples:\s*(\d+)\s*\|\s*elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|\s*learning rate:\s*([\d.E+-]+)\s*\|\s*global batch size:\s*(\d+)\s*\|\s*lm loss:\s*([\d.E+-]+)'
    )

    # Phase B completion with elapsed
    RE_PHASE_B_COMPLETE = re.compile(
        r'\[Phase B\] Parameter recovery COMPLETED\s*\(\s*path=(\w+),\s*elapsed=([\d.]+)s'
    )

    # Convergence completed with total_elapsed
    RE_CONVERGENCE = re.compile(
        r'CONVERGENCE COMPLETED.*total_elapsed=([\d.]+)s'
    )

    # Timestamp recovery from BSR lines  
    RE_TIMESTAMP = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')

    # Checkpoint save lines
    RE_CHECKPOINT_SAVE = re.compile(r'saving checkpoint at iteration\s+(\d+)')

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Parse safe-point repair completions
            m = RE_REPAIR_COMPLETE.search(line)
            if m:
                path, step, failed, replacement, total, pa, pb, pc = m.groups()
                # Extract timestamp
                ts_m = RE_TIMESTAMP.search(line)
                ts = ts_m.group(1) if ts_m else None
                repairs.append(SafePointRepair(
                    step=int(step),
                    failed_rank=int(failed),
                    replacement_rank=int(replacement),
                    path=path,
                    total_s=float(total),
                    phase_a_s=float(pa),
                    phase_b_s=float(pb),
                    phase_c_s=float(pc),
                    timestamp=ts,
                ))

            # Parse checkpoint_restart_fn elapsed
            m = RE_CKPT_RESTART.search(line)
            if m:
                elapsed, step = m.groups()
                # Attach to the most recent repair for this step
                for r in reversed(repairs):
                    if r.step == int(step) and r.ckpt_restart_fn_s is None:
                        r.ckpt_restart_fn_s = float(elapsed)
                        break

            # Parse fault injections (deduplicate — only from rank-0)
            m = RE_FAULT_INJECT.search(line)
            if m:
                num, ftype, rank, step, experts_str = m.groups()
                ts_m = RE_TIMESTAMP.search(line)
                ts = ts_m.group(1) if ts_m else None
                experts = [int(x.strip()) for x in experts_str.split(',') if x.strip()]
                faults.append(FaultInjection(
                    inject_num=int(num),
                    step=int(step),
                    fault_type=ftype,
                    rank=int(rank),
                    experts=experts,
                    timestamp=ts,
                ))

            # Parse training iterations
            m = RE_ITERATION.search(line)
            if m:
                ts, step, total, consumed, elapsed_ms, lr, bs, lm_loss = m.groups()
                iterations.append(IterationRecord(
                    step=int(step),
                    elapsed_ms=float(elapsed_ms),
                    lm_loss=float(lm_loss),
                    timestamp=ts,
                ))

    return repairs, faults, iterations


def print_summary(repairs: List[SafePointRepair], faults: List[FaultInjection], iterations: List[IterationRecord]):
    """Print a comprehensive timing summary."""

    # Deduplicate faults (multiple ranks log same injection)
    seen_faults = set()
    unique_faults = []
    for f in faults:
        key = (f.inject_num, f.step, f.rank)
        if key not in seen_faults:
            seen_faults.add(key)
            unique_faults.append(f)

    # Deduplicate repairs (multiple ranks report same repair)
    seen_repairs = set()
    unique_repairs = []
    for r in repairs:
        key = (r.step, r.failed_rank, r.replacement_rank, r.total_s)
        if key not in seen_repairs:
            seen_repairs.add(key)
            unique_repairs.append(r)

    print("=" * 90)
    print("BSR-MoE 故障恢复日志分析报告")
    print("=" * 90)

    # === 1. Overview ===
    print("\n## 1. 基本概览")
    print(f"  训练迭代总数:       {len(iterations)}")
    if iterations:
        print(f"  迭代范围:           step {iterations[0].step} ~ step {iterations[-1].step}")
    print(f"  故障注入次数 (去重): {len(unique_faults)}")
    print(f"  恢复完成次数 (去重): {len(unique_repairs)}")
    if unique_faults:
        ranks_hit = sorted(set(f.rank for f in unique_faults))
        print(f"  被注入故障的rank:    {ranks_hit}")
        inject_steps = sorted(set(f.step for f in unique_faults))
        print(f"  故障注入step列表:    {inject_steps}")

    # === 2. Fault injection details ===
    print("\n## 2. 故障注入详情")
    print(f"  {'#':<4} {'Step':<7} {'Type':<20} {'Rank':<6} {'Expert数':<10} {'时间戳'}")
    print(f"  {'-'*4} {'-'*7} {'-'*20} {'-'*6} {'-'*10} {'-'*20}")
    for f in unique_faults[:60]:  # limit display
        print(f"  {f.inject_num:<4} {f.step:<7} {f.fault_type:<20} {f.rank:<6} {len(f.experts):<10} {f.timestamp or 'N/A'}")
    if len(unique_faults) > 60:
        print(f"  ... (省略 {len(unique_faults) - 60} 条)")

    # === 3. Recovery timing summary ===
    print("\n## 3. 恢复阶段耗时统计 (基于 SAFE-POINT REPAIR COMPLETED)")
    if not unique_repairs:
        print("  未找到恢复完成记录")
        return

    phase_a = [r.phase_a_s for r in unique_repairs]
    phase_b = [r.phase_b_s for r in unique_repairs]
    phase_c = [r.phase_c_s for r in unique_repairs]
    totals = [r.total_s for r in unique_repairs]
    ckpt_times = [r.ckpt_restart_fn_s for r in unique_repairs if r.ckpt_restart_fn_s is not None]

    def stats(vals):
        if not vals:
            return "N/A"
        return f"min={min(vals):.3f}s  avg={sum(vals)/len(vals):.3f}s  max={max(vals):.3f}s  P50={sorted(vals)[len(vals)//2]:.3f}s"

    print(f"  修复次数:           {len(unique_repairs)}")
    print(f"  恢复路径:           {set(r.path for r in unique_repairs)}")
    print()
    print(f"  Phase A (基础设施修复):  {stats(phase_a)}")
    print(f"  Phase B (参数恢复):      {stats(phase_b)}")
    print(f"  Phase C (修复后处理):    {stats(phase_c)}")
    print(f"  总耗时:                   {stats(totals)}")
    if ckpt_times:
        print(f"  checkpoint_restart_fn:   {stats(ckpt_times)}")

    # === 4. Per-step recovery breakdown (deduplicated — take median per step) ===
    print("\n## 4. 每步恢复详情 (每 step 取所有 rank 中位数)")
    # Group repairs by step, then take median values per step
    import statistics
    repairs_by_step = defaultdict(list)
    for r in unique_repairs:
        repairs_by_step[r.step].append(r)

    print(f"  {'Step':<7} {'Failed':<7} {'Path':<20} {'PhaseA(s)':<11} {'PhaseB(s)':<11} {'PhaseC(s)':<11} {'Total(s)':<11} {'Ranks':<7}")
    print(f"  {'-'*7} {'-'*7} {'-'*20} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*7}")
    per_step_summary = []  # (step, failed_rank, path, median_pa, median_pb, median_pc, median_total, n_ranks)
    for step in sorted(repairs_by_step.keys()):
        rs = repairs_by_step[step]
        n = len(rs)
        pa = statistics.median([r.phase_a_s for r in rs])
        pb = statistics.median([r.phase_b_s for r in rs])
        pc = statistics.median([r.phase_c_s for r in rs])
        tot = statistics.median([r.total_s for r in rs])
        failed = rs[0].failed_rank
        path = rs[0].path
        per_step_summary.append((step, failed, path, pa, pb, pc, tot, n))
        print(f"  {step:<7} {failed:<7} {path:<20} {pa:<11.3f} {pb:<11.3f} {pc:<11.3f} {tot:<11.3f} {n:<7}")

    # === 5. Iteration elapsed time comparison ===
    print("\n## 5. 训练迭代耗时分析")
    if iterations:
        # Find fault injection steps
        fault_steps = set(f.step for f in unique_faults)
        normal_iters = [i for i in iterations if i.step not in fault_steps and i.step - 1 not in fault_steps]
        recovery_iters = [i for i in iterations if i.step in fault_steps or i.step + 1 in fault_steps or i.step - 1 in fault_steps]

        if normal_iters:
            normal_times = [i.elapsed_ms for i in normal_iters]
            avg_normal = sum(normal_times) / len(normal_times)
            print(f"  正常迭代 (去除了故障步附近3步):")
            print(f"    数量: {len(normal_iters)}")
            print(f"    平均耗时: {avg_normal:.1f} ms")
            print(f"    最小/最大: {min(normal_times):.1f} / {max(normal_times):.1f} ms")
            print(f"    P50/P95: {sorted(normal_times)[len(normal_times)//2]:.1f} / {sorted(normal_times)[int(len(normal_times)*0.95)]:.1f} ms")

        # Show iterations around fault injection steps
        print(f"\n  故障注入步附近的迭代耗时:")
        for f_step in sorted(fault_steps):
            nearby = [i for i in iterations if f_step - 5 <= i.step <= f_step + 5]
            for i in nearby:
                marker = " <<< FAULT" if i.step == f_step else ""
                print(f"    step {i.step}: {i.elapsed_ms:.1f} ms (lm_loss={i.lm_loss:.4f}){marker}")
            print()

    # === 6. Recovery overhead estimation ===
    print("\n## 6. 恢复开销估算")
    if unique_repairs and iterations:
        normal_times_ms = [i.elapsed_ms for i in iterations if i.step not in fault_steps]
        if normal_times_ms:
            avg_normal_ms = sum(normal_times_ms) / len(normal_times_ms)
            # Use per-step median for recovery overhead
            total_recovery_s = sum(t[6] for t in per_step_summary)  # sum of median totals
            n_recoveries = len(per_step_summary)
            total_iters = len(iterations)
            total_training_s = sum(i.elapsed_ms for i in iterations) / 1000.0
            total_wall_s = total_training_s + total_recovery_s
            print(f"  平均正常迭代耗时:     {avg_normal_ms:.1f} ms = {avg_normal_ms/1000:.3f} s/iter")
            print(f"  恢复次数 (去重按step): {n_recoveries}")
            print(f"  总恢复时间:          {total_recovery_s:.1f} s = {total_recovery_s/60:.1f} min")
            print(f"  总训练迭代时间:      {total_training_s:.1f} s = {total_training_s/60:.1f} min")
            print(f"  恢复开销占比:        {total_recovery_s/total_wall_s*100:.1f}%")
            print(f"  等价丢失迭代数:      {total_recovery_s/(avg_normal_ms/1000):.1f} iters")
            print(f"  每次恢复平均耗时:    {total_recovery_s/n_recoveries:.1f} s ≈ {total_recovery_s/n_recoveries/(avg_normal_ms/1000):.1f} iters")
            for step, failed, path, pa, pb, pc, tot, n in per_step_summary:
                equiv = tot / (avg_normal_ms / 1000)
                print(f"    step {step}: {tot:.1f}s ≈ {equiv:.1f} iters overhead (path={path}; PhaseA={pa:.1f}s, PhaseB={pb:.1f}s, PhaseC={pc:.1f}s)")

    # === 7. Time between fault injection and recovery completion ===
    print("\n## 7. 故障注入→恢复完成 时间差 (基于时间戳)")
    for f_step in sorted(fault_steps):
        f_ts = None
        for f in unique_faults:
            if f.step == f_step and f.timestamp:
                f_ts = datetime.strptime(f.timestamp, "%Y-%m-%d %H:%M:%S")
                break
        repair = None
        for r in unique_repairs:
            if r.step == f_step and r.timestamp:
                repair = r
                break
        if f_ts and repair:
            r_ts = datetime.strptime(repair.timestamp, "%Y-%m-%d %H:%M:%S")
            delta = (r_ts - f_ts).total_seconds()
            # Find the iteration that was running just before fault
            prev_iters = [i for i in iterations if i.step == f_step]
            iter_time = prev_iters[0].elapsed_ms if prev_iters else 0
            print(f"  step {f_step}: fault@{f.timestamp} → repair_done@{repair.timestamp} = {delta:.1f}s (iter_time={iter_time:.1f}ms)")
        else:
            print(f"  step {f_step}: 缺少时间戳，无法计算")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    repairs, faults, iterations = parse_log(filepath)
    print_summary(repairs, faults, iterations)