#!/usr/bin/env python3
"""BSR-MoE training log timing report.

This parser is intentionally conservative:
- duplicate multi-rank event lines are folded before statistics are computed;
- clean iteration time is estimated from a robust median after excluding warmup,
  skipped iterations, checkpoint iterations, and fault/recovery neighborhoods;
- recovery wall time is reported two ways: controller self-time and observed
  training-iteration overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RE_ITER = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"iteration\s+(?P<iter>\d+)\s*/\s*(?P<target>\d+)\s*\|(?P<body>.*)"
)
RE_ELAPSED = re.compile(r"elapsed time per iteration \(ms\):\s*(?P<v>[0-9.]+)")
RE_SKIPPED = re.compile(r"number of skipped iterations:\s*(?P<v>\d+)")
RE_LOSS = re.compile(r"lm loss:\s*(?P<v>[0-9.eE+-]+)")
RE_GRAD_NORM = re.compile(r"grad norm:\s*(?P<v>[0-9.eE+-]+)")
RE_FAULT = re.compile(
    r"FAULT INJECTION\s+#(?P<num>\d+):\s+type=(?P<kind>[^,]+),\s+rank=(?P<rank>\d+),\s+step=(?P<step>\d+)"
)
RE_NEXT_FAULT = re.compile(r"next periodic fault scheduled at step\s+(?P<step>\d+)\s+\(interval=(?P<interval>\d+)")
RE_REPAIR = re.compile(
    r"SAFE-POINT REPAIR COMPLETED\s+\(path=(?P<path>[^,]+),\s+step=(?P<step>\d+),\s+"
    r"failed=(?P<failed>\d+),\s+replacement=(?P<replacement>\d+)\)\s+\|\s+"
    r"Total=(?P<total>[0-9.]+)s\s+\|\s+Breakdown:\s+"
    r"PhaseA=(?P<a>[0-9.]+)s,\s+PhaseB=(?P<b>[0-9.]+)s,\s+PhaseC=(?P<c>[0-9.]+)s"
)
RE_PHASE = re.compile(r"controller:\s+(?P<name>dense_sync|expert_restore.*?)\s+elapsed=(?P<s>[0-9.]+)s\s+\(step=(?P<step>\d+)\)")
RE_EXPERT_SUCCESS = re.compile(
    r"(?:stale expert restore|expert_restore_fn):\s+SUCCESS.*?restored\s+(?P<done>\d+)(?:/(?P<total>\d+))?\s+experts.*?(?P<s>[0-9.]+)s"
)
RE_DENSE_SUCCESS = re.compile(
    r"dense param sync:\s+SUCCESS.*?synced\s+(?P<params>\d+)\s+params\s+\((?P<scalars>\d+)\s+scalars\).*?"
    r"from\s+(?:single\s+source\s+)?rank\s+(?P<src>\d+).*?(?P<s>[0-9.]+)s"
)
RE_OPT_SUBMIT = re.compile(r"submitted\s+(?P<count>\d+)\s+optimizer\s+state\s+load\s+requests")
RE_SYNC_OPT = re.compile(
    r"synchronous\s+optimizer\s+state\s+load\s+completed.*?"
    r"elapsed=(?P<s>[0-9.]+)s,\s+step=(?P<step>\d+)"
)
RE_OPT_COMPLETE = re.compile(r"async\s+optimizer\s+load\s+completed.*?(?P<count>\d+)\s+requests.*?\(step=(?P<step>\d+)\)")
RE_OPT_FINALIZE = re.compile(r"finalized\s+(?P<count>\d+)\s+expert\s+optimizer\s+loads.*?\(step=(?P<step>\d+)\)")
RE_CKPT_RESTART_TOTAL = re.compile(
    r"checkpoint_restart_fn:.*?TOTAL RECOVERY TIME:\s+(?P<s>[0-9.]+)s\s+"
    r"\(step=(?P<step>\d+),"
)
RE_CKPT_RESTART_LOAD = re.compile(
    r"checkpoint_restart_fn:.*?CHECKPOINT LOADED.*?load_time=(?P<s>[0-9.]+)s"
)
RE_OPT_SKIP = re.compile(r"optimizer\.step\(\)\s+SKIPPED\s+at\s+step\s+(?P<step>\d+)")
RE_ZERO = re.compile(r"restart-in-place:\s+invalidated\s+(?P<params>\d+)\s+params,\s+(?P<states>\d+)\s+optimizer\s+states.*?\(step=(?P<step>\d+)\)")
RE_REINTEGRATED = re.compile(r"REINTEGRATED\s+(?:->|→)\s+HEALTHY_TRAINING.*?step=(?P<step>\d+)")
RE_SAVE_START = re.compile(r"saving checkpoint at iteration\s+(?P<iter>\d+)")
RE_SAVE_DONE = re.compile(r"successfully saved checkpoint from iteration\s+(?P<iter>\d+)")
RE_CKPT_META = re.compile(r"checkpoint metadata injected\s+\(iteration=(?P<iter>\d+),")
RE_LOAD_CKPT = re.compile(r"loading checkpoint(?: state dict)? from\s+(?P<path>\S+)")
RE_STATE_LOAD = re.compile(r"loading checkpoint state dict from\s+(?P<path>\S+)")
RE_TIMER = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.\-/ ]+?)\s+\.{3,}\s*:\s*\((?P<min>[0-9.]+),\s*(?P<max>[0-9.]+)\)")
RE_NO_MATCH = re.compile(r"no matching params found for expert")
RE_DIR_UPDATE_FAIL = re.compile(r"directory update failed for expert")

SEVERE_MARKERS = (
    "Traceback",
    "RuntimeError:",
    "OutOfMemoryError",
    "CUDA out of memory",
    "NCCL error",
    "ncclRemoteError",
    "Connection closed by remote peer",
    "Treating restore as failed",
)


@dataclass
class Iteration:
    step: int
    target: int
    ts: str
    elapsed_ms: float
    skipped: int
    loss: float | None
    grad_norm: float | None
    line: int


@dataclass
class Fault:
    number: int
    step: int
    rank: int
    kind: str
    line: int


@dataclass
class Repair:
    step: int
    failed: int
    replacement: int
    path: str
    total_s: float
    phase_a_s: float
    phase_b_s: float
    phase_c_s: float
    line: int


@dataclass
class Phase:
    step: int
    name: str
    seconds: float
    line: int


@dataclass
class Timer:
    name: str
    min_ms: float
    max_ms: float
    line: int
    step: int | None = None


def parse_optional_float(regex: re.Pattern[str], text: str) -> float | None:
    match = regex.search(text)
    return float(match.group("v")) if match else None


def parse_optional_int(regex: re.Pattern[str], text: str, default: int = 0) -> int:
    match = regex.search(text)
    return int(match.group("v")) if match else default


def pctl(values: Iterable[float], pct: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.median(vals) if vals else None


def stat(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = list(values)
    return {
        "count": len(vals),
        "min": min(vals) if vals else None,
        "mean": statistics.fmean(vals) if vals else None,
        "p50": median(vals),
        "p90": pctl(vals, 90),
        "p95": pctl(vals, 95),
        "p99": pctl(vals, 99),
        "max": max(vals) if vals else None,
    }


def mad_sigma(values: Iterable[float]) -> float:
    vals = list(values)
    center = median(vals)
    if center is None:
        return 0.0
    deviations = [abs(v - center) for v in vals]
    mad = median(deviations)
    if mad and mad > 0:
        return mad * 1.4826
    q25 = pctl(vals, 25)
    q75 = pctl(vals, 75)
    if q25 is not None and q75 is not None and q75 > q25:
        return (q75 - q25) / 1.349
    return 0.0


def unique(items: Iterable[Any], key_fn) -> list[Any]:
    seen = set()
    out = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_log(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "iterations": [],
        "faults": [],
        "next_faults": [],
        "repairs": [],
        "phases": [],
        "expert_success": [],
        "dense_success": [],
        "opt_submit": [],
        "sync_opt_load": [],
        "opt_complete": [],
        "opt_finalize": [],
        "checkpoint_restart_total": [],
        "checkpoint_restart_load": [],
        "opt_skip_steps": [],
        "zero": [],
        "reintegrated_steps": [],
        "save_start_steps": [],
        "save_done_steps": [],
        "checkpoint_meta_steps": [],
        "load_paths": [],
        "state_load_paths": [],
        "timers": [],
        "severe": [],
        "no_match_count": 0,
        "directory_update_fail_count": 0,
        "line_count": 0,
    }
    last_checkpoint_step: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            data["line_count"] = line_no
            text = line.rstrip("\n")

            if any(marker in text for marker in SEVERE_MARKERS):
                data["severe"].append({"line": line_no, "text": text.strip()})
            if RE_NO_MATCH.search(text):
                data["no_match_count"] += 1
            if RE_DIR_UPDATE_FAIL.search(text):
                data["directory_update_fail_count"] += 1

            match = RE_ITER.search(text)
            if match:
                body = match.group("body")
                elapsed = RE_ELAPSED.search(body)
                if elapsed:
                    data["iterations"].append(
                        Iteration(
                            step=int(match.group("iter")),
                            target=int(match.group("target")),
                            ts=match.group("ts"),
                            elapsed_ms=float(elapsed.group("v")),
                            skipped=parse_optional_int(RE_SKIPPED, body),
                            loss=parse_optional_float(RE_LOSS, body),
                            grad_norm=parse_optional_float(RE_GRAD_NORM, body),
                            line=line_no,
                        )
                    )

            match = RE_FAULT.search(text)
            if match:
                data["faults"].append(
                    Fault(
                        number=int(match.group("num")),
                        step=int(match.group("step")),
                        rank=int(match.group("rank")),
                        kind=match.group("kind"),
                        line=line_no,
                    )
                )

            match = RE_NEXT_FAULT.search(text)
            if match:
                data["next_faults"].append(
                    {"step": int(match.group("step")), "interval": int(match.group("interval")), "line": line_no}
                )

            match = RE_REPAIR.search(text)
            if match:
                data["repairs"].append(
                    Repair(
                        step=int(match.group("step")),
                        failed=int(match.group("failed")),
                        replacement=int(match.group("replacement")),
                        path=match.group("path"),
                        total_s=float(match.group("total")),
                        phase_a_s=float(match.group("a")),
                        phase_b_s=float(match.group("b")),
                        phase_c_s=float(match.group("c")),
                        line=line_no,
                    )
                )

            match = RE_PHASE.search(text)
            if match:
                data["phases"].append(
                    Phase(
                        step=int(match.group("step")),
                        name=" ".join(match.group("name").split()),
                        seconds=float(match.group("s")),
                        line=line_no,
                    )
                )

            match = RE_EXPERT_SUCCESS.search(text)
            if match:
                data["expert_success"].append(
                    {
                        "done": int(match.group("done")),
                        "total": int(match.group("total")) if match.group("total") else None,
                        "seconds": float(match.group("s")),
                        "line": line_no,
                    }
                )

            match = RE_DENSE_SUCCESS.search(text)
            if match:
                data["dense_success"].append(
                    {
                        "source": int(match.group("src")),
                        "params": int(match.group("params")),
                        "scalars": int(match.group("scalars")),
                        "seconds": float(match.group("s")),
                        "line": line_no,
                    }
                )

            match = RE_OPT_SUBMIT.search(text)
            if match:
                data["opt_submit"].append({"count": int(match.group("count")), "line": line_no})
            match = RE_SYNC_OPT.search(text)
            if match:
                data["sync_opt_load"].append(
                    {
                        "step": int(match.group("step")),
                        "seconds": float(match.group("s")),
                        "line": line_no,
                    }
                )
            match = RE_OPT_COMPLETE.search(text)
            if match:
                data["opt_complete"].append({"step": int(match.group("step")), "count": int(match.group("count")), "line": line_no})
            match = RE_OPT_FINALIZE.search(text)
            if match:
                data["opt_finalize"].append({"step": int(match.group("step")), "count": int(match.group("count")), "line": line_no})
            match = RE_CKPT_RESTART_TOTAL.search(text)
            if match:
                data["checkpoint_restart_total"].append(
                    {
                        "step": int(match.group("step")),
                        "seconds": float(match.group("s")),
                        "line": line_no,
                    }
                )
            match = RE_CKPT_RESTART_LOAD.search(text)
            if match:
                data["checkpoint_restart_load"].append(
                    {"seconds": float(match.group("s")), "line": line_no}
                )
            match = RE_OPT_SKIP.search(text)
            if match:
                data["opt_skip_steps"].append(int(match.group("step")))
            match = RE_ZERO.search(text)
            if match:
                data["zero"].append(
                    {
                        "step": int(match.group("step")),
                        "params": int(match.group("params")),
                        "states": int(match.group("states")),
                        "line": line_no,
                    }
                )
            match = RE_REINTEGRATED.search(text)
            if match:
                data["reintegrated_steps"].append(int(match.group("step")))
            match = RE_SAVE_START.search(text)
            if match:
                last_checkpoint_step = int(match.group("iter"))
                data["save_start_steps"].append(last_checkpoint_step)
            match = RE_SAVE_DONE.search(text)
            if match:
                last_checkpoint_step = int(match.group("iter"))
                data["save_done_steps"].append(last_checkpoint_step)
            match = RE_CKPT_META.search(text)
            if match:
                last_checkpoint_step = int(match.group("iter"))
                data["checkpoint_meta_steps"].append(last_checkpoint_step)
            match = RE_LOAD_CKPT.search(text)
            if match:
                data["load_paths"].append(match.group("path"))
            match = RE_STATE_LOAD.search(text)
            if match:
                data["state_load_paths"].append(match.group("path"))
            match = RE_TIMER.search(text)
            if match:
                data["timers"].append(
                    Timer(
                        name=" ".join(match.group("name").strip().split()),
                        min_ms=float(match.group("min")),
                        max_ms=float(match.group("max")),
                        line=line_no,
                        step=last_checkpoint_step if "checkpoint" in match.group("name") else None,
                    )
                )
    return data


def fold_repairs(repairs: list[Repair]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, str], list[Repair]] = defaultdict(list)
    for repair in repairs:
        grouped[(repair.step, repair.failed, repair.replacement, repair.path)].append(repair)
    rows = []
    for (step, failed, replacement, path), records in sorted(grouped.items()):
        rows.append(
            {
                "step": step,
                "failed": failed,
                "replacement": replacement,
                "path": path,
                "duplicate_lines": len(records),
                "total_s_max": max(r.total_s for r in records),
                "total_s_p50": median(r.total_s for r in records),
                "phase_a_s_max": max(r.phase_a_s for r in records),
                "phase_b_s_max": max(r.phase_b_s for r in records),
                "phase_c_s_max": max(r.phase_c_s for r in records),
            }
        )
    return rows


def event_steps(data: dict[str, Any], repairs: list[dict[str, Any]], neighborhood: int) -> set[int]:
    raw = {f.step for f in data["faults"]}
    raw.update(r["step"] for r in repairs)
    raw.update(data["save_start_steps"])
    raw.update(data["save_done_steps"])
    raw.update(data["checkpoint_meta_steps"])
    out = set()
    for step in raw:
        for offset in range(-neighborhood, neighborhood + 1):
            out.add(step + offset)
    return out


def clean_baseline(
    iterations: list[Iteration],
    data: dict[str, Any],
    repairs: list[dict[str, Any]],
    warmup: int,
    neighborhood: int,
    outlier_sigma: float,
) -> dict[str, Any]:
    sorted_iters = sorted(iterations, key=lambda item: (item.step, item.line))
    warmup_lines = {item.line for item in sorted_iters[:warmup]}
    excluded_steps = event_steps(data, repairs, neighborhood)

    candidates = []
    exclusions = Counter()
    for item in sorted_iters:
        if item.line in warmup_lines:
            exclusions["warmup"] += 1
            continue
        if item.skipped:
            exclusions["skipped"] += 1
            continue
        if item.step in excluded_steps:
            exclusions["event_window"] += 1
            continue
        candidates.append(item.elapsed_ms)

    center = median(candidates)
    sigma = mad_sigma(candidates)
    clean = list(candidates)
    if center is not None and sigma > 0 and outlier_sigma > 0:
        lo = max(0.0, center - outlier_sigma * sigma)
        hi = center + outlier_sigma * sigma
        clean = [value for value in candidates if lo <= value <= hi]
        exclusions["mad_outlier"] += len(candidates) - len(clean)

    return {
        "baseline_ms": median(clean),
        "raw_stats_ms": stat(item.elapsed_ms for item in sorted_iters),
        "candidate_stats_ms": stat(candidates),
        "clean_stats_ms": stat(clean),
        "exclusions": dict(exclusions),
        "first_iteration": asdict(sorted_iters[0]) if sorted_iters else None,
        "last_iteration": asdict(sorted_iters[-1]) if sorted_iters else None,
    }


def by_step(iterations: list[Iteration]) -> dict[int, Iteration]:
    result = {}
    for item in iterations:
        result[item.step] = item
    return result


def fault_windows(
    faults: list[Fault],
    repairs: list[dict[str, Any]],
    iterations: list[Iteration],
    baseline_ms: float | None,
    lookahead: int,
    save_ms_by_iter: dict[int, float],
) -> list[dict[str, Any]]:
    iteration_by_step = by_step(iterations)
    repair_by_step = {item["step"]: item for item in repairs}
    rows = []
    for fault in faults:
        iter_rows = []
        overhead_sum = 0.0
        corrected_overhead_sum = 0.0
        for step in range(fault.step, fault.step + lookahead + 1):
            item = iteration_by_step.get(step)
            if item is None:
                continue
            overhead = max(0.0, item.elapsed_ms - baseline_ms) if baseline_ms is not None else None
            checkpoint_ms = save_ms_by_iter.get(step, 0.0)
            corrected_overhead = (
                max(0.0, item.elapsed_ms - baseline_ms - checkpoint_ms) if baseline_ms is not None else None
            )
            if overhead is not None:
                overhead_sum += overhead
            if corrected_overhead is not None:
                corrected_overhead_sum += corrected_overhead
            iter_rows.append(
                {
                    "step": step,
                    "elapsed_ms": item.elapsed_ms,
                    "overhead_ms": overhead,
                    "checkpoint_ms_from_prev_iter": checkpoint_ms,
                    "corrected_overhead_ms": corrected_overhead,
                    "skipped": item.skipped,
                    "loss": item.loss,
                }
            )
        repair = repair_by_step.get(fault.step)
        rows.append(
            {
                "number": fault.number,
                "step": fault.step,
                "rank": fault.rank,
                "kind": fault.kind,
                "repair_total_s_max": repair["total_s_max"] if repair else None,
                "repair_phase_b_s_max": repair["phase_b_s_max"] if repair else None,
                "window_overhead_ms": overhead_sum,
                "corrected_window_overhead_ms": corrected_overhead_sum,
                "iterations": iter_rows,
            }
        )
    return rows


def loss_spikes(iterations: list[Iteration], faults: list[Fault], lookahead: int, sigma_mult: float) -> dict[str, Any]:
    valid = [item.loss for item in iterations if item.loss is not None and item.skipped == 0]
    center = median(valid)
    sigma = mad_sigma(valid)
    threshold = center + sigma_mult * sigma if center is not None and sigma > 0 else None
    iteration_by_step = by_step(iterations)
    spikes = []
    if threshold is not None:
        for fault in faults:
            for step in range(fault.step, fault.step + lookahead + 1):
                item = iteration_by_step.get(step)
                if item and item.loss is not None and item.loss > threshold:
                    spikes.append(
                        {
                            "fault": fault.number,
                            "fault_step": fault.step,
                            "step": step,
                            "loss": item.loss,
                            "threshold": threshold,
                        }
                    )
    return {"baseline_loss": center, "sigma": sigma, "threshold": threshold, "spikes": spikes}


def timer_summary(timers: list[Timer]) -> dict[str, Any]:
    grouped: dict[str, list[Timer]] = defaultdict(list)
    for timer in timers:
        grouped[timer.name].append(timer)
    return {
        name: {
            "count": len(records),
            "min_ms": stat(r.min_ms for r in records),
            "max_ms": stat(r.max_ms for r in records),
        }
        for name, records in sorted(grouped.items())
    }


def save_checkpoint_by_next_iter(timers: list[Timer]) -> dict[int, float]:
    """Map iteration N+1 to the checkpoint save timer printed after iteration N."""
    result: dict[int, float] = {}
    for timer in timers:
        if timer.name != "save-checkpoint" or timer.step is None:
            continue
        result[timer.step + 1] = max(result.get(timer.step + 1, 0.0), timer.max_ms)
    return result


def summary_row(metric: str, unit: str, stats: dict[str, Any], note: str = "") -> dict[str, Any]:
    return {
        "metric": metric,
        "unit": unit,
        "count": stats.get("count", 0),
        "min": stats.get("min"),
        "mean": stats.get("mean"),
        "p50": stats.get("p50"),
        "p90": stats.get("p90"),
        "p95": stats.get("p95"),
        "p99": stats.get("p99"),
        "max": stats.get("max"),
        "note": note,
    }


def build_average_summary(
    data: dict[str, Any],
    baseline: dict[str, Any],
    repairs: list[dict[str, Any]],
    fault_windows_rows: list[dict[str, Any]],
    timers: dict[str, Any],
    dense_stats_s: dict[str, Any],
    expert_stats_s: dict[str, Any],
    sync_opt_stats_s: dict[str, Any],
    checkpoint_restart_stats_s: dict[str, Any],
    checkpoint_restart_load_stats_s: dict[str, Any],
    phase_stats_s: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        summary_row(
            "iteration.raw",
            "ms",
            baseline["raw_stats_ms"],
            "All parsed iteration elapsed times, including warmup/fault/checkpoint effects.",
        ),
        summary_row(
            "iteration.candidate",
            "ms",
            baseline["candidate_stats_ms"],
            "After excluding warmup, skipped iterations, checkpoints, and fault neighborhoods.",
        ),
        summary_row(
            "iteration.clean",
            "ms",
            baseline["clean_stats_ms"],
            "Candidate iterations after MAD outlier filtering; best estimate of normal iteration cost.",
        ),
        summary_row(
            "iteration.skipped",
            "ms",
            stat(item.elapsed_ms for item in data["iterations"] if item.skipped),
            "Iterations where optimizer step was skipped.",
        ),
        summary_row(
            "iteration.non_skipped",
            "ms",
            stat(item.elapsed_ms for item in data["iterations"] if not item.skipped),
            "Iterations where optimizer step was not skipped.",
        ),
        summary_row(
            "fault.window.raw_overhead",
            "ms",
            stat(row["window_overhead_ms"] for row in fault_windows_rows),
            "Sum of iteration overheads in each fault window before checkpoint correction.",
        ),
        summary_row(
            "fault.window.checkpoint_corrected_overhead",
            "ms",
            stat(row["corrected_window_overhead_ms"] for row in fault_windows_rows),
            "Fault-window overhead after subtracting save-checkpoint time attributed to the next iteration.",
        ),
        summary_row(
            "recovery.controller_total",
            "s",
            stat(row["total_s_max"] for row in repairs),
            "SAFE-POINT REPAIR COMPLETED total time, using max duplicate value per recovery step.",
        ),
        summary_row(
            "recovery.controller_phase_b",
            "s",
            stat(row["phase_b_s_max"] for row in repairs),
            "Phase B parameter recovery time, using max duplicate value per recovery step.",
        ),
        summary_row(
            "recovery.dense_sync_success",
            "s",
            dense_stats_s,
            "Dense parameter sync success records.",
        ),
        summary_row(
            "recovery.expert_restore_success",
            "s",
            expert_stats_s,
            "Stale expert restore / expert_restore_fn success records.",
        ),
        summary_row(
            "recovery.sync_optimizer_load",
            "s",
            sync_opt_stats_s,
            "Synchronous expert optimizer state load in the recovery critical path.",
        ),
        summary_row(
            "recovery.checkpoint_restart_total",
            "s",
            checkpoint_restart_stats_s,
            "Full checkpoint restart callback total recovery time.",
        ),
        summary_row(
            "recovery.checkpoint_restart_load",
            "s",
            checkpoint_restart_load_stats_s,
            "Full checkpoint restart callback checkpoint load time.",
        ),
    ]

    for phase_name, phase_stats in sorted(phase_stats_s.items()):
        rows.append(
            summary_row(
                f"recovery.phase.{phase_name}",
                "s",
                phase_stats,
                "Controller phase elapsed log records.",
            )
        )

    for timer_name, timer_stats in sorted(timers.items()):
        rows.append(
            summary_row(
                f"timer.{timer_name}.max_rank",
                "ms",
                timer_stats["max_ms"],
                "Timer max across ranks.",
            )
        )

    return rows


def build_report(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = parse_log(path)
    faults = unique(data["faults"], lambda item: (item.number, item.step, item.rank, item.kind))
    repairs = fold_repairs(data["repairs"])
    base = clean_baseline(data["iterations"], data, repairs, args.warmup_iters, args.event_neighborhood, args.outlier_sigma)
    baseline_ms = base["baseline_ms"]
    save_ms_by_iter = save_checkpoint_by_next_iter(data["timers"])
    windows = fault_windows(faults, repairs, data["iterations"], baseline_ms, args.fault_lookahead, save_ms_by_iter)
    timers = timer_summary(data["timers"])
    phase_stats_s = {
        name: stat(item.seconds for item in data["phases"] if item.name == name)
        for name in sorted({item.name for item in data["phases"]})
    }
    dense_stats_s = stat(item["seconds"] for item in data["dense_success"])
    expert_stats_s = stat(item["seconds"] for item in data["expert_success"])
    sync_opt_stats_s = stat(item["seconds"] for item in data["sync_opt_load"])
    checkpoint_restart_stats_s = stat(item["seconds"] for item in data["checkpoint_restart_total"])
    checkpoint_restart_load_stats_s = stat(item["seconds"] for item in data["checkpoint_restart_load"])
    average_summary = build_average_summary(
        data=data,
        baseline=base,
        repairs=repairs,
        fault_windows_rows=windows,
        timers=timers,
        dense_stats_s=dense_stats_s,
        expert_stats_s=expert_stats_s,
        sync_opt_stats_s=sync_opt_stats_s,
        checkpoint_restart_stats_s=checkpoint_restart_stats_s,
        checkpoint_restart_load_stats_s=checkpoint_restart_load_stats_s,
        phase_stats_s=phase_stats_s,
    )

    fault_steps = {f.step for f in faults}
    repair_steps = {r["step"] for r in repairs}
    missing_repairs = sorted(fault_steps - repair_steps)
    unique_skip_steps = sorted(set(data["opt_skip_steps"]))
    reintegrated_steps = sorted(set(data["reintegrated_steps"]))
    state_load_counter = Counter(data["state_load_paths"])
    load_counter = Counter(data["load_paths"])

    status = "ok"
    notes = []
    if data["line_count"] == 0:
        status = "unknown"
        notes.append("log is empty")
    if data["severe"]:
        status = "bad"
        notes.append(f"found {len(data['severe'])} severe error lines")
    if data["no_match_count"]:
        status = "bad"
        notes.append(f"found {data['no_match_count']} no-matching expert param lines")
    if missing_repairs:
        status = "bad"
        notes.append(f"fault steps without repair: {missing_repairs[:10]}")
    if status == "ok":
        notes.append("no fatal NCCL/OOM/Traceback/no-matching-param signal found")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "line_count": data["line_count"],
        "status": status,
        "notes": notes,
        "counts": {
            "iterations": len(data["iterations"]),
            "faults": len(faults),
            "repairs": len(repairs),
            "optimizer_skip_steps": len(unique_skip_steps),
            "reintegrated_steps": len(reintegrated_steps),
            "directory_update_fail_lines": data["directory_update_fail_count"],
            "severe_error_lines": len(data["severe"]),
            "no_match_lines": data["no_match_count"],
        },
        "baseline": base,
        "faults": [asdict(item) for item in faults],
        "repairs": repairs,
        "fault_windows": windows,
        "fault_window_stats_ms": stat(row["window_overhead_ms"] for row in windows),
        "corrected_fault_window_stats_ms": stat(row["corrected_window_overhead_ms"] for row in windows),
        "average_summary": average_summary,
        "loss": loss_spikes(data["iterations"], faults, args.loss_lookahead, args.loss_sigma),
        "timers": timers,
        "phase_stats_s": phase_stats_s,
        "expert_success_stats_s": expert_stats_s,
        "dense_success_stats_s": dense_stats_s,
        "sync_optimizer_load_stats_s": sync_opt_stats_s,
        "checkpoint_restart_total_stats_s": checkpoint_restart_stats_s,
        "checkpoint_restart_load_stats_s": checkpoint_restart_load_stats_s,
        "dense_sources": sorted({item["source"] for item in data["dense_success"]}),
        "optimizer_submit_counts": stat(item["count"] for item in data["opt_submit"]),
        "optimizer_complete_counts": stat(item["count"] for item in data["opt_complete"]),
        "optimizer_finalize_counts": stat(item["count"] for item in data["opt_finalize"]),
        "zero_invalidations": unique(data["zero"], lambda item: (item["step"], item["params"], item["states"])),
        "checkpoint": {
            "save_start_steps": sorted(set(data["save_start_steps"])),
            "save_done_steps": sorted(set(data["save_done_steps"])),
            "checkpoint_meta_steps": sorted(set(data["checkpoint_meta_steps"])),
            "save_ms_by_next_iter": dict(sorted(save_ms_by_iter.items())),
            "state_load_paths_top": state_load_counter.most_common(10),
            "load_paths_top": load_counter.most_common(10),
        },
        "severe_examples": data["severe"][: args.top],
    }


def fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def fmt_s(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def fmt_by_unit(value: Any, unit: str) -> str:
    if unit == "s":
        return fmt_s(value)
    if unit == "ms":
        return fmt_ms(value)
    return "n/a" if value is None else f"{float(value):.3f} {unit}"


def metric_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in report.get("average_summary", [])}


def print_average_line(row: dict[str, Any]) -> None:
    unit = row["unit"]
    print(
        f"- {row['metric']}: count={row['count']}, "
        f"mean={fmt_by_unit(row['mean'], unit)}, "
        f"p50={fmt_by_unit(row['p50'], unit)}, "
        f"p95={fmt_by_unit(row['p95'], unit)}, "
        f"max={fmt_by_unit(row['max'], unit)}"
    )


def print_report(report: dict[str, Any], top: int) -> None:
    print("BSR-MoE Timing Report")
    print(f"Log: {report['path']}")
    print(f"Size: {report['size_bytes']} bytes, lines: {report['line_count']}")
    print(f"Status: {report['status'].upper()}")
    for note in report["notes"]:
        print(f"- {note}")
    print(
        "- counts: "
        f"iters={report['counts']['iterations']}, faults={report['counts']['faults']}, "
        f"repairs={report['counts']['repairs']}, opt_skip_steps={report['counts']['optimizer_skip_steps']}, "
        f"reintegrated_steps={report['counts']['reintegrated_steps']}"
    )
    if report["counts"]["directory_update_fail_lines"]:
        print(f"- directory update failed lines: {report['counts']['directory_update_fail_lines']} (warning lines, restore may still succeed)")
    print()

    metrics = metric_map(report)
    print("Average Summary")
    for metric in (
        "iteration.raw",
        "iteration.clean",
        "iteration.skipped",
        "iteration.non_skipped",
        "timer.save-checkpoint.max_rank",
        "timer.load-checkpoint.max_rank",
        "timer.evaluate.max_rank",
        "fault.window.raw_overhead",
        "fault.window.checkpoint_corrected_overhead",
        "recovery.controller_total",
        "recovery.controller_phase_b",
        "recovery.dense_sync_success",
        "recovery.expert_restore_success",
    ):
        row = metrics.get(metric)
        if row and row["count"]:
            print_average_line(row)
    print()

    base = report["baseline"]
    first = base["first_iteration"]
    last = base["last_iteration"]
    print("Iteration Timing")
    if first:
        print(f"- first parsed iter: {first['step']} elapsed={fmt_ms(first['elapsed_ms'])}, skipped={first['skipped']}")
    if last:
        print(f"- last parsed iter:  {last['step']} elapsed={fmt_ms(last['elapsed_ms'])}, skipped={last['skipped']}")
    print(f"- clean baseline: {fmt_ms(base['baseline_ms'])} (exclusions={base['exclusions']})")
    print(
        f"- raw elapsed: p50={fmt_ms(base['raw_stats_ms']['p50'])}, "
        f"p95={fmt_ms(base['raw_stats_ms']['p95'])}, max={fmt_ms(base['raw_stats_ms']['max'])}"
    )
    print(
        f"- clean elapsed: p50={fmt_ms(base['clean_stats_ms']['p50'])}, "
        f"p95={fmt_ms(base['clean_stats_ms']['p95'])}, max={fmt_ms(base['clean_stats_ms']['max'])}"
    )
    print()

    print("Fault/Recovery")
    for fault in report["faults"][:top]:
        print(f"- fault #{fault['number']}: step={fault['step']}, rank={fault['rank']}, type={fault['kind']}")
    if len(report["faults"]) > top:
        print(f"- ... {len(report['faults']) - top} more faults omitted")
    repair_stats = stat(item["total_s_max"] for item in report["repairs"])
    print(
        f"- controller repair total(max per step): count={repair_stats['count']}, "
        f"p50={fmt_s(repair_stats['p50'])}, p95={fmt_s(repair_stats['p95'])}, max={fmt_s(repair_stats['max'])}"
    )
    print(
        f"- observed fault-window overhead: p50={fmt_ms(report['fault_window_stats_ms']['p50'])}, "
        f"p95={fmt_ms(report['fault_window_stats_ms']['p95'])}, max={fmt_ms(report['fault_window_stats_ms']['max'])}"
    )
    print(
        f"- checkpoint-corrected fault-window overhead: "
        f"p50={fmt_ms(report['corrected_fault_window_stats_ms']['p50'])}, "
        f"p95={fmt_ms(report['corrected_fault_window_stats_ms']['p95'])}, "
        f"max={fmt_ms(report['corrected_fault_window_stats_ms']['max'])}"
    )
    worst = sorted(report["fault_windows"], key=lambda item: item["window_overhead_ms"], reverse=True)[:top]
    print("- worst fault windows:")
    for row in worst:
        iter_text = ", ".join(
            f"{item['step']}:{fmt_ms(item['elapsed_ms'])}/ckpt{fmt_ms(item['checkpoint_ms_from_prev_iter'])}/skip{item['skipped']}"
            for item in row["iterations"][:4]
        )
        print(
            f"  #{row['number']} step={row['step']} rank={row['rank']} "
            f"repair={fmt_s(row['repair_total_s_max'])}, raw={fmt_ms(row['window_overhead_ms'])}, "
            f"corrected={fmt_ms(row['corrected_window_overhead_ms'])} | {iter_text}"
        )
    print()

    print("Recovery Components")
    print(f"- dense success: sources={report['dense_sources']}, {report['dense_success_stats_s']}")
    print(f"- expert restore success seconds: {report['expert_success_stats_s']}")
    for name, stats in report["phase_stats_s"].items():
        print(f"- controller phase {name}: p50={fmt_s(stats['p50'])}, p95={fmt_s(stats['p95'])}, max={fmt_s(stats['max'])}")
    if report["zero_invalidations"]:
        sample = report["zero_invalidations"][:top]
        print(f"- zero invalidations sample: {sample}")
    print()

    print("Checkpoint/Timer")
    save = report["timers"].get("save-checkpoint")
    load = report["timers"].get("load-checkpoint")
    evaluate = report["timers"].get("evaluate")
    if save:
        print(
            f"- save-checkpoint max-rank timer: count={save['count']}, "
            f"mean={fmt_ms(save['max_ms']['mean'])}, p50={fmt_ms(save['max_ms']['p50'])}, "
            f"max={fmt_ms(save['max_ms']['max'])}"
        )
    if load:
        print(
            f"- load-checkpoint max-rank timer: count={load['count']}, "
            f"mean={fmt_ms(load['max_ms']['mean'])}, p50={fmt_ms(load['max_ms']['p50'])}, "
            f"max={fmt_ms(load['max_ms']['max'])}"
        )
    if evaluate:
        print(
            f"- evaluate max-rank timer: count={evaluate['count']}, "
            f"mean={fmt_ms(evaluate['max_ms']['mean'])}, p50={fmt_ms(evaluate['max_ms']['p50'])}, "
            f"max={fmt_ms(evaluate['max_ms']['max'])}"
        )
    if report["checkpoint"]["state_load_paths_top"]:
        print("- top checkpoint state_dict loads:")
        for path, count in report["checkpoint"]["state_load_paths_top"][:top]:
            print(f"  {count}x {path}")
    print()

    loss = report["loss"]
    print("Loss")
    print(
        f"- baseline={loss['baseline_loss']}, sigma={loss['sigma']}, "
        f"threshold={loss['threshold']}, spikes={len(loss['spikes'])}"
    )
    for spike in loss["spikes"][:top]:
        print(f"  fault#{spike['fault']} step={spike['step']} loss={spike['loss']:.6g} threshold={spike['threshold']:.6g}")


def write_fault_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "number",
                "step",
                "rank",
                "kind",
                "repair_total_s_max",
                "repair_phase_b_s_max",
                "window_overhead_ms",
                "corrected_window_overhead_ms",
                "iterations",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in writer.fieldnames if key != "iterations"},
                    "iterations": json.dumps(row["iterations"], ensure_ascii=False),
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["metric", "unit", "count", "min", "mean", "p50", "p90", "p95", "p99", "max", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Path to training log")
    parser.add_argument("--json", help="Write full JSON report")
    parser.add_argument("--fault-csv", help="Write per-fault timing CSV")
    parser.add_argument("--summary-csv", help="Write average timing summary CSV")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--event-neighborhood", type=int, default=1)
    parser.add_argument("--outlier-sigma", type=float, default=4.0)
    parser.add_argument("--fault-lookahead", type=int, default=2)
    parser.add_argument("--loss-lookahead", type=int, default=5)
    parser.add_argument("--loss-sigma", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    path = Path(args.log).expanduser().resolve()
    if not path.exists():
        print(f"error: log not found: {path}", file=sys.stderr)
        return 2
    report = build_report(path, args)
    if args.json:
        Path(args.json).expanduser().resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.fault_csv:
        write_fault_csv(Path(args.fault_csv).expanduser().resolve(), report["fault_windows"])
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv).expanduser().resolve(), report["average_summary"])
    if not args.quiet:
        print_report(report, args.top)
    if report["status"] == "bad":
        return 1
    if report["status"] == "unknown":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
