#!/usr/bin/env python3
"""Parse ablation.log and emit per-mode statistics in a clean, paper-ready form.

Recovery timings (Phase A / Phase B / Phase B->C) come from controller log
lines and are already in seconds. Per-iteration `elapsed time per iteration`
fields from the train log are already in milliseconds.
"""
from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOG = Path("/Users/zds/bsr/ablation.log")
OUT_CSV = Path("/Users/zds/bsr/ablation_summary.csv")

MODES = ["selective_deferred", "selective_sync_opt", "full_checkpoint"]

RE_MODE_BEGIN = re.compile(r"^\[ablation\] mode=(\S+)$")
RE_MODE_END = re.compile(r"^\[ablation\] mode=(\S+) finished normally$")
# elapsed= ... s (seconds, float)
RE_PHASE_A = re.compile(r"\[Phase A\] Infrastructure repair COMPLETED \(elapsed=([\d.]+)s\)")
RE_PHASE_B = re.compile(
    r"\[Phase B\] Parameter recovery COMPLETED \(path=(\w+), elapsed=([\d.]+)s\)"
)
RE_PHASE_BC = re.compile(
    r"\[Phase B→C\] Post-recovery convergence completed "
    r"\(path=(\w+), elapsed=([\d.]+)s, step=(\d+)\)"
)
# Per-iter wall time in ms.
RE_ITER = re.compile(
    r"iteration\s+(\d+)/\s*\d+ \|.*elapsed time per iteration \(ms\): ([\d.]+) "
    r"\|.*lm loss: ([\d.E+\-]+)"
)


def stats(label: str, vals_s: list[float]) -> dict:
    """vals are seconds; report ms."""
    if not vals_s:
        return {"label": label, "n": 0}
    n = len(vals_s)
    ms = [v * 1000.0 for v in vals_s]
    return {
        "label": label,
        "n": n,
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "std_ms": statistics.stdev(ms) if n > 1 else 0.0,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "p95_ms": sorted(ms)[int(0.95 * (n - 1))],
    }


def stats_ms(label: str, ms: list[float]) -> dict:
    if not ms:
        return {"label": label, "n": 0}
    n = len(ms)
    return {
        "label": label,
        "n": n,
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "std_ms": statistics.stdev(ms) if n > 1 else 0.0,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "p95_ms": sorted(ms)[int(0.95 * (n - 1))],
    }


def fmt_row(s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"  {s['label']:<46}  (no samples)"
    return (f"  {s['label']:<46}  n={s['n']:>4}  "
            f"mean={s['mean_ms']:9.2f}  med={s['median_ms']:9.2f}  "
            f"p95={s['p95_ms']:9.2f}  std={s['std_ms']:9.2f}  "
            f"min={s['min_ms']:8.2f}  max={s['max_ms']:9.2f}   (ms)")


def parse_modes(text: str) -> dict[str, tuple[int, int]]:
    bounds: dict[str, tuple[int, int]] = {}
    starts: dict[str, int] = {}
    for i, line in enumerate(text.splitlines(), 1):
        m = RE_MODE_BEGIN.match(line.strip())
        if m and m.group(1) in MODES and m.group(1) not in starts:
            starts[m.group(1)] = i
            continue
        m = RE_MODE_END.match(line.strip())
        if m and m.group(1) in MODES:
            bounds[m.group(1)] = (starts[m.group(1)], i)
    return bounds


def analyze(mode: str, lines: list[str]) -> dict:
    phase_a: list[float] = []
    phase_b: dict[str, list[float]] = defaultdict(list)
    phase_bc: dict[str, list[float]] = defaultdict(list)
    phase_b_steps: list[int] = []
    iter_records: list[tuple[int, float, float]] = []  # (iter, t_ms, loss)

    for ln in lines:
        if (m := RE_PHASE_A.search(ln)):
            phase_a.append(float(m.group(1)))
            continue
        if (m := RE_PHASE_B.search(ln)):
            phase_b[m.group(1)].append(float(m.group(2)))
            continue
        if (m := RE_PHASE_BC.search(ln)):
            phase_bc[m.group(1)].append(float(m.group(2)))
            phase_b_steps.append(int(m.group(3)))
            continue
        if (m := RE_ITER.search(ln)):
            iter_records.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    return {
        "mode": mode,
        "phase_a_s": phase_a,
        "phase_b_s_per_path": dict(phase_b),
        "phase_bc_s_per_path": dict(phase_bc),
        "phase_b_steps": phase_b_steps,
        "fault_steps_unique": sorted(set(phase_b_steps)),
        "iter_records": iter_records,
    }


def main() -> None:
    text = LOG.read_text()
    lines = text.splitlines()
    bounds = parse_modes(text)

    print("=" * 100)
    print("Mode boundaries (lines, inclusive):")
    for m in MODES:
        s, e = bounds[m]
        print(f"  {m:<22}  [{s:>5}..{e:>5}]   length={e-s+1:>5}")
    print()

    rows: list[dict] = []
    summaries: list[dict] = []

    for m in MODES:
        s, e = bounds[m]
        section = lines[s - 1:e]
        r = analyze(m, section)

        print("=" * 100)
        print(f"mode = {m}")
        print("=" * 100)

        # Recovery phase timings (raw seconds -> ms).
        a_stat = stats("Phase A (infra repair)", r["phase_a_s"])
        print(fmt_row(a_stat))
        for path, vals in r["phase_b_s_per_path"].items():
            print(fmt_row(stats(f"Phase B  ({path})", vals)))
        for path, vals in r["phase_bc_s_per_path"].items():
            print(fmt_row(stats(f"Phase B->C ({path})", vals)))

        # Phase A+B per-event critical path (sum across phase samples / event count).
        # Each fault event triggers one Phase A and one Phase B per rank; we
        # aggregate the per-rank elapsed and report the mean as the synchronous
        # critical-path estimator for that event class.
        a_mean_s = statistics.fmean(r["phase_a_s"]) if r["phase_a_s"] else 0.0
        all_b_s = [v for vs in r["phase_b_s_per_path"].values() for v in vs]
        b_mean_s = statistics.fmean(all_b_s) if all_b_s else 0.0
        crit_s = a_mean_s + b_mean_s

        n_events = len(r["fault_steps_unique"])
        print(f"  fault events (unique steps): {n_events}  "
              f"(first={r['fault_steps_unique'][0]}, last={r['fault_steps_unique'][-1]}, "
              f"interval=40)")
        print(f"  recovery critical-path estimator (Phase A + Phase B mean): "
              f"{crit_s*1000:.2f} ms")

        # Iteration-time analysis.
        iters = r["iter_records"]
        # Drop warmup (iter 1 has 207s init bias, iter 2-5 jumpy).
        body = [(i, t, l) for (i, t, l) in iters if i >= 6]
        fault_steps = set(r["fault_steps_unique"])
        # Recovery overhead lands on the iter immediately AFTER fault step.
        # In Megatron, fault is injected at step k; the recovery work runs
        # before step k+1's forward, so iter k+1 carries the wall.
        fwin_t = [t for (i, t, _) in body if (i - 1) in fault_steps]
        clean_t = [t for (i, t, _) in body if (i - 1) not in fault_steps]
        if clean_t and fwin_t:
            cm = statistics.fmean(clean_t)
            fm = statistics.fmean(fwin_t)
            print()
            print(fmt_row(stats_ms("per-iter wall (clean iters)", clean_t)))
            print(fmt_row(stats_ms("per-iter wall (fault-window iters)", fwin_t)))
            print(f"  fault-window OVERHEAD per event (mean fwin - mean clean): "
                  f"{fm - cm:.2f} ms ({(fm - cm)/1000:.3f} s)")
            total_overhead_s = (fm - cm) * n_events / 1000.0
            print(f"  total recovery overhead across {n_events} events: "
                  f"{total_overhead_s:.2f} s")

        loss_last = body[-1][2] if body else float("nan")
        total_wall_s = sum(t for (_, t, _) in body) / 1000.0
        print(f"  body iters analyzed: {len(body)}   total wall: {total_wall_s:.1f}s "
              f"({total_wall_s/60:.2f}min)")
        print(f"  final lm loss: {loss_last:.4f}")
        print()

        summaries.append({
            "mode": m,
            "events": n_events,
            "phase_a_mean_ms": a_mean_s * 1000,
            "phase_b_mean_ms": b_mean_s * 1000,
            "phase_b_median_ms": statistics.median(all_b_s) * 1000 if all_b_s else 0,
            "phase_b_p95_ms": (sorted(all_b_s)[int(0.95 * (len(all_b_s)-1))] * 1000) if all_b_s else 0,
            "phase_b_max_ms": max(all_b_s) * 1000 if all_b_s else 0,
            "crit_path_mean_ms": crit_s * 1000,
            "path_label": ",".join(sorted(r["phase_b_s_per_path"].keys())),
            "iter_clean_mean_ms": statistics.fmean(clean_t) if clean_t else 0,
            "iter_fault_mean_ms": statistics.fmean(fwin_t) if fwin_t else 0,
            "iter_overhead_per_event_ms": (statistics.fmean(fwin_t) - statistics.fmean(clean_t))
                                          if (clean_t and fwin_t) else 0,
            "final_lm_loss": loss_last,
            "total_wall_s": total_wall_s,
        })

    # Cross-mode comparison.
    print("=" * 100)
    print("CROSS-MODE COMPARISON")
    print("=" * 100)
    hdr = (f"{'mode':<22} {'events':>7} {'crit_mean_ms':>14} {'crit_med_ms':>13} "
           f"{'crit_p95_ms':>13} {'iter_clean':>11} {'iter_fault':>11} {'over/ev_ms':>12} "
           f"{'final_loss':>11} {'wall_s':>9}  path")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(f"{s['mode']:<22} {s['events']:>7} {s['phase_b_mean_ms']:>14.2f} "
              f"{s['phase_b_median_ms']:>13.2f} {s['phase_b_p95_ms']:>13.2f} "
              f"{s['iter_clean_mean_ms']:>11.2f} {s['iter_fault_mean_ms']:>11.2f} "
              f"{s['iter_overhead_per_event_ms']:>12.2f} "
              f"{s['final_lm_loss']:>11.4f} {s['total_wall_s']:>9.1f}  {s['path_label']}")

    # Attribution.
    print()
    print("Attribution decomposition (using Phase B mean as critical path; Phase A is ~0):")
    by_mode = {s["mode"]: s for s in summaries}
    f = by_mode["full_checkpoint"]["phase_b_mean_ms"]
    so = by_mode["selective_sync_opt"]["phase_b_mean_ms"]
    sd = by_mode["selective_deferred"]["phase_b_mean_ms"]
    sel_gain = f - so
    opt_gain = so - sd
    tot_gain = f - sd
    print(f"  full_checkpoint        : {f:>11.2f} ms  (CHECKPOINT_RESTART, baseline)")
    print(f"  selective_sync_opt     : {so:>11.2f} ms  (HYBRID, opt loaded sync)")
    print(f"  selective_deferred     : {sd:>11.2f} ms  (HYBRID, opt deferred)")
    print(f"  selective_restore_gain = full − sync_opt   = "
          f"{sel_gain:.2f} ms  ({sel_gain/f*100:5.2f} %)")
    print(f"  optimizer_defer_gain   = sync_opt − defer  = "
          f"{opt_gain:.2f} ms  ({opt_gain/f*100:5.2f} %)")
    print(f"  total_gain             = full − defer      = "
          f"{tot_gain:.2f} ms  ({tot_gain/f*100:5.2f} %)")
    print(f"  speedup vs full        : {f/sd:6.1f}×")

    # Iteration-time view of the same.
    print()
    print("Same attribution from per-iter fault-window overhead (rank-0 train log):")
    f = by_mode["full_checkpoint"]["iter_overhead_per_event_ms"]
    so = by_mode["selective_sync_opt"]["iter_overhead_per_event_ms"]
    sd = by_mode["selective_deferred"]["iter_overhead_per_event_ms"]
    print(f"  full_checkpoint        : {f:>11.2f} ms / event  "
          f"(includes data-loader resume, NCCL re-init, etc.)")
    print(f"  selective_sync_opt     : {so:>11.2f} ms / event")
    print(f"  selective_deferred     : {sd:>11.2f} ms / event")

    # Loss / quality.
    print()
    print("Final lm loss after 2060 iters with 50 faults:")
    for s in summaries:
        print(f"  {s['mode']:<22}  {s['final_lm_loss']:.4f}")

    # CSV.
    with OUT_CSV.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"\nWrote summary CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
