#!/usr/bin/env python3
"""Recompute architecture comparisons and plot the supplied paired losses."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from paths import DATA, OUT

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = DATA
ROOT = OUT
from plot_style import apply_style, grid

PROFILES = ("small_plain_24x1024", "small_deepseek_mla_24x1024")
STEPS = (1050, 2450, 4050)
RANKS = (41, 24, 62)
OFFSETS = (0, 10, 20, 50, 100)
COLORS = ("#23658D", "#CC6B2B")
MARKERS = ("o", "s")
NAMES = ("GQA MoE", "DeepSeek-style MLA MoE")


def audit():
    summary = json.loads((HERE / "architecture_summary.json").read_text())
    csv_rows = list(csv.DictReader((HERE / "architecture_quality_results.csv").open()))
    reports = summary["results"]
    expected = {(p, t, r) for p in PROFILES for t in STEPS for r in ("nofault", "restart")}
    keys = [(r["profile"], r["fault_step"], r["reference_arm"]) for r in reports]
    if len(keys) != 12 or set(keys) != expected or len(csv_rows) != 12:
        raise ValueError("Expected 12 unique pairs from six Hybrid branches")
    trajectories = []
    for report in reports:
        p, t, arm = report["profile"], report["fault_step"], report["reference_arm"]
        if (report["seed"] != 1234 or report["state_treatment"] != "hybrid_full_state"
                or report["old_step"] != t - 50 or report["old_step"] % 200
                or report["failed_rank"] != RANKS[STEPS.index(t)]
                or report["terminal_step"] != t + 100
                or report["whole_run_Y"] is not None
                or report["calibrated_run_risk"] is not None):
            raise ValueError(f"Protocol mismatch: {p}, {t}, {arm}")
        rows = report["rows"]
        if [r["step"] - t for r in rows] != list(OFFSETS):
            raise ValueError("Unexpected fixed evaluation grid")
        for row in rows:
            if not all(math.isfinite(row[k]) and row[k] > 0
                       for k in ("hybrid_loss", "reference_loss")):
                raise ValueError("Invalid loss")
            delta = (row["hybrid_loss"] - row["reference_loss"]) / row["reference_loss"]
            if not math.isclose(delta, row["signed_relative_degradation"], abs_tol=1e-14):
                raise ValueError("Paired loss does not reproduce its reported change")
            trajectories.append(dict(profile=p, fault_step=t, reference_arm=arm,
                                     **row, signed_percent_change=100 * delta))
        peak = max(0, *(r["signed_relative_degradation"] for r in rows))
        endpoint = rows[-1]["signed_relative_degradation"]
        if not math.isclose(peak, report["sampled_peak_positive_relative_degradation"], abs_tol=1e-14):
            raise ValueError("Peak mismatch")
        if not math.isclose(endpoint, report["step_100_signed_relative_degradation"], abs_tol=1e-14):
            raise ValueError("Endpoint mismatch")
        c = next(r for r in csv_rows if (r["profile"], int(r["fault_step"]), r["reference_arm"]) == (p, t, arm))
        for field in ("sampled_peak_positive_relative_degradation", "step_100_signed_relative_degradation"):
            if not math.isclose(float(c[field]), report[field], abs_tol=1e-14):
                raise ValueError("CSV differs from JSON")
        other = next(r for r in reports if r["profile"] == p and r["fault_step"] == t and r["reference_arm"] != arm)
        if [r["hybrid_loss"] for r in rows] != [r["hybrid_loss"] for r in other["rows"]]:
            raise ValueError("Comparators do not share the same Hybrid trajectory")
    with (OUT / "paired_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    analysis = dict(
        comparison_rows=12, distinct_hybrid_branches=6, paired_evaluation_rows=60,
        seed=1234, expert_age=50, evaluation_offsets=list(OFFSETS),
        source_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in (HERE / "architecture_summary.json", HERE / "architecture_quality_results.csv")},
        aggregate_values_reproduced=True, same_actual_training_sample_ids_proven=False,
        scope="100-step paired architecture diagnostics; not whole-run Y or calibrated risk",
        maxima={}
    )
    for arm in ("nofault", "restart"):
        analysis["maxima"][arm] = {}
        for profile in PROFILES:
            group = [r for r in reports if r["reference_arm"] == arm and r["profile"] == profile]
            analysis["maxima"][arm][profile] = dict(
                peak_percent=max(r["sampled_peak_positive_relative_degradation"] for r in group) * 100,
                min_endpoint_percent=min(r["step_100_signed_relative_degradation"] for r in group) * 100,
                max_endpoint_percent=max(r["step_100_signed_relative_degradation"] for r in group) * 100,
            )
    (OUT / "architecture_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return reports


def main():
    reports = audit()
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.1))
    x = np.arange(3)
    for col, arm in enumerate(("nofault", "restart")):
        peak_ax, end_ax = axes[:, col]
        for i, profile in enumerate(PROFILES):
            rows = [next(r for r in reports if (r["profile"], r["fault_step"], r["reference_arm"]) == (profile, t, arm)) for t in STEPS]
            xp = x + (-0.065 if i == 0 else 0.065)
            peaks = [r["sampled_peak_positive_relative_degradation"] * 100 for r in rows]
            ends = [r["step_100_signed_relative_degradation"] * 100 for r in rows]
            for ax, values in ((peak_ax, peaks), (end_ax, ends)):
                ax.scatter(xp, values, c=COLORS[i], marker=MARKERS[i], s=26, zorder=4,
                           edgecolors="white", linewidths=0.45)
            peak_index = int(np.argmax(peaks))
            peak_ax.annotate(f"{peaks[peak_index]:.3g}%", (xp[peak_index], peaks[peak_index]),
                             xytext=(7, 8), textcoords="offset points",
                             ha="left", color=COLORS[i], fontsize=7)
        peak_ax.set_yscale("symlog", linthresh=0.0001, linscale=0.6)
        peak_ax.set_ylim(-0.00003, 2.2)
        peak_ax.set_yticks([0, 0.001, 0.01, 0.1, 1])
        peak_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        peak_ax.axhline(1, color="#758391", ls="--", lw=0.8)
        peak_ax.text(2.34, 1.07, "1% peak tolerance", ha="right", va="bottom", fontsize=6.8)
        end_ax.axhline(0, color="#758391", ls=":", lw=0.75)
        peak_ax.set_title(f"({chr(97+col)}) Peak vs {'NoFault' if arm=='nofault' else 'Restart'}", loc="left", fontweight="bold")
        end_ax.set_title(f"({chr(99+col)}) Step-100 endpoint vs {'NoFault' if arm=='nofault' else 'Restart'}", loc="left", fontweight="bold")
        peak_ax.set_ylabel("Positive peak loss change (%)")
        end_ax.set_ylabel("Signed endpoint loss change (%)")
        end_ax.set_ylim((-0.013, 0.016) if arm == "nofault" else (-0.128, 0.018))
        for ax in (peak_ax, end_ax):
            ax.set_xlim(-0.4, 2.4)
            ax.set_xticks(x, [f"{t:,}" for t in STEPS])
            grid(ax)
        end_ax.set_xlabel("Fault step (expert age = 50)")
    handles = [Line2D([], [], color=c, marker=m, linestyle="None", markersize=5, label=n)
               for c, m, n in zip(COLORS, MARKERS, NAMES)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.13, top=0.85, hspace=0.58, wspace=0.3)
    fig.savefig(ROOT / "quality_architecture_transfer.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "quality_architecture_transfer.png", dpi=200, bbox_inches="tight", pad_inches=0.04)
    print(json.dumps(json.loads((OUT / "architecture_analysis.json").read_text())["maxima"], indent=2))


if __name__ == "__main__":
    main()
