#!/usr/bin/env python3
"""Reproduce the checkpoint-splice quality figure from the archived summary."""

import csv
import sys
from paths import DATA, OUT
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import PHASE_COLORS, PHASE_MARKERS, apply_style, decimal_ticks, grid
ROOT = DATA
ROWS = list(csv.DictReader((ROOT / "checkpoint_results.csv").open(newline="")))
SHORT = [r for r in ROWS if r["horizon"] == "500"]
FULL = [r for r in ROWS if r["horizon"] == "full"]
assert len(SHORT) == 12 and len(FULL) == 3


def pct(row, key):
    return 100.0 * float(row[key])


apply_style()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.55), layout="constrained")
colors = PHASE_COLORS
labels = {"1000": "Early (1,000)", "5050": "Middle (5,050)", "9050": "Late (9,050)"}

for stage in ("1000", "5050", "9050"):
    items = [r for r in SHORT if r["case_id"].startswith(f"s{stage}_")]
    items.sort(key=lambda r: ["one", "two", "three", "four"].index(r["case_id"].split("_")[2]))
    offset = {"1000": -0.07, "5050": 0, "9050": 0.07}[stage]
    # Rank identities and placements differ: these are categorical cases,
    # not a continuous dose-response curve.
    ax.scatter([rank + offset for rank in range(1, 5)],
               [pct(r, "peak_relative_degradation") for r in items],
               marker=PHASE_MARKERS[stage], s=24, color=colors[stage],
               label=labels[stage], zorder=3)
ax.set_yscale("log")
ax.set_ylim(1e-5, 2)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Affected ranks")
ax.set_ylabel("Peak increase (%)")
ax.set_title("(a) First 500 recovery steps", loc="left")
ax.axhline(1, color="#697586", linestyle="--", linewidth=0.8)
ax.text(4.2, 1.13, "Peak tolerance: 1%", ha="right", fontsize=6.3, color="#55606F")
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.78), frameon=False)
grid(ax)
decimal_ticks(ax.yaxis)

full_order = [
    "s1000_g800_four_balanced",
    "s5050_g50_one_high_load",
    "s9050_g50_one_low_load",
]
items = [next(r for r in FULL if r["case_id"] == case) for case in full_order]
for i, row in enumerate(items):
    bx.plot(i, pct(row, "peak_relative_degradation"), "o", color=colors[row["case_id"].split("_")[0][1:]], markersize=5)
    bx.plot(i, pct(row, "endpoint_relative_degradation"), "D", color=colors[row["case_id"].split("_")[0][1:]], markersize=4)
bx.set_yscale("log")
bx.set_ylim(1e-4, 2)
bx.set_xticks(range(3), ["Early\n4 ranks", "Middle\n1 rank", "Late\n1 rank"])
bx.set_ylabel("Increase (%)")
bx.set_title("(b) Step-10,000 outcomes", loc="left")
bx.axhline(1, color="#697586", linestyle="--", linewidth=0.8)
bx.axhline(0.5, color="#697586", linestyle=":", linewidth=0.8)
bx.text(2.15, 1.12, "Peak: 1%", ha="right", fontsize=6.2, color="#55606F")
bx.text(2.15, 0.53, "Final: 0.5%", ha="right", fontsize=6.2, color="#55606F")
bx.plot([], [], "ko", markersize=4, label="Peak in first 500 steps")
bx.plot([], [], "kD", markersize=4, label="Step-10,000 endpoint")
bx.legend(fontsize=6.4, loc="lower left", frameon=False)
grid(bx)
decimal_ticks(bx.yaxis)
fig.savefig(OUT / "quality_checkpoint_study.pdf")

fig.savefig(OUT / "plot_checkpoint.png", dpi=190)
