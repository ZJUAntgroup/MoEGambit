#!/usr/bin/env python3
"""Reproduce the rank-local scaling and independent MoC-port aggregates.

The CSV field ``full_load_seconds`` denotes rank-local checkpoint loading,
not a measured whole-job rollback. These are rounded means, not per-event
observations, and do not support splitting peer and shard path costs.
"""

import csv
from pathlib import Path
from paths import DATA, OUT as OUTPUT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_style, grid


OUT = OUTPUT / "recovery_scaling.pdf"
DATA = DATA / "recovery_scaling.csv"
with DATA.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
SCALE = [(int(r["gpus"]), float(r["full_load_seconds"]),
          float(r["repair_seconds"])) for r in rows if r["panel"] == "scale"]
LAYOUTS = [(f'{r["label"]} · EDP {r["edp"]}', float(r["full_load_seconds"]),
            float(r["repair_seconds"]), float(r["pec_sync_seconds"]),
            float(r["pec_2level_async_seconds"]))
           for r in rows if r["panel"] == "layout"]
assert len(SCALE) == 2 and len(LAYOUTS) == 4
assert [r["repair_path"] for r in rows if r["panel"] == "layout"] == [
    "hybrid", "hybrid", "full_peer", "full_peer"]
assert [round(r / h, 2) for _, r, h in SCALE] == [1.26, 1.40]
assert [round(r / h, 2) for _, r, h, _, _ in LAYOUTS] == [1.26, 1.62, 3.52, 3.56]
assert all(min(r[1:]) > 0 for r in LAYOUTS)

apply_style()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.6, 2.8), layout="constrained")
restart_color, repair_color = "#b44e48", "#326f90"

x = np.arange(len(SCALE))
width = 0.30
restart = [row[1] for row in SCALE]
repair = [row[2] for row in SCALE]
ax.bar(x - width / 2, restart, width, color=restart_color, label="FullLoad")
ax.bar(x + width / 2, repair, width, color=repair_color, label="MoEGambit")
for i, (_, r, h) in enumerate(SCALE):
    ax.text(i - width / 2, r + 0.7, f"{r:.1f}", ha="center", fontsize=6.8)
    ax.text(i + width / 2, h + 0.7, f"{h:.1f}", ha="center", fontsize=6.8)
ax.set_ylim(0, 54)
ax.set_xticks(x, [f"{row[0]} GPUs" for row in SCALE])
ax.set_ylabel("Recovery-event latency (s)")
ax.set_title("(a) GPU count", loc="left", fontsize=9)
ax.legend(loc="upper left", frameon=False, fontsize=7.0)
grid(ax)

labels = [row[0] for row in LAYOUTS]
y = np.arange(len(LAYOUTS))
series = [
    ("MoEGambit", 2, -0.21, "#326f90", None),
    ("PEC-sync", 3, 0.00, "#c78b55", None),
    ("PEC-2L async", 4, 0.21, "#807598", "//"),
]
for name, column, offset, color, hatch in series:
    ratios = [row[1] / row[column] for row in LAYOUTS]
    bx.barh(y + offset, ratios, height=0.18, color=color, label=name,
            hatch=hatch, edgecolor="white", linewidth=0.3)
    for yi, ratio in zip(y + offset, ratios):
        bx.text(ratio + 0.035, yi, f"{ratio:.2f}×", va="center", fontsize=6.4)
bx.set_yticks(y, labels)
bx.invert_yaxis()
bx.set_xlim(0, 4.25)
bx.set_xticks([0, 1, 2, 3, 4])
bx.set_xlabel("FullLoad / recovery latency")
bx.set_title("(b) 64-GPU layouts", loc="left", fontsize=9)
bx.axvline(1, color="#697586", linestyle="--", linewidth=0.8)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2,
          frameon=False, fontsize=6.4, handlelength=1.3,
          columnspacing=0.9, labelspacing=0.3, borderaxespad=0)
grid(bx, axis="x")

fig.savefig(OUT)
# Keep the standalone vector artifact and the manuscript asset synchronized.
export = OUT.parent / "output/pdf/recovery_scaling.pdf"
export.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(export)
print(OUT)

fig.savefig(OUTPUT / "recovery_scaling.png", dpi=200)
