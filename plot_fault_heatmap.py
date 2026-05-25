#!/usr/bin/env python3
"""
Visualization of 50 single-fault injection eval losses (iter 600) vs the
10-run NoFault baseline. Produces a composite figure:

  (a) Top: 10x5 heatmap of (loss - mu_baseline) for the 50 fault runs,
      ordered by injection step 351..400.
  (b) Bottom: per-run deviation scatter with baseline +/-1 sigma and
      +/-2 sigma bands; baseline 10-run distribution shown as a small
      strip on the right.

Outputs:
  fault_heatmap.pdf  (vector, for LaTeX \includegraphics)
  fault_heatmap.png  (raster, for previews)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# 50 single-fault hybrid-recovery validation losses at iteration 600
# (faults injected at iterations 351..400, one per run).
fault_losses = np.array([
    # findmax1.log (25 runs, faults 351..375)
    4.8662, 4.8130, 4.8247, 4.8229, 4.8617,
    4.8214, 4.8755, 4.7858, 4.8602, 4.8580,
    4.8457, 4.8633, 4.8053, 4.8277, 4.8646,
    4.8385, 4.8287, 4.8305, 4.8044, 4.8402,
    4.8375, 4.9203, 4.8195, 4.8623, 4.8244,
    # findmax2.log (25 runs, faults 376..400)
    4.8257, 4.8308, 4.8320, 4.8545, 4.8256,
    4.8294, 4.8102, 4.8314, 4.8596, 4.8023,
    4.8031, 4.8137, 4.8184, 4.8211, 4.8137,
    4.8269, 4.8478, 4.8562, 4.8301, 4.8337,
    4.8326, 4.8218, 4.8156, 4.8469, 4.8193,
])

# 10 NoFault baseline runs (iter 600 eval loss)
baseline_losses = np.array([
    4.8118, 4.8674, 4.8528, 4.8199, 4.8456,
    4.8686, 4.8739, 4.8903, 4.8570, 4.8556,
])

mu_b = baseline_losses.mean()
sigma_b = baseline_losses.std(ddof=1)
mu_f = fault_losses.mean()
sigma_f = fault_losses.std(ddof=1)

print(f"Baseline: n=10  mu={mu_b:.4f}  sigma={sigma_b:.4f}  "
      f"min={baseline_losses.min():.4f}  max={baseline_losses.max():.4f}")
print(f"Fault   : n=50  mu={mu_f:.4f}  sigma={sigma_f:.4f}  "
      f"min={fault_losses.min():.4f}  max={fault_losses.max():.4f}")
print(f"Diff of means: {mu_f - mu_b:+.4f}  ({(mu_f - mu_b)/sigma_b:+.2f} sigma_baseline)")

# Deviation from baseline mean
dev = fault_losses - mu_b           # shape (50,)
inj_steps = np.arange(351, 401)     # 351..400

within_1s = np.sum(np.abs(dev) <= 1 * sigma_b)
within_2s = np.sum(np.abs(dev) <= 2 * sigma_b)
print(f"|dev| <= 1*sigma_base : {within_1s}/50")
print(f"|dev| <= 2*sigma_base : {within_2s}/50")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig = plt.figure(figsize=(7.0, 5.6))
gs = gridspec.GridSpec(
    2, 2,
    height_ratios=[1.0, 1.0],
    width_ratios=[1.0, 0.04],
    hspace=0.55, wspace=0.05,
)

# ---- (a) Heatmap: 5 rows x 10 cols, ordered by injection step ----------
ax_hm = fig.add_subplot(gs[0, 0])
ax_cb = fig.add_subplot(gs[0, 1])

# Reshape to 5x10 (rows: injection-step blocks of 10; cols: position 0..9)
grid = dev.reshape(5, 10)
step_grid = inj_steps.reshape(5, 10)

vmax = max(abs(dev.min()), abs(dev.max()))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

im = ax_hm.imshow(grid, cmap="RdBu_r", norm=norm, aspect="auto")
ax_hm.set_xticks(np.arange(10))
ax_hm.set_xticklabels([f"+{i}" for i in range(10)])
ax_hm.set_yticks(np.arange(5))
ax_hm.set_yticklabels([f"{351 + 10 * i}-{360 + 10 * i}" for i in range(5)])
ax_hm.set_xlabel("offset within injection block")
ax_hm.set_ylabel("injection-step block")
ax_hm.set_title(
    "(a) Per-run eval-loss deviation vs NoFault baseline mean "
    f"($\\mu_{{base}}={mu_b:.4f}$)"
)

# Annotate each cell with the actual loss value (small font)
for i in range(5):
    for j in range(10):
        val = grid[i, j]
        # Pick text color based on cell darkness
        col = "white" if abs(val) > 0.55 * vmax else "black"
        ax_hm.text(j, i, f"{fault_losses.reshape(5,10)[i,j]:.3f}",
                   ha="center", va="center", color=col, fontsize=6.5)

cb = fig.colorbar(im, cax=ax_cb)
cb.set_label("loss $-$ $\\mu_{base}$")

# ---- (b) Per-run deviation scatter with baseline sigma bands -----------
ax_sc = fig.add_subplot(gs[1, :])

# +/-1 sigma and +/-2 sigma bands (centered at 0 = baseline mean)
ax_sc.axhspan(-2 * sigma_b, 2 * sigma_b, color="#cfe3ff", alpha=0.55,
              label=f"baseline $\\pm 2\\sigma$ ({2*sigma_b:.3f})")
ax_sc.axhspan(-1 * sigma_b, 1 * sigma_b, color="#7fb2ff", alpha=0.55,
              label=f"baseline $\\pm 1\\sigma$ ({sigma_b:.3f})")
ax_sc.axhline(0, color="black", linewidth=1.0,
              label=f"baseline mean ($\\mu_{{base}}={mu_b:.4f}$)")

# Color points by sign of deviation
colors = ["#b22222" if d > 0 else "#1f4e79" for d in dev]
ax_sc.scatter(inj_steps, dev, c=colors, s=22, edgecolor="black",
              linewidth=0.4, zorder=3, label="fault-injection run (n=50)")

# Mean line for fault runs
ax_sc.axhline(mu_f - mu_b, color="#2ca02c", linewidth=1.2, linestyle="--",
              label=f"fault-run mean ($\\mu_{{fault}}={mu_f:.4f}$)")

ax_sc.set_xlim(350, 401)
ax_sc.set_xlabel("fault-injection iteration")
ax_sc.set_ylabel("loss $-$ $\\mu_{base}$")
ax_sc.set_title(
    "(b) 50 single-fault runs vs 10-run NoFault baseline "
    f"({within_2s}/50 within $\\pm 2\\sigma$, "
    f"{within_1s}/50 within $\\pm 1\\sigma$)"
)
ax_sc.grid(True, axis="y", alpha=0.3)
ax_sc.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=7.5)

fig.suptitle(
    "Hybrid-recovery quality stays inside the NoFault noise floor",
    fontsize=11, y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("/Users/zds/bsr/fault_heatmap.pdf", bbox_inches="tight")
fig.savefig("/Users/zds/bsr/fault_heatmap.png", dpi=200, bbox_inches="tight")
print("Saved: fault_heatmap.pdf, fault_heatmap.png")
