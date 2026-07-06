#!/usr/bin/env python3
"""
Compact heatmap for fault-injection eval-loss results.
Outputs a true vector PDF plus a raster PNG preview.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

fault_losses = np.array([
    4.8662, 4.8130, 4.8247, 4.8229, 4.8617,
    4.8214, 4.8755, 4.7858, 4.8602, 4.8580,
    4.8457, 4.8633, 4.8053, 4.8277, 4.8646,
    4.8385, 4.8287, 4.8305, 4.8044, 4.8402,
    4.8375, 4.9203, 4.8195, 4.8623, 4.8244,
    4.8257, 4.8308, 4.8320, 4.8545, 4.8256,
    4.8294, 4.8102, 4.8314, 4.8596, 4.8023,
    4.8031, 4.8137, 4.8184, 4.8211, 4.8137,
    4.8269, 4.8478, 4.8562, 4.8301, 4.8337,
    4.8326, 4.8218, 4.8156, 4.8469, 4.8193,
])

baseline_losses = np.array([
    4.8118, 4.8674, 4.8528, 4.8199, 4.8456,
    4.8686, 4.8739, 4.8903, 4.8570, 4.8556,
])

mu_b = baseline_losses.mean()
sigma_b = baseline_losses.std(ddof=1)
dev = fault_losses - mu_b

within_1s = np.sum(np.abs(dev) <= 1 * sigma_b)
within_2s = np.sum(np.abs(dev) <= 2 * sigma_b)
print(f"Baseline: mu={mu_b:.4f}  sigma={sigma_b:.4f}")
print(f"|dev| <= 1*sigma: {within_1s}/50  |  <= 2*sigma: {within_2s}/50")

# ---------------------------------------------------------------------------
# Custom colormap: soft blue → white → warm salmon
# Matched to fault_heatmap_compact.png by pixel sampling
# ---------------------------------------------------------------------------

cmap_colors = [
    (0.00, (0.23, 0.47, 0.74)),   # deep blue
    (0.15, (0.33, 0.57, 0.80)),   # medium-dark blue
    (0.30, (0.45, 0.67, 0.84)),   # medium blue
    (0.42, (0.56, 0.76, 0.87)),   # light blue
    (0.50, (0.93, 0.93, 0.93)),   # near-white center
    (0.58, (0.95, 0.85, 0.78)),   # pale salmon
    (0.70, (0.94, 0.74, 0.63)),   # light salmon
    (0.85, (0.90, 0.56, 0.43)),   # medium salmon
    (1.00, (0.75, 0.22, 0.17)),   # deep red
]

cdict = {"red": [], "green": [], "blue": []}
for pos, (r, g, b) in cmap_colors:
    cdict["red"].append((pos, r, r))
    cdict["green"].append((pos, g, g))
    cdict["blue"].append((pos, b, b))

custom_cmap = LinearSegmentedColormap("SoftBlueWhiteSalmon", cdict, N=256)

# ---------------------------------------------------------------------------
# Figure — native IEEE single-column size.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2,
    "axes.labelsize": 6.4,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.30, 0.98))

grid = dev.reshape(5, 10)
loss_grid = fault_losses.reshape(5, 10)

vmax = max(abs(dev.min()), abs(dev.max()))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

for i in range(5):
    for j in range(10):
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=custom_cmap(norm(grid[i, j])),
                edgecolor="#f2f2f2",
                linewidth=0.10,
            )
        )

ax.set_xlim(-0.5, 9.5)
ax.set_ylim(4.5, -0.5)
ax.set_aspect("auto")
ax.set_xticks(np.arange(10))
ax.set_xticklabels([str(i) for i in range(10)])
ax.set_yticks(np.arange(5))
ax.set_yticklabels([f"{351 + 10 * i}-{360 + 10 * i}" for i in range(5)])
ax.set_xlabel("Offset")
ax.set_ylabel("Step")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", which="both", length=0, pad=1.5)

for i in range(5):
    for j in range(10):
        val = grid[i, j]
        col = "white" if abs(val) > 0.55 * vmax else "black"
        ax.text(j, i, f"{loss_grid[i, j]:.3f}",
                ha="center", va="center", color=col, fontsize=3.7)

fig.subplots_adjust(left=0.13, right=0.875, bottom=0.32, top=0.96)
cax = fig.add_axes([0.895, 0.32, 0.018, 0.64])
steps = 96
for k in range(steps):
    y0 = k / steps
    value = norm.vmin + (norm.vmax - norm.vmin) * (k + 0.5) / steps
    cax.add_patch(
        Rectangle(
            (0, y0),
            1,
            1 / steps,
            facecolor=custom_cmap(norm(value)),
            edgecolor="none",
        )
    )
cax.set_xlim(0, 1)
cax.set_ylim(0, 1)
cax.set_xticks([])
tick_values = np.linspace(norm.vmin, norm.vmax, 3)
cax.set_yticks((tick_values - norm.vmin) / (norm.vmax - norm.vmin))
cax.set_yticklabels([f"{v:.2f}" for v in tick_values])
cax.yaxis.tick_right()
cax.yaxis.set_label_position("right")
cax.tick_params(axis="y", which="both", length=0, pad=1)
cax.set_ylabel("$\\Delta L$", labelpad=1)
for spine in cax.spines.values():
    spine.set_visible(False)

fig.savefig("./fault_heatmap.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig("./fault_heatmap_compact.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig("./fault_heatmap_new.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
print("Saved: fault_heatmap.pdf, fault_heatmap_compact.pdf, fault_heatmap_new.png")
