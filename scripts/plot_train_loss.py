#!/usr/bin/env python3
"""Render a clean training-loss figure for paper.tex.

Design (v3):
- single full-width axes; tail comparison via an inset, not a second panel
- log-y so the 12 -> 2.7 dynamic range stays visually balanced
- smoothed-only foreground (no raw shadow); three colors, three dash styles
- fault injections shown as a *rug* along the bottom x-axis (short ticks),
  not vertical lines crossing the whole figure
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

ROOT = Path("/Users/zds/bsr")
LOGS = {
    "Restart (Megatron)": ROOT / "megatron.log",
    "MoC-System":         ROOT / "mocsystem.log",
    "MoEGambit":          ROOT / "moegambit.log",
}

FAULT_STEPS = [723, 1188, 2461, 3517, 4309, 5640, 6855, 7912, 8689, 9304]

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/")
LOSS_RE = re.compile(r"lm loss:\s*([0-9.+\-eE]+)")


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    last: dict[int, float] = {}
    with path.open() as f:
        for line in f:
            if "lm loss:" not in line:
                continue
            mi, ml = ITER_RE.search(line), LOSS_RE.search(line)
            if not (mi and ml):
                continue
            try:
                last[int(mi.group(1))] = float(ml.group(1))
            except ValueError:
                continue
    iters = np.array(sorted(last))
    losses = np.array([last[i] for i in iters])
    return iters, losses


def ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        y[i] = acc
    return y


def stride_avg(it: np.ndarray, ls: np.ndarray, stride: int):
    n = (len(it) // stride) * stride
    it2 = it[:n].reshape(-1, stride).mean(axis=1)
    ls2 = ls[:n].reshape(-1, stride).mean(axis=1)
    if n < len(it):
        it2 = np.append(it2, it[n:].mean())
        ls2 = np.append(ls2, ls[n:].mean())
    return it2, ls2


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    colors = {
        "Restart (Megatron)": "#1f77b4",
        "MoC-System":         "#d62728",
        "MoEGambit":          "#2ca02c",
    }
    dashes = {
        "Restart (Megatron)": (None, None),
        "MoC-System":         (5, 2),
        "MoEGambit":          (1.6, 1.6),
    }
    fault_color = "#555555"

    fig, ax = plt.subplots(figsize=(7.2, 2.8))

    series = {name: parse_log(p) for name, p in LOGS.items()}

    # --- Main curves: smoothed only, log-y ---
    for name, (it, ls) in series.items():
        # stride=10 keeps the first averaged point close to the true
        # ~12 starting loss; stronger EMA (alpha=0.3) preserves the
        # sharp early descent rather than flattening it.
        it_s, ls_s = stride_avg(it, ls, stride=10)
        ls_s = ema(ls_s, alpha=0.30)
        line, = ax.plot(it_s, ls_s, color=colors[name], linewidth=1.5,
                        label=name, solid_capstyle="round")
        if dashes[name] != (None, None):
            line.set_dashes(dashes[name])

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("LM loss (log scale)", labelpad=4)
    ax.set_xlim(0, 10000)
    ax.set_yscale("log")
    # Explicit ylim so the high-loss start (~12) is not clipped by the
    # auto-rescaling that the major-tick FixedLocator would otherwise impose.
    ax.set_ylim(2.5, 14)
    # Major ticks carry labels; minor ticks are visual aids without text,
    # so the squeezed 2.6/2.7/2.8/2.9 band doesn't overlap.
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([3, 4, 5, 7, 10, 13]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(FixedLocator(
        [2.6, 2.7, 2.8, 2.9, 3.5, 4.5, 5.5, 6, 6.5, 7.5, 8, 8.5, 9, 9.5,
         11, 12]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=2, labelleft=False)
    ax.tick_params(which="major", pad=2.5)

    ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Faults as full-height dashed verticals across the plot, plus a
    # short solid tick at the bottom for emphasis. Using axes-fraction y
    # avoids any collision with log-scale ylim / tick labels.
    trans = ax.get_xaxis_transform()  # x: data; y: axes [0,1]
    for s in FAULT_STEPS:
        # full-height faint dashed line crossing the curves
        ax.plot([s, s], [0.0, 1.0], transform=trans,
                color=fault_color, linestyle=(0, (2, 3)),
                linewidth=0.6, alpha=0.45, zorder=1, clip_on=True)
        # short solid bottom tick for unambiguous marker
        ax.plot([s, s], [0.0, 0.025], transform=trans,
                color=fault_color, linewidth=1.1, alpha=0.95,
                solid_capstyle="butt", zorder=4, clip_on=True)
    fault_proxy = plt.Line2D([0], [0], color=fault_color, linewidth=0.9,
                             linestyle=(0, (2, 3)),
                             label=f"Injected fault ($\\times {len(FAULT_STEPS)}$)")

    # --- Inset: tail 9000..10000 (linear scale, real differences visible) ---
    axin = inset_axes(ax, width="36%", height="44%",
                      loc="upper right", borderpad=1.2)
    zoom_lo, zoom_hi = 9000, 10000
    for name, (it, ls) in series.items():
        mask = (it >= zoom_lo) & (it <= zoom_hi)
        it_t, ls_t = it[mask], ls[mask]
        it_s, ls_s = stride_avg(it_t, ls_t, stride=4)
        ls_s = ema(ls_s, alpha=0.18)
        line, = axin.plot(it_s, ls_s, color=colors[name], linewidth=1.2)
        if dashes[name] != (None, None):
            line.set_dashes(dashes[name])
    axin.set_xlim(zoom_lo, zoom_hi)
    axin.set_xticks([9000, 9500, 10000])
    axin.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
    axin.set_title("Tail: iter 9k--10k", fontsize=7.5, pad=1.5)
    for sp in ("top", "right"):
        axin.spines[sp].set_visible(False)
    axin.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)
    # Mark faults inside the inset (axes-fraction y, same trick as main)
    trans_in = axin.get_xaxis_transform()
    for s in FAULT_STEPS:
        if zoom_lo <= s <= zoom_hi:
            axin.plot([s, s], [0.0, 0.07], transform=trans_in,
                      color=fault_color, linewidth=0.9, alpha=0.9,
                      zorder=4)

    # --- Legend placed ABOVE the axes (horizontal), never overlaps curves ---
    handles, labels = ax.get_legend_handles_labels()
    handles.append(fault_proxy)
    labels.append(fault_proxy.get_label())
    ax.legend(handles, labels,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), frameon=False,
              handlelength=2.4, columnspacing=1.6, borderaxespad=0.0)

    out = ROOT / "train_loss.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out}")

    print("\nFinal-iteration loss (last 200 iters mean):")
    for name, (it, ls) in series.items():
        tail = ls[it >= (it.max() - 200)]
        print(f"  {name:20s}: mean={tail.mean():.4f}  last={ls[-1]:.4f}  "
              f"n={len(ls)}")
    print(f"\nFault steps marked: {FAULT_STEPS}")


if __name__ == "__main__":
    main()
