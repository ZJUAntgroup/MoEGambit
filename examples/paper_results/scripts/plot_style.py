"""Shared typography and colors for the paper's aggregate-data figures."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PHASE_COLORS = {"1000": "#B3543D", "5050": "#267D92", "9050": "#665A9B"}
PHASE_MARKERS = {"1000": "o", "5050": "s", "9050": "^"}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 6.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.titlepad": 7,
    })


def decimal_ticks(axis):
    axis.set_major_formatter(FuncFormatter(
        lambda value, _position: f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"))


def grid(ax, axis="y"):
    ax.grid(axis=axis, which="major", color="#DDE2E7", linewidth=0.45)
    ax.set_axisbelow(True)
