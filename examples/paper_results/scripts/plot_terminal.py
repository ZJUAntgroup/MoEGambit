#!/usr/bin/env python3
"""Validate the 102result export and reproduce its selected-history figure.

Connect only supplied validation points. The four histories share seed 1234
and prefixes; they are not independent trials or calibrated risk estimates.
"""
import csv
import hashlib
import json
import math
import sys
from paths import DATA, OUT
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = DATA
from plot_style import apply_style, grid

POLICIES = ("single", "late", "repeat_same", "repeat_other")
NAMES = ("Single", "Late", "Repeat: same rank", "Repeat: other rank")
COLORS = ("#265F84", "#C46B2E", "#677A45", "#805993")
MARKERS = ("o", "s", "^", "D")


def close(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-14):
        raise ValueError(f"Inconsistent {label}: {actual} != {expected}")


def main():
    summary_path, trajectory_path = HERE / "risk_summary.json", HERE / "risk_trajectory.csv"
    summary = json.loads(summary_path.read_text())
    if summary["optimizer_steps"] != 10450 or summary["risk_calibrated"] or summary["alpha_run_target"] != .05:
        raise ValueError("Unexpected experiment budget or risk-calibration claim")
    reports = {r["policy"]: r for r in summary["results"]}
    if len(summary["results"]) != 4 or set(reports) != set(POLICIES):
        raise ValueError("Expected four selected histories")
    with trajectory_path.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    keys = [(r["policy"], int(r["step"])) for r in exported]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate trajectory rows")
    expected_keys, outcomes = set(), []
    for policy in POLICIES:
        report = reports[policy]
        expected_faults = [9050] if policy == "late" else [8200, 9050] if policy.startswith("repeat") else [8200]
        expected_second = 57 if policy == "repeat_same" else 7 if policy == "repeat_other" else None
        if (report["seed"] != 1234 or report["terminal_step"] != 10000 or
                report["state_treatment"] != "hybrid_full_state" or report["failed_rank"] != 57 or
                report["fault_steps"] != expected_faults or report["second_failed_rank"] != expected_second or
                report["eta_final"] != .005 or report["eta_peak"] != .01 or
                report["calibrated_run_risk"] is not None or report["same_actual_training_sample_ids_proven"]):
            raise ValueError(f"Protocol or evidence scope changed: {policy}")
        rows = report["rows"]
        expected_grid = [8200, 8210, 8300, 8700, 9050, 9060, 9150, 9550, 10000]
        if policy == "late":
            expected_grid = expected_grid[4:]
        if [r["step"] for r in rows] != expected_grid or report["validation_steps"] != expected_grid:
            raise ValueError(f"Unexpected evaluation grid: {policy}")
        for row in rows:
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("hybrid_loss", "reference_loss")):
                raise ValueError("Invalid validation loss")
            close(row["signed_relative_degradation"],
                  (row["hybrid_loss"] - row["reference_loss"]) / row["reference_loss"], policy)
            expected_keys.add((policy, row["step"]))
        final, peak = rows[-1]["signed_relative_degradation"], max(r["signed_relative_degradation"] for r in rows)
        y = max(max(0, final) / .005, max(0, peak) / .01)
        close(report["final_signed_relative_degradation"], final, f"{policy} final")
        close(report["sampled_peak_signed_relative_degradation"], peak, f"{policy} peak")
        close(report["Y"], y, f"{policy} Y")
        if report["quality_boundary_exceeded"] != (y > 1):
            raise ValueError("Incorrect exceedance flag")
        outcomes.append(dict(policy=policy, final_percent=100 * final, sampled_peak_percent=100 * peak,
                             final_tolerance_fraction=max(0, final) / .005,
                             peak_tolerance_fraction=max(0, peak) / .01,
                             selected_history_Y=y, boundary_exceeded=y > 1))
    if set(keys) != expected_keys:
        raise ValueError("CSV and JSON trajectory coverage differs")
    for row in exported:
        expected = next(r for r in reports[row["policy"]]["rows"] if r["step"] == int(row["step"]))
        for key in ("hybrid_loss", "reference_loss", "signed_relative_degradation"):
            close(float(row[key]), expected[key], f"CSV {key}")
    for policy in ("repeat_same", "repeat_other"):
        if reports[policy]["rows"][:4] != reports["single"]["rows"][:4]:
            raise ValueError("Repeated histories no longer share the first-fault prefix")
    for group in ("nofault_vs_saved_baseline", "full_remaining_trajectory_nofault_control"):
        for row in summary[group]:
            close(row["signed_relative_degradation"],
                  (row["hybrid_loss"] - row["reference_loss"]) / row["reference_loss"], group)

    analysis = dict(source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (summary_path, trajectory_path)},
                    trajectory_rows=len(exported), seed=1234, results=outcomes,
                    maximum_terminal_percent=max(r["final_percent"] for r in outcomes),
                    maximum_sampled_peak_percent=max(r["sampled_peak_percent"] for r in outcomes),
                    max_abs_nofault_baseline_replay_percent=100 * max(abs(r["signed_relative_degradation"])
                                                                    for r in summary["nofault_vs_saved_baseline"]),
                    risk_calibrated=False, raw_state_audits_in_export=False,
                    interpretation="Selected post-fault histories to terminal step; common-prefix evaluations before the first fault are not exported. Y is a measured normalized loss outcome, not admission score R or a failure probability.")
    (OUT / "terminal_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    with (OUT / "terminal_outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)

    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6})
    fig, (ax, budget) = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.095, right=.98, bottom=.22, top=.86, wspace=.34)
    ax.set_title("(a) Quality through step 10,000", loc="left", fontweight="bold")
    for policy, name, color, marker in zip(POLICIES, NAMES, COLORS, MARKERS):
        rows = reports[policy]["rows"]
        if policy.startswith("repeat"):
            rows = rows[3:]  # Shared earlier samples are drawn once by Single.
        ax.plot([r["step"] for r in rows], [100 * r["signed_relative_degradation"] for r in rows],
                label=name, color=color, marker=marker, markersize=3.8, linewidth=1.1,
                markeredgecolor="white", markeredgewidth=.35,
                linestyle="--" if policy.startswith("repeat") else "-")
    ax.axhline(0, color="#89939D", linewidth=.8, linestyle=":")
    for step in (8200, 9050):
        ax.axvline(step, color="#C6CDD3", linewidth=.8, linestyle=":", zorder=0)
    ax.set_xlim(8130, 10070)
    ax.set_ylim(-.0105, .009)
    ax.set_xticks([8200, 8700, 9050, 9550, 10000], ["8,200", "8,700", "9,050", "9,550", "10,000"])
    ax.set_yticks([-.01, -.005, 0, .005], ["-0.010", "-0.005", "0", "0.005"])
    ax.set_xlabel("Committed training step")
    ax.set_ylabel("Loss change vs restart (%)")
    ax.legend(loc="upper left", ncol=2, frameon=False, columnspacing=.9, handlelength=1.5)
    grid(ax)
    budget.set_title("(b) Fixed quality-tolerance use", loc="left", fontweight="bold")
    for i, (row, color) in enumerate(zip(outcomes, COLORS)):
        for shift, key, marker in ((-.12, "final_tolerance_fraction", "o"),
                                   (.12, "peak_tolerance_fraction", "s")):
            budget.scatter(i + shift, row[key], s=35, marker=marker, color=color,
                           edgecolors="white", linewidths=.45, zorder=3)
        budget.text(i, max(row["final_tolerance_fraction"], row["peak_tolerance_fraction"]) * 1.35,
                    f"{row['selected_history_Y']:.3g}", ha="center", fontsize=7.4, color=color)
    budget.set_yscale("log")
    budget.set_ylim(.00075, 2.5)
    budget.set_yticks([.001, .01, .1, 1], ["0.001", "0.01", "0.1", "1"])
    budget.axhline(1, color="#67717A", linestyle="--", linewidth=.9)
    budget.text(3.4, 1.1, "Quality boundary = 1", ha="right", va="bottom", fontsize=7.6)
    budget.set_xlim(-.45, 3.45)
    budget.set_xticks(range(4), ["Single", "Late", "Repeat\nsame", "Repeat\nother"])
    budget.set_ylabel("Positive change / tolerance")
    budget.legend(handles=[Line2D([], [], color="#56616C", marker="o", linestyle="None", label="Final / 0.5%"),
                           Line2D([], [], color="#56616C", marker="s", linestyle="None", label="Peak / 1%")],
                  loc="upper left", bbox_to_anchor=(.01, .86), ncol=1, frameon=False,
                  columnspacing=1, handletextpad=.35)
    grid(budget)
    output = OUT / "quality_full_state_terminal.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=.045)
    fig.savefig(OUT / "quality_full_state_terminal.png", dpi=200, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
