#!/usr/bin/env python3
"""Plot ten full-state Hybrid/Restart publication comparisons.

Uses author-confirmed rerun protocol and existing Restart rows for duplicate
middle/high-load comparisons. Does not infer missing raw trajectories or a
statistical risk certificate; original exports remain preserved.
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


def main():
    source = HERE / "expert_count_results.csv"
    summary_path = HERE / "summary.json"
    summary = json.loads(summary_path.read_text())
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or summary["rows"] != 10:
        raise ValueError("Expected ten unique full-state Hybrid/Restart comparisons")
    if summary["independent_risk_certificate_available"] or summary["same_actual_sample_ids_proven"]:
        raise ValueError("Unexpected scope change in source summary")
    if summary["validation_offsets"] != [0, 10, 20, 50, 100, 200, 500]:
        raise ValueError("Unexpected fixed evaluation grid")
    index = {}
    for row in rows:
        key = (int(row["seed"]), row["case_id"], row["reference_arm"])
        if key in index:
            raise ValueError(f"Duplicate comparison: {key}")
        index[key] = row
        peak = float(row["sampled_peak_positive_relative_increase"])
        endpoint = float(row["step_500_signed_relative_loss_change"])
        if not math.isfinite(peak) or not math.isfinite(endpoint) or peak < 0:
            raise ValueError("Nonfinite or negative positive-part peak")
        if peak + 1e-14 < max(0, endpoint):
            raise ValueError("Peak smaller than endpoint")
        if not math.isclose(float(row["eta_peak_fraction"]), peak / .01, abs_tol=1e-12):
            raise ValueError("Peak-tolerance fraction does not match source data")
        if (int(row["expert_gap_steps"]) != 50 or int(row["old_step"]) % 200
                or int(row["fault_step"]) - int(row["old_step"]) != 50):
            raise ValueError("Source checkpoints violate the registered age/grid")
        if row["whole_run_Y"] or row["calibrated_run_risk"]:
            raise ValueError("500-step aggregates must not contain run-level risk")
        if int(row["num_experts"]) != int(summary["experts_by_seed"][row["seed"]]):
            raise ValueError("Expert count differs from summary")
        if row["same_actual_sample_ids_proven"].lower() != "false":
            raise ValueError("Unexpected sample-ID guarantee")
    if sum(r["reference_arm"] == "restart" for r in rows) != 10:
        raise ValueError("All current comparisons must use actual Restart")
    configs = [(1235, 128, "#265F84", "o", -.10),
               (1236, 64, "#C46B2E", "s", .10)]
    def get(seed, step, kind="one_high_load", reference="restart"):
        return index[seed, f"s{step}_g50_{kind}", reference]
    def percent(row, metric):
        return 100 * float(row[metric])
    peak_key = "sampled_peak_positive_relative_increase"
    end_key = "step_500_signed_relative_loss_change"
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 9.3, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.titlepad": 9})
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.5))
    fig.subplots_adjust(left=.13, right=.98, bottom=.12, top=.80,
                        hspace=.78, wspace=.39)
    titles = ["(a) Peak vs training point", "(b) Loss at t + 500",
              "(c) Load: loss at t + 500", "(d) Load: sampled peak"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0, color="#87929B", linestyle=":", linewidth=.8, zorder=1)
        grid(ax)
    for seed, experts, color, marker, shift in configs:
        stage = [get(seed, t) for t in (1050, 5050, 9050)]
        for ax, metric in ((axes[0, 0], peak_key), (axes[0, 1], end_key)):
            ax.scatter([x + shift for x in range(3)], [percent(r, metric) for r in stage],
                       color=color, marker=marker, s=39, edgecolor="white", linewidth=.5, zorder=3)
        middle = [get(seed, 5050, kind) for kind in
                  ("one_high_load", "one_low_load", "four_balanced")]
        axes[1, 0].scatter([x + shift for x in range(3)],
                          [percent(r, end_key) for r in middle],
                          color=color, marker=marker, s=39, edgecolor="white", linewidth=.5, zorder=3)
        axes[1, 1].scatter([x + shift for x in range(3)],
                          [percent(r, peak_key) for r in middle],
                          color=color, marker=marker, s=39, edgecolor="white", linewidth=.5, zorder=3)
    for ax in axes[0]:
        ax.set_xticks(range(3), ["1,050", "5,050", "9,050"])
        ax.set_xlabel("Fault step (expert age = 50)")
        ax.set_xlim(-.4, 2.4)
        ax.set_yscale("symlog", linthresh=.001, linscale=.8)
    axes[0, 0].set_ylabel("Peak loss increase (%)")
    axes[0, 0].set_ylim(-.0004, 2.0)
    axes[0, 0].set_yticks([0, .001, .01, .1, 1], ["0", "0.001", "0.01", "0.1", "1"])
    axes[0, 0].axhline(1, linestyle="--", linewidth=.9, color="#67717A")
    axes[0, 0].text(2.3, 1.07, "1% peak tolerance", ha="right", va="bottom", fontsize=7.5)
    early64 = percent(get(1236, 1050), peak_key)
    early128 = percent(get(1235, 1050), peak_key)
    axes[0, 0].annotate(f"{early64:.3g}%", (0.1, early64), xytext=(8, -11),
                        textcoords="offset points", fontsize=8, color="#A7531E")
    axes[0, 0].annotate(f"{early128:.3g}%", (-.1, early128), xytext=(8, -7),
                        textcoords="offset points", fontsize=8, color="#265F84")
    axes[0, 1].set_ylabel("Loss change (%)")
    axes[0, 1].set_ylim(-.015, .7)
    axes[0, 1].set_yticks([-.006, -.002, 0, .002, .02, .2], ["-0.006", "-0.002", "0", "0.002", "0.02", "0.2"])
    axes[1, 0].set_xticks(range(3), ["1 rank\nhigh load", "1 rank\nlow load", "4 ranks\nbalanced"])
    axes[1, 0].set_xlim(-.4, 2.4)
    axes[1, 0].set_ylim(-.007, .0055)
    axes[1, 0].set_yticks([-.006, -.004, -.002, 0, .002, .004])
    axes[1, 0].set_ylabel("Loss change (%)")
    axes[1, 0].set_xlabel("Case (reference: Restart)")
    axes[1, 1].set_xticks(range(3), ["1 rank\nhigh load", "1 rank\nlow load", "4 ranks\nbalanced"])
    axes[1, 1].set_xlim(-.4, 2.4)
    axes[1, 1].set_ylim(-.0003, .0054)
    axes[1, 1].set_yticks([0, .001, .002, .003, .004, .005])
    axes[1, 1].set_ylabel("Peak loss increase (%)")
    axes[1, 1].set_xlabel("Case (reference: Restart)")
    for ax in axes[1]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    handles = [Line2D([], [], linestyle="none", marker=m, color=c, markersize=6,
                      label=f"{e} experts / seed {s}") for s, e, c, m, _ in configs]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .98),
               ncol=2, frameon=False, fontsize=9)
    fig.text(.55, .91, "Full Hybrid vs Restart; old expert weights and optimizer",
             ha="center", fontsize=8.5, color="#4B5560")
    pdf = OUT / "quality_full_state_500.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / "plot_experts.png", dpi=180)
    plt.close(fig)
    report = dict(source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (source, summary_path)},
                  comparisons=len(rows), distinct_hybrid_branches=10, actual_restart_pairs=10,
                  expert_age_steps=50, continuation_steps=500,
                  protocol_confirmation="Author confirmed all reruns use full-state Hybrid and whole-job Restart on 2026-10-02; raw rerun exports not supplied.",
                  state_audit_files_available=False, complete_run_Y_available=False,
                  calibrated_run_risk_available=False, configuration_results={})
    for seed, experts, *_ in configs:
        these = [r for r in rows if int(r["seed"]) == seed and r["reference_arm"] == "restart"]
        restart = get(seed, 5050, reference="restart")
        report["configuration_results"][str(seed)] = dict(
            num_experts=experts, maximum_sampled_peak_percent=max(percent(r, peak_key) for r in these),
            maximum_peak_tolerance_fraction=max(float(r["eta_peak_fraction"]) for r in these),
            stage_peak_percent={str(t): percent(get(seed, t), peak_key) for t in (1050, 5050, 9050)},
            stage_endpoint_percent={str(t): percent(get(seed, t), end_key) for t in (1050, 5050, 9050)},
            restart_sampled_peak_percent=percent(restart, peak_key),
            restart_endpoint_percent=percent(restart, end_key))
    (OUT / "experts_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
