#!/usr/bin/env python3
"""Summarize BSR-MoE recovery ablation results.

Expected input layout:

    <log_root>/full_checkpoint/timing_report.json
    <log_root>/selective_sync_opt/timing_report.json
    <log_root>/selective_deferred/timing_report.json

The default attribution uses checkpoint-corrected fault-window overhead:

    selective_restore_gain = full_checkpoint - selective_sync_opt
    optimizer_defer_gain   = selective_sync_opt - selective_deferred
    total_gain             = full_checkpoint - selective_deferred

This gives an additive decomposition of why the production BSR path is
faster than a full checkpoint load under the same fault-injection schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_MODES = ("full_checkpoint", "selective_sync_opt", "selective_deferred")


def load_report(log_root: Path, mode: str) -> dict[str, Any]:
    path = log_root / mode / "timing_report.json"
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["_report_path"] = str(path)
    return report


def average_summary_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in report.get("average_summary", [])}


def metric_mean(report: dict[str, Any], metric: str) -> tuple[float | None, str]:
    rows = average_summary_map(report)
    row = rows.get(metric)
    if row is None:
        return None, ""
    mean = row.get("mean")
    unit = row.get("unit", "")
    if mean is None:
        return None, unit
    return float(mean), unit


def fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "ms":
        return f"{value / 1000.0:.3f}s"
    if unit == "s":
        return f"{value:.3f}s"
    return f"{value:.3f}{unit}"


def build_summary(
    reports: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    values: dict[str, float | None] = {}
    unit = ""
    for mode, report in reports.items():
        value, row_unit = metric_mean(report, metric)
        values[mode] = value
        unit = unit or row_unit

    full = values.get("full_checkpoint")
    sync = values.get("selective_sync_opt")
    deferred = values.get("selective_deferred")

    scope_gain = full - sync if full is not None and sync is not None else None
    defer_gain = sync - deferred if sync is not None and deferred is not None else None
    total_gain = full - deferred if full is not None and deferred is not None else None

    def pct(part: float | None) -> float | None:
        if part is None or total_gain is None or total_gain == 0:
            return None
        return 100.0 * part / total_gain

    return {
        "metric": metric,
        "unit": unit,
        "mode_means": values,
        "gains": {
            "selective_restore_gain": scope_gain,
            "optimizer_defer_gain": defer_gain,
            "total_gain": total_gain,
        },
        "gain_percent": {
            "selective_restore_gain": pct(scope_gain),
            "optimizer_defer_gain": pct(defer_gain),
        },
        "reports": {
            mode: {
                "status": report.get("status"),
                "notes": report.get("notes", []),
                "counts": report.get("counts", {}),
                "path": report.get("_report_path"),
            }
            for mode, report in reports.items()
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    unit = summary["unit"]
    print("BSR-MoE Ablation Summary")
    print(f"metric: {summary['metric']} ({unit})")
    print("")
    print("Mode means")
    for mode in DEFAULT_MODES:
        print(f"- {mode}: {fmt(summary['mode_means'].get(mode), unit)}")
    print("")
    print("Attribution")
    gains = summary["gains"]
    pcts = summary["gain_percent"]
    print(
        "- selective stale-expert restore gain: "
        f"{fmt(gains['selective_restore_gain'], unit)} "
        f"({pcts['selective_restore_gain']:.1f}% of total)"
        if pcts["selective_restore_gain"] is not None
        else "- selective stale-expert restore gain: n/a"
    )
    print(
        "- optimizer defer/weight-first gain: "
        f"{fmt(gains['optimizer_defer_gain'], unit)} "
        f"({pcts['optimizer_defer_gain']:.1f}% of total)"
        if pcts["optimizer_defer_gain"] is not None
        else "- optimizer defer/weight-first gain: n/a"
    )
    print(f"- total gain: {fmt(gains['total_gain'], unit)}")
    print("")
    print("Status")
    for mode, meta in summary["reports"].items():
        notes = "; ".join(meta.get("notes") or [])
        print(f"- {mode}: {meta.get('status')} ({notes})")


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    unit = summary["unit"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "name", "value", "unit", "percent_of_total"],
        )
        writer.writeheader()
        for mode, value in summary["mode_means"].items():
            writer.writerow(
                {
                    "kind": "mode_mean",
                    "name": mode,
                    "value": value,
                    "unit": unit,
                    "percent_of_total": "",
                }
            )
        for name, value in summary["gains"].items():
            writer.writerow(
                {
                    "kind": "gain",
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "percent_of_total": summary["gain_percent"].get(name, ""),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", help="Ablation log root containing mode subdirectories")
    parser.add_argument(
        "--metric",
        default="fault.window.checkpoint_corrected_overhead",
        help="average_summary metric to compare",
    )
    parser.add_argument("--json", help="Write combined ablation JSON report")
    parser.add_argument("--csv", help="Write combined ablation CSV report")
    args = parser.parse_args()

    log_root = Path(args.log_root).expanduser().resolve()
    reports = {mode: load_report(log_root, mode) for mode in DEFAULT_MODES}
    summary = build_summary(reports, args.metric)
    print_summary(summary)

    if args.json:
        out = Path(args.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv:
        out = Path(args.csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
