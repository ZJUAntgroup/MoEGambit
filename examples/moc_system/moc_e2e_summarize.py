"""Strict evidence-based end-to-end results, not a sum of component timers."""
import csv
import json
import math
from pathlib import Path
import statistics
from moc_timing_launch import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def valid_time(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid measurement: {value}")
    return value


def validate_job(cfg, job, root):
    rank_count = cfg["nnodes"] * cfg["per_node"]
    fault = read(root / "fault.json")
    if fault["step"] != cfg["failure_step"] or fault["failed_ranks"] != job["failed_ranks"]:
        raise ValueError("failure boundary mismatch")
    before, after, cp, restores = [], [], [], []
    data_checks, replay_loss = 0, []
    for rank in range(rank_count):
        prefix = read(root / "prefix" / f"training.rank_{rank:03d}.json")
        resume = read(root / "resume" / f"training.rank_{rank:03d}.json")
        checkpoint = read(root / "prefix" / f"checkpoint.rank_{rank:03d}.json")
        restoration = read(root / "resume" / f"restore.rank_{rank:03d}.json")
        if prefix["rank"] != rank or resume["rank"] != rank or restoration["rank"] != rank:
            raise ValueError("rank identity mismatch")
        if prefix["pid"] == resume["pid"]:
            raise ValueError("workers did not actually restart")
        for rows, start, end in ((prefix["rows"], cfg["step"], cfg["failure_step"]),
                                 (resume["rows"], cfg["checkpoint_step"], cfg["endpoint"])):
            if [row["step"] for row in rows] != list(range(start + 1, end + 1)):
                raise ValueError("missing, duplicated or out-of-order commits")
            for row in rows:
                valid_time(row["elapsed_s"])
                if any(not math.isfinite(value) for value in row["loss"].values()):
                    raise ValueError("nonfinite loss")
        if restoration["checkpoint_step"] != cfg["checkpoint_step"]:
            raise ValueError("wrong restart point")
        source = "native_full_checkpoint" if job["arm"] == "full_sync_native" else (
            "supervisor_cpu" if job["arm"] == "pec_2level_async" and rank not in job["failed_ranks"]
            else "durable_partial_checkpoint")
        if restoration["source"] != source:
            raise ValueError("wrong failed/healthy recovery source")
        if source != "native_full_checkpoint" and restoration.get("verified_fields", 0) < 1:
            raise ValueError("partial state recovery was not verified")
        if job["arm"] != "full_sync_native":
            if checkpoint.get("persisted_expert_units", 0) < 1 or checkpoint.get("snapshotted_expert_units", 0) < 1:
                raise ValueError("no actual selected experts written/snapshotted")
            if restoration["expert_units"] != (checkpoint["snapshotted_expert_units"] if source == "supervisor_cpu"
                                               else checkpoint["persisted_expert_units"]):
                raise ValueError("partial expert recovery count mismatch")
        for field in ("checkpoint_blocking_s", "written_bytes_local", "failure_boundary_drain_s"):
            valid_time(checkpoint[field])
        relevant = lambda rows: [row for row in rows if cfg["checkpoint_step"] < row["step"] <= cfg["failure_step"]]
        left, right = relevant(prefix["batches"]), relevant(resume["batches"])
        if left != right:
            raise ValueError(f"actual replay tokens/labels differ on rank {rank}")
        if prefix["batches"] and not left:
            raise ValueError("no data fingerprint for replay interval")
        data_checks += len(left)
        p_losses = {row["step"]: row["loss"] for row in prefix["rows"]}
        p_schedule = {row["step"]: row for row in prefix["rows"]}
        for row in resume["rows"]:
            if row["step"] <= cfg["failure_step"]:
                for field in ("consumed_samples_before", "learning_rates_before", "learning_rates_after"):
                    if row[field] != p_schedule[row["step"]][field]:
                        raise ValueError(f"replay sample counter/LR schedule differs on rank {rank}")
                for key, value in row["loss"].items():
                    baseline = p_losses[row["step"]][key]
                    replay_loss.append(abs(value - baseline) / max(abs(baseline), 1e-12))
        before.append(prefix)
        after.append(resume)
        cp.append(checkpoint)
        restores.append(restoration)
    if not data_checks or not replay_loss:
        raise ValueError("missing actual data fingerprints or measured training losses")
    start = read(root / "prefix" / "first_step_start.json")["monotonic"]
    failure = fault["monotonic"]
    marks = {step: read(root / "resume" / f"commit_{step}.json")["monotonic"]
             for step in (cfg["checkpoint_step"] + 1, cfg["failure_step"], cfg["failure_step"] + 1, cfg["endpoint"])}
    if not start < failure < marks[cfg["checkpoint_step"] + 1] <= marks[cfg["failure_step"]] <= marks[cfg["failure_step"] + 1] <= marks[cfg["endpoint"]]:
        raise ValueError("inconsistent node-0 monotonic timeline")
    elapsed = valid_time(marks[cfg["endpoint"]] - start)
    if not elapsed:
        raise ValueError("zero end-to-end duration")
    gpu_overlap = [valid_time(row.get("gpu_timeline_overlap_ms", 0)) for row in cp]
    return {"job_id": job["id"], "arm": job["arm"], "repeat": job["repeat"],
            "failed_rank_count": len(job["failed_ranks"]), "failed_ranks": job["failed_ranks"],
            "window_e2e_s": elapsed, "before_failure_s": failure - start,
            "recovery_first_resumed_commit_s": marks[cfg["checkpoint_step"] + 1] - failure,
            "recovery_caught_up_s": marks[cfg["failure_step"]] - failure,
            "recovery_first_new_commit_s": marks[cfg["failure_step"] + 1] - failure,
            "recovery_to_endpoint_s": marks[cfg["endpoint"]] - failure,
            "replayed_steps": cfg["failure_step"] - cfg["checkpoint_step"],
            "actual_training_updates": sum(len(x["rows"]) for x in (before[0], after[0])),
            "unique_committed_steps": cfg["endpoint"] - cfg["step"],
            "effective_steps_per_s": (cfg["endpoint"] - cfg["step"]) / elapsed,
            "checkpoint_blocking_max_rank_s": max(row["checkpoint_blocking_s"] for row in cp),
            "checkpoint_written_bytes": sum(row["written_bytes_local"] + row.get("context_bytes", 0) for row in cp),
            "failure_boundary_drain_max_rank_s": max(row["failure_boundary_drain_s"] for row in cp),
            "gpu_overlap_observed_ranks": sum(value > 0 for value in gpu_overlap),
            "gpu_overlap_max_ms": max(gpu_overlap),
            "replay_data_fingerprints_verified": data_checks,
            "replay_sample_counters_and_lr_match": True,
            "max_relative_replay_training_loss_difference": max(replay_loss),
            "all_workers_restarted": True, "all_rank_records_complete": True,
            "interpretation": "Controlled short-run end-to-end timing for independent fixed-K port; no validation-quality guarantee"}


def summarize(root):
    root = Path(root)
    cfg = read(root / "plan.json")
    results = [validate_job(cfg, job, root / job["id"]) for job in cfg["jobs"]]
    for result in results:
        baseline = next(row for row in results if row["arm"] == "full_sync_native" and row["repeat"] == result["repeat"]
                        and row["failed_rank_count"] == result["failed_rank_count"])
        result["window_speedup_vs_native_full"] = baseline["window_e2e_s"] / result["window_e2e_s"]
        result["recovery_speedup_vs_native_full"] = baseline["recovery_caught_up_s"] / result["recovery_caught_up_s"]
        for phase in ("prefix", "resume"):
            for rank in range(cfg["nnodes"] * cfg["per_node"]):
                baseline_data = read(root / baseline["job_id"] / phase / f"training.rank_{rank:03d}.json")["batches"]
                treatment_data = read(root / result["job_id"] / phase / f"training.rank_{rank:03d}.json")["batches"]
                if treatment_data != baseline_data:
                    raise ValueError(f"cross-arm data alignment failed: {result['job_id']}/{phase}/rank_{rank}")
        result["cross_arm_data_fingerprints_match"] = True
    columns = [key for key, value in results[0].items() if isinstance(value, (str, float, int, bool))]
    with (root / "e2e_results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    groups = []
    for count in cfg["failure_counts"]:
        for arm in ("full_sync_native", "pec_sync", "pec_2level_async"):
            rows = [row for row in results if row["arm"] == arm and row["failed_rank_count"] == count]
            groups.append({"arm": arm, "failed_rank_count": count, "n": len(rows),
                           **{field + "_median": statistics.median(row[field] for row in rows) for field in
                              ("window_e2e_s", "recovery_caught_up_s", "recovery_first_new_commit_s",
                               "checkpoint_written_bytes", "window_speedup_vs_native_full")}})
    atomic_json(root / "e2e_summary.json", {"scope": cfg["scope"], "complete": True, "results": results,
                "groups": groups, "steps": cfg["benchmark_training_steps"], "excluded": cfg["excluded"],
                "measurement": cfg["clock"], "checkpoint_interval": cfg["checkpoint_interval"],
                "cache_policy": cfg["cache_policy"], "cadence_note": cfg["cadence_note"]})
    lines = ["# MoC port: measured end-to-end results", "", cfg["scope"], "",
             "| Arm | Failed ranks | Runs | Window (s) | Recovery + replay (s) | Window speedup |",
             "|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(f"| {group['arm']} | {group['failed_rank_count']} | {group['n']} | {group['window_e2e_s_median']:.3f} | "
                     f"{group['recovery_caught_up_s_median']:.3f} | {group['window_speedup_vs_native_full_median']:.3f}× |")
    lines += ["", "Window: first training step to matched endpoint, including checkpoint blocking, persistence drain, worker exit/relaunch, "
              "model loading, communication-group initialization, data rebuilding and real replay.", "",
              "The failure is controlled and delayed until the checkpoint is durable. Detection and cluster scheduler delay are excluded. "
              "CPU cache transfer and correctness checks are included in observed time. Absolute timings describe this port, not the authors' implementation.",
              "", "One repetition is a functionality/timing observation. Training loss is diagnostic; this benchmark does not prove validation quality or statistical risk."]
    (root / "e2e_report.md").write_text("\n".join(lines) + "\n")
    plot_status = "matplotlib unavailable; CSV/JSON/Markdown results are complete"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
        labels = [f"{row['arm']}\nf={row['failed_rank_count']}" for row in groups]
        for axis, field, title in ((axes[0], "window_e2e_s_median", "Matched training window"),
                                   (axes[1], "recovery_caught_up_s_median", "Recovery and replay")):
            axis.bar(range(len(groups)), [row[field] for row in groups], color=["#777777", "#4c78a8", "#2a9d8f"] * len(cfg["failure_counts"]))
            axis.set_xticks(range(len(groups)), labels, rotation=35, ha="right", fontsize=7)
            axis.set_ylabel("Wall time (s)")
            axis.set_title(title)
            axis.spines[["top", "right"]].set_visible(False)
        fig.savefig(root / "e2e_timing.pdf")
        fig.savefig(root / "e2e_timing.svg")
        plt.close(fig)
        plot_status = "e2e_timing.pdf and e2e_timing.svg generated"
    atomic_json(root / "COMPLETE.json", {"jobs": len(results), "results": "e2e_summary.json", "plot": plot_status})
    print((root / "e2e_report.md").read_text(), flush=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    summarize(parser.parse_args().result_dir)
