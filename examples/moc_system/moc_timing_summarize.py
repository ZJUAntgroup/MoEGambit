"""Strict coverage checks; summarize max-rank durations, never average ranks."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from moc_timing_launch import atomic_json


def quantile(values, p):
    values = sorted(values)
    if not values:
        raise ValueError("empty timing series")
    i = (len(values) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def summarize(root):
    plan = json.loads((root / "plan.json").read_text())
    world = plan["nnodes"] * plan["per_node"]
    expected_ranks = set(range(world))
    records = []
    for rank in range(world):
        path = root / "ranks" / f"rank_{rank:03d}.jsonl"
        if not path.is_file():
            raise ValueError(f"rank timing file missing: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(row.get("rank") != rank for row in rows):
            raise ValueError(f"rank identity mismatch: {path}")
        for row in rows:
            for key, value in row.items():
                if key.endswith("_s") or "bytes" in key and isinstance(value, (int, float)):
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid timing/byte measurement: {path}/{key}")
        smoke = json.loads((root / "ranks" / f"smoke.rank_{rank:03d}.json").read_text())
        if smoke["rank"] != rank or smoke["count"] != plan["smoke_steps"]:
            raise ValueError(f"incomplete smoke check on rank {rank}")
        if [r["step"] for r in smoke["rows"]] != list(range(plan["step"] + 1, plan["step"] + plan["smoke_steps"] + 1)):
            raise ValueError(f"smoke step alignment differs on rank {rank}")
        records.extend(rows)
    keys, sample_rows = set(), []
    for arm in plan["arms"]:
        name = arm["name"]
        for event in ("snapshot", "persist", "restore"):
            for repeat in range(plan["repeats"]):
                counts = plan["failure_counts"] if event == "restore" else [0]
                for count in counts:
                    group = [r for r in records if r.get("arm") == name and r["event"] == event
                             and r.get("repeat") == repeat and r.get("failed_count", 0) == count]
                    if len(group) != world or {r["rank"] for r in group} != expected_ranks:
                        raise ValueError(f"missing/duplicate ranks: {name}/{event}/{repeat}/{count}")
                    for row in group:
                        key = (name, event, repeat, count, row["rank"])
                        if key in keys:
                            raise ValueError(f"duplicate measurement {key}")
                        keys.add(key)
                    out = {"arm": name, "event": event, "repeat": repeat, "failed_count": count}
                    if event == "snapshot":
                        preflights = [r for r in records if r["event"] == "preflight"]
                        if preflights:
                            layers = {r["expert_layer_count"] for r in preflights}
                            if len(layers) != 1:
                                raise ValueError("MoE layer count differs across ranks")
                            layers = layers.pop()
                            if (sum(r["snapshot_expert_units_local"] for r in group) != layers * arm["snapshot_k"]
                                    or sum(r["persist_expert_units_local"] for r in group) != layers * arm["persist_k"]):
                                raise ValueError("snapshot/persist did not select exactly K experts per layer")
                        out.update(snapshot_max_s=max(r["snapshot_local_s"] for r in group),
                                   enqueue_max_s=max(r["enqueue_local_s"] for r in group),
                                   backpressure_max_s=max(r["backpressure_local_s"] for r in group),
                                   snapshot_tensor_bytes=sum(r["snapshot_tensor_bytes_local"] for r in group))
                    elif event == "persist":
                        out.update(persist_max_s=max(r["persist_local_s"] for r in group),
                                   written_bytes=sum(r["written_bytes_local"] for r in group))
                    else:
                        if (len({tuple(r["failed_ranks"]) for r in group}) != 1
                                or any(r["verified_entries"] <= 0 or r.get("dense_replicas_verified") is not True for r in group)
                                or sum(r["source"] == "storage" for r in group) != count):
                            raise ValueError(f"invalid simulated-failure coverage: {name}/{repeat}/{count}")
                        out.update(restore_max_s=max(r["restore_global_max_s"] for r in group),
                                   storage_read_bytes=sum(r["storage_read_bytes_local"] for r in group),
                                   failed_ranks=",".join(map(str, group[0]["failed_ranks"])))
                    sample_rows.append(out)
    timed = [r for r in records if r["event"] in ("snapshot", "persist", "restore")]
    if len(timed) != len(keys):
        raise ValueError("unexpected timed records outside the registered plan")
    def write_csv(path, rows):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_csv(root / "timing_samples.csv", sample_rows)
    summaries = []
    for arm in plan["arms"]:
        name = arm["name"]
        saving = [r for r in sample_rows if r["arm"] == name and r["event"] == "snapshot"]
        persisting = [r for r in sample_rows if r["arm"] == name and r["event"] == "persist"]
        for count in plan["failure_counts"]:
            values = [r["restore_max_s"] for r in sample_rows
                      if r["arm"] == name and r["event"] == "restore" and r["failed_count"] == count]
            summaries.append({"arm": name, "failed_count": count, "n": len(values),
                              "snapshot_median_s": statistics.median(r["snapshot_max_s"] for r in saving),
                              "enqueue_median_s": statistics.median(r["enqueue_max_s"] for r in saving),
                              "persist_median_s": statistics.median(r["persist_max_s"] for r in persisting),
                              "written_bytes_median": statistics.median(r["written_bytes"] for r in persisting),
                              "restore_median_s": statistics.median(values),
                              "restore_p95_s": quantile(values, .95),
                              "restore_min_s": min(values), "restore_max_s": max(values)})
    write_csv(root / "timing_summary.csv", summaries)
    output = {"complete": True, "scope": plan["scope"], "plan": plan,
              "summary": summaries, "ranks": world, "smoke_steps": plan["smoke_steps"],
              "all_state_checks_passed": True,
              "interpretation": "descriptive timing only; fixed weights, uncontrolled cache, no full-system speedup or quality conclusion"}
    atomic_json(root / "timing_summary.json", output)
    atomic_json(root / "COMPLETE.json", {"complete": True, "summary": "timing_summary.json", "ranks": world})
    print(f"[moc-timing] COMPLETE: {root}/timing_summary.csv", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    summarize(parser.parse_args().result_dir)
