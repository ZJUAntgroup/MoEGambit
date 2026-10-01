"""Frozen short, real-training experiment; no synthetic performance numbers."""
import hashlib
import json
import os
from pathlib import Path
from moc_timing_plan import Plan


def manifest():
    base = Plan.from_env()
    cadence = int(os.environ.get("MOC_CKPT_INTERVAL", "200"))
    gap = int(os.environ.get("MOC_FAILURE_GAP", "5"))
    tail = int(os.environ.get("MOC_POST_FAILURE_STEPS", "10"))
    counts = [int(x) for x in os.environ.get("MOC_FAILURE_COUNTS", "1").split(",")]
    if cadence < 1 or gap < 1 or tail < 1 or base.step + cadence + gap + tail > 10000:
        raise ValueError("invalid checkpoint/failure/endpoint schedule")
    if len(set(counts)) != len(counts) or not counts or any(x not in (1, 2, 3, 4) for x in counts):
        raise ValueError("MOC_FAILURE_COUNTS must contain distinct counts from 1..4")
    if base.persist_k % 8 or base.snapshot_k % 8:
        raise ValueError("K must be divisible by EP=8 for balanced checkpoint work")
    checkpoint = base.step + cadence
    jobs = []
    for repeat in range(base.repeats):
        for count in counts:
            failed = [(repeat * 17 + i) % 64 for i in range(count)]
            # Rotate treatment order to reduce systematic cache/order confounding.
            arms = ["full_sync_native", "pec_sync", "pec_2level_async"]
            shift = repeat % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                jobs.append({"id": f"r{repeat}_f{count}_{arm}", "arm": arm,
                             "repeat": repeat, "failed_ranks": failed})
    cfg = base.manifest()
    cfg.update(schema_version=2, scope="independent Megatron MoC mechanism port: controlled-restart end-to-end training",
               arms=[{"name": "full_sync_native", "snapshot_k": base.experts, "persist_k": base.experts},
                     {"name": "pec_sync", "snapshot_k": base.persist_k, "persist_k": base.persist_k},
                     {"name": "pec_2level_async", "snapshot_k": base.snapshot_k, "persist_k": base.persist_k}],
               checkpoint_step=checkpoint, failure_step=checkpoint + gap,
               endpoint=checkpoint + gap + tail, checkpoint_interval=cadence,
               failure_counts=counts, jobs=jobs,
               benchmark_training_steps=len(jobs) * (cadence + 2 * gap + tail),
               excluded=["physical failure detection", "cluster scheduler/replacement-node allocation",
                         "original authors' ZeRO-2 integration", "Dynamic-K", "quality/risk guarantee"],
               cpu_cache="node supervisor survives worker relaunch; failed-rank expert cache is discarded",
               replay="all workers restart at globally committed checkpoint; healthy snapshot-PEC, failed persist-PEC",
               clock="node-0 monotonic clock; fault boundary to first resumed commit, caught-up commit, and endpoint",
               cadence_note="paper cadence" if cadence == 200 else "compressed timing smoke; not paper cadence")
    root = Path(__file__).resolve().parents[2]
    files = [root / "examples/moc_system/run_moc_e2e.sh", *sorted((root / "examples/moc_system").glob("moc_e2e_*.py"))]
    cfg["source_sha256"].update({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return cfg


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2))
