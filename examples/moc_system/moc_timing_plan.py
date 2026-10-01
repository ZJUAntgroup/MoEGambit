"""Dependency-free configuration and ownership rules for the MoC timing port."""
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import re
import hashlib
import subprocess

EXPERT = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.local_experts\.(\d+)\.")


def selected_experts(n, ep, k, round_index, layer, stride=None):
    """EP-interleaved sequential selection, staggered between layers.

    Megatron assigns contiguous expert IDs to EP ranks. Walk EP ranks first,
    then local experts, so small K does not always burden EP rank zero.
    """
    if not (n > 0 and ep > 0 and n % ep == 0 and 1 <= k <= n):
        raise ValueError("invalid N/EP/K")
    width = n // ep
    start = (round_index + layer) * (k if stride is None else stride)
    return {(j % ep) * width + j // ep
            for j in ((start + i) % n for i in range(k))}


def dense_owners(unit_bytes, count):
    """Coarse module sharding, largest module first; deterministic ties."""
    if count < 1:
        raise ValueError("empty DP group")
    loads = [0] * count
    result = {}
    for key, size in sorted(unit_bytes.items(), key=lambda item: (-item[1], item[0])):
        owner = min(range(count), key=lambda i: (loads[i], i))
        result[key] = owner
        loads[owner] += size
    return result


def outside(child, parent):
    a, b = Path(child).resolve(), Path(parent).resolve()
    return a != b and b not in a.parents


@dataclass(frozen=True)
class Plan:
    run_id: str
    base_dir: str
    result_dir: str
    scratch: str
    step: int
    repeats: int
    smoke_steps: int
    experts: int
    snapshot_k: int
    persist_k: int
    nnodes: int
    per_node: int
    fsync: bool
    keep_scratch: bool
    profile: str

    @classmethod
    def from_env(cls):
        run = os.environ.get("RUN_ID", "moc_timing_01")
        log = os.environ.get("LOG_DIR", "/personal/moegambit/moc_timing")
        plan = cls(
            run, os.environ.get("BASE_DIR", "/shared/moegambit/baseline/seed_1234"),
            str(Path(log) / run),
            os.environ.get("MOC_CKPT_ROOT", os.environ.get("FSE_CKPT_ROOT", f"/shared/moegambit/moc_timing_ckpts/{run}")),
            int(os.environ.get("MOC_START_STEP", "4000")),
            int(os.environ.get("MOC_REPEATS", "1")),
            int(os.environ.get("MOC_SMOKE_STEPS", "5")),
            int(os.environ.get("NUM_EXPERTS", "128")),
            int(os.environ.get("MOC_SNAPSHOT_K", "32")),
            int(os.environ.get("MOC_PERSIST_K", "16")),
            int(os.environ.get("NNODES", "8")),
            int(os.environ.get("NPROC_PER_NODE", "8")),
            os.environ.get("MOC_FSYNC", "1") == "1",
            os.environ.get("MOC_KEEP_SCRATCH", "0") == "1",
            os.environ.get("MOC_MODEL_PROFILE", os.environ.get("FSE_MODEL_PROFILE", "qwen3_48x2048")),
        )
        plan.validate()
        return plan

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("invalid RUN_ID")
        log = Path(self.result_dir)
        if not log.is_absolute() or not log.is_relative_to("/personal"):
            raise ValueError("results must be under /personal")
        if not Path(self.base_dir).is_absolute() or not Path(self.scratch).is_absolute():
            raise ValueError("BASE_DIR and MOC_CKPT_ROOT must be absolute")
        paths = [self.base_dir, self.result_dir, self.scratch]
        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                if not outside(a, b) or not outside(b, a):
                    raise ValueError("baseline, results and scratch must be disjoint")
        if self.nnodes * self.per_node != 64 or not 1 <= self.per_node <= 8:
            raise ValueError("existing PP=8 / EP=8 baseline requires 64 ranks")
        if not 0 < self.step < 10000 or self.step % 200:
            raise ValueError("MOC_START_STEP must be a positive 200-step checkpoint")
        if not 1 <= self.repeats <= 10 or not 1 <= self.smoke_steps <= 20:
            raise ValueError("repeats must be 1..10 and smoke steps 1..20")
        if self.step + self.smoke_steps > 10000:
            raise ValueError("smoke steps exceed the baseline schedule")
        if not 1 <= self.persist_k <= self.snapshot_k <= self.experts or self.experts % 8:
            raise ValueError("require 1 <= Kpersist <= Ksnapshot <= N and N divisible by EP=8")
        for name in ("MOC_FSYNC", "MOC_KEEP_SCRATCH"):
            if os.environ.get(name, "1" if name == "MOC_FSYNC" else "0") not in ("0", "1"):
                raise ValueError(f"{name} must be 0 or 1")

    def arms(self):
        return [
            {"name": "full_sync", "snapshot_k": self.experts, "persist_k": self.experts, "async": False},
            {"name": "pec_sync", "snapshot_k": self.persist_k, "persist_k": self.persist_k, "async": False},
            {"name": "pec_2level_async", "snapshot_k": self.snapshot_k, "persist_k": self.persist_k, "async": True},
        ]

    def manifest(self):
        repository = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        sources = ["examples/moc_system/run_moc_timing_short.sh", "examples/moc_system/common.sh",
                   "examples/moc_system/model_profile.sh", "Megatron-LM/megatron/training/training.py",
                   "Megatron-LM/megatron/training/checkpointing.py"]
        sources += [str(p.relative_to(repository)) for p in sorted((repository / "examples/moc_system").glob("moc_timing_*.py"))]
        digests = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in sources}
        return {**asdict(self), "schema_version": 1, "arms": self.arms(),
                "source_revision": revision, "source_sha256": digests,
                "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
                "master_port": int(os.environ.get("MASTER_PORT", "20130")),
                "dataset_prefix": os.environ.get("DATA_PATH", "/shared/moegambit/data/my_qwen3_data_text_document"),
                "seed": int(os.environ.get("SEED", "1234")),
                "failure_counts": [1, 2, 3, 4], "benchmark_training_steps": 0,
                "scope": "fixed-state MoC core-mechanism timing port; not original MoC-System",
                "paper": "https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf",
                "cache_policy": "uncontrolled OS cache; no cold-cache claim",
                "excluded": ["failure detection", "process/group restart", "training replay",
                             "scheduler/RNG/data-iterator reconstruction",
                             "GPU snapshot overlap with forward/backward", "Dynamic-K", "quality risk guarantee"]}


if __name__ == "__main__":
    import json
    print(json.dumps(Plan.from_env().manifest(), indent=2))
