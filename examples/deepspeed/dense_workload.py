#!/usr/bin/env python3
"""Synthetic dense DeepSpeed workload for the hot-spare example.

No external model, tokenizer, or dataset is required. The workload deliberately
uses a small dense MLP so the example exercises recovery rather than MoE routing.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--train-iters", type=int, default=8)
    parser.add_argument("--fault-step", type=int, default=5)
    parser.add_argument("--fault-rank", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.fault_step < args.train_iters:
        parser.error("fault step must precede train-iters")
    return args


def epoch() -> int:
    return int(
        os.environ.get(
            "MOEGAMBIT_RECOVERY_EPOCH",
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        )
    )


def replacement_startup() -> bool:
    return (
        epoch() > 0
        and os.environ.get("MOEGAMBIT_DEEPSPEED_INPROCESS_REPLACEMENT", "0") == "1"
    )


def notify(kind: str, rank: int, **payload: object) -> bool:
    from moegambit.runtime.hot_spare import send_worker_event

    return send_worker_event(kind, rank, **payload)


def prepare_process_group(rank: int, world_size: int) -> None:
    """Use the same replacement gate as the regular DeepSpeed workload."""
    if not replacement_startup():
        return
    from moegambit.runtime.distributed import wait_for_recovery_group_barrier

    timeout = float(
        os.environ.get("MOEGAMBIT_DEEPSPEED_GROUP_BARRIER_TIMEOUT", "300")
    )
    notify("rank_recovery_phase", rank, phase="replacement_default_pg_retired_gate_start")
    wait_for_recovery_group_barrier(
        "default_pg_retired", ordinal=-2, rank=rank,
        world_size=world_size, timeout_seconds=timeout,
    )
    notify("rank_recovery_phase", rank, phase="replacement_default_pg_init_start")


def finish_process_group(rank: int, world_size: int) -> None:
    if not replacement_startup():
        return
    from moegambit.runtime.distributed import wait_for_recovery_group_barrier

    timeout = float(
        os.environ.get("MOEGAMBIT_DEEPSPEED_GROUP_BARRIER_TIMEOUT", "300")
    )
    notify("rank_recovery_phase", rank, phase="replacement_default_pg_init_done")
    wait_for_recovery_group_barrier(
        "default_pg_initialized", ordinal=-1, rank=rank,
        world_size=world_size, timeout_seconds=timeout,
    )
    notify("rank_recovery_phase", rank, phase="replacement_default_pg_initialized_gate_done")


def inject_failure(args: argparse.Namespace, engine: object, rank: int) -> None:
    step = int(engine.global_steps)
    if epoch() != 0 or step != args.fault_step:
        return
    runtime = getattr(engine, "_moegambit_runtime", None)
    if runtime is None:
        raise RuntimeError("DeepSpeed recovery runtime was not attached")
    runtime.wait_for_failure_commit(step)
    dist.barrier()
    if rank != args.fault_rank:
        runtime.wait_for_recovery_preemption(step)
        if not runtime.settings.inprocess_recovery:
            raise RuntimeError("survivor preemption returned without in-process recovery")
        return
    marker = Path(args.state_dir) / "fault_injected.json"
    marker.write_text(
        json.dumps({"rank": rank, "step": step, "recovery_epoch": epoch()}) + "\n",
        encoding="utf-8",
    )
    notify("rank_failure", rank, reason="injected_sigkill", global_step=step)
    os.kill(os.getpid(), signal.SIGKILL)


class DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def main() -> None:
    args = args_from_cli()
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    startup_rank = int(os.environ["RANK"])
    startup_world = int(os.environ["WORLD_SIZE"])
    prepare_process_group(startup_rank, startup_world)

    import deepspeed
    from moegambit.runtime.hot_spare import report_worker_phase

    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    finish_process_group(rank, world_size)
    report_worker_phase("distributed_ready", rank)
    torch.manual_seed(1234)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)
    report_worker_phase("model_build_start", rank)
    if replacement_startup():
        notify("rank_recovery_phase", rank, phase="replacement_model_build_start")
    model = DenseMLP()
    report_worker_phase("model_build_done", rank)
    if replacement_startup():
        notify("rank_recovery_phase", rank, phase="replacement_model_build_done")

    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 4 * world_size,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8},
        },
        "zero_optimization": {"stage": 1},
    }
    report_worker_phase("engine_init_start", rank)
    if replacement_startup():
        notify("rank_recovery_phase", rank, phase="replacement_engine_init_start")
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    report_worker_phase("engine_init_done", rank)
    if replacement_startup():
        notify("rank_recovery_phase", rank, phase="replacement_engine_init_done")
    runtime = getattr(engine, "_moegambit_runtime", None)
    if runtime is None:
        raise RuntimeError("DeepSpeed recovery runtime was not attached")
    if not getattr(runtime.settings, "inprocess_replacement", False):
        if local_rank == 0:
            notify("worker_ready", rank, global_step=int(engine.global_steps))
        report_worker_phase("train_barrier_start", rank)
        dist.barrier()
        report_worker_phase("train_barrier_done", rank)

    while int(engine.global_steps) < args.train_iters:
        step = int(engine.global_steps) + 1
        if step == 1:
            report_worker_phase("first_iteration_start", rank)
        generator = torch.Generator(device="cpu").manual_seed(1234 + step * 1000 + rank)
        values = torch.randn(4, 128, generator=generator).to(engine.device)
        prediction = engine(values)
        loss = prediction.float().square().mean()
        engine.backward(loss)
        engine.step()
        if step == 1:
            report_worker_phase("first_iteration_done", rank)
        if rank == 0:
            print(
                f"[dense-deepspeed] step={engine.global_steps} loss={float(loss):.6f}",
                flush=True,
            )
        inject_failure(args, engine, rank)

    if runtime.zero2 is not None:
        for manager in runtime.zero2.managers.values():
            manager.wait_until_replicated(int(engine.global_steps))
    dist.barrier()
    if rank == 0:
        result = {
            "global_step": int(engine.global_steps),
            "restart_count": epoch(),
            "model_kind": "dense",
            "recovery_contract": runtime.recovery_contract,
        }
        path = Path(args.state_dir) / "completed.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[dense-deepspeed] complete: {path}", flush=True)
    runtime.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"[dense-deepspeed] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
