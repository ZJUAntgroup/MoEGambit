"""Small, deterministic DDP workload with recovery hooks and atomic checkpoints.

Run with torchrun for synchronized training, or directly for a CPU smoke test.
The dedicated fault_replacement.py harness exercises real rank replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moegambit.config import RuntimeConfig  # noqa: E402
from moegambit.runtime.runtime import initialize  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--checkpoint-dir", default=os.getenv("MOEGAMBIT_CHECKPOINT_DIR", ""))
    parser.add_argument("--checkpoint-interval", type=int,
                        default=int(os.getenv("MOEGAMBIT_CHECKPOINT_INTERVAL", "5")))
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.checkpoint_interval <= 0:
        parser.error("steps and checkpoint interval must be positive")
    return args


def raw_model(model):
    model = getattr(model, "current", model)
    return getattr(model, "module", model)


def build_everything(torch, device, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    ).to(device)
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def load_cold_relaunch_checkpoint(torch, model, optimizer, seed):
    if os.environ.get("MOEGAMBIT_CHECKPOINT_RELAUNCH", "0").lower() not in {"1", "true", "yes", "on"}:
        return 0
    locator = os.environ.get("MOEGAMBIT_CHECKPOINT_LOCATOR", "")
    if not locator:
        raise RuntimeError("cold relaunch has no checkpoint locator")
    payload = torch.load(locator, map_location="cpu", weights_only=True)
    resume_step = int(payload["resume_step"])
    if resume_step < 0 or payload["seed"] != seed:
        raise ValueError("checkpoint cursor or seed does not match this workload")
    expected = os.environ.get("MOEGAMBIT_CHECKPOINT_STEP")
    if expected is not None and int(expected) != resume_step:
        raise ValueError("checkpoint step does not match the relaunch directive")
    # The saved cursor is the number of committed updates, also the next batch.
    os.environ["MOEGAMBIT_CHECKPOINT_STEP"] = str(resume_step)
    raw_model(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return resume_step


def commit_checkpoint(torch, runtime, model, optimizer, resume_step, args):
    if not args.checkpoint_dir or resume_step % args.checkpoint_interval:
        return
    target = Path(args.checkpoint_dir).expanduser().resolve() / f"step-{resume_step}.pt"
    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    status = [None]
    if rank == 0:
        temporary = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save({"resume_step": resume_step, "seed": args.seed,
                            "model": raw_model(model).state_dict(),
                            "optimizer": optimizer.state_dict()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            status[0] = f"checkpoint publication failed: {type(exc).__name__}: {exc}"
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    status[0] = status[0] or f"checkpoint cleanup failed: {exc}"
    if distributed:
        # All ranks learn whether the single writer published before advertising it.
        torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])
    runtime.record_checkpoint(str(target), resume_step)


def report_completion(torch, record):
    """Collect both rank results and print them through a single stdout writer."""
    records = [record]
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        records = [None] * torch.distributed.get_world_size() if rank == 0 else None
        torch.distributed.gather_object(record, records, dst=0)
        if rank != 0:
            return
    for result in records:
        print("completed: " + json.dumps(result), flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        print("PyTorch is required; install moegambit[torch].", file=sys.stderr)
        return 2

    from moegambit.adapters.generic_ddp import RebindableModel, build_generic_ddp_adapter

    device = torch.device("cpu")
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requires a CUDA-enabled PyTorch installation")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    owns_group = "RANK" in os.environ and not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(args.backend, timeout=timedelta(seconds=60))
    try:
        distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        model, optimizer = build_everything(torch, device, args.seed)
        cursor = load_cold_relaunch_checkpoint(torch, model, optimizer, args.seed)
        if cursor > args.steps:
            raise ValueError("checkpoint is beyond the requested total steps")
        if distributed:
            model = RebindableModel(model)
        adapter = build_generic_ddp_adapter(module=model, optimizer=optimizer,
                                            backend=args.backend, committed_step=cursor)
        runtime = initialize(adapter=adapter, config=RuntimeConfig.from_env())
        print(f"rank={rank} ddp={distributed} enabled={runtime.enabled} resume_step={cursor}", flush=True)
        while cursor < args.steps:
            # Steps are one-based committed update versions; cursor indexes batches.
            step = runtime.iteration_boundary(cursor + 1)
            generator = torch.Generator().manual_seed(args.seed + (step - 1) * world_size + rank)
            inputs = torch.randn(8, 16, generator=generator).to(device)
            targets = torch.randint(0, 2, (8,), generator=generator).to(device)
            try:
                optimizer.zero_grad(set_to_none=True)
                loss = functional.cross_entropy(model(inputs), targets)
                loss.backward()
                runtime.before_optimizer_step(step)
                optimizer.step()
                runtime.after_optimizer_step(step, committed=True)
                runtime.commit_iteration(step)
            except RuntimeError as exc:
                if not runtime.on_distributed_error(exc):
                    raise
                cursor = runtime.resume_step
                continue
            cursor = step
            # A filesystem error must not replay an already committed optimizer step.
            commit_checkpoint(torch, runtime, model, optimizer, cursor, args)
            if rank == 0:
                print(f"step {cursor} loss {loss.item():.4f}", flush=True)
        digest = hashlib.sha256()
        for tensor in raw_model(model).state_dict().values():
            digest.update(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
        report_completion(torch, {"rank": rank, "ddp": distributed,
              "completed_steps": cursor, "model_digest": digest.hexdigest()})
        return 0
    finally:
        if owns_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
