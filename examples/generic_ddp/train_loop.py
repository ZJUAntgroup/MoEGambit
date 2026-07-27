"""Worked example: elastic recovery around a plain PyTorch DDP loop.

This is the DualPipe-style part of the project.  DualPipe does not hide its
scheduling inside a framework fork; it shows the loop you are expected to
write.  Same idea here: recovery needs to know where your safe points are, and
only your training loop knows that.

Run (single process, no recovery -- proves the hooks are inert when disabled)::

    python examples/generic_ddp/train_loop.py

The five hook calls below are the *entire* public integration surface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from a source checkout without `pip install -e .`
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moegambit.config import RuntimeConfig  # noqa: E402
from moegambit.runtime.runtime import initialize  # noqa: E402


def build_everything():
    """Build model/optimizer/loader. Torch is imported lazily on purpose."""
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    if os.environ.get("WORLD_SIZE") and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [(torch.randn(8, 16), torch.randint(0, 2, (8,))) for _ in range(20)]
    return model, optimizer, batches


def load_cold_relaunch_checkpoint(torch, model, optimizer) -> None:
    """Consume the NodeAgent checkpoint contract before training resumes."""

    if os.environ.get("MOEGAMBIT_CHECKPOINT_RELAUNCH", "0") != "1":
        return
    locator = os.environ.get("MOEGAMBIT_CHECKPOINT_LOCATOR", "")
    if not locator:
        raise RuntimeError("cold relaunch has no checkpoint locator")
    payload = torch.load(locator, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])


def commit_checkpoint(torch, runtime, model, optimizer, resume_step: int) -> None:
    """Atomically publish a checkpoint, then make it eligible for fallback."""

    directory = os.environ.get("MOEGAMBIT_CHECKPOINT_DIR", "")
    interval = int(os.environ.get("MOEGAMBIT_CHECKPOINT_INTERVAL", "0"))
    if not directory or interval <= 0 or resume_step % interval:
        return
    target_dir = Path(directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"step-{resume_step}.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save(
        {
            "resume_step": resume_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, target)
    # This call happens only after the atomic rename.  A partially written
    # file can therefore never become a relaunch target.
    runtime.record_checkpoint(str(target), resume_step)


def main() -> int:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("torch is not installed; this example needs it to run a real step.")
        print("The hook *shape* below is what matters -- see the source.")
        return 0

    from moegambit.adapters.generic_ddp import (
        RebindableModel,
        build_generic_ddp_adapter,
    )

    model, optimizer, batches = build_everything()
    load_cold_relaunch_checkpoint(torch, model, optimizer)
    if hasattr(model, "reducer"):
        # DDP reconstruction creates a new wrapper/reducer.  The loop keeps a
        # stable owner and therefore automatically calls the rebuilt wrapper.
        old_optimizer = optimizer
        model = RebindableModel(model)
        optimizer = old_optimizer

    config = RuntimeConfig.from_env()
    adapter = build_generic_ddp_adapter(module=model, optimizer=optimizer)
    runtime = initialize(adapter=adapter, config=config)

    print(f"moegambit enabled={runtime.enabled} resume_step={runtime.resume_step}")

    for step, (inputs, targets) in enumerate(
        batches[runtime.resume_step :], start=runtime.resume_step
    ):
        # 1. Safe stopping point: no collective is in flight here.
        step = runtime.iteration_boundary(step)

        try:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()

            # 2. Last moment the pre-step state still exists.
            runtime.before_optimizer_step(step)

            optimizer.step()

            # 3. New state exists -- publish its version.
            runtime.after_optimizer_step(step, committed=True)

            # 4. One full iteration done: commit the recovery epoch.
            runtime.commit_iteration(step)

            # The saved state resumes at the next loop cursor.  Publishing it
            # after commit_iteration preserves optimizer/step consistency.
            commit_checkpoint(
                torch,
                runtime,
                model,
                optimizer,
                resume_step=step + 1,
            )

        except RuntimeError as exc:
            # 5. Ask the runtime whether this was a recoverable failure.
            #    Re-raising when it says no is the point: an unclassified error
            #    is a bug, and swallowing it would hide that bug.
            if not runtime.on_distributed_error(exc):
                raise
            continue

        if step % 5 == 0:
            print(f"step {step:>3}  loss {loss.item():.4f}")

    print("done:", dict(runtime.describe()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
