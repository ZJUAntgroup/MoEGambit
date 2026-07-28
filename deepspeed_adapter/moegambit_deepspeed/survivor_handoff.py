"""Current-step survivor state preserved across a DeepSpeed epoch relaunch."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

from moegambit.runtime.recovery_handoff import (
    PROTOCOL_VERSION,
    rank_state_path,
)


class DeepSpeedSurvivorHandoffError(RuntimeError):
    pass


def _cpu_model_state(module: Any) -> dict[str, Any]:
    state = {}
    for name, tensor in module.state_dict().items():
        source = tensor.detach()
        state[name] = source.to(device="cpu", copy=True)
    return state


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(
            torch.cuda.current_device()
        )
    try:
        import numpy

        state["numpy"] = numpy.random.get_state()
    except ImportError:
        pass
    return state


def apply_rng_state(state: dict[str, Any]) -> None:
    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(
            state["torch_cuda"], torch.cuda.current_device()
        )
    if "numpy" in state:
        try:
            import numpy

            numpy.random.set_state(state["numpy"])
        except ImportError:
            pass


def capture_engine_state(engine: Any) -> dict[str, Any]:
    scheduler = getattr(engine, "lr_scheduler", None)
    scheduler_state = (
        scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )
    zero_optimizer = getattr(engine, "optimizer", None)
    base_optimizer = getattr(zero_optimizer, "optimizer", None)
    param_groups = []
    if base_optimizer is not None:
        for group in base_optimizer.param_groups:
            param_groups.append(
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
            )
    return {
        "global_steps": int(getattr(engine, "global_steps", -1)),
        "global_samples": int(getattr(engine, "global_samples", 0)),
        "skipped_steps": int(getattr(engine, "skipped_steps", 0)),
        "engine_runtime": {
            name: getattr(engine, name)
            for name in (
                "micro_steps",
                "gas_boundary_ctr",
                "_is_gradient_accumulation_boundary",
                "_force_grad_boundary",
            )
            if hasattr(engine, name)
        },
        "scheduler": scheduler_state,
        "optimizer_param_groups": param_groups,
        "optimizer_runtime": {
            name: getattr(zero_optimizer, name)
            for name in (
                "dynamic_loss_scale",
                "overflow",
                "clip_grad",
                "loss_scaler",
            )
            if zero_optimizer is not None
            and hasattr(zero_optimizer, name)
        },
    }


def apply_engine_state(engine: Any, state: dict[str, Any]) -> None:
    step = int(state["global_steps"])
    engine.global_steps = step
    if hasattr(engine, "global_samples"):
        engine.global_samples = int(state.get("global_samples", 0))
    if hasattr(engine, "skipped_steps"):
        engine.skipped_steps = int(state.get("skipped_steps", 0))
    for name, value in state.get("engine_runtime", {}).items():
        if hasattr(engine, name):
            setattr(engine, name, value)
    zero_optimizer = getattr(engine, "optimizer", None)
    base_optimizer = getattr(zero_optimizer, "optimizer", None)
    saved_groups = state.get("optimizer_param_groups", [])
    if base_optimizer is not None and saved_groups:
        if len(base_optimizer.param_groups) != len(saved_groups):
            raise DeepSpeedSurvivorHandoffError(
                "optimizer param-group count changed across handoff: "
                f"current={len(base_optimizer.param_groups)} "
                f"saved={len(saved_groups)}"
            )
        for current, saved in zip(
            base_optimizer.param_groups, saved_groups
        ):
            for key, value in saved.items():
                current[key] = value
    for name, value in state.get("optimizer_runtime", {}).items():
        if zero_optimizer is not None and hasattr(zero_optimizer, name):
            setattr(zero_optimizer, name, value)
    scheduler = getattr(engine, "lr_scheduler", None)
    scheduler_state = state.get("scheduler")
    if (
        scheduler is not None
        and scheduler_state is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(scheduler_state)


def capture_survivor_handoff(
    engine: Any,
    optimizer_replica: Any,
    *,
    root: Path,
    recovery_epoch: int,
    failure_step: int,
) -> dict[str, Any]:
    """Write one rank's current state before its old process is retired."""
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    actual_step = int(getattr(engine, "global_steps", -1))
    if actual_step != int(failure_step):
        raise DeepSpeedSurvivorHandoffError(
            "survivor is not at the requested safe step: "
            f"rank={rank} actual_step={actual_step} "
            f"failure_step={failure_step}"
        )

    optimizer_replica.wait_until_replicated(actual_step)
    started = time.monotonic()
    payload = {
        "protocol": PROTOCOL_VERSION,
        "recovery_epoch": int(recovery_epoch),
        "source_epoch": int(
            os.environ.get("MOEGAMBIT_RECOVERY_EPOCH", "0")
        ),
        "rank": rank,
        "logical_node": int(
            os.environ.get("MOEGAMBIT_LOGICAL_NODE_RANK", "0")
        ),
        "physical_node": int(
            os.environ.get("MOEGAMBIT_PHYSICAL_NODE_RANK", "0")
        ),
        "step": actual_step,
        "engine_state": capture_engine_state(engine),
        "rng_state": capture_rng_state(),
        "model": _cpu_model_state(engine.module),
        "optimizer": optimizer_replica.capture_local_handoff(actual_step),
        "peer_optimizer": optimizer_replica.export_peer_handoff(actual_step),
    }
    path = rank_state_path(root, recovery_epoch, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    size = path.stat().st_size
    return {
        "rank": rank,
        "step": actual_step,
        "path": str(path),
        "bytes": size,
        "elapsed_s": time.monotonic() - started,
    }


def load_survivor_handoff(
    *,
    root: Path,
    recovery_epoch: int,
    rank: int,
    expected_step: int,
    expected_source_epoch: int,
    expected_logical_node: int,
    expected_physical_node: int,
) -> dict[str, Any]:
    import torch

    path = rank_state_path(root, recovery_epoch, rank)
    if not path.is_file():
        raise DeepSpeedSurvivorHandoffError(
            f"survivor handoff is missing for rank={rank}: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "protocol": PROTOCOL_VERSION,
        "recovery_epoch": int(recovery_epoch),
        "source_epoch": int(expected_source_epoch),
        "rank": int(rank),
        "logical_node": int(expected_logical_node),
        "physical_node": int(expected_physical_node),
        "step": int(expected_step),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise DeepSpeedSurvivorHandoffError(
            f"survivor handoff version mismatch for {path}: {mismatches}"
        )
    return payload


def restore_survivor_handoff(
    engine: Any,
    optimizer_replica: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    incompatible = engine.module.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DeepSpeedSurvivorHandoffError(
            "survivor model handoff is incompatible: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    optimizer_summary = optimizer_replica.restore_handoff_snapshot(
        payload["optimizer"], restore_expert=True
    )
    apply_engine_state(engine, payload["engine_state"])
    apply_rng_state(payload.get("rng_state", {}))
    return {
        "rank": int(payload["rank"]),
        "step": int(payload["step"]),
        "model_tensors": len(payload["model"]),
        "optimizer": optimizer_summary,
    }
