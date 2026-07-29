"""Small public facade used by the Megatron training-loop patch."""

from __future__ import annotations

import os
from typing import Any, Optional

from ...config import RuntimeConfig
from ...runtime.runtime import RecoveryRuntime, initialize
from .adapter import build_megatron_adapter, _client

__all__ = [
    "bootstrap_control_plane",
    "initialize_runtime",
    "finalize_replacement",
    "get_runtime",
    "iteration_boundary",
    "before_optimizer_step",
    "after_optimizer_step",
    "on_distributed_error",
    "commit_iteration",
]

_RUNTIME: Optional[RecoveryRuntime] = None


def bootstrap_control_plane() -> None:
    client = _client()
    client.elastic_sanitize_recovery_env_for_startup()
    client.elastic_client_start()


def initialize_runtime(
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
    args: Any = None,
    config: Optional[RuntimeConfig] = None,
) -> RecoveryRuntime:
    global _RUNTIME
    adapter = build_megatron_adapter(model, optimizer, scheduler, args)
    resolved = config or RuntimeConfig.from_env()
    enabled = bool(
        getattr(args, "moe_moegambit_enable", resolved.enabled)
        if args is not None
        else resolved.enabled
    )
    if enabled != resolved.enabled or resolved.framework != "megatron":
        resolved = resolved.merged_with(enabled=enabled, framework="megatron")
    _RUNTIME = initialize(adapter=adapter, config=resolved)

    client = _client()
    rebuild_mode = bool(client.is_rebuild_mode())
    client.elastic_zero2_initialize(
        model,
        optimizer,
        initial_step=int(getattr(args, "iteration", 0) if args is not None else 0),
        start_transport=not rebuild_mode,
    )
    if rebuild_mode:
        client.elastic_report_recovery_phase("model_optimizer_ready")
    return _RUNTIME


def finalize_replacement(
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
) -> None:
    """Complete replacement state transfer after caller restores load flags."""

    client = _client()
    if client.is_rebuild_mode():
        client.elastic_replacement_sync_params(model, optimizer, scheduler)


def get_runtime() -> Optional[RecoveryRuntime]:
    return _RUNTIME


def iteration_boundary(step: int) -> int:
    if _RUNTIME is None:
        return step
    return _RUNTIME.iteration_boundary(step)


def before_optimizer_step(step: int) -> None:
    if _RUNTIME is not None:
        _RUNTIME.before_optimizer_step(step)


def after_optimizer_step(step: int, committed: bool = True) -> None:
    if _RUNTIME is not None:
        _RUNTIME.after_optimizer_step(step, committed=committed)


def on_distributed_error(exc: BaseException) -> bool:
    return bool(_RUNTIME is not None and _RUNTIME.on_distributed_error(exc))


def commit_iteration(step: int) -> bool:
    return bool(_RUNTIME is not None and _RUNTIME.commit_iteration(step))
