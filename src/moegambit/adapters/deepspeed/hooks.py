"""Stable hook surface imported by patched DeepSpeed source files."""

from __future__ import annotations

import os
from functools import wraps
from typing import Callable

from .config import prepare_deepspeed_config
from .group_creation import ordered_new_group
from .integration import attach_engine
from .launcher_recovery import confirm_rank_recovery
from .packed_moe import (
    PACKED_EXPERT_FORMAT,
    PACKED_EXPERT_FORMAT_VERSION,
    PackedExpertPrefetcher,
    build_packed_expert_state,
    packed_expert_cache_enabled,
    packed_expert_checkpoint_enabled,
    packed_expert_checkpoint_name,
    packed_expert_tensor_bytes,
    take_cached_packed_expert,
    validate_packed_expert_state,
)


_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in _TRUE_VALUES
        for name in (
            "MOEGAMBIT_HOT_SWAP",
            "MOEGAMBIT_ZERO2",
            "DEEPSPEED_MOEGAMBIT_HOT_SWAP",
            "DEEPSPEED_MOEGAMBIT_ZERO2",
        )
    )


def is_inprocess_replacement() -> bool:
    value = os.environ.get(
        "MOEGAMBIT_DEEPSPEED_INPROCESS_REPLACEMENT", "0"
    )
    return value.strip().lower() in _TRUE_VALUES


def track_training_phase(phase: str) -> Callable:
    """Decorate a DeepSpeed boundary without placing policy in DeepSpeed."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(engine, *args, **kwargs):
            runtime = getattr(engine, "_moegambit_runtime", None)
            if runtime is None:
                return function(engine, *args, **kwargs)
            runtime.record_training_phase(phase)
            try:
                return function(engine, *args, **kwargs)
            except BaseException as exc:
                runtime.record_training_failure(exc)
                raise

        return wrapped

    return decorate


__all__ = [
    "PACKED_EXPERT_FORMAT",
    "PACKED_EXPERT_FORMAT_VERSION",
    "PackedExpertPrefetcher",
    "attach_engine",
    "build_packed_expert_state",
    "confirm_rank_recovery",
    "is_enabled",
    "is_inprocess_replacement",
    "ordered_new_group",
    "packed_expert_cache_enabled",
    "packed_expert_checkpoint_enabled",
    "packed_expert_checkpoint_name",
    "packed_expert_tensor_bytes",
    "prepare_deepspeed_config",
    "take_cached_packed_expert",
    "track_training_phase",
    "validate_packed_expert_state",
]
