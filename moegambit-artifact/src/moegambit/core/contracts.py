"""Stable data contracts shared by runtimes and engine adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class RecoveryMode(str, Enum):
    NOOP = "noop"
    HOT_SWAP = "hot_swap"
    CHECKPOINT_RELAUNCH = "checkpoint_relaunch"
    ABORT = "abort"


@dataclass(frozen=True)
class FeatureSwitches:
    """Orthogonal runtime features.

    ZeRO-2 replication may run without hot replacement to pre-stage optimizer
    state for another recovery system. Hot replacement may run without ZeRO-2
    and obtain all optimizer state from an engine-specific checkpoint path.
    """

    hot_swap: bool = False
    zero2: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {"hot_swap": self.hot_swap, "zero2": self.zero2}


@dataclass(frozen=True)
class FailureEvent:
    rank: int
    step: int
    checkpoint_step: int
    node: Optional[int] = None
    local_rank: Optional[int] = None
    scope: str = "rank"
    reason: str = "fail_stop"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryContext:
    failure: FailureEvent
    peer_available: bool
    spare_available: bool
    affected_experts: Tuple[int, ...] = ()
    num_experts: int = 0
    exposure_debt: int = 0

    @property
    def gap(self) -> int:
        return max(0, self.failure.step - self.failure.checkpoint_step)


@dataclass(frozen=True)
class RecoveryDecision:
    mode: RecoveryMode
    reason: str
    projected_exposure: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
