"""Three-threshold degraded-training safety policy."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

__all__ = ["DegradedDecision", "DegradedModePolicy"]


@dataclass(frozen=True)
class DegradedDecision:
    allowed: bool
    reason: str
    healthy_capacity: float
    degraded_steps: int
    staleness: int
    stale_load_token_ratio: float


class DegradedModePolicy:
    _instance: Optional["DegradedModePolicy"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        capacity_threshold: float = 0.5,
        max_degraded_steps: int = 1000,
        max_staleness: int = 500,
    ) -> None:
        if not 0.0 <= capacity_threshold <= 1.0:
            raise ValueError("capacity_threshold must be in [0, 1]")
        if max_degraded_steps < 0 or max_staleness < 0:
            raise ValueError("degraded thresholds must be non-negative")
        self.capacity_threshold = float(capacity_threshold)
        self.max_degraded_steps = int(max_degraded_steps)
        self.max_staleness = int(max_staleness)
        self.healthy_experts = 0
        self.total_experts = 0
        self.degraded_since_step: Optional[int] = None
        self.current_step = 0
        self.checkpoint_step = 0
        self.stale_tokens = 0
        self.total_tokens = 0

    @classmethod
    def get_instance(cls, **kwargs) -> "DegradedModePolicy":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def update(
        self,
        *,
        healthy_experts: int,
        total_experts: int,
        current_step: int,
        checkpoint_step: int,
        stale_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.healthy_experts = int(healthy_experts)
        self.total_experts = int(total_experts)
        self.current_step = int(current_step)
        self.checkpoint_step = int(checkpoint_step)
        self.stale_tokens = int(stale_tokens)
        self.total_tokens = int(total_tokens)
        if healthy_experts < total_experts and self.degraded_since_step is None:
            self.degraded_since_step = self.current_step
        elif healthy_experts >= total_experts:
            self.degraded_since_step = None

    def evaluate(self) -> DegradedDecision:
        capacity = (
            0.0
            if self.total_experts <= 0
            else self.healthy_experts / self.total_experts
        )
        degraded_steps = (
            0
            if self.degraded_since_step is None
            else max(0, self.current_step - self.degraded_since_step)
        )
        staleness = max(0, self.current_step - self.checkpoint_step)
        slt = 0.0 if self.total_tokens <= 0 else self.stale_tokens / self.total_tokens
        if capacity < self.capacity_threshold:
            allowed, reason = False, "healthy expert capacity below threshold"
        elif degraded_steps > self.max_degraded_steps:
            allowed, reason = False, "degraded iteration budget exceeded"
        elif staleness > self.max_staleness:
            allowed, reason = False, "checkpoint staleness threshold exceeded"
        else:
            allowed, reason = True, "degraded training remains inside all thresholds"
        return DegradedDecision(
            allowed, reason, capacity, degraded_steps, staleness, slt
        )
