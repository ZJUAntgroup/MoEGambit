"""Recovery path selection with no training-engine dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from moegambit.core.contracts import (
    FeatureSwitches,
    RecoveryContext,
    RecoveryDecision,
    RecoveryMode,
)


class RecoveryPolicy(Protocol):
    def decide(
        self, context: RecoveryContext, features: FeatureSwitches
    ) -> RecoveryDecision:
        ...


@dataclass(frozen=True)
class StalenessDensityPolicy:
    min_gap: int = 0
    max_gap: int = 200
    exposure_window_steps: int = 2000
    max_density: float = 0.1

    def __post_init__(self) -> None:
        if self.min_gap < 0 or self.max_gap < self.min_gap:
            raise ValueError("recovery gap bounds are invalid")
        if self.exposure_window_steps <= 0:
            raise ValueError("exposure_window_steps must be positive")
        if not 0.0 <= self.max_density <= 1.0:
            raise ValueError("max_density must be between zero and one")

    def decide(
        self, context: RecoveryContext, features: FeatureSwitches
    ) -> RecoveryDecision:
        if not features.hot_swap:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH, "hot_swap_disabled"
            )
        if not context.spare_available:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH, "no_spare"
            )
        if not context.peer_available:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH, "no_peer"
            )
        if context.gap < self.min_gap:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH, "gap_below_minimum"
            )
        if context.gap > self.max_gap:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH, "gap_above_maximum"
            )
        if context.num_experts <= 0:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH,
                "invalid_exposure_domain",
            )

        projected_debt = (
            context.exposure_debt
            + len(context.affected_experts) * context.gap
        )
        denominator = context.num_experts * self.exposure_window_steps
        density = projected_debt / float(denominator)
        if density > self.max_density:
            return RecoveryDecision(
                RecoveryMode.CHECKPOINT_RELAUNCH,
                "exposure_budget_exceeded",
                projected_exposure=density,
            )
        return RecoveryDecision(
            RecoveryMode.HOT_SWAP,
            "hybrid_recovery_admitted",
            projected_exposure=density,
            metadata={"zero2_enabled": features.zero2},
        )
