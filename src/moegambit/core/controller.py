"""Engine-neutral recovery state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from moegambit.core.contracts import (
    FeatureSwitches,
    RecoveryContext,
    RecoveryDecision,
)
from moegambit.core.events import EventBus, RuntimeEvent
from moegambit.core.policy import RecoveryPolicy


class ControllerState(str, Enum):
    HEALTHY = "healthy"
    DECIDING = "deciding"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class RecoveryController:
    policy: RecoveryPolicy
    features: FeatureSwitches
    events: EventBus
    state: ControllerState = ControllerState.HEALTHY
    epoch: int = 0

    def decide(self, context: RecoveryContext) -> RecoveryDecision:
        if self.state is not ControllerState.HEALTHY:
            raise RuntimeError(
                f"cannot start recovery while controller is {self.state.value}"
            )
        self.state = ControllerState.DECIDING
        self.epoch += 1
        self.events.emit(
            RuntimeEvent(
                "recovery.deciding",
                {"epoch": self.epoch, "rank": context.failure.rank},
            )
        )
        decision = self.policy.decide(context, self.features)
        self.state = ControllerState.RECOVERING
        self.events.emit(
            RuntimeEvent(
                "recovery.decided",
                {
                    "epoch": self.epoch,
                    "mode": decision.mode.value,
                    "reason": decision.reason,
                    "projected_exposure": decision.projected_exposure,
                },
            )
        )
        return decision

    def complete(self) -> None:
        self.state = ControllerState.HEALTHY
        self.events.emit(RuntimeEvent("recovery.completed", {"epoch": self.epoch}))

    def fail(self, reason: str) -> None:
        self.state = ControllerState.FAILED
        self.events.emit(
            RuntimeEvent(
                "recovery.failed", {"epoch": self.epoch, "reason": reason}
            )
        )
