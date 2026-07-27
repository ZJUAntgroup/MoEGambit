"""Recovery epoch lifecycle and training-loop boundary names."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from ..errors import ContractViolation

__all__ = ["LifecyclePhase", "EpochState", "RecoveryEpochTracker"]


class LifecyclePhase(Enum):
    ITERATION_BOUNDARY = "iteration_boundary"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"
    DISTRIBUTED_ERROR = "on_distributed_error"
    COMMIT_ITERATION = "commit_iteration"


class EpochState(Enum):
    CLEAN = "clean"
    RECOVERING = "recovering"
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass
class RecoveryEpochTracker:
    """Strict monotonic state machine for local recovery participation."""

    epoch: int = 0
    state: EpochState = EpochState.CLEAN
    last_committed_step: int = -1
    provisional_since_step: Optional[int] = None
    history: list = field(default_factory=list)

    def begin_recovery(self, failed_ranks: Tuple[int, ...], at_step: int) -> int:
        if self.state not in (EpochState.CLEAN, EpochState.COMMITTED):
            raise ContractViolation(
                f"cannot begin recovery while epoch state is {self.state.value}"
            )
        if not failed_ranks:
            raise ContractViolation("recovery requires at least one failed rank")
        if at_step < 0:
            raise ContractViolation("recovery step must be non-negative")
        self.epoch += 1
        self.state = EpochState.RECOVERING
        self.provisional_since_step = None
        self.history.append(
            {
                "event": "begin_recovery",
                "epoch": self.epoch,
                "failed_ranks": list(failed_ranks),
                "at_step": int(at_step),
                "at_ns": time.time_ns(),
            }
        )
        return self.epoch

    def adopt_recovery(
        self,
        recovery_epoch: int,
        failed_ranks: Tuple[int, ...],
        at_step: int,
    ) -> int:
        recovery_epoch = int(recovery_epoch)
        if recovery_epoch <= self.epoch:
            raise ContractViolation(
                "adopted recovery epoch must be newer than the local epoch: "
                f"{recovery_epoch} <= {self.epoch}"
            )
        if not failed_ranks:
            raise ContractViolation("adopted recovery must name failed ranks")
        if at_step < 0:
            raise ContractViolation("adopted recovery step must be non-negative")
        self.epoch = recovery_epoch
        self.state = EpochState.RECOVERING
        self.provisional_since_step = None
        self.history.append(
            {
                "event": "adopt_recovery",
                "epoch": self.epoch,
                "failed_ranks": list(failed_ranks),
                "at_step": int(at_step),
                "at_ns": time.time_ns(),
            }
        )
        return self.epoch

    def mark_provisional(self, resume_step: int) -> None:
        if self.state is not EpochState.RECOVERING:
            raise ContractViolation(
                "only a recovering epoch can become provisional"
            )
        if resume_step < 0:
            raise ContractViolation("resume step must be non-negative")
        self.state = EpochState.PROVISIONAL
        self.provisional_since_step = int(resume_step)
        self.history.append(
            {
                "event": "provisional",
                "epoch": self.epoch,
                "resume_step": int(resume_step),
            }
        )

    def can_commit(self, step: int) -> bool:
        return (
            self.state is EpochState.PROVISIONAL
            and self.provisional_since_step is not None
            and int(step) > self.provisional_since_step
        )

    def commit(self, step: int) -> bool:
        if not self.can_commit(step):
            if self.state in (EpochState.CLEAN, EpochState.COMMITTED):
                self.last_committed_step = max(self.last_committed_step, int(step))
            return False
        self.state = EpochState.COMMITTED
        self.last_committed_step = int(step)
        self.history.append(
            {"event": "commit", "epoch": self.epoch, "step": int(step)}
        )
        return True

    def fail(self, reason: str) -> None:
        self.state = EpochState.FAILED
        self.history.append(
            {
                "event": "fail",
                "epoch": self.epoch,
                "reason": str(reason)[:1000],
            }
        )

    @property
    def is_provisional(self) -> bool:
        return self.state is EpochState.PROVISIONAL

    def snapshot(self) -> Mapping[str, Any]:
        data: Dict[str, Any] = {
            "epoch": self.epoch,
            "state": self.state.value,
            "last_committed_step": self.last_committed_step,
        }
        if self.provisional_since_step is not None:
            data["provisional_since_step"] = self.provisional_since_step
        return data
