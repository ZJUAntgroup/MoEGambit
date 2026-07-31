"""Framework-neutral training-step transaction semantics.

The runtime must never infer rollback safety from an iteration number alone.
An iteration can have the same number before and after the optimizer has
started mutating parameters.  This module records that distinction and turns a
failure location into a fail-closed recovery decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ContractViolation

__all__ = [
    "FailureAction",
    "FailureDecision",
    "StepPhase",
    "StepTransaction",
]


class StepPhase(str, Enum):
    IDLE = "idle"
    SAFE_POINT = "safe_point"
    FORWARD = "forward"
    BACKWARD = "backward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIMIZER_BEFORE = "optimizer_before"
    OPTIMIZER_DURING = "optimizer_during"
    OPTIMIZER_AFTER = "optimizer_after"
    STEP_COMMITTED = "step_committed"
    CHECKPOINT_BEFORE = "checkpoint_before"
    CHECKPOINT_COMMIT = "checkpoint_commit"
    CHECKPOINT_COMMITTED = "checkpoint_committed"


class FailureAction(str, Enum):
    """Required action after a failure at a particular phase."""

    REPLAY_STEP = "replay_step"
    RESTORE_AND_REPLAY = "restore_and_replay"
    RESUME_COMMITTED = "resume_committed"
    CHECKPOINT_RESTART = "checkpoint_restart"


@dataclass(frozen=True)
class FailureDecision:
    action: FailureAction
    resume_step: int
    phase: StepPhase
    optimizer_dirty: bool
    reason: str

    @property
    def permits_bookkeeping_rollback(self) -> bool:
        return self.action is FailureAction.REPLAY_STEP


class StepTransaction:
    """Track one rank's last proven state version and current mutation phase."""

    def __init__(self, committed_step: int = 0) -> None:
        committed_step = int(committed_step)
        if committed_step < 0:
            raise ValueError("committed step must be non-negative")
        self.committed_step = committed_step
        self.checkpoint_step = -1
        self.current_step = committed_step
        self.phase = StepPhase.IDLE
        self.optimizer_snapshot_step = -1
        self.pending_checkpoint_steps: set[int] = set()

    def begin_step(self, step: int) -> None:
        step = int(step)
        if step != self.committed_step:
            raise ContractViolation(
                "new step must start at the last committed version: "
                f"step={step} committed={self.committed_step}"
            )
        self.current_step = step
        self.phase = StepPhase.SAFE_POINT

    def enter(self, phase: StepPhase | str) -> None:
        phase = StepPhase(phase)
        if phase in {
            StepPhase.IDLE,
            StepPhase.STEP_COMMITTED,
            StepPhase.CHECKPOINT_COMMITTED,
        }:
            raise ContractViolation(
                f"phase {phase.value} requires an explicit commit operation"
            )
        self.phase = phase

    def mark_optimizer_snapshot(self, step: int) -> None:
        step = int(step)
        if step < self.optimizer_snapshot_step:
            raise ContractViolation(
                "optimizer snapshot version cannot move backwards: "
                f"{step} < {self.optimizer_snapshot_step}"
            )
        if step > self.committed_step:
            raise ContractViolation(
                "optimizer snapshot cannot be newer than committed state: "
                f"snapshot={step} committed={self.committed_step}"
            )
        self.optimizer_snapshot_step = step

    def mark_optimizer_applied(self, committed_step: int) -> None:
        committed_step = int(committed_step)
        if self.phase is not StepPhase.OPTIMIZER_DURING:
            raise ContractViolation(
                "optimizer can commit only from optimizer_during; "
                f"phase={self.phase.value}"
            )
        if committed_step != self.current_step + 1:
            raise ContractViolation(
                "optimizer commit must advance exactly one version: "
                f"current={self.current_step} committed={committed_step}"
            )
        self.committed_step = committed_step
        self.current_step = committed_step
        self.phase = StepPhase.OPTIMIZER_AFTER

    def mark_optimizer_skipped(self, committed_step: int) -> None:
        if self.phase is not StepPhase.OPTIMIZER_DURING:
            raise ContractViolation(
                "optimizer skip can be recorded only from optimizer_during; "
                f"phase={self.phase.value}"
            )
        committed_step = int(committed_step)
        if committed_step != self.current_step + 1:
            raise ContractViolation(
                "skipped optimizer step must advance one logical version: "
                f"current={self.current_step} committed={committed_step}"
            )
        self.committed_step = committed_step
        self.current_step = committed_step
        self.phase = StepPhase.OPTIMIZER_AFTER

    def mark_optimizer_noop(self) -> None:
        if self.phase is not StepPhase.OPTIMIZER_DURING:
            raise ContractViolation(
                "optimizer no-op can be recorded only from optimizer_during; "
                f"phase={self.phase.value}"
            )
        self.phase = StepPhase.STEP_COMMITTED

    def mark_step_safe(self, step: int) -> None:
        step = int(step)
        if step != self.committed_step:
            raise ContractViolation(
                "safe step disagrees with committed state: "
                f"safe={step} committed={self.committed_step}"
            )
        if self.optimizer_snapshot_step < step:
            raise ContractViolation(
                "step is not safe until its optimizer snapshot is committed: "
                f"snapshot={self.optimizer_snapshot_step} step={step}"
            )
        self.phase = StepPhase.STEP_COMMITTED

    def begin_checkpoint(self, step: int) -> None:
        step = int(step)
        if step > self.committed_step:
            raise ContractViolation(
                "checkpoint must start from committed training state: "
                f"checkpoint={step} committed={self.committed_step}"
            )
        self.pending_checkpoint_steps.add(step)
        self.phase = StepPhase.CHECKPOINT_BEFORE

    def begin_checkpoint_commit(self, step: int) -> None:
        step = int(step)
        if step not in self.pending_checkpoint_steps:
            raise ContractViolation(
                f"checkpoint {step} was not prepared before commit"
            )
        if step > self.committed_step:
            raise ContractViolation("checkpoint commit version mismatch")
        self.phase = StepPhase.CHECKPOINT_COMMIT

    def mark_checkpoint_committed(self, step: int) -> None:
        step = int(step)
        if step not in self.pending_checkpoint_steps:
            raise ContractViolation(
                f"checkpoint {step} was not prepared before publication"
            )
        if step > self.committed_step:
            raise ContractViolation("checkpoint version mismatch")
        self.pending_checkpoint_steps.discard(step)
        self.checkpoint_step = max(self.checkpoint_step, step)
        self.phase = StepPhase.CHECKPOINT_COMMITTED

    @property
    def optimizer_dirty(self) -> bool:
        return self.phase is StepPhase.OPTIMIZER_DURING

    def decide_failure(self) -> FailureDecision:
        phase = self.phase
        if phase in {
            StepPhase.SAFE_POINT,
            StepPhase.FORWARD,
            StepPhase.BACKWARD,
            StepPhase.FORWARD_BACKWARD,
            StepPhase.OPTIMIZER_BEFORE,
        }:
            return FailureDecision(
                FailureAction.REPLAY_STEP,
                self.committed_step,
                phase,
                False,
                "optimizer state was not mutated",
            )
        if phase is StepPhase.OPTIMIZER_DURING:
            if self.optimizer_snapshot_step == self.committed_step:
                return FailureDecision(
                    FailureAction.RESTORE_AND_REPLAY,
                    self.committed_step,
                    phase,
                    True,
                    "optimizer may be partially committed",
                )
            return FailureDecision(
                FailureAction.CHECKPOINT_RESTART,
                self.checkpoint_step,
                phase,
                True,
                "no committed optimizer snapshot can undo a partial step",
            )
        if phase is StepPhase.OPTIMIZER_AFTER:
            return FailureDecision(
                FailureAction.CHECKPOINT_RESTART,
                self.checkpoint_step,
                phase,
                False,
                "optimizer advanced but the new replica is not committed",
            )
        if phase in {
            StepPhase.STEP_COMMITTED,
            StepPhase.CHECKPOINT_BEFORE,
            StepPhase.CHECKPOINT_COMMIT,
            StepPhase.CHECKPOINT_COMMITTED,
        }:
            return FailureDecision(
                FailureAction.RESUME_COMMITTED,
                self.committed_step,
                phase,
                False,
                "training state is at a committed optimizer version",
            )
        return FailureDecision(
            FailureAction.CHECKPOINT_RESTART,
            self.checkpoint_step,
            phase,
            False,
            "failure occurred outside a proven training safe point",
        )
