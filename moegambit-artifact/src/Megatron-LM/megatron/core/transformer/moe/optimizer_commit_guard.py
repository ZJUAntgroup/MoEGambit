# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoEGambit Optimizer Commit Guard.

Ensures that ``optimizer.step()`` is not executed when the current
iteration has been invalidated by a hard failure, preventing partial
parameter updates that would leave the model in an inconsistent state.

Commit boundary analysis
------------------------
Megatron's ``MixedPrecisionOptimizer.step()`` has three internal phases:

1. ``prepare_grads()`` — copy model grads → main grads, unscale, check
   inf/nan.  **No parameter mutation.**
2. ``clip_grad_norm()`` — clip gradients.  **No parameter mutation.**
3. ``step_with_ready_grads()`` — the **commit point**:
   a. ``self.optimizer.step()`` — updates fp32 main params + optimizer
      state (momentum, variance).  **Main params mutated.**
   b. ``_copy_main_params_to_model_params()`` — copies fp32 main params
      back to fp16/bf16 model params.  **Model params mutated.**

If a hard failure (NCCL error) occurs during ``forward_backward_func()``,
``optimizer.step()`` has NOT yet been called, so both main params and
model params are still at the state of the previous completed iteration.
This is the common case — NCCL errors happen during collectives in
forward/backward, not during the purely-local optimizer step.

Guard strategy (v1)
-------------------
The guard provides two mechanisms:

1. **Pre-step check** (``should_commit()``): Called by ``train_step``
   before ``optimizer.step()``.  Returns False if the iteration has been
   invalidated, causing the optimizer step to be skipped entirely.

2. **Post-failure recovery** (``recover_from_partial_commit()``):
   If a failure occurs *during* ``optimizer.step()`` (extremely rare —
   only possible if a CUDA error happens during the local Adam kernel
   or the copy-back), this method restores model params from fp32 main
   params using ``optimizer.reload_model_params()``.  In v1 this is an
   interface-only placeholder; the actual implementation requires
   careful handling of optimizer state rollback.

Staged apply (future)
---------------------
A future version may split ``step_with_ready_grads()`` into:
- ``compute_updates()`` — compute new param values in a staging buffer
- ``commit_updates()`` — atomically swap staging buffer into model params

This would provide true atomic commit semantics.  The
``staged_apply_fn`` callback is reserved for this purpose.

Integration with training.py
-----------------------------
The guard is checked in ``train_step`` between ``forward_backward_func``
and ``optimizer.step()``::

    # After forward_backward_func:
    if not commit_guard.should_commit():
        # Skip optimizer.step() — iteration is invalid
        return {}, 1, False, False, 0, None, None

    # Normal path:
    update_successful, grad_norm, ... = optimizer.step()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Commit phase tracking
# =====================================================================

class CommitPhase(IntEnum):
    """Tracks where we are in the optimizer commit lifecycle."""

    NOT_STARTED = 0
    """optimizer.step() has not been called yet."""

    GRADS_PREPARED = 1
    """prepare_grads() completed (grads copied, unscaled, checked)."""

    GRADS_CLIPPED = 2
    """clip_grad_norm() completed."""

    MAIN_PARAMS_UPDATED = 3
    """self.optimizer.step() completed — fp32 main params updated,
    optimizer state (momentum/variance) updated.  Model params NOT
    yet updated."""

    MODEL_PARAMS_WRITTEN = 4
    """_copy_main_params_to_model_params() completed — model params
    now reflect the new values.  This is the commit point."""

    COMMITTED = 5
    """optimizer.step() returned successfully.  The iteration's
    parameter update is fully committed."""


# =====================================================================
# Commit record
# =====================================================================

@dataclass
class CommitRecord:
    """Records the outcome of an optimizer commit attempt."""

    step: int = -1
    """Training step."""

    phase_reached: CommitPhase = CommitPhase.NOT_STARTED
    """Highest phase reached before completion or failure."""

    committed: bool = False
    """True if the commit completed successfully."""

    skipped: bool = False
    """True if the commit was skipped (iteration invalidated)."""

    blocked: bool = False
    """True if should_commit() returned False (guard blocked)."""

    recovery_needed: bool = False
    """True if a partial commit was detected and recovery is needed."""

    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "phase_reached": self.phase_reached.name,
            "committed": self.committed,
            "skipped": self.skipped,
            "blocked": self.blocked,
            "recovery_needed": self.recovery_needed,
        }


# =====================================================================
# OptimizerCommitGuard
# =====================================================================

class OptimizerCommitGuard:
    """Guards optimizer.step() against execution during invalidated iterations.

    Lifecycle per iteration::

        begin_iteration(step)
        ... forward / backward ...
        # Check before optimizer:
        if guard.should_commit():
            optimizer.step()
            guard.mark_committed(step)
        else:
            guard.mark_skipped(step)
        end_iteration(step)

    Thread safety: not thread-safe.  All calls from the main training thread.
    """

    def __init__(self) -> None:
        # Current iteration state
        self._current_step: int = -1
        self._phase: CommitPhase = CommitPhase.NOT_STARTED
        self._blocked: bool = False
        self._block_reason: str = ""
        self._block_step: int = -1
        self._in_iteration: bool = False

        # Invalidation check function
        self._is_iteration_invalid_fn: Optional[Callable[[], bool]] = None

        # Callbacks
        self._recover_fn: Optional[Callable] = None
        self._staged_apply_fn: Optional[Callable] = None  # future

        # History
        self._current_record: Optional[CommitRecord] = None
        self._history: List[CommitRecord] = []
        self._max_history: int = 100

        # Counters
        self._total_commits: int = 0
        self._total_skips: int = 0
        self._total_blocks: int = 0
        self._total_recoveries: int = 0

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def register_callbacks(
        self,
        *,
        is_iteration_invalid_fn: Optional[Callable[[], bool]] = None,
        recover_fn: Optional[Callable] = None,
        staged_apply_fn: Optional[Callable] = None,
    ) -> None:
        """Register callback functions.

        Args:
            is_iteration_invalid_fn: Returns True if the current iteration
                is invalid (e.g. ``IterationInvalidator.is_current_iteration_invalid``).
            recover_fn: Called when a partial commit is detected and
                recovery is needed.  Signature: ``recover_fn(step, phase)``.
                In v1 this typically calls ``optimizer.reload_model_params()``.
            staged_apply_fn: Future placeholder for staged apply.
        """
        if is_iteration_invalid_fn is not None:
            self._is_iteration_invalid_fn = is_iteration_invalid_fn
        if recover_fn is not None:
            self._recover_fn = recover_fn
        if staged_apply_fn is not None:
            self._staged_apply_fn = staged_apply_fn

    # -----------------------------------------------------------------
    # Iteration lifecycle
    # -----------------------------------------------------------------

    def begin_iteration(self, step: int) -> None:
        """Called at the start of each training iteration."""
        self._current_step = step
        self._phase = CommitPhase.NOT_STARTED
        self._blocked = False
        self._block_reason = ""
        self._block_step = -1
        self._in_iteration = True
        self._current_record = CommitRecord(step=step, timestamp=time.time())

    def end_iteration(self, step: int) -> None:
        """Called at the end of each training iteration."""
        if self._current_record is not None:
            self._history.append(self._current_record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        self._in_iteration = False
        self._current_record = None

    # -----------------------------------------------------------------
    # Explicit block API
    # -----------------------------------------------------------------

    def block(self, reason: str = "", step: int = -1) -> None:
        """Explicitly block optimizer commit for the current iteration.

        Called by RecoveryController.on_hard_rank_failure() to ensure
        the optimizer step is skipped even if the iteration invalidation
        flag hasn't propagated yet.
        """
        self._blocked = True
        self._block_reason = reason
        self._block_step = step
        logger.warning(
            "OptimizerCommitGuard: commit BLOCKED (step=%d, reason=%s)",
            step if step >= 0 else self._current_step, reason,
        )

    # -----------------------------------------------------------------
    # Pre-step guard
    # -----------------------------------------------------------------

    def should_commit(self) -> bool:
        """Check whether optimizer.step() should proceed.

        Returns False if the current iteration has been invalidated
        or explicitly blocked, preventing the optimizer from committing
        a tainted update.

        This should be called AFTER forward_backward_func() completes
        (or after catching an exception from it) and BEFORE
        optimizer.step().
        """
        # Check explicit block (set by RecoveryController.on_hard_rank_failure)
        if self._blocked:
            self._total_blocks += 1
            if self._current_record is not None:
                self._current_record.blocked = True
            log_step = self._block_step if self._block_step >= 0 else self._current_step
            logger.warning(
                "MoEGambit OptimizerCommitGuard: BLOCKING optimizer.step() "
                "at step %d — explicitly blocked (reason=%s)",
                log_step, self._block_reason,
            )
            return False

        # Check invalidation
        if self._is_iteration_invalid_fn is not None:
            if self._is_iteration_invalid_fn():
                self._blocked = True
                self._total_blocks += 1
                if self._current_record is not None:
                    self._current_record.blocked = True
                logger.warning(
                    "MoEGambit OptimizerCommitGuard: BLOCKING optimizer.step() "
                    "at step %d — iteration is invalid",
                    self._current_step,
                )
                return False

        return True

    # -----------------------------------------------------------------
    # Post-step tracking
    # -----------------------------------------------------------------

    def mark_phase(self, phase: CommitPhase) -> None:
        """Update the current commit phase (for fine-grained tracking).

        Called by instrumented optimizer code to track progress through
        the commit phases.  In v1, this is optional — the guard works
        without phase tracking.
        """
        self._phase = phase
        if self._current_record is not None:
            self._current_record.phase_reached = phase

    def mark_committed(self, step: int = -1) -> None:
        """Mark the optimizer step as successfully committed.

        Called after optimizer.step() returns successfully.
        """
        self._phase = CommitPhase.COMMITTED
        self._total_commits += 1
        if self._current_record is not None:
            self._current_record.committed = True
            self._current_record.phase_reached = CommitPhase.COMMITTED

    def mark_skipped(self, step: int = -1, reason: str = "") -> None:
        """Mark the optimizer step as skipped.

        Called when optimizer.step() is skipped due to invalidation
        or other reasons (e.g. grad overflow).
        """
        self._total_skips += 1
        if self._current_record is not None:
            self._current_record.skipped = True

        log_step = self._block_step if self._block_step >= 0 else self._current_step
        logger.info(
            "MoEGambit OptimizerCommitGuard: optimizer.step() SKIPPED "
            "at step %d (reason: %s)",
            log_step, reason or "iteration_invalid",
        )

    # -----------------------------------------------------------------
    # Recovery (v1: interface only)
    # -----------------------------------------------------------------

    def recover_from_partial_commit(self, step: int = -1) -> bool:
        """Attempt to recover from a partial optimizer commit.

        This handles the rare case where a failure occurs DURING
        optimizer.step() — after self.optimizer.step() has updated
        fp32 main params but before _copy_main_params_to_model_params()
        has completed.

        Recovery strategy (v1):
        - Call optimizer.reload_model_params() to re-copy fp32 main
          params to model params, ensuring consistency.
        - Note: this does NOT undo the optimizer state update (momentum,
          variance).  A future version may implement full optimizer
          state rollback.

        Returns True if recovery was attempted.
        """
        if self._phase.value < CommitPhase.MAIN_PARAMS_UPDATED.value:
            # Failure before optimizer.step() — no recovery needed
            logger.debug(
                "MoEGambit OptimizerCommitGuard: no recovery needed at "
                "step %d (phase=%s, before MAIN_PARAMS_UPDATED)",
                self._current_step, self._phase.name,
            )
            return False

        if self._phase == CommitPhase.COMMITTED:
            # Already committed — no recovery needed
            return False

        logger.warning(
            "MoEGambit OptimizerCommitGuard: PARTIAL COMMIT detected at "
            "step %d (phase=%s). Attempting recovery.",
            self._current_step, self._phase.name,
        )

        self._total_recoveries += 1
        if self._current_record is not None:
            self._current_record.recovery_needed = True

        if self._recover_fn is not None:
            try:
                self._recover_fn(step=step or self._current_step,
                                 phase=self._phase)
                logger.warning(
                    "MoEGambit OptimizerCommitGuard: recovery completed "
                    "at step %d", self._current_step,
                )
            except Exception as e:
                logger.error(
                    "MoEGambit OptimizerCommitGuard: recovery FAILED "
                    "at step %d: %s", self._current_step, e,
                )
                return False

        return True

    # -----------------------------------------------------------------
    # Inverse compensation (future interface)
    # -----------------------------------------------------------------

    def request_inverse_compensation(
        self,
        step: int = -1,
        *,
        rollback_optimizer_state: bool = False,
        rollback_main_params: bool = False,
    ) -> bool:
        """Request inverse compensation for a committed step.

        This is a **future interface** — not implemented in v1.

        In a future version, this would:
        1. Undo the optimizer state update (restore previous momentum/
           variance from a snapshot).
        2. Undo the main param update (restore previous fp32 values).
        3. Re-copy main params to model params.

        Args:
            step: The step to compensate.
            rollback_optimizer_state: If True, restore optimizer state.
            rollback_main_params: If True, restore main param values.

        Returns:
            True if compensation was performed (always False in v1).
        """
        logger.warning(
            "MoEGambit OptimizerCommitGuard: inverse compensation "
            "requested for step %d but NOT IMPLEMENTED in v1. "
            "(rollback_optimizer_state=%s, rollback_main_params=%s)",
            step, rollback_optimizer_state, rollback_main_params,
        )
        return False

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    @property
    def phase(self) -> CommitPhase:
        return self._phase

    @property
    def is_blocked(self) -> bool:
        """True if should_commit() returned False for this iteration."""
        return self._blocked

    @property
    def in_iteration(self) -> bool:
        return self._in_iteration

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_commits(self) -> int:
        return self._total_commits

    @property
    def total_skips(self) -> int:
        return self._total_skips

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def total_recoveries(self) -> int:
        return self._total_recoveries

    def summary(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "phase": self._phase.name,
            "is_blocked": self._blocked,
            "in_iteration": self._in_iteration,
            "total_commits": self._total_commits,
            "total_skips": self._total_skips,
            "total_blocks": self._total_blocks,
            "total_recoveries": self._total_recoveries,
        }

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Full reset (for testing)."""
        self._current_step = -1
        self._phase = CommitPhase.NOT_STARTED
        self._blocked = False
        self._block_reason = ""
        self._in_iteration = False
        self._current_record = None
        self._history.clear()
        self._total_commits = 0
        self._total_skips = 0
        self._total_blocks = 0
        self._total_recoveries = 0


# =====================================================================
# Global singleton
# =====================================================================

_GUARD: Optional[OptimizerCommitGuard] = None


def get_optimizer_commit_guard() -> OptimizerCommitGuard:
    """Get or create the global OptimizerCommitGuard singleton."""
    global _GUARD
    if _GUARD is None:
        _GUARD = OptimizerCommitGuard()
    return _GUARD


def clear_optimizer_commit_guard() -> None:
    """Reset the global guard (for testing)."""
    global _GUARD
    if _GUARD is not None:
        _GUARD.reset()
    _GUARD = None
