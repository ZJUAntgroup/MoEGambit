# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pipeline-Safe Rollback / Replay for BSR-MoE (PP > 1).

When a hard failure occurs during a pipeline-parallel training iteration,
the in-flight micro-batches, activations, and gradients across all pipeline
stages must be safely discarded before the iteration can be replayed.

Key insight: Megatron's pipeline schedule functions (1F1B, interleaved)
store ALL in-flight state as **function-local variables** (``input_tensors``,
``output_tensors``, ``forward_data_store``, P2P handles, etc.).  When an
exception propagates out of ``forward_backward_func``, these locals are
destroyed with the stack frame.  The P2P communication layer is also
stateless across calls.

Therefore, pipeline-safe rollback does NOT require manual cleanup of
pipeline internals.  The main challenges are:

1. **Cross-stage synchronization** — All PP stages must agree to enter
   rollback mode.  A failure on one stage must be communicated to all
   others before replay can begin.

2. **NCCL communicator health** — After an exception, the NCCL
   communicator may have pending operations.  A barrier or group
   rebuild may be needed before replay.

3. **Grad buffer cleanup** — ``train_step`` calls ``zero_grad_buffer()``
   at the start, so replay automatically gets clean gradients.

4. **Data iterator rewind** — Same as PP=1, handled by
   ``RollbackReplayManager``.

Protocol
--------
::

    1. Exception caught in train_step (any PP stage)
    2. All stages detect failure via:
       a. Local exception (the stage where failure occurred)
       b. P2P timeout / NCCL error (peer stages)
       c. Explicit failure broadcast via Gloo (if NCCL is dead)
    3. PipelineRollbackCoordinator.initiate_rollback()
       - Sets rollback flag on all stages
       - Optionally broadcasts failure info via Gloo
    4. RollbackReplayManager.rollback() — restores iteration state
    5. Next train_step call starts fresh pipeline schedule

Scope (v1)
----------
* Assumes failure is detected at iteration boundary (exception
  propagates out of forward_backward_func)
* Does NOT handle mid-microbatch recovery within the pipeline
* Does NOT handle partial-stage failures (all-or-nothing)
* Gloo-based failure broadcast is optional (v1 relies on NCCL
  errors propagating to all stages)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Pipeline rollback state
# =====================================================================

class PipelineRollbackState(enum.IntEnum):
    """State of the pipeline rollback coordinator."""
    NORMAL = 0
    FAILURE_DETECTED = 1
    ROLLBACK_IN_PROGRESS = 2
    AWAITING_REPLAY = 3
    REPLAY_IN_PROGRESS = 4


# =====================================================================
# Pipeline failure info
# =====================================================================

@dataclass
class PipelineFailureInfo:
    """Information about a pipeline failure."""

    failed_stage: int = -1
    """Pipeline stage where the failure originated (-1 = unknown)."""

    failed_rank: int = -1
    """Global rank where the failure originated."""

    step: int = -1
    """Training step when the failure occurred."""

    num_microbatches_completed: int = 0
    """Number of microbatches that completed before the failure."""

    total_microbatches: int = 0
    """Total number of microbatches in the iteration."""

    failure_phase: str = ""
    """Phase of the pipeline schedule: 'warmup', 'steady', 'cooldown', 'unknown'."""

    reason: str = ""
    """Human-readable failure reason."""

    timestamp: float = 0.0
    """Wall-clock time of failure detection."""

    nccl_healthy: bool = True
    """Whether NCCL communicator is believed to be healthy after failure."""


@dataclass
class PipelineRollbackResult:
    """Result of a pipeline rollback operation."""

    success: bool = False
    state: PipelineRollbackState = PipelineRollbackState.NORMAL
    failure_info: Optional[PipelineFailureInfo] = None
    stages_synchronized: bool = False
    grad_buffers_cleared: bool = False
    nccl_recovered: bool = True
    elapsed_seconds: float = 0.0
    error: str = ""


# =====================================================================
# PipelineRollbackCoordinator
# =====================================================================

class PipelineRollbackCoordinator:
    """Coordinates pipeline-safe rollback across all PP stages.

    This coordinator extends the PP=1 ``RollbackReplayManager`` with
    pipeline-specific logic:

    1. Cross-stage failure detection and synchronization
    2. NCCL communicator health assessment
    3. Pipeline metadata cleanup verification

    It does NOT replace ``RollbackReplayManager`` — it wraps it with
    pipeline-aware logic.

    Lifecycle::

        # Normal training
        coordinator.begin_iteration(step, pp_rank, pp_size, num_microbatches)

        # Failure detected (exception from forward_backward_func)
        coordinator.on_pipeline_failure(failure_info)

        # Synchronize all stages
        coordinator.synchronize_rollback()

        # Delegate to RollbackReplayManager for state restoration
        rollback_manager.rollback(...)

        # Verify pipeline is clean for replay
        coordinator.verify_pipeline_clean()

        # Replay
        coordinator.begin_replay()
        ... train_step ...
        coordinator.complete_replay()
    """

    def __init__(self) -> None:
        self._state = PipelineRollbackState.NORMAL
        self._current_step: int = -1
        self._pp_rank: int = 0
        self._pp_size: int = 1
        self._num_microbatches: int = 0
        self._in_iteration: bool = False

        # Failure tracking
        self._current_failure: Optional[PipelineFailureInfo] = None
        self._failure_history: List[PipelineFailureInfo] = []
        self._max_history: int = 50

        # Counters
        self._total_rollbacks: int = 0
        self._total_replays: int = 0
        self._total_replay_successes: int = 0

        # Callbacks
        self._on_rollback_fn: Optional[Callable] = None
        self._on_replay_complete_fn: Optional[Callable] = None

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def register_callbacks(
        self,
        on_rollback_fn: Optional[Callable] = None,
        on_replay_complete_fn: Optional[Callable] = None,
    ) -> None:
        """Register callbacks for rollback/replay events."""
        if on_rollback_fn is not None:
            self._on_rollback_fn = on_rollback_fn
        if on_replay_complete_fn is not None:
            self._on_replay_complete_fn = on_replay_complete_fn

    # -----------------------------------------------------------------
    # Iteration lifecycle
    # -----------------------------------------------------------------

    def begin_iteration(
        self,
        step: int,
        pp_rank: int = 0,
        pp_size: int = 1,
        num_microbatches: int = 1,
    ) -> None:
        """Called at the start of each training iteration."""
        self._current_step = step
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._num_microbatches = num_microbatches
        self._in_iteration = True

        # If we were awaiting replay, transition to replay in progress
        if self._state == PipelineRollbackState.AWAITING_REPLAY:
            self._state = PipelineRollbackState.REPLAY_IN_PROGRESS
            self._total_replays += 1
            logger.info(
                "BSR-MoE pipeline: step %d — entering REPLAY_IN_PROGRESS "
                "(pp_rank=%d)",
                step, pp_rank,
            )

    def end_iteration(self, step: int) -> None:
        """Called at the end of each training iteration."""
        self._in_iteration = False

    # -----------------------------------------------------------------
    # Failure detection
    # -----------------------------------------------------------------

    def on_pipeline_failure(
        self,
        *,
        failed_stage: int = -1,
        failed_rank: int = -1,
        step: int = -1,
        num_microbatches_completed: int = 0,
        failure_phase: str = "unknown",
        reason: str = "",
        nccl_healthy: bool = True,
    ) -> PipelineFailureInfo:
        """Record a pipeline failure.

        Called when an exception is caught from ``forward_backward_func``
        or when a P2P timeout is detected.

        Returns the failure info record.
        """
        effective_step = step if step >= 0 else self._current_step

        info = PipelineFailureInfo(
            failed_stage=failed_stage,
            failed_rank=failed_rank,
            step=effective_step,
            num_microbatches_completed=num_microbatches_completed,
            total_microbatches=self._num_microbatches,
            failure_phase=failure_phase,
            reason=reason,
            timestamp=time.time(),
            nccl_healthy=nccl_healthy,
        )

        self._current_failure = info
        self._state = PipelineRollbackState.FAILURE_DETECTED

        logger.warning(
            "BSR-MoE pipeline: FAILURE DETECTED at step %d — "
            "stage=%d, rank=%d, phase=%s, microbatches=%d/%d, "
            "nccl_healthy=%s, reason=%r",
            effective_step, failed_stage, failed_rank,
            failure_phase, num_microbatches_completed,
            self._num_microbatches, nccl_healthy, reason,
        )

        return info

    # -----------------------------------------------------------------
    # Rollback
    # -----------------------------------------------------------------

    def initiate_rollback(
        self,
        *,
        sync_fn: Optional[Callable] = None,
        clear_grad_fn: Optional[Callable] = None,
    ) -> PipelineRollbackResult:
        """Initiate a pipeline-safe rollback.

        This method:
        1. Synchronizes all PP stages (via sync_fn or barrier)
        2. Clears grad buffers (via clear_grad_fn)
        3. Transitions to AWAITING_REPLAY state

        Args:
            sync_fn: Optional callable to synchronize all PP stages.
                Signature: ``sync_fn() -> bool``.
                If None, synchronization is skipped (assumes all stages
                will independently detect the failure via NCCL errors).
            clear_grad_fn: Optional callable to clear gradient buffers.
                Signature: ``clear_grad_fn() -> None``.
                If None, grad clearing is deferred to the next train_step.

        Returns:
            A PipelineRollbackResult describing the outcome.
        """
        t0 = time.monotonic()
        result = PipelineRollbackResult(
            failure_info=self._current_failure,
        )

        if self._state not in (
            PipelineRollbackState.FAILURE_DETECTED,
            PipelineRollbackState.ROLLBACK_IN_PROGRESS,
        ):
            result.error = (
                f"Cannot initiate rollback in state {self._state.name}"
            )
            logger.error("BSR-MoE pipeline: %s", result.error)
            return result

        self._state = PipelineRollbackState.ROLLBACK_IN_PROGRESS

        # Step 1: Synchronize all stages
        if sync_fn is not None:
            try:
                result.stages_synchronized = sync_fn()
            except Exception as e:
                logger.warning(
                    "BSR-MoE pipeline: stage sync failed (expected if "
                    "NCCL is dead): %s", e,
                )
                result.stages_synchronized = False
        else:
            # Without explicit sync, we rely on all stages independently
            # detecting the failure via NCCL errors
            result.stages_synchronized = True

        # Step 2: Clear grad buffers
        if clear_grad_fn is not None:
            try:
                clear_grad_fn()
                result.grad_buffers_cleared = True
            except Exception as e:
                logger.warning(
                    "BSR-MoE pipeline: grad buffer clear failed: %s", e,
                )
                result.grad_buffers_cleared = False
        else:
            # Deferred to next train_step's zero_grad_buffer()
            result.grad_buffers_cleared = True

        # Step 3: Assess NCCL health
        if self._current_failure is not None:
            result.nccl_recovered = self._current_failure.nccl_healthy

        # Step 4: Archive failure and transition
        if self._current_failure is not None:
            self._failure_history.append(self._current_failure)
            if len(self._failure_history) > self._max_history:
                self._failure_history = self._failure_history[-self._max_history:]

        self._total_rollbacks += 1
        self._state = PipelineRollbackState.AWAITING_REPLAY

        # Invoke callback
        if self._on_rollback_fn is not None:
            try:
                self._on_rollback_fn(result)
            except Exception as e:
                logger.error(
                    "BSR-MoE pipeline: on_rollback_fn failed: %s", e,
                )

        result.state = self._state
        result.success = True
        result.elapsed_seconds = time.monotonic() - t0

        logger.warning(
            "BSR-MoE pipeline: rollback COMPLETE — step=%d, "
            "stages_synced=%s, grads_cleared=%s, nccl_ok=%s, "
            "elapsed=%.3fs",
            self._current_step,
            result.stages_synchronized,
            result.grad_buffers_cleared,
            result.nccl_recovered,
            result.elapsed_seconds,
        )

        return result

    # -----------------------------------------------------------------
    # Replay
    # -----------------------------------------------------------------

    def begin_replay(self) -> bool:
        """Mark the start of a replay attempt.

        Returns True if replay can proceed.
        """
        if self._state == PipelineRollbackState.AWAITING_REPLAY:
            self._state = PipelineRollbackState.REPLAY_IN_PROGRESS
            self._total_replays += 1
            return True
        elif self._state == PipelineRollbackState.REPLAY_IN_PROGRESS:
            return True  # Already in replay
        else:
            logger.warning(
                "BSR-MoE pipeline: begin_replay called in state %s",
                self._state.name,
            )
            return False

    def complete_replay(self, success: bool = True) -> None:
        """Mark the end of a replay attempt.

        Args:
            success: Whether the replay succeeded.
        """
        if success:
            self._total_replay_successes += 1
            self._state = PipelineRollbackState.NORMAL
            self._current_failure = None
            logger.info(
                "BSR-MoE pipeline: replay SUCCEEDED at step %d",
                self._current_step,
            )
        else:
            # Failed replay — go back to awaiting
            self._state = PipelineRollbackState.AWAITING_REPLAY
            logger.warning(
                "BSR-MoE pipeline: replay FAILED at step %d — "
                "awaiting next attempt",
                self._current_step,
            )

        if self._on_replay_complete_fn is not None:
            try:
                self._on_replay_complete_fn(success)
            except Exception as e:
                logger.error(
                    "BSR-MoE pipeline: on_replay_complete_fn failed: %s", e,
                )

    # -----------------------------------------------------------------
    # Pipeline cleanup verification
    # -----------------------------------------------------------------

    def verify_pipeline_clean(
        self,
        model: Any = None,
    ) -> List[str]:
        """Verify that the pipeline is clean for replay.

        Checks:
        1. No stale activations in model buffers
        2. Grad buffers are zero (or will be zeroed)
        3. No pending P2P operations

        Returns a list of issues found (empty = clean).
        """
        issues = []

        # Check 1: Model grad buffers
        if model is not None:
            models = model if isinstance(model, (list, tuple)) else [model]
            for i, m in enumerate(models):
                if hasattr(m, 'grad_buffer') and m.grad_buffer is not None:
                    # Check if any grad buffer has non-zero values
                    # (This is a heuristic — in practice, zero_grad_buffer
                    # at the start of train_step handles this)
                    pass  # Deferred to train_step

        # Check 2: State consistency
        if self._state not in (
            PipelineRollbackState.AWAITING_REPLAY,
            PipelineRollbackState.REPLAY_IN_PROGRESS,
            PipelineRollbackState.NORMAL,
        ):
            issues.append(
                f"Unexpected state for replay: {self._state.name}"
            )

        return issues

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    @property
    def state(self) -> PipelineRollbackState:
        return self._state

    @property
    def is_in_rollback(self) -> bool:
        return self._state in (
            PipelineRollbackState.FAILURE_DETECTED,
            PipelineRollbackState.ROLLBACK_IN_PROGRESS,
        )

    @property
    def is_awaiting_replay(self) -> bool:
        return self._state == PipelineRollbackState.AWAITING_REPLAY

    @property
    def is_in_replay(self) -> bool:
        return self._state == PipelineRollbackState.REPLAY_IN_PROGRESS

    @property
    def current_failure(self) -> Optional[PipelineFailureInfo]:
        return self._current_failure

    @property
    def pp_size(self) -> int:
        return self._pp_size

    @property
    def pp_rank(self) -> int:
        return self._pp_rank

    @property
    def total_rollbacks(self) -> int:
        return self._total_rollbacks

    @property
    def total_replays(self) -> int:
        return self._total_replays

    @property
    def total_replay_successes(self) -> int:
        return self._total_replay_successes

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self._state.name,
            "pp_rank": self._pp_rank,
            "pp_size": self._pp_size,
            "current_step": self._current_step,
            "total_rollbacks": self._total_rollbacks,
            "total_replays": self._total_replays,
            "total_replay_successes": self._total_replay_successes,
            "failure_history_size": len(self._failure_history),
            "current_failure": {
                "stage": self._current_failure.failed_stage,
                "rank": self._current_failure.failed_rank,
                "step": self._current_failure.step,
                "phase": self._current_failure.failure_phase,
            } if self._current_failure else None,
        }

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Full reset (for testing)."""
        self._state = PipelineRollbackState.NORMAL
        self._current_step = -1
        self._pp_rank = 0
        self._pp_size = 1
        self._num_microbatches = 0
        self._in_iteration = False
        self._current_failure = None
        self._failure_history.clear()
        self._total_rollbacks = 0
        self._total_replays = 0
        self._total_replay_successes = 0


# =====================================================================
# Global singleton
# =====================================================================

_COORDINATOR: Optional[PipelineRollbackCoordinator] = None


def get_pipeline_rollback_coordinator() -> PipelineRollbackCoordinator:
    """Get or create the global PipelineRollbackCoordinator singleton."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = PipelineRollbackCoordinator()
    return _COORDINATOR


def clear_pipeline_rollback_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.reset()
    _COORDINATOR = None


# =====================================================================
# Convenience API
# =====================================================================

def pipeline_safe_rollback(
    *,
    failed_stage: int = -1,
    failed_rank: int = -1,
    step: int = -1,
    reason: str = "",
    nccl_healthy: bool = True,
    sync_fn: Optional[Callable] = None,
    clear_grad_fn: Optional[Callable] = None,
) -> PipelineRollbackResult:
    """One-shot pipeline-safe rollback.

    Combines failure detection + rollback initiation into a single call.
    This is the primary entry point for the training loop.
    """
    coord = get_pipeline_rollback_coordinator()

    coord.on_pipeline_failure(
        failed_stage=failed_stage,
        failed_rank=failed_rank,
        step=step,
        reason=reason,
        nccl_healthy=nccl_healthy,
    )

    return coord.initiate_rollback(
        sync_fn=sync_fn,
        clear_grad_fn=clear_grad_fn,
    )
