# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MOEGAMBIT-MoE Iteration Rollback & Replay Manager.

When a hard failure is detected mid-iteration (inside forward/backward/
optimizer), the current iteration's intermediate state is tainted.  This
module provides the mechanism to:

1. **Snapshot** the iteration boundary state *before* each iteration begins.
2. **Rollback** to that snapshot when a failure invalidates the current
   iteration (restore ``consumed_train_samples``, ``iteration``, etc.).
3. **Replay** the same iteration using the same data (via
   ``RerunDataIterator.rewind()``).

Design principles
-----------------
* **No model-parameter deep-copy.**  When a failure occurs mid-iteration,
  ``optimizer.step()`` has NOT yet been called, so model parameters are
  still at the state of the *previous* completed iteration.  We only need
  to restore bookkeeping counters (``consumed_train_samples``, ``iteration``,
  ``num_floating_point_operations_so_far``).

* **Data replay via RerunDataIterator.**  Megatron's ``RerunDataIterator``
  already buffers microbatches consumed during the current iteration.
  Calling ``rewind()`` replays the same data; calling ``advance()`` drops
  the buffer and moves to the next batch.  We leverage this directly.

* **Grad buffer cleanup.**  After a failed iteration, gradient buffers
  contain partial/corrupt data.  The next ``train_step`` call will
  ``zero_grad_buffer()`` + ``optimizer.zero_grad()`` at the top, so no
  explicit cleanup is needed from the rollback manager.  However, we
  provide a ``cleanup_fn`` callback for any additional cleanup that may
  be required (e.g. clearing CUDA caches, resetting NCCL communicators).

* **PP > 1 placeholder.**  The ``RollbackReplayManager`` accepts an
  optional ``pipeline_rollback_fn`` callback.  In v1 (PP=1) this is
  unused.  Future phases will implement pipeline-safe rollback by
  coordinating across pipeline stages.

Snapshot contents
-----------------
::

    IterationSnapshot:
        iteration: int              # training step number
        consumed_train_samples: int # total samples consumed so far
        consumed_valid_samples: int # total valid samples consumed
        num_fp_ops_so_far: int      # floating-point operations counter
        optimizer_step_committed: bool  # always False at snapshot time

The snapshot is taken at the *beginning* of each iteration (before
``train_step``), so it represents the last consistent boundary.

Rollback protocol (PP=1)
------------------------
1. Restore ``args.consumed_train_samples`` from snapshot.
2. Restore ``args.iteration`` from snapshot (no increment).
3. Restore ``num_floating_point_operations_so_far`` from snapshot.
4. Call ``RerunDataIterator.rewind()`` to replay the same microbatches.
5. Call ``cleanup_fn()`` if registered (optional).
6. Set ``replay_pending = True`` so the training loop knows to re-execute
   ``train_step`` for the same iteration.

Replay protocol
---------------
The training loop checks ``is_replay_pending()`` at the top of each
iteration.  If True:
- The iteration counter is NOT incremented (it was already restored).
- ``train_step`` is called normally — ``RerunDataIterator`` will serve
  the buffered microbatches via ``rewind()``.
- After successful completion, ``complete_replay()`` is called to clear
  the replay flag and call ``RerunDataIterator.advance()``.

Integration with training.py
-----------------------------
::

    rollback_mgr = get_rollback_replay_manager()

    while iteration < train_iters:
        # 1. Snapshot at iteration boundary
        rollback_mgr.snapshot(iteration, args, num_fp_ops)

        # 2. Run train_step (may fail)
        try:
            loss, ... = train_step(...)
        except RuntimeError as e:
            if is_comm_error(e):
                moegambit_report_hard_failure(...)
                # Rollback is triggered by the invalidation path

        # 3. Check invalidation → rollback
        if moegambit_is_current_iteration_invalid():
            rollback_mgr.rollback(args, data_iterator)
            continue  # re-enter loop, replay_pending=True

        # 4. Check replay completion
        if rollback_mgr.is_replay_pending():
            # train_step succeeded on replay
            rollback_mgr.complete_replay(data_iterator)

        # 5. Normal path: advance
        rollback_mgr.advance(data_iterator)
        iteration += 1
        ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Iteration Snapshot
# =====================================================================

@dataclass
class IterationSnapshot:
    """Immutable snapshot of iteration-boundary state.

    Taken at the *beginning* of each iteration, before ``train_step``.
    Represents the last consistent boundary that we can roll back to.
    """

    iteration: int = -1
    """Training step number at snapshot time."""

    consumed_train_samples: int = 0
    """Total training samples consumed up to this point."""

    consumed_valid_samples: int = 0
    """Total validation samples consumed up to this point."""

    num_floating_point_operations_so_far: int = 0
    """Cumulative floating-point operations counter."""

    timestamp: float = 0.0
    """Wall-clock time when the snapshot was taken."""

    valid: bool = False
    """True if this snapshot contains meaningful data (has been taken)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "consumed_train_samples": self.consumed_train_samples,
            "consumed_valid_samples": self.consumed_valid_samples,
            "num_floating_point_operations_so_far": self.num_floating_point_operations_so_far,
            "timestamp": self.timestamp,
            "valid": self.valid,
        }


# =====================================================================
# Rollback / Replay Manager
# =====================================================================

class RollbackReplayManager:
    """Manages iteration rollback and replay for MOEGAMBIT-MoE hard failure recovery.

    Lifecycle per iteration::

        snapshot(step, args, num_fp_ops)   # save boundary state
        ... train_step() ...               # may fail
        # on failure:
        rollback(args, data_iterator)      # restore state, rewind data
        # on next loop iteration (replay):
        train_step() succeeds
        complete_replay(data_iterator)     # clear replay flag, advance data
        # on normal success:
        advance(data_iterator)             # drop buffered data, move forward

    Thread safety: not thread-safe.  All calls from the main training thread.
    """

    def __init__(self) -> None:
        # Current snapshot
        self._snapshot: IterationSnapshot = IterationSnapshot()

        # Replay state
        self._replay_pending: bool = False
        self._replay_count: int = 0  # how many times current iteration replayed

        # History
        self._total_rollbacks: int = 0
        self._total_replays_completed: int = 0
        self._rollback_history: List[Dict[str, Any]] = []
        self._max_history: int = 100

        # Callbacks
        self._cleanup_fn: Optional[Callable] = None
        self._pipeline_rollback_fn: Optional[Callable] = None  # PP > 1 placeholder

        # Configuration
        self._max_replay_attempts: int = 3  # max replays before giving up

    # -----------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------

    def register_callbacks(
        self,
        *,
        cleanup_fn: Optional[Callable] = None,
        pipeline_rollback_fn: Optional[Callable] = None,
    ) -> None:
        """Register optional callbacks.

        Args:
            cleanup_fn: Called during rollback for additional cleanup
                (e.g. clearing CUDA caches, resetting communicators).
                Signature: ``cleanup_fn(snapshot: IterationSnapshot)``.
            pipeline_rollback_fn: PP > 1 placeholder.  Called during
                rollback to coordinate across pipeline stages.
                Signature: ``pipeline_rollback_fn(snapshot: IterationSnapshot)``.
        """
        if cleanup_fn is not None:
            self._cleanup_fn = cleanup_fn
        if pipeline_rollback_fn is not None:
            self._pipeline_rollback_fn = pipeline_rollback_fn

    # -----------------------------------------------------------------
    # Snapshot
    # -----------------------------------------------------------------

    def take_snapshot(
        self,
        iteration: int,
        consumed_train_samples: int,
        consumed_valid_samples: int = 0,
        num_floating_point_operations_so_far: int = 0,
    ) -> None:
        """Take a snapshot of the current iteration boundary state.

        Must be called at the *beginning* of each iteration, before
        ``train_step()``.  This captures the "last known good" state
        that we can roll back to if the iteration fails.

        Args:
            iteration: Current training step number.
            consumed_train_samples: ``args.consumed_train_samples``.
            consumed_valid_samples: ``args.consumed_valid_samples``.
            num_floating_point_operations_so_far: FP ops counter.
        """
        self._snapshot = IterationSnapshot(
            iteration=iteration,
            consumed_train_samples=consumed_train_samples,
            consumed_valid_samples=consumed_valid_samples,
            num_floating_point_operations_so_far=num_floating_point_operations_so_far,
            timestamp=time.time(),
            valid=True,
        )

        logger.debug(
            "MOEGAMBIT-MoE RollbackReplayManager: snapshot taken at iteration %d "
            "(consumed_train_samples=%d)",
            iteration, consumed_train_samples,
        )

    # -----------------------------------------------------------------
    # Rollback
    # -----------------------------------------------------------------

    def rollback(
        self,
        args: Any,
        data_iterators: Any = None,
        num_fp_ops_ref: Optional[List[int]] = None,
    ) -> bool:
        """Roll back to the last snapshot.

        Restores ``args.consumed_train_samples``, ``args.iteration``,
        and optionally ``num_floating_point_operations_so_far``.
        Rewinds the data iterator to replay the same microbatches.

        Args:
            args: Megatron args object (will be mutated in-place).
                Must have ``consumed_train_samples`` and ``iteration``
                attributes.  Can also be a simple namespace/dict-like
                object for testing.
            data_iterators: The ``RerunDataIterator`` (or list of them
                for virtual PP).  ``rewind()`` will be called on each.
                Can be None if data replay is handled externally.
            num_fp_ops_ref: Optional mutable reference (single-element
                list) to ``num_floating_point_operations_so_far``.
                If provided, ``num_fp_ops_ref[0]`` is restored.

        Returns:
            True if rollback was performed, False if no valid snapshot.
        """
        if not self._snapshot.valid:
            logger.error(
                "MOEGAMBIT-MoE RollbackReplayManager: rollback requested but "
                "no valid snapshot exists!"
            )
            return False

        snap = self._snapshot

        # Check replay attempt limit
        self._replay_count += 1
        if self._replay_count > self._max_replay_attempts:
            logger.error(
                "MOEGAMBIT-MoE RollbackReplayManager: max replay attempts (%d) "
                "exceeded for iteration %d. Giving up.",
                self._max_replay_attempts, snap.iteration,
            )
            return False

        logger.warning(
            "MOEGAMBIT-MoE RollbackReplayManager: ROLLBACK to iteration %d "
            "(consumed_train_samples: %d → %d, attempt %d/%d)",
            snap.iteration,
            getattr(args, 'consumed_train_samples', -1),
            snap.consumed_train_samples,
            self._replay_count,
            self._max_replay_attempts,
        )

        # 1. Restore bookkeeping counters
        if hasattr(args, 'consumed_train_samples'):
            args.consumed_train_samples = snap.consumed_train_samples
        if hasattr(args, 'consumed_valid_samples'):
            args.consumed_valid_samples = snap.consumed_valid_samples

        # NOTE: we do NOT restore args.iteration here because the training
        # loop variable `iteration` is a local variable, not args.iteration.
        # The training loop's `continue` statement will re-enter the loop
        # without incrementing `iteration`.

        # 2. Restore FP ops counter
        if num_fp_ops_ref is not None and len(num_fp_ops_ref) > 0:
            num_fp_ops_ref[0] = snap.num_floating_point_operations_so_far

        # 3. Rewind data iterator(s) for replay
        if data_iterators is not None:
            self._rewind_data_iterators(data_iterators)

        # 4. Invoke cleanup callback
        if self._cleanup_fn is not None:
            try:
                self._cleanup_fn(snap)
            except Exception as e:
                logger.error(
                    "MOEGAMBIT-MoE RollbackReplayManager: cleanup_fn failed: %s", e
                )

        # 5. PP > 1 placeholder
        if self._pipeline_rollback_fn is not None:
            try:
                self._pipeline_rollback_fn(snap)
            except Exception as e:
                logger.error(
                    "MOEGAMBIT-MoE RollbackReplayManager: pipeline_rollback_fn "
                    "failed: %s", e
                )

        # 6. Set replay pending
        self._replay_pending = True

        # 7. Record history
        self._total_rollbacks += 1
        record = {
            "iteration": snap.iteration,
            "consumed_train_samples": snap.consumed_train_samples,
            "replay_attempt": self._replay_count,
            "timestamp": time.time(),
        }
        self._rollback_history.append(record)
        if len(self._rollback_history) > self._max_history:
            self._rollback_history = self._rollback_history[-self._max_history:]

        return True

    # -----------------------------------------------------------------
    # Replay
    # -----------------------------------------------------------------

    def is_replay_pending(self) -> bool:
        """True if the current iteration is a replay attempt."""
        return self._replay_pending

    def complete_replay(self, data_iterators: Any = None) -> None:
        """Mark the replay as successfully completed.

        Called after ``train_step`` succeeds on a replayed iteration.
        Advances the data iterator past the replayed microbatches.

        Args:
            data_iterators: The ``RerunDataIterator`` (or list).
                ``advance()`` will be called to drop the replay buffer.
        """
        if not self._replay_pending:
            logger.debug(
                "MOEGAMBIT-MoE RollbackReplayManager: complete_replay called "
                "but no replay was pending"
            )
            return

        logger.warning(
            "MOEGAMBIT-MoE RollbackReplayManager: replay COMPLETED for "
            "iteration %d (attempt %d)",
            self._snapshot.iteration, self._replay_count,
        )

        self._replay_pending = False
        self._replay_count = 0
        self._total_replays_completed += 1

        # Advance data iterator past the replayed microbatches
        if data_iterators is not None:
            self._advance_data_iterators(data_iterators)

    # -----------------------------------------------------------------
    # Advance (normal path)
    # -----------------------------------------------------------------

    def advance(self, data_iterators: Any = None) -> None:
        """Advance past the current iteration (normal success path).

        Called after a successful ``train_step`` on a non-replay iteration.
        Drops the data iterator's microbatch buffer.

        Args:
            data_iterators: The ``RerunDataIterator`` (or list).
        """
        if self._replay_pending:
            # This shouldn't happen — advance is for normal path only
            logger.warning(
                "MOEGAMBIT-MoE RollbackReplayManager: advance() called while "
                "replay is pending — treating as complete_replay()"
            )
            self.complete_replay(data_iterators)
            return

        # Reset replay counter for the next iteration
        self._replay_count = 0

        # Advance data iterator
        if data_iterators is not None:
            self._advance_data_iterators(data_iterators)

    # -----------------------------------------------------------------
    # Internal: data iterator manipulation
    # -----------------------------------------------------------------

    @staticmethod
    def _rewind_data_iterators(data_iterators: Any) -> None:
        """Call rewind() on data iterator(s) to replay microbatches."""
        if data_iterators is None:
            return

        if isinstance(data_iterators, (list, tuple)):
            for di in data_iterators:
                if di is not None and hasattr(di, 'rewind'):
                    di.rewind()
        elif hasattr(data_iterators, 'rewind'):
            data_iterators.rewind()

    @staticmethod
    def _advance_data_iterators(data_iterators: Any) -> None:
        """Call advance() on data iterator(s) to drop replay buffer."""
        if data_iterators is None:
            return

        if isinstance(data_iterators, (list, tuple)):
            for di in data_iterators:
                if di is not None and hasattr(di, 'advance'):
                    di.advance()
        elif hasattr(data_iterators, 'advance'):
            data_iterators.advance()

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    @property
    def has_valid_snapshot(self) -> bool:
        """True if a valid snapshot exists."""
        return self._snapshot.valid

    @property
    def snapshot(self) -> IterationSnapshot:
        """Current snapshot (may be invalid if not yet taken)."""
        return self._snapshot

    @property
    def replay_count(self) -> int:
        """Number of replay attempts for the current iteration."""
        return self._replay_count

    @property
    def max_replay_attempts(self) -> int:
        return self._max_replay_attempts

    @max_replay_attempts.setter
    def max_replay_attempts(self, value: int) -> None:
        self._max_replay_attempts = max(1, value)

    @property
    def total_rollbacks(self) -> int:
        return self._total_rollbacks

    @property
    def total_replays_completed(self) -> int:
        return self._total_replays_completed

    @property
    def exceeded_max_replays(self) -> bool:
        """True if the current iteration has exceeded max replay attempts."""
        return self._replay_count > self._max_replay_attempts

    def summary(self) -> Dict[str, Any]:
        return {
            "has_valid_snapshot": self._snapshot.valid,
            "snapshot_iteration": self._snapshot.iteration,
            "replay_pending": self._replay_pending,
            "replay_count": self._replay_count,
            "max_replay_attempts": self._max_replay_attempts,
            "total_rollbacks": self._total_rollbacks,
            "total_replays_completed": self._total_replays_completed,
        }

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Full reset (for testing)."""
        self._snapshot = IterationSnapshot()
        self._replay_pending = False
        self._replay_count = 0
        self._total_rollbacks = 0
        self._total_replays_completed = 0
        self._rollback_history.clear()


# =====================================================================
# Global singleton
# =====================================================================

_MANAGER: Optional[RollbackReplayManager] = None


def get_rollback_replay_manager() -> RollbackReplayManager:
    """Get or create the global RollbackReplayManager singleton."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = RollbackReplayManager()
    return _MANAGER


def clear_rollback_replay_manager() -> None:
    """Reset the global manager (for testing)."""
    global _MANAGER
    if _MANAGER is not None:
        _MANAGER.reset()
    _MANAGER = None
