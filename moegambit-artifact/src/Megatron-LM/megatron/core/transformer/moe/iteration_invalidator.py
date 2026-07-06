# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoEGambit Iteration Invalidator.

When a hard failure is detected **inside** an ongoing forward/backward pass
or optimizer step, the current iteration's intermediate state (gradients,
activations, partially-updated parameters) is tainted and must not be
committed.

Lightweight flag-based mechanism:

1. Marks the current iteration as **invalid** (``invalidate()``).
2. Blocks the optimizer from committing the update (``should_skip_optimizer_step()``).
3. Signals the training loop to discard the current iteration's results
   and proceed to the safe-point recovery path (``is_current_iteration_invalid()``).

The invalidator does NOT perform rollback itself — it only sets flags.
The actual rollback (restoring ``consumed_train_samples``, resetting the
data iterator, etc.) is handled by the training loop based on these flags.

Integration with training.py
-----------------------------
::

    # Before train_step:
    invalidator.begin_iteration(step)

    try:
        loss, ... = train_step(...)
    except Exception as e:
        # Hard failure detected inside forward/backward
        detector.report_collective_failure(...)
        # invalidator.invalidate() is called by the detector callback

    if invalidator.is_current_iteration_invalid():
        # Skip optimizer commit, discard loss, proceed to safe-point
        ...
    else:
        iteration += 1
        consumed_train_samples += ...

    # After iteration:
    invalidator.end_iteration(step)

Relationship to existing modules
---------------------------------
* ``HardFailureDetector`` calls ``invalidator.invalidate()`` via its
  ``on_iteration_invalid_fn`` callback.
* ``RecoveryController`` reads ``invalidator.is_current_iteration_invalid()``
  in its ``before_iteration()`` hook to decide whether to enter the
  safe-point repair path.
* The training loop (``training.py``) checks ``should_skip_optimizer_step()``
  to decide whether to call ``optimizer.step()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Invalidation record
# =====================================================================

@dataclass
class InvalidationRecord:
    """Records why an iteration was invalidated."""

    step: int = -1
    """Training step that was invalidated."""

    failed_rank: int = -1
    """The rank whose failure caused the invalidation."""

    reason: str = ""
    """Human-readable reason."""

    timestamp: float = 0.0
    """Wall-clock time of invalidation."""


# =====================================================================
# IterationInvalidator
# =====================================================================

class IterationInvalidator:
    """Tracks whether the current training iteration is valid.

    Lifecycle per iteration::

        begin_iteration(step)   # reset flags for new iteration
        ... forward / backward / optimizer ...
        # if failure detected:  invalidate(step, rank, reason)
        is_current_iteration_invalid()  → True
        should_skip_optimizer_step()    → True
        end_iteration(step)     # archive record, reset flags
    """

    def __init__(self) -> None:
        # Current iteration state
        self._current_step: int = -1
        self._invalid: bool = False
        self._in_iteration: bool = False
        self._current_record: Optional[InvalidationRecord] = None

        # History (bounded)
        self._history: List[InvalidationRecord] = []
        self._max_history: int = 100

        # Counters
        self._total_invalidations: int = 0

    # -----------------------------------------------------------------
    # Iteration lifecycle
    # -----------------------------------------------------------------

    def begin_iteration(self, step: int) -> None:
        """Called at the start of each training iteration.

        Resets the invalid flag for the new iteration.
        """
        self._current_step = step
        self._invalid = False
        self._in_iteration = True
        self._current_record = None

    def end_iteration(self, step: int) -> None:
        """Called at the end of each training iteration.

        If the iteration was invalidated, archives the record.
        """
        if self._invalid and self._current_record is not None:
            self._history.append(self._current_record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self._in_iteration = False
        self._current_record = None
        # NOTE: we do NOT reset _invalid here — the training loop may
        # still need to read it after end_iteration() to decide whether
        # to skip iteration increment.  It is reset by begin_iteration().

    # -----------------------------------------------------------------
    # Invalidation API
    # -----------------------------------------------------------------

    def invalidate(
        self,
        *,
        step: int = -1,
        failed_rank: int = -1,
        reason: str = "",
    ) -> bool:
        """Mark the current iteration as invalid.

        Idempotent: calling multiple times for the same iteration is safe.

        Returns:
            True if this is the first invalidation for the current iteration.
        """
        if self._invalid:
            logger.debug(
                "IterationInvalidator: step %d already invalid, "
                "skipping duplicate (new reason: %s)",
                self._current_step, reason,
            )
            return False

        self._invalid = True
        self._total_invalidations += 1

        effective_step = step if step >= 0 else self._current_step
        self._current_record = InvalidationRecord(
            step=effective_step,
            failed_rank=failed_rank,
            reason=reason,
            timestamp=time.time(),
        )

        logger.warning(
            "MoEGambit IterationInvalidator: step %d INVALIDATED "
            "(failed_rank=%d, reason=%r)",
            effective_step, failed_rank, reason,
        )

        return True

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    def is_current_iteration_invalid(self) -> bool:
        """Return True if the current iteration has been invalidated."""
        return self._invalid

    def should_skip_optimizer_step(self) -> bool:
        """Return True if the optimizer step should be skipped.

        This is equivalent to ``is_current_iteration_invalid()`` but
        named explicitly for clarity at the call site in training.py.
        """
        return self._invalid

    @property
    def in_iteration(self) -> bool:
        """True if between begin_iteration and end_iteration."""
        return self._in_iteration

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_invalidations(self) -> int:
        return self._total_invalidations

    @property
    def last_invalidation(self) -> Optional[InvalidationRecord]:
        """Most recent invalidation record (from history or current)."""
        if self._current_record is not None:
            return self._current_record
        if self._history:
            return self._history[-1]
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "is_invalid": self._invalid,
            "in_iteration": self._in_iteration,
            "total_invalidations": self._total_invalidations,
            "history_size": len(self._history),
        }

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Full reset (for testing)."""
        self._current_step = -1
        self._invalid = False
        self._in_iteration = False
        self._current_record = None
        self._history.clear()
        self._total_invalidations = 0


# =====================================================================
# Global singleton
# =====================================================================

_INVALIDATOR: Optional[IterationInvalidator] = None


def get_iteration_invalidator() -> IterationInvalidator:
    """Get or create the global IterationInvalidator singleton."""
    global _INVALIDATOR
    if _INVALIDATOR is None:
        _INVALIDATOR = IterationInvalidator()
    return _INVALIDATOR


def clear_iteration_invalidator() -> None:
    """Reset the global invalidator (for testing)."""
    global _INVALIDATOR
    if _INVALIDATOR is not None:
        _INVALIDATOR.reset()
    _INVALIDATOR = None
