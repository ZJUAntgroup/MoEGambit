# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MOEGAMBIT-MoE expert staleness exposure tracker.

Tracks hybrid-recovery-induced stale iterations per logical rank within a
sliding window.  This module is used by the ``RankExposureGuardedHybridPolicy``
to decide whether a rank has been "stale too often" and should trigger a
checkpoint restart instead of another hybrid recovery.

Key concepts
------------

* **Stale iterations**: When a hybrid recovery occurs at step *s* with gap
  *g*, the recovered rank has *g* stale iterations (its weights were
  *g* steps behind the rest of the cluster).  These *g* iterations are
  added to the rank's cumulative stale-iter counter within the window.

* **Stale exposure ratio**: ``stale_iters / window_steps``.  If this
  exceeds ``max_rank_stale_exposure``, the policy prefers checkpoint
  restart for the next fault on that rank.

* **Only hybrid recovery counts**: Checkpoint restarts reload *all* ranks
  uniformly from the same checkpoint, so no rank is relatively "stale".
  Therefore, checkpoint restart events are **not** recorded.

Thread safety
-------------

This class is **not** thread-safe.  It is designed to be called from the
RecoveryController's event loop, which is single-threaded (training
iterations are sequential).  If concurrent access is needed, external
locking must be provided.

Persistence
-----------

RankExposureTracker supports serialization via ``to_state_dict()`` and
``from_state_dict()`` so that stale-iteration history can survive
controller restarts (e.g., after a checkpoint restart).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HybridRecoveryEvent:
    """A single hybrid recovery event record.

    Attributes:
        step: Training iteration when recovery occurred.
        rank: Logical rank that was recovered.
        gap: Number of stale iterations (current_step - checkpoint_step).
        recovery_path: Always ``"hybrid_recovery"``.
    """

    step: int
    rank: int
    gap: int
    recovery_path: str = "hybrid_recovery"
    num_affected_experts: int = 1

    @property
    def expert_iteration_debt(self) -> int:
        """Return this event's contribution to the paper's S(t)."""
        return self.gap * self.num_affected_experts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "step": self.step,
            "rank": self.rank,
            "gap": self.gap,
            "recovery_path": self.recovery_path,
            "num_affected_experts": self.num_affected_experts,
            "expert_iteration_debt": self.expert_iteration_debt,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HybridRecoveryEvent":
        """Deserialize from a dict."""
        return cls(
            step=d["step"],
            rank=d["rank"],
            gap=d["gap"],
            recovery_path=d.get("recovery_path", "hybrid_recovery"),
            # Version-1 tracker checkpoints did not persist the number of
            # affected experts.  Treat each old event as one expert so that
            # restoring the tracker is conservative and backward compatible.
            num_affected_experts=max(1, int(d.get("num_affected_experts", 1))),
        )


class RankExposureTracker:
    """Tracks hybrid-recovery-induced stale iterations per logical rank.

    Usage::

        tracker = RankExposureTracker()

        # Record a hybrid recovery event
        tracker.record_hybrid_recovery(step=500, rank=3, gap=30)

        # Query stale iterations for a rank
        stale = tracker.get_rank_stale_iters(rank=3, current_step=600,
                                              window_steps=1000)

        # Query stale exposure ratio
        exposure = tracker.get_rank_stale_exposure(rank=3,
                                                     current_step=600,
                                                     window_steps=1000)

        # Get all ranks' exposures
        all_exp = tracker.get_all_rank_exposures(current_step=600,
                                                   window_steps=1000)

    Args:
        max_events: Maximum number of events to keep in memory.
            Older events are pruned when this limit is exceeded.
            Set to 0 or negative for unlimited.  Default 10000.
    """

    def __init__(self, max_events: int = 10000) -> None:
        self._events: List[HybridRecoveryEvent] = []
        self._max_events = max_events
        # _lock is provided for optional thread-safety if the tracker
        # is ever accessed from multiple threads.  The RecoveryController
        # is single-threaded, so this lock is not strictly necessary
        # under normal operation.  Using a non-reentrant lock for
        # minimal overhead.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_hybrid_recovery(
        self,
        step: int,
        rank: int,
        gap: int,
        num_affected_experts: int = 1,
    ) -> None:
        """Record a hybrid recovery event.

        Args:
            step: Training iteration when the recovery occurred.
            rank: Logical rank that was recovered.
            gap: Number of stale iterations introduced
                (``current_step - checkpoint_step``).
        """
        if gap < 0:
            logger.warning(
                "RankExposureTracker: ignoring event with negative gap "
                "(step=%d, rank=%d, gap=%d)",
                step, rank, gap,
            )
            return

        if num_affected_experts <= 0:
            raise ValueError(
                "num_affected_experts must be positive, "
                f"got {num_affected_experts}"
            )

        event = HybridRecoveryEvent(
            step=step,
            rank=rank,
            gap=gap,
            num_affected_experts=num_affected_experts,
        )
        with self._lock:
            self._events.append(event)
            # Enforce max_events limit
            if self._max_events > 0 and len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        logger.info(
            "RankExposureTracker: recorded hybrid recovery "
            "(step=%d, rank=%d, gap=%d, affected_experts=%d, "
            "expert_iteration_debt=%d, total_events=%d)",
            step, rank, gap, num_affected_experts,
            event.expert_iteration_debt, len(self._events),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_rank_stale_iters(
        self,
        rank: int,
        current_step: int,
        window_steps: int,
    ) -> int:
        """Get the total stale iterations for a rank within the window.

        Args:
            rank: Logical rank to query.
            current_step: Current training iteration.
            window_steps: Size of the sliding window.

        Returns:
            Total number of stale iterations for this rank within
            ``[current_step - window_steps, current_step]``.
            Returns 0 if ``window_steps <= 0``.
        """
        if window_steps <= 0:
            return 0
        with self._lock:
            self._prune_unlocked(current_step, window_steps)
            window_start = max(0, current_step - window_steps)
            total = 0
            for ev in self._events:
                if ev.rank == rank and ev.step >= window_start:
                    total += ev.gap
            return total

    def get_rank_stale_exposure(
        self,
        rank: int,
        current_step: int,
        window_steps: int,
    ) -> float:
        """Get the stale exposure ratio for a rank within the window.

        Args:
            rank: Logical rank to query.
            current_step: Current training iteration.
            window_steps: Size of the sliding window.

        Returns:
            ``stale_iters / window_steps``, or 0 if window_steps <= 0
            or no events found.
        """
        if window_steps <= 0:
            return 0.0
        stale_iters = self.get_rank_stale_iters(rank, current_step, window_steps)
        return stale_iters / window_steps

    def get_all_rank_exposures(
        self,
        current_step: int,
        window_steps: int,
    ) -> Dict[int, float]:
        """Get stale exposure ratios for all ranks within the window.

        Args:
            current_step: Current training iteration.
            window_steps: Size of the sliding window.

        Returns:
            Dict mapping rank -> stale exposure ratio.
            Only ranks with non-zero exposure are included.
        """
        with self._lock:
            self._prune_unlocked(current_step, window_steps)
            window_start = max(0, current_step - window_steps)
            stale_by_rank: Dict[int, int] = {}
            for ev in self._events:
                if ev.step >= window_start:
                    stale_by_rank[ev.rank] = stale_by_rank.get(ev.rank, 0) + ev.gap
            if window_steps <= 0:
                return {}
            return {
                rank: stale / window_steps
                for rank, stale in stale_by_rank.items()
            }

    def get_window_expert_iteration_debt(
        self,
        current_step: int,
        window_steps: int,
    ) -> int:
        """Return S(t), summed across all hybrid recoveries in the window."""
        if window_steps <= 0:
            return 0
        with self._lock:
            self._prune_unlocked(current_step, window_steps)
            window_start = max(0, current_step - window_steps)
            return sum(
                ev.expert_iteration_debt
                for ev in self._events
                if ev.step >= window_start
            )

    def get_expert_staleness_density(
        self,
        current_step: int,
        window_steps: int,
        num_experts: int,
    ) -> float:
        """Return Phi(t) = S(t) / (N_expert * W_exp)."""
        if window_steps <= 0 or num_experts <= 0:
            return 0.0
        debt = self.get_window_expert_iteration_debt(current_step, window_steps)
        return debt / (num_experts * window_steps)

    def get_event_count(self) -> int:
        """Return the total number of recorded events."""
        with self._lock:
            return len(self._events)

    def get_events_for_rank(
        self,
        rank: int,
        current_step: int,
        window_steps: int,
    ) -> List[HybridRecoveryEvent]:
        """Return all events for a specific rank within the window.

        Useful for debugging and testing.
        """
        with self._lock:
            self._prune_unlocked(current_step, window_steps)
            window_start = max(0, current_step - window_steps)
            return [
                ev for ev in self._events
                if ev.rank == rank and ev.step >= window_start
            ]

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, current_step: int, window_steps: int) -> int:
        """Remove events outside the sliding window.

        Args:
            current_step: Current training iteration.
            window_steps: Size of the sliding window.

        Returns:
            Number of events pruned.
        """
        with self._lock:
            return self._prune_unlocked(current_step, window_steps)

    def _prune_unlocked(self, current_step: int, window_steps: int) -> int:
        """Internal prune (caller must hold self._lock)."""
        if window_steps <= 0:
            return 0
        window_start = max(0, current_step - window_steps)
        before = len(self._events)
        self._events = [ev for ev in self._events if ev.step >= window_start]
        pruned = before - len(self._events)
        return pruned

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for checkpoint persistence.

        Returns:
            A dict containing all events, suitable for JSON serialization.
        """
        with self._lock:
            return {
                "version": 2,
                "max_events": self._max_events,
                "events": [ev.to_dict() for ev in self._events],
            }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "RankExposureTracker":
        """Deserialize tracker state from a checkpoint.

        Args:
            state: A dict previously returned by ``to_state_dict()``.

        Returns:
            A new ``RankExposureTracker`` with the restored state.
        """
        max_events = state.get("max_events", 10000)
        tracker = cls(max_events=max_events)
        events_data = state.get("events", [])
        tracker._events = [
            HybridRecoveryEvent.from_dict(d) for d in events_data
        ]
        logger.info(
            "RankExposureTracker: restored %d events from state_dict",
            len(tracker._events),
        )
        return tracker

    def reset(self) -> None:
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()

    def __repr__(self) -> str:
        return (
            f"RankExposureTracker(events={len(self._events)}, "
            f"max_events={self._max_events})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_TRACKER: Optional[RankExposureTracker] = None


def get_rank_exposure_tracker() -> RankExposureTracker:
    """Get or create the global tracker singleton."""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = RankExposureTracker()
    return _TRACKER


def set_rank_exposure_tracker(tracker: RankExposureTracker) -> None:
    """Replace the global tracker singleton (for testing or restoration)."""
    global _TRACKER
    _TRACKER = tracker


def reset_rank_exposure_tracker() -> None:
    """Reset the global tracker singleton (for testing)."""
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.reset()
    _TRACKER = None
