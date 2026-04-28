# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BSR-MoE Hard Failure Detector.

Detects rank-level hard failures — situations where a rank has exited,
crashed, or can no longer participate in NCCL collectives.

Hard failure sources
--------------------
1. **Rank process exit** — the OS process on a remote node has terminated.
2. **Collective failure** — an all-reduce, all-to-all, or other collective
   raises a RuntimeError / NCCL error.
3. **P2P / pipeline failure** — a send/recv operation times out or errors
   (PP > 1 path; interface only in v1).

Detection strategy (v1 — PP = 1)
---------------------------------
The detector does NOT run a background heartbeat thread.  Instead it
provides two entry points that the training loop calls:

* ``report_failure(rank, reason)`` — called by any code path that catches
  a communication exception (e.g. a try/except around ``forward_backward_func``).
* ``check_via_gloo(gloo_group)`` — an optional cooperative check at a safe
  point using a Gloo all-reduce (Gloo is more resilient to NCCL failures).

Both entry points funnel into the same ``RecoveryController.on_hard_rank_failure()``
path, reusing the existing quarantine / health-mask / expert-directory stack.

PP > 1 (interface only)
-----------------------
``report_pipeline_stage_failure(stage_rank, reason)`` is provided as a
placeholder.  The actual P2P timeout detection will be added in a later
phase when pipeline schedule modifications are implemented.

Relationship to soft failure
----------------------------
A hard failure **always** implies quarantine.  The detector calls
``RecoveryController.on_hard_rank_failure()`` which internally invokes
the quarantine callback, so the existing health-mask / routing isolation
is automatically applied.  There is no separate "hard quarantine" state —
the ``FaultRecord.fault_type`` field distinguishes soft vs hard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =====================================================================
# Failure source classification
# =====================================================================

class FailureSource(IntEnum):
    """How the hard failure was detected."""

    UNKNOWN = 0
    """Source not specified."""

    COLLECTIVE_ERROR = 1
    """A collective operation (all-reduce, all-to-all, reduce-scatter, etc.)
    raised a fatal error (typically NCCL timeout or internal error)."""

    P2P_TIMEOUT = 2
    """A point-to-point send/recv timed out (PP > 1 path)."""

    RANK_EXIT = 3
    """The remote rank's process has exited (detected via Gloo probe or
    external monitor)."""

    GLOO_PROBE = 4
    """Detected via a cooperative Gloo all-reduce health check."""

    EXTERNAL_MONITOR = 5
    """Reported by an external fault-tolerance monitor (e.g. nvidia-resiliency-ext
    RankMonitorClient, or a custom watchdog)."""

    MANUAL_INJECTION = 6
    """Injected manually for testing (via ``FaultInjector`` or API call)."""


# =====================================================================
# Failure record
# =====================================================================

@dataclass
class HardFailureRecord:
    """Immutable record of a single hard failure detection event."""

    failed_rank: int = -1
    """Global rank that failed."""

    source: FailureSource = FailureSource.UNKNOWN
    """How the failure was detected."""

    reason: str = ""
    """Human-readable description."""

    step: int = -1
    """Training step at which the failure was detected (-1 = unknown)."""

    timestamp: float = 0.0
    """Wall-clock time of detection."""

    mid_iteration: bool = False
    """True if the failure was detected inside forward/backward/optimizer
    (i.e. the current iteration is tainted and must be invalidated)."""

    exception_type: str = ""
    """String representation of the exception class, if any."""


# =====================================================================
# HardFailureDetector
# =====================================================================

class HardFailureDetector:
    """Detects and records hard rank failures.

    This class is a **local bookkeeper** — it does not perform distributed
    communication itself.  It records failure reports and, when a new
    failure is detected, invokes a registered callback (typically
    ``RecoveryController.on_hard_rank_failure``).

    Thread safety: not thread-safe.  All calls are expected from the main
    training thread.
    """

    def __init__(self) -> None:
        self._detected: Dict[int, HardFailureRecord] = {}  # rank → record
        self._on_hard_failure_fn: Optional[Callable] = None
        self._on_iteration_invalid_fn: Optional[Callable] = None

    # -----------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------

    def register_callbacks(
        self,
        *,
        on_hard_failure_fn: Optional[Callable] = None,
        on_iteration_invalid_fn: Optional[Callable] = None,
    ) -> None:
        """Register callbacks invoked when a hard failure is detected.

        Args:
            on_hard_failure_fn: Called with ``(failed_rank, reason, step,
                expert_ids, ep_group_ranks, dp_group_ranks)`` keyword args.
                Typically wired to ``RecoveryController.on_hard_rank_failure``.
            on_iteration_invalid_fn: Called with ``(step, failed_rank, reason)``
                keyword args when the failure occurred mid-iteration.
                Typically wired to ``IterationInvalidator.invalidate``.
        """
        if on_hard_failure_fn is not None:
            self._on_hard_failure_fn = on_hard_failure_fn
        if on_iteration_invalid_fn is not None:
            self._on_iteration_invalid_fn = on_iteration_invalid_fn

    # -----------------------------------------------------------------
    # Failure reporting API
    # -----------------------------------------------------------------

    def report_failure(
        self,
        failed_rank: int,
        *,
        source: FailureSource = FailureSource.UNKNOWN,
        reason: str = "",
        step: int = -1,
        mid_iteration: bool = False,
        exception_type: str = "",
        expert_ids: Optional[List[int]] = None,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
    ) -> bool:
        """Report a hard failure for a rank.

        Idempotent: if the rank is already recorded, the callback is NOT
        re-invoked and ``False`` is returned.

        Args:
            failed_rank: Global rank that failed.
            source: How the failure was detected.
            reason: Human-readable description.
            step: Current training step.
            mid_iteration: True if inside forward/backward/optimizer.
            exception_type: Exception class name (for diagnostics).
            expert_ids: Global expert IDs on the failed rank (optional;
                the callback can compute this from placement if not given).
            ep_group_ranks: EP group ranks (for group rebuild context).
            dp_group_ranks: DP group ranks (for group rebuild context).

        Returns:
            True if this is a *new* failure (callback was invoked).
            False if the rank was already recorded.
        """
        if failed_rank in self._detected:
            logger.debug(
                "HardFailureDetector: rank %d already recorded, skipping "
                "(step=%d)", failed_rank, step,
            )
            return False

        record = HardFailureRecord(
            failed_rank=failed_rank,
            source=source,
            reason=reason,
            step=step,
            timestamp=time.time(),
            mid_iteration=mid_iteration,
            exception_type=exception_type,
        )
        self._detected[failed_rank] = record

        logger.warning(
            "BSR-MoE HardFailureDetector: rank %d HARD FAILED "
            "(source=%s, reason=%r, step=%d, mid_iteration=%s)",
            failed_rank, source.name, reason, step, mid_iteration,
        )

        # --- Trigger iteration invalidation if mid-iteration ---
        if mid_iteration and self._on_iteration_invalid_fn is not None:
            self._on_iteration_invalid_fn(
                step=step,
                failed_rank=failed_rank,
                reason=f"hard_failure:{source.name}:{reason}",
            )

        # --- Trigger recovery controller ---
        if self._on_hard_failure_fn is not None:
            self._on_hard_failure_fn(
                failed_rank=failed_rank,
                reason=reason or f"hard_failure:{source.name}",
                step=step,
                expert_ids=expert_ids,
                ep_group_ranks=ep_group_ranks,
                dp_group_ranks=dp_group_ranks,
                mid_iteration=mid_iteration,
            )

        return True

    def report_collective_failure(
        self,
        failed_rank: int,
        *,
        reason: str = "",
        step: int = -1,
        mid_iteration: bool = True,
        exception: Optional[BaseException] = None,
        expert_ids: Optional[List[int]] = None,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
    ) -> bool:
        """Convenience: report a failure detected via a collective error.

        Typically called from a try/except around forward_backward_func or
        finalize_model_grads.
        """
        exc_type = type(exception).__name__ if exception else ""
        return self.report_failure(
            failed_rank,
            source=FailureSource.COLLECTIVE_ERROR,
            reason=reason or str(exception),
            step=step,
            mid_iteration=mid_iteration,
            exception_type=exc_type,
            expert_ids=expert_ids,
            ep_group_ranks=ep_group_ranks,
            dp_group_ranks=dp_group_ranks,
        )

    def report_pipeline_stage_failure(
        self,
        stage_rank: int,
        *,
        reason: str = "",
        step: int = -1,
        mid_iteration: bool = True,
        expert_ids: Optional[List[int]] = None,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
    ) -> bool:
        """Report a pipeline stage failure (PP > 1 placeholder).

        In v1 this has the same implementation as ``report_failure`` with
        ``source=P2P_TIMEOUT``.  Future phases will add P2P timeout
        detection in the pipeline schedule.
        """
        return self.report_failure(
            stage_rank,
            source=FailureSource.P2P_TIMEOUT,
            reason=reason,
            step=step,
            mid_iteration=mid_iteration,
            expert_ids=expert_ids,
            ep_group_ranks=ep_group_ranks,
            dp_group_ranks=dp_group_ranks,
        )

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    def is_failed(self, rank: int) -> bool:
        """Return True if the given rank has been recorded as hard-failed."""
        return rank in self._detected

    @property
    def failed_ranks(self) -> Set[int]:
        """Set of all hard-failed ranks."""
        return set(self._detected.keys())

    @property
    def num_failures(self) -> int:
        return len(self._detected)

    def get_record(self, rank: int) -> Optional[HardFailureRecord]:
        return self._detected.get(rank)

    @property
    def has_mid_iteration_failure(self) -> bool:
        """True if any recorded failure occurred mid-iteration."""
        return any(r.mid_iteration for r in self._detected.values())

    def summary(self) -> Dict:
        return {
            "num_failures": self.num_failures,
            "failed_ranks": sorted(self._detected.keys()),
            "has_mid_iteration_failure": self.has_mid_iteration_failure,
        }

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def clear_rank(self, rank: int) -> None:
        """Remove a rank from the detected set (after successful recovery)."""
        self._detected.pop(rank, None)

    def reset(self) -> None:
        """Clear all records (for testing)."""
        self._detected.clear()


# =====================================================================
# Global singleton
# =====================================================================

_DETECTOR: Optional[HardFailureDetector] = None


def get_hard_failure_detector() -> HardFailureDetector:
    """Get or create the global HardFailureDetector singleton."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = HardFailureDetector()
    return _DETECTOR


def clear_hard_failure_detector() -> None:
    """Reset the global detector (for testing)."""
    global _DETECTOR
    if _DETECTOR is not None:
        _DETECTOR.reset()
    _DETECTOR = None
