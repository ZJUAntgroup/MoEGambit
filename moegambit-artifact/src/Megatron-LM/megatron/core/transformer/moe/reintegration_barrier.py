# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Safe Reintegration Barrier for MoEGambit (Step 14).

This module implements a **controlled reintegration barrier** that ensures a
recovered node only re-joins training when ALL preconditions are satisfied
and the system is at a safe point (iteration boundary).

State machine (3 phases)
------------------------
::

    ISOLATED_READY          ->  REPAIRED_NOT_ROUTED       ->  ROUTED_INTEGRATED
    (replacement ready       (group repaired,              (routing re-enabled,
     but not yet in           topology refreshed,           expert in candidate set,
     training collectives)    expert not yet routable)      normal training)

Precondition gates
------------------
Five preconditions must ALL be satisfied before reintegration can proceed:

1. **replacement_ready** -- replacement rank is online and ready.
2. **groups_repaired** -- affected NCCL process groups have been rebuilt.
3. **directory_refreshed** -- ``ActiveExpertDirectory`` reflects the new
   rank mapping.
4. **topology_refreshed** -- ``DispatchTopology`` has been refreshed with
   the updated rank<->expert mapping.
5. **experts_restorable** -- restored experts satisfy the conditions to
   re-enter the routing candidate set (weights loaded, optimizer state
   available or deferred).

Only when all five gates are marked does ``can_reintegrate()`` return
``True``.  The actual reintegration is executed by
``execute_reintegration()`` at a safe point (iteration boundary).

Design principles
-----------------
* **Gate-based** -- each precondition is independently markable and
  queryable.  The barrier never assumes ordering among preconditions.
* **Self-contained** -- no imports from other MoEGambit modules.  All
  state is fed in via explicit API calls.
* **Deterministic** -- given the same inputs, the barrier always
  produces the same decisions.
* **Auditable** -- every state transition is logged to an event list.

Integration with training loop
------------------------------
::

    # At each iteration boundary (safe point):
    barrier = get_reintegration_barrier()
    barrier.check_safe_point(step)

    # The above will automatically execute reintegration for any
    # failed_rank whose preconditions are all satisfied.

Scope (v1)
----------
* Three-phase state machine (ISOLATED_READY -> REPAIRED_NOT_ROUTED ->
  ROUTED_INTEGRATED)
* Five precondition gates
* Safe-point execution with ordered callbacks
* Event log for audit trail
* Global singleton pattern
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Reintegration phase enum
# =====================================================================

class ReintegrationPhase(enum.IntEnum):
    """Three-phase state machine for reintegration."""

    ISOLATED_READY = 0
    """Replacement rank is ready but not yet in training collectives."""

    REPAIRED_NOT_ROUTED = 1
    """Process groups repaired, topology refreshed, but expert is not
    yet routable (routing still bypasses this expert)."""

    ROUTED_INTEGRATED = 2
    """Routing re-enabled, expert is in the candidate set, normal
    training resumes."""


# =====================================================================
# Required preconditions
# =====================================================================

REQUIRED_PRECONDITIONS: FrozenSet[str] = frozenset({
    "replacement_ready",
    "groups_repaired",
    "directory_refreshed",
    "topology_refreshed",
    "experts_restorable",
})
"""The five precondition gates that must ALL be satisfied."""


# =====================================================================
# Reintegration record (per failed_rank)
# =====================================================================

@dataclass
class ReintegrationRecord:
    """Tracks the reintegration state for a single failed rank."""

    failed_rank: int = -1
    """The rank that failed."""

    replacement_rank: int = -1
    """The replacement rank that took over."""

    expert_ids: List[int] = field(default_factory=list)
    """Global expert IDs hosted by the failed/replacement rank."""

    phase: ReintegrationPhase = ReintegrationPhase.ISOLATED_READY
    """Current phase in the reintegration state machine."""

    begin_step: int = -1
    """Training step when reintegration was initiated."""

    complete_step: int = -1
    """Training step when reintegration completed (-1 if not yet)."""

    met_preconditions: Set[str] = field(default_factory=set)
    """Set of precondition names that have been satisfied."""

    created_at: float = field(default_factory=time.time)
    """Wall-clock time when this record was created."""

    completed_at: float = -1.0
    """Wall-clock time when reintegration completed."""

    def all_preconditions_met(self) -> bool:
        """Return True if all required preconditions are satisfied."""
        return self.met_preconditions >= REQUIRED_PRECONDITIONS

    def missing_preconditions(self) -> FrozenSet[str]:
        """Return the set of preconditions not yet satisfied."""
        return REQUIRED_PRECONDITIONS - self.met_preconditions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "expert_ids": list(self.expert_ids),
            "phase": self.phase.name,
            "begin_step": self.begin_step,
            "complete_step": self.complete_step,
            "met_preconditions": sorted(self.met_preconditions),
            "missing_preconditions": sorted(self.missing_preconditions()),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# =====================================================================
# Reintegration event (audit log)
# =====================================================================

@dataclass
class ReintegrationEvent:
    """A single event in the reintegration audit trail."""

    step: int = -1
    event_type: str = ""
    failed_rank: int = -1
    replacement_rank: int = -1
    phase: str = ""
    precondition: str = ""
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "event_type": self.event_type,
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "phase": self.phase,
            "precondition": self.precondition,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


# =====================================================================
# Reintegration Barrier
# =====================================================================

class ReintegrationBarrier:
    """Controlled barrier for safe expert reintegration.

    Manages per-failed-rank reintegration state machines and enforces
    that all preconditions are met before reintegration proceeds at a
    safe point (iteration boundary).
    """

    def __init__(self) -> None:
        # Per-failed-rank records
        self._records: Dict[int, ReintegrationRecord] = {}

        # Event log
        self._event_log: List[ReintegrationEvent] = []

    # -----------------------------------------------------------------
    # Begin reintegration
    # -----------------------------------------------------------------

    def begin_reintegration(
        self,
        failed_rank: int,
        replacement_rank: int,
        expert_ids: List[int],
        step: int = -1,
    ) -> ReintegrationRecord:
        """Create an ISOLATED_READY record for a failed rank.

        This is called when a replacement rank is identified and ready
        to begin the reintegration process.

        Args:
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            expert_ids: Global expert IDs hosted by this rank.
            step: Current training step.

        Returns:
            The newly created ReintegrationRecord.

        Raises:
            ValueError: If a record already exists for this failed_rank
                and it has not yet completed.
        """
        if failed_rank in self._records:
            existing = self._records[failed_rank]
            if existing.phase != ReintegrationPhase.ROUTED_INTEGRATED:
                raise ValueError(
                    f"Reintegration already in progress for rank {failed_rank} "
                    f"(phase={existing.phase.name}). Cannot begin a new one."
                )

        record = ReintegrationRecord(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            expert_ids=list(expert_ids),
            phase=ReintegrationPhase.ISOLATED_READY,
            begin_step=step,
        )
        self._records[failed_rank] = record

        self._log_event(
            "begin_reintegration",
            step=step,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            phase=ReintegrationPhase.ISOLATED_READY.name,
            details={"expert_ids": list(expert_ids)},
        )

        logger.info(
            "MoEGambit barrier: begin reintegration for rank %d -> %d "
            "(experts=%s, step=%d)",
            failed_rank, replacement_rank, expert_ids, step,
        )

        return record

    # -----------------------------------------------------------------
    # Mark preconditions
    # -----------------------------------------------------------------

    def mark_precondition(
        self,
        failed_rank: int,
        precondition_name: str,
        step: int = -1,
    ) -> bool:
        """Mark a precondition as satisfied for a failed rank.

        Args:
            failed_rank: The rank whose precondition is being marked.
            precondition_name: One of the REQUIRED_PRECONDITIONS.
            step: Current training step.

        Returns:
            True if this was a new precondition (not previously marked).

        Raises:
            KeyError: If no record exists for this failed_rank.
            ValueError: If precondition_name is not in REQUIRED_PRECONDITIONS.
        """
        if failed_rank not in self._records:
            raise KeyError(
                f"No reintegration record for rank {failed_rank}. "
                f"Call begin_reintegration() first."
            )

        if precondition_name not in REQUIRED_PRECONDITIONS:
            raise ValueError(
                f"Unknown precondition: {precondition_name!r}. "
                f"Valid: {sorted(REQUIRED_PRECONDITIONS)}"
            )

        record = self._records[failed_rank]

        if record.phase == ReintegrationPhase.ROUTED_INTEGRATED:
            # Already fully integrated, no-op
            return False

        was_new = precondition_name not in record.met_preconditions
        record.met_preconditions.add(precondition_name)

        self._log_event(
            "mark_precondition",
            step=step,
            failed_rank=failed_rank,
            replacement_rank=record.replacement_rank,
            phase=record.phase.name,
            precondition=precondition_name,
            details={
                "was_new": was_new,
                "met": sorted(record.met_preconditions),
                "missing": sorted(record.missing_preconditions()),
            },
        )

        # Phase transition: ISOLATED_READY -> REPAIRED_NOT_ROUTED
        # when all preconditions are met
        if (record.phase == ReintegrationPhase.ISOLATED_READY
                and record.all_preconditions_met()):
            record.phase = ReintegrationPhase.REPAIRED_NOT_ROUTED
            self._log_event(
                "phase_transition",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=record.replacement_rank,
                phase=ReintegrationPhase.REPAIRED_NOT_ROUTED.name,
                details={"from": ReintegrationPhase.ISOLATED_READY.name},
            )
            logger.info(
                "MoEGambit barrier: rank %d -> REPAIRED_NOT_ROUTED "
                "(all preconditions met, step=%d)",
                failed_rank, step,
            )

        return was_new

    # -----------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------

    def can_reintegrate(self, failed_rank: int) -> bool:
        """Check if a failed rank is ready for reintegration.

        Returns True only if all preconditions are met (phase is at
        least REPAIRED_NOT_ROUTED) and the rank has not yet been
        fully integrated.
        """
        if failed_rank not in self._records:
            return False
        record = self._records[failed_rank]
        return (
            record.all_preconditions_met()
            and record.phase == ReintegrationPhase.REPAIRED_NOT_ROUTED
        )

    def is_integrated(self, failed_rank: int) -> bool:
        """Check if a failed rank has completed reintegration."""
        if failed_rank not in self._records:
            return False
        return self._records[failed_rank].phase == ReintegrationPhase.ROUTED_INTEGRATED

    def get_record(self, failed_rank: int) -> Optional[ReintegrationRecord]:
        """Get the reintegration record for a failed rank."""
        return self._records.get(failed_rank)

    def get_pending_ranks(self) -> List[int]:
        """Get all failed ranks that are pending reintegration.

        Returns ranks in ISOLATED_READY or REPAIRED_NOT_ROUTED phase.
        """
        return [
            rank for rank, rec in self._records.items()
            if rec.phase != ReintegrationPhase.ROUTED_INTEGRATED
        ]

    def get_ready_ranks(self) -> List[int]:
        """Get all failed ranks that are ready for reintegration.

        Returns ranks in REPAIRED_NOT_ROUTED phase (all preconditions met).
        """
        return [
            rank for rank, rec in self._records.items()
            if rec.phase == ReintegrationPhase.REPAIRED_NOT_ROUTED
        ]

    # -----------------------------------------------------------------
    # Execute reintegration
    # -----------------------------------------------------------------

    def execute_reintegration(
        self,
        failed_rank: int,
        step: int,
        *,
        re_enable_routing_fn: Optional[Callable[[List[int], int], None]] = None,
        clear_degraded_fn: Optional[Callable[[List[int], int], None]] = None,
        verify_topology_fn: Optional[Callable[[int, int], bool]] = None,
        notify_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> bool:
        """Execute the reintegration sequence at a safe point.

        This transitions the rank from REPAIRED_NOT_ROUTED to
        ROUTED_INTEGRATED by executing the following ordered steps:

        1. Verify topology consistency (if callback provided).
        2. Re-enable routing for the expert IDs.
        3. Clear degraded mode markers.
        4. Mark phase as ROUTED_INTEGRATED.
        5. Notify completion (if callback provided).

        Args:
            failed_rank: The rank to reintegrate.
            step: Current training step (must be at safe point).
            re_enable_routing_fn: Callback to re-enable routing.
                Signature: (expert_ids, step) -> None
            clear_degraded_fn: Callback to clear degraded mode.
                Signature: (expert_ids, step) -> None
            verify_topology_fn: Callback to verify topology.
                Signature: (failed_rank, replacement_rank) -> bool
            notify_fn: Callback to notify completion.
                Signature: (failed_rank, replacement_rank, step) -> None

        Returns:
            True if reintegration was executed successfully.

        Raises:
            KeyError: If no record exists for this failed_rank.
            RuntimeError: If preconditions are not met or topology
                verification fails.
        """
        if failed_rank not in self._records:
            raise KeyError(
                f"No reintegration record for rank {failed_rank}."
            )

        record = self._records[failed_rank]

        if not record.all_preconditions_met():
            raise RuntimeError(
                f"Cannot reintegrate rank {failed_rank}: preconditions not met. "
                f"Missing: {sorted(record.missing_preconditions())}"
            )

        if record.phase == ReintegrationPhase.ROUTED_INTEGRATED:
            # Already integrated
            return True

        if record.phase == ReintegrationPhase.ISOLATED_READY:
            # Should have transitioned to REPAIRED_NOT_ROUTED when
            # preconditions were met, but handle gracefully
            record.phase = ReintegrationPhase.REPAIRED_NOT_ROUTED

        # Step 1: Verify topology consistency
        if verify_topology_fn is not None:
            ok = verify_topology_fn(failed_rank, record.replacement_rank)
            if not ok:
                self._log_event(
                    "topology_verification_failed",
                    step=step,
                    failed_rank=failed_rank,
                    replacement_rank=record.replacement_rank,
                    phase=record.phase.name,
                )
                raise RuntimeError(
                    f"Topology verification failed for rank {failed_rank} "
                    f"(replacement={record.replacement_rank})."
                )

        # Step 2: Re-enable routing
        if re_enable_routing_fn is not None:
            re_enable_routing_fn(record.expert_ids, step)
            self._log_event(
                "routing_re_enabled",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=record.replacement_rank,
                phase=record.phase.name,
                details={"expert_ids": list(record.expert_ids)},
            )

        # Step 3: Clear degraded mode
        if clear_degraded_fn is not None:
            clear_degraded_fn(record.expert_ids, step)
            self._log_event(
                "degraded_mode_cleared",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=record.replacement_rank,
                phase=record.phase.name,
                details={"expert_ids": list(record.expert_ids)},
            )

        # Step 4: Mark as ROUTED_INTEGRATED
        record.phase = ReintegrationPhase.ROUTED_INTEGRATED
        record.complete_step = step
        record.completed_at = time.time()

        self._log_event(
            "reintegration_complete",
            step=step,
            failed_rank=failed_rank,
            replacement_rank=record.replacement_rank,
            phase=ReintegrationPhase.ROUTED_INTEGRATED.name,
            details={
                "begin_step": record.begin_step,
                "complete_step": step,
            },
        )

        logger.warning(
            "MoEGambit barrier: rank %d reintegrated at step %d "
            "(replacement=%d, experts=%s)",
            failed_rank, step, record.replacement_rank, record.expert_ids,
        )

        # Step 5: Notify
        if notify_fn is not None:
            notify_fn(failed_rank, record.replacement_rank, step)

        return True

    # -----------------------------------------------------------------
    # Safe-point hook
    # -----------------------------------------------------------------

    def check_safe_point(
        self,
        step: int,
        *,
        re_enable_routing_fn: Optional[Callable[[List[int], int], None]] = None,
        clear_degraded_fn: Optional[Callable[[List[int], int], None]] = None,
        verify_topology_fn: Optional[Callable[[int, int], bool]] = None,
        notify_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[int]:
        """Training loop hook: check and execute all ready reintegrations.

        Called at each iteration boundary (safe point).  For every
        failed rank whose preconditions are all met, executes the
        reintegration sequence.

        Args:
            step: Current training step.
            re_enable_routing_fn: Callback to re-enable routing.
            clear_degraded_fn: Callback to clear degraded mode.
            verify_topology_fn: Callback to verify topology.
            notify_fn: Callback to notify completion.

        Returns:
            List of failed_ranks that were successfully reintegrated.
        """
        ready = self.get_ready_ranks()
        integrated = []

        for failed_rank in ready:
            try:
                ok = self.execute_reintegration(
                    failed_rank,
                    step,
                    re_enable_routing_fn=re_enable_routing_fn,
                    clear_degraded_fn=clear_degraded_fn,
                    verify_topology_fn=verify_topology_fn,
                    notify_fn=notify_fn,
                )
                if ok:
                    integrated.append(failed_rank)
            except RuntimeError as e:
                logger.error(
                    "MoEGambit barrier: reintegration failed for rank %d "
                    "at step %d: %s",
                    failed_rank, step, e,
                )

        return integrated

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    @property
    def records(self) -> Dict[int, ReintegrationRecord]:
        """All reintegration records (read-only copy)."""
        return dict(self._records)

    @property
    def event_log(self) -> List[ReintegrationEvent]:
        """Event log (read-only copy)."""
        return list(self._event_log)

    @property
    def num_pending(self) -> int:
        """Number of ranks pending reintegration."""
        return len(self.get_pending_ranks())

    @property
    def num_integrated(self) -> int:
        """Number of ranks that have completed reintegration."""
        return sum(
            1 for rec in self._records.values()
            if rec.phase == ReintegrationPhase.ROUTED_INTEGRATED
        )

    # -----------------------------------------------------------------
    # Summary / reset
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the barrier state."""
        return {
            "num_records": len(self._records),
            "num_pending": self.num_pending,
            "num_integrated": self.num_integrated,
            "pending_ranks": self.get_pending_ranks(),
            "ready_ranks": self.get_ready_ranks(),
            "num_events": len(self._event_log),
        }

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._records.clear()
        self._event_log.clear()

    def __repr__(self) -> str:
        return (
            f"ReintegrationBarrier("
            f"records={len(self._records)}, "
            f"pending={self.num_pending}, "
            f"integrated={self.num_integrated})"
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _log_event(
        self,
        event_type: str,
        *,
        step: int = -1,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        phase: str = "",
        precondition: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = ReintegrationEvent(
            step=step,
            event_type=event_type,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            phase=phase,
            precondition=precondition,
            timestamp=time.time(),
            details=details or {},
        )
        self._event_log.append(event)


# =====================================================================
# Global singleton
# =====================================================================

_BARRIER: Optional[ReintegrationBarrier] = None


def get_reintegration_barrier() -> ReintegrationBarrier:
    """Get or create the global reintegration barrier singleton."""
    global _BARRIER
    if _BARRIER is None:
        _BARRIER = ReintegrationBarrier()
    return _BARRIER


def clear_reintegration_barrier() -> None:
    """Reset the global barrier (for testing)."""
    global _BARRIER
    if _BARRIER is not None:
        _BARRIER.reset()
    _BARRIER = None
