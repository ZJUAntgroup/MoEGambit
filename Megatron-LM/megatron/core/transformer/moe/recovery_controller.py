# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BSR-MoE Recovery Controller (Step 11).

This module provides a **single orchestration controller** that coordinates
the entire BSR-MoE fault recovery lifecycle.  It wires together all previous
steps (1–10) into a coherent state machine driven by training-loop hooks.

The controller itself does **not** directly modify routers, dispatchers, or
process groups.  Instead, it invokes registered callbacks at the right time
in the right order.  This keeps the controller testable without distributed
init and avoids tight coupling to Megatron internals.

State machine
-------------
::

    HEALTHY_TRAINING
        │
        └── on_hard_rank_failure() ──▶ PENDING_GROUP_REPAIR
                                            │
                                            ▼
                                    on_replacement_assigned() ──▶ WAITING_FOR_REPLACEMENT
                                                                        │
                                                                        ▼
                                                            on_replacement_ready() ──▶ SAFE_POINT_REPAIR
                                                                                            │
                                                                                            ▼
                                                                                before_iteration()
                                                                                (safe-point repair)
                                                                                            │
                                                                                            ▼
                                                                                    REINTEGRATED
                                                                                            │
                                                                                            ▼
                                                                                before_iteration()
                                                                                (finalize)
                                                                                            │
                                                                                            ▼
                                                                                HEALTHY_TRAINING

Safe-point definition
---------------------
A safe point is an **iteration boundary** — the moment between the end of
one training step (after optimizer.step()) and the beginning of the next
(before the next forward pass).  The ``before_iteration()`` hook is the
primary safe-point entry.

Training loop integration
-------------------------
::

    for step in range(num_steps):
        controller.before_iteration(step=step)   # ← safe-point hook
        forward()
        backward()
        optimizer.step()
        controller.after_iteration(step=step)     # ← post-step hook (v1: no-op)

Callback registration
---------------------
The controller delegates all actual work to registered callbacks.  This
allows testing without any Megatron dependencies::

    controller.register_callbacks(
        replacement_announce_fn=...,
        group_rebuild_request_fn=...,
        group_rebuild_execute_fn=...,
        group_rebuild_finish_fn=...,
        topology_refresh_fn=...,
        dense_sync_fn=...,
        expert_restore_fn=...,
    )

Scope (v1)
----------
* ✅ Full lifecycle orchestration: fault → degraded → repair → reintegrate
* ✅ Callback-based — no direct Megatron dependencies
* ✅ Event logging for audit trail
* ✅ Query API for monitoring
* ❌ Multi-fault concurrent recovery (v1 handles one fault at a time)
* ❌ Automatic fault detection (caller must invoke on_rank_quarantined/on_hard_rank_failure)
* ❌ Automatic replacement spawning (caller must invoke on_replacement_assigned)

Hard failure & iteration invalidation (v1.1)
---------------------------------------------
When ``on_hard_rank_failure()`` is called with ``mid_iteration=True``, the
controller records that the current iteration is tainted.  The training
loop should check ``iteration_was_invalidated`` and skip the optimizer
commit + iteration increment for that step.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _ts() -> str:
    """Return current timestamp string for structured logging."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# =====================================================================
# Recovery phase (state machine)
# =====================================================================

class RecoveryPhase(enum.IntEnum):
    """Phases of the BSR-MoE recovery lifecycle."""

    HEALTHY_TRAINING = 0
    """Normal training — no faults, no recovery in progress."""

    DEGRADED_ISOLATION = 1
    """DEPRECATED: Soft-fault quarantine phase.  No longer used in the
    hard-failure-only architecture.  Kept for enum value stability."""

    PENDING_GROUP_REPAIR = 2
    """A hard rank failure has been detected.  The failed rank cannot
    participate in collectives.  Waiting for a replacement rank to be
    assigned."""

    WAITING_FOR_REPLACEMENT = 3
    """A replacement rank has been assigned but is still bootstrapping
    (loading model, pulling dense params, etc.)."""

    SAFE_POINT_REPAIR = 4
    """The replacement rank is ready.  Waiting for the next safe point
    (iteration boundary) to execute the repair sequence."""

    REINTEGRATED = 5
    """Repair has been executed at a safe point.  The replacement rank
    is now integrated.  Will transition to HEALTHY_TRAINING at the next
    before_iteration call."""

    PIPELINE_REBINDING = 6
    """PP group rebuild + P2P rebinding phase (PP>1 only).  Entered after
    SAFE_POINT_REPAIR completes the EP/DP group repair.  Transitions to
    REINTEGRATED once pipeline groups and P2P communicators are repaired."""

    ROLLBACK_PENDING = 7
    """PP>1 mid-iteration failure: waiting for all pipeline stages to
    complete a coordinated rollback before proceeding to group repair."""


# =====================================================================
# Expert recovery state tracker (replaces ExpertHealthManager)
# =====================================================================

class ExpertRecoveryState(enum.IntEnum):
    """Lightweight recovery states for experts on a replacement rank.

    Unlike the deleted ``ExpertState`` (which coupled to health masks and
    routing exclusion), this enum is **observability-only** — it does NOT
    affect routing decisions.  All experts participate in routing at all
    times; the state merely records where they are in the recovery pipeline.
    """

    HEALTHY = 0
    """Fully up-to-date, normal training."""

    RECOVERING = 1
    """Expert is being restored (weights loading from checkpoint).
    The expert is NOT yet participating in training — the replacement
    rank has not finished its recovery sequence."""

    STALE_RUNNABLE = 2
    """Expert weights restored from a (possibly stale) checkpoint.
    The expert participates in training but its weights may lag behind."""

    FULLY_RECOVERED = 3
    """Warmup / catch-up complete.  Ready to be promoted to HEALTHY."""


class ExpertRecoveryTracker:
    """Tracks per-expert recovery state without coupling to routing.

    This replaces the deleted ``ExpertHealthManager`` + ``ExpertHealthMask``
    stack.  It provides:
    - State tracking for preferential routing (read-only query)
    - Logging / observability
    - No health mask, no routing exclusion

    The tracker is embedded in ``RecoveryController`` rather than being
    a per-layer singleton, because in the new architecture recovery is
    coordinated at the controller level (one fault at a time).
    """

    def __init__(self, num_experts: int = 0) -> None:
        self._num_experts = num_experts
        self._states: Dict[int, ExpertRecoveryState] = {}
        # Only experts that are NOT HEALTHY are tracked.
        # Missing key → HEALTHY.

    def set_num_experts(self, num_experts: int) -> None:
        self._num_experts = num_experts

    # -- Mutation --

    def mark_recovering(self, expert_ids: List[int], step: int = 0) -> None:
        """Mark experts as RECOVERING (fault detected, weights being loaded)."""
        for eid in expert_ids:
            self._states[eid] = ExpertRecoveryState.RECOVERING
        if expert_ids:
            logger.info(
                "ExpertRecoveryTracker: experts %s → RECOVERING (step=%d)",
                expert_ids, step,
            )

    def mark_stale_runnable(self, expert_ids: List[int], step: int = 0) -> None:
        """Mark experts as STALE_RUNNABLE (checkpoint loaded, participating)."""
        for eid in expert_ids:
            self._states[eid] = ExpertRecoveryState.STALE_RUNNABLE
        if expert_ids:
            logger.info(
                "ExpertRecoveryTracker: experts %s → STALE_RUNNABLE (step=%d)",
                expert_ids, step,
            )

    def mark_fully_recovered(self, expert_ids: List[int], step: int = 0) -> None:
        """Mark experts as FULLY_RECOVERED (warmup done)."""
        for eid in expert_ids:
            self._states[eid] = ExpertRecoveryState.FULLY_RECOVERED
        if expert_ids:
            logger.info(
                "ExpertRecoveryTracker: experts %s → FULLY_RECOVERED (step=%d)",
                expert_ids, step,
            )

    def mark_healthy(self, expert_ids: List[int], step: int = 0) -> None:
        """Promote experts back to HEALTHY (remove from tracker)."""
        for eid in expert_ids:
            self._states.pop(eid, None)
        if expert_ids:
            logger.info(
                "ExpertRecoveryTracker: experts %s → HEALTHY (step=%d)",
                expert_ids, step,
            )

    # -- Query --

    def get_state(self, expert_id: int) -> ExpertRecoveryState:
        return self._states.get(expert_id, ExpertRecoveryState.HEALTHY)

    def get_stale_experts(self) -> List[int]:
        """Return expert IDs in STALE_RUNNABLE state."""
        return sorted(
            eid for eid, s in self._states.items()
            if s == ExpertRecoveryState.STALE_RUNNABLE
        )

    def get_recovering_experts(self) -> List[int]:
        """Return expert IDs in RECOVERING state."""
        return sorted(
            eid for eid, s in self._states.items()
            if s == ExpertRecoveryState.RECOVERING
        )

    def all_healthy(self) -> bool:
        return len(self._states) == 0

    def summary(self) -> Dict[str, Any]:
        by_state: Dict[str, List[int]] = {}
        for eid, s in self._states.items():
            by_state.setdefault(s.name, []).append(eid)
        return {k: sorted(v) for k, v in by_state.items()}

    def reset(self) -> None:
        self._states.clear()


# Valid phase transitions
_VALID_TRANSITIONS = {
    RecoveryPhase.HEALTHY_TRAINING: {
        RecoveryPhase.PENDING_GROUP_REPAIR,
        RecoveryPhase.ROLLBACK_PENDING,
    },
    RecoveryPhase.ROLLBACK_PENDING: {
        RecoveryPhase.PENDING_GROUP_REPAIR,
    },
    RecoveryPhase.PENDING_GROUP_REPAIR: {
        RecoveryPhase.WAITING_FOR_REPLACEMENT,
    },
    RecoveryPhase.WAITING_FOR_REPLACEMENT: {
        RecoveryPhase.SAFE_POINT_REPAIR,
    },
    RecoveryPhase.SAFE_POINT_REPAIR: {
        RecoveryPhase.REINTEGRATED,
        RecoveryPhase.PIPELINE_REBINDING,
    },
    RecoveryPhase.PIPELINE_REBINDING: {
        RecoveryPhase.REINTEGRATED,
    },
    RecoveryPhase.REINTEGRATED: {
        RecoveryPhase.HEALTHY_TRAINING,
    },
}


# =====================================================================
# Event log
# =====================================================================

@dataclass
class RecoveryEvent:
    """A single event in the recovery lifecycle."""

    event_type: str = ""
    """Type of event (e.g. 'rank_quarantined', 'replacement_ready')."""

    phase_from: str = ""
    """Phase before the event."""

    phase_to: str = ""
    """Phase after the event."""

    step: int = -1
    """Training step when the event occurred."""

    timestamp: float = 0.0
    """Wall-clock timestamp."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional event-specific details."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "phase_from": self.phase_from,
            "phase_to": self.phase_to,
            "step": self.step,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


# =====================================================================
# Fault record
# =====================================================================

@dataclass
class FaultRecord:
    """Tracks a single fault and its recovery progress."""

    failed_rank: int = -1
    """The rank that failed."""

    replacement_rank: int = -1
    """The replacement rank (set when assigned)."""

    fault_type: str = ""
    """'soft' (quarantined but alive) or 'hard' (cannot participate)."""

    reason: str = ""
    """Human-readable fault reason."""

    fault_step: int = -1
    """Training step when the fault was detected."""

    replacement_assigned_step: int = -1
    """Training step when replacement was assigned."""

    replacement_ready_step: int = -1
    """Training step when replacement reported ready."""

    repair_step: int = -1
    """Training step when safe-point repair was executed."""

    reintegration_step: int = -1
    """Training step when reintegration was finalized."""

    expert_ids: List[int] = field(default_factory=list)
    """Global expert IDs affected by this fault."""

    ep_group_ranks: List[int] = field(default_factory=list)
    """EP group ranks (for group rebuild)."""

    dp_group_ranks: List[int] = field(default_factory=list)
    """DP group ranks (for group rebuild)."""

    mid_iteration: bool = False
    """True if the fault was detected inside forward/backward/optimizer."""

    pp_group_ranks: List[int] = field(default_factory=list)
    """PP group ranks (for pipeline stage repair)."""

    failed_stage: int = -1
    """The pipeline stage that failed (-1 if not a PP failure)."""

    pipeline_repaired: bool = False
    """True if PP group rebuild + P2P rebinding has been completed."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "fault_type": self.fault_type,
            "reason": self.reason,
            "fault_step": self.fault_step,
            "replacement_assigned_step": self.replacement_assigned_step,
            "replacement_ready_step": self.replacement_ready_step,
            "repair_step": self.repair_step,
            "reintegration_step": self.reintegration_step,
            "expert_ids": list(self.expert_ids),
            "ep_group_ranks": list(self.ep_group_ranks),
            "dp_group_ranks": list(self.dp_group_ranks),
            "mid_iteration": self.mid_iteration,
            "pp_group_ranks": list(self.pp_group_ranks),
            "failed_stage": self.failed_stage,
            "pipeline_repaired": self.pipeline_repaired,
        }


# =====================================================================
# Recovery Controller
# =====================================================================

class RecoveryController:
    """Orchestrates the BSR-MoE fault recovery lifecycle.

    The controller maintains a state machine and invokes registered
    callbacks at the appropriate times.  It does NOT directly modify
    any Megatron data structures.

    Usage::

        ctrl = RecoveryController()
        ctrl.register_callbacks(
            ...
        )

        # In training loop:
        for step in range(num_steps):
            ctrl.before_iteration(step=step)
            train_step()
            ctrl.after_iteration(step=step)

        # On fault detection:
        ctrl.on_hard_rank_failure(failed_rank=1, step=100)
    """

    def __init__(self) -> None:
        self._phase = RecoveryPhase.HEALTHY_TRAINING
        self._active_faults: Dict[int, FaultRecord] = {}  # failed_rank → record
        self._completed_recoveries: List[FaultRecord] = []
        self._event_log: List[RecoveryEvent] = []

        # Iteration invalidation tracking
        self._iteration_invalidated: bool = False
        self._invalidated_step: int = -1

        # Async recovery tracking (Phase 15)
        self._async_recovery_pending: bool = False
        self._async_expert_request_ids: List[str] = []
        self._async_recovery_expert_ids: List[int] = []
        self._async_recovery_failed_rank: int = -1
        self._async_recovery_replacement_rank: int = -1

        # Callbacks for async recovery (set by bsr_integration)
        self._async_expert_restore_fn: Optional[Callable] = None
        self._poll_async_recovery_fn: Optional[Callable] = None

        # Callbacks (all optional)
        self._health_mark_stale_runnable_fn: Optional[Callable] = None
        self._health_mark_healthy_fn: Optional[Callable] = None
        self._replacement_announce_fn: Optional[Callable] = None
        self._replacement_query_fn: Optional[Callable] = None
        self._replacement_integrate_fn: Optional[Callable] = None
        self._group_rebuild_request_fn: Optional[Callable] = None
        self._group_rebuild_execute_fn: Optional[Callable] = None
        self._group_rebuild_finish_fn: Optional[Callable] = None
        self._topology_refresh_fn: Optional[Callable] = None
        self._dense_sync_fn: Optional[Callable] = None
        self._expert_restore_fn: Optional[Callable] = None

        # PP>1 recovery callbacks
        self._pipeline_stage_repair_fn: Optional[Callable] = None
        self._pipeline_rollback_fn: Optional[Callable] = None
        self._microbatch_invalidation_fn: Optional[Callable] = None

        # Gap-aware recovery policy
        self._gap_aware_policy_manager = None  # Set via set_gap_aware_policy_manager
        self._checkpoint_restart_fn: Optional[Callable] = None

        # Dual-path recovery callbacks (hard failure unified preconditions)
        self._optimizer_commit_block_fn: Optional[Callable] = None
        self._enter_waiting_fn: Optional[Callable] = None

        # Reintegration barrier (optional, set via set_reintegration_barrier)
        self._reintegration_barrier = None

        # Unified post-recovery convergence callback (Step 8)
        self._post_recovery_convergence_fn: Optional[Callable] = None

        # Stage-safe recovery protocol callback (PP>1 closed-loop)
        self._stage_safe_recovery_fn: Optional[Callable] = None

        # Restart-in-place test mode
        self._invalidate_tensor_fn: Optional[Callable] = None
        self._restart_in_place_mode: bool = False

        # Force checkpoint restart callback (e.g., when checkpoint format
        # does not support selective per-rank loading)
        self._force_checkpoint_restart_fn: Optional[Callable] = None

        # Recovery path tracking
        self._last_recovery_path: str = ""  # "CHECKPOINT_RESTART" or "HYBRID_RECOVERY"

        # Expert recovery state tracker (replaces ExpertHealthManager)
        self._expert_tracker = ExpertRecoveryTracker()

        # PP>1 tracking
        self._inflight_microbatches_invalidated: bool = False
        self._pipeline_rollback_completed: bool = False

    # -----------------------------------------------------------------
    # Reintegration barrier
    # -----------------------------------------------------------------

    def set_reintegration_barrier(self, barrier) -> None:
        """Set the reintegration barrier for controlled reintegration.

        Args:
            barrier: A ``ReintegrationBarrier`` instance.
        """
        self._reintegration_barrier = barrier

    @property
    def reintegration_barrier(self):
        """Get the reintegration barrier (or None)."""
        return self._reintegration_barrier

    # -----------------------------------------------------------------
    # Gap-aware recovery policy
    # -----------------------------------------------------------------

    def set_gap_aware_policy_manager(self, manager) -> None:
        """Set the gap-aware recovery policy manager.

        Args:
            manager: A ``GapAwareRecoveryPolicyManager`` instance.
        """
        self._gap_aware_policy_manager = manager

    @property
    def gap_aware_policy_manager(self):
        """Get the gap-aware recovery policy manager (or None)."""
        return self._gap_aware_policy_manager

    # -----------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------

    def register_callbacks(
        self,
        *,
        health_mark_stale_runnable_fn: Optional[Callable] = None,
        health_mark_healthy_fn: Optional[Callable] = None,
        replacement_announce_fn: Optional[Callable] = None,
        replacement_query_fn: Optional[Callable] = None,
        replacement_integrate_fn: Optional[Callable] = None,
        group_rebuild_request_fn: Optional[Callable] = None,
        group_rebuild_execute_fn: Optional[Callable] = None,
        group_rebuild_finish_fn: Optional[Callable] = None,
        topology_refresh_fn: Optional[Callable] = None,
        dense_sync_fn: Optional[Callable] = None,
        expert_restore_fn: Optional[Callable] = None,
        pipeline_stage_repair_fn: Optional[Callable] = None,
        pipeline_rollback_fn: Optional[Callable] = None,
        microbatch_invalidation_fn: Optional[Callable] = None,
        async_expert_restore_fn: Optional[Callable] = None,
        poll_async_recovery_fn: Optional[Callable] = None,
        checkpoint_restart_fn: Optional[Callable] = None,
        optimizer_commit_block_fn: Optional[Callable] = None,
        enter_waiting_fn: Optional[Callable] = None,
        post_recovery_convergence_fn: Optional[Callable] = None,
        stage_safe_recovery_fn: Optional[Callable] = None,
        invalidate_tensor_fn: Optional[Callable] = None,
        force_checkpoint_restart_fn: Optional[Callable] = None,
    ) -> None:
        """Register callback functions for each recovery action.

        All callbacks accept keyword arguments.  The controller passes
        relevant context (failed_rank, replacement_rank, step, etc.)
        as keyword arguments.
        """
        if health_mark_stale_runnable_fn is not None:
            self._health_mark_stale_runnable_fn = health_mark_stale_runnable_fn
        if health_mark_healthy_fn is not None:
            self._health_mark_healthy_fn = health_mark_healthy_fn
        if replacement_announce_fn is not None:
            self._replacement_announce_fn = replacement_announce_fn
        if replacement_query_fn is not None:
            self._replacement_query_fn = replacement_query_fn
        if replacement_integrate_fn is not None:
            self._replacement_integrate_fn = replacement_integrate_fn
        if group_rebuild_request_fn is not None:
            self._group_rebuild_request_fn = group_rebuild_request_fn
        if group_rebuild_execute_fn is not None:
            self._group_rebuild_execute_fn = group_rebuild_execute_fn
        if group_rebuild_finish_fn is not None:
            self._group_rebuild_finish_fn = group_rebuild_finish_fn
        if topology_refresh_fn is not None:
            self._topology_refresh_fn = topology_refresh_fn
        if dense_sync_fn is not None:
            self._dense_sync_fn = dense_sync_fn
        if expert_restore_fn is not None:
            self._expert_restore_fn = expert_restore_fn
        if pipeline_stage_repair_fn is not None:
            self._pipeline_stage_repair_fn = pipeline_stage_repair_fn
        if pipeline_rollback_fn is not None:
            self._pipeline_rollback_fn = pipeline_rollback_fn
        if microbatch_invalidation_fn is not None:
            self._microbatch_invalidation_fn = microbatch_invalidation_fn
        if async_expert_restore_fn is not None:
            self._async_expert_restore_fn = async_expert_restore_fn
        if poll_async_recovery_fn is not None:
            self._poll_async_recovery_fn = poll_async_recovery_fn
        if checkpoint_restart_fn is not None:
            self._checkpoint_restart_fn = checkpoint_restart_fn
        if optimizer_commit_block_fn is not None:
            self._optimizer_commit_block_fn = optimizer_commit_block_fn
        if enter_waiting_fn is not None:
            self._enter_waiting_fn = enter_waiting_fn
        if post_recovery_convergence_fn is not None:
            self._post_recovery_convergence_fn = post_recovery_convergence_fn
        if stage_safe_recovery_fn is not None:
            self._stage_safe_recovery_fn = stage_safe_recovery_fn
        if invalidate_tensor_fn is not None:
            self._invalidate_tensor_fn = invalidate_tensor_fn
        if force_checkpoint_restart_fn is not None:
            self._force_checkpoint_restart_fn = force_checkpoint_restart_fn

    # -----------------------------------------------------------------
    # Phase transitions
    # -----------------------------------------------------------------

    def _transition_to(
        self,
        new_phase: RecoveryPhase,
        event_type: str = "",
        step: int = -1,
        **details,
    ) -> None:
        """Transition to a new phase with validation and logging."""
        old_phase = self._phase
        valid = _VALID_TRANSITIONS.get(old_phase, set())
        if new_phase not in valid:
            raise ValueError(
                f"Invalid phase transition: {old_phase.name} → {new_phase.name}. "
                f"Valid targets: {[p.name for p in valid]}"
            )

        self._phase = new_phase
        event = RecoveryEvent(
            event_type=event_type or f"transition_{old_phase.name}_to_{new_phase.name}",
            phase_from=old_phase.name,
            phase_to=new_phase.name,
            step=step,
            timestamp=time.time(),
            details=details,
        )
        self._event_log.append(event)

        logger.warning(
            "[%s] BSR-MoE controller: %s → %s (event=%s, step=%d)",
            _ts(), old_phase.name, new_phase.name, event.event_type, step,
        )

    # -----------------------------------------------------------------
    # Fault injection API
    # -----------------------------------------------------------------

    def on_hard_rank_failure(
        self,
        failed_rank: int,
        *,
        reason: str = "",
        step: int = -1,
        expert_ids: Optional[List[int]] = None,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
        mid_iteration: bool = False,
        restart_in_place: bool = False,
    ) -> None:
        """Handle a hard fault: rank cannot participate in collectives.

        This transitions to PENDING_GROUP_REPAIR.  A replacement rank
        must be assigned before recovery can proceed.

        Args:
            failed_rank: The global rank that failed.
            reason: Human-readable fault reason.
            step: Training step when the fault was detected.
            expert_ids: Global expert IDs affected by this fault.
            ep_group_ranks: EP group ranks (for group rebuild).
            dp_group_ranks: DP group ranks (for group rebuild).
            mid_iteration: If True, the failure occurred inside
                forward/backward/optimizer.  The current iteration is
                marked invalid and the optimizer commit must be skipped.
            restart_in_place: If True, the failed rank restarts in-place
                (replacement_rank == failed_rank).  This triggers a fast
                path that skips group rebuild and topology refresh, and
                automatically progresses through replacement assignment
                and readiness.  Used for testing restart-in-place recovery.

        Idempotent: if the rank is already tracked as a hard fault, this is a no-op.
        """
        record = self._active_faults.get(failed_rank)
        if record is not None and record.fault_type == "hard":
            # Already tracked as hard fault — no-op
            logger.debug(
                "BSR-MoE controller: rank %d already tracked as hard fault, "
                "skipping duplicate (step=%d)",
                failed_rank, step,
            )
            return

        if record is None:
            # New fault
            record = FaultRecord(
                failed_rank=failed_rank,
                fault_type="hard",
                reason=reason,
                fault_step=step,
                expert_ids=list(expert_ids or []),
                ep_group_ranks=list(ep_group_ranks or []),
                dp_group_ranks=list(dp_group_ranks or []),
                mid_iteration=mid_iteration,
            )
            self._active_faults[failed_rank] = record
        else:
            # Already tracked — update fields
            record.fault_type = "hard"
            record.mid_iteration = record.mid_iteration or mid_iteration
            if reason:
                record.reason = reason
            if expert_ids:
                record.expert_ids = list(expert_ids)
            if ep_group_ranks:
                record.ep_group_ranks = list(ep_group_ranks)
            if dp_group_ranks:
                record.dp_group_ranks = list(dp_group_ranks)

        logger.warning(
            "[%s] BSR-MoE controller: rank %d HARD FAILURE (step=%d, reason=%s, "
            "experts=%s, mid_iteration=%s)",
            _ts(), failed_rank, step, reason, expert_ids, mid_iteration,
        )

        # Track experts as recovering
        if expert_ids:
            self._expert_tracker.mark_recovering(expert_ids, step=step)

        # Mark current iteration as invalid if failure is mid-iteration
        if mid_iteration:
            self._iteration_invalidated = True
            self._invalidated_step = step
            logger.warning(
                "BSR-MoE controller: iteration %d INVALIDATED due to "
                "hard failure of rank %d (mid_iteration=True)",
                step, failed_rank,
            )

        # Block optimizer commit for this iteration
        if self._optimizer_commit_block_fn is not None:
            try:
                self._optimizer_commit_block_fn(step=step, failed_rank=failed_rank)
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: optimizer_commit_block_fn failed: %s", e,
                )

        # Notify external system that we are entering waiting-for-replacement
        if self._enter_waiting_fn is not None:
            try:
                self._enter_waiting_fn(
                    failed_rank=failed_rank,
                    step=step,
                    expert_ids=expert_ids,
                )
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: enter_waiting_fn failed: %s", e,
                )

        # Transition
        if self._phase == RecoveryPhase.HEALTHY_TRAINING:
            self._transition_to(
                RecoveryPhase.PENDING_GROUP_REPAIR,
                event_type="hard_rank_failure",
                step=step,
                failed_rank=failed_rank,
            )

        # Restart-in-place fast path: auto-assign replacement = failed_rank
        # and immediately progress to SAFE_POINT_REPAIR
        if restart_in_place:
            self._restart_in_place_mode = True

            # Call invalidate_tensor_fn if registered
            if self._invalidate_tensor_fn is not None:
                try:
                    self._invalidate_tensor_fn(
                        failed_rank=failed_rank,
                        step=step,
                    )
                except Exception as e:
                    logger.error(
                        "BSR-MoE controller: invalidate_tensor_fn failed: %s", e,
                    )

            # Auto-assign replacement_rank = failed_rank
            self.on_replacement_assigned(
                failed_rank=failed_rank,
                replacement_rank=failed_rank,
                step=step,
            )
            # Auto-mark replacement ready
            self.on_replacement_ready(
                failed_rank=failed_rank,
                step=step,
            )

            logger.warning(
                "BSR-MoE controller: restart-in-place fast path for rank %d "
                "(replacement=self, step=%d)",
                failed_rank, step,
            )

    # -----------------------------------------------------------------
    # PP>1 fault injection API
    # -----------------------------------------------------------------

    def on_pipeline_stage_failure(
        self,
        failed_stage: int,
        failed_rank: int,
        *,
        step: int = -1,
        pp_group_ranks: Optional[List[int]] = None,
        reason: str = "",
        expert_ids: Optional[List[int]] = None,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
        mid_iteration: bool = True,
    ) -> None:
        """Handle a pipeline stage failure (PP>1).

        This is the PP-aware entry point for hard failures.  It records
        the pipeline-specific information (failed_stage, pp_group_ranks)
        and transitions to ROLLBACK_PENDING if mid-iteration, or directly
        to PENDING_GROUP_REPAIR otherwise.

        Args:
            failed_stage: The pipeline stage index that failed.
            failed_rank: The global rank that failed.
            step: Training step when the fault was detected.
            pp_group_ranks: Ranks in the PP group.
            reason: Human-readable fault reason.
            expert_ids: Global expert IDs affected.
            ep_group_ranks: EP group ranks.
            dp_group_ranks: DP group ranks.
            mid_iteration: If True, failure occurred mid-iteration.
        """
        # Delegate to on_hard_rank_failure for the common path, but
        # override the transition target for PP>1 mid-iteration failures.
        record = self._active_faults.get(failed_rank)
        if record is not None and record.fault_type == "hard":
            # Already tracked — update PP fields only
            record.failed_stage = failed_stage
            record.pp_group_ranks = list(pp_group_ranks or [])
            return

        # Create or escalate via on_hard_rank_failure (but suppress its
        # transition — we handle it ourselves).
        saved_phase = self._phase

        # Temporarily block transition by setting phase to a non-source state
        # Actually, just call the common setup without transition:
        if record is None:
            record = FaultRecord(
                failed_rank=failed_rank,
                fault_type="hard",
                reason=reason or f"pipeline_stage_{failed_stage}_failure",
                fault_step=step,
                expert_ids=list(expert_ids or []),
                ep_group_ranks=list(ep_group_ranks or []),
                dp_group_ranks=list(dp_group_ranks or []),
                mid_iteration=mid_iteration,
                pp_group_ranks=list(pp_group_ranks or []),
                failed_stage=failed_stage,
            )
            self._active_faults[failed_rank] = record
        else:
            record.fault_type = "hard"
            record.mid_iteration = record.mid_iteration or mid_iteration
            record.failed_stage = failed_stage
            record.pp_group_ranks = list(pp_group_ranks or [])
            if reason:
                record.reason = reason

        # Track experts as recovering
        if expert_ids:
            self._expert_tracker.mark_recovering(expert_ids, step=step)

        # Mark iteration invalid
        if mid_iteration:
            self._iteration_invalidated = True
            self._invalidated_step = step

        # Block optimizer commit for this iteration
        if self._optimizer_commit_block_fn is not None:
            try:
                self._optimizer_commit_block_fn(step=step, failed_rank=failed_rank)
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: optimizer_commit_block_fn failed: %s", e,
                )

        # Notify external system that we are entering waiting-for-replacement
        if self._enter_waiting_fn is not None:
            try:
                self._enter_waiting_fn(
                    failed_rank=failed_rank,
                    step=step,
                    expert_ids=expert_ids,
                )
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: enter_waiting_fn failed: %s", e,
                )

        # Invalidate in-flight microbatches
        if mid_iteration:
            self.invalidate_inflight_microbatches(
                step=step,
                pp_size=len(pp_group_ranks) if pp_group_ranks else 1,
            )

        # Transition: mid-iteration PP failure → ROLLBACK_PENDING
        if self._phase == RecoveryPhase.HEALTHY_TRAINING:
            if mid_iteration and (pp_group_ranks and len(pp_group_ranks) > 1):
                self._transition_to(
                    RecoveryPhase.ROLLBACK_PENDING,
                    event_type="pipeline_stage_failure",
                    step=step,
                    failed_rank=failed_rank,
                    failed_stage=failed_stage,
                )
                # Invoke pipeline rollback callback
                if self._pipeline_rollback_fn is not None:
                    try:
                        self._pipeline_rollback_fn(
                            failed_rank=failed_rank,
                            failed_stage=failed_stage,
                            step=step,
                            pp_group_ranks=pp_group_ranks,
                        )
                    except Exception as e:
                        logger.error(
                            "BSR-MoE controller: pipeline_rollback_fn "
                            "failed: %s", e,
                        )
                self._pipeline_rollback_completed = True
                # After rollback, transition to PENDING_GROUP_REPAIR
                self._transition_to(
                    RecoveryPhase.PENDING_GROUP_REPAIR,
                    event_type="pipeline_rollback_completed",
                    step=step,
                    failed_rank=failed_rank,
                )
            else:
                self._transition_to(
                    RecoveryPhase.PENDING_GROUP_REPAIR,
                    event_type="pipeline_stage_failure",
                    step=step,
                    failed_rank=failed_rank,
                    failed_stage=failed_stage,
                )

        logger.warning(
            "BSR-MoE controller: pipeline stage failure — "
            "stage=%d, rank=%d, step=%d, pp_ranks=%s",
            failed_stage, failed_rank, step, pp_group_ranks,
        )

    def invalidate_inflight_microbatches(
        self,
        step: int = -1,
        pp_size: int = 1,
    ) -> None:
        """Mark all in-flight microbatches as invalid.

        Called when a PP stage failure is detected mid-iteration.
        All microbatches currently in the pipeline are tainted and
        must be discarded.

        Args:
            step: Training step.
            pp_size: Pipeline parallel size.
        """
        self._inflight_microbatches_invalidated = True

        if self._microbatch_invalidation_fn is not None:
            try:
                self._microbatch_invalidation_fn(
                    step=step, pp_size=pp_size,
                )
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: microbatch_invalidation_fn "
                    "failed: %s", e,
                )

        logger.warning(
            "BSR-MoE controller: in-flight microbatches invalidated "
            "(step=%d, pp_size=%d)",
            step, pp_size,
        )

    # -----------------------------------------------------------------
    # PP>1 query helpers
    # -----------------------------------------------------------------

    def maybe_repair_pipeline_groups(self, step: int = -1) -> bool:
        """Check if PP group repair is needed and ready.

        Returns True if the controller is in PIPELINE_REBINDING phase.
        """
        return self._phase == RecoveryPhase.PIPELINE_REBINDING

    def maybe_rebind_p2p(self, step: int = -1) -> bool:
        """Check if P2P rebinding is needed.

        Returns True if the controller is in PIPELINE_REBINDING phase
        (P2P rebinding is part of the pipeline repair sequence).
        """
        return self._phase == RecoveryPhase.PIPELINE_REBINDING

    @property
    def inflight_microbatches_invalidated(self) -> bool:
        """True if in-flight microbatches have been invalidated."""
        return self._inflight_microbatches_invalidated

    @property
    def pipeline_rollback_completed(self) -> bool:
        """True if pipeline rollback has been completed."""
        return self._pipeline_rollback_completed

    # -----------------------------------------------------------------
    # Replacement lifecycle
    # -----------------------------------------------------------------

    def on_replacement_assigned(
        self,
        failed_rank: int,
        replacement_rank: int,
        step: int = -1,
    ) -> None:
        """A replacement rank has been assigned for the failed rank.

        Transitions from PENDING_GROUP_REPAIR → WAITING_FOR_REPLACEMENT.
        """
        record = self._active_faults.get(failed_rank)
        if record is None:
            raise ValueError(
                f"Cannot assign replacement for rank {failed_rank}: "
                f"no active fault record."
            )

        record.replacement_rank = replacement_rank
        record.replacement_assigned_step = step

        logger.warning(
            "[%s] BSR-MoE controller: replacement assigned — "
            "failed_rank=%d, replacement_rank=%d (step=%d)",
            _ts(), failed_rank, replacement_rank, step,
        )

        # Invoke replacement announce callback
        if self._replacement_announce_fn is not None:
            self._replacement_announce_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )

        if self._phase == RecoveryPhase.PENDING_GROUP_REPAIR:
            self._transition_to(
                RecoveryPhase.WAITING_FOR_REPLACEMENT,
                event_type="replacement_assigned",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
            )

    def on_replacement_ready(
        self,
        failed_rank: int,
        step: int = -1,
    ) -> None:
        """The replacement rank has finished bootstrapping and is ready.

        Transitions from WAITING_FOR_REPLACEMENT → SAFE_POINT_REPAIR.
        """
        record = self._active_faults.get(failed_rank)
        if record is None:
            raise ValueError(
                f"Cannot mark replacement ready for rank {failed_rank}: "
                f"no active fault record."
            )

        record.replacement_ready_step = step

        logger.warning(
            "[%s] BSR-MoE controller: replacement ready — "
            "failed_rank=%d (step=%d)",
            _ts(), failed_rank, step,
        )

        if self._phase == RecoveryPhase.WAITING_FOR_REPLACEMENT:
            self._transition_to(
                RecoveryPhase.SAFE_POINT_REPAIR,
                event_type="replacement_ready",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=record.replacement_rank,
            )

    # -----------------------------------------------------------------
    # Training loop hooks
    # -----------------------------------------------------------------

    def before_iteration(self, step: int = -1) -> bool:
        """Safe-point hook: called at the beginning of each iteration.

        This is the primary safe-point entry.  If the controller is in
        SAFE_POINT_REPAIR phase, it executes the full repair sequence.

        If the controller is in REINTEGRATED phase, it finalizes the
        recovery and transitions to HEALTHY_TRAINING.

        Also clears any iteration-invalidation flag from the previous step.

        Returns:
            True if a repair was executed at this safe point.
        """
        # Clear iteration invalidation from previous step
        self.clear_iteration_invalidation()
        # Finalize reintegration from previous iteration
        if self._phase == RecoveryPhase.REINTEGRATED:
            self._finalize_reintegration(step=step)
            return False

        # Execute repair at safe point
        if self._phase == RecoveryPhase.SAFE_POINT_REPAIR:
            return self._execute_safe_point_repair(step=step)

        return False

    def after_iteration(self, step: int = -1) -> bool:
        """Post-step hook: called after each iteration.

        Polls async recovery operations (expert weight loading, deferred
        optimizer state loading) and finalizes them when complete.

        Returns:
            True if any async recovery action was completed.
        """
        if not self._async_recovery_pending:
            return False

        # Delegate to the registered poll callback
        if self._poll_async_recovery_fn is not None:
            try:
                completed = self._poll_async_recovery_fn(step=step)
                if completed:
                    self._async_recovery_pending = False
                    self._async_expert_request_ids.clear()
                    logger.info(
                        "BSR-MoE controller: async recovery completed at "
                        "step %d (failed_rank=%d, replacement=%d)",
                        step, self._async_recovery_failed_rank,
                        self._async_recovery_replacement_rank,
                    )
                    return True
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: poll_async_recovery_fn failed: %s", e,
                )

        return False

    # -----------------------------------------------------------------
    # maybe_* helpers
    # -----------------------------------------------------------------

    def maybe_enter_degraded_mode(self) -> bool:
        """Check if the system is in a non-healthy recovery phase."""
        return self._phase in (
            RecoveryPhase.PENDING_GROUP_REPAIR,
            RecoveryPhase.WAITING_FOR_REPLACEMENT,
            RecoveryPhase.SAFE_POINT_REPAIR,
            RecoveryPhase.PIPELINE_REBINDING,
            RecoveryPhase.ROLLBACK_PENDING,
        )

    def maybe_repair_groups(self) -> bool:
        """Check if group repair is pending and ready."""
        return self._phase == RecoveryPhase.SAFE_POINT_REPAIR

    def maybe_reintegrate(self) -> bool:
        """Check if reintegration is pending."""
        return self._phase == RecoveryPhase.REINTEGRATED

    # -----------------------------------------------------------------
    # Internal: safe-point repair sequence
    # -----------------------------------------------------------------

    def _execute_safe_point_repair(self, step: int = -1) -> bool:
        """Execute the full repair sequence at a safe point.

        The repair is structured as a three-phase pipeline so that both
        the checkpoint-restart and hybrid-recovery paths share the same
        infrastructure repair steps:

        Phase A — Shared pre-repair (always executed):
            1. Integrate replacement rank
            2. Request group rebuild
            3. Execute group rebuild
            4. Finish group rebuild
            5. Refresh dispatch topology

        Phase B — Path-specific parameter recovery (branched):
            Gap-aware policy evaluation → select path
            B1. CHECKPOINT_RESTART path:
                - Load all params + optimizer state from checkpoint
            B2. HYBRID_RECOVERY path (default):
                - Dense/shared/router params from healthy DP peer
                - MoE expert weights from distributed checkpoint

        Phase C — Shared post-repair (always executed):
            6. PP group repair + P2P rebinding (PP>1 only)
            7. Reintegration barrier setup
            8. State transition → REINTEGRATED
        """
        # Find the active fault record that is ready for repair
        ready_record = None
        for record in self._active_faults.values():
            if record.replacement_ready_step >= 0:
                ready_record = record
                break

        if ready_record is None:
            logger.warning(
                "BSR-MoE controller: SAFE_POINT_REPAIR but no ready record"
            )
            return False

        failed_rank = ready_record.failed_rank
        replacement_rank = ready_record.replacement_rank

        repair_start = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  START safe-point repair at step %d "
            "(failed=%d, replacement=%d)",
            _ts(), step, failed_rank, replacement_rank,
        )

        # =============================================================
        # Phase A: Shared pre-repair (infrastructure)
        # =============================================================
        phase_a_start = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  [Phase A] Starting infrastructure repair...",
            _ts(),
        )

        # A1. Integrate replacement rank
        t0 = time.time()
        if self._replacement_integrate_fn is not None:
            self._replacement_integrate_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: step1 replacement_integrate elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )

        # Restart-in-place: skip group rebuild and topology refresh
        # because the rank hasn't changed — no process group repair needed.
        _skip_infra = (self._restart_in_place_mode
                       and replacement_rank == failed_rank)

        # A2. Request group rebuild
        t0 = time.time()
        if self._group_rebuild_request_fn is not None and not _skip_infra:
            self._group_rebuild_request_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                ep_group_ranks=ready_record.ep_group_ranks,
                dp_group_ranks=ready_record.dp_group_ranks,
            )
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: step2 group_rebuild_request elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )

        # A3. Execute group rebuild
        t0 = time.time()
        if self._group_rebuild_execute_fn is not None and not _skip_infra:
            try:
                self._group_rebuild_execute_fn(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                )
            except Exception as e:
                t1 = time.time()
                logger.error(
                    "[%s] BSR-MoE controller: step3 group_rebuild_execute "
                    "FAILED (elapsed=%.3fs, step=%d): %s",
                    _ts(), t1 - t0, step, e,
                )
                ready_record.repair_step = -1
                logger.error(
                    "[%s] BSR-MoE controller: safe-point repair ABORTED — "
                    "group rebuild failed, training cannot continue safely. "
                    "(step=%d, failed=%d, replacement=%d)",
                    _ts(), step, failed_rank, replacement_rank,
                )
                return False
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: step3 group_rebuild_execute elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )

        # A4. Finish group rebuild
        t0 = time.time()
        if self._group_rebuild_finish_fn is not None and not _skip_infra:
            self._group_rebuild_finish_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: step4 group_rebuild_finish elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )

        # A5. Refresh dispatch topology
        t0 = time.time()
        if self._topology_refresh_fn is not None and not _skip_infra:
            self._topology_refresh_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                expert_ids=ready_record.expert_ids,
            )
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: step5 topology_refresh elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )
        
        phase_a_elapsed = time.time() - phase_a_start
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  [Phase A] Infrastructure repair COMPLETED (elapsed=%.3fs)",
            _ts(), phase_a_elapsed,
        )

        # =============================================================
        # Phase B: Path-specific parameter recovery
        # =============================================================
        phase_b_start = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  [Phase B] Starting parameter recovery...",
            _ts(),
        )
        
        # Evaluate gap-aware policy to choose between checkpoint restart
        # and hybrid recovery.  Only applies to hard failures when the
        # gap-aware policy manager is configured and enabled.
        # Soft failures (no gap-aware manager) always use hybrid path.

        recovery_path_name = "HYBRID_RECOVERY"  # default
        decision = None

        if (self._gap_aware_policy_manager is not None
                and self._gap_aware_policy_manager.enabled):
            from megatron.core.transformer.moe.gap_aware_recovery_policy import (
                RecoveryPath,
            )
            decision = self._gap_aware_policy_manager.evaluate(
                current_iteration=step,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                num_affected_experts=(
                    len(ready_record.expert_ids)
                    if ready_record.expert_ids else 0
                ),
            )
            recovery_path_name = decision.path.name
            logger.warning(
                "[%s] BSR-MoE controller: gap-aware policy selected %s "
                "(gap=%d, step=%d, failed=%d, replacement=%d, reason=%s)",
                _ts(), recovery_path_name, decision.gap,
                step, failed_rank, replacement_rank, decision.reason,
            )

        # Override: force CHECKPOINT_RESTART if the registered callback
        # says so (e.g., checkpoint is in torch_dist format which does
        # not support selective per-rank loading).
        if (recovery_path_name != "CHECKPOINT_RESTART"
                and self._force_checkpoint_restart_fn is not None):
            try:
                force_result = self._force_checkpoint_restart_fn(step=step)
                if force_result:
                    recovery_path_name = "CHECKPOINT_RESTART"
                    # Build a RecoveryDecision so that checkpoint_restart_fn
                    # and the training loop know the checkpoint iteration
                    # for proper rollback.
                    _force_reason = (
                        force_result.get("reason", str(force_result))
                        if isinstance(force_result, dict)
                        else str(force_result)
                    )
                    _force_ckpt_iter = (
                        force_result.get("checkpoint_iteration", -1)
                        if isinstance(force_result, dict)
                        else -1
                    )
                    from megatron.core.transformer.moe.gap_aware_recovery_policy import (
                        RecoveryPath,
                        RecoveryDecision,
                    )
                    decision = RecoveryDecision(
                        path=RecoveryPath.CHECKPOINT_RESTART,
                        current_step=step,
                        latest_checkpoint_step=_force_ckpt_iter,
                        gap=(step - _force_ckpt_iter
                             if _force_ckpt_iter >= 0 else -1),
                        failed_rank=failed_rank,
                        reason="forced_checkpoint_restart",
                        reason_detail=_force_reason,
                        metadata={
                            "forced": True,
                            "failed_rank": failed_rank,
                            "replacement_rank": replacement_rank,
                        },
                    )
                    logger.warning(
                        "[%s] BSR-MoE controller: CHECKPOINT_RESTART forced "
                        "by force_checkpoint_restart_fn: %s "
                        "(step=%d, ckpt_iter=%d, gap=%d)",
                        _ts(), _force_reason, step,
                        _force_ckpt_iter,
                        decision.gap,
                    )
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: force_checkpoint_restart_fn "
                    "failed: %s (continuing with %s)",
                    e, recovery_path_name,
                )

        if recovery_path_name == "CHECKPOINT_RESTART":
            self._execute_checkpoint_restart_path(
                ready_record=ready_record,
                step=step,
                decision=decision,
            )
        else:
            self._execute_hybrid_recovery_path(
                ready_record=ready_record,
                step=step,
                decision=decision,
            )

        phase_b_elapsed = time.time() - phase_b_start
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  [Phase B] Parameter recovery COMPLETED "
            "(path=%s, elapsed=%.3fs)",
            _ts(), recovery_path_name, phase_b_elapsed,
        )

        # _last_recovery_path is now set inside _execute_checkpoint_restart_path
        # and _execute_hybrid_recovery_path, so no outer assignment needed.

        # =============================================================
        # Phase B→C convergence: unified post-recovery verification
        # =============================================================
        # Both paths have now completed their parameter loading.  Before
        # entering Phase C we call the optional post_recovery_convergence_fn
        # so that directory / health-manager / dispatch-topology views are
        # verified to be consistent regardless of which path was taken.
        if self._post_recovery_convergence_fn is not None:
            t0 = time.time()
            try:
                self._post_recovery_convergence_fn(
                    path=self._last_recovery_path or recovery_path_name,
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                    expert_ids=ready_record.expert_ids or [],
                )
                logger.warning(
                    "[%s] BSR-MoE controller: ⏱️  [Phase B→C] Post-recovery convergence "
                    "completed (path=%s, elapsed=%.3fs, step=%d)",
                    _ts(), self._last_recovery_path, time.time() - t0, step,
                )
            except Exception as conv_e:
                logger.error(
                    "[%s] BSR-MoE controller: post-recovery convergence "
                    "failed (non-fatal): %s", _ts(), conv_e,
                )

        # =============================================================
        # Phase C: Shared post-repair
        # =============================================================
        phase_c_start = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  [Phase C] Starting post-repair...",
            _ts(),
        )

        # C1. PP group repair + P2P rebinding (PP>1 only)
        has_pp = (bool(ready_record.pp_group_ranks)
                  and len(ready_record.pp_group_ranks) > 1)
        if has_pp and self._pipeline_stage_repair_fn is not None:
            self._transition_to(
                RecoveryPhase.PIPELINE_REBINDING,
                event_type="pipeline_rebinding_started",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
            )

            t0 = time.time()
            try:
                self._pipeline_stage_repair_fn(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                    pp_group_ranks=ready_record.pp_group_ranks,
                    failed_stage=ready_record.failed_stage,
                )
                ready_record.pipeline_repaired = True
                logger.warning(
                    "[%s] BSR-MoE controller: pipeline_stage_repair "
                    "completed for rank %d elapsed=%.3fs (step=%d)",
                    _ts(), failed_rank, time.time() - t0, step,
                )
            except Exception as e:
                logger.error(
                    "[%s] BSR-MoE controller: pipeline_stage_repair_fn "
                    "failed for rank %d: %s (elapsed=%.3fs)",
                    _ts(), failed_rank, e, time.time() - t0,
                )

            self._transition_to(
                RecoveryPhase.REINTEGRATED,
                event_type="pipeline_rebinding_completed",
                step=step,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
            )

            ready_record.repair_step = step
            self._inflight_microbatches_invalidated = False
            self._pipeline_rollback_completed = False

            repair_elapsed = time.time() - repair_start
            phase_c_elapsed = time.time() - phase_c_start
            logger.warning(
                "[%s] BSR-MoE controller: ⏱️  ✅ SAFE-POINT REPAIR COMPLETED "
                "(PP>1, path=%s, step=%d) "
                "| Total=%.3fs | Breakdown: PhaseA=%.3fs, PhaseB=%.3fs, PhaseC=%.3fs",
                _ts(), recovery_path_name, step,
                repair_elapsed, phase_a_elapsed, phase_b_elapsed, phase_c_elapsed,
            )
            return True

        # C2. Reintegration barrier setup
        if self._reintegration_barrier is not None:
            try:
                self._reintegration_barrier.begin_reintegration(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    expert_ids=ready_record.expert_ids or [],
                    step=step,
                )
                self._reintegration_barrier.mark_precondition(
                    failed_rank, "replacement_ready", step=step,
                )
                self._reintegration_barrier.mark_precondition(
                    failed_rank, "groups_repaired", step=step,
                )
                self._reintegration_barrier.mark_precondition(
                    failed_rank, "directory_refreshed", step=step,
                )
                self._reintegration_barrier.mark_precondition(
                    failed_rank, "topology_refreshed", step=step,
                )
                self._reintegration_barrier.mark_precondition(
                    failed_rank, "experts_restorable", step=step,
                )
                logger.warning(
                    "[%s] BSR-MoE controller: reintegration barrier initialized "
                    "for rank %d (all preconditions marked, step=%d)",
                    _ts(), failed_rank, step,
                )
            except Exception as e:
                logger.error(
                    "BSR-MoE controller: reintegration barrier setup "
                    "failed for rank %d: %s", failed_rank, e,
                )

        # C3. Record repair step and transition
        ready_record.repair_step = step

        self._transition_to(
            RecoveryPhase.REINTEGRATED,
            event_type="safe_point_repair_completed",
            step=step,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            recovery_path=recovery_path_name,
        )

        repair_elapsed = time.time() - repair_start
        phase_c_elapsed = time.time() - phase_c_start
        logger.warning(
            "[%s] BSR-MoE controller: ⏱️  ✅ SAFE-POINT REPAIR COMPLETED "
            "(path=%s, step=%d, failed=%d, replacement=%d) "
            "| Total=%.3fs | Breakdown: PhaseA=%.3fs, PhaseB=%.3fs, PhaseC=%.3fs",
            _ts(), recovery_path_name, step, failed_rank, replacement_rank,
            repair_elapsed, phase_a_elapsed, phase_b_elapsed, phase_c_elapsed,
        )

        return True

    # -----------------------------------------------------------------
    # Path B1: Checkpoint restart
    # -----------------------------------------------------------------

    def _execute_checkpoint_restart_path(
        self,
        ready_record: FaultRecord,
        step: int,
        decision: Any = None,
    ) -> None:
        """Checkpoint restart path: load ALL params + optimizer state
        from the latest distributed checkpoint.

        This path is selected when the gap between the current iteration
        and the latest checkpoint is small, making a full checkpoint
        restart cheaper than hybrid recovery.

        Protocol (Step 3 minimal flow):
            1. replacement rank ready                  (precondition)
            2. load latest distributed checkpoint      ← this method
            3. wait for safe point                     (precondition)
            4. process-group repair                    (Phase A, already done)
            5. active expert directory refresh         (Phase A, already done)
            6. dispatch topology refresh               (Phase A, already done)
            7. reintegration                           (Phase C, follows)

        After this method returns, the replacement rank has:
        - Dense/shared/router params from checkpoint
        - MoE expert weights from checkpoint
        - Optimizer state from checkpoint (if available)

        Hybrid-recovery semantics (dense-from-peer / expert-from-checkpoint)
        are NOT used in this path.  The training loop is responsible for
        rolling iteration back to the checkpoint iteration so that all
        ranks resume from a consistent boundary.
        """
        failed_rank = ready_record.failed_rank
        replacement_rank = ready_record.replacement_rank

        gap = decision.gap if decision is not None else -1
        ckpt_iter = (
            decision.latest_checkpoint_step if decision is not None else -1
        )

        logger.warning(
            "[%s] BSR-MoE controller: CHECKPOINT_RESTART path SELECTED — "
            "loading all params from checkpoint "
            "(step=%d, failed=%d, replacement=%d, gap=%d, ckpt_iter=%d)",
            _ts(), step, failed_rank, replacement_rank, gap, ckpt_iter,
        )

        t0 = time.time()
        if self._checkpoint_restart_fn is not None:
            try:
                self._checkpoint_restart_fn(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                    decision=decision,
                )
                elapsed = time.time() - t0
                logger.warning(
                    "[%s] BSR-MoE controller: CHECKPOINT_RESTART path "
                    "COMPLETED — checkpoint_restart_fn elapsed=%.3fs "
                    "(step=%d, ckpt_iter=%d, gap=%d)",
                    _ts(), elapsed, step, ckpt_iter, gap,
                )
                # Record event so the path appears in the audit log
                self._event_log.append(RecoveryEvent(
                    event_type="checkpoint_restart_executed",
                    phase_from=self._phase.name,
                    phase_to=self._phase.name,
                    step=step,
                    timestamp=time.time(),
                    details={
                        "failed_rank": failed_rank,
                        "replacement_rank": replacement_rank,
                        "checkpoint_iteration": ckpt_iter,
                        "gap": gap,
                        "elapsed_seconds": elapsed,
                    },
                ))
                self._last_recovery_path = "CHECKPOINT_RESTART"
            except Exception as e:
                logger.error(
                    "[%s] BSR-MoE controller: checkpoint_restart_fn "
                    "FAILED: %s — falling back to hybrid recovery "
                    "(step=%d)",
                    _ts(), e, step,
                )
                self._event_log.append(RecoveryEvent(
                    event_type="checkpoint_restart_failed_fallback",
                    phase_from=self._phase.name,
                    phase_to=self._phase.name,
                    step=step,
                    timestamp=time.time(),
                    details={
                        "failed_rank": failed_rank,
                        "replacement_rank": replacement_rank,
                        "error": str(e),
                    },
                ))
                # Fallback: execute hybrid recovery path instead
                self._execute_hybrid_recovery_path(
                    ready_record=ready_record,
                    step=step,
                    decision=decision,
                )
                # Mark that we actually fell back to hybrid
                self._last_recovery_path = "HYBRID_RECOVERY"
        else:
            logger.warning(
                "[%s] BSR-MoE controller: CHECKPOINT_RESTART selected "
                "but no checkpoint_restart_fn registered — falling "
                "back to hybrid recovery (step=%d)",
                _ts(), step,
            )
            self._execute_hybrid_recovery_path(
                ready_record=ready_record,
                step=step,
                decision=decision,
            )
            self._last_recovery_path = "HYBRID_RECOVERY"

    # -----------------------------------------------------------------
    # Path B2: Hybrid recovery
    # -----------------------------------------------------------------

    def _execute_hybrid_recovery_path(
        self,
        ready_record: FaultRecord,
        step: int,
        decision: Optional["RecoveryDecision"] = None,
    ) -> None:
        """Hybrid recovery path: dense params from DP peer, expert
        weights from distributed checkpoint.

        This path is selected when the gap is large, making it cheaper
        to pull dense params from a healthy DP neighbor (which has the
        current version) and only load MoE expert weights from the
        checkpoint.

        After this method returns, the replacement rank has:
        - Dense/shared/router params from healthy DP peer (current version)
        - MoE expert weights from distributed checkpoint
        - Optimizer state: dense from DP peer, expert from checkpoint

        Args:
            ready_record: The fault record for the rank being recovered.
            step: Current training iteration.
            decision: The ``RecoveryDecision`` from gap-aware policy
                evaluation, if available.  Used to record the hybrid
                recovery event in the ``RankExposureTracker``.
        """
        failed_rank = ready_record.failed_rank
        replacement_rank = ready_record.replacement_rank

        logger.warning(
            "[%s] BSR-MoE controller: HYBRID_RECOVERY path — "
            "dense from DP peer, experts from checkpoint (step=%d, "
            "failed=%d, replacement=%d)",
            _ts(), step, failed_rank, replacement_rank,
        )

        # Record this hybrid recovery event in the RankExposureTracker
        # so that the RankExposureGuardedPolicy can track per-rank
        # stale iterations and decide whether future faults on this rank
        # should trigger a checkpoint restart instead.
        if decision is not None:
            try:
                from megatron.core.transformer.moe.rank_exposure_tracker import (
                    get_rank_exposure_tracker,
                )
                tracker = get_rank_exposure_tracker()
                tracker.record_hybrid_recovery(
                    step=step,
                    rank=failed_rank,
                    gap=decision.gap,
                )
                logger.info(
                    "[%s] BSR-MoE controller: recorded hybrid recovery "
                    "in RankExposureTracker (step=%d, rank=%d, gap=%d)",
                    _ts(), step, failed_rank, decision.gap,
                )
            except Exception as e:
                logger.warning(
                    "BSR-MoE controller: failed to record hybrid recovery "
                    "in RankExposureTracker: %s", e,
                )

        # B2a. Pull dense params from healthy DP peer
        t0 = time.time()
        if self._dense_sync_fn is not None:
            self._dense_sync_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )
        t1 = time.time()
        logger.warning(
            "[%s] BSR-MoE controller: dense_sync elapsed=%.3fs (step=%d)",
            _ts(), t1 - t0, step,
        )

        # B2b. Restore expert weights from checkpoint
        #      Prefer async path if available; fall back to sync.
        t0 = time.time()
        if self._async_expert_restore_fn is not None:
            try:
                request_ids = self._async_expert_restore_fn(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                    expert_ids=ready_record.expert_ids,
                )
                if request_ids:
                    self._async_recovery_pending = True
                    self._async_expert_request_ids = list(request_ids)
                    self._async_recovery_expert_ids = list(
                        ready_record.expert_ids or []
                    )
                    self._async_recovery_failed_rank = failed_rank
                    self._async_recovery_replacement_rank = replacement_rank
                    logger.warning(
                        "[%s] BSR-MoE controller: async expert restore "
                        "submitted (%d requests, elapsed=%.3fs, step=%d)",
                        _ts(), len(request_ids), time.time() - t0, step,
                    )
            except Exception as e:
                logger.error(
                    "[%s] BSR-MoE controller: async_expert_restore_fn "
                    "failed, falling back to sync: %s", _ts(), e,
                )
                if self._expert_restore_fn is not None:
                    self._expert_restore_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                        expert_ids=ready_record.expert_ids,
                    )
                    logger.warning(
                        "[%s] BSR-MoE controller: expert_restore (sync "
                        "fallback) elapsed=%.3fs (step=%d)",
                        _ts(), time.time() - t0, step,
                    )
        elif self._expert_restore_fn is not None:
            self._expert_restore_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                expert_ids=ready_record.expert_ids,
            )
            logger.warning(
                "[%s] BSR-MoE controller: expert_restore (sync) "
                "elapsed=%.3fs (step=%d)",
                _ts(), time.time() - t0, step,
            )

        # Mark the recovery path for external query
        self._last_recovery_path = "HYBRID_RECOVERY"

    def _finalize_reintegration(self, step: int = -1) -> None:
        """Finalize reintegration: move fault records to completed list.

        This is the final step of the recovery sequence.  It:
        1. Checks the reintegration barrier (if set) to ensure all
           preconditions are met before proceeding.
        2. Executes barrier reintegration (re-enable routing, clear
           degraded mode).
        3. Marks recovered experts as HEALTHY (via health_mark_healthy_fn).
        4. Moves fault records from active to completed.
        5. Transitions to HEALTHY_TRAINING phase.
        """
        completed = []
        for rank, record in list(self._active_faults.items()):
            if record.repair_step >= 0:
                # Check barrier if available
                if self._reintegration_barrier is not None:
                    if not self._reintegration_barrier.can_reintegrate(rank):
                        logger.warning(
                            "BSR-MoE controller: barrier blocks reintegration "
                            "for rank %d at step %d (missing preconditions: %s)",
                            rank, step,
                            sorted(self._reintegration_barrier.get_record(rank).missing_preconditions())
                            if self._reintegration_barrier.get_record(rank) else "no record",
                        )
                        continue

                    # Execute barrier reintegration
                    try:
                        def _re_enable_routing(expert_ids, s):
                            """Re-enable routing for recovered experts."""
                            if self._health_mark_healthy_fn is not None:
                                self._health_mark_healthy_fn(
                                    expert_ids=expert_ids, step=s,
                                )

                        self._reintegration_barrier.execute_reintegration(
                            rank,
                            step,
                            re_enable_routing_fn=_re_enable_routing,
                        )
                        logger.info(
                            "BSR-MoE controller: barrier reintegration "
                            "executed for rank %d at step %d",
                            rank, step,
                        )
                    except Exception as e:
                        logger.error(
                            "BSR-MoE controller: barrier reintegration "
                            "failed for rank %d: %s", rank, e,
                        )
                        continue
                else:
                    # No barrier — mark experts healthy directly
                    if record.expert_ids and self._health_mark_healthy_fn is not None:
                        try:
                            self._health_mark_healthy_fn(
                                expert_ids=record.expert_ids,
                                step=step,
                            )
                            logger.info(
                                "BSR-MoE controller: marked experts %s as HEALTHY "
                                "at step %d (failed_rank=%d)",
                                record.expert_ids, step, rank,
                            )
                        except Exception as e:
                            logger.error(
                                "BSR-MoE controller: failed to mark experts "
                                "healthy: %s", e,
                            )

                record.reintegration_step = step
                completed.append(rank)
                self._completed_recoveries.append(record)

        for rank in completed:
            del self._active_faults[rank]

        self._transition_to(
            RecoveryPhase.HEALTHY_TRAINING,
            event_type="reintegration_finalized",
            step=step,
        )

        logger.warning(
            "[%s] BSR-MoE controller: reintegration finalized at step %d "
            "(%d recoveries completed)",
            _ts(), step, len(completed),
        )

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    @property
    def phase(self) -> RecoveryPhase:
        """Current recovery phase."""
        return self._phase

    @property
    def expert_tracker(self) -> ExpertRecoveryTracker:
        """Access the expert recovery state tracker."""
        return self._expert_tracker

    @property
    def is_healthy(self) -> bool:
        """True if in HEALTHY_TRAINING phase."""
        return self._phase == RecoveryPhase.HEALTHY_TRAINING

    @property
    def is_degraded(self) -> bool:
        """True if in any non-healthy phase."""
        return self._phase != RecoveryPhase.HEALTHY_TRAINING

    @property
    def async_recovery_pending(self) -> bool:
        """True if async expert recovery is in progress."""
        return self._async_recovery_pending

    @property
    def last_recovery_path(self) -> str:
        """The recovery path used in the most recent safe-point repair.

        Returns ``"CHECKPOINT_RESTART"`` or ``"HYBRID_RECOVERY"`` or ``""``
        if no repair has been executed yet.
        """
        return self._last_recovery_path

    @property
    def restart_in_place_mode(self) -> bool:
        """True if the controller is in restart-in-place test mode."""
        return self._restart_in_place_mode

    @property
    def iteration_was_invalidated(self) -> bool:
        """True if the current iteration was invalidated by a hard failure.

        The training loop should check this after ``train_step()`` returns
        (or after catching an exception from ``train_step()``) to decide
        whether to skip the optimizer commit and iteration increment.
        """
        return self._iteration_invalidated

    @property
    def reintegration_barrier(self):
        """The ReintegrationBarrier instance, or None."""
        return self._reintegration_barrier

    @property
    def invalidated_step(self) -> int:
        """The step that was invalidated, or -1 if none."""
        return self._invalidated_step

    def clear_iteration_invalidation(self) -> None:
        """Clear the iteration-invalidated flag.

        Called by the training loop after it has handled the invalidation
        (skipped optimizer commit, discarded loss, etc.).  Typically called
        from ``bsr_before_iteration()`` at the start of the next iteration.
        """
        if self._iteration_invalidated:
            logger.info(
                "BSR-MoE controller: clearing iteration invalidation "
                "(was step %d)", self._invalidated_step,
            )
        self._iteration_invalidated = False
        self._invalidated_step = -1

    @property
    def num_active_faults(self) -> int:
        """Number of active (unresolved) faults."""
        return len(self._active_faults)

    @property
    def active_faults(self) -> Dict[int, FaultRecord]:
        """Copy of active fault records."""
        return dict(self._active_faults)

    @property
    def completed_recoveries(self) -> List[FaultRecord]:
        """Copy of completed recovery records."""
        return list(self._completed_recoveries)

    @property
    def num_completed_recoveries(self) -> int:
        """Number of completed recoveries."""
        return len(self._completed_recoveries)

    @property
    def event_log(self) -> List[RecoveryEvent]:
        """Copy of the event log."""
        return list(self._event_log)

    def get_fault_record(self, failed_rank: int) -> Optional[FaultRecord]:
        """Get the fault record for a specific rank."""
        return self._active_faults.get(failed_rank)

    def summary(self) -> Dict[str, Any]:
        """Summary of the controller state."""
        result = {
            "phase": self._phase.name,
            "num_active_faults": self.num_active_faults,
            "active_failed_ranks": sorted(self._active_faults.keys()),
            "num_completed_recoveries": self.num_completed_recoveries,
            "num_events": len(self._event_log),
            "last_recovery_path": self._last_recovery_path,
        }
        if self._gap_aware_policy_manager is not None:
            result["gap_aware_policy"] = self._gap_aware_policy_manager.summary()
        return result

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset the controller to initial state."""
        self._phase = RecoveryPhase.HEALTHY_TRAINING
        self._active_faults.clear()
        self._completed_recoveries.clear()
        self._event_log.clear()
        self._iteration_invalidated = False
        self._invalidated_step = -1
        self._inflight_microbatches_invalidated = False
        self._pipeline_rollback_completed = False
        # Reset async recovery state
        self._async_recovery_pending = False
        self._async_expert_request_ids.clear()
        self._async_recovery_expert_ids.clear()
        self._async_recovery_failed_rank = -1
        self._async_recovery_replacement_rank = -1
        if self._reintegration_barrier is not None:
            self._reintegration_barrier.reset()
        self._reintegration_barrier = None
        # Reset gap-aware policy
        if self._gap_aware_policy_manager is not None:
            self._gap_aware_policy_manager.reset()
        self._gap_aware_policy_manager = None
        self._checkpoint_restart_fn = None
        self._force_checkpoint_restart_fn = None
        self._last_recovery_path = ""
        self._restart_in_place_mode = False

    def __repr__(self) -> str:
        return (
            f"RecoveryController(phase={self._phase.name}, "
            f"active_faults={self.num_active_faults}, "
            f"completed={self.num_completed_recoveries})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_CONTROLLER: Optional[RecoveryController] = None


def get_recovery_controller() -> RecoveryController:
    """Get or create the global recovery controller singleton."""
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = RecoveryController()
    return _CONTROLLER


def clear_recovery_controller() -> None:
    """Reset the global controller (for testing)."""
    global _CONTROLLER
    if _CONTROLLER is not None:
        _CONTROLLER.reset()
    _CONTROLLER = None
