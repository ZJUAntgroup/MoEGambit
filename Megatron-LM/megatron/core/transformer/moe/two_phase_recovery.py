# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-Phase Weights-First Recovery Protocol.

This module implements a **weights-first, optimizer-later** recovery
protocol for MoE experts.  The key insight is that expert weights are
needed to resume forward/backward passes, but optimizer state (momentum,
variance) can be loaded in the background while training continues.

Two-phase recovery state machine (per recovery session)
-------------------------------------------------------
::

    NOT_STARTED
        │
        ▼  begin_recovery()
    WEIGHTS_LOADING
        │
        ▼  on_weights_restored()
    WEIGHTS_READY          ← expert re-joins forward/backward here
        │
        ▼  on_optimizer_submitted()
    OPTIMIZER_PENDING      ← optimizer state loading in background
        │
        ▼  on_optimizer_loaded()
    FULLY_RECOVERED        ← optimizer barrier lifted
        │
        ▼  on_promoted_healthy()
    COMPLETED

Update barrier semantics
------------------------
+-----------------------+----------+----------+----------------+
| Phase                 | Forward  | Backward | Optimizer Step |
+=======================+==========+==========+================+
| NOT_STARTED /         |    ✗     |    ✗     |       ✗        |
| WEIGHTS_LOADING       |          |          |                |
+-----------------------+----------+----------+----------------+
| WEIGHTS_READY /       |    ✓     |    ✓     |       ✗        |
| OPTIMIZER_PENDING     |          |          |  (barrier)     |
+-----------------------+----------+----------+----------------+
| FULLY_RECOVERED /     |    ✓     |    ✓     |       ✓        |
| COMPLETED             |          |          |                |
+-----------------------+----------+----------+----------------+

Dense vs Expert optimizer state
-------------------------------
* **Dense / shared / router** optimizer state is restored **synchronously**
  during reintegration (before training resumes).  All DP peers must hold
  identical dense optimizer state for gradient averaging to be correct.
* **Expert** optimizer state is restored **asynchronously** (deferred).
  Each expert's optimizer state is independent across EP ranks, so
  loading it in the background does not break consistency.  The
  ``OptimizerUpdateBarrier`` blocks gradient updates for the affected
  expert parameters until the optimizer state is loaded.

Integration with existing modules
----------------------------------
* ``ExpertHealthManager`` — maps to per-expert states:
    - UNAVAILABLE → WEIGHTS_LOADING (begin_recovery)
    - STALE_RUNNABLE → WEIGHTS_READY (on_weights_restored)
    - FULLY_RECOVERED → FULLY_RECOVERED (on_optimizer_loaded)
    - HEALTHY → COMPLETED (on_promoted_healthy)
* ``OptimizerUpdateBarrier`` — blocks expert optimizer step during
  WEIGHTS_READY and OPTIMIZER_PENDING phases.
* ``DeferredOptimizerLoader`` — manages the actual optimizer state
  loading during OPTIMIZER_PENDING phase.
* ``RecoveryController`` — orchestrates the overall recovery flow;
  this module provides the per-session tracking layer on top.

Scope (v1)
----------
* ✅ Per-session two-phase state machine
* ✅ Explicit update barrier semantics
* ✅ Dense optimizer state: synchronous (before reintegration)
* ✅ Expert optimizer state: deferred (async after weights restored)
* ✅ Timing metrics for each phase
* ✅ Unified deferred-load hook (extensible for future dense deferral)
* ❌ True async weight loading (uses existing sync/async paths)
* ❌ Automatic FULLY_RECOVERED → HEALTHY promotion (separate barrier)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Recovery phase enum (per-session)
# =====================================================================

class TwoPhaseState(enum.IntEnum):
    """States for a single two-phase recovery session."""

    NOT_STARTED = 0
    """Recovery has not begun."""

    WEIGHTS_LOADING = 1
    """Expert weights are being loaded from checkpoint.
    Forward/backward: BLOCKED.  Optimizer step: BLOCKED."""

    WEIGHTS_READY = 2
    """Expert weights have been restored.  The expert can participate
    in forward and backward passes.
    Forward/backward: ALLOWED.  Optimizer step: BLOCKED (barrier)."""

    OPTIMIZER_PENDING = 3
    """Optimizer state load has been submitted and is in progress.
    Forward/backward: ALLOWED.  Optimizer step: BLOCKED (barrier)."""

    FULLY_RECOVERED = 4
    """Optimizer state has been loaded and the barrier is lifted.
    Forward/backward: ALLOWED.  Optimizer step: ALLOWED.
    Awaiting promotion to HEALTHY at the next safe barrier."""

    COMPLETED = 5
    """Recovery is complete.  Expert has been promoted to HEALTHY."""


# Valid transitions
_VALID_TWO_PHASE_TRANSITIONS = {
    TwoPhaseState.NOT_STARTED:       {TwoPhaseState.WEIGHTS_LOADING},
    TwoPhaseState.WEIGHTS_LOADING:   {TwoPhaseState.WEIGHTS_READY,
                                      TwoPhaseState.NOT_STARTED},  # abort
    TwoPhaseState.WEIGHTS_READY:     {TwoPhaseState.OPTIMIZER_PENDING,
                                      TwoPhaseState.FULLY_RECOVERED},  # skip opt
    TwoPhaseState.OPTIMIZER_PENDING: {TwoPhaseState.FULLY_RECOVERED,
                                      TwoPhaseState.WEIGHTS_READY},  # retry
    TwoPhaseState.FULLY_RECOVERED:   {TwoPhaseState.COMPLETED},
    TwoPhaseState.COMPLETED:         set(),  # terminal
}

# States where forward/backward is allowed
_FORWARD_ALLOWED_STATES = frozenset({
    TwoPhaseState.WEIGHTS_READY,
    TwoPhaseState.OPTIMIZER_PENDING,
    TwoPhaseState.FULLY_RECOVERED,
    TwoPhaseState.COMPLETED,
})

# States where optimizer step is allowed
_OPTIMIZER_STEP_ALLOWED_STATES = frozenset({
    TwoPhaseState.FULLY_RECOVERED,
    TwoPhaseState.COMPLETED,
})


# =====================================================================
# Per-expert recovery session
# =====================================================================

@dataclass
class ExpertRecoverySession:
    """Tracks a single expert's two-phase recovery progress."""

    layer_id: int = -1
    """MoE layer index."""

    expert_id: int = -1
    """Global expert index."""

    state: TwoPhaseState = TwoPhaseState.NOT_STARTED
    """Current recovery phase."""

    failed_rank: int = -1
    """The rank that failed."""

    replacement_rank: int = -1
    """The replacement rank."""

    # Timing
    recovery_start_time: float = 0.0
    """Wall-clock time when recovery began."""

    weights_ready_time: float = 0.0
    """Wall-clock time when weights were restored."""

    optimizer_submit_time: float = 0.0
    """Wall-clock time when optimizer load was submitted."""

    fully_recovered_time: float = 0.0
    """Wall-clock time when optimizer state was loaded."""

    completed_time: float = 0.0
    """Wall-clock time when promoted to HEALTHY."""

    # Step tracking
    recovery_start_step: int = -1
    weights_ready_step: int = -1
    optimizer_submit_step: int = -1
    fully_recovered_step: int = -1
    completed_step: int = -1

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.expert_id)

    def transition_to(self, new_state: TwoPhaseState) -> None:
        """Perform a validated state transition."""
        allowed = _VALID_TWO_PHASE_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid two-phase transition for expert "
                f"(layer={self.layer_id}, id={self.expert_id}): "
                f"{self.state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        self.state = new_state

    @property
    def is_forward_allowed(self) -> bool:
        """Whether this expert can participate in forward/backward."""
        return self.state in _FORWARD_ALLOWED_STATES

    @property
    def is_optimizer_step_allowed(self) -> bool:
        """Whether optimizer updates are allowed for this expert."""
        return self.state in _OPTIMIZER_STEP_ALLOWED_STATES

    @property
    def weights_restore_elapsed(self) -> float:
        """Seconds from recovery start to weights ready."""
        if self.weights_ready_time > 0 and self.recovery_start_time > 0:
            return self.weights_ready_time - self.recovery_start_time
        return 0.0

    @property
    def optimizer_load_elapsed(self) -> float:
        """Seconds from optimizer submit to fully recovered."""
        if self.fully_recovered_time > 0 and self.optimizer_submit_time > 0:
            return self.fully_recovered_time - self.optimizer_submit_time
        return 0.0

    @property
    def total_recovery_elapsed(self) -> float:
        """Total seconds from recovery start to completion."""
        end = self.completed_time or self.fully_recovered_time
        if end > 0 and self.recovery_start_time > 0:
            return end - self.recovery_start_time
        return 0.0

    @property
    def time_to_trainable(self) -> float:
        """Seconds from recovery start to first trainable state
        (WEIGHTS_READY).  This is the key metric: how quickly the
        expert can re-join training."""
        return self.weights_restore_elapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "state": self.state.name,
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "weights_restore_elapsed": self.weights_restore_elapsed,
            "optimizer_load_elapsed": self.optimizer_load_elapsed,
            "total_recovery_elapsed": self.total_recovery_elapsed,
            "time_to_trainable": self.time_to_trainable,
            "recovery_start_step": self.recovery_start_step,
            "weights_ready_step": self.weights_ready_step,
            "optimizer_submit_step": self.optimizer_submit_step,
            "fully_recovered_step": self.fully_recovered_step,
            "completed_step": self.completed_step,
        }


# =====================================================================
# Two-Phase Recovery Coordinator
# =====================================================================

class TwoPhaseRecoveryCoordinator:
    """Coordinates the weights-first, optimizer-later recovery protocol.

    This coordinator sits on top of the existing recovery infrastructure
    (RecoveryController, ExpertHealthManager, DeferredOptimizerLoader)
    and provides per-session tracking of the two-phase recovery flow.

    Usage::

        coord = TwoPhaseRecoveryCoordinator()

        # Phase 1: begin recovery (expert weights loading)
        coord.begin_recovery(
            expert_ids=[(1, 2), (1, 3)],
            failed_rank=1, replacement_rank=10, step=100,
        )

        # ... weights are loaded by StaleExpertRestoreCoordinator ...

        # Phase 1 complete: weights restored
        coord.on_weights_restored(
            expert_ids=[(1, 2), (1, 3)], step=105,
        )
        # → experts are now WEIGHTS_READY (forward/backward allowed)

        # Phase 2: submit optimizer state loads
        coord.on_optimizer_submitted(
            expert_ids=[(1, 2), (1, 3)], step=105,
        )
        # → experts are now OPTIMIZER_PENDING

        # ... optimizer state loaded by DeferredOptimizerLoader ...

        # Phase 2 complete: optimizer state loaded
        coord.on_optimizer_loaded(
            expert_ids=[(1, 2), (1, 3)], step=110,
        )
        # → experts are now FULLY_RECOVERED (optimizer step allowed)

        # Promotion to HEALTHY (at safe barrier)
        coord.on_promoted_healthy(
            expert_ids=[(1, 2), (1, 3)], step=111,
        )
    """

    def __init__(self, *, defer_optimizer: bool = True) -> None:
        self._sessions: Dict[Tuple[int, int], ExpertRecoverySession] = {}
        self._completed_history: List[ExpertRecoverySession] = []
        self._defer_optimizer = defer_optimizer

    # -----------------------------------------------------------------
    # Phase 1: Weight recovery
    # -----------------------------------------------------------------

    def begin_recovery(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        step: int = -1,
    ) -> List[ExpertRecoverySession]:
        """Begin two-phase recovery for a set of experts.

        Creates recovery sessions and transitions them to WEIGHTS_LOADING.

        Args:
            expert_ids: List of (layer_id, expert_id) tuples.
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            step: Current training step.

        Returns:
            List of created sessions.
        """
        now = time.monotonic()
        sessions = []
        for layer_id, expert_id in expert_ids:
            key = (layer_id, expert_id)
            # If there's an existing non-completed session, abort it
            existing = self._sessions.get(key)
            if existing is not None and existing.state != TwoPhaseState.COMPLETED:
                logger.warning(
                    "BSR-MoE two-phase: aborting existing session for "
                    "expert (layer=%d, id=%d) in state %s",
                    layer_id, expert_id, existing.state.name,
                )

            session = ExpertRecoverySession(
                layer_id=layer_id,
                expert_id=expert_id,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                recovery_start_time=now,
                recovery_start_step=step,
            )
            session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
            self._sessions[key] = session
            sessions.append(session)

        logger.info(
            "BSR-MoE two-phase: began recovery for %d experts "
            "(failed_rank=%d, replacement=%d, step=%d)",
            len(sessions), failed_rank, replacement_rank, step,
        )
        return sessions

    def on_weights_restored(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        step: int = -1,
    ) -> int:
        """Mark experts as WEIGHTS_READY (Phase 1 complete).

        After this call, experts can participate in forward/backward
        but optimizer updates remain blocked.

        Returns:
            Number of experts transitioned.
        """
        now = time.monotonic()
        count = 0
        for key in expert_ids:
            session = self._sessions.get(key)
            if session is None:
                continue
            if session.state != TwoPhaseState.WEIGHTS_LOADING:
                logger.debug(
                    "BSR-MoE two-phase: skipping on_weights_restored for "
                    "expert %s (state=%s)", key, session.state.name,
                )
                continue
            session.transition_to(TwoPhaseState.WEIGHTS_READY)
            session.weights_ready_time = now
            session.weights_ready_step = step
            count += 1

        if count > 0:
            logger.warning(
                "BSR-MoE two-phase: %d experts → WEIGHTS_READY at step %d "
                "(forward/backward ALLOWED, optimizer step BLOCKED)",
                count, step,
            )
        return count

    # -----------------------------------------------------------------
    # Phase 2: Optimizer state recovery
    # -----------------------------------------------------------------

    def on_optimizer_submitted(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        step: int = -1,
    ) -> int:
        """Mark experts as OPTIMIZER_PENDING (optimizer load submitted).

        Returns:
            Number of experts transitioned.
        """
        now = time.monotonic()
        count = 0
        for key in expert_ids:
            session = self._sessions.get(key)
            if session is None:
                continue
            if session.state != TwoPhaseState.WEIGHTS_READY:
                continue
            session.transition_to(TwoPhaseState.OPTIMIZER_PENDING)
            session.optimizer_submit_time = now
            session.optimizer_submit_step = step
            count += 1

        if count > 0:
            logger.info(
                "BSR-MoE two-phase: %d experts → OPTIMIZER_PENDING at step %d",
                count, step,
            )
        return count

    def on_optimizer_loaded(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        step: int = -1,
    ) -> int:
        """Mark experts as FULLY_RECOVERED (Phase 2 complete).

        After this call, the optimizer update barrier is lifted and
        optimizer steps are allowed.

        Returns:
            Number of experts transitioned.
        """
        now = time.monotonic()
        count = 0
        for key in expert_ids:
            session = self._sessions.get(key)
            if session is None:
                continue
            if session.state not in (
                TwoPhaseState.OPTIMIZER_PENDING,
                TwoPhaseState.WEIGHTS_READY,  # skip-opt path
            ):
                continue
            session.transition_to(TwoPhaseState.FULLY_RECOVERED)
            session.fully_recovered_time = now
            session.fully_recovered_step = step
            count += 1

        if count > 0:
            # Compute average time-to-trainable for logging
            sessions = [self._sessions[k] for k in expert_ids
                        if k in self._sessions]
            avg_ttt = sum(s.time_to_trainable for s in sessions) / max(len(sessions), 1)
            avg_opt = sum(s.optimizer_load_elapsed for s in sessions) / max(len(sessions), 1)
            logger.warning(
                "BSR-MoE two-phase: %d experts → FULLY_RECOVERED at step %d "
                "(avg time-to-trainable=%.3fs, avg optimizer-load=%.3fs, "
                "optimizer step now ALLOWED)",
                count, step, avg_ttt, avg_opt,
            )
        return count

    def on_promoted_healthy(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        step: int = -1,
    ) -> int:
        """Mark experts as COMPLETED (promoted to HEALTHY).

        Returns:
            Number of experts transitioned.
        """
        now = time.monotonic()
        count = 0
        for key in expert_ids:
            session = self._sessions.get(key)
            if session is None:
                continue
            if session.state != TwoPhaseState.FULLY_RECOVERED:
                continue
            session.transition_to(TwoPhaseState.COMPLETED)
            session.completed_time = now
            session.completed_step = step
            self._completed_history.append(session)
            count += 1

        if count > 0:
            logger.info(
                "BSR-MoE two-phase: %d experts → COMPLETED at step %d",
                count, step,
            )
        return count

    # -----------------------------------------------------------------
    # Skip-optimizer shortcut
    # -----------------------------------------------------------------

    def skip_optimizer_phase(
        self,
        expert_ids: List[Tuple[int, int]],
        *,
        step: int = -1,
    ) -> int:
        """Skip the optimizer loading phase (WEIGHTS_READY → FULLY_RECOVERED).

        Used when ``moe_bsr_defer_optimizer_load = False`` or when
        optimizer state is not available.  The expert transitions directly
        from WEIGHTS_READY to FULLY_RECOVERED.

        Returns:
            Number of experts transitioned.
        """
        now = time.monotonic()
        count = 0
        for key in expert_ids:
            session = self._sessions.get(key)
            if session is None:
                continue
            if session.state != TwoPhaseState.WEIGHTS_READY:
                continue
            session.transition_to(TwoPhaseState.FULLY_RECOVERED)
            session.fully_recovered_time = now
            session.fully_recovered_step = step
            count += 1

        if count > 0:
            logger.info(
                "BSR-MoE two-phase: %d experts skipped optimizer phase "
                "→ FULLY_RECOVERED at step %d",
                count, step,
            )
        return count

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    def get_session(
        self, layer_id: int, expert_id: int,
    ) -> Optional[ExpertRecoverySession]:
        """Get the recovery session for a specific expert."""
        return self._sessions.get((layer_id, expert_id))

    def get_state(
        self, layer_id: int, expert_id: int,
    ) -> TwoPhaseState:
        """Get the two-phase state for a specific expert."""
        session = self._sessions.get((layer_id, expert_id))
        if session is None:
            return TwoPhaseState.NOT_STARTED
        return session.state

    def is_forward_allowed(self, layer_id: int, expert_id: int) -> bool:
        """Check if forward/backward is allowed for an expert."""
        session = self._sessions.get((layer_id, expert_id))
        if session is None:
            return True  # No recovery session → normal operation
        return session.is_forward_allowed

    def is_optimizer_step_allowed(self, layer_id: int, expert_id: int) -> bool:
        """Check if optimizer step is allowed for an expert."""
        session = self._sessions.get((layer_id, expert_id))
        if session is None:
            return True  # No recovery session → normal operation
        return session.is_optimizer_step_allowed

    def get_experts_in_state(
        self, state: TwoPhaseState,
    ) -> List[Tuple[int, int]]:
        """Get all experts currently in a given state."""
        return [
            key for key, session in self._sessions.items()
            if session.state == state
        ]

    @property
    def num_active(self) -> int:
        """Number of active (non-completed) recovery sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.state != TwoPhaseState.COMPLETED
        )

    @property
    def num_weights_ready(self) -> int:
        """Number of experts in WEIGHTS_READY state."""
        return sum(
            1 for s in self._sessions.values()
            if s.state == TwoPhaseState.WEIGHTS_READY
        )

    @property
    def num_optimizer_pending(self) -> int:
        """Number of experts in OPTIMIZER_PENDING state."""
        return sum(
            1 for s in self._sessions.values()
            if s.state == TwoPhaseState.OPTIMIZER_PENDING
        )

    @property
    def num_fully_recovered(self) -> int:
        """Number of experts in FULLY_RECOVERED state."""
        return sum(
            1 for s in self._sessions.values()
            if s.state == TwoPhaseState.FULLY_RECOVERED
        )

    @property
    def all_completed(self) -> bool:
        """True if all sessions have completed."""
        return all(
            s.state == TwoPhaseState.COMPLETED
            for s in self._sessions.values()
        )

    @property
    def all_weights_ready(self) -> bool:
        """True if all active sessions have at least reached WEIGHTS_READY."""
        return all(
            s.state in _FORWARD_ALLOWED_STATES
            for s in self._sessions.values()
        )

    @property
    def defer_optimizer(self) -> bool:
        """Whether optimizer state loading is deferred."""
        return self._defer_optimizer

    def summary(self) -> Dict[str, Any]:
        """Summary of all recovery sessions."""
        by_state: Dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.state.name] = by_state.get(s.state.name, 0) + 1

        # Compute average metrics for completed sessions
        completed = [s for s in self._sessions.values()
                     if s.state in (TwoPhaseState.FULLY_RECOVERED,
                                    TwoPhaseState.COMPLETED)]
        avg_ttt = 0.0
        avg_total = 0.0
        if completed:
            avg_ttt = sum(s.time_to_trainable for s in completed) / len(completed)
            avg_total = sum(s.total_recovery_elapsed for s in completed) / len(completed)

        return {
            "total_sessions": len(self._sessions),
            "active": self.num_active,
            "by_state": by_state,
            "defer_optimizer": self._defer_optimizer,
            "avg_time_to_trainable": avg_ttt,
            "avg_total_recovery": avg_total,
            "completed_history": len(self._completed_history),
        }

    def reset(self) -> None:
        """Clear all sessions and history."""
        self._sessions.clear()
        self._completed_history.clear()

    def __repr__(self) -> str:
        return (
            f"TwoPhaseRecoveryCoordinator("
            f"sessions={len(self._sessions)}, "
            f"active={self.num_active}, "
            f"defer_optimizer={self._defer_optimizer})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_COORDINATOR: Optional[TwoPhaseRecoveryCoordinator] = None


def get_two_phase_recovery_coordinator(
    *, defer_optimizer: bool = True,
) -> TwoPhaseRecoveryCoordinator:
    """Get or create the global two-phase recovery coordinator."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = TwoPhaseRecoveryCoordinator(
            defer_optimizer=defer_optimizer,
        )
    return _COORDINATOR


def clear_two_phase_recovery_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.reset()
    _COORDINATOR = None
