# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# BSR-MoE: Stage-Safe Recovery Protocol for PP > 1
#
# This module provides the minimal end-to-end protocol that ties together
# the existing PP recovery building blocks into a single, testable
# closed-loop sequence:
#
#   detect → invalidate → rollback → wait-for-replacement →
#   safe-point repair → P2P rebind → resume training
#
# It does NOT duplicate logic from the building blocks; instead it
# orchestrates them in the correct order and verifies pre/post-conditions
# at each step.
#
# Design principles:
#   1. Reuse existing modules (PipelineRollbackCoordinator,
#      PipelineStageRepairer, RecoveryController, etc.)
#   2. No mid-microbatch recovery — failure is detected at iteration
#      boundary (exception from forward_backward_func).
#   3. Replacement rank takes over the same logical stage.
#   4. Dense/expert dual-path recovery semantics are unchanged from PP=1.

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Protocol phase
# =====================================================================

class StageRecoveryPhase(enum.IntEnum):
    """Phases of the stage-safe recovery protocol."""
    IDLE = 0
    FAILURE_DETECTED = 1
    ITERATION_INVALIDATED = 2
    ROLLBACK_COMPLETED = 3
    WAITING_FOR_REPLACEMENT = 4
    SAFE_POINT_REPAIR = 5
    P2P_REBOUND = 6
    TRAINING_RESUMED = 7


_VALID_PHASE_TRANSITIONS = {
    StageRecoveryPhase.IDLE: {
        StageRecoveryPhase.FAILURE_DETECTED,
    },
    StageRecoveryPhase.FAILURE_DETECTED: {
        StageRecoveryPhase.ITERATION_INVALIDATED,
        StageRecoveryPhase.IDLE,  # abort
    },
    StageRecoveryPhase.ITERATION_INVALIDATED: {
        StageRecoveryPhase.ROLLBACK_COMPLETED,
        StageRecoveryPhase.IDLE,
    },
    StageRecoveryPhase.ROLLBACK_COMPLETED: {
        StageRecoveryPhase.WAITING_FOR_REPLACEMENT,
        StageRecoveryPhase.IDLE,
    },
    StageRecoveryPhase.WAITING_FOR_REPLACEMENT: {
        StageRecoveryPhase.SAFE_POINT_REPAIR,
        StageRecoveryPhase.IDLE,
    },
    StageRecoveryPhase.SAFE_POINT_REPAIR: {
        StageRecoveryPhase.P2P_REBOUND,
        StageRecoveryPhase.IDLE,
    },
    StageRecoveryPhase.P2P_REBOUND: {
        StageRecoveryPhase.TRAINING_RESUMED,
        StageRecoveryPhase.IDLE,
    },
    StageRecoveryPhase.TRAINING_RESUMED: {
        StageRecoveryPhase.IDLE,
    },
}


# =====================================================================
# Protocol result
# =====================================================================

@dataclass
class StageRecoveryResult:
    """Result of a stage-safe recovery sequence."""

    phase_reached: StageRecoveryPhase = StageRecoveryPhase.IDLE
    failed_rank: int = -1
    failed_stage: int = -1
    replacement_rank: int = -1
    step: int = -1

    # Per-step outcomes
    iteration_invalidated: bool = False
    rollback_success: bool = False
    replacement_ready: bool = False
    groups_repaired: bool = False
    p2p_rebound: bool = False
    training_resumed: bool = False

    # Recovery path
    recovery_path: str = ""  # "CHECKPOINT_RESTART", "HYBRID_RECOVERY", or "FULL_PEER_RECOVERY"

    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            self.phase_reached == StageRecoveryPhase.TRAINING_RESUMED
            and len(self.errors) == 0
        )

    def to_dict(self) -> Dict:
        return {
            "phase_reached": self.phase_reached.name,
            "failed_rank": self.failed_rank,
            "failed_stage": self.failed_stage,
            "replacement_rank": self.replacement_rank,
            "step": self.step,
            "iteration_invalidated": self.iteration_invalidated,
            "rollback_success": self.rollback_success,
            "replacement_ready": self.replacement_ready,
            "groups_repaired": self.groups_repaired,
            "p2p_rebound": self.p2p_rebound,
            "training_resumed": self.training_resumed,
            "recovery_path": self.recovery_path,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "success": self.success,
            "errors": self.errors,
        }


# =====================================================================
# StageSafeRecoveryProtocol
# =====================================================================

class StageSafeRecoveryProtocol:
    """Orchestrates the minimal stage-safe recovery closed loop for PP > 1.

    This class does NOT own any of the building blocks — it receives them
    as injectable callbacks so that each step can be tested in isolation
    and the real modules can be swapped for mocks.

    Usage::

        protocol = StageSafeRecoveryProtocol()
        result = protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=1000,
            pp_group_ranks=[0, 2, 4, 6],
            ep_group_ranks=[0, 1, 2, 3],
            # ... callbacks ...
        )
    """

    def __init__(self) -> None:
        self._phase = StageRecoveryPhase.IDLE
        self._history: List[StageRecoveryResult] = []

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def _transition(self, new_phase: StageRecoveryPhase) -> None:
        valid = _VALID_PHASE_TRANSITIONS.get(self._phase, set())
        if new_phase not in valid:
            raise ValueError(
                f"Invalid phase transition: {self._phase.name} → {new_phase.name}"
            )
        old = self._phase
        self._phase = new_phase
        logger.info(
            "BSR-MoE stage-safe: %s → %s", old.name, new_phase.name,
        )

    @property
    def phase(self) -> StageRecoveryPhase:
        return self._phase

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        failed_rank: int,
        failed_stage: int,
        replacement_rank: int,
        step: int,
        pp_group_ranks: List[int],
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
        expert_ids: Optional[List[int]] = None,
        recovery_path: str = "HYBRID_RECOVERY",
        # Callbacks — all optional, each step is skipped if callback is None
        invalidate_iteration_fn: Optional[Callable] = None,
        rollback_fn: Optional[Callable] = None,
        wait_replacement_fn: Optional[Callable] = None,
        group_repair_fn: Optional[Callable] = None,
        topology_refresh_fn: Optional[Callable] = None,
        p2p_rebind_fn: Optional[Callable] = None,
        dense_sync_fn: Optional[Callable] = None,
        expert_restore_fn: Optional[Callable] = None,
        expert_peer_sync_fn: Optional[Callable] = None,
        checkpoint_restart_fn: Optional[Callable] = None,
        convergence_fn: Optional[Callable] = None,
        reintegration_fn: Optional[Callable] = None,
    ) -> StageRecoveryResult:
        """Execute the full stage-safe recovery protocol.

        Steps:
            1. Detect failure → FAILURE_DETECTED
            2. Invalidate iteration + in-flight microbatches → ITERATION_INVALIDATED
            3. Rollback (data iterator rewind, counter restore) → ROLLBACK_COMPLETED
            4. Wait for replacement rank → WAITING_FOR_REPLACEMENT
            5. Safe-point repair (group rebuild + topology refresh + param recovery) → SAFE_POINT_REPAIR
            6. P2P rebind → P2P_REBOUND
            7. Resume training → TRAINING_RESUMED
        """
        t_start = time.time()
        result = StageRecoveryResult(
            failed_rank=failed_rank,
            failed_stage=failed_stage,
            replacement_rank=replacement_rank,
            step=step,
            recovery_path=recovery_path,
        )

        logger.warning(
            "[%s] BSR-MoE stage-safe: starting recovery "
            "(failed_rank=%d, stage=%d, replacement=%d, "
            "path=%s, step=%d, pp_ranks=%s)",
            _ts(), failed_rank, failed_stage, replacement_rank,
            recovery_path, step, pp_group_ranks,
        )

        try:
            # ---- Step 1: Detect failure ----
            self._transition(StageRecoveryPhase.FAILURE_DETECTED)
            result.phase_reached = StageRecoveryPhase.FAILURE_DETECTED

            # ---- Step 2: Invalidate iteration ----
            if invalidate_iteration_fn is not None:
                try:
                    invalidate_iteration_fn(
                        step=step,
                        failed_rank=failed_rank,
                        failed_stage=failed_stage,
                    )
                except Exception as e:
                    result.errors.append(f"invalidate_iteration failed: {e}")
            result.iteration_invalidated = True
            self._transition(StageRecoveryPhase.ITERATION_INVALIDATED)
            result.phase_reached = StageRecoveryPhase.ITERATION_INVALIDATED

            # ---- Step 3: Rollback ----
            if rollback_fn is not None:
                try:
                    rb_result = rollback_fn(
                        failed_rank=failed_rank,
                        failed_stage=failed_stage,
                        step=step,
                    )
                    result.rollback_success = (
                        rb_result if isinstance(rb_result, bool) else True
                    )
                except Exception as e:
                    result.errors.append(f"rollback failed: {e}")
                    result.rollback_success = False
            else:
                result.rollback_success = True  # No rollback needed (test mode)
            self._transition(StageRecoveryPhase.ROLLBACK_COMPLETED)
            result.phase_reached = StageRecoveryPhase.ROLLBACK_COMPLETED

            # ---- Step 4: Wait for replacement ----
            if wait_replacement_fn is not None:
                try:
                    wait_replacement_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                    )
                except Exception as e:
                    result.errors.append(f"wait_replacement failed: {e}")
            result.replacement_ready = True
            self._transition(StageRecoveryPhase.WAITING_FOR_REPLACEMENT)
            result.phase_reached = StageRecoveryPhase.WAITING_FOR_REPLACEMENT

            # ---- Step 5: Safe-point repair ----
            # 5a. Group rebuild
            if group_repair_fn is not None:
                try:
                    group_repair_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                        pp_group_ranks=pp_group_ranks,
                    )
                    result.groups_repaired = True
                except Exception as e:
                    result.errors.append(f"group_repair failed: {e}")

            # 5b. Topology refresh
            if topology_refresh_fn is not None:
                try:
                    topology_refresh_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                    )
                except Exception as e:
                    result.errors.append(f"topology_refresh failed: {e}")

            # 5c. Parameter recovery (path-dependent)
            if recovery_path == "CHECKPOINT_RESTART":
                if checkpoint_restart_fn is not None:
                    try:
                        checkpoint_restart_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                        )
                    except Exception as e:
                        result.errors.append(
                            f"checkpoint_restart failed: {e}"
                        )
            elif recovery_path == "FULL_PEER_RECOVERY":
                # Dense from DP peer + experts from Expert-DP peer
                if dense_sync_fn is not None:
                    try:
                        dense_sync_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                        )
                    except Exception as e:
                        result.errors.append(f"dense_sync failed: {e}")

                if expert_peer_sync_fn is not None:
                    try:
                        expert_peer_sync_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                            expert_ids=expert_ids,
                        )
                    except Exception as e:
                        result.errors.append(
                            f"expert_peer_sync failed: {e}"
                        )
                elif expert_restore_fn is not None:
                    # Fallback to checkpoint-based expert restore
                    try:
                        expert_restore_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                            expert_ids=expert_ids,
                        )
                    except Exception as e:
                        result.errors.append(
                            f"expert_restore (fallback) failed: {e}"
                        )
            else:  # HYBRID_RECOVERY
                if dense_sync_fn is not None:
                    try:
                        dense_sync_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                        )
                    except Exception as e:
                        result.errors.append(f"dense_sync failed: {e}")

                if expert_restore_fn is not None:
                    try:
                        expert_restore_fn(
                            failed_rank=failed_rank,
                            replacement_rank=replacement_rank,
                            step=step,
                            expert_ids=expert_ids,
                        )
                    except Exception as e:
                        result.errors.append(f"expert_restore failed: {e}")

            # 5d. Unified convergence
            if convergence_fn is not None:
                try:
                    convergence_fn(
                        path=recovery_path,
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                        expert_ids=expert_ids or [],
                    )
                except Exception as e:
                    result.errors.append(f"convergence failed: {e}")

            self._transition(StageRecoveryPhase.SAFE_POINT_REPAIR)
            result.phase_reached = StageRecoveryPhase.SAFE_POINT_REPAIR

            # ---- Step 6: P2P rebind ----
            if p2p_rebind_fn is not None:
                try:
                    p2p_rebind_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        pp_group_ranks=pp_group_ranks,
                        step=step,
                    )
                    result.p2p_rebound = True
                except Exception as e:
                    result.errors.append(f"p2p_rebind failed: {e}")
            else:
                result.p2p_rebound = True  # Lazy rebuild mode

            self._transition(StageRecoveryPhase.P2P_REBOUND)
            result.phase_reached = StageRecoveryPhase.P2P_REBOUND

            # ---- Step 7: Reintegration + resume ----
            if reintegration_fn is not None:
                try:
                    reintegration_fn(
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                        expert_ids=expert_ids or [],
                    )
                except Exception as e:
                    result.errors.append(f"reintegration failed: {e}")

            result.training_resumed = True
            self._transition(StageRecoveryPhase.TRAINING_RESUMED)
            result.phase_reached = StageRecoveryPhase.TRAINING_RESUMED

        except Exception as e:
            result.errors.append(f"protocol error: {e}")
            logger.error(
                "BSR-MoE stage-safe: protocol error at phase %s: %s",
                self._phase.name, e,
            )

        # Return to IDLE for next recovery
        self._phase = StageRecoveryPhase.IDLE

        result.elapsed_seconds = time.time() - t_start
        self._history.append(result)

        logger.warning(
            "[%s] BSR-MoE stage-safe: recovery %s "
            "(phase=%s, path=%s, errors=%d, elapsed=%.3fs, step=%d)",
            _ts(),
            "SUCCEEDED" if result.success else "FAILED",
            result.phase_reached.name,
            recovery_path,
            len(result.errors),
            result.elapsed_seconds,
            step,
        )

        return result

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[StageRecoveryResult]:
        return list(self._history)

    @property
    def total_recoveries(self) -> int:
        return len(self._history)

    @property
    def total_successes(self) -> int:
        return sum(1 for r in self._history if r.success)

    def summary(self) -> Dict:
        return {
            "phase": self._phase.name,
            "total_recoveries": self.total_recoveries,
            "total_successes": self.total_successes,
            "last_result": self._history[-1].to_dict() if self._history else None,
        }

    def reset(self) -> None:
        self._phase = StageRecoveryPhase.IDLE
        self._history.clear()


# =====================================================================
# Utility: compute new PP ranks after replacement
# =====================================================================

def compute_new_pp_ranks(
    pp_group_ranks: List[int],
    failed_rank: int,
    replacement_rank: int,
) -> List[int]:
    """Compute the new PP group ranks after replacing a failed rank.

    The replacement rank takes the same position (same logical stage)
    as the failed rank.

    Args:
        pp_group_ranks: Original PP group ranks, e.g. [0, 2, 4, 6].
        failed_rank: The rank that failed.
        replacement_rank: The replacement rank.

    Returns:
        New PP group ranks with failed_rank replaced.

    Raises:
        ValueError: If failed_rank is not in pp_group_ranks.
    """
    if failed_rank not in pp_group_ranks:
        raise ValueError(
            f"failed_rank {failed_rank} not in pp_group_ranks {pp_group_ranks}"
        )
    return [
        replacement_rank if r == failed_rank else r
        for r in pp_group_ranks
    ]


def identify_failed_stage(
    failed_rank: int,
    pp_group_ranks: List[int],
) -> int:
    """Identify which pipeline stage the failed rank belongs to.

    Args:
        failed_rank: The rank that failed.
        pp_group_ranks: PP group ranks, e.g. [0, 2, 4, 6].

    Returns:
        Stage index (0-based), or -1 if not found.
    """
    try:
        return pp_group_ranks.index(failed_rank)
    except ValueError:
        return -1


# =====================================================================
# Global singleton
# =====================================================================

_PROTOCOL: Optional[StageSafeRecoveryProtocol] = None


def get_stage_safe_recovery_protocol() -> StageSafeRecoveryProtocol:
    """Get or create the global StageSafeRecoveryProtocol singleton."""
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = StageSafeRecoveryProtocol()
    return _PROTOCOL


def clear_stage_safe_recovery_protocol() -> None:
    """Clear the global singleton (for testing)."""
    global _PROTOCOL
    _PROTOCOL = None


# =====================================================================
# Helpers
# =====================================================================

def _ts() -> str:
    return time.strftime("%H:%M:%S")
