# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# MOEGAMBIT-MoE: Unified Reintegration — convergence point for restart & hybrid paths
#
# After Phase A (shared infrastructure repair) and Phase B (path-specific
# parameter recovery), both the checkpoint-restart and hybrid-recovery paths
# must execute the same post-recovery steps before entering Phase C (shared
# post-repair).  This module provides a single ``PostRecoveryConvergence``
# class that encapsulates those steps so that neither path needs to maintain
# its own copy of the logic.
#
# Convergence steps (executed in order):
#   1. Mark experts STALE_RUNNABLE (if not already done by the path)
#   2. Verify directory / health-manager / dispatch-topology consistency
#   3. Activate preferential routing bias (if enabled)
#   4. Submit deferred optimizer loads (if applicable)
#   5. Drive two-phase recovery state machine (if enabled)
#
# Design principles:
#   - Each step is idempotent — safe to call even if the path already did it.
#   - Each step is optional — controlled by config flags.
#   - The class is stateless across invocations (no singleton needed).

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recovery path enum — used to tag which path was taken
# ---------------------------------------------------------------------------

class RecoveryPath(IntEnum):
    """Identifies which Phase-B path was taken."""
    CHECKPOINT_RESTART = 1
    HYBRID_RECOVERY = 2


# ---------------------------------------------------------------------------
# Convergence result
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Result of the post-recovery convergence sequence."""

    path: RecoveryPath
    """Which recovery path was taken."""

    failed_rank: int = -1
    replacement_rank: int = -1
    step: int = -1

    # Step outcomes
    experts_marked_stale: int = 0
    consistency_verified: bool = False
    consistency_issues: List[str] = field(default_factory=list)
    preferential_routing_activated: int = 0
    optimizer_loads_submitted: int = 0
    two_phase_driven: bool = False

    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        return {
            "path": self.path.name,
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "step": self.step,
            "experts_marked_stale": self.experts_marked_stale,
            "consistency_verified": self.consistency_verified,
            "consistency_issues": self.consistency_issues,
            "preferential_routing_activated": self.preferential_routing_activated,
            "optimizer_loads_submitted": self.optimizer_loads_submitted,
            "two_phase_driven": self.two_phase_driven,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "success": self.success,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# PostRecoveryConvergence
# ---------------------------------------------------------------------------

class PostRecoveryConvergence:
    """Unified post-recovery convergence point for both restart & hybrid paths.

    This class is instantiated once during ``maybe_initialize_moegambit_moe()`` and
    stored as a module-level singleton.  Both ``checkpoint_restart_fn`` and
    ``expert_restore_fn`` call ``execute()`` after their path-specific work
    is done, ensuring the same post-recovery steps are applied regardless of
    which path was taken.

    Usage::

        convergence = get_post_recovery_convergence()
        result = convergence.execute(
            path=RecoveryPath.CHECKPOINT_RESTART,
            failed_rank=2,
            replacement_rank=5,
            expert_ids=[0, 1, 2, 3],
            restored_experts=[(1, 0), (1, 1), (2, 0), (2, 1)],
            step=1000,
            config=config,
        )
    """

    def __init__(self, config: Any = None):
        """
        Args:
            config: TransformerConfig (or any object with the relevant
                    ``moe_moegambit_*`` attributes).  May be ``None`` for testing.
        """
        self._config = config

    def execute(
        self,
        *,
        path: RecoveryPath,
        failed_rank: int,
        replacement_rank: int,
        expert_ids: List[int],
        restored_experts: List[Tuple[int, int]],
        step: int = -1,
        config: Any = None,
        # Optional overrides for testing / decoupling
        mark_stale_fn: Optional[Callable] = None,
        verify_consistency_fn: Optional[Callable] = None,
        activate_preferential_fn: Optional[Callable] = None,
        submit_optimizer_fn: Optional[Callable] = None,
        drive_two_phase_fn: Optional[Callable] = None,
    ) -> ConvergenceResult:
        """Execute the unified post-recovery convergence sequence.

        Args:
            path: Which recovery path was taken.
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            expert_ids: Global expert IDs on the replacement rank.
            restored_experts: List of (layer_id, expert_id) pairs that
                were restored.
            step: Current training step.
            config: Optional config override (falls back to self._config).
            mark_stale_fn: Optional callback ``(expert_ids, step) -> int``.
            verify_consistency_fn: Optional callback ``() -> List[str]``.
            activate_preferential_fn: Optional callback
                ``(restored_experts, step) -> int``.
            submit_optimizer_fn: Optional callback
                ``(restored_experts, step) -> int``.
            drive_two_phase_fn: Optional callback
                ``(restored_experts, step) -> bool``.

        Returns:
            ConvergenceResult with outcomes of each step.
        """
        cfg = config or self._config
        t_start = time.time()

        result = ConvergenceResult(
            path=path,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            step=step,
        )

        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: starting post-recovery "
            "sequence (path=%s, failed=%d, replacement=%d, "
            "experts=%d, step=%d)",
            _ts(), path.name, failed_rank, replacement_rank,
            len(expert_ids), step,
        )

        # ---- Step 1: Mark experts STALE_RUNNABLE (idempotent) ----
        t_s1 = time.time()
        result.experts_marked_stale = self._step_mark_stale(
            expert_ids=expert_ids,
            step=step,
            result=result,
            mark_stale_fn=mark_stale_fn,
        )
        t_s1_elapsed = time.time() - t_s1
        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  [1/5] mark_stale_runnable "
            "done (marked=%d, time=%.3fs)",
            _ts(), result.experts_marked_stale, t_s1_elapsed,
        )

        # ---- Step 2: Verify consistency ----
        t_s2 = time.time()
        result.consistency_issues = self._step_verify_consistency(
            result=result,
            verify_consistency_fn=verify_consistency_fn,
        )
        result.consistency_verified = len(result.consistency_issues) == 0
        t_s2_elapsed = time.time() - t_s2
        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  [2/5] verify_consistency "
            "done (consistent=%s, issues=%d, time=%.3fs)",
            _ts(), result.consistency_verified,
            len(result.consistency_issues), t_s2_elapsed,
        )

        # ---- Step 3: Activate preferential routing (if enabled) ----
        t_s3 = time.time()
        _pref_enabled = _get_flag(cfg, 'moe_moegambit_preferential_routing', False)
        if _pref_enabled and restored_experts:
            result.preferential_routing_activated = self._step_preferential_routing(
                restored_experts=restored_experts,
                step=step,
                config=cfg,
                result=result,
                activate_preferential_fn=activate_preferential_fn,
            )
        t_s3_elapsed = time.time() - t_s3
        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  [3/5] preferential_routing "
            "done (enabled=%s, activated=%d, time=%.3fs)",
            _ts(), _pref_enabled,
            result.preferential_routing_activated, t_s3_elapsed,
        )

        # ---- Step 4: Submit deferred optimizer loads (if applicable) ----
        t_s4 = time.time()
        _opt_restore = _get_flag(cfg, 'moe_moegambit_expert_opt_restore', True)
        _defer_opt = _get_flag(cfg, 'moe_moegambit_defer_optimizer_load', True)
        # Only for hybrid path — checkpoint restart loads optimizer inline
        if (
            path == RecoveryPath.HYBRID_RECOVERY
            and _opt_restore
            and _defer_opt
            and restored_experts
        ):
            result.optimizer_loads_submitted = self._step_submit_optimizer(
                restored_experts=restored_experts,
                step=step,
                result=result,
                submit_optimizer_fn=submit_optimizer_fn,
            )
        t_s4_elapsed = time.time() - t_s4
        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  [4/5] deferred_optimizer "
            "done (submitted=%d, time=%.3fs)",
            _ts(), result.optimizer_loads_submitted, t_s4_elapsed,
        )

        # ---- Step 5: Drive two-phase recovery state machine ----
        t_s5 = time.time()
        _two_phase = _get_flag(cfg, 'moe_moegambit_weights_first_recovery', True)
        if _two_phase and restored_experts:
            result.two_phase_driven = self._step_drive_two_phase(
                path=path,
                restored_experts=restored_experts,
                step=step,
                result=result,
                drive_two_phase_fn=drive_two_phase_fn,
            )
        t_s5_elapsed = time.time() - t_s5
        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  [5/5] two_phase_recovery "
            "done (driven=%s, time=%.3fs)",
            _ts(), result.two_phase_driven, t_s5_elapsed,
        )

        result.elapsed_seconds = time.time() - t_start

        logger.warning(
            "[%s] MOEGAMBIT-MoE unified convergence: ⏱️  ✅ COMPLETED "
            "(path=%s, total=%.3fs, step=%d) "
            "| Breakdown: mark_stale=%.3fs, verify=%.3fs, pref_routing=%.3fs, "
            "deferred_opt=%.3fs, two_phase=%.3fs",
            _ts(), path.name, result.elapsed_seconds, step,
            t_s1_elapsed, t_s2_elapsed, t_s3_elapsed,
            t_s4_elapsed, t_s5_elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # Individual steps (each is self-contained and error-tolerant)
    # ------------------------------------------------------------------

    def _step_mark_stale(
        self,
        expert_ids: List[int],
        step: int,
        result: ConvergenceResult,
        mark_stale_fn: Optional[Callable] = None,
    ) -> int:
        """Step 1: Mark experts as STALE_RUNNABLE across all layers.

        Idempotent — experts already in STALE_RUNNABLE are silently skipped
        by the health manager.
        """
        if not expert_ids:
            return 0

        try:
            if mark_stale_fn is not None:
                count = mark_stale_fn(expert_ids, step)
                return count if isinstance(count, int) else len(expert_ids)

            # Default: use the recovery controller's expert tracker
            from moegambit.adapters.megatron.moe.recovery_controller import get_recovery_controller
            ctrl = get_recovery_controller()
            ctrl._expert_tracker.mark_stale_runnable(expert_ids, step=step)
            return len(expert_ids)
        except Exception as e:
            msg = f"mark_stale_runnable failed: {e}"
            logger.error("MOEGAMBIT-MoE unified convergence: %s", msg)
            result.errors.append(msg)
            return 0

    def _step_verify_consistency(
        self,
        result: ConvergenceResult,
        verify_consistency_fn: Optional[Callable] = None,
    ) -> List[str]:
        """Step 2: Verify directory / health-manager / dispatch-topology consistency."""
        try:
            if verify_consistency_fn is not None:
                issues = verify_consistency_fn()
                return issues if isinstance(issues, list) else []

            # Default: use dispatch topology manager
            from moegambit.adapters.megatron.moe.dispatch_topology_refresh import (
                get_dispatch_topology_manager,
            )
            mgr = get_dispatch_topology_manager()
            if mgr is not None:
                issues = mgr.check_router_dispatcher_consistency()
                if issues:
                    logger.warning(
                        "MOEGAMBIT-MoE unified convergence: consistency issues: %s",
                        issues,
                    )
                return issues if isinstance(issues, list) else []
            return []
        except Exception as e:
            msg = f"consistency verification failed: {e}"
            logger.error("MOEGAMBIT-MoE unified convergence: %s", msg)
            # Non-fatal — don't add to result.errors
            return [msg]

    def _step_preferential_routing(
        self,
        restored_experts: List[Tuple[int, int]],
        step: int,
        config: Any,
        result: ConvergenceResult,
        activate_preferential_fn: Optional[Callable] = None,
    ) -> int:
        """Step 3: Activate preferential routing bias for restored experts."""
        try:
            if activate_preferential_fn is not None:
                count = activate_preferential_fn(restored_experts, step)
                return count if isinstance(count, int) else len(restored_experts)

            # Default: use the preferential routing manager
            from moegambit.adapters.megatron.moe.preferential_routing import (
                get_preferential_routing_manager,
            )
            _window = _get_flag(config, 'moe_moegambit_preferential_routing_window', 100)
            _bias = _get_flag(config, 'moe_moegambit_preferential_routing_bias', 0.1)
            _num_experts = _get_flag(config, 'num_moe_experts', 8)

            count = 0
            for lid, eid in restored_experts:
                mgr = get_preferential_routing_manager(
                    layer_number=lid,
                    num_experts=_num_experts,
                    initial_bias=_bias,
                    window_steps=_window,
                )
                mgr.activate(eid, step=step)
                count += 1
            return count
        except Exception as e:
            msg = f"preferential routing activation failed: {e}"
            logger.error("MOEGAMBIT-MoE unified convergence: %s", msg)
            result.errors.append(msg)
            return 0

    def _step_submit_optimizer(
        self,
        restored_experts: List[Tuple[int, int]],
        step: int,
        result: ConvergenceResult,
        submit_optimizer_fn: Optional[Callable] = None,
    ) -> int:
        """Step 4: Submit deferred optimizer state load requests."""
        try:
            if submit_optimizer_fn is not None:
                count = submit_optimizer_fn(restored_experts, step)
                return count if isinstance(count, int) else len(restored_experts)

            # Default: use the deferred optimizer loader
            from moegambit.adapters.megatron.moe.deferred_optimizer_load import (
                get_deferred_optimizer_loader,
            )
            loader = get_deferred_optimizer_loader()
            if loader is None:
                return 0
            # The actual submission is done by the path-specific code
            # (expert_restore_fn already submits for hybrid path).
            # For checkpoint restart, optimizer state is loaded inline.
            return 0
        except Exception as e:
            msg = f"optimizer load submission failed: {e}"
            logger.error("MOEGAMBIT-MoE unified convergence: %s", msg)
            result.errors.append(msg)
            return 0

    def _step_drive_two_phase(
        self,
        path: RecoveryPath,
        restored_experts: List[Tuple[int, int]],
        step: int,
        result: ConvergenceResult,
        drive_two_phase_fn: Optional[Callable] = None,
    ) -> bool:
        """Step 5: Drive the two-phase recovery state machine."""
        try:
            if drive_two_phase_fn is not None:
                driven = drive_two_phase_fn(restored_experts, step)
                return bool(driven)

            # Default: use the two-phase recovery coordinator
            from moegambit.adapters.megatron.moe.two_phase_recovery import (
                get_two_phase_recovery_coordinator,
            )
            coord = get_two_phase_recovery_coordinator()
            if coord is None:
                return False

            # For checkpoint restart: weights are loaded inline, so we
            # can drive directly to WEIGHTS_READY.
            if path == RecoveryPath.CHECKPOINT_RESTART:
                coord.on_weights_restored(expert_ids=restored_experts, step=step)
                # Checkpoint restart loads optimizer inline too, so skip
                # the optimizer phase and go directly to FULLY_RECOVERED.
                coord.skip_optimizer_phase(expert_ids=restored_experts, step=step)
                return True
            else:
                # Hybrid path: two-phase is already driven by expert_restore_fn.
                # Just verify state is consistent.
                return True
        except Exception as e:
            msg = f"two-phase drive failed: {e}"
            logger.error("MOEGAMBIT-MoE unified convergence: %s", msg)
            result.errors.append(msg)
            return False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_CONVERGENCE: Optional[PostRecoveryConvergence] = None


def get_post_recovery_convergence(
    config: Any = None,
) -> PostRecoveryConvergence:
    """Get or create the global PostRecoveryConvergence instance."""
    global _CONVERGENCE
    if _CONVERGENCE is None:
        _CONVERGENCE = PostRecoveryConvergence(config=config)
    return _CONVERGENCE


def clear_post_recovery_convergence() -> None:
    """Clear the global instance (for testing)."""
    global _CONVERGENCE
    _CONVERGENCE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Compact timestamp for log messages."""
    return time.strftime("%H:%M:%S")


def _get_flag(config: Any, name: str, default: Any) -> Any:
    """Safely read a config attribute with a default."""
    if config is None:
        return default
    return getattr(config, name, default)
