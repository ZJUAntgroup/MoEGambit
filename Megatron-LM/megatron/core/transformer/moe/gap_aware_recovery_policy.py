# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BSR-MoE Gap-Aware Recovery Policy.

This module implements a **recovery path selector** that chooses between
checkpoint restart and hybrid recovery based on the "gap" between the
current training iteration and the latest available checkpoint.

Gap definition
--------------
::

    gap = current_iteration - latest_checkpoint_iteration

Decision logic (v1: simple threshold)
--------------------------------------
::

    if gap <= gap_threshold:
        -> CHECKPOINT_RESTART   (small gap: cheaper to restart from ckpt)
    else:
        -> HYBRID_RECOVERY      (large gap: online recovery is cheaper)

The threshold-based policy is the default.  A ``CostModelPolicy`` base
class is provided for future extensions (e.g., cost-model-based selection
that considers checkpoint I/O bandwidth, number of affected experts,
recomputation cost, etc.).

Integration
-----------
The policy is consulted by ``RecoveryController._execute_safe_point_repair``
when a hard failure is being repaired.  The selected path determines
whether the controller:

* **CHECKPOINT_RESTART**: signals the training loop to stop and reload
  from the latest checkpoint (full restart).
* **HYBRID_RECOVERY**: proceeds with the existing online repair sequence
  (group rebuild -> topology refresh -> dense sync -> expert restore).

Scope (v1)
----------
* Simple threshold policy
* Structured decision logging
* Extension point for cost-model policy
* Does NOT implement the actual checkpoint restart logic (that is in
  ``bsr_integration.py`` and ``training.py``)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Recovery path enum
# =====================================================================

class RecoveryPath(enum.Enum):
    """The two recovery paths available after a hard failure."""

    CHECKPOINT_RESTART = "checkpoint_restart"
    """Small gap: stop training and restart from the latest checkpoint.
    All ranks reload model + optimizer state from disk."""

    HYBRID_RECOVERY = "hybrid_recovery"
    """Large gap: online recovery without full restart.
    Dense/shared/router params synced from healthy DP peer;
    MoE expert weights restored from distributed checkpoint."""


# =====================================================================
# Recovery decision
# =====================================================================

@dataclass
class RecoveryDecision:
    """Immutable record of a recovery path decision.

    Attributes:
        path: The selected recovery path.
        current_iteration: Training iteration when the decision was made.
        checkpoint_iteration: Iteration of the latest available checkpoint.
        gap: ``current_iteration - checkpoint_iteration``.
        gap_threshold: The threshold used for the decision.
        reason: Human-readable explanation.
        timestamp: ISO-format timestamp of the decision.
        metadata: Arbitrary extra info (for cost-model extensions).
    """

    path: RecoveryPath
    current_iteration: int
    checkpoint_iteration: int
    gap: int
    gap_threshold: int
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict (for logging / manifest)."""
        return {
            "selected_path": self.path.value,
            "current_iteration": self.current_iteration,
            "checkpoint_iteration": self.checkpoint_iteration,
            "gap": self.gap,
            "gap_threshold": self.gap_threshold,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =====================================================================
# Base policy (extension point)
# =====================================================================

class RecoveryPolicyBase:
    """Abstract base for recovery path selection policies.

    Subclass this and override ``select_path`` to implement custom
    decision logic (e.g., cost-model-based).
    """

    def select_path(
        self,
        *,
        current_iteration: int,
        checkpoint_iteration: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Choose a recovery path.

        Args:
            current_iteration: Current training step.
            checkpoint_iteration: Step of the latest available checkpoint.
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            num_affected_experts: Number of experts on the failed rank.
            **kwargs: Reserved for future cost-model inputs.

        Returns:
            A ``RecoveryDecision`` describing the chosen path.
        """
        raise NotImplementedError


# =====================================================================
# Threshold policy (v1 default)
# =====================================================================

class ThresholdRecoveryPolicy(RecoveryPolicyBase):
    """Simple threshold-based recovery path selector.

    If ``gap <= gap_threshold``, select CHECKPOINT_RESTART.
    Otherwise, select HYBRID_RECOVERY.

    Args:
        gap_threshold: The gap threshold.  Default 100.
    """

    def __init__(self, gap_threshold: int = 100) -> None:
        if gap_threshold < 0:
            raise ValueError(
                f"gap_threshold must be >= 0, got {gap_threshold}"
            )
        self._gap_threshold = gap_threshold

    @property
    def gap_threshold(self) -> int:
        """The configured gap threshold."""
        return self._gap_threshold

    def select_path(
        self,
        *,
        current_iteration: int,
        checkpoint_iteration: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Select recovery path based on gap vs threshold.

        Returns:
            A ``RecoveryDecision`` with the selected path.
        """
        gap = current_iteration - checkpoint_iteration

        if checkpoint_iteration < 0:
            # No checkpoint available — must use hybrid recovery
            path = RecoveryPath.HYBRID_RECOVERY
            reason = (
                f"no checkpoint available (checkpoint_iteration={checkpoint_iteration}), "
                f"forced hybrid recovery"
            )
        elif gap <= self._gap_threshold:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = (
                f"gap={gap} <= threshold={self._gap_threshold}, "
                f"checkpoint restart is cheaper"
            )
        else:
            path = RecoveryPath.HYBRID_RECOVERY
            reason = (
                f"gap={gap} > threshold={self._gap_threshold}, "
                f"hybrid recovery avoids full restart"
            )

        decision = RecoveryDecision(
            path=path,
            current_iteration=current_iteration,
            checkpoint_iteration=checkpoint_iteration,
            gap=gap,
            gap_threshold=self._gap_threshold,
            reason=reason,
            metadata={
                "failed_rank": failed_rank,
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
                "policy": "threshold",
            },
        )

        # Structured decision log
        logger.warning(
            "BSR-MoE gap-aware recovery decision: "
            "path=%s | current_iteration=%d | checkpoint_iteration=%d | "
            "gap=%d | threshold=%d | reason=%s | "
            "failed_rank=%d | replacement_rank=%d",
            decision.path.value,
            decision.current_iteration,
            decision.checkpoint_iteration,
            decision.gap,
            decision.gap_threshold,
            decision.reason,
            failed_rank,
            replacement_rank,
        )

        return decision


# =====================================================================
# Rank-Exposure Guarded Hybrid Policy
# =====================================================================

@dataclass
class RankExposureGuardedConfig:
    """Configuration for the rank-exposure guarded hybrid recovery policy.

    This policy uses multi-boundary gap thresholds and rank stale exposure
    tracking to decide between checkpoint restart and hybrid recovery.

    Attributes:
        delta_time_min_gap: Gap below this → checkpoint restart (hybrid not
            cost-effective).
        max_single_gap: Gap above this → checkpoint restart (stale state too
            far behind).
        exposure_window_steps: Window (in training steps) for tracking rank
            stale exposure.
        max_rank_stale_exposure: Maximum stale exposure ratio per rank within
            the window (e.g. 0.02 = 2%).
        policy_margin: Hybrid must be at least this fraction faster than
            restart to be selected.  Default 0.10 (10%).
        fixed_gap_threshold: Baseline fixed-gap threshold for comparison
            experiments (e.g. FixedGapThreshold-32).
    """

    delta_time_min_gap: int = 32
    max_single_gap: int = 192
    exposure_window_steps: int = 20000
    max_rank_stale_exposure: float = 0.02
    policy_margin: float = 0.10
    fixed_gap_threshold: int = 32

    def validate(self) -> None:
        """Validate configuration constraints.  Raises ValueError on failure."""
        if self.delta_time_min_gap < 0:
            raise ValueError(
                f"delta_time_min_gap must be >= 0, got {self.delta_time_min_gap}"
            )
        if self.max_single_gap < self.delta_time_min_gap:
            raise ValueError(
                f"max_single_gap ({self.max_single_gap}) must be >= "
                f"delta_time_min_gap ({self.delta_time_min_gap})"
            )
        if self.exposure_window_steps <= 0:
            raise ValueError(
                f"exposure_window_steps must be > 0, got {self.exposure_window_steps}"
            )
        if not (0 < self.max_rank_stale_exposure <= 1):
            raise ValueError(
                f"max_rank_stale_exposure must be in (0, 1], "
                f"got {self.max_rank_stale_exposure}"
            )
        if self.policy_margin < 0:
            raise ValueError(
                f"policy_margin must be >= 0, got {self.policy_margin}"
            )
        if self.fixed_gap_threshold < 0:
            raise ValueError(
                f"fixed_gap_threshold must be >= 0, got {self.fixed_gap_threshold}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "delta_time_min_gap": self.delta_time_min_gap,
            "max_single_gap": self.max_single_gap,
            "exposure_window_steps": self.exposure_window_steps,
            "max_rank_stale_exposure": self.max_rank_stale_exposure,
            "policy_margin": self.policy_margin,
            "fixed_gap_threshold": self.fixed_gap_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RankExposureGuardedConfig":
        """Create a config from a dict (ignoring unknown keys)."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known_keys})


class RankExposureGuardedHybridPolicy(RecoveryPolicyBase):
    """Rank-exposure guarded hybrid recovery policy.

    Decision logic::

        if no checkpoint available:
            → HYBRID_RECOVERY  (no choice)
        elif gap < delta_time_min_gap:
            → CHECKPOINT_RESTART  (hybrid too expensive for tiny gap)
        elif gap > max_single_gap:
            → CHECKPOINT_RESTART  (stale state too far behind)
        elif rank_stale_exposure_in_window > max_rank_stale_exposure:
            → CHECKPOINT_RESTART  (rank has been stale too often)
        else:
            → HYBRID_RECOVERY  (hybrid is safe and cost-effective)

    The ``rank_stale_exposure`` check uses a ``RankExposureTracker`` to
    track cumulative stale iterations per rank within a sliding window.
    Only hybrid recovery events are recorded; checkpoint restarts are
    not because they reload all ranks uniformly.

    Thread safety: This class is **not** thread-safe.  It is designed
    for the RecoveryController's single-threaded event loop.

    Args:
        config: A ``RankExposureGuardedConfig`` with all parameters.
        tracker: Optional ``RankExposureTracker`` instance.  If None,
            the global singleton is used.
    """

    def __init__(
        self,
        config: Optional[RankExposureGuardedConfig] = None,
        tracker: Any = None,  # RankExposureTracker, import-avoided
    ) -> None:
        self._config = config or RankExposureGuardedConfig()
        self._config.validate()

        # Use provided tracker or fall back to global singleton
        if tracker is not None:
            self._tracker = tracker
        else:
            from megatron.core.transformer.moe.rank_exposure_tracker import (
                get_rank_exposure_tracker,
            )
            self._tracker = get_rank_exposure_tracker()

        logger.warning(
            "BSR-MoE RankExposureGuardedHybridPolicy initialized: %s",
            self._config.to_dict(),
        )

    @property
    def config(self) -> RankExposureGuardedConfig:
        """The policy configuration."""
        return self._config

    @property
    def gap_threshold(self) -> int:
        """Backward-compatible: returns fixed_gap_threshold."""
        return self._config.fixed_gap_threshold

    # ------------------------------------------------------------------
    # Rank stale exposure (delegates to RankExposureTracker)
    # ------------------------------------------------------------------

    def get_rank_stale_exposure(self, rank: int, current_step: int) -> float:
        """Compute the stale exposure ratio for a rank within the window.

        Returns ``stale_iters / exposure_window_steps``.
        """
        return self._tracker.get_rank_stale_exposure(
            rank=rank,
            current_step=current_step,
            window_steps=self._config.exposure_window_steps,
        )

    def get_all_rank_exposures(self, current_step: int) -> Dict[int, float]:
        """Get stale exposure ratios for all ranks."""
        return self._tracker.get_all_rank_exposures(
            current_step=current_step,
            window_steps=self._config.exposure_window_steps,
        )

    # ------------------------------------------------------------------
    # Policy selection
    # ------------------------------------------------------------------

    def select_path(
        self,
        *,
        current_iteration: int,
        checkpoint_iteration: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Select recovery path based on gap boundaries and rank exposure."""
        cfg = self._config
        gap = current_iteration - checkpoint_iteration

        # --- Decision logic ---
        if checkpoint_iteration < 0:
            # No checkpoint available
            path = RecoveryPath.HYBRID_RECOVERY
            reason = (
                f"no checkpoint available (checkpoint_iteration={checkpoint_iteration}), "
                f"forced hybrid recovery"
            )
            gap_threshold_used = -1

        elif gap < cfg.delta_time_min_gap:
            # Gap too small: hybrid overhead not worthwhile
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = (
                f"gap={gap} < delta_time_min_gap={cfg.delta_time_min_gap}, "
                f"checkpoint restart (hybrid too expensive for small gap)"
            )
            gap_threshold_used = cfg.delta_time_min_gap

        elif gap > cfg.max_single_gap:
            # Gap too large: stale state too far behind
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = (
                f"gap={gap} > max_single_gap={cfg.max_single_gap}, "
                f"checkpoint restart (stale state too far behind)"
            )
            gap_threshold_used = cfg.max_single_gap

        else:
            # Gap is in [delta_time_min_gap, max_single_gap]
            # Check rank stale exposure
            exposure = self.get_rank_stale_exposure(
                failed_rank, current_iteration,
            )
            if exposure > cfg.max_rank_stale_exposure:
                path = RecoveryPath.CHECKPOINT_RESTART
                reason = (
                    f"gap={gap} in [{cfg.delta_time_min_gap}, {cfg.max_single_gap}], "
                    f"but rank {failed_rank} stale exposure={exposure:.6f} > "
                    f"max={cfg.max_rank_stale_exposure}, "
                    f"checkpoint restart (rank too frequently stale)"
                )
                gap_threshold_used = cfg.delta_time_min_gap
            else:
                path = RecoveryPath.HYBRID_RECOVERY
                reason = (
                    f"gap={gap} in [{cfg.delta_time_min_gap}, {cfg.max_single_gap}], "
                    f"rank {failed_rank} stale exposure={exposure:.6f} <= "
                    f"max={cfg.max_rank_stale_exposure}, "
                    f"hybrid recovery (cost-effective and safe)"
                )
                gap_threshold_used = cfg.delta_time_min_gap

        # Record hybrid recovery events for exposure tracking.
        # Only hybrid recovery produces stale iterations; checkpoint
        # restart reloads all ranks uniformly so no rank is "stale".
        if path == RecoveryPath.HYBRID_RECOVERY and failed_rank >= 0:
            self._tracker.record_hybrid_recovery(
                step=current_iteration,
                rank=failed_rank,
                gap=gap,
            )

        decision = RecoveryDecision(
            path=path,
            current_iteration=current_iteration,
            checkpoint_iteration=checkpoint_iteration,
            gap=gap,
            gap_threshold=gap_threshold_used,
            reason=reason,
            metadata={
                "failed_rank": failed_rank,
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
                "policy": "rank_exposure_guarded_hybrid",
                "policy_config": cfg.to_dict(),
                "delta_time_min_gap": cfg.delta_time_min_gap,
                "max_single_gap": cfg.max_single_gap,
                "rank_stale_exposure": (
                    self.get_rank_stale_exposure(failed_rank, current_iteration)
                    if failed_rank >= 0 else 0.0
                ),
            },
        )

        # Structured decision log
        logger.warning(
            "BSR-MoE rank-exposure-guarded recovery decision: "
            "path=%s | iter=%d | ckpt_iter=%d | gap=%d | "
            "delta_time_min_gap=%d | max_single_gap=%d | "
            "reason=%s | failed_rank=%d | replacement_rank=%d",
            decision.path.value,
            decision.current_iteration,
            decision.checkpoint_iteration,
            decision.gap,
            cfg.delta_time_min_gap,
            cfg.max_single_gap,
            decision.reason,
            failed_rank,
            replacement_rank,
        )

        return decision


# =====================================================================
# Gap-Aware Recovery Policy Manager
# =====================================================================

class GapAwareRecoveryPolicyManager:
    """Manages the gap-aware recovery policy lifecycle.

    This is the main entry point used by ``RecoveryController``.  It:
    1. Holds a reference to the active policy (default: ThresholdRecoveryPolicy).
    2. Accepts a ``get_checkpoint_iteration_fn`` callback to query the
       latest checkpoint iteration at decision time.
    3. Records decision history for audit.

    Usage::

        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=my_fn,
        )
        decision = mgr.evaluate(current_iteration=500, failed_rank=0)
        if decision.path == RecoveryPath.CHECKPOINT_RESTART:
            ...  # signal training loop to restart
        else:
            ...  # proceed with hybrid recovery
    """

    def __init__(
        self,
        *,
        gap_threshold: int = 100,
        policy: Optional[RecoveryPolicyBase] = None,
        get_checkpoint_iteration_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        """Initialize the policy manager.

        Args:
            gap_threshold: Threshold for the default ThresholdRecoveryPolicy.
                Ignored if a custom ``policy`` is provided.
            policy: Optional custom policy.  If None, a
                ``ThresholdRecoveryPolicy(gap_threshold)`` is created.
            get_checkpoint_iteration_fn: Callback that returns the iteration
                of the latest available checkpoint.  If None, checkpoint
                iteration defaults to -1 (no checkpoint → forced hybrid).
        """
        if policy is not None:
            self._policy = policy
        else:
            self._policy = ThresholdRecoveryPolicy(gap_threshold=gap_threshold)

        self._get_checkpoint_iteration_fn = get_checkpoint_iteration_fn
        self._decision_history: list[RecoveryDecision] = []
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        """Whether gap-aware recovery is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def policy(self) -> RecoveryPolicyBase:
        """The active recovery policy."""
        return self._policy

    @policy.setter
    def policy(self, value: RecoveryPolicyBase) -> None:
        self._policy = value

    @property
    def decision_history(self) -> list[RecoveryDecision]:
        """List of all past decisions (for audit)."""
        return list(self._decision_history)

    @property
    def last_decision(self) -> Optional[RecoveryDecision]:
        """The most recent decision, or None."""
        return self._decision_history[-1] if self._decision_history else None

    def set_checkpoint_iteration_fn(
        self, fn: Callable[[], int],
    ) -> None:
        """Set or replace the checkpoint iteration query callback."""
        self._get_checkpoint_iteration_fn = fn

    def get_checkpoint_iteration(self) -> int:
        """Query the latest checkpoint iteration.

        Returns -1 if no callback is registered or the callback fails.
        """
        if self._get_checkpoint_iteration_fn is None:
            return -1
        try:
            return self._get_checkpoint_iteration_fn()
        except Exception as e:
            logger.warning(
                "BSR-MoE gap-aware: get_checkpoint_iteration_fn failed: %s", e,
            )
            return -1

    def evaluate(
        self,
        *,
        current_iteration: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Evaluate the recovery policy and return a decision.

        If gap-aware recovery is disabled, always returns HYBRID_RECOVERY
        (preserving the existing behavior).

        Args:
            current_iteration: Current training step.
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            num_affected_experts: Number of experts on the failed rank.

        Returns:
            A ``RecoveryDecision``.
        """
        if not self._enabled:
            decision = RecoveryDecision(
                path=RecoveryPath.HYBRID_RECOVERY,
                current_iteration=current_iteration,
                checkpoint_iteration=-1,
                gap=-1,
                gap_threshold=-1,
                reason="gap-aware recovery disabled, defaulting to hybrid",
                metadata={"policy": "disabled"},
            )
            logger.info(
                "BSR-MoE gap-aware recovery: DISABLED, using hybrid recovery"
            )
            self._decision_history.append(decision)
            return decision

        checkpoint_iteration = self.get_checkpoint_iteration()

        decision = self._policy.select_path(
            current_iteration=current_iteration,
            checkpoint_iteration=checkpoint_iteration,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            num_affected_experts=num_affected_experts,
            **kwargs,
        )

        self._decision_history.append(decision)
        return decision

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for monitoring / logging."""
        result: Dict[str, Any] = {
            "enabled": self._enabled,
            "policy_type": type(self._policy).__name__,
            "num_decisions": len(self._decision_history),
            "last_decision": (
                self.last_decision.to_dict()
                if self.last_decision is not None
                else None
            ),
        }
        if isinstance(self._policy, ThresholdRecoveryPolicy):
            result["gap_threshold"] = self._policy.gap_threshold
        elif isinstance(self._policy, RankExposureGuardedHybridPolicy):
            result["gap_threshold"] = self._policy.config.fixed_gap_threshold
            result["policy_config"] = self._policy.config.to_dict()
        return result

    def reset(self) -> None:
        """Clear decision history (for testing)."""
        self._decision_history.clear()


# =====================================================================
# Global singleton
# =====================================================================

_POLICY_MANAGER: Optional[GapAwareRecoveryPolicyManager] = None


def get_gap_aware_recovery_policy_manager() -> GapAwareRecoveryPolicyManager:
    """Get or create the global policy manager singleton."""
    global _POLICY_MANAGER
    if _POLICY_MANAGER is None:
        _POLICY_MANAGER = GapAwareRecoveryPolicyManager()
    return _POLICY_MANAGER


def initialize_gap_aware_recovery_policy(
    *,
    gap_threshold: int = 100,
    policy: Optional[RecoveryPolicyBase] = None,
    policy_type: str = "threshold",
    rank_exposure_config: Optional[RankExposureGuardedConfig] = None,
    get_checkpoint_iteration_fn: Optional[Callable[[], int]] = None,
    enabled: bool = True,
) -> GapAwareRecoveryPolicyManager:
    """Initialize the global policy manager with configuration.

    This should be called once during BSR-MoE initialization
    (``maybe_initialize_bsr_moe``).

    Args:
        gap_threshold: Gap threshold for the default threshold policy.
            Also used as ``fixed_gap_threshold`` for the rank-exposure
            guarded policy.  Ignored if a custom ``policy`` is provided.
        policy: Optional custom policy override.  If provided, all
            other policy-creation arguments are ignored.
        policy_type: Policy type to create.  One of:
            - ``"threshold"``: simple threshold policy (default).
            - ``"rank_exposure_guarded_hybrid"``: multi-boundary policy
              with rank stale exposure tracking.
        rank_exposure_config: Optional ``RankExposureGuardedConfig`` for
            the rank-exposure guarded policy.  If None and
            ``policy_type="rank_exposure_guarded_hybrid"``, a default
            config is created (using ``gap_threshold`` as
            ``fixed_gap_threshold``).
        get_checkpoint_iteration_fn: Callback to query checkpoint iteration.
        enabled: Whether gap-aware recovery is enabled.

    Returns:
        The initialized ``GapAwareRecoveryPolicyManager``.
    """
    global _POLICY_MANAGER

    # Build the policy instance if not explicitly provided
    if policy is None:
        if policy_type == "rank_exposure_guarded_hybrid":
            cfg = rank_exposure_config
            if cfg is None:
                cfg = RankExposureGuardedConfig(
                    fixed_gap_threshold=gap_threshold,
                )
            cfg.validate()
            policy = RankExposureGuardedHybridPolicy(config=cfg)
            logger.warning(
                "BSR-MoE gap-aware recovery policy: "
                "using RankExposureGuardedHybridPolicy "
                "(delta_time_min_gap=%d, max_single_gap=%d, "
                "exposure_window_steps=%d, max_rank_stale_exposure=%.4f, "
                "policy_margin=%.4f, fixed_gap_threshold=%d)",
                cfg.delta_time_min_gap, cfg.max_single_gap,
                cfg.exposure_window_steps, cfg.max_rank_stale_exposure,
                cfg.policy_margin, cfg.fixed_gap_threshold,
            )
        else:
            # Default: threshold policy (backward compatible)
            policy = ThresholdRecoveryPolicy(gap_threshold=gap_threshold)

    _POLICY_MANAGER = GapAwareRecoveryPolicyManager(
        gap_threshold=gap_threshold,
        policy=policy,
        get_checkpoint_iteration_fn=get_checkpoint_iteration_fn,
    )
    _POLICY_MANAGER.enabled = enabled
    logger.warning(
        "BSR-MoE gap-aware recovery policy initialized: "
        "enabled=%s, threshold=%d, policy=%s",
        enabled, gap_threshold, type(_POLICY_MANAGER.policy).__name__,
    )
    return _POLICY_MANAGER


def clear_gap_aware_recovery_policy() -> None:
    """Reset the global policy manager (for testing)."""
    global _POLICY_MANAGER
    if _POLICY_MANAGER is not None:
        _POLICY_MANAGER.reset()
    _POLICY_MANAGER = None
