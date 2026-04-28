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
from typing import Any, Callable, Dict, Optional

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
        return {
            "enabled": self._enabled,
            "policy_type": type(self._policy).__name__,
            "gap_threshold": (
                self._policy.gap_threshold
                if isinstance(self._policy, ThresholdRecoveryPolicy)
                else None
            ),
            "num_decisions": len(self._decision_history),
            "last_decision": (
                self.last_decision.to_dict()
                if self.last_decision is not None
                else None
            ),
        }

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
    get_checkpoint_iteration_fn: Optional[Callable[[], int]] = None,
    enabled: bool = True,
) -> GapAwareRecoveryPolicyManager:
    """Initialize the global policy manager with configuration.

    This should be called once during BSR-MoE initialization
    (``maybe_initialize_bsr_moe``).

    Args:
        gap_threshold: Gap threshold for the default threshold policy.
        policy: Optional custom policy override.
        get_checkpoint_iteration_fn: Callback to query checkpoint iteration.
        enabled: Whether gap-aware recovery is enabled.

    Returns:
        The initialized ``GapAwareRecoveryPolicyManager``.
    """
    global _POLICY_MANAGER
    _POLICY_MANAGER = GapAwareRecoveryPolicyManager(
        gap_threshold=gap_threshold,
        policy=policy,
        get_checkpoint_iteration_fn=get_checkpoint_iteration_fn,
    )
    _POLICY_MANAGER.enabled = enabled
    logger.info(
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
