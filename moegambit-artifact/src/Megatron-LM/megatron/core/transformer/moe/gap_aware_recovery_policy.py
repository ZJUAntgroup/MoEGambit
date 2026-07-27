# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MOEGAMBIT-MoE Recovery Policy Framework.

This module implements a **recovery path selector** that chooses between
checkpoint restart and hybrid recovery based on policy-specific criteria.

Supported policies
------------------
1. ``RestartAndSparePolicy``  — always CHECKPOINT_RESTART (baseline).
2. ``AlwaysHybridPolicy``     — always HYBRID_RECOVERY (baseline).
3. ``FixedGapThresholdPolicy`` — single gap threshold (baseline).
4. ``TwoThresholdPolicy``     — gap lower bound + upper bound (baseline).
5. ``RankExposureGuardedPolicy`` — gap bounds + rank stale exposure (main).

Decision input
--------------
::

    gap = current_step - latest_checkpoint_step
    rank = failed_logical_rank

Decision logic (RankExposureGuardedPolicy)
------------------------------------------
::

    rank_stale_iters_before =
        tracker.get_rank_stale_iters(rank, current_step, window_steps)
    rank_stale_iters_after = rank_stale_iters_before + gap
    rank_exposure_after = rank_stale_iters_after / window_steps

    if gap < delta_time_min_gap:
        path = checkpoint_restart
    elif gap > max_single_gap:
        path = checkpoint_restart
    elif rank_exposure_after > max_rank_stale_exposure:
        path = checkpoint_restart
    else:
        path = hybrid_recovery

Integration
-----------
The policy is consulted by ``RecoveryController._execute_safe_point_repair``
when a hard failure is being repaired.
"""

from __future__ import annotations

import abc
import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_DECISION_REASON_ALIASES = {
    "gap_below_threshold": "fixed_gap_restart",
    "gap_at_or_above_threshold": "fixed_gap_hybrid",
    "forced_checkpoint_restart": "forced_restart",
    "checkpoint_restart_forced": "forced_restart",
    "no_checkpoint_available": "forced_hybrid",
    "gap_aware_disabled": "forced_hybrid",
    "within_two_threshold_safe_region": "within_rank_exposure_safe_region",
}

_OPTIONAL_COST_FIELDS = (
    "estimated_restart_cost",
    "estimated_hybrid_cost",
    "policy_margin",
)


def _default_run_id() -> str:
    """Return a stable run id for structured logs."""
    for key in ("MOEGAMBIT_RUN_ID", "MEGATRON_RUN_ID", "RUN_ID"):
        value = os.environ.get(key)
        if value:
            return value
    return "unknown"


def _policy_type_from_policy(policy: Any) -> str:
    """Return a stable machine-readable policy type."""
    if isinstance(policy, RankExposureGuardedPolicy):
        return "rank_exposure_guarded"
    if isinstance(policy, FixedGapThresholdPolicy):
        return "fixed_gap_threshold"
    if isinstance(policy, TwoThresholdPolicy):
        return "two_threshold"
    if isinstance(policy, RestartAndSparePolicy):
        return "restart_and_spare"
    if isinstance(policy, AlwaysHybridPolicy):
        return "always_hybrid"
    return type(policy).__name__


def normalize_decision_reason(decision: "RecoveryDecision") -> str:
    """Normalize internal policy reasons to the experiment log enum."""
    reason = getattr(decision, "reason", "")
    if (
        reason == "no_checkpoint_available"
        and decision.path == RecoveryPath.CHECKPOINT_RESTART
    ):
        return "no_checkpoint"
    return _DECISION_REASON_ALIASES.get(reason, reason)


def build_recovery_path_chosen_log(
    decision: "RecoveryDecision",
    *,
    run_id: Optional[str] = None,
    policy_type: str = "",
) -> Dict[str, Any]:
    """Build the stable JSON payload for a recovery path decision."""
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    payload: Dict[str, Any] = {
        "event": "recovery_path_chosen",
        "run_id": run_id or _default_run_id(),
        "current_step": decision.current_step,
        "latest_checkpoint_step": decision.latest_checkpoint_step,
        "checkpoint_gap": decision.gap,
        "failed_rank": decision.failed_rank,
        "policy_type": policy_type or metadata.get("policy", "unknown"),
        "selected_path": decision.path.value,
        "decision_reason": normalize_decision_reason(decision),
        "delta_time_min_gap": decision.delta_time_min_gap,
        "max_single_gap": decision.max_single_gap,
        "exposure_window_steps": decision.exposure_window_steps,
        "max_rank_stale_exposure": decision.max_rank_stale_exposure,
        "rank_stale_iters_before": decision.rank_stale_iters_before,
        "rank_stale_iters_after": decision.rank_stale_iters_after,
        "rank_stale_exposure_before": decision.rank_stale_exposure_before,
        "rank_stale_exposure_after": decision.rank_stale_exposure_after,
        "num_affected_experts": decision.num_affected_experts,
        "num_experts": decision.num_experts,
        "window_expert_iteration_debt_before": (
            decision.window_expert_iteration_debt_before
        ),
        "window_expert_iteration_debt_after": (
            decision.window_expert_iteration_debt_after
        ),
        "expert_staleness_density_before": decision.expert_staleness_density_before,
        "expert_staleness_density_after": decision.expert_staleness_density_after,
        "max_expert_staleness_density": decision.max_expert_staleness_density,
    }

    for field_name in _OPTIONAL_COST_FIELDS:
        value = getattr(decision, field_name, None)
        if value is None:
            value = metadata.get(field_name)
        if value is not None:
            payload[field_name] = value

    return payload


def log_recovery_path_chosen_json(
    decision: "RecoveryDecision",
    *,
    run_id: Optional[str] = None,
    policy_type: str = "",
) -> Dict[str, Any]:
    """Emit a machine-parseable JSON log for the final path decision."""
    payload = build_recovery_path_chosen_log(
        decision,
        run_id=run_id,
        policy_type=policy_type,
    )
    logger.warning(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    return payload


# =====================================================================
# Recovery path enum
# =====================================================================

class RecoveryPath(enum.Enum):
    """The three recovery paths available after a hard failure."""

    CHECKPOINT_RESTART = "checkpoint_restart"
    """Stop training and restart from the latest checkpoint.
    All ranks reload model + optimizer state from disk."""

    HYBRID_RECOVERY = "hybrid_recovery"
    """Online recovery without full restart.
    Dense/shared/router params synced from healthy DP peer;
    MoE expert weights restored from distributed checkpoint."""

    FULL_PEER_RECOVERY = "full_peer_recovery"
    """Zero-disk, zero-staleness recovery when EDP > 1.
    Dense/shared/router params synced from healthy DP peer;
    MoE expert weights + optimizer state synced from healthy Expert-DP
    peer.  Because the expert state comes from a peer at step t (not
    from a checkpoint at step c), the recovered expert state has zero
    staleness and contributes nothing to Phi'(t).  No update barrier
    or deferred optimizer load is needed."""


# =====================================================================
# Structured decision
# =====================================================================

@dataclass
class RecoveryDecision:
    """Immutable record of a recovery path decision.

    Attributes:
        path: The selected recovery path.
        current_step: Training step when the decision was made.
        latest_checkpoint_step: Step of the latest available checkpoint.
        gap: ``current_step - latest_checkpoint_step``.
        failed_rank: The logical rank that failed.
        reason: Short machine-readable reason string.
        reason_detail: Human-readable explanation.
        timestamp: ISO-format timestamp of the decision.

        delta_time_min_gap: Gap lower bound (TwoThreshold / RankExposureGuarded).
        max_single_gap: Gap upper bound (TwoThreshold / RankExposureGuarded).
        exposure_window_steps: Window size for exposure tracking.
        rank_stale_iters_before: Stale iters for this rank *before* this recovery.
        rank_stale_iters_after: Stale iters for this rank *after* this recovery.
        rank_stale_exposure_before: Exposure ratio before.
        rank_stale_exposure_after: Exposure ratio after.
        max_rank_stale_exposure: Maximum allowed exposure ratio.

        metadata: Arbitrary extra info (for cost-model extensions).
    """

    # --- Core fields (always populated) ---
    path: RecoveryPath
    current_step: int
    latest_checkpoint_step: int
    gap: int
    failed_rank: int
    reason: str
    reason_detail: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # --- Policy-specific fields (populated by relevant policies) ---
    delta_time_min_gap: int = -1
    max_single_gap: int = -1
    exposure_window_steps: int = -1
    rank_stale_iters_before: int = 0
    rank_stale_iters_after: int = 0
    rank_stale_exposure_before: float = 0.0
    rank_stale_exposure_after: float = 0.0
    max_rank_stale_exposure: float = -1.0

    # --- Paper contract R2: Phi'(t) = (S(t) + |E_new| * Delta) / (N * W) ---
    num_affected_experts: int = 0
    num_experts: int = 0
    window_expert_iteration_debt_before: int = 0
    window_expert_iteration_debt_after: int = 0
    expert_staleness_density_before: float = 0.0
    expert_staleness_density_after: float = 0.0
    max_expert_staleness_density: float = -1.0

    # --- Arbitrary metadata ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict (for logging / manifest)."""
        return {
            "selected_path": self.path.value,
            "current_step": self.current_step,
            "latest_checkpoint_step": self.latest_checkpoint_step,
            "gap": self.gap,
            "failed_rank": self.failed_rank,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "timestamp": self.timestamp,
            "delta_time_min_gap": self.delta_time_min_gap,
            "max_single_gap": self.max_single_gap,
            "exposure_window_steps": self.exposure_window_steps,
            "rank_stale_iters_before": self.rank_stale_iters_before,
            "rank_stale_iters_after": self.rank_stale_iters_after,
            "rank_stale_exposure_before": self.rank_stale_exposure_before,
            "rank_stale_exposure_after": self.rank_stale_exposure_after,
            "max_rank_stale_exposure": self.max_rank_stale_exposure,
            "num_affected_experts": self.num_affected_experts,
            "num_experts": self.num_experts,
            "window_expert_iteration_debt_before": (
                self.window_expert_iteration_debt_before
            ),
            "window_expert_iteration_debt_after": (
                self.window_expert_iteration_debt_after
            ),
            "expert_staleness_density_before": self.expert_staleness_density_before,
            "expert_staleness_density_after": self.expert_staleness_density_after,
            "max_expert_staleness_density": self.max_expert_staleness_density,
            "metadata": self.metadata,
        }


# =====================================================================
# Abstract base policy
# =====================================================================

class RecoveryPolicyBase(abc.ABC):
    """Abstract base for recovery path selection policies.

    Subclass this and override ``choose`` to implement custom
    decision logic.
    """

    @abc.abstractmethod
    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Choose a recovery path.

        Args:
            current_step: Current training step.
            latest_checkpoint_step: Step of the latest available checkpoint.
            failed_rank: The logical rank that failed.
            replacement_rank: The replacement rank.
            num_affected_experts: Number of experts on the failed rank.

        Returns:
            A ``RecoveryDecision`` describing the chosen path.
        """
        ...

    # Backward compatibility alias
    def select_path(self, **kwargs: Any) -> RecoveryDecision:
        """Alias for ``choose`` (backward compatibility)."""
        return self.choose(**kwargs)


# =====================================================================
# Baseline 1: Restart+Spare (always checkpoint restart)
# =====================================================================

class RestartAndSparePolicy(RecoveryPolicyBase):
    """Always selects CHECKPOINT_RESTART.

    This is the most conservative baseline: on any hard failure,
    roll the entire training job back to the latest checkpoint.
    """

    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        gap = current_step - latest_checkpoint_step

        if latest_checkpoint_step < 0:
            # No checkpoint: forced hybrid recovery (cannot restart)
            return RecoveryDecision(
                path=RecoveryPath.HYBRID_RECOVERY,
                current_step=current_step,
                latest_checkpoint_step=latest_checkpoint_step,
                gap=gap,
                failed_rank=failed_rank,
                reason="no_checkpoint_available",
                reason_detail="no checkpoint available, forced hybrid recovery",
                metadata={
                    "policy": "restart_and_spare",
                    "replacement_rank": replacement_rank,
                    "num_affected_experts": num_affected_experts,
                },
            )

        return RecoveryDecision(
            path=RecoveryPath.CHECKPOINT_RESTART,
            current_step=current_step,
            latest_checkpoint_step=latest_checkpoint_step,
            gap=gap,
            failed_rank=failed_rank,
            reason="always_restart",
            reason_detail="restart_and_spare policy: always checkpoint restart",
            metadata={
                "policy": "restart_and_spare",
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
            },
        )


# =====================================================================
# Baseline 2: AlwaysHybrid (always hybrid recovery)
# =====================================================================

class AlwaysHybridPolicy(RecoveryPolicyBase):
    """Always selects HYBRID_RECOVERY.

    This is the most aggressive baseline: on any hard failure,
    attempt online recovery without full checkpoint restart.
    """

    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        gap = current_step - latest_checkpoint_step

        return RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_step=current_step,
            latest_checkpoint_step=latest_checkpoint_step,
            gap=gap,
            failed_rank=failed_rank,
            reason="always_hybrid",
            reason_detail="always_hybrid policy: always hybrid recovery",
            metadata={
                "policy": "always_hybrid",
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
            },
        )


# =====================================================================
# Baseline 3: FixedGapThreshold (single gap threshold)
# =====================================================================

class FixedGapThresholdPolicy(RecoveryPolicyBase):
    """Single fixed gap threshold policy.

    Decision logic::

        if gap < fixed_gap_threshold:  checkpoint_restart
        else:                          hybrid_recovery

    Args:
        fixed_gap_threshold: The gap threshold.  Default 32.
    """

    def __init__(self, fixed_gap_threshold: int = 32) -> None:
        if fixed_gap_threshold < 0:
            raise ValueError(
                f"fixed_gap_threshold must be >= 0, got {fixed_gap_threshold}"
            )
        self._fixed_gap_threshold = fixed_gap_threshold

    @property
    def gap_threshold(self) -> int:
        """Backward-compatible property."""
        return self._fixed_gap_threshold

    @property
    def fixed_gap_threshold(self) -> int:
        return self._fixed_gap_threshold

    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        gap = current_step - latest_checkpoint_step

        if latest_checkpoint_step < 0:
            path = RecoveryPath.HYBRID_RECOVERY
            reason = "no_checkpoint_available"
            reason_detail = "no checkpoint available, forced hybrid recovery"
        elif gap < self._fixed_gap_threshold:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "gap_below_threshold"
            reason_detail = (
                f"gap={gap} < threshold={self._fixed_gap_threshold}, "
                f"checkpoint restart"
            )
        else:
            path = RecoveryPath.HYBRID_RECOVERY
            reason = "gap_at_or_above_threshold"
            reason_detail = (
                f"gap={gap} >= threshold={self._fixed_gap_threshold}, "
                f"hybrid recovery"
            )

        return RecoveryDecision(
            path=path,
            current_step=current_step,
            latest_checkpoint_step=latest_checkpoint_step,
            gap=gap,
            failed_rank=failed_rank,
            reason=reason,
            reason_detail=reason_detail,
            delta_time_min_gap=self._fixed_gap_threshold,
            metadata={
                "policy": "fixed_gap_threshold",
                "fixed_gap_threshold": self._fixed_gap_threshold,
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
            },
        )


# =====================================================================
# Baseline 4: TwoThreshold (gap lower + upper bound)
# =====================================================================

class TwoThresholdPolicy(RecoveryPolicyBase):
    """Two-threshold policy with gap lower and upper bounds.

    Decision logic::

        if gap < delta_time_min_gap:   checkpoint_restart
        elif gap > max_single_gap:     checkpoint_restart
        else:                          hybrid_recovery

    Args:
        delta_time_min_gap: Gap below this → checkpoint restart.
        max_single_gap: Gap above this → checkpoint restart.
    """

    def __init__(
        self,
        delta_time_min_gap: int = 32,
        max_single_gap: int = 192,
    ) -> None:
        if delta_time_min_gap < 0:
            raise ValueError(
                f"delta_time_min_gap must be >= 0, got {delta_time_min_gap}"
            )
        if max_single_gap < delta_time_min_gap:
            raise ValueError(
                f"max_single_gap ({max_single_gap}) must be >= "
                f"delta_time_min_gap ({delta_time_min_gap})"
            )
        self._delta_time_min_gap = delta_time_min_gap
        self._max_single_gap = max_single_gap

    @property
    def delta_time_min_gap(self) -> int:
        return self._delta_time_min_gap

    @property
    def max_single_gap(self) -> int:
        return self._max_single_gap

    @property
    def gap_threshold(self) -> int:
        """Backward-compatible: returns delta_time_min_gap."""
        return self._delta_time_min_gap

    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        gap = current_step - latest_checkpoint_step

        if latest_checkpoint_step < 0:
            path = RecoveryPath.HYBRID_RECOVERY
            reason = "no_checkpoint_available"
            reason_detail = "no checkpoint available, forced hybrid recovery"
        elif gap < self._delta_time_min_gap:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "gap_below_time_threshold"
            reason_detail = (
                f"gap={gap} < delta_time_min_gap={self._delta_time_min_gap}, "
                f"checkpoint restart (hybrid too expensive for tiny gap)"
            )
        elif gap > self._max_single_gap:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "gap_above_single_gap_threshold"
            reason_detail = (
                f"gap={gap} > max_single_gap={self._max_single_gap}, "
                f"checkpoint restart (stale state too far behind)"
            )
        else:
            path = RecoveryPath.HYBRID_RECOVERY
            reason = "within_two_threshold_safe_region"
            reason_detail = (
                f"gap={gap} in [{self._delta_time_min_gap}, "
                f"{self._max_single_gap}], hybrid recovery"
            )

        return RecoveryDecision(
            path=path,
            current_step=current_step,
            latest_checkpoint_step=latest_checkpoint_step,
            gap=gap,
            failed_rank=failed_rank,
            reason=reason,
            reason_detail=reason_detail,
            delta_time_min_gap=self._delta_time_min_gap,
            max_single_gap=self._max_single_gap,
            metadata={
                "policy": "two_threshold",
                "delta_time_min_gap": self._delta_time_min_gap,
                "max_single_gap": self._max_single_gap,
                "replacement_rank": replacement_rank,
                "num_affected_experts": num_affected_experts,
            },
        )


# =====================================================================
# Policy 5: RankExposureGuardedPolicy (main policy)
# =====================================================================

@dataclass
class RankExposureGuardedConfig:
    """Configuration for the rank-exposure guarded recovery policy.

    Attributes:
        delta_time_min_gap: Gap below this → checkpoint restart.
        max_single_gap: Gap above this → checkpoint restart.
        exposure_window_steps: Sliding window for tracking rank stale
            exposure.
        max_rank_stale_exposure: Maximum stale exposure ratio per rank
            within the window (e.g. 0.02 = 2%).
    """

    delta_time_min_gap: int = 32
    max_single_gap: int = 192
    exposure_window_steps: int = 20000
    # Kept under the old field name for CLI/checkpoint compatibility.  It now
    # bounds the paper's global expert staleness density Phi, not one rank's
    # unweighted stale-iteration ratio.
    max_rank_stale_exposure: float = 0.1

    @property
    def max_expert_staleness_density(self) -> float:
        return self.max_rank_stale_exposure

    def validate(self) -> None:
        """Validate configuration constraints.  Raises ValueError."""
        if self.delta_time_min_gap < 0:
            raise ValueError(
                f"delta_time_min_gap must be >= 0, "
                f"got {self.delta_time_min_gap}"
            )
        if self.max_single_gap < self.delta_time_min_gap:
            raise ValueError(
                f"max_single_gap ({self.max_single_gap}) must be >= "
                f"delta_time_min_gap ({self.delta_time_min_gap})"
            )
        if self.exposure_window_steps <= 0:
            raise ValueError(
                f"exposure_window_steps must be > 0, "
                f"got {self.exposure_window_steps}"
            )
        if not (0 < self.max_rank_stale_exposure <= 1):
            raise ValueError(
                f"max_rank_stale_exposure must be in (0, 1], "
                f"got {self.max_rank_stale_exposure}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_time_min_gap": self.delta_time_min_gap,
            "max_single_gap": self.max_single_gap,
            "exposure_window_steps": self.exposure_window_steps,
            "max_rank_stale_exposure": self.max_rank_stale_exposure,
            "max_expert_staleness_density": self.max_expert_staleness_density,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RankExposureGuardedConfig":
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known_keys})


class RankExposureGuardedPolicy(RecoveryPolicyBase):
    """Rank-exposure guarded recovery policy.

    Decision logic::

        rank_stale_iters_before =
            tracker.get_rank_stale_iters(rank, current_step, window_steps)
        rank_stale_iters_after = rank_stale_iters_before + gap
        rank_exposure_after = rank_stale_iters_after / window_steps

        if gap < delta_time_min_gap:
            path = checkpoint_restart   (reason: gap_below_time_threshold)
        elif gap > max_single_gap:
            path = checkpoint_restart   (reason: gap_above_single_gap_threshold)
        elif rank_exposure_after > max_rank_stale_exposure:
            path = checkpoint_restart   (reason: rank_stale_exposure_exceeded)
        else:
            path = hybrid_recovery      (reason: within_rank_exposure_safe_region)

    **Important**: The exposure check uses ``rank_stale_iters_after``
    (i.e. *including* the current gap), not ``rank_stale_iters_before``.
    This ensures the policy accounts for the stale iterations the current
    hybrid recovery would introduce.

    Only hybrid recovery events are recorded in the tracker; checkpoint
    restarts reload all ranks uniformly, so no rank is relatively
    "stale".

    Args:
        config: A ``RankExposureGuardedConfig`` with all parameters.
        tracker: Optional ``RankExposureTracker`` instance.  If None,
            the global singleton is used.
    """

    def __init__(
        self,
        config: Optional[RankExposureGuardedConfig] = None,
        tracker: Any = None,
    ) -> None:
        self._config = config or RankExposureGuardedConfig()
        self._config.validate()

        if tracker is not None:
            self._tracker = tracker
        else:
            try:
                from megatron.core.transformer.moe.rank_exposure_tracker import (
                    get_rank_exposure_tracker,
                )
                self._tracker = get_rank_exposure_tracker()
            except (ImportError, ModuleNotFoundError):
                # Fallback for test environments where the full megatron
                # package tree is not importable.  Create a standalone
                # RankExposureTracker directly.  This tracker will NOT
                # be the global singleton, but the policy will function
                # correctly for decision-making purposes.
                import rank_exposure_tracker as _ret
                self._tracker = _ret.RankExposureTracker()
                logger.info(
                    "MOEGAMBIT-MoE RankExposureGuardedPolicy: created standalone "
                    "RankExposureTracker (global singleton not available)"
                )

        logger.warning(
            "MOEGAMBIT-MoE RankExposureGuardedPolicy initialized: %s",
            self._config.to_dict(),
        )

    @property
    def config(self) -> RankExposureGuardedConfig:
        return self._config

    @property
    def gap_threshold(self) -> int:
        """Backward-compatible: returns delta_time_min_gap."""
        return self._config.delta_time_min_gap

    def get_rank_stale_exposure(self, rank: int, current_step: int) -> float:
        """Compute stale exposure ratio for a rank within the window."""
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

    def choose(
        self,
        *,
        current_step: int,
        latest_checkpoint_step: int,
        failed_rank: int = -1,
        replacement_rank: int = -1,
        num_affected_experts: int = 0,
        **kwargs: Any,
    ) -> RecoveryDecision:
        """Select recovery path based on gap boundaries and rank exposure.

        When ``expert_dp_peer_available=True`` is passed (indicating EDP > 1
        and at least one healthy Expert-DP peer exists for the failed rank),
        the policy selects ``FULL_PEER_RECOVERY`` — a zero-disk, zero-staleness
        path that pulls *all* state (dense + expert) from in-memory peers.
        This path bypasses the gap and exposure checks entirely because the
        recovered expert state is at step *t* (current), not step *c*
        (checkpoint), so it introduces zero staleness.
        """
        cfg = self._config
        gap = current_step - latest_checkpoint_step
        affected_experts = max(1, int(num_affected_experts or 1))
        num_experts = max(
            affected_experts,
            int(kwargs.get("num_total_experts", 0) or 0),
            1,
        )
        debt_before = self._tracker.get_window_expert_iteration_debt(
            current_step=current_step,
            window_steps=cfg.exposure_window_steps,
        )
        current_debt = (
            affected_experts * max(0, gap)
            if latest_checkpoint_step >= 0
            else 0
        )
        debt_after = debt_before + current_debt
        denominator = num_experts * cfg.exposure_window_steps
        density_before = debt_before / denominator if denominator > 0 else 0.0
        density_after = debt_after / denominator if denominator > 0 else 0.0

        # Preserve the old fields as aliases so existing telemetry consumers
        # continue to work while the decision uses the paper's expert-weighted
        # global debt.
        stale_iters_before = debt_before
        stale_iters_after = debt_after
        exposure_before = density_before
        exposure_after = density_after

        # --- Full-peer fast path (EDP > 1) ---
        expert_dp_peer_available = kwargs.get("expert_dp_peer_available", False)
        dense_peer_available = kwargs.get("dense_peer_available", True)
        edp = kwargs.get("expert_data_parallel_size", 1)

        if expert_dp_peer_available:
            current_debt = 0
            debt_after = debt_before
            density_after = density_before
            stale_iters_after = debt_after
            exposure_after = density_after
            path = RecoveryPath.FULL_PEER_RECOVERY
            reason = "full_peer_edp_available"
            reason_detail = (
                f"EDP={edp} > 1, expert-DP peer available for rank "
                f"{failed_rank} — zero-disk full-peer recovery "
                f"(zero staleness, Phi'(t) contribution = 0)"
            )

        # --- MoEGambit R2 admission contract ---
        elif not dense_peer_available:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "dense_peer_unavailable"
            reason_detail = (
                "no current-step dense/non-expert peer is available; "
                "the intentional mixed-version state cannot be constructed"
            )

        elif latest_checkpoint_step < 0:
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "no_checkpoint_available"
            reason_detail = (
                f"no checkpoint available (latest_checkpoint_step="
                f"{latest_checkpoint_step}); expert weights and optimizer "
                "cannot satisfy the state-source contract"
            )

        elif gap < cfg.delta_time_min_gap:
            # Gap too small: hybrid overhead not worthwhile
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "gap_below_time_threshold"
            reason_detail = (
                f"gap={gap} < delta_time_min_gap={cfg.delta_time_min_gap}, "
                f"checkpoint restart (hybrid too expensive for tiny gap)"
            )

        elif gap > cfg.max_single_gap:
            # Gap too large: stale state too far behind
            path = RecoveryPath.CHECKPOINT_RESTART
            reason = "gap_above_single_gap_threshold"
            reason_detail = (
                f"gap={gap} > max_single_gap={cfg.max_single_gap}, "
                f"checkpoint restart (stale state too far behind)"
            )

        else:
            if density_after > cfg.max_expert_staleness_density:
                path = RecoveryPath.CHECKPOINT_RESTART
                reason = "rank_stale_exposure_exceeded"
                reason_detail = (
                    f"gap={gap} in [{cfg.delta_time_min_gap}, "
                    f"{cfg.max_single_gap}], but Phi'(t)={density_after:.6f} "
                    f"> Phi_max={cfg.max_expert_staleness_density:.6f}; "
                    f"S(t)={debt_before}, |E_new|={affected_experts}, "
                    f"N_expert={num_experts}, W_exp={cfg.exposure_window_steps}"
                )
            else:
                path = RecoveryPath.HYBRID_RECOVERY
                reason = "within_rank_exposure_safe_region"
                reason_detail = (
                    f"gap={gap} in [{cfg.delta_time_min_gap}, "
                    f"{cfg.max_single_gap}], Phi'(t)={density_after:.6f} "
                    f"<= Phi_max={cfg.max_expert_staleness_density:.6f}; "
                    "intentional mixed-version recovery admitted"
                )

        # NOTE: Recording of hybrid recovery events is deferred to the
        # RecoveryController, which calls tracker.record_hybrid_recovery()
        # only AFTER hybrid recovery SUCCEEDS.  This ensures that failed
        # hybrid attempts that fall back to checkpoint restart do not
        # pollute the exposure tracker with fictitious stale iterations.

        decision = RecoveryDecision(
            path=path,
            current_step=current_step,
            latest_checkpoint_step=latest_checkpoint_step,
            gap=gap,
            failed_rank=failed_rank,
            reason=reason,
            reason_detail=reason_detail,
            delta_time_min_gap=cfg.delta_time_min_gap,
            max_single_gap=cfg.max_single_gap,
            exposure_window_steps=cfg.exposure_window_steps,
            rank_stale_iters_before=stale_iters_before,
            rank_stale_iters_after=stale_iters_after,
            rank_stale_exposure_before=exposure_before,
            rank_stale_exposure_after=exposure_after,
            max_rank_stale_exposure=cfg.max_rank_stale_exposure,
            num_affected_experts=affected_experts,
            num_experts=num_experts,
            window_expert_iteration_debt_before=debt_before,
            window_expert_iteration_debt_after=debt_after,
            expert_staleness_density_before=density_before,
            expert_staleness_density_after=density_after,
            max_expert_staleness_density=cfg.max_expert_staleness_density,
            metadata={
                "policy": "rank_exposure_guarded",
                "contract": "moegambit_r2_expert_staleness_density",
                "contract_reason": (
                    "admit" if path == RecoveryPath.HYBRID_RECOVERY else reason
                ),
                "replacement_rank": replacement_rank,
                "num_affected_experts": affected_experts,
                "num_experts": num_experts,
                "policy_config": cfg.to_dict(),
            },
        )

        # Structured decision log
        logger.warning(
            "MOEGAMBIT-MoE recovery decision: "
            "path=%s | step=%d | ckpt_step=%d | gap=%d | "
            "delta_time_min_gap=%d | max_single_gap=%d | "
            "debt_before=%d | debt_after=%d | phi_after=%.6f | "
            "phi_max=%.4f | affected_experts=%d/%d | reason=%s | failed_rank=%d",
            decision.path.value,
            decision.current_step,
            decision.latest_checkpoint_step,
            decision.gap,
            cfg.delta_time_min_gap,
            cfg.max_single_gap,
            stale_iters_before,
            stale_iters_after,
            exposure_after,
            cfg.max_rank_stale_exposure,
            affected_experts,
            num_experts,
            decision.reason,
            failed_rank,
        )

        return decision


# =====================================================================
# Backward-compatible aliases
# =====================================================================

# Old class names kept for backward compatibility with existing code
# that imports these names.
ThresholdRecoveryPolicy = FixedGapThresholdPolicy
RankExposureGuardedHybridPolicy = RankExposureGuardedPolicy


# =====================================================================
# Gap-Aware Recovery Policy Manager
# =====================================================================

class GapAwareRecoveryPolicyManager:
    """Manages the recovery policy lifecycle.

    This is the main entry point used by ``RecoveryController``.  It:
    1. Holds a reference to the active policy.
    2. Accepts a ``get_checkpoint_iteration_fn`` callback to query the
       latest checkpoint iteration at decision time.
    3. Records decision history for audit.

    Usage::

        mgr = GapAwareRecoveryPolicyManager(
            policy=RankExposureGuardedPolicy(config=cfg),
            get_checkpoint_iteration_fn=my_fn,
        )
        decision = mgr.evaluate(current_iteration=500, failed_rank=0)
    """

    def __init__(
        self,
        *,
        gap_threshold: int = 100,
        policy: Optional[RecoveryPolicyBase] = None,
        get_checkpoint_iteration_fn: Optional[Callable[[], int]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if policy is not None:
            self._policy = policy
        else:
            self._policy = FixedGapThresholdPolicy(
                fixed_gap_threshold=gap_threshold,
            )

        self._get_checkpoint_iteration_fn = get_checkpoint_iteration_fn
        self._decision_history: list[RecoveryDecision] = []
        self._enabled: bool = True
        self._run_id = run_id or _default_run_id()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def policy(self) -> RecoveryPolicyBase:
        return self._policy

    @policy.setter
    def policy(self, value: RecoveryPolicyBase) -> None:
        self._policy = value

    @property
    def policy_type(self) -> str:
        return _policy_type_from_policy(self._policy)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def decision_history(self) -> list[RecoveryDecision]:
        return list(self._decision_history)

    @property
    def last_decision(self) -> Optional[RecoveryDecision]:
        return self._decision_history[-1] if self._decision_history else None

    def set_checkpoint_iteration_fn(
        self, fn: Callable[[], int],
    ) -> None:
        self._get_checkpoint_iteration_fn = fn

    def get_checkpoint_iteration(self) -> int:
        """Query the latest checkpoint iteration.  Returns -1 on failure."""
        if self._get_checkpoint_iteration_fn is None:
            return -1
        try:
            return self._get_checkpoint_iteration_fn()
        except Exception as e:
            logger.warning(
                "MOEGAMBIT-MoE gap-aware: get_checkpoint_iteration_fn failed: %s", e,
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

        Decision precedence:
          1. ``MOEGAMBIT_FORCE_CHECKPOINT_RESTART=1`` env var (highest priority):
             force every event to ``CHECKPOINT_RESTART``. Used by the
             MoC-System emulation script to faithfully reproduce MoC-System's
             full-ckpt-reload recovery semantics regardless of which MOEGAMBIT
             switches happen to be on in the argv. The PEC byte-overlay
             still rewrites which expert shard gets loaded; the path itself
             is forced to restart.
          2. If gap-aware recovery is disabled, return ``HYBRID_RECOVERY``
             (legacy default; matches MoEGuard runs that disable the policy
             but want the fast path).
          3. Otherwise consult the configured policy.
        """
        if os.environ.get("MOEGAMBIT_FORCE_CHECKPOINT_RESTART", "0") == "1":
            decision = RecoveryDecision(
                path=RecoveryPath.CHECKPOINT_RESTART,
                current_step=current_iteration,
                latest_checkpoint_step=-1,
                gap=-1,
                failed_rank=failed_rank,
                reason="forced_checkpoint_restart",
                reason_detail=(
                    "MOEGAMBIT_FORCE_CHECKPOINT_RESTART=1 — recovery path forced "
                    "to CHECKPOINT_RESTART for MoC-System emulation"
                ),
                metadata={"policy": "forced_restart"},
            )
            self._decision_history.append(decision)
            return decision

        if not self._enabled:
            decision = RecoveryDecision(
                path=RecoveryPath.HYBRID_RECOVERY,
                current_step=current_iteration,
                latest_checkpoint_step=-1,
                gap=-1,
                failed_rank=failed_rank,
                reason="gap_aware_disabled",
                reason_detail="gap-aware recovery disabled, defaulting to hybrid",
                metadata={"policy": "disabled"},
            )
            self._decision_history.append(decision)
            return decision

        checkpoint_iteration = self.get_checkpoint_iteration()

        decision = self._policy.choose(
            current_step=current_iteration,
            latest_checkpoint_step=checkpoint_iteration,
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
            "structured_policy_type": self.policy_type,
            "run_id": self._run_id,
            "num_decisions": len(self._decision_history),
            "last_decision": (
                self.last_decision.to_dict()
                if self.last_decision is not None
                else None
            ),
        }
        # Add policy-specific info
        if isinstance(self._policy, FixedGapThresholdPolicy):
            result["gap_threshold"] = self._policy.fixed_gap_threshold
        elif isinstance(self._policy, TwoThresholdPolicy):
            result["delta_time_min_gap"] = self._policy.delta_time_min_gap
            result["max_single_gap"] = self._policy.max_single_gap
        elif isinstance(self._policy, RankExposureGuardedPolicy):
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


# Policy type name → factory function
_POLICY_REGISTRY: Dict[str, Callable[..., RecoveryPolicyBase]] = {
    "restart_and_spare": lambda **kw: RestartAndSparePolicy(),
    "always_hybrid": lambda **kw: AlwaysHybridPolicy(),
    "fixed_gap_threshold": lambda **kw: FixedGapThresholdPolicy(
        fixed_gap_threshold=kw.get("gap_threshold", 32),
    ),
    "two_threshold": lambda **kw: TwoThresholdPolicy(
        delta_time_min_gap=kw.get("delta_time_min_gap", 32),
        max_single_gap=kw.get("max_single_gap", 192),
    ),
    "rank_exposure_guarded": lambda **kw: RankExposureGuardedPolicy(
        config=kw.get("rank_exposure_config"),
    ),
    "expert_staleness_guarded": lambda **kw: RankExposureGuardedPolicy(
        config=kw.get("rank_exposure_config"),
    ),
    # Backward-compatible aliases
    "threshold": lambda **kw: FixedGapThresholdPolicy(
        fixed_gap_threshold=kw.get("gap_threshold", 100),
    ),
    "rank_exposure_guarded_hybrid": lambda **kw: RankExposureGuardedPolicy(
        config=kw.get("rank_exposure_config"),
    ),
}


def initialize_gap_aware_recovery_policy(
    *,
    gap_threshold: int = 100,
    policy: Optional[RecoveryPolicyBase] = None,
    policy_type: str = "threshold",
    rank_exposure_config: Optional[RankExposureGuardedConfig] = None,
    delta_time_min_gap: Optional[int] = None,
    max_single_gap: Optional[int] = None,
    get_checkpoint_iteration_fn: Optional[Callable[[], int]] = None,
    enabled: bool = True,
    run_id: Optional[str] = None,
) -> GapAwareRecoveryPolicyManager:
    """Initialize the global policy manager with configuration.

    Args:
        gap_threshold: Gap threshold for fixed-gap policies.
            Also used as ``fixed_gap_threshold`` when
            ``policy_type="fixed_gap_threshold"``.
        policy: Optional custom policy override.
        policy_type: Policy type string.  One of:
            - ``"restart_and_spare"``: always checkpoint restart.
            - ``"always_hybrid"``: always hybrid recovery.
            - ``"fixed_gap_threshold"``: single gap threshold.
            - ``"two_threshold"``: gap lower + upper bound.
            - ``"rank_exposure_guarded"``: gap bounds + exposure tracking.
            - ``"threshold"``: alias for ``fixed_gap_threshold`` (backward compat).
            - ``"rank_exposure_guarded_hybrid"``: alias for
              ``rank_exposure_guarded`` (backward compat).
        rank_exposure_config: Optional config for RankExposureGuardedPolicy.
        delta_time_min_gap: Override for TwoThreshold / RankExposureGuarded.
        max_single_gap: Override for TwoThreshold / RankExposureGuarded.
        get_checkpoint_iteration_fn: Callback to query checkpoint iteration.
        enabled: Whether gap-aware recovery is enabled.
        run_id: Optional stable run id for structured JSON decision logs.

    Returns:
        The initialized ``GapAwareRecoveryPolicyManager``.
    """
    global _POLICY_MANAGER

    if policy is not None:
        _POLICY_MANAGER = GapAwareRecoveryPolicyManager(
            gap_threshold=gap_threshold,
            policy=policy,
            get_checkpoint_iteration_fn=get_checkpoint_iteration_fn,
            run_id=run_id,
        )
        _POLICY_MANAGER.enabled = enabled
        return _POLICY_MANAGER

    # Build RankExposureGuardedConfig if needed
    if policy_type in (
        "rank_exposure_guarded",
        "rank_exposure_guarded_hybrid",
        "expert_staleness_guarded",
    ):
        cfg = rank_exposure_config
        if cfg is None:
            cfg = RankExposureGuardedConfig(
                delta_time_min_gap=(
                    delta_time_min_gap
                    if delta_time_min_gap is not None else 32
                ),
                max_single_gap=(
                    max_single_gap
                    if max_single_gap is not None else 192
                ),
            )
        cfg.validate()
        policy = RankExposureGuardedPolicy(config=cfg)
        logger.warning(
            "MOEGAMBIT-MoE gap-aware recovery policy: "
            "using RankExposureGuardedPolicy "
            "(delta_time_min_gap=%d, max_single_gap=%d, "
            "exposure_window_steps=%d, max_rank_stale_exposure=%.4f)",
            cfg.delta_time_min_gap, cfg.max_single_gap,
            cfg.exposure_window_steps, cfg.max_rank_stale_exposure,
        )
    elif policy_type == "two_threshold":
        dt = delta_time_min_gap if delta_time_min_gap is not None else 32
        msg = max_single_gap if max_single_gap is not None else 192
        policy = TwoThresholdPolicy(
            delta_time_min_gap=dt,
            max_single_gap=msg,
        )
        logger.warning(
            "MOEGAMBIT-MoE gap-aware recovery policy: "
            "using TwoThresholdPolicy "
            "(delta_time_min_gap=%d, max_single_gap=%d)",
            dt, msg,
        )
    else:
        # Default / fixed_threshold / threshold
        factory = _POLICY_REGISTRY.get(policy_type, _POLICY_REGISTRY["threshold"])
        policy = factory(gap_threshold=gap_threshold)
        logger.warning(
            "MOEGAMBIT-MoE gap-aware recovery policy: "
            "using %s (gap_threshold=%d)",
            type(policy).__name__, gap_threshold,
        )

    _POLICY_MANAGER = GapAwareRecoveryPolicyManager(
        gap_threshold=gap_threshold,
        policy=policy,
        get_checkpoint_iteration_fn=get_checkpoint_iteration_fn,
        run_id=run_id,
    )
    _POLICY_MANAGER.enabled = enabled
    logger.warning(
        "MOEGAMBIT-MoE gap-aware recovery policy initialized: "
        "enabled=%s, policy=%s",
        enabled, type(_POLICY_MANAGER.policy).__name__,
    )
    return _POLICY_MANAGER


def clear_gap_aware_recovery_policy() -> None:
    """Reset the global policy manager (for testing)."""
    global _POLICY_MANAGER
    if _POLICY_MANAGER is not None:
        _POLICY_MANAGER.reset()
    _POLICY_MANAGER = None
