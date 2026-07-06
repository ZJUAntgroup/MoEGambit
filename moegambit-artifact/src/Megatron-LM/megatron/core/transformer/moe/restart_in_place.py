# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Restart-In-Place Recovery Module for MoEGambit.

Core infrastructure for simulating and testing restart-in-place fault recovery with real tensor operations.  Unlike the
existing fault injection framework (which uses pure state-machine operations),
this module actually invalidates tensor data (fills with NaN sentinels) and
verifies recovery correctness at the tensor level.

Key components:

1. ``RestartInPlaceRecoveryState`` — Recovery state enum:
   HEALTHY → INVALIDATED → WEIGHTS_RESTORING → WEIGHTS_READY →
   OPTIMIZER_RESTORING → FULLY_RECOVERED

2. ``invalidate_rank_tensors()`` — Fills all parameters and optimizer states
   with NaN sentinels to simulate GPU memory loss.

3. ``verify_recovery()`` — Checks all recovered tensors for NaN/Inf/shape/dtype
   correctness.  Acts as a hard gate before reintegration.

4. ``RestartInPlaceCoordinator`` — Orchestrates the full restart-in-place
   recovery flow: fault injection → invalidation → recovery → verification.

Design principles:
- Real tensor operations (not just state machine transitions)
- NaN sentinel for invalidation (not zeros, which could be valid)
- Fail-closed verification (any NaN/Inf blocks reintegration)
- Single-process testable (no distributed init required)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Recovery state enum
# =====================================================================

class RestartInPlaceRecoveryState(enum.IntEnum):
    """States for restart-in-place recovery.

    Permission matrix:
    +---------------------+---------+----------+----------------+
    | State               | forward | backward | optimizer step |
    +=====================+=========+==========+================+
    | HEALTHY             |    ✓    |    ✓     |       ✓        |
    | INVALIDATED         |    ✗    |    ✗     |       ✗        |
    | WEIGHTS_RESTORING   |    ✗    |    ✗     |       ✗        |
    | WEIGHTS_READY       |    ✓    |    ✓     |       ✗        |
    | OPTIMIZER_RESTORING |    ✓    |    ✓     |       ✗        |
    | FULLY_RECOVERED     |    ✓    |    ✓     |       ✓        |
    +---------------------+---------+----------+----------------+
    """

    HEALTHY = 0
    INVALIDATED = 1
    WEIGHTS_RESTORING = 2
    WEIGHTS_READY = 3
    OPTIMIZER_RESTORING = 4
    FULLY_RECOVERED = 5


# Valid transitions
_VALID_RIP_TRANSITIONS = {
    RestartInPlaceRecoveryState.HEALTHY: {
        RestartInPlaceRecoveryState.INVALIDATED,
    },
    RestartInPlaceRecoveryState.INVALIDATED: {
        RestartInPlaceRecoveryState.WEIGHTS_RESTORING,
    },
    RestartInPlaceRecoveryState.WEIGHTS_RESTORING: {
        RestartInPlaceRecoveryState.WEIGHTS_READY,
    },
    RestartInPlaceRecoveryState.WEIGHTS_READY: {
        RestartInPlaceRecoveryState.OPTIMIZER_RESTORING,
    },
    RestartInPlaceRecoveryState.OPTIMIZER_RESTORING: {
        RestartInPlaceRecoveryState.FULLY_RECOVERED,
    },
    RestartInPlaceRecoveryState.FULLY_RECOVERED: {
        RestartInPlaceRecoveryState.HEALTHY,
    },
}

# States where forward/backward is allowed
_FORWARD_ALLOWED = frozenset({
    RestartInPlaceRecoveryState.HEALTHY,
    RestartInPlaceRecoveryState.WEIGHTS_READY,
    RestartInPlaceRecoveryState.OPTIMIZER_RESTORING,
    RestartInPlaceRecoveryState.FULLY_RECOVERED,
})

# States where optimizer step is allowed
_OPTIMIZER_ALLOWED = frozenset({
    RestartInPlaceRecoveryState.HEALTHY,
    RestartInPlaceRecoveryState.FULLY_RECOVERED,
})


# =====================================================================
# Verification result
# =====================================================================

class RecoveryVerificationError(RuntimeError):
    """Raised when recovery verification fails."""
    pass


@dataclass
class VerificationResult:
    """Result of a recovery verification check."""

    success: bool = False
    """Whether all checks passed."""

    num_params_checked: int = 0
    """Number of parameters checked."""

    num_opt_states_checked: int = 0
    """Number of optimizer state tensors checked."""

    errors: List[str] = field(default_factory=list)
    """List of error messages for failed checks."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "num_params_checked": self.num_params_checked,
            "num_opt_states_checked": self.num_opt_states_checked,
            "errors": list(self.errors),
        }


# =====================================================================
# Tensor invalidation
# =====================================================================

def invalidate_rank_tensors(model, optimizer=None, fill_value: float = float('nan')) -> Dict[str, int]:
    """Invalidate all parameters and optimizer states with a sentinel value.

    This simulates GPU memory loss by overwriting tensor contents.  The
    default NaN sentinel makes accidental stale use obvious; tests can pass
    ``fill_value=0.0`` to model a device whose memory comes back zeroed.

    Args:
        model: The model whose parameters to invalidate.
        optimizer: Optional optimizer whose states to invalidate.

    Returns:
        Dict with counts: {"params_invalidated", "opt_states_invalidated"}.
    """
    import torch

    stats = {"params_invalidated": 0, "opt_states_invalidated": 0}

    # Invalidate all model parameters
    for name, param in model.named_parameters():
        param.data.fill_(fill_value)
        stats["params_invalidated"] += 1

    # Invalidate optimizer states
    if optimizer is not None:
        for param_key, state in optimizer.state.items():
            if isinstance(state, dict):
                for state_name, state_val in state.items():
                    if hasattr(state_val, 'fill_'):
                        state_val.fill_(fill_value)
                        stats["opt_states_invalidated"] += 1

    logger.warning(
        "MoEGambit restart-in-place: invalidated %d params, %d optimizer states "
        "(fill_value=%s)",
        stats["params_invalidated"], stats["opt_states_invalidated"], fill_value,
    )
    return stats


def zero_rank_tensors(model, optimizer=None) -> Dict[str, int]:
    """Overwrite all parameters and optimizer states with zeros."""
    return invalidate_rank_tensors(model, optimizer, fill_value=0.0)


def invalidate_dense_params_only(model, optimizer=None, classification=None):
    """Invalidate only dense-like parameters (allreduce=True).

    Args:
        model: The model.
        optimizer: Optional optimizer.
        classification: Optional ParamClassification. If None, computed.

    Returns:
        Dict with counts.
    """
    import torch

    if classification is None:
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        classification = classify_model_parameters(model)

    stats = {"params_invalidated": 0, "opt_states_invalidated": 0}

    for name, param in classification.all_dense_like.items():
        param.data.fill_(float('nan'))
        stats["params_invalidated"] += 1

    if optimizer is not None:
        dense_param_ids = set(id(p) for p in classification.all_dense_like.values())
        for p in classification.all_dense_like.values():
            if hasattr(p, 'main_param'):
                dense_param_ids.add(id(p.main_param))

        for param_key, state in optimizer.state.items():
            param_id = id(param_key)
            if param_id not in dense_param_ids:
                continue
            if isinstance(state, dict):
                for state_name, state_val in state.items():
                    if hasattr(state_val, 'fill_'):
                        state_val.fill_(float('nan'))
                        stats["opt_states_invalidated"] += 1

    return stats


def invalidate_expert_params_only(model, optimizer=None, classification=None):
    """Invalidate only expert parameters (allreduce=False).

    Args:
        model: The model.
        optimizer: Optional optimizer.
        classification: Optional ParamClassification. If None, computed.

    Returns:
        Dict with counts.
    """
    import torch

    if classification is None:
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        classification = classify_model_parameters(model)

    stats = {"params_invalidated": 0, "opt_states_invalidated": 0}

    for name, param in classification.expert_params.items():
        param.data.fill_(float('nan'))
        stats["params_invalidated"] += 1

    if optimizer is not None:
        expert_param_ids = set(id(p) for p in classification.expert_params.values())
        for p in classification.expert_params.values():
            if hasattr(p, 'main_param'):
                expert_param_ids.add(id(p.main_param))

        for param_key, state in optimizer.state.items():
            param_id = id(param_key)
            if param_id not in expert_param_ids:
                continue
            if isinstance(state, dict):
                for state_name, state_val in state.items():
                    if hasattr(state_val, 'fill_'):
                        state_val.fill_(float('nan'))
                        stats["opt_states_invalidated"] += 1

    return stats


# =====================================================================
# Verification
# =====================================================================

def verify_recovery(
    model,
    optimizer=None,
    classification=None,
    *,
    check_dense: bool = True,
    check_expert: bool = True,
    check_optimizer: bool = True,
    reference_shapes: Optional[Dict[str, Tuple]] = None,
    reference_dtypes: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    """Verify that recovery produced valid tensors.

    Checks:
    1. No NaN values in any parameter
    2. No Inf values in any parameter
    3. Shape matches reference (if provided)
    4. Dtype matches reference (if provided)
    5. Optimizer states are non-empty and contain no NaN/Inf
       (for parameters that should have been recovered)

    Args:
        model: The model to verify.
        optimizer: Optional optimizer to verify.
        classification: Optional ParamClassification.
        check_dense: Whether to check dense-like parameters.
        check_expert: Whether to check expert parameters.
        check_optimizer: Whether to check optimizer states.
        reference_shapes: Optional dict of param_name -> expected shape.
        reference_dtypes: Optional dict of param_name -> expected dtype.

    Returns:
        VerificationResult.

    Raises:
        RecoveryVerificationError: If any check fails.
    """
    import torch

    if classification is None:
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        classification = classify_model_parameters(model)

    result = VerificationResult()

    # Determine which params to check
    params_to_check: Dict[str, Any] = {}
    if check_dense:
        params_to_check.update(classification.all_dense_like)
    if check_expert:
        params_to_check.update(classification.expert_params)

    # Check parameters
    for name, param in params_to_check.items():
        result.num_params_checked += 1

        if torch.isnan(param.data).any():
            result.errors.append(f"NaN found in param '{name}'")

        if torch.isinf(param.data).any():
            result.errors.append(f"Inf found in param '{name}'")

        if reference_shapes is not None and name in reference_shapes:
            expected_shape = reference_shapes[name]
            if param.data.shape != expected_shape:
                result.errors.append(
                    f"Shape mismatch in param '{name}': "
                    f"expected {expected_shape}, got {param.data.shape}"
                )

        if reference_dtypes is not None and name in reference_dtypes:
            expected_dtype = reference_dtypes[name]
            if param.data.dtype != expected_dtype:
                result.errors.append(
                    f"Dtype mismatch in param '{name}': "
                    f"expected {expected_dtype}, got {param.data.dtype}"
                )

    # Check optimizer states
    if check_optimizer and optimizer is not None:
        # Build set of param ids we care about
        check_param_ids = set(id(p) for p in params_to_check.values())
        for p in params_to_check.values():
            if hasattr(p, 'main_param'):
                check_param_ids.add(id(p.main_param))

        state_dict = optimizer.state
        if hasattr(state_dict, 'items'):
            found_any_state = False
            for param_key, state in state_dict.items():
                param_id = id(param_key)
                if param_id not in check_param_ids:
                    continue

                if isinstance(state, dict):
                    for state_name, state_val in state.items():
                        if hasattr(state_val, 'numel') and hasattr(state_val, 'data'):
                            found_any_state = True
                            result.num_opt_states_checked += 1

                            if torch.isnan(state_val).any():
                                result.errors.append(
                                    f"NaN in optimizer state "
                                    f"'{state_name}' for param id={param_id}"
                                )
                            if torch.isinf(state_val).any():
                                result.errors.append(
                                    f"Inf in optimizer state "
                                    f"'{state_name}' for param id={param_id}"
                                )

            # Fail if optimizer state is expected but empty
            if not found_any_state and len(params_to_check) > 0:
                result.errors.append(
                    "Optimizer state is empty but parameters require "
                    "recovered optimizer state"
                )

    result.success = len(result.errors) == 0

    if not result.success:
        error_msg = (
            f"Recovery verification failed with {len(result.errors)} errors: "
            + "; ".join(result.errors[:5])
        )
        if len(result.errors) > 5:
            error_msg += f" ... and {len(result.errors) - 5} more"
        logger.error("MoEGambit restart-in-place: %s", error_msg)
        raise RecoveryVerificationError(error_msg)

    logger.info(
        "MoEGambit restart-in-place: verification PASSED "
        "(%d params, %d opt states checked)",
        result.num_params_checked, result.num_opt_states_checked,
    )
    return result


# =====================================================================
# RestartInPlaceCoordinator
# =====================================================================

class RestartInPlaceCoordinator:
    """Orchestrates restart-in-place recovery with real tensor operations.

    This coordinator combines the RecoveryController (in restart-in-place
    test mode) with tensor invalidation, gap-aware policy selection, and
    recovery verification.

    Usage (single-process test)::

        coord = RestartInPlaceCoordinator()
        model, optimizer = make_model_and_optimizer()

        # Save ground truth
        gt = {n: p.data.clone() for n, p in model.named_parameters()}

        # Inject fault + invalidate tensors
        coord.inject_fault_and_invalidate(model, optimizer, step=100)

        # Execute recovery (providing restore callbacks)
        coord.execute_recovery(
            model, optimizer,
            dense_broadcast_fn=lambda p, **kw: p.data.copy_(gt[name]),
            expert_load_fn=lambda entry, m: ...,
            step=100,
        )

        # Verify and reintegrate
        coord.verify_and_reintegrate(model, optimizer, step=101)
    """

    def __init__(
        self,
        *,
        gap_threshold: int = 100,
        checkpoint_iteration: int = -1,
    ) -> None:
        self._state = RestartInPlaceRecoveryState.HEALTHY
        self._gap_threshold = gap_threshold
        self._checkpoint_iteration = checkpoint_iteration
        self._fault_step: int = -1
        self._recovery_path: str = ""
        self._event_log: List[Dict[str, Any]] = []
        self._classification = None

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    @property
    def state(self) -> RestartInPlaceRecoveryState:
        return self._state

    @property
    def recovery_path(self) -> str:
        return self._recovery_path

    @property
    def is_forward_allowed(self) -> bool:
        return self._state in _FORWARD_ALLOWED

    @property
    def is_optimizer_step_allowed(self) -> bool:
        return self._state in _OPTIMIZER_ALLOWED

    def _transition_to(self, new_state: RestartInPlaceRecoveryState, **details):
        old = self._state
        valid = _VALID_RIP_TRANSITIONS.get(old, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid restart-in-place transition: "
                f"{old.name} → {new_state.name}. "
                f"Valid: {[s.name for s in valid]}"
            )
        self._state = new_state
        event = {
            "from": old.name,
            "to": new_state.name,
            "timestamp": time.time(),
            **details,
        }
        self._event_log.append(event)
        logger.info(
            "MoEGambit restart-in-place: %s → %s",
            old.name, new_state.name,
        )

    # -----------------------------------------------------------------
    # Phase 1: Fault injection + invalidation
    # -----------------------------------------------------------------

    def inject_fault_and_invalidate(
        self,
        model,
        optimizer=None,
        step: int = -1,
    ) -> Dict[str, int]:
        """Inject a fault and invalidate all tensors.

        Transitions: HEALTHY → INVALIDATED

        Args:
            model: The model.
            optimizer: Optional optimizer.
            step: Training step when the fault occurs.

        Returns:
            Invalidation stats dict.
        """
        self._transition_to(
            RestartInPlaceRecoveryState.INVALIDATED,
            step=step,
        )
        self._fault_step = step

        # Classify parameters
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        self._classification = classify_model_parameters(model)

        # Invalidate tensors
        stats = invalidate_rank_tensors(model, optimizer)

        logger.warning(
            "MoEGambit restart-in-place: fault injected at step %d, "
            "invalidated %d params, %d opt states",
            step, stats["params_invalidated"], stats["opt_states_invalidated"],
        )
        return stats

    # -----------------------------------------------------------------
    # Phase 2: Recovery execution
    # -----------------------------------------------------------------

    def execute_recovery(
        self,
        model,
        optimizer=None,
        *,
        dense_restore_fn: Optional[Callable] = None,
        expert_restore_fn: Optional[Callable] = None,
        checkpoint_restore_fn: Optional[Callable] = None,
        dense_optimizer_restore_fn: Optional[Callable] = None,
        expert_optimizer_restore_fn: Optional[Callable] = None,
        step: int = -1,
    ) -> str:
        """Execute the recovery based on gap-aware policy.

        Transitions:
        - INVALIDATED → WEIGHTS_RESTORING → WEIGHTS_READY →
          OPTIMIZER_RESTORING → FULLY_RECOVERED

        For small gap (checkpoint restart): calls checkpoint_restore_fn
        which should restore everything at once.

        For large gap (hybrid recovery):
        1. dense_restore_fn: restore dense params from DP peer
        2. expert_restore_fn: restore expert params from checkpoint
        3. dense_optimizer_restore_fn: restore dense optimizer state
        4. expert_optimizer_restore_fn: restore expert optimizer state

        Args:
            model: The model.
            optimizer: Optional optimizer.
            dense_restore_fn: Callback to restore dense params.
                Signature: (model, classification) -> None
            expert_restore_fn: Callback to restore expert params.
                Signature: (model, classification) -> None
            checkpoint_restore_fn: Callback for full checkpoint restore.
                Signature: (model, optimizer) -> None
            dense_optimizer_restore_fn: Callback to restore dense optimizer.
                Signature: (optimizer, classification) -> None
            expert_optimizer_restore_fn: Callback to restore expert optimizer.
                Signature: (optimizer, classification) -> None
            step: Training step.

        Returns:
            The recovery path name: "CHECKPOINT_RESTART" or "HYBRID_RECOVERY".
        """
        if self._classification is None:
            from megatron.core.transformer.moe.dense_param_sync import (
                classify_model_parameters,
            )
            self._classification = classify_model_parameters(model)

        # Determine recovery path
        gap = step - self._checkpoint_iteration if self._checkpoint_iteration >= 0 else -1
        if self._checkpoint_iteration >= 0 and gap <= self._gap_threshold:
            path = "CHECKPOINT_RESTART"
        else:
            path = "HYBRID_RECOVERY"

        self._recovery_path = path

        # Phase: INVALIDATED → WEIGHTS_RESTORING
        self._transition_to(
            RestartInPlaceRecoveryState.WEIGHTS_RESTORING,
            step=step, path=path,
        )

        if path == "CHECKPOINT_RESTART":
            # Full checkpoint restore
            if checkpoint_restore_fn is not None:
                checkpoint_restore_fn(model, optimizer)
            else:
                logger.warning(
                    "MoEGambit restart-in-place: CHECKPOINT_RESTART selected "
                    "but no checkpoint_restore_fn provided"
                )
        else:
            # Hybrid: restore dense from peer, expert from checkpoint
            if dense_restore_fn is not None:
                dense_restore_fn(model, self._classification)

            if expert_restore_fn is not None:
                expert_restore_fn(model, self._classification)

        # Phase: WEIGHTS_RESTORING → WEIGHTS_READY
        self._transition_to(
            RestartInPlaceRecoveryState.WEIGHTS_READY,
            step=step,
        )

        # Phase: WEIGHTS_READY → OPTIMIZER_RESTORING
        self._transition_to(
            RestartInPlaceRecoveryState.OPTIMIZER_RESTORING,
            step=step,
        )

        if path == "CHECKPOINT_RESTART":
            # Checkpoint already restored optimizer state
            pass
        else:
            # Hybrid: restore optimizer states
            if dense_optimizer_restore_fn is not None:
                dense_optimizer_restore_fn(optimizer, self._classification)

            if expert_optimizer_restore_fn is not None:
                expert_optimizer_restore_fn(optimizer, self._classification)

        # Phase: OPTIMIZER_RESTORING → FULLY_RECOVERED
        self._transition_to(
            RestartInPlaceRecoveryState.FULLY_RECOVERED,
            step=step,
        )

        logger.warning(
            "MoEGambit restart-in-place: recovery completed via %s at step %d",
            path, step,
        )
        return path

    # -----------------------------------------------------------------
    # Phase 3: Verification + reintegration
    # -----------------------------------------------------------------

    def verify_and_reintegrate(
        self,
        model,
        optimizer=None,
        step: int = -1,
        *,
        check_optimizer: bool = True,
    ) -> VerificationResult:
        """Verify recovery and transition to HEALTHY.

        Transitions: FULLY_RECOVERED → HEALTHY

        Args:
            model: The model.
            optimizer: Optional optimizer.
            step: Training step.
            check_optimizer: Whether to verify optimizer states.

        Returns:
            VerificationResult.

        Raises:
            RecoveryVerificationError: If verification fails.
        """
        result = verify_recovery(
            model,
            optimizer=optimizer,
            classification=self._classification,
            check_optimizer=check_optimizer and optimizer is not None,
        )

        # Only transition if verification passed
        self._transition_to(
            RestartInPlaceRecoveryState.HEALTHY,
            step=step,
        )

        logger.warning(
            "MoEGambit restart-in-place: verification PASSED, "
            "transitioned to HEALTHY at step %d",
            step,
        )
        return result

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return list(self._event_log)

    @property
    def classification(self):
        return self._classification

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self._state.name,
            "recovery_path": self._recovery_path,
            "fault_step": self._fault_step,
            "gap_threshold": self._gap_threshold,
            "checkpoint_iteration": self._checkpoint_iteration,
            "num_events": len(self._event_log),
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = RestartInPlaceRecoveryState.HEALTHY
        self._recovery_path = ""
        self._fault_step = -1
        self._event_log.clear()
        self._classification = None
