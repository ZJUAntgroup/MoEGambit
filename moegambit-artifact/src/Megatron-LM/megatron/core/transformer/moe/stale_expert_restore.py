# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stale Expert Restore from Checkpoint (Step 10).

After a replacement rank comes online and receives current dense parameters
from a healthy DP peer (Step 9), the **expert parameters** still need to be
restored.  Selective expert weight recovery from the most recent distributed checkpoint.

Recovery flow
-------------
::

    RecoveryManifest (from checkpoint dir)
            │
            ▼
    identify_experts_to_restore(manifest, failed_rank)
            │  → list of (layer_id, expert_id) with checkpoint locations
            ▼
    restore_expert_weights(model, restore_plan, load_fn)
            │
            ├─ for each affected expert:
            │     load_fn(weight_location) → tensor
            │     copy tensor into model expert parameters
            │
            ├─ health manager: UNAVAILABLE → STALE_RUNNABLE
            │
            ├─ expert directory: update recovery_state
            │
            └─ optimizer update barrier: block optimizer step for stale experts

State flow
----------
::

    fault detected
        │
        ▼
    UNAVAILABLE  (router excludes, no forward/backward)
        │
        ▼  checkpoint restore (this module)
    STALE_RUNNABLE  (router includes, forward ✓, backward ✓, optimizer step ✗)
        │
        ▼  (future step: deferred optimizer state loading)
    FULLY_RECOVERED  (optimizer step ✓)
        │
        ▼  (safe barrier)
    HEALTHY

Update barrier semantics
------------------------
When an expert is in STALE_RUNNABLE state:

* **Forward**: ALLOWED — the expert participates in token dispatch and
  computes activations.  Weights are stale but usable.
* **Backward**: ALLOWED — gradients flow through the expert normally.
  This is necessary for the rest of the model to train correctly.
* **Optimizer step**: BLOCKED — the expert's parameters are NOT updated
  by the optimizer.  This prevents corrupting the stale weights with
  gradient updates computed from a mismatched optimizer state.

The barrier is implemented as a set of parameter names that the optimizer
should skip.  The caller is responsible for checking this set before
applying optimizer updates.

Scope (v1)
----------
* ✅ Selective expert weight restore from checkpoint
* ✅ Health manager state transition (UNAVAILABLE → STALE_RUNNABLE)
* ✅ Expert directory update
* ✅ Optimizer update barrier for stale experts
* ❌ Actual Megatron dist_checkpointing integration (uses callback)
* ❌ Deferred optimizer state loading (later step)
* ❌ Automatic STALE_RUNNABLE → FULLY_RECOVERED promotion (later step)
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.moe.async_recovery_worker import (
        AsyncRecoveryWorker,
    )

logger = logging.getLogger(__name__)


# =====================================================================
# Layer id helpers
# =====================================================================

def _directory_layer_id_from_restore_entry(
    directory: Any,
    layer_id: int,
    *,
    layer_id_base: int = 1,
) -> int:
    """Map a restore-plan layer id to ActiveExpertDirectory's 0-based id.

    moegambit runtime restore plans use 1-based global MoE layer ids because the
    checkpoint loader maps them back to PP-local checkpoint keys.  Manifest
    plans may already be 0-based.  The active expert directory is built from
    ``range(num_layers)``, so its valid ids are 0..num_layers-1.
    """
    num_layers = getattr(directory, "num_layers", None)
    if callable(num_layers):
        num_layers = num_layers()

    if (
        layer_id_base == 1
        and isinstance(num_layers, int)
        and 1 <= layer_id <= num_layers
    ):
        return layer_id - 1

    return layer_id


# =====================================================================
# Expert restore plan
# =====================================================================

@dataclass
class ExpertRestoreEntry:
    """Describes a single expert that needs to be restored from checkpoint."""

    layer_id: int
    """MoE layer index."""

    expert_id: int
    """Global expert index."""

    checkpoint_step: int = -1
    """Training step of the checkpoint containing this expert's weights."""

    checkpoint_dir: str = ""
    """Path to the checkpoint directory."""

    weight_location: str = ""
    """Key/path reference for the weight shard in the checkpoint."""

    optimizer_location: str = ""
    """Key/path reference for the optimizer state shard (for future use)."""

    host_rank: int = -1
    """The rank that will host this expert after restore."""

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.expert_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_dir": self.checkpoint_dir,
            "weight_location": self.weight_location,
            "optimizer_location": self.optimizer_location,
            "host_rank": self.host_rank,
        }


@dataclass
class ExpertRestorePlan:
    """A plan for restoring multiple experts from checkpoint."""

    entries: List[ExpertRestoreEntry] = field(default_factory=list)
    """Experts to restore."""

    failed_rank: int = -1
    """The rank that failed."""

    replacement_rank: int = -1
    """The replacement rank that will host the restored experts."""

    step: int = -1
    """Current training step when the plan was created."""

    layer_id_base: int = 1
    """Base of ``ExpertRestoreEntry.layer_id`` values."""

    @property
    def num_experts(self) -> int:
        return len(self.entries)

    @property
    def affected_layers(self) -> List[int]:
        return sorted(set(e.layer_id for e in self.entries))

    @property
    def affected_expert_ids(self) -> List[int]:
        return sorted(set(e.expert_id for e in self.entries))

    def get_entries_for_layer(self, layer_id: int) -> List[ExpertRestoreEntry]:
        return [e for e in self.entries if e.layer_id == layer_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "step": self.step,
            "layer_id_base": self.layer_id_base,
            "num_experts": self.num_experts,
            "entries": [e.to_dict() for e in self.entries],
        }


# =====================================================================
# Expert restore result
# =====================================================================

@dataclass
class ExpertRestoreResult:
    """Result of an expert restore operation."""

    success: bool = False
    """Whether all experts were restored successfully."""

    num_restored: int = 0
    """Number of experts successfully restored."""

    num_failed: int = 0
    """Number of experts that failed to restore."""

    num_state_transitions: int = 0
    """Number of health manager state transitions performed."""

    num_directory_updates: int = 0
    """Number of expert directory entries updated."""

    num_barrier_params: int = 0
    """Number of parameters added to the optimizer update barrier."""

    optimizer_load_submitted: int = 0
    """Number of optimizer state load requests submitted to DeferredOptimizerLoader."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the restore operation."""

    errors: List[str] = field(default_factory=list)
    """Error messages for failed restores."""

    restored_experts: List[Tuple[int, int]] = field(default_factory=list)
    """(layer_id, expert_id) pairs that were successfully restored."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "num_restored": self.num_restored,
            "num_failed": self.num_failed,
            "num_state_transitions": self.num_state_transitions,
            "num_directory_updates": self.num_directory_updates,
            "num_barrier_params": self.num_barrier_params,
            "optimizer_load_submitted": self.optimizer_load_submitted,
            "elapsed_seconds": self.elapsed_seconds,
            "errors": list(self.errors),
            "restored_experts": list(self.restored_experts),
        }


# =====================================================================
# Optimizer update barrier
# =====================================================================

class OptimizerUpdateBarrier:
    """Tracks which expert parameters should be skipped during optimizer step.

    When an expert is in STALE_RUNNABLE state, its parameters should NOT
    be updated by the optimizer because the optimizer state (momentum,
    variance) is not yet loaded.  Applying gradient updates with stale/zero
    optimizer state would corrupt the weights.

    The barrier maintains a set of parameter names.  The training loop
    should check ``is_blocked(param_name)`` before applying optimizer
    updates.

    Usage::

        barrier = get_optimizer_update_barrier()

        # After restoring expert weights:
        barrier.block_expert_params(model, expert_ids=[2, 3])

        # In optimizer step:
        for name, param in model.named_parameters():
            if barrier.is_blocked(name):
                continue  # skip this param
            optimizer.step(param)

        # After deferred optimizer state loading:
        barrier.unblock_expert_params(expert_ids=[2, 3])
    """

    def __init__(self) -> None:
        self._blocked_params: Set[str] = set()
        self._blocked_expert_ids: Set[int] = set()
        self._block_step: int = -1

    def block_params(self, param_names: List[str], step: int = -1) -> None:
        """Add parameter names to the blocked set."""
        self._blocked_params.update(param_names)
        if step >= 0:
            self._block_step = step
        logger.info(
            "MoEGambit optimizer barrier: blocked %d params (total %d)",
            len(param_names), len(self._blocked_params),
        )

    def block_expert_params(
        self,
        model,
        expert_ids: List[int],
        step: int = -1,
    ) -> int:
        """Block optimizer updates for all parameters belonging to given experts.

        Expert parameters are identified by:
        1. ``param.allreduce == False`` (expert parallel parameter)
        2. Parameter name contains 'experts' (naming convention)

        Args:
            model: The model.
            expert_ids: Global expert IDs to block.
            step: Training step.

        Returns:
            Number of parameters blocked.
        """
        blocked = []
        expert_id_set = set(expert_ids)
        self._blocked_expert_ids.update(expert_ids)

        for name, param in model.named_parameters():
            is_expert = not getattr(param, 'allreduce', True)
            if not is_expert:
                continue
            # Check if this param belongs to one of the blocked experts
            # Expert params typically have names like:
            #   layers.X.moe.experts.weight1, layers.X.moe.experts.weight2
            # In Megatron, expert weights are stored as a single tensor
            # covering all local experts, so we block the entire param
            # if ANY of the local experts are in the blocked set.
            if 'experts' in name or 'expert' in name:
                blocked.append(name)

        self._blocked_params.update(blocked)
        if step >= 0:
            self._block_step = step

        logger.info(
            "MoEGambit optimizer barrier: blocked %d expert params for "
            "expert_ids=%s (total blocked=%d)",
            len(blocked), sorted(expert_ids), len(self._blocked_params),
        )
        return len(blocked)

    def is_blocked(self, param_name: str) -> bool:
        """Check if a parameter should be skipped during optimizer step."""
        return param_name in self._blocked_params

    def is_expert_blocked(self, expert_id: int) -> bool:
        """Check if an expert's parameters are blocked."""
        return expert_id in self._blocked_expert_ids

    def unblock_params(self, param_names: List[str]) -> None:
        """Remove parameter names from the blocked set."""
        self._blocked_params -= set(param_names)

    def unblock_expert_params(self, expert_ids: List[int]) -> None:
        """Remove all blocked params for given expert IDs.

        This removes expert IDs from the blocked set and clears any
        param names that were blocked for those experts.
        """
        self._blocked_expert_ids -= set(expert_ids)
        # If no more experts are blocked, clear all blocked params
        if not self._blocked_expert_ids:
            self._blocked_params.clear()
        logger.info(
            "MoEGambit optimizer barrier: unblocked expert_ids=%s "
            "(remaining blocked experts=%d, params=%d)",
            sorted(expert_ids), len(self._blocked_expert_ids),
            len(self._blocked_params),
        )

    def unblock_all(self) -> None:
        """Clear all blocked parameters."""
        self._blocked_params.clear()
        self._blocked_expert_ids.clear()
        self._block_step = -1

    @property
    def num_blocked_params(self) -> int:
        return len(self._blocked_params)

    @property
    def num_blocked_experts(self) -> int:
        return len(self._blocked_expert_ids)

    @property
    def blocked_param_names(self) -> FrozenSet[str]:
        return frozenset(self._blocked_params)

    @property
    def blocked_expert_ids(self) -> FrozenSet[int]:
        return frozenset(self._blocked_expert_ids)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_blocked_params": self.num_blocked_params,
            "num_blocked_experts": self.num_blocked_experts,
            "block_step": self._block_step,
            "blocked_expert_ids": sorted(self._blocked_expert_ids),
        }

    def reset(self) -> None:
        self.unblock_all()

    def __repr__(self) -> str:
        return (
            f"OptimizerUpdateBarrier(blocked_params={self.num_blocked_params}, "
            f"blocked_experts={self.num_blocked_experts})"
        )


# =====================================================================
# Plan construction from manifest
# =====================================================================

def identify_experts_to_restore(
    manifest,
    failed_rank: int,
    replacement_rank: int = -1,
    *,
    target_states: Optional[Set[str]] = None,
) -> ExpertRestorePlan:
    """Identify which experts need to be restored from a recovery manifest.

    Scans the manifest for experts that were hosted on the failed rank
    and are in a recoverable state.

    Args:
        manifest: A ``RecoveryManifest`` instance.
        failed_rank: The rank that failed.
        replacement_rank: The rank that will host the restored experts.
        target_states: Set of recovery states to consider for restore.
            Defaults to ``{"HEALTHY", "STALE_RUNNABLE", "FULLY_RECOVERED"}``.
            (We restore from any state that had valid weights at checkpoint time.)

    Returns:
        An ``ExpertRestorePlan``.
    """
    if target_states is None:
        target_states = {"HEALTHY", "STALE_RUNNABLE", "FULLY_RECOVERED"}

    plan = ExpertRestorePlan(
        failed_rank=failed_rank,
        replacement_rank=replacement_rank,
        step=manifest.step,
        layer_id_base=0,
    )

    for entry in manifest.entries:
        if entry.host_rank != failed_rank:
            continue
        if entry.recovery_state not in target_states:
            continue

        restore_entry = ExpertRestoreEntry(
            layer_id=entry.layer_id,
            expert_id=entry.expert_id,
            checkpoint_step=entry.checkpoint_step,
            checkpoint_dir=manifest.checkpoint_dir,
            weight_location=entry.weight_location,
            optimizer_location=entry.optimizer_location,
            host_rank=replacement_rank if replacement_rank >= 0 else entry.host_rank,
        )
        plan.entries.append(restore_entry)

    logger.info(
        "MoEGambit: identified %d experts to restore for failed_rank=%d "
        "(replacement=%d, checkpoint_step=%d)",
        plan.num_experts, failed_rank, replacement_rank, manifest.step,
    )
    return plan


# =====================================================================
# Core restore logic
# =====================================================================

def restore_expert_weights(
    model,
    plan: ExpertRestorePlan,
    *,
    load_fn: Optional[Callable] = None,
    health_managers: Optional[Dict[int, Any]] = None,
    directory=None,
    barrier: Optional[OptimizerUpdateBarrier] = None,
    step: int = -1,
) -> ExpertRestoreResult:
    """Restore expert weights from checkpoint and transition to STALE_RUNNABLE.

    This is the core restore function that:
    1. Loads expert weights from checkpoint (via ``load_fn`` callback).
    2. Transitions health manager state: UNAVAILABLE → STALE_RUNNABLE.
    3. Updates expert directory recovery state.
    4. Adds expert params to optimizer update barrier.

    Args:
        model: The model whose expert parameters to restore.
        plan: The restore plan (from ``identify_experts_to_restore``).
        load_fn: Callback to load expert weights from checkpoint.
            Signature: ``load_fn(entry: ExpertRestoreEntry, model) -> bool``.
            Returns ``True`` on success.  If ``None``, weights are not
            actually loaded (dry-run mode for testing).
        health_managers: Dict mapping layer_id → ExpertHealthManager.
            If ``None``, attempts to use the global registry.
        directory: An ``ActiveExpertDirectory`` instance.
            If ``None``, attempts to use the global singleton.
        barrier: An ``OptimizerUpdateBarrier`` instance.
            If ``None``, attempts to use the global singleton.
        step: Current training step.

    Returns:
        An ``ExpertRestoreResult``.
    """
    start_time = time.monotonic()
    result = ExpertRestoreResult()

    if not plan.entries:
        result.success = True
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    effective_step = step if step >= 0 else plan.step

    # --- Phase 1: Load expert weights ---
    for entry in plan.entries:
        try:
            if load_fn is not None:
                success = load_fn(entry, model)
                if not success:
                    result.num_failed += 1
                    result.errors.append(
                        f"load_fn returned False for expert "
                        f"(layer={entry.layer_id}, id={entry.expert_id})"
                    )
                    continue
            else:
                # Dry-run mode: check if parameters contain NaN sentinel.
                # If they do, it means tensors have been invalidated and
                # we must NOT silently succeed — a real load_fn is required.
                import torch
                has_nan_sentinel = False
                for name, param in model.named_parameters():
                    is_expert = not getattr(param, 'allreduce', True)
                    if is_expert and torch.isnan(param.data).any():
                        has_nan_sentinel = True
                        break
                if has_nan_sentinel:
                    result.num_failed += 1
                    result.errors.append(
                        f"Dry-run mode but expert params contain NaN sentinel "
                        f"(layer={entry.layer_id}, id={entry.expert_id}). "
                        f"A real load_fn is required for recovery."
                    )
                    continue
            # If no load_fn and no NaN, treat as dry-run success
            result.num_restored += 1
            result.restored_experts.append((entry.layer_id, entry.expert_id))
        except Exception as e:
            result.num_failed += 1
            result.errors.append(
                f"Exception restoring expert (layer={entry.layer_id}, "
                f"id={entry.expert_id}): {e}"
            )

    # --- Phase 2: Health manager transitions ---
    if health_managers is None:
        try:
            from megatron.core.transformer.moe.expert_health_manager import (
                _MANAGER_REGISTRY,
            )
            health_managers = _MANAGER_REGISTRY
        except ImportError:
            health_managers = {}

    # Group restored experts by layer
    restored_by_layer: Dict[int, List[int]] = {}
    for layer_id, expert_id in result.restored_experts:
        restored_by_layer.setdefault(layer_id, []).append(expert_id)

    for layer_id, expert_ids in restored_by_layer.items():
        mgr = health_managers.get(layer_id)
        if mgr is None:
            continue
        try:
            mgr.mark_stale_runnable(expert_ids, step=effective_step)
            result.num_state_transitions += len(expert_ids)
        except Exception as e:
            logger.warning(
                "MoEGambit: health manager transition failed for layer %d, "
                "experts %s: %s", layer_id, expert_ids, e,
            )

    # --- Phase 3: Update expert directory ---
    if directory is None:
        try:
            from megatron.core.transformer.moe.expert_directory import (
                get_active_expert_directory,
            )
            directory = get_active_expert_directory()
        except ImportError:
            directory = None

    if directory is not None:
        for layer_id, expert_id in result.restored_experts:
            directory_layer_id = _directory_layer_id_from_restore_entry(
                directory,
                layer_id,
                layer_id_base=getattr(plan, "layer_id_base", 1),
            )
            try:
                directory.update_recovery_state(
                    directory_layer_id, expert_id, "STALE_RUNNABLE",
                )
                if plan.replacement_rank >= 0:
                    directory.update_host_rank(
                        directory_layer_id, expert_id, plan.replacement_rank,
                    )
                result.num_directory_updates += 1
            except Exception as e:
                logger.warning(
                    "MoEGambit: directory update failed for expert "
                    "(restore_layer=%d, directory_layer=%d, id=%d): %s",
                    layer_id, directory_layer_id, expert_id, e,
                )

    # --- Phase 4: Optimizer update barrier ---
    if barrier is None:
        barrier = get_optimizer_update_barrier()

    if model is not None and result.restored_experts:
        all_expert_ids = sorted(set(eid for _, eid in result.restored_experts))
        blocked = barrier.block_expert_params(
            model, expert_ids=all_expert_ids, step=effective_step,
        )
        result.num_barrier_params = blocked

    # --- Finalize ---
    result.success = result.num_failed == 0
    result.elapsed_seconds = time.monotonic() - start_time

    logger.warning(
        "MoEGambit stale expert restore: %s — restored %d/%d experts "
        "(transitions=%d, directory=%d, barrier=%d, %.2fs)",
        "SUCCESS" if result.success else "PARTIAL",
        result.num_restored, result.num_restored + result.num_failed,
        result.num_state_transitions, result.num_directory_updates,
        result.num_barrier_params, result.elapsed_seconds,
    )
    return result


# =====================================================================
# High-level coordinator
# =====================================================================

class StaleExpertRestoreCoordinator:
    """Coordinates stale expert restore from checkpoint.

    This is the high-level coordinator that:
    1. Builds a restore plan from the recovery manifest.
    2. Executes the restore (load weights, transition states).
    3. Tracks results for auditing.
    """

    def __init__(self) -> None:
        self._plans: List[ExpertRestorePlan] = []
        self._results: List[ExpertRestoreResult] = []
        self._async_plan_lock = threading.Lock()
        self._async_plan_signatures: Set[Tuple[Any, ...]] = set()

    def plan_restore(
        self,
        manifest,
        failed_rank: int,
        replacement_rank: int = -1,
    ) -> ExpertRestorePlan:
        """Build a restore plan from a recovery manifest."""
        plan = identify_experts_to_restore(
            manifest, failed_rank, replacement_rank,
        )
        self._plans.append(plan)
        return plan

    def execute_restore(
        self,
        model,
        plan: ExpertRestorePlan,
        *,
        load_fn: Optional[Callable] = None,
        health_managers: Optional[Dict[int, Any]] = None,
        directory=None,
        barrier: Optional[OptimizerUpdateBarrier] = None,
        step: int = -1,
    ) -> ExpertRestoreResult:
        """Execute a restore plan (synchronous)."""
        result = restore_expert_weights(
            model, plan,
            load_fn=load_fn,
            health_managers=health_managers,
            directory=directory,
            barrier=barrier,
            step=step,
        )
        self._results.append(result)
        return result

    def execute_restore_async(
        self,
        plan: ExpertRestorePlan,
        *,
        load_fn: Optional[Callable] = None,
        worker: Optional['AsyncRecoveryWorker'] = None,
        step: int = -1,
    ) -> List[str]:
        """Submit expert weight loads to the async worker (non-blocking).

        This submits each expert's load request to the ``AsyncRecoveryWorker``
        and returns immediately.  The caller is responsible for polling the
        worker for completed loads and applying them to GPU.

        State transitions (health manager, directory, barrier) are NOT
        performed here — they must be done after the async loads complete
        and the tensors are copied to GPU.

        Args:
            plan: The restore plan.
            load_fn: Callback to load expert weights.  Signature:
                ``load_fn(request: AsyncLoadRequest) -> torch.Tensor``.
                The returned tensor should be on CPU.
            worker: The ``AsyncRecoveryWorker`` to submit to.
                If ``None``, uses the global singleton.
            step: Current training step.

        Returns:
            List of request_id strings for tracking.
        """
        if worker is None:
            from megatron.core.transformer.moe.async_recovery_worker import (
                get_async_recovery_worker,
            )
            worker = get_async_recovery_worker()

        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        plan_signature = (
            plan.failed_rank,
            plan.replacement_rank,
            step,
            tuple(sorted(
                (
                    entry.layer_id,
                    entry.expert_id,
                    entry.checkpoint_dir,
                    entry.checkpoint_step,
                    entry.host_rank,
                )
                for entry in plan.entries
            )),
        )
        with self._async_plan_lock:
            if plan_signature in self._async_plan_signatures:
                logger.info(
                    "MoEGambit stale expert restore: duplicate async restore "
                    "plan ignored (failed_rank=%d, replacement=%d, step=%d)",
                    plan.failed_rank, plan.replacement_rank, step,
                )
                return []
            self._async_plan_signatures.add(plan_signature)

        request_ids: List[str] = []
        for entry in plan.entries:
            req = AsyncLoadRequest(
                expert_id=entry.expert_id,
                layer_id=entry.layer_id,
                checkpoint_path=entry.checkpoint_dir,
                weight_location=entry.weight_location,
                request_type="expert_weight",
                load_fn=load_fn,
                metadata={
                    "checkpoint_step": entry.checkpoint_step,
                    "host_rank": entry.host_rank,
                    "plan_step": step,
                },
            )
            req_id = worker.submit_expert_load(req)
            request_ids.append(req_id)

        logger.info(
            "MoEGambit stale expert restore: submitted %d async load requests "
            "for plan (failed_rank=%d, replacement=%d, step=%d)",
            len(request_ids), plan.failed_rank, plan.replacement_rank, step,
        )
        return request_ids

    @property
    def plans(self) -> List[ExpertRestorePlan]:
        return list(self._plans)

    @property
    def results(self) -> List[ExpertRestoreResult]:
        return list(self._results)

    @property
    def last_result(self) -> Optional[ExpertRestoreResult]:
        return self._results[-1] if self._results else None

    def summary(self) -> Dict[str, Any]:
        return {
            "num_plans": len(self._plans),
            "num_results": len(self._results),
            "total_restored": sum(r.num_restored for r in self._results),
            "total_failed": sum(r.num_failed for r in self._results),
            "last_success": self._results[-1].success if self._results else None,
        }

    def reset(self) -> None:
        self._plans.clear()
        self._results.clear()

    def __repr__(self) -> str:
        return (
            f"StaleExpertRestoreCoordinator("
            f"plans={len(self._plans)}, "
            f"results={len(self._results)})"
        )


# =====================================================================
# Global singletons
# =====================================================================

_COORDINATOR: Optional[StaleExpertRestoreCoordinator] = None
_BARRIER: Optional[OptimizerUpdateBarrier] = None


def get_stale_expert_restore_coordinator() -> StaleExpertRestoreCoordinator:
    """Get or create the global coordinator singleton."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = StaleExpertRestoreCoordinator()
    return _COORDINATOR


def clear_stale_expert_restore_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.reset()
    _COORDINATOR = None


def get_optimizer_update_barrier() -> OptimizerUpdateBarrier:
    """Get or create the global optimizer update barrier."""
    global _BARRIER
    if _BARRIER is None:
        _BARRIER = OptimizerUpdateBarrier()
    return _BARRIER


def clear_optimizer_update_barrier() -> None:
    """Reset the global barrier (for testing)."""
    global _BARRIER
    if _BARRIER is not None:
        _BARRIER.reset()
    _BARRIER = None
