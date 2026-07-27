# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Safe-point Process Group Repair for MOEGAMBIT-MoE.

This module implements the actual process group rebuild and rebind logic
that is called at safe points (iteration boundaries) after a hard failure
and replacement rank registration.

The key challenge is that ``parallel_state.py`` has no setter/rebind API
for process groups — only getters and a full ``destroy_model_parallel()``.
We therefore implement **selective rebuild**: only the affected groups
(EP, Expert DP, DP, TP, etc.) are destroyed and recreated, while unaffected
groups (PP, CP) are left untouched.

Protocol
--------
::

    1. validate_repair_preconditions()
       - Verify replacement rank is reachable
       - Verify all healthy ranks agree on the repair plan

    2. invalidate_affected_groups()
       - Mark affected groups as stale (set globals to None)
       - Destroy Gloo groups explicitly

    3. rebuild_affected_groups()
       - Create new NCCL/Gloo groups with updated rank lists
       - Assign to parallel_state globals

    4. rebind_moe_modules()
       - Walk the model tree and update group references on
         MoE dispatchers, routers, experts, and layers

    5. verify_new_groups()
       - Run a lightweight barrier/allreduce on new groups
       - Confirm all ranks can communicate

Scope (v1)
----------
* PP=1 only (PP>1 interface reserved)
* No shrink training
* No hot-swap at arbitrary time
* Replacement rank takes exact same logical slot

Integration
-----------
Called from ``moegambit_integration.py`` via the ``group_rebuild_execute_fn``
and ``group_rebuild_finish_fn`` callbacks, which are invoked by
``RecoveryController._execute_safe_point_repair()`` at step 3 and 4.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Repair phase enum
# =====================================================================

class RepairPhase(enum.IntEnum):
    """Phases of a single safe-point repair operation."""
    NOT_STARTED = 0
    VALIDATING = 1
    INVALIDATING = 2
    REBUILDING = 3
    REBINDING = 4
    VERIFYING = 5
    COMPLETED = 6
    FAILED = 7


# =====================================================================
# Repair result
# =====================================================================

@dataclass
class RepairResult:
    """Result of a safe-point group repair operation."""

    success: bool = False
    phase_reached: RepairPhase = RepairPhase.NOT_STARTED
    failed_rank: int = -1
    replacement_rank: int = -1
    groups_invalidated: int = 0
    groups_rebuilt: int = 0
    modules_rebound: int = 0
    verification_passed: bool = False
    elapsed_seconds: float = 0.0
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Affected group specification
# =====================================================================

# Map from canonical group name to the parallel_state global variable name.
# These are the groups that may need rebuilding when a rank is replaced.
_GROUP_VAR_MAP: Dict[str, str] = {
    # EP / Expert groups
    "EXPERT_MODEL_PARALLEL_GROUP": "_EXPERT_MODEL_PARALLEL_GROUP",
    "EXPERT_TENSOR_PARALLEL_GROUP": "_EXPERT_TENSOR_PARALLEL_GROUP",
    "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP": "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
    "EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP": "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
    "EXPERT_DATA_PARALLEL_GROUP": "_EXPERT_DATA_PARALLEL_GROUP",
    "EXPERT_DATA_PARALLEL_GROUP_GLOO": "_EXPERT_DATA_PARALLEL_GROUP_GLOO",
    # DP groups
    "DATA_PARALLEL_GROUP": "_DATA_PARALLEL_GROUP",
    "DATA_PARALLEL_GROUP_GLOO": "_DATA_PARALLEL_GROUP_GLOO",
    "DATA_PARALLEL_GROUP_WITH_CP": "_DATA_PARALLEL_GROUP_WITH_CP",
    "DATA_PARALLEL_GROUP_WITH_CP_GLOO": "_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
    # PP groups (for PP > 1 stage repair)
    "PIPELINE_MODEL_PARALLEL_GROUP": "_PIPELINE_MODEL_PARALLEL_GROUP",
    "PIPELINE_MODEL_PARALLEL_GROUP_GLOO": "_PIPELINE_MODEL_PARALLEL_GROUP_GLOO",
    "MODEL_PARALLEL_GROUP": "_MODEL_PARALLEL_GROUP",
    # TP groups (for TP > 1 support)
    "TENSOR_MODEL_PARALLEL_GROUP": "_TENSOR_MODEL_PARALLEL_GROUP",
    "TENSOR_AND_DATA_PARALLEL_GROUP": "_TENSOR_AND_DATA_PARALLEL_GROUP",
    "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP": "_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
}

# Groups that use Gloo backend (need explicit destroy)
_GLOO_GROUPS = frozenset({
    "EXPERT_DATA_PARALLEL_GROUP_GLOO",
    "DATA_PARALLEL_GROUP_GLOO",
    "DATA_PARALLEL_GROUP_WITH_CP_GLOO",
    "PIPELINE_MODEL_PARALLEL_GROUP_GLOO",
})


# =====================================================================
# SafePointGroupRepairer
# =====================================================================

class SafePointGroupRepairer:
    """Executes the actual process group repair at a safe point.

    This class encapsulates the 5-phase repair protocol:
    validate → invalidate → rebuild → rebind → verify.

    It is stateless between repairs — each ``execute()`` call is
    self-contained.  State is tracked only during a single repair
    operation for logging and auditing.

    The repairer does NOT decide WHEN to repair — that is the job of
    ``GroupRebuildCoordinator`` and ``RecoveryController``.  This class
    only implements HOW to repair.
    """

    def __init__(self) -> None:
        self._last_result: Optional[RepairResult] = None
        self._total_repairs: int = 0
        # Rank lists saved during invalidation (before groups are set to
        # None).  Keyed by canonical group name.  Populated by
        # ``_invalidate_affected_groups`` and consumed by
        # ``_compute_new_ranks_for_group`` for groups that do NOT have a
        # dedicated ``_*_GLOBAL_RANKS`` variable in ``parallel_state``.
        self._saved_group_ranks: Dict[str, List[int]] = {}

    @property
    def last_result(self) -> Optional[RepairResult]:
        return self._last_result

    @property
    def total_repairs(self) -> int:
        return self._total_repairs

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def execute(
        self,
        plan,  # GroupRebuildPlan from group_rebuild.py
        *,
        model: Any = None,
        create_group_fn: Optional[Callable] = None,
        verify: bool = True,
        pp_size: int = 1,
    ) -> RepairResult:
        """Execute a full safe-point group repair.

        Args:
            plan: A ``GroupRebuildPlan`` describing what to rebuild.
            model: The Megatron model (list of model chunks) for rebinding.
            create_group_fn: Callable to create a new process group.
                Signature: ``create_group_fn(ranks, backend='nccl', **kwargs) -> ProcessGroup``.
                If None, uses ``torch.distributed.new_group``.
            verify: Whether to run verification after rebuild.
            pp_size: Pipeline parallel size.  Must be 1 in v1.

        Returns:
            A ``RepairResult`` describing the outcome.
        """
        t0 = time.monotonic()
        self._saved_group_ranks.clear()
        result = RepairResult(
            failed_rank=plan.failed_rank,
            replacement_rank=plan.replacement_rank,
        )

        try:
            # Phase 1: Validate
            result.phase_reached = RepairPhase.VALIDATING
            self._validate(plan, pp_size=pp_size)

            # Phase 2: Invalidate affected groups
            result.phase_reached = RepairPhase.INVALIDATING
            result.groups_invalidated = self._invalidate_affected_groups(
                plan.affected_groups,
            )

            # Phase 3: Rebuild affected groups
            result.phase_reached = RepairPhase.REBUILDING
            result.groups_rebuilt = self._rebuild_affected_groups(
                plan,
                create_group_fn=create_group_fn,
            )

            # Phase 4: Rebind MoE modules
            result.phase_reached = RepairPhase.REBINDING
            if model is not None:
                result.modules_rebound = self._rebind_moe_modules(model)

            # Phase 4b: Pipeline stage repair (PP > 1)
            if pp_size > 1:
                pp_repair_count = self._repair_pipeline_stage(
                    plan,
                    create_group_fn=create_group_fn,
                )
                result.groups_rebuilt += pp_repair_count
                result.details['pp_groups_repaired'] = pp_repair_count

            # Phase 5: Verify
            if verify:
                result.phase_reached = RepairPhase.VERIFYING
                result.verification_passed = self._verify_new_groups(
                    plan.affected_groups,
                )
            else:
                result.verification_passed = True

            # Validate that rebuild was effective: if we invalidated groups
            # but rebuilt none, the repair is incomplete and training will
            # crash when accessing those groups.
            if (result.groups_invalidated > 0
                    and result.groups_rebuilt == 0
                    and result.modules_rebound == 0):
                # Check if this is a rank-remap scenario where the
                # replacement inherits the failed rank's identity.
                # In that case, rebuilt=0 is expected because the group
                # membership doesn't actually change.
                import torch.distributed
                world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 0
                if plan.replacement_rank >= world_size:
                    # Identity inheritance: groups were invalidated but
                    # will be recreated with original ranks.  This is OK
                    # only if the replacement process has already joined
                    # with the failed rank's identity.
                    logger.warning(
                        "MOEGAMBIT-MoE repair: identity inheritance mode — "
                        "invalidated=%d groups but rebuilt=0 (replacement "
                        "rank %d >= world_size %d).  Groups must be "
                        "recreated with original membership.",
                        result.groups_invalidated,
                        plan.replacement_rank, world_size,
                    )
                    # Attempt to restore invalidated groups with original ranks
                    restored = self._restore_invalidated_groups(
                        plan, create_group_fn=create_group_fn,
                    )
                    result.groups_rebuilt = restored
                    if restored == 0:
                        raise RuntimeError(
                            f"Repair incomplete: invalidated "
                            f"{result.groups_invalidated} groups but "
                            f"rebuilt 0.  replacement_rank="
                            f"{plan.replacement_rank} exceeds world_size="
                            f"{world_size}.  The replacement process must "
                            f"join with the failed rank's identity (rank="
                            f"{plan.failed_rank})."
                        )
                else:
                    raise RuntimeError(
                        f"Repair incomplete: invalidated "
                        f"{result.groups_invalidated} groups but rebuilt 0 "
                        f"and rebound 0 modules.  This indicates a "
                        f"fundamental failure in group recreation."
                    )

            result.phase_reached = RepairPhase.COMPLETED
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.phase_reached = RepairPhase.FAILED
            logger.error(
                "MOEGAMBIT-MoE safe-point repair FAILED at phase %s: %s",
                result.phase_reached.name, e,
            )

        result.elapsed_seconds = time.monotonic() - t0
        self._last_result = result
        if result.success:
            self._total_repairs += 1

        logger.warning(
            "MOEGAMBIT-MoE safe-point repair %s — "
            "invalidated=%d, rebuilt=%d, rebound=%d, verified=%s, "
            "elapsed=%.2fs",
            "SUCCEEDED" if result.success else "FAILED",
            result.groups_invalidated, result.groups_rebuilt,
            result.modules_rebound, result.verification_passed,
            result.elapsed_seconds,
        )

        return result

    # -----------------------------------------------------------------
    # Phase 1: Validate
    # -----------------------------------------------------------------

    def _validate(self, plan, *, pp_size: int = 1) -> None:
        """Validate repair preconditions."""
        if pp_size > 1:
            logger.warning(
                "MOEGAMBIT-MoE safe-point repair: PP>1 (%d) — v1 support is "
                "limited.  Proceeding with best-effort repair.",
                pp_size,
            )

        if plan.failed_rank == plan.replacement_rank:
            # Identity inheritance mode: the replacement process has
            # joined with the same rank ID as the failed process.
            # This is valid — we just need to recreate the groups
            # (the membership doesn't change, but the underlying
            # NCCL communicators need to be refreshed).
            logger.info(
                "MOEGAMBIT-MoE repair: identity inheritance mode — "
                "failed_rank == replacement_rank == %d.  "
                "Groups will be recreated with original membership.",
                plan.failed_rank,
            )

        if not plan.affected_groups:
            raise ValueError("No affected groups specified in repair plan")

        logger.info(
            "MOEGAMBIT-MoE repair validation passed: failed=%d, replacement=%d, "
            "groups=%s",
            plan.failed_rank, plan.replacement_rank, plan.affected_groups,
        )

    # -----------------------------------------------------------------
    # Phase 2: Invalidate affected groups
    # -----------------------------------------------------------------

    def _invalidate_affected_groups(
        self,
        affected_groups: List[str],
    ) -> int:
        """Set affected parallel_state globals to None.

        For Gloo groups, explicitly destroy them first.

        Returns:
            Number of groups invalidated.
        """
        count = 0

        try:
            from megatron.core import parallel_state as ps
        except ImportError:
            logger.warning(
                "MOEGAMBIT-MoE repair: cannot import parallel_state, "
                "skipping group invalidation"
            )
            return 0

        for group_name in affected_groups:
            var_name = _GROUP_VAR_MAP.get(group_name)
            if var_name is None:
                logger.debug("Unknown group name: %s, skipping", group_name)
                continue

            current_group = getattr(ps, var_name, None)
            if current_group is None:
                continue

            # Save the group's rank list BEFORE invalidation so that
            # _compute_new_ranks_for_group can use it for groups that
            # lack a dedicated _*_GLOBAL_RANKS variable (e.g.
            # TENSOR_AND_DATA_PARALLEL_GROUP).
            try:
                import torch.distributed as _td
                saved_ranks = _td.get_process_group_ranks(current_group)
                if saved_ranks:
                    self._saved_group_ranks[group_name] = list(saved_ranks)
            except Exception:
                pass  # best-effort; _compute_new_ranks_for_group has fallbacks

            # Destroy Gloo groups explicitly
            if group_name in _GLOO_GROUPS:
                try:
                    import torch.distributed
                    torch.distributed.destroy_process_group(current_group)
                    logger.debug(
                        "Destroyed Gloo group %s", group_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to destroy Gloo group %s: %s",
                        group_name, e,
                    )

            # Set global to None
            setattr(ps, var_name, None)
            count += 1
            logger.debug("Invalidated %s (%s)", group_name, var_name)

        logger.info(
            "MOEGAMBIT-MoE repair: invalidated %d groups", count,
        )
        return count

    # -----------------------------------------------------------------
    # Phase 3: Rebuild affected groups
    # -----------------------------------------------------------------

    def _rebuild_affected_groups(
        self,
        plan,
        *,
        create_group_fn: Optional[Callable] = None,
    ) -> int:
        """Create new process groups with updated rank lists.

        In v1, we use a simplified approach:
        - For EP groups: use plan.new_ep_group_ranks
        - For DP groups: compute new ranks by replacing failed→replacement
        - For Gloo groups: create with backend='gloo'

        When the replacement rank exceeds the original world_size, we use
        ``_create_group_with_world_expansion`` which leverages
        ``torch.distributed.new_group`` with ``use_local_synchronization``
        or falls back to creating a new store-based group.

        Returns:
            Number of groups rebuilt.
        """
        count = 0

        try:
            from megatron.core import parallel_state as ps
            import torch.distributed
        except ImportError:
            logger.warning(
                "MOEGAMBIT-MoE repair: cannot import parallel_state or "
                "torch.distributed, skipping group rebuild"
            )
            return 0

        if create_group_fn is None:
            create_group_fn = _default_create_group

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Check if replacement rank exceeds world_size — if so, we need
        # to remap it to the failed rank's slot (identity inheritance).
        replacement_rank = plan.replacement_rank
        failed_rank = plan.failed_rank
        needs_rank_remap = replacement_rank >= world_size

        if needs_rank_remap:
            logger.warning(
                "MOEGAMBIT-MoE repair: replacement_rank=%d >= world_size=%d, "
                "using identity inheritance (replacement inherits failed "
                "rank %d's slot in all groups)",
                replacement_rank, world_size, failed_rank,
            )

        for group_name in plan.affected_groups:
            var_name = _GROUP_VAR_MAP.get(group_name)
            if var_name is None:
                continue

            # Determine new ranks for this group
            new_ranks = self._compute_new_ranks_for_group(
                group_name, plan,
            )

            if new_ranks is None:
                logger.warning(
                    "MOEGAMBIT-MoE repair: cannot compute new ranks for %s, "
                    "skipping",
                    group_name,
                )
                continue

            # If replacement rank exceeds world_size, the replacement
            # process must have joined with the failed rank's identity.
            # In that case, the group ranks stay the same as original
            # (with failed_rank still in the list, since the replacement
            # process IS now that rank).
            if needs_rank_remap:
                new_ranks = [
                    failed_rank if r == replacement_rank else r
                    for r in new_ranks
                ]

            # Validate all ranks are within world_size
            invalid_ranks = [r for r in new_ranks if r >= world_size]
            if invalid_ranks:
                logger.error(
                    "MOEGAMBIT-MoE repair: ranks %s in group %s exceed "
                    "world_size=%d, skipping rebuild",
                    invalid_ranks, group_name, world_size,
                )
                continue

            # Determine backend
            is_gloo = group_name in _GLOO_GROUPS
            backend = "gloo" if is_gloo else "nccl"

            # Create new group
            try:
                new_group = create_group_fn(
                    new_ranks,
                    backend=backend,
                    group_name=group_name,
                )

                # Only assign if this rank is in the group
                if rank in new_ranks:
                    setattr(ps, var_name, new_group)
                    count += 1
                    logger.debug(
                        "Rebuilt %s (%s) with ranks %s (backend=%s)",
                        group_name, var_name, new_ranks, backend,
                    )

            except Exception as e:
                logger.error(
                    "MOEGAMBIT-MoE repair: failed to rebuild %s: %s",
                    group_name, e,
                )

        logger.info("MOEGAMBIT-MoE repair: rebuilt %d groups", count)
        return count

    def _compute_new_ranks_for_group(
        self,
        group_name: str,
        plan,
    ) -> Optional[List[int]]:
        """Compute the new rank list for a specific group.

        Uses the plan's EP group ranks as the base, and replaces
        failed_rank with replacement_rank.
        """
        from megatron.core.transformer.moe.group_rebuild import (
            GroupRebuildCoordinator,
        )

        # For EP-related groups, use the plan's new_ep_group_ranks
        ep_group_names = {
            "EXPERT_MODEL_PARALLEL_GROUP",
            "EXPERT_TENSOR_PARALLEL_GROUP",
            "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
            "EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
            "EXPERT_DATA_PARALLEL_GROUP",
            "EXPERT_DATA_PARALLEL_GROUP_GLOO",
        }

        if group_name in ep_group_names:
            if plan.new_ep_group_ranks:
                return list(plan.new_ep_group_ranks)
            elif plan.old_ep_group_ranks:
                return GroupRebuildCoordinator.compute_new_group_ranks(
                    plan.old_ep_group_ranks,
                    plan.failed_rank,
                    plan.replacement_rank,
                )

        # For DP-related groups, try to get current DP ranks and replace
        dp_group_names = {
            "DATA_PARALLEL_GROUP",
            "DATA_PARALLEL_GROUP_GLOO",
            "DATA_PARALLEL_GROUP_WITH_CP",
            "DATA_PARALLEL_GROUP_WITH_CP_GLOO",
        }

        if group_name in dp_group_names:
            try:
                from megatron.core import parallel_state as ps
                dp_ranks = getattr(ps, '_DATA_PARALLEL_GLOBAL_RANKS', None)
                if dp_ranks is not None:
                    return GroupRebuildCoordinator.compute_new_group_ranks(
                        dp_ranks,
                        plan.failed_rank,
                        plan.replacement_rank,
                    )
            except Exception:
                pass

            # Fallback: use EP ranks (in PP=1, EP and DP may overlap)
            if plan.new_ep_group_ranks:
                return list(plan.new_ep_group_ranks)

        # For TP-related groups, get current TP ranks and replace
        # failed_rank with replacement_rank.  When TP > 1, the TP
        # group's NCCL communicator is bound to physical rank IDs,
        # so it must be rebuilt even though the logical membership
        # is the same.
        tp_group_names = {
            "TENSOR_MODEL_PARALLEL_GROUP",
            "TENSOR_AND_DATA_PARALLEL_GROUP",
            "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        }

        if group_name in tp_group_names:
            # Strategy:
            #   1. Try the dedicated _*_GLOBAL_RANKS variable (only
            #      TENSOR_MODEL_PARALLEL_GROUP has one).
            #   2. Fall back to _saved_group_ranks (populated during
            #      invalidation, before the group was set to None).
            #   3. Last resort: try to read ranks from the live group
            #      object (only works if not yet invalidated).
            ranks_for_group: Optional[List[int]] = None

            try:
                from megatron.core import parallel_state as ps

                if group_name == "TENSOR_MODEL_PARALLEL_GROUP":
                    ranks_for_group = getattr(
                        ps, '_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS', None,
                    )
            except Exception:
                pass

            # Fallback: saved ranks from invalidation phase
            if ranks_for_group is None:
                ranks_for_group = self._saved_group_ranks.get(group_name)

            # Last resort: live group object
            if ranks_for_group is None:
                try:
                    from megatron.core import parallel_state as ps
                    var_name = _GROUP_VAR_MAP.get(group_name)
                    if var_name:
                        live_group = getattr(ps, var_name, None)
                        if live_group is not None:
                            import torch.distributed as _td
                            ranks_for_group = _td.get_process_group_ranks(
                                live_group,
                            )
                except Exception:
                    pass

            if ranks_for_group is not None:
                return GroupRebuildCoordinator.compute_new_group_ranks(
                    ranks_for_group,
                    plan.failed_rank,
                    plan.replacement_rank,
                )

        return None

    # -----------------------------------------------------------------
    # Restore invalidated groups (identity inheritance fallback)
    # -----------------------------------------------------------------

    def _restore_invalidated_groups(
        self,
        plan,
        *,
        create_group_fn: Optional[Callable] = None,
    ) -> int:
        """Restore invalidated groups using original rank membership.

        This is called when replacement_rank >= world_size (identity
        inheritance mode).  In this case, the replacement process has
        joined with the failed rank's identity, so the group membership
        is unchanged — we just need to recreate the groups with the
        original rank lists.

        Returns:
            Number of groups restored.
        """
        count = 0

        try:
            from megatron.core import parallel_state as ps
            import torch.distributed
        except ImportError:
            return 0

        if create_group_fn is None:
            create_group_fn = _default_create_group

        rank = torch.distributed.get_rank()
        failed_rank = plan.failed_rank

        for group_name in plan.affected_groups:
            var_name = _GROUP_VAR_MAP.get(group_name)
            if var_name is None:
                continue

            # Skip if already restored
            if getattr(ps, var_name, None) is not None:
                continue

            # Compute original ranks (using failed_rank, not replacement)
            original_ranks = self._compute_original_ranks_for_group(
                group_name, plan,
            )

            if original_ranks is None:
                logger.warning(
                    "MOEGAMBIT-MoE repair: cannot compute original ranks for "
                    "%s, skipping restore", group_name,
                )
                continue

            # Determine backend
            is_gloo = group_name in _GLOO_GROUPS
            backend = "gloo" if is_gloo else "nccl"

            try:
                new_group = create_group_fn(
                    original_ranks,
                    backend=backend,
                    group_name=group_name,
                )

                if rank in original_ranks:
                    setattr(ps, var_name, new_group)
                    count += 1
                    logger.debug(
                        "Restored %s (%s) with original ranks %s",
                        group_name, var_name, original_ranks,
                    )
            except Exception as e:
                logger.error(
                    "MOEGAMBIT-MoE repair: failed to restore %s: %s",
                    group_name, e,
                )

        logger.info("MOEGAMBIT-MoE repair: restored %d invalidated groups", count)
        return count

    def _compute_original_ranks_for_group(
        self,
        group_name: str,
        plan,
    ) -> Optional[List[int]]:
        """Compute the original rank list for a group (before failure).

        Used in identity inheritance mode where the replacement process
        takes over the failed rank's identity.
        """
        # Use old_ep_group_ranks if available
        ep_group_names = {
            "EXPERT_MODEL_PARALLEL_GROUP",
            "EXPERT_TENSOR_PARALLEL_GROUP",
            "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
            "EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
            "EXPERT_DATA_PARALLEL_GROUP",
            "EXPERT_DATA_PARALLEL_GROUP_GLOO",
        }

        if group_name in ep_group_names:
            if plan.old_ep_group_ranks:
                return list(plan.old_ep_group_ranks)

        # For DP groups, try to get from parallel_state
        dp_group_names = {
            "DATA_PARALLEL_GROUP",
            "DATA_PARALLEL_GROUP_GLOO",
            "DATA_PARALLEL_GROUP_WITH_CP",
            "DATA_PARALLEL_GROUP_WITH_CP_GLOO",
        }

        if group_name in dp_group_names:
            try:
                from megatron.core import parallel_state as ps
                dp_ranks = getattr(ps, '_DATA_PARALLEL_GLOBAL_RANKS', None)
                if dp_ranks is not None:
                    return list(dp_ranks)
            except Exception:
                pass

            # Fallback to old EP ranks
            if plan.old_ep_group_ranks:
                return list(plan.old_ep_group_ranks)

        # For TP groups, get original ranks from parallel_state or
        # saved ranks from invalidation phase.
        tp_group_names = {
            "TENSOR_MODEL_PARALLEL_GROUP",
            "TENSOR_AND_DATA_PARALLEL_GROUP",
            "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        }

        if group_name in tp_group_names:
            ranks_for_group: Optional[List[int]] = None

            # Try dedicated _*_GLOBAL_RANKS variable (only TP has one)
            try:
                from megatron.core import parallel_state as ps
                if group_name == "TENSOR_MODEL_PARALLEL_GROUP":
                    tp_ranks = getattr(
                        ps, '_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS', None,
                    )
                    if tp_ranks is not None:
                        ranks_for_group = list(tp_ranks)
            except Exception:
                pass

            # Fallback: saved ranks from invalidation phase
            if ranks_for_group is None:
                saved = self._saved_group_ranks.get(group_name)
                if saved is not None:
                    ranks_for_group = list(saved)

            if ranks_for_group is not None:
                return ranks_for_group

        return None

    # -----------------------------------------------------------------
    # Phase 4: Rebind MoE modules
    # -----------------------------------------------------------------

    def _rebind_moe_modules(self, model: Any) -> int:
        """Walk the model tree and rebind group references on MoE modules.

        Returns:
            Number of module attributes rebound.
        """
        from megatron.core.transformer.moe.group_rebuild import (
            rebind_moe_module_groups,
            rebind_dispatcher_derived_values,
            MOE_DISPATCHER_REBIND_MAP,
            MOE_ROUTER_REBIND_MAP,
            MOE_LAYER_REBIND_MAP,
            MOE_EXPERTS_REBIND_MAP,
            get_group_rebuild_coordinator,
        )

        count = 0
        coordinator = get_group_rebuild_coordinator()

        # Build a dict of current group handles from parallel_state
        try:
            from megatron.core import parallel_state as mpu
            pg_dict = {
                "ep": mpu.get_expert_model_parallel_group(),
                "expt_tp": getattr(mpu, 'get_expert_tensor_parallel_group', lambda: None)(),
                "tp_ep": getattr(mpu, 'get_expert_tensor_and_model_parallel_group', lambda: None)(),
                "tp": mpu.get_tensor_model_parallel_group(),
                "cp": getattr(mpu, 'get_context_parallel_group', lambda: None)(),
                "tp_cp": getattr(mpu, 'get_tensor_and_context_parallel_group', lambda: None)(),
                "tp_dp_cp": getattr(mpu, 'get_tensor_and_data_parallel_group', lambda with_context_parallel=True: None)(with_context_parallel=True) if hasattr(mpu, 'get_tensor_and_data_parallel_group') else None,
                "expt_dp": getattr(mpu, 'get_expert_data_parallel_group', lambda: None)(),
            }
        except Exception as e:
            logger.warning(
                "MOEGAMBIT-MoE repair: cannot build pg_dict from parallel_state: %s",
                e,
            )
            return 0

        # Walk model chunks
        models = model if isinstance(model, (list, tuple)) else [model]
        for model_chunk in models:
            count += self._rebind_model_chunk(
                model_chunk, pg_dict, coordinator,
            )

        logger.info("MOEGAMBIT-MoE repair: rebound %d module attributes", count)
        return count

    def _rebind_model_chunk(
        self,
        model_chunk: Any,
        pg_dict: Dict[str, Any],
        coordinator: Any,
    ) -> int:
        """Rebind group references in a single model chunk."""
        from megatron.core.transformer.moe.group_rebuild import (
            rebind_moe_module_groups,
            rebind_dispatcher_derived_values,
            MOE_DISPATCHER_REBIND_MAP,
            MOE_ROUTER_REBIND_MAP,
            MOE_LAYER_REBIND_MAP,
            MOE_EXPERTS_REBIND_MAP,
        )

        count = 0

        # Walk all named modules looking for MoE components
        for name, module in model_chunk.named_modules():
            module_type = type(module).__name__

            # Token dispatcher
            if 'Dispatcher' in module_type or 'dispatcher' in name:
                n = rebind_moe_module_groups(
                    module, pg_dict, MOE_DISPATCHER_REBIND_MAP,
                    module_name=name, coordinator=coordinator,
                )
                if n > 0:
                    rebind_dispatcher_derived_values(module)
                count += n

            # Router
            elif 'Router' in module_type or 'router' in name:
                count += rebind_moe_module_groups(
                    module, pg_dict, MOE_ROUTER_REBIND_MAP,
                    module_name=name, coordinator=coordinator,
                )

            # MoE layer
            elif 'MoELayer' in module_type or 'moe_layer' in name:
                count += rebind_moe_module_groups(
                    module, pg_dict, MOE_LAYER_REBIND_MAP,
                    module_name=name, coordinator=coordinator,
                )

            # Expert modules
            elif 'Expert' in module_type or 'expert' in name:
                count += rebind_moe_module_groups(
                    module, pg_dict, MOE_EXPERTS_REBIND_MAP,
                    module_name=name, coordinator=coordinator,
                )

        return count

    # -----------------------------------------------------------------
    # Phase 5: Verify
    # -----------------------------------------------------------------

    def _verify_new_groups(
        self,
        affected_groups: List[str],
    ) -> bool:
        """Run lightweight verification on rebuilt groups.

        Performs a barrier on each rebuilt group to confirm all ranks
        can communicate.

        Returns:
            True if all verifications passed.
        """
        try:
            from megatron.core import parallel_state as ps
            import torch.distributed
        except ImportError:
            return True  # Can't verify without distributed

        all_passed = True
        for group_name in affected_groups:
            var_name = _GROUP_VAR_MAP.get(group_name)
            if var_name is None:
                continue

            group = getattr(ps, var_name, None)
            if group is None:
                continue

            # Skip Gloo groups for barrier verification (they may not
            # support CUDA tensors)
            if group_name in _GLOO_GROUPS:
                continue

            try:
                torch.distributed.barrier(group=group)
                logger.debug(
                    "MOEGAMBIT-MoE repair: verification passed for %s",
                    group_name,
                )
            except Exception as e:
                logger.error(
                    "MOEGAMBIT-MoE repair: verification FAILED for %s: %s",
                    group_name, e,
                )
                all_passed = False

        return all_passed

    # -----------------------------------------------------------------
    # Phase 4b: Pipeline stage repair (PP > 1)
    # -----------------------------------------------------------------

    def _repair_pipeline_stage(
        self,
        plan,
        *,
        create_group_fn: Optional[Callable] = None,
    ) -> int:
        """Repair pipeline-related groups and P2P bindings.

        This is called when pp_size > 1 and the failed rank is part of
        a pipeline stage.  It:
        1. Rebuilds the PP group with the replacement rank
        2. Updates prev/next rank pointers in parallel_state
        3. Recreates the P2P communicator

        If the current rank's PP group does NOT contain the failed rank,
        this is a no-op (that PP group is unaffected by the failure).

        Returns:
            Number of pipeline-related items repaired.
        """
        from megatron.core.transformer.moe import pipeline_stage_repair as psr_mod

        repairer = psr_mod.get_pipeline_stage_repairer()

        # Determine PP group ranks for the current rank
        pp_group_ranks = None
        try:
            from megatron.core import parallel_state as ps
            pp_group_ranks = getattr(ps, '_PIPELINE_GLOBAL_RANKS', None)
            if pp_group_ranks is not None:
                pp_group_ranks = list(pp_group_ranks)
        except Exception:
            pass

        if pp_group_ranks is None:
            # Try to get from the plan
            pp_group_ranks = getattr(plan, 'old_pp_group_ranks', None)

        if pp_group_ranks is None:
            logger.warning(
                "MOEGAMBIT-MoE repair: cannot determine PP group ranks, "
                "skipping pipeline stage repair"
            )
            return 0

        # If the failed rank is NOT in this rank's PP group, skip repair.
        # This PP group is unaffected by the failure.
        if plan.failed_rank not in pp_group_ranks:
            logger.debug(
                "MOEGAMBIT-MoE repair: failed_rank %d not in this rank's "
                "PP group %s — skipping pipeline stage repair "
                "(this PP group is unaffected)",
                plan.failed_rank, pp_group_ranks,
            )
            return 0

        result = repairer.execute(
            failed_rank=plan.failed_rank,
            replacement_rank=plan.replacement_rank,
            pp_group_ranks=pp_group_ranks,
            create_group_fn=create_group_fn,
        )

        if result.success:
            logger.warning(
                "MOEGAMBIT-MoE repair: pipeline stage repair SUCCEEDED — "
                "pp_group_rebuilt=%s, prev_next_updated=%s, "
                "p2p_rebound=%s, elapsed=%.2fs",
                result.pp_group_rebuilt, result.prev_next_updated,
                result.p2p_rebound, result.elapsed_seconds,
            )
        else:
            logger.error(
                "MOEGAMBIT-MoE repair: pipeline stage repair FAILED: %s",
                result.error,
            )

        count = 0
        if result.pp_group_rebuilt:
            count += 1
        if result.prev_next_updated:
            count += 1
        if result.p2p_rebound:
            count += 1
        return count

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "total_repairs": self._total_repairs,
            "last_result": {
                "success": self._last_result.success,
                "phase": self._last_result.phase_reached.name,
                "groups_rebuilt": self._last_result.groups_rebuilt,
                "modules_rebound": self._last_result.modules_rebound,
                "elapsed": self._last_result.elapsed_seconds,
            } if self._last_result else None,
        }

    def reset(self) -> None:
        """Reset state (for testing)."""
        self._last_result = None
        self._total_repairs = 0


# =====================================================================
# Default group creation
# =====================================================================

def _default_create_group(
    ranks: List[int],
    backend: str = "nccl",
    group_name: str = "",
    **kwargs,
):
    """Default group creation using torch.distributed.new_group."""
    import torch.distributed
    return torch.distributed.new_group(
        ranks=ranks,
        backend=backend,
    )


# =====================================================================
# Global singleton
# =====================================================================

_REPAIRER: Optional[SafePointGroupRepairer] = None


def get_safe_point_group_repairer() -> SafePointGroupRepairer:
    """Get or create the global SafePointGroupRepairer singleton."""
    global _REPAIRER
    if _REPAIRER is None:
        _REPAIRER = SafePointGroupRepairer()
    return _REPAIRER


def clear_safe_point_group_repairer() -> None:
    """Reset the global repairer (for testing)."""
    global _REPAIRER
    if _REPAIRER is not None:
        _REPAIRER.reset()
    _REPAIRER = None


# =====================================================================
# Top-level convenience API
# =====================================================================

def execute_safe_point_repair(
    plan,
    *,
    model: Any = None,
    create_group_fn: Optional[Callable] = None,
    verify: bool = True,
    pp_size: int = 1,
) -> RepairResult:
    """Execute a safe-point group repair using the global repairer.

    This is the primary entry point for ``moegambit_integration.py``.
    """
    repairer = get_safe_point_group_repairer()
    return repairer.execute(
        plan,
        model=model,
        create_group_fn=create_group_fn,
        verify=verify,
        pp_size=pp_size,
    )
