# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pipeline Stage Repair for MoEGambit (PP > 1).

When a hard failure occurs on a pipeline stage and a replacement rank
takes over, the PP-related process groups and P2P communicator must be
rebuilt so that the pipeline schedule can resume.

Handles:

1. **PP group rebuild** — Replace the failed rank in the PP group and
   recreate the NCCL process group.
2. **Prev/next rank update** — Update the cached prev/next PP rank
   in ``parallel_state``.
3. **P2P communicator rebind** — Destroy and recreate the P2P
   communicator singleton so it picks up the new PP group.
4. **Compound group rebuild** — Rebuild all groups that include the
   PP dimension (e.g., TP-PP, DP-PP, etc.).

Protocol
--------
::

    1. Identify which PP stage the failed rank belongs to
    2. Compute new PP group ranks (replace failed → replacement)
    3. At safe point:
       a. Invalidate old PP group
       b. Rebuild PP group with new ranks
       c. Update prev/next rank cache
       d. Recreate P2P communicator
       e. Rebuild compound groups
       f. Verify with barrier

Scope (v1)
----------
* Single PP group (no virtual pipeline stages in the repair path)
* Replacement rank takes the exact same logical PP stage
* No dynamic PP topology changes
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# PP group variable map
# =====================================================================

# Map from canonical name to parallel_state global variable name
# for PP-related groups.
_PP_GROUP_VAR_MAP: Dict[str, str] = {
    "PIPELINE_MODEL_PARALLEL_GROUP": "_PIPELINE_MODEL_PARALLEL_GROUP",
    "PIPELINE_MODEL_PARALLEL_GROUP_FOR_EMBEDDING": "_EMBEDDING_GROUP",
    "PIPELINE_MODEL_PARALLEL_GROUP_FOR_POSITION_EMBEDDING": "_POSITION_EMBEDDING_GROUP",
    "MODEL_PARALLEL_GROUP": "_MODEL_PARALLEL_GROUP",
    "MODEL_AND_EXPERT_PARALLEL_GROUP": "_MODEL_AND_EXPERT_PARALLEL_GROUP",
    "TENSOR_AND_EXPERT_PARALLEL_GROUP": "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
}

# Compound groups that include PP dimension and may need rebuild
_PP_COMPOUND_GROUPS: List[str] = [
    "MODEL_PARALLEL_GROUP",
    "MODEL_AND_EXPERT_PARALLEL_GROUP",
    "TENSOR_AND_EXPERT_PARALLEL_GROUP",
    "PIPELINE_MODEL_PARALLEL_GROUP_FOR_EMBEDDING",
    "PIPELINE_MODEL_PARALLEL_GROUP_FOR_POSITION_EMBEDDING",
]


# =====================================================================
# Repair result
# =====================================================================

@dataclass
class PipelineStageRepairResult:
    """Result of a pipeline stage repair operation."""

    success: bool = False
    failed_rank: int = -1
    replacement_rank: int = -1
    failed_stage: int = -1
    pp_group_rebuilt: bool = False
    prev_next_updated: bool = False
    p2p_rebound: bool = False
    compound_groups_rebuilt: int = 0
    verification_passed: bool = False
    elapsed_seconds: float = 0.0
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# PipelineStageRepairer
# =====================================================================

class PipelineStageRepairer:
    """Repairs a single pipeline stage after rank replacement.

    PP-specific aspects of safe-point repair
    that are NOT covered by ``SafePointGroupRepairer`` (which focuses
    on EP/DP groups).

    The repairer is stateless between repairs — each ``execute()``
    call is self-contained.
    """

    def __init__(self) -> None:
        self._last_result: Optional[PipelineStageRepairResult] = None
        self._total_repairs: int = 0

    @property
    def last_result(self) -> Optional[PipelineStageRepairResult]:
        return self._last_result

    @property
    def total_repairs(self) -> int:
        return self._total_repairs

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def execute(
        self,
        *,
        failed_rank: int,
        replacement_rank: int,
        pp_group_ranks: List[int],
        create_group_fn: Optional[Callable] = None,
        verify: bool = True,
    ) -> PipelineStageRepairResult:
        """Execute a pipeline stage repair.

        Args:
            failed_rank: The global rank that failed.
            replacement_rank: The global rank of the replacement.
            pp_group_ranks: The current PP group ranks (ordered by stage).
            create_group_fn: Optional callable to create a new process group.
            verify: Whether to run verification after rebuild.

        Returns:
            A PipelineStageRepairResult describing the outcome.
        """
        t0 = time.monotonic()
        result = PipelineStageRepairResult(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
        )

        try:
            # Step 0: Validate
            if failed_rank not in pp_group_ranks:
                raise ValueError(
                    f"failed_rank {failed_rank} not in pp_group_ranks "
                    f"{pp_group_ranks}"
                )
            failed_stage = pp_group_ranks.index(failed_rank)
            result.failed_stage = failed_stage

            # Compute new PP group ranks
            new_pp_ranks = list(pp_group_ranks)
            new_pp_ranks[failed_stage] = replacement_rank

            logger.warning(
                "MoEGambit pipeline repair: stage=%d, failed=%d, "
                "replacement=%d, old_ranks=%s, new_ranks=%s",
                failed_stage, failed_rank, replacement_rank,
                pp_group_ranks, new_pp_ranks,
            )

            # Step 1: Invalidate old PP group
            self._invalidate_pp_group()

            # Step 2: Rebuild PP group
            result.pp_group_rebuilt = self._rebuild_pp_group(
                new_pp_ranks,
                create_group_fn=create_group_fn,
            )

            # Step 3: Update prev/next rank cache
            result.prev_next_updated = self._update_prev_next_ranks(
                new_pp_ranks,
            )

            # Step 4: Rebind P2P communicator
            result.p2p_rebound = self._rebind_p2p_communicator()

            # Step 5: Rebuild compound groups
            result.compound_groups_rebuilt = self._rebuild_compound_groups(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                create_group_fn=create_group_fn,
            )

            # Step 6: Verify
            if verify:
                result.verification_passed = self._verify(new_pp_ranks)
            else:
                result.verification_passed = True

            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(
                "MoEGambit pipeline repair FAILED: %s", e,
            )

        result.elapsed_seconds = time.monotonic() - t0
        self._last_result = result
        if result.success:
            self._total_repairs += 1

        logger.warning(
            "MoEGambit pipeline repair %s — "
            "stage=%d, pp_rebuilt=%s, prev_next=%s, p2p=%s, "
            "compound=%d, verified=%s, elapsed=%.2fs",
            "SUCCEEDED" if result.success else "FAILED",
            result.failed_stage,
            result.pp_group_rebuilt,
            result.prev_next_updated,
            result.p2p_rebound,
            result.compound_groups_rebuilt,
            result.verification_passed,
            result.elapsed_seconds,
        )

        return result

    # -----------------------------------------------------------------
    # Step 1: Invalidate old PP group
    # -----------------------------------------------------------------

    def _invalidate_pp_group(self) -> None:
        """Set the PP group global to None."""
        try:
            from megatron.core import parallel_state as ps
            current = getattr(ps, '_PIPELINE_MODEL_PARALLEL_GROUP', None)
            if current is not None:
                try:
                    import torch.distributed
                    torch.distributed.destroy_process_group(current)
                except Exception as e:
                    logger.warning(
                        "MoEGambit pipeline repair: failed to destroy old "
                        "PP group: %s", e,
                    )
                ps._PIPELINE_MODEL_PARALLEL_GROUP = None
                logger.debug("MoEGambit pipeline repair: old PP group invalidated")
        except ImportError:
            logger.warning(
                "MoEGambit pipeline repair: cannot import parallel_state"
            )

    # -----------------------------------------------------------------
    # Step 2: Rebuild PP group
    # -----------------------------------------------------------------

    def _rebuild_pp_group(
        self,
        new_pp_ranks: List[int],
        create_group_fn: Optional[Callable] = None,
    ) -> bool:
        """Create a new PP group with updated ranks.

        If any rank in new_pp_ranks exceeds world_size, falls back to
        identity inheritance mode where the replacement process has
        already joined with the failed rank's identity, so we recreate
        the group with the original rank list.
        """
        try:
            from megatron.core import parallel_state as ps
            import torch.distributed

            if create_group_fn is None:
                create_group_fn = _default_create_group

            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Check if any rank exceeds world_size
            invalid_ranks = [r for r in new_pp_ranks if r >= world_size]
            if invalid_ranks:
                logger.warning(
                    "MoEGambit pipeline repair: ranks %s exceed world_size=%d "
                    "in PP group.  Using identity inheritance — the "
                    "replacement process must have joined with the failed "
                    "rank's identity.",
                    invalid_ranks, world_size,
                )
                # In identity inheritance mode, the PP group ranks don't
                # actually change — the replacement process IS the failed
                # rank now.  Get the original PP ranks.
                original_pp_ranks = getattr(ps, '_PIPELINE_GLOBAL_RANKS', None)
                if original_pp_ranks is not None:
                    new_pp_ranks = list(original_pp_ranks)
                else:
                    # Remap: replace out-of-range ranks back to failed rank
                    # This assumes there's only one replacement happening
                    failed_rank = invalid_ranks[0] - world_size  # heuristic
                    new_pp_ranks = [
                        r if r < world_size else (r - world_size)
                        for r in new_pp_ranks
                    ]
                    logger.warning(
                        "MoEGambit pipeline repair: remapped PP ranks to %s",
                        new_pp_ranks,
                    )

            new_group = create_group_fn(
                new_pp_ranks,
                backend="nccl",
                group_name="PIPELINE_MODEL_PARALLEL_GROUP",
            )

            if rank in new_pp_ranks:
                ps._PIPELINE_MODEL_PARALLEL_GROUP = new_group
                # Also update the global ranks list
                ps._PIPELINE_GLOBAL_RANKS = new_pp_ranks
                logger.debug(
                    "MoEGambit pipeline repair: PP group rebuilt with "
                    "ranks %s", new_pp_ranks,
                )
                return True
            return False

        except Exception as e:
            logger.error(
                "MoEGambit pipeline repair: failed to rebuild PP group: %s", e,
            )
            return False

    # -----------------------------------------------------------------
    # Step 3: Update prev/next rank cache
    # -----------------------------------------------------------------

    def _update_prev_next_ranks(
        self,
        new_pp_ranks: List[int],
    ) -> bool:
        """Update the cached prev/next PP ranks in parallel_state."""
        try:
            from megatron.core import parallel_state as ps
            import torch.distributed

            rank = torch.distributed.get_rank()

            if rank not in new_pp_ranks:
                return False

            my_idx = new_pp_ranks.index(rank)
            pp_size = len(new_pp_ranks)

            # Compute prev and next ranks
            if pp_size > 1:
                prev_rank = new_pp_ranks[(my_idx - 1) % pp_size]
                next_rank = new_pp_ranks[(my_idx + 1) % pp_size]
            else:
                prev_rank = rank
                next_rank = rank

            # Update parallel_state globals
            if hasattr(ps, '_PREV_PIPELINE_MODEL_PARALLEL_RANK'):
                ps._PREV_PIPELINE_MODEL_PARALLEL_RANK = prev_rank
            if hasattr(ps, '_NEXT_PIPELINE_MODEL_PARALLEL_RANK'):
                ps._NEXT_PIPELINE_MODEL_PARALLEL_RANK = next_rank

            logger.debug(
                "MoEGambit pipeline repair: prev/next updated — "
                "rank=%d, prev=%d, next=%d",
                rank, prev_rank, next_rank,
            )
            return True

        except Exception as e:
            logger.error(
                "MoEGambit pipeline repair: failed to update prev/next: %s", e,
            )
            return False

    # -----------------------------------------------------------------
    # Step 4: Rebind P2P communicator
    # -----------------------------------------------------------------

    def _rebind_p2p_communicator(self) -> bool:
        """Destroy and recreate the P2P communicator singleton.

        The P2P communicator caches the PP group reference, so it must
        be recreated after the PP group is rebuilt.
        """
        try:
            from megatron.core.pipeline_parallel import p2p_communication

            # Check if there's a global P2P communicator to reset
            if hasattr(p2p_communication, '_P2P_COMMUNICATOR'):
                old = p2p_communication._P2P_COMMUNICATOR
                if old is not None:
                    # The communicator will be lazily recreated on next use
                    p2p_communication._P2P_COMMUNICATOR = None
                    logger.debug(
                        "MoEGambit pipeline repair: P2P communicator cleared "
                        "(will be recreated on next use)"
                    )
                    return True

            # If no global singleton, check for module-level creation
            # In some versions, the communicator is created per-call
            # and doesn't need explicit reset
            logger.debug(
                "MoEGambit pipeline repair: no P2P communicator singleton "
                "found (per-call creation mode)"
            )
            return True

        except ImportError:
            logger.warning(
                "MoEGambit pipeline repair: cannot import p2p_communication"
            )
            return False
        except Exception as e:
            logger.error(
                "MoEGambit pipeline repair: failed to rebind P2P: %s", e,
            )
            return False

    # -----------------------------------------------------------------
    # Step 5: Rebuild compound groups
    # -----------------------------------------------------------------

    def _rebuild_compound_groups(
        self,
        *,
        failed_rank: int,
        replacement_rank: int,
        create_group_fn: Optional[Callable] = None,
    ) -> int:
        """Rebuild compound groups that include the PP dimension.

        These groups combine PP with other dimensions (TP, DP, etc.)
        and need to be updated when a PP rank changes.

        When replacement_rank >= world_size (identity inheritance mode),
        the group membership doesn't actually change, so we recreate
        groups with their original rank lists.

        Returns the number of groups rebuilt.
        """
        count = 0

        try:
            from megatron.core import parallel_state as ps
            import torch.distributed

            if create_group_fn is None:
                create_group_fn = _default_create_group

            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Identity inheritance mode check
            needs_identity_inheritance = replacement_rank >= world_size

            for group_name in _PP_COMPOUND_GROUPS:
                var_name = _PP_GROUP_VAR_MAP.get(group_name)
                if var_name is None:
                    continue

                current_group = getattr(ps, var_name, None)
                if current_group is None:
                    continue

                # Get current ranks
                try:
                    current_ranks = torch.distributed.get_process_group_ranks(
                        current_group
                    )
                except Exception:
                    continue

                if failed_rank not in current_ranks:
                    continue

                # Compute new ranks
                if needs_identity_inheritance:
                    # In identity inheritance mode, the replacement process
                    # has the failed rank's identity, so ranks don't change
                    new_ranks = list(current_ranks)
                else:
                    new_ranks = [
                        replacement_rank if r == failed_rank else r
                        for r in current_ranks
                    ]

                # Validate all ranks are within world_size
                invalid_ranks = [r for r in new_ranks if r >= world_size]
                if invalid_ranks:
                    logger.warning(
                        "MoEGambit pipeline repair: compound group %s has "
                        "ranks %s exceeding world_size=%d, using original "
                        "ranks instead",
                        group_name, invalid_ranks, world_size,
                    )
                    new_ranks = list(current_ranks)

                # Destroy old group
                try:
                    torch.distributed.destroy_process_group(current_group)
                except Exception as e:
                    logger.warning(
                        "MoEGambit pipeline repair: failed to destroy "
                        "compound group %s: %s", group_name, e,
                    )

                # Create new group
                try:
                    new_group = create_group_fn(
                        new_ranks,
                        backend="nccl",
                        group_name=group_name,
                    )
                    if rank in new_ranks:
                        setattr(ps, var_name, new_group)
                        count += 1
                        logger.debug(
                            "MoEGambit pipeline repair: rebuilt compound "
                            "group %s with ranks %s",
                            group_name, new_ranks,
                        )
                except Exception as e:
                    logger.error(
                        "MoEGambit pipeline repair: failed to rebuild "
                        "compound group %s: %s", group_name, e,
                    )

        except ImportError:
            logger.warning(
                "MoEGambit pipeline repair: cannot import parallel_state"
            )
        except Exception as e:
            logger.error(
                "MoEGambit pipeline repair: compound group rebuild failed: %s",
                e,
            )

        logger.info(
            "MoEGambit pipeline repair: rebuilt %d compound groups", count,
        )
        return count

    # -----------------------------------------------------------------
    # Step 6: Verify
    # -----------------------------------------------------------------

    def _verify(self, new_pp_ranks: List[int]) -> bool:
        """Verify the rebuilt PP group with a barrier."""
        try:
            from megatron.core import parallel_state as ps
            import torch.distributed

            pp_group = getattr(ps, '_PIPELINE_MODEL_PARALLEL_GROUP', None)
            if pp_group is None:
                return False

            torch.distributed.barrier(group=pp_group)
            logger.debug("MoEGambit pipeline repair: verification passed")
            return True

        except Exception as e:
            logger.error(
                "MoEGambit pipeline repair: verification FAILED: %s", e,
            )
            return False

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "total_repairs": self._total_repairs,
            "last_result": {
                "success": self._last_result.success,
                "failed_stage": self._last_result.failed_stage,
                "pp_rebuilt": self._last_result.pp_group_rebuilt,
                "p2p_rebound": self._last_result.p2p_rebound,
                "compound_groups": self._last_result.compound_groups_rebuilt,
                "elapsed": self._last_result.elapsed_seconds,
            } if self._last_result else None,
        }

    def reset(self) -> None:
        """Reset state (for testing)."""
        self._last_result = None
        self._total_repairs = 0


# =====================================================================
# Helper: identify failed stage
# =====================================================================

def identify_failed_stage(
    failed_rank: int,
    pp_group_ranks: Optional[List[int]] = None,
) -> int:
    """Identify which pipeline stage a failed rank belongs to.

    Args:
        failed_rank: The global rank that failed.
        pp_group_ranks: The PP group ranks (ordered by stage).
            If None, attempts to read from parallel_state.

    Returns:
        The pipeline stage index (0-based), or -1 if not found.
    """
    if pp_group_ranks is None:
        try:
            from megatron.core import parallel_state as ps
            pp_group_ranks = getattr(ps, '_PIPELINE_GLOBAL_RANKS', None)
        except ImportError:
            pass

    if pp_group_ranks is None:
        return -1

    if failed_rank in pp_group_ranks:
        return pp_group_ranks.index(failed_rank)

    return -1


def compute_new_pp_ranks(
    pp_group_ranks: List[int],
    failed_rank: int,
    replacement_rank: int,
) -> List[int]:
    """Compute new PP group ranks after replacement.

    Returns a new list with failed_rank replaced by replacement_rank,
    preserving the stage ordering.
    """
    return [
        replacement_rank if r == failed_rank else r
        for r in pp_group_ranks
    ]


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

_REPAIRER: Optional[PipelineStageRepairer] = None


def get_pipeline_stage_repairer() -> PipelineStageRepairer:
    """Get or create the global PipelineStageRepairer singleton."""
    global _REPAIRER
    if _REPAIRER is None:
        _REPAIRER = PipelineStageRepairer()
    return _REPAIRER


def clear_pipeline_stage_repairer() -> None:
    """Reset the global repairer (for testing)."""
    global _REPAIRER
    if _REPAIRER is not None:
        _REPAIRER.reset()
    _REPAIRER = None


# =====================================================================
# Top-level convenience API
# =====================================================================

def execute_pipeline_stage_repair(
    *,
    failed_rank: int,
    replacement_rank: int,
    pp_group_ranks: List[int],
    create_group_fn: Optional[Callable] = None,
    verify: bool = True,
) -> PipelineStageRepairResult:
    """Execute a pipeline stage repair using the global repairer.

    This is the primary entry point for ``moegambit_integration.py``.
    """
    repairer = get_pipeline_stage_repairer()
    return repairer.execute(
        failed_rank=failed_rank,
        replacement_rank=replacement_rank,
        pp_group_ranks=pp_group_ranks,
        create_group_fn=create_group_fn,
        verify=verify,
    )
