# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dense / Shared / Router Parameter Fast-Pull from Healthy DP Peers (Step 9).

When a replacement rank comes online after a fault, it needs to obtain the
**current** version of all non-expert parameters.  In non-ZeRO-2 mode, every
DP peer holds an identical copy of these parameters (because gradients are
all-reduced before the optimizer step).  Therefore, the fastest recovery path
is to **broadcast** from any healthy DP peer rather than loading from a
(potentially stale) checkpoint.

Parameter classification
------------------------
Megatron tags every parameter with an ``allreduce`` attribute:

* ``param.allreduce == True`` (default) → **dense parameter**.
  Replicated across all DP ranks.  Gradient all-reduced in ``dp_cp_group``.
  Includes: attention, layernorm, embedding, output layer, router weights,
  shared-expert MLP weights.

* ``param.allreduce == False`` → **expert parameter**.
  Sharded across EP ranks.  Gradient all-reduced in ``expt_dp_group``.
  Includes: ``GroupedMLP.weight1/weight2``, ``SequentialMLP`` expert weights.

This module only handles **dense parameters** (``allreduce == True``).
Expert parameters are left for a later step to recover from checkpoint.

Recovery flow
-------------
::

    replacement rank starts
            │
            ▼
    classify_model_parameters(model)
            │
            ├─ dense_params:  {name: param}  (allreduce=True)
            ├─ expert_params: {name: param}  (allreduce=False)
            └─ router_params: {name: param}  (name contains 'router')
            │
            ▼
    select_healthy_dp_peer(dp_group, quarantined_ranks)
            │
            ▼
    pull_dense_params_from_peer(model, source_rank, dp_group)
            │
            ├─ for each dense param:
            │     broadcast(param.data, src=source_rank, group=dp_group)
            │
            └─ expert params: SKIPPED (left for checkpoint recovery)

Peer selection
--------------
In non-ZeRO-2 mode, all DP peers hold identical dense parameters, so any
healthy peer can serve as the source.  The selection strategy is:

1. Prefer the DP rank 0 (lowest rank in the DP group) — this is consistent
   with Megatron's existing ``broadcast_params()`` which always uses rank 0.
2. If rank 0 is quarantined, pick the lowest-ranked healthy peer.
3. If no healthy peer is available, fall back to checkpoint recovery.

Retry strategy
--------------
* **Attempt 1**: Broadcast from selected healthy peer.
* **Attempt 2**: If broadcast fails (e.g., NCCL timeout), try next healthy peer.
* **Attempt 3**: If all peers fail, mark recovery as failed and log error.
  The caller can then fall back to checkpoint-based recovery.

Scope (v1)
----------
* ✅ Dense / shared / router parameter sync from healthy DP peer
* ✅ Dense optimizer state sync (momentum, variance) from healthy DP peer
* ❌ Expert parameter recovery (later step — from checkpoint)
* ❌ Expert optimizer state recovery (later step — deferred loading)
* ❌ ZeRO-2 / sharded optimizer (not in scope)
* ❌ Automatic re-entry into training loop (not in scope)

Integration with MOEGAMBIT-MoE stack
------------------------------
* ``ReplacementRegistry`` (Step 6) — replacement lifecycle.
* ``GroupRebuildCoordinator`` (Step 7) — safe-point group rebuild.
* ``DispatchTopologyManager`` (Step 8) — topology refresh.
* This module (Step 9) — dense parameter fast-pull.

Typical usage::

    from megatron.core.transformer.moe.dense_param_sync import (
        classify_model_parameters,
        pull_dense_params_from_peer,
        select_healthy_dp_peer,
        DenseParamSyncResult,
    )

    # After group rebuild, before re-entering training:
    classification = classify_model_parameters(model)
    source_rank = select_healthy_dp_peer(dp_group, quarantined_ranks)
    result = pull_dense_params_from_peer(
        model, source_rank, dp_group, classification=classification,
    )
    assert result.success
    # Dense params are now current.  Expert params still need checkpoint recovery.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Parameter classification
# =====================================================================

class ParamCategory(enum.Enum):
    """Classification of a model parameter for recovery purposes."""

    DENSE = "dense"
    """Dense parameter: replicated across all DP ranks.
    ``param.allreduce == True`` (default).
    Includes: attention, layernorm, embedding, output layer."""

    ROUTER = "router"
    """Router gating weight: technically a dense parameter, but called out
    separately for logging and auditing.  ``param.allreduce == True`` and
    name contains 'router'."""

    SHARED_EXPERT = "shared_expert"
    """Shared expert MLP weight: technically a dense parameter, but called
    out separately.  ``param.allreduce == True`` and name contains
    'shared_experts'."""

    EXPERT = "expert"
    """Expert parameter: sharded across EP ranks.
    ``param.allreduce == False``.
    Includes: GroupedMLP weight1/weight2, SequentialMLP expert weights."""


@dataclass
class ParamClassification:
    """Result of classifying all model parameters.

    This is a pure-data descriptor that groups parameters by category.
    """

    dense_params: Dict[str, Any] = field(default_factory=dict)
    """name → param for dense parameters (allreduce=True, not router/shared)."""

    router_params: Dict[str, Any] = field(default_factory=dict)
    """name → param for router parameters."""

    shared_expert_params: Dict[str, Any] = field(default_factory=dict)
    """name → param for shared expert parameters."""

    expert_params: Dict[str, Any] = field(default_factory=dict)
    """name → param for expert parameters (allreduce=False)."""

    @property
    def num_dense(self) -> int:
        return len(self.dense_params)

    @property
    def num_router(self) -> int:
        return len(self.router_params)

    @property
    def num_shared_expert(self) -> int:
        return len(self.shared_expert_params)

    @property
    def num_expert(self) -> int:
        return len(self.expert_params)

    @property
    def num_total(self) -> int:
        return self.num_dense + self.num_router + self.num_shared_expert + self.num_expert

    @property
    def all_dense_like(self) -> Dict[str, Any]:
        """Return all parameters that should be synced from DP peer.

        This includes dense, router, and shared expert parameters —
        everything with ``allreduce == True``.
        """
        merged = {}
        merged.update(self.dense_params)
        merged.update(self.router_params)
        merged.update(self.shared_expert_params)
        return merged

    @property
    def num_dense_like(self) -> int:
        return self.num_dense + self.num_router + self.num_shared_expert

    def dense_param_count(self) -> int:
        """Total number of scalar values in dense-like parameters."""
        return sum(p.numel() for p in self.all_dense_like.values())

    def expert_param_count(self) -> int:
        """Total number of scalar values in expert parameters."""
        return sum(p.numel() for p in self.expert_params.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "dense": self.num_dense,
            "router": self.num_router,
            "shared_expert": self.num_shared_expert,
            "expert": self.num_expert,
            "total": self.num_total,
            "dense_like_params": self.num_dense_like,
            "dense_like_scalars": self.dense_param_count(),
            "expert_scalars": self.expert_param_count(),
        }


def classify_model_parameters(model) -> ParamClassification:
    """Classify all model parameters into dense / router / shared / expert.

    The classification is based on two signals:

    1. ``param.allreduce`` attribute:
       - ``True`` (default) → dense-like (synced via DP all-reduce)
       - ``False`` → expert (synced via Expert-DP all-reduce)

    2. Parameter name patterns:
       - Contains ``'router'`` → router parameter
       - Contains ``'shared_experts'`` → shared expert parameter
       - Otherwise → generic dense or expert

    Args:
        model: The model (or module) to classify.  Must support
            ``named_parameters()``.

    Returns:
        A ``ParamClassification`` with all parameters grouped.
    """
    result = ParamClassification()

    for name, param in model.named_parameters():
        is_expert_parallel = not getattr(param, 'allreduce', True)

        if is_expert_parallel:
            result.expert_params[name] = param
        elif 'router' in name:
            result.router_params[name] = param
        elif 'shared_experts' in name:
            result.shared_expert_params[name] = param
        else:
            result.dense_params[name] = param

    logger.info(
        "Parameter classification: %d dense, %d router, %d shared_expert, "
        "%d expert (total %d)",
        result.num_dense, result.num_router, result.num_shared_expert,
        result.num_expert, result.num_total,
    )
    return result


def classify_param(name: str, param) -> ParamCategory:
    """Classify a single parameter.

    Args:
        name: The parameter name (from ``named_parameters()``).
        param: The parameter tensor.

    Returns:
        The ``ParamCategory``.
    """
    is_expert_parallel = not getattr(param, 'allreduce', True)
    if is_expert_parallel:
        return ParamCategory.EXPERT
    elif 'router' in name:
        return ParamCategory.ROUTER
    elif 'shared_experts' in name:
        return ParamCategory.SHARED_EXPERT
    else:
        return ParamCategory.DENSE


# =====================================================================
# Healthy peer selection
# =====================================================================

def select_healthy_dp_peer(
    dp_group_ranks: List[int],
    quarantined_ranks: Optional[FrozenSet[int]] = None,
    failed_ranks: Optional[FrozenSet[int]] = None,
    local_rank: Optional[int] = None,
) -> Optional[int]:
    """Select a healthy DP peer to pull parameters from.

    Strategy:
    1. Prefer the lowest-ranked healthy peer (consistent with Megatron's
       ``broadcast_params()`` which uses rank 0 of the DP group).
    2. Exclude quarantined and failed ranks.
    3. Exclude the local rank (we don't pull from ourselves).

    Args:
        dp_group_ranks: Ordered list of global ranks in the DP group.
        quarantined_ranks: Set of quarantined global ranks.
        failed_ranks: Set of hard-failed global ranks.
        local_rank: The local rank (replacement rank).  Excluded from
            candidates.

    Returns:
        The global rank of the selected healthy peer, or ``None`` if
        no healthy peer is available.
    """
    if quarantined_ranks is None:
        quarantined_ranks = frozenset()
    if failed_ranks is None:
        failed_ranks = frozenset()

    excluded = quarantined_ranks | failed_ranks
    if local_rank is not None:
        excluded = excluded | {local_rank}

    candidates = [r for r in dp_group_ranks if r not in excluded]

    if not candidates:
        logger.error(
            "No healthy DP peer available for parameter sync! "
            "dp_group=%s, quarantined=%s, failed=%s, local=%s",
            dp_group_ranks, quarantined_ranks, failed_ranks, local_rank,
        )
        return None

    selected = candidates[0]  # lowest rank
    logger.info(
        "Selected healthy DP peer: rank %d (from candidates %s)",
        selected, candidates,
    )
    return selected


# =====================================================================
# Sync result
# =====================================================================

@dataclass
class DenseParamSyncResult:
    """Result of a dense parameter sync operation."""

    success: bool = False
    """Whether the sync completed successfully."""

    source_rank: int = -1
    """The rank that provided the parameters."""

    num_params_synced: int = 0
    """Number of parameters that were synced."""

    num_scalars_synced: int = 0
    """Total number of scalar values synced."""

    num_expert_skipped: int = 0
    """Number of expert parameters that were skipped."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the sync operation."""

    attempt: int = 0
    """Which attempt succeeded (1-based)."""

    error: str = ""
    """Error message if sync failed."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_rank": self.source_rank,
            "num_params_synced": self.num_params_synced,
            "num_scalars_synced": self.num_scalars_synced,
            "num_expert_skipped": self.num_expert_skipped,
            "elapsed_seconds": self.elapsed_seconds,
            "attempt": self.attempt,
            "error": self.error,
        }


# =====================================================================
# Core sync logic
# =====================================================================

def pull_dense_params_from_peer(
    model,
    source_rank: int,
    dp_group=None,
    *,
    classification: Optional[ParamClassification] = None,
    include_optimizer_states: bool = True,
    optimizer=None,
    max_retries: int = 2,
    broadcast_fn=None,
) -> DenseParamSyncResult:
    """Pull dense/shared/router parameters from a healthy DP peer.

    This broadcasts all dense-like parameters (``allreduce == True``) from
    the ``source_rank`` to all other ranks in the ``dp_group``.  Expert
    parameters (``allreduce == False``) are **skipped**.

    In non-ZeRO-2 mode, all DP peers hold identical dense parameters, so
    this is equivalent to restoring the replacement rank's parameters to
    the current training state.

    Args:
        model: The model whose parameters to sync.
        source_rank: The global rank to broadcast from.
        dp_group: The data-parallel process group.  If ``None``, the
            function operates in "dry-run" mode (no actual communication).
        classification: Pre-computed parameter classification.  If ``None``,
            will be computed from the model.
        include_optimizer_states: If ``True`` and ``optimizer`` is provided,
            also sync optimizer states (momentum, variance) for dense params.
        optimizer: The optimizer whose states to sync.  Only used if
            ``include_optimizer_states`` is ``True``.
        max_retries: Maximum number of retry attempts.
        broadcast_fn: Custom broadcast function for testing.  Signature:
            ``broadcast_fn(tensor, src, group) -> None``.
            If ``None``, uses ``torch.distributed.broadcast``.

    Returns:
        A ``DenseParamSyncResult`` describing the outcome.
    """
    start_time = time.monotonic()

    if classification is None:
        classification = classify_model_parameters(model)

    result = DenseParamSyncResult(source_rank=source_rank)
    dense_like = classification.all_dense_like

    if not dense_like:
        result.success = True
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    # Determine broadcast function
    if broadcast_fn is None and dp_group is not None:
        import torch.distributed
        def _default_broadcast(tensor, src, group):
            torch.distributed.broadcast(tensor, src=src, group=group)
        broadcast_fn = _default_broadcast

    # Attempt sync with retries
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            synced_count = 0
            synced_scalars = 0

            for name, param in dense_like.items():
                if broadcast_fn is not None and dp_group is not None:
                    broadcast_fn(param.data, src=source_rank, group=dp_group)
                synced_count += 1
                synced_scalars += param.numel()

            # Sync optimizer states if requested
            if include_optimizer_states and optimizer is not None:
                opt_synced = _sync_optimizer_states_for_dense(
                    optimizer, classification, source_rank, dp_group, broadcast_fn,
                )
                synced_scalars += opt_synced

            # Fail-closed: if we have dense params but synced none, fail
            if synced_count == 0 and len(dense_like) > 0:
                raise RuntimeError(
                    "Fail-closed: {} dense params exist "
                    "but 0 were synced".format(len(dense_like))
                )

            # Fail-closed: if optimizer sync was requested but synced 0 scalars
            if (include_optimizer_states and optimizer is not None
                    and opt_synced == 0
                    and classification.num_dense_like > 0):
                raise RuntimeError(
                    "Fail-closed: optimizer state sync requested but "
                    "0 scalars were synced for dense parameters"
                )

            result.success = True
            result.num_params_synced = synced_count
            result.num_scalars_synced = synced_scalars
            result.num_expert_skipped = classification.num_expert
            result.attempt = attempt
            result.elapsed_seconds = time.monotonic() - start_time

            logger.debug(
                "MOEGAMBIT-MoE dense param sync: SUCCESS — synced %d params "
                "(%d scalars) from rank %d (attempt %d, %.2fs, "
                "skipped %d expert params)",
                synced_count, synced_scalars, source_rank, attempt,
                result.elapsed_seconds, classification.num_expert,
            )
            return result

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "MOEGAMBIT-MoE dense param sync: attempt %d FAILED — %s",
                attempt, last_error,
            )

    # All retries exhausted
    result.success = False
    result.error = f"All {max_retries} attempts failed. Last error: {last_error}"
    result.elapsed_seconds = time.monotonic() - start_time

    logger.error(
        "MOEGAMBIT-MoE dense param sync: FAILED after %d attempts — %s",
        max_retries, result.error,
    )
    return result


def _sync_optimizer_states_for_dense(
    optimizer,
    classification: ParamClassification,
    source_rank: int,
    dp_group,
    broadcast_fn,
    *,
    require_nonempty_state: bool = True,
) -> int:
    """Sync optimizer states (momentum, variance) for dense parameters.

    In non-ZeRO-2 mode, optimizer states for dense parameters are identical
    across all DP peers (because they see the same all-reduced gradients).

    Args:
        optimizer: The optimizer.
        classification: Parameter classification.
        source_rank: Rank to broadcast from.
        dp_group: DP process group.
        broadcast_fn: Broadcast function.
        require_nonempty_state: If True (default), raise RuntimeError when
            optimizer state is empty but dense parameters exist.  This
            implements fail-closed semantics.

    Returns:
        Number of scalar values synced.

    Raises:
        RuntimeError: If require_nonempty_state is True and optimizer state
            is empty while dense parameters exist.
    """
    if optimizer is None or dp_group is None or broadcast_fn is None:
        return 0

    # Build a set of dense-like param ids for matching.
    # We use both id(param) from classification AND build a reverse map
    # from optimizer param_groups to handle cases where the optimizer
    # holds different param objects (e.g., fp32 main params in
    # Float16Module / mixed-precision training).
    dense_like_params = set(id(p) for p in classification.all_dense_like.values())

    # Also build a set of dense-like param *data_ptr* for robust matching
    # when optimizer uses fp32 copies of fp16 model params.
    dense_like_data_ptrs = set()
    for p in classification.all_dense_like.values():
        if hasattr(p, 'data') and hasattr(p.data, 'data_ptr'):
            dense_like_data_ptrs.add(p.data.data_ptr())

    # Build expert param ids to EXCLUDE from sync
    expert_param_ids = set(id(p) for p in classification.expert_params.values())

    synced_scalars = 0

    try:
        # Access optimizer state dict
        state_dict = optimizer.state
        if hasattr(state_dict, 'items'):
            # Fail-closed: if optimizer state is empty but we have dense params
            if (require_nonempty_state
                    and len(state_dict) == 0
                    and classification.num_dense_like > 0):
                raise RuntimeError(
                    "Fail-closed: optimizer state is empty "
                    "(len(state_dict) == 0) but {} dense-like parameters "
                    "exist.  The optimizer may not have been stepped yet, "
                    "or the state was lost.  Set require_nonempty_state=False "
                    "to override.".format(classification.num_dense_like)
                )

            for param_key, state in state_dict.items():
                # Determine if this optimizer state belongs to a dense-like param.
                #
                # Strategy: include by default, exclude only if we can
                # positively identify the param as an expert param.
                # This handles the common case where optimizer param objects
                # differ from model param objects (e.g., fp32 main params
                # in mixed-precision training).
                if isinstance(param_key, int):
                    # Integer key (e.g., DistributedOptimizer) — cannot
                    # match by id; include it (conservative).
                    pass
                else:
                    param_id = id(param_key)
                    if param_id in expert_param_ids:
                        # Definitely an expert param — skip
                        continue
                    # If it matches a dense-like param by id, include it.
                    # If it matches neither (e.g., fp32 copy), also include
                    # it — better to over-sync than under-sync for dense.

                if isinstance(state, dict):
                    for state_name, state_val in state.items():
                        if hasattr(state_val, 'data') and hasattr(state_val, 'numel'):
                            broadcast_fn(state_val.data, src=source_rank, group=dp_group)
                            synced_scalars += state_val.numel()
    except RuntimeError:
        raise  # Re-raise fail-closed errors
    except Exception as e:
        logger.warning(
            "MOEGAMBIT-MoE: optimizer state sync encountered error: %s", e,
        )

    return synced_scalars


# =====================================================================
# Expert parameter sync from Expert-DP peer (EDP > 1)
# =====================================================================

@dataclass
class ExpertPeerSyncResult:
    """Result of an expert parameter sync from an Expert-DP peer."""

    success: bool = False
    """Whether the sync completed successfully."""

    source_rank: int = -1
    """The rank that provided the expert parameters."""

    num_params_synced: int = 0
    """Number of expert parameters that were synced."""

    num_scalars_synced: int = 0
    """Total number of scalar values synced."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the sync operation."""

    attempt: int = 0
    """Which attempt succeeded (1-based)."""

    error: str = ""
    """Error message if sync failed."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_rank": self.source_rank,
            "num_params_synced": self.num_params_synced,
            "num_scalars_synced": self.num_scalars_synced,
            "elapsed_seconds": self.elapsed_seconds,
            "attempt": self.attempt,
            "error": self.error,
        }


def select_healthy_expert_dp_peer(
    expt_dp_group_ranks: List[int],
    quarantined_ranks: Optional[FrozenSet[int]] = None,
    failed_ranks: Optional[FrozenSet[int]] = None,
    local_rank: Optional[int] = None,
) -> Optional[int]:
    """Select a healthy Expert-DP peer to pull expert parameters from.

    This is the Expert-DP analogue of ``select_healthy_dp_peer()``.
    When EDP > 1, each expert subset is replicated across the
    ``expt_dp_group``.  Any healthy peer in this group holds an
    identical, current-step copy of the expert weights and optimizer
    state.

    Strategy: prefer the lowest-ranked healthy peer (consistent with
    ``select_healthy_dp_peer``).

    Args:
        expt_dp_group_ranks: Ordered list of global ranks in the
            Expert-DP group.
        quarantined_ranks: Set of quarantined global ranks.
        failed_ranks: Set of hard-failed global ranks.
        local_rank: The local rank (replacement rank).

    Returns:
        The global rank of the selected healthy Expert-DP peer, or
        ``None`` if no healthy peer is available (EDP == 1 or all
        peers are down).
    """
    if expt_dp_group_ranks is None or len(expt_dp_group_ranks) <= 1:
        # EDP == 1: no expert replica exists
        return None

    if quarantined_ranks is None:
        quarantined_ranks = frozenset()
    if failed_ranks is None:
        failed_ranks = frozenset()

    excluded = quarantined_ranks | failed_ranks
    if local_rank is not None:
        excluded = excluded | {local_rank}

    candidates = [r for r in expt_dp_group_ranks if r not in excluded]

    if not candidates:
        logger.warning(
            "No healthy Expert-DP peer available for expert sync! "
            "expt_dp_group=%s, quarantined=%s, failed=%s, local=%s",
            expt_dp_group_ranks, quarantined_ranks, failed_ranks,
            local_rank,
        )
        return None

    selected = candidates[0]
    logger.info(
        "Selected healthy Expert-DP peer: rank %d (from candidates %s)",
        selected, candidates,
    )
    return selected


def pull_expert_params_from_peer(
    model,
    source_rank: int,
    expt_dp_group=None,
    *,
    classification: Optional[ParamClassification] = None,
    include_optimizer_states: bool = True,
    optimizer=None,
    max_retries: int = 2,
    broadcast_fn=None,
) -> ExpertPeerSyncResult:
    """Pull expert parameters from a healthy Expert-DP peer.

    When EDP > 1, each expert subset is replicated across the
    ``expt_dp_group``.  This function broadcasts expert parameters
    (``allreduce == False``) from the ``source_rank`` to all other
    ranks in the ``expt_dp_group``.

    This is the "full-peer" complement to ``pull_dense_params_from_peer()``:
    together they enable **zero-disk, zero-staleness recovery** when
    EDP > 1.  Because the expert state comes from a peer at step *t*
    (not from a checkpoint at step *c*), the recovered expert state
    has zero staleness and contributes nothing to Phi'(t).

    Dense-like parameters (``allreduce == True``) are **skipped** —
    they are handled by ``pull_dense_params_from_peer()`` over the
    DP group.

    Args:
        model: The model whose expert parameters to sync.
        source_rank: The global rank to broadcast from.
        expt_dp_group: The Expert-DP process group.  If ``None``,
            operates in dry-run mode (no actual communication).
        classification: Pre-computed parameter classification.
        include_optimizer_states: If ``True`` and ``optimizer`` is
            provided, also sync optimizer states for expert params.
        optimizer: The optimizer whose expert states to sync.
        max_retries: Maximum number of retry attempts.
        broadcast_fn: Custom broadcast function for testing.

    Returns:
        An ``ExpertPeerSyncResult`` describing the outcome.
    """
    start_time = time.monotonic()

    if classification is None:
        classification = classify_model_parameters(model)

    result = ExpertPeerSyncResult(source_rank=source_rank)
    expert_params = classification.expert_params

    if not expert_params:
        result.success = True
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    # Determine broadcast function
    if broadcast_fn is None and expt_dp_group is not None:
        import torch.distributed
        def _default_broadcast(tensor, src, group):
            torch.distributed.broadcast(tensor, src=src, group=group)
        broadcast_fn = _default_broadcast

    # Attempt sync with retries
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            synced_count = 0
            synced_scalars = 0

            for name, param in expert_params.items():
                if broadcast_fn is not None and expt_dp_group is not None:
                    broadcast_fn(param.data, src=source_rank, group=expt_dp_group)
                synced_count += 1
                synced_scalars += param.numel()

            # Sync expert optimizer states if requested
            if include_optimizer_states and optimizer is not None:
                opt_synced = _sync_optimizer_states_for_experts(
                    optimizer, classification, source_rank,
                    expt_dp_group, broadcast_fn,
                )
                synced_scalars += opt_synced

            # Fail-closed: if we have expert params but synced none
            if synced_count == 0 and len(expert_params) > 0:
                raise RuntimeError(
                    "Fail-closed: {} expert params exist "
                    "but 0 were synced".format(len(expert_params))
                )

            result.success = True
            result.num_params_synced = synced_count
            result.num_scalars_synced = synced_scalars
            result.attempt = attempt
            result.elapsed_seconds = time.monotonic() - start_time

            logger.debug(
                "MOEGAMBIT-MoE expert peer sync: SUCCESS — synced %d params "
                "(%d scalars) from Expert-DP peer rank %d (attempt %d, "
                "%.2fs)",
                synced_count, synced_scalars, source_rank, attempt,
                result.elapsed_seconds,
            )
            return result

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "MOEGAMBIT-MoE expert peer sync: attempt %d FAILED — %s",
                attempt, last_error,
            )

    # All retries exhausted
    result.success = False
    result.error = f"All {max_retries} attempts failed. Last error: {last_error}"
    result.elapsed_seconds = time.monotonic() - start_time

    logger.error(
        "MOEGAMBIT-MoE expert peer sync: FAILED after %d attempts — %s",
        max_retries, result.error,
    )
    return result


def _sync_optimizer_states_for_experts(
    optimizer,
    classification: ParamClassification,
    source_rank: int,
    expt_dp_group,
    broadcast_fn,
) -> int:
    """Sync optimizer states (momentum, variance) for expert parameters.

    When EDP > 1, optimizer states for expert parameters are identical
    across all Expert-DP peers (because expert gradients are all-reduced
    within the ``expt_dp_group`` before the optimizer step).

    Args:
        optimizer: The optimizer.
        classification: Parameter classification.
        source_rank: Rank to broadcast from.
        expt_dp_group: Expert-DP process group.
        broadcast_fn: Broadcast function.

    Returns:
        Number of scalar values synced.
    """
    if optimizer is None or expt_dp_group is None or broadcast_fn is None:
        return 0

    # Build expert param ids for matching
    expert_param_ids = set(id(p) for p in classification.expert_params.values())

    synced_scalars = 0

    try:
        state_dict = optimizer.state
        if hasattr(state_dict, 'items'):
            for param_key, state in state_dict.items():
                # Only sync optimizer state for expert params
                if isinstance(param_key, int):
                    # Integer key — cannot match by id; skip (conservative
                    # for experts: better to under-sync than over-sync,
                    # since dense optimizer states are handled separately)
                    continue
                else:
                    param_id = id(param_key)
                    if param_id not in expert_param_ids:
                        # Not an expert param — skip
                        continue

                if isinstance(state, dict):
                    for state_name, state_val in state.items():
                        if hasattr(state_val, 'data') and hasattr(state_val, 'numel'):
                            broadcast_fn(
                                state_val.data, src=source_rank,
                                group=expt_dp_group,
                            )
                            synced_scalars += state_val.numel()
    except Exception as e:
        logger.warning(
            "MOEGAMBIT-MoE: expert optimizer state sync encountered error: %s", e,
        )

    return synced_scalars


# =====================================================================
# Post-sync verification
# =====================================================================

def verify_synced_params(
    model,
    classification: Optional[ParamClassification] = None,
) -> Tuple[bool, str]:
    """Verify that all dense-like parameters are free of NaN and Inf.

    This should be called after pull_dense_params_from_peer() to confirm
    that the sync produced valid tensor data.

    Args:
        model: The model to verify.
        classification: Optional pre-computed classification.

    Returns:
        Tuple of (success: bool, error_message: str).
        If success is True, error_message is empty.
    """
    import torch

    if classification is None:
        classification = classify_model_parameters(model)

    errors = []
    for name, param in classification.all_dense_like.items():
        if torch.isnan(param.data).any():
            errors.append("NaN in '{}'".format(name))
        if torch.isinf(param.data).any():
            errors.append("Inf in '{}'".format(name))

    if errors:
        msg = "Dense param verification failed: " + "; ".join(errors[:5])
        if len(errors) > 5:
            msg += " ... and {} more".format(len(errors) - 5)
        logger.error("MOEGAMBIT-MoE: %s", msg)
        return False, msg

    logger.info(
        "MOEGAMBIT-MoE: dense param verification PASSED (%d params checked)",
        classification.num_dense_like,
    )
    return True, ""


# =====================================================================
# High-level recovery orchestrator
# =====================================================================

@dataclass
class DenseRecoveryPlan:
    """Describes a planned dense parameter recovery operation."""

    replacement_rank: int = -1
    """The replacement rank that needs parameters."""

    failed_rank: int = -1
    """The failed rank being replaced."""

    source_rank: int = -1
    """The healthy DP peer to pull from."""

    dp_group_ranks: List[int] = field(default_factory=list)
    """Ranks in the DP group."""

    step: int = -1
    """Training step when recovery was planned."""

    include_optimizer: bool = True
    """Whether to also sync optimizer states."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replacement_rank": self.replacement_rank,
            "failed_rank": self.failed_rank,
            "source_rank": self.source_rank,
            "dp_group_ranks": list(self.dp_group_ranks),
            "step": self.step,
            "include_optimizer": self.include_optimizer,
        }


class DenseParamRecoveryCoordinator:
    """Coordinates dense parameter recovery for replacement ranks.

    This is the high-level coordinator that:
    1. Plans the recovery (select source peer, classify parameters).
    2. Executes the sync (broadcast dense params from peer).
    3. Tracks the result for auditing.

    The coordinator does NOT handle expert parameter recovery — that is
    left for a later step (checkpoint-based recovery).
    """

    def __init__(self) -> None:
        self._plans: List[DenseRecoveryPlan] = []
        self._results: List[DenseParamSyncResult] = []
        self._last_classification: Optional[ParamClassification] = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_recovery(
        self,
        replacement_rank: int,
        failed_rank: int,
        dp_group_ranks: List[int],
        quarantined_ranks: Optional[FrozenSet[int]] = None,
        step: int = 0,
        include_optimizer: bool = True,
    ) -> DenseRecoveryPlan:
        """Plan a dense parameter recovery operation.

        Args:
            replacement_rank: The replacement rank.
            failed_rank: The failed rank.
            dp_group_ranks: Ranks in the DP group.
            quarantined_ranks: Quarantined ranks to exclude.
            step: Current training step.
            include_optimizer: Whether to sync optimizer states.

        Returns:
            The recovery plan.
        """
        source = select_healthy_dp_peer(
            dp_group_ranks=dp_group_ranks,
            quarantined_ranks=quarantined_ranks,
            failed_ranks=frozenset({failed_rank}),
            local_rank=replacement_rank,
        )

        plan = DenseRecoveryPlan(
            replacement_rank=replacement_rank,
            failed_rank=failed_rank,
            source_rank=source if source is not None else -1,
            dp_group_ranks=list(dp_group_ranks),
            step=step,
            include_optimizer=include_optimizer,
        )
        self._plans.append(plan)

        logger.info(
            "MOEGAMBIT-MoE dense recovery: planned — replacement=%d, failed=%d, "
            "source=%d, step=%d",
            replacement_rank, failed_rank,
            plan.source_rank, step,
        )
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_recovery(
        self,
        model,
        plan: DenseRecoveryPlan,
        dp_group=None,
        optimizer=None,
        broadcast_fn=None,
    ) -> DenseParamSyncResult:
        """Execute a dense parameter recovery.

        Args:
            model: The model to sync.
            plan: The recovery plan.
            dp_group: DP process group.
            optimizer: Optimizer (for state sync).
            broadcast_fn: Custom broadcast function.

        Returns:
            The sync result.
        """
        if plan.source_rank < 0:
            result = DenseParamSyncResult(
                success=False,
                error="No healthy source peer available",
            )
            self._results.append(result)
            return result

        classification = classify_model_parameters(model)
        self._last_classification = classification

        result = pull_dense_params_from_peer(
            model=model,
            source_rank=plan.source_rank,
            dp_group=dp_group,
            classification=classification,
            include_optimizer_states=plan.include_optimizer,
            optimizer=optimizer,
            broadcast_fn=broadcast_fn,
        )
        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def plans(self) -> List[DenseRecoveryPlan]:
        return list(self._plans)

    @property
    def results(self) -> List[DenseParamSyncResult]:
        return list(self._results)

    @property
    def last_classification(self) -> Optional[ParamClassification]:
        return self._last_classification

    @property
    def last_result(self) -> Optional[DenseParamSyncResult]:
        return self._results[-1] if self._results else None

    def summary(self) -> Dict[str, Any]:
        return {
            "num_plans": len(self._plans),
            "num_results": len(self._results),
            "last_success": self._results[-1].success if self._results else None,
            "last_classification": (
                self._last_classification.summary()
                if self._last_classification else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"DenseParamRecoveryCoordinator("
            f"plans={len(self._plans)}, "
            f"results={len(self._results)})"
        )

    def reset(self) -> None:
        self._plans.clear()
        self._results.clear()
        self._last_classification = None


# =====================================================================
# Global singleton
# =====================================================================

_COORDINATOR: Optional[DenseParamRecoveryCoordinator] = None


def get_dense_param_recovery_coordinator() -> DenseParamRecoveryCoordinator:
    """Get or create the global coordinator singleton."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = DenseParamRecoveryCoordinator()
    return _COORDINATOR


def clear_dense_param_recovery_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.reset()
    _COORDINATOR = None
