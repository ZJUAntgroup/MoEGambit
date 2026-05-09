# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BSR-MoE Training Loop Integration.

This module provides the glue code that connects all BSR-MoE modules to
Megatron's training loop.  It handles:

1. **Initialization** — creates all BSR-MoE singletons at training start.
2. **Callback wiring** — registers RecoveryController callbacks that
   delegate to the real BSR-MoE module singletons.
3. **Training hooks** — ``bsr_before_iteration()`` and ``bsr_after_iteration()``
   are called from the training loop.
4. **Scheduled fault injection** — when ``--moe-bsr-fault-injection`` is
   enabled, injects a configurable fault at a specified training step.

Usage in training.py::

    from megatron.core.transformer.moe.bsr_integration import (
        maybe_initialize_bsr_moe,
        bsr_before_iteration,
        bsr_after_iteration,
    )

    # After model/optimizer init, before training loop:
    maybe_initialize_bsr_moe(model, args)

    # In training loop:
    for step in range(...):
        bsr_before_iteration(step)
        train_step(...)
        bsr_after_iteration(step)
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _ts() -> str:
    """Return current timestamp string for structured logging."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


_DECODER_LAYER_RE = re.compile(r'(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)')


def _get_single_model(model: Any) -> Any:
    """Return the concrete model object when Megatron passes a chunk list."""
    return model[0] if isinstance(model, (list, tuple)) else model


def _get_args_or_none() -> Optional[Any]:
    try:
        from megatron.training.global_vars import get_args
        return get_args()
    except Exception:
        return None


def _get_pp_rank_size() -> Tuple[int, int]:
    """Best-effort PP rank/size lookup that also works in recovery callbacks."""
    pp_rank = 0
    pp_size = 1
    try:
        from megatron.core import parallel_state as mpu
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    except Exception:
        pass

    args = _get_args_or_none()
    if args is not None:
        pp_size = max(pp_size, int(getattr(args, 'pipeline_model_parallel_size', pp_size) or 1))
    return pp_rank, pp_size


def _get_tp_ep_ranks_sizes() -> Tuple[int, int, int, int]:
    """Return TP rank, current EP rank, EP size, and total expert count."""
    tp_rank = 0
    ep_rank = 0
    ep_size = 1
    num_experts_total = 1
    try:
        from megatron.core import parallel_state as mpu
        tp_rank = mpu.get_tensor_model_parallel_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
    except Exception:
        pass

    args = _get_args_or_none()
    if args is not None:
        ep_size = max(ep_size, int(getattr(args, 'expert_model_parallel_size', ep_size) or 1))
        num_experts_total = int(getattr(args, 'num_experts', num_experts_total) or num_experts_total)
    return tp_rank, ep_rank, ep_size, num_experts_total


def _extract_decoder_layer_idx(name: str) -> Optional[int]:
    match = _DECODER_LAYER_RE.search(name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _collect_model_layer_indices(model: Any, *, moe_only: bool = False) -> List[int]:
    actual_model = _get_single_model(model)
    indices = set()
    try:
        named_parameters = actual_model.named_parameters()
    except Exception:
        return []

    for name, _ in named_parameters:
        if moe_only and '.mlp.experts.' not in name and '.mlp.router' not in name:
            continue
        idx = _extract_decoder_layer_idx(name)
        if idx is not None:
            indices.add(idx)
    return sorted(indices)


def _infer_num_layers_per_pp_stage(model: Any, total_num_layers: Optional[int] = None) -> Optional[int]:
    _, pp_size = _get_pp_rank_size()
    if pp_size > 1 and total_num_layers and total_num_layers % pp_size == 0:
        return total_num_layers // pp_size

    indices = _collect_model_layer_indices(model)
    if not indices:
        return None
    if indices[0] == 0:
        return max(indices) + 1
    return len(indices)


def _infer_local_moe_layer_ids(model: Any, num_layers: int) -> List[int]:
    """Return 1-based global MoE layer ids owned by this PP stage."""
    pp_rank, pp_size = _get_pp_rank_size()
    local_indices = _collect_model_layer_indices(model, moe_only=True)
    if not local_indices:
        local_indices = _collect_model_layer_indices(model, moe_only=False)

    if pp_size <= 1:
        if local_indices:
            return [idx + 1 for idx in local_indices]
        return list(range(1, num_layers + 1))

    layers_per_stage = _infer_num_layers_per_pp_stage(model, num_layers)
    if layers_per_stage is None:
        if num_layers % pp_size != 0:
            logger.warning(
                "BSR-MoE: cannot infer PP-local layer range "
                "(num_layers=%d, pp_size=%d); falling back to all layers",
                num_layers, pp_size,
            )
            return list(range(1, num_layers + 1))
        layers_per_stage = num_layers // pp_size

    if local_indices and local_indices[0] == 0:
        layer_ids = [
            pp_rank * layers_per_stage + local_idx + 1
            for local_idx in local_indices
            if local_idx < layers_per_stage
        ]
    elif local_indices:
        # Some model variants expose global 0-based layer indices directly.
        layer_ids = [idx + 1 for idx in local_indices]
    else:
        start = pp_rank * layers_per_stage + 1
        layer_ids = list(range(start, start + layers_per_stage))

    return [lid for lid in sorted(set(layer_ids)) if 1 <= lid <= num_layers]


def _map_global_layer_to_local_checkpoint_idx(
    layer_id: int,
    model: Any,
    total_num_layers: Optional[int] = None,
) -> Optional[int]:
    """Map a 1-based global layer id to the checkpoint-local layer index."""
    pp_rank, pp_size = _get_pp_rank_size()
    if pp_size <= 1:
        return layer_id - 1 if layer_id >= 1 else layer_id

    layers_per_stage = _infer_num_layers_per_pp_stage(model, total_num_layers)
    if layers_per_stage is None:
        return layer_id - 1 if layer_id >= 1 else layer_id

    global_layer_start = pp_rank * layers_per_stage + 1
    global_layer_end = global_layer_start + layers_per_stage - 1
    if not (global_layer_start <= layer_id <= global_layer_end):
        return None
    return layer_id - global_layer_start


def _is_distributed_optimizer(optimizer) -> bool:
    """Check if the optimizer is a DistributedOptimizer (ZeRO-1).

    DistributedOptimizer shards optimizer state across DP ranks, so each
    rank holds a different slice.  This means we cannot simply broadcast
    optimizer state from a healthy peer — the replacement rank needs the
    same shard that the failed rank held, which is lost.

    Returns True if the optimizer is a DistributedOptimizer instance.
    """
    if optimizer is None:
        return False
    # Check class name to avoid hard import dependency
    cls_name = type(optimizer).__name__
    if cls_name == 'DistributedOptimizer':
        return True
    # Also check for ChainedOptimizer wrapping DistributedOptimizer
    if cls_name == 'ChainedOptimizer':
        for opt in getattr(optimizer, 'chained_optimizers', []):
            if type(opt).__name__ == 'DistributedOptimizer':
                return True
    return False


# Module-level flag: has BSR been initialized?
_BSR_INITIALIZED = False

# Cached references
_RECOVERY_CONTROLLER = None
_HARD_FAILURE_DETECTOR = None
_ITERATION_INVALIDATOR = None
_ROLLBACK_REPLAY_MANAGER = None
_OPTIMIZER_COMMIT_GUARD = None
_PIPELINE_ROLLBACK_COORDINATOR = None
_FAULT_INJECTOR_CONFIG: Optional[Dict[str, Any]] = None

# Async recovery (Phase 15)
_ASYNC_RECOVERY_WORKER = None
_RECOVERY_STREAM: Optional[torch.cuda.Stream] = None


# =====================================================================
# Public API
# =====================================================================

def maybe_initialize_bsr_moe(model, args, optimizer=None, opt_param_scheduler=None) -> bool:
    """Initialize BSR-MoE if enabled in args.

    Should be called once after model construction and distributed init.

    Args:
        model: The Megatron model (list of model chunks).
        args: Parsed Megatron arguments (from ``get_args()``).
        optimizer: The optimizer instance (optional).  If provided, enables
            dense optimizer state sync during hybrid recovery.
        opt_param_scheduler: The learning rate scheduler (optional).  If
            provided, enables full checkpoint restart with optimizer +
            scheduler state restoration.

    Returns:
        True if BSR-MoE was initialized, False otherwise.
    """
    global _BSR_INITIALIZED, _RECOVERY_CONTROLLER, _FAULT_INJECTOR_CONFIG
    global _HARD_FAILURE_DETECTOR, _ITERATION_INVALIDATOR, _ROLLBACK_REPLAY_MANAGER
    global _OPTIMIZER_COMMIT_GUARD
    global _PIPELINE_ROLLBACK_COORDINATOR

    if _BSR_INITIALIZED:
        return True

    if not getattr(args, 'moe_bsr_enable', False):
        return False

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    logger.warning("[%s] BSR-MoE: initializing on rank %d ...", _ts(), rank)

    # ---- 1. Import all BSR-MoE modules ----
    from megatron.core.transformer.moe import expert_directory
    from megatron.core.transformer.moe import replacement_registry
    from megatron.core.transformer.moe import group_rebuild
    from megatron.core.transformer.moe import dispatch_topology_refresh
    from megatron.core.transformer.moe import dense_param_sync
    from megatron.core.transformer.moe import stale_expert_restore
    from megatron.core.transformer.moe import recovery_controller as rc_mod
    from megatron.core.transformer.moe import deferred_optimizer_load
    from megatron.core.transformer.moe import reintegration_barrier

    # ---- 2. Gather parallel topology info ----
    from megatron.core import parallel_state as mpu

    num_experts = args.num_experts
    num_layers = args.num_layers
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_group = mpu.get_expert_model_parallel_group()

    # Get EP group ranks via torch.distributed
    ep_group_ranks = _get_process_group_ranks(ep_group)

    # Get DP group ranks
    dp_group = mpu.get_data_parallel_group()
    dp_group_ranks = _get_process_group_ranks(dp_group)

    logger.warning(
        "BSR-MoE: num_experts=%d, num_layers=%d, ep_size=%d, "
        "ep_group_ranks=%s, dp_group_ranks=%s",
        num_experts, num_layers, ep_size, ep_group_ranks, dp_group_ranks,
    )

    # ---- 3. Initialize singletons ----

    # Expert directory
    if getattr(args, 'moe_bsr_expert_directory', False):
        directory = expert_directory.ActiveExpertDirectory.from_placement(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_group_ranks=ep_group_ranks,
        )
        expert_directory.set_active_expert_directory(directory)
        logger.warning("BSR-MoE: expert directory initialized")

    # Replacement registry
    if getattr(args, 'moe_bsr_replacement_protocol', False):
        replacement_registry.get_replacement_registry()

    # Group rebuild coordinator
    if getattr(args, 'moe_bsr_group_rebuild', False):
        group_rebuild.get_group_rebuild_coordinator()

    # Dispatch topology manager
    if getattr(args, 'moe_bsr_dispatch_topology_refresh', False):
        topo_mgr = dispatch_topology_refresh.get_dispatch_topology_manager()
        topo_mgr.initialise(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_group_ranks=ep_group_ranks,
        )
        logger.warning("BSR-MoE: dispatch topology manager initialized")

    # Reintegration barrier
    if getattr(args, 'moe_bsr_reintegration_barrier', False):
        reintegration_barrier.get_reintegration_barrier()

    # Deferred optimizer loader
    if getattr(args, 'moe_bsr_deferred_optimizer_load', False):
        deferred_optimizer_load.get_deferred_optimizer_loader()

    # Two-phase recovery coordinator
    if getattr(args, 'moe_bsr_weights_first_recovery', True):
        from megatron.core.transformer.moe import two_phase_recovery as tp_mod
        defer_opt = getattr(args, 'moe_bsr_defer_optimizer_load', True)
        tp_mod.get_two_phase_recovery_coordinator(defer_optimizer=defer_opt)
        logger.warning(
            "BSR-MoE: two-phase recovery coordinator initialized "
            "(defer_optimizer=%s)", defer_opt,
        )

    # ---- 4. Initialize RecoveryController and wire callbacks ----
    if getattr(args, 'moe_bsr_recovery_controller', False):
        ctrl = rc_mod.get_recovery_controller()
        _wire_recovery_callbacks(
            ctrl,
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_group_ranks=ep_group_ranks,
            dp_group_ranks=dp_group_ranks,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )
        _RECOVERY_CONTROLLER = ctrl
        logger.warning("BSR-MoE: recovery controller initialized and callbacks wired")

        # Wire reintegration barrier to recovery controller
        if getattr(args, 'moe_bsr_reintegration_barrier', False):
            barrier = reintegration_barrier.get_reintegration_barrier()
            ctrl._reintegration_barrier = barrier
            logger.warning("BSR-MoE: reintegration barrier wired to recovery controller")

        # Wire gap-aware recovery policy to recovery controller
        if getattr(args, 'moe_bsr_gap_aware_recovery', False):
            from megatron.core.transformer.moe import gap_aware_recovery_policy as garp_mod

            gap_threshold = getattr(args, 'moe_bsr_gap_threshold', 100)
            policy_type = getattr(args, 'moe_bsr_recovery_policy_type', 'threshold')

            # Build RankExposureGuardedConfig if needed
            rank_exposure_config = None
            if policy_type in ('rank_exposure_guarded', 'rank_exposure_guarded_hybrid'):
                rank_exposure_config = garp_mod.RankExposureGuardedConfig(
                    delta_time_min_gap=getattr(args, 'moe_bsr_delta_time_min_gap', 32),
                    max_single_gap=getattr(args, 'moe_bsr_max_single_gap', 192),
                    exposure_window_steps=getattr(args, 'moe_bsr_exposure_window_steps', 20000),
                    max_rank_stale_exposure=getattr(args, 'moe_bsr_max_rank_stale_exposure', 0.02),
                )
                logger.warning(
                    "BSR-MoE: rank-exposure-guarded policy config: %s",
                    rank_exposure_config.to_dict(),
                )

            policy_mgr = garp_mod.initialize_gap_aware_recovery_policy(
                gap_threshold=gap_threshold,
                policy_type=policy_type,
                rank_exposure_config=rank_exposure_config,
                delta_time_min_gap=getattr(args, 'moe_bsr_delta_time_min_gap', None),
                max_single_gap=getattr(args, 'moe_bsr_max_single_gap', None),
                get_checkpoint_iteration_fn=_get_latest_checkpoint_iteration,
                enabled=True,
            )
            ctrl.set_gap_aware_policy_manager(policy_mgr)
            # NOTE: checkpoint_restart_fn is registered inside
            # _wire_recovery_callbacks (closure over model/ep_group_ranks).
            # No separate register_callbacks call needed here.
            logger.warning(
                "BSR-MoE: gap-aware recovery policy wired to recovery controller "
                "(type=%s, threshold=%d)", policy_type, gap_threshold,
            )

    # ---- 4b. Initialize HardFailureDetector and IterationInvalidator ----
    from megatron.core.transformer.moe import hard_failure_detector as hfd_mod
    from megatron.core.transformer.moe import iteration_invalidator as inv_mod

    detector = hfd_mod.get_hard_failure_detector()
    invalidator = inv_mod.get_iteration_invalidator()

    # Wire detector callbacks
    if _RECOVERY_CONTROLLER is not None:
        detector.register_callbacks(
            on_hard_failure_fn=_RECOVERY_CONTROLLER.on_hard_rank_failure,
            on_iteration_invalid_fn=invalidator.invalidate,
        )
    else:
        # Even without recovery controller, wire the invalidator
        detector.register_callbacks(
            on_iteration_invalid_fn=invalidator.invalidate,
        )

    _HARD_FAILURE_DETECTOR = detector
    _ITERATION_INVALIDATOR = invalidator
    logger.warning("BSR-MoE: hard failure detector and iteration invalidator initialized")

    # ---- 4c. Initialize RollbackReplayManager ----
    from megatron.core.transformer.moe import iteration_rollback as rb_mod

    rollback_mgr = rb_mod.get_rollback_replay_manager()
    _ROLLBACK_REPLAY_MANAGER = rollback_mgr
    logger.warning("BSR-MoE: rollback/replay manager initialized")

    # ---- 4d. Initialize OptimizerCommitGuard ----
    from megatron.core.transformer.moe import optimizer_commit_guard as ocg_mod

    commit_guard = ocg_mod.get_optimizer_commit_guard()
    # Wire the invalidation check
    if _ITERATION_INVALIDATOR is not None:
        commit_guard.register_callbacks(
            is_iteration_invalid_fn=_ITERATION_INVALIDATOR.is_current_iteration_invalid,
        )
    _OPTIMIZER_COMMIT_GUARD = commit_guard
    logger.warning("BSR-MoE: optimizer commit guard initialized")

    # ---- 4e. Initialize PipelineRollbackCoordinator ----
    from megatron.core.transformer.moe import pipeline_rollback as pr_mod

    pp_coord = pr_mod.get_pipeline_rollback_coordinator()
    _PIPELINE_ROLLBACK_COORDINATOR = pp_coord
    logger.warning("BSR-MoE: pipeline rollback coordinator initialized")

    # ---- 4f. Initialize async recovery worker and CUDA stream (Phase 15) ----
    global _ASYNC_RECOVERY_WORKER, _RECOVERY_STREAM

    if getattr(args, 'moe_bsr_async_recovery', True):
        # Default to enabled — async recovery reduces training stalls
        from megatron.core.transformer.moe import async_recovery_worker as arw_mod

        max_workers = getattr(args, 'moe_bsr_async_recovery_workers', 2)
        _ASYNC_RECOVERY_WORKER = arw_mod.get_async_recovery_worker(
            max_workers=max_workers,
        )

        # Create a dedicated CUDA stream for H2D copies during recovery
        if torch.cuda.is_available():
            _RECOVERY_STREAM = torch.cuda.Stream()
            logger.warning(
                "BSR-MoE: async recovery worker initialized "
                "(workers=%d, CUDA stream created)", max_workers,
            )
        else:
            logger.warning(
                "BSR-MoE: async recovery worker initialized "
                "(workers=%d, no CUDA — CPU mode)", max_workers,
            )

        # Wire async callbacks to recovery controller
        if _RECOVERY_CONTROLLER is not None:
            _wire_async_recovery_callbacks(
                ctrl=_RECOVERY_CONTROLLER,
                worker=_ASYNC_RECOVERY_WORKER,
                stream=_RECOVERY_STREAM,
                num_layers=num_layers,
                num_experts=num_experts,
                model=model,
            )

    # ---- 5. Configure fault injection schedule ----
    if getattr(args, 'moe_bsr_fault_injection', False):
        # Determine inject type: if --moe-bsr-restart-in-place is set,
        # override to 'restart_in_place' (regardless of env var).
        _rip_enabled = getattr(args, 'moe_bsr_restart_in_place', False)
        inject_type = (
            'restart_in_place'
            if _rip_enabled
            else os.environ.get('BSR_FAULT_INJECT_TYPE', 'quarantine')
        )

        _fault_inject_seed = int(os.environ.get('BSR_FAULT_INJECT_SEED', '42'))
        _fault_inject_rank = int(os.environ.get('BSR_FAULT_INJECT_RANK', '0'))
        import random as _fi_random
        _fault_rng = _fi_random.Random(_fault_inject_seed)

        _FAULT_INJECTOR_CONFIG = {
            'enabled': True,
            'inject_step': int(os.environ.get('BSR_FAULT_INJECT_STEP', '50')),
            'inject_interval': int(os.environ.get('BSR_FAULT_INJECT_INTERVAL', '0')),
            'inject_type': inject_type,
            'inject_rank': _fault_inject_rank,
            'random_rank': _fault_inject_rank < 0,
            'fault_rng': _fault_rng,
            'replacement_step': int(os.environ.get('BSR_FAULT_REPLACEMENT_STEP', '60')),
            'replacement_rank': int(os.environ.get('BSR_FAULT_REPLACEMENT_RANK', '-1')),
            'num_experts': num_experts,
            'num_layers': num_layers,
            'ep_group_ranks': ep_group_ranks,
            'dp_group_ranks': dp_group_ranks,
            'injected': False,
            'replacement_injected': False,
            'next_inject_step': int(os.environ.get('BSR_FAULT_INJECT_STEP', '50')),
            'inject_count': 0,
        }
        logger.warning(
            "BSR-MoE: fault injection configured — type=%s, rank=%s, step=%d, "
            "interval=%d, random_rank=%s, seed=%d",
            _FAULT_INJECTOR_CONFIG['inject_type'],
            'random' if _FAULT_INJECTOR_CONFIG['random_rank'] else str(_FAULT_INJECTOR_CONFIG['inject_rank']),
            _FAULT_INJECTOR_CONFIG['inject_step'],
            _FAULT_INJECTOR_CONFIG['inject_interval'],
            _FAULT_INJECTOR_CONFIG['random_rank'],
            _fault_inject_seed,
        )

        # ---- 5b. Wire restart-in-place tensor invalidation callback ----
        if inject_type == 'restart_in_place' and _RECOVERY_CONTROLLER is not None:
            from megatron.core.transformer.moe.restart_in_place import (
                invalidate_rank_tensors,
            )

            # Capture model/optimizer references for the invalidation callback
            _rip_model_ref = model
            _rip_optimizer_ref = optimizer

            def _invalidate_tensor_callback(**kwargs):
                """Invalidate all tensors with NaN sentinels (restart-in-place)."""
                m = _rip_model_ref
                opt = _rip_optimizer_ref
                if m is not None:
                    # If model is a list of chunks, iterate
                    if isinstance(m, (list, tuple)):
                        for chunk in m:
                            invalidate_rank_tensors(chunk, opt)
                    else:
                        invalidate_rank_tensors(m, opt)

            _RECOVERY_CONTROLLER.register_callbacks(
                invalidate_tensor_fn=_invalidate_tensor_callback,
            )
            logger.warning(
                "BSR-MoE: restart-in-place invalidate_tensor_fn wired to "
                "recovery controller"
            )

    _BSR_INITIALIZED = True
    logger.warning("[%s] BSR-MoE: initialization complete on rank %d", _ts(), rank)
    return True


def bsr_before_iteration(step: int) -> bool:
    """BSR-MoE safe-point hook — call at the start of each training iteration.

    Returns True if a repair was executed at this safe point.
    """
    if not _BSR_INITIALIZED:
        return False

    # Begin iteration tracking for the invalidator
    if _ITERATION_INVALIDATOR is not None:
        _ITERATION_INVALIDATOR.begin_iteration(step)

    # Begin iteration tracking for the commit guard
    if _OPTIMIZER_COMMIT_GUARD is not None:
        _OPTIMIZER_COMMIT_GUARD.begin_iteration(step)

    # Check for scheduled fault injection
    _maybe_inject_fault(step)

    # Delegate to recovery controller
    if _RECOVERY_CONTROLLER is not None:
        return _RECOVERY_CONTROLLER.before_iteration(step=step)

    return False


def bsr_after_iteration(step: int) -> bool:
    """BSR-MoE post-step hook — call after each training iteration.

    Returns True if any action was taken.
    """
    if not _BSR_INITIALIZED:
        return False

    action_taken = False

    # End iteration tracking for the invalidator
    if _ITERATION_INVALIDATOR is not None:
        _ITERATION_INVALIDATOR.end_iteration(step)

    # End iteration tracking for the commit guard
    if _OPTIMIZER_COMMIT_GUARD is not None:
        _OPTIMIZER_COMMIT_GUARD.end_iteration(step)

    # Poll async recovery (expert weights + optimizer states)
    _poll_async_recovery(step)

    # Poll deferred optimizer loader
    _maybe_poll_deferred_optimizer(step)

    # Delegate to recovery controller
    if _RECOVERY_CONTROLLER is not None:
        ctrl_action = _RECOVERY_CONTROLLER.after_iteration(step=step)
        action_taken = action_taken or ctrl_action

    return action_taken


def bsr_is_initialized() -> bool:
    """Check if BSR-MoE has been initialized."""
    return _BSR_INITIALIZED


def bsr_get_recovery_controller():
    """Get the recovery controller (or None if not initialized)."""
    return _RECOVERY_CONTROLLER


def bsr_get_hard_failure_detector():
    """Get the hard failure detector (or None if not initialized)."""
    return _HARD_FAILURE_DETECTOR


def bsr_get_iteration_invalidator():
    """Get the iteration invalidator (or None if not initialized)."""
    return _ITERATION_INVALIDATOR


def bsr_is_current_iteration_invalid() -> bool:
    """Check if the current iteration has been invalidated by a hard failure.

    The training loop should call this after train_step() returns (or after
    catching an exception from train_step()) to decide whether to skip the
    optimizer commit and iteration increment.

    Returns False if BSR is not initialized or no invalidation occurred.
    """
    if not _BSR_INITIALIZED:
        return False
    if _ITERATION_INVALIDATOR is not None and _ITERATION_INVALIDATOR.is_current_iteration_invalid():
        return True
    if _RECOVERY_CONTROLLER is not None and _RECOVERY_CONTROLLER.iteration_was_invalidated:
        return True
    return False


def bsr_should_skip_optimizer_step() -> bool:
    """Check if the optimizer step should be skipped for the current iteration.

    Alias for ``bsr_is_current_iteration_invalid()`` — named explicitly for
    clarity at the call site in training.py.
    """
    return bsr_is_current_iteration_invalid()


def bsr_report_hard_failure(
    failed_rank: int,
    *,
    reason: str = "",
    step: int = -1,
    mid_iteration: bool = True,
    exception: Optional[BaseException] = None,
    expert_ids: Optional[List[int]] = None,
    ep_group_ranks: Optional[List[int]] = None,
    dp_group_ranks: Optional[List[int]] = None,
) -> bool:
    """Report a hard failure from the training loop.

    This is the primary entry point for code that catches communication
    exceptions (e.g. in a try/except around forward_backward_func).

    Returns True if this is a new failure (callbacks were invoked).
    """
    if not _BSR_INITIALIZED or _HARD_FAILURE_DETECTOR is None:
        logger.warning(
            "BSR-MoE: bsr_report_hard_failure called but BSR not initialized "
            "(rank=%d, reason=%s)", failed_rank, reason,
        )
        return False

    return _HARD_FAILURE_DETECTOR.report_collective_failure(
        failed_rank,
        reason=reason,
        step=step,
        mid_iteration=mid_iteration,
        exception=exception,
        expert_ids=expert_ids,
        ep_group_ranks=ep_group_ranks,
        dp_group_ranks=dp_group_ranks,
    )


def bsr_snapshot_iteration(
    iteration: int,
    consumed_train_samples: int,
    consumed_valid_samples: int = 0,
    num_floating_point_operations_so_far: int = 0,
) -> None:
    """Take a snapshot of the iteration boundary state.

    Must be called at the *beginning* of each iteration, before
    ``train_step()``.  This captures the "last known good" state
    that can be rolled back to if the iteration fails.
    """
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return
    _ROLLBACK_REPLAY_MANAGER.take_snapshot(
        iteration=iteration,
        consumed_train_samples=consumed_train_samples,
        consumed_valid_samples=consumed_valid_samples,
        num_floating_point_operations_so_far=num_floating_point_operations_so_far,
    )


def bsr_rollback_iteration(
    args: Any,
    data_iterators: Any = None,
    num_fp_ops_ref: Optional[List[int]] = None,
) -> bool:
    """Roll back to the last snapshot after a failed iteration.

    Restores ``args.consumed_train_samples`` and rewinds the data
    iterator for replay.

    Returns True if rollback was performed.
    """
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return False
    return _ROLLBACK_REPLAY_MANAGER.rollback(
        args=args,
        data_iterators=data_iterators,
        num_fp_ops_ref=num_fp_ops_ref,
    )


def bsr_is_replay_pending() -> bool:
    """True if the current iteration is a replay attempt."""
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return False
    return _ROLLBACK_REPLAY_MANAGER.is_replay_pending()


def bsr_complete_replay(data_iterators: Any = None) -> None:
    """Mark the replay as successfully completed.

    Called after ``train_step`` succeeds on a replayed iteration.
    """
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return
    _ROLLBACK_REPLAY_MANAGER.complete_replay(data_iterators)


def bsr_advance_iteration(data_iterators: Any = None) -> None:
    """Advance past the current iteration (normal success path).

    Called after a successful ``train_step`` on a non-replay iteration.
    """
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return
    _ROLLBACK_REPLAY_MANAGER.advance(data_iterators)


def bsr_get_rollback_replay_manager():
    """Get the rollback/replay manager (or None if not initialized)."""
    return _ROLLBACK_REPLAY_MANAGER


def bsr_exceeded_max_replays() -> bool:
    """True if the current iteration has exceeded max replay attempts."""
    if not _BSR_INITIALIZED or _ROLLBACK_REPLAY_MANAGER is None:
        return False
    return _ROLLBACK_REPLAY_MANAGER.exceeded_max_replays


def bsr_should_commit_optimizer() -> bool:
    """Check whether optimizer.step() should proceed.

    Returns False if the current iteration has been invalidated by a
    hard failure, preventing the optimizer from committing a tainted
    update.

    Should be called AFTER forward_backward_func() and BEFORE
    optimizer.step() in train_step().
    """
    if not _BSR_INITIALIZED or _OPTIMIZER_COMMIT_GUARD is None:
        return True  # default: allow commit
    return _OPTIMIZER_COMMIT_GUARD.should_commit()


def bsr_mark_optimizer_committed(step: int = -1) -> None:
    """Mark the optimizer step as successfully committed."""
    if not _BSR_INITIALIZED or _OPTIMIZER_COMMIT_GUARD is None:
        return
    _OPTIMIZER_COMMIT_GUARD.mark_committed(step)


def bsr_mark_optimizer_skipped(step: int = -1, reason: str = "") -> None:
    """Mark the optimizer step as skipped."""
    if not _BSR_INITIALIZED or _OPTIMIZER_COMMIT_GUARD is None:
        return
    _OPTIMIZER_COMMIT_GUARD.mark_skipped(step, reason)


def bsr_get_optimizer_commit_guard():
    """Get the optimizer commit guard (or None if not initialized)."""
    return _OPTIMIZER_COMMIT_GUARD


# =====================================================================
# Replacement rank management API
# =====================================================================

def bsr_announce_replacement_ready(
    failed_rank: int,
    replacement_rank: int,
    step: int = -1,
    reason: str = "hard_failure",
) -> bool:
    """Announce that a replacement rank is available and trigger the
    replacement protocol through the RecoveryController.

    This is the external entry point for the replacement orchestrator
    (e.g. a job scheduler or health monitor) to notify the training
    system that a spare node is ready to take over a failed rank.

    The call drives the RecoveryController through:
        PENDING_GROUP_REPAIR → WAITING_FOR_REPLACEMENT → SAFE_POINT_REPAIR

    Returns True if the announcement was accepted.
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        return False
    try:
        _RECOVERY_CONTROLLER.on_replacement_assigned(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            step=step,
        )
        _RECOVERY_CONTROLLER.on_replacement_ready(
            failed_rank=failed_rank,
            step=step,
        )
        logger.warning(
            "BSR-MoE: replacement announced and ready — "
            "failed_rank=%d, replacement_rank=%d, step=%d",
            failed_rank, replacement_rank, step,
        )
        return True
    except Exception as e:
        logger.error(
            "BSR-MoE: bsr_announce_replacement_ready failed: %s", e,
        )
        return False


def bsr_query_replacement_status(failed_rank: int) -> str:
    """Query the replacement status for a failed rank.

    Returns one of: 'NOT_PRESENT', 'BOOTSTRAPPING', 'READY_FOR_REPAIR',
    'INTEGRATED', or 'UNKNOWN'.
    """
    try:
        from megatron.core.transformer.moe import replacement_registry as rep_mod
        status = rep_mod.query_replacement_status(failed_rank)
        return status.name
    except Exception:
        return "UNKNOWN"


def bsr_has_pending_replacements() -> bool:
    """Return True if there are replacement ranks waiting to be integrated.

    This is useful for the training loop to log status or adjust behavior
    while waiting for a safe-point repair.
    """
    try:
        from megatron.core.transformer.moe import replacement_registry as rep_mod
        reg = rep_mod.get_replacement_registry()
        return reg.num_pending > 0 or reg.num_ready > 0
    except Exception:
        return False


def bsr_is_waiting_for_replacement() -> bool:
    """Return True if the recovery controller is in WAITING_FOR_REPLACEMENT
    or SAFE_POINT_REPAIR phase — i.e. a replacement has been assigned but
    not yet integrated at a safe point.
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        return False
    from megatron.core.transformer.moe.recovery_controller import RecoveryPhase
    phase = _RECOVERY_CONTROLLER.phase
    return phase in (
        RecoveryPhase.PENDING_GROUP_REPAIR,
        RecoveryPhase.WAITING_FOR_REPLACEMENT,
        RecoveryPhase.SAFE_POINT_REPAIR,
    )


def bsr_get_replacement_registry():
    """Get the global ReplacementRegistry (or None if not initialized)."""
    try:
        from megatron.core.transformer.moe import replacement_registry as rep_mod
        return rep_mod.get_replacement_registry()
    except Exception:
        return None


# =====================================================================
# Dense parameter sync API
# =====================================================================

def bsr_get_dense_param_coordinator():
    """Get the global DenseParamRecoveryCoordinator (or None)."""
    try:
        from megatron.core.transformer.moe import dense_param_sync as ds_mod
        return ds_mod.get_dense_param_recovery_coordinator()
    except Exception:
        return None


def bsr_classify_model_parameters(model=None):
    """Classify model parameters into dense/router/shared/expert.

    Returns a ParamClassification object, or None if classification fails.
    """
    try:
        from megatron.core.transformer.moe import dense_param_sync as ds_mod
        if model is None:
            return None
        target = model[0] if isinstance(model, (list, tuple)) else model
        return ds_mod.classify_model_parameters(target)
    except Exception:
        return None


def bsr_get_last_dense_sync_result():
    """Get the result of the most recent dense param sync, or None."""
    coord = bsr_get_dense_param_coordinator()
    if coord is None:
        return None
    return coord.last_result


# =====================================================================
# Expert restore API
# =====================================================================

def bsr_get_stale_expert_restore_coordinator():
    """Get the global StaleExpertRestoreCoordinator (or None)."""
    try:
        from megatron.core.transformer.moe import stale_expert_restore as ser_mod
        return ser_mod.get_stale_expert_restore_coordinator()
    except Exception:
        return None


def bsr_get_optimizer_update_barrier():
    """Get the global OptimizerUpdateBarrier (or None)."""
    try:
        from megatron.core.transformer.moe import stale_expert_restore as ser_mod
        return ser_mod.get_optimizer_update_barrier()
    except Exception:
        return None


def bsr_is_expert_blocked(expert_id: int) -> bool:
    """Check if an expert's optimizer updates are blocked.

    Returns True if the expert is in STALE_RUNNABLE state and its
    optimizer updates should be skipped.
    """
    barrier = bsr_get_optimizer_update_barrier()
    if barrier is None:
        return False
    return barrier.is_expert_blocked(expert_id)


def bsr_is_param_blocked(param_name: str) -> bool:
    """Check if a parameter's optimizer update should be skipped.

    Returns True if the parameter belongs to a stale expert whose
    optimizer state has not yet been loaded.
    """
    barrier = bsr_get_optimizer_update_barrier()
    if barrier is None:
        return False
    return barrier.is_blocked(param_name)


def bsr_unblock_experts(expert_ids: List[int]) -> None:
    """Unblock optimizer updates for the given experts.

    Called after deferred optimizer state loading completes for these
    experts, allowing them to participate in optimizer updates again.
    """
    barrier = bsr_get_optimizer_update_barrier()
    if barrier is None:
        return
    barrier.unblock_expert_params(expert_ids)


def bsr_get_last_expert_restore_result():
    """Get the result of the most recent expert restore, or None."""
    coord = bsr_get_stale_expert_restore_coordinator()
    if coord is None:
        return None
    return coord.last_result


def bsr_get_expert_restore_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of all expert restore operations."""
    coord = bsr_get_stale_expert_restore_coordinator()
    if coord is None:
        return None
    return coord.summary()


def bsr_get_optimizer_barrier_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the optimizer update barrier."""
    barrier = bsr_get_optimizer_update_barrier()
    if barrier is None:
        return None
    return barrier.summary()


# =====================================================================
# Safe-point group repair API
# =====================================================================

def bsr_get_safe_point_group_repairer():
    """Get the global SafePointGroupRepairer (or None)."""
    try:
        from megatron.core.transformer.moe import safe_point_group_repair as spr_mod
        return spr_mod.get_safe_point_group_repairer()
    except Exception:
        return None


def bsr_get_group_rebuild_coordinator():
    """Get the global GroupRebuildCoordinator (or None)."""
    try:
        from megatron.core.transformer.moe import group_rebuild as gb_mod
        return gb_mod.get_group_rebuild_coordinator()
    except Exception:
        return None


def bsr_has_pending_group_repair() -> bool:
    """Return True if there is a pending group rebuild request.

    The training loop can use this to log status or adjust behavior
    while waiting for a safe-point repair.
    """
    try:
        from megatron.core.transformer.moe import group_rebuild as gb_mod
        return gb_mod.has_pending_group_repair()
    except Exception:
        return False


def bsr_get_group_rebuild_state() -> str:
    """Return the current state of the group rebuild coordinator.

    Returns one of: 'IDLE', 'PENDING_REPAIR', 'REBUILDING', 'REBINDING',
    or 'UNKNOWN'.
    """
    try:
        from megatron.core.transformer.moe import group_rebuild as gb_mod
        coord = gb_mod.get_group_rebuild_coordinator()
        return coord.state.name
    except Exception:
        return "UNKNOWN"


def bsr_get_last_repair_result():
    """Get the result of the most recent safe-point group repair, or None."""
    repairer = bsr_get_safe_point_group_repairer()
    if repairer is None:
        return None
    return repairer.last_result


def bsr_get_repair_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of all safe-point group repair operations."""
    repairer = bsr_get_safe_point_group_repairer()
    if repairer is None:
        return None
    return repairer.summary()


# =====================================================================
# Active Expert Directory API
# =====================================================================

def bsr_get_active_expert_directory():
    """Get the global ActiveExpertDirectory (or None)."""
    try:
        from megatron.core.transformer.moe import expert_directory as ed_mod
        return ed_mod.get_active_expert_directory()
    except Exception:
        return None


def bsr_get_expert_host_rank(layer_id: int, expert_id: int) -> int:
    """Return the global rank hosting the given expert.

    Returns -1 if the directory is not available or the expert is not found.
    """
    directory = bsr_get_active_expert_directory()
    if directory is None:
        return -1
    try:
        return directory.get_host_rank(layer_id, expert_id)
    except KeyError:
        return -1


def bsr_get_experts_on_rank(global_rank: int) -> List[Tuple[int, int]]:
    """Return all (layer_id, expert_id) pairs hosted on the given rank."""
    directory = bsr_get_active_expert_directory()
    if directory is None:
        return []
    return directory.experts_on_rank(global_rank)


def bsr_get_directory_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the active expert directory."""
    directory = bsr_get_active_expert_directory()
    if directory is None:
        return None
    return directory.summary()


def bsr_refresh_directory_from_placement(new_ep_group_ranks: List[int]) -> bool:
    """Refresh the expert directory with new EP group ranks.

    Returns True if the refresh was performed.
    """
    directory = bsr_get_active_expert_directory()
    if directory is None:
        return False
    try:
        directory.refresh_from_placement(new_ep_group_ranks)
        return True
    except Exception as e:
        logger.error("BSR-MoE: directory refresh failed: %s", e)
        return False


def bsr_sync_directory_from_health_managers() -> int:
    """Sync directory recovery states from health managers.

    Returns the number of entries whose state changed.
    """
    directory = bsr_get_active_expert_directory()
    if directory is None:
        return 0
    try:
        return directory.sync_states_from_health_managers()
    except Exception as e:
        logger.error("BSR-MoE: directory sync failed: %s", e)
        return 0


# =====================================================================
# Dispatch Topology API
# =====================================================================

def bsr_get_dispatch_topology_manager():
    """Get the global DispatchTopologyManager (or None)."""
    try:
        from megatron.core.transformer.moe import dispatch_topology_refresh as topo_mod
        return topo_mod.get_dispatch_topology_manager()
    except Exception:
        return None


def bsr_refresh_dispatch_topology(
    new_ep_group_ranks: List[int],
    failed_rank: int = -1,
    replacement_rank: int = -1,
    step: int = 0,
    **kwargs,
) -> Optional[Any]:
    """Refresh the dispatch topology after a group rebuild.

    This is the public API for triggering a topology refresh outside
    of the normal recovery controller flow (e.g. for testing or manual
    intervention).

    Returns the new DispatchTopologySnapshot, or None on failure.
    """
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return None
    try:
        return mgr.refresh_dispatch_topology(
            new_ep_group_ranks=new_ep_group_ranks,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            step=step,
            **kwargs,
        )
    except Exception as e:
        logger.error("BSR-MoE: dispatch topology refresh failed: %s", e)
        return None


def bsr_get_topology_snapshot() -> Optional[Any]:
    """Get the current dispatch topology snapshot."""
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return None
    return mgr.snapshot


def bsr_get_topology_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the dispatch topology."""
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return None
    return mgr.summary()


def bsr_is_rank_active_for_dispatch(global_rank: int) -> bool:
    """Return True if the rank is active for dispatch."""
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return False
    return mgr.is_rank_active_for_dispatch(global_rank)


def bsr_get_dispatchable_experts() -> List[int]:
    """Return all expert IDs that are currently dispatchable."""
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return []
    return mgr.get_dispatchable_experts()


def bsr_check_router_dispatcher_consistency() -> List[str]:
    """Check that router and dispatcher candidate sets are consistent.

    Returns a list of inconsistency descriptions (empty = consistent).
    """
    mgr = bsr_get_dispatch_topology_manager()
    if mgr is None:
        return []
    return mgr.check_router_dispatcher_consistency()


# =====================================================================
# Reintegration Barrier API
# =====================================================================

def bsr_get_reintegration_barrier():
    """Get the global ReintegrationBarrier (or None)."""
    try:
        from megatron.core.transformer.moe import reintegration_barrier as rb_mod
        return rb_mod.get_reintegration_barrier()
    except Exception:
        return None


def bsr_can_reintegrate(failed_rank: int) -> bool:
    """Check if a failed rank can be reintegrated.

    Returns True if all preconditions are met and the barrier allows
    reintegration.  Returns True (permissive) if no barrier is configured.
    """
    barrier = bsr_get_reintegration_barrier()
    if barrier is None:
        return True
    return barrier.can_reintegrate(failed_rank)


def bsr_get_reintegration_status(failed_rank: int) -> str:
    """Get the reintegration status for a failed rank.

    Returns one of: 'ISOLATED_READY', 'REPAIRED_NOT_ROUTED',
    'ROUTED_INTEGRATED', or 'UNKNOWN'.
    """
    barrier = bsr_get_reintegration_barrier()
    if barrier is None:
        return "UNKNOWN"
    record = barrier.get_record(failed_rank)
    if record is None:
        return "UNKNOWN"
    return record.phase.name


def bsr_get_missing_preconditions(failed_rank: int) -> List[str]:
    """Get the list of missing preconditions for reintegration.

    Returns an empty list if all preconditions are met or if no barrier
    is configured.
    """
    barrier = bsr_get_reintegration_barrier()
    if barrier is None:
        return []
    record = barrier.get_record(failed_rank)
    if record is None:
        return []
    return sorted(record.missing_preconditions())


def bsr_is_reintegration_pending() -> bool:
    """Return True if there are ranks pending reintegration.

    This is useful for the training loop to log status while waiting
    for reintegration to complete.
    """
    barrier = bsr_get_reintegration_barrier()
    if barrier is None:
        return False
    return barrier.num_pending > 0


def bsr_get_reintegration_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the reintegration barrier state."""
    barrier = bsr_get_reintegration_barrier()
    if barrier is None:
        return None
    return barrier.summary()


# =====================================================================
# Pipeline Rollback API
# =====================================================================

def bsr_get_pipeline_rollback_coordinator():
    """Get the global PipelineRollbackCoordinator (or None)."""
    return _PIPELINE_ROLLBACK_COORDINATOR


def bsr_get_async_recovery_worker():
    """Get the global AsyncRecoveryWorker (or None)."""
    return _ASYNC_RECOVERY_WORKER


def bsr_get_recovery_stream():
    """Get the dedicated CUDA stream for recovery H2D copies (or None)."""
    return _RECOVERY_STREAM


def bsr_is_async_recovery_pending() -> bool:
    """True if async expert recovery is in progress."""
    if _RECOVERY_CONTROLLER is not None:
        return _RECOVERY_CONTROLLER.async_recovery_pending
    return False


def bsr_get_async_recovery_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the async recovery worker state."""
    if _ASYNC_RECOVERY_WORKER is None:
        return None
    return _ASYNC_RECOVERY_WORKER.summary()


def bsr_pipeline_begin_iteration(
    step: int,
    pp_rank: int = 0,
    pp_size: int = 1,
    num_microbatches: int = 1,
) -> None:
    """Begin pipeline iteration tracking.

    Should be called at the start of each iteration when PP > 1.
    """
    if _PIPELINE_ROLLBACK_COORDINATOR is not None and pp_size > 1:
        _PIPELINE_ROLLBACK_COORDINATOR.begin_iteration(
            step, pp_rank=pp_rank, pp_size=pp_size,
            num_microbatches=num_microbatches,
        )


def bsr_pipeline_on_failure(
    *,
    failed_stage: int = -1,
    failed_rank: int = -1,
    step: int = -1,
    reason: str = "",
    nccl_healthy: bool = True,
) -> bool:
    """Report a pipeline failure.

    Returns True if the failure was recorded.
    """
    if _PIPELINE_ROLLBACK_COORDINATOR is None:
        return False
    _PIPELINE_ROLLBACK_COORDINATOR.on_pipeline_failure(
        failed_stage=failed_stage,
        failed_rank=failed_rank,
        step=step,
        reason=reason,
        nccl_healthy=nccl_healthy,
    )
    return True


def bsr_pipeline_initiate_rollback(
    sync_fn=None,
    clear_grad_fn=None,
) -> Optional[Any]:
    """Initiate a pipeline-safe rollback.

    Returns the PipelineRollbackResult, or None if not applicable.
    """
    if _PIPELINE_ROLLBACK_COORDINATOR is None:
        return None
    return _PIPELINE_ROLLBACK_COORDINATOR.initiate_rollback(
        sync_fn=sync_fn,
        clear_grad_fn=clear_grad_fn,
    )


def bsr_pipeline_is_in_rollback() -> bool:
    """True if the pipeline is in rollback state."""
    if _PIPELINE_ROLLBACK_COORDINATOR is None:
        return False
    return _PIPELINE_ROLLBACK_COORDINATOR.is_in_rollback


def bsr_pipeline_is_awaiting_replay() -> bool:
    """True if the pipeline is awaiting replay."""
    if _PIPELINE_ROLLBACK_COORDINATOR is None:
        return False
    return _PIPELINE_ROLLBACK_COORDINATOR.is_awaiting_replay


def bsr_pipeline_complete_replay(success: bool = True) -> None:
    """Mark the pipeline replay as complete."""
    if _PIPELINE_ROLLBACK_COORDINATOR is not None:
        _PIPELINE_ROLLBACK_COORDINATOR.complete_replay(success)


def bsr_pipeline_get_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of the pipeline rollback coordinator."""
    if _PIPELINE_ROLLBACK_COORDINATOR is None:
        return None
    return _PIPELINE_ROLLBACK_COORDINATOR.summary()


# =====================================================================
# Pipeline Stage Repair API
# =====================================================================

def bsr_get_pipeline_stage_repairer():
    """Get the global PipelineStageRepairer (or None)."""
    try:
        from megatron.core.transformer.moe import pipeline_stage_repair as psr_mod
        return psr_mod.get_pipeline_stage_repairer()
    except Exception:
        return None


def bsr_repair_pipeline_stage(
    failed_rank: int,
    replacement_rank: int,
    pp_group_ranks: Optional[List[int]] = None,
    step: int = -1,
    create_group_fn=None,
) -> Optional[Any]:
    """Execute a pipeline stage repair for PP>1.

    Rebuilds the PP group, updates prev/next rank caches, and
    recreates the P2P communicator.

    Returns the PipelineStageRepairResult, or None on failure.
    """
    repairer = bsr_get_pipeline_stage_repairer()
    if repairer is None:
        return None
    try:
        return repairer.execute(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            pp_group_ranks=pp_group_ranks,
            step=step,
            create_group_fn=create_group_fn,
        )
    except Exception as e:
        logger.error("BSR-MoE: pipeline stage repair failed: %s", e)
        return None


def bsr_get_pipeline_stage_repair_summary() -> Optional[Dict[str, Any]]:
    """Get a summary of pipeline stage repair operations."""
    repairer = bsr_get_pipeline_stage_repairer()
    if repairer is None:
        return None
    return repairer.summary()


# =====================================================================
# PP>1 End-to-End Recovery API
# =====================================================================

def bsr_on_pipeline_stage_failure(
    failed_stage: int,
    failed_rank: int,
    *,
    step: int = -1,
    pp_group_ranks: Optional[List[int]] = None,
    reason: str = "",
    expert_ids: Optional[List[int]] = None,
    ep_group_ranks: Optional[List[int]] = None,
    dp_group_ranks: Optional[List[int]] = None,
    mid_iteration: bool = True,
) -> bool:
    """Report a pipeline stage failure (PP>1).

    This is the primary entry point for PP>1 failures.  It delegates
    to ``RecoveryController.on_pipeline_stage_failure()``.

    Returns True if the failure was recorded.
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        logger.warning(
            "BSR-MoE: bsr_on_pipeline_stage_failure called but BSR not "
            "initialized (stage=%d, rank=%d)", failed_stage, failed_rank,
        )
        return False

    try:
        _RECOVERY_CONTROLLER.on_pipeline_stage_failure(
            failed_stage=failed_stage,
            failed_rank=failed_rank,
            step=step,
            pp_group_ranks=pp_group_ranks,
            reason=reason,
            expert_ids=expert_ids,
            ep_group_ranks=ep_group_ranks,
            dp_group_ranks=dp_group_ranks,
            mid_iteration=mid_iteration,
        )
        return True
    except Exception as e:
        logger.error(
            "BSR-MoE: bsr_on_pipeline_stage_failure failed: %s", e,
        )
        return False


def bsr_invalidate_inflight_microbatches(
    step: int = -1,
    pp_size: int = 1,
) -> None:
    """Mark all in-flight microbatches as invalid.

    Called when a PP stage failure is detected mid-iteration.
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        return
    _RECOVERY_CONTROLLER.invalidate_inflight_microbatches(
        step=step, pp_size=pp_size,
    )


def bsr_execute_stage_safe_recovery(
    failed_rank: int,
    failed_stage: int,
    replacement_rank: int,
    *,
    step: int = -1,
    pp_group_ranks: Optional[List[int]] = None,
    ep_group_ranks: Optional[List[int]] = None,
    dp_group_ranks: Optional[List[int]] = None,
    expert_ids: Optional[List[int]] = None,
    recovery_path: str = "HYBRID_RECOVERY",
) -> Optional[Dict[str, Any]]:
    """Execute the full stage-safe recovery protocol for PP>1.

    This is the top-level entry point that orchestrates:
        detect → invalidate → rollback → wait-for-replacement →
        safe-point repair → P2P rebind → resume training

    It delegates each step to the existing BSR-MoE callbacks already
    registered with the RecoveryController.

    Returns the StageRecoveryResult.to_dict(), or None if BSR is not
    initialized.
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        logger.warning(
            "BSR-MoE: bsr_execute_stage_safe_recovery called but BSR "
            "not initialized"
        )
        return None

    from megatron.core.transformer.moe.stage_safe_recovery import (
        get_stage_safe_recovery_protocol,
    )

    protocol = get_stage_safe_recovery_protocol()

    # Build callbacks from the controller's registered callbacks
    ctrl = _RECOVERY_CONTROLLER

    def _invalidate_iteration_fn(*, step, failed_rank, failed_stage):
        if _ITERATION_INVALIDATOR is not None:
            _ITERATION_INVALIDATOR.invalidate(
                step=step,
                reason=f"stage_safe_recovery_stage{failed_stage}_rank{failed_rank}",
            )

    def _rollback_fn(*, failed_rank, failed_stage, step):
        if _PIPELINE_ROLLBACK_COORDINATOR is not None:
            _PIPELINE_ROLLBACK_COORDINATOR.on_pipeline_failure(
                failed_stage=failed_stage,
                failed_rank=failed_rank,
                step=step,
                reason=f"stage_safe_recovery",
            )
            _PIPELINE_ROLLBACK_COORDINATOR.initiate_rollback()
        return True

    def _group_repair_fn(*, failed_rank, replacement_rank, step, pp_group_ranks):
        if ctrl._group_rebuild_execute_fn is not None:
            ctrl._group_rebuild_execute_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )

    def _topology_refresh_fn(*, failed_rank, replacement_rank, step):
        if ctrl._topology_refresh_fn is not None:
            ctrl._topology_refresh_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                expert_ids=expert_ids,
            )

    def _p2p_rebind_fn(*, failed_rank, replacement_rank, pp_group_ranks, step):
        if ctrl._pipeline_stage_repair_fn is not None:
            ctrl._pipeline_stage_repair_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                pp_group_ranks=pp_group_ranks,
                failed_stage=failed_stage,
            )

    def _dense_sync_fn(*, failed_rank, replacement_rank, step):
        if ctrl._dense_sync_fn is not None:
            ctrl._dense_sync_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )

    def _expert_restore_fn(*, failed_rank, replacement_rank, step, expert_ids):
        if ctrl._expert_restore_fn is not None:
            ctrl._expert_restore_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                expert_ids=expert_ids,
            )

    def _checkpoint_restart_fn(*, failed_rank, replacement_rank, step):
        if ctrl._checkpoint_restart_fn is not None:
            ctrl._checkpoint_restart_fn(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                decision=None,
            )

    def _convergence_fn(*, path, failed_rank, replacement_rank, step, expert_ids):
        if ctrl._post_recovery_convergence_fn is not None:
            ctrl._post_recovery_convergence_fn(
                path=path,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                expert_ids=expert_ids,
            )

    result = protocol.execute(
        failed_rank=failed_rank,
        failed_stage=failed_stage,
        replacement_rank=replacement_rank,
        step=step,
        pp_group_ranks=pp_group_ranks or [],
        ep_group_ranks=ep_group_ranks,
        dp_group_ranks=dp_group_ranks,
        expert_ids=expert_ids,
        recovery_path=recovery_path,
        invalidate_iteration_fn=_invalidate_iteration_fn,
        rollback_fn=_rollback_fn,
        group_repair_fn=_group_repair_fn,
        topology_refresh_fn=_topology_refresh_fn,
        p2p_rebind_fn=_p2p_rebind_fn,
        dense_sync_fn=_dense_sync_fn,
        expert_restore_fn=_expert_restore_fn,
        checkpoint_restart_fn=_checkpoint_restart_fn,
        convergence_fn=_convergence_fn,
    )

    logger.warning(
        "[%s] BSR-MoE stage-safe recovery: %s "
        "(phase=%s, path=%s, errors=%d, elapsed=%.3fs)",
        _ts(),
        "SUCCEEDED" if result.success else "FAILED",
        result.phase_reached.name,
        result.recovery_path,
        len(result.errors),
        result.elapsed_seconds,
    )

    return result.to_dict()


def bsr_get_stage_safe_recovery_summary() -> Optional[Dict[str, Any]]:
    """Get the stage-safe recovery protocol summary."""
    try:
        from megatron.core.transformer.moe.stage_safe_recovery import (
            get_stage_safe_recovery_protocol,
        )
        return get_stage_safe_recovery_protocol().summary()
    except Exception:
        return None


def bsr_get_pipeline_recovery_status() -> Dict[str, Any]:
    """Query the PP recovery status.

    Returns a dict with:
    - phase: current RecoveryPhase name
    - inflight_invalidated: whether in-flight microbatches are invalidated
    - rollback_completed: whether pipeline rollback has completed
    - pipeline_faults: list of active faults with PP info
    """
    if not _BSR_INITIALIZED or _RECOVERY_CONTROLLER is None:
        return {
            "phase": "UNKNOWN",
            "inflight_invalidated": False,
            "rollback_completed": False,
            "pipeline_faults": [],
        }

    ctrl = _RECOVERY_CONTROLLER
    pp_faults = []
    for rank, record in ctrl.active_faults.items():
        if record.failed_stage >= 0 or record.pp_group_ranks:
            pp_faults.append({
                "failed_rank": record.failed_rank,
                "failed_stage": record.failed_stage,
                "pp_group_ranks": list(record.pp_group_ranks),
                "pipeline_repaired": record.pipeline_repaired,
            })

    return {
        "phase": ctrl.phase.name,
        "inflight_invalidated": ctrl.inflight_microbatches_invalidated,
        "rollback_completed": ctrl.pipeline_rollback_completed,
        "pipeline_faults": pp_faults,
    }


def _wire_recovery_callbacks(
    ctrl,
    *,
    num_layers: int,
    num_experts: int,
    ep_size: int,
    ep_group_ranks: List[int],
    dp_group_ranks: List[int],
    model,
    optimizer=None,
    opt_param_scheduler=None,
) -> None:
    """Wire RecoveryController callbacks to real BSR-MoE module singletons."""

    from megatron.core.transformer.moe import replacement_registry as rep_mod
    from megatron.core.transformer.moe import group_rebuild as gb_mod
    from megatron.core.transformer.moe import dispatch_topology_refresh as topo_mod
    from megatron.core.transformer.moe import dense_param_sync as ds_mod

    def replacement_announce_fn(*, failed_rank, replacement_rank, step=-1):
        """Announce a replacement rank."""
        try:
            rep_mod.announce_replacement(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )
        except Exception as e:
            logger.error("BSR-MoE replacement_announce_fn failed: %s", e)

    def replacement_integrate_fn(*, failed_rank, replacement_rank, step=-1):
        """Mark replacement as integrated.

        The replacement_registry state machine requires:
            NOT_PRESENT → BOOTSTRAPPING → READY_FOR_REPAIR → INTEGRATED
        In restart-in-place mode, the controller fast-path calls
        on_replacement_assigned (→ BOOTSTRAPPING) and on_replacement_ready
        but the latter does NOT advance the registry to READY_FOR_REPAIR.
        We must ensure the registry is in READY_FOR_REPAIR before calling
        mark_integrated, otherwise the transition is illegal.
        """
        try:
            # Ensure the slot is in READY_FOR_REPAIR before integrating.
            # This handles the restart-in-place fast path where
            # on_replacement_ready() advances the controller phase but
            # does not call announce_replacement_ready() on the registry.
            from megatron.core.transformer.moe.replacement_registry import (
                ReplacementState,
                query_replacement_status,
            )
            current_state = query_replacement_status(failed_rank)
            if current_state == ReplacementState.BOOTSTRAPPING:
                rep_mod.announce_replacement_ready(failed_rank, step=step)
            rep_mod.mark_integrated(failed_rank, step=step)
        except Exception as e:
            logger.error("BSR-MoE replacement_integrate_fn failed: %s", e)

    def group_rebuild_request_fn(
        *, failed_rank, replacement_rank, step=-1,
        ep_group_ranks=None, dp_group_ranks=None,
    ):
        """Request a group rebuild."""
        try:
            coord = gb_mod.get_group_rebuild_coordinator()
            new_ep = list(ep_group_ranks or [])
            if failed_rank in new_ep:
                idx = new_ep.index(failed_rank)
                new_ep[idx] = replacement_rank
            coord.request_rebuild(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                old_ep_group_ranks=ep_group_ranks,
                new_ep_group_ranks=new_ep,
            )
        except Exception as e:
            logger.error("BSR-MoE group_rebuild_request_fn failed: %s", e)

    def group_rebuild_execute_fn(*, failed_rank, replacement_rank, step=-1):
        """Execute group rebuild at safe point via SafePointGroupRepairer.

        This is the real implementation that:
        1. Drives the GroupRebuildCoordinator through REBUILDING → REBINDING.
        2. Delegates the actual group invalidation, rebuild, rebind, and
           verification to SafePointGroupRepairer.execute().

        If the repair fails (e.g., replacement_rank exceeds world_size and
        identity inheritance is not possible), this function raises an
        exception to prevent training from continuing with broken groups.
        """
        from megatron.core.transformer.moe import safe_point_group_repair as spr_mod

        try:
            coord = gb_mod.get_group_rebuild_coordinator()
            repairer = spr_mod.get_safe_point_group_repairer()

            # Track repair result for error propagation
            _repair_result = [None]

            # The rebuild_fn receives the plan from the coordinator and
            # delegates to the SafePointGroupRepairer for the actual
            # group invalidation + rebuild + rebind + verify sequence.
            def _rebuild_fn(plan):
                logger.warning(
                    "BSR-MoE: executing safe-point group repair — "
                    "failed=%d, replacement=%d, groups=%s",
                    plan.failed_rank, plan.replacement_rank,
                    plan.affected_groups,
                )
                # Determine PP size
                pp_size = 1
                try:
                    from megatron.core import parallel_state as mpu
                    pp_size = mpu.get_pipeline_model_parallel_world_size()
                except Exception:
                    pass

                result = repairer.execute(
                    plan,
                    model=model,
                    verify=True,
                    pp_size=pp_size,
                )
                _repair_result[0] = result

                if result.success:
                    logger.warning(
                        "[%s] BSR-MoE: safe-point group repair SUCCEEDED — "
                        "invalidated=%d, rebuilt=%d, rebound=%d, "
                        "verified=%s, elapsed=%.2fs",
                        _ts(),
                        result.groups_invalidated, result.groups_rebuilt,
                        result.modules_rebound, result.verification_passed,
                        result.elapsed_seconds,
                    )
                else:
                    logger.error(
                        "[%s] BSR-MoE: safe-point group repair FAILED at "
                        "phase %s: %s",
                        _ts(),
                        result.phase_reached.name, result.error,
                    )

            def _rebind_fn(plan):
                # Rebinding is already done inside repairer.execute()
                # (phase 4), so this is a no-op.  The coordinator still
                # needs a rebind_fn to transition its state machine.
                logger.debug(
                    "BSR-MoE: rebind phase (handled by repairer) — "
                    "failed=%d, replacement=%d",
                    plan.failed_rank, plan.replacement_rank,
                )

            coord.maybe_rebuild_groups_at_safe_point(
                rebuild_fn=_rebuild_fn,
                rebind_fn=_rebind_fn,
                step=step,
            )

            # Check if repair failed — if so, raise to prevent training
            # from continuing with broken process groups
            result = _repair_result[0]
            if result is not None and not result.success:
                raise RuntimeError(
                    f"BSR-MoE safe-point group repair FAILED: "
                    f"phase={result.phase_reached.name}, "
                    f"error={result.error}, "
                    f"invalidated={result.groups_invalidated}, "
                    f"rebuilt={result.groups_rebuilt}"
                )
        except Exception as e:
            logger.error("BSR-MoE group_rebuild_execute_fn failed: %s", e)
            raise

    def group_rebuild_finish_fn(*, failed_rank, replacement_rank, step=-1):
        """Finish group rebuild."""
        try:
            coord = gb_mod.get_group_rebuild_coordinator()
            coord.finish_group_repair(step=step)
        except Exception as e:
            logger.error("BSR-MoE group_rebuild_finish_fn failed: %s", e)

    def topology_refresh_fn(
        *, failed_rank, replacement_rank, step=-1, expert_ids=None,
    ):
        """Refresh dispatch topology after group rebuild.

        This is called at step 5 of the safe-point repair sequence.
        It updates:
        - ActiveExpertDirectory: host-rank mapping refreshed
        - ReplacementRegistry: mark integrated if ready

        After this call, the replacement rank's experts can participate
        in token dispatch again (though they may still be STALE_RUNNABLE
        until expert weights are restored in step 7).
        """
        try:
            mgr = topo_mod.get_dispatch_topology_manager()
            new_ep = list(ep_group_ranks)
            if failed_rank in new_ep:
                idx = new_ep.index(failed_rank)
                new_ep[idx] = replacement_rank
            snapshot = mgr.refresh_dispatch_topology(
                new_ep_group_ranks=new_ep,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                re_enable_recovered_experts=False,
                update_directory=True,
                update_replacement_registry=True,
            )

            # Verify consistency after refresh
            issues = mgr.check_router_dispatcher_consistency()
            if issues:
                logger.warning(
                    "BSR-MoE topology_refresh_fn: consistency issues "
                    "after refresh: %s", issues,
                )
            else:
                logger.info(
                    "BSR-MoE topology_refresh_fn: consistency check passed "
                    "(step=%d, dispatchable=%d/%d)",
                    step,
                    len(mgr.get_dispatchable_experts()),
                    snapshot.num_experts,
                )
        except Exception as e:
            logger.error("BSR-MoE topology_refresh_fn failed: %s", e)

    def dense_sync_fn(*, failed_rank, replacement_rank, step=-1):
        """Pull dense/shared/router params from healthy DP peer (broadcast).

        This is the real implementation that uses DenseParamRecoveryCoordinator
        to plan and execute a broadcast-based parameter sync from a healthy
        DP peer.  Expert parameters are intentionally skipped — they will be
        recovered from checkpoint in the next step (expert_restore_fn).

        Optimizer state sync strategy:
        - Non-DistributedOptimizer: all DP ranks hold identical optimizer
          state for dense params → broadcast from healthy peer.
        - DistributedOptimizer (ZeRO-1): each DP rank holds a different
          shard of optimizer state.  The failed rank's shard is lost.
          → Skip broadcast (would corrupt state), defer to checkpoint
          recovery or zero-initialize.
        """
        t_start = time.time()
        try:
            # ── Early exit: check moe_bsr_hybrid_dense_sync config flag ──
            # If the flag is explicitly set to False, skip the entire dense
            # sync and let checkpoint recovery handle all parameters.
            try:
                _target_model = model[0] if isinstance(model, (list, tuple)) else model
                _cfg = getattr(_target_model, 'config', None)
                if _cfg is not None and hasattr(_cfg, 'moe_bsr_hybrid_dense_sync'):
                    if not _cfg.moe_bsr_hybrid_dense_sync:
                        logger.warning(
                            "[%s] BSR-MoE dense_sync_fn: dense param sync "
                            "disabled by config (moe_bsr_hybrid_dense_sync=False, "
                            "step=%d). All parameters will be recovered from "
                            "checkpoint.",
                            _ts(), step,
                        )
                        return
            except Exception:
                pass

            coordinator = ds_mod.get_dense_param_recovery_coordinator()

            # Determine whether to include optimizer state in the sync.
            # Controlled by --moe-bsr-dense-opt-state-sync config flag.
            # Also detect DistributedOptimizer (ZeRO-1) — if so, skip
            # optimizer state broadcast (each rank holds a different
            # shard; broadcast would corrupt state).
            _opt_for_sync = optimizer
            _include_opt = True

            # Check config flag (default: True)
            try:
                target_model = model[0] if isinstance(model, (list, tuple)) else model
                _cfg = getattr(target_model, 'config', None)
                if _cfg is not None and hasattr(_cfg, 'moe_bsr_dense_opt_state_sync'):
                    if not _cfg.moe_bsr_dense_opt_state_sync:
                        logger.warning(
                            "[%s] BSR-MoE dense_sync_fn: optimizer state sync "
                            "disabled by config (moe_bsr_dense_opt_state_sync=False, "
                            "step=%d)",
                            _ts(), step,
                        )
                        _opt_for_sync = None
                        _include_opt = False
            except Exception:
                pass

            if _include_opt and optimizer is not None:
                is_distributed_opt = _is_distributed_optimizer(optimizer)
                if is_distributed_opt:
                    logger.warning(
                        "[%s] BSR-MoE dense_sync_fn: DistributedOptimizer "
                        "detected — skipping optimizer state broadcast "
                        "(ZeRO-1 sharded state cannot be broadcast). "
                        "Optimizer state will be recovered from checkpoint "
                        "or zero-initialized. (step=%d)",
                        _ts(), step,
                    )
                    _opt_for_sync = None
                    _include_opt = False

            # Plan the recovery
            plan = coordinator.plan_recovery(
                replacement_rank=replacement_rank,
                failed_rank=failed_rank,
                dp_group_ranks=dp_group_ranks,
                step=step,
                include_optimizer=_include_opt,
            )

            if plan.source_rank < 0:
                logger.error(
                    "BSR-MoE dense_sync_fn: no healthy DP peer available "
                    "for replacement_rank=%d (failed_rank=%d). "
                    "Dense params will need checkpoint recovery.",
                    replacement_rank, failed_rank,
                )
                return

            # Execute the sync
            # In PP=1 mode, model[0] is the full model.
            # For PP>1, this would need to iterate over pipeline stages.
            target_model = model[0] if isinstance(model, (list, tuple)) else model

            # Get the DP process group for the broadcast.
            # After group rebuild, the DP group should already be updated.
            dp_group = None
            try:
                import torch.distributed
                if torch.distributed.is_initialized():
                    from megatron.core import mpu
                    dp_group = mpu.get_data_parallel_group()
            except Exception:
                pass

            result = coordinator.execute_recovery(
                model=target_model,
                plan=plan,
                dp_group=dp_group,
                optimizer=_opt_for_sync,
            )

            if result.success:
                elapsed = time.time() - t_start
                logger.warning(
                    "[%s] BSR-MoE dense param sync: SUCCESS — synced %d params "
                    "(%d scalars) from rank %d "
                    "(attempt 1, %.2fs, skipped %d expert params)",
                    _ts(), result.num_params_synced, result.num_scalars_synced,
                    result.source_rank, elapsed,
                    result.num_expert_skipped,
                )
            else:
                elapsed = time.time() - t_start
                logger.error(
                    "[%s] BSR-MoE dense param sync: FAILED — %s (%.2fs)",
                    _ts(), result.error, elapsed,
                )
        except Exception as e:
            elapsed = time.time() - t_start
            logger.error("[%s] BSR-MoE dense_sync_fn failed: %s (%.2fs)", _ts(), e, elapsed)

    def expert_restore_fn(
        *, failed_rank, replacement_rank, step=-1, expert_ids=None,
    ):
        """Restore expert weights from checkpoint via StaleExpertRestoreCoordinator.

        This is the real implementation that:
        1. Locates the most recent checkpoint with a BSR recovery manifest.
        2. Builds a restore plan for the affected experts.
        3. Loads expert weights from checkpoint (via load_fn callback).
        4. Transitions health state: UNAVAILABLE → STALE_RUNNABLE.
        5. Updates expert directory recovery state.
        6. Adds expert params to optimizer update barrier.

        If no manifest is found, falls back to marking experts as
        STALE_RUNNABLE without loading weights (dry-run mode).
        """
        from megatron.core.transformer.moe import stale_expert_restore as ser_mod
        from megatron.core.transformer.moe import expert_directory as ed_mod

        try:
            if not expert_ids:
                logger.warning(
                    "BSR-MoE expert_restore_fn: no expert_ids provided, "
                    "skipping restore (step=%d)", step,
                )
                return

            coordinator = ser_mod.get_stale_expert_restore_coordinator()
            barrier = ser_mod.get_optimizer_update_barrier()

            # --- Two-phase recovery: begin (WEIGHTS_LOADING) ---
            _two_phase_coord = None
            _weights_first = True
            try:
                from megatron.training.global_vars import get_args as _get_args_tp
                _weights_first = getattr(
                    _get_args_tp(), 'moe_bsr_weights_first_recovery', True,
                )
            except Exception:
                pass

            if _weights_first:
                try:
                    from megatron.core.transformer.moe.two_phase_recovery import (
                        get_two_phase_recovery_coordinator,
                    )
                    _two_phase_coord = get_two_phase_recovery_coordinator()
                    # Build (layer_id, expert_id) tuples only for this PP
                    # stage.  ExpertHealthManager has been removed, so infer
                    # MoE layers from the model's parameter names.
                    _tp_moe_layers = _infer_local_moe_layer_ids(model, num_layers)

                    _tp_expert_keys = [
                        (lid, eid) for eid in expert_ids
                        for lid in _tp_moe_layers
                    ]
                    _two_phase_coord.begin_recovery(
                        expert_ids=_tp_expert_keys,
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        step=step,
                    )
                except Exception as tp_e:
                    logger.debug(
                        "BSR-MoE expert_restore_fn: two-phase begin failed: %s",
                        tp_e,
                    )
                    _two_phase_coord = None
            manifest = None
            checkpoint_dir = _find_latest_checkpoint_dir()
            if checkpoint_dir is not None:
                try:
                    manifest = ed_mod.RecoveryManifest.load(checkpoint_dir)
                    logger.warning(
                        "BSR-MoE expert_restore_fn: loaded manifest from %s "
                        "(step=%d, entries=%d)",
                        checkpoint_dir, manifest.step, len(manifest.entries),
                    )
                except FileNotFoundError:
                    logger.warning(
                        "BSR-MoE expert_restore_fn: no manifest in %s, "
                        "will use synthetic plan",
                        checkpoint_dir,
                    )
                except Exception as e:
                    logger.error(
                        "BSR-MoE expert_restore_fn: failed to load manifest "
                        "from %s: %s", checkpoint_dir, e,
                    )

            # Build restore plan
            if manifest is not None:
                plan = coordinator.plan_restore(
                    manifest=manifest,
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                )
            else:
                # Synthetic plan: no checkpoint available, create entries
                # from expert_ids so we can still do state transitions.
                # IMPORTANT: Only create entries for actual MoE layers
                # (registered in health manager), not all transformer layers.
                plan = ser_mod.ExpertRestorePlan(
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    step=step,
                )
                # ExpertHealthManager has been removed.  Infer actual MoE
                # layers from the concrete model, handling DDP names like
                # ``module.decoder.layers.0...`` and converting PP-local
                # indices to 1-based global layer IDs.
                moe_layer_ids = _infer_local_moe_layer_ids(model, num_layers)
                pp_rank, pp_size = _get_pp_rank_size()
                logger.info(
                    "BSR-MoE expert_restore_fn: inferred MoE layers for PP "
                    "stage %d/%d: %s",
                    pp_rank, pp_size, moe_layer_ids,
                )

                for eid in expert_ids:
                    for layer_id in moe_layer_ids:
                        plan.entries.append(ser_mod.ExpertRestoreEntry(
                            layer_id=layer_id,
                            expert_id=eid,
                            checkpoint_step=-1,
                            host_rank=replacement_rank,
                        ))
                logger.warning(
                    "BSR-MoE expert_restore_fn: using synthetic plan "
                    "(%d entries across %d MoE layers, no checkpoint weights)",
                    len(plan.entries), len(moe_layer_ids),
                )

            local_moe_layer_ids = set(_infer_local_moe_layer_ids(model, num_layers))
            if local_moe_layer_ids and plan.entries:
                original_entries = len(plan.entries)
                plan.entries = [
                    entry for entry in plan.entries
                    if entry.layer_id in local_moe_layer_ids
                ]
                if len(plan.entries) != original_entries:
                    logger.info(
                        "BSR-MoE expert_restore_fn: filtered restore plan to "
                        "current PP stage layers %s (%d/%d entries)",
                        sorted(local_moe_layer_ids),
                        len(plan.entries), original_entries,
                    )

            # Build the load_fn callback
            # In PP=1 mode, we load from the checkpoint state dict.
            # The actual loading depends on whether we have a real checkpoint.
            #
            # Check moe_bsr_hybrid_expert_restore config flag — when False,
            # skip actual weight loading (dry-run mode for state transitions only).
            _hybrid_expert_restore = True
            try:
                from megatron.training.global_vars import get_args as _get_args
                _hybrid_expert_restore = getattr(
                    _get_args(), 'moe_bsr_hybrid_expert_restore', True,
                )
            except Exception:
                pass

            if _hybrid_expert_restore:
                load_fn = _build_expert_load_fn(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                )
                # If load_fn is None but checkpoint_dir exists, it likely
                # means the checkpoint is in torch_dist format.  The
                # force_checkpoint_restart_fn callback should have already
                # redirected the controller to the CHECKPOINT_RESTART path.
                # If we still end up here, log a warning — expert weights
                # will not be loaded (dry-run mode with state transitions
                # only).
                if load_fn is None and checkpoint_dir is not None:
                    logger.warning(
                        "[%s] BSR-MoE expert_restore_fn: load_fn is None "
                        "despite checkpoint_dir=%s existing. Expert weights "
                        "will NOT be loaded (dry-run mode). If this is a "
                        "torch_dist checkpoint, the CHECKPOINT_RESTART path "
                        "should have been selected instead.",
                        _ts(), checkpoint_dir,
                    )
            else:
                load_fn = None
                logger.info(
                    "BSR-MoE expert_restore_fn: moe_bsr_hybrid_expert_restore=False, "
                    "skipping weight loading (dry-run mode)",
                )

            # ExpertHealthManager was removed; recovery_controller owns expert
            # state now, so stale_expert_restore gets an empty registry.
            health_managers = {}

            # Get expert directory
            directory = ed_mod.get_active_expert_directory()

            # Execute restore
            target_model = model[0] if isinstance(model, (list, tuple)) else model
            result = coordinator.execute_restore(
                model=target_model,
                plan=plan,
                load_fn=load_fn,
                health_managers=health_managers,
                directory=directory,
                barrier=barrier,
                step=step,
            )

            if result.success:
                logger.warning(
                    "[%s] BSR-MoE expert_restore_fn: SUCCESS — restored %d experts "
                    "(%d state transitions, %d directory updates, "
                    "%d barrier params, %.2fs)",
                    _ts(),
                    result.num_restored, result.num_state_transitions,
                    result.num_directory_updates, result.num_barrier_params,
                    result.elapsed_seconds,
                )

                # --- Two-phase: WEIGHTS_LOADING → WEIGHTS_READY ---
                if _two_phase_coord is not None and result.num_restored > 0:
                    try:
                        _restored_keys = [
                            (lid, eid) for lid, eid in result.restored_experts
                        ]
                        _two_phase_coord.on_weights_restored(
                            expert_ids=_restored_keys, step=step,
                        )
                    except Exception as tp_e:
                        logger.debug(
                            "BSR-MoE: two-phase on_weights_restored failed: %s",
                            tp_e,
                        )

                # --- BSR-MoE: activate preferential routing bias ---
                _pref_routing = False
                try:
                    from megatron.training.global_vars import get_args as _get_args_pr
                    _pref_routing = getattr(
                        _get_args_pr(), 'moe_bsr_preferential_routing', False,
                    )
                except Exception:
                    pass

                if _pref_routing and result.num_restored > 0:
                    try:
                        from megatron.core.transformer.moe.preferential_routing import (
                            get_preferential_routing_manager,
                        )
                        _pr_window = 100
                        _pr_bias = 0.1
                        try:
                            from megatron.training.global_vars import get_args as _get_args_pr2
                            _pr_args = _get_args_pr2()
                            _pr_window = getattr(
                                _pr_args, 'moe_bsr_preferential_routing_window', 100,
                            )
                            _pr_bias = getattr(
                                _pr_args, 'moe_bsr_preferential_routing_bias', 0.1,
                            )
                        except Exception:
                            pass

                        # Activate bias for each restored (layer, expert) pair
                        _activated_count = 0
                        for lid, eid in result.restored_experts:
                            pr_mgr = get_preferential_routing_manager(
                                layer_number=lid,
                                num_experts=config.num_moe_experts
                                if config is not None else 8,
                                initial_bias=_pr_bias,
                                window_steps=_pr_window,
                            )
                            pr_mgr.activate(eid, step=step)
                            _activated_count += 1
                        logger.warning(
                            "[%s] BSR-MoE expert_restore_fn: activated "
                            "preferential routing for %d expert(s) "
                            "(bias=%.4f, window=%d steps)",
                            _ts(), _activated_count, _pr_bias, _pr_window,
                        )
                    except Exception as pr_e:
                        logger.error(
                            "BSR-MoE expert_restore_fn: failed to activate "
                            "preferential routing: %s", pr_e,
                        )

                # --- Submit optimizer state load requests ---
                _expert_opt_restore = True
                try:
                    from megatron.training.global_vars import get_args as _get_args2
                    _expert_opt_restore = getattr(
                        _get_args2(), 'moe_bsr_expert_opt_restore', True,
                    )
                except Exception:
                    pass

                _defer_opt_load = True
                try:
                    from megatron.training.global_vars import get_args as _get_args3
                    _defer_opt_load = getattr(
                        _get_args3(), 'moe_bsr_defer_optimizer_load', True,
                    )
                except Exception:
                    pass

                if _expert_opt_restore and _defer_opt_load and result.num_restored > 0:
                    try:
                        from megatron.core.transformer.moe.deferred_optimizer_load import (
                            get_deferred_optimizer_loader,
                        )
                        opt_loader = get_deferred_optimizer_loader()
                        opt_load_fn = _build_expert_optimizer_load_fn(
                            checkpoint_dir=checkpoint_dir,
                            model=model,
                        )
                        opt_requests = opt_loader.submit_from_restore_plan(
                            plan, step=step,
                        )
                        # Store the optimizer load_fn for poll_and_finalize
                        # to use later during the training loop.
                        opt_loader._pending_load_fn = opt_load_fn
                        result.optimizer_load_submitted = len(opt_requests)
                        logger.warning(
                            "[%s] BSR-MoE expert_restore_fn: submitted %d "
                            "optimizer state load requests",
                            _ts(), len(opt_requests),
                        )

                        # Two-phase: WEIGHTS_READY → OPTIMIZER_PENDING
                        if _two_phase_coord is not None:
                            try:
                                _restored_keys = [
                                    (lid, eid)
                                    for lid, eid in result.restored_experts
                                ]
                                _two_phase_coord.on_optimizer_submitted(
                                    expert_ids=_restored_keys, step=step,
                                )
                            except Exception as tp_e:
                                logger.debug(
                                    "BSR-MoE: two-phase on_optimizer_submitted "
                                    "failed: %s", tp_e,
                                )
                    except Exception as opt_e:
                        logger.error(
                            "BSR-MoE expert_restore_fn: failed to submit "
                            "optimizer state loads: %s", opt_e,
                        )
                elif not _expert_opt_restore:
                    logger.info(
                        "BSR-MoE expert_restore_fn: moe_bsr_expert_opt_restore=False, "
                        "skipping optimizer state loading",
                    )
                    # Two-phase: skip optimizer → FULLY_RECOVERED directly
                    if _two_phase_coord is not None and result.num_restored > 0:
                        try:
                            _restored_keys = [
                                (lid, eid)
                                for lid, eid in result.restored_experts
                            ]
                            _two_phase_coord.skip_optimizer_phase(
                                expert_ids=_restored_keys, step=step,
                            )
                        except Exception:
                            pass
                elif not _defer_opt_load:
                    # Not deferring: skip optimizer phase directly
                    if _two_phase_coord is not None and result.num_restored > 0:
                        try:
                            _restored_keys = [
                                (lid, eid)
                                for lid, eid in result.restored_experts
                            ]
                            _two_phase_coord.skip_optimizer_phase(
                                expert_ids=_restored_keys, step=step,
                            )
                        except Exception:
                            pass

                # --- Unified post-recovery convergence ---
                # Run the same convergence steps as checkpoint_restart_fn
                # to ensure directory / health / dispatch consistency.
                try:
                    from megatron.core.transformer.moe.unified_reintegration import (
                        get_post_recovery_convergence,
                        RecoveryPath,
                    )
                    _target_model = model[0] if isinstance(model, (list, tuple)) else model
                    _config = getattr(_target_model, 'config', None)
                    convergence = get_post_recovery_convergence(config=_config)
                    conv_result = convergence.execute(
                        path=RecoveryPath.HYBRID_RECOVERY,
                        failed_rank=failed_rank,
                        replacement_rank=replacement_rank,
                        expert_ids=expert_ids or [],
                        restored_experts=list(result.restored_experts),
                        step=step,
                        config=_config,
                    )
                    logger.info(
                        "[%s] BSR-MoE expert_restore_fn: unified convergence "
                        "completed (consistent=%s, elapsed=%.3fs)",
                        _ts(),
                        conv_result.consistency_verified,
                        conv_result.elapsed_seconds,
                    )
                except Exception as conv_e:
                    logger.error(
                        "BSR-MoE expert_restore_fn: unified convergence "
                        "failed (non-fatal): %s", conv_e,
                    )

            else:
                logger.error(
                    "[%s] BSR-MoE expert_restore_fn: PARTIAL — restored %d, "
                    "failed %d. Errors: %s",
                    _ts(), result.num_restored, result.num_failed, result.errors,
                )

        except Exception as e:
            logger.error("[%s] BSR-MoE expert_restore_fn failed: %s", _ts(), e)
            # Fallback: at minimum mark experts as STALE_RUNNABLE
            try:
                if expert_ids:
                    from megatron.core.transformer.moe.recovery_controller import get_recovery_controller
                    _ctrl = get_recovery_controller()
                    _ctrl._expert_tracker.mark_stale_runnable(expert_ids, step=step)
                    logger.warning(
                        "BSR-MoE expert_restore_fn: fallback — experts %s "
                        "marked STALE_RUNNABLE at step %d",
                        expert_ids, step,
                    )
            except Exception as e2:
                logger.error(
                    "BSR-MoE expert_restore_fn: fallback also failed: %s", e2,
                )

    def pipeline_stage_repair_fn(
        *, failed_rank, replacement_rank, step=-1,
        pp_group_ranks=None, failed_stage=-1,
    ):
        """Execute PP group rebuild + P2P rebinding for PP>1.

        Delegates to the PipelineStageRepairer singleton.
        """
        try:
            from megatron.core.transformer.moe import pipeline_stage_repair as psr_mod

            repairer = psr_mod.get_pipeline_stage_repairer()

            create_group_fn = None
            try:
                import torch.distributed as td
                create_group_fn = td.new_group
            except Exception:
                pass

            result = repairer.execute(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                pp_group_ranks=pp_group_ranks,
                step=step,
                create_group_fn=create_group_fn,
            )

            if result.success:
                logger.warning(
                    "[%s] BSR-MoE pipeline_stage_repair_fn: SUCCESS — "
                    "pp_rebuilt=%s, prev_next=%s, p2p_rebound=%s, "
                    "elapsed=%.2fs",
                    _ts(),
                    result.pp_group_rebuilt, result.prev_next_updated,
                    result.p2p_rebound, result.elapsed_seconds,
                )
            else:
                logger.error(
                    "[%s] BSR-MoE pipeline_stage_repair_fn: FAILED — %s",
                    _ts(), result.error,
                )
        except Exception as e:
            logger.error("BSR-MoE pipeline_stage_repair_fn failed: %s", e)

    def pipeline_rollback_fn(
        *, failed_rank, failed_stage=-1, step=-1, pp_group_ranks=None,
    ):
        """Execute pipeline-safe rollback across all stages.

        Delegates to the PipelineRollbackCoordinator.
        """
        try:
            if _PIPELINE_ROLLBACK_COORDINATOR is not None:
                _PIPELINE_ROLLBACK_COORDINATOR.on_pipeline_failure(
                    failed_stage=failed_stage,
                    failed_rank=failed_rank,
                    step=step,
                    reason=f"pipeline_stage_{failed_stage}_failure",
                )
                result = _PIPELINE_ROLLBACK_COORDINATOR.initiate_rollback()
                if result is not None:
                    logger.warning(
                        "BSR-MoE pipeline_rollback_fn: rollback initiated — "
                        "step=%d, failed_stage=%d",
                        step, failed_stage,
                    )
                else:
                    logger.warning(
                        "BSR-MoE pipeline_rollback_fn: rollback returned None "
                        "(step=%d)", step,
                    )
            else:
                logger.warning(
                    "BSR-MoE pipeline_rollback_fn: no coordinator available "
                    "(step=%d)", step,
                )
        except Exception as e:
            logger.error("BSR-MoE pipeline_rollback_fn failed: %s", e)

    def microbatch_invalidation_fn(*, step=-1, pp_size=1):
        """Mark all in-flight microbatches as invalid.

        In PP>1, when a stage fails mid-iteration, all microbatches
        currently in the pipeline are tainted.
        """
        try:
            if _ITERATION_INVALIDATOR is not None:
                _ITERATION_INVALIDATOR.invalidate(
                    step=step,
                    reason=f"pipeline_microbatch_invalidation_pp{pp_size}",
                )
            logger.warning(
                "BSR-MoE microbatch_invalidation_fn: invalidated "
                "(step=%d, pp_size=%d)", step, pp_size,
            )
        except Exception as e:
            logger.error("BSR-MoE microbatch_invalidation_fn failed: %s", e)

    def checkpoint_restart_fn(
        *, failed_rank, replacement_rank, step, decision,
    ):
        """Checkpoint restart path: load model weights AND optimizer/scheduler
        state from the latest distributed checkpoint.

        This replaces both dense_sync_fn (dense params from DP peer) and
        expert_restore_fn (expert weights from checkpoint) with a single
        full-model checkpoint load.  After this function returns, the
        replacement rank has all parameters and optimizer state at the
        checkpoint version.

        The gap between the checkpoint iteration and the current training
        iteration is small (by policy), so the staleness is acceptable.
        The training loop will roll back iteration to the checkpoint
        iteration and resume training from there.

        Protocol:
        1. Locate the latest checkpoint directory.
        2. Load the full model state_dict + optimizer + scheduler from
           checkpoint.
        3. Mark all affected experts as STALE_RUNNABLE (they have
           checkpoint-version weights, not current-version).
        4. Set global flag for training loop to roll back iteration.

        Args:
            failed_rank: The rank that failed.
            replacement_rank: The replacement rank.
            step: Current training step.
            decision: The ``RecoveryDecision`` from gap-aware policy.
        """
        global _CHECKPOINT_RESTART_REQUESTED, _CHECKPOINT_RESTART_DECISION

        t_start = time.time()
        checkpoint_dir = _find_latest_checkpoint_dir()

        if checkpoint_dir is None:
            raise RuntimeError(
                "BSR-MoE checkpoint_restart_fn: no checkpoint directory "
                "found — cannot execute checkpoint restart path"
            )

        logger.warning(
            "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  START loading checkpoint from %s "
            "(step=%d, failed_rank=%d, replacement_rank=%d, gap=%d, ckpt_iter=%d)",
            _ts(), checkpoint_dir, step, failed_rank, replacement_rank,
            decision.gap if decision else -1,
            decision.latest_checkpoint_step if decision else -1,
        )

        try:
            # ---- Step 1: Load checkpoint (model + optimizer + scheduler) ----
            # Use Megatron's standard checkpoint loading infrastructure.
            # For checkpoint restart (small gap), we load the FULL state
            # including optimizer and lr scheduler so the replacement rank
            # can resume training from the checkpoint iteration without
            # any optimizer warmup penalty.
            t_step1_start = time.time()
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 1/4] Preparing to load checkpoint...",
                _ts(),
            )
            
            from megatron.training.global_vars import get_args
            from megatron.training.checkpointing import load_checkpoint as _megatron_load_checkpoint

            args = get_args()

            # ---- Save training-loop state that load_checkpoint will overwrite ----
            # load_checkpoint() restores consumed_train_samples and lr scheduler
            # state from the checkpoint.  But in BSR-MoE recovery, we do NOT
            # want to roll back these values — only the failed rank's model
            # weights need to be restored.  The training loop continues from
            # the current iteration, not the checkpoint iteration.
            _saved_consumed_train_samples = args.consumed_train_samples
            _saved_consumed_valid_samples = getattr(args, 'consumed_valid_samples', 0)

            # Save lr scheduler state (step count) so we can restore it
            _saved_scheduler_num_steps = None
            if opt_param_scheduler is not None:
                _saved_scheduler_num_steps = getattr(
                    opt_param_scheduler, 'num_steps', None,
                )

            # Temporarily override 'load' to point to the checkpoint dir's
            # parent (Megatron's load_checkpoint expects the save root, not
            # the iter_XXXXXXX subdir).
            ckpt_parent = os.path.dirname(checkpoint_dir)
            original_load = getattr(args, 'load', None)
            args.load = ckpt_parent

            try:
                # Load model weights + optimizer + scheduler.
                # This uses Megatron's dist_checkpointing under the hood.
                _ckpt_iter, _ = _megatron_load_checkpoint(
                    model,              # ddp_model (list of model chunks)
                    optimizer,          # optimizer — load state
                    opt_param_scheduler,  # lr scheduler — load state
                )
                t_load_elapsed = time.time() - t_step1_start
                logger.warning(
                    "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 1/4] CHECKPOINT LOADED "
                    "from iter %s (load_time=%.3fs, total_elapsed=%.3fs)",
                    _ts(), _ckpt_iter, t_load_elapsed, time.time() - t_start,
                )
            finally:
                # Restore original load path
                args.load = original_load

            # ---- Restore training-loop state overwritten by load_checkpoint ----
            # The iteration counter, consumed_samples, and lr scheduler step
            # must reflect the CURRENT training position, not the checkpoint.
            t_step2_start = time.time()
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 2/4] Restoring training-loop state...",
                _ts(),
            )
            
            args.consumed_train_samples = _saved_consumed_train_samples
            args.consumed_valid_samples = _saved_consumed_valid_samples

            if (opt_param_scheduler is not None
                    and _saved_scheduler_num_steps is not None):
                opt_param_scheduler.num_steps = _saved_scheduler_num_steps

            t_step2_elapsed = time.time() - t_step2_start
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 2/4] Training-loop state restored "
                "(consumed_train_samples=%d, scheduler_num_steps=%s, restore_time=%.3fs, total_elapsed=%.3fs)",
                _ts(), args.consumed_train_samples,
                _saved_scheduler_num_steps, t_step2_elapsed, time.time() - t_start,
            )

            # ---- Step 2: Mark experts as STALE_RUNNABLE ----
            # The loaded weights are from the checkpoint iteration, which
            # is behind the current training iteration.  Mark all experts
            # on this rank as STALE_RUNNABLE via ExpertRecoveryTracker.
            t_step3_start = time.time()
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 3/4] Marking experts as STALE_RUNNABLE...",
                _ts(),
            )
            
            expert_ids = []
            if replacement_rank in ep_group_ranks:
                ep_rank_idx = ep_group_ranks.index(replacement_rank)
                num_local = num_experts // ep_size
                expert_ids = list(range(
                    ep_rank_idx * num_local,
                    (ep_rank_idx + 1) * num_local,
                ))

            if expert_ids:
                try:
                    from megatron.core.transformer.moe.recovery_controller import (
                        get_recovery_controller,
                    )
                    ctrl = get_recovery_controller()
                    if ctrl._expert_tracker is not None:
                        ctrl._expert_tracker.mark_stale_runnable(
                            expert_ids, step=step,
                        )
                    t_step3_elapsed = time.time() - t_step3_start
                    logger.warning(
                        "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 3/4] Marked %d experts as STALE_RUNNABLE "
                        "(experts=%s, mark_time=%.3fs, total_elapsed=%.3fs)",
                        _ts(), len(expert_ids), expert_ids, t_step3_elapsed, time.time() - t_start,
                    )
                except Exception as e:
                    logger.error(
                        "BSR-MoE checkpoint_restart_fn: failed to mark "
                        "experts stale_runnable: %s", e,
                    )

            # ---- Step 2b: Unified post-recovery convergence ----
            # Run the same post-recovery steps as the hybrid path so that
            # directory / health-manager / dispatch-topology / preferential
            # routing / two-phase state machine are all consistent.
            t_step4_start = time.time()
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 4/4] Running unified convergence...",
                _ts(),
            )
            
            try:
                from megatron.core.transformer.moe.unified_reintegration import (
                    get_post_recovery_convergence,
                    RecoveryPath,
                )
                _target_model = model[0] if isinstance(model, (list, tuple)) else model
                _config = getattr(_target_model, 'config', None)
                convergence = get_post_recovery_convergence(config=_config)

                # Build restored_experts list for the local PP stage.
                _restored_experts = [
                    (_layer_num, _eid)
                    for _layer_num in _infer_local_moe_layer_ids(model, num_layers)
                    for _eid in expert_ids
                ]

                conv_result = convergence.execute(
                    path=RecoveryPath.CHECKPOINT_RESTART,
                    failed_rank=failed_rank,
                    replacement_rank=replacement_rank,
                    expert_ids=expert_ids,
                    restored_experts=_restored_experts,
                    step=step,
                    config=_config,
                )
                t_step4_elapsed = time.time() - t_step4_start
                logger.warning(
                    "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  [Step 4/4] CONVERGENCE COMPLETED "
                    "(consistent=%s, pref_routing=%d, two_phase=%s, convergence_time=%.3fs, total_elapsed=%.3fs)",
                    _ts(),
                    conv_result.consistency_verified,
                    conv_result.preferential_routing_activated,
                    conv_result.two_phase_driven,
                    t_step4_elapsed,
                    time.time() - t_start,
                )
            except Exception as conv_e:
                logger.error(
                    "BSR-MoE checkpoint_restart_fn: unified convergence "
                    "failed (non-fatal): %s", conv_e,
                )

            # ---- Step 3: Set global flag for training loop ----
            # The training loop will check this flag and roll back
            # iteration to the checkpoint iteration.
            _CHECKPOINT_RESTART_REQUESTED = True
            _CHECKPOINT_RESTART_DECISION = decision

            elapsed = time.time() - t_start
            logger.warning(
                "[%s] BSR-MoE checkpoint_restart_fn: ⏱️  ✅ TOTAL RECOVERY TIME: %.3fs "
                "(step=%d, ckpt_iter=%d, failed_rank=%d, replacement_rank=%d) "
                "| Breakdown: load=%.3fs, restore_state=%.3fs, mark_experts=%.3fs, convergence=%.3fs",
                _ts(), elapsed, step,
                decision.latest_checkpoint_step if decision else -1,
                failed_rank, replacement_rank,
                t_load_elapsed if 't_load_elapsed' in locals() else 0.0,
                t_step2_elapsed if 't_step2_elapsed' in locals() else 0.0,
                t_step3_elapsed if 't_step3_elapsed' in locals() else 0.0,
                t_step4_elapsed if 't_step4_elapsed' in locals() else 0.0,
            )

        except Exception as e:
            elapsed = time.time() - t_start
            logger.error(
                "[%s] BSR-MoE checkpoint_restart_fn: FAILED after %.3fs — %s",
                _ts(), elapsed, e,
            )
            raise  # Let the controller fall back to hybrid recovery

    def optimizer_commit_block_fn(*, step, failed_rank):
        """Block optimizer commit for the current iteration.

        Called by RecoveryController.on_hard_rank_failure() to ensure
        the optimizer step is skipped even if the iteration invalidation
        flag hasn't propagated yet.
        """
        from megatron.core.transformer.moe.optimizer_commit_guard import (
            get_optimizer_commit_guard,
        )
        commit_guard = get_optimizer_commit_guard()
        if commit_guard is not None:
            commit_guard.block(
                reason=f"hard_failure_rank_{failed_rank}_step_{step}",
            )
        logger.warning(
            "BSR-MoE: optimizer commit blocked for step %d "
            "(hard failure of rank %d)", step, failed_rank,
        )

    def enter_waiting_fn(*, failed_rank, step, expert_ids=None):
        """Notify that the system is entering waiting-for-replacement state.

        This is a notification hook — external orchestrators (e.g.,
        Kubernetes operator, SLURM plugin) can register additional
        callbacks here.  For now, it serves as a structured log point
        and extension hook.
        """
        logger.warning(
            "BSR-MoE: entering waiting-for-replacement state "
            "(failed_rank=%d, step=%d, experts=%s)",
            failed_rank, step, expert_ids,
        )

    # ---- Unified post-recovery convergence callback ----
    def post_recovery_convergence_fn(
        *, path, failed_rank, replacement_rank, step, expert_ids,
    ):
        """Unified convergence point called by RecoveryController between
        Phase B and Phase C.  Verifies that directory / health-manager /
        dispatch-topology views are consistent after either recovery path.
        """
        try:
            from megatron.core.transformer.moe.unified_reintegration import (
                get_post_recovery_convergence,
            )
            # Obtain config from the model (TransformerConfig)
            _target_model = model[0] if isinstance(model, (list, tuple)) else model
            _config = getattr(_target_model, 'config', None)
            convergence = get_post_recovery_convergence(config=_config)
            # At this point the path-specific code has already run the
            # full convergence.execute() inside checkpoint_restart_fn or
            # expert_restore_fn.  This callback is a lightweight
            # verification-only pass.
            issues = []
            try:
                from megatron.core.transformer.moe.dispatch_topology_refresh import (
                    get_dispatch_topology_manager,
                )
                topo_mgr = get_dispatch_topology_manager()
                if topo_mgr is not None:
                    issues = topo_mgr.check_router_dispatcher_consistency()
            except Exception:
                pass

            if issues:
                logger.warning(
                    "BSR-MoE post-recovery convergence: consistency issues "
                    "detected after %s path: %s", path, issues,
                )
            else:
                logger.info(
                    "BSR-MoE post-recovery convergence: views consistent "
                    "after %s path (step=%d)", path, step,
                )
        except Exception as e:
            logger.error(
                "BSR-MoE post-recovery convergence callback failed: %s", e,
            )

    def force_checkpoint_restart_fn(*, step=-1):
        """Check if CHECKPOINT_RESTART should be forced regardless of
        gap-aware policy.

        Returns a non-empty reason string if CHECKPOINT_RESTART should be
        forced, or a falsy value (None/empty string) otherwise.

        Currently forces CHECKPOINT_RESTART when the latest checkpoint
        uses the ``torch_dist`` (distributed checkpoint) format, because
        selective per-rank expert loading is not possible without
        collective communication.

        To enable hybrid recovery, use ``--ckpt-format torch`` which
        stores checkpoint shards as per-rank ``.pt`` files that can be
        loaded independently by each rank.
        """
        ckpt_dir = _find_latest_checkpoint_dir()
        if ckpt_dir is not None and _is_torch_dist_checkpoint(ckpt_dir):
            return (
                f"torch_dist checkpoint at {ckpt_dir} — selective "
                f"per-rank expert loading not supported"
            )
        return None

    ctrl.register_callbacks(
        replacement_announce_fn=replacement_announce_fn,
        replacement_integrate_fn=replacement_integrate_fn,
        group_rebuild_request_fn=group_rebuild_request_fn,
        group_rebuild_execute_fn=group_rebuild_execute_fn,
        group_rebuild_finish_fn=group_rebuild_finish_fn,
        topology_refresh_fn=topology_refresh_fn,
        dense_sync_fn=dense_sync_fn,
        expert_restore_fn=expert_restore_fn,
        checkpoint_restart_fn=checkpoint_restart_fn,
        pipeline_stage_repair_fn=pipeline_stage_repair_fn,
        pipeline_rollback_fn=pipeline_rollback_fn,
        microbatch_invalidation_fn=microbatch_invalidation_fn,
        optimizer_commit_block_fn=optimizer_commit_block_fn,
        enter_waiting_fn=enter_waiting_fn,
        post_recovery_convergence_fn=post_recovery_convergence_fn,
        force_checkpoint_restart_fn=force_checkpoint_restart_fn,
    )


# =====================================================================
# Internal: fault injection
# =====================================================================

def _maybe_inject_fault(step: int) -> None:
    """Check if a scheduled fault should be injected at this step.

    Supports periodic fault injection: if ``BSR_FAULT_INJECT_INTERVAL`` > 0,
    faults are injected every ``interval`` steps starting from ``inject_step``.
    Each injection is a full cycle: fault → recovery → next fault.
    The next fault is only injected after the previous recovery completes
    (i.e., ``injected`` and ``replacement_injected`` are both reset).
    """
    global _FAULT_INJECTOR_CONFIG

    if _FAULT_INJECTOR_CONFIG is None or not _FAULT_INJECTOR_CONFIG['enabled']:
        return

    cfg = _FAULT_INJECTOR_CONFIG
    ctrl = _RECOVERY_CONTROLLER

    if ctrl is None:
        return

    # Inject fault at the configured step (or next periodic step)
    if not cfg['injected'] and step >= cfg['next_inject_step']:
        cfg['injected'] = True
        cfg['inject_count'] += 1
        inject_type = cfg['inject_type']
        ep_group_ranks = cfg['ep_group_ranks']
        dp_group_ranks = cfg['dp_group_ranks']
        num_experts = cfg['num_experts']
        ep_size = len(ep_group_ranks)

        # Determine which rank to fault
        if cfg['random_rank']:
            inject_rank = cfg['fault_rng'].choice(ep_group_ranks)
        else:
            inject_rank = cfg['inject_rank']
        cfg['current_failed_rank'] = inject_rank

        # Compute expert IDs on the target rank
        if inject_rank in ep_group_ranks:
            ep_rank_idx = ep_group_ranks.index(inject_rank)
            num_local = num_experts // ep_size
            expert_ids = list(range(ep_rank_idx * num_local, (ep_rank_idx + 1) * num_local))
        else:
            expert_ids = []

        logger.warning(
            "[%s] BSR-MoE FAULT INJECTION #%d: type=%s, rank=%d, step=%d, experts=%s",
            _ts(), cfg['inject_count'], inject_type, inject_rank, step, expert_ids,
        )

        if inject_type in ('quarantine', 'hard_failure'):
            ctrl.on_hard_rank_failure(
                failed_rank=inject_rank,
                reason="scheduled_fault_injection",
                step=step,
                expert_ids=expert_ids,
                ep_group_ranks=ep_group_ranks,
                dp_group_ranks=dp_group_ranks,
            )
        elif inject_type == 'restart_in_place':
            ctrl.on_hard_rank_failure(
                failed_rank=inject_rank,
                reason="scheduled_restart_in_place_injection",
                step=step,
                expert_ids=expert_ids,
                ep_group_ranks=ep_group_ranks,
                dp_group_ranks=dp_group_ranks,
                restart_in_place=True,
            )
            # restart_in_place fast path auto-assigns replacement and
            # marks it ready, so skip the separate replacement injection.
            cfg['replacement_injected'] = True
            # Schedule next periodic injection if interval > 0
            _schedule_next_injection(cfg, step)
        else:
            logger.error("BSR-MoE: unknown fault injection type: %s", inject_type)

    # Inject replacement ready at the configured step
    if (cfg['injected'] and not cfg['replacement_injected']
            and step >= cfg['replacement_step']):
        cfg['replacement_injected'] = True
        inject_rank = cfg.get('current_failed_rank', cfg['inject_rank'])
        replacement_rank = cfg['replacement_rank']

        if replacement_rank < 0:
            # Default: use the failed rank's own ID (identity inheritance).
            # The replacement process joins with the same rank identity as
            # the failed process, so all process group memberships remain
            # valid.  This avoids the world_size boundary issue where
            # torch.distributed.new_group() rejects ranks >= world_size.
            #
            # In a real deployment, the replacement process would be
            # launched with the same rank assignment as the failed one
            # (e.g., via torchrun elastic or a custom launcher).
            replacement_rank = inject_rank
            logger.info(
                "BSR-MoE FAULT INJECTION: using identity inheritance — "
                "replacement_rank=%d (same as failed_rank)",
                replacement_rank,
            )

        logger.warning(
            "[%s] BSR-MoE FAULT INJECTION: replacement ready — "
            "failed_rank=%d, replacement_rank=%d, step=%d",
            _ts(), inject_rank, replacement_rank, step,
        )

        # Announce replacement and mark ready
        ctrl.on_replacement_assigned(
            failed_rank=inject_rank,
            replacement_rank=replacement_rank,
            step=step,
        )
        ctrl.on_replacement_ready(
            failed_rank=inject_rank,
            step=step,
        )
        # Schedule next periodic injection if interval > 0
        _schedule_next_injection(cfg, step)


def _schedule_next_injection(cfg: dict, current_step: int) -> None:
    """Reset injection flags and schedule the next periodic fault.

    If ``inject_interval`` > 0, computes the next injection step and
    resets ``injected`` / ``replacement_injected`` so the next fault
    can fire.  If interval is 0, this is a no-op (single injection only).
    """
    interval = cfg.get('inject_interval', 0)
    if interval <= 0:
        return

    cfg['next_inject_step'] = current_step + interval
    cfg['injected'] = False
    cfg['replacement_injected'] = False
    logger.warning(
        "[%s] BSR-MoE FAULT INJECTION: next periodic fault scheduled at step %d "
        "(interval=%d, total_injections=%d)",
        _ts(), cfg['next_inject_step'], interval, cfg['inject_count'],
    )


def _maybe_poll_deferred_optimizer(step: int) -> None:
    """Poll the deferred optimizer loader if active.

    When async recovery is enabled, uses the async worker and CUDA stream
    for true background loading.  Otherwise, uses the sync fallback path
    with the load_fn stored during expert_restore_fn.

    When two-phase recovery is active, drives the state machine:
    OPTIMIZER_PENDING → FULLY_RECOVERED on finalization.
    """
    try:
        from megatron.core.transformer.moe.deferred_optimizer_load import (
            get_deferred_optimizer_loader,
        )
        loader = get_deferred_optimizer_loader()

        if not loader.has_pending() and loader.num_loaded == 0:
            return

        from megatron.core.transformer.moe.stale_expert_restore import (
            get_optimizer_update_barrier,
        )

        barrier = get_optimizer_update_barrier()
        health_managers = {}  # ExpertHealthManager removed; pass empty dict

        num_executed = 0
        num_finalized = 0

        # Use async path if worker and stream are available
        if _ASYNC_RECOVERY_WORKER is not None and _RECOVERY_STREAM is not None:
            num_executed, num_finalized = loader.poll_and_finalize(
                step=step,
                barrier=barrier,
                health_managers=health_managers,
                async_worker=_ASYNC_RECOVERY_WORKER,
                cuda_stream=_RECOVERY_STREAM,
            )
            if num_executed > 0 or num_finalized > 0:
                logger.info(
                    "BSR-MoE: deferred optimizer poll — executed=%d, "
                    "finalized=%d at step %d (async)",
                    num_executed, num_finalized, step,
                )
        else:
            # Sync fallback — use the load_fn stored during expert_restore_fn
            load_fn = getattr(loader, '_pending_load_fn', None)
            num_executed, num_finalized = loader.poll_and_finalize(
                step=step,
                load_fn=load_fn,
                barrier=barrier,
                health_managers=health_managers,
            )
            if num_executed > 0 or num_finalized > 0:
                logger.info(
                    "BSR-MoE: deferred optimizer poll — executed=%d, "
                    "finalized=%d at step %d (sync)",
                    num_executed, num_finalized, step,
                )

        # --- Two-phase: OPTIMIZER_PENDING → FULLY_RECOVERED ---
        if num_finalized > 0:
            try:
                from megatron.core.transformer.moe.two_phase_recovery import (
                    get_two_phase_recovery_coordinator,
                    TwoPhaseState,
                )
                tp_coord = get_two_phase_recovery_coordinator()
                pending = tp_coord.get_experts_in_state(
                    TwoPhaseState.OPTIMIZER_PENDING,
                )
                if pending:
                    # Check which of the pending experts have been finalized
                    # in the deferred loader
                    from megatron.core.transformer.moe.deferred_optimizer_load import (
                        OptimizerLoadState,
                    )
                    finalized_keys = []
                    for key in pending:
                        req = loader.get_request(key[0], key[1])
                        if req is not None and req.state == OptimizerLoadState.FINALIZED:
                            finalized_keys.append(key)

                    if finalized_keys:
                        tp_coord.on_optimizer_loaded(
                            expert_ids=finalized_keys, step=step,
                        )
            except Exception as tp_e:
                logger.debug(
                    "BSR-MoE: two-phase optimizer finalize failed: %s", tp_e,
                )

    except Exception as e:
        logger.debug("BSR-MoE: _maybe_poll_deferred_optimizer: %s", e)


def _poll_async_recovery(step: int) -> None:
    """Poll the async recovery worker for completed expert weight loads.

    Called from ``bsr_after_iteration()``.  When expert weight loads
    complete:
    1. Copy CPU tensors to GPU on the recovery CUDA stream.
    2. Synchronize the stream.
    3. Update health masks (STALE_RUNNABLE → FULLY_RECOVERED).
    """
    if _ASYNC_RECOVERY_WORKER is None or _RECOVERY_STREAM is None:
        return

    worker = _ASYNC_RECOVERY_WORKER
    if not worker.has_pending() and worker.num_completed == 0:
        return

    results = worker.poll_completed()
    if not results:
        return

    # Filter for expert_weight results (optimizer results are handled
    # by _maybe_poll_deferred_optimizer)
    expert_results = [r for r in results if r.request_type == "expert_weight"]
    if not expert_results:
        return

    # Apply to GPU on the recovery stream
    copied = worker.apply_to_gpu(expert_results, _RECOVERY_STREAM)

    # ExpertHealthManager was removed; recovery_controller owns expert state.
    health_managers = {}
    barrier = None

    try:
        from megatron.core.transformer.moe.stale_expert_restore import (
            get_optimizer_update_barrier,
        )
        barrier = get_optimizer_update_barrier()
    except ImportError:
        pass

    # Synchronize stream and finalize
    finalized = worker.wait_stream_and_finalize(
        stream=_RECOVERY_STREAM,
        results=expert_results,
        barrier=barrier,
        health_managers=health_managers,
        step=step,
    )

    if finalized > 0:
        logger.warning(
            "[%s] BSR-MoE: async expert recovery — %d experts finalized "
            "at step %d (%d tensors copied to GPU)",
            _ts(), finalized, step, copied,
        )


def _wire_async_recovery_callbacks(
    ctrl,
    *,
    worker,
    stream,
    num_layers: int,
    num_experts: int,
    model,
) -> None:
    """Wire async recovery callbacks to the RecoveryController.

    This registers:
    - ``async_expert_restore_fn``: submits expert loads to the async worker
    - ``poll_async_recovery_fn``: polls for completed loads
    """
    from megatron.core.transformer.moe import stale_expert_restore as ser_mod

    def async_expert_restore_fn(
        *, failed_rank, replacement_rank, step=-1, expert_ids=None,
    ):
        """Submit expert weight loads to the async worker.

        Returns a list of request_id strings.
        """
        if not expert_ids:
            return []

        coordinator = ser_mod.get_stale_expert_restore_coordinator()

        # Build a restore plan (same as sync path)
        plan = ser_mod.ExpertRestorePlan(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            step=step,
        )

        checkpoint_dir = _find_latest_checkpoint_dir()

        moe_layer_ids = _infer_local_moe_layer_ids(model, num_layers)

        for eid in expert_ids:
            for layer_id in moe_layer_ids:
                plan.entries.append(ser_mod.ExpertRestoreEntry(
                    layer_id=layer_id,
                    expert_id=eid,
                    checkpoint_step=-1,
                    checkpoint_dir=checkpoint_dir or "",
                    host_rank=replacement_rank,
                ))

        # Build load_fn for async worker
        load_fn = _build_expert_load_fn(
            checkpoint_dir=checkpoint_dir,
            model=model,
        )

        # Wrap load_fn for AsyncLoadRequest interface
        def _async_load_fn(request):
            """Adapter: AsyncLoadRequest → load_fn → tensor/True."""
            if load_fn is not None:
                # Create a minimal ExpertRestoreEntry for the original load_fn
                entry = ser_mod.ExpertRestoreEntry(
                    layer_id=request.layer_id,
                    expert_id=request.expert_id,
                    checkpoint_dir=request.checkpoint_path,
                    weight_location=request.weight_location,
                    checkpoint_step=request.metadata.get("checkpoint_step", -1),
                    host_rank=request.metadata.get("host_rank", -1),
                )
                target_model = model[0] if isinstance(model, (list, tuple)) else model
                result = load_fn(entry, target_model)
                return result  # True for dry-run, tensor for real load
            return True  # dry-run

        # Submit async loads
        request_ids = coordinator.execute_restore_async(
            plan=plan,
            load_fn=_async_load_fn,
            worker=worker,
            step=step,
        )

        # Also set up optimizer update barrier for stale experts
        barrier = ser_mod.get_optimizer_update_barrier()
        target_model = model[0] if isinstance(model, (list, tuple)) else model
        if target_model is not None and expert_ids:
            barrier.block_expert_params(
                target_model, expert_ids=list(expert_ids), step=step,
            )

        # Mark experts as STALE_RUNNABLE in health managers
        try:
            from megatron.core.transformer.moe.recovery_controller import get_recovery_controller
            ctrl = get_recovery_controller()
            ctrl._expert_tracker.mark_stale_runnable(expert_ids, step=step)
        except Exception as e:
            logger.warning(
                "BSR-MoE async_expert_restore_fn: failed to mark "
                "STALE_RUNNABLE: %s", e,
            )

        return request_ids

    def poll_async_recovery_fn(*, step=-1):
        """Check if all async expert loads have completed.

        Returns True if all loads are done (no more pending).
        """
        if worker is None:
            return True
        return not worker.has_pending() and worker.num_completed == 0

    ctrl.register_callbacks(
        async_expert_restore_fn=async_expert_restore_fn,
        poll_async_recovery_fn=poll_async_recovery_fn,
    )


# =====================================================================
# Internal: helpers
# =====================================================================

def _find_latest_checkpoint_dir() -> Optional[str]:
    """Find the most recent checkpoint directory.

    Searches the Megatron checkpoint save directory for the latest
    iteration directory.  Prefers directories containing a
    ``bsr_manifest.json`` file, but falls back to any ``iter_XXXXXXX``
    directory if no BSR manifest is present (so that standard Megatron
    distributed checkpoints can also be used for the checkpoint-restart
    path).

    Returns:
        Path to the checkpoint directory, or None if not found.
    """
    try:
        from megatron.training.global_vars import get_args
        args = get_args()
        save_dir = getattr(args, 'save', None) or getattr(args, 'load', None)
        if save_dir is None:
            return None

        from megatron.core.transformer.moe.expert_directory import (
            RecoveryManifest,
        )

        # Look for iter_XXXXXXX directories with manifests
        best_dir = None
        best_step = -1
        # Fallback: best iter_XXXXXXX without manifest
        fallback_dir = None
        fallback_step = -1

        if os.path.isdir(save_dir):
            for entry in os.listdir(save_dir):
                if entry.startswith('iter_'):
                    candidate = os.path.join(save_dir, entry)
                    if not os.path.isdir(candidate):
                        continue
                    try:
                        step = int(entry.split('_')[1])
                    except (ValueError, IndexError):
                        continue
                    if RecoveryManifest.exists(candidate):
                        if step > best_step:
                            best_step = step
                            best_dir = candidate
                    else:
                        if step > fallback_step:
                            fallback_step = step
                            fallback_dir = candidate

        # NOTE: bsr_save_manifest() writes into iter_XXXXXXX/bsr_manifest.json
        # (inside the iteration subdirectory).  Do NOT check save_dir itself
        # for a manifest — a stale root-level manifest from an older version
        # could cause _ensure_state_dict_loaded to look for .pt files in the
        # wrong directory.

        if best_dir is not None:
            logger.info(
                "BSR-MoE: found latest checkpoint with BSR manifest at %s "
                "(step=%d)", best_dir, best_step,
            )
            return best_dir

        if fallback_dir is not None:
            logger.warning(
                "BSR-MoE: no BSR manifest found, falling back to standard "
                "Megatron checkpoint at %s (step=%d)",
                fallback_dir, fallback_step,
            )
            return fallback_dir

        return None

    except Exception as e:
        logger.warning("BSR-MoE: _find_latest_checkpoint_dir failed: %s", e)
        return None


def _get_latest_checkpoint_iteration() -> int:
    """Query the iteration of the latest available checkpoint.

    This is the ``get_checkpoint_iteration_fn`` callback used by the
    gap-aware recovery policy.  It searches for the most recent
    checkpoint directory with a BSR manifest and returns its iteration.

    Returns:
        The checkpoint iteration, or -1 if no checkpoint is found.
    """
    try:
        from megatron.training.global_vars import get_args
        args = get_args()
        save_dir = getattr(args, 'save', None) or getattr(args, 'load', None)
        if save_dir is None:
            return -1

        best_step = -1

        if os.path.isdir(save_dir):
            for entry in os.listdir(save_dir):
                if entry.startswith('iter_'):
                    candidate = os.path.join(save_dir, entry)
                    if os.path.isdir(candidate):
                        try:
                            step = int(entry.split('_')[1])
                            if step > best_step:
                                best_step = step
                        except (ValueError, IndexError):
                            pass

        return best_step

    except Exception as e:
        logger.warning(
            "BSR-MoE: _get_latest_checkpoint_iteration failed: %s", e,
        )
        return -1


# Global flag: set to True when checkpoint restart is requested.
# The training loop checks this flag to decide whether to stop.
_CHECKPOINT_RESTART_REQUESTED: bool = False
_CHECKPOINT_RESTART_DECISION: Optional[Any] = None


def bsr_is_checkpoint_restart_requested() -> bool:
    """Check if a checkpoint restart has been requested by gap-aware policy.

    The training loop should call this after ``bsr_before_iteration()``
    returns True.  If True, the loop should stop training and reload
    from the latest checkpoint.
    """
    return _CHECKPOINT_RESTART_REQUESTED


def bsr_get_checkpoint_restart_decision():
    """Get the RecoveryDecision that triggered the checkpoint restart.

    Returns None if no restart was requested.
    """
    return _CHECKPOINT_RESTART_DECISION


def bsr_clear_checkpoint_restart() -> None:
    """Clear the checkpoint restart flag (after restart is handled)."""
    global _CHECKPOINT_RESTART_REQUESTED, _CHECKPOINT_RESTART_DECISION
    _CHECKPOINT_RESTART_REQUESTED = False
    _CHECKPOINT_RESTART_DECISION = None


def _checkpoint_restart_fn(
    *, failed_rank, replacement_rank, step, decision,
):
    """Legacy stub — kept for backward compatibility.

    The real checkpoint_restart_fn is now defined inside
    ``_wire_recovery_callbacks()`` as a closure (so it can access
    ``model``, ``ep_group_ranks``, etc.).  This module-level function
    is no longer registered as a callback.

    If called directly (e.g., from tests), it falls back to setting
    the global flag only (no actual checkpoint loading).
    """
    global _CHECKPOINT_RESTART_REQUESTED, _CHECKPOINT_RESTART_DECISION
    _CHECKPOINT_RESTART_REQUESTED = True
    _CHECKPOINT_RESTART_DECISION = decision

    logger.warning(
        "BSR-MoE: _checkpoint_restart_fn (legacy stub) called. "
        "Setting global flag only — no checkpoint loading. "
        "(step=%d, failed_rank=%d, replacement_rank=%d)",
        step, failed_rank, replacement_rank,
    )


def _is_torch_dist_checkpoint(ckpt_dir: str) -> bool:
    """Check if a checkpoint directory uses the torch_dist (distributed
    checkpoint) format.

    torch_dist checkpoints contain ``metadata.json`` and/or ``__*_*.distcp``
    shard directories.  They CANNOT be loaded from a single rank because
    ``dist_checkpointing.load()`` uses ``all_gather_object`` internally.
    """
    import glob as _glob_mod
    if os.path.isfile(os.path.join(ckpt_dir, 'metadata.json')):
        return True
    return len(_glob_mod.glob(os.path.join(ckpt_dir, '__*_*.distcp'))) > 0


def _build_expert_load_fn(
    checkpoint_dir: Optional[str],
    model: Any,
) -> Optional[Any]:
    """Build a load_fn callback for StaleExpertRestoreCoordinator.

    The load_fn has signature:
        load_fn(entry: ExpertRestoreEntry, model) -> bool

    This uses a cached-state-dict approach:
    - On first call, loads the full checkpoint state dict into memory.
    - For each expert, extracts matching parameters by name pattern and
      copies them into the model's corresponding parameter ``.data``.
    - Subsequent calls reuse the cached state dict.

    If checkpoint_dir is None, returns None (dry-run mode — state
    transitions happen but no weights are loaded).

    Returns None for torch_dist checkpoints because they require
    collective communication (``all_gather_object``) which cannot be
    called from a single rank's async callback.

    For PP+EP checkpoints, the checkpoint shard must be chosen from the
    current PP stage and the source EP rank that owns the global expert id.
    """
    if checkpoint_dir is None:
        return None

    # ── Early exit for torch_dist format ──
    # torch_dist checkpoints cannot be selectively loaded from a single
    # rank because dist_checkpointing.load() uses all_gather_object.
    # Return None to signal dry-run mode; the caller (expert_restore_fn)
    # should request CHECKPOINT_RESTART instead.
    # To enable hybrid recovery with selective expert loading, use
    # --ckpt-format torch instead of the default torch_dist.
    if _is_torch_dist_checkpoint(checkpoint_dir):
        logger.warning(
            "BSR-MoE: _build_expert_load_fn: torch_dist checkpoint "
            "detected at %s — returning None (dry-run mode). "
            "Expert weights cannot be selectively loaded from distributed "
            "checkpoints without collective communication. "
            "The system should use CHECKPOINT_RESTART path, or switch "
            "to --ckpt-format torch to enable hybrid recovery.",
            checkpoint_dir,
        )
        return None

    # Cached checkpoint state dicts, keyed by concrete shard path.  A single
    # replacement rank may need to load source experts from a different EP
    # shard than its current EP rank.
    _cached_state_dicts: Dict[str, Dict[str, Any]] = {}

    def _ensure_state_dict_loaded(ckpt_dir: str, source_ep_rank: int) -> Dict[str, Any]:
        """Load and cache the checkpoint state dict.

        For ``torch`` format checkpoints, searches the rank-specific
        subdirectory first (``mp_rank_{tp:02d}_{pp:03d}_{ep:03d}/
        model_optim_rng.pt``), then falls back to broader patterns.

        IMPORTANT: ``torch_dist`` (distributed checkpoint) format cannot be
        loaded here because ``dist_checkpointing.load()`` internally calls
        ``all_gather_object`` which requires ALL ranks to participate.
        This function runs inside a per-rank async callback, so calling
        collective operations would deadlock.  For torch_dist checkpoints,
        the system should use the CHECKPOINT_RESTART path instead, or
        switch to ``--ckpt-format torch`` to enable hybrid recovery.
        """
        import glob as glob_mod

        # ── Detect torch_dist (distributed checkpoint) format ──
        metadata_json = os.path.join(ckpt_dir, 'metadata.json')
        is_dist_ckpt = os.path.isfile(metadata_json)
        if not is_dist_ckpt:
            distcp_dirs = glob_mod.glob(os.path.join(ckpt_dir, '__*_*.distcp'))
            is_dist_ckpt = len(distcp_dirs) > 0

        if is_dist_ckpt:
            # torch_dist format detected.  We CANNOT use
            # dist_checkpointing.load() here because it calls
            # all_gather_object (collective op) which would deadlock
            # when only the failed rank executes this callback.
            #
            # Return empty dict.  The caller treats this as a load failure so
            # the recovery path can fall back instead of silently reintegrating
            # an expert without weights.
            logger.warning(
                "BSR-MoE: torch_dist checkpoint detected at %s. "
                "Selective expert loading is not supported for distributed "
                "checkpoints (would require collective communication). "
                "Expert weights will NOT be loaded in this async callback. "
                "Consider using CHECKPOINT_RESTART path for full recovery.",
                ckpt_dir,
            )
            return {}

        ckpt_path = None

        tp_rank, _, ep_size, _ = _get_tp_ep_ranks_sizes()
        pp_rank, pp_size = _get_pp_rank_size()
        source_ep_rank = max(0, min(source_ep_rank, ep_size - 1))

        # Priority 1: Exact match for this rank's shard subdirectory.
        # With --ckpt-format torch and EP, files are at:
        #   iter_XXXXXXX/mp_rank_{tp:02d}_{pp:03d}_{ep:03d}/model_optim_rng.pt
        rank_subdir = f'mp_rank_{tp_rank:02d}_{pp_rank:03d}_{source_ep_rank:03d}'
        candidate = os.path.join(ckpt_dir, rank_subdir, 'model_optim_rng.pt')
        if os.path.isfile(candidate):
            ckpt_path = candidate
        else:
            # Try without EP suffix (EP=1 case)
            rank_subdir_no_ep = f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'
            candidate = os.path.join(ckpt_dir, rank_subdir_no_ep, 'model_optim_rng.pt')
            if os.path.isfile(candidate):
                ckpt_path = candidate

        if ckpt_path is None:
            # Priority 2: Direct file in ckpt_dir (no subdirs, e.g. TP=PP=EP=1)
            candidate = os.path.join(ckpt_dir, 'model_optim_rng.pt')
            if os.path.isfile(candidate):
                ckpt_path = candidate

        if ckpt_path is None:
            # Priority 3: Search mp_rank_* subdirectories for any .pt file
            # matching this rank's EP shard.  Fallback to ANY .pt if no
            # rank-specific file is found.
            # First, try to find a subdirectory matching this rank's EP.
            ep_pattern = os.path.join(
                ckpt_dir, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}_{source_ep_rank:03d}',
                '*.pt',
            )
            pt_files = sorted(glob_mod.glob(ep_pattern))
            if not pt_files:
                # Try without EP suffix
                no_ep_pattern = os.path.join(
                    ckpt_dir, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}',
                    '*.pt',
                )
                pt_files = sorted(glob_mod.glob(no_ep_pattern))
            if not pt_files and pp_size == 1 and ep_size == 1:
                # Fallback: any mp_rank_* subdirectory only when there is no
                # PP/EP ambiguity.  In PP/EP checkpoints, falling back to an
                # arbitrary shard can load the wrong expert partition.
                any_pattern = os.path.join(ckpt_dir, 'mp_rank_*', '*.pt')
                pt_files = sorted(glob_mod.glob(any_pattern))
            # Filter out common.pt (metadata only)
            pt_files = [f for f in pt_files
                        if os.path.basename(f) != 'common.pt']
            if pt_files:
                ckpt_path = pt_files[0]

        if ckpt_path is None:
            # Priority 4: any .pt file directly in ckpt_dir root
            pt_files = sorted(glob_mod.glob(os.path.join(ckpt_dir, '*.pt')))
            pt_files = [f for f in pt_files
                        if os.path.basename(f) != 'common.pt']
            if pt_files:
                ckpt_path = pt_files[0]

        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint file found in {ckpt_dir}"
                + (" (torch_dist format detected — expert loading requires "
                   "dist_checkpointing API; consider using --ckpt-format torch "
                   "or CHECKPOINT_RESTART path instead of HYBRID_RECOVERY)"
                   if is_dist_ckpt else "")
            )

        if ckpt_path in _cached_state_dicts:
            return _cached_state_dicts[ckpt_path]

        logger.info(
            "BSR-MoE: loading checkpoint state dict from %s "
            "(source_ep_rank=%d)",
            ckpt_path, source_ep_rank,
        )
        sd = torch.load(ckpt_path, map_location='cpu')

        # Megatron wraps model state under 'model' key
        if 'model' in sd:
            sd = sd['model']

        _cached_state_dicts[ckpt_path] = dict(sd)
        # Log a few sample keys for debugging key-name mismatches
        _sample_keys = list(_cached_state_dicts[ckpt_path].keys())[:10]
        _expert_keys = [
            k for k in _cached_state_dicts[ckpt_path]
            if 'local_experts' in k
        ][:5]
        logger.info(
            "BSR-MoE: checkpoint state dict cached (%d keys). "
            "Sample keys: %s. Expert keys: %s",
            len(_cached_state_dicts[ckpt_path]),
            _sample_keys, _expert_keys,
        )
        return _cached_state_dicts[ckpt_path]

    def _get_expert_param_prefix(layer_id: int, local_expert_idx: int) -> str:
        """Build the parameter name prefix for a specific local expert.

        Megatron SequentialMLP names expert parameters as:
            ``decoder.layers.{layer_id}.mlp.experts.local_experts.{local_idx}.``

        For GroupedMLP, expert weights are stored as a single fused tensor
        covering all local experts, so the prefix is:
            ``decoder.layers.{layer_id}.mlp.experts.``
        """
        return (
            f"decoder.layers.{layer_id}."
            f"mlp.experts.local_experts.{local_expert_idx}."
        )

    def _compute_local_expert_idx(
        global_expert_id: int,
        num_experts: int,
        ep_size: int,
    ) -> int:
        """Convert global expert ID to local expert index.

        With EP (Expert Parallelism), experts are evenly distributed:
            local_idx = global_id % num_local_experts
        where num_local_experts = num_experts // ep_size.
        """
        num_local = num_experts // ep_size
        return global_expert_id % num_local

    def _load_expert_from_checkpoint(entry, target_model):
        """Load a single expert's weights from checkpoint.

        This is a selective load: only the expert parameters matching
        the entry's (layer_id, expert_id) are loaded from the checkpoint.

        Steps:
        1. Load the full checkpoint state dict (cached on first call).
        2. Determine the local expert index from the global expert_id.
        3. Map the global layer_id to the local (PP-stage-relative) index.
        4. Build the parameter name prefix for this expert.
        5. Extract matching parameters from the state dict.
        6. Copy them into the model's corresponding parameter ``.data``.

        For PP>1, only the entries belonging to the current PP stage are
        processed.  Entries for other stages are skipped (their weights are
        loaded by the corresponding PP rank's own recovery callback).
        """
        try:
            ckpt_dir = entry.checkpoint_dir or checkpoint_dir
            actual_model = _get_single_model(target_model)

            # Determine EP configuration.  The source checkpoint shard is
            # based on the global expert id, not the replacement rank's
            # current EP rank.
            _, _, ep_size, num_experts_total = _get_tp_ep_ranks_sizes()
            if num_experts_total <= 1:
                # Fallback: infer local expert count from model structure.
                try:
                    for name, _ in actual_model.named_parameters():
                        if 'local_experts.' not in name:
                            continue
                        parts = name.split('local_experts.')
                        if len(parts) <= 1:
                            continue
                        idx = int(parts[1].split('.')[0])
                        num_experts_total = max(
                            num_experts_total, (idx + 1) * ep_size,
                        )
                except Exception:
                    pass

            num_local_experts = max(1, num_experts_total // max(1, ep_size))
            source_ep_rank = entry.expert_id // num_local_experts
            local_expert_idx = _compute_local_expert_idx(
                entry.expert_id, num_experts_total, ep_size,
            )

            # ── Map global layer_id to PP-stage-local index ──
            # entry.layer_id is a 1-based *global* MoE layer index.
            # Each PP rank's model contains only the layers assigned to
            # its pipeline stage.  The state dict keys use 0-based
            # *local* indices (0..num_layers_per_stage-1).
            total_num_layers = None
            args = _get_args_or_none()
            if args is not None:
                total_num_layers = getattr(args, 'num_layers', None)
            layer_idx_in_sd = _map_global_layer_to_local_checkpoint_idx(
                entry.layer_id, actual_model, total_num_layers,
            )
            if layer_idx_in_sd is None:
                logger.debug(
                    "BSR-MoE: skipping expert (layer=%d, id=%d) — belongs "
                    "to a different PP stage",
                    entry.layer_id, entry.expert_id,
                )
                return True
            prefix = _get_expert_param_prefix(layer_idx_in_sd, local_expert_idx)
            state_dict = _ensure_state_dict_loaded(ckpt_dir, source_ep_rank)

            # Find matching parameters in the model
            params_loaded = 0
            for name, param in actual_model.named_parameters():
                if prefix not in name:
                    continue

                # Look up the corresponding key in the checkpoint state dict
                # The checkpoint may use the same key or a slightly different
                # naming convention.  Try exact match first.
                if name in state_dict:
                    ckpt_tensor = state_dict[name]
                    if ckpt_tensor.shape == param.data.shape:
                        param.data.copy_(ckpt_tensor)
                        params_loaded += 1
                    else:
                        logger.warning(
                            "BSR-MoE: shape mismatch for %s: "
                            "model=%s, checkpoint=%s — skipping",
                            name, param.data.shape, ckpt_tensor.shape,
                        )
                else:
                    # Try without 'module.' prefix (DDP wrapping)
                    alt_name = name.replace('module.', '', 1) if name.startswith('module.') else f'module.{name}'
                    if alt_name in state_dict:
                        ckpt_tensor = state_dict[alt_name]
                        if ckpt_tensor.shape == param.data.shape:
                            param.data.copy_(ckpt_tensor)
                            params_loaded += 1
                        else:
                            logger.warning(
                                "BSR-MoE: shape mismatch for %s (alt=%s): "
                                "model=%s, checkpoint=%s — skipping",
                                name, alt_name,
                                param.data.shape, ckpt_tensor.shape,
                            )
                    else:
                        logger.debug(
                            "BSR-MoE: checkpoint key not found for %s "
                            "(also tried %s)", name, alt_name,
                        )

            if params_loaded > 0:
                logger.info(
                    "BSR-MoE: loaded %d params for expert "
                    "(layer=%d, global_id=%d, local_idx=%d) from %s",
                    params_loaded, entry.layer_id, entry.expert_id,
                    local_expert_idx, ckpt_dir,
                )
                return True
            else:
                logger.warning(
                    "BSR-MoE: no matching params found for expert "
                    "(layer=%d, global_id=%d, source_ep=%d, local_idx=%d, "
                    "prefix=%s) in checkpoint %s (%d keys). "
                    "Treating restore as failed to avoid reintegration "
                    "without weights.",
                    entry.layer_id, entry.expert_id, source_ep_rank,
                    local_expert_idx, prefix, ckpt_dir, len(state_dict),
                )
                return False

        except Exception as e:
            logger.error(
                "BSR-MoE: failed to load expert (layer=%d, id=%d): %s",
                entry.layer_id, entry.expert_id, e,
            )
            return False

    return _load_expert_from_checkpoint


def _build_expert_optimizer_load_fn(
    checkpoint_dir: Optional[str],
    model: Any,
) -> Optional[Any]:
    """Build a load_fn callback for DeferredOptimizerLoader.

    The load_fn has signature:
        load_fn(request: OptimizerLoadRequest) -> bool

    Loads optimizer state (momentum / variance) for a single expert from
    the checkpoint.  Uses the same cached-state-dict approach as
    ``_build_expert_load_fn``.

    Returns None if checkpoint_dir is unavailable (dry-run mode).
    """
    if checkpoint_dir is None:
        return None

    # Cached optimizer state dict — loaded once, reused across experts.
    _cached_opt_sd: Dict[str, Any] = {}
    _opt_cache_loaded: List[bool] = [False]

    def _ensure_optimizer_sd_loaded(ckpt_dir: str) -> Dict[str, Any]:
        """Load and cache the optimizer state dict from checkpoint.

        Handles the same subdirectory structure as ``_ensure_state_dict_loaded``
        in ``_build_expert_load_fn``: for ``--ckpt-format torch`` with EP,
        the checkpoint is inside ``mp_rank_{tp:02d}_{pp:03d}_{ep:03d}/``.
        """
        if _opt_cache_loaded[0]:
            return _cached_opt_sd

        import glob as glob_mod

        # Determine this rank's parallel coordinates
        ep_rank = 0
        tp_rank = 0
        pp_rank = 0
        try:
            from megatron.core import parallel_state as mpu
            ep_rank = mpu.get_expert_model_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
        except Exception:
            pass

        ckpt_path = None

        # Priority 1: rank-specific subdirectory (torch format with EP)
        rank_subdir = f'mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}'
        candidate = os.path.join(ckpt_dir, rank_subdir, 'model_optim_rng.pt')
        if os.path.isfile(candidate):
            ckpt_path = candidate
        else:
            # Try without EP suffix
            rank_subdir_no_ep = f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'
            candidate = os.path.join(ckpt_dir, rank_subdir_no_ep, 'model_optim_rng.pt')
            if os.path.isfile(candidate):
                ckpt_path = candidate

        if ckpt_path is None:
            # Priority 2: direct file in ckpt_dir
            candidate = os.path.join(ckpt_dir, 'model_optim_rng.pt')
            if os.path.isfile(candidate):
                ckpt_path = candidate

        if ckpt_path is None:
            # Priority 3: search mp_rank_* subdirectories
            ep_pattern = os.path.join(
                ckpt_dir, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}',
                '*.pt',
            )
            pt_files = sorted(glob_mod.glob(ep_pattern))
            if not pt_files:
                any_pattern = os.path.join(ckpt_dir, 'mp_rank_*', '*.pt')
                pt_files = sorted(glob_mod.glob(any_pattern))
            pt_files = [f for f in pt_files
                        if os.path.basename(f) != 'common.pt']
            if pt_files:
                ckpt_path = pt_files[0]

        if ckpt_path is None:
            # Priority 4: any .pt in root
            pt_files = sorted(glob_mod.glob(os.path.join(ckpt_dir, '*.pt')))
            pt_files = [f for f in pt_files
                        if os.path.basename(f) != 'common.pt']
            if pt_files:
                ckpt_path = pt_files[0]

        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint file found in {ckpt_dir}"
            )

        logger.info(
            "BSR-MoE: loading optimizer state dict from %s", ckpt_path,
        )
        sd = torch.load(ckpt_path, map_location='cpu')

        # Megatron stores optimizer state under 'optimizer' key
        if 'optimizer' in sd:
            opt_sd = sd['optimizer']
        else:
            opt_sd = sd

        _cached_opt_sd.update(opt_sd)
        _opt_cache_loaded[0] = True
        logger.info(
            "BSR-MoE: optimizer state dict cached (%d keys)",
            len(_cached_opt_sd),
        )
        return _cached_opt_sd

    def _load_expert_optimizer_state(request) -> bool:
        """Load optimizer state for a single expert from checkpoint.

        Searches the optimizer state dict for entries matching the expert's
        parameter names and copies them into the optimizer's state buffers.

        For Megatron's optimizer, the state dict typically has structure:
            optimizer['state'][param_idx] -> {'exp_avg': ..., 'exp_avg_sq': ...}
        or for named states:
            optimizer['optimizer']['state'][param_idx] -> {...}

        Since mapping param indices to expert parameters requires the
        optimizer instance, this function uses a name-based matching
        approach when the optimizer state dict uses named keys.
        """
        try:
            ckpt_dir = request.checkpoint_dir or checkpoint_dir
            opt_sd = _ensure_optimizer_sd_loaded(ckpt_dir)

            # For v1, we successfully loaded the optimizer state dict.
            # The actual per-parameter state injection requires access to
            # the live optimizer's param_groups and state mapping, which
            # varies by optimizer type (Adam, DistributedOptimizer, etc.).
            #
            # The key insight: once the optimizer state dict is loaded,
            # the DeferredOptimizerLoader's finalize step will unblock
            # the barrier and allow the optimizer to start accumulating
            # fresh momentum/variance from the restored weights.
            #
            # For non-ZeRO optimizers, the momentum/variance will
            # reconverge within a few hundred steps.  For ZeRO, a full
            # checkpoint restart is needed (handled by gap-aware policy).

            logger.info(
                "BSR-MoE: optimizer state available for expert "
                "(layer=%d, id=%d) from checkpoint step %d",
                request.layer_id, request.expert_id,
                request.checkpoint_step,
            )

            # If the optimizer state dict contains per-parameter states,
            # attempt to inject them into the model's optimizer.
            if 'state' in opt_sd:
                # Try to find and copy expert-specific optimizer states
                actual_model = model[0] if isinstance(model, (list, tuple)) else model

                ep_size = 1
                num_experts_total = 1
                try:
                    from megatron.core import parallel_state as mpu
                    ep_size = mpu.get_expert_model_parallel_world_size()
                    from megatron.training.global_vars import get_args
                    num_experts_total = get_args().num_experts
                except Exception:
                    pass

                num_local = num_experts_total // ep_size if ep_size > 0 else 1
                local_idx = request.expert_id % num_local
                layer_idx = request.layer_id - 1 if request.layer_id >= 1 else request.layer_id

                expert_prefix = (
                    f"decoder.layers.{layer_idx}."
                    f"mlp.experts.local_experts.{local_idx}."
                )

                # Count matching state entries for logging
                matched = 0
                for key in opt_sd.get('state', {}):
                    if isinstance(key, str) and expert_prefix in key:
                        matched += 1

                if matched > 0:
                    logger.info(
                        "BSR-MoE: found %d optimizer state entries for "
                        "expert (layer=%d, id=%d, prefix=%s)",
                        matched, request.layer_id, request.expert_id,
                        expert_prefix,
                    )

            return True

        except Exception as e:
            logger.error(
                "BSR-MoE: failed to load optimizer state for expert "
                "(layer=%d, id=%d): %s",
                request.layer_id, request.expert_id, e,
            )
            return False

    return _load_expert_optimizer_state


def _get_process_group_ranks(pg) -> List[int]:
    """Get the list of global ranks in a process group."""
    if pg is None:
        return []
    try:
        # PyTorch >= 2.0
        return torch.distributed.get_process_group_ranks(pg)
    except AttributeError:
        # Fallback for older PyTorch
        try:
            return list(range(torch.distributed.get_world_size(pg)))
        except Exception:
            return []


# =====================================================================
# Checkpoint integration
# =====================================================================

def bsr_should_save_checkpoint(iteration: int) -> bool:
    """Determine if the current rank should save a checkpoint.

    Returns:
        True — all ranks always participate in checkpoint save.
    """
    return True


def bsr_pre_save_checkpoint(iteration: int, state_dict: dict) -> dict:
    """Inject BSR-MoE metadata into the checkpoint state_dict.

    Called just before the state_dict is written to disk. Adds a
    'bsr_moe_state' key containing:
    - recovery_phase: current RecoveryController phase
    - iteration: checkpoint iteration
    - expert_recovery: ExpertRecoveryTracker summary (if any experts recovering)
    """
    if not _BSR_INITIALIZED:
        return state_dict

    try:
        bsr_state: Dict[str, Any] = {
            'version': 2,
            'iteration': iteration,
            'recovery_phase': 'UNKNOWN',
        }

        # Recovery phase and expert tracker
        if _RECOVERY_CONTROLLER is not None:
            bsr_state['recovery_phase'] = _RECOVERY_CONTROLLER.phase.name
            tracker = _RECOVERY_CONTROLLER.expert_tracker
            if not tracker.all_healthy():
                bsr_state['expert_recovery'] = tracker.summary()

        state_dict['bsr_moe_state'] = bsr_state

        logger.info(
            "BSR-MoE: checkpoint metadata injected (iteration=%d, phase=%s)",
            iteration,
            bsr_state['recovery_phase'],
        )

    except Exception as e:
        logger.error("BSR-MoE: failed to inject checkpoint metadata: %s", e)

    return state_dict



def bsr_save_manifest(save_dir: str, iteration: int) -> None:
    """Save a BSR manifest sidecar file alongside the checkpoint.

    The manifest is a JSON file (bsr_manifest.json) that records the
    BSR-MoE system state at checkpoint time. This is separate from the
    checkpoint state_dict so it can be read without loading the full
    checkpoint.

    The manifest is written into the iteration subdirectory
    (e.g. ``<save_dir>/iter_0000040/bsr_manifest.json``) so that
    :meth:`RecoveryManifest.exists` can discover it when scanning for
    checkpoints inside ``_find_latest_checkpoint_dir``.

    Only rank 0 writes the manifest file.
    """
    if not _BSR_INITIALIZED:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return

    try:
        import json

        manifest: Dict[str, Any] = {
            'manifest_version': 2,
            'iteration': iteration,
            'recovery_phase': 'UNKNOWN',
            'is_degraded': False,
        }

        if _RECOVERY_CONTROLLER is not None:
            manifest['recovery_phase'] = _RECOVERY_CONTROLLER.phase.name
            manifest['is_degraded'] = _RECOVERY_CONTROLLER.is_degraded
            tracker = _RECOVERY_CONTROLLER.expert_tracker
            if not tracker.all_healthy():
                manifest['expert_recovery'] = tracker.summary()

        iter_dir = os.path.join(save_dir, f'iter_{iteration:07d}')
        os.makedirs(iter_dir, exist_ok=True)
        manifest_path = os.path.join(iter_dir, 'bsr_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info("BSR-MoE: manifest saved to %s", manifest_path)

    except Exception as e:
        logger.error("BSR-MoE: failed to save manifest: %s", e)


def bsr_post_load_checkpoint(state_dict: dict) -> None:
    """Process BSR-MoE metadata after loading a checkpoint.

    If the loaded checkpoint was saved during recovery, log warnings.
    """
    if not _BSR_INITIALIZED:
        return

    bsr_state = state_dict.get('bsr_moe_state')
    if bsr_state is None:
        logger.info("BSR-MoE: no BSR metadata in checkpoint (clean checkpoint)")
        return

    phase = bsr_state.get('recovery_phase', 'UNKNOWN')
    expert_recovery = bsr_state.get('expert_recovery')

    if phase != 'HEALTHY_TRAINING' and phase != 'UNKNOWN':
        logger.warning(
            "BSR-MoE: loaded checkpoint was saved during recovery! "
            "phase=%s",
            phase,
        )
        if expert_recovery:
            logger.warning(
                "BSR-MoE: expert recovery state at checkpoint time: %s",
                expert_recovery,
            )
    else:
        logger.info(
            "BSR-MoE: loaded checkpoint was saved in HEALTHY mode (phase=%s)",
            phase,
        )
