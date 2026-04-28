# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deferred Optimizer-State Loading for Stale Experts (Step 12).

After stale expert weights are restored from checkpoint (Step 10), the
experts enter STALE_RUNNABLE state: they can participate in forward and
backward, but their optimizer state (momentum, variance, etc.) is not yet
loaded.  The ``OptimizerUpdateBarrier`` (Step 10) blocks optimizer updates
for these experts to prevent weight corruption.

This module implements a **deferred loader** that:

1. Accepts load requests for expert optimizer states.
2. Tracks each request through a logical state machine.
3. Executes the actual load via a caller-provided callback (``load_fn``).
4. On completion, unblocks the optimizer barrier and transitions the
   expert to FULLY_RECOVERED.

State machine (per expert)
--------------------------
::

    NOT_SUBMITTED
        │
        ▼  submit_load()
    SUBMITTED
        │
        ▼  load_fn() called (at poll or explicit execute)
    LOADING
        │
        ├── success ──▶ LOADED
        │                  │
        │                  ▼  finalize_recovery()
        │              FINALIZED
        │
        └── failure ──▶ FAILED
                           │
                           ▼  retry / resubmit
                       SUBMITTED

Integration with training loop
------------------------------
::

    for step in range(num_steps):
        controller.before_iteration(step)
        forward()
        backward()
        optimizer.step()  # barrier skips stale expert params
        controller.after_iteration(step)

        # Deferred loader hook (can be in after_iteration or separate):
        deferred_loader.poll_and_finalize(step)

The ``poll_and_finalize()`` call is the minimal training-loop hook.  It:
1. Checks if any submitted loads are ready to execute.
2. Executes pending loads (via ``load_fn`` callback).
3. For completed loads: unblocks barrier, transitions health state.

Design principles
-----------------
* **Logically async** — the state machine models async loading without
  requiring actual threads.  The ``load_fn`` callback is called
  synchronously during ``poll_and_finalize()``.  Future versions can
  make this truly async.
* **Callback-based** — no direct Megatron checkpoint dependency.  The
  caller provides ``load_fn(entry) -> bool`` that does the actual I/O.
* **Testable without distributed** — all state transitions are pure Python.

Scope (v1)
----------
* ✅ Logical async state machine for deferred optimizer loading
* ✅ Integration with OptimizerUpdateBarrier (unblock on completion)
* ✅ Integration with ExpertHealthManager (STALE_RUNNABLE → FULLY_RECOVERED)
* ✅ Per-expert tracking and audit trail
* ❌ True async / background thread loading (future)
* ❌ Sharded optimizer state (ZeRO-2) support (future)
* ❌ Automatic FULLY_RECOVERED → HEALTHY promotion (separate safe barrier)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from megatron.core.transformer.moe.async_recovery_worker import (
        AsyncRecoveryWorker,
    )

logger = logging.getLogger(__name__)


# =====================================================================
# Load state enum
# =====================================================================

class OptimizerLoadState(enum.IntEnum):
    """States for a single expert's deferred optimizer-state load."""

    NOT_SUBMITTED = 0
    """No load request has been submitted."""

    SUBMITTED = 1
    """Load request submitted, waiting to be executed."""

    LOADING = 2
    """Load is in progress (for future async support)."""

    LOADED = 3
    """Optimizer state has been loaded successfully."""

    FINALIZED = 4
    """Recovery finalized: barrier unblocked, health state updated."""

    FAILED = 5
    """Load failed.  Can be resubmitted."""


# Valid transitions
_VALID_LOAD_TRANSITIONS = {
    OptimizerLoadState.NOT_SUBMITTED: {OptimizerLoadState.SUBMITTED},
    OptimizerLoadState.SUBMITTED: {
        OptimizerLoadState.LOADING,
        OptimizerLoadState.LOADED,   # direct success (sync load)
        OptimizerLoadState.FAILED,
    },
    OptimizerLoadState.LOADING: {
        OptimizerLoadState.LOADED,
        OptimizerLoadState.FAILED,
    },
    OptimizerLoadState.LOADED: {OptimizerLoadState.FINALIZED},
    OptimizerLoadState.FINALIZED: set(),  # terminal
    OptimizerLoadState.FAILED: {OptimizerLoadState.SUBMITTED},  # retry
}


# =====================================================================
# Per-expert load request
# =====================================================================

@dataclass
class OptimizerLoadRequest:
    """Tracks a single expert's deferred optimizer-state load."""

    layer_id: int = -1
    """MoE layer index."""

    expert_id: int = -1
    """Global expert index."""

    state: OptimizerLoadState = OptimizerLoadState.NOT_SUBMITTED
    """Current load state."""

    optimizer_location: str = ""
    """Key/path reference for the optimizer state in the checkpoint."""

    checkpoint_dir: str = ""
    """Path to the checkpoint directory."""

    checkpoint_step: int = -1
    """Training step of the checkpoint."""

    submit_step: int = -1
    """Training step when the load was submitted."""

    complete_step: int = -1
    """Training step when the load completed."""

    finalize_step: int = -1
    """Training step when recovery was finalized."""

    attempt_count: int = 0
    """Number of load attempts (including retries)."""

    error_message: str = ""
    """Error message from the last failed attempt."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the load operation."""

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.expert_id)

    def transition_to(self, new_state: OptimizerLoadState) -> None:
        """Perform a validated state transition."""
        allowed = _VALID_LOAD_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid optimizer load transition: "
                f"{self.state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        self.state = new_state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "state": self.state.name,
            "optimizer_location": self.optimizer_location,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_step": self.checkpoint_step,
            "submit_step": self.submit_step,
            "complete_step": self.complete_step,
            "finalize_step": self.finalize_step,
            "attempt_count": self.attempt_count,
            "error_message": self.error_message,
            "elapsed_seconds": self.elapsed_seconds,
        }


# =====================================================================
# Deferred Optimizer Loader
# =====================================================================

class DeferredOptimizerLoader:
    """Manages deferred loading of optimizer states for stale experts.

    Usage::

        loader = DeferredOptimizerLoader()

        # After expert weights are restored (Step 10):
        loader.submit_load(
            layer_id=0, expert_id=2,
            optimizer_location="ckpt/expert_2_optim",
            checkpoint_dir="/path/to/ckpt",
            step=100,
        )

        # In training loop (each iteration or safe point):
        loader.poll_and_finalize(
            step=101,
            load_fn=my_load_fn,
            barrier=my_barrier,
            health_managers=my_managers,
        )

        # When all loads are done:
        assert loader.all_finalized()
    """

    def __init__(self) -> None:
        self._requests: Dict[Tuple[int, int], OptimizerLoadRequest] = {}
        self._finalized_history: List[OptimizerLoadRequest] = []

    # -----------------------------------------------------------------
    # Submit API
    # -----------------------------------------------------------------

    def submit_load(
        self,
        layer_id: int,
        expert_id: int,
        *,
        optimizer_location: str = "",
        checkpoint_dir: str = "",
        checkpoint_step: int = -1,
        step: int = -1,
    ) -> OptimizerLoadRequest:
        """Submit a deferred load request for an expert's optimizer state.

        Args:
            layer_id: MoE layer index.
            expert_id: Global expert index.
            optimizer_location: Key/path for the optimizer state shard.
            checkpoint_dir: Path to the checkpoint directory.
            checkpoint_step: Training step of the checkpoint.
            step: Current training step.

        Returns:
            The created ``OptimizerLoadRequest``.
        """
        key = (layer_id, expert_id)
        existing = self._requests.get(key)

        if existing is not None:
            if existing.state == OptimizerLoadState.FAILED:
                # Retry: transition back to SUBMITTED
                existing.transition_to(OptimizerLoadState.SUBMITTED)
                existing.submit_step = step
                existing.error_message = ""
                logger.info(
                    "BSR-MoE deferred loader: resubmitted load for "
                    "expert (layer=%d, id=%d) at step %d",
                    layer_id, expert_id, step,
                )
                return existing
            elif existing.state in (
                OptimizerLoadState.SUBMITTED,
                OptimizerLoadState.LOADING,
            ):
                # Already pending — no-op
                logger.debug(
                    "BSR-MoE deferred loader: load already pending for "
                    "expert (layer=%d, id=%d), state=%s",
                    layer_id, expert_id, existing.state.name,
                )
                return existing
            elif existing.state in (
                OptimizerLoadState.LOADED,
                OptimizerLoadState.FINALIZED,
            ):
                # Already done — no-op
                return existing

        req = OptimizerLoadRequest(
            layer_id=layer_id,
            expert_id=expert_id,
            state=OptimizerLoadState.NOT_SUBMITTED,
            optimizer_location=optimizer_location,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            submit_step=step,
        )
        req.transition_to(OptimizerLoadState.SUBMITTED)
        self._requests[key] = req

        logger.info(
            "BSR-MoE deferred loader: submitted load for expert "
            "(layer=%d, id=%d) at step %d (location=%s)",
            layer_id, expert_id, step, optimizer_location,
        )
        return req

    def submit_from_restore_plan(
        self,
        restore_plan,
        step: int = -1,
    ) -> List[OptimizerLoadRequest]:
        """Submit load requests for all experts in a restore plan.

        Args:
            restore_plan: An ``ExpertRestorePlan`` from Step 10.
            step: Current training step.

        Returns:
            List of created requests.
        """
        requests = []
        for entry in restore_plan.entries:
            req = self.submit_load(
                layer_id=entry.layer_id,
                expert_id=entry.expert_id,
                optimizer_location=entry.optimizer_location,
                checkpoint_dir=entry.checkpoint_dir,
                checkpoint_step=entry.checkpoint_step,
                step=step,
            )
            requests.append(req)
        return requests

    # -----------------------------------------------------------------
    # Poll / execute API
    # -----------------------------------------------------------------

    def poll_ready(self, layer_id: int, expert_id: int) -> OptimizerLoadState:
        """Check the load state for a specific expert.

        Returns:
            The current ``OptimizerLoadState``.
        """
        key = (layer_id, expert_id)
        req = self._requests.get(key)
        if req is None:
            return OptimizerLoadState.NOT_SUBMITTED
        return req.state

    def execute_pending_loads(
        self,
        load_fn: Optional[Callable] = None,
        step: int = -1,
        async_worker: Optional['AsyncRecoveryWorker'] = None,
    ) -> int:
        """Execute all SUBMITTED loads.

        When ``async_worker`` is provided, loads are submitted to the
        background thread pool and the request transitions to LOADING
        (truly async).  When ``async_worker`` is None, loads are executed
        synchronously and transition directly to LOADED.

        Args:
            load_fn: Callback ``load_fn(request: OptimizerLoadRequest) -> bool``.
                Returns True on success.  If None, all loads succeed (dry-run).
            step: Current training step.
            async_worker: Optional ``AsyncRecoveryWorker`` for true async loading.

        Returns:
            Number of loads executed (or submitted to async worker).
        """
        executed = 0
        for req in list(self._requests.values()):
            if req.state != OptimizerLoadState.SUBMITTED:
                continue

            req.attempt_count += 1

            # --- Async path ---
            if async_worker is not None:
                from megatron.core.transformer.moe.async_recovery_worker import (
                    AsyncLoadRequest,
                )
                async_req = AsyncLoadRequest(
                    expert_id=req.expert_id,
                    layer_id=req.layer_id,
                    checkpoint_path=req.checkpoint_dir,
                    weight_location=req.optimizer_location,
                    request_type="optimizer_state",
                    load_fn=load_fn,
                    metadata={
                        "checkpoint_step": req.checkpoint_step,
                        "submit_step": req.submit_step,
                    },
                )
                req_id = async_worker.submit_optimizer_load(async_req)
                req.transition_to(OptimizerLoadState.LOADING)
                # Store the async request_id for tracking
                req.error_message = f"async:{req_id}"  # reuse field for tracking
                executed += 1
                logger.info(
                    "BSR-MoE deferred loader: submitted async optimizer load "
                    "for expert (layer=%d, id=%d) at step %d (req_id=%s)",
                    req.layer_id, req.expert_id, step, req_id,
                )
                continue

            # --- Sync path (original) ---
            start_time = time.monotonic()

            try:
                if load_fn is not None:
                    success = load_fn(req)
                else:
                    success = True  # dry-run

                if success:
                    req.transition_to(OptimizerLoadState.LOADED)
                    req.complete_step = step
                    req.elapsed_seconds = time.monotonic() - start_time
                    executed += 1
                    logger.info(
                        "BSR-MoE deferred loader: loaded optimizer state for "
                        "expert (layer=%d, id=%d) at step %d (%.2fs)",
                        req.layer_id, req.expert_id, step, req.elapsed_seconds,
                    )
                else:
                    req.transition_to(OptimizerLoadState.FAILED)
                    req.error_message = "load_fn returned False"
                    req.elapsed_seconds = time.monotonic() - start_time
                    logger.warning(
                        "BSR-MoE deferred loader: load failed for expert "
                        "(layer=%d, id=%d): load_fn returned False",
                        req.layer_id, req.expert_id,
                    )
            except Exception as e:
                req.transition_to(OptimizerLoadState.FAILED)
                req.error_message = str(e)
                req.elapsed_seconds = time.monotonic() - start_time
                logger.warning(
                    "BSR-MoE deferred loader: load exception for expert "
                    "(layer=%d, id=%d): %s",
                    req.layer_id, req.expert_id, e,
                )

        return executed

    def poll_async_loads(
        self,
        worker: 'AsyncRecoveryWorker',
        stream: 'torch.cuda.Stream',
        step: int = -1,
    ) -> int:
        """Poll the async worker for completed optimizer loads.

        For each completed load:
        1. Execute H2D copy on the provided CUDA stream.
        2. Transition the request from LOADING → LOADED.

        This must be called from the main thread.

        Args:
            worker: The ``AsyncRecoveryWorker`` to poll.
            stream: CUDA stream for H2D copies.
            step: Current training step.

        Returns:
            Number of loads that transitioned to LOADED.
        """
        results = worker.poll_completed()
        if not results:
            return 0

        # Filter for optimizer_state results
        optim_results = [r for r in results if r.request_type == "optimizer_state"]
        if not optim_results:
            return 0

        # Apply to GPU
        import torch as th
        copied = worker.apply_to_gpu(optim_results, stream)

        # Synchronize stream before updating state
        stream.synchronize()

        # Transition LOADING → LOADED
        loaded_count = 0
        for result in optim_results:
            key = (result.layer_id, result.expert_id)
            req = self._requests.get(key)
            if req is None:
                continue
            if req.state != OptimizerLoadState.LOADING:
                continue

            if result.success:
                req.transition_to(OptimizerLoadState.LOADED)
                req.complete_step = step
                req.elapsed_seconds = result.elapsed_seconds
                req.error_message = ""
                loaded_count += 1
                logger.info(
                    "BSR-MoE deferred loader: async optimizer load completed "
                    "for expert (layer=%d, id=%d) at step %d (%.3fs)",
                    req.layer_id, req.expert_id, step, result.elapsed_seconds,
                )
            else:
                req.transition_to(OptimizerLoadState.FAILED)
                req.error_message = result.error or "async load failed"
                logger.warning(
                    "BSR-MoE deferred loader: async optimizer load failed "
                    "for expert (layer=%d, id=%d): %s",
                    req.layer_id, req.expert_id, result.error,
                )

        return loaded_count

    def finalize_loaded(
        self,
        *,
        barrier=None,
        health_managers: Optional[Dict[int, Any]] = None,
        step: int = -1,
    ) -> int:
        """Finalize all LOADED requests: unblock barrier, update health state.

        For each expert in LOADED state:
        1. Unblock the optimizer barrier for that expert.
        2. Transition health manager: STALE_RUNNABLE → FULLY_RECOVERED.
        3. Mark the request as FINALIZED.

        Args:
            barrier: An ``OptimizerUpdateBarrier`` instance.
            health_managers: Dict mapping layer_id → ExpertHealthManager.
            step: Current training step.

        Returns:
            Number of requests finalized.
        """
        finalized = 0

        # Group loaded requests by layer for batch health transitions
        loaded_by_layer: Dict[int, List[OptimizerLoadRequest]] = {}
        for req in self._requests.values():
            if req.state == OptimizerLoadState.LOADED:
                loaded_by_layer.setdefault(req.layer_id, []).append(req)

        if not loaded_by_layer:
            return 0

        # Collect all expert IDs to unblock
        all_expert_ids = []
        for reqs in loaded_by_layer.values():
            for req in reqs:
                all_expert_ids.append(req.expert_id)

        # 1. Unblock optimizer barrier
        if barrier is not None and all_expert_ids:
            barrier.unblock_expert_params(all_expert_ids)
            logger.info(
                "BSR-MoE deferred loader: unblocked optimizer barrier for "
                "expert_ids=%s at step %d",
                sorted(set(all_expert_ids)), step,
            )

        # 2. Transition health manager: STALE_RUNNABLE → FULLY_RECOVERED
        if health_managers is not None:
            for layer_id, reqs in loaded_by_layer.items():
                mgr = health_managers.get(layer_id)
                if mgr is None:
                    continue
                expert_ids = [req.expert_id for req in reqs]
                try:
                    mgr.mark_fully_recovered(expert_ids, step=step)
                except Exception as e:
                    logger.warning(
                        "BSR-MoE deferred loader: health transition failed "
                        "for layer %d, experts %s: %s",
                        layer_id, expert_ids, e,
                    )

        # 3. Mark as FINALIZED
        for reqs in loaded_by_layer.values():
            for req in reqs:
                req.transition_to(OptimizerLoadState.FINALIZED)
                req.finalize_step = step
                finalized += 1
                self._finalized_history.append(req)

        logger.warning(
            "BSR-MoE deferred loader: finalized %d expert optimizer loads "
            "at step %d",
            finalized, step,
        )
        return finalized

    def poll_and_finalize(
        self,
        step: int = -1,
        *,
        load_fn: Optional[Callable] = None,
        barrier=None,
        health_managers: Optional[Dict[int, Any]] = None,
        async_worker: Optional['AsyncRecoveryWorker'] = None,
        cuda_stream: Optional['torch.cuda.Stream'] = None,
    ) -> Tuple[int, int]:
        """Combined poll: execute pending loads, then finalize completed ones.

        This is the minimal training-loop hook.  Call once per iteration
        or at each safe point.

        When ``async_worker`` and ``cuda_stream`` are provided, the async
        path is used:
        1. Submit SUBMITTED loads to the async worker (SUBMITTED → LOADING).
        2. Poll the worker for completed loads (LOADING → LOADED).
        3. Finalize LOADED loads (LOADED → FINALIZED).

        Args:
            step: Current training step.
            load_fn: Callback for loading optimizer state.
            barrier: OptimizerUpdateBarrier instance.
            health_managers: Dict mapping layer_id → ExpertHealthManager.
            async_worker: Optional async worker for true async loading.
            cuda_stream: Optional CUDA stream for H2D copies (required
                when async_worker is provided).

        Returns:
            Tuple of (num_executed, num_finalized).
        """
        num_executed = self.execute_pending_loads(
            load_fn=load_fn, step=step, async_worker=async_worker,
        )

        # Poll async loads if worker is provided
        if async_worker is not None and cuda_stream is not None:
            num_async_loaded = self.poll_async_loads(
                worker=async_worker, stream=cuda_stream, step=step,
            )
            if num_async_loaded > 0:
                logger.info(
                    "BSR-MoE deferred loader: %d async loads completed at step %d",
                    num_async_loaded, step,
                )

        num_finalized = self.finalize_loaded(
            barrier=barrier,
            health_managers=health_managers,
            step=step,
        )
        return num_executed, num_finalized

    # -----------------------------------------------------------------
    # Finalize recovery (explicit per-expert)
    # -----------------------------------------------------------------

    def finalize_recovery(
        self,
        layer_id: int,
        expert_id: int,
        *,
        barrier=None,
        health_managers: Optional[Dict[int, Any]] = None,
        step: int = -1,
    ) -> bool:
        """Finalize recovery for a single expert.

        This is an alternative to ``finalize_loaded()`` for fine-grained
        control.  It only finalizes the specified expert.

        Returns:
            True if the expert was finalized, False otherwise.
        """
        key = (layer_id, expert_id)
        req = self._requests.get(key)
        if req is None or req.state != OptimizerLoadState.LOADED:
            return False

        # Unblock barrier
        if barrier is not None:
            barrier.unblock_expert_params([expert_id])

        # Health transition
        if health_managers is not None:
            mgr = health_managers.get(layer_id)
            if mgr is not None:
                try:
                    mgr.mark_fully_recovered([expert_id], step=step)
                except Exception as e:
                    logger.warning(
                        "BSR-MoE deferred loader: health transition failed "
                        "for expert (layer=%d, id=%d): %s",
                        layer_id, expert_id, e,
                    )

        req.transition_to(OptimizerLoadState.FINALIZED)
        req.finalize_step = step
        self._finalized_history.append(req)
        return True

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    def get_request(
        self, layer_id: int, expert_id: int,
    ) -> Optional[OptimizerLoadRequest]:
        """Get the load request for a specific expert."""
        return self._requests.get((layer_id, expert_id))

    @property
    def num_submitted(self) -> int:
        """Number of requests in SUBMITTED state."""
        return sum(
            1 for r in self._requests.values()
            if r.state == OptimizerLoadState.SUBMITTED
        )

    @property
    def num_loaded(self) -> int:
        """Number of requests in LOADED state (awaiting finalization)."""
        return sum(
            1 for r in self._requests.values()
            if r.state == OptimizerLoadState.LOADED
        )

    @property
    def num_finalized(self) -> int:
        """Number of requests in FINALIZED state."""
        return sum(
            1 for r in self._requests.values()
            if r.state == OptimizerLoadState.FINALIZED
        )

    @property
    def num_failed(self) -> int:
        """Number of requests in FAILED state."""
        return sum(
            1 for r in self._requests.values()
            if r.state == OptimizerLoadState.FAILED
        )

    @property
    def num_pending(self) -> int:
        """Number of requests not yet finalized or failed."""
        return sum(
            1 for r in self._requests.values()
            if r.state in (
                OptimizerLoadState.SUBMITTED,
                OptimizerLoadState.LOADING,
            )
        )

    def all_finalized(self) -> bool:
        """True if all submitted requests have been finalized."""
        return all(
            r.state == OptimizerLoadState.FINALIZED
            for r in self._requests.values()
        )

    def has_pending(self) -> bool:
        """True if there are any pending (non-finalized, non-failed) requests."""
        return self.num_pending > 0

    @property
    def finalized_history(self) -> List[OptimizerLoadRequest]:
        """List of finalized requests (audit trail)."""
        return list(self._finalized_history)

    def summary(self) -> Dict[str, Any]:
        """Summary of the loader state."""
        return {
            "total_requests": len(self._requests),
            "submitted": self.num_submitted,
            "loaded": self.num_loaded,
            "finalized": self.num_finalized,
            "failed": self.num_failed,
            "pending": self.num_pending,
            "all_finalized": self.all_finalized(),
        }

    def reset(self) -> None:
        """Clear all requests and history."""
        self._requests.clear()
        self._finalized_history.clear()

    def __repr__(self) -> str:
        return (
            f"DeferredOptimizerLoader("
            f"total={len(self._requests)}, "
            f"submitted={self.num_submitted}, "
            f"loaded={self.num_loaded}, "
            f"finalized={self.num_finalized}, "
            f"failed={self.num_failed})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_LOADER: Optional[DeferredOptimizerLoader] = None


def get_deferred_optimizer_loader() -> DeferredOptimizerLoader:
    """Get or create the global deferred optimizer loader singleton."""
    global _LOADER
    if _LOADER is None:
        _LOADER = DeferredOptimizerLoader()
    return _LOADER


def clear_deferred_optimizer_loader() -> None:
    """Reset the global loader (for testing)."""
    global _LOADER
    if _LOADER is not None:
        _LOADER.reset()
    _LOADER = None
