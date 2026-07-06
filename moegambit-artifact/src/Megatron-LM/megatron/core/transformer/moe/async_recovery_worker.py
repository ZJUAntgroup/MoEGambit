# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Async Recovery Worker — Background Thread + CUDA Stream (Phase 15).

Provides a background-thread pool for disk I/O (checkpoint reads) and a
dedicated CUDA stream for H2D tensor copies, so that expert weight
restoration and deferred optimizer-state loading do not block the main
training loop.

Architecture
------------
::

    Main thread                     Background thread(s)
    ───────────                     ────────────────────
    submit_expert_load(req)  ──►    _worker_loop():
                                      load_fn(req) → cpu_tensor
                                      put result in _completed_queue
    poll_completed()         ◄──    (non-blocking check)
    apply_to_gpu(results, stream)   (CUDA stream H2D copy)
    wait_stream_and_finalize(...)   (sync stream, update masks)

Key constraints
---------------
* **CUDA tensor ops must happen on the main thread** — background threads
  only do disk I/O and CPU tensor manipulation.
* **NCCL collectives are synchronous** — they are NOT submitted to the
  async worker.
* **GIL is released during I/O waits** — so background threads can make
  progress while the main thread runs CUDA kernels.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class AsyncLoadRequest:
    """Describes a single async load request (expert weight or optimizer state)."""

    request_id: str = ""
    """Unique identifier for this request."""

    expert_id: int = -1
    """Global expert index."""

    layer_id: int = -1
    """MoE layer index."""

    checkpoint_path: str = ""
    """Path to the checkpoint file/directory."""

    weight_location: str = ""
    """Key/path reference for the weight shard in the checkpoint."""

    target_param_name: str = ""
    """Name of the target parameter in the model."""

    request_type: str = "expert_weight"
    """Type of request: 'expert_weight' or 'optimizer_state'."""

    load_fn: Optional[Callable] = None
    """Callback to execute the actual load.  Signature:
    ``load_fn(request: AsyncLoadRequest) -> torch.Tensor`` (CPU tensor).
    If None, a dry-run is performed."""

    # Metadata for the caller
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata passed through to the result."""

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.expert_id)


@dataclass
class AsyncLoadResult:
    """Result of an async load operation."""

    request_id: str = ""
    """Matches the request's request_id."""

    expert_id: int = -1
    """Global expert index."""

    layer_id: int = -1
    """MoE layer index."""

    request_type: str = "expert_weight"
    """Type of request: 'expert_weight' or 'optimizer_state'."""

    cpu_tensor: Optional[torch.Tensor] = None
    """The loaded tensor on CPU.  None if load failed or dry-run."""

    cpu_tensors: Optional[Dict[str, torch.Tensor]] = None
    """Multiple loaded tensors (e.g. optimizer state has exp_avg, exp_avg_sq).
    None if not applicable."""

    target_param_name: str = ""
    """Name of the target parameter in the model."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the load operation."""

    error: Optional[str] = None
    """Error message if load failed.  None on success."""

    success: bool = False
    """Whether the load succeeded."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata passed through from the request."""


# =====================================================================
# Async Recovery Worker
# =====================================================================

class AsyncRecoveryWorker:
    """Background thread pool for async checkpoint loading + CUDA stream H2D copy.

    Usage::

        worker = AsyncRecoveryWorker(max_workers=2)

        # Submit load requests (non-blocking)
        req_id = worker.submit_expert_load(request)

        # In training loop, poll for completed loads
        results = worker.poll_completed()
        if results:
            worker.apply_to_gpu(results, stream)
            stream.synchronize()
            # update health masks, barriers, etc.

        # Cleanup
        worker.shutdown()
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._new_work_event = threading.Event()
        self._shutdown_event = threading.Event()

        # Queues protected by _lock
        self._pending_queue: List[AsyncLoadRequest] = []
        self._completed_queue: List[AsyncLoadResult] = []

        # Track in-flight requests
        self._inflight: Dict[str, AsyncLoadRequest] = {}

        # Worker threads
        self._workers: List[threading.Thread] = []
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"moegambit-async-recovery-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        logger.info(
            "MoEGambit AsyncRecoveryWorker: started %d worker threads",
            max_workers,
        )

    # -----------------------------------------------------------------
    # Submit API
    # -----------------------------------------------------------------

    def submit_expert_load(self, request: AsyncLoadRequest) -> str:
        """Submit an async expert weight load request.

        Args:
            request: The load request.

        Returns:
            The request_id (auto-generated if empty).
        """
        if not request.request_id:
            request.request_id = f"expert_{request.layer_id}_{request.expert_id}_{uuid.uuid4().hex[:8]}"
        request.request_type = "expert_weight"
        return self._submit(request)

    def submit_optimizer_load(self, request: AsyncLoadRequest) -> str:
        """Submit an async optimizer state load request.

        Args:
            request: The load request.

        Returns:
            The request_id.
        """
        if not request.request_id:
            request.request_id = f"optim_{request.layer_id}_{request.expert_id}_{uuid.uuid4().hex[:8]}"
        request.request_type = "optimizer_state"
        return self._submit(request)

    def _submit(self, request: AsyncLoadRequest) -> str:
        """Internal: add request to pending queue."""
        with self._lock:
            self._pending_queue.append(request)
            self._inflight[request.request_id] = request
        self._new_work_event.set()
        logger.debug(
            "MoEGambit async worker: submitted %s request %s "
            "(layer=%d, expert=%d)",
            request.request_type, request.request_id,
            request.layer_id, request.expert_id,
        )
        return request.request_id

    # -----------------------------------------------------------------
    # Poll API (main thread)
    # -----------------------------------------------------------------

    def poll_completed(self) -> List[AsyncLoadResult]:
        """Non-blocking check for completed loads.

        Returns:
            List of completed results (may be empty).
        """
        with self._lock:
            results = list(self._completed_queue)
            self._completed_queue.clear()
            # Remove from inflight
            for r in results:
                self._inflight.pop(r.request_id, None)
        return results

    def has_pending(self) -> bool:
        """Check if there are any pending or in-flight requests."""
        with self._lock:
            return bool(self._pending_queue) or bool(self._inflight)

    @property
    def num_pending(self) -> int:
        """Number of pending requests (not yet picked up by workers)."""
        with self._lock:
            return len(self._pending_queue)

    @property
    def num_inflight(self) -> int:
        """Number of in-flight requests (pending + being processed)."""
        with self._lock:
            return len(self._inflight)

    @property
    def num_completed(self) -> int:
        """Number of completed results waiting to be polled."""
        with self._lock:
            return len(self._completed_queue)

    # -----------------------------------------------------------------
    # GPU apply (main thread only)
    # -----------------------------------------------------------------

    def apply_to_gpu(
        self,
        results: List[AsyncLoadResult],
        stream: torch.cuda.Stream,
        target_params: Optional[Dict[str, torch.nn.Parameter]] = None,
    ) -> int:
        """Copy loaded CPU tensors to GPU on the given CUDA stream.

        This must be called from the main thread.

        Args:
            results: Completed load results with cpu_tensor set.
            stream: The CUDA stream to use for H2D copies.
            target_params: Optional dict mapping param_name -> Parameter.
                If provided, the loaded tensor is copied into the parameter's
                data buffer.  If None, the cpu_tensor is moved to GPU in-place
                on the result object (result.cpu_tensor becomes a GPU tensor).

        Returns:
            Number of tensors copied to GPU.
        """
        copied = 0
        with torch.cuda.stream(stream):
            for result in results:
                if not result.success:
                    continue

                # Handle single tensor (expert weights)
                if result.cpu_tensor is not None:
                    if target_params and result.target_param_name in target_params:
                        param = target_params[result.target_param_name]
                        param.data.copy_(result.cpu_tensor, non_blocking=True)
                    else:
                        # Move tensor to GPU (replace cpu_tensor with gpu version)
                        result.cpu_tensor = result.cpu_tensor.to(
                            device='cuda', non_blocking=True,
                        )
                    copied += 1

                # Handle multiple tensors (optimizer states)
                if result.cpu_tensors:
                    for key, tensor in result.cpu_tensors.items():
                        result.cpu_tensors[key] = tensor.to(
                            device='cuda', non_blocking=True,
                        )
                    copied += len(result.cpu_tensors)

        return copied

    def wait_stream_and_finalize(
        self,
        stream: torch.cuda.Stream,
        results: List[AsyncLoadResult],
        *,
        health_mask_updates: Optional[List[Tuple[int, List[int]]]] = None,
        barrier=None,
        health_managers: Optional[Dict[int, Any]] = None,
        step: int = -1,
    ) -> int:
        """Synchronize the CUDA stream and perform post-copy finalization.

        This must be called from the main thread after ``apply_to_gpu()``.

        Finalization includes:
        - Updating health masks (STALE_RUNNABLE -> FULLY_RECOVERED)
        - Unblocking optimizer barrier
        - Updating health managers

        Args:
            stream: The CUDA stream to synchronize.
            results: The results that were applied to GPU.
            health_mask_updates: List of (layer_id, expert_indices) to mark healthy.
            barrier: OptimizerUpdateBarrier to unblock.
            health_managers: Dict mapping layer_id -> ExpertHealthManager.
            step: Current training step.

        Returns:
            Number of results finalized.
        """
        stream.synchronize()

        finalized = 0
        expert_ids_to_unblock = []

        for result in results:
            if not result.success:
                continue
            expert_ids_to_unblock.append(result.expert_id)
            finalized += 1

        # Update health managers
        if health_managers and finalized > 0:
            # Group by layer
            by_layer: Dict[int, List[int]] = {}
            for result in results:
                if result.success:
                    by_layer.setdefault(result.layer_id, []).append(result.expert_id)

            for layer_id, expert_ids in by_layer.items():
                mgr = health_managers.get(layer_id)
                if mgr is not None:
                    try:
                        mgr.mark_fully_recovered(expert_ids, step=step)
                    except Exception as e:
                        logger.warning(
                            "MoEGambit async worker: health manager transition "
                            "failed for layer %d, experts %s: %s",
                            layer_id, expert_ids, e,
                        )

        # Unblock optimizer barrier
        if barrier is not None and expert_ids_to_unblock:
            try:
                barrier.unblock_expert_params(expert_ids_to_unblock)
            except Exception as e:
                logger.warning(
                    "MoEGambit async worker: barrier unblock failed: %s", e,
                )

        # Update health masks
        if health_mask_updates:
            try:
                from megatron.core.transformer.moe.expert_health import (
                    get_expert_health_mask,
                )
                for layer_id, expert_indices in health_mask_updates:
                    mask = get_expert_health_mask(layer_id, num_experts=0)
                    mask.mark_healthy(expert_indices)
            except Exception as e:
                logger.warning(
                    "MoEGambit async worker: health mask update failed: %s", e,
                )

        if finalized > 0:
            logger.info(
                "MoEGambit async worker: finalized %d results at step %d "
                "(unblocked experts: %s)",
                finalized, step, sorted(set(expert_ids_to_unblock)),
            )

        return finalized

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    def shutdown(self, timeout: float = 10.0) -> None:
        """Shut down all worker threads.

        Args:
            timeout: Maximum seconds to wait for each thread.
        """
        self._shutdown_event.set()
        self._new_work_event.set()  # Wake up sleeping workers

        for t in self._workers:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    "MoEGambit async worker: thread %s did not shut down "
                    "within %.1fs", t.name, timeout,
                )

        with self._lock:
            remaining = len(self._pending_queue) + len(self._inflight)
            if remaining > 0:
                logger.warning(
                    "MoEGambit async worker: shutdown with %d pending/inflight "
                    "requests", remaining,
                )
            self._pending_queue.clear()
            self._inflight.clear()

        self._workers.clear()
        logger.info("MoEGambit AsyncRecoveryWorker: shutdown complete")

    @property
    def is_shutdown(self) -> bool:
        """True if shutdown has been requested."""
        return self._shutdown_event.is_set()

    # -----------------------------------------------------------------
    # Worker loop (background thread)
    # -----------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background worker loop: pick requests from pending queue and execute."""
        thread_name = threading.current_thread().name
        logger.debug("MoEGambit async worker: %s started", thread_name)

        while not self._shutdown_event.is_set():
            # Wait for work or shutdown
            self._new_work_event.wait(timeout=1.0)

            while not self._shutdown_event.is_set():
                # Try to get a request
                request = None
                with self._lock:
                    if self._pending_queue:
                        request = self._pending_queue.pop(0)
                    else:
                        self._new_work_event.clear()
                        break

                if request is None:
                    break

                # Execute the load
                result = self._execute_load(request)

                # Put result in completed queue
                with self._lock:
                    self._completed_queue.append(result)

        logger.debug("MoEGambit async worker: %s exiting", thread_name)

    def _execute_load(self, request: AsyncLoadRequest) -> AsyncLoadResult:
        """Execute a single load request (runs in background thread).

        Only CPU operations are performed here.  No CUDA tensor ops.
        """
        start_time = time.monotonic()
        result = AsyncLoadResult(
            request_id=request.request_id,
            expert_id=request.expert_id,
            layer_id=request.layer_id,
            request_type=request.request_type,
            target_param_name=request.target_param_name,
            metadata=dict(request.metadata),
        )

        try:
            if request.load_fn is not None:
                loaded = request.load_fn(request)
                if isinstance(loaded, torch.Tensor):
                    # Ensure tensor is on CPU
                    if loaded.is_cuda:
                        loaded = loaded.cpu()
                    result.cpu_tensor = loaded
                    result.success = True
                elif isinstance(loaded, dict):
                    # Multiple tensors (optimizer state)
                    cpu_tensors = {}
                    for key, tensor in loaded.items():
                        if isinstance(tensor, torch.Tensor):
                            cpu_tensors[key] = tensor.cpu() if tensor.is_cuda else tensor
                    result.cpu_tensors = cpu_tensors
                    result.success = True
                elif loaded is True:
                    # Dry-run success (load_fn returned True)
                    result.success = True
                elif loaded is False or loaded is None:
                    result.error = "load_fn returned False/None"
                    result.success = False
                else:
                    result.error = f"Unexpected load_fn return type: {type(loaded)}"
                    result.success = False
            else:
                # Dry-run: no load_fn provided
                result.success = True

        except Exception as e:
            result.error = f"Exception in load_fn: {e}"
            result.success = False
            logger.warning(
                "MoEGambit async worker: load failed for %s "
                "(layer=%d, expert=%d): %s",
                request.request_id, request.layer_id,
                request.expert_id, e,
            )

        result.elapsed_seconds = time.monotonic() - start_time
        logger.debug(
            "MoEGambit async worker: completed %s %s "
            "(layer=%d, expert=%d, success=%s, %.3fs)",
            request.request_type, request.request_id,
            request.layer_id, request.expert_id,
            result.success, result.elapsed_seconds,
        )
        return result

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the worker state."""
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "num_workers_alive": sum(1 for t in self._workers if t.is_alive()),
                "num_pending": len(self._pending_queue),
                "num_inflight": len(self._inflight),
                "num_completed": len(self._completed_queue),
                "is_shutdown": self._shutdown_event.is_set(),
            }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"AsyncRecoveryWorker(workers={s['num_workers_alive']}/{s['max_workers']}, "
            f"pending={s['num_pending']}, inflight={s['num_inflight']}, "
            f"completed={s['num_completed']})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_WORKER: Optional[AsyncRecoveryWorker] = None
_WORKER_LOCK = threading.Lock()


def get_async_recovery_worker(max_workers: int = 2) -> AsyncRecoveryWorker:
    """Get or create the global AsyncRecoveryWorker singleton.

    Args:
        max_workers: Number of background threads (only used on first call).

    Returns:
        The global worker instance.
    """
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.is_shutdown:
            _WORKER = AsyncRecoveryWorker(max_workers=max_workers)
    return _WORKER


def clear_async_recovery_worker() -> None:
    """Shut down and clear the global worker (for testing)."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None:
            _WORKER.shutdown()
        _WORKER = None
