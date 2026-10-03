"""Overlap compact telemetry transfer and publication with subsequent training."""
from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Mapping

from ..audit.io import digest
from ..control.protocol import MessageType


class AsyncQualityOffloader:
    """Capture immutable features on the caller's stream, publish in background.

    No parameters/optimizer checkpoints are copied here: pass already-computed
    compact routing, sensitivity and parameter/moment-drift telemetry. CUDA
    inputs are cloned before returning to prevent the next step mutating them.
    Completion waits and control RPCs run in one background thread. The bounded
    queue drops a submission when full; no missing step may authorize Hybrid.
    """

    def __init__(self, client, *, model_id: str, telemetry_version: str,
                 policy_version: str, max_pending: int = 2, max_elements: int = 65536):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be positive")
        if type(max_elements) is not int or max_elements < 1:
            raise ValueError("max_elements must be positive")
        for value in (model_id, telemetry_version, policy_version):
            if not isinstance(value, str) or not value:
                raise ValueError("model/telemetry/policy identities are required")
        self.client = client
        self.identity = dict(model_id=model_id, telemetry_version=telemetry_version,
                             policy_version=policy_version)
        self.max_elements = max_elements
        self._slots = threading.BoundedSemaphore(max_pending)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quality-cpu")
        self._lock = threading.RLock()
        self._closed = False
        self._cuda_streams = {}
        self._stats = {"submitted": 0, "acknowledged": 0, "dropped": 0, "failed": 0,
                       "last_ack_step": None, "last_error": None}

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def _capture(self, value, events, count):
        # Keep torch optional for CPU/control deployments.
        if type(value).__module__.startswith("torch"):
            import torch
            if not isinstance(value, torch.Tensor):
                raise TypeError("quality telemetry must contain tensors or JSON values")
            count[0] += value.numel()
            if count[0] > self.max_elements:
                raise ValueError("quality feature element limit exceeded")
            if value.layout != torch.strided:
                raise ValueError("quality tensors must be dense")
            if value.device.type not in ("cpu", "cuda"):
                raise ValueError("quality offload supports CPU and CUDA tensors")
            frozen = value.detach().clone()  # ordered on the producer's current stream
            if not frozen.is_cuda:
                return frozen
            with torch.cuda.device(frozen.device):
                producer = torch.cuda.current_stream(frozen.device)
                transfer = self._cuda_streams.get(frozen.device)
                if transfer is None:
                    transfer = torch.cuda.Stream(device=frozen.device)
                    self._cuda_streams[frozen.device] = transfer
                transfer.wait_stream(producer)
                host = torch.empty_like(frozen, device="cpu", pin_memory=True)
                with torch.cuda.stream(transfer):
                    host.copy_(frozen, non_blocking=True)
                    frozen.record_stream(transfer)
                    event = torch.cuda.Event()
                    event.record(transfer)
                events.append((event, frozen, transfer))
            return host
        if isinstance(value, Mapping):
            count[0] += len(value) + 1
            if count[0] > self.max_elements:
                raise ValueError("quality feature element limit exceeded")
            if any(not isinstance(key, str) for key in value):
                raise ValueError("feature keys must be strings")
            return {key: self._capture(item, events, count) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            count[0] += 1
            if count[0] > self.max_elements:
                raise ValueError("quality feature element limit exceeded")
            return [self._capture(item, events, count) for item in value]
        if value is None or type(value) in (str, bool, int, float):
            count[0] += 1
            if count[0] > self.max_elements:
                raise ValueError("quality feature element limit exceeded")
            return value
        raise TypeError("quality features must contain tensors or JSON values")

    @staticmethod
    def _json_value(value):
        if isinstance(value, dict):
            return {key: AsyncQualityOffloader._json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AsyncQualityOffloader._json_value(item) for item in value]
        if type(value).__module__.startswith("torch"):
            return value.tolist()
        return value

    def submit(self, features: Mapping[str, Any], *, committed_step: int,
               checkpoint_step: int, topology_generation: int,
               group_manifest_hash: str, world_size: int,
               run_history: list[Mapping[str, Any]],
               recovery_epoch: int = 0) -> Future | None:
        """Call only after the global commit boundary; None means backpressure.

        The Future succeeds only after the independent watcher acknowledged the
        immutable snapshot. Its result contains the retained record digest.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("quality offloader is closed")
            if not self._slots.acquire(blocking=False):
                self._stats["dropped"] += 1
                return None
            events = []
            try:
                if not isinstance(features, Mapping):
                    raise TypeError("features must be a mapping")
                captured = self._capture(features, events, [0])
                record = {"schema_version": 1, **self.identity,
                          "owner_rank": self.client.config.sender.get("global_rank"),
                          "recovery_epoch": recovery_epoch,
                          "committed_step": committed_step, "checkpoint_step": checkpoint_step,
                          "topology_generation": topology_generation,
                          "group_manifest_hash": group_manifest_hash, "world_size": world_size}
                # History is lightweight JSON and must be frozen before the
                # caller appends another recovery event.
                record["run_history"] = json.loads(json.dumps(run_history, allow_nan=False))
                future = self._pool.submit(self._publish, record, captured, events, recovery_epoch)
            except Exception:
                self._slots.release()
                raise
            self._stats["submitted"] += 1
            return future

    def _publish(self, record, captured, events, epoch):
        try:
            for event, _source, _stream in events:
                event.synchronize()  # background wait, never on the training thread
            record["features"] = self._json_value(captured)
            json.dumps(record, allow_nan=False)
            response = self.client.request(MessageType.QUALITY_FEATURES, record, recovery_epoch=epoch)
            if (response.get("record_digest") != digest(record) or
                    response.get("committed_step") != record["committed_step"]):
                raise ValueError("watcher did not acknowledge the submitted feature snapshot")
            with self._lock:
                self._stats["acknowledged"] += 1
                self._stats["last_ack_step"] = record["committed_step"]
            return response
        except Exception as exc:
            with self._lock:
                self._stats["failed"] += 1
                self._stats["last_error"] = f"{type(exc).__name__}: {exc}"[:1000]
            import logging
            logging.getLogger(__name__).exception("quality CPU publication failed at step %s",
                                                  record["committed_step"])
            raise
        finally:
            self._slots.release()

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
