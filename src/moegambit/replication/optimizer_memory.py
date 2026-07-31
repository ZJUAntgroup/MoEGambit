"""Framework-neutral host-memory replication for optimizer-owned state.

Framework adapters enumerate tensor and scalar references.  This module owns
buffer reuse, version commit, a TCP ring transport, integrity checks and stable
memory-replica locators; it does not inspect Megatron, DeepSpeed or DDP types.
PyTorch is imported lazily so watcher/control-only installations can still
import :mod:`moegambit.replication` without a training framework installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import logging
import socket
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from ..state.catalog import Placement, StateSource, StateSourceKind
from ..state.version import StateVersion
from .locator import MemoryReplicaLocator
from .transport import (
    FrameLimits,
    TransportIntegrityError,
    canonical_json,
    connect_with_retry,
    payload_digest,
    receive_into,
    receive_json,
    send_json,
)

__all__ = [
    "OptimizerTensorRef",
    "OptimizerScalarRef",
    "OptimizerMemorySnapshot",
    "Zero2MemoryReplicaManager",
    "OptimizerMemoryReplicaManager",
    "apply_optimizer_snapshot",
    "backup_holder_for_owner",
    "capture_optimizer_snapshot",
    "memory_replica_sources",
    "ring_neighbors",
]


logger = logging.getLogger(__name__)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "optimizer memory replication requires PyTorch"
        ) from exc
    return torch


@dataclass(frozen=True)
class OptimizerTensorRef:
    identity: str
    tensor: Any
    is_expert: bool

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("optimizer tensor identity must be non-empty")


@dataclass(frozen=True)
class OptimizerScalarRef:
    identity: str
    state: Mapping[Any, Any]
    key: Any
    is_expert: bool

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("optimizer scalar identity must be non-empty")


@dataclass
class OptimizerMemorySnapshot:
    owner_rank: int
    holder_rank: int
    step: int
    manifest_hash: str
    manifest: list[dict]
    scalars: dict[str, Any]
    segments: list[dict]
    buffers: dict[str, Any]
    generation: int = 0

    def _wire_segments(self) -> list[dict]:
        wire = []
        for segment in self.segments:
            item = dict(segment)
            dtype_name = str(item["dtype"])
            if dtype_name not in self.buffers:
                raise TransportIntegrityError(
                    f"snapshot lacks buffer for {dtype_name!r}"
                )
            item["sha256"] = payload_digest(
                _tensor_byte_view(self.buffers[dtype_name])
            )
            wire.append(item)
        return wire

    def wire_header(self) -> dict:
        return {
            "protocol": 2,
            "generation": int(self.generation),
            "owner_rank": int(self.owner_rank),
            "holder_rank": int(self.holder_rank),
            "step": int(self.step),
            "manifest_hash": str(self.manifest_hash),
            "manifest": list(self.manifest),
            "scalars": dict(self.scalars),
            "segments": self._wire_segments(),
        }

    @classmethod
    def from_wire(
        cls,
        header: Mapping[str, Any],
        buffers: Mapping[str, Any],
    ) -> "OptimizerMemorySnapshot":
        protocol = int(header.get("protocol", 1))
        if protocol not in (1, 2):
            raise TransportIntegrityError(
                f"unsupported optimizer replica protocol {protocol}"
            )
        snapshot = cls(
            owner_rank=int(header["owner_rank"]),
            holder_rank=int(header["holder_rank"]),
            step=int(header["step"]),
            manifest_hash=str(header["manifest_hash"]),
            manifest=list(header["manifest"]),
            scalars=dict(header.get("scalars", {})),
            segments=[dict(item) for item in header["segments"]],
            buffers=dict(buffers),
            generation=int(header.get("generation", 0)),
        )
        _validate_snapshot(snapshot, require_checksums=protocol >= 2)
        return snapshot


@dataclass
class _LocalSlot:
    step: int = -1
    manifest_hash: str = ""
    manifest: Optional[list[dict]] = None
    scalars: Optional[dict[str, Any]] = None
    segments: Optional[list[dict]] = None
    buffers: Optional[dict[str, Any]] = None
    event: Any = None
    in_flight: bool = False
    staged_at: float = 0.0


def _manifest_for_refs(
    refs: Iterable[OptimizerTensorRef],
) -> tuple[list[dict], dict[str, int], str]:
    offsets: Dict[str, int] = {}
    manifest = []
    identities = set()
    for ref in refs:
        if ref.identity in identities:
            raise RuntimeError(
                f"duplicate optimizer snapshot identity: {ref.identity}"
            )
        identities.add(ref.identity)
        tensor = ref.tensor.detach()
        dtype = str(tensor.dtype)
        offset = offsets.get(dtype, 0)
        numel = int(tensor.numel())
        manifest.append(
            {
                "identity": ref.identity,
                "shape": [int(size) for size in tensor.shape],
                "dtype": dtype,
                "numel": numel,
                "offset": offset,
                "is_expert": bool(ref.is_expert),
            }
        )
        offsets[dtype] = offset + numel
    digest = "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest, offsets, digest


def _torch_dtype_by_name(name: str) -> Any:
    torch = _require_torch()
    short_name = str(name).removeprefix("torch.")
    dtype = getattr(torch, short_name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported optimizer snapshot dtype: {name}")
    return dtype


def _tensor_byte_view(tensor: Any) -> memoryview:
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        try:
            return memoryview(tensor).cast("B")
        except TypeError as exc:
            raise TypeError(
                "optimizer snapshot buffer must be a tensor or buffer object"
            ) from exc
    cpu_tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return memoryview(cpu_tensor.numpy()).cast("B")


def _allocate_host_tensor(dtype_name: str, numel: int) -> Any:
    torch = _require_torch()
    dtype = _torch_dtype_by_name(dtype_name)
    if torch.cuda.is_available():
        try:
            return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(numel, dtype=dtype, device="cpu")


def _validate_segments(
    segments: Sequence[Mapping[str, Any]],
    limits: FrameLimits,
) -> int:
    if len(segments) > limits.max_segments:
        raise TransportIntegrityError(
            f"replica contains {len(segments)} segments; limit is {limits.max_segments}"
        )
    total = 0
    dtypes = set()
    for segment in segments:
        dtype = str(segment.get("dtype", ""))
        numel = int(segment.get("numel", -1))
        byte_count = int(segment.get("byte_count", -1))
        if not dtype or dtype in dtypes:
            raise TransportIntegrityError(
                "replica segment dtypes must be non-empty and unique"
            )
        if numel < 0 or byte_count < 0:
            raise TransportIntegrityError("replica segment sizes must be non-negative")
        dtypes.add(dtype)
        total += byte_count
        if total > limits.max_payload_bytes:
            raise TransportIntegrityError(
                f"replica payload exceeds {limits.max_payload_bytes} bytes"
            )
    return total


def _receive_buffer(
    sock: socket.socket,
    segment: Mapping[str, Any],
    *,
    limits: FrameLimits,
    existing: Any = None,
) -> Any:
    _validate_segments((segment,), limits)
    dtype_name = str(segment["dtype"])
    numel = int(segment["numel"])
    size = int(segment["byte_count"])
    if (
        existing is None
        or existing.dtype != _torch_dtype_by_name(dtype_name)
        or int(existing.numel()) != numel
    ):
        existing = _allocate_host_tensor(dtype_name, numel)
    view = _tensor_byte_view(existing)
    if len(view) != size:
        raise TransportIntegrityError(
            f"optimizer replica segment size mismatch: allocated={len(view)} expected={size}"
        )
    receive_into(sock, view)
    expected = str(segment.get("sha256", ""))
    if expected and not hmac.compare_digest(payload_digest(view), expected):
        raise TransportIntegrityError(
            f"optimizer replica checksum mismatch for {dtype_name}"
        )
    return existing


def _validate_snapshot(
    snapshot: OptimizerMemorySnapshot,
    *,
    require_checksums: bool,
) -> None:
    if snapshot.owner_rank < 0 or snapshot.holder_rank < 0 or snapshot.step < 0:
        raise TransportIntegrityError("snapshot rank/step fields must be non-negative")
    manifest_hash = "sha256:" + hashlib.sha256(
        canonical_json(snapshot.manifest)
    ).hexdigest()
    # Protocol 1 used the bare hex representation.  Accept it only while
    # reading legacy snapshots; all new snapshots emit the prefixed digest.
    if snapshot.manifest_hash not in (manifest_hash, manifest_hash.removeprefix("sha256:")):
        raise TransportIntegrityError("optimizer snapshot manifest hash mismatch")
    segment_by_dtype = {str(item["dtype"]): item for item in snapshot.segments}
    if len(segment_by_dtype) != len(snapshot.segments):
        raise TransportIntegrityError("optimizer snapshot has duplicate segments")
    for dtype_name, segment in segment_by_dtype.items():
        if dtype_name not in snapshot.buffers:
            raise TransportIntegrityError(
                f"optimizer snapshot lacks payload for {dtype_name}"
            )
        view = _tensor_byte_view(snapshot.buffers[dtype_name])
        if len(view) != int(segment["byte_count"]):
            raise TransportIntegrityError(
                f"optimizer snapshot payload size mismatch for {dtype_name}"
            )
        expected = str(segment.get("sha256", ""))
        if require_checksums and not expected:
            raise TransportIntegrityError(
                f"optimizer snapshot lacks checksum for {dtype_name}"
            )
        if expected and not hmac.compare_digest(payload_digest(view), expected):
            raise TransportIntegrityError(
                f"optimizer snapshot checksum mismatch for {dtype_name}"
            )


def ring_neighbors(group_ranks: Sequence[int], rank: int) -> tuple[int, int]:
    normalized = [int(item) for item in group_ranks]
    if len(normalized) < 2:
        raise RuntimeError("optimizer memory replication requires DP size >= 2")
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("optimizer replication group contains duplicate ranks")
    try:
        index = normalized.index(int(rank))
    except ValueError as exc:
        raise RuntimeError(
            f"rank {rank} is not in DP group {normalized}"
        ) from exc
    return (
        normalized[(index - 1) % len(normalized)],
        normalized[(index + 1) % len(normalized)],
    )


def backup_holder_for_owner(
    group_ranks: Sequence[int], owner_rank: int
) -> int:
    return ring_neighbors(group_ranks, owner_rank)[1]


def memory_replica_sources(
    snapshot: OptimizerMemorySnapshot,
    *,
    recovery_epoch: int = 0,
    placement: Placement = Placement.SHARDED,
) -> Mapping[str, StateSource]:
    """Project a committed snapshot into policy-consumable source records."""

    version = StateVersion(
        committed_step=int(snapshot.step),
        optimizer_generation=int(snapshot.step),
        recovery_epoch=int(recovery_epoch),
    )
    sources: Dict[str, StateSource] = {}
    for item in snapshot.manifest:
        identity = str(item["identity"])
        sources[identity] = StateSource(
            kind=StateSourceKind.MEMORY_REPLICA,
            version=version,
            locator=str(
                MemoryReplicaLocator(
                    generation=int(snapshot.generation),
                    owner_rank=int(snapshot.owner_rank),
                    holder_rank=int(snapshot.holder_rank),
                    identity=identity,
                )
            ),
            metadata={
                "kind": "optimizer_tensor",
                "placement": placement.value,
                "owner": int(snapshot.owner_rank),
                "shape": list(item.get("shape", ())),
                "dtype": str(item.get("dtype", "")),
                "tags": ["expert"] if item.get("is_expert") else [],
                "manifest_hash": snapshot.manifest_hash,
            },
        )
    for identity in snapshot.scalars:
        sources[str(identity)] = StateSource(
            kind=StateSourceKind.MEMORY_REPLICA,
            version=version,
            locator=str(
                MemoryReplicaLocator(
                    generation=int(snapshot.generation),
                    owner_rank=int(snapshot.owner_rank),
                    holder_rank=int(snapshot.holder_rank),
                    identity=str(identity),
                )
            ),
            metadata={
                "kind": "optimizer_scalar",
                "placement": placement.value,
                "owner": int(snapshot.owner_rank),
                "shape": [],
                "dtype": "",
                "tags": [],
                "manifest_hash": snapshot.manifest_hash,
            },
        )
    return sources


def capture_optimizer_snapshot(
    *,
    owner_rank: int,
    holder_rank: int,
    step: int,
    tensor_refs: Iterable[OptimizerTensorRef],
    scalar_refs: Iterable[OptimizerScalarRef],
    generation: int = 0,
) -> OptimizerMemorySnapshot:
    """Synchronously capture adapter-enumerated optimizer state."""

    torch = _require_torch()
    refs = list(tensor_refs)
    scalar_sources = list(scalar_refs)
    manifest, totals, manifest_hash = _manifest_for_refs(refs)
    if not manifest:
        raise RuntimeError("optimizer recovery snapshot has no tensor state")

    buffers = {
        dtype_name: _allocate_host_tensor(dtype_name, int(numel))
        for dtype_name, numel in totals.items()
    }
    segments = [
        {
            "dtype": dtype_name,
            "numel": int(numel),
            "byte_count": int(numel)
            * int(buffers[dtype_name].element_size()),
        }
        for dtype_name, numel in sorted(totals.items())
    ]
    _validate_segments(segments, FrameLimits())
    used_cuda = False
    with torch.no_grad():
        for ref, item in zip(refs, manifest):
            destination = buffers[item["dtype"]].narrow(
                0, int(item["offset"]), int(item["numel"])
            )
            non_blocking = bool(
                ref.tensor.is_cuda and destination.is_pinned()
            )
            destination.copy_(
                ref.tensor.detach().view(-1), non_blocking=non_blocking
            )
            used_cuda = used_cuda or bool(ref.tensor.is_cuda)
    if used_cuda:
        torch.cuda.current_stream().synchronize()

    return OptimizerMemorySnapshot(
        owner_rank=int(owner_rank),
        holder_rank=int(holder_rank),
        step=int(step),
        manifest_hash=manifest_hash,
        manifest=manifest,
        scalars={
            ref.identity: ref.state[ref.key] for ref in scalar_sources
        },
        segments=segments,
        buffers=buffers,
        generation=int(generation),
    )


class Zero2MemoryReplicaManager:
    """Double-buffered D2H/H2H replication for adapter-enumerated state."""

    def __init__(
        self,
        rank: int,
        tensor_refs_fn: Callable[[], list[OptimizerTensorRef]],
        scalar_refs_fn: Callable[[], list[OptimizerScalarRef]],
        publish_endpoint_fn: Callable[[str, int, int, int], bool],
        wait_endpoint_fn: Callable[[str, float], Optional[Mapping[str, Any]]],
        timeout: float = 300.0,
        *,
        buffer_slots: int = 2,
        bind_host: str = "0.0.0.0",
        retry_interval: float = 0.2,
        limits: Optional[FrameLimits] = None,
    ) -> None:
        if int(rank) < 0:
            raise ValueError("replication rank must be non-negative")
        if timeout <= 0:
            raise ValueError("replication timeout must be positive")
        if int(buffer_slots) not in (1, 2):
            raise ValueError("buffer_slots must be 1 or 2")
        if not bind_host:
            raise ValueError("replication bind_host must be non-empty")
        self.rank = int(rank)
        self._tensor_refs_fn = tensor_refs_fn
        self._scalar_refs_fn = scalar_refs_fn
        self._publish_endpoint_fn = publish_endpoint_fn
        self._wait_endpoint_fn = wait_endpoint_fn
        self.timeout = float(timeout)
        self.buffer_slots = int(buffer_slots)
        self.bind_host = str(bind_host)
        self.retry_interval = float(retry_interval)
        self.limits = limits or FrameLimits()

        self.group_ranks: list[int] = []
        self.generation = 0
        self.owner_to_receive = -1
        self.holder_for_local = -1
        self._listener: Optional[socket.socket] = None
        self._incoming: Optional[socket.socket] = None
        self._outgoing: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._incoming_ready = threading.Event()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._pending_slots: list[int] = []
        self._slots = [_LocalSlot() for _ in range(self.buffer_slots)]
        self._peer_slots: list[Optional[OptimizerMemorySnapshot]] = [
            None for _ in range(self.buffer_slots)
        ]
        self._peer_committed_step = -1
        self._local_replicated_step = -1
        self._failure: Optional[BaseException] = None
        torch = _require_torch()
        self._d2h_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

    @property
    def local_replicated_step(self) -> int:
        with self._cv:
            return self._local_replicated_step

    @property
    def peer_committed_step(self) -> int:
        with self._cv:
            return self._peer_committed_step

    def _peer_id(self, owner_rank: int, holder_rank: int) -> str:
        return (
            # Keep the historical rendezvous key so existing watcher state
            # and mixed-version workers remain interoperable during rollout.
            f"zero2-memory:{self.generation}:"
            f"owner={owner_rank}:holder={holder_rank}"
        )

    def start_transport(
        self, group_ranks: Sequence[int], generation: int = 0
    ) -> None:
        self.stop_transport(clear_failure=True)
        self.group_ranks = [int(item) for item in group_ranks]
        self.generation = int(generation)
        if self.generation < 0:
            raise ValueError("replication generation must be non-negative")
        self.owner_to_receive, self.holder_for_local = ring_neighbors(
            self.group_ranks, self.rank
        )
        self._stop.clear()
        self._incoming_ready.clear()
        with self._cv:
            self._local_replicated_step = -1
            self._peer_committed_step = -1
            self._peer_slots = [None for _ in range(self.buffer_slots)]

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind_host, 0))
        listener.listen(1)
        listener.settimeout(self.timeout)
        self._listener = listener
        port = int(listener.getsockname()[1])
        incoming_id = self._peer_id(self.owner_to_receive, self.rank)
        if not self._publish_endpoint_fn(
            incoming_id, port, self.owner_to_receive, self.rank
        ):
            self.stop_transport()
            raise RuntimeError(
                f"failed to publish optimizer replica endpoint {incoming_id}"
            )

        self._accept_thread = threading.Thread(
            target=self._accept_incoming,
            name=f"optimizer-replica-accept-r{self.rank}",
            daemon=True,
        )
        self._accept_thread.start()

        outgoing_id = self._peer_id(self.rank, self.holder_for_local)
        endpoint = self._wait_endpoint_fn(outgoing_id, self.timeout)
        if endpoint is None:
            self.stop_transport()
            raise RuntimeError(
                f"timed out waiting for optimizer replica endpoint {outgoing_id}"
            )
        try:
            outgoing = connect_with_retry(
                (str(endpoint["host"]), int(endpoint["port"])),
                timeout=self.timeout,
                retry_interval=self.retry_interval,
            )
            outgoing.settimeout(self.timeout)
            self._outgoing = outgoing
            if not self._incoming_ready.wait(self.timeout):
                raise RuntimeError(
                    "timed out accepting optimizer replica ring predecessor"
                )
            self._raise_if_failed()
        except BaseException:
            self.stop_transport()
            raise

        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name=f"optimizer-replica-send-r{self.rank}",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name=f"optimizer-replica-recv-r{self.rank}",
            daemon=True,
        )
        self._sender_thread.start()
        self._receiver_thread.start()

    def _accept_incoming(self) -> None:
        try:
            if self._listener is None:
                raise RuntimeError("replication listener is not initialized")
            incoming, _address = self._listener.accept()
            incoming.settimeout(self.timeout)
            self._incoming = incoming
        except BaseException as exc:
            self._set_failure(exc)
        finally:
            self._incoming_ready.set()

    def _set_failure(self, exc: BaseException) -> None:
        with self._cv:
            if self._failure is None and not self._stop.is_set():
                self._failure = exc
                logger.exception(
                    "optimizer memory replication failed on rank %d", self.rank
                )
            self._cv.notify_all()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("optimizer memory replication failed") from self._failure

    def _allocate_slot(
        self, slot: _LocalSlot, totals: Mapping[str, int]
    ) -> None:
        buffers = {}
        segments = []
        total_bytes = 0
        for dtype_name in sorted(totals):
            numel = int(totals[dtype_name])
            buffer = _allocate_host_tensor(dtype_name, numel)
            byte_count = numel * int(buffer.element_size())
            total_bytes += byte_count
            buffers[dtype_name] = buffer
            segments.append(
                {
                    "dtype": dtype_name,
                    "numel": numel,
                    "byte_count": byte_count,
                }
            )
        _validate_segments(segments, self.limits)
        if total_bytes > self.limits.max_payload_bytes:
            raise RuntimeError("optimizer snapshot exceeds payload limit")
        slot.buffers = buffers
        slot.segments = segments

    def schedule_snapshot(self, step: int) -> Mapping[str, Any]:
        torch = _require_torch()
        step = int(step)
        if step < 0:
            raise ValueError("optimizer snapshot step must be non-negative")
        refs = self._tensor_refs_fn()
        manifest, totals, manifest_hash = _manifest_for_refs(refs)
        if not manifest:
            raise RuntimeError("optimizer memory snapshot has no tensor state")
        if len(manifest) > self.limits.max_segments * 1_000:
            raise RuntimeError("optimizer snapshot manifest is unreasonably large")
        scalar_refs = self._scalar_refs_fn()
        scalar_identities = [ref.identity for ref in scalar_refs]
        if len(scalar_identities) != len(set(scalar_identities)):
            raise RuntimeError("optimizer snapshot scalar identities are not unique")
        scalars = {ref.identity: ref.state[ref.key] for ref in scalar_refs}
        # Reject non-JSON scalar payloads before marking a buffer in-flight.
        canonical_json(scalars)
        slot_index = step % len(self._slots)
        slot = self._slots[slot_index]

        with self._cv:
            self._raise_if_failed()
            deadline = time.monotonic() + self.timeout
            while slot.in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"optimizer snapshot slot {slot_index} is still in flight"
                    )
                self._cv.wait(min(remaining, 0.5))
                self._raise_if_failed()

        if slot.manifest_hash != manifest_hash or slot.buffers is None:
            self._allocate_slot(slot, totals)
        slot.step = step
        slot.manifest_hash = manifest_hash
        slot.manifest = manifest
        slot.scalars = scalars
        slot.staged_at = time.monotonic()

        with torch.no_grad():
            if self._d2h_stream is not None:
                self._d2h_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._d2h_stream):
                    for ref, item in zip(refs, manifest):
                        destination = slot.buffers[item["dtype"]].narrow(
                            0, int(item["offset"]), int(item["numel"])
                        )
                        destination.copy_(
                            ref.tensor.detach().view(-1), non_blocking=True
                        )
                    event = torch.cuda.Event()
                    event.record(self._d2h_stream)
                    slot.event = event
            else:
                for ref, item in zip(refs, manifest):
                    destination = slot.buffers[item["dtype"]].narrow(
                        0, int(item["offset"]), int(item["numel"])
                    )
                    destination.copy_(ref.tensor.detach().view(-1))
                slot.event = None

        with self._cv:
            slot.in_flight = True
            self._pending_slots.append(slot_index)
            self._cv.notify_all()
        return {
            "step": step,
            "tensor_count": len(manifest),
            "scalar_count": len(scalars),
            "bytes": sum(
                int(segment["byte_count"]) for segment in slot.segments or ()
            ),
            "manifest_hash": manifest_hash,
            "generation": self.generation,
        }

    def wait_until_replicated(
        self, step: int, timeout: Optional[float] = None
    ) -> None:
        deadline = time.monotonic() + (
            self.timeout if timeout is None else float(timeout)
        )
        with self._cv:
            while self._local_replicated_step < int(step):
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"optimizer snapshot step {step} not replicated; "
                        f"latest={self._local_replicated_step}"
                    )
                self._cv.wait(min(remaining, 0.5))
            self._raise_if_failed()

    def wait_until_peer_committed(
        self, step: int, timeout: Optional[float] = None
    ) -> None:
        """Wait until this rank holds a complete peer snapshot for ``step``."""

        deadline = time.monotonic() + (
            self.timeout if timeout is None else float(timeout)
        )
        with self._cv:
            while self._peer_committed_step < int(step):
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"peer optimizer snapshot step {step} not committed; "
                        f"latest={self._peer_committed_step}"
                    )
                self._cv.wait(min(remaining, 0.5))
            self._raise_if_failed()

    def _sender_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._cv:
                    while not self._pending_slots and not self._stop.is_set():
                        self._cv.wait(0.5)
                    if self._stop.is_set():
                        return
                    slot_index = self._pending_slots.pop(0)
                slot = self._slots[slot_index]
                if slot.event is not None:
                    slot.event.synchronize()
                if self._outgoing is None:
                    raise RuntimeError("outgoing replication socket is unavailable")
                snapshot = OptimizerMemorySnapshot(
                    owner_rank=self.rank,
                    holder_rank=self.holder_for_local,
                    step=slot.step,
                    manifest_hash=slot.manifest_hash,
                    manifest=list(slot.manifest or ()),
                    scalars=dict(slot.scalars or {}),
                    segments=list(slot.segments or ()),
                    buffers=dict(slot.buffers or {}),
                    generation=self.generation,
                )
                header = snapshot.wire_header()
                send_json(self._outgoing, header, limits=self.limits)
                for segment in snapshot.segments:
                    self._outgoing.sendall(
                        _tensor_byte_view(snapshot.buffers[segment["dtype"]])
                    )
                ack = receive_json(self._outgoing, limits=self.limits)
                if (
                    int(ack.get("step", -1)) != slot.step
                    or str(ack.get("manifest_hash", "")) != slot.manifest_hash
                    or not ack.get("committed")
                ):
                    raise RuntimeError(
                        f"bad optimizer replica acknowledgement: {ack}"
                    )
                with self._cv:
                    self._local_replicated_step = max(
                        self._local_replicated_step, slot.step
                    )
                    slot.in_flight = False
                    self._cv.notify_all()
        except BaseException as exc:
            self._set_failure(exc)

    def _receiver_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self._incoming is None:
                    raise RuntimeError("incoming replication socket is unavailable")
                header = receive_json(self._incoming, limits=self.limits)
                if int(header.get("protocol", -1)) != 2:
                    raise TransportIntegrityError(
                        "live optimizer replication requires protocol 2"
                    )
                if int(header.get("generation", -1)) != self.generation:
                    raise TransportIntegrityError(
                        "optimizer replica generation does not match"
                    )
                if int(header.get("owner_rank", -1)) != self.owner_to_receive:
                    raise TransportIntegrityError(
                        f"optimizer replica owner mismatch: {header}"
                    )
                if int(header.get("holder_rank", -1)) != self.rank:
                    raise TransportIntegrityError(
                        f"optimizer replica holder mismatch: {header}"
                    )
                segments = header.get("segments", ())
                if not isinstance(segments, list):
                    raise TransportIntegrityError(
                        "optimizer replica segments must be a list"
                    )
                _validate_segments(segments, self.limits)
                slot_index = int(header["step"]) % len(self._peer_slots)
                previous = self._peer_slots[slot_index]
                buffers = {}
                for segment in segments:
                    dtype_name = str(segment["dtype"])
                    reusable = None
                    if (
                        previous is not None
                        and previous.manifest_hash == header.get("manifest_hash")
                    ):
                        reusable = previous.buffers.get(dtype_name)
                    buffers[dtype_name] = _receive_buffer(
                        self._incoming,
                        segment,
                        limits=self.limits,
                        existing=reusable,
                    )
                snapshot = OptimizerMemorySnapshot.from_wire(header, buffers)
                with self._cv:
                    self._peer_slots[slot_index] = snapshot
                    self._peer_committed_step = max(
                        self._peer_committed_step, snapshot.step
                    )
                    self._cv.notify_all()
                send_json(
                    self._incoming,
                    {
                        "step": snapshot.step,
                        "manifest_hash": snapshot.manifest_hash,
                        "committed": True,
                    },
                    limits=self.limits,
                )
        except (ConnectionError, OSError) as exc:
            with self._cv:
                has_snapshot = self._peer_committed_step >= 0
            if self._stop.is_set() or has_snapshot:
                return
            self._set_failure(exc)
        except BaseException as exc:
            self._set_failure(exc)

    def get_peer_snapshot(
        self, owner_rank: int, step: int
    ) -> OptimizerMemorySnapshot:
        with self._cv:
            self._raise_if_failed()
            candidates = [
                snapshot
                for snapshot in self._peer_slots
                if snapshot is not None
                and snapshot.owner_rank == int(owner_rank)
                and snapshot.step == int(step)
                and snapshot.generation == self.generation
            ]
            if len(candidates) != 1:
                available = [
                    (snapshot.owner_rank, snapshot.step, snapshot.generation)
                    for snapshot in self._peer_slots
                    if snapshot is not None
                ]
                raise RuntimeError(
                    f"optimizer replica owner={owner_rank} step={step} unavailable; "
                    f"available={available}"
                )
            return candidates[0]

    def stop_transport(self, clear_failure: bool = False) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for sock in (self._incoming, self._outgoing, self._listener):
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        for thread in (
            self._accept_thread,
            self._sender_thread,
            self._receiver_thread,
        ):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._listener = None
        self._incoming = None
        self._outgoing = None
        self._accept_thread = None
        self._sender_thread = None
        self._receiver_thread = None
        self._pending_slots.clear()
        for slot in self._slots:
            slot.in_flight = False
        if clear_failure:
            self._failure = None


OptimizerMemoryReplicaManager = Zero2MemoryReplicaManager


def apply_optimizer_snapshot(
    snapshot: OptimizerMemorySnapshot,
    tensor_refs: list[OptimizerTensorRef],
    scalar_refs: list[OptimizerScalarRef],
    *,
    restore_expert: bool,
) -> Mapping[str, Any]:
    torch = _require_torch()
    _validate_snapshot(snapshot, require_checksums=False)
    manifest, _totals, manifest_hash = _manifest_for_refs(tensor_refs)
    compatible_hashes = {
        manifest_hash,
        manifest_hash.removeprefix("sha256:"),
    }
    if snapshot.manifest_hash not in compatible_hashes or manifest != snapshot.manifest:
        raise RuntimeError(
            "optimizer memory snapshot manifest mismatch: "
            f"local={manifest_hash} remote={snapshot.manifest_hash}"
        )

    segment_by_dtype = {str(item["dtype"]): item for item in snapshot.segments}
    cpu_tensors = {}
    for dtype_name, segment in segment_by_dtype.items():
        dtype = _torch_dtype_by_name(dtype_name)
        buffer = snapshot.buffers[dtype_name]
        if isinstance(buffer, torch.Tensor):
            if buffer.dtype != dtype or int(buffer.numel()) != int(segment["numel"]):
                raise RuntimeError(
                    f"optimizer snapshot buffer mismatch for {dtype_name}: "
                    f"shape={tuple(buffer.shape)} dtype={buffer.dtype}"
                )
            cpu_tensors[dtype_name] = buffer.view(-1)
        else:
            cpu_tensors[dtype_name] = torch.frombuffer(
                buffer, dtype=dtype, count=int(segment["numel"])
            )

    copied_tensors = 0
    copied_bytes = 0
    used_cuda = False
    with torch.no_grad():
        for ref, item in zip(tensor_refs, manifest):
            if ref.is_expert and not restore_expert:
                continue
            source = cpu_tensors[item["dtype"]].narrow(
                0, int(item["offset"]), int(item["numel"])
            ).view(item["shape"])
            non_blocking = bool(ref.tensor.is_cuda and source.is_pinned())
            ref.tensor.copy_(source, non_blocking=non_blocking)
            used_cuda = used_cuda or bool(ref.tensor.is_cuda)
            copied_tensors += 1
            copied_bytes += int(source.numel()) * int(source.element_size())
    if used_cuda:
        torch.cuda.current_stream().synchronize()

    scalar_by_identity = {ref.identity: ref for ref in scalar_refs}
    copied_scalars = 0
    for identity, ref in scalar_by_identity.items():
        if ref.is_expert and not restore_expert:
            continue
        if identity not in snapshot.scalars:
            raise RuntimeError(f"optimizer snapshot scalar missing: {identity}")
        ref.state[ref.key] = snapshot.scalars[identity]
        copied_scalars += 1

    return {
        "owner_rank": snapshot.owner_rank,
        "holder_rank": snapshot.holder_rank,
        "step": snapshot.step,
        "generation": snapshot.generation,
        "manifest_hash": snapshot.manifest_hash,
        "tensor_count": copied_tensors,
        "scalar_count": copied_scalars,
        "bytes": copied_bytes,
        "restore_scope": "all" if restore_expert else "non_expert",
    }
