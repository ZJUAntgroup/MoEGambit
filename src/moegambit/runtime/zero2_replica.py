"""Engine-neutral in-memory replication for rank-local optimizer shards.

The manager keeps two pinned host buffers for the local optimizer shard and
replicates completed snapshots to one data-parallel ring neighbor over a
dedicated TCP connection. Engine adapters provide stable tensor/scalar
references and endpoint discovery; this module has no Megatron or DeepSpeed
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import socket
import struct
import threading
import time
from typing import Any, Callable, Iterable, Optional

import torch


logger = logging.getLogger(__name__)


@dataclass
class OptimizerTensorRef:
    identity: str
    tensor: torch.Tensor
    is_expert: bool


@dataclass
class OptimizerScalarRef:
    identity: str
    state: dict
    key: Any
    is_expert: bool


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

    def wire_header(self) -> dict:
        return {
            "protocol": 1,
            "owner_rank": self.owner_rank,
            "holder_rank": self.holder_rank,
            "step": self.step,
            "manifest_hash": self.manifest_hash,
            "manifest": self.manifest,
            "scalars": self.scalars,
            "segments": self.segments,
        }

    @classmethod
    def from_wire(cls, header: dict, buffers: dict[str, Any]):
        return cls(
            owner_rank=int(header["owner_rank"]),
            holder_rank=int(header["holder_rank"]),
            step=int(header["step"]),
            manifest_hash=str(header["manifest_hash"]),
            manifest=list(header["manifest"]),
            scalars=dict(header.get("scalars", {})),
            segments=list(header["segments"]),
            buffers=buffers,
        )


@dataclass
class _LocalSlot:
    step: int = -1
    manifest_hash: str = ""
    manifest: Optional[list[dict]] = None
    scalars: Optional[dict[str, Any]] = None
    segments: Optional[list[dict]] = None
    buffers: Optional[dict[str, torch.Tensor]] = None
    event: Optional[torch.cuda.Event] = None
    in_flight: bool = False
    staged_at: float = 0.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_for_refs(refs: Iterable[OptimizerTensorRef]):
    offsets: dict[str, int] = {}
    manifest = []
    for ref in refs:
        tensor = ref.tensor.detach()
        dtype = str(tensor.dtype)
        offset = offsets.get(dtype, 0)
        manifest.append(
            {
                "identity": ref.identity,
                "shape": list(tensor.shape),
                "dtype": dtype,
                "numel": tensor.numel(),
                "offset": offset,
                "is_expert": bool(ref.is_expert),
            }
        )
        offsets[dtype] = offset + tensor.numel()
    digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return manifest, offsets, digest


def _torch_dtype_by_name(name: str) -> torch.dtype:
    short_name = name.removeprefix("torch.")
    dtype = getattr(torch, short_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported optimizer snapshot dtype: {name}")
    return dtype


def _tensor_byte_view(tensor: torch.Tensor) -> memoryview:
    cpu_tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return memoryview(cpu_tensor.numpy()).cast("B")


def _send_json(sock: socket.socket, payload: dict):
    data = _canonical_json(payload)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:], size - offset)
        if received <= 0:
            raise ConnectionError("optimizer replica socket closed")
        offset += received
    return bytes(data)


def _recv_json(sock: socket.socket) -> dict:
    (size,) = struct.unpack("!Q", _recv_exact(sock, 8))
    if size > 128 * 1024 * 1024:
        raise RuntimeError(f"optimizer replica header is unreasonably large: {size}")
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


def _allocate_host_tensor(dtype_name: str, numel: int) -> torch.Tensor:
    dtype = _torch_dtype_by_name(dtype_name)
    if torch.cuda.is_available():
        try:
            return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(numel, dtype=dtype, device="cpu")


def _recv_buffer(
    sock: socket.socket, segment: dict, existing: Optional[torch.Tensor] = None
) -> torch.Tensor:
    dtype_name = str(segment["dtype"])
    numel = int(segment["numel"])
    size = int(segment["byte_count"])
    if (
        existing is None
        or existing.dtype != _torch_dtype_by_name(dtype_name)
        or existing.numel() != numel
    ):
        existing = _allocate_host_tensor(dtype_name, numel)
    view = _tensor_byte_view(existing)
    if len(view) != size:
        raise RuntimeError(
            f"optimizer replica segment size mismatch: allocated={len(view)} expected={size}"
        )
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:], size - offset)
        if received <= 0:
            raise ConnectionError("optimizer replica socket closed during payload")
        offset += received
    return existing


def ring_neighbors(group_ranks: list[int], rank: int) -> tuple[int, int]:
    if len(group_ranks) < 2:
        raise RuntimeError("optimizer memory replication requires DP size >= 2")
    try:
        index = group_ranks.index(rank)
    except ValueError as exc:
        raise RuntimeError(f"rank {rank} is not in DP group {group_ranks}") from exc
    owner_to_receive = group_ranks[(index - 1) % len(group_ranks)]
    holder_for_local = group_ranks[(index + 1) % len(group_ranks)]
    return owner_to_receive, holder_for_local


def backup_holder_for_owner(group_ranks: list[int], owner_rank: int) -> int:
    _, holder = ring_neighbors(group_ranks, owner_rank)
    return holder


@torch.no_grad()
def capture_optimizer_snapshot(
    *,
    owner_rank: int,
    holder_rank: int,
    step: int,
    tensor_refs: Iterable[OptimizerTensorRef],
    scalar_refs: Iterable[OptimizerScalarRef],
) -> OptimizerMemorySnapshot:
    """Synchronously capture optimizer references in host memory.

    This is used only at a recovery boundary. The steady-state replica path
    remains asynchronous; recovery freezes an immutable snapshot while the
    survivor process stays resident.
    """
    refs = list(tensor_refs)
    scalars_source = list(scalar_refs)
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
            "byte_count": int(numel) * buffers[dtype_name].element_size(),
        }
        for dtype_name, numel in sorted(totals.items())
    ]
    used_cuda = False
    for ref, item in zip(refs, manifest):
        destination = buffers[item["dtype"]].narrow(
            0, int(item["offset"]), int(item["numel"])
        )
        non_blocking = bool(ref.tensor.is_cuda and destination.is_pinned())
        destination.copy_(ref.tensor.detach().view(-1), non_blocking=non_blocking)
        used_cuda = used_cuda or ref.tensor.is_cuda
    if used_cuda:
        torch.cuda.current_stream().synchronize()

    return OptimizerMemorySnapshot(
        owner_rank=int(owner_rank),
        holder_rank=int(holder_rank),
        step=int(step),
        manifest_hash=manifest_hash,
        manifest=manifest,
        scalars={
            ref.identity: ref.state[ref.key]
            for ref in scalars_source
        },
        segments=segments,
        buffers=buffers,
    )


class Zero2MemoryReplicaManager:
    """Double-buffered D2H/H2H optimizer-state replication."""

    def __init__(
        self,
        rank: int,
        tensor_refs_fn: Callable[[], list[OptimizerTensorRef]],
        scalar_refs_fn: Callable[[], list[OptimizerScalarRef]],
        publish_endpoint_fn: Callable[[str, int, int, int], bool],
        wait_endpoint_fn: Callable[[str, float], Optional[dict]],
        timeout: float = 300.0,
        buffer_slots: int = 2,
    ):
        self.rank = int(rank)
        self._tensor_refs_fn = tensor_refs_fn
        self._scalar_refs_fn = scalar_refs_fn
        self._publish_endpoint_fn = publish_endpoint_fn
        self._wait_endpoint_fn = wait_endpoint_fn
        self.timeout = float(timeout)
        self.buffer_slots = int(buffer_slots)
        if self.buffer_slots not in (1, 2):
            raise ValueError("buffer_slots must be 1 or 2")

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
        self._d2h_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

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
            f"zero2-memory:{self.generation}:"
            f"owner={owner_rank}:holder={holder_rank}"
        )

    def start_transport(self, group_ranks: list[int], generation: int = 0):
        self.stop_transport(clear_failure=True)
        self.group_ranks = list(group_ranks)
        self.generation = int(generation)
        self.owner_to_receive, self.holder_for_local = ring_neighbors(
            self.group_ranks, self.rank
        )
        self._stop.clear()
        self._incoming_ready.clear()
        with self._cv:
            self._local_replicated_step = -1

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        listener.settimeout(self.timeout)
        self._listener = listener
        port = int(listener.getsockname()[1])
        incoming_id = self._peer_id(self.owner_to_receive, self.rank)
        if not self._publish_endpoint_fn(
            incoming_id, port, self.owner_to_receive, self.rank
        ):
            raise RuntimeError(f"failed to publish optimizer replica endpoint {incoming_id}")

        self._accept_thread = threading.Thread(
            target=self._accept_incoming,
            name=f"zero2-accept-r{self.rank}",
            daemon=True,
        )
        self._accept_thread.start()

        outgoing_id = self._peer_id(self.rank, self.holder_for_local)
        endpoint = self._wait_endpoint_fn(outgoing_id, self.timeout)
        if endpoint is None:
            raise RuntimeError(f"timed out waiting for optimizer replica endpoint {outgoing_id}")
        endpoint_address = (str(endpoint["host"]), int(endpoint["port"]))
        deadline = time.monotonic() + self.timeout
        last_error = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    "timed out connecting to optimizer replica endpoint "
                    f"{endpoint_address[0]}:{endpoint_address[1]}: {last_error}"
                ) from last_error
            try:
                outgoing = socket.create_connection(
                    endpoint_address, timeout=min(10.0, remaining)
                )
                break
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.2, remaining))
        outgoing.settimeout(self.timeout)
        self._outgoing = outgoing
        if not self._incoming_ready.wait(self.timeout):
            raise RuntimeError("timed out accepting optimizer replica ring predecessor")
        self._raise_if_failed()

        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name=f"zero2-h2h-send-r{self.rank}",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name=f"zero2-h2h-recv-r{self.rank}",
            daemon=True,
        )
        self._sender_thread.start()
        self._receiver_thread.start()
        logger.warning(
            "[elastic-zero2] rank=%d transport ready owner=%d holder=%d generation=%d",
            self.rank,
            self.owner_to_receive,
            self.holder_for_local,
            self.generation,
        )

    def _accept_incoming(self):
        try:
            incoming, _ = self._listener.accept()
            incoming.settimeout(self.timeout)
            self._incoming = incoming
            self._incoming_ready.set()
        except BaseException as exc:
            self._set_failure(exc)
            self._incoming_ready.set()

    def _set_failure(self, exc: BaseException):
        with self._cv:
            if self._failure is None and not self._stop.is_set():
                self._failure = exc
                logger.exception("[elastic-zero2] rank=%d replication failed", self.rank)
            self._cv.notify_all()

    def _raise_if_failed(self):
        if self._failure is not None:
            raise RuntimeError("optimizer memory replication failed") from self._failure

    def _allocate_slot(self, slot: _LocalSlot, totals: dict[str, int]):
        buffers = {}
        segments = []
        for dtype_name in sorted(totals):
            dtype = _torch_dtype_by_name(dtype_name)
            numel = int(totals[dtype_name])
            buffer = _allocate_host_tensor(dtype_name, numel)
            buffers[dtype_name] = buffer
            segments.append(
                {
                    "dtype": dtype_name,
                    "numel": numel,
                    "byte_count": numel * buffer.element_size(),
                }
            )
        slot.buffers = buffers
        slot.segments = segments

    @torch.no_grad()
    def schedule_snapshot(self, step: int):
        step = int(step)
        refs = self._tensor_refs_fn()
        manifest, totals, manifest_hash = _manifest_for_refs(refs)
        if not manifest:
            raise RuntimeError("optimizer memory snapshot has no tensor state")
        scalar_refs = self._scalar_refs_fn()
        scalars = {ref.identity: ref.state[ref.key] for ref in scalar_refs}
        slot_index = step % len(self._slots)
        slot = self._slots[slot_index]

        with self._cv:
            self._raise_if_failed()
            deadline = time.monotonic() + self.timeout
            while slot.in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"optimizer snapshot slot {slot_index} is still in flight")
                self._cv.wait(min(remaining, 0.5))
                self._raise_if_failed()

        if slot.manifest_hash != manifest_hash or slot.buffers is None:
            self._allocate_slot(slot, totals)
        slot.step = step
        slot.manifest_hash = manifest_hash
        slot.manifest = manifest
        slot.scalars = scalars
        slot.staged_at = time.monotonic()

        if self._d2h_stream is not None:
            # optimizer.step() launches updates asynchronously on the current
            # stream.  The offload stream must observe those writes before it
            # reads the rank-local shard.
            self._d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._d2h_stream):
                for ref, item in zip(refs, manifest):
                    destination = slot.buffers[item["dtype"]].narrow(
                        0, int(item["offset"]), int(item["numel"])
                    )
                    destination.copy_(ref.tensor.detach().view(-1), non_blocking=True)
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
            "bytes": sum(int(segment["byte_count"]) for segment in slot.segments),
            "manifest_hash": manifest_hash,
        }

    def wait_until_replicated(self, step: int, timeout: Optional[float] = None):
        deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
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
    ):
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

    def _sender_loop(self):
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
                d2h_done_at = time.monotonic()
                snapshot = OptimizerMemorySnapshot(
                    owner_rank=self.rank,
                    holder_rank=self.holder_for_local,
                    step=slot.step,
                    manifest_hash=slot.manifest_hash,
                    manifest=slot.manifest,
                    scalars=slot.scalars,
                    segments=slot.segments,
                    buffers=slot.buffers,
                )
                _send_json(self._outgoing, snapshot.wire_header())
                for segment in snapshot.segments:
                    self._outgoing.sendall(
                        _tensor_byte_view(snapshot.buffers[segment["dtype"]])
                    )
                ack = _recv_json(self._outgoing)
                if int(ack.get("step", -1)) != slot.step or not ack.get("committed"):
                    raise RuntimeError(f"bad optimizer replica acknowledgement: {ack}")
                h2h_done_at = time.monotonic()
                logger.info(
                    "[elastic-zero2] rank=%d step=%d committed "
                    "d2h_ms=%.1f h2h_ms=%.1f bytes=%d",
                    self.rank,
                    slot.step,
                    (d2h_done_at - slot.staged_at) * 1000.0,
                    (h2h_done_at - d2h_done_at) * 1000.0,
                    sum(int(item["byte_count"]) for item in slot.segments),
                )
                with self._cv:
                    self._local_replicated_step = max(
                        self._local_replicated_step, slot.step
                    )
                    slot.in_flight = False
                    self._cv.notify_all()
        except BaseException as exc:
            self._set_failure(exc)

    def _receiver_loop(self):
        try:
            while not self._stop.is_set():
                header = _recv_json(self._incoming)
                if int(header.get("owner_rank", -1)) != self.owner_to_receive:
                    raise RuntimeError(f"optimizer replica owner mismatch: {header}")
                if int(header.get("holder_rank", -1)) != self.rank:
                    raise RuntimeError(f"optimizer replica holder mismatch: {header}")
                slot_index = int(header["step"]) % len(self._peer_slots)
                previous = self._peer_slots[slot_index]
                buffers = {}
                for segment in header["segments"]:
                    dtype_name = str(segment["dtype"])
                    reusable = None
                    if (
                        previous is not None
                        and previous.manifest_hash == header.get("manifest_hash")
                    ):
                        candidate = previous.buffers.get(dtype_name)
                        if isinstance(candidate, torch.Tensor):
                            reusable = candidate
                    buffers[dtype_name] = _recv_buffer(
                        self._incoming, segment, reusable
                    )
                snapshot = OptimizerMemorySnapshot.from_wire(header, buffers)
                with self._cv:
                    self._peer_slots[slot_index] = snapshot
                    self._peer_committed_step = max(
                        self._peer_committed_step, snapshot.step
                    )
                    self._cv.notify_all()
                _send_json(self._incoming, {"step": snapshot.step, "committed": True})
        except (ConnectionError, OSError) as exc:
            # During recovery, ranks retire the old replication generation at
            # slightly different times.  Once a complete peer snapshot exists,
            # a clean socket close is expected; version checks at restore time
            # still reject an older snapshot.
            with self._cv:
                has_committed_snapshot = self._peer_committed_step >= 0
            if self._stop.is_set() or has_committed_snapshot:
                logger.info(
                    "[elastic-zero2] rank=%d peer transport closed after step=%d: %s",
                    self.rank,
                    self._peer_committed_step,
                    exc,
                )
                return
            self._set_failure(exc)
        except BaseException as exc:
            self._set_failure(exc)

    def get_peer_snapshot(self, owner_rank: int, step: int) -> OptimizerMemorySnapshot:
        with self._cv:
            self._raise_if_failed()
            candidates = [
                snapshot
                for snapshot in self._peer_slots
                if snapshot is not None
                and snapshot.owner_rank == int(owner_rank)
                and snapshot.step == int(step)
            ]
            if len(candidates) != 1:
                available = [
                    (snapshot.owner_rank, snapshot.step)
                    for snapshot in self._peer_slots
                    if snapshot is not None
                ]
                raise RuntimeError(
                    f"optimizer replica owner={owner_rank} step={step} unavailable; "
                    f"available={available}"
                )
            return candidates[0]

    def stop_transport(self, clear_failure: bool = False):
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
        for thread in (self._accept_thread, self._sender_thread, self._receiver_thread):
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


@torch.no_grad()
def apply_optimizer_snapshot(
    snapshot: OptimizerMemorySnapshot,
    tensor_refs: list[OptimizerTensorRef],
    scalar_refs: list[OptimizerScalarRef],
    *,
    restore_expert: bool,
):
    manifest, _, manifest_hash = _manifest_for_refs(tensor_refs)
    if manifest_hash != snapshot.manifest_hash or manifest != snapshot.manifest:
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
            if buffer.dtype != dtype or buffer.numel() != int(segment["numel"]):
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
    for ref, item in zip(tensor_refs, manifest):
        if ref.is_expert and not restore_expert:
            continue
        source = cpu_tensors[item["dtype"]].narrow(
            0, int(item["offset"]), int(item["numel"])
        )
        source = source.view(item["shape"])
        non_blocking = bool(ref.tensor.is_cuda and source.is_pinned())
        ref.tensor.copy_(source, non_blocking=non_blocking)
        used_cuda = used_cuda or ref.tensor.is_cuda
        copied_tensors += 1
        copied_bytes += source.numel() * source.element_size()

    if used_cuda:
        torch.cuda.current_stream().synchronize()

    copied_scalars = 0
    for ref in scalar_refs:
        if ref.is_expert and not restore_expert:
            continue
        if ref.identity not in snapshot.scalars:
            raise RuntimeError(f"optimizer snapshot scalar missing: {ref.identity}")
        ref.state[ref.key] = snapshot.scalars[ref.identity]
        copied_scalars += 1

    return {
        "owner_rank": snapshot.owner_rank,
        "holder_rank": snapshot.holder_rank,
        "step": snapshot.step,
        "manifest_hash": snapshot.manifest_hash,
        "tensor_count": copied_tensors,
        "scalar_count": copied_scalars,
        "bytes": copied_bytes,
        "restore_scope": "all" if restore_expert else "non_expert",
    }
