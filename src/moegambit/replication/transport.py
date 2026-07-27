"""Framework-neutral framing helpers for host-memory replication.

The transport deliberately does not use c10d/NCCL.  Optimizer replication can
therefore progress independently without changing a framework's collective
creation or launch order.  Payload allocation remains the caller's concern;
this module only provides bounded framing, exact I/O, retry and integrity
primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import socket
import struct
import time
from typing import Any, Mapping, Optional

__all__ = [
    "FrameLimits",
    "TransportIntegrityError",
    "canonical_json",
    "connect_with_retry",
    "payload_digest",
    "receive_exact",
    "receive_into",
    "receive_json",
    "send_json",
]


class TransportIntegrityError(RuntimeError):
    """A received frame violates size, schema or checksum constraints."""


@dataclass(frozen=True)
class FrameLimits:
    max_header_bytes: int = 4 * 1024 * 1024
    max_payload_bytes: int = 64 * 1024 * 1024 * 1024
    max_segments: int = 1024

    def __post_init__(self) -> None:
        if self.max_header_bytes <= 0:
            raise ValueError("max_header_bytes must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if self.max_segments <= 0:
            raise ValueError("max_segments must be positive")


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransportIntegrityError(
            f"replication header is not canonical JSON: {exc}"
        ) from exc


def send_json(
    sock: socket.socket,
    payload: Mapping[str, Any],
    *,
    limits: Optional[FrameLimits] = None,
) -> None:
    limits = limits or FrameLimits()
    data = canonical_json(dict(payload))
    if len(data) > limits.max_header_bytes:
        raise TransportIntegrityError(
            f"replication header exceeds {limits.max_header_bytes} bytes"
        )
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def receive_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise TransportIntegrityError("negative receive size")
    data = bytearray(size)
    receive_into(sock, memoryview(data))
    return bytes(data)


def receive_into(sock: socket.socket, destination: memoryview) -> None:
    view = destination.cast("B")
    offset = 0
    while offset < len(view):
        received = sock.recv_into(view[offset:], len(view) - offset)
        if received <= 0:
            raise ConnectionError("replication socket closed during a frame")
        offset += received


def receive_json(
    sock: socket.socket,
    *,
    limits: Optional[FrameLimits] = None,
) -> Mapping[str, Any]:
    limits = limits or FrameLimits()
    (size,) = struct.unpack("!Q", receive_exact(sock, 8))
    if size <= 0 or size > limits.max_header_bytes:
        raise TransportIntegrityError(
            f"replication header size {size} is outside the allowed range"
        )
    try:
        decoded = json.loads(receive_exact(sock, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportIntegrityError("replication header is invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise TransportIntegrityError("replication header must be an object")
    return decoded


def payload_digest(payload: Any) -> str:
    try:
        view = memoryview(payload).cast("B")
    except TypeError as exc:
        raise TypeError("replication payload must expose the buffer protocol") from exc
    return "sha256:" + hashlib.sha256(view).hexdigest()


def connect_with_retry(
    address: tuple[str, int],
    *,
    timeout: float,
    retry_interval: float = 0.2,
) -> socket.socket:
    if timeout <= 0:
        raise ValueError("replication connection timeout must be positive")
    if retry_interval <= 0:
        raise ValueError("replication retry interval must be positive")
    deadline = time.monotonic() + float(timeout)
    last_error: Optional[OSError] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectionError(
                "timed out connecting to replication endpoint "
                f"{address[0]}:{address[1]}: {last_error}"
            ) from last_error
        try:
            return socket.create_connection(
                address,
                timeout=min(10.0, remaining),
            )
        except OSError as exc:
            last_error = exc
            time.sleep(min(float(retry_interval), remaining))
