"""Bounded newline-delimited control server for RecoveryCoordinatorService."""

from __future__ import annotations

import hashlib
import socketserver
import threading
from collections import deque
from typing import Deque, Dict, Mapping, Optional, Tuple

from ..errors import ContractViolation, RecoveryRejected
from .protocol import Envelope, MessageType, decode_message
from .service import RecoveryCoordinatorService

__all__ = ["ControlRequestProcessor", "ControlServer"]


class _ResponseCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("response cache capacity must be positive")
        self.capacity = int(capacity)
        self._values: Dict[Tuple[str, str, str], Tuple[str, bytes]] = {}
        self._order: Deque[Tuple[str, str, str]] = deque()
        self._lock = threading.Lock()

    def lookup(
        self,
        key: Tuple[str, str, str],
        request_digest: str,
    ) -> Optional[bytes]:
        with self._lock:
            cached = self._values.get(key)
            if cached is None:
                return None
            if cached[0] != request_digest:
                raise ContractViolation(
                    "message_id was reused with different request contents"
                )
            return cached[1]

    def store(
        self,
        key: Tuple[str, str, str],
        request_digest: str,
        response: bytes,
    ) -> None:
        with self._lock:
            if key not in self._values:
                self._order.append(key)
            self._values[key] = (request_digest, response)
            while len(self._order) > self.capacity:
                expired = self._order.popleft()
                self._values.pop(expired, None)


class ControlRequestProcessor:
    """Authenticate, de-duplicate and dispatch one versioned request."""

    def __init__(
        self,
        service: RecoveryCoordinatorService,
        *,
        job_token: Optional[str] = None,
        require_token: bool = False,
        max_message_bytes: int = 1 << 20,
        max_clock_skew_s: float = 300.0,
        dedup_capacity: int = 10000,
    ) -> None:
        if require_token and not job_token:
            raise ValueError("require_token needs a non-empty job_token")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if max_clock_skew_s <= 0:
            raise ValueError("max_clock_skew_s must be positive")
        self.service = service
        self.job_token = job_token
        self.require_token = bool(require_token)
        self.max_message_bytes = int(max_message_bytes)
        self.max_clock_skew_s = float(max_clock_skew_s)
        self._responses = _ResponseCache(dedup_capacity)

    def _authenticate(self, envelope: Envelope) -> None:
        signed = bool(envelope.auth)
        if self.require_token and not signed:
            raise ContractViolation("control request is unsigned")
        if signed:
            if not self.job_token:
                raise ContractViolation(
                    "control request is signed but no verification token is configured"
                )
            try:
                envelope.verify(
                    self.job_token,
                    max_clock_skew_s=self.max_clock_skew_s,
                )
            except ValueError as exc:
                raise ContractViolation(
                    f"control request authentication failed: {exc}"
                ) from exc

    @staticmethod
    def _rank(envelope: Envelope) -> int:
        try:
            return int(envelope.sender.get("global_rank", -1))
        except (TypeError, ValueError):
            return -1

    def _dispatch(self, envelope: Envelope) -> Mapping[str, object]:
        common = {
            "job_id": envelope.job_id,
            "attempt_id": envelope.attempt_id,
        }
        if envelope.type == MessageType.RECOVERY_REQUEST:
            return self.service.prepare(envelope.payload, **common)
        if envelope.type == MessageType.RECOVERY_ASSIGNMENT:
            return self.service.get_assignment(
                envelope.payload,
                recovery_epoch=envelope.recovery_epoch,
                **common,
            )
        if envelope.type == MessageType.RECOVERY_COMMITTED:
            return self.service.committed(
                envelope.payload,
                rank=self._rank(envelope),
                recovery_epoch=envelope.recovery_epoch,
                **common,
            )
        if envelope.type == MessageType.RECOVERY_FAILED:
            return self.service.failed(
                envelope.payload,
                rank=self._rank(envelope),
                recovery_epoch=envelope.recovery_epoch,
                **common,
            )
        if envelope.type == MessageType.CHECKPOINT_RELAUNCH_REQUEST:
            return self.service.request_checkpoint_relaunch(
                envelope.payload,
                rank=self._rank(envelope),
                recovery_epoch=envelope.recovery_epoch,
                **common,
            )
        if envelope.type == MessageType.HEARTBEAT:
            return self.service.heartbeat(**common)
        raise RecoveryRejected(f"unsupported control message type {envelope.type!r}")

    def process(self, raw: bytes) -> bytes:
        if not isinstance(raw, bytes):
            raise TypeError("control request must be bytes")
        if len(raw) > self.max_message_bytes:
            raise ValueError("control request exceeds max_message_bytes")
        line = raw[:-1] if raw.endswith(b"\n") else raw
        if b"\n" in line or b"\r" in line:
            raise ValueError("control request must contain exactly one JSON line")
        try:
            decoded = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("control request must be UTF-8") from exc
        _payload, envelope = decode_message(decoded)
        if envelope is None:
            raise ContractViolation("new control service requires a v1 envelope")
        self._authenticate(envelope)
        canonical = envelope.to_json().encode("utf-8")
        request_digest = hashlib.sha256(canonical).hexdigest()
        key = (envelope.job_id, envelope.attempt_id, envelope.message_id)
        cached = self._responses.lookup(key, request_digest)
        if cached is not None:
            return cached
        try:
            payload = dict(self._dispatch(envelope))
        except Exception as exc:
            payload = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        response = envelope.response(payload)
        if self.job_token:
            response.sign(self.job_token)
        encoded = (response.to_json() + "\n").encode("utf-8")
        if len(encoded) > self.max_message_bytes:
            raise ValueError("control response exceeds max_message_bytes")
        self._responses.store(key, request_digest, encoded)
        return encoded


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ControlServer:
    """Small TCP host around :class:`ControlRequestProcessor`."""

    def __init__(
        self,
        host: str,
        port: int,
        processor: ControlRequestProcessor,
    ) -> None:
        if not host or host.strip() in ("0.0.0.0", "::"):
            raise ValueError("control server requires an explicit bind interface")
        if not 0 <= int(port) <= 65535:
            raise ValueError("control server port is out of range")
        self.processor = processor
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                maximum = outer.processor.max_message_bytes
                raw = self.rfile.readline(maximum + 2)
                if not raw:
                    return
                if len(raw) > maximum:
                    return
                try:
                    response = outer.processor.process(raw)
                except Exception:
                    return
                self.wfile.write(response)

        self._server = _ThreadingTCPServer((host, int(port)), Handler)
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> "ControlServer":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="moegambit-control",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "ControlServer":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False
