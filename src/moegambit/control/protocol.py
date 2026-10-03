"""Versioned, authenticated control-plane message envelopes."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

from ..errors import ProtocolVersionMismatch

PROTOCOL_VERSION = 1

__all__ = [
    "PROTOCOL_VERSION",
    "MessageType",
    "Envelope",
    "MessageDeduplicator",
    "check_protocol_compatibility",
    "decode_message",
]


class MessageType:
    HEARTBEAT = "heartbeat"
    QUALITY_FEATURES = "quality_features"
    RECOVERY_REQUEST = "recovery_request"
    RECOVERY_ASSIGNMENT = "recovery_assignment"
    RECOVERY_COMMITTED = "recovery_committed"
    RECOVERY_FAILED = "recovery_failed"
    CHECKPOINT_RELAUNCH_REQUEST = "checkpoint_relaunch_request"
    CHECKPOINT_RELAUNCH_ACK = "checkpoint_relaunch_ack"


@dataclass
class Envelope:
    """Metadata wrapper shared by every new control RPC."""

    protocol_version: int
    message_id: str
    type: str
    job_id: str
    attempt_id: str
    recovery_epoch: int
    sender: Dict[str, Any]
    sent_at_ns: int
    payload: Dict[str, Any]
    auth: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.auth is None:
            self.auth = {}

    @classmethod
    def new(
        cls,
        message_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        job_id: str,
        attempt_id: str,
        recovery_epoch: int,
        sender: Mapping[str, Any],
    ) -> "Envelope":
        return cls(
            protocol_version=PROTOCOL_VERSION,
            message_id=str(uuid.uuid4()),
            type=str(message_type),
            job_id=str(job_id),
            attempt_id=str(attempt_id),
            recovery_epoch=int(recovery_epoch),
            sender=dict(sender),
            sent_at_ns=time.time_ns(),
            payload=dict(payload or {}),
        )

    def validate(self) -> None:
        check_protocol_compatibility(self.protocol_version)
        bounded_strings = (
            ("message_id", self.message_id, 128),
            ("type", self.type, 128),
            ("job_id", self.job_id, 256),
            ("attempt_id", self.attempt_id, 256),
        )
        for name, value, limit in bounded_strings:
            if not isinstance(value, str) or not value:
                raise ValueError(f"control {name} must be a non-empty string")
            if len(value) > limit:
                raise ValueError(f"control {name} exceeds {limit} characters")
        if not isinstance(self.recovery_epoch, int) or self.recovery_epoch < 0:
            raise ValueError("control recovery_epoch must be a non-negative integer")
        if not isinstance(self.sender, dict):
            raise ValueError("control sender must be an object")
        if not isinstance(self.sent_at_ns, int) or self.sent_at_ns < 0:
            raise ValueError("control sent_at_ns must be a non-negative integer")
        if not isinstance(self.payload, dict):
            raise ValueError("control payload must be an object")
        if not isinstance(self.auth, dict):
            raise ValueError("control auth must be an object")

    def _signing_bytes(self) -> bytes:
        values = asdict(self)
        values.pop("auth", None)
        return json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def sign(self, secret: str, *, key_id: str = "job") -> "Envelope":
        if not isinstance(secret, str) or not secret:
            raise ValueError("control signing secret must be non-empty")
        digest = hmac.new(
            secret.encode("utf-8"), self._signing_bytes(), hashlib.sha256
        ).hexdigest()
        self.auth = {
            "algorithm": "hmac-sha256",
            "key_id": str(key_id),
            "digest": digest,
        }
        return self

    def verify(
        self,
        secret: str,
        *,
        max_clock_skew_s: float = 300.0,
        now_ns: Optional[int] = None,
    ) -> None:
        if not isinstance(secret, str) or not secret:
            raise ValueError("control verification secret must be non-empty")
        if max_clock_skew_s <= 0:
            raise ValueError("max_clock_skew_s must be positive")
        auth = self.auth or {}
        if auth.get("algorithm") != "hmac-sha256":
            raise ValueError("control envelope is not HMAC-SHA256 signed")
        supplied = auth.get("digest")
        if not isinstance(supplied, str) or not supplied:
            raise ValueError("control envelope signature is missing")
        expected = hmac.new(
            secret.encode("utf-8"), self._signing_bytes(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("control envelope signature is invalid")
        current = time.time_ns() if now_ns is None else int(now_ns)
        skew_s = abs(current - self.sent_at_ns) / 1_000_000_000.0
        if skew_s > float(max_clock_skew_s):
            raise ValueError(
                "control envelope timestamp is outside the "
                f"{max_clock_skew_s}s window"
            )

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "Envelope":
        values = json.loads(data)
        if not isinstance(values, dict):
            raise ValueError("control envelope must be a JSON object")
        try:
            envelope = cls(**values)
        except TypeError as exc:
            raise ValueError(f"control envelope fields are invalid: {exc}") from exc
        envelope.validate()
        return envelope

    def response(
        self,
        payload: Mapping[str, Any],
        *,
        message_type: str = "response",
        sender: Optional[Mapping[str, Any]] = None,
    ) -> "Envelope":
        body = dict(payload)
        body["request_id"] = self.message_id
        return Envelope.new(
            message_type,
            body,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            recovery_epoch=self.recovery_epoch,
            sender=sender or {"role": "watcher"},
        )


class MessageDeduplicator:
    """Bounded, thread-safe message-id set for idempotent side effects."""

    def __init__(self, capacity: int = 10000) -> None:
        if capacity <= 0:
            raise ValueError("message deduplication capacity must be positive")
        self.capacity = int(capacity)
        self._seen = set()
        self._order: Deque[str] = deque()
        self._lock = threading.Lock()

    def accept(self, message_id: str) -> bool:
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("message_id must be a non-empty string")
        with self._lock:
            if message_id in self._seen:
                return False
            self._seen.add(message_id)
            self._order.append(message_id)
            while len(self._order) > self.capacity:
                expired = self._order.popleft()
                self._seen.discard(expired)
            return True


def decode_message(data: str) -> Tuple[Dict[str, Any], Optional[Envelope]]:
    """Decode a v1 envelope or one legacy flat JSON object."""

    values = json.loads(data)
    if not isinstance(values, dict):
        raise ValueError("control message must be a JSON object")
    if "protocol_version" not in values:
        return values, None
    try:
        envelope = Envelope(**values)
    except TypeError as exc:
        raise ValueError(f"control envelope fields are invalid: {exc}") from exc
    envelope.validate()
    payload = dict(envelope.payload)
    payload.setdefault("type", envelope.type)
    payload.setdefault("recovery_epoch", envelope.recovery_epoch)
    for field in ("node_rank", "global_rank", "local_rank"):
        if field in envelope.sender:
            payload.setdefault(field, envelope.sender[field])
    return payload, envelope


def check_protocol_compatibility(protocol_version: int) -> None:
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolVersionMismatch(
            "incompatible control protocol major version: "
            f"local={PROTOCOL_VERSION}, peer={protocol_version}"
        )
