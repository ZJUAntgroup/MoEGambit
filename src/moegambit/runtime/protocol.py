"""Versioned JSON-line control-plane protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class WireMessage:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def encode(self) -> bytes:
        return (
            json.dumps(
                {
                    "version": self.version,
                    "kind": self.kind,
                    "payload": dict(self.payload),
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes | str) -> "WireMessage":
        value = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
        version = int(value.get("version", -1))
        if version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
            )
        return cls(
            kind=str(value["kind"]),
            payload=dict(value.get("payload") or {}),
            version=version,
        )
