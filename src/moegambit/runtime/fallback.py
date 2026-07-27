"""Explicit checkpoint-relaunch fallback control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .client import ControlClient

__all__ = [
    "FallbackRequest",
    "FallbackController",
    "ControlPlaneFallbackController",
]


@dataclass(frozen=True)
class FallbackRequest:
    recovery_epoch: int
    at_step: int
    reason: str
    error_type: str
    evidence: Mapping[str, Any]


@runtime_checkable
class FallbackController(Protocol):
    def request_checkpoint_relaunch(self, request: FallbackRequest) -> bool: ...


class ControlPlaneFallbackController:
    def __init__(self, client: ControlClient) -> None:
        self.client = client

    def request_checkpoint_relaunch(self, request: FallbackRequest) -> bool:
        response = self.client.notify(
            "checkpoint_relaunch_request",
            {
                "at_step": request.at_step,
                "reason": request.reason,
                "error_type": request.error_type,
                "evidence": dict(request.evidence),
            },
            recovery_epoch=request.recovery_epoch,
        )
        return bool(response.get("ok", False))
