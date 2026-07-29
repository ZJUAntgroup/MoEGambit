"""Typed checkpoint-relaunch directive shared by watcher and node agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Optional

from ..errors import ContractViolation

__all__ = ["RelaunchDirective"]


@dataclass(frozen=True)
class RelaunchDirective:
    directive_id: str
    recovery_epoch: int
    parent_attempt_id: str
    next_attempt_id: str
    checkpoint_locator: str
    checkpoint_step: int
    reason: str
    command_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.directive_id.startswith("sha256:"):
            raise ValueError("relaunch directive_id must be a SHA-256 digest")
        if self.recovery_epoch < 0 or self.checkpoint_step < 0:
            raise ValueError("relaunch epoch and checkpoint step must be non-negative")
        if not self.parent_attempt_id or not self.next_attempt_id:
            raise ValueError("relaunch attempt identities must be non-empty")
        if self.parent_attempt_id == self.next_attempt_id:
            raise ValueError("relaunch must create a new attempt identity")
        if not self.checkpoint_locator.strip() or not self.reason.strip():
            raise ValueError("relaunch checkpoint locator and reason are required")

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        attempt_id: str,
        recovery_epoch: int,
        checkpoint_locator: str,
        checkpoint_step: int,
        reason: str,
        command_digest: Optional[str] = None,
    ) -> "RelaunchDirective":
        next_attempt_id = f"{attempt_id}.r{int(recovery_epoch)}"
        identity = {
            "job_id": str(job_id),
            "parent_attempt_id": str(attempt_id),
            "next_attempt_id": next_attempt_id,
            "recovery_epoch": int(recovery_epoch),
            "checkpoint_locator": str(checkpoint_locator),
            "checkpoint_step": int(checkpoint_step),
            "command_digest": command_digest,
        }
        blob = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return cls(
            directive_id="sha256:" + hashlib.sha256(blob).hexdigest(),
            recovery_epoch=int(recovery_epoch),
            parent_attempt_id=str(attempt_id),
            next_attempt_id=next_attempt_id,
            checkpoint_locator=str(checkpoint_locator),
            checkpoint_step=int(checkpoint_step),
            reason=str(reason),
            command_digest=(
                None if not command_digest else str(command_digest)
            ),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RelaunchDirective":
        if not isinstance(values, Mapping):
            raise ContractViolation("checkpoint relaunch directive must be an object")
        try:
            return cls(
                directive_id=str(values["directive_id"]),
                recovery_epoch=int(values["recovery_epoch"]),
                parent_attempt_id=str(values["parent_attempt_id"]),
                next_attempt_id=str(values["next_attempt_id"]),
                checkpoint_locator=str(values["checkpoint_locator"]),
                checkpoint_step=int(values["checkpoint_step"]),
                reason=str(values["reason"]),
                command_digest=(
                    None
                    if values.get("command_digest") in (None, "")
                    else str(values["command_digest"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractViolation(
                f"checkpoint relaunch directive is invalid: {exc}"
            ) from exc
