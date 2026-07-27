"""Structured, framework-neutral recovery observability."""

from __future__ import annotations

import json
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

__all__ = [
    "RECOVERY_PHASES",
    "RecoveryOutcome",
    "RecoveryRecord",
    "PhaseLatencyTimer",
]

RECOVERY_PHASES = (
    "fault_detection",
    "quiesce",
    "group_rebuild",
    "state_restore",
    "first_forward",
    "first_full_step",
)


class RecoveryOutcome(Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    FALLBACK = "fallback"
    ABORTED = "aborted"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class RecoveryRecord:
    job_id: str = ""
    attempt_id: str = ""
    recovery_epoch: int = 0
    failed_ranks: Tuple[int, ...] = ()
    failure_class: str = ""
    adapter: str = ""
    adapter_version: str = ""
    topology_manifest: str = ""
    resume_step: int = -1
    checkpoint_step: int = -1
    decision: Any = ""
    policy_evidence: MutableMapping[str, Any] = field(default_factory=dict)
    state_sources: MutableMapping[str, Any] = field(default_factory=dict)
    phase_latency_ms: MutableMapping[str, float] = field(default_factory=dict)
    validation: MutableMapping[str, Any] = field(default_factory=dict)
    result: RecoveryOutcome = RecoveryOutcome.PROVISIONAL

    def record_fallback(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fallback reason must be a non-empty string")
        self.result = RecoveryOutcome.FALLBACK
        self.validation["fallback_reason"] = reason

    def measure_phase(
        self,
        phase: str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "PhaseLatencyTimer":
        return PhaseLatencyTimer(self.phase_latency_ms, phase, clock=clock)

    def to_dict(self) -> Mapping[str, Any]:
        return _json_safe(
            {
                "job_id": self.job_id,
                "attempt_id": self.attempt_id,
                "recovery_epoch": self.recovery_epoch,
                "failed_ranks": self.failed_ranks,
                "failure_class": self.failure_class,
                "adapter": self.adapter,
                "adapter_version": self.adapter_version,
                "topology_manifest": self.topology_manifest,
                "resume_step": self.resume_step,
                "checkpoint_step": self.checkpoint_step,
                "decision": self.decision,
                "policy_evidence": self.policy_evidence,
                "state_sources": self.state_sources,
                "phase_latency_ms": self.phase_latency_ms,
                "validation": self.validation,
                "result": self.result,
            }
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )


class PhaseLatencyTimer(AbstractContextManager):
    def __init__(
        self,
        phase_latency_ms: MutableMapping[str, float],
        phase: str,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a non-empty string")
        self._latencies = phase_latency_ms
        self._phase = phase
        self._clock = clock
        self._started_at: Optional[float] = None

    def __enter__(self) -> "PhaseLatencyTimer":
        if self._started_at is not None:
            raise RuntimeError("phase timer cannot be entered more than once")
        self._started_at = self._clock()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._started_at is None:
            raise RuntimeError("phase timer exited before it was entered")
        elapsed_ms = max(0.0, (self._clock() - self._started_at) * 1000.0)
        self._latencies[self._phase] = (
            float(self._latencies.get(self._phase, 0.0)) + elapsed_ms
        )
        return False
