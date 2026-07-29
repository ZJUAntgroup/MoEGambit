"""The single authoritative description of one recovery attempt.

Design doc section 7.4.  A ``RecoveryPlan`` is produced by the control plane,
hashed, and broadcast to every participant.  Adapters *execute* plans; they
never derive their own.  This is the mechanism that prevents the classic
split-brain failure where each rank independently concludes a different
recovery mode and the job deadlocks in a collective.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple

from ..state.catalog import StateSource
from ..state.version import StateVersion

__all__ = ["RecoveryMode", "WorkerEndpoint", "RecoveryPlan"]


_NON_DECISION_SOURCE_KEYS = frozenset(
    {
        "evidence",
        "tensor",
        "value",
        "payload",
        "callable",
    }
)


def _canonical_state_source(value: Any) -> Any:
    """Return the decision-bearing, JSON-safe form of a state source.

    Recovery source selection is part of the plan, not diagnostic metadata:
    choosing rank 0 instead of rank 7 (or a checkpoint instead of a peer) must
    change the plan digest.  Runtime payloads such as tensors and callables are
    deliberately excluded because their bytes/addresses are not stable plan
    identity.  A source that contains *only* an opaque payload is rejected
    rather than silently hashing as an empty descriptor.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, StateVersion):
        return {
            "committed_step": value.committed_step,
            "optimizer_generation": value.optimizer_generation,
            "recovery_epoch": value.recovery_epoch,
        }
    if isinstance(value, StateSource):
        return {
            "kind": value.kind.value,
            "version": _canonical_state_source(value.version),
            "locator": value.locator,
            "metadata": _canonical_state_source(value.metadata),
        }
    if isinstance(value, Mapping):
        descriptor = {
            str(key): _canonical_state_source(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _NON_DECISION_SOURCE_KEYS
        }
        if not descriptor and value:
            raise TypeError(
                "state source must contain a stable descriptor in addition to "
                "runtime tensor/callable payloads"
            )
        return descriptor
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonical_state_source(item) for item in value]
    raise TypeError(
        f"state source descriptor {type(value).__name__!r} is not canonicalizable"
    )


class RecoveryMode(Enum):
    """Recovery strategies, ordered from cheapest to most disruptive."""

    PEER = "peer"
    HYBRID = "hybrid"
    CHECKPOINT = "checkpoint"
    ABORT = "abort"


@dataclass(frozen=True)
class WorkerEndpoint:
    """Where a replacement worker will run."""

    host: str
    port: int
    node_rank: int
    local_rank: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "node_rank": self.node_rank,
            "local_rank": self.local_rank,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "WorkerEndpoint":
        return cls(
            host=str(values["host"]),
            port=int(values["port"]),
            node_rank=int(values["node_rank"]),
            local_rank=int(values["local_rank"]),
        )


@dataclass(frozen=True)
class RecoveryPlan:
    """An immutable, hashable instruction set for one recovery epoch."""

    protocol_version: int
    recovery_epoch: int
    failed_ranks: Tuple[int, ...]
    resume_step: int
    mode: RecoveryMode
    replacements: Mapping[int, WorkerEndpoint] = field(default_factory=dict)
    topology_generation: int = 0
    group_manifest_hash: str = ""
    state_sources: Mapping[str, Any] = field(default_factory=dict)
    policy_evidence: Mapping[str, Any] = field(default_factory=dict)
    timeout_budget_s: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_fast_path(self) -> bool:
        return self.mode in (RecoveryMode.PEER, RecoveryMode.HYBRID)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "recovery_epoch": self.recovery_epoch,
            "failed_ranks": list(self.failed_ranks),
            "resume_step": self.resume_step,
            "mode": self.mode.value,
            "replacements": {
                str(k): dict(v.as_dict()) for k, v in sorted(self.replacements.items())
            },
            "topology_generation": self.topology_generation,
            "group_manifest_hash": self.group_manifest_hash,
            "state_sources": {
                str(identity): _canonical_state_source(source)
                for identity, source in sorted(self.state_sources.items())
            },
            "policy_evidence": dict(self.policy_evidence),
            "timeout_budget_s": dict(sorted(self.timeout_budget_s.items())),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RecoveryPlan":
        """Parse the stable control-plane representation of a plan."""

        if not isinstance(values, Mapping):
            raise TypeError("recovery plan must be a mapping")
        replacements = {
            int(rank): WorkerEndpoint.from_dict(endpoint)
            for rank, endpoint in dict(values.get("replacements", {})).items()
        }
        sources = {}
        for identity, raw_source in dict(values.get("state_sources", {})).items():
            if isinstance(raw_source, StateSource):
                sources[str(identity)] = raw_source
                continue
            if not isinstance(raw_source, Mapping) or "kind" not in raw_source:
                sources[str(identity)] = raw_source
                continue
            raw_version = raw_source.get("version", {})
            if isinstance(raw_version, StateVersion):
                version = raw_version
            elif isinstance(raw_version, Mapping):
                version = StateVersion(
                    committed_step=int(raw_version.get("committed_step", -1)),
                    optimizer_generation=int(
                        raw_version.get("optimizer_generation", 0)
                    ),
                    recovery_epoch=int(raw_version.get("recovery_epoch", 0)),
                )
            else:
                raise TypeError("state source version must be a mapping")
            from ..state.catalog import StateSourceKind

            sources[str(identity)] = StateSource(
                kind=StateSourceKind(str(raw_source["kind"])),
                version=version,
                locator=str(raw_source.get("locator", "")),
                metadata=dict(raw_source.get("metadata", {})),
            )
        mode = values.get("mode", RecoveryMode.ABORT.value)
        if not isinstance(mode, RecoveryMode):
            mode = RecoveryMode(str(mode))
        return cls(
            protocol_version=int(values["protocol_version"]),
            recovery_epoch=int(values["recovery_epoch"]),
            failed_ranks=tuple(int(rank) for rank in values.get("failed_ranks", ())),
            resume_step=int(values["resume_step"]),
            mode=mode,
            replacements=replacements,
            topology_generation=int(values.get("topology_generation", 0)),
            group_manifest_hash=str(values.get("group_manifest_hash", "")),
            state_sources=sources,
            policy_evidence=dict(values.get("policy_evidence", {})),
            timeout_budget_s={
                str(name): float(timeout)
                for name, timeout in dict(values.get("timeout_budget_s", {})).items()
            },
        )

    def digest(self) -> str:
        """Hash of the *decision-bearing* fields.

        ``policy_evidence`` is diagnostic and excluded.  ``state_sources`` is
        canonicalized and included because it decides which peer/checkpoint a
        rank will execute; excluding it would allow split-brain plans to share
        the same digest.
        """
        decision = dict(self.as_dict())
        decision.pop("policy_evidence", None)
        blob = json.dumps(decision, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def timeout_for(self, phase: str, default: float = 60.0) -> float:
        return float(self.timeout_budget_s.get(phase, default))
