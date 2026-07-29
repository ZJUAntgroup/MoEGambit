"""Framework-neutral recovery policy contracts.

Policies consume facts normalized by an adapter or the control service.  They
never inspect framework objects and never move payloads.  Their sole job is to
choose a deterministic recovery mode from versioned source availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..capabilities import AdapterCapabilities
from ..runtime.recovery_plan import RecoveryMode
from ..state.catalog import StateCatalog, StateSource, StateSourceKind
from ..state.version import StateVersion, newest_common_version

__all__ = [
    "ExposureEvent",
    "RecoveryFacts",
    "PolicyDecision",
    "RecoveryPolicy",
    "PeerOrCheckpointPolicy",
]


@dataclass(frozen=True)
class ExposureEvent:
    """A previously committed recovery exposure in the policy window."""

    step: int
    ranks: Tuple[int, ...]
    reason: str
    checkpoint_step: Optional[int] = None
    expert_state_count: Optional[int] = None


@dataclass(frozen=True)
class RecoveryFacts:
    """Complete standardized evidence available to a recovery policy."""

    failed_ranks: Tuple[int, ...]
    resume_step: int
    latest_checkpoint_step: int
    available_state_sources: Mapping[str, Tuple[StateSource, ...]]
    exposure_history: Sequence[ExposureEvent]
    capabilities: AdapterCapabilities
    state_catalog: Optional[StateCatalog] = None


@dataclass(frozen=True)
class PolicyDecision:
    """A deterministic mode choice plus auditable evidence."""

    mode: RecoveryMode
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RecoveryPolicy(Protocol):
    def decide(self, facts: RecoveryFacts) -> PolicyDecision: ...


def sources_of_kind(
    facts: RecoveryFacts, *kinds: StateSourceKind
) -> Mapping[str, Tuple[StateSource, ...]]:
    allowed = frozenset(kinds)
    return {
        identity: tuple(source for source in sources if source.kind in allowed)
        for identity, sources in facts.available_state_sources.items()
    }


def consistent_version(
    sources_by_identity: Mapping[str, Tuple[StateSource, ...]],
    required_step: int,
) -> Optional[StateVersion]:
    """Return one parameter/optimizer-consistent version for every identity."""

    if not sources_by_identity or any(
        not sources for sources in sources_by_identity.values()
    ):
        return None
    versions = [
        tuple(source.version for source in sources)
        for sources in sources_by_identity.values()
    ]
    exact = newest_common_version(versions)
    if exact is not None and exact.committed_step == required_step:
        return exact
    candidates = sorted(
        {
            version
            for group in versions
            for version in group
            if version.committed_step == required_step
        },
        reverse=True,
    )
    for candidate in candidates:
        if all(
            any(candidate.is_consistent_with(version) for version in group)
            for group in versions
        ):
            return candidate
    return None


def source_evidence(facts: RecoveryFacts) -> Mapping[str, Any]:
    return {
        identity: [
            {
                "kind": source.kind.value,
                "committed_step": source.version.committed_step,
                "optimizer_generation": source.version.optimizer_generation,
                "recovery_epoch": source.version.recovery_epoch,
                "locator": source.locator,
                "metadata": dict(source.metadata),
            }
            for source in sources
        ]
        for identity, sources in sorted(facts.available_state_sources.items())
    }


def version_evidence(
    version: Optional[StateVersion],
) -> Optional[Mapping[str, int]]:
    if version is None:
        return None
    return {
        "committed_step": version.committed_step,
        "optimizer_generation": version.optimizer_generation,
        "recovery_epoch": version.recovery_epoch,
    }


def checkpoint_version(facts: RecoveryFacts) -> Optional[StateVersion]:
    if facts.latest_checkpoint_step < 0:
        return None
    return consistent_version(
        sources_of_kind(facts, StateSourceKind.CHECKPOINT),
        facts.latest_checkpoint_step,
    )


class PeerOrCheckpointPolicy:
    """Prefer a complete live copy, otherwise a complete checkpoint.

    A memory replica is a live fast-path source just like a peer tensor copy,
    but only when the adapter explicitly declares optimizer replication.  The
    later source planner still validates each selected source and its version.
    """

    def decide(self, facts: RecoveryFacts) -> PolicyDecision:
        common = {
            "policy": type(self).__name__,
            "failed_ranks": list(facts.failed_ranks),
            "resume_step": facts.resume_step,
            "latest_checkpoint_step": facts.latest_checkpoint_step,
            "required_state_count": len(facts.available_state_sources),
            "capability_digest": facts.capabilities.digest(),
            "state_sources": source_evidence(facts),
        }
        if not facts.available_state_sources:
            return PolicyDecision(
                RecoveryMode.ABORT,
                "no required state inventory was provided",
                common,
            )

        live_kinds = [StateSourceKind.PEER]
        if facts.capabilities.optimizer_memory_replication:
            live_kinds.append(StateSourceKind.MEMORY_REPLICA)
        live_version = consistent_version(
            sources_of_kind(facts, *live_kinds), facts.resume_step
        )
        if live_version is not None and facts.capabilities.peer_parameter_restore:
            return PolicyDecision(
                RecoveryMode.PEER,
                "every required state identity has a consistent live source",
                dict(common, live_version=version_evidence(live_version)),
            )

        checkpoint = checkpoint_version(facts)
        if checkpoint is not None:
            return PolicyDecision(
                RecoveryMode.CHECKPOINT,
                "live state is incomplete or inconsistent; a complete checkpoint is available",
                dict(
                    common,
                    live_version=None,
                    checkpoint_version=version_evidence(checkpoint),
                    fallback_reason="live_state_unavailable_or_inconsistent",
                ),
            )
        return PolicyDecision(
            RecoveryMode.ABORT,
            "neither consistent live state nor a complete checkpoint is available",
            dict(
                common,
                live_version=None,
                checkpoint_version=None,
                fallback_reason="no_consistent_recovery_source",
            ),
        )
