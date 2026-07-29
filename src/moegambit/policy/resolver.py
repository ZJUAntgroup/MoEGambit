"""Deterministic planning and runtime resolution of state sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..adapters.base import StateSources
from ..errors import RecoveryRejected, StateUnavailable
from ..runtime.recovery_plan import RecoveryMode, RecoveryPlan
from ..state.catalog import Placement, StateCatalog, StateSource, StateSourceKind
from .base import PolicyDecision, RecoveryFacts, consistent_version

__all__ = [
    "SourceQuery",
    "StateSourceCandidateProvider",
    "RecoveryEvidenceProvider",
    "StateSourcePlanner",
    "DeterministicStateSourcePlanner",
    "StateSourceResolver",
    "MappingStateSourceResolver",
    "serialize_source_candidates",
    "parse_source_candidates",
]


@dataclass(frozen=True)
class SourceQuery:
    failed_ranks: Tuple[int, ...]
    resume_step: int
    world_size: int
    state_catalog: StateCatalog


@runtime_checkable
class StateSourceCandidateProvider(Protocol):
    def source_candidates(
        self, query: SourceQuery
    ) -> Mapping[str, Sequence[StateSource]]: ...


@runtime_checkable
class RecoveryEvidenceProvider(Protocol):
    """Optional adapter hook for policy facts beyond source availability."""

    def recovery_exposure_history(
        self, query: SourceQuery
    ) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class StateSourcePlanner(Protocol):
    def select(
        self, facts: RecoveryFacts, decision: PolicyDecision
    ) -> Mapping[str, StateSource]: ...


def _placement(source: StateSource) -> Optional[Placement]:
    raw = source.metadata.get("placement")
    if isinstance(raw, Placement):
        return raw
    try:
        return Placement(str(raw))
    except (TypeError, ValueError):
        return None


def _source_dict(source: StateSource) -> Mapping[str, Any]:
    return {
        "kind": source.kind.value,
        "version": {
            "committed_step": source.version.committed_step,
            "optimizer_generation": source.version.optimizer_generation,
            "recovery_epoch": source.version.recovery_epoch,
        },
        "locator": source.locator,
        "metadata": dict(source.metadata),
    }


def serialize_source_candidates(
    candidates: Mapping[str, Sequence[StateSource]],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    """Return the canonical JSON control-plane representation."""

    return {
        str(identity): [
            _source_dict(source)
            for source in sorted(
                sources,
                key=lambda item: (
                    item.kind.value,
                    item.version,
                    item.locator,
                ),
            )
        ]
        for identity, sources in sorted(candidates.items())
    }


def parse_source_candidates(
    values: Mapping[str, Any],
) -> Mapping[str, Tuple[StateSource, ...]]:
    """Parse and validate a serialized source inventory."""

    from ..state.version import StateVersion

    if not isinstance(values, Mapping):
        raise RecoveryRejected("available_state_sources must be a mapping")
    parsed: Dict[str, Tuple[StateSource, ...]] = {}
    for identity, raw_sources in sorted(values.items()):
        if not isinstance(identity, str) or not identity:
            raise RecoveryRejected("source identity must be a non-empty string")
        if isinstance(raw_sources, (str, bytes)) or not isinstance(
            raw_sources, (list, tuple)
        ):
            raise RecoveryRejected(
                f"source candidates for {identity!r} must be a sequence"
            )
        sources = []
        for raw in raw_sources:
            if not isinstance(raw, Mapping):
                raise RecoveryRejected(
                    f"source candidate for {identity!r} must be an object"
                )
            version = raw.get("version")
            if not isinstance(version, Mapping):
                raise RecoveryRejected(
                    f"source candidate for {identity!r} lacks a version"
                )
            try:
                sources.append(
                    StateSource(
                        kind=StateSourceKind(str(raw["kind"])),
                        version=StateVersion(
                            committed_step=int(version["committed_step"]),
                            optimizer_generation=int(
                                version.get("optimizer_generation", 0)
                            ),
                            recovery_epoch=int(version.get("recovery_epoch", 0)),
                        ),
                        locator=str(raw.get("locator", "")),
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RecoveryRejected(
                    f"invalid source candidate for {identity!r}: {exc}"
                ) from exc
        parsed[identity] = tuple(sources)
    return parsed


def _stable_choice(
    sources: Sequence[StateSource],
    *,
    version: Any,
    kinds: Sequence[StateSourceKind],
) -> StateSource | None:
    priority = {kind: index for index, kind in enumerate(kinds)}
    compatible = [
        source
        for source in sources
        if source.kind in priority and source.version.is_consistent_with(version)
    ]
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda source: (
            priority[source.kind],
            source.locator,
            source.version.recovery_epoch,
        ),
    )


class DeterministicStateSourcePlanner:
    """Choose one digest-safe source for every required state identity."""

    def select(
        self, facts: RecoveryFacts, decision: PolicyDecision
    ) -> Mapping[str, StateSource]:
        if decision.mode is RecoveryMode.ABORT:
            return {}
        sources = facts.available_state_sources
        if not sources:
            raise RecoveryRejected("policy selected recovery without state sources")

        if decision.mode is RecoveryMode.CHECKPOINT:
            version = consistent_version(
                {
                    identity: tuple(
                        source
                        for source in candidates
                        if source.kind is StateSourceKind.CHECKPOINT
                    )
                    for identity, candidates in sources.items()
                },
                facts.latest_checkpoint_step,
            )
            kinds = (StateSourceKind.CHECKPOINT,)
        elif decision.mode is RecoveryMode.PEER:
            kinds = (
                (
                    StateSourceKind.MEMORY_REPLICA,
                    StateSourceKind.PEER,
                    StateSourceKind.LOCAL,
                )
                if facts.capabilities.optimizer_memory_replication
                else (StateSourceKind.PEER, StateSourceKind.LOCAL)
            )
            version = consistent_version(
                {
                    identity: tuple(
                        source for source in candidates if source.kind in kinds
                    )
                    for identity, candidates in sources.items()
                },
                facts.resume_step,
            )
        elif decision.mode is RecoveryMode.HYBRID:
            version = None
            kinds = ()
        else:
            raise RecoveryRejected(f"unsupported recovery mode: {decision.mode.value}")

        selected: Dict[str, StateSource] = {}
        if decision.mode is not RecoveryMode.HYBRID:
            if version is None:
                raise RecoveryRejected(
                    f"policy selected {decision.mode.value} without one consistent version"
                )
            for identity, candidates in sorted(sources.items()):
                source = _stable_choice(candidates, version=version, kinds=kinds)
                if source is None:
                    raise RecoveryRejected(
                        f"no {decision.mode.value} source covers {identity!r}"
                    )
                selected[identity] = source
            return selected

        refs = (
            {ref.identity: ref for ref in facts.state_catalog}
            if facts.state_catalog is not None
            else {}
        )
        placements: Dict[str, Placement] = {}
        for identity, candidates in sources.items():
            if identity in refs:
                placements[identity] = refs[identity].placement
                continue
            classified = {_placement(source) for source in candidates}
            classified.discard(None)
            if len(classified) != 1:
                raise RecoveryRejected(
                    f"hybrid placement is missing or conflicting for {identity!r}"
                )
            placement = next(iter(classified))
            assert isinstance(placement, Placement)
            placements[identity] = placement
        live_kinds = (
            (StateSourceKind.MEMORY_REPLICA, StateSourceKind.PEER)
            if facts.capabilities.optimizer_memory_replication
            else (StateSourceKind.PEER,)
        )
        live_version = consistent_version(
            {
                identity: tuple(
                    source
                    for source in sources[identity]
                    if source.kind
                    in live_kinds
                )
                for identity, placement in placements.items()
                if placement is not Placement.UNIQUE
            },
            facts.resume_step,
        )
        checkpoint = consistent_version(
            {
                identity: tuple(
                    source
                    for source in sources[identity]
                    if source.kind is StateSourceKind.CHECKPOINT
                )
                for identity, placement in placements.items()
                if placement is Placement.UNIQUE
            },
            facts.latest_checkpoint_step,
        )
        if live_version is None or checkpoint is None:
            raise RecoveryRejected("hybrid source groups are version-inconsistent")
        for identity, placement in sorted(placements.items()):
            if placement is Placement.UNIQUE:
                source = _stable_choice(
                    sources[identity],
                    version=checkpoint,
                    kinds=(StateSourceKind.CHECKPOINT,),
                )
            else:
                source = _stable_choice(
                    sources[identity],
                    version=live_version,
                    kinds=live_kinds,
                )
            if source is None:
                raise RecoveryRejected(f"hybrid source is missing for {identity!r}")
            selected[identity] = source
        return selected


@runtime_checkable
class StateSourceResolver(Protocol):
    def resolve(self, plan: RecoveryPlan) -> StateSources: ...


class MappingStateSourceResolver:
    """Resolve stable locators with explicitly registered loader functions."""

    def __init__(self, loaders: Mapping[str, Callable[[StateSource], Any]]) -> None:
        self._loaders = dict(loaders)

    def resolve(self, plan: RecoveryPlan) -> StateSources:
        resolved: Dict[str, Any] = {}
        for identity, source in sorted(plan.state_sources.items()):
            if not isinstance(source, StateSource):
                raise StateUnavailable(
                    f"state source {identity!r} is not a typed descriptor"
                )
            loader = self._loaders.get(source.locator)
            if loader is None:
                raise StateUnavailable(
                    f"no loader is registered for {identity!r} at {source.locator!r}"
                )
            resolved[identity] = loader(source)
        return StateSources(resolved)
