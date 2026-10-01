"""MoE-aware hybrid recovery policy.

Current replicated/sharded state is restored from peers or memory replicas;
unique expert state is restored from a complete checkpoint only while the
resulting expert-staleness exposure remains inside an auditable budget.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ..capabilities import AdapterCapabilities
from ..runtime.recovery_plan import RecoveryMode
from ..state.catalog import Placement, StateSource, StateSourceKind
from ..state.version import StateVersion
from .base import (
    PolicyDecision,
    RecoveryFacts,
    checkpoint_version,
    consistent_version,
    source_evidence,
    sources_of_kind,
    version_evidence,
)

__all__ = ["MoeHybridPolicy"]


def _classification(
    facts: RecoveryFacts,
) -> Optional[Mapping[str, Tuple[Placement, bool]]]:
    if facts.state_catalog is not None:
        refs = {ref.identity: ref for ref in facts.state_catalog}
        if set(refs) != set(facts.available_state_sources):
            return None
        return {
            identity: (ref.placement, ref.is_expert)
            for identity, ref in refs.items()
        }
    result: Dict[str, Tuple[Placement, bool]] = {}
    for identity, sources in facts.available_state_sources.items():
        placements = set()
        expert = set()
        for source in sources:
            raw_placement = source.metadata.get("placement")
            try:
                placements.add(
                    raw_placement
                    if isinstance(raw_placement, Placement)
                    else Placement(str(raw_placement))
                )
            except (TypeError, ValueError):
                pass
            tags = source.metadata.get("tags", ())
            if isinstance(tags, (list, tuple, set, frozenset)):
                expert.add("expert" in tags)
            elif "is_expert" in source.metadata:
                expert.add(bool(source.metadata["is_expert"]))
        if len(placements) != 1 or len(expert) != 1:
            return None
        result[identity] = (next(iter(placements)), next(iter(expert)))
    return result


def _only(
    facts: RecoveryFacts,
    identities: Tuple[str, ...],
    *kinds: StateSourceKind,
) -> Mapping[str, Tuple[StateSource, ...]]:
    allowed = frozenset(kinds)
    return {
        identity: tuple(
            source
            for source in facts.available_state_sources[identity]
            if source.kind in allowed
        )
        for identity in identities
    }


class MoeHybridPolicy:
    DEFAULT_MAX_EXPERT_STALENESS_DENSITY = 0.1
    DEFAULT_EXPOSURE_WINDOW_STEPS = 2_000

    def __init__(
        self,
        max_expert_staleness_density: float = DEFAULT_MAX_EXPERT_STALENESS_DENSITY,
        exposure_window_steps: int = DEFAULT_EXPOSURE_WINDOW_STEPS,
    ) -> None:
        if max_expert_staleness_density < 0:
            raise ValueError("max_expert_staleness_density must be non-negative")
        if exposure_window_steps <= 0:
            raise ValueError("exposure_window_steps must be positive")
        self.max_expert_staleness_density = float(
            max_expert_staleness_density
        )
        self.exposure_window_steps = int(exposure_window_steps)

    @staticmethod
    def _fallback(
        checkpoint: Optional[StateVersion],
        reason: str,
        evidence: Mapping[str, Any],
        code: str,
    ) -> PolicyDecision:
        details = dict(evidence)
        details["fallback_reason"] = code
        details["checkpoint_version"] = version_evidence(checkpoint)
        if checkpoint is not None:
            return PolicyDecision(RecoveryMode.CHECKPOINT, reason, details)
        details["abort_reason"] = "complete_checkpoint_unavailable"
        return PolicyDecision(
            RecoveryMode.ABORT,
            reason + "; no complete checkpoint is available",
            details,
        )

    def decide(self, facts: RecoveryFacts) -> PolicyDecision:
        evidence: Dict[str, Any] = {
            "policy": type(self).__name__,
            "failed_ranks": list(facts.failed_ranks),
            "resume_step": facts.resume_step,
            "latest_checkpoint_step": facts.latest_checkpoint_step,
            "state_sources": source_evidence(facts),
            **self._policy_evidence(),
        }
        checkpoint = checkpoint_version(facts)
        if not facts.available_state_sources:
            return self._fallback(
                checkpoint,
                "no required state inventory was provided",
                evidence,
                "missing_state_inventory",
            )

        live_kinds = [StateSourceKind.PEER]
        if facts.capabilities.optimizer_memory_replication:
            live_kinds.append(StateSourceKind.MEMORY_REPLICA)
        live = consistent_version(
            sources_of_kind(facts, *live_kinds), facts.resume_step
        )
        if live is not None and facts.capabilities.peer_parameter_restore:
            evidence["live_version"] = version_evidence(live)
            return PolicyDecision(
                RecoveryMode.PEER,
                "every required identity has a consistent current live source",
                evidence,
            )

        classified = _classification(facts)
        if classified is None:
            return self._fallback(
                checkpoint,
                "state placement/expert classification is incomplete",
                evidence,
                "state_classification_incomplete",
            )
        unique = tuple(
            sorted(
                identity
                for identity, (placement, _expert) in classified.items()
                if placement is Placement.UNIQUE
            )
        )
        non_unique = tuple(sorted(set(classified) - set(unique)))
        nonexpert_unique = [
            identity for identity in unique if not classified[identity][1]
        ]
        if nonexpert_unique:
            evidence["nonexpert_unique_identities"] = nonexpert_unique
            return self._fallback(
                checkpoint,
                "unique non-expert state cannot use the MoE hybrid path",
                evidence,
                "nonexpert_unique_state",
            )
        if not unique:
            return self._fallback(
                checkpoint,
                "full live restore is unavailable and no unique expert state requires hybrid recovery",
                evidence,
                "live_state_unavailable_or_inconsistent",
            )
        required = AdapterCapabilities(
            peer_parameter_restore=True,
            moe_state_classification=True,
        )
        missing = facts.capabilities.missing(required)
        if missing:
            evidence["missing_hybrid_capabilities"] = list(missing)
            return self._fallback(
                checkpoint,
                "adapter lacks hybrid capabilities: " + ", ".join(missing),
                evidence,
                "adapter_missing_hybrid_capabilities",
            )

        dense = consistent_version(
            _only(facts, non_unique, *live_kinds), facts.resume_step
        )
        expert_checkpoint = consistent_version(
            _only(facts, unique, StateSourceKind.CHECKPOINT),
            facts.latest_checkpoint_step,
        )
        if dense is None or expert_checkpoint is None:
            return self._fallback(
                checkpoint,
                "hybrid source groups are incomplete or version-inconsistent",
                evidence,
                "hybrid_sources_unavailable_or_inconsistent",
            )

        checkpoint_gap = facts.resume_step - facts.latest_checkpoint_step
        if checkpoint_gap < 0:
            return self._fallback(
                checkpoint,
                "checkpoint is newer than the agreed resume step",
                evidence,
                "invalid_checkpoint_gap",
            )
        evidence.update({
            "dense_live_version": version_evidence(dense),
            "expert_checkpoint_version": version_evidence(expert_checkpoint),
            "unique_expert_identities": list(unique),
            "checkpoint_gap": checkpoint_gap,
        })
        return self._admit_hybrid(facts, checkpoint, evidence, classified, unique)

    def _policy_evidence(self) -> Mapping[str, Any]:
        return {
            "staleness_threshold": self.max_expert_staleness_density,
            "exposure_window_steps": self.exposure_window_steps,
        }

    def _admit_hybrid(self, facts, checkpoint, evidence, classified, unique):
        """Legacy density gate; subclasses reuse source checks, not this gate."""
        checkpoint_gap = facts.resume_step - facts.latest_checkpoint_step
        window_start = facts.resume_step - self.exposure_window_steps
        historic_debt = 0
        for event in facts.exposure_history:
            if event.step < window_start or event.step >= facts.resume_step:
                continue
            if event.checkpoint_step is None or event.expert_state_count is None:
                return self._fallback(
                    checkpoint,
                    "exposure history lacks expert debt metadata",
                    evidence,
                    "exposure_history_incomplete",
                )
            gap = event.step - event.checkpoint_step
            if gap < 0 or event.expert_state_count < 0:
                return self._fallback(
                    checkpoint,
                    "exposure history contains invalid expert debt metadata",
                    evidence,
                    "exposure_history_invalid",
                )
            historic_debt += event.expert_state_count * gap
        expert_count = sum(
            1 for _identity, (_placement, flag) in classified.items() if flag
        )
        if expert_count <= 0:
            return self._fallback(
                checkpoint,
                "expert population is unknown",
                evidence,
                "expert_population_unknown",
            )
        current_debt = len(unique) * checkpoint_gap
        denominator = expert_count * self.exposure_window_steps
        density = (historic_debt + current_debt) / denominator
        evidence.update(
            {
                "historic_expert_step_debt": historic_debt,
                "current_expert_step_debt": current_debt,
                "projected_expert_staleness_density": density,
            }
        )
        if density > self.max_expert_staleness_density:
            return self._fallback(
                checkpoint,
                "projected expert staleness density exceeds the configured threshold",
                evidence,
                "expert_staleness_density_exceeded",
            )
        return PolicyDecision(
            RecoveryMode.HYBRID,
            "current non-unique state and checkpoint expert state are consistent within the staleness budget",
            evidence,
        )
