"""Watcher-side service that freezes one plan per recovery epoch."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..capabilities import AdapterCapabilities
from ..errors import ContractViolation, RecoveryRejected
from ..policy import (
    DeterministicStateSourcePlanner,
    ExposureEvent,
    PeerOrCheckpointPolicy,
    RecoveryFacts,
    RecoveryPolicy,
    StateSourcePlanner,
    parse_source_candidates,
)
from ..runtime.recovery_plan import RecoveryMode, RecoveryPlan, WorkerEndpoint
from ..state.catalog import StateSource, StateSourceKind
from ..state.version import StateVersion
from .protocol import PROTOCOL_VERSION
from .state_store import ControlStore, InMemoryControlStore

__all__ = ["RecoveryCoordinatorService"]


def _stable_request_digest(payload: Mapping[str, Any]) -> str:
    """Hash every plan-bearing fact while excluding rank-local diagnostics."""

    classification = dict(payload.get("classification", {}))
    decision = {
        "at_step": payload.get("at_step"),
        "recovery_epoch": payload.get("recovery_epoch"),
        "topology_generation": payload.get("topology_generation"),
        "group_manifest_hash": payload.get("group_manifest_hash"),
        "classification": {
            "recoverable": classification.get("recoverable"),
            "failure_class": classification.get("failure_class"),
            "failed_ranks": classification.get("failed_ranks"),
        },
        "capabilities": payload.get("capabilities"),
        "world_size": payload.get("world_size"),
        "state_manifest": payload.get("state_manifest"),
        "state_catalog": payload.get("state_catalog"),
        "available_state_sources": payload.get("available_state_sources"),
        "latest_checkpoint_step": payload.get("latest_checkpoint_step"),
        "exposure_history": payload.get("exposure_history"),
    }
    try:
        blob = json.dumps(
            decision,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecoveryRejected(
            f"recovery facts are not JSON-canonicalizable: {exc}"
        ) from exc
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class _FrozenAssignment:
    request_digest: str
    response: Mapping[str, Any]
    committed_ranks: Dict[int, int] = field(default_factory=dict)
    failures: list = field(default_factory=list)


class RecoveryCoordinatorService:
    """Policy-driven plan freezer for versioned state-source inventories."""

    def __init__(
        self,
        *,
        store_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        replacement_provider: Optional[
            Callable[[Mapping[str, Any]], Mapping[int, WorkerEndpoint]]
        ] = None,
        control_store: Optional[ControlStore] = None,
        policy: Optional[RecoveryPolicy] = None,
        source_planner: Optional[StateSourcePlanner] = None,
    ) -> None:
        self.store_provider = store_provider
        self.replacement_provider = replacement_provider or (lambda payload: {})
        self.control_store = control_store or InMemoryControlStore()
        self.policy = policy or PeerOrCheckpointPolicy()
        self.source_planner = source_planner or DeterministicStateSourcePlanner()
        self._assignments: Dict[Tuple[str, str, int], _FrozenAssignment] = {}
        self._latest_epochs: Dict[Tuple[str, str], int] = {}
        self._fallback_requests: Dict[Tuple[str, str], list] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _scope(job_id: str, attempt_id: str) -> Tuple[str, str]:
        if not job_id or not attempt_id:
            raise RecoveryRejected("job_id and attempt_id are required")
        return str(job_id), str(attempt_id)

    @classmethod
    def _key(
        cls,
        job_id: str,
        attempt_id: str,
        epoch: int,
    ) -> Tuple[str, str, int]:
        scope = cls._scope(job_id, attempt_id)
        if int(epoch) <= 0:
            raise RecoveryRejected("a positive recovery epoch is required")
        return scope[0], scope[1], int(epoch)

    @staticmethod
    def _validate_capabilities(
        capabilities: AdapterCapabilities,
        mode: RecoveryMode,
    ) -> None:
        if not isinstance(capabilities, AdapterCapabilities):
            raise RecoveryRejected("adapter capabilities are missing")
        if not capabilities.static_world_replacement:
            raise RecoveryRejected("adapter cannot retain a failed logical rank")
        if not (
            capabilities.full_group_rebuild
            or capabilities.selective_group_rebuild
        ):
            raise RecoveryRejected("adapter cannot rebuild an affected group")
        if mode in (RecoveryMode.PEER, RecoveryMode.HYBRID) and not (
            capabilities.peer_parameter_restore
        ):
            raise RecoveryRejected("adapter cannot restore state from a peer")
        if mode is RecoveryMode.HYBRID and not capabilities.moe_state_classification:
            raise RecoveryRejected("adapter cannot classify MoE state for hybrid restore")

    @staticmethod
    def _catalog_descriptors(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]:
        raw_catalog = payload.get("state_catalog")
        if not isinstance(raw_catalog, list) or not raw_catalog:
            raise RecoveryRejected("state catalog is empty")
        descriptors: Dict[str, Mapping[str, Any]] = {}
        for descriptor in raw_catalog:
            if not isinstance(descriptor, Mapping):
                raise RecoveryRejected("state catalog entry must be an object")
            identity = str(descriptor.get("identity", ""))
            if not identity or identity in descriptors:
                raise RecoveryRejected(
                    "state identities must be non-empty and unique"
                )
            version_data = descriptor.get("version")
            if not isinstance(version_data, Mapping):
                raise RecoveryRejected(f"state version is missing for {identity!r}")
            try:
                StateVersion(
                    committed_step=int(version_data["committed_step"]),
                    optimizer_generation=int(version_data["optimizer_generation"]),
                    recovery_epoch=int(version_data.get("recovery_epoch", 0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RecoveryRejected(
                    f"state version is invalid for {identity!r}: {exc}"
                ) from exc
            descriptors[identity] = descriptor
        return descriptors

    @classmethod
    def _legacy_candidates(
        cls,
        payload: Mapping[str, Any],
        failed_ranks: Tuple[int, ...],
        world_size: int,
    ) -> Mapping[str, Tuple[StateSource, ...]]:
        """Conservative compatibility path for adapters without a provider.

        Only replicated state is inferred.  Sharded and unique state require
        explicit adapter-normalized candidates and therefore fail closed.
        """

        candidates: Dict[str, Tuple[StateSource, ...]] = {}
        survivors = tuple(
            rank for rank in range(world_size) if rank not in failed_ranks
        )
        for identity, descriptor in cls._catalog_descriptors(payload).items():
            placement = str(descriptor.get("placement", ""))
            version_data = dict(descriptor["version"])
            version = StateVersion(
                committed_step=int(version_data["committed_step"]),
                optimizer_generation=int(version_data["optimizer_generation"]),
                recovery_epoch=int(version_data.get("recovery_epoch", 0)),
            )
            metadata = dict(descriptor.get("metadata", {}))
            metadata.update(
                {
                    "kind": descriptor.get("kind"),
                    "placement": placement,
                    "owner": int(descriptor.get("owner", -1)),
                    "shape": list(descriptor.get("shape", ())),
                    "dtype": descriptor.get("dtype", ""),
                    "tags": list(descriptor.get("tags", ())),
                }
            )
            if placement == "replicated":
                candidates[identity] = tuple(
                    StateSource(
                        StateSourceKind.PEER,
                        version,
                        f"rank://{rank}",
                        metadata=metadata,
                    )
                    for rank in survivors
                )
            else:
                candidates[identity] = ()
        return candidates

    @classmethod
    def _facts(
        cls,
        payload: Mapping[str, Any],
        failed_ranks: Tuple[int, ...],
        world_size: int,
        capabilities: AdapterCapabilities,
    ) -> RecoveryFacts:
        raw_candidates = payload.get("available_state_sources")
        candidates = (
            parse_source_candidates(raw_candidates)
            if isinstance(raw_candidates, Mapping)
            else cls._legacy_candidates(payload, failed_ranks, world_size)
        )
        descriptor_ids = set(cls._catalog_descriptors(payload))
        if set(candidates) != descriptor_ids:
            raise RecoveryRejected(
                "state catalog and available source identities differ"
            )
        checkpoint_steps = {
            source.version.committed_step
            for sources in candidates.values()
            for source in sources
            if source.kind is StateSourceKind.CHECKPOINT
        }
        try:
            latest_checkpoint_step = int(
                payload.get(
                    "latest_checkpoint_step",
                    max(checkpoint_steps) if checkpoint_steps else -1,
                )
            )
        except (TypeError, ValueError) as exc:
            raise RecoveryRejected(
                "latest_checkpoint_step must be an integer"
            ) from exc
        exposure = []
        raw_exposure = payload.get("exposure_history", ())
        if isinstance(raw_exposure, (str, bytes)) or not isinstance(
            raw_exposure, (list, tuple)
        ):
            raise RecoveryRejected("exposure_history must be a sequence")
        for item in raw_exposure:
            if not isinstance(item, Mapping):
                raise RecoveryRejected("exposure event must be an object")
            try:
                exposure.append(
                    ExposureEvent(
                        step=int(item["step"]),
                        ranks=tuple(int(rank) for rank in item.get("ranks", ())),
                        reason=str(item.get("reason", "")),
                        checkpoint_step=(
                            None
                            if item.get("checkpoint_step") is None
                            else int(item["checkpoint_step"])
                        ),
                        expert_state_count=(
                            None
                            if item.get("expert_state_count") is None
                            else int(item["expert_state_count"])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RecoveryRejected(f"invalid exposure event: {exc}") from exc
        return RecoveryFacts(
            failed_ranks=failed_ranks,
            resume_step=int(payload["at_step"]),
            latest_checkpoint_step=latest_checkpoint_step,
            available_state_sources=candidates,
            exposure_history=tuple(exposure),
            capabilities=capabilities,
        )

    def _reject_stale(self, scope: Tuple[str, str], epoch: int) -> None:
        latest = self._latest_epochs.get(scope, 0)
        if epoch < latest:
            raise ContractViolation(
                f"stale recovery epoch {epoch}; latest frozen epoch is {latest}"
            )

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
    ) -> Mapping[str, Any]:
        try:
            epoch = int(payload.get("recovery_epoch", -1))
        except (TypeError, ValueError) as exc:
            raise RecoveryRejected("recovery_epoch must be an integer") from exc
        key = self._key(job_id, attempt_id, epoch)
        scope = key[:2]
        classification = payload.get("classification")
        if not isinstance(classification, Mapping) or not classification.get(
            "recoverable"
        ):
            raise RecoveryRejected(
                "failure was not positively classified as recoverable"
            )
        failed_ranks = tuple(
            int(rank) for rank in classification.get("failed_ranks", ())
        )
        if len(failed_ranks) != 1:
            raise RecoveryRejected("Phase C service requires one failed rank")
        world_size = int(payload.get("world_size", 0))
        if world_size <= 1 or any(
            rank < 0 or rank >= world_size for rank in failed_ranks
        ):
            raise RecoveryRejected("failed rank/world_size contract is invalid")
        raw_capabilities = payload.get("capabilities")
        if not isinstance(raw_capabilities, Mapping):
            raise RecoveryRejected("adapter capabilities are missing")
        try:
            capabilities = AdapterCapabilities.from_dict(raw_capabilities)
        except (TypeError, ValueError) as exc:
            raise RecoveryRejected(f"adapter capabilities are invalid: {exc}") from exc
        request_digest = _stable_request_digest(payload)

        with self._lock:
            self._reject_stale(scope, epoch)
            existing = self._assignments.get(key)
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise ContractViolation(
                        "ranks submitted different recovery facts for one epoch"
                    )
                return dict(existing.response)

            facts = self._facts(payload, failed_ranks, world_size, capabilities)
            decision = self.policy.decide(facts)
            if decision.mode is RecoveryMode.ABORT:
                raise RecoveryRejected(
                    "recovery policy aborted: " + decision.reason
                )
            self._validate_capabilities(capabilities, decision.mode)
            sources = self.source_planner.select(facts, decision)
            resume_step = (
                facts.latest_checkpoint_step
                if decision.mode is RecoveryMode.CHECKPOINT
                else facts.resume_step
            )
            if resume_step < 0:
                raise RecoveryRejected("recovery policy selected an invalid resume step")
            plan = RecoveryPlan(
                protocol_version=PROTOCOL_VERSION,
                recovery_epoch=epoch,
                failed_ranks=failed_ranks,
                replacements=dict(self.replacement_provider(payload)),
                resume_step=resume_step,
                mode=decision.mode,
                topology_generation=int(payload["topology_generation"]),
                group_manifest_hash=str(payload["group_manifest_hash"]),
                state_sources=sources,
                policy_evidence=dict(
                    decision.evidence,
                    policy=type(self.policy).__name__,
                    reason=decision.reason,
                    request_digest=request_digest,
                ),
            )
            store = dict(self.store_provider(payload))
            if "host" not in store or "port" not in store:
                raise RecoveryRejected("store provider omitted host or port")
            response = {
                "ok": True,
                "plan": plan.as_dict(),
                "plan_digest": plan.digest(),
                "store": store,
            }
            self._assignments[key] = _FrozenAssignment(request_digest, response)
            store_key = f"assignment/{scope[0]}/{scope[1]}"
            if not self.control_store.put_if_epoch(store_key, response, epoch):
                del self._assignments[key]
                raise ContractViolation("control store rejected the frozen epoch")
            self._latest_epochs[scope] = epoch
            return dict(response)

    def get_assignment(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        key = self._key(job_id, attempt_id, recovery_epoch)
        with self._lock:
            self._reject_stale(key[:2], int(recovery_epoch))
            frozen = self._assignments.get(key)
            if frozen is None:
                raise RecoveryRejected("recovery assignment is not frozen yet")
            expected = str(payload.get("plan_digest", ""))
            if expected and expected != frozen.response["plan_digest"]:
                raise ContractViolation(
                    "requested recovery plan digest does not match"
                )
            return dict(frozen.response)

    def committed(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
        rank: int,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        key = self._key(job_id, attempt_id, recovery_epoch)
        with self._lock:
            self._reject_stale(key[:2], int(recovery_epoch))
            frozen = self._assignments.get(key)
            if frozen is None:
                raise RecoveryRejected("commit references an unknown recovery epoch")
            if payload.get("plan_digest") != frozen.response["plan_digest"]:
                raise ContractViolation(
                    "commit plan digest does not match frozen plan"
                )
            step = int(payload["step"])
            plan_step = int(frozen.response["plan"]["resume_step"])
            if step <= plan_step:
                raise ContractViolation(
                    "commit must follow a complete post-recovery iteration"
                )
            existing_steps = set(frozen.committed_ranks.values())
            if existing_steps and step not in existing_steps:
                raise ContractViolation(
                    "ranks committed different post-recovery steps"
                )
            previous = frozen.committed_ranks.get(int(rank))
            if previous is not None and previous != step:
                raise ContractViolation("rank changed its committed recovery step")
            frozen.committed_ranks[int(rank)] = step
            return {"ok": True, "committed_count": len(frozen.committed_ranks)}

    def failed(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
        rank: int,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        key = self._key(job_id, attempt_id, recovery_epoch)
        with self._lock:
            frozen = self._assignments.get(key)
            if frozen is None:
                raise RecoveryRejected("failure references an unknown recovery epoch")
            supplied_digest = str(payload.get("plan_digest", ""))
            if supplied_digest and supplied_digest != frozen.response["plan_digest"]:
                raise ContractViolation(
                    "failure plan digest does not match frozen plan"
                )
            frozen.failures.append(
                {
                    "rank": int(rank),
                    "error_type": str(payload.get("error_type", "unknown")),
                    "error": str(payload.get("error", ""))[:1000],
                }
            )
            return {"ok": True, "failure_count": len(frozen.failures)}

    def request_checkpoint_relaunch(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
        rank: int,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id)
        record = {
            "rank": int(rank),
            "recovery_epoch": int(recovery_epoch),
            "at_step": int(payload.get("at_step", -1)),
            "reason": str(payload.get("reason", ""))[:1000],
            "error_type": str(payload.get("error_type", "unknown")),
            "evidence": dict(payload.get("evidence", {})),
        }
        if not record["reason"]:
            raise RecoveryRejected("checkpoint relaunch reason is required")
        with self._lock:
            self._fallback_requests.setdefault(scope, []).append(record)
        return {"ok": True, "action": "checkpoint_relaunch"}

    def heartbeat(
        self,
        *,
        job_id: str,
        attempt_id: str,
    ) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id)
        with self._lock:
            return {
                "ok": True,
                "latest_recovery_epoch": self._latest_epochs.get(scope, 0),
                "fallback_requests": len(self._fallback_requests.get(scope, ())),
            }

    def snapshot(self, job_id: str, attempt_id: str) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id)
        with self._lock:
            assignments = {
                epoch: dict(frozen.response)
                for (job, attempt, epoch), frozen in self._assignments.items()
                if (job, attempt) == scope
            }
            return {
                "latest_epoch": self._latest_epochs.get(scope, 0),
                "assignments": assignments,
                "fallback_requests": list(self._fallback_requests.get(scope, ())),
            }
