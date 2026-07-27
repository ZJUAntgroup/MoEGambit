"""Watcher-side service that freezes one plan per recovery epoch."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import quote

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
from .relaunch import RelaunchDirective
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
            allow_nan=False,
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
    def _scope_token(scope: Tuple[str, str]) -> str:
        return f"{quote(scope[0], safe='')}/{quote(scope[1], safe='')}"

    @classmethod
    def _assignment_store_key(cls, key: Tuple[str, str, int]) -> str:
        return f"assignment/{cls._scope_token(key[:2])}/{int(key[2])}"

    @classmethod
    def _latest_store_key(cls, scope: Tuple[str, str]) -> str:
        return f"latest/{cls._scope_token(scope)}"

    @classmethod
    def _fallback_store_key(cls, scope: Tuple[str, str]) -> str:
        return f"fallback/{cls._scope_token(scope)}"

    @classmethod
    def _relaunch_store_key(cls, scope: Tuple[str, str]) -> str:
        return f"relaunch/{cls._scope_token(scope)}"

    @staticmethod
    def _frozen_record(frozen: _FrozenAssignment) -> Mapping[str, Any]:
        return {
            "request_digest": str(frozen.request_digest),
            "response": dict(frozen.response),
            "committed_ranks": {
                str(rank): int(step)
                for rank, step in sorted(frozen.committed_ranks.items())
            },
            "failures": [dict(item) for item in frozen.failures],
        }

    @staticmethod
    def _frozen_from_record(value: Any) -> _FrozenAssignment:
        if not isinstance(value, Mapping):
            raise ContractViolation("persisted recovery assignment is not an object")
        response = value.get("response")
        committed = value.get("committed_ranks", {})
        failures = value.get("failures", [])
        if not isinstance(response, Mapping):
            raise ContractViolation("persisted recovery response is missing")
        if not isinstance(committed, Mapping) or not isinstance(failures, list):
            raise ContractViolation("persisted recovery assignment state is invalid")
        if any(not isinstance(item, Mapping) for item in failures):
            raise ContractViolation("persisted recovery failures are invalid")
        request_digest = str(value.get("request_digest", ""))
        if not request_digest:
            raise ContractViolation("persisted recovery request digest is missing")
        try:
            committed_ranks = {
                int(rank): int(step) for rank, step in committed.items()
            }
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "persisted recovery commit state is invalid"
            ) from exc
        return _FrozenAssignment(
            request_digest=request_digest,
            response=dict(response),
            committed_ranks=committed_ranks,
            failures=[dict(item) for item in failures],
        )

    def _load_latest_epoch(self, scope: Tuple[str, str]) -> int:
        raw = self.control_store.get(self._latest_store_key(scope))
        persisted = 0
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ContractViolation("persisted latest epoch is not an object")
            try:
                persisted = int(raw["epoch"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractViolation("persisted latest epoch is invalid") from exc
        latest = max(self._latest_epochs.get(scope, 0), persisted)
        self._latest_epochs[scope] = latest
        return latest

    def _load_frozen(
        self, key: Tuple[str, str, int]
    ) -> Optional[_FrozenAssignment]:
        raw = self.control_store.get(self._assignment_store_key(key))
        if raw is None:
            return self._assignments.get(key)
        frozen = self._frozen_from_record(raw)
        self._assignments[key] = frozen
        return frozen

    def _persist_new_frozen(
        self,
        key: Tuple[str, str, int],
        candidate: _FrozenAssignment,
    ) -> _FrozenAssignment:
        assignment_key = self._assignment_store_key(key)
        epoch = int(key[2])
        record = self._frozen_record(candidate)
        if not self.control_store.put_if_epoch(assignment_key, record, epoch):
            winner_raw = self.control_store.get(assignment_key)
            if winner_raw is None:
                raise ContractViolation("control store rejected the frozen epoch")
            winner = self._frozen_from_record(winner_raw)
            if winner.request_digest != candidate.request_digest:
                raise ContractViolation(
                    "ranks submitted different recovery facts for one epoch"
                )
            candidate = winner

        latest_record = {
            "epoch": epoch,
            "plan_digest": str(candidate.response["plan_digest"]),
        }
        if not self.control_store.put_if_epoch(
            self._latest_store_key(key[:2]), latest_record, epoch
        ):
            latest = self._load_latest_epoch(key[:2])
            if latest > epoch:
                raise ContractViolation(
                    f"stale recovery epoch {epoch}; latest frozen epoch is {latest}"
                )
            raise ContractViolation("control store rejected latest epoch publication")
        self._assignments[key] = candidate
        self._latest_epochs[key[:2]] = epoch
        return candidate

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
        latest = self._load_latest_epoch(scope)
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
            existing = self._load_frozen(key)
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
            frozen = self._persist_new_frozen(
                key, _FrozenAssignment(request_digest, response)
            )
            return dict(frozen.response)

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
            frozen = self._load_frozen(key)
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
            store_key = self._assignment_store_key(key)
            for _attempt in range(16):
                current_raw = self.control_store.get(store_key)
                if current_raw is None:
                    raise RecoveryRejected(
                        "commit references an unknown recovery epoch"
                    )
                frozen = self._frozen_from_record(current_raw)
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
                if previous is not None:
                    if previous != step:
                        raise ContractViolation(
                            "rank changed its committed recovery step"
                        )
                    return {
                        "ok": True,
                        "committed_count": len(frozen.committed_ranks),
                    }
                committed_ranks = dict(frozen.committed_ranks)
                committed_ranks[int(rank)] = step
                updated = _FrozenAssignment(
                    frozen.request_digest,
                    frozen.response,
                    committed_ranks,
                    list(frozen.failures),
                )
                if self.control_store.compare_and_set(
                    store_key,
                    current_raw,
                    self._frozen_record(updated),
                    int(recovery_epoch),
                ):
                    self._assignments[key] = updated
                    return {
                        "ok": True,
                        "committed_count": len(updated.committed_ranks),
                    }
            raise ContractViolation("concurrent recovery commits did not converge")

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
            self._reject_stale(key[:2], int(recovery_epoch))
            store_key = self._assignment_store_key(key)
            record = {
                "rank": int(rank),
                "error_type": str(payload.get("error_type", "unknown")),
                "error": str(payload.get("error", ""))[:1000],
            }
            for _attempt in range(16):
                current_raw = self.control_store.get(store_key)
                if current_raw is None:
                    raise RecoveryRejected(
                        "failure references an unknown recovery epoch"
                    )
                frozen = self._frozen_from_record(current_raw)
                supplied_digest = str(payload.get("plan_digest", ""))
                if supplied_digest and supplied_digest != frozen.response["plan_digest"]:
                    raise ContractViolation(
                        "failure plan digest does not match frozen plan"
                    )
                if record in frozen.failures:
                    return {"ok": True, "failure_count": len(frozen.failures)}
                updated = _FrozenAssignment(
                    frozen.request_digest,
                    frozen.response,
                    dict(frozen.committed_ranks),
                    [*frozen.failures, record],
                )
                if self.control_store.compare_and_set(
                    store_key,
                    current_raw,
                    self._frozen_record(updated),
                    int(recovery_epoch),
                ):
                    self._assignments[key] = updated
                    return {"ok": True, "failure_count": len(updated.failures)}
            raise ContractViolation("concurrent recovery failures did not converge")

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
        try:
            at_step = int(payload.get("at_step", -1))
            checkpoint_step = int(payload.get("checkpoint_step", -1))
        except (TypeError, ValueError) as exc:
            raise RecoveryRejected(
                "checkpoint relaunch steps must be integers"
            ) from exc
        checkpoint_locator = str(payload.get("checkpoint_locator") or "").strip()
        if at_step < 0:
            raise RecoveryRejected("checkpoint relaunch at_step is required")
        if checkpoint_step < 0 or not checkpoint_locator:
            raise RecoveryRejected(
                "checkpoint relaunch requires a committed checkpoint locator and step"
            )
        if checkpoint_step > at_step:
            raise RecoveryRejected(
                "checkpoint relaunch target cannot be newer than the failed step"
            )
        reason = str(payload.get("reason", ""))[:1000]
        if not reason:
            raise RecoveryRejected("checkpoint relaunch reason is required")
        directive = RelaunchDirective.create(
            job_id=job_id,
            attempt_id=attempt_id,
            recovery_epoch=int(recovery_epoch),
            checkpoint_locator=checkpoint_locator,
            checkpoint_step=checkpoint_step,
            reason=reason,
            command_digest=(
                str(payload["command_digest"])
                if payload.get("command_digest")
                else None
            ),
        )
        record = {
            "rank": int(rank),
            "recovery_epoch": int(recovery_epoch),
            "at_step": at_step,
            "reason": reason,
            "error_type": str(payload.get("error_type", "unknown")),
            "evidence": dict(payload.get("evidence", {})),
            "checkpoint_locator": checkpoint_locator,
            "checkpoint_step": checkpoint_step,
            "directive_id": directive.directive_id,
        }
        with self._lock:
            self._reject_stale(scope, int(recovery_epoch))
            directive_record = {
                "directive": dict(directive.as_dict()),
                "status": "pending",
                "acknowledged_nodes": {},
            }
            relaunch_key = self._relaunch_store_key(scope)
            if not self.control_store.put_if_epoch(
                relaunch_key,
                directive_record,
                max(0, int(recovery_epoch)),
            ):
                existing = self.control_store.get(relaunch_key)
                if not isinstance(existing, Mapping) or not isinstance(
                    existing.get("directive"), Mapping
                ):
                    raise ContractViolation(
                        "persisted checkpoint relaunch directive is invalid"
                    )
                frozen = RelaunchDirective.from_dict(existing["directive"])
                if frozen.directive_id != directive.directive_id:
                    raise ContractViolation(
                        "ranks requested different checkpoint relaunch targets for one epoch"
                    )
                directive = frozen

            store_key = self._fallback_store_key(scope)
            for _attempt in range(16):
                current = self.control_store.get(store_key)
                if current is None:
                    records = []
                elif isinstance(current, list):
                    records = [
                        dict(item) for item in current if isinstance(item, Mapping)
                    ]
                else:
                    raise ContractViolation(
                        "persisted checkpoint relaunch requests are invalid"
                    )
                if record in records:
                    break
                updated = [*records, record]
                if self.control_store.compare_and_set(
                    store_key,
                    current,
                    updated,
                    max(0, int(recovery_epoch)),
                ):
                    records = updated
                    break
            else:
                raise ContractViolation(
                    "concurrent checkpoint relaunch requests did not converge"
                )
            self._fallback_requests[scope] = records
        return {
            "ok": True,
            "action": "checkpoint_relaunch",
            "directive": dict(directive.as_dict()),
        }

    def acknowledge_checkpoint_relaunch(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        attempt_id: str,
        node_rank: int,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        if int(node_rank) < 0:
            raise RecoveryRejected("checkpoint relaunch acknowledgement needs node_rank")
        scope = self._scope(job_id, attempt_id)
        store_key = self._relaunch_store_key(scope)
        for _attempt in range(16):
            current = self.control_store.get(store_key)
            if not isinstance(current, Mapping) or not isinstance(
                current.get("directive"), Mapping
            ):
                raise RecoveryRejected("checkpoint relaunch directive is not pending")
            directive = RelaunchDirective.from_dict(current["directive"])
            if directive.recovery_epoch != int(recovery_epoch):
                raise ContractViolation(
                    "checkpoint relaunch acknowledgement epoch does not match"
                )
            if str(payload.get("directive_id", "")) != directive.directive_id:
                raise ContractViolation(
                    "checkpoint relaunch acknowledgement digest does not match"
                )
            if str(payload.get("next_attempt_id", "")) != directive.next_attempt_id:
                raise ContractViolation(
                    "checkpoint relaunch acknowledgement attempt does not match"
                )
            command_digest = str(payload.get("command_digest", ""))
            if directive.command_digest and command_digest != directive.command_digest:
                raise ContractViolation(
                    "checkpoint relaunch command digest does not match"
                )
            acknowledgements = dict(current.get("acknowledged_nodes", {}))
            node_key = str(int(node_rank))
            acknowledgement = {
                "next_attempt_id": directive.next_attempt_id,
                "command_digest": command_digest,
                "worker_count": int(payload.get("worker_count", 0)),
            }
            previous = acknowledgements.get(node_key)
            if previous is not None:
                if previous != acknowledgement:
                    raise ContractViolation(
                        "node changed its checkpoint relaunch acknowledgement"
                    )
                return {
                    "ok": True,
                    "acknowledged_nodes": len(acknowledgements),
                }
            acknowledgements[node_key] = acknowledgement
            updated = dict(current)
            updated["acknowledged_nodes"] = acknowledgements
            updated["status"] = "acknowledged"
            if self.control_store.compare_and_set(
                store_key,
                current,
                updated,
                max(0, int(recovery_epoch)),
            ):
                return {
                    "ok": True,
                    "acknowledged_nodes": len(acknowledgements),
                }
        raise ContractViolation(
            "concurrent checkpoint relaunch acknowledgements did not converge"
        )

    def heartbeat(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        job_id: str,
        attempt_id: str,
    ) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id)
        with self._lock:
            fallback = self.control_store.get(self._fallback_store_key(scope))
            fallback_count = len(fallback) if isinstance(fallback, list) else 0
            response = {
                "ok": True,
                "latest_recovery_epoch": self._load_latest_epoch(scope),
                "fallback_requests": fallback_count,
            }
            heartbeat = dict(payload or {})
            if str(heartbeat.get("role", "")) == "node_agent":
                raw = self.control_store.get(self._relaunch_store_key(scope))
                if isinstance(raw, Mapping) and isinstance(
                    raw.get("directive"), Mapping
                ):
                    directive = RelaunchDirective.from_dict(raw["directive"])
                    node_rank = int(heartbeat.get("node_rank", -1))
                    command_digest = str(heartbeat.get("command_digest", ""))
                    if (
                        directive.command_digest
                        and command_digest != directive.command_digest
                    ):
                        raise ContractViolation(
                            "node agent command digest differs from relaunch directive"
                        )
                    acknowledged = dict(raw.get("acknowledged_nodes", {}))
                    if node_rank >= 0 and str(node_rank) not in acknowledged:
                        response["relaunch"] = dict(directive.as_dict())
            return response

    def snapshot(self, job_id: str, attempt_id: str) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id)
        with self._lock:
            latest = self._load_latest_epoch(scope)
            if latest > 0:
                self._load_frozen((scope[0], scope[1], latest))
            assignments = {
                epoch: dict(frozen.response)
                for (job, attempt, epoch), frozen in self._assignments.items()
                if (job, attempt) == scope
            }
            fallback = self.control_store.get(self._fallback_store_key(scope))
            fallback_records = (
                [dict(item) for item in fallback if isinstance(item, Mapping)]
                if isinstance(fallback, list)
                else []
            )
            self._fallback_requests[scope] = fallback_records
            relaunch = self.control_store.get(self._relaunch_store_key(scope))
            return {
                "latest_epoch": latest,
                "assignments": assignments,
                "fallback_requests": fallback_records,
                "relaunch": relaunch,
            }
