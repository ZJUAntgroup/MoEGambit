"""Bounded, watcher-owned CPU snapshots that outlive a training process."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Mapping

from ..audit.io import digest


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class CPUQualityFeatureStore:
    """Keep complete immutable rank records, never synthesize missing features.

    The watcher must run outside the failed worker/node's failure domain. This
    RAM store does not survive watcher death; an empty store forces fallback.
    Each topology/attempt is a separate bounded scope. Exact-step snapshots
    alone are usable; a pending transfer or an older snapshot is insufficient.
    """

    def __init__(self, *, retain_steps: int = 4, max_scopes: int = 8,
                 max_record_bytes: int = 262144, max_total_bytes: int = 67108864):
        for name, value in locals().copy().items():
            if name != "self":
                _integer(value, name, 1)
        self.retain_steps = retain_steps
        self.max_scopes = max_scopes
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self._scopes: OrderedDict[tuple, dict[int, dict[int, str]]] = OrderedDict()
        self._bytes = 0
        self._conflicts = set()
        self._lock = threading.RLock()

    @staticmethod
    def _scope(job_id, attempt_id, record):
        for key, value in (("job_id", job_id), ("attempt_id", attempt_id),
                           *((key, record.get(key)) for key in
                             ("model_id", "telemetry_version", "policy_version",
                              "group_manifest_hash"))):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a non-empty string")
        generation = _integer(record.get("topology_generation"), "topology_generation")
        world = _integer(record.get("world_size"), "world_size", 1)
        epoch = _integer(record.get("recovery_epoch"), "recovery_epoch")
        if world > 4096:
            raise ValueError("world_size exceeds the feature-store rank limit")
        return (job_id, attempt_id, generation, record["group_manifest_hash"], world,
                record["model_id"], record["telemetry_version"], record["policy_version"], epoch)

    def publish(self, record: Mapping[str, Any], *, job_id: str, attempt_id: str,
                rank: int) -> Mapping[str, Any]:
        scope = self._scope(job_id, attempt_id, record)
        step = _integer(record.get("committed_step"), "committed_step")
        checkpoint = _integer(record.get("checkpoint_step"), "checkpoint_step")
        if checkpoint > step or not 0 <= _integer(rank, "rank") < scope[4]:
            raise ValueError("invalid checkpoint step or owner rank")
        if record.get("owner_rank") != rank or type(record.get("owner_rank")) is not int:
            raise ValueError("feature owner differs from authenticated sender rank")
        if record.get("schema_version") != 1 or type(record.get("schema_version")) is not int:
            raise ValueError("unsupported feature schema")
        if not isinstance(record.get("features"), Mapping):
            raise ValueError("features must be an object")
        history = record.get("run_history")
        if not isinstance(history, list) or any(not isinstance(item, Mapping) for item in history):
            raise ValueError("run_history must be a list of whole-run event objects")
        encoded = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        size = len(encoded.encode("utf-8"))
        if size > self.max_record_bytes:
            raise ValueError("quality feature record exceeds byte limit")
        with self._lock:
            if scope not in self._scopes and len(self._scopes) >= self.max_scopes:
                raise ValueError("feature-store scope limit reached; restart or release old scope")
            steps = self._scopes.get(scope, {})
            previous = steps.get(step, {}).get(rank)
            if previous is not None:
                if previous != encoded:
                    self._conflicts.add((scope, step, rank))
                    raise ValueError("conflicting features for the same rank and committed step")
                return {"ok": True, "record_digest": digest(json.loads(previous)), "committed_step": step}
            # A checkpoint rollback must never reset the quality history. Compare
            # this logical owner across retained topology generations as well.
            for other_scope, other_steps in self._scopes.items():
                if other_scope[0] != scope[0]:
                    continue
                for rank_records in other_steps.values():
                    old = rank_records.get(rank)
                    if old is None:
                        continue
                    old_history = json.loads(old)["run_history"]
                    shorter = min(len(history), len(old_history))
                    if history[:shorter] != old_history[:shorter]:
                        raise ValueError("quality run history is not append-only")
                    if len(history) < len(old_history) and (other_scope != scope or step >= max(other_steps)):
                        raise ValueError("quality run history was truncated")
            # Delayed older records cannot evict newer retained steps.
            kept = sorted(set(steps) | {step})[-self.retain_steps:]
            if step not in kept:
                raise ValueError("feature step is outside the retained window")
            expired = [key for key in steps if key not in kept]
            freed = sum(len(value.encode("utf-8")) for key in expired for value in steps[key].values())
            if self._bytes - freed + size > self.max_total_bytes:
                raise ValueError("feature-store total byte limit reached")
            self._scopes.setdefault(scope, steps)
            for key in expired:
                del steps[key]
                self._conflicts = {item for item in self._conflicts if item[:2] != (scope, key)}
            steps.setdefault(step, {})[rank] = encoded
            self._bytes += size - freed
        return {"ok": True, "record_digest": digest(json.loads(encoded)), "committed_step": step}

    def recovery_context(self, payload: Mapping[str, Any], *, job_id: str,
                         attempt_id: str) -> Mapping[str, Any]:
        context = dict(payload.get("quality_context", {}))
        # Never retain or trust a worker's post-failure reconstruction of features.
        context.pop("features", None)
        context["retained_features_complete"] = False
        try:
            scope = self._scope(job_id, attempt_id, {**context,
                "topology_generation": payload["quality_source_topology_generation"],
                "recovery_epoch": payload["quality_source_recovery_epoch"],
                "group_manifest_hash": payload["group_manifest_hash"],
                "world_size": payload["world_size"]})
            step = _integer(payload["at_step"], "at_step")
            checkpoint = _integer(payload["latest_checkpoint_step"], "latest_checkpoint_step")
            if _integer(payload["topology_generation"], "target topology generation") != scope[2] + 1:
                raise ValueError("recovery must identify its preceding source topology")
            if _integer(payload["recovery_epoch"], "target recovery epoch") != scope[8] + 1:
                raise ValueError("recovery must identify its preceding source epoch")
        except (KeyError, TypeError, ValueError):
            context["feature_retention_error"] = "invalid_or_missing_feature_scope"
            return context
        with self._lock:
            records = self._scopes.get(scope, {}).get(step, {})
            if any(item[:2] == (scope, step) for item in self._conflicts):
                context["feature_retention_error"] = "conflicting_feature_snapshot"
                return context
            missing = sorted(set(range(scope[4])) - set(records))
            if missing:
                context["feature_retention_error"] = "exact_step_not_fully_acknowledged"
                context["missing_feature_ranks"] = missing
                return context
            decoded = {str(rank): json.loads(records[rank]) for rank in range(scope[4])}
        if any(item["checkpoint_step"] != checkpoint for item in decoded.values()):
            context["feature_retention_error"] = "checkpoint_reference_mismatch"
            return context
        histories = [item["run_history"] for item in decoded.values()]
        if any(history != histories[0] for history in histories):
            context["feature_retention_error"] = "rank_history_mismatch"
            return context
        context.update({
            "retained_features_complete": True,
            "features": {"by_rank": {rank: item["features"] for rank, item in decoded.items()}},
            "run_history": histories[0],
            "feature_snapshot": {"committed_step": step, "checkpoint_step": checkpoint,
                                 "source_topology_generation": scope[2],
                                 "source_recovery_epoch": scope[8],
                                 "record_digests": {rank: digest(item) for rank, item in decoded.items()}},
        })
        return context
