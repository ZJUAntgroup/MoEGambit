"""Versioned control client and watcher-backed recovery coordinator."""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ..adapters.base import FrameworkAdapter, StateSources, StoreHandle
from ..control.coordinator import (
    RecoveryAssignment,
    RecoveryCoordinator,
    RecoveryRequest,
)
from ..control.protocol import Envelope, decode_message
from ..errors import ContractViolation, RecoveryRejected, RecoveryTimeout
from ..policy import (
    RecoveryEvidenceProvider,
    SourceQuery,
    StateSourceCandidateProvider,
    StateSourceResolver,
    serialize_source_candidates,
)
from ..runtime.recovery_plan import RecoveryPlan
from ..state.catalog import Placement, StateRef, StateSource, StateSourceKind

__all__ = [
    "ControlClientConfig",
    "ControlClient",
    "WatcherRecoveryCoordinator",
    "build_recovery_request_payload",
]


@dataclass(frozen=True)
class ControlClientConfig:
    host: str
    port: int
    job_id: str
    attempt_id: str
    sender: Mapping[str, Any]
    job_token: Optional[str] = None
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 300.0
    max_message_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if not self.host or not 0 < int(self.port) <= 65535:
            raise ValueError("control endpoint must contain a host and valid port")
        if not self.job_id or not self.attempt_id:
            raise ValueError("control client job_id and attempt_id are required")
        if self.connect_timeout_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError("control client timeouts must be positive")
        if self.max_message_bytes <= 0:
            raise ValueError("control max_message_bytes must be positive")


class ControlClient:
    def __init__(
        self,
        config: ControlClientConfig,
        *,
        socket_factory: Callable[..., socket.socket] = socket.create_connection,
    ) -> None:
        self.config = config
        self._socket_factory = socket_factory
        self._lock = threading.Lock()

    def with_attempt(self, attempt_id: str) -> "ControlClient":
        """Return a client for the next cold-relaunch attempt."""

        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("control attempt_id must be non-empty")
        return type(self)(
            replace(self.config, attempt_id=attempt_id),
            socket_factory=self._socket_factory,
        )

    def _encode(self, envelope: Envelope) -> bytes:
        if self.config.job_token:
            envelope.sign(self.config.job_token)
        data = (envelope.to_json() + "\n").encode("utf-8")
        if len(data) > self.config.max_message_bytes:
            raise ValueError("control request exceeds max_message_bytes")
        return data

    def _read_line(self, sock: socket.socket) -> str:
        buffer = bytearray()
        while True:
            chunk = sock.recv(min(65536, self.config.max_message_bytes + 1))
            if not chunk:
                raise ConnectionError("control server closed before responding")
            buffer.extend(chunk)
            if len(buffer) > self.config.max_message_bytes:
                raise ValueError("control response exceeds max_message_bytes")
            newline = buffer.find(b"\n")
            if newline >= 0:
                return bytes(buffer[:newline]).decode("utf-8")

    def request(
        self,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        request = Envelope.new(
            message_type,
            payload,
            job_id=self.config.job_id,
            attempt_id=self.config.attempt_id,
            recovery_epoch=int(recovery_epoch),
            sender=self.config.sender,
        )
        with self._lock:
            sock = self._socket_factory(
                (self.config.host, int(self.config.port)),
                timeout=float(self.config.connect_timeout_s),
            )
            try:
                sock.settimeout(float(self.config.request_timeout_s))
                sock.sendall(self._encode(request))
                raw = self._read_line(sock)
            except socket.timeout as exc:
                raise RecoveryTimeout(
                    f"control request {message_type!r} timed out"
                ) from exc
            finally:
                sock.close()
        decoded, response = decode_message(raw)
        if response is None:
            raise ContractViolation("control server returned an unversioned response")
        if self.config.job_token:
            try:
                response.verify(self.config.job_token)
            except ValueError as exc:
                raise ContractViolation(
                    f"control response authentication failed: {exc}"
                ) from exc
        if response.job_id != request.job_id:
            raise ContractViolation("control response job_id does not match request")
        if response.attempt_id != request.attempt_id:
            raise ContractViolation("control response attempt_id does not match request")
        if response.recovery_epoch != request.recovery_epoch:
            raise ContractViolation("control response epoch does not match request")
        if decoded.get("request_id") != request.message_id:
            raise ContractViolation(
                "control response does not correlate to the current request"
            )
        if decoded.get("ok") is False:
            raise RecoveryRejected(
                f"{decoded.get('error_type', 'RecoveryRejected')}: "
                f"{decoded.get('error', 'recovery rejected')}"
            )
        return decoded

    def notify(
        self,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        recovery_epoch: int,
    ) -> Mapping[str, Any]:
        return self.request(
            message_type,
            payload,
            recovery_epoch=recovery_epoch,
        )


def build_recovery_request_payload(
    request: RecoveryRequest,
    adapter: FrameworkAdapter,
    *, quality_identity: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    classification = request.classification
    topology = adapter.topology.inspect()
    catalog = adapter.state.catalog()
    state_catalog = []
    for ref in sorted(catalog, key=lambda item: item.identity):
        tensor = ref.tensor
        state_catalog.append(
            {
                "identity": ref.identity,
                "kind": ref.kind.value,
                "placement": ref.placement.value,
                # A replicated value has no unique logical owner.  Normalizing
                # this field prevents otherwise identical ranks from producing
                # different plan-bearing request digests.
                "owner": (
                    -1
                    if ref.placement is Placement.REPLICATED
                    else int(ref.owner)
                ),
                "version": {
                    "committed_step": ref.version.committed_step,
                    "optimizer_generation": ref.version.optimizer_generation,
                    "recovery_epoch": ref.version.recovery_epoch,
                },
                "shape": (
                    [int(size) for size in getattr(tensor, "shape", ())]
                    if tensor is not None
                    else []
                ),
                "dtype": (
                    str(getattr(tensor, "dtype", ""))
                    if tensor is not None
                    else ""
                ),
                "tags": sorted(ref.tags),
                "metadata": dict(ref.metadata),
            }
        )
    query = SourceQuery(
        failed_ranks=tuple(int(rank) for rank in classification.failed_ranks),
        resume_step=int(request.at_step),
        world_size=int(topology.world_size),
        state_catalog=catalog,
    )
    if isinstance(adapter.state, StateSourceCandidateProvider):
        raw_candidates = adapter.state.source_candidates(query)
    else:
        # Compatibility is intentionally conservative.  Replicated state can
        # be inferred from the global rank set; sharded/unique ownership and
        # checkpoint locators must be described by an adapter provider.
        survivors = tuple(
            rank
            for rank in range(int(topology.world_size))
            if rank not in query.failed_ranks
        )
        raw_candidates = {
            ref.identity: tuple(
                StateSource(
                    kind=StateSourceKind.PEER,
                    version=ref.version,
                    locator=f"rank://{rank}",
                    metadata={
                        "kind": ref.kind.value,
                        "placement": ref.placement.value,
                        "owner": -1,
                        "shape": [
                            int(size)
                            for size in getattr(ref.tensor, "shape", ())
                        ],
                        "dtype": (
                            str(getattr(ref.tensor, "dtype", ""))
                            if ref.tensor is not None
                            else ""
                        ),
                        "tags": sorted(ref.tags),
                        **dict(ref.metadata),
                    },
                )
                for rank in survivors
            )
            if ref.placement is Placement.REPLICATED
            else ()
            for ref in catalog
        }
    candidates = serialize_source_candidates(raw_candidates)
    checkpoint_steps = [
        int(source["version"]["committed_step"])
        for sources in candidates.values()
        for source in sources
        if source["kind"] == StateSourceKind.CHECKPOINT.value
    ]
    exposure_history: Sequence[Mapping[str, Any]] = ()
    if isinstance(adapter.state, RecoveryEvidenceProvider):
        exposure_history = adapter.state.recovery_exposure_history(query)
    context_provider = getattr(adapter.state, "quality_recovery_context", None)
    quality_context = context_provider(query) if callable(context_provider) else {}
    if not isinstance(quality_context, Mapping):
        raise ContractViolation("quality_recovery_context must return an object")
    quality_context = dict(quality_context)
    for key, value in (quality_identity or {}).items():
        if key in quality_context and quality_context[key] != value:
            raise ContractViolation("offloader and adapter quality identities differ")
        quality_context[key] = value
    return {
        "quality_context": dict(quality_context),
        "quality_source_topology_generation": int(topology.generation),
        "quality_source_recovery_epoch": int(request.recovery_epoch) - 1,
        "at_step": request.at_step,
        "recovery_epoch": request.recovery_epoch,
        "topology_generation": request.topology_generation,
        "group_manifest_hash": request.group_manifest_hash,
        "classification": {
            "recoverable": classification.recoverable,
            "failure_class": classification.failure_class,
            "failed_ranks": list(classification.failed_ranks),
            "evidence": dict(classification.evidence),
        },
        "adapter": dict(adapter.describe()),
        "capabilities": dict(adapter.capabilities.as_dict()),
        "world_size": int(topology.world_size),
        "rank": int(topology.rank),
        "state_manifest": catalog.manifest_digest(),
        "state_catalog": state_catalog,
        "available_state_sources": candidates,
        "latest_checkpoint_step": (
            max(checkpoint_steps) if checkpoint_steps else -1
        ),
        "exposure_history": [dict(item) for item in exposure_history],
    }


class WatcherRecoveryCoordinator(RecoveryCoordinator):
    def __init__(
        self,
        client: ControlClient,
        *,
        source_resolver: Optional[StateSourceResolver] = None,
        quality_identity: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.client = client
        self.source_resolver = source_resolver
        self.quality_identity = dict(quality_identity or {})

    def configure_quality_identity(self, identity: Mapping[str, str]) -> None:
        self.quality_identity = dict(identity)

    def _assignment(self, response: Mapping[str, Any]) -> RecoveryAssignment:
        raw_plan = response.get("plan")
        raw_store = response.get("store")
        if not isinstance(raw_plan, Mapping) or not isinstance(raw_store, Mapping):
            raise ContractViolation("recovery response lacks plan or store")
        plan = RecoveryPlan.from_dict(raw_plan)
        store = StoreHandle(
            host=str(raw_store["host"]),
            port=int(raw_store["port"]),
            prefix=str(raw_store.get("prefix", "moegambit")),
            timeout_s=float(raw_store.get("timeout_s", 300.0)),
        )
        sources = (
            StateSources(dict(plan.state_sources))
            if self.source_resolver is None
            else self.source_resolver.resolve(plan)
        )
        if set(sources.by_identity) != set(plan.state_sources):
            raise ContractViolation(
                "resolved state-source identities differ from frozen plan"
            )
        return RecoveryAssignment(
            plan=plan,
            store=store,
            sources=sources,
            plan_digest=str(response.get("plan_digest", "")),
        )

    def prepare(
        self,
        request: RecoveryRequest,
        adapter: FrameworkAdapter,
    ) -> RecoveryAssignment:
        response = self.client.request(
            "recovery_request",
            build_recovery_request_payload(request, adapter, quality_identity=self.quality_identity),
            recovery_epoch=request.recovery_epoch,
        )
        assignment = self._assignment(response)
        if assignment.plan.recovery_epoch != request.recovery_epoch:
            raise ContractViolation("watcher returned a plan for another epoch")
        return assignment

    def fetch_assignment(
        self,
        recovery_epoch: int,
        *,
        plan_digest: str = "",
    ) -> RecoveryAssignment:
        response = self.client.request(
            "recovery_assignment",
            {"plan_digest": plan_digest},
            recovery_epoch=int(recovery_epoch),
        )
        assignment = self._assignment(response)
        if assignment.plan.recovery_epoch != int(recovery_epoch):
            raise ContractViolation("fetched assignment belongs to another epoch")
        return assignment

    def committed(self, assignment: RecoveryAssignment, step: int) -> None:
        self.client.notify(
            "recovery_committed",
            {"plan_digest": assignment.plan_digest, "step": int(step)},
            recovery_epoch=assignment.plan.recovery_epoch,
        )

    def failed(
        self,
        assignment: Optional[RecoveryAssignment],
        exc: BaseException,
    ) -> None:
        epoch = 0 if assignment is None else assignment.plan.recovery_epoch
        payload: Dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        if assignment is not None:
            payload["plan_digest"] = assignment.plan_digest
        self.client.notify("recovery_failed", payload, recovery_epoch=epoch)
