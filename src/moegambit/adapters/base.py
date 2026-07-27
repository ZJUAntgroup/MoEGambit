"""The framework adapter contract.

Design doc section 8.  Rather than one god-object base class with dozens of
methods, the contract is split into four narrow ``Protocol`` interfaces which a
``FrameworkAdapter`` *composes*:

* :class:`TopologyAdapter`  -- process groups and parallel layout
* :class:`StateAdapter`     -- what state exists and how to restore it
* :class:`OptimizerAdapter` -- optimizer shard versioning around a step
* :class:`TrainingAdapter`  -- driving the training loop across a pause

Two deliberate design choices:

1. ``Protocol`` (structural typing) rather than an abstract base class.  A
   framework object satisfies the contract by having the right methods; it does
   not have to inherit from us.  That matters because adapter authors often
   wrap objects they do not own.
2. Composition over inheritance.  A framework that supports topology rebuild
   but not optimizer replication implements three protocols and declares the
   missing capability, instead of inheriting a method that raises.

The core receives a ``FrameworkAdapter`` by dependency injection and calls only
these methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

try:  # pragma: no cover - Protocol is stdlib from 3.8, guard is belt-and-braces
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from ..capabilities import AdapterCapabilities
from ..distributed.topology import TopologySpec, ValidationReport
from ..runtime.recovery_plan import RecoveryPlan
from ..state.catalog import StateCatalog, StateRef

__all__ = [
    "TopologyAdapter",
    "StateAdapter",
    "OptimizerAdapter",
    "TrainingAdapter",
    "FrameworkAdapter",
    "ProgressToken",
    "PauseRequest",
    "QuiescenceProof",
    "RebuildHandle",
    "StoreHandle",
    "StateSources",
    "RecoveryDriver",
]


@dataclass(frozen=True)
class ProgressToken:
    """Where the training loop currently is."""

    step: int
    micro_step: int = 0
    recovery_epoch: int = 0
    committed: bool = False


@dataclass(frozen=True)
class PauseRequest:
    """Instruction to reach a safe stopping point."""

    recovery_epoch: int
    reason: str
    deadline_s: float = 300.0


@dataclass(frozen=True)
class QuiescenceProof:
    """Evidence that a rank actually reached a safe point.

    We require positive proof rather than assuming a pause succeeded: tearing
    down process groups while a collective is still in flight is precisely how
    a "recovery" turns into a hang.
    """

    rank: int
    progress: ProgressToken
    collectives_drained: bool
    process_groups_destroyed: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        # Quiescence is proven *before* the topology adapter performs the
        # destructive rebuild.  Some adapters retire groups while quiescing,
        # others do so in ``TopologyAdapter.rebuild``; both are valid as long
        # as no collective remains in flight.
        return self.collectives_drained


@dataclass(frozen=True)
class RebuildHandle:
    """Opaque token returned by ``prepare_rebuild`` and consumed by ``rebuild``."""

    recovery_epoch: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoreHandle:
    """Rendezvous information for re-forming process groups."""

    host: str
    port: int
    prefix: str = "moegambit"
    timeout_s: float = 300.0


@dataclass(frozen=True)
class StateSources:
    """Resolved provenance for each state identity in a plan."""

    by_identity: Mapping[str, Any] = field(default_factory=dict)

    def for_ref(self, ref: StateRef) -> Optional[Any]:
        return self.by_identity.get(ref.identity)


@runtime_checkable
class RecoveryDriver(Protocol):
    """Optional bridge for an existing framework-owned recovery sequence.

    New adapters should normally let :class:`RecoveryExecutor` invoke the four
    narrow protocols directly.  The driver exists so a mature, already tested
    integration can be moved behind the package boundary without first
    rewriting its destructive sequence.  It is deliberately small and is not
    consulted by policy or state planning.
    """

    def classify_error(self, exc: BaseException) -> bool:
        """Return true only for an explicitly recognized distributed failure."""

    def recover(self, exc: BaseException) -> Optional[int]:
        """Recover and return the resume step, or ``None`` when not handled."""

    def poll(self, step: int) -> Optional[int]:
        """Execute a pending safe-point recovery and return its resume step."""

    def commit(self, step: int) -> bool:
        """Commit the first complete iteration after recovery."""


@runtime_checkable
class TopologyAdapter(Protocol):
    """Owns process groups and the parallel layout."""

    def inspect(self) -> TopologySpec:
        """Report the current layout, including the group manifest."""

    def prepare_rebuild(self, plan: RecoveryPlan) -> RebuildHandle:
        """Do everything possible before the destructive rebuild step."""

    def rebuild(self, plan: RecoveryPlan, store: StoreHandle) -> TopologySpec:
        """Re-form process groups; must preserve creation order."""

    def rebind(self, topology: TopologySpec) -> None:
        """Point framework-internal references at the new communicators."""

    def validate(self, expected: TopologySpec) -> ValidationReport:
        """Confirm the rebuilt layout matches what the plan expected."""


@runtime_checkable
class StateAdapter(Protocol):
    """Enumerates and restores training state."""

    def catalog(self) -> StateCatalog:
        """Enumerate recoverable state with stable identities."""

    def load_replacement_base(self, plan: RecoveryPlan) -> None:
        """Give a replacement worker a loadable baseline before peer restore."""

    def restore(self, plan: RecoveryPlan, sources: StateSources) -> None:
        """Restore state from the plan's chosen sources."""

    def validate_state(self, plan: RecoveryPlan) -> ValidationReport:
        """Confirm restored state is complete and version-consistent."""


@runtime_checkable
class OptimizerAdapter(Protocol):
    """Versions optimizer shards around the step boundary."""

    def local_state_refs(self) -> Sequence[StateRef]:
        """Optimizer state owned by this rank."""

    def before_step(self, step: int) -> None:
        """Called before state is overwritten; a committed copy must exist."""

    def after_step(self, step: int, committed: bool) -> None:
        """Publish (or roll back) the new state version."""

    def rebind(self, topology: TopologySpec) -> None:
        """Re-point optimizer collectives at rebuilt groups."""


@runtime_checkable
class TrainingAdapter(Protocol):
    """Drives the training loop across a pause/resume boundary."""

    def current_progress(self) -> ProgressToken:
        """Report the current step for cross-rank agreement."""

    def quiesce(self, request: PauseRequest) -> QuiescenceProof:
        """Reach a safe point and prove it."""

    def apply_resume(self, plan: RecoveryPlan) -> None:
        """Adopt the plan's resume step and epoch."""

    def reset_transients(self, plan: RecoveryPlan) -> None:
        """Clear caches//buffers that must not survive a rebuild."""

    def warmup_and_validate(self, plan: RecoveryPlan) -> ValidationReport:
        """Run the first post-recovery forward and check it is sane."""


@dataclass(frozen=True)
class FrameworkAdapter:
    """Composition of the four protocols plus declared capabilities.

    This is what gets injected into the runtime.  ``name`` and ``version`` are
    recorded in every ``RecoveryRecord`` so a post-mortem can tell which
    adapter produced a given outcome.
    """

    name: str
    topology: TopologyAdapter
    state: StateAdapter
    optimizer: OptimizerAdapter
    training: TrainingAdapter
    capabilities: AdapterCapabilities = field(default_factory=AdapterCapabilities)
    version: str = "unknown"
    support_level: str = "experimental"
    backend: Any = None
    recovery_driver: Optional[RecoveryDriver] = None

    def describe(self) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "support_level": self.support_level,
            "capability_digest": self.capabilities.digest(),
        }
