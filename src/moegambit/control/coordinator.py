"""Typed recovery coordination contracts used by the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..adapters.base import FrameworkAdapter, StateSources, StoreHandle
from ..distributed.c10d_backend import FailureClassification
from ..errors import ContractViolation
from ..runtime.recovery_plan import RecoveryPlan

__all__ = [
    "RecoveryRequest",
    "RecoveryAssignment",
    "RecoveryCoordinator",
    "StaticRecoveryCoordinator",
]


@dataclass(frozen=True)
class RecoveryRequest:
    classification: FailureClassification
    at_step: int
    recovery_epoch: int
    topology_generation: int
    group_manifest_hash: str

    def __post_init__(self) -> None:
        if self.at_step < 0:
            raise ValueError("recovery request step must be non-negative")
        if self.recovery_epoch <= 0:
            raise ValueError("recovery request epoch must be positive")
        if self.topology_generation < 0:
            raise ValueError("topology generation must be non-negative")
        if not self.group_manifest_hash:
            raise ValueError("group manifest hash is required")
        if not self.classification.recoverable:
            raise ValueError("recovery request must carry a recoverable classification")
        if not self.classification.failed_ranks:
            raise ValueError("recovery request must name at least one failed rank")


@dataclass(frozen=True)
class RecoveryAssignment:
    plan: RecoveryPlan
    store: StoreHandle
    sources: StateSources = field(default_factory=StateSources)
    plan_digest: str = ""

    def __post_init__(self) -> None:
        actual = self.plan.digest()
        if self.plan_digest and self.plan_digest != actual:
            raise ContractViolation(
                f"recovery plan digest mismatch: {self.plan_digest} != {actual}"
            )
        object.__setattr__(self, "plan_digest", actual)


@runtime_checkable
class RecoveryCoordinator(Protocol):
    def prepare(
        self,
        request: RecoveryRequest,
        adapter: FrameworkAdapter,
    ) -> RecoveryAssignment: ...

    def committed(self, assignment: RecoveryAssignment, step: int) -> None: ...

    def failed(
        self,
        assignment: Optional[RecoveryAssignment],
        exc: BaseException,
    ) -> None: ...


class StaticRecoveryCoordinator:
    """Inject one frozen assignment for embedding and deterministic tests."""

    def __init__(self, assignment: RecoveryAssignment) -> None:
        self.assignment = assignment
        self.commits = []
        self.failures = []

    def prepare(
        self,
        request: RecoveryRequest,
        adapter: FrameworkAdapter,
    ) -> RecoveryAssignment:
        del adapter
        plan = self.assignment.plan
        if request.recovery_epoch != plan.recovery_epoch:
            raise ContractViolation("assignment belongs to another recovery epoch")
        if request.at_step != plan.resume_step:
            raise ContractViolation("assignment resume step differs from request")
        if request.topology_generation != plan.topology_generation:
            raise ContractViolation("assignment topology generation differs from request")
        if request.group_manifest_hash != plan.group_manifest_hash:
            raise ContractViolation("assignment topology manifest differs from request")
        if tuple(request.classification.failed_ranks) != tuple(plan.failed_ranks):
            raise ContractViolation("assignment failed ranks differ from request")
        return self.assignment

    def committed(self, assignment: RecoveryAssignment, step: int) -> None:
        if assignment.plan_digest != self.assignment.plan_digest:
            raise ContractViolation("commit references another recovery plan")
        if int(step) <= assignment.plan.resume_step:
            raise ContractViolation("recovery commit must follow a complete iteration")
        self.commits.append((assignment.plan.recovery_epoch, int(step)))

    def failed(
        self,
        assignment: Optional[RecoveryAssignment],
        exc: BaseException,
    ) -> None:
        epoch = None if assignment is None else assignment.plan.recovery_epoch
        self.failures.append((epoch, type(exc).__name__, str(exc)))
