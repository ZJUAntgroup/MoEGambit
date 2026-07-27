"""Framework-neutral distributed failure classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, runtime_checkable

from ..errors import RecoverableDistributedError

__all__ = [
    "FailureClassification",
    "DistributedBackend",
    "ConservativeDistributedBackend",
]


@dataclass(frozen=True)
class FailureClassification:
    recoverable: bool
    failure_class: str = "unclassified"
    failed_ranks: Tuple[int, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DistributedBackend(Protocol):
    def classify_error(self, exc: BaseException) -> FailureClassification: ...


class ConservativeDistributedBackend:
    """Accept typed failures and an optional explicit backend classifier."""

    def __init__(
        self,
        classifier: Optional[
            Callable[[BaseException], Optional[FailureClassification]]
        ] = None,
    ) -> None:
        self._classifier = classifier

    def classify_error(self, exc: BaseException) -> FailureClassification:
        if isinstance(exc, RecoverableDistributedError):
            return FailureClassification(
                True,
                failure_class=getattr(exc, "failure_class", "fail_stop"),
                failed_ranks=tuple(getattr(exc, "failed_ranks", ())),
                evidence=dict(getattr(exc, "evidence", {})),
            )
        if self._classifier is not None:
            result = self._classifier(exc)
            if result is not None:
                return result
        return FailureClassification(
            False,
            evidence={"exception_type": type(exc).__name__},
        )
