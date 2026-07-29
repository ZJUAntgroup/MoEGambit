"""Typed failures for the MoEGambit recovery runtime.

Recovery decisions are driven by typed errors rather than broad substring
matching.  Unknown exceptions remain application failures and propagate.
"""

from __future__ import annotations

__all__ = [
    "MoEGambitError",
    "RecoverableDistributedError",
    "ContractViolation",
    "AdapterUnsupportedError",
    "RecoveryTimeout",
    "StateUnavailable",
    "FatalInfrastructureError",
    "ProtocolVersionMismatch",
    "RecoveryRejected",
]


class MoEGambitError(Exception):
    """Base class for every error raised by MoEGambit itself."""


class RecoverableDistributedError(MoEGambitError):
    """A distributed failure positively classified as fail-stop."""

    def __init__(
        self,
        message: str,
        *,
        failed_ranks: tuple = (),
        failure_class: str = "fail_stop",
        evidence: object = None,
    ) -> None:
        super().__init__(message)
        self.failed_ranks = tuple(int(rank) for rank in failed_ranks)
        self.failure_class = str(failure_class)
        self.evidence = dict(evidence or {})


class ContractViolation(MoEGambitError):
    """Topology, state, or step bookkeeping disagreed across participants."""


class AdapterUnsupportedError(MoEGambitError):
    """The active adapter lacks a capability required by a plan."""


class RecoveryTimeout(MoEGambitError):
    """A recovery phase exceeded its timeout budget."""


class StateUnavailable(MoEGambitError):
    """No source satisfies the required state identity and version."""


class FatalInfrastructureError(MoEGambitError):
    """Watcher, store, or replacement machinery cannot continue."""


class ProtocolVersionMismatch(MoEGambitError):
    """A peer spoke an incompatible control-protocol version."""


class RecoveryRejected(MoEGambitError):
    """A classified failure was rejected before destructive work."""
