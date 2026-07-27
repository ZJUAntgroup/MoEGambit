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
