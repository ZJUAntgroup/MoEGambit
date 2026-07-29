"""Standardised description of a framework's parallel layout.

Design doc section 7.1.  Every framework must project its own layout onto
:class:`TopologySpec` so the core never hard-codes Megatron's group getters.

The important property is the *manifest*: process-group creation order is part
of the contract, because c10d assigns communicator identity by creation order.
If ranks disagree on that order, rebuilding will deadlock inside NCCL rather
than fail cleanly.  We therefore hash the manifest and compare it up front, and
treat divergence as a contract violation instead of waiting for a timeout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

from ..errors import ContractViolation

__all__ = ["GroupSpec", "TopologySpec", "ValidationReport"]


@dataclass(frozen=True)
class GroupSpec:
    """One process group, identified by *purpose* rather than by object."""

    name: str
    ranks: Tuple[int, ...]
    backend: str
    purpose: str
    creation_ordinal: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "ranks": list(self.ranks),
            "backend": self.backend,
            "purpose": self.purpose,
            "creation_ordinal": self.creation_ordinal,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Result of comparing observed state against what a plan expected."""

    ok: bool
    findings: Tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def success(cls, **details: object) -> "ValidationReport":
        return cls(ok=True, findings=(), details=details)

    @classmethod
    def failure(cls, *findings: str, **details: object) -> "ValidationReport":
        return cls(ok=False, findings=tuple(findings), details=details)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ContractViolation("; ".join(self.findings) or "validation failed")


@dataclass(frozen=True)
class TopologySpec:
    """A rank's view of the global parallel layout.

    ``logical_axes`` holds axis sizes (dp/tp/pp/ep/etp/cp and any framework
    extension); ``coordinates`` holds this rank's position on each axis.
    Adapters may add custom axes -- the core treats them opaquely.
    """

    world_size: int
    rank: int
    logical_axes: Mapping[str, int] = field(default_factory=dict)
    coordinates: Mapping[str, int] = field(default_factory=dict)
    groups: Tuple[GroupSpec, ...] = ()
    generation: int = 0
    manifest_hash: str = ""

    def compute_manifest_hash(self) -> str:
        """Hash the group manifest in creation order.

        Creation order is included on purpose: two layouts with identical group
        membership but different creation sequence are *not* interchangeable.
        """
        ordered = sorted(self.groups, key=lambda g: (g.creation_ordinal, g.name))
        blob = json.dumps(
            {
                "world_size": self.world_size,
                "axes": dict(sorted(self.logical_axes.items())),
                "groups": [g.as_dict() for g in ordered],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def sealed(self) -> "TopologySpec":
        """Return a copy whose ``manifest_hash`` matches its contents."""
        from dataclasses import replace

        return replace(self, manifest_hash=self.compute_manifest_hash())

    def agrees_with(self, other: "TopologySpec") -> ValidationReport:
        """Compare two ranks' views. Rank/coordinates are allowed to differ."""
        findings = []
        if self.world_size != other.world_size:
            findings.append(
                f"world_size mismatch: {self.world_size} != {other.world_size}"
            )
        if dict(self.logical_axes) != dict(other.logical_axes):
            findings.append("logical axes mismatch")
        if self.generation != other.generation:
            findings.append(
                f"topology generation mismatch: {self.generation} != {other.generation}"
            )
        mine = self.manifest_hash or self.compute_manifest_hash()
        theirs = other.manifest_hash or other.compute_manifest_hash()
        if mine != theirs:
            findings.append(f"group manifest mismatch: {mine} != {theirs}")
        if findings:
            return ValidationReport.failure(*findings)
        return ValidationReport.success(manifest_hash=mine)

    def group_by_purpose(self, purpose: str) -> Sequence[GroupSpec]:
        return tuple(g for g in self.groups if g.purpose == purpose)
