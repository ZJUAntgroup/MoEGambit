"""Framework-neutral inventory of recoverable training state.

Design doc section 7.2.  The catalog is how a framework tells the core *what
state exists and how it is placed*, without leaking framework types.

The single most important rule here is ``StateRef.identity``: it must stay
stable across cold start, replacement, and framework rebuild under the same
configuration.  Python ``id()``, iteration order, and memory addresses are
forbidden as identity, because all three silently change across a rebuild and
would cause state to be restored into the wrong slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Tuple

from .version import StateVersion

__all__ = [
    "StateKind",
    "Placement",
    "StateRef",
    "StateCatalog",
    "StateSource",
    "StateSourceKind",
]


class StateKind(Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"
    OPTIMIZER_TENSOR = "optimizer_tensor"
    OPTIMIZER_SCALAR = "optimizer_scalar"
    RNG = "rng"
    SCHEDULER = "scheduler"
    DATALOADER = "dataloader"
    FRAMEWORK_TRANSIENT = "framework_transient"


class Placement(Enum):
    """How a piece of state is distributed across ranks.

    This drives recovery strategy directly:

    * ``REPLICATED`` -- some peer holds an identical copy, so peer restore works.
    * ``SHARDED``    -- reconstructable from the owning group's shards.
    * ``UNIQUE``     -- no peer copy exists.  For MoE this is the hard case:
      a unique expert shard can only come from a checkpoint, which is exactly
      why the hybrid policy exists.
    """

    REPLICATED = "replicated"
    SHARDED = "sharded"
    UNIQUE = "unique"


class StateSourceKind(Enum):
    PEER = "peer"
    CHECKPOINT = "checkpoint"
    MEMORY_REPLICA = "memory_replica"
    LOCAL = "local"
    REINITIALIZE = "reinitialize"


@dataclass(frozen=True)
class StateSource:
    """Where one slice of state will come from during recovery."""

    kind: StateSourceKind
    version: StateVersion
    locator: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in (
            StateSourceKind.PEER,
            StateSourceKind.CHECKPOINT,
            StateSourceKind.MEMORY_REPLICA,
        ) and (
            not isinstance(self.locator, str) or not self.locator.strip()
        ):
            raise ValueError(
                f"{self.kind.value} state sources require a stable locator"
            )


@dataclass
class StateRef:
    """A handle to one addressable piece of training state.

    Tensors are referenced, never copied, at catalog time.  ``scalar_get`` and
    ``scalar_set`` exist because optimizer scalars (step counters, loss scale)
    are not tensors but still must be versioned and restored.
    """

    identity: str
    kind: StateKind
    placement: Placement
    owner: int
    version: StateVersion
    tensor: Optional[Any] = None
    scalar_get: Optional[Callable[[], Any]] = None
    scalar_set: Optional[Callable[[Any], None]] = None
    tags: FrozenSet[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity or not isinstance(self.identity, str):
            raise ValueError("StateRef.identity must be a non-empty string")
        if self.tensor is None and self.scalar_get is None:
            raise ValueError(
                f"StateRef {self.identity!r} carries neither a tensor nor a scalar accessor"
            )

    @property
    def is_expert(self) -> bool:
        """MoE expert state, the state most likely to have no peer replica."""
        return "expert" in self.tags

    @property
    def recoverable_from_peer(self) -> bool:
        return self.placement in (Placement.REPLICATED, Placement.SHARDED)


@dataclass
class StateCatalog:
    """The set of ``StateRef`` objects an adapter exposes for one rank."""

    refs: Tuple[StateRef, ...] = ()

    def __iter__(self) -> Iterator[StateRef]:
        return iter(self.refs)

    def __len__(self) -> int:
        return len(self.refs)

    @classmethod
    def from_iterable(cls, refs: Iterable[StateRef]) -> "StateCatalog":
        materialized = tuple(refs)
        seen = set()
        duplicates = set()
        for ref in materialized:
            if ref.identity in seen:
                duplicates.add(ref.identity)
            seen.add(ref.identity)
        if duplicates:
            raise ValueError(
                "duplicate StateRef identities: " + ", ".join(sorted(duplicates))
            )
        return cls(refs=materialized)

    def by_identity(self, identity: str) -> Optional[StateRef]:
        for ref in self.refs:
            if ref.identity == identity:
                return ref
        return None

    def of_placement(self, placement: Placement) -> Tuple[StateRef, ...]:
        return tuple(r for r in self.refs if r.placement == placement)

    def of_kind(self, kind: StateKind) -> Tuple[StateRef, ...]:
        return tuple(r for r in self.refs if r.kind == kind)

    def tagged(self, tag: str) -> Tuple[StateRef, ...]:
        return tuple(r for r in self.refs if tag in r.tags)

    def unique_refs(self) -> Tuple[StateRef, ...]:
        """State with no peer copy -- the checkpoint-only set."""
        return self.of_placement(Placement.UNIQUE)

    def identity_digest(self) -> str:
        """Digest of the identity set, for cross-rank agreement checks."""
        import hashlib

        joined = "\n".join(sorted(r.identity for r in self.refs))
        return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def manifest_digest(self) -> str:
        """Digest stable identity, kind, placement, shape and dtype metadata.

        Identity-only agreement is insufficient: broadcasting a 32-element
        tensor into a 64-element destination can hang a collective rather than
        raise a useful Python error.  Shape/dtype are therefore part of the
        pre-transfer contract whenever a tensor exposes them.
        """

        import hashlib
        import json

        descriptors = []
        for ref in sorted(self.refs, key=lambda item: item.identity):
            tensor = ref.tensor
            shape = getattr(tensor, "shape", ()) if tensor is not None else ()
            dtype = getattr(tensor, "dtype", None) if tensor is not None else None
            descriptors.append(
                {
                    "identity": ref.identity,
                    "kind": ref.kind.value,
                    "placement": ref.placement.value,
                    "shape": [int(size) for size in shape],
                    "dtype": "" if dtype is None else str(dtype),
                    "tags": sorted(ref.tags),
                    "metadata": dict(ref.metadata),
                }
            )
        blob = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()
