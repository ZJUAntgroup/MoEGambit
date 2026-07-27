"""Stable locators for optimizer snapshots held by another worker."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse

__all__ = ["MemoryReplicaLocator"]


@dataclass(frozen=True)
class MemoryReplicaLocator:
    generation: int
    owner_rank: int
    holder_rank: int
    identity: str

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("replica generation must be non-negative")
        if self.owner_rank < 0 or self.holder_rank < 0:
            raise ValueError("replica ranks must be non-negative")
        if not self.identity:
            raise ValueError("replica identity must be non-empty")

    def __str__(self) -> str:
        identity = quote(self.identity, safe="")
        return (
            f"memory://generation/{self.generation}/owner/{self.owner_rank}/"
            f"holder/{self.holder_rank}/state/{identity}"
        )

    @classmethod
    def parse(cls, value: str) -> "MemoryReplicaLocator":
        parsed = urlparse(value)
        if parsed.scheme != "memory" or parsed.netloc != "generation":
            raise ValueError(f"invalid memory replica locator: {value!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if (
            len(parts) < 7
            or parts[1] != "owner"
            or parts[3] != "holder"
            or parts[5] != "state"
        ):
            raise ValueError(f"invalid memory replica locator: {value!r}")
        # urlparse places ``generation`` in netloc and the generation value in
        # the first path component.
        return cls(
            generation=int(parts[0]),
            owner_rank=int(parts[2]),
            holder_rank=int(parts[4]),
            identity=unquote("/".join(parts[6:])),
        )
