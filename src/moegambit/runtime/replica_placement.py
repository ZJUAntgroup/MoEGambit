"""Failure-domain-aware placement for in-memory recovery replicas."""

from __future__ import annotations

from collections.abc import Iterable


class ReplicaPlacementError(RuntimeError):
    pass


def failure_domain_ring_order(
    group_ranks: Iterable[int],
    *,
    ranks_per_failure_domain: int,
) -> list[int]:
    """Return a deterministic ring whose successor is on another node.

    ``ranks_per_failure_domain`` is normally ``LOCAL_WORLD_SIZE``. Sorting by
    local rank first interleaves physical nodes for both flat DP groups and
    pipeline-stage DP groups.
    """
    domain_size = int(ranks_per_failure_domain)
    if domain_size <= 0:
        raise ReplicaPlacementError(
            "ranks_per_failure_domain must be positive"
        )
    ranks = sorted({int(rank) for rank in group_ranks})
    if len(ranks) < 2:
        raise ReplicaPlacementError(
            "optimizer replication requires at least two ranks"
        )

    ring = sorted(
        ranks,
        key=lambda rank: (
            rank % domain_size,
            rank // domain_size,
            rank,
        ),
    )
    collisions = []
    for index, owner in enumerate(ring):
        holder = ring[(index + 1) % len(ring)]
        if owner // domain_size == holder // domain_size:
            collisions.append((owner, holder, owner // domain_size))
    if collisions:
        raise ReplicaPlacementError(
            "cannot place every optimizer replica outside its owner's "
            f"failure domain: ranks={ranks} domain_size={domain_size} "
            f"collisions={collisions}"
        )
    return ring
