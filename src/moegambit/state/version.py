"""State version algebra.

Design doc section 7.3.  The core refuses the vague judgement "parameters and
optimizer each look recent enough".  A recovery plan must name one legal pair::

    (parameters@step=k, optimizer@step=k)

``StateVersion`` is ordered so that "newest consistent version" is a total
comparison rather than an ad-hoc heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = ["StateVersion", "newest_common_version"]


@dataclass(frozen=True, order=True)
class StateVersion:
    """Ordering is (committed_step, optimizer_generation, recovery_epoch).

    ``committed_step`` leads deliberately: a state is only as good as the last
    step that was actually committed, regardless of how many times the job has
    since been rebuilt.
    """

    committed_step: int
    optimizer_generation: int = 0
    recovery_epoch: int = 0

    def is_consistent_with(self, other: "StateVersion") -> bool:
        """Two refs may be restored together only at the same committed step."""
        return (
            self.committed_step == other.committed_step
            and self.optimizer_generation == other.optimizer_generation
        )


def newest_common_version(
    versions: Iterable[Iterable[StateVersion]],
) -> Optional[StateVersion]:
    """Newest version present in *every* group, or ``None``.

    Used to answer "is there a step at which all required state is available
    from some source?".  Returning ``None`` must lead to checkpoint relaunch
    rather than a partial restore.
    """
    groups = [set(group) for group in versions]
    if not groups:
        return None
    common = groups[0]
    for group in groups[1:]:
        common &= group
    return max(common) if common else None
