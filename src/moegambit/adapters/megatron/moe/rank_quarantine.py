"""Thread-safe rank quarantine registry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

__all__ = [
    "QuarantineRecord",
    "RankQuarantineRegistry",
    "get_rank_quarantine_registry",
    "quarantine_rank",
    "release_rank",
]


@dataclass(frozen=True)
class QuarantineRecord:
    rank: int
    step: int
    reason: str
    quarantined_at_ns: int
    released_at_ns: Optional[int] = None


class RankQuarantineRegistry:
    _instance: Optional["RankQuarantineRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._active: Dict[int, QuarantineRecord] = {}
        self._history = []
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "RankQuarantineRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def quarantine(self, rank: int, step: int = -1, reason: str = "") -> bool:
        rank = int(rank)
        if rank < 0:
            raise ValueError("quarantined rank must be non-negative")
        with self._lock:
            if rank in self._active:
                return False
            record = QuarantineRecord(rank, int(step), str(reason), time.time_ns())
            self._active[rank] = record
            self._history.append(record)
            return True

    def release(self, rank: int) -> bool:
        rank = int(rank)
        with self._lock:
            record = self._active.pop(rank, None)
            if record is None:
                return False
            self._history.append(
                QuarantineRecord(
                    record.rank,
                    record.step,
                    record.reason,
                    record.quarantined_at_ns,
                    time.time_ns(),
                )
            )
            return True

    def is_quarantined(self, rank: int) -> bool:
        with self._lock:
            return int(rank) in self._active

    @property
    def quarantined_ranks(self) -> FrozenSet[int]:
        with self._lock:
            return frozenset(self._active)

    def reset(self) -> None:
        with self._lock:
            self._active.clear()
            self._history.clear()

    def summary(self) -> Mapping[str, Any]:
        return {
            "quarantined_ranks": sorted(self.quarantined_ranks),
            "active_count": len(self.quarantined_ranks),
            "history_count": len(self._history),
        }


def get_rank_quarantine_registry() -> RankQuarantineRegistry:
    return RankQuarantineRegistry.get_instance()


def quarantine_rank(
    failed_rank: int,
    step: int = -1,
    reason: str = "",
) -> bool:
    return get_rank_quarantine_registry().quarantine(failed_rank, step, reason)


def release_rank(rank: int) -> bool:
    return get_rank_quarantine_registry().release(rank)
