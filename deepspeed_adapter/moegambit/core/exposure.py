"""Sliding-window stale-expert exposure accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class ExposureRecord:
    step: int
    debt: int


class ExposureTracker:
    def __init__(self, window_steps: int) -> None:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self.window_steps = int(window_steps)
        self._records: Deque[ExposureRecord] = deque()
        self._debt = 0

    @property
    def debt(self) -> int:
        return self._debt

    def advance(self, step: int) -> int:
        cutoff = int(step) - self.window_steps
        while self._records and self._records[0].step <= cutoff:
            self._debt -= self._records.popleft().debt
        return self._debt

    def add(self, step: int, affected_experts: int, gap: int) -> int:
        self.advance(step)
        debt = max(0, int(affected_experts)) * max(0, int(gap))
        self._records.append(ExposureRecord(int(step), debt))
        self._debt += debt
        return self._debt

    def density(self, step: int, num_experts: int) -> float:
        if num_experts <= 0:
            return 0.0
        self.advance(step)
        return self._debt / float(num_experts * self.window_steps)
