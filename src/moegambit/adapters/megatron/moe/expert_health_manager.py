"""Per-layer expert health state machine used by routing and recovery."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "ExpertState",
    "ExpertTransition",
    "ExpertHealthManager",
    "get_expert_health_manager",
    "clear_manager_registry",
]


class ExpertState(enum.IntEnum):
    HEALTHY = 0
    UNAVAILABLE = 1
    STALE_RUNNABLE = 2
    FULLY_RECOVERED = 3


@dataclass(frozen=True)
class ExpertTransition:
    expert_id: int
    previous: ExpertState
    current: ExpertState
    step: int


_ALLOWED = {
    ExpertState.HEALTHY: frozenset({ExpertState.HEALTHY, ExpertState.UNAVAILABLE}),
    ExpertState.UNAVAILABLE: frozenset(
        {ExpertState.UNAVAILABLE, ExpertState.STALE_RUNNABLE}
    ),
    ExpertState.STALE_RUNNABLE: frozenset(
        {ExpertState.STALE_RUNNABLE, ExpertState.FULLY_RECOVERED}
    ),
    ExpertState.FULLY_RECOVERED: frozenset(
        {ExpertState.FULLY_RECOVERED, ExpertState.HEALTHY, ExpertState.UNAVAILABLE}
    ),
}


class ExpertHealthManager:
    _instances: Dict[int, "ExpertHealthManager"] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        layer_number: int,
        num_experts: int,
        *,
        device: Any = None,
    ) -> None:
        if layer_number < 0:
            raise ValueError("layer_number must be non-negative")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.layer_number = int(layer_number)
        self.num_experts = int(num_experts)
        self.device = device
        self._states = [ExpertState.HEALTHY] * self.num_experts
        self._last_step = [-1] * self.num_experts
        self._history = []
        self._lock = threading.RLock()

    @classmethod
    def get_instance(
        cls,
        layer_number: int,
        num_experts: Optional[int] = None,
        *,
        device: Any = None,
    ) -> "ExpertHealthManager":
        with cls._instances_lock:
            existing = cls._instances.get(int(layer_number))
            if existing is not None:
                if num_experts is not None and existing.num_experts != int(num_experts):
                    raise ValueError("existing health manager has a different expert count")
                return existing
            if num_experts is None:
                raise ValueError("num_experts is required when creating a manager")
            manager = cls(layer_number, num_experts, device=device)
            cls._instances[int(layer_number)] = manager
            return manager

    def _transition(
        self,
        expert_ids: Iterable[int],
        target: ExpertState,
        step: int,
    ) -> Tuple[ExpertTransition, ...]:
        transitions = []
        with self._lock:
            for raw_id in expert_ids:
                expert_id = int(raw_id)
                if expert_id < 0 or expert_id >= self.num_experts:
                    raise IndexError(f"expert id {expert_id} is out of range")
                previous = self._states[expert_id]
                if target not in _ALLOWED[previous]:
                    raise ValueError(
                        f"invalid expert transition {previous.name} -> {target.name}"
                    )
                self._states[expert_id] = target
                self._last_step[expert_id] = int(step)
                transition = ExpertTransition(expert_id, previous, target, int(step))
                transitions.append(transition)
                self._history.append(transition)
        return tuple(transitions)

    def mark_unavailable(self, expert_ids: Iterable[int], step: int = -1):
        return self._transition(expert_ids, ExpertState.UNAVAILABLE, step)

    def mark_stale_runnable(self, expert_ids: Iterable[int], step: int = -1):
        return self._transition(expert_ids, ExpertState.STALE_RUNNABLE, step)

    def mark_fully_recovered(self, expert_ids: Iterable[int], step: int = -1):
        return self._transition(expert_ids, ExpertState.FULLY_RECOVERED, step)

    def mark_healthy(self, expert_ids: Iterable[int], step: int = -1):
        return self._transition(expert_ids, ExpertState.HEALTHY, step)

    def get_state(self, expert_id: int) -> ExpertState:
        with self._lock:
            return self._states[int(expert_id)]

    def get_routable_mask(self):
        values = [state is not ExpertState.UNAVAILABLE for state in self._states]
        try:
            import torch

            return torch.tensor(values, dtype=torch.bool, device=self.device)
        except ImportError:
            return tuple(values)

    def get_layer_summary(self) -> Mapping[str, Any]:
        with self._lock:
            counts = {
                state.name: sum(item is state for item in self._states)
                for state in ExpertState
            }
            return {
                "layer_number": self.layer_number,
                "num_experts": self.num_experts,
                "counts": counts,
                "states": [state.name for state in self._states],
                "last_steps": list(self._last_step),
            }

    @property
    def history(self) -> Tuple[ExpertTransition, ...]:
        return tuple(self._history)


_MANAGER_REGISTRY = ExpertHealthManager._instances


def get_expert_health_manager(
    layer_number: int,
    num_experts: Optional[int] = None,
    *,
    device: Any = None,
) -> ExpertHealthManager:
    return ExpertHealthManager.get_instance(
        layer_number, num_experts, device=device
    )


def clear_manager_registry() -> None:
    with ExpertHealthManager._instances_lock:
        ExpertHealthManager._instances.clear()
