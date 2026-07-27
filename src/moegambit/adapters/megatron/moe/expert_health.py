"""Compatibility health-mask facade backed by ExpertHealthManager."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .expert_health_manager import get_expert_health_manager

__all__ = ["ExpertHealthMask", "get_expert_health_mask", "clear_health_masks"]


class ExpertHealthMask:
    def __init__(self, layer_id: int, num_experts: int, device: Any = None) -> None:
        self.manager = get_expert_health_manager(
            layer_id, num_experts, device=device
        )

    def mark_unavailable(self, expert_indices: Iterable[int], step: int = -1):
        return self.manager.mark_unavailable(expert_indices, step)

    def mark_healthy(self, expert_indices: Iterable[int], step: int = -1):
        states = [self.manager.get_state(int(index)) for index in expert_indices]
        # Recovery may update a legacy mask directly from UNAVAILABLE.  Preserve
        # the four-state invariant by advancing through intermediate states.
        ids = [int(index) for index in expert_indices]
        for expert_id, state in zip(ids, states):
            if state.name == "UNAVAILABLE":
                self.manager.mark_stale_runnable([expert_id], step)
                self.manager.mark_fully_recovered([expert_id], step)
            elif state.name == "STALE_RUNNABLE":
                self.manager.mark_fully_recovered([expert_id], step)
            self.manager.mark_healthy([expert_id], step)

    @property
    def mask(self):
        return self.manager.get_routable_mask()


_MASKS = {}


def get_expert_health_mask(
    layer_id: int,
    num_experts: int,
    device: Any = None,
) -> ExpertHealthMask:
    key = int(layer_id)
    existing = _MASKS.get(key)
    if existing is not None:
        return existing
    if num_experts <= 0:
        manager = get_expert_health_manager(key)
        num_experts = manager.num_experts
    mask = ExpertHealthMask(key, int(num_experts), device=device)
    _MASKS[key] = mask
    return mask


def clear_health_masks() -> None:
    _MASKS.clear()
