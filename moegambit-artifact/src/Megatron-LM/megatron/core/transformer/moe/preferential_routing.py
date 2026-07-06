# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# MoEGambit: Preferential Routing for Recovered Experts
#
# After an expert is restored from checkpoint and transitions to
# STALE_RUNNABLE / FULLY_RECOVERED, it has been absent from training for
# some time.  This module optionally applies a small, time-windowed,
# linearly-decayed routing bias to the recovered expert's sigmoid scores
# so that the router preferentially routes a few more tokens to it,
# accelerating its reintegration into the active training workload.
#
# Design principles:
#   1. Opt-in only — disabled by default.
#   2. Bounded — bias decays linearly to zero over a configurable window.
#   3. Non-destructive — does not modify capacity factors, aux loss, or
#      the existing DeepSeek-V3 expert_bias mechanism.
#   4. Per-layer, per-expert granularity.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-expert bias session
# ---------------------------------------------------------------------------

@dataclass
class _ExpertBiasSession:
    """Tracks the bias lifecycle for a single recovered expert."""

    expert_id: int
    """Global expert ID within the layer."""

    initial_bias: float
    """Starting magnitude of the additive bias."""

    window_steps: int
    """Total number of training steps the bias should remain active."""

    activation_step: int
    """Global training step at which the bias was activated."""

    # ---- derived / mutable ----
    elapsed_steps: int = 0
    """Number of steps elapsed since activation."""

    active: bool = True
    """Whether this session is still producing non-zero bias."""

    def current_bias(self) -> float:
        """Return the linearly-decayed bias for the current step."""
        if not self.active or self.window_steps <= 0:
            return 0.0
        remaining_frac = max(0.0, 1.0 - self.elapsed_steps / self.window_steps)
        return self.initial_bias * remaining_frac

    def step(self) -> None:
        """Advance one training step and expire if window exhausted."""
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.window_steps:
            self.active = False

    def remaining_steps(self) -> int:
        return max(0, self.window_steps - self.elapsed_steps)


# ---------------------------------------------------------------------------
# Per-layer manager
# ---------------------------------------------------------------------------

class PreferentialRoutingManager:
    """Manages recovery-bias sessions for one transformer layer.

    Typical lifecycle:
        1. ``activate(expert_id, step)`` — called when an expert transitions
           to STALE_RUNNABLE.
        2. ``get_bias_tensor(device)`` — called by the router every forward
           pass to obtain the additive bias vector ``[num_experts]``.
        3. ``step_all()`` — called once per global batch (from
           ``finalize_model_grads``) to advance the decay.
        4. Bias auto-expires after ``window_steps``; no manual deactivation
           needed (though ``deactivate`` is provided for early cancellation).
    """

    def __init__(
        self,
        layer_number: int,
        num_experts: int,
        initial_bias: float = 0.1,
        window_steps: int = 100,
    ):
        self.layer_number = layer_number
        self.num_experts = num_experts
        self.initial_bias = initial_bias
        self.window_steps = window_steps

        # expert_id → session
        self._sessions: Dict[int, _ExpertBiasSession] = {}

        # Cached tensor — invalidated on activate / deactivate / step
        self._cached_tensor: Optional[torch.Tensor] = None
        self._cache_device: Optional[torch.device] = None
        self._cache_dirty: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(self, expert_id: int, step: int = 0) -> None:
        """Start a new bias session for *expert_id*.

        If a session already exists and is still active, it is replaced
        (restarted).
        """
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(
                f"expert_id {expert_id} out of range [0, {self.num_experts})"
            )
        self._sessions[expert_id] = _ExpertBiasSession(
            expert_id=expert_id,
            initial_bias=self.initial_bias,
            window_steps=self.window_steps,
            activation_step=step,
        )
        self._cache_dirty = True
        logger.info(
            "MoEGambit layer %d: preferential routing activated for expert %d "
            "(bias=%.4f, window=%d steps, step=%d)",
            self.layer_number, expert_id, self.initial_bias, self.window_steps, step,
        )

    def deactivate(self, expert_id: int) -> None:
        """Immediately cancel the bias session for *expert_id*."""
        session = self._sessions.pop(expert_id, None)
        if session is not None:
            self._cache_dirty = True
            logger.info(
                "MoEGambit layer %d: preferential routing deactivated for expert %d",
                self.layer_number, expert_id,
            )

    def get_bias_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return a ``[num_experts]`` float32 tensor of additive biases.

        Returns ``None`` if no sessions are active (fast path — avoids
        allocation and addition in the router hot path).
        """
        # Prune expired sessions
        expired = [eid for eid, s in self._sessions.items() if not s.active]
        for eid in expired:
            del self._sessions[eid]
            self._cache_dirty = True

        if not self._sessions:
            self._cached_tensor = None
            return None

        if self._cache_dirty or self._cache_device != device:
            bias = torch.zeros(self.num_experts, dtype=torch.float32, device=device)
            for eid, session in self._sessions.items():
                bias[eid] = session.current_bias()
            self._cached_tensor = bias
            self._cache_device = device
            self._cache_dirty = False

        return self._cached_tensor

    def step_all(self) -> None:
        """Advance all active sessions by one training step.

        Called once per global batch from ``finalize_model_grads``.
        """
        if not self._sessions:
            return
        for session in self._sessions.values():
            if session.active:
                session.step()
        self._cache_dirty = True

    def has_active_sessions(self) -> bool:
        """Return ``True`` if at least one session is still active."""
        return any(s.active for s in self._sessions.values())

    def get_active_expert_ids(self) -> List[int]:
        """Return sorted list of expert IDs with active bias sessions."""
        return sorted(eid for eid, s in self._sessions.items() if s.active)

    def get_session_info(self, expert_id: int) -> Optional[Dict]:
        """Return diagnostic info for one expert's session, or ``None``."""
        session = self._sessions.get(expert_id)
        if session is None:
            return None
        return {
            "expert_id": session.expert_id,
            "initial_bias": session.initial_bias,
            "current_bias": session.current_bias(),
            "elapsed_steps": session.elapsed_steps,
            "remaining_steps": session.remaining_steps(),
            "active": session.active,
            "activation_step": session.activation_step,
        }

    def summary(self) -> Dict:
        """Return a summary dict for logging / diagnostics."""
        active = []
        for eid, s in sorted(self._sessions.items()):
            if s.active:
                active.append({
                    "expert_id": eid,
                    "current_bias": round(s.current_bias(), 6),
                    "remaining_steps": s.remaining_steps(),
                })
        return {
            "layer_number": self.layer_number,
            "num_active_sessions": len(active),
            "sessions": active,
        }

    def reset(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()
        self._cached_tensor = None
        self._cache_dirty = True


# ---------------------------------------------------------------------------
# Global registry — one manager per (layer_number)
# ---------------------------------------------------------------------------

_MANAGER_REGISTRY: Dict[int, PreferentialRoutingManager] = {}


def get_preferential_routing_manager(
    layer_number: int,
    num_experts: int,
    initial_bias: float = 0.1,
    window_steps: int = 100,
) -> PreferentialRoutingManager:
    """Get or create the :class:`PreferentialRoutingManager` for *layer_number*.

    The manager is created on first access and cached for the lifetime of the
    training run (or until ``clear_preferential_routing_managers`` is called).
    """
    if layer_number not in _MANAGER_REGISTRY:
        _MANAGER_REGISTRY[layer_number] = PreferentialRoutingManager(
            layer_number=layer_number,
            num_experts=num_experts,
            initial_bias=initial_bias,
            window_steps=window_steps,
        )
    return _MANAGER_REGISTRY[layer_number]


def get_all_preferential_routing_managers() -> Dict[int, PreferentialRoutingManager]:
    """Return the full registry (for use in ``finalize_model_grads``)."""
    return _MANAGER_REGISTRY


def clear_preferential_routing_managers() -> None:
    """Clear all managers (for testing or shutdown)."""
    _MANAGER_REGISTRY.clear()
