# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert Directory & Recovery Manifest for MOEGAMBIT-MoE.

This module provides two complementary data structures for MoE fault recovery:

1. **ActiveExpertDirectory** — a live, in-memory mapping from
   ``(layer_id, expert_id)`` to the global rank that currently hosts it,
   plus the expert's recovery state.  This is the "phone book" that the
   recovery coordinator consults to decide *where* an expert lives and
   *what state* it is in.

2. **RecoveryManifest** — a serialisable snapshot of recovery metadata that
   is saved as a **side-car JSON file** alongside the Megatron distributed
   checkpoint.  It records, for every expert, the checkpoint step, weight
   location reference, optimizer-state location reference, and recovery
   state at the time the checkpoint was taken.

Design principles
-----------------
* **No modification to the checkpoint format itself** — the manifest is a
  separate ``moegambit_manifest.json`` file written into the same checkpoint
  directory (e.g. ``iter_0001000/moegambit_manifest.json``).
* **Minimal coupling** — the directory is a pure-Python / torch-CPU data
  structure that can be tested without distributed init.
* **Integration with existing MOEGAMBIT-MoE stack**:
  - ``ExpertHealthManager`` (Step 2) owns per-expert lifecycle states.
  - ``RankQuarantineRegistry`` (Step 4) owns rank-level quarantine.
  - ``ActiveExpertDirectory`` (this module) owns the rank ↔ expert mapping
    and provides a unified query surface.
  - ``RecoveryManifest`` (this module) owns the checkpoint-time snapshot.

Checkpoint interface boundary
-----------------------------
* **Megatron dist_checkpointing** is responsible for saving / loading the
  actual model weights and optimizer states.
* **RecoveryManifest** is responsible for saving / loading the *index* that
  tells the recovery coordinator which checkpoint step, which shard file,
  and which state each expert was in.
* **ActiveExpertDirectory** is responsible for the *live* mapping that is
  rebuilt after recovery and kept in sync during training.

Typical usage::

    from moegambit.adapters.megatron.moe.expert_directory import (
        ActiveExpertDirectory,
        RecoveryManifest,
    )

    # --- During training ---
    directory = ActiveExpertDirectory.from_placement(
        num_layers=4, num_experts=8, ep_size=4,
        ep_group_ranks=[0, 2, 4, 6],
    )
    # Query: which rank hosts expert 5 in layer 2?
    rank = directory.get_host_rank(layer_id=2, expert_id=5)

    # --- At checkpoint save ---
    manifest = RecoveryManifest.from_directory(directory, step=1000,
                                               checkpoint_dir="/ckpt/iter_0001000")
    manifest.save()   # writes /ckpt/iter_0001000/moegambit_manifest.json

    # --- At recovery ---
    manifest = RecoveryManifest.load("/ckpt/iter_0001000")
    entry = manifest.get_entry(layer_id=2, expert_id=5)
    print(entry.checkpoint_step, entry.weight_location, entry.recovery_state)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from moegambit.runtime.checkpoint_commit import atomic_write_json

logger = logging.getLogger(__name__)

# Manifest file name — written as a side-car alongside Megatron checkpoint files.
MANIFEST_FILENAME = "moegambit_manifest.json"
MANIFEST_VERSION = 1


# =====================================================================
# ExpertDirectoryEntry — per-(layer, expert) metadata
# =====================================================================

@dataclass
class ExpertDirectoryEntry:
    """Metadata for a single expert in a single layer.

    This is the atomic unit of both the live directory and the persisted
    manifest.  All fields are plain Python types so that the entry can be
    trivially serialised to JSON.
    """

    layer_id: int
    """Transformer layer index that owns this MoE sub-layer."""

    expert_id: int
    """Global expert index (0-based, across all EP ranks)."""

    host_rank: int
    """Global rank that currently hosts this expert's parameters."""

    ep_rank: int = -1
    """EP-rank index within the expert-parallel group."""

    # --- Checkpoint / recovery fields ---

    checkpoint_step: int = -1
    """Training step of the latest checkpoint that contains this expert's
    weights.  ``-1`` means no checkpoint has been recorded yet."""

    weight_location: str = ""
    """Reference to the weight shard within the checkpoint directory.
    For Megatron dist_checkpointing this is typically the sharded key
    prefix, e.g. ``model.decoder.layers.2.mlp.experts.linear_fc1.weight``.
    Empty string means not yet recorded."""

    optimizer_location: str = ""
    """Reference to the optimizer-state shard.  For distributed optimizer
    this is the key prefix in the optimizer state dict.  Empty string means
    not yet recorded."""

    recovery_state: str = "HEALTHY"
    """Current recovery state as a string.  One of the ``ExpertState`` names:
    ``HEALTHY``, ``UNAVAILABLE``, ``STALE_RUNNABLE``, ``FULLY_RECOVERED``.
    Stored as string for JSON serialisation friendliness."""

    # --- Convenience ---

    def key(self) -> Tuple[int, int]:
        """Return the ``(layer_id, expert_id)`` key."""
        return (self.layer_id, self.expert_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExpertDirectoryEntry":
        """Deserialise from a plain dict."""
        return ExpertDirectoryEntry(**d)


# =====================================================================
# ActiveExpertDirectory — live in-memory mapping
# =====================================================================

class ActiveExpertDirectory:
    """Live mapping from ``(layer_id, expert_id)`` → host rank + state.

    This is the single source of truth for "where does expert X live right
    now?" during training.  It is populated once from the parallel
    configuration and updated when:

    * A rank is quarantined (experts become UNAVAILABLE).
    * A checkpoint is loaded for recovery (experts become STALE_RUNNABLE).
    * A group rebuild reassigns experts to a replacement rank.
    * A safe barrier promotes experts back to HEALTHY.

    The directory does NOT own the ``ExpertHealthMask`` or the
    ``ExpertHealthManager`` — those are separate layers in the MOEGAMBIT-MoE
    stack.  The directory is a *passive data store* that other components
    query.
    """

    def __init__(self) -> None:
        # Primary storage: (layer_id, expert_id) → ExpertDirectoryEntry
        self._entries: Dict[Tuple[int, int], ExpertDirectoryEntry] = {}
        self._num_layers: int = 0
        self._num_experts: int = 0
        self._ep_size: int = 1

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_placement(
        num_layers: int,
        num_experts: int,
        ep_size: int,
        ep_group_ranks: Optional[List[int]] = None,
    ) -> "ActiveExpertDirectory":
        """Build a directory from the parallel configuration.

        Args:
            num_layers: Number of MoE layers (only layers that have MoE).
            num_experts: Total number of global experts per layer.
            ep_size: Expert-parallel world size.
            ep_group_ranks: Ordered list of global ranks in the EP group.
                If ``None``, defaults to ``[0, 1, ..., ep_size-1]``.

        Returns:
            A fully populated ``ActiveExpertDirectory``.
        """
        if ep_group_ranks is None:
            ep_group_ranks = list(range(ep_size))
        assert len(ep_group_ranks) == ep_size
        assert num_experts % ep_size == 0
        num_local = num_experts // ep_size

        directory = ActiveExpertDirectory()
        directory._num_layers = num_layers
        directory._num_experts = num_experts
        directory._ep_size = ep_size

        for layer_id in range(num_layers):
            for ep_rank_idx, global_rank in enumerate(ep_group_ranks):
                for local_idx in range(num_local):
                    expert_id = ep_rank_idx * num_local + local_idx
                    entry = ExpertDirectoryEntry(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        host_rank=global_rank,
                        ep_rank=ep_rank_idx,
                        recovery_state="HEALTHY",
                    )
                    directory._entries[(layer_id, expert_id)] = entry

        logger.info(
            "ActiveExpertDirectory: built %d entries "
            "(%d layers × %d experts, ep_size=%d, ranks=%s)",
            len(directory._entries), num_layers, num_experts, ep_size, ep_group_ranks,
        )
        return directory

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_entry(self, layer_id: int, expert_id: int) -> Optional[ExpertDirectoryEntry]:
        """Return the directory entry, or ``None`` if not found."""
        return self._entries.get((layer_id, expert_id))

    def get_host_rank(self, layer_id: int, expert_id: int) -> int:
        """Return the global rank hosting the given expert.

        Raises:
            KeyError: If the ``(layer_id, expert_id)`` pair is not in the directory.
        """
        entry = self._entries.get((layer_id, expert_id))
        if entry is None:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not in directory")
        return entry.host_rank

    def get_recovery_state(self, layer_id: int, expert_id: int) -> str:
        """Return the recovery state string for the given expert."""
        entry = self._entries.get((layer_id, expert_id))
        if entry is None:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not in directory")
        return entry.recovery_state

    def experts_on_rank(self, global_rank: int) -> List[Tuple[int, int]]:
        """Return all ``(layer_id, expert_id)`` pairs hosted on the given rank."""
        return [
            (e.layer_id, e.expert_id)
            for e in self._entries.values()
            if e.host_rank == global_rank
        ]

    def experts_in_state(self, state: str) -> List[Tuple[int, int]]:
        """Return all ``(layer_id, expert_id)`` pairs in the given state."""
        return [
            (e.layer_id, e.expert_id)
            for e in self._entries.values()
            if e.recovery_state == state
        ]

    def all_entries(self) -> List[ExpertDirectoryEntry]:
        """Return all entries as a flat list (sorted by layer, expert)."""
        return [self._entries[k] for k in sorted(self._entries.keys())]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def ep_size(self) -> int:
        return self._ep_size

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary."""
        state_counts: Dict[str, int] = {}
        for e in self._entries.values():
            state_counts[e.recovery_state] = state_counts.get(e.recovery_state, 0) + 1
        return {
            "num_layers": self._num_layers,
            "num_experts": self._num_experts,
            "ep_size": self._ep_size,
            "total_entries": len(self._entries),
            "state_counts": state_counts,
        }

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def update_recovery_state(
        self, layer_id: int, expert_id: int, new_state: str
    ) -> None:
        """Update the recovery state for a single expert."""
        entry = self._entries.get((layer_id, expert_id))
        if entry is None:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not in directory")
        old = entry.recovery_state
        entry.recovery_state = new_state
        logger.debug(
            "Directory: expert (layer=%d, id=%d) state %s → %s",
            layer_id, expert_id, old, new_state,
        )

    def update_host_rank(
        self, layer_id: int, expert_id: int, new_rank: int
    ) -> None:
        """Update the host rank for a single expert (after group rebuild)."""
        entry = self._entries.get((layer_id, expert_id))
        if entry is None:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not in directory")
        old = entry.host_rank
        entry.host_rank = new_rank
        logger.debug(
            "Directory: expert (layer=%d, id=%d) host rank %d → %d",
            layer_id, expert_id, old, new_rank,
        )

    def update_checkpoint_info(
        self,
        layer_id: int,
        expert_id: int,
        checkpoint_step: int,
        weight_location: str = "",
        optimizer_location: str = "",
    ) -> None:
        """Record checkpoint metadata for a single expert."""
        entry = self._entries.get((layer_id, expert_id))
        if entry is None:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not in directory")
        entry.checkpoint_step = checkpoint_step
        if weight_location:
            entry.weight_location = weight_location
        if optimizer_location:
            entry.optimizer_location = optimizer_location

    def bulk_update_recovery_state(
        self, expert_ids: List[int], new_state: str, *, layer_id: Optional[int] = None
    ) -> None:
        """Update recovery state for multiple experts.

        Args:
            expert_ids: List of global expert IDs.
            new_state: Target state string.
            layer_id: If given, only update that layer.  If ``None``, update
                all layers.
        """
        layers = [layer_id] if layer_id is not None else list(range(self._num_layers))
        for lid in layers:
            for eid in expert_ids:
                key = (lid, eid)
                if key in self._entries:
                    self._entries[key].recovery_state = new_state

    def bulk_update_host_rank(
        self, global_rank: int, new_rank: int
    ) -> int:
        """Reassign all experts from ``global_rank`` to ``new_rank``.

        Returns:
            Number of entries updated.
        """
        count = 0
        for entry in self._entries.values():
            if entry.host_rank == global_rank:
                entry.host_rank = new_rank
                count += 1
        if count > 0:
            logger.info(
                "Directory: reassigned %d entries from rank %d → rank %d",
                count, global_rank, new_rank,
            )
        return count

    def refresh_from_placement(
        self,
        ep_group_ranks: List[int],
    ) -> None:
        """Refresh host-rank mapping after a group rebuild.

        This re-computes the ``host_rank`` and ``ep_rank`` fields for every
        entry based on the new EP group membership, while preserving all
        other metadata (checkpoint_step, recovery_state, etc.).

        Args:
            ep_group_ranks: New ordered list of global ranks in the EP group.
        """
        assert len(ep_group_ranks) == self._ep_size, (
            f"ep_group_ranks length ({len(ep_group_ranks)}) != ep_size ({self._ep_size})"
        )
        num_local = self._num_experts // self._ep_size

        for (layer_id, expert_id), entry in self._entries.items():
            ep_rank_idx = expert_id // num_local
            entry.ep_rank = ep_rank_idx
            entry.host_rank = ep_group_ranks[ep_rank_idx]

        logger.info(
            "Directory: refreshed placement for %d entries with new ranks %s",
            len(self._entries), ep_group_ranks,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ActiveExpertDirectory(layers={self._num_layers}, "
            f"experts={self._num_experts}, ep_size={self._ep_size}, "
            f"entries={len(self._entries)})"
        )


# =====================================================================
# RecoveryManifest — serialisable checkpoint-time snapshot
# =====================================================================

class RecoveryManifest:
    """Serialisable snapshot of expert recovery metadata.

    The manifest is saved as ``moegambit_manifest.json`` inside the Megatron
    checkpoint directory.  It contains:

    * Global metadata (version, step, parallel config).
    * Per-expert entries with checkpoint step, weight/optimizer location
      references, host rank, and recovery state.

    The manifest does NOT contain the actual weights or optimizer states —
    those are stored by Megatron's distributed checkpoint system.  The
    manifest is an *index* that tells the recovery coordinator where to
    find each expert's data and what state it was in.
    """

    def __init__(
        self,
        step: int = -1,
        checkpoint_dir: str = "",
        entries: Optional[List[ExpertDirectoryEntry]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.version: int = MANIFEST_VERSION
        self.step: int = step
        self.checkpoint_dir: str = checkpoint_dir
        self.entries: List[ExpertDirectoryEntry] = entries or []
        self.metadata: Dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Construction from directory
    # ------------------------------------------------------------------

    @staticmethod
    def from_directory(
        directory: ActiveExpertDirectory,
        step: int,
        checkpoint_dir: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RecoveryManifest":
        """Create a manifest from the current state of an ActiveExpertDirectory.

        This is the primary way to create a manifest at checkpoint-save time.

        Args:
            directory: The live expert directory.
            step: Current training step (= checkpoint iteration).
            checkpoint_dir: Path to the checkpoint directory
                (e.g. ``/path/to/checkpoints/iter_0001000``).
            extra_metadata: Optional dict of additional metadata to include.

        Returns:
            A ``RecoveryManifest`` ready to be saved.
        """
        entries = directory.all_entries()
        # Stamp every entry with the checkpoint step
        for e in entries:
            if e.checkpoint_step < 0:
                e.checkpoint_step = step

        meta: Dict[str, Any] = {
            "num_layers": directory.num_layers,
            "num_experts": directory.num_experts,
            "ep_size": directory.ep_size,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return RecoveryManifest(
            step=step,
            checkpoint_dir=checkpoint_dir,
            entries=entries,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire manifest to a JSON-safe dict."""
        return {
            "version": self.version,
            "step": self.step,
            "checkpoint_dir": self.checkpoint_dir,
            "metadata": self.metadata,
            "entries": [e.to_dict() for e in self.entries],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RecoveryManifest":
        """Deserialise from a dict (as loaded from JSON)."""
        version = d.get("version", 1)
        if version != MANIFEST_VERSION:
            logger.warning(
                "RecoveryManifest version mismatch: expected %d, got %d",
                MANIFEST_VERSION, version,
            )
        entries = [ExpertDirectoryEntry.from_dict(e) for e in d.get("entries", [])]
        return RecoveryManifest(
            step=d.get("step", -1),
            checkpoint_dir=d.get("checkpoint_dir", ""),
            entries=entries,
            metadata=d.get("metadata", {}),
        )

    def save(self, checkpoint_dir: Optional[str] = None) -> str:
        """Save the manifest as JSON to the checkpoint directory.

        Args:
            checkpoint_dir: Override the checkpoint directory.  If ``None``,
                uses ``self.checkpoint_dir``.

        Returns:
            The full path to the saved manifest file.
        """
        target_dir = checkpoint_dir or self.checkpoint_dir
        if not target_dir:
            raise ValueError("No checkpoint_dir specified for manifest save")

        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, MANIFEST_FILENAME)

        atomic_write_json(Path(path), self.to_dict())

        logger.info(
            "RecoveryManifest: saved %d entries to %s (step=%d)",
            len(self.entries), path, self.step,
        )
        return path

    @staticmethod
    def load(checkpoint_dir: str) -> "RecoveryManifest":
        """Load a manifest from a checkpoint directory.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        Returns:
            The loaded ``RecoveryManifest``.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
        """
        path = os.path.join(checkpoint_dir, MANIFEST_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No manifest found at {path}")

        with open(path, "r") as f:
            data = json.load(f)

        manifest = RecoveryManifest.from_dict(data)
        # Override checkpoint_dir with the actual load path
        manifest.checkpoint_dir = checkpoint_dir
        logger.info(
            "RecoveryManifest: loaded %d entries from %s (step=%d)",
            len(manifest.entries), path, manifest.step,
        )
        return manifest

    @staticmethod
    def exists(checkpoint_dir: str) -> bool:
        """Check if a manifest file exists in the given directory."""
        return os.path.exists(os.path.join(checkpoint_dir, MANIFEST_FILENAME))

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_entry(self, layer_id: int, expert_id: int) -> Optional[ExpertDirectoryEntry]:
        """Find the entry for a specific ``(layer_id, expert_id)``."""
        for e in self.entries:
            if e.layer_id == layer_id and e.expert_id == expert_id:
                return e
        return None

    def get_entries_for_layer(self, layer_id: int) -> List[ExpertDirectoryEntry]:
        """Return all entries for a given layer."""
        return [e for e in self.entries if e.layer_id == layer_id]

    def get_entries_for_expert(self, expert_id: int) -> List[ExpertDirectoryEntry]:
        """Return entries for a given expert across all layers."""
        return [e for e in self.entries if e.expert_id == expert_id]

    def get_entries_in_state(self, state: str) -> List[ExpertDirectoryEntry]:
        """Return all entries in the given recovery state."""
        return [e for e in self.entries if e.recovery_state == state]

    def get_recovery_source(
        self, layer_id: int, expert_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return the recovery source info for a specific expert.

        This is the primary query for the recovery coordinator: given a
        ``(layer_id, expert_id)`` that needs to be restored, return the
        checkpoint step and location references.

        Returns:
            A dict with keys ``checkpoint_step``, ``checkpoint_dir``,
            ``weight_location``, ``optimizer_location``, or ``None`` if
            the expert is not in the manifest.
        """
        entry = self.get_entry(layer_id, expert_id)
        if entry is None:
            return None
        return {
            "checkpoint_step": entry.checkpoint_step,
            "checkpoint_dir": self.checkpoint_dir,
            "weight_location": entry.weight_location,
            "optimizer_location": entry.optimizer_location,
            "host_rank": entry.host_rank,
            "recovery_state": entry.recovery_state,
        }

    def get_stale_experts(self) -> List[Tuple[int, int]]:
        """Return ``(layer_id, expert_id)`` pairs that need recovery.

        An expert "needs recovery" if its ``recovery_state`` is
        ``UNAVAILABLE`` or ``STALE_RUNNABLE``.
        """
        return [
            (e.layer_id, e.expert_id)
            for e in self.entries
            if e.recovery_state in ("UNAVAILABLE", "STALE_RUNNABLE")
        ]

    # ------------------------------------------------------------------
    # Populate weight locations from Megatron key conventions
    # ------------------------------------------------------------------

    def populate_weight_locations(
        self,
        model_prefix: str = "model.decoder.layers",
        expert_key_pattern: str = "{prefix}.{layer_id}.mlp.experts.linear_fc1.weight",
        optimizer_key_pattern: str = "",
    ) -> None:
        """Fill in ``weight_location`` and ``optimizer_location`` fields
        using Megatron's standard key naming conventions.

        This uses the non-singleton sharded key pattern where all experts
        in a layer share a single ShardedTensor with an expert-dimension
        offset.  The location string encodes the key prefix and the expert's
        offset within the sharded tensor.

        Args:
            model_prefix: Prefix for the model state dict keys.
            expert_key_pattern: Pattern for weight keys.  ``{prefix}`` is
                replaced with ``model_prefix``, ``{layer_id}`` with the
                layer index.
            optimizer_key_pattern: Pattern for optimizer keys.  If empty,
                optimizer_location is set to a default based on weight_location.
        """
        for entry in self.entries:
            weight_key = expert_key_pattern.format(
                prefix=model_prefix, layer_id=entry.layer_id,
            )
            entry.weight_location = f"{weight_key}[expert_offset={entry.expert_id}]"

            if optimizer_key_pattern:
                opt_key = optimizer_key_pattern.format(
                    prefix=model_prefix, layer_id=entry.layer_id,
                )
                entry.optimizer_location = f"{opt_key}[expert_offset={entry.expert_id}]"
            else:
                entry.optimizer_location = f"optimizer.state.{weight_key}[expert_offset={entry.expert_id}]"

    # ------------------------------------------------------------------
    # Apply manifest to a live directory
    # ------------------------------------------------------------------

    def apply_to_directory(self, directory: ActiveExpertDirectory) -> int:
        """Apply manifest entries to a live directory.

        Updates checkpoint_step, weight_location, optimizer_location, and
        recovery_state for every entry that exists in both the manifest and
        the directory.

        Returns:
            Number of entries updated.
        """
        count = 0
        for me in self.entries:
            de = directory.get_entry(me.layer_id, me.expert_id)
            if de is None:
                continue
            de.checkpoint_step = me.checkpoint_step
            de.weight_location = me.weight_location
            de.optimizer_location = me.optimizer_location
            de.recovery_state = me.recovery_state
            count += 1
        logger.info(
            "RecoveryManifest: applied %d entries to directory (step=%d)",
            count, self.step,
        )
        return count

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RecoveryManifest(step={self.step}, "
            f"entries={len(self.entries)}, "
            f"dir={self.checkpoint_dir!r})"
        )


# =====================================================================
# Global singleton for the active directory
# =====================================================================

_ACTIVE_DIRECTORY: Optional[ActiveExpertDirectory] = None


def get_active_expert_directory() -> Optional[ActiveExpertDirectory]:
    """Return the global ``ActiveExpertDirectory``, or ``None`` if not set."""
    return _ACTIVE_DIRECTORY


def set_active_expert_directory(directory: ActiveExpertDirectory) -> None:
    """Set the global ``ActiveExpertDirectory``."""
    global _ACTIVE_DIRECTORY
    _ACTIVE_DIRECTORY = directory
    logger.info("Global ActiveExpertDirectory set: %s", directory)


def clear_active_expert_directory() -> None:
    """Clear the global directory (for testing)."""
    global _ACTIVE_DIRECTORY
    _ACTIVE_DIRECTORY = None
