"""Atomic publication and validation for DeepSpeed recovery checkpoints."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from moegambit.runtime.checkpoint_commit import (
    CHECKPOINT_COMMIT_NAME,
    atomic_write_json,
    atomic_write_text,
)


MANIFEST_NAME = CHECKPOINT_COMMIT_NAME
_STEP_TAG = re.compile(r"^global_step(\d+)$")
_PACKED_EXPERT_ENV = "DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT"
_PACKED_EXPERT_PATTERN = (
    "layer_*_ep_rank_*_mp_rank_*_packed_expert_states.pt"
)


def _dense_shards(tag_dir: Path, zero3: bool) -> list[Path]:
    pattern = (
        "zero_pp_rank_*_mp_rank_*_model_states.pt"
        if zero3
        else "mp_rank_*_model_states.pt"
    )
    return sorted(tag_dir.glob(pattern))


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _expected_packed_expert_shards(engine: Any) -> int:
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    layers = [
        module
        for module in engine.module.modules()
        if isinstance(module, AutoEPMoELayer)
    ]
    if not layers:
        return 0
    ep_sizes = {int(module.ep_size) for module in layers}
    if len(ep_sizes) != 1:
        raise RuntimeError(
            "packed expert checkpoint requires one AutoEP group size; "
            f"found {sorted(ep_sizes)}"
        )
    return int(engine.mp_world_size) * ep_sizes.pop() * len(layers)


def build_checkpoint_manifest(
    engine: Any,
    checkpoint_dir: Path,
    tag: str,
) -> dict[str, Any]:
    tag_dir = checkpoint_dir / tag
    zero3 = bool(engine.zero_optimization_partition_weights())
    expected = int(engine.mp_world_size)
    if zero3:
        expected *= int(engine.dp_world_size)
    shards = _dense_shards(tag_dir, zero3)
    invalid = [path.name for path in shards if path.stat().st_size <= 0]
    if len(shards) != expected or invalid:
        raise RuntimeError(
            "incomplete DeepSpeed checkpoint "
            f"{tag}: dense_shards={len(shards)}/{expected} "
            f"empty={invalid}"
        )
    manifest = {
        "format_version": 1,
        "tag": tag,
        "created_at": time.time(),
        "expected_dense_shards": expected,
        "dense_shards": {
            path.name: path.stat().st_size for path in shards
        },
    }
    packed_required = (
        _env_enabled(_PACKED_EXPERT_ENV)
        and not zero3
        and os.environ.get("MOEGAMBIT_MODEL_KIND", "moe").lower() != "dense"
    )
    if packed_required:
        packed = sorted(tag_dir.glob(_PACKED_EXPERT_PATTERN))
        expected_packed = _expected_packed_expert_shards(engine)
        invalid_packed = [
            path.name for path in packed if path.stat().st_size <= 0
        ]
        if len(packed) != expected_packed or invalid_packed:
            raise RuntimeError(
                "incomplete packed expert checkpoint "
                f"{tag}: packed_shards={len(packed)}/{expected_packed} "
                f"empty={invalid_packed}"
            )
        manifest["expected_packed_expert_shards"] = expected_packed
        manifest["packed_expert_shards"] = {
            path.name: path.stat().st_size for path in packed
        }
    return manifest


def publish_checkpoint(
    engine: Any,
    checkpoint_dir: Path,
    tag: str,
) -> None:
    import torch.distributed as dist

    outcome: list[str | None] = [None]
    if dist.get_rank() == 0:
        try:
            manifest = build_checkpoint_manifest(engine, checkpoint_dir, tag)
            tag_dir = checkpoint_dir / tag
            atomic_write_json(tag_dir / MANIFEST_NAME, manifest)
            atomic_write_text(checkpoint_dir / "latest", f"{tag}\n")
        except Exception as exc:
            outcome[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(
            f"checkpoint publication failed on rank 0: {outcome[0]}"
        )
    dist.barrier()


def validate_checkpoint_manifest(
    checkpoint_dir: Path,
    tag: str,
) -> dict[str, Any]:
    manifest_path = checkpoint_dir / tag / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"missing checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag:
        raise RuntimeError(
            f"checkpoint manifest tag mismatch: {manifest.get('tag')!r} != {tag!r}"
        )
    expected = int(manifest["expected_dense_shards"])
    recorded = manifest.get("dense_shards", {})
    missing_or_changed = []
    for name, saved_size in recorded.items():
        path = checkpoint_dir / tag / name
        if not path.is_file() or path.stat().st_size != int(saved_size):
            missing_or_changed.append(name)
    if len(recorded) != expected or missing_or_changed:
        raise RuntimeError(
            "checkpoint manifest validation failed "
            f"{tag}: dense_shards={len(recorded)}/{expected} "
            f"missing_or_changed={missing_or_changed}"
        )
    expected_packed = int(
        manifest.get("expected_packed_expert_shards", 0)
    )
    recorded_packed = manifest.get("packed_expert_shards", {})
    changed_packed = []
    for name, saved_size in recorded_packed.items():
        path = checkpoint_dir / tag / name
        if not path.is_file() or path.stat().st_size != int(saved_size):
            changed_packed.append(name)
    if (
        len(recorded_packed) != expected_packed
        or changed_packed
    ):
        raise RuntimeError(
            "checkpoint manifest validation failed "
            f"{tag}: packed_shards="
            f"{len(recorded_packed)}/{expected_packed} "
            f"missing_or_changed={changed_packed}"
        )
    return manifest


def resolve_committed_checkpoint(checkpoint_dir: Path) -> str:
    candidates: list[str] = []
    latest_path = checkpoint_dir / "latest"
    if latest_path.is_file():
        latest = latest_path.read_text(encoding="utf-8").strip()
        if latest:
            candidates.append(latest)

    committed = []
    for manifest_path in checkpoint_dir.glob(f"*/{MANIFEST_NAME}"):
        tag = manifest_path.parent.name
        match = _STEP_TAG.match(tag)
        committed.append((int(match.group(1)) if match else -1, tag))
    for _, tag in sorted(committed, reverse=True):
        if tag not in candidates:
            candidates.append(tag)

    failures = []
    for tag in candidates:
        try:
            validate_checkpoint_manifest(checkpoint_dir, tag)
            return tag
        except Exception as exc:
            failures.append(f"{tag}: {exc}")
    detail = "; ".join(failures) if failures else "no committed tags"
    raise RuntimeError(
        f"no complete DeepSpeed checkpoint in {checkpoint_dir}: {detail}"
    )


def checkpoint_step_from_tag(tag: str) -> int:
    match = _STEP_TAG.match(str(tag))
    if match is None:
        raise RuntimeError(
            "rank in-process recovery requires a global_step checkpoint "
            f"tag; got {tag!r}"
        )
    return int(match.group(1))
