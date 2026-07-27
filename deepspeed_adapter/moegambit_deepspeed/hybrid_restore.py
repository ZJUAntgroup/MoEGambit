"""Single-stage MoE recovery from checkpoint and a compatible DP peer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


class DeepSpeedHybridRestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeerRestorePlan:
    replacement_rank: int
    source_rank: int
    data_parallel_ranks: tuple[int, ...]


class _FlatDataParallelTopology:
    def __init__(self, world_size: int) -> None:
        self._world_size = int(world_size)

    def get_coord(self, rank: int) -> Any:
        if not 0 <= rank < self._world_size:
            raise ValueError(f"rank {rank} is outside the data topology")
        return type("DataCoord", (), {"data": rank})()

    def get_axis_names(self) -> tuple[str, ...]:
        return ("data",)

    def filter_match(self, **filters: int) -> list[int]:
        if filters:
            raise ValueError(
                f"flat data topology has no filter axes: {filters}"
            )
        return list(range(self._world_size))


def is_expert_parameter(parameter: Any) -> bool:
    return (
        getattr(parameter, "ds_zero_placement_family", "replicated")
        == "autoep_expert"
    )


def _is_expert_buffer_name(name: str) -> bool:
    # AutoEP currently stores expert weights as Parameters. Keep the buffer
    # rule explicit so future expert-local buffers are not copied as dense
    # state by accident.
    return ".experts." in f".{name}."


def non_expert_model_tensors(module: Any) -> list[tuple[str, Any]]:
    tensors = [
        (name, parameter.data)
        for name, parameter in module.named_parameters()
        if not is_expert_parameter(parameter)
    ]
    persistent_names = set(module.state_dict().keys())
    tensors.extend(
        (name, buffer)
        for name, buffer in module.named_buffers()
        if name in persistent_names and not _is_expert_buffer_name(name)
    )
    tensors.sort(key=lambda item: item[0])
    return tensors


def _tensor_manifest(
    tensors: Iterable[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    manifest = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in tensors
    ]
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return manifest, hashlib.sha256(encoded).hexdigest()


def build_peer_restore_plans(
    topology: Any,
    replacement_ranks: Iterable[int],
) -> list[PeerRestorePlan]:
    replacement_set = {int(rank) for rank in replacement_ranks}
    if not replacement_set:
        raise DeepSpeedHybridRestoreError("replacement rank set is empty")

    plans = []
    for replacement_rank in sorted(replacement_set):
        coord = topology.get_coord(replacement_rank)
        filters = {
            axis: getattr(coord, axis)
            for axis in topology.get_axis_names()
            if axis != "data"
        }
        data_parallel_ranks = tuple(
            sorted(topology.filter_match(**filters))
        )
        donors = [
            rank
            for rank in data_parallel_ranks
            if rank not in replacement_set
        ]
        if not donors:
            raise DeepSpeedHybridRestoreError(
                "no live DP peer for replacement rank "
                f"{replacement_rank}; data_parallel_ranks="
                f"{list(data_parallel_ranks)} replacement_ranks="
                f"{sorted(replacement_set)}. Use a data-major physical-node "
                "layout or a rank-granular replacement."
            )
        plans.append(
            PeerRestorePlan(
                replacement_rank=replacement_rank,
                source_rank=donors[0],
                data_parallel_ranks=data_parallel_ranks,
            )
        )
    return plans


def validate_relaunch_checkpoint_steps(
    checkpoint_steps: Iterable[int],
    *,
    failure_step: int,
) -> int:
    """Select the common checkpoint version for a full-epoch relaunch.

    A recovery epoch reconstructs every worker, so a donor's in-memory state
    comes from the checkpoint it just loaded, not from the later failure step.
    """
    steps = tuple(int(step) for step in checkpoint_steps)
    if not steps:
        raise DeepSpeedHybridRestoreError(
            "recovery checkpoint step set is empty"
        )
    if failure_step < 0:
        raise DeepSpeedHybridRestoreError(
            f"failure step must be non-negative; got {failure_step}"
        )
    invalid = sorted({step for step in steps if step < 0})
    if invalid:
        raise DeepSpeedHybridRestoreError(
            "recovery ranks reported invalid checkpoint steps: "
            f"{invalid}"
        )
    unique_steps = sorted(set(steps))
    if len(unique_steps) != 1:
        raise DeepSpeedHybridRestoreError(
            "recovery ranks loaded different checkpoint versions: "
            f"{unique_steps}"
        )
    checkpoint_step = unique_steps[0]
    if checkpoint_step > failure_step:
        raise DeepSpeedHybridRestoreError(
            "recovery checkpoint is newer than the recorded failure: "
            f"checkpoint_step={checkpoint_step} failure_step={failure_step}"
        )
    return checkpoint_step


def _validate_peer_header(
    header: dict[str, Any],
    *,
    plan: PeerRestorePlan,
    expected_step: int,
    manifest: list[dict[str, Any]],
    manifest_hash: str,
) -> None:
    expected = {
        "source_rank": plan.source_rank,
        "replacement_rank": plan.replacement_rank,
        "step": expected_step,
        "manifest_hash": manifest_hash,
        "manifest": manifest,
    }
    mismatches = {
        key: {"expected": value, "actual": header.get(key)}
        for key, value in expected.items()
        if header.get(key) != value
    }
    if mismatches:
        raise DeepSpeedHybridRestoreError(
            "non-expert peer state is incompatible: "
            + json.dumps(mismatches, sort_keys=True)
        )


def restore_non_expert_model_from_peer(
    engine: Any,
    *,
    replacement_ranks: Iterable[int],
    expected_step: int,
) -> dict[str, Any]:
    """Overwrite non-expert state from a checkpoint-restored DP peer.

    All world ranks must call this function. Process groups are created in a
    global deterministic order, while only each source/replacement pair moves
    tensor data.
    """
    import torch
    import torch.distributed as dist

    if expected_step < 0:
        raise DeepSpeedHybridRestoreError(
            "expected recovery step must be non-negative"
        )
    if not dist.is_initialized():
        raise DeepSpeedHybridRestoreError(
            "torch.distributed must be initialized before peer restore"
        )

    topology = getattr(getattr(engine, "grid", None), "_topo", None)
    if topology is None:
        topology = _FlatDataParallelTopology(dist.get_world_size())
    plans = build_peer_restore_plans(topology, replacement_ranks)
    rank = dist.get_rank()
    copied_tensors = 0
    copied_bytes = 0
    local_roles = []

    for ordinal, plan in enumerate(plans):
        pair_group = dist.new_group(
            ranks=[plan.source_rank, plan.replacement_rank]
        )
        if rank not in (plan.source_rank, plan.replacement_rank):
            continue

        tensors = non_expert_model_tensors(engine.module)
        manifest, manifest_hash = _tensor_manifest(tensors)
        header = {
            "ordinal": ordinal,
            "source_rank": plan.source_rank,
            "replacement_rank": plan.replacement_rank,
            "step": int(getattr(engine, "global_steps", -1)),
            "manifest_hash": manifest_hash,
            "manifest": manifest,
        }
        object_list = [header if rank == plan.source_rank else None]
        dist.broadcast_object_list(
            object_list, src=plan.source_rank, group=pair_group
        )
        peer_header = object_list[0]
        if not isinstance(peer_header, dict):
            raise DeepSpeedHybridRestoreError(
                f"peer restore group {ordinal} returned an invalid header"
            )
        _validate_peer_header(
            peer_header,
            plan=plan,
            expected_step=expected_step,
            manifest=manifest,
            manifest_hash=manifest_hash,
        )

        for _, tensor in tensors:
            dist.broadcast(
                tensor, src=plan.source_rank, group=pair_group
            )
            if rank == plan.replacement_rank:
                copied_tensors += 1
                copied_bytes += tensor.numel() * tensor.element_size()
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        local_roles.append(
            "source" if rank == plan.source_rank else "replacement"
        )

    dist.barrier()
    return {
        "expected_step": expected_step,
        "replacement_ranks": [plan.replacement_rank for plan in plans],
        "source_ranks": [plan.source_rank for plan in plans],
        "local_roles": local_roles,
        "copied_tensors": copied_tensors,
        "copied_bytes": copied_bytes,
        "expert_source": "checkpoint",
        "non_expert_source": "live_dp_peer",
        "peer_state_origin": "checkpoint_relaunch",
        "optimizer_source": "checkpoint",
        "two_phase": False,
    }
