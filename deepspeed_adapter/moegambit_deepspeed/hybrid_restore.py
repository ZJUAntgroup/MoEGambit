"""Single-stage MoE recovery from checkpoint and a compatible DP peer."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


class DeepSpeedHybridRestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeerRestorePlan:
    replacement_rank: int
    source_rank: int
    data_parallel_ranks: tuple[int, ...]


@dataclass(frozen=True)
class OptimizerPeerRestorePlan:
    namespace: str
    replacement_rank: int
    source_rank: int
    replica_ring_ranks: tuple[int, ...]


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


def build_optimizer_peer_restore_plans(
    group_manifest: dict[str, Iterable[int]],
    replacement_ranks: Iterable[int],
) -> list[OptimizerPeerRestorePlan]:
    """Map each failed optimizer shard to its actual replica holder."""
    replacement_set = {int(rank) for rank in replacement_ranks}
    if not replacement_set:
        raise DeepSpeedHybridRestoreError("replacement rank set is empty")

    normalized_manifest = {}
    for namespace, values in sorted(group_manifest.items()):
        ranks = tuple(int(rank) for rank in values)
        if len(ranks) < 2 or len(set(ranks)) != len(ranks):
            raise DeepSpeedHybridRestoreError(
                "invalid optimizer replica ring: "
                f"namespace={namespace} ranks={list(ranks)}"
            )
        normalized_manifest[str(namespace)] = ranks

    plans = []
    for replacement_rank in sorted(replacement_set):
        matching = [
            (namespace, ranks)
            for namespace, ranks in normalized_manifest.items()
            if replacement_rank in ranks
        ]
        if not matching:
            raise DeepSpeedHybridRestoreError(
                "replacement has no non-expert optimizer replica group: "
                f"rank={replacement_rank}"
            )
        for namespace, ranks in matching:
            owner_index = ranks.index(replacement_rank)
            holder_rank = ranks[(owner_index + 1) % len(ranks)]
            if holder_rank in replacement_set:
                raise DeepSpeedHybridRestoreError(
                    "failed optimizer shard's backup holder is also being "
                    f"replaced: owner={replacement_rank} "
                    f"holder={holder_rank} ring={list(ranks)}"
                )
            plans.append(
                OptimizerPeerRestorePlan(
                    namespace=namespace,
                    replacement_rank=replacement_rank,
                    source_rank=holder_rank,
                    replica_ring_ranks=ranks,
                )
            )
    return plans


def validate_checkpoint_base_steps(
    checkpoint_steps: Iterable[int],
    *,
    failure_step: int,
) -> int:
    """Select the common checkpoint version used only as the expert base."""
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


def validate_mixed_version_steps(
    checkpoint_steps: Iterable[int],
    survivor_steps: Iterable[int],
    *,
    failure_step: int,
) -> int:
    """Validate checkpoint experts and current-step survivor state."""
    checkpoint_step = validate_checkpoint_base_steps(
        checkpoint_steps, failure_step=failure_step
    )
    survivor_values = tuple(int(step) for step in survivor_steps)
    if not survivor_values:
        raise DeepSpeedHybridRestoreError(
            "mixed-version recovery has no survivor handoff"
        )
    invalid = sorted(
        {step for step in survivor_values if step != int(failure_step)}
    )
    if invalid:
        raise DeepSpeedHybridRestoreError(
            "survivor handoff is not at the failure safe step: "
            f"expected={failure_step} actual={invalid}"
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
    """Overwrite replacement non-expert state from a current-step DP peer.

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
        if rank == plan.source_rank:
            from moegambit_deepspeed.survivor_handoff import (
                capture_engine_state,
                capture_rng_state,
            )

            header["engine_state"] = capture_engine_state(engine)
            header["rng_state"] = capture_rng_state()
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
        if rank == plan.replacement_rank:
            from moegambit_deepspeed.survivor_handoff import (
                apply_engine_state,
                apply_rng_state,
            )

            engine_state = peer_header.get("engine_state")
            rng_state = peer_header.get("rng_state")
            if not isinstance(engine_state, dict):
                raise DeepSpeedHybridRestoreError(
                    f"peer restore group {ordinal} omitted engine state"
                )
            if not isinstance(rng_state, dict):
                raise DeepSpeedHybridRestoreError(
                    f"peer restore group {ordinal} omitted RNG state"
                )
            apply_engine_state(engine, engine_state)
            apply_rng_state(rng_state)
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
        "peer_state_origin": "survivor_handoff",
        "two_phase": False,
    }


def _snapshot_tensor(
    snapshot: Any, item: dict[str, Any]
) -> Any:
    source = snapshot.buffers[item["dtype"]].view(-1).narrow(
        0, int(item["offset"]), int(item["numel"])
    )
    return source.view(item["shape"])


def _optimizer_manifest_for_refs(
    refs: Iterable[Any],
) -> tuple[list[dict[str, Any]], str]:
    offsets: dict[str, int] = {}
    manifest = []
    for ref in refs:
        tensor = ref.tensor.detach()
        dtype = str(tensor.dtype)
        offset = offsets.get(dtype, 0)
        manifest.append(
            {
                "identity": ref.identity,
                "shape": list(tensor.shape),
                "dtype": dtype,
                "numel": int(tensor.numel()),
                "offset": offset,
                "is_expert": bool(ref.is_expert),
            }
        )
        offsets[dtype] = offset + int(tensor.numel())
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return manifest, hashlib.sha256(encoded).hexdigest()


def _validate_optimizer_peer_entry(
    entry: dict[str, Any],
    *,
    plan: OptimizerPeerRestorePlan,
    expected_step: int,
    local_manifest: list[dict[str, Any]],
    local_manifest_hash: str,
) -> None:
    remote_manifest = entry.get("manifest")
    if not isinstance(remote_manifest, list):
        raise DeepSpeedHybridRestoreError(
            "optimizer peer entry omitted its tensor manifest"
        )
    encoded = json.dumps(
        remote_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded_hash = hashlib.sha256(encoded).hexdigest()
    expected = {
        "namespace": plan.namespace,
        "owner_rank": plan.replacement_rank,
        "holder_rank": plan.source_rank,
        "step": int(expected_step),
        "replica_ring_ranks": list(plan.replica_ring_ranks),
        "manifest_hash": local_manifest_hash,
        "manifest": local_manifest,
    }
    mismatches = {
        key: {"expected": value, "actual": entry.get(key)}
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if entry.get("manifest_hash") != encoded_hash:
        mismatches["encoded_manifest_hash"] = {
            "expected": entry.get("manifest_hash"),
            "actual": encoded_hash,
        }
    if mismatches:
        raise DeepSpeedHybridRestoreError(
            "replacement optimizer state is incompatible: "
            + json.dumps(mismatches, sort_keys=True)
        )


def restore_non_expert_optimizer_from_peer(
    engine: Any,
    optimizer_replica: Any,
    *,
    survivor_payload: dict[str, Any] | None,
    replacement_ranks: Iterable[int],
    expected_step: int,
) -> dict[str, Any]:
    """Restore a failed ZeRO shard from the current replica held by its peer."""
    import torch
    import torch.distributed as dist

    replacement_set = {int(rank) for rank in replacement_ranks}
    local_groups = dict(optimizer_replica.group_ranks)
    gathered_groups: list[dict[str, list[int]] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(gathered_groups, local_groups)
    group_manifest: dict[str, list[int]] = {}
    for rank_groups in gathered_groups:
        if not rank_groups:
            continue
        for namespace, ranks in rank_groups.items():
            normalized = [int(item) for item in ranks]
            previous = group_manifest.setdefault(namespace, normalized)
            if previous != normalized:
                raise DeepSpeedHybridRestoreError(
                    "optimizer replica group manifest is inconsistent: "
                    f"namespace={namespace} first={previous} "
                    f"other={normalized}"
                )
    plans = build_optimizer_peer_restore_plans(
        group_manifest, replacement_set
    )
    rank = dist.get_rank()
    copied_tensors = 0
    copied_scalars = 0
    copied_bytes = 0
    local_roles = []

    for ordinal, plan in enumerate(plans):
        pair_group = dist.new_group(
            ranks=[plan.source_rank, plan.replacement_rank]
        )
        if rank not in (plan.source_rank, plan.replacement_rank):
            continue

        header: dict[str, Any] | None = None
        snapshots: dict[str, Any] = {}
        if rank == plan.source_rank:
            if survivor_payload is None:
                raise DeepSpeedHybridRestoreError(
                    f"source rank {rank} has no survivor handoff"
                )
            peer_snapshots = survivor_payload.get("peer_optimizer", {})
            if plan.namespace not in peer_snapshots:
                raise DeepSpeedHybridRestoreError(
                    "source peer does not hold the failed optimizer shard: "
                    f"source={plan.source_rank} "
                    f"replacement={plan.replacement_rank} "
                    f"namespace={plan.namespace}"
                )
            snapshots = {
                plan.namespace: peer_snapshots[plan.namespace]
            }
            entries = []
            for namespace, snapshot in snapshots.items():
                if (
                    int(snapshot.owner_rank) != plan.replacement_rank
                    or int(snapshot.holder_rank) != plan.source_rank
                    or int(snapshot.step) != int(expected_step)
                ):
                    raise DeepSpeedHybridRestoreError(
                        "optimizer handoff provenance mismatch: "
                        f"namespace={namespace} owner={snapshot.owner_rank} "
                        f"holder={snapshot.holder_rank} step={snapshot.step}"
                    )
                entries.append(
                    {
                        "namespace": namespace,
                        "owner_rank": int(snapshot.owner_rank),
                        "holder_rank": int(snapshot.holder_rank),
                        "step": int(snapshot.step),
                        "replica_ring_ranks": list(
                            plan.replica_ring_ranks
                        ),
                        "manifest_hash": snapshot.manifest_hash,
                        "manifest": snapshot.manifest,
                        "scalars": snapshot.scalars,
                    }
                )
            header = {
                "protocol": 1,
                "ordinal": ordinal,
                "source_rank": plan.source_rank,
                "replacement_rank": plan.replacement_rank,
                "step": int(expected_step),
                "entries": entries,
            }

        object_list = [header]
        dist.broadcast_object_list(
            object_list, src=plan.source_rank, group=pair_group
        )
        peer_header = object_list[0]
        if not isinstance(peer_header, dict):
            raise DeepSpeedHybridRestoreError(
                f"optimizer peer group {ordinal} returned an invalid header"
            )
        expected_header = {
            "protocol": 1,
            "ordinal": ordinal,
            "source_rank": plan.source_rank,
            "replacement_rank": plan.replacement_rank,
            "step": int(expected_step),
        }
        if any(
            peer_header.get(key) != value
            for key, value in expected_header.items()
        ):
            raise DeepSpeedHybridRestoreError(
                f"optimizer peer header mismatch: {peer_header}"
            )

        entries = peer_header.get("entries")
        if not isinstance(entries, list) or len(entries) != 1:
            raise DeepSpeedHybridRestoreError(
                "optimizer peer header must contain exactly one namespace: "
                f"{peer_header}"
            )
        for entry in entries:
            namespace = str(entry["namespace"])
            local_ref_list = [
                ref
                for ref in optimizer_replica.tensor_refs_for_namespace(
                    namespace
                )
                if not ref.is_expert
            ]
            local_refs = {
                ref.identity: ref
                for ref in local_ref_list
            }
            scalar_refs = {
                ref.identity: ref
                for ref in optimizer_replica.scalar_refs_for_namespace(
                    namespace
                )
                if not ref.is_expert
            }
            scalars = entry.get("scalars")
            if not isinstance(scalars, dict):
                raise DeepSpeedHybridRestoreError(
                    "optimizer peer entry omitted scalar state: "
                    f"namespace={namespace}"
                )
            if set(scalars) != set(scalar_refs):
                raise DeepSpeedHybridRestoreError(
                    "replacement optimizer scalar manifest mismatch: "
                    f"namespace={namespace} "
                    f"local={sorted(scalar_refs)} "
                    f"peer={sorted(scalars)}"
                )
            manifest, manifest_hash = _optimizer_manifest_for_refs(
                local_ref_list
            )
            _validate_optimizer_peer_entry(
                entry,
                plan=plan,
                expected_step=expected_step,
                local_manifest=manifest,
                local_manifest_hash=manifest_hash,
            )
            if any(bool(item["is_expert"]) for item in manifest):
                raise DeepSpeedHybridRestoreError(
                    "non-expert optimizer replica contains expert state: "
                    f"namespace={namespace}"
                )
            snapshot = snapshots.get(namespace)
            for item in manifest:
                identity = item["identity"]
                destination = local_refs[identity].tensor.detach().view(-1)
                source_cpu = (
                    _snapshot_tensor(snapshot, item).view(-1)
                    if rank == plan.source_rank
                    else None
                )
                chunk_bytes = max(
                    1,
                    int(
                        os.environ.get(
                            "MOEGAMBIT_PEER_TRANSFER_CHUNK_MB", "256"
                        )
                    )
                    * 1024
                    * 1024,
                )
                chunk_numel = max(
                    1, chunk_bytes // destination.element_size()
                )
                staging_numel = min(chunk_numel, destination.numel())
                staging = torch.empty(
                    staging_numel,
                    dtype=destination.dtype,
                    device=engine.device,
                )
                for offset in range(0, destination.numel(), chunk_numel):
                    count = min(
                        chunk_numel, destination.numel() - offset
                    )
                    transfer = staging.narrow(0, 0, count)
                    if rank == plan.source_rank:
                        transfer.copy_(
                            source_cpu.narrow(0, offset, count),
                            non_blocking=False,
                        )
                    dist.broadcast(
                        transfer, src=plan.source_rank, group=pair_group
                    )
                    if rank == plan.replacement_rank:
                        destination.narrow(0, offset, count).copy_(
                            transfer, non_blocking=False
                        )
                        copied_bytes += (
                            transfer.numel() * transfer.element_size()
                        )
                if rank == plan.replacement_rank:
                    copied_tensors += 1

            if rank == plan.replacement_rank:
                for identity, ref in scalar_refs.items():
                    if identity not in scalars:
                        raise DeepSpeedHybridRestoreError(
                            "replacement optimizer scalar missing: "
                            f"{identity}"
                        )
                    ref.state[ref.key] = scalars[identity]
                    copied_scalars += 1
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        local_roles.append(
            "source" if rank == plan.source_rank else "replacement"
        )

    dist.barrier()
    return {
        "expected_step": int(expected_step),
        "replacement_ranks": sorted(replacement_set),
        "source_ranks": sorted(
            {plan.source_rank for plan in plans}
        ),
        "local_roles": local_roles,
        "copied_tensors": copied_tensors,
        "copied_scalars": copied_scalars,
        "copied_bytes": copied_bytes,
        "optimizer_source": "survivor_peer_replica",
        "restore_scope": "non_expert",
    }
