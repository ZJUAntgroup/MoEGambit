"""DeepSpeed process-group rebuild without recreating survivor engines."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Callable, Mapping


class DeepSpeedInProcessRecoveryError(RuntimeError):
    pass


_CONTROL_ORDINALS = {
    "topology_commit": -10,
    "checkpoint_selected": -11,
    "checkpoint_restored": -12,
    "survivor_state_committed": -13,
    "model_peer_restored": -14,
    "optimizer_peer_restored": -15,
    "recovery_state_committed": -16,
    "optimizer_replica_started": -17,
}


def wait_for_inprocess_recovery_gate(
    phase: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Commit one recovery-epoch phase through the watcher control plane."""
    try:
        ordinal = _CONTROL_ORDINALS[phase]
    except KeyError as exc:
        raise DeepSpeedInProcessRecoveryError(
            f"unknown in-process recovery control phase: {phase}"
        ) from exc

    import torch.distributed as dist
    from moegambit.runtime.distributed import (
        wait_for_recovery_group_barrier,
    )

    timeout = float(
        os.environ.get(
            "MOEGAMBIT_DEEPSPEED_GROUP_BARRIER_TIMEOUT", "300"
        )
    )
    wait_for_recovery_group_barrier(
        f"deepspeed_{phase}",
        ordinal=ordinal,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        timeout_seconds=timeout,
        metadata=metadata,
    )


def _c10d_generation_state() -> dict[str, int | bool]:
    import torch

    state: dict[str, int | bool] = {
        "initialized": torch.distributed.is_initialized(),
        "pg_map_count": -1,
        "pg_name_count": -1,
        "group_count": -1,
    }
    try:
        world = torch.distributed.distributed_c10d._world
        state.update(
            pg_map_count=len(world.pg_map),
            pg_name_count=len(world.pg_names),
            group_count=int(world.group_count),
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return state


def _retire_process_group_generation(dist: Any, rank: int) -> None:
    before = _c10d_generation_state()
    if not before["initialized"]:
        raise DeepSpeedInProcessRecoveryError(
            f"rank {rank} cannot retire an uninitialized c10d generation"
        )
    unavailable = [
        key
        for key in ("pg_map_count", "pg_name_count", "group_count")
        if int(before[key]) < 0
    ]
    if unavailable:
        raise DeepSpeedInProcessRecoveryError(
            "cannot validate c10d generation before teardown: "
            + ",".join(unavailable)
        )

    dist.destroy_process_group()
    after = _c10d_generation_state()
    errors = []
    if after["initialized"]:
        errors.append("default process group remains initialized")
    for key in ("pg_map_count", "pg_name_count", "group_count"):
        if int(after[key]) not in (-1, 0):
            errors.append(f"{key}={after[key]}")
    if errors:
        raise DeepSpeedInProcessRecoveryError(
            "incomplete c10d generation teardown: " + "; ".join(errors)
        )


def _group_ranks(group: Any) -> list[int] | None:
    if group is None:
        return None
    import torch

    return [
        int(rank)
        for rank in torch.distributed.get_process_group_ranks(group)
    ]


def _group_dict_ranks(groups: Any) -> dict[str, list[int] | None]:
    if not isinstance(groups, Mapping):
        return {}
    return {
        str(name): _group_ranks(group)
        for name, group in sorted(groups.items())
    }


def _local_group_manifest(engine: Any) -> dict[str, Any]:
    module = engine.module
    grid = getattr(engine, "grid", None)
    optimizer = engine.optimizer
    tied_comms = getattr(module, "tied_comms", {})
    return {
        "data_parallel": _group_ranks(engine.data_parallel_group),
        "sequence_data_parallel": _group_ranks(
            engine.seq_data_parallel_group
        ),
        "sequence_parallel": _group_ranks(
            getattr(engine, "seq_parallel_group", None)
        ),
        "expert_parallel": _group_dict_ranks(
            engine.expert_parallel_group
        ),
        "expert_data_parallel": _group_dict_ranks(
            engine.expert_data_parallel_group
        ),
        "model_parallel": _group_ranks(
            getattr(optimizer, "model_parallel_group", None)
        ),
        "optimizer_data_parallel": _group_ranks(
            optimizer.dp_process_group
        ),
        "optimizer_real_data_parallel": [
            _group_ranks(group)
            for group in optimizer.real_dp_process_group
        ],
        "pipeline": (
            _group_ranks(grid.get_pipe_parallel_group())
            if grid is not None
            else None
        ),
        "pipeline_data_parallel": (
            _group_ranks(grid.get_data_parallel_group())
            if grid is not None
            else None
        ),
        "pipeline_model_parallel": (
            _group_ranks(grid.get_model_parallel_group())
            if grid is not None
            else None
        ),
        "pipeline_slice_parallel": (
            _group_ranks(grid.get_slice_parallel_group())
            if grid is not None
            else None
        ),
        "tied": {
            str(name): _group_ranks(value.get("group"))
            for name, value in sorted(tied_comms.items())
        },
    }


def _recovery_communicator_warmup_enabled() -> bool:
    return os.environ.get(
        "MOEGAMBIT_DEEPSPEED_RECOVERY_COMM_WARMUP", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _local_recovery_warmup_groups(
    engine: Any,
) -> dict[tuple[str, tuple[int, ...]], dict[str, Any]]:
    """Collect the distinct local groups used by the first training step."""
    groups_by_handle: dict[int, dict[str, Any]] = {}

    def add(name: str, group: Any) -> None:
        if group is None:
            return
        ranks = _group_ranks(group)
        if ranks is None or len(ranks) <= 1:
            return
        handle_key = id(group)
        entry = groups_by_handle.setdefault(
            handle_key,
            {
                "group": group,
                "names": set(),
                "ranks": tuple(int(rank) for rank in ranks),
            },
        )
        if (
            entry["group"] is not group
            or entry["ranks"] != tuple(int(rank) for rank in ranks)
        ):
            raise DeepSpeedInProcessRecoveryError(
                "local process-group identity changed while building the "
                f"warmup manifest: existing={entry} new={name}"
            )
        entry["names"].add(name)

    add("engine.data_parallel", engine.data_parallel_group)
    add("engine.sequence_data_parallel", engine.seq_data_parallel_group)
    add("engine.sequence_parallel", engine.seq_parallel_group)

    optimizer = engine.optimizer
    add("optimizer.data_parallel", optimizer.dp_process_group)
    add(
        "optimizer.model_parallel",
        getattr(optimizer, "model_parallel_group", None),
    )
    for index, group in enumerate(
        getattr(optimizer, "real_dp_process_group", ())
    ):
        add(f"optimizer.real_data_parallel.{index}", group)

    for family, values in (
        ("expert_parallel", engine.expert_parallel_group),
        ("expert_data_parallel", engine.expert_data_parallel_group),
    ):
        if isinstance(values, Mapping):
            for name, group in sorted(values.items()):
                add(f"{family}.{name}", group)

    grid = getattr(engine, "grid", None)
    if grid is not None:
        for name, getter_name in (
            ("pipeline", "get_pipe_parallel_group"),
            ("pipeline_data_parallel", "get_data_parallel_group"),
            ("pipeline_model_parallel", "get_model_parallel_group"),
            ("pipeline_slice_parallel", "get_slice_parallel_group"),
        ):
            getter = getattr(grid, getter_name, None)
            if callable(getter):
                add(name, getter())

    tied_comms = getattr(engine.module, "tied_comms", {})
    if isinstance(tied_comms, Mapping):
        for name, value in sorted(tied_comms.items()):
            if isinstance(value, Mapping):
                add(f"tied.{name}", value.get("group"))

    folding = getattr(engine, "_autoep_folding_group_handles", None)
    if folding is not None:
        for name in ("tp_group", "dense_dp_group", "ep_group", "edp_group"):
            add(f"folding.{name}", getattr(folding, name, None))

    groups: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for entry in groups_by_handle.values():
        canonical_name = sorted(entry["names"])[0]
        descriptor = (canonical_name, entry["ranks"])
        if descriptor in groups:
            raise DeepSpeedInProcessRecoveryError(
                "communicator warmup has duplicate semantic descriptors: "
                f"{descriptor}"
            )
        groups[descriptor] = entry
    return groups


def _global_recovery_warmup_manifest(
    dist: Any,
    local_groups: Mapping[
        tuple[str, tuple[int, ...]], Mapping[str, Any]
    ],
) -> list[tuple[str, tuple[int, ...]]]:
    """Agree on a deterministic warmup order before issuing collectives."""
    world_size = int(dist.get_world_size())
    local_manifest = [
        {
            "name": name,
            "ranks": list(ranks),
            "aliases": sorted(entry["names"]),
        }
        for (name, ranks), entry in sorted(local_groups.items())
    ]
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_manifest)

    parsed: list[set[tuple[str, tuple[int, ...]]]] = []
    union: set[tuple[str, tuple[int, ...]]] = set()
    for reporter, manifest in enumerate(gathered):
        if not isinstance(manifest, list):
            raise DeepSpeedInProcessRecoveryError(
                "communicator warmup manifest is missing for "
                f"rank {reporter}: {manifest!r}"
            )
        local_descriptors: set[tuple[str, tuple[int, ...]]] = set()
        for item in manifest:
            try:
                name = str(item["name"])
                ranks = tuple(int(value) for value in item["ranks"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DeepSpeedInProcessRecoveryError(
                    "invalid communicator warmup descriptor "
                    f"from rank {reporter}: {item!r}"
                ) from exc
            if (
                not name
                or len(ranks) <= 1
                or len(set(ranks)) != len(ranks)
                or any(rank < 0 or rank >= world_size for rank in ranks)
            ):
                raise DeepSpeedInProcessRecoveryError(
                    "invalid communicator warmup group "
                    f"from rank {reporter}: name={name!r} ranks={ranks}"
                )
            descriptor = (name, ranks)
            local_descriptors.add(descriptor)
            union.add(descriptor)
        parsed.append(local_descriptors)

    for descriptor in union:
        _, ranks = descriptor
        missing = [
            rank for rank in ranks if descriptor not in parsed[rank]
        ]
        if missing:
            raise DeepSpeedInProcessRecoveryError(
                "communicator warmup group is not visible to every member: "
                f"group={descriptor} missing={missing}"
            )
    return sorted(
        union,
        key=lambda descriptor: (
            descriptor[0],
            len(descriptor[1]),
            descriptor[1],
        ),
    )


def _warm_pipeline_p2p(dist: Any, engine: Any, token: Any) -> int:
    """Eagerly connect both directions of every adjacent pipeline edge."""
    grid = getattr(engine, "grid", None)
    if grid is None or int(grid.pipe_parallel_size) <= 1:
        return 0

    stage = int(grid.get_stage_id())
    stages = int(grid.pipe_parallel_size)
    if stages % 2:
        return 0
    operations = 0
    for parity in (0, 1):
        if parity == 0:
            peer_stage = stage + 1 if stage % 2 == 0 else stage - 1
        elif stage == 0:
            peer_stage = stages - 1
        elif stage == stages - 1:
            peer_stage = 0
        elif stage % 2:
            peer_stage = stage + 1
        else:
            peer_stage = stage - 1
        peer = int(grid.stage_to_global(peer_stage))
        for forward in (True, False):
            send = (stage < peer_stage) == forward
            if send:
                dist.send(token, dst=peer)
            else:
                dist.recv(token, src=peer)
            operations += 1
            dist.all_reduce(token)
    return operations


def warm_recovery_communicators(
    engine: Any,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Initialize only communicators used by the first post-recovery step."""
    if not _recovery_communicator_warmup_enabled():
        return {"enabled": False, "groups": 0, "p2p_operations": 0}

    import torch
    import torch.distributed as dist

    if phase_callback is not None:
        phase_callback("communicator_warmup_start")
    local_groups = _local_recovery_warmup_groups(engine)
    manifest = _global_recovery_warmup_manifest(dist, local_groups)
    rank = int(dist.get_rank())
    token = torch.ones(1, dtype=torch.int32, device=engine.device)

    for descriptor in manifest:
        _, ranks = descriptor
        if rank in ranks:
            dist.all_reduce(
                token,
                group=local_groups[descriptor]["group"],
            )
        # Non-members wait here so overlapping groups can never initialize in
        # a different order on different ranks.
        dist.all_reduce(token)

    p2p_operations = _warm_pipeline_p2p(dist, engine, token)
    torch.cuda.synchronize(engine.device)
    if phase_callback is not None:
        phase_callback("communicator_warmup_done")
    return {
        "enabled": True,
        "groups": len(manifest),
        "local_groups": len(local_groups),
        "p2p_operations": p2p_operations,
    }


def _require_rank_command(
    command: Mapping[str, Any], rank: int, world_size: int
) -> tuple[int, int, str, int]:
    if command.get("recovery_mode") != "rank_in_process":
        raise DeepSpeedInProcessRecoveryError(
            f"unexpected recovery mode: {command.get('recovery_mode')}"
        )
    failed_rank = int(command.get("failed_rank", -1))
    epoch = int(command.get("epoch", -1))
    master_addr = str(command.get("master_addr", "")).strip()
    master_port = int(command.get("master_port", -1))
    if not 0 <= failed_rank < world_size:
        raise DeepSpeedInProcessRecoveryError(
            f"failed rank {failed_rank} is outside world size {world_size}"
        )
    if epoch <= 0 or not master_addr or master_port <= 0:
        raise DeepSpeedInProcessRecoveryError(
            "rank recovery command has no valid epoch or rendezvous endpoint"
        )
    rank_mapping = command.get("rank_mapping")
    if not isinstance(rank_mapping, Mapping):
        raise DeepSpeedInProcessRecoveryError(
            "rank recovery command omitted rank_mapping"
        )
    expected_ranks = {str(item) for item in range(world_size)}
    if set(rank_mapping) != expected_ranks:
        raise DeepSpeedInProcessRecoveryError(
            "rank recovery mapping does not cover the complete world"
        )
    if not 0 <= rank < world_size:
        raise DeepSpeedInProcessRecoveryError(
            f"survivor rank {rank} is outside world size {world_size}"
        )
    return failed_rank, epoch, master_addr, master_port


def _create_expert_groups(engine: Any, grid: Any | None) -> None:
    from deepspeed.utils import groups

    autoep_config = engine._config.expert_parallel_config
    if autoep_config is None or not autoep_config.enabled:
        return
    ep_size = int(autoep_config.autoep_size)
    tp_size = int(engine.autotp_size())
    sp_size = int(engine._autoep_sequence_parallel_world_size())
    mp_size = max(tp_size, sp_size, 1)
    groups._create_expert_and_data_parallel(
        expert_parallel_size_=ep_size,
        mp_size=mp_size,
        pp_size=int(grid.pipe_parallel_size) if grid is not None else 1,
        mp_mode="tp" if tp_size > 1 else "sp",
        use_data_before_expert_parallel_=(
            engine._config.use_data_before_expert_parallel_
        ),
        folding_spec=(
            engine._autoep_folding_spec if tp_size > 1 else None
        ),
        pipeline_mpu=grid,
    )


def _folding_group_handles(engine: Any) -> Any:
    folding_spec = getattr(engine, "_autoep_folding_spec", None)
    if folding_spec is None or folding_spec.tp_size <= 1:
        return None
    from deepspeed.module_inject.auto_ep_folding import (
        FoldingGroupHandles,
        local_folding_ranks,
    )
    from deepspeed import comm as dist
    from deepspeed.utils import groups

    group_name = f"ep_size_{folding_spec.ep_size}"
    local_ranks = local_folding_ranks(dist.get_rank(), folding_spec)
    return FoldingGroupHandles(
        spec=folding_spec,
        tp_group=groups.get_tensor_model_parallel_group(),
        dense_dp_group=groups._get_data_parallel_group(),
        ep_group=groups._get_expert_parallel_group(group_name),
        edp_group=groups._get_expert_data_parallel_group(group_name),
        ep_group_name=group_name,
        tp_ranks=local_ranks["tp"],
        dense_dp_ranks=local_ranks["dense_dp"],
        ep_ranks=local_ranks["ep"],
        edp_ranks=local_ranks["edp"],
    )


def _rebind_optimizer_process_groups(
    engine: Any, grid: Any | None, dist: Any
) -> None:
    optimizer = engine.optimizer
    dense_group = engine.seq_data_parallel_group
    optimizer.mpu = grid
    optimizer.dp_process_group = dense_group
    optimizer.ep_process_group = engine.expert_parallel_group
    optimizer.expert_dp_process_group = (
        engine.expert_data_parallel_group
    )

    param_groups = optimizer.optimizer.param_groups
    group_count = len(param_groups)
    dense_size = dist.get_world_size(group=dense_group)
    optimizer.real_dp_process_group = [
        dense_group for _ in range(group_count)
    ]
    optimizer.partition_count = [
        dense_size for _ in range(group_count)
    ]
    optimizer.sequence_parallel_size = engine.sequence_parallel_size

    if engine.has_moe_layers:
        # ZeRO replaces param_group["params"] with FP32 master shards after
        # its constructor. Reuse its original MoE classification here.
        moe_layout = getattr(optimizer, "is_moe_param_group", None)
        if not isinstance(moe_layout, (list, tuple)):
            raise DeepSpeedInProcessRecoveryError(
                "ZeRO optimizer omitted its initialized MoE group layout"
            )
        if len(moe_layout) != group_count:
            raise DeepSpeedInProcessRecoveryError(
                "ZeRO MoE group layout no longer matches optimizer groups: "
                f"layout={len(moe_layout)} groups={group_count}"
            )
        expert_groups = engine.expert_data_parallel_group
        if not isinstance(expert_groups, Mapping):
            raise DeepSpeedInProcessRecoveryError(
                "rebuilt expert data-parallel groups are unavailable"
            )
        for index, is_moe_group in enumerate(moe_layout):
            if not is_moe_group:
                continue
            group_name = param_groups[index].get("name")
            if not group_name or group_name not in expert_groups:
                raise DeepSpeedInProcessRecoveryError(
                    "rebuilt expert data-parallel group is missing for "
                    f"optimizer group {index}: name={group_name!r}"
                )
            expert_group = expert_groups[group_name]
            if expert_group is None:
                raise DeepSpeedInProcessRecoveryError(
                    "rebuilt expert data-parallel group is empty for "
                    f"optimizer group {index}: name={group_name!r}"
                )
            optimizer.real_dp_process_group[index] = expert_group
            optimizer.partition_count[index] = dist.get_world_size(
                group=expert_group
            )

    optimizer.configure_autoep_folding_tp_gradient_reduction(
        getattr(engine, "_autoep_folding_spec", None)
    )
    if grid is None:
        optimizer.model_parallel_group = None
        optimizer.model_parallel_world_size = 1
        optimizer.model_parallel_rank = 0
    else:
        optimizer.model_parallel_group = grid.get_model_parallel_group()
        optimizer.model_parallel_world_size = (
            grid.get_model_parallel_world_size()
        )
        optimizer.model_parallel_rank = grid.get_model_parallel_rank()


def _rebind_engine(engine: Any, grid: Any | None) -> None:
    from deepspeed import comm as dist
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
    from deepspeed.runtime.pipe import p2p
    from deepspeed.utils import groups

    engine.mpu = grid
    if grid is not None:
        engine.grid = grid
    groups.mpu = grid
    folding_handles = _folding_group_handles(engine)
    engine._autoep_folding_group_handles = folding_handles
    for module in engine.module.modules():
        if isinstance(module, AutoEPMoELayer):
            module.folding_group_handles = folding_handles
            if folding_handles is None:
                module.tp_group = None
            module.set_deepspeed_parallelism(
                engine._config.use_data_before_expert_parallel_,
                folding_group_handles=folding_handles,
            )
        elif hasattr(module, "set_deepspeed_parallelism"):
            module.set_deepspeed_parallelism(
                engine._config.use_data_before_expert_parallel_
            )

    engine.global_rank = dist.get_rank()
    engine.world_size = dist.get_world_size()
    engine.local_all_to_all_group = None
    if engine.zero_quantized_gradients():
        engine.local_all_to_all_group = groups._get_local_all_to_all_group()
    engine.data_parallel_group = groups._get_data_parallel_group()
    engine.dp_world_size = groups._get_data_parallel_world_size()
    engine.seq_data_parallel_group = (
        groups._get_sequence_data_parallel_group()
    )
    engine.seq_dp_world_size = (
        groups._get_sequence_data_parallel_world_size()
    )
    engine.mp_world_size = groups._get_model_parallel_world_size()
    engine.expert_parallel_group = (
        groups._get_expert_parallel_group_dict()
    )
    engine.expert_data_parallel_group = (
        groups._get_expert_data_parallel_group_dict()
    )
    engine.sequence_parallel_size = (
        groups._get_sequence_parallel_world_size()
    )
    engine.seq_parallel_group = (
        groups._get_sequence_parallel_group()
        if engine.sequence_parallel_size > 1
        else None
    )

    if grid is not None:
        engine.num_stages = grid.pipe_parallel_size
        engine.stage_id = grid.get_stage_id()
        engine.prev_stage = engine.stage_id - 1
        engine.next_stage = engine.stage_id + 1
        engine.is_pipe_parallel = grid.pipe_parallel_size > 1
        engine.is_data_parallel = grid.data_parallel_size > 1
        engine.is_model_parallel = grid.model_parallel_size > 1
        p2p._groups = None
        p2p._grid = None
        p2p._async = []
        if engine.is_pipe_parallel:
            p2p.init_process_groups(grid)
        engine.reset_activation_shape()
        engine.first_gradient_send = True

    _rebind_optimizer_process_groups(engine, grid, dist)


def _validate_manifest(engine: Any, failed_rank: int, epoch: int) -> None:
    from deepspeed import comm as dist

    grid = getattr(engine, "grid", None)
    topology = getattr(grid, "_topo", None)
    local = {
        "epoch": epoch,
        "failed_rank": failed_rank,
        "world_size": dist.get_world_size(),
        "pipe_size": int(grid.pipe_parallel_size) if grid is not None else 1,
        "data_size": (
            int(grid.data_parallel_size)
            if grid is not None
            else dist.get_world_size()
        ),
        "topology_axes": (
            list(topology.get_axis_names())
            if topology is not None
            else ["data"]
        ),
        "topology_dims": (
            [
                int(topology.get_dim(axis))
                for axis in topology.get_axis_names()
            ]
            if topology is not None
            else [dist.get_world_size()]
        ),
    }
    wait_for_inprocess_recovery_gate(
        "topology_commit",
        metadata=local,
    )


def rebuild_engine_process_groups(
    engine: Any,
    command: Mapping[str, Any],
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild the complete DeepSpeed topology around resident tensors."""
    from deepspeed import comm as dist
    from deepspeed.runtime.pipe.module import PipelineModule
    from deepspeed.utils import groups
    import deepspeed

    pipeline_engine = isinstance(engine.module, PipelineModule)
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None or not hasattr(
        optimizer, "real_dp_process_group"
    ):
        raise DeepSpeedInProcessRecoveryError(
            "rank in-process recovery requires DeepSpeed ZeRO stage 1 or 2"
        )
    rank = int(engine.global_rank)
    world_size = int(dist.get_world_size())
    failed_rank, epoch, master_addr, master_port = _require_rank_command(
        command, rank, world_size
    )
    failure_step = int(command.get("failure_step", -1))
    if failure_step < 0:
        raise DeepSpeedInProcessRecoveryError(
            "rank recovery command omitted the committed failure step"
        )

    def report(phase: str) -> None:
        if phase_callback is not None:
            phase_callback(phase)

    old_group_manifest = _local_group_manifest(engine)
    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "MOEGAMBIT_RECOVERY_EPOCH": str(epoch),
            "TORCHELASTIC_RESTART_COUNT": str(epoch),
            "MOEGAMBIT_RECOVERY_FAILED_RANK": str(failed_rank),
            "MOEGAMBIT_RECOVERY_FAILURE_STEP": str(failure_step),
        }
    )
    timeout_seconds = float(
        os.environ.get("MOEGAMBIT_INPROCESS_PG_TIMEOUT", "600")
    )
    barrier_timeout = float(
        os.environ.get(
            "MOEGAMBIT_DEEPSPEED_GROUP_BARRIER_TIMEOUT", "300"
        )
    )
    report("default_pg_teardown_start")
    _retire_process_group_generation(dist, rank)
    report("default_pg_teardown_done")
    from moegambit.runtime.distributed import (
        wait_for_recovery_group_barrier,
    )

    report("default_pg_retired_gate_start")
    wait_for_recovery_group_barrier(
        "default_pg_retired",
        ordinal=-2,
        rank=rank,
        world_size=world_size,
        timeout_seconds=barrier_timeout,
    )
    report("default_pg_retired_gate_done")
    groups.reset_for_recovery()
    report("default_pg_init_start")
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        timeout=timedelta(seconds=timeout_seconds),
    )
    report("default_pg_init_done")
    report("default_pg_initialized_gate_start")
    wait_for_recovery_group_barrier(
        "default_pg_initialized",
        ordinal=-1,
        rank=rank,
        world_size=world_size,
        timeout_seconds=barrier_timeout,
    )
    report("default_pg_initialized_gate_done")
    report("pipeline_groups_start")
    grid = (
        engine.module.rebuild_process_groups()
        if pipeline_engine
        else None
    )
    report("pipeline_groups_done")
    groups.mpu = grid
    report("expert_groups_start")
    _create_expert_groups(engine, grid)
    report("expert_groups_done")
    report("engine_group_rebind_start")
    _rebind_engine(engine, grid)
    report("engine_group_rebind_done")
    report("local_group_manifest_start")
    new_group_manifest = _local_group_manifest(engine)
    if new_group_manifest != old_group_manifest:
        raise DeepSpeedInProcessRecoveryError(
            "rebuilt DeepSpeed groups changed logical membership: "
            f"before={old_group_manifest} after={new_group_manifest}"
        )
    report("local_group_manifest_done")
    report("global_group_manifest_start")
    _validate_manifest(engine, failed_rank, epoch)
    report("global_group_manifest_done")
    warmup_summary = warm_recovery_communicators(
        engine,
        phase_callback=phase_callback,
    )
    return {
        "rank": rank,
        "failed_rank": failed_rank,
        "epoch": epoch,
        "world_size": world_size,
        "pipe_size": int(grid.pipe_parallel_size) if grid is not None else 1,
        "data_size": (
            int(grid.data_parallel_size)
            if grid is not None
            else world_size
        ),
        "model_parameters_preserved": True,
        "optimizer_parameters_preserved": True,
        "group_membership_preserved": True,
        "communicator_warmup": warmup_summary,
    }


def validate_replacement_process_groups(
    engine: Any,
    phase_callback: Callable[[str], None] | None = None,
) -> None:
    """Join the same post-rebuild manifest collective as all survivors."""
    failed_rank = int(
        os.environ["MOEGAMBIT_RECOVERY_FAILED_RANK"]
    )
    epoch = int(os.environ["MOEGAMBIT_RECOVERY_EPOCH"])
    if phase_callback is not None:
        phase_callback("global_group_manifest_start")
    _validate_manifest(engine, failed_rank, epoch)
    if phase_callback is not None:
        phase_callback("global_group_manifest_done")
    warm_recovery_communicators(
        engine,
        phase_callback=phase_callback,
    )
