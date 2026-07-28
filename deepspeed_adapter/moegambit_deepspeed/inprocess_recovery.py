"""DeepSpeed process-group rebuild without recreating survivor engines."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Mapping


class DeepSpeedInProcessRecoveryError(RuntimeError):
    pass


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

    optimizer = engine.optimizer
    optimizer.mpu = grid
    optimizer.dp_process_group = engine.seq_data_parallel_group
    optimizer.ep_process_group = engine.expert_parallel_group
    optimizer.expert_dp_process_group = (
        engine.expert_data_parallel_group
    )
    group_count = len(optimizer.optimizer.param_groups)
    optimizer.real_dp_process_group = [
        engine.seq_data_parallel_group for _ in range(group_count)
    ]
    optimizer.partition_count = [
        dist.get_world_size(group=engine.seq_data_parallel_group)
        for _ in range(group_count)
    ]
    optimizer.sequence_parallel_size = engine.sequence_parallel_size
    if engine.has_moe_layers:
        optimizer._configure_moe_settings()
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
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    mismatches = [
        {"rank": rank, "manifest": manifest}
        for rank, manifest in enumerate(gathered)
        if manifest != local
    ]
    if mismatches:
        raise DeepSpeedInProcessRecoveryError(
            f"rebuilt DeepSpeed group manifest mismatch: {mismatches}"
        )


def rebuild_engine_process_groups(
    engine: Any, command: Mapping[str, Any]
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
    _retire_process_group_generation(dist, rank)
    groups.reset_for_recovery()
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        timeout=timedelta(
            seconds=float(
                os.environ.get(
                    "MOEGAMBIT_INPROCESS_PG_TIMEOUT", "600"
                )
            )
        ),
    )
    grid = (
        engine.module.rebuild_process_groups()
        if pipeline_engine
        else None
    )
    groups.mpu = grid
    _create_expert_groups(engine, grid)
    _rebind_engine(engine, grid)
    new_group_manifest = _local_group_manifest(engine)
    if new_group_manifest != old_group_manifest:
        raise DeepSpeedInProcessRecoveryError(
            "rebuilt DeepSpeed groups changed logical membership: "
            f"before={old_group_manifest} after={new_group_manifest}"
        )
    _validate_manifest(engine, failed_rank, epoch)
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
    }


def validate_replacement_process_groups(engine: Any) -> None:
    """Join the same post-rebuild manifest collective as all survivors."""
    failed_rank = int(
        os.environ["MOEGAMBIT_RECOVERY_FAILED_RANK"]
    )
    epoch = int(os.environ["MOEGAMBIT_RECOVERY_EPOCH"])
    _validate_manifest(engine, failed_rank, epoch)
