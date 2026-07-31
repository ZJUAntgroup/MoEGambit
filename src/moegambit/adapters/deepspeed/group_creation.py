"""Recovery-only ordering for DeepSpeed process-group construction."""

from __future__ import annotations

import os
from typing import Any, Iterable

from moegambit.runtime.distributed import GroupSpec, WatcherOrdinalBarrier


_TRUE_VALUES = {"1", "true", "yes", "on"}
_GROUP_EPOCH: int | None = None
_GROUP_ORDINAL = 0
_GROUP_BARRIER: WatcherOrdinalBarrier | None = None


def _enabled() -> bool:
    ordered = os.environ.get(
        "MOEGAMBIT_DEEPSPEED_ORDERED_GROUP_REBUILD", "0"
    )
    inprocess = os.environ.get(
        "MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY", "0"
    )
    if ordered.strip().lower() not in _TRUE_VALUES:
        return False
    if inprocess.strip().lower() not in _TRUE_VALUES:
        return False
    epoch_value = os.environ.get("MOEGAMBIT_RECOVERY_EPOCH")
    if epoch_value is None:
        return False
    try:
        epoch = int(epoch_value)
    except ValueError as exc:
        raise RuntimeError("invalid MoEGambit recovery epoch") from exc
    return epoch > 0


def ordered_new_group(backend: Any, ranks: Iterable[int]) -> Any:
    """Create a DeepSpeed group in the coordinator-approved ordinal order."""

    global _GROUP_BARRIER
    global _GROUP_EPOCH
    global _GROUP_ORDINAL

    rank_list = [int(item) for item in ranks]
    if not _enabled():
        return backend.new_group(rank_list)

    epoch = int(os.environ["MOEGAMBIT_RECOVERY_EPOCH"])
    if _GROUP_EPOCH != epoch:
        timeout = float(
            os.environ.get(
                "MOEGAMBIT_DEEPSPEED_GROUP_BARRIER_TIMEOUT", "300"
            )
        )
        _GROUP_EPOCH = epoch
        _GROUP_ORDINAL = 0
        _GROUP_BARRIER = WatcherOrdinalBarrier.from_environment(
            int(backend.get_rank()),
            int(backend.get_world_size()),
            timeout_seconds=timeout,
        )

    if _GROUP_BARRIER is None:
        raise RuntimeError("DeepSpeed recovery group barrier is not initialized")
    group_spec = GroupSpec(
        ordinal=_GROUP_ORDINAL,
        name=f"deepspeed_group_{_GROUP_ORDINAL:04d}",
        ranks=tuple(rank_list),
        backend="nccl",
    )
    _GROUP_BARRIER.wait("start", group_spec, group_spec.fingerprint)
    group = backend.new_group(rank_list)
    _GROUP_BARRIER.wait("done", group_spec, group_spec.fingerprint)
    _GROUP_ORDINAL += 1
    return group
