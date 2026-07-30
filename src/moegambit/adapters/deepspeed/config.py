"""DeepSpeed configuration validation for MoEGambit features."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from moegambit.runtime.config import env_bool


class DeepSpeedAdapterConfigError(ValueError):
    pass


def prepare_deepspeed_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy a DeepSpeed config before engine construction."""
    prepared = deepcopy(dict(config))
    hot_swap = env_bool(
        "MOEGAMBIT_HOT_SWAP",
        env_bool("DEEPSPEED_MOEGAMBIT_HOT_SWAP", False),
    )
    zero2 = env_bool(
        "MOEGAMBIT_ZERO2",
        env_bool("DEEPSPEED_MOEGAMBIT_ZERO2", False),
    )
    zero = prepared.get("zero_optimization")
    if isinstance(zero, bool):
        zero = {"stage": 1 if zero else 0}
        prepared["zero_optimization"] = zero
    elif zero is None:
        zero = {}
        prepared["zero_optimization"] = zero
    elif not isinstance(zero, dict):
        raise DeepSpeedAdapterConfigError(
            "zero_optimization must be a mapping"
        )

    try:
        stage = int(zero.get("stage", 0))
    except (TypeError, ValueError) as exc:
        raise DeepSpeedAdapterConfigError(
            "zero_optimization.stage must be an integer"
        ) from exc

    if zero2 and stage != 2:
        raise DeepSpeedAdapterConfigError(
            "MoEGambit ZeRO-2 replication requires "
            "zero_optimization.stage=2"
        )
    if hot_swap and stage == 2:
        zero["elastic_checkpoint"] = True
    return prepared
