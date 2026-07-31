"""DeepSpeed launcher callback for rank-granular recovery."""

from __future__ import annotations

import logging
import os

from moegambit.runtime.hot_spare import request_worker_command


logger = logging.getLogger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}


def confirm_rank_recovery(dist_rank: int, return_code: int) -> bool:
    """Ask the external coordinator whether sibling workers should survive."""

    enabled = os.environ.get(
        "MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY", "0"
    )
    if enabled.strip().lower() not in _TRUE_VALUES:
        return False
    try:
        command = request_worker_command(
            "rank_failure",
            int(dist_rank),
            reason=f"launcher_observed_exit_{return_code}",
            return_code=int(return_code),
        )
    except Exception as exc:
        logger.error(
            "rank %s recovery confirmation failed: %s",
            dist_rank,
            exc,
        )
        return False
    return bool(
        command
        and command.get("recovery_mode") == "rank_in_process"
        and int(command.get("failed_rank", -1)) == int(dist_rank)
        and command.get("failure_step") is not None
        and int(command.get("epoch", 0)) > 0
    )
