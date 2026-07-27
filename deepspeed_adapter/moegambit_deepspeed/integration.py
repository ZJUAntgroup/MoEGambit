"""Optional DeepSpeed engine hooks activated by MoEGambit feature flags."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moegambit.runtime.config import env_bool


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSpeedRuntimeSettings:
    hot_swap: bool
    zero2: bool
    application_checkpoint: bool
    checkpoint_dir: Path | None
    checkpoint_interval: int
    restart_count: int
    recovery_epoch: int
    replica_timeout: float
    hybrid_restore: bool = False

    @classmethod
    def from_env(cls) -> "DeepSpeedRuntimeSettings":
        checkpoint = os.environ.get("MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR")
        interval = int(
            os.environ.get("MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL", "0")
        )
        if interval < 0:
            raise ValueError(
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL cannot be negative"
            )
        return cls(
            hot_swap=env_bool(
                "MOEGAMBIT_HOT_SWAP",
                env_bool("DEEPSPEED_MOEGAMBIT_HOT_SWAP", False),
            ),
            zero2=env_bool(
                "MOEGAMBIT_ZERO2",
                env_bool("DEEPSPEED_MOEGAMBIT_ZERO2", False),
            ),
            application_checkpoint=env_bool(
                "MOEGAMBIT_DEEPSPEED_APPLICATION_CHECKPOINT", False
            ),
            checkpoint_dir=Path(checkpoint) if checkpoint else None,
            checkpoint_interval=interval,
            restart_count=int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")),
            recovery_epoch=int(
                os.environ.get(
                    "MOEGAMBIT_RECOVERY_EPOCH",
                    os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
                )
            ),
            replica_timeout=float(
                os.environ.get("MOEGAMBIT_ZERO2_REPLICATION_TIMEOUT", "300")
            ),
            hybrid_restore=env_bool(
                "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE", False
            ),
        )


class DeepSpeedRecoveryRuntime:
    def __init__(
        self, engine: Any, settings: DeepSpeedRuntimeSettings
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.zero2 = None
        self._original_take_model_step = engine._take_model_step
        self._checkpoint_in_progress = False

    def start(self) -> "DeepSpeedRecoveryRuntime":
        self._restore_after_elastic_restart()
        if self.settings.zero2:
            from moegambit_deepspeed.zero2 import DeepSpeedZero2Replica

            self.zero2 = DeepSpeedZero2Replica(
                self.engine, timeout=self.settings.replica_timeout
            )
            self.zero2.start(int(getattr(self.engine, "global_steps", 0)))
        self._install_optimizer_step_hook()
        return self

    def _restore_after_elastic_restart(self) -> None:
        if (
            not self.settings.hot_swap
            or self.settings.application_checkpoint
            or self.settings.recovery_epoch <= 0
            or self.settings.checkpoint_dir is None
        ):
            return
        self._report_phase("checkpoint_restore_start")
        load_path, _ = self.engine.load_checkpoint(
            str(self.settings.checkpoint_dir)
        )
        if load_path is None:
            raise RuntimeError(
                "DeepSpeed elastic restart could not load a checkpoint from "
                f"{self.settings.checkpoint_dir}"
            )
        self._report_phase("checkpoint_restore_done")
        if self.settings.hybrid_restore:
            self._restore_non_expert_from_peer()
        logger.warning(
            "MoEGambit restored DeepSpeed recovery epoch %d from %s",
            self.settings.recovery_epoch,
            load_path,
        )

    def _restore_non_expert_from_peer(self) -> None:
        from moegambit_deepspeed.hybrid_restore import (
            restore_non_expert_model_from_peer,
            validate_relaunch_checkpoint_steps,
        )
        import torch.distributed as dist

        failed_logical_node = int(
            os.environ["MOEGAMBIT_RECOVERY_FAILED_LOGICAL_NODE"]
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        failure_step = int(os.environ["MOEGAMBIT_RECOVERY_FAILURE_STEP"])
        local_checkpoint_step = int(getattr(self.engine, "global_steps", -1))
        checkpoint_steps: list[int | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(checkpoint_steps, local_checkpoint_step)
        expected_step = validate_relaunch_checkpoint_steps(
            (
                step
                for step in checkpoint_steps
                if step is not None
            ),
            failure_step=failure_step,
        )
        logger.warning(
            "MoEGambit recovery version selected checkpoint_step=%d "
            "failure_step=%d rollback_steps=%d",
            expected_step,
            failure_step,
            failure_step - expected_step,
        )
        self._report_phase("recovery_version_validated")
        first_rank = failed_logical_node * local_world_size
        replacement_ranks = range(
            first_rank, first_rank + local_world_size
        )
        self._report_phase("non_expert_peer_restore_start")
        summary = restore_non_expert_model_from_peer(
            self.engine,
            replacement_ranks=replacement_ranks,
            expected_step=expected_step,
        )
        summary["failure_step"] = failure_step
        summary["rollback_steps"] = failure_step - expected_step
        self._report_phase("non_expert_peer_restore_done")
        logger.warning(
            "MoEGambit single-stage hybrid restore complete: %s", summary
        )

    def _report_phase(self, phase: str) -> None:
        from moegambit.runtime.hot_spare import report_worker_phase

        rank = int(getattr(self.engine, "global_rank", 0))
        report_worker_phase(phase, rank)

    def _install_optimizer_step_hook(self) -> None:
        runtime = self

        def wrapped_take_model_step(*args, **kwargs):
            before = int(getattr(runtime.engine, "global_steps", 0))
            if runtime.zero2 is not None:
                runtime.zero2.before_step(before)
            result = runtime._original_take_model_step(*args, **kwargs)
            after = int(getattr(runtime.engine, "global_steps", before))
            if after > before:
                if runtime.zero2 is not None:
                    runtime.zero2.after_step(after)
                runtime._maybe_checkpoint(after)
            return result

        # PipelineEngine disables its public step() and invokes
        # _take_model_step() from the pipeline schedule. Hooking the common
        # optimizer boundary keeps ordinary and pipeline engines equivalent.
        self.engine._take_model_step = wrapped_take_model_step

    def _maybe_checkpoint(self, step: int) -> None:
        interval = self.settings.checkpoint_interval
        if (
            not self.settings.hot_swap
            or self.settings.application_checkpoint
            or self.settings.checkpoint_dir is None
            or interval <= 0
            or step % interval
            or self._checkpoint_in_progress
        ):
            return
        self._checkpoint_in_progress = True
        try:
            self.engine.save_checkpoint(
                str(self.settings.checkpoint_dir),
                tag=f"global_step{step}",
                client_state={
                    "moegambit_checkpoint_step": step,
                    "moegambit_recovery_epoch": self.settings.recovery_epoch,
                },
            )
        finally:
            self._checkpoint_in_progress = False

    def close(self) -> None:
        self.engine._take_model_step = self._original_take_model_step
        if self.zero2 is not None:
            self.zero2.close()


def attach_engine(engine: Any) -> DeepSpeedRecoveryRuntime | None:
    """Attach once after ``deepspeed.initialize`` constructs the engine."""
    existing = getattr(engine, "_moegambit_runtime", None)
    if existing is not None:
        return existing
    settings = DeepSpeedRuntimeSettings.from_env()
    if not settings.hot_swap and not settings.zero2:
        return None
    runtime = DeepSpeedRecoveryRuntime(engine, settings).start()
    engine._moegambit_runtime = runtime
    return runtime
