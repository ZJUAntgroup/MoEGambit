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
        load_path, _ = self.engine.load_checkpoint(
            str(self.settings.checkpoint_dir)
        )
        if load_path is None:
            raise RuntimeError(
                "DeepSpeed elastic restart could not load a checkpoint from "
                f"{self.settings.checkpoint_dir}"
            )
        logger.warning(
            "MoEGambit restored DeepSpeed recovery epoch %d from %s",
            self.settings.recovery_epoch,
            load_path,
        )

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
