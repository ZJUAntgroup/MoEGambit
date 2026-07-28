"""Optional DeepSpeed engine hooks activated by MoEGambit feature flags."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moegambit.runtime.config import env_bool
from moegambit.runtime.recovery_handoff import (
    PROTOCOL_VERSION,
    handoff_root,
    rank_error_path,
    rank_ready_path,
    read_json,
    request_path,
    write_json_atomic,
)


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
    survivor_handoff: bool = False

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
        hot_swap = env_bool(
            "MOEGAMBIT_HOT_SWAP",
            env_bool("DEEPSPEED_MOEGAMBIT_HOT_SWAP", False),
        )
        recovery_strategy = os.environ.get(
            "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
            "torch_elastic_checkpoint_relaunch",
        )
        mixed_version_strategy = (
            recovery_strategy == "mixed_version_survivor_handoff"
        )
        hybrid_requested = env_bool(
            "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE",
            hot_swap and mixed_version_strategy,
        )
        if hybrid_requested and not mixed_version_strategy:
            raise ValueError(
                "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE requires "
                "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY="
                "mixed_version_survivor_handoff"
            )
        hybrid_restore = hot_swap and hybrid_requested
        return cls(
            hot_swap=hot_swap,
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
            hybrid_restore=hybrid_restore,
            survivor_handoff=(
                hybrid_restore
                and env_bool(
                    "MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF", True
                )
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
        self._step_lock = threading.RLock()
        self._recovery_freeze = threading.Event()
        self._handoff_stop = threading.Event()
        self._handoff_thread: threading.Thread | None = None
        self._survivor_payload: dict[str, Any] | None = None
        self.recovery_contract: dict[str, Any] | None = None

    def start(self) -> "DeepSpeedRecoveryRuntime":
        needs_optimizer_replica = (
            self.settings.zero2 or self.settings.survivor_handoff
        )
        if needs_optimizer_replica:
            from moegambit_deepspeed.zero2 import DeepSpeedZero2Replica

            self.zero2 = DeepSpeedZero2Replica(
                self.engine, timeout=self.settings.replica_timeout
            )
            self.zero2.prepare()
        self._restore_after_elastic_restart()
        if self.zero2 is not None:
            self.zero2.start(int(getattr(self.engine, "global_steps", 0)))
        self._install_optimizer_step_hook()
        if (
            self.settings.survivor_handoff
            and self.settings.recovery_epoch == 0
        ):
            self._start_handoff_watcher()
        return self

    def _restore_after_elastic_restart(self) -> None:
        if (
            not self.settings.hot_swap
            or self.settings.application_checkpoint
            or self.settings.recovery_epoch <= 0
            or self.settings.checkpoint_dir is None
        ):
            return
        from moegambit_deepspeed.checkpoint_commit import (
            resolve_committed_checkpoint,
        )
        import torch.distributed as dist

        selection: list[tuple[str | None, str | None]] = [(None, None)]
        if dist.get_rank() == 0:
            try:
                selection[0] = (
                    resolve_committed_checkpoint(self.settings.checkpoint_dir),
                    None,
                )
            except Exception as exc:
                selection[0] = (None, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(selection, src=0)
        tag, error = selection[0]
        if error is not None or tag is None:
            raise RuntimeError(
                f"DeepSpeed recovery checkpoint selection failed: {error}"
            )
        self._report_phase("checkpoint_restore_start")
        load_path, _ = self.engine.load_checkpoint(
            str(self.settings.checkpoint_dir),
            tag=tag,
        )
        if load_path is None:
            raise RuntimeError(
                "DeepSpeed elastic restart could not load a checkpoint from "
                f"{self.settings.checkpoint_dir}"
            )
        self._report_phase("checkpoint_restore_done")
        if self.settings.hybrid_restore:
            self._restore_mixed_version_from_handoff()
        logger.warning(
            "MoEGambit restored DeepSpeed recovery epoch %d from %s",
            self.settings.recovery_epoch,
            load_path,
        )

    def _restore_mixed_version_from_handoff(self) -> None:
        from moegambit_deepspeed.hybrid_restore import (
            restore_non_expert_model_from_peer,
            restore_non_expert_optimizer_from_peer,
            validate_mixed_version_steps,
        )
        from moegambit_deepspeed.survivor_handoff import (
            load_survivor_handoff,
            restore_survivor_handoff,
        )
        import torch.distributed as dist

        if not self.settings.survivor_handoff or self.zero2 is None:
            raise RuntimeError(
                "DeepSpeed mixed-version recovery requires survivor handoff "
                "and optimizer peer replication"
            )
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
        first_rank = failed_logical_node * local_world_size
        replacement_ranks = tuple(
            range(first_rank, first_rank + local_world_size)
        )
        rank = dist.get_rank()

        self._report_phase("survivor_handoff_restore_start")
        survivor_step = None
        if rank not in replacement_ranks:
            self._survivor_payload = load_survivor_handoff(
                root=handoff_root(),
                recovery_epoch=self.settings.recovery_epoch,
                rank=rank,
                expected_step=failure_step,
                expected_source_epoch=self.settings.recovery_epoch - 1,
                expected_logical_node=int(
                    os.environ["MOEGAMBIT_LOGICAL_NODE_RANK"]
                ),
                expected_physical_node=int(
                    os.environ["MOEGAMBIT_PHYSICAL_NODE_RANK"]
                ),
            )
            summary = restore_survivor_handoff(
                self.engine, self.zero2, self._survivor_payload
            )
            survivor_step = int(summary["step"])
            logger.warning(
                "MoEGambit restored survivor rank=%d from current-step "
                "handoff step=%d",
                rank,
                survivor_step,
            )
        survivor_steps: list[int | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(survivor_steps, survivor_step)
        reported_survivors = [
            step for step in survivor_steps if step is not None
        ]
        expected_survivors = (
            dist.get_world_size() - len(replacement_ranks)
        )
        if len(reported_survivors) != expected_survivors:
            raise RuntimeError(
                "mixed-version recovery is missing survivor handoffs: "
                f"expected={expected_survivors} "
                f"actual={len(reported_survivors)}"
            )
        checkpoint_step = validate_mixed_version_steps(
            (
                step
                for step in checkpoint_steps
                if step is not None
            ),
            reported_survivors,
            failure_step=failure_step,
        )
        self._report_phase("survivor_handoff_restore_done")
        logger.warning(
            "MoEGambit mixed-version contract checkpoint_step=%d "
            "resume_step=%d expert_staleness=%d rollback_steps=0",
            checkpoint_step,
            failure_step,
            failure_step - checkpoint_step,
        )
        self._report_phase("recovery_version_validated")
        self._report_phase("non_expert_peer_restore_start")
        model_summary = restore_non_expert_model_from_peer(
            self.engine,
            replacement_ranks=replacement_ranks,
            expected_step=failure_step,
        )
        optimizer_summary = restore_non_expert_optimizer_from_peer(
            self.engine,
            self.zero2,
            survivor_payload=self._survivor_payload,
            replacement_ranks=replacement_ranks,
            expected_step=failure_step,
        )
        local_steps: list[int | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(
            local_steps, int(getattr(self.engine, "global_steps", -1))
        )
        if set(local_steps) != {failure_step}:
            raise RuntimeError(
                "mixed-version recovery did not converge on the resume step: "
                f"expected={failure_step} actual={sorted(set(local_steps))}"
            )
        self.recovery_contract = {
            "mode": "mixed_version",
            "checkpoint_step": checkpoint_step,
            "resume_step": failure_step,
            "expert_staleness": failure_step - checkpoint_step,
            "rollback_steps": 0,
            "survivor_state": "current_step_handoff",
            "replacement_non_expert_model": "current_step_peer",
            "replacement_non_expert_optimizer": (
                "current_step_peer_replica"
            ),
            "replacement_rng": "current_step_peer",
            "replacement_expert_model": "checkpoint",
            "replacement_expert_optimizer": "checkpoint",
        }
        self._report_phase("non_expert_peer_restore_done")
        logger.warning(
            "MoEGambit mixed-version hybrid restore complete: %s",
            {
                **self.recovery_contract,
                "replacement_non_expert_model": model_summary,
                "replacement_non_expert_optimizer": optimizer_summary,
                "two_phase": False,
            },
        )

    def _start_handoff_watcher(self) -> None:
        root = handoff_root()
        next_epoch = self.settings.recovery_epoch + 1
        request = request_path(root, next_epoch)
        runtime = self
        watcher_started_at = time.time()

        def watch() -> None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(runtime.engine.device)
            while not runtime._handoff_stop.wait(0.1):
                if not request.is_file():
                    continue
                try:
                    if request.stat().st_mtime < watcher_started_at:
                        continue
                except OSError:
                    continue
                try:
                    command = read_json(request)
                    expected_identity = {
                        "protocol": PROTOCOL_VERSION,
                        "recovery_epoch": next_epoch,
                        "source_epoch": runtime.settings.recovery_epoch,
                        "physical_node": int(
                            os.environ.get(
                                "MOEGAMBIT_PHYSICAL_NODE_RANK", "-1"
                            )
                        ),
                        "logical_node": int(
                            os.environ.get(
                                "MOEGAMBIT_LOGICAL_NODE_RANK", "-1"
                            )
                        ),
                    }
                    mismatches = {
                        key: {
                            "expected": value,
                            "actual": command.get(key),
                        }
                        for key, value in expected_identity.items()
                        if command.get(key) != value
                    }
                    if mismatches:
                        raise RuntimeError(
                            "survivor handoff request identity mismatch: "
                            f"{mismatches}"
                        )
                    rank = int(getattr(runtime.engine, "global_rank", -1))
                    requested_ranks = {
                        int(item) for item in command.get("ranks", ())
                    }
                    if rank not in requested_ranks:
                        raise RuntimeError(
                            "survivor handoff request omitted local rank: "
                            f"rank={rank} requested={sorted(requested_ranks)}"
                        )
                    failure_step = int(command["failure_step"])
                    runtime._recovery_freeze.set()
                    with runtime._step_lock:
                        from moegambit_deepspeed.survivor_handoff import (
                            capture_survivor_handoff,
                        )

                        result = capture_survivor_handoff(
                            runtime.engine,
                            runtime.zero2,
                            root=root,
                            recovery_epoch=next_epoch,
                            failure_step=failure_step,
                        )
                    write_json_atomic(
                        rank_ready_path(
                            root,
                            next_epoch,
                            int(result["rank"]),
                        ),
                        result,
                    )
                    logger.warning(
                        "MoEGambit survivor handoff ready rank=%d step=%d "
                        "bytes=%d elapsed_s=%.2f",
                        result["rank"],
                        result["step"],
                        result["bytes"],
                        result["elapsed_s"],
                    )
                except BaseException as exc:
                    rank = int(getattr(runtime.engine, "global_rank", -1))
                    write_json_atomic(
                        rank_error_path(root, next_epoch, rank),
                        {
                            "rank": rank,
                            "recovery_epoch": next_epoch,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    logger.exception(
                        "MoEGambit survivor handoff failed rank=%d", rank
                    )
                return

        self._handoff_thread = threading.Thread(
            target=watch,
            name="moegambit-survivor-handoff",
            daemon=True,
        )
        self._handoff_thread.start()

    def _report_phase(self, phase: str) -> None:
        from moegambit.runtime.hot_spare import report_worker_phase

        rank = int(getattr(self.engine, "global_rank", 0))
        report_worker_phase(phase, rank)

    def _install_optimizer_step_hook(self) -> None:
        runtime = self

        def wrapped_take_model_step(*args, **kwargs):
            while True:
                while runtime._recovery_freeze.is_set():
                    time.sleep(0.1)
                runtime._step_lock.acquire()
                if runtime._recovery_freeze.is_set():
                    runtime._step_lock.release()
                    continue
                break
            try:
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
            finally:
                runtime._step_lock.release()

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
            tag = f"global_step{step}"
            self.engine.save_checkpoint(
                str(self.settings.checkpoint_dir),
                tag=tag,
                client_state={
                    "moegambit_checkpoint_step": step,
                    "moegambit_recovery_epoch": self.settings.recovery_epoch,
                },
                save_latest=False,
            )
            from moegambit_deepspeed.checkpoint_commit import (
                publish_checkpoint,
            )

            publish_checkpoint(
                self.engine,
                self.settings.checkpoint_dir,
                tag,
            )
        finally:
            self._checkpoint_in_progress = False

    def close(self) -> None:
        self._handoff_stop.set()
        if (
            self._handoff_thread is not None
            and self._handoff_thread is not threading.current_thread()
        ):
            self._handoff_thread.join(timeout=1.0)
        self.engine._take_model_step = self._original_take_model_step
        if self.zero2 is not None:
            self.zero2.close()

    def wait_for_failure_commit(self, step: int) -> None:
        if self.settings.survivor_handoff:
            if self.zero2 is None:
                raise RuntimeError(
                    "failure injection requires optimizer peer replication"
                )
            self.zero2.wait_until_replicated(int(step))

    def wait_for_recovery_preemption(self, step: int) -> None:
        """Hold a synthetic safe-point fault at ``step`` until retirement."""
        if not self.settings.survivor_handoff:
            raise RuntimeError(
                "safe-point preemption requires survivor handoff"
            )
        timeout = float(
            os.environ.get("MOEGAMBIT_RECOVERY_HANDOFF_TIMEOUT", "180")
        ) + 30.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            actual_step = int(getattr(self.engine, "global_steps", -1))
            if actual_step != int(step):
                raise RuntimeError(
                    "worker advanced beyond the committed failure step: "
                    f"expected={step} actual={actual_step}"
                )
            if self._handoff_stop.is_set():
                raise RuntimeError(
                    "recovery runtime closed before worker preemption"
                )
            if self._recovery_freeze.is_set():
                time.sleep(0.1)
            else:
                self._recovery_freeze.wait(0.1)
        raise TimeoutError(
            "worker was not retired after the committed failure step: "
            f"step={step} timeout_s={timeout:.1f}"
        )


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
