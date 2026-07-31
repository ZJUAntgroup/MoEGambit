"""Optional DeepSpeed engine hooks activated by MoEGambit feature flags."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moegambit.core.step_transaction import (
    FailureDecision,
    StepPhase,
    StepTransaction,
)
from moegambit.runtime.config import env_bool


logger = logging.getLogger(__name__)
RECOVERY_STRATEGY = "rank_in_process_hybrid"


@dataclass(frozen=True)
class DeepSpeedRuntimeSettings:
    hot_swap: bool
    zero2: bool
    checkpoint_dir: Path | None
    checkpoint_interval: int
    recovery_epoch: int
    replica_timeout: float
    hybrid_restore: bool = False
    inprocess_recovery: bool = False
    inprocess_replacement: bool = False

    @classmethod
    def from_env(cls) -> "DeepSpeedRuntimeSettings":
        hot_swap = env_bool(
            "MOEGAMBIT_HOT_SWAP",
            env_bool("DEEPSPEED_MOEGAMBIT_HOT_SWAP", False),
        )
        zero2 = env_bool(
            "MOEGAMBIT_ZERO2",
            env_bool("DEEPSPEED_MOEGAMBIT_ZERO2", False),
        )
        replica_timeout = (
            float(
                os.environ.get(
                    "MOEGAMBIT_ZERO2_REPLICATION_TIMEOUT", "300"
                )
            )
            if hot_swap or zero2
            else 300.0
        )
        if not hot_swap:
            return cls(
                hot_swap=False,
                zero2=zero2,
                checkpoint_dir=None,
                checkpoint_interval=0,
                recovery_epoch=0,
                replica_timeout=replica_timeout,
            )

        checkpoint = os.environ.get("MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR")
        interval = int(
            os.environ.get("MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL", "0")
        )
        if interval <= 0:
            raise ValueError(
                "DeepSpeed hot swap requires a positive "
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL"
            )
        recovery_strategy = os.environ.get(
            "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
            RECOVERY_STRATEGY,
        )
        inprocess_recovery = env_bool(
            "MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY", True
        )
        hybrid_restore = env_bool(
            "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE", True
        )
        if recovery_strategy != RECOVERY_STRATEGY:
            raise ValueError(
                "DeepSpeed hot swap supports only "
                f"MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY={RECOVERY_STRATEGY}; "
                f"got {recovery_strategy!r}"
            )
        if not inprocess_recovery:
            raise ValueError(
                "DeepSpeed hot swap requires "
                "MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY=1"
            )
        if not hybrid_restore:
            raise ValueError(
                "DeepSpeed hot swap requires "
                "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE=1"
            )
        if checkpoint is None:
            raise ValueError(
                "DeepSpeed hot swap requires "
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR and a positive "
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL"
            )
        if env_bool("MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF", False):
            raise ValueError(
                "MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF was removed; "
                f"use {RECOVERY_STRATEGY}"
            )
        inprocess_replacement = env_bool(
            "MOEGAMBIT_DEEPSPEED_INPROCESS_REPLACEMENT", False
        )
        if inprocess_replacement and not inprocess_recovery:
            raise ValueError(
                "MOEGAMBIT_DEEPSPEED_INPROCESS_REPLACEMENT is valid only "
                "inside rank-granular recovery"
            )
        return cls(
            hot_swap=hot_swap,
            zero2=zero2,
            checkpoint_dir=Path(checkpoint) if checkpoint else None,
            checkpoint_interval=interval,
            recovery_epoch=int(
                os.environ.get("MOEGAMBIT_RECOVERY_EPOCH", "0")
            ),
            replica_timeout=replica_timeout,
            hybrid_restore=hybrid_restore,
            inprocess_recovery=inprocess_recovery,
            inprocess_replacement=inprocess_replacement,
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
        self._peer_optimizer_state: dict[str, Any] | None = None
        self.recovery_contract: dict[str, Any] | None = None
        self.current_recovery_epoch = settings.recovery_epoch
        initial_step = int(getattr(engine, "global_steps", 0))
        self.step_transaction = StepTransaction(initial_step)
        self.last_failure_decision: FailureDecision | None = None

    def start(self) -> "DeepSpeedRecoveryRuntime":
        needs_optimizer_replica = (
            self.settings.zero2 or self.settings.hybrid_restore
        )
        if needs_optimizer_replica:
            from .zero2 import DeepSpeedZero2Replica

            if (
                self.settings.inprocess_recovery
                and self.settings.inprocess_replacement
            ):
                self._report_rank_phase(
                    "replacement_optimizer_prepare_start"
                )
            self.zero2 = DeepSpeedZero2Replica(
                self.engine, timeout=self.settings.replica_timeout
            )
            self.zero2.prepare()
            if (
                self.settings.inprocess_recovery
                and self.settings.inprocess_replacement
            ):
                self._report_rank_phase(
                    "replacement_optimizer_prepare_done"
                )
        if (
            self.settings.inprocess_recovery
            and self.settings.inprocess_replacement
        ):
            from .inprocess_recovery import (
                validate_replacement_process_groups,
            )

            validate_replacement_process_groups(
                self.engine,
                phase_callback=self._report_rank_phase,
            )
            self._restore_inprocess_hybrid()
        if self.zero2 is not None:
            initial_step = int(getattr(self.engine, "global_steps", 0))
            self.zero2.start(initial_step)
            self.zero2.wait_until_replicated(initial_step)
            self.step_transaction.mark_optimizer_snapshot(initial_step)
        self.step_transaction.begin_step(
            int(getattr(self.engine, "global_steps", 0))
        )
        if self.zero2 is not None:
            self.step_transaction.mark_step_safe(
                int(getattr(self.engine, "global_steps", 0))
            )
        if (
            self.settings.inprocess_recovery
            and self.settings.inprocess_replacement
        ):
            self._report_rank_event("rank_recovery_ready")
        self._install_optimizer_step_hook()
        return self

    def _restore_inprocess_hybrid(self) -> None:
        """Restore only the replacement while survivors retain live state."""
        from .checkpoint_commit import (
            checkpoint_step_from_tag,
            resolve_committed_checkpoint,
        )
        from .hybrid_restore import (
            restore_non_expert_model_from_peer,
            restore_non_expert_optimizer_from_peer,
            validate_checkpoint_base_steps,
        )
        from .inprocess_recovery import (
            wait_for_inprocess_recovery_gate,
        )
        import torch.distributed as dist

        if (
            not self.settings.hybrid_restore
            or self.settings.checkpoint_dir is None
            or self.zero2 is None
        ):
            raise RuntimeError(
                "in-process hybrid recovery requires checkpoint, "
                "hybrid restore, and optimizer peer replication"
            )
        failed_rank = int(
            os.environ["MOEGAMBIT_RECOVERY_FAILED_RANK"]
        )
        failure_step = int(
            os.environ["MOEGAMBIT_RECOVERY_FAILURE_STEP"]
        )
        rank = dist.get_rank()
        is_replacement = rank == failed_rank
        if is_replacement != self.settings.inprocess_replacement:
            raise RuntimeError(
                "in-process replacement identity mismatch: "
                f"rank={rank} failed_rank={failed_rank} "
                f"replacement_env={self.settings.inprocess_replacement}"
            )

        self._report_rank_phase("checkpoint_selection_start")
        tag = resolve_committed_checkpoint(
            self.settings.checkpoint_dir
        )
        checkpoint_step_value = validate_checkpoint_base_steps(
            (checkpoint_step_from_tag(tag),),
            failure_step=failure_step,
        )
        wait_for_inprocess_recovery_gate(
            "checkpoint_selected",
            metadata={
                "epoch": self.current_recovery_epoch,
                "failed_rank": failed_rank,
                "failure_step": failure_step,
                "checkpoint_tag": tag,
                "checkpoint_step": checkpoint_step_value,
            },
        )
        self._report_rank_phase("checkpoint_selection_done")

        if is_replacement:
            self._report_rank_phase("replacement_checkpoint_restore_start")
            load_path, _ = self.engine.load_checkpoint(
                str(self.settings.checkpoint_dir),
                tag=tag,
            )
            if load_path is None:
                raise RuntimeError(
                    "replacement could not load committed checkpoint "
                    f"{tag} from {self.settings.checkpoint_dir}"
                )
            loaded_checkpoint_step = int(
                getattr(self.engine, "global_steps", -1)
            )
            if loaded_checkpoint_step != checkpoint_step_value:
                raise RuntimeError(
                    "replacement loaded an unexpected checkpoint step: "
                    f"tag={tag} expected={checkpoint_step_value} "
                    f"actual={loaded_checkpoint_step}"
                )
            self._report_rank_phase("replacement_checkpoint_restore_done")
        self._report_rank_phase("checkpoint_restore_gate_start")
        wait_for_inprocess_recovery_gate(
            "checkpoint_restored",
            metadata={
                "epoch": self.current_recovery_epoch,
                "failed_rank": failed_rank,
                "checkpoint_tag": tag,
                "checkpoint_step": checkpoint_step_value,
            },
        )
        self._report_rank_phase("checkpoint_restore_gate_done")

        expected_local_step = (
            checkpoint_step_value if is_replacement else failure_step
        )
        actual_local_step = int(
            getattr(self.engine, "global_steps", -1)
        )
        if actual_local_step != expected_local_step:
            raise RuntimeError(
                "local hybrid recovery state is not aligned: "
                f"rank={rank} replacement={is_replacement} "
                f"expected={expected_local_step} "
                f"actual={actual_local_step}"
            )
        self._report_rank_phase("survivor_state_commit_start")
        wait_for_inprocess_recovery_gate(
            "survivor_state_committed",
            metadata={
                "epoch": self.current_recovery_epoch,
                "failed_rank": failed_rank,
                "failure_step": failure_step,
                "checkpoint_step": checkpoint_step_value,
            },
        )
        self._report_rank_phase("survivor_state_commit_done")

        self._report_rank_phase("non_expert_peer_restore_start")
        model_summary = restore_non_expert_model_from_peer(
            self.engine,
            replacement_ranks=(failed_rank,),
            expected_step=failure_step,
        )
        self._report_rank_phase("non_expert_model_restore_done")
        optimizer_summary = restore_non_expert_optimizer_from_peer(
            self.engine,
            self.zero2,
            peer_optimizer_state=self._peer_optimizer_state,
            replacement_ranks=(failed_rank,),
            expected_step=failure_step,
        )
        self._report_rank_phase("non_expert_optimizer_restore_done")

        # Pipeline replacement construction deliberately deferred this
        # collective so rank 0 can never broadcast random standby weights.
        synchronize_tied_weights = getattr(
            self.engine.module, "_synchronize_tied_weights", None
        )
        if callable(synchronize_tied_weights):
            synchronize_tied_weights()
        final_local_step = int(
            getattr(self.engine, "global_steps", -1)
        )
        if final_local_step != failure_step:
            raise RuntimeError(
                "in-process recovery did not converge on the resume step: "
                f"rank={rank} expected={failure_step} "
                f"actual={final_local_step}"
            )
        self._report_rank_phase("recovery_state_commit_start")
        wait_for_inprocess_recovery_gate(
            "recovery_state_committed",
            metadata={
                "epoch": self.current_recovery_epoch,
                "failed_rank": failed_rank,
                "failure_step": failure_step,
                "checkpoint_step": checkpoint_step_value,
            },
        )
        self._report_rank_phase("recovery_state_commit_done")
        self.recovery_contract = {
            "mode": "rank_in_process_hybrid",
            "checkpoint_step": checkpoint_step_value,
            "resume_step": failure_step,
            "expert_staleness": (
                failure_step - checkpoint_step_value
            ),
            "rollback_steps": 0,
            "survivor_state": "resident_cuda",
            "survivor_processes_restarted": 0,
            "replacement_ranks": [failed_rank],
            "replacement_non_expert_model": "current_step_peer",
            "replacement_non_expert_optimizer": (
                "current_step_peer_replica"
            ),
            "replacement_rng": "current_step_peer",
            "replacement_expert_model": "checkpoint",
            "replacement_expert_optimizer": "checkpoint",
        }
        self._report_rank_phase("non_expert_peer_restore_done")
        logger.warning(
            "MoEGambit in-process hybrid restore complete: %s",
            {
                **self.recovery_contract,
                "replacement_non_expert_model_summary": model_summary,
                "replacement_non_expert_optimizer_summary": (
                    optimizer_summary
                ),
                "two_phase": False,
            },
        )

    def _report_rank_event(
        self, kind: str, *, epoch: int | None = None, **payload: Any
    ) -> dict[str, Any] | None:
        from moegambit.runtime.hot_spare import request_worker_command

        rank = int(getattr(self.engine, "global_rank", 0))
        if epoch is not None:
            payload["epoch"] = int(epoch)
        command = request_worker_command(kind, rank, **payload)
        return dict(command) if command is not None else None

    def _report_rank_phase(self, phase: str) -> None:
        self._report_rank_event(
            "rank_recovery_phase",
            epoch=self.current_recovery_epoch,
            phase=phase,
        )

    def _wait_for_rank_recovery_command(
        self, step: int
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(
            os.environ.get(
                "MOEGAMBIT_INPROCESS_RECOVERY_TIMEOUT", "300"
            )
        )
        old_epoch = self.current_recovery_epoch
        while time.monotonic() < deadline:
            command = self._report_rank_event(
                "poll",
                epoch=old_epoch,
                state="rank_quiesced",
                global_step=int(step),
            )
            if command is None:
                raise RuntimeError(
                    "rank recovery has no configured coordinator"
                )
            if command.get("action") == "abort":
                raise RuntimeError(
                    "rank recovery aborted by coordinator: "
                    f"{command.get('reason', 'unknown reason')}"
                )
            if (
                command.get("recovery_mode") == "rank_in_process"
                and int(command.get("epoch", old_epoch)) > old_epoch
                and command.get("failed_rank") is not None
            ):
                if int(command.get("failure_step", -1)) != int(step):
                    raise RuntimeError(
                        "coordinator failure step does not match the local "
                        f"safe point: command={command.get('failure_step')} "
                        f"local={step}"
                    )
                return command
            time.sleep(0.2)
        raise TimeoutError(
            "timed out waiting for rank-granular recovery command at "
            f"step {step}"
        )

    def recover_in_process(self, step: int) -> dict[str, Any]:
        """Keep survivor tensors resident while replacing one failed rank."""
        if not self.settings.inprocess_recovery:
            raise RuntimeError("in-process DeepSpeed recovery is disabled")
        if self.settings.inprocess_replacement:
            raise RuntimeError(
                "replacement rank cannot enter the survivor rebuild path"
            )
        actual_step = int(getattr(self.engine, "global_steps", -1))
        if actual_step != int(step):
            raise RuntimeError(
                "survivor is not at the committed recovery step: "
                f"expected={step} actual={actual_step}"
            )
        if self.zero2 is None:
            raise RuntimeError(
                "in-process recovery requires optimizer peer replication"
            )

        self._recovery_freeze.set()
        command: dict[str, Any] | None = None
        old_zero2 = self.zero2
        try:
            command = self._wait_for_rank_recovery_command(step)
            recovery_epoch = int(command["epoch"])
            self.current_recovery_epoch = recovery_epoch
            self._report_rank_event(
                "rank_recovery_phase",
                epoch=recovery_epoch,
                phase="optimizer_replica_capture_start",
            )
            old_zero2.wait_until_replicated(int(step))
            peer_optimizer = old_zero2.export_peer_snapshots(int(step))
            for manager in old_zero2.managers.values():
                manager.stop_transport()
            self._peer_optimizer_state = peer_optimizer
            self._report_rank_event(
                "rank_recovery_phase",
                epoch=recovery_epoch,
                phase="process_group_rebuild_start",
            )
            from .inprocess_recovery import (
                rebuild_engine_process_groups,
            )

            rebuild_summary = rebuild_engine_process_groups(
                self.engine,
                command,
                phase_callback=self._report_rank_phase,
            )
            self._report_rank_phase("process_group_rebuild_done")
            self._restore_inprocess_hybrid()

            old_zero2.close()
            self._peer_optimizer_state = None
            from .zero2 import DeepSpeedZero2Replica

            self.zero2 = DeepSpeedZero2Replica(
                self.engine, timeout=self.settings.replica_timeout
            )
            self.zero2.prepare()
            self.zero2.start(int(step))
            self._report_rank_event(
                "rank_recovery_ready",
                epoch=recovery_epoch,
                global_step=int(step),
            )
            self._recovery_freeze.clear()
            return {
                **rebuild_summary,
                "recovery_contract": self.recovery_contract,
            }
        except BaseException as exc:
            logger.exception(
                "DeepSpeed in-process recovery failed at step %d", step
            )
            try:
                self._report_rank_event(
                    "rank_recovery_failed",
                    epoch=(
                        int(command["epoch"])
                        if command is not None
                        else self.current_recovery_epoch
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception(
                    "could not report in-process recovery failure"
                )
            raise

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
                if runtime.step_transaction.committed_step != before:
                    runtime.step_transaction = StepTransaction(before)
                    if runtime.zero2 is not None:
                        runtime.step_transaction.mark_optimizer_snapshot(before)
                runtime.step_transaction.begin_step(before)
                runtime.step_transaction.enter(StepPhase.OPTIMIZER_BEFORE)
                if runtime.zero2 is not None:
                    runtime.zero2.before_step(before)
                    runtime.step_transaction.mark_optimizer_snapshot(before)
                runtime.step_transaction.enter(StepPhase.OPTIMIZER_DURING)
                result = runtime._original_take_model_step(*args, **kwargs)
                after = int(getattr(runtime.engine, "global_steps", before))
                if after > before:
                    runtime.step_transaction.mark_optimizer_applied(after)
                    if runtime.zero2 is not None:
                        runtime.zero2.after_step(after)
                        runtime.zero2.wait_until_replicated(after)
                        runtime.step_transaction.mark_optimizer_snapshot(after)
                        runtime.step_transaction.mark_step_safe(after)
                    runtime._maybe_checkpoint(after)
                else:
                    runtime.step_transaction.mark_optimizer_noop()
                return result
            except BaseException as exc:
                runtime._record_training_failure(exc)
                raise
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
            or self.settings.checkpoint_dir is None
            or interval <= 0
            or step % interval
            or self._checkpoint_in_progress
        ):
            return
        self._checkpoint_in_progress = True
        self.step_transaction.begin_checkpoint(step)
        try:
            tag = f"global_step{step}"
            self.engine.save_checkpoint(
                str(self.settings.checkpoint_dir),
                tag=tag,
                client_state={
                    "moegambit_checkpoint_step": step,
                    "moegambit_recovery_epoch": (
                        self.current_recovery_epoch
                    ),
                },
                save_latest=False,
            )
            from .checkpoint_commit import (
                publish_checkpoint,
            )

            self.step_transaction.begin_checkpoint_commit(step)
            publish_checkpoint(
                self.engine,
                self.settings.checkpoint_dir,
                tag,
            )
            self.step_transaction.mark_checkpoint_committed(step)
        except BaseException as exc:
            self._record_training_failure(exc)
            raise
        finally:
            self._checkpoint_in_progress = False

    def record_training_phase(self, phase: StepPhase | str) -> None:
        """Record forward/backward boundaries exposed by a workload wrapper."""

        self.step_transaction.enter(phase)

    def _record_training_failure(
        self, exc: BaseException
    ) -> FailureDecision:
        decision = self.step_transaction.decide_failure()
        if self.last_failure_decision is not None and id(exc) == getattr(
            self, "_last_failure_id", None
        ):
            return self.last_failure_decision
        self._last_failure_id = id(exc)
        self.last_failure_decision = decision
        logger.error(
            "DeepSpeed training failure phase=%s action=%s resume_step=%d: %s",
            decision.phase.value,
            decision.action.value,
            decision.resume_step,
            exc,
        )
        try:
            self._report_rank_event(
                "rank_recovery_phase",
                epoch=self.current_recovery_epoch,
                phase="training_failure_classified",
                training_phase=decision.phase.value,
                failure_action=decision.action.value,
                safe_step=decision.resume_step,
                optimizer_dirty=decision.optimizer_dirty,
            )
        except Exception:
            logger.exception("could not report DeepSpeed failure phase")
        return decision

    def record_training_failure(
        self, exc: BaseException
    ) -> FailureDecision:
        return self._record_training_failure(exc)

    def close(self) -> None:
        self.engine._take_model_step = self._original_take_model_step
        if self.zero2 is not None:
            self.zero2.close()

    def wait_for_failure_commit(self, step: int) -> None:
        if not self.settings.inprocess_recovery:
            raise RuntimeError("rank in-process recovery is disabled")
        if self.zero2 is None:
            raise RuntimeError(
                "failure injection requires optimizer peer replication"
            )
        self.zero2.wait_until_replicated(int(step))

    def wait_for_recovery_preemption(self, step: int) -> None:
        """Hold the synthetic safe point during rank-granular recovery."""
        if not self.settings.inprocess_recovery:
            raise RuntimeError("rank in-process recovery is disabled")
        self.recover_in_process(step)


def attach_engine(engine: Any) -> DeepSpeedRecoveryRuntime | None:
    """Attach once after ``deepspeed.initialize`` constructs the engine."""
    existing = getattr(engine, "_moegambit_runtime", None)
    if existing is not None:
        return existing
    settings = DeepSpeedRuntimeSettings.from_env()
    if not settings.hot_swap and not settings.zero2:
        return None
    runtime = DeepSpeedRecoveryRuntime(engine, settings)
    if settings.inprocess_recovery and settings.inprocess_replacement:
        runtime._report_rank_phase("replacement_runtime_attach_start")
    runtime.start()
    if settings.inprocess_recovery and settings.inprocess_replacement:
        runtime._report_rank_phase("replacement_runtime_attach_done")
    engine._moegambit_runtime = runtime
    return runtime
