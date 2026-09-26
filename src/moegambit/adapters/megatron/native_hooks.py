"""High-level hooks used by the small Megatron source patch.

The methods in this module own MoEGambit policy.  Patched Megatron files only
publish framework lifecycle events and pass the framework objects that the
adapter cannot discover safely on its own.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from inspect import signature
import logging
import os
import random
import sys
from typing import Any, Callable, Optional

from ...core.step_transaction import (
    FailureAction,
    FailureDecision,
    StepPhase,
    StepTransaction,
)

logger = logging.getLogger(__name__)


@dataclass
class StartupState:
    rebuild: bool
    saved_load: Any = None
    saved_no_load_optim: Any = None
    saved_no_load_rng: Any = None
    use_expert_sidecar: bool = False


@dataclass
class ErrorDisposition:
    handled: bool
    resume_iteration: Optional[int] = None


@dataclass
class IterationDisposition:
    continue_loop: bool
    floating_point_operations: Any


class MegatronNativeHooks:
    """Adapter-owned implementation behind Megatron's hook-only patch."""

    def __init__(self) -> None:
        self._crash_next_step = int(os.environ.get("CRASH_AT_STEP", "-1"))
        self._crash_interval = int(os.environ.get("CRASH_INTERVAL", "0"))
        self._crash_rank = int(os.environ.get("CRASH_RANK", "0"))
        self._crash_count = 0
        self._crash_rng = random.Random(int(os.environ.get("CRASH_SEED", "42")))
        self._step_transaction = StepTransaction()
        self._last_failure_decision: Optional[FailureDecision] = None

    def add_moe_arguments(self, group: Any) -> None:
        """Register compatibility CLI options outside Megatron's argument code."""

        boolean_options = {
            "--moe-moegambit-enable": "Master switch for MoEGambit recovery.",
            "--moe-moegambit-health-mask": "Enable expert health masking.",
            "--moe-moegambit-rank-quarantine": "Enable rank quarantine.",
            "--moe-moegambit-dispatch-quarantine-assert": (
                "Assert that quarantined EP ranks receive no tokens."
            ),
            "--moe-moegambit-dispatch-sanitize": (
                "Sanitize dispatch splits for quarantined EP ranks."
            ),
            "--moe-moegambit-expert-directory": "Enable the expert directory.",
            "--moe-moegambit-replacement-protocol": "Enable replacement registration.",
            "--moe-moegambit-group-rebuild": "Enable safe-point group rebuild.",
            "--moe-moegambit-dispatch-topology-refresh": (
                "Refresh dispatch topology after group rebuild."
            ),
            "--moe-moegambit-dense-param-sync": "Synchronize dense parameters from a peer.",
            "--moe-moegambit-stale-expert-restore": "Restore stale expert weights.",
            "--moe-moegambit-recovery-controller": "Enable recovery orchestration.",
            "--moe-moegambit-deferred-optimizer-load": (
                "Load expert optimizer state asynchronously."
            ),
            "--moe-moegambit-force-checkpoint-restart": (
                "Force full-checkpoint recovery."
            ),
            "--moe-moegambit-degraded-mode-policy": "Enable degraded-mode policy.",
            "--moe-moegambit-reintegration-barrier": "Enable reintegration gating.",
            "--moe-moegambit-gap-aware-recovery": "Enable gap-aware path selection.",
            "--moe-moegambit-fault-injection": "Enable fault injection.",
            "--moe-moegambit-restart-in-place": "Enable restart-in-place simulation.",
            "--moe-moegambit-full-peer-recovery": "Restore all state from a DP peer.",
            "--moe-moegambit-hot-spare-pool": "Enable the hot-spare pool.",
        }
        for option, help_text in boolean_options.items():
            group.add_argument(option, action="store_true", help=help_text)

        default_on_options = {
            "hybrid-expert-restore": (
                "Restore expert weights on the hybrid path.",
                "Skip expert weight restore on the hybrid path.",
            ),
            "expert-opt-restore": (
                "Restore expert optimizer state.",
                "Skip expert optimizer-state restore.",
            ),
            "weights-first-recovery": (
                "Restore weights before optimizer state.",
                "Disable weights-first recovery.",
            ),
            "defer-optimizer-load": (
                "Defer expert optimizer-state loading.",
                "Load expert optimizer state synchronously.",
            ),
        }
        for suffix, (enable_help, disable_help) in default_on_options.items():
            dest = f"moe_moegambit_{suffix.replace('-', '_')}"
            group.add_argument(
                f"--moe-moegambit-{suffix}",
                dest=dest,
                action="store_true",
                default=True,
                help=enable_help,
            )
            group.add_argument(
                f"--no-moe-moegambit-{suffix}",
                dest=dest,
                action="store_false",
                help=disable_help,
            )

        group.add_argument("--moe-moegambit-gap-threshold", type=int, default=100)
        group.add_argument(
            "--moe-moegambit-recovery-policy-type",
            choices=[
                "threshold",
                "rank_exposure_guarded_hybrid",
                "expert_staleness_guarded",
            ],
            default="threshold",
        )
        group.add_argument("--moe-moegambit-delta-time-min-gap", type=int, default=32)
        group.add_argument("--moe-moegambit-max-single-gap", type=int, default=192)
        group.add_argument(
            "--moe-moegambit-exposure-window-steps", type=int, default=20000
        )
        group.add_argument(
            "--moe-moegambit-max-expert-staleness-density",
            "--moe-moegambit-max-rank-stale-exposure",
            dest="moe_moegambit_max_rank_stale_exposure",
            type=float,
            default=0.1,
        )
        group.add_argument("--moe-moegambit-policy-margin", type=float, default=0.10)
        group.add_argument("--moe-moegambit-degraded-tau-c", type=float, default=0.5)
        group.add_argument("--moe-moegambit-degraded-t-max", type=int, default=1000)
        group.add_argument("--moe-moegambit-degraded-s-max", type=int, default=500)
        group.add_argument("--moe-moegambit-num-hot-spares", type=int, default=0)

    def routing_expert_bias(
        self,
        expert_bias: Optional[Any],
        *,
        layer_number: Optional[int],
        device: Any,
    ) -> Optional[Any]:
        """Combine Megatron's load-balancing bias with recovery preference."""

        if layer_number is None:
            return expert_bias
        from .moe.preferential_routing import get_all_preferential_routing_managers

        manager = get_all_preferential_routing_managers().get(layer_number)
        if manager is None or not manager.has_active_sessions():
            return expert_bias
        recovery_bias = manager.get_bias_tensor(device)
        if expert_bias is None:
            return recovery_bias
        return expert_bias + recovery_bias

    def after_grad_finalize(self) -> None:
        """Advance adapter-owned routing recovery windows."""

        from .moe.preferential_routing import get_all_preferential_routing_managers

        for manager in get_all_preferential_routing_managers().values():
            manager.step_all()

    # Training lifecycle -------------------------------------------------

    def is_rebuild_mode(self) -> bool:
        from .elastic_client import is_rebuild_mode

        return bool(is_rebuild_mode())

    def allow_world_collectives(self) -> bool:
        """Whether framework cold-start collectives are safe on this worker."""

        return not self.is_rebuild_mode()

    def prepare_pretrain(self, args: Any) -> StartupState:
        """Prepare normal or replacement startup without leaking policy to Megatron."""

        if getattr(args, "moe_moegambit_enable", False):
            actual_kind = "moe" if getattr(args, "num_experts", None) else "dense"
            configured_kind = os.environ.get("MOEGAMBIT_MODEL_KIND", "moe").lower()
            if configured_kind != actual_kind:
                raise ValueError(
                    "Megatron model kind mismatch: training arguments select "
                    f"{actual_kind}, but MOEGAMBIT_MODEL_KIND={configured_kind!r}; "
                    "set MOEGAMBIT_MODEL_KIND=dense on every training and "
                    "watcher node for a dense job"
                )

        from .elastic_client import (
            elastic_expert_sidecar_available,
            elastic_sanitize_recovery_env_for_startup,
        )
        from .integration import bootstrap_control_plane

        rebuild = self.is_rebuild_mode()
        elastic_sanitize_recovery_env_for_startup()
        bootstrap_control_plane()
        state = StartupState(rebuild=rebuild)
        if not rebuild:
            return state

        state.saved_load = args.load
        state.saved_no_load_optim = args.no_load_optim
        state.saved_no_load_rng = args.no_load_rng
        state.use_expert_sidecar = bool(
            getattr(args, "num_experts", None)
        ) and elastic_expert_sidecar_available()
        args.no_load_optim = True
        args.no_load_rng = True
        if state.use_expert_sidecar:
            args.load = None
        args.enable_gloo_process_groups = False
        args.moe_moegambit_weights_first_recovery = False
        args.moe_moegambit_async_recovery = False
        logger.warning(
            "[elastic] REBUILD MODE: model=%s checkpoint_base=%s; "
            "peer state will be synchronized at the current step",
            "moe" if getattr(args, "num_experts", None) else "dense",
            "packed_rank_sidecar" if state.use_expert_sidecar else "full_checkpoint",
        )
        return state

    def complete_model_setup(
        self,
        state: StartupState,
        *,
        args: Any,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        load_checkpoint: Any,
        checkpointing_context: Any,
    ) -> None:
        """Initialize the runtime and complete replacement state transfer."""

        from .elastic_client import (
            elastic_report_recovery_phase,
            elastic_restore_expert_sidecar,
        )
        from .integration import finalize_replacement, initialize_runtime

        initialize_runtime(model, optimizer, scheduler, args=args)
        if not state.rebuild:
            return

        args.load = state.saved_load
        if state.use_expert_sidecar:
            try:
                summary = elastic_restore_expert_sidecar(model, optimizer)
                elastic_report_recovery_phase(
                    "checkpoint_loaded",
                    checkpoint_source="expert_packed_sidecar",
                    expert_sidecar=summary,
                )
            except Exception:
                logger.exception(
                    "[elastic] packed expert sidecar restore failed; loading the "
                    "authoritative full checkpoint"
                )
                args.no_load_optim = True
                args.no_load_rng = True
                (
                    args.iteration,
                    args.num_floating_point_operations_so_far,
                ) = load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    checkpointing_context=checkpointing_context,
                )
                elastic_report_recovery_phase(
                    "checkpoint_loaded",
                    checkpoint_source="full_checkpoint_after_sidecar_fallback",
                )
        elastic_report_recovery_phase("model_optimizer_ready")
        args.no_load_optim = state.saved_no_load_optim
        args.no_load_rng = state.saved_no_load_rng
        finalize_replacement(model, optimizer, scheduler)
        logger.warning("[elastic] REBUILD MODE: param sync complete, joining training loop")

    def align_resume_state(self, args: Any, scheduler: Any = None) -> None:
        from .elastic_client import elastic_align_resume_state

        if not self.is_rebuild_mode():
            return
        resume_iteration = int(os.environ.get("ELASTIC_RESUME_ITERATION", "-1"))
        if resume_iteration < 0:
            return
        resume_iteration = elastic_align_resume_state(args, scheduler, resume_iteration)
        if resume_iteration is None or resume_iteration < 0:
            return
        args.num_floating_point_operations_so_far = getattr(
            args, "num_floating_point_operations_so_far", 0
        )
        logger.warning(
            "[elastic] REBUILD MODE: resuming replacement at iteration %d "
            "(consumed_train_samples=%d)",
            args.iteration,
            args.consumed_train_samples,
        )

    def initialize_training(
        self, model: Any, args: Any, optimizer: Any, scheduler: Any
    ) -> None:
        from .moe_integration import maybe_initialize_moegambit_moe

        maybe_initialize_moegambit_moe(
            model, args, optimizer=optimizer, opt_param_scheduler=scheduler
        )

    def maybe_inject_checkpoint_restart_crash(
        self, step: int, *, torch_module: Any
    ) -> None:
        if self._crash_next_step < 0 or step < self._crash_next_step:
            return
        distributed = torch_module.distributed
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        world_size = distributed.get_world_size() if distributed.is_initialized() else 1
        target_rank = (
            self._crash_rng.choice(range(world_size))
            if self._crash_rank < 0
            else self._crash_rank
        )
        self._crash_count += 1
        print(
            f"[CRASH INJECT #{self._crash_count}] target_rank={target_rank}, "
            f"my_rank={rank}, step={step}",
            flush=True,
        )
        self._crash_next_step = (
            step + self._crash_interval if self._crash_interval > 0 else -1
        )
        sys.exit(1)

    def iteration_prologue(self, iteration: int) -> None:
        from .elastic_client import (
            elastic_post_rebuild_iteration_barrier,
            elastic_trace_post_rebuild_phase,
        )

        elastic_post_rebuild_iteration_barrier(iteration)
        elastic_trace_post_rebuild_phase("iteration_prologue_start", iteration)

    def iteration_safe_point(
        self,
        iteration: int,
        *,
        args: Any,
        num_floating_point_operations: Any,
        num_microbatches: int,
    ) -> int:
        """Run recovery work at the framework-owned iteration boundary."""

        from megatron.core import mpu

        from .elastic_client import (
            elastic_post_rebuild_iteration_barrier,
            elastic_trace_post_rebuild_phase,
        )
        from .integration import iteration_boundary
        from .moe_integration import (
            moegambit_before_iteration,
            moegambit_clear_checkpoint_restart,
            moegambit_get_checkpoint_restart_decision,
            moegambit_is_checkpoint_restart_requested,
            moegambit_pipeline_begin_iteration,
            moegambit_snapshot_iteration,
        )

        elastic_trace_post_rebuild_phase("iteration_safe_point_start", iteration)
        resume_iteration = iteration_boundary(iteration)
        if resume_iteration != iteration:
            logger.warning(
                "[elastic] Iteration %d: pause requested, entering rebuild...", iteration
            )
            if resume_iteration is not None and resume_iteration >= 0:
                iteration = resume_iteration
                args.curr_iteration = iteration
            logger.warning(
                "[elastic] Rebuild complete, resuming at iteration %d", iteration
            )
        elastic_trace_post_rebuild_phase("iteration_safe_point_done", iteration)
        if self._step_transaction.committed_step != int(iteration):
            self._step_transaction = StepTransaction(int(iteration))
        self._step_transaction.begin_step(iteration)
        self._last_failure_decision = None

        elastic_trace_post_rebuild_phase("moegambit_before_iteration_start", iteration)
        moegambit_before_iteration(iteration)
        elastic_trace_post_rebuild_phase("moegambit_before_iteration_done", iteration)
        elastic_post_rebuild_iteration_barrier(iteration)

        if moegambit_is_checkpoint_restart_requested():
            decision = moegambit_get_checkpoint_restart_decision()
            checkpoint_iteration = (
                decision.latest_checkpoint_step
                if decision is not None and decision.latest_checkpoint_step >= 0
                else -1
            )
            logger.warning(
                "MOEGAMBIT-MoE: checkpoint restart completed at iteration %d "
                "(ckpt_iter=%d, gap=%d); continuing with restored experts",
                iteration,
                checkpoint_iteration,
                iteration - checkpoint_iteration if checkpoint_iteration >= 0 else -1,
            )
            moegambit_clear_checkpoint_restart()

        moegambit_snapshot_iteration(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
            consumed_valid_samples=getattr(args, "consumed_valid_samples", 0),
            num_floating_point_operations_so_far=num_floating_point_operations,
        )
        pipeline_size = mpu.get_pipeline_model_parallel_world_size()
        if pipeline_size > 1:
            moegambit_pipeline_begin_iteration(
                step=iteration,
                pp_rank=mpu.get_pipeline_model_parallel_rank(),
                pp_size=pipeline_size,
                num_microbatches=num_microbatches,
            )
        return iteration

    def handle_train_step_error(
        self, exc: RuntimeError, *, args: Any, iteration: int
    ) -> ErrorDisposition:
        """Classify a distributed failure and invoke the selected recovery path."""

        message = str(exc).lower()
        is_communication_error = any(
            marker in message
            for marker in (
                "nccl",
                "ncclsystemerror",
                "ncclremoteerror",
                "ncclinternalerror",
                "unhandled system error",
                "connection reset",
                "broken pipe",
                "timed out",
                "peer failure",
                "remote process exited",
            )
        )
        if not is_communication_error or not getattr(
            args, "moe_moegambit_enable", False
        ):
            return ErrorDisposition(handled=False)

        logger.exception(
            "MOEGAMBIT-MoE: communication error in iteration %d: %s",
            iteration,
            exc,
        )
        from .integration import get_runtime

        decision = self._step_transaction.decide_failure()
        self._last_failure_decision = decision
        if decision.action is not FailureAction.REPLAY_STEP:
            runtime = get_runtime()
            if runtime is not None:
                runtime.fail_closed(
                    exc,
                    reason="unsafe_megatron_step_replay",
                    evidence={
                        "phase": decision.phase.value,
                        "action": decision.action.value,
                        "resume_step": decision.resume_step,
                        "optimizer_dirty": decision.optimizer_dirty,
                    },
                )
            logger.critical(
                "MOEGAMBIT-MoE: refusing bookkeeping-only rollback at "
                "iteration %d phase=%s action=%s",
                iteration,
                decision.phase.value,
                decision.action.value,
            )
            return ErrorDisposition(handled=False)

        from megatron.core import mpu
        import torch

        from .integration import on_distributed_error
        from .moe_integration import (
            moegambit_pipeline_on_failure,
            moegambit_report_hard_failure,
        )

        if on_distributed_error(exc):
            runtime = get_runtime()
            resume = runtime.resume_step if runtime is not None else iteration
            self._step_transaction = StepTransaction(int(resume))
            return ErrorDisposition(handled=True, resume_iteration=resume)

        moegambit_report_hard_failure(
            failed_rank=-1,
            reason=str(exc),
            step=iteration,
            mid_iteration=True,
            exception=exc,
        )
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            moegambit_pipeline_on_failure(
                failed_stage=mpu.get_pipeline_model_parallel_rank(),
                failed_rank=torch.distributed.get_rank(),
                step=iteration,
                reason=str(exc),
                nccl_healthy=False,
            )
        return ErrorDisposition(handled=True)

    def finish_train_step(
        self,
        iteration: int,
        *,
        args: Any,
        data_iterators: Any,
        model: Any,
        optimizer: Any,
        num_floating_point_operations: Any,
    ) -> IterationDisposition:
        """Apply rollback/replay policy after Megatron finishes a train step."""

        from megatron.core import mpu

        from .moe_integration import (
            moegambit_after_iteration,
            moegambit_complete_replay,
            moegambit_exceeded_max_replays,
            moegambit_get_reintegration_summary,
            moegambit_has_pending_replacements,
            moegambit_is_current_iteration_invalid,
            moegambit_is_reintegration_pending,
            moegambit_is_replay_pending,
            moegambit_is_waiting_for_replacement,
            moegambit_pipeline_complete_replay,
            moegambit_pipeline_initiate_rollback,
            moegambit_rollback_iteration,
        )

        pipeline_size = mpu.get_pipeline_model_parallel_world_size()
        if moegambit_is_current_iteration_invalid():
            fp_ops_ref = [num_floating_point_operations]
            moegambit_rollback_iteration(
                args=args,
                data_iterators=data_iterators,
                num_fp_ops_ref=fp_ops_ref,
            )
            if pipeline_size > 1:

                def clear_grad() -> None:
                    for model_chunk in model:
                        model_chunk.zero_grad_buffer()
                    optimizer.zero_grad()

                result = moegambit_pipeline_initiate_rollback(
                    clear_grad_fn=clear_grad
                )
                if result is not None:
                    logger.warning(
                        "MOEGAMBIT-MoE: pipeline rollback at iteration %d "
                        "(sync=%s, grad_cleared=%s, nccl_reset=%s)",
                        iteration,
                        result.all_stages_synced,
                        result.grad_buffers_cleared,
                        result.nccl_communicator_reset,
                    )
            if moegambit_exceeded_max_replays():
                logger.error(
                    "MOEGAMBIT-MoE: iteration %d exceeded max replay attempts",
                    iteration,
                )
            if moegambit_is_waiting_for_replacement():
                logger.warning(
                    "MOEGAMBIT-MoE: iteration %d is waiting for replacement",
                    iteration,
                )
            elif moegambit_has_pending_replacements():
                logger.warning(
                    "MOEGAMBIT-MoE: iteration %d has pending replacements",
                    iteration,
                )
            if moegambit_is_reintegration_pending():
                logger.warning(
                    "MOEGAMBIT-MoE: iteration %d reintegration pending: %s",
                    iteration,
                    moegambit_get_reintegration_summary(),
                )
            moegambit_after_iteration(iteration)
            return IterationDisposition(
                continue_loop=True,
                floating_point_operations=fp_ops_ref[0],
            )

        if moegambit_is_replay_pending():
            logger.warning("MOEGAMBIT-MoE: iteration %d replay succeeded", iteration)
            moegambit_complete_replay(data_iterators)
            if pipeline_size > 1:
                moegambit_pipeline_complete_replay(success=True)
        moegambit_after_iteration(iteration)
        return IterationDisposition(
            continue_loop=False,
            floating_point_operations=num_floating_point_operations,
        )

    # Thin event forwarding ---------------------------------------------

    def update_step(self, step: int, *, phase: str, step_tag: int) -> None:
        from .elastic_client import elastic_client_update_step

        phase_map = {
            "iteration_safe_point": StepPhase.SAFE_POINT,
            "forward_backward": StepPhase.FORWARD_BACKWARD,
            "optimizer_step": StepPhase.OPTIMIZER_BEFORE,
            "checkpoint": StepPhase.CHECKPOINT_BEFORE,
        }
        transaction_phase = phase_map.get(phase)
        if transaction_phase is not None:
            self._step_transaction.enter(transaction_phase)
        elastic_client_update_step(step, phase=phase, step_tag=step_tag)

    def trace(self, phase: str, step: int, **extra: Any) -> None:
        from .elastic_client import elastic_trace_post_rebuild_phase

        elastic_trace_post_rebuild_phase(phase, step, **extra)

    def clear_trace(self) -> None:
        from .elastic_client import elastic_clear_post_rebuild_trace

        elastic_clear_post_rebuild_trace()

    def report_phase(self, phase: str, **extra: Any) -> None:
        from .elastic_client import elastic_report_recovery_phase

        elastic_report_recovery_phase(phase, **extra)

    def data_ready(self) -> None:
        if self.is_rebuild_mode():
            self.report_phase("data_ready")

    def optimizer_may_commit(self) -> bool:
        from .moe_integration import (
            moegambit_mark_optimizer_skipped,
            moegambit_should_commit_optimizer,
        )

        allowed = moegambit_should_commit_optimizer()
        if not allowed:
            moegambit_mark_optimizer_skipped(reason="iteration_invalidated")
        return allowed

    def before_optimizer_step(self, step: int) -> None:
        from .integration import before_optimizer_step

        self._step_transaction.enter(StepPhase.OPTIMIZER_BEFORE)
        before_optimizer_step(step)
        # From this point until the framework reports success, optimizer state
        # is conservatively treated as partially mutated.
        self._step_transaction.enter(StepPhase.OPTIMIZER_DURING)

    def mark_optimizer_committed(self, step: int, *, committed: bool) -> None:
        from .moe_integration import moegambit_mark_optimizer_committed

        if committed:
            moegambit_mark_optimizer_committed()
            self._step_transaction.mark_optimizer_applied(int(step) + 1)
        else:
            self._step_transaction.mark_optimizer_skipped(int(step) + 1)

    def after_optimizer_step(self, step: int, *, committed: bool) -> None:
        from .integration import after_optimizer_step

        after_optimizer_step(step, committed=committed)
        next_step = int(step) + 1
        self._step_transaction.mark_optimizer_snapshot(next_step)
        self._step_transaction.mark_step_safe(next_step)

    def commit_iteration(self, step: int) -> bool:
        from .integration import commit_iteration

        return bool(commit_iteration(step))

    # Checkpoint lifecycle ----------------------------------------------

    def checkpoint_pre_save(self, iteration: int, state_dict: Any) -> Any:
        from .moe_integration import (
            moegambit_is_initialized,
            moegambit_pre_save_checkpoint,
        )

        self._step_transaction.begin_checkpoint(iteration)
        if not moegambit_is_initialized():
            return state_dict
        return moegambit_pre_save_checkpoint(iteration, state_dict)

    def checkpoint_save_expert_sidecar(
        self,
        save_dir: str,
        iteration: int,
        model: Any,
        optimizer: Any,
        floating_point_operations: Any,
    ) -> None:
        from .moe_integration import moegambit_is_initialized

        if not moegambit_is_initialized():
            return
        if os.environ.get("ELASTIC_EXPERT_SIDECAR", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        from .elastic_client import elastic_save_expert_sidecar

        try:
            elastic_save_expert_sidecar(
                save_dir,
                iteration,
                model,
                optimizer,
                floating_point_operations,
            )
        except Exception:
            logger.exception(
                "Failed to save expert recovery sidecar; replacement will use "
                "the full checkpoint fallback"
            )

    def checkpoint_post_save(self, save_dir: str, iteration: int) -> None:
        from .moe_integration import (
            moegambit_is_initialized,
            moegambit_save_manifest,
        )

        from .integration import get_runtime

        runtime = get_runtime()
        if runtime is not None or moegambit_is_initialized():
            self._step_transaction.begin_checkpoint_commit(iteration)
            try:
                if moegambit_is_initialized():
                    moegambit_save_manifest(save_dir, iteration)
            except BaseException as exc:
                self.checkpoint_failed(iteration, exc)
                raise
            self._step_transaction.mark_checkpoint_committed(iteration)
            if runtime is not None:
                runtime.record_checkpoint(
                    os.path.join(save_dir, f"iter_{int(iteration):07d}"),
                    int(iteration),
                )

    def checkpoint_failed(self, iteration: int, exc: BaseException) -> None:
        if id(exc) == getattr(self, "_last_checkpoint_failure_id", None):
            return
        self._last_checkpoint_failure_id = id(exc)
        decision = self._step_transaction.decide_failure()
        self._last_failure_decision = decision
        from .integration import get_runtime

        runtime = get_runtime()
        if runtime is not None:
            runtime.fail_closed(
                exc,
                reason="megatron_checkpoint_failed",
                evidence={
                    "phase": decision.phase.value,
                    "resume_step": decision.resume_step,
                    "checkpoint_step": self._step_transaction.checkpoint_step,
                    "iteration": int(iteration),
                },
            )

    def checkpoint_post_load(self, state_dict: Any) -> None:
        from .moe_integration import (
            moegambit_is_initialized,
            moegambit_post_load_checkpoint,
        )

        if state_dict is not None and moegambit_is_initialized():
            moegambit_post_load_checkpoint(state_dict)

    # Distributed initialization ---------------------------------------

    def complete_replacement_dependencies(self, compile_dependencies: Any) -> bool:
        if not self.is_rebuild_mode():
            return False
        logger.warning(
            "[elastic] REBUILD MODE: skipping cold-start barriers after "
            "model-parallel setup"
        )
        compile_dependencies(skip_distributed_barriers=True)
        self.report_phase("cold_start_deps_ready")
        return True

    def prepare_process_group_init(
        self,
        *,
        args: Any,
        device_id: Any,
        store: Any,
        init_kwargs: dict[str, Any],
        torch_module: Any,
    ) -> dict[str, Any]:
        """Prepare replacement PG initialization and return final kwargs."""

        rebuild = self.is_rebuild_mode()
        standby_runtime = {"enabled": False}
        if rebuild:
            from .elastic_client import (
                elastic_configure_recovery_nccl_transport,
                elastic_create_rebuild_store,
                elastic_prearm_standby_cuda_runtime,
                elastic_refresh_prearmed_standby_assignment,
            )

            elastic_configure_recovery_nccl_transport()
            if device_id is not None:
                standby_runtime = elastic_prearm_standby_cuda_runtime(device_id)

        if rebuild:
            use_device_id = (
                os.environ.get(
                    "ELASTIC_REBUILD_INIT_PG_DEVICE_ID",
                    os.environ.get("ELASTIC_INIT_PG_DEVICE_ID", "0"),
                )
                == "1"
            )
        else:
            use_device_id = os.environ.get("ELASTIC_INIT_PG_DEVICE_ID", "0") == "1"
        if (
            device_id is not None
            and args.distributed_backend == "nccl"
            and use_device_id
        ):
            try:
                if (
                    "device_id"
                    in signature(torch_module.distributed.init_process_group).parameters
                ):
                    init_kwargs["device_id"] = device_id
            except (TypeError, ValueError):
                pass

        if rebuild:
            self.report_phase(
                "init_pg_start",
                master_addr=os.environ.get("MASTER_ADDR"),
                master_port=os.environ.get("MASTER_PORT"),
                world_size=args.world_size,
                pg_device_id_enabled=use_device_id,
                pg_device_id=str(device_id) if device_id is not None else None,
                store_connect_pending=store is None,
                standby_prearmed=os.environ.get("ELASTIC_PREARMED_STANDBY", "0")
                == "1",
                standby_runtime=standby_runtime,
            )
            if os.environ.get("ELASTIC_PREARMED_STANDBY", "0") == "1":
                activation = elastic_refresh_prearmed_standby_assignment()
                self.report_phase("standby_activated", **activation)
            if store is None:
                timeout_minutes = int(
                    os.environ.get(
                        "ELASTIC_REBUILD_TIMEOUT_MINUTES",
                        str(args.distributed_timeout_minutes),
                    )
                )
                store = elastic_create_rebuild_store(
                    os.environ["MASTER_ADDR"],
                    os.environ["MASTER_PORT"],
                    args.world_size,
                    args.rank,
                    timedelta(minutes=timeout_minutes),
                )
                init_kwargs["store"] = store
                self.report_phase(
                    "rebuild_store_ready",
                    master_addr=os.environ.get("MASTER_ADDR"),
                    master_port=os.environ.get("MASTER_PORT"),
                    store_type=type(store).__name__,
                )
        return init_kwargs

    def process_group_ready(self) -> None:
        if self.is_rebuild_mode():
            self.report_phase("pg_ready")

    def create_gloo_process_groups(self, configured: bool) -> bool:
        return False if self.is_rebuild_mode() else configured

    def _phase_timeout_seconds(
        self, args: Any = None, default_seconds: float = 300.0
    ) -> float:
        value = os.environ.get(
            "ELASTIC_PHASE_TIMEOUT_SECONDS",
            os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT"),
        )
        if value:
            return float(value)
        timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
        if timeout_minutes is None:
            timeout_minutes = os.environ.get("DISTRIBUTED_TIMEOUT_MINUTES")
        try:
            group_timeout = float(timeout_minutes) * 60.0
        except (TypeError, ValueError):
            group_timeout = 0.0
        margin = float(os.environ.get("ELASTIC_PHASE_TIMEOUT_MARGIN_SECONDS", "120"))
        return max(float(default_seconds), group_timeout + margin)

    def _wait_phase(self, phase: str, world_size: int, timeout: float) -> None:
        from .elastic_client import elastic_wait_for_recovery_phase_count

        if not elastic_wait_for_recovery_phase_count(phase, world_size, timeout):
            raise RuntimeError(
                f"[elastic] Not all {world_size} ranks reached {phase} "
                f"within {timeout}s"
            )

    @contextmanager
    def model_parallel_initialization(self, args: Any, mpu: Any):
        """Coordinate model-parallel group creation for a recovery epoch."""

        if not self.is_rebuild_mode():
            yield
            return

        self.report_phase("mpu_init_start")
        timeout = self._phase_timeout_seconds(args)
        self._wait_phase("mpu_init_start", args.world_size, timeout)
        old_trace = os.environ.get("ELASTIC_TRACE_MPU_GROUPS")
        if old_trace is None:
            os.environ["ELASTIC_TRACE_MPU_GROUPS"] = "1"
        try:
            if hasattr(mpu, "reset_elastic_mpu_group_ordinal"):
                mpu.reset_elastic_mpu_group_ordinal()
            yield
            if hasattr(mpu, "finalize_elastic_selective_group_rebuild"):
                mpu.finalize_elastic_selective_group_rebuild()
        finally:
            if old_trace is None:
                os.environ.pop("ELASTIC_TRACE_MPU_GROUPS", None)
        self.report_phase("mpu_init_done")
        self.report_phase("mpu_ready")
        self._wait_phase("mpu_ready", args.world_size, timeout)

    def reset_model_parallel_group_ordinal(self) -> None:
        from .parallel_state_hooks import parallel_state_hooks

        parallel_state_hooks.reset_ordinal()

    def prepare_selective_group_rebuild(
        self, replacement_rank: int, *, torch_module: Any
    ) -> dict[str, Any]:
        from .parallel_state_hooks import parallel_state_hooks

        return parallel_state_hooks.prepare_selective_rebuild(
            replacement_rank, torch_module=torch_module
        )

    def finalize_selective_group_rebuild(
        self,
        *,
        torch_module: Any,
        track_group: Callable[[Any], None],
    ) -> dict[str, Any]:
        from .parallel_state_hooks import parallel_state_hooks

        return parallel_state_hooks.finalize_selective_rebuild(
            torch_module=torch_module, track_group=track_group
        )

    def create_process_group(self, **kwargs: Any) -> Any:
        from .parallel_state_hooks import parallel_state_hooks

        return parallel_state_hooks.create_group(**kwargs)

    def token_dispatch(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from .token_dispatch_hooks import TokenDispatcherRecoveryHooks

        return getattr(TokenDispatcherRecoveryHooks, method)(*args, **kwargs)

    def trace_pipeline_p2p(
        self,
        communicator: Any,
        phase: str,
        *,
        tensor_send_next: Any,
        tensor_send_prev: Any,
        recv_prev: bool,
        recv_next: bool,
    ) -> None:
        """Report first post-rebuild P2P calls without changing ordering."""

        try:
            from .elastic_client import (
                elastic_is_post_rebuild_trace_active,
                elastic_report_recovery_phase,
            )

            if (
                os.environ.get("ELASTIC_RECOVERY_STATE")
                != "post_rebuild_stabilization_trace"
                or not elastic_is_post_rebuild_trace_active()
            ):
                return
            token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "rebuild")
            key = (token, phase)
            seen = getattr(communicator, "_elastic_p2p_trace", set())
            if key in seen:
                return
            seen.add(key)
            communicator._elastic_p2p_trace = seen
            elastic_report_recovery_phase(
                phase,
                p2p_send_prev=tensor_send_prev is not None,
                p2p_recv_prev=bool(recv_prev),
                p2p_send_next=tensor_send_next is not None,
                p2p_recv_next=bool(recv_next),
                p2p_prev_rank=communicator.prev_rank,
                p2p_next_rank=communicator.next_rank,
                p2p_group_name=str(
                    getattr(communicator.pp_group, "group_name", "unavailable")
                ),
            )
        except Exception:
            return

    def reset_rerun_state(self, machine: Any, current_iteration: int) -> dict[str, Any]:
        """Discard Megatron's transient rerun state after external recovery."""

        previous_state = machine.state
        previous_iteration = machine.current_iteration
        machine.state = type(machine.state).NOT_RUNNING_YET
        machine.current_iteration = int(current_iteration)
        machine.first_iteration_complete = machine.current_iteration > 0
        machine.rerun_requested = False
        machine.checkpoint_requested = False
        machine.restart_again_requested = False
        machine.continue_requested = False
        machine.validation_counts = defaultdict(int)
        machine.failed_validation_call = None
        machine.initial_result = None
        machine.suspicious_node = None
        machine.suspicious_device = None
        machine.saved_state = None
        machine.data_iterator_checkpoints = None
        machine.saved_results = {}
        machine.large_value_counts = {}
        machine.max_values = {}
        machine.error_injector.injected_error_type = None
        return {
            "mode": machine.mode.value,
            "previous_state": previous_state.value,
            "previous_iteration": previous_iteration,
            "state": machine.state.value,
            "current_iteration": machine.current_iteration,
            "first_iteration_complete": machine.first_iteration_complete,
        }


megatron_hooks = MegatronNativeHooks()
