"""Public training-loop runtime for coordinated in-process recovery."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from ..config import FallbackMode, RuntimeConfig
from ..control.coordinator import RecoveryAssignment, RecoveryCoordinator, RecoveryRequest
from ..distributed.c10d_backend import FailureClassification
from ..errors import MoEGambitError, RecoverableDistributedError
from ..observability.events import RecoveryOutcome, RecoveryRecord
from ..observability.metrics import RecoveryMetrics
from .executor import RecoveryExecutionResult, RecoveryExecutor
from .fallback import FallbackController, FallbackRequest
from .lifecycle import EpochState, LifecyclePhase, RecoveryEpochTracker

__all__ = ["RecoveryRuntime", "initialize"]

logger = logging.getLogger(__name__)


class RecoveryRuntime:
    def __init__(
        self,
        adapter: Any = None,
        config: Optional[RuntimeConfig] = None,
        *,
        coordinator: Optional[RecoveryCoordinator] = None,
        executor: Optional[RecoveryExecutor] = None,
        fallback_controller: Optional[FallbackController] = None,
        metrics: Optional[RecoveryMetrics] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or RuntimeConfig()
        self.coordinator = coordinator
        self.fallback_controller = fallback_controller
        self.metrics = metrics or RecoveryMetrics()
        self.epochs = RecoveryEpochTracker()
        self.executor = executor
        self._resume_step = 0
        self._enabled = bool(self.config.enabled)
        self._last_assignment: Optional[RecoveryAssignment] = None
        self._last_execution: Optional[RecoveryExecutionResult] = None
        self._last_record: Optional[RecoveryRecord] = None
        self._last_record_observed = False
        if self._enabled and adapter is None:
            raise MoEGambitError(
                "elastic recovery is enabled but no framework adapter was provided"
            )
        if self._enabled and self.executor is None and adapter is not None:
            self.executor = RecoveryExecutor(
                adapter,
                self.epochs,
                job_id=self.config.job_id,
                attempt_id=self.config.attempt_id,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def resume_step(self) -> int:
        return self._resume_step

    @property
    def recovery_epoch(self) -> int:
        return self.epochs.epoch

    def iteration_boundary(self, step: int) -> int:
        if not self._enabled:
            return int(step)
        self.trace(LifecyclePhase.ITERATION_BOUNDARY, step=step)
        driver = getattr(self.adapter, "recovery_driver", None)
        if driver is None:
            return int(step)
        try:
            resume_step = driver.poll(int(step))
        except Exception as exc:
            logger.exception("framework recovery driver poll failed")
            self._request_fallback("framework_poll_recovery_failed", exc)
            return int(step)
        if resume_step is None:
            return int(step)
        self._adopt_driver_recovery(int(resume_step), at_step=int(step))
        return int(resume_step)

    def before_optimizer_step(self, step: int) -> None:
        if not self._enabled:
            return
        self.trace(LifecyclePhase.BEFORE_OPTIMIZER_STEP, step=step)
        self.adapter.optimizer.before_step(int(step))

    def after_optimizer_step(self, step: int, committed: bool = True) -> None:
        if not self._enabled:
            return
        self.trace(
            LifecyclePhase.AFTER_OPTIMIZER_STEP,
            step=step,
            committed=committed,
        )
        self.adapter.optimizer.after_step(int(step), bool(committed))

    def commit_iteration(self, step: int) -> bool:
        if not self._enabled:
            return False
        self.trace(LifecyclePhase.COMMIT_ITERATION, step=step)
        driver = getattr(self.adapter, "recovery_driver", None)
        driver_committed = bool(driver.commit(int(step))) if driver is not None else False
        if not self.epochs.can_commit(int(step)):
            self.epochs.commit(int(step))
            return driver_committed
        if self.coordinator is not None and self._last_assignment is not None:
            try:
                self.coordinator.committed(self._last_assignment, int(step))
            except Exception as exc:
                self.epochs.fail(f"control-plane commit rejected: {exc}")
                self._request_fallback("recovery_commit_rejected", exc)
                return False
        promoted = self.epochs.commit(int(step))
        if promoted and self._last_record is not None:
            self._last_record.result = RecoveryOutcome.COMMITTED
            self._last_record.validation["provisional"] = False
            self._last_record.validation["committed_step"] = int(step)
            if not self._last_record_observed:
                self.metrics.observe_record(self._last_record)
                self._last_record_observed = True
        return promoted or driver_committed

    def on_distributed_error(self, exc: BaseException) -> bool:
        if not self._enabled:
            return False
        self.trace(LifecyclePhase.DISTRIBUTED_ERROR, error=type(exc).__name__)
        classification = self._classify_error(exc)
        if not classification.recoverable:
            return False
        if self.epochs.state is EpochState.PROVISIONAL:
            self.epochs.fail("failure inside provisional recovery epoch")
            self._request_fallback(
                "failure_inside_provisional_recovery_epoch", exc
            )
            return False

        driver = getattr(self.adapter, "recovery_driver", None)
        if driver is not None:
            try:
                resume_step = driver.recover(exc)
            except Exception as recovery_exc:
                logger.exception("framework recovery driver failed")
                self._request_fallback(
                    "framework_recovery_driver_failed", recovery_exc
                )
                return False
            if resume_step is None:
                self._request_fallback("framework_recovery_not_handled", exc)
                return False
            self._adopt_driver_recovery(
                int(resume_step),
                at_step=self._current_step(),
                classification=classification,
            )
            return True

        if self.coordinator is None or self.executor is None:
            self._request_fallback("recovery_coordinator_unavailable", exc)
            return False

        assignment: Optional[RecoveryAssignment] = None
        try:
            at_step = self._current_step()
            epoch = self.epochs.begin_recovery(
                tuple(classification.failed_ranks), at_step
            )
            topology = self.adapter.topology.inspect()
            request = RecoveryRequest(
                classification=classification,
                at_step=at_step,
                recovery_epoch=epoch,
                topology_generation=topology.generation + 1,
                group_manifest_hash=(
                    topology.manifest_hash or topology.compute_manifest_hash()
                ),
            )
            assignment = self.coordinator.prepare(request, self.adapter)
            result = self.executor.execute(
                assignment,
                failure_class=classification.failure_class,
                at_step=at_step,
            )
        except Exception as recovery_exc:
            self.epochs.fail(str(recovery_exc))
            try:
                self.coordinator.failed(assignment, recovery_exc)
            except Exception:
                logger.exception("coordinator failed while recording recovery failure")
            self._request_fallback(
                "in_process_recovery_failed",
                recovery_exc,
                evidence={"original_error_type": type(exc).__name__},
            )
            return False

        self._last_assignment = assignment
        self._last_execution = result
        self._last_record = result.record
        self._last_record_observed = False
        self._resume_step = result.resume_step
        return True

    def execute_assignment(
        self,
        assignment: RecoveryAssignment,
        *,
        failure_class: str = "replacement_start",
        at_step: Optional[int] = None,
    ) -> int:
        if not self._enabled:
            raise MoEGambitError(
                "cannot execute an assignment while MoEGambit is disabled"
            )
        if self.executor is None:
            raise MoEGambitError("no recovery executor is configured")
        plan = assignment.plan
        current_step = plan.resume_step if at_step is None else int(at_step)
        self.epochs.adopt_recovery(
            plan.recovery_epoch,
            tuple(plan.failed_ranks),
            current_step,
        )
        try:
            result = self.executor.execute(
                assignment,
                failure_class=failure_class,
                at_step=current_step,
            )
        except Exception as exc:
            self.epochs.fail(str(exc))
            if self.coordinator is not None:
                try:
                    self.coordinator.failed(assignment, exc)
                except Exception:
                    logger.exception(
                        "coordinator failed while recording replacement failure"
                    )
            self._request_fallback("replacement_assignment_failed", exc)
            raise
        self._last_assignment = assignment
        self._last_execution = result
        self._last_record = result.record
        self._last_record_observed = False
        self._resume_step = result.resume_step
        return result.resume_step

    def _request_fallback(
        self,
        reason: str,
        exc: BaseException,
        *,
        evidence: Optional[Mapping[str, object]] = None,
    ) -> bool:
        if self.config.fallback != FallbackMode.CHECKPOINT_RELAUNCH:
            self._record_terminal_failure(reason, exc, fallback=False)
            return False
        if self.fallback_controller is None:
            self._record_terminal_failure(
                reason + ":fallback_controller_unavailable",
                exc,
                fallback=False,
            )
            return False
        request = FallbackRequest(
            recovery_epoch=self.epochs.epoch,
            at_step=self._current_step(),
            reason=reason,
            error_type=type(exc).__name__,
            evidence=dict(evidence or {}),
        )
        try:
            accepted = bool(
                self.fallback_controller.request_checkpoint_relaunch(request)
            )
        except Exception:
            logger.exception("checkpoint relaunch request failed")
            accepted = False
        self._record_terminal_failure(reason, exc, fallback=accepted)
        return accepted

    def _record_terminal_failure(
        self,
        reason: str,
        exc: BaseException,
        *,
        fallback: bool,
    ) -> None:
        failed_ranks = ()
        for event in reversed(self.epochs.history):
            if "failed_ranks" in event:
                failed_ranks = tuple(int(rank) for rank in event["failed_ranks"])
                break
        record = RecoveryRecord(
            job_id=self.config.job_id,
            attempt_id=self.config.attempt_id,
            recovery_epoch=self.epochs.epoch,
            failed_ranks=failed_ranks,
            failure_class="recovery_failure",
            adapter=getattr(self.adapter, "name", "unknown"),
            adapter_version=getattr(self.adapter, "version", "unknown"),
            resume_step=self._current_step(),
            decision="checkpoint_relaunch" if fallback else "abort",
            validation={
                "fallback_reason": reason,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            },
            result=(
                RecoveryOutcome.FALLBACK if fallback else RecoveryOutcome.ABORTED
            ),
        )
        self._last_record = record
        self.metrics.observe_record(record)
        self._last_record_observed = True

    def _classify_error(self, exc: BaseException) -> FailureClassification:
        driver = getattr(self.adapter, "recovery_driver", None)
        if driver is not None:
            try:
                if driver.classify_error(exc):
                    return FailureClassification(
                        True,
                        failure_class="fail_stop",
                        failed_ranks=tuple(getattr(exc, "failed_ranks", ())),
                        evidence={"exception_type": type(exc).__name__},
                    )
            except Exception:
                logger.exception("recovery-driver classifier raised")
                return FailureClassification(False)
        if isinstance(exc, RecoverableDistributedError):
            return FailureClassification(
                True,
                failure_class=exc.failure_class,
                failed_ranks=exc.failed_ranks,
                evidence=dict(exc.evidence),
            )
        backend = getattr(self.adapter, "backend", None)
        classify = getattr(backend, "classify_error", None)
        if classify is None:
            return FailureClassification(False)
        try:
            result = classify(exc)
            if isinstance(result, FailureClassification):
                return result
            return FailureClassification(bool(result), failure_class="fail_stop")
        except Exception:
            logger.exception("error classifier raised; treating as unrecoverable")
            return FailureClassification(False)

    def _current_step(self) -> int:
        current_progress = getattr(self.adapter.training, "current_progress", None)
        if callable(current_progress):
            try:
                return int(current_progress().step)
            except Exception:
                logger.exception("adapter current_progress failed")
        return max(0, int(self.epochs.last_committed_step))

    def _adopt_driver_recovery(
        self,
        resume_step: int,
        *,
        at_step: int,
        classification: Optional[FailureClassification] = None,
    ) -> None:
        classified = classification or FailureClassification(
            True,
            failure_class="fail_stop",
            failed_ranks=(-1,),
        )
        if self.epochs.state not in (EpochState.RECOVERING, EpochState.PROVISIONAL):
            self.epochs.begin_recovery(classified.failed_ranks, at_step)
        if self.epochs.state is EpochState.RECOVERING:
            self.epochs.mark_provisional(resume_step)
        self._resume_step = int(resume_step)
        self._last_record = RecoveryRecord(
            job_id=self.config.job_id,
            attempt_id=self.config.attempt_id,
            recovery_epoch=self.epochs.epoch,
            failed_ranks=classified.failed_ranks,
            failure_class=classified.failure_class,
            adapter=getattr(self.adapter, "name", "unknown"),
            adapter_version=getattr(self.adapter, "version", "unknown"),
            resume_step=int(resume_step),
            decision="framework_bridge",
            validation={"provisional": True},
        )
        self._last_record_observed = False

    def trace(self, phase: Any, **fields: object) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            name = phase.value if isinstance(phase, LifecyclePhase) else str(phase)
            logger.debug("moegambit phase=%s %s", name, fields)

    def describe(self) -> Mapping[str, object]:
        info = {"enabled": self._enabled, "framework": self.config.framework}
        info.update(self.epochs.snapshot())
        if self.adapter is not None and hasattr(self.adapter, "describe"):
            info.update(self.adapter.describe())
        if self._last_record is not None:
            info["last_recovery"] = self._last_record.to_dict()
        info["metrics"] = self.metrics.snapshot()
        return info


def initialize(
    adapter: Any = None,
    config: Optional[RuntimeConfig] = None,
    *,
    coordinator: Optional[RecoveryCoordinator] = None,
    executor: Optional[RecoveryExecutor] = None,
    fallback_controller: Optional[FallbackController] = None,
    metrics: Optional[RecoveryMetrics] = None,
    **overrides: object,
) -> RecoveryRuntime:
    resolved = config or RuntimeConfig.from_env()
    if overrides:
        resolved = resolved.merged_with(**overrides)
    if resolved.enabled:
        problems = resolved.validate()
        if problems:
            raise MoEGambitError("invalid runtime configuration: " + "; ".join(problems))
    control_client = None
    if (
        resolved.enabled
        and coordinator is None
        and adapter is not None
        and getattr(adapter, "recovery_driver", None) is None
    ):
        from .client import ControlClient, ControlClientConfig, WatcherRecoveryCoordinator

        def _rank(name: str) -> int:
            try:
                return int(os.environ.get(name, "-1"))
            except ValueError:
                return -1

        control_client = ControlClient(
            ControlClientConfig(
                host=resolved.watcher.host,
                port=resolved.watcher.port,
                job_id=resolved.job_id,
                attempt_id=resolved.attempt_id,
                sender={
                    "role": "worker",
                    "node_rank": _rank("NODE_RANK"),
                    "global_rank": _rank("RANK"),
                    "local_rank": _rank("LOCAL_RANK"),
                },
                job_token=resolved.security.job_token,
                connect_timeout_s=min(10.0, resolved.recovery_timeout_s),
                request_timeout_s=resolved.recovery_timeout_s,
                max_message_bytes=resolved.security.max_message_bytes,
            )
        )
        coordinator = WatcherRecoveryCoordinator(control_client)
    if (
        fallback_controller is None
        and resolved.fallback == FallbackMode.CHECKPOINT_RELAUNCH
    ):
        if control_client is None:
            control_client = getattr(coordinator, "client", None)
        if control_client is not None:
            from .fallback import ControlPlaneFallbackController

            fallback_controller = ControlPlaneFallbackController(control_client)
    runtime = RecoveryRuntime(
        adapter=adapter,
        config=resolved,
        coordinator=coordinator,
        executor=executor,
        fallback_controller=fallback_controller,
        metrics=metrics,
    )
    replacement_mode = os.environ.get("MOEGAMBIT_REPLACEMENT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    fetch_assignment = getattr(coordinator, "fetch_assignment", None)
    if runtime.enabled and replacement_mode and callable(fetch_assignment):
        raw_epoch = os.environ.get("MOEGAMBIT_RECOVERY_EPOCH", "")
        if not raw_epoch:
            raise MoEGambitError(
                "replacement mode requires MOEGAMBIT_RECOVERY_EPOCH"
            )
        try:
            replacement_epoch = int(raw_epoch)
        except ValueError as exc:
            raise MoEGambitError(
                f"invalid replacement recovery epoch: {raw_epoch!r}"
            ) from exc
        assignment = fetch_assignment(
            replacement_epoch,
            plan_digest=os.environ.get("MOEGAMBIT_RECOVERY_PLAN_DIGEST", ""),
        )
        runtime.execute_assignment(assignment, at_step=assignment.plan.resume_step)
    return runtime
