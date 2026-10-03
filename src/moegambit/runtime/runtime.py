"""Public training-loop runtime for coordinated in-process recovery."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from ..config import FallbackMode, RuntimeConfig
from ..adapters.base import CheckpointResumeAdapter
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
        event_sink: Any = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or RuntimeConfig()
        self.coordinator = coordinator
        self.fallback_controller = fallback_controller
        self.metrics = metrics or RecoveryMetrics()
        self.event_sink = event_sink
        self.epochs = RecoveryEpochTracker()
        self.executor = executor
        self._resume_step = 0
        self._enabled = bool(self.config.enabled)
        self._last_assignment: Optional[RecoveryAssignment] = None
        self._last_execution: Optional[RecoveryExecutionResult] = None
        self._last_record: Optional[RecoveryRecord] = None
        self._last_record_observed = False
        self._latest_checkpoint: Optional[tuple[str, int]] = None
        self._quality_offloader = None
        self._quality_feature_provider = None
        self._quality_history_provider = None
        self._quality_committed_optimizer_step = None
        configured_locator = os.environ.get("MOEGAMBIT_CHECKPOINT_LOCATOR", "")
        configured_step = os.environ.get("MOEGAMBIT_CHECKPOINT_STEP", "")
        if configured_locator and configured_step:
            try:
                self.record_checkpoint(configured_locator, int(configured_step))
            except (TypeError, ValueError) as exc:
                raise MoEGambitError(
                    "invalid MOEGAMBIT_CHECKPOINT_LOCATOR/STEP configuration"
                ) from exc
            if os.environ.get("MOEGAMBIT_CHECKPOINT_RELAUNCH", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                self._resume_step = int(configured_step)
                self.epochs.last_committed_step = int(configured_step)
                if adapter is not None and isinstance(
                    adapter.training, CheckpointResumeAdapter
                ):
                    adapter.training.apply_checkpoint_resume(int(configured_step))
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

    def record_checkpoint(self, locator: str, step: int) -> None:
        """Publish the latest fully committed checkpoint for fallback use.

        Training integrations call this only after the checkpoint manifest and
        all shards are durable.  Merely reaching a training step is not proof
        that a restartable checkpoint exists.
        """

        if not isinstance(locator, str) or not locator.strip():
            raise ValueError("checkpoint locator must be non-empty")
        step = int(step)
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")
        if self._latest_checkpoint is not None and step < self._latest_checkpoint[1]:
            raise ValueError("latest checkpoint step cannot move backwards")
        self._latest_checkpoint = (locator.strip(), step)

    def iteration_boundary(self, step: int) -> int:
        if not self._enabled:
            return int(step)
        self.trace(LifecyclePhase.ITERATION_BOUNDARY, step=step)
        boundary = getattr(self.adapter.training, "iteration_boundary", None)
        if callable(boundary):
            boundary(int(step))
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
        self._quality_committed_optimizer_step = None
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
        self._quality_committed_optimizer_step = int(step) if committed else None

    def configure_quality_offload(self, offloader, feature_provider, *, history_provider) -> None:
        """Attach precomputed telemetry to the normal committed-step hook.

        feature_provider(step, checkpoint_step) returns compact features. It
        must not compute a new sensitivity estimate from a failed rank. The
        caller owns offloader.close() and periodically records durable checkpoints.
        """
        if not callable(feature_provider) or not callable(history_provider):
            raise TypeError("quality feature/history providers must be callable")
        self._quality_offloader = offloader
        self._quality_feature_provider = feature_provider
        self._quality_history_provider = history_provider
        configure = getattr(self.coordinator, "configure_quality_identity", None)
        if callable(configure):
            configure(offloader.identity)

    def _offload_quality_features(self, step: int) -> None:
        if (self._quality_offloader is None or self._latest_checkpoint is None or
                self._quality_committed_optimizer_step != int(step) or
                self.epochs.state not in (EpochState.CLEAN, EpochState.COMMITTED)):
            return
        self._quality_committed_optimizer_step = None
        try:
            topology = self.adapter.topology.inspect()
            checkpoint_step = self._latest_checkpoint[1]
            features = self._quality_feature_provider(int(step), checkpoint_step)
            pending = self._quality_offloader.submit(
                features, committed_step=int(step), checkpoint_step=checkpoint_step,
                topology_generation=topology.generation,
                group_manifest_hash=topology.manifest_hash or topology.compute_manifest_hash(),
                world_size=topology.world_size, recovery_epoch=self.recovery_epoch,
                run_history=self._quality_history_provider(),
            )
            if pending is None:
                logger.warning("quality CPU upload queue full at committed step %s", step)
        except Exception:
            # Training continues, but missing telemetry cannot authorize Hybrid.
            logger.exception("quality CPU capture failed at committed step %s", step)

    def commit_iteration(self, step: int) -> bool:
        if not self._enabled:
            return False
        self.trace(LifecyclePhase.COMMIT_ITERATION, step=step)
        driver = getattr(self.adapter, "recovery_driver", None)
        driver_committed = bool(driver.commit(int(step))) if driver is not None else False
        if not self.epochs.can_commit(int(step)):
            self.epochs.commit(int(step))
            self._offload_quality_features(int(step))
            return driver_committed
        if self.coordinator is not None and self._last_assignment is not None:
            try:
                self.coordinator.committed(self._last_assignment, int(step))
            except Exception as exc:
                self.epochs.fail(f"control-plane commit rejected: {exc}")
                self._request_fallback("recovery_commit_rejected", exc)
                return False
        promoted = self.epochs.commit(int(step))
        self._offload_quality_features(int(step))
        if promoted and self._last_record is not None:
            self._last_record.result = RecoveryOutcome.COMMITTED
            self._last_record.validation["provisional"] = False
            self._last_record.validation["committed_step"] = int(step)
            if not self._last_record_observed:
                self.metrics.observe_record(self._last_record)
                self._last_record_observed = True
                self._emit_record(self._last_record)
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

    def fail_closed(
        self,
        exc: BaseException,
        *,
        reason: str,
        evidence: Optional[Mapping[str, object]] = None,
    ) -> bool:
        """Reject in-process continuation and request checkpoint relaunch.

        Adapters call this when their phase tracker proves that framework state
        may already be partially mutated.  Such a failure must not enter the
        ordinary iteration replay path, even when the underlying exception is
        otherwise classified as recoverable.
        """

        if not self._enabled:
            return False
        self.epochs.fail(reason)
        return self._request_fallback(
            reason,
            exc,
            evidence=evidence,
        )

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
            checkpoint_locator=(
                None
                if self._latest_checkpoint is None
                else self._latest_checkpoint[0]
            ),
            checkpoint_step=(
                -1
                if self._latest_checkpoint is None
                else self._latest_checkpoint[1]
            ),
            command_digest=os.environ.get("MOEGAMBIT_COMMAND_DIGEST") or None,
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
        self._emit_record(record)

    def _emit_record(self, record: RecoveryRecord) -> None:
        if self.event_sink is not None:
            try:
                self.event_sink(record)
            except Exception:
                # Evidence I/O cannot undo a committed training iteration.
                logger.exception("failed to publish recovery evidence for epoch %s", record.recovery_epoch)

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
        if self._quality_offloader is not None:
            info["quality_cpu_offload"] = self._quality_offloader.stats()
        if self._latest_checkpoint is not None:
            info["latest_checkpoint"] = {
                "locator": self._latest_checkpoint[0],
                "step": self._latest_checkpoint[1],
            }
        return info


def initialize(
    adapter: Any = None,
    config: Optional[RuntimeConfig] = None,
    *,
    coordinator: Optional[RecoveryCoordinator] = None,
    executor: Optional[RecoveryExecutor] = None,
    fallback_controller: Optional[FallbackController] = None,
    state_source_resolver: Any = None,
    metrics: Optional[RecoveryMetrics] = None,
    event_sink: Any = None,
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
        coordinator = WatcherRecoveryCoordinator(
            control_client,
            source_resolver=state_source_resolver,
        )
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
        event_sink=event_sink,
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
