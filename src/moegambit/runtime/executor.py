"""Execute one frozen RecoveryPlan through the adapter protocols."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.base import FrameworkAdapter, PauseRequest
from ..control.coordinator import RecoveryAssignment
from ..control.protocol import PROTOCOL_VERSION
from ..errors import ContractViolation, ProtocolVersionMismatch
from ..observability.events import RecoveryOutcome, RecoveryRecord
from .lifecycle import RecoveryEpochTracker
from .recovery_plan import RecoveryMode

__all__ = ["RecoveryExecutionResult", "RecoveryExecutor"]


@dataclass(frozen=True)
class RecoveryExecutionResult:
    resume_step: int
    record: RecoveryRecord
    assignment: RecoveryAssignment


class RecoveryExecutor:
    """Synchronous, fail-closed recovery executor."""

    def __init__(
        self,
        adapter: FrameworkAdapter,
        epochs: RecoveryEpochTracker,
        *,
        job_id: str = "",
        attempt_id: str = "",
    ) -> None:
        self.adapter = adapter
        self.epochs = epochs
        self.job_id = job_id
        self.attempt_id = attempt_id

    def execute(
        self,
        assignment: RecoveryAssignment,
        *,
        failure_class: str,
        at_step: int,
    ) -> RecoveryExecutionResult:
        plan = assignment.plan
        if plan.protocol_version != PROTOCOL_VERSION:
            raise ProtocolVersionMismatch(
                f"local={PROTOCOL_VERSION}, plan={plan.protocol_version}"
            )
        if plan.recovery_epoch != self.epochs.epoch:
            raise ContractViolation(
                "executor epoch does not match the frozen recovery plan"
            )
        if assignment.plan_digest != plan.digest():
            raise ContractViolation("assignment digest changed before execution")
        if plan.mode is RecoveryMode.ABORT:
            raise ContractViolation("an abort plan cannot enter RecoveryExecutor")
        capabilities = self.adapter.capabilities
        if not capabilities.static_world_replacement:
            raise ContractViolation(
                "adapter cannot retain the failed logical rank"
            )
        if not (
            capabilities.selective_group_rebuild
            or capabilities.full_group_rebuild
        ):
            raise ContractViolation("adapter cannot rebuild an affected group")
        if plan.mode in (RecoveryMode.PEER, RecoveryMode.HYBRID):
            if not capabilities.peer_parameter_restore:
                raise ContractViolation("adapter cannot restore state from a peer")
        if plan.mode is RecoveryMode.HYBRID and not capabilities.moe_state_classification:
            raise ContractViolation(
                "adapter cannot classify state required by a hybrid plan"
            )
        planned_sources = set(plan.state_sources)
        resolved_sources = set(assignment.sources.by_identity)
        if not planned_sources:
            raise ContractViolation("recovery plan contains no state sources")
        if planned_sources != resolved_sources:
            raise ContractViolation(
                "resolved state-source identities differ from the frozen plan"
            )

        record = RecoveryRecord(
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            recovery_epoch=plan.recovery_epoch,
            failed_ranks=plan.failed_ranks,
            failure_class=failure_class,
            adapter=self.adapter.name,
            adapter_version=self.adapter.version,
            topology_manifest=plan.group_manifest_hash,
            resume_step=plan.resume_step,
            decision=plan.mode.value,
            policy_evidence=dict(plan.policy_evidence),
            state_sources=dict(plan.state_sources),
            result=RecoveryOutcome.PROVISIONAL,
        )

        with record.measure_phase("quiesce"):
            proof = self.adapter.training.quiesce(
                PauseRequest(
                    recovery_epoch=plan.recovery_epoch,
                    reason=failure_class,
                    deadline_s=plan.timeout_for("quiesce", 300.0),
                )
            )
            if not proof.is_safe:
                raise ContractViolation(
                    "adapter did not prove that in-flight collectives were drained"
                )
            record.validation["quiescence"] = {
                "rank": proof.rank,
                "collectives_drained": proof.collectives_drained,
                "process_groups_destroyed": proof.process_groups_destroyed,
                "at_step": int(at_step),
            }

        rebuild_handle = self.adapter.topology.prepare_rebuild(plan)
        if rebuild_handle.recovery_epoch != plan.recovery_epoch:
            raise ContractViolation("rebuild handle belongs to another epoch")
        self.adapter.state.load_replacement_base(plan)

        with record.measure_phase("group_rebuild"):
            topology = self.adapter.topology.rebuild(plan, assignment.store)
            if topology.generation != plan.topology_generation:
                raise ContractViolation(
                    "rebuilt topology generation differs from frozen plan: "
                    f"{topology.generation} != {plan.topology_generation}"
                )
            self.adapter.topology.rebind(topology)
            self.adapter.optimizer.rebind(topology)
            observed_manifest = (
                topology.manifest_hash or topology.compute_manifest_hash()
            )
            if observed_manifest != plan.group_manifest_hash:
                raise ContractViolation(
                    "rebuilt topology manifest differs from frozen plan: "
                    f"{observed_manifest} != {plan.group_manifest_hash}"
                )
            topology_report = self.adapter.topology.validate(topology)
            topology_report.raise_for_status()

        with record.measure_phase("state_restore"):
            self.adapter.state.restore(plan, assignment.sources)
            state_report = self.adapter.state.validate_state(plan)
            state_report.raise_for_status()

        self.adapter.training.reset_transients(plan)
        self.adapter.training.apply_resume(plan)
        with record.measure_phase("first_forward"):
            warmup = self.adapter.training.warmup_and_validate(plan)
            warmup.raise_for_status()
        self.epochs.mark_provisional(plan.resume_step)
        record.validation.update(
            {
                "topology": dict(topology_report.details),
                "state": dict(state_report.details),
                "warmup": dict(warmup.details),
                "plan_digest": assignment.plan_digest,
                "provisional": True,
            }
        )
        return RecoveryExecutionResult(plan.resume_step, record, assignment)
