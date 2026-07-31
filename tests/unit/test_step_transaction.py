from __future__ import annotations

import pytest

from moegambit.core.step_transaction import (
    FailureAction,
    StepPhase,
    StepTransaction,
)
from moegambit.errors import ContractViolation
from moegambit.runtime.checkpoint_commit import (
    CHECKPOINT_COMMIT_NAME,
    load_checkpoint_commit,
    publish_checkpoint_commit,
)


@pytest.mark.parametrize(
    "phase",
    [
        StepPhase.FORWARD,
        StepPhase.BACKWARD,
        StepPhase.FORWARD_BACKWARD,
        StepPhase.OPTIMIZER_BEFORE,
    ],
)
def test_failure_before_optimizer_mutation_replays_step(phase):
    transaction = StepTransaction(7)
    transaction.begin_step(7)
    transaction.enter(phase)

    decision = transaction.decide_failure()

    assert decision.action is FailureAction.REPLAY_STEP
    assert decision.resume_step == 7
    assert decision.permits_bookkeeping_rollback
    assert not decision.optimizer_dirty


def test_failure_during_optimizer_requires_state_restore():
    transaction = StepTransaction(7)
    transaction.mark_optimizer_snapshot(7)
    transaction.begin_step(7)
    transaction.enter(StepPhase.OPTIMIZER_BEFORE)
    transaction.enter(StepPhase.OPTIMIZER_DURING)

    decision = transaction.decide_failure()

    assert decision.action is FailureAction.RESTORE_AND_REPLAY
    assert decision.resume_step == 7
    assert decision.optimizer_dirty
    assert not decision.permits_bookkeeping_rollback


def test_failure_after_optimizer_before_replica_forces_checkpoint_restart():
    transaction = StepTransaction(7)
    transaction.mark_optimizer_snapshot(7)
    transaction.begin_step(7)
    transaction.enter(StepPhase.OPTIMIZER_DURING)
    transaction.mark_optimizer_applied(8)

    decision = transaction.decide_failure()

    assert decision.action is FailureAction.CHECKPOINT_RESTART
    assert decision.resume_step == -1
    assert not decision.permits_bookkeeping_rollback


def test_optimizer_version_becomes_safe_only_after_replica_commit():
    transaction = StepTransaction(7)
    transaction.mark_optimizer_snapshot(7)
    transaction.begin_step(7)
    transaction.enter(StepPhase.OPTIMIZER_DURING)
    transaction.mark_optimizer_applied(8)

    with pytest.raises(ContractViolation, match="snapshot"):
        transaction.mark_step_safe(8)

    transaction.mark_optimizer_snapshot(8)
    transaction.mark_step_safe(8)
    assert transaction.decide_failure().action is FailureAction.RESUME_COMMITTED


def test_checkpoint_before_during_and_after_commit_resume_committed_step():
    transaction = StepTransaction(8)
    transaction.mark_optimizer_snapshot(8)
    transaction.begin_step(8)
    transaction.mark_step_safe(8)

    transaction.begin_checkpoint(8)
    assert transaction.decide_failure().action is FailureAction.RESUME_COMMITTED

    transaction.begin_checkpoint_commit(8)
    assert transaction.decide_failure().action is FailureAction.RESUME_COMMITTED

    transaction.mark_checkpoint_committed(8)
    decision = transaction.decide_failure()
    assert decision.action is FailureAction.RESUME_COMMITTED
    assert transaction.checkpoint_step == 8


def test_checkpoint_commit_record_is_atomic_and_versioned(tmp_path):
    target = publish_checkpoint_commit(
        tmp_path,
        framework="megatron",
        step=12,
        metadata={"manifest": "moegambit_manifest.json"},
    )

    assert target == tmp_path / CHECKPOINT_COMMIT_NAME
    assert not list(tmp_path.glob("*.tmp"))
    record = load_checkpoint_commit(
        tmp_path,
        framework="megatron",
        step=12,
    )
    assert record["metadata"]["manifest"] == "moegambit_manifest.json"


def test_checkpoint_commit_record_rejects_wrong_framework_or_step(tmp_path):
    publish_checkpoint_commit(
        tmp_path,
        framework="deepspeed",
        step=4,
    )

    with pytest.raises(RuntimeError, match="framework mismatch"):
        load_checkpoint_commit(tmp_path, framework="megatron")
    with pytest.raises(RuntimeError, match="step mismatch"):
        load_checkpoint_commit(tmp_path, step=5)
