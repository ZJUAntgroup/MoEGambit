"""Risk gate negative cases and coordinator context integrity."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import pytest

from moegambit.audit.io import digest, write_json
from moegambit.capabilities import AdapterCapabilities
from moegambit.policy import (QualityRiskPolicy, FileRiskEvidenceProvider, RecoveryFacts,
                             risk_context_digest, ExposureEvent)
from moegambit.runtime.recovery_plan import RecoveryMode
from moegambit.state.catalog import StateSource, StateSourceKind
from moegambit.state.version import StateVersion
from moegambit.control.service import _stable_request_digest


def facts():
    def source(kind, step, placement, expert=False):
        return StateSource(kind, StateVersion(step, step, 1), f"{kind.value}://{step}",
                           metadata={"placement": placement, "tags": ["expert"] if expert else []})
    return RecoveryFacts(
        failed_ranks=(1,), resume_step=1000, latest_checkpoint_step=200,
        available_state_sources={
            "dense": (source(StateSourceKind.PEER, 1000, "replicated"),
                      source(StateSourceKind.CHECKPOINT, 200, "replicated")),
            "expert": (source(StateSourceKind.CHECKPOINT, 200, "unique", True),)},
        exposure_history=(), capabilities=AdapterCapabilities(
            peer_parameter_restore=True, moe_state_classification=True),
        quality_context={"model_id": "moe-a", "policy_version": "v1", "telemetry_version": "v1",
                         "features": {"exposed_mass": .125}, "run_history": []},
        control_scope={"job_id": "job", "attempt_id": "a1", "recovery_epoch": 1,
                       "topology_generation": 1, "group_manifest_hash": "topology"})


def calibration():
    return {"schema_version": 1, "expires_at": "2030-01-01T00:00:00Z",
            "bound_scope": "whole_run_conditional", "qualification": "operator_reviewed",
            "independent_audit": True, "method": "external qualified predictor",
            "assumptions": "declared support and conditioning protocol",
            "audit_report_sha256": "a" * 64,
            "outcome_definition": "paired-whole-run-final-and-peak-v1",
            "eta_final": .005, "eta_peak": .01, "alpha_run": .05,
            "support": {"model_id": ["moe-a"], "policy_version": ["v1"], "telemetry_version": ["v1"]}}


def evidence(f=None):
    return {"schema_version": 1, "calibration": calibration(),
            "calibration_digest": digest(calibration()), "context_digest": risk_context_digest(f or facts()),
            "expires_at": "2030-01-01T00:00:00Z", "risk_upper": .04,
            "support_checked": True, "in_support": True, "run_history_checked": True}


class Provider:
    def __init__(self, value): self.value = value
    def evaluate(self, facts): return deepcopy(self.value)


def policy(value, mode="enforce"):
    return QualityRiskPolicy(Provider(value), mode=mode,
                             clock=lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))


def test_admit_risk_with_no_age_gate_and_planner_sources():
    result = policy(evidence()).decide(facts())
    assert result.mode is RecoveryMode.HYBRID  # gap 800; not an age cutoff
    assert result.evidence["R"] == pytest.approx(.8)
    assert "staleness_threshold" not in result.evidence
    assert "projected_expert_staleness_density" not in result.evidence


@pytest.mark.parametrize("upper,expected", [(.05, RecoveryMode.HYBRID),
                                           (.05001, RecoveryMode.CHECKPOINT)])
def test_inclusive_risk_boundary(upper, expected):
    value = evidence(); value["risk_upper"] = upper
    assert policy(value).decide(facts()).mode is expected


def test_audit_only_never_executes_hybrid():
    result = policy(evidence(), "audit-only").decide(facts())
    assert result.mode is RecoveryMode.CHECKPOINT
    assert result.evidence["would_admit"] is True


@pytest.mark.parametrize("key,value", [
    ("risk_upper", float("nan")), ("risk_upper", -1), ("risk_upper", True),
    ("risk_upper", 1.1), ("context_digest", "stale"), ("calibration_digest", "stale"),
    ("in_support", False), ("support_checked", False), ("run_history_checked", False),
    ("expires_at", "2020-01-01T00:00:00Z"), ("expires_at", "2030-01-01"),
    ("schema_version", True),
])
def test_invalid_prediction_fails_closed(key, value):
    e = evidence(); e[key] = value
    result = policy(e).decide(facts())
    assert result.mode is RecoveryMode.CHECKPOINT
    assert result.evidence["fallback_reason"] == "quality_evidence_invalid"


@pytest.mark.parametrize("key,value", [("bound_scope", "marginal"),
    ("qualification", "unreviewed"), ("independent_audit", False),
    ("alpha_run", .01), ("eta_final", .1), ("eta_peak", .5),
    ("expires_at", "2020-01-01T00:00:00Z"), ("outcome_definition", "per-fault")])
def test_incompatible_calibration_fails_closed(key, value):
    e = evidence(); e["calibration"][key] = value
    e["calibration_digest"] = digest(e["calibration"])
    assert policy(e).decide(facts()).mode is RecoveryMode.CHECKPOINT


@pytest.mark.parametrize("change", ["model", "history", "epoch", "step", "source", "exposure"])
def test_prediction_is_bound_to_complete_context(change):
    f = facts(); e = evidence(f)
    if change == "model":
        f.quality_context["model_id"] = "unseen-architecture"
    elif change == "history":
        f.quality_context["run_history"].append({"violation": True, "fallback": True})
    elif change == "epoch": f.control_scope["recovery_epoch"] = 2
    elif change == "step": f = replace(f, resume_step=1001)
    elif change == "source":
        f.available_state_sources["dense"][0].metadata["changed"] = True
    else:
        f = replace(f, exposure_history=(ExposureEvent(900, (1,), "hybrid", 800, 1),))
    assert risk_context_digest(f) != e["context_digest"]
    assert policy(e).decide(f).mode is not RecoveryMode.HYBRID


def test_missing_provider_and_incomplete_checkpoint_abort():
    assert QualityRiskPolicy().decide(facts()).mode is RecoveryMode.CHECKPOINT
    f = facts(); f.available_state_sources["dense"] = f.available_state_sources["dense"][:1]
    assert QualityRiskPolicy().decide(f).mode is RecoveryMode.ABORT
    with pytest.raises(ValueError): QualityRiskPolicy(mode="enforce")


def test_file_provider_pins_calibration_and_reads_new_prediction(tmp_path):
    c, p = tmp_path / "calibration.json", tmp_path / "prediction.json"
    write_json(c, calibration()); write_json(p, evidence())
    provider = FileRiskEvidenceProvider(c, p)
    first = provider.evaluate(facts())
    write_json(c, {"corrupted": True})
    assert provider.evaluate(facts())["calibration"] == first["calibration"]
    e = evidence(); e["risk_upper"] = .1; write_json(p, e)
    assert provider.evaluate(facts())["risk_upper"] == .1


def test_worker_context_changes_coordinator_request_digest():
    payload = {"quality_context": {"features": {"x": 1}}}
    first = _stable_request_digest(payload)
    payload["quality_context"]["features"]["x"] = 2
    assert _stable_request_digest(payload) != first


def test_predictor_exception_fails_closed():
    class Broken:
        def evaluate(self, facts): raise RuntimeError("predictor temporarily unavailable")
    result = QualityRiskPolicy(Broken(), mode="enforce").decide(facts())
    assert result.mode is RecoveryMode.CHECKPOINT
    assert "temporarily unavailable" in result.evidence["risk_evidence_error"]


def test_coordinator_binds_authoritative_scope_and_rejects_rank_disagreement():
    from moegambit.control.service import RecoveryCoordinatorService
    from moegambit.policy import serialize_source_candidates
    from moegambit.errors import ContractViolation
    class Trusted:
        def evaluate(self, normalized):
            assert normalized.control_scope == facts().control_scope
            return evidence(normalized)
    f = facts()
    payload = {"at_step": f.resume_step, "recovery_epoch": 1, "world_size": 2,
               "topology_generation": 1, "group_manifest_hash": "topology",
               "classification": {"recoverable": True, "failure_class": "fail_stop", "failed_ranks": [1]},
               "capabilities": AdapterCapabilities(static_world_replacement=True,
                    full_group_rebuild=True, peer_parameter_restore=True,
                    moe_state_classification=True).as_dict(),
               "latest_checkpoint_step": 200, "exposure_history": [],
               "quality_context": f.quality_context,
               "control_scope": {"job_id": "spoofed"},
               "available_state_sources": serialize_source_candidates(f.available_state_sources),
               "state_catalog": [
                   {"identity": identity, "version": {"committed_step": 1000,
                    "optimizer_generation": 1000, "recovery_epoch": 1}}
                   for identity in f.available_state_sources]}
    service = RecoveryCoordinatorService(
        store_provider=lambda payload: {"host": "127.0.0.1", "port": 23000},
        policy=QualityRiskPolicy(Trusted(), mode="enforce"))
    response = service.prepare(payload, job_id="job", attempt_id="a1")
    assert response["plan"]["mode"] == "hybrid"
    payload["quality_context"]["features"]["exposed_mass"] = .25
    with pytest.raises(ContractViolation, match="different recovery facts"):
        service.prepare(payload, job_id="job", attempt_id="a1")


def test_cli_enforce_requires_artifacts_before_creating_store(tmp_path):
    from moegambit.cli.watcher import main
    path = tmp_path / "never_created.sqlite"
    with pytest.raises(SystemExit, match="requires qualified"):
        main(["--rendezvous-host", "127.0.0.1", "--rendezvous-port", "12345",
              "--policy", "quality-risk", "--quality-mode", "enforce",
              "--control-store-path", str(path)])
    assert not path.exists()


def test_observed_peak_violation_cannot_be_erased_by_restart_or_predictor():
    f = facts()
    f.quality_context["run_history"] = [{"peak_violation_observed": True,
                                         "restarted_afterward": True}]
    e = evidence(f); e["risk_upper"] = 0
    result = policy(e).decide(f)
    assert result.mode is RecoveryMode.CHECKPOINT
    assert result.evidence["risk_upper"] == 1
    assert result.evidence["R"] == 20
    assert result.evidence["would_admit"] is False
