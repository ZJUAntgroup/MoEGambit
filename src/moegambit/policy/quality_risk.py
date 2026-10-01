"""Quality-risk admission with coordinator-owned, context-bound evidence.

This module validates evidence contracts; it does not fit a risk predictor or
prove that an externally supplied conditional bound is statistically valid.
Legacy checkpoint-age/density thresholds are not used by this policy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..audit.io import canonical, digest, nonempty, read_json
from ..runtime.recovery_plan import RecoveryMode
from .base import PolicyDecision, RecoveryFacts, source_evidence
from .moe_hybrid import MoeHybridPolicy


def risk_context(facts: RecoveryFacts) -> Mapping[str, Any]:
    """All ranks must agree on this snapshot; retain the entire run history.

    quality_context includes model_id, telemetry_version, policy_version,
    features, and run_history (including any previous quality violations).
    A fallback must not truncate that history.
    """
    return {
        "control_scope": dict(facts.control_scope),
        "failed_ranks": list(facts.failed_ranks),
        "resume_step": facts.resume_step,
        "latest_checkpoint_step": facts.latest_checkpoint_step,
        "state_sources": source_evidence(facts),
        "capabilities": dict(facts.capabilities.as_dict()),
        "quality_context": dict(facts.quality_context),
        "exposure_history": [
            {"step": e.step, "ranks": list(e.ranks), "reason": e.reason,
             "checkpoint_step": e.checkpoint_step,
             "expert_state_count": e.expert_state_count}
            for e in facts.exposure_history
        ],
    }


def risk_context_digest(facts: RecoveryFacts) -> str:
    return digest(risk_context(facts))


class RiskEvidenceProvider(Protocol):
    """Trusted coordinator plugin, not a worker-supplied probability."""
    def evaluate(self, facts: RecoveryFacts) -> Mapping[str, Any]: ...


def _probability(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


def _expiry(value: Any) -> datetime:
    parsed = datetime.fromisoformat(nonempty(value, "expires_at").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("expires_at must include a timezone")
    return parsed


class FileRiskEvidenceProvider:
    """A snapshot of operator-managed calibration and prediction artifacts.

    The prediction file is read on each decision to allow an external predictor
    to publish a fresh atomic snapshot. Calibration is pinned until restart.
    A schema-valid artifact is NOT a statistical certificate by itself.
    """
    def __init__(self, calibration_path: str | Path, evidence_path: str | Path):
        self.calibration = read_json(calibration_path)
        self.calibration_digest = digest(self.calibration)
        self.evidence_path = Path(evidence_path)

    def evaluate(self, facts: RecoveryFacts) -> Mapping[str, Any]:
        raw = read_json(self.evidence_path)
        if not isinstance(raw, Mapping):
            raise ValueError("prediction must be an object")
        result = dict(raw)
        result["calibration"] = self.calibration
        result["loaded_calibration_digest"] = self.calibration_digest
        return result


class QualityRiskPolicy(MoeHybridPolicy):
    """Admit hybrid only when a qualified whole-run bound / alpha_run <= 1.

    audit-only never admits a candidate hybrid restoration. Complete live state
    still follows the peer path, which introduces no stale expert exposure.
    Existing source completeness/capability checks apply before risk admission.
    """
    def __init__(self, provider: RiskEvidenceProvider | None = None, *,
                 mode: str = "audit-only", eta_final: float = .005,
                 eta_peak: float = .01, alpha_run: float = .05,
                 clock: Callable[[], datetime] | None = None):
        super().__init__()
        if mode not in ("audit-only", "enforce"):
            raise ValueError("quality mode must be audit-only or enforce")
        self.mode = mode
        self.provider = provider
        self.eta_final = _probability(eta_final, "eta_final")
        self.eta_peak = _probability(eta_peak, "eta_peak")
        self.alpha_run = _probability(alpha_run, "alpha_run")
        if min(self.eta_final, self.eta_peak, self.alpha_run) <= 0:
            raise ValueError("quality tolerances and risk budget must be positive")
        if mode == "enforce" and provider is None:
            raise ValueError("enforce mode requires a coordinator-owned risk provider")
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _policy_evidence(self):
        return {"quality_mode": self.mode, "eta_final": self.eta_final,
                "eta_peak": self.eta_peak, "alpha_run": self.alpha_run,
                "admission_threshold": 1.0, "bound_scope": "whole_run_conditional"}

    def _validate(self, facts, prediction):
        context = facts.quality_context
        for key in ("model_id", "telemetry_version", "policy_version"):
            nonempty(context.get(key), key)
        if not isinstance(context.get("features"), Mapping):
            raise ValueError("quality features are missing")
        if not isinstance(context.get("run_history"), list):
            raise ValueError("whole-run recovery/violation history is missing")
        for item in context["run_history"]:
            if not isinstance(item, Mapping):
                raise ValueError("run history entries must be objects")
            if "peak_violation_observed" in item and type(item["peak_violation_observed"]) is not bool:
                raise ValueError("peak_violation_observed must be a boolean")
        for key in ("job_id", "attempt_id", "group_manifest_hash"):
            nonempty(facts.control_scope.get(key), key)
        for key in ("recovery_epoch", "topology_generation"):
            value = facts.control_scope.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"authoritative {key} is missing or invalid")
        canonical(context)
        calibration = prediction["calibration"]
        if (type(calibration["schema_version"]) is not int or
                type(prediction["schema_version"]) is not int or
                calibration["schema_version"] != 1 or prediction["schema_version"] != 1):
            raise ValueError("unsupported quality evidence schema")
        expected_digest = digest(calibration)
        if prediction["calibration_digest"] != expected_digest:
            raise ValueError("prediction was made for a different calibration artifact")
        if prediction["context_digest"] != risk_context_digest(facts):
            raise ValueError("prediction context is stale or mismatched")
        if _expiry(calibration["expires_at"]) <= self.clock():
            raise ValueError("calibration has expired")
        if _expiry(prediction["expires_at"]) <= self.clock():
            raise ValueError("prediction has expired")
        if calibration["bound_scope"] != "whole_run_conditional":
            raise ValueError("a marginal/per-fault bound cannot authorize whole-run admission")
        if calibration["qualification"] != "operator_reviewed":
            raise ValueError("calibration has not been qualified by the operator")
        if calibration["independent_audit"] is not True:
            raise ValueError("independent audit is not declared")
        for key in ("method", "assumptions", "audit_report_sha256", "outcome_definition"):
            nonempty(calibration.get(key), key)
        report_hash = calibration["audit_report_sha256"]
        if len(report_hash) != 64 or any(c not in "0123456789abcdef" for c in report_hash):
            raise ValueError("audit report digest must be a lowercase SHA-256 hash")
        if calibration["outcome_definition"] != "paired-whole-run-final-and-peak-v1":
            raise ValueError("incompatible quality outcome definition")
        for key, expected in (("eta_final", self.eta_final), ("eta_peak", self.eta_peak),
                              ("alpha_run", self.alpha_run)):
            if _probability(calibration[key], key) != expected:
                raise ValueError(f"calibration {key} differs from the configured target")
        support = calibration["support"]
        for key in ("model_id", "telemetry_version", "policy_version"):
            if not isinstance(support[key], list) or context[key] not in support[key]:
                raise ValueError(f"{key} is outside the calibrated support")
        if prediction["support_checked"] is not True or prediction["in_support"] is not True:
            raise ValueError("feature support was not checked or is outside support")
        if prediction["run_history_checked"] is not True:
            raise ValueError("predictor did not account for prior run history")
        upper = _probability(prediction["risk_upper"], "risk_upper")
        # Y contains a maximum over the full evaluation schedule: an already
        # observed peak violation cannot be undone by a later checkpoint restart.
        if any(item.get("peak_violation_observed") is True for item in context["run_history"]):
            upper = 1.0
        return upper, expected_digest

    def _admit_hybrid(self, facts, checkpoint, evidence, classified, unique):
        try:
            evidence["context_digest"] = risk_context_digest(facts)
            if self.provider is None:
                raise ValueError("no calibrated risk provider is configured")
            prediction = self.provider.evaluate(facts)
            upper, calibration_digest = self._validate(facts, prediction)
            risk = upper / self.alpha_run
            evidence.update({"risk_upper": upper, "R": risk,
                             "calibration_digest": calibration_digest,
                             "would_admit": risk <= 1})
        except Exception as exc:  # An unavailable external predictor cannot authorize hybrid.
            evidence["risk_evidence_error"] = str(exc)
            return self._fallback(checkpoint, "quality-risk evidence is unavailable or invalid",
                                  evidence, "quality_evidence_invalid")
        if self.mode == "audit-only":
            return self._fallback(checkpoint, "quality rule audited without hybrid execution",
                                  evidence, "quality_audit_only")
        if risk > 1:
            return self._fallback(checkpoint, "whole-run quality-risk bound exceeds the budget",
                                  evidence, "quality_risk_exceeded")
        return PolicyDecision(RecoveryMode.HYBRID,
                              "qualified whole-run quality-risk bound is within budget", evidence)
