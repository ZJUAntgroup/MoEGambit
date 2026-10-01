"""Independent Bernoulli whole-run audit; never a per-state conditional proof."""
from __future__ import annotations

import math
from .io import integer


def binomial_upper(trials: int, violations: int, confidence: float = .95) -> float:
    """One-sided exact Clopper--Pearson upper bound, without scipy.

    Each trial must be an independent complete run of the frozen policy under
    the declared run/fault distribution, not a correlated checkpoint splice.
    """
    integer(trials, "trials", 1)
    integer(violations, "violations")
    if violations > trials:
        raise ValueError("violations cannot exceed trials")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if violations == trials:
        return 1.
    if violations == 0:
        return -math.expm1(math.log1p(-confidence) / trials)
    target = 1 - confidence
    lo, hi = violations / trials, 1.
    coefficients = [math.lgamma(trials + 1) - math.lgamma(i + 1) -
                    math.lgamma(trials - i + 1) for i in range(violations + 1)]
    for _ in range(80):
        p = (lo + hi) / 2
        if p == 1.:
            break
        terms = [c + i * math.log(p) + (trials - i) * math.log1p(-p)
                 for i, c in enumerate(coefficients)]
        largest = max(terms)
        cdf = math.exp(largest) * sum(math.exp(t - largest) for t in terms)
        if cdf > target:
            lo = p
        else:
            hi = p
    return hi


def risk_audit(trials: int, violations: int, *, confidence: float = .95,
               alpha_run: float = .05) -> dict:
    if type(alpha_run) not in (int, float) or not 0 < alpha_run < 1:
        raise ValueError("alpha_run must be between zero and one")
    upper = binomial_upper(trials, violations, confidence)
    return {"schema_version": 1, "trials": trials, "violations": violations,
            "confidence": confidence, "alpha_run": alpha_run, "risk_upper": upper,
            "within_budget": upper <= alpha_run,
            "scope": "marginal whole-run risk under independent identically distributed audit trials",
            "conditional_guarantee": False,
            "requirements": ["frozen policy and support before audit", "independent whole runs",
                             "no audit reuse for tuning or acceptance selection",
                             "all predeclared final/peak measurements complete"]}
