#!/usr/bin/env python3
"""Audit recorded quality-admission decisions; never infer decisions from losses.

All decisions are job-wide, not one record per affected rank. Missing labels
remain unknown. Independence/frozen-policy declarations are explicit assumptions.
Only Python's standard library is needed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


def number(value, name, lower=0.0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected a number')
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise ValueError(f'{name}: invalid value {value!r}')
    return float(value)


def records(paths, key):
    """Deduplicate identical job-wide records; conflicting duplicates are errors."""
    found, duplicates = {}, 0
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = tuple(row[k] for k in key)
            if any(not isinstance(v, str) or not v for v in identity):
                raise ValueError(f'{path}:{line_no}: missing identity')
            if identity in found:
                if found[identity] != row:
                    raise ValueError(f'conflicting record {identity}; reconcile node/process logs')
                duplicates += 1
            else:
                found[identity] = row
    return list(found.values()), duplicates


def ratio(a, b):
    return dict(numerator=a, denominator=b, fraction=a / b if b else None)


def summarize_decisions(rows, alpha):
    reasons = Counter()
    hybrid, eligible, quality_reject = [], [], []
    missing_scores = 0
    policies = set()
    for row in rows:
        action = row['action']
        if action not in ('Hybrid', 'Restart'):
            raise ValueError('action must be Hybrid or Restart')
        if not isinstance(row['guard_eligible'], bool):
            raise ValueError('guard_eligible must be JSON true/false')
        if not row.get('policy_id') or not row.get('reason'):
            raise ValueError('policy_id and first failed guard/reason are required')
        policies.add(row['policy_id'])
        if 'alpha_run' in row and row['alpha_run'] != alpha:
            raise ValueError('decision alpha_run differs from frozen audit target')
        if isinstance(row.get('step'), bool) or not isinstance(row.get('step'), int) or row['step'] < 0:
            raise ValueError('step must be a nonnegative integer')
        score = row.get('risk_upper')
        if score is None:
            missing_scores += 1
        else:
            score = number(score, 'risk_upper', upper=1.0)
            if row.get('risk_score_R') is not None and not math.isclose(
                    number(row['risk_score_R'], 'R'), score / alpha, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError('R must equal risk_upper / alpha_run')
        if action == 'Hybrid':
            if not row['guard_eligible']:
                raise ValueError('Hybrid admitted with an integrity/source/support guard false')
            if score is not None and score > alpha:
                raise ValueError('Hybrid admitted above the declared risk budget')
            hybrid.append(row)
        else:
            reasons[row['reason']] += 1
            if row['guard_eligible']:
                if score is not None and score <= alpha:
                    raise ValueError('eligible Restart below budget: log additional guard or policy override')
                quality_reject.append(row)
        if row['guard_eligible']:
            eligible.append(row)
        label = row.get('hybrid_Y')
        if label is not None:
            number(label, 'hybrid_Y')
            if row.get('label_scope') != 'complete_candidate_continuation':
                raise ValueError('Hybrid Y must describe the complete candidate continuation, not a short endpoint')
            if not row.get('label_source'):
                raise ValueError('Hybrid Y needs a paired-validation source')
    known_h = [r for r in hybrid if r.get('hybrid_Y') is not None]
    known_r = [r for r in quality_reject if r.get('hybrid_Y') is not None]
    # Rejected states need a counterfactual Hybrid continuation; a Restart loss
    # cannot reveal whether the candidate Hybrid action was safe.
    unsafe_h = sum(r['hybrid_Y'] > 1 for r in known_h)
    safe_r = sum(r['hybrid_Y'] <= 1 for r in known_r)
    safe_all = [r for r in known_h + known_r if r['hybrid_Y'] <= 1]
    return dict(
        job_wide_decisions=len(rows), policy_ids=sorted(policies),
        distinct_training_runs=len({r['run_id'] for r in rows}),
        hybrid_count=len(hybrid), restart_count=len(rows)-len(hybrid),
        guard_ineligible_count=len(rows)-len(eligible),
        quality_gate_rejections=len(quality_reject),
        admission=ratio(len(hybrid), len(rows)),
        eligible_admission=ratio(len(hybrid), len(eligible)),
        fallback_reasons=dict(reasons), missing_risk_bounds=missing_scores,
        hybrid_label_coverage=ratio(len(known_h), len(hybrid)),
        rejected_counterfactual_coverage=ratio(len(known_r), len(quality_reject)),
        unsafe_fraction_among_labelled_admissions=ratio(unsafe_h, len(known_h)),
        safe_fraction_among_labelled_quality_rejections=ratio(safe_r, len(known_r)),
        rejection_fraction_among_labelled_safe_candidates=ratio(safe_r, len(safe_all)),
        complete_decision_labels=(bool(rows) and len(known_h)==len(hybrid) and len(known_r)==len(quality_reject)),
        uncertainty='Descriptive decision ratios; no iid event-level interval is computed. '
                    'Multiple decisions and shared-prefix branches may be correlated.',
    )


def binomial_upper(k, n, delta=0.05):
    """One-sided exact Clopper--Pearson upper limit, by binomial CDF inversion."""
    if not 0 < delta < 1 or n < 1 or not 0 <= k <= n:
        raise ValueError('invalid binomial inputs')
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(delta) / n)
    def cdf(p):
        terms = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                 + j*math.log(p)+(n-j)*math.log1p(-p) for j in range(k+1)]
        maximum = max(terms)
        return math.exp(maximum)*sum(math.exp(t-maximum) for t in terms)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo+hi)/2
        if cdf(mid) > delta:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def summarize_runs(rows, manifest, alpha, delta):
    reasons = []
    completed = []
    units = []
    for r in rows:
        if r.get('complete') is not True or r.get('policy_Y') is None:
            reasons.append(f'incomplete whole-run record: {r["run_id"]}')
            continue
        number(r['policy_Y'], 'policy_Y')
        completed.append(r)
        unit = r.get('independent_unit_id')
        if not unit:
            reasons.append(f'missing independent unit: {r["run_id"]}')
        units.append(unit)
    if not completed:
        reasons.append('no complete whole-run outcomes')
    if len(set(units)) != len(units):
        reasons.append('multiple rows reuse an independent unit/shared prefix')
    if not manifest:
        reasons.append('missing frozen audit manifest')
    else:
        for field in ('iid_complete_runs_declared', 'policy_frozen_before_audit',
                      'audit_size_fixed_before_outcomes', 'no_audit_tuning_declared'):
            if manifest.get(field) is not True:
                reasons.append(f'{field} is not declared')
        if manifest.get('alpha_run') != alpha or manifest.get('delta') != delta:
            reasons.append('manifest alpha/delta differs from target')
        for field in ('policy_id', 'policy_artifact_sha256', 'audit_protocol_id',
                      'distribution_description', 'fixed_evaluation_grid_id'):
            if not manifest.get(field):
                reasons.append(f'missing {field}')
        for field, value in [('eta_final', 0.005), ('eta_peak', 0.01)]:
            if manifest.get(field) != value:
                reasons.append(f'manifest {field} differs from common tolerance')
        for field in ('training_unit_ids', 'calibration_unit_ids'):
            if not isinstance(manifest.get(field), list):
                reasons.append(f'missing {field}')
            elif set(units) & set(manifest[field]):
                reasons.append(f'audit overlaps {field}')
        if len(completed) != manifest.get('planned_independent_audit_runs'):
            reasons.append('audit incomplete or size differs from predeclared design')
        for r in completed:
            if r.get('policy_id') != manifest.get('policy_id'):
                reasons.append('audit contains a different/tuned policy')
            if r.get('evaluation_grid_id') != manifest.get('fixed_evaluation_grid_id'):
                reasons.append('audit evaluation grid differs')
            if r.get('reference_policy') != 'whole_job_Restart' or not r.get('outcome_source'):
                reasons.append('unverified quality comparator/source')
    k = sum(r['policy_Y'] > 1 for r in completed)
    # No interval is reported for correlated, selected, incomplete, or mixed-policy data.
    upper = binomial_upper(k, len(completed), delta) if completed and not reasons else None
    return dict(whole_runs_supplied=len(rows), complete_runs=len(completed),
                observed_exceedances=k, exceedance=ratio(k, len(completed)),
                one_sided_confidence=1-delta, iid_binomial_upper=upper,
                alpha_run=alpha,
                certificate_under_declared_assumptions=(upper <= alpha if upper is not None else None),
                unavailable_reasons=sorted(set(reasons)),
                assumption_note='Declarations are audit prerequisites, not proof of independence. '
                                'The bound applies to this frozen policy and declared whole-run distribution; '
                                'it is not a conditional guarantee for every fault, architecture, or horizon.')


def inventory(roots, out):
    """Locate evidence, without reading checkpoint tensors or assuming log formats."""
    matches = []
    skip = {'ckpt', 'checkpoints', '.git', '__pycache__', 'Megatron-LM', 'DeepSpeed'}
    keywords = ('risk_upper', 'calibrated_run_risk', 'calibration', 'risk_score',
                'recovery_path_chosen', 'recovery.decided', 'admission', 'policy_id')
    for root in roots:
        for directory, names, files in os.walk(root):
            names[:] = sorted(n for n in names if n not in skip and not n.startswith('iter_'))
            for name in sorted(files):
                path = Path(directory)/name
                if path.suffix not in ('.log', '.jsonl', '.json', '.csv'):
                    continue
                try:
                    size = path.stat().st_size
                    # Inventory only: head/tail sampling of large logs is explicit.
                    with path.open('rb') as f:
                        chunk = f.read(1024*1024)
                        if size > 1024*1024:
                            f.seek(max(1024*1024, size-1024*1024)); chunk += f.read()
                    body = chunk.decode('utf-8', errors='replace')
                    present = [k for k in keywords if k in body]
                    if present:
                        matches.append(dict(path=str(path), bytes=size, keywords=present,
                                            head_tail_only=size > 2*1024*1024))
                except OSError as e:
                    matches.append(dict(path=str(path), error=str(e)))
    (out/'evidence_inventory.json').write_text(json.dumps(matches, indent=2)+'\n')
    return len(matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, nargs='*', default=[])
    parser.add_argument('--runs', type=Path, nargs='*', default=[])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--scan-root', type=Path, nargs='*', default=[])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads(args.manifest.read_text()) if args.manifest else None
    decisions, dd = records(args.decisions, ('run_id', 'event_id'))
    runs, rd = records(args.runs, ('run_id',))
    result = dict(schema_version=1,
                  decision_audit=summarize_decisions(decisions, 0.05),
                  whole_run_audit=summarize_runs(runs, m, 0.05, 0.05),
                  identical_decision_duplicates_removed=dd,
                  identical_run_duplicates_removed=rd,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in args.decisions+args.runs+([args.manifest] if args.manifest else [])})
    result['decision_audit_by_policy'] = {pid: summarize_decisions(
        [row for row in decisions if row['policy_id'] == pid], 0.05)
        for pid in result['decision_audit']['policy_ids']}
    result['missing_evidence'] = []
    if not decisions:
        result['missing_evidence'].append('No recorded R2 decisions; admission/error rates unavailable.')
    if not runs:
        result['missing_evidence'].append('No complete independent policy-run audit; certificate unavailable.')
    if args.scan_root:
        result['inventory_candidates'] = inventory(args.scan_root, args.out_dir)
    (args.out_dir/'r2_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    d, r = result['decision_audit'], result['whole_run_audit']
    lines = ['# R2 decision audit', '',
             f'Job-wide decisions: {d["job_wide_decisions"]}; Hybrid: {d["hybrid_count"]}; Restart: {d["restart_count"]}.',
             f'Admission fraction: {d["admission"]["fraction"]}; eligible admission: {d["eligible_admission"]["fraction"]}.',
             f'Labelled admitted outcomes: {d["hybrid_label_coverage"]}.',
             f'Unsafe admitted outcomes / labelled admissions: {d["unsafe_fraction_among_labelled_admissions"]}.',
             f'Safe rejected Hybrid counterfactuals / labelled quality rejections: {d["safe_fraction_among_labelled_quality_rejections"]}.',
             f'Complete whole-run audit: {r["complete_runs"]}; exceedances: {r["observed_exceedances"]}; upper bound: {r["iid_binomial_upper"]}.',
             '', 'Missing prerequisites:', *['- '+x for x in result['missing_evidence']+r['unavailable_reasons']],
             '', 'Unknown values are unavailable, not zero. Decision ratios do not certify alpha_run.']
    (args.out_dir/'r2_audit.md').write_text('\n'.join(lines)+'\n')
    print(args.out_dir/'r2_audit.json')


if __name__ == '__main__':
    main()
