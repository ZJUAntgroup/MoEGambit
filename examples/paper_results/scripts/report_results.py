#!/usr/bin/env python3
"""Recompute published aggregates without inventing per-event/per-run records."""
import csv
import hashlib
import json
import math
from audit_r2 import binomial_upper
from paths import DATA, OUT


def audit_aggregate():
    source = json.loads((DATA / 'r2_audit_aggregate.json').read_text())
    c = source['candidate_decisions']
    if set(c) != {'safe_admitted', 'safe_rejected', 'unsafe_admitted', 'unsafe_rejected'}:
        raise ValueError('Unexpected candidate categories')
    if any(type(v) is not int or v < 0 for v in c.values()):
        raise ValueError('Counts must be nonnegative integers')
    n, k = source['n_runs'], source['run_exceedances']
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or n == 0:
        raise ValueError('Invalid complete-run counts')
    if sum(c.values()) != n or source['run_safe'] + k != n:
        raise ValueError('Aggregate denominators disagree')
    admitted = c['safe_admitted'] + c['unsafe_admitted']
    unsafe = c['unsafe_admitted'] + c['unsafe_rejected']
    safe = c['safe_admitted'] + c['safe_rejected']
    delta = 1 - source['confidence_level']
    metrics = {
        'accuracy': (c['safe_admitted'] + c['unsafe_rejected']) / n,
        'admission_rate': admitted / n,
        'unsafe_interception_rate': c['unsafe_rejected'] / unsafe,
        'safe_rejection_rate': c['safe_rejected'] / safe,
        'run_exceedance_rate': k / n,
        'run_upper_one_sided_95': binomial_upper(k, n, delta),
        'admitted_candidate_exceedance_rate': c['unsafe_admitted'] / admitted,
        'admitted_candidate_upper_one_sided_95': binomial_upper(c['unsafe_admitted'], admitted, delta),
    }
    for key, value in metrics.items():
        if not math.isclose(value, source['derived_metrics'][key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'Published metric mismatch: {key}')
    return {
        'source': source['source'],
        'raw_per_run_records_supplied_locally': source['raw_per_run_records_supplied_locally'],
        'scope': source['scope'], 'n_runs': n, 'run_exceedances': k,
        'admitted_candidates': admitted, 'unsafe_admitted': c['unsafe_admitted'],
        'alpha_run': source['alpha_run'], 'confidence_level': source['confidence_level'],
        'metrics': metrics,
        'run_upper_within_target': metrics['run_upper_one_sided_95'] <= source['alpha_run'],
        'admitted_upper_within_target': metrics['admitted_candidate_upper_one_sided_95'] <= source['alpha_run'],
        'assumptions': source['author_confirmed_protocol'],
        'missing_reproducibility_details': source['unknown_details'],
    }


def main():
    recovery = []
    with (DATA / 'recovery_scaling.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            baseline = float(row['full_load_seconds'])
            if baseline <= 0:
                raise ValueError('FullLoad latency must be positive')
            item = dict(panel=row['panel'], label=row['label'], edp=int(row['edp']),
                        comparison='rank-local recovery, excluding replay',
                        full_load_seconds=baseline, repair_path=row['repair_path'])
            for column, name in [('repair_seconds', 'moegambit'),
                                 ('pec_sync_seconds', 'pec_sync'),
                                 ('pec_2level_async_seconds', 'pec_2level_async')]:
                if row[column]:
                    seconds = float(row[column])
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError('Recovery latency must be finite and positive')
                    item[name] = dict(seconds=seconds, speedup=baseline / seconds,
                                      reduction_percent=100 * (1 - seconds / baseline))
            recovery.append(item)
    with (DATA / 'moc_e2e_aggregate.csv').open(newline='') as f:
        moc = list(csv.DictReader(f))
    window_baseline = float(moc[0]['window_seconds'])
    for row in moc:
        speedup = window_baseline / float(row['window_seconds'])
        if abs(speedup - float(row['reported_window_speedup'])) > .00051:
            raise ValueError('Rounded MoC end-to-end speedup inconsistent')
        row['computed_window_speedup'] = speedup
    r2 = audit_aggregate()
    result = dict(
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(DATA.glob('*')) if p.suffix in ('.json', '.csv')},
        recovery=recovery, moc_controlled_restart=moc, r2=r2,
        timing_note='FullLoad is rank-local; 35.6x is a separate replay-inclusive manuscript result. These rounded means cannot reconstruct per-event confidence intervals.',
        quality_note='Short/shared-prefix quality cases are trajectory diagnostics; only the separate independent-run audit uses a binomial run-risk bound.',
    )
    (OUT / 'aggregate_report.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'r2_aggregate_report.json').write_text(json.dumps(r2, indent=2) + '\n')
    print(f'Recovery: {len(recovery)} cells; MoC window: {len(moc)} arms')
    print(f"R2: {r2['run_exceedances']}/{r2['n_runs']} complete runs; one-sided 95% upper = {100*r2['metrics']['run_upper_one_sided_95']:.2f}%")
    print(f"Admitted: {r2['unsafe_admitted']}/{r2['admitted_candidates']}; upper = {100*r2['metrics']['admitted_candidate_upper_one_sided_95']:.2f}% (different denominator)")
    print(f'Reports: {OUT}')


if __name__ == '__main__':
    main()
