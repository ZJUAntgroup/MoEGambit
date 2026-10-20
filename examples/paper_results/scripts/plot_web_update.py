#!/usr/bin/env python3
"""Replot measured Web update and FiQA exports, without starting training.

Outputs use MOEGAMBIT_PAPER_RESULTS_OUT (default /personal/moegambit/paper_results).
Reported query-bootstrap intervals are retained, not inferred from means.
"""
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import DATA, OUT


def read_csv(name):
    with (DATA / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def load_results():
    meta = json.loads((DATA / 'web_update_plan.json').read_text())
    checked((meta['completed_arms'], meta['completed_paired_cases'],
             meta['expected_all_cases']) == (12, 6, 6), 'Incomplete experiment')
    checked(meta['w1_completed'] and meta['w2_all_cases_completed'], 'Incomplete suite')
    plan = meta['plan']
    checked(plan['world_size'] == 64 and plan['r2_enabled'] is False,
            'Expected 64 GPUs with R2 disabled')
    checked(plan['state_treatment'] == 'hybrid_full_state', 'Expected full-state Hybrid')
    raw = read_csv('web_update_pipeline.csv')
    index = {(r['case_id'], r['arm']): r for r in raw}
    expected = {(c['id'], a) for c in plan['cases'] for a in ('restart', 'hybrid')}
    checked(len(raw) == len(index) == 12 and set(index) == expected,
            'Missing, unexpected or duplicate arm')
    pairs = []
    for case in plan['cases']:
        restart, hybrid = [index[(case['id'], arm)] for arm in ('restart', 'hybrid')]
        for row in (restart, hybrid):
            checked(row['r2_enabled'] == 'False' and
                    row['harness'] == plan['harness'], 'Measurement scope mismatch')
            checked(int(row['failed_ranks']) == len(case['ranks']) and
                    int(row['age']) == case['step'] - case['source'] and
                    int(row['endpoint']) == case['endpoint'], 'Case configuration mismatch')
            for key in ('update_seconds', 'gpu_hours', 'validation_loss'):
                value = float(row[key])
                checked(math.isfinite(value) and value > 0, 'Invalid ' + key)
            checked(math.isclose(float(row['gpu_hours']),
                                 float(row['update_seconds']) * 64 / 3600),
                    'GPU-hours differ from allocated time')
        age = int(restart['age'])
        checked(int(restart['replayed_updates']) == age and
                int(hybrid['replayed_updates']) == 0, 'Wrong replay accounting')
        rt, ht = float(restart['update_seconds']), float(hybrid['update_seconds'])
        checked(math.isclose(rt / ht, float(hybrid['pipeline_speedup'])), 'Wrong speedup')
        pairs.append(dict(case_id=case['id'], age=age, ranks=len(case['ranks']),
            restart_seconds=rt, hybrid_seconds=ht, speedup=rt / ht,
            gpu_hours_restart=float(restart['gpu_hours']),
            gpu_hours_hybrid=float(hybrid['gpu_hours']),
            saved_percent=100 * (1 - ht / rt),
            loss_change_percent=100 * (float(hybrid['validation_loss']) /
                                       float(restart['validation_loss']) - 1)))
    quality = read_csv('fiqa_quality.csv')
    for row in quality:
        for key in ('offset', 'queries'):
            row[key] = int(row[key])
        for key in ('restart', 'hybrid', 'difference', 'low95', 'high95',
                    'query_candidate_coverage', 'mean_candidate_recall',
                    'validation_loss_change'):
            row[key] = float(row[key])
            checked(math.isfinite(row[key]), 'Non-finite task measurement')
        checked(row['queries'] == 500, 'Expected 500 paired queries')
        checked(0 <= row['restart'] <= 1 and 0 <= row['hybrid'] <= 1,
                'Invalid task score')
        checked(math.isclose(row['hybrid'] - row['restart'], row['difference']),
                'Incorrect task difference')
        checked(row['low95'] <= row['difference'] <= row['high95'], 'Invalid interval')
    groups = [('s250_g50_one', 0), ('s250_g50_one', 100),
              ('s350_g150_four', 0), ('s350_g150_four', 100)]
    keys = {(r['case_id'], r['offset'], r['metric']) for r in quality}
    checked(len(keys) == len(quality) == 8 and keys ==
            {(case, off, metric) for case, off in groups for metric in ('mrr10', 'ndcg10')},
            'Missing, duplicate or unexpected task measurements')
    return meta, pairs, quality, groups


def save(fig, name):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=200,
                    bbox_inches='tight', pad_inches=.10)
    plt.close(fig)


def plot_timing(pairs):
    fig, ax = plt.subplots(figsize=(6.5, 2.7), layout='constrained')
    x = np.arange(len(pairs)); width = .34
    for offset, field, label, color in [(-width / 2, 'restart_seconds', 'Restart', '#B4BDC6'),
                                       (width / 2, 'hybrid_seconds', 'Full-state Hybrid', '#246B92')]:
        ax.bar(x + offset, [r[field] / 60 for r in pairs], width,
               color=color, label=label, edgecolor='#50616E', linewidth=.4)
    for i, row in enumerate(pairs):
        ax.text(i, max(row['restart_seconds'], row['hybrid_seconds']) / 60 + .6,
                f"{row['speedup']:.2f}×", ha='center', fontsize=10)
    ax.set_xticks(x, [f"{r['age']} steps\n{r['ranks']} rank" + ('s' if r['ranks'] > 1 else '')
                       for r in pairs])
    ax.set_xlabel('Restored expert age / affected ranks')
    ax.set_ylabel('Update completion time (min)')
    ax.set_ylim(0, 30); ax.set_yticks([0, 10, 20, 30])
    ax.legend(loc='upper left', frameon=False, ncol=2)
    ax.grid(axis='y', color='#E1E5E9', linewidth=.5); ax.set_axisbelow(True)
    save(fig, 'web_update_pipeline')


def plot_quality(quality, groups):
    fig, ax = plt.subplots(figsize=(6.5, 3.1), layout='constrained')
    for metric, shift, marker, color, label in [
            ('mrr10', -.10, 'o', '#246B92', 'MRR@10'),
            ('ndcg10', .10, 's', '#9A590B', 'nDCG@10')]:
        rows = [next(r for r in quality if (r['case_id'], r['offset'], r['metric']) ==
                     (case, off, metric)) for case, off in groups]
        errors = [[100 * (r['difference'] - r['low95']) for r in rows],
                  [100 * (r['high95'] - r['difference']) for r in rows]]
        ax.errorbar([100 * r['difference'] for r in rows], np.arange(4) + shift,
                    xerr=errors, fmt=marker, color=color, markersize=5,
                    linewidth=1.2, capsize=3, label=label)
    ax.set_yticks(range(4), ['50 steps / 1 rank · offset 0', '50 steps / 1 rank · offset 100',
                              '150 steps / 4 ranks · offset 0', '150 steps / 4 ranks · offset 100'])
    ax.set_ylim(3.5, -.7); ax.set_xlim(-3.5, 5.5)
    ax.set_xlabel('Hybrid − Restart (percentage points)')
    ax.axvline(0, color='#78848D', linestyle='--', linewidth=.9)
    ax.grid(axis='x', color='#E1E5E9', linewidth=.5); ax.set_axisbelow(True)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=2, frameon=False)
    save(fig, 'fiqa_reranking_quality')


def main():
    meta, pairs, quality, groups = load_results()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
        'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7})
    plot_timing(pairs)
    plot_quality(quality, groups)
    with (OUT / 'web_update_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    names = ('web_update_pipeline.csv', 'fiqa_quality.csv', 'web_update_plan.json')
    report = dict(input_sha256={n: hashlib.sha256((DATA / n).read_bytes()).hexdigest() for n in names},
        config_hash=meta['config_hash'], completed_pairs=6, timing_repeats_per_case=1,
        state_treatment='hybrid_full_state', r2_enabled=False, harness=meta['plan']['harness'],
        timing_scope=meta['plan']['timing_scope'],
        speedup_range=[min(r['speedup'] for r in pairs), max(r['speedup'] for r in pairs)],
        saved_gpu_hours_percent_range=[min(r['saved_percent'] for r in pairs), max(r['saved_percent'] for r in pairs)],
        endpoint_loss_change_percent_range=[min(r['loss_change_percent'] for r in pairs), max(r['loss_change_percent'] for r in pairs)],
        endpoint_task_intervals_cross_zero=all(r['low95'] <= 0 <= r['high95'] for r in quality if r['offset'] == 100),
        task_uncertainty='Reported 2000-replicate paired-query bootstrap intervals; 500 fixed queries',
        scope='Cold-start checkpoint-simulated update workflow; no hot-recovery or R2 risk estimate')
    (OUT / 'web_update_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
