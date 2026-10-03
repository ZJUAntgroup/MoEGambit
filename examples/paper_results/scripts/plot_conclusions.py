#!/usr/bin/env python3
"""Plot manuscript conclusions from supplied tables; never synthesize samples."""
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from paths import DATA, OUT
from plot_style import apply_style, grid

BLUE, ORANGE, PURPLE = '#265F84', '#C46B2E', '#805993'


def rows(name):
    with (DATA / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'Expected finite positive measurement: {value}')
    return result


def restoration():
    source = rows('restoration_ablation.csv')
    keys = [(r['path'], r['attachment']) for r in source]
    expected = {(p, a) for p in ('FullLoad', 'Hybrid') for a in ('Single-phase', 'Two-phase')}
    if len(keys) != 4 or set(keys) != expected or any(int(r['events']) != 500 for r in source):
        raise ValueError('Expected the four 500-event restoration cells')
    values = {(r['path'], r['attachment']): positive(r['mean_seconds']) for r in source}
    fig, ax = plt.subplots(figsize=(6.2, 3), layout='constrained')
    x = np.arange(2)
    for offset, attachment, color in [(-.18, 'Single-phase', ORANGE), (.18, 'Two-phase', BLUE)]:
        y = [values[p, attachment] for p in ('FullLoad', 'Hybrid')]
        bars = ax.bar(x + offset, y, .34, label=attachment, color=color)
        ax.bar_label(bars, fmt='%.3f s', padding=4, fontsize=8)
    base, best = values['FullLoad', 'Single-phase'], values['Hybrid', 'Two-phase']
    reduction = 100 * (1 - best / base)
    ax.set_xticks(x, ['FullLoad (rank-local)', 'Hybrid (selective)'])
    ax.set_ylabel('Restoration latency (s; replay excluded)')
    ax.set_ylim(0, 46); grid(ax)
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('Selective restoration + two-phase attachment', loc='left', fontweight='bold')
    ax.text(.02, .95, f'Combined reduction: {reduction:.1f}%\n{base - best:.3f} s saved',
            transform=ax.transAxes, va='top', fontsize=8, color='#46515B')
    save(fig, 'restoration_ablation')
    source_effect = (values['FullLoad', 'Single-phase'] + values['FullLoad', 'Two-phase']
                     - values['Hybrid', 'Single-phase'] - values['Hybrid', 'Two-phase']) / 2
    phase_effect = (values['FullLoad', 'Single-phase'] + values['Hybrid', 'Single-phase']
                    - values['FullLoad', 'Two-phase'] - values['Hybrid', 'Two-phase']) / 2
    return dict(combined_reduction_percent=reduction, saved_seconds=base-best,
                marginal_selective_seconds=source_effect, marginal_two_phase_seconds=phase_effect,
                scope='Four cell means, rank-local latency; no event-level confidence intervals')


def burst():
    source = rows('burst_cell_means.csv')
    keys = [(int(r['affected_ranks']), int(r['expert_age_steps'])) for r in source]
    if len(keys) != 12 or set(keys) != {(r,a) for r in (8,16,24) for a in (50,100,150,200)}:
        raise ValueError('Unexpected burst sweep coverage')
    values = {k: float(r['validation_loss_deviation_baseline_sd']) for k,r in zip(keys,source)}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite burst cell')
    matrix = np.array([[values[r,a] for a in (50,100,150,200)] for r in (8,16,24)])
    fig, ax = plt.subplots(figsize=(6.3, 2.8), layout='constrained')
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(4), ['50','100','150','200']); ax.set_yticks(range(3), ['8','16','24'])
    ax.set_xlabel('Expert age (training steps)'); ax.set_ylabel('Affected ranks')
    ax.set_title('Burst-fault full-state recovery', loc='left', fontweight='bold')
    for i in range(3):
        for j in range(4):
            ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=12,
                    color='white' if abs(matrix[i,j]) > 1.3 else '#243644')
    fig.colorbar(image, ax=ax, label='Signed deviation (baseline SD)')
    save(fig,'burst_quality')
    return dict(cells=12,maximum_signed_baseline_sd=float(matrix.max()),
                age_50_maximum_absolute_baseline_sd=float(np.abs(matrix[:,0]).max()),
                scope='Baseline-SD units, not percent quality loss or R2 risk probability')


def downstream():
    source=rows('downstream_accuracy.csv')
    if [r['task'] for r in source] != ['ARC-E','BoolQ','MathQA','OBQA','PIQA','RACE','SWAG','WG']:
        raise ValueError('Unexpected downstream tasks')
    arms=('Restart','MoC PEC','MoEGambit'); colors=('#677581',ORANGE,BLUE)
    values={arm:np.array([float(r[arm]) for r in source]) for arm in arms}
    if any(not np.isfinite(v).all() or ((v<0)|(v>100)).any() for v in values.values()):
        raise ValueError('Invalid downstream score')
    metrics={arm:float(v.mean()) for arm,v in values.items()}
    for arm, expected in zip(arms,(45.06,44.67,45.32)):
        if abs(metrics[arm]-expected) > .00501:
            raise ValueError('Downstream means do not match rounded manuscript values')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.2,3.1),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    x=np.arange(8)
    for arm,color,offset in zip(arms,colors,(-.24,0,.24)):
        ax.bar(x+offset,values[arm],.23,color=color,label=arm)
    labels=[r['task']+'\n'+r['metric'].replace('acc_norm','norm') for r in source]
    ax.set_xticks(x,labels,fontsize=7); ax.set_ylim(0,78)
    ax.set_ylabel('Zero-shot task score (%)'); ax.set_title('(a) Task scores at step 10,000',loc='left',fontweight='bold')
    ax.legend(frameon=False,ncol=3,fontsize=7,loc='upper left'); grid(ax)
    for arm,color,offset in [('MoC PEC',ORANGE,-.17),('MoEGambit',BLUE,.17)]:
        bx.bar(x+offset,values[arm]-values['Restart'],.32,color=color,label=arm)
    bx.axhline(0,color='#687581',linewidth=.8); grid(bx)
    bx.set_xticks(x,[r['task'] for r in source],rotation=35,ha='right',fontsize=7)
    bx.set_ylabel('Difference vs Restart (percentage points)')
    bx.set_ylim(-2.0,2.4); bx.set_title('(b) Improvements and regressions',loc='left',fontweight='bold')
    bx.text(.02,.96,'Equal-task means: '+ ' / '.join(f'{metrics[a]:.2f}' for a in arms),
            transform=bx.transAxes,va='top',fontsize=7,color='#46515B')
    save(fig,'downstream_accuracy')
    return dict(equal_task_mean_percent=metrics,metrics_by_task={r['task']:r['metric'] for r in source},
                moegambit_mean_change_percentage_points=metrics['MoEGambit']-metrics['Restart'],
                scope='Mixed acc/acc_norm metrics, equal task weights; no significance claim')


def overhead():
    s=json.loads((DATA/'control_path_overhead.json').read_text())
    lo=s['minimum_step_time_change_percent']; mean=s['mean_step_time_change_percent']; hi=s['maximum_step_time_change_percent']
    if not lo <= mean <= hi or s['range_is_confidence_interval'] or s['raw_per_repetition_samples_included']:
        raise ValueError('Unexpected overhead scope')
    if s['repetitions'] !=20 or s['steps_per_repetition'] !=1000 or not s['all_rank_fence_included'] or not s['device_completion_confirmation_included']:
        raise ValueError('Unexpected complete-control-path protocol')
    fig,ax=plt.subplots(figsize=(7.1,2),layout='constrained')
    ax.hlines(0,lo,hi,color='#8799A9',linewidth=5,zorder=1)
    ax.scatter([lo,hi],[0,0],color='#8799A9',marker='|',s=180,zorder=2)
    ax.scatter([mean],[0],color=BLUE,s=65,zorder=3,label='Mean of 20 repetitions')
    for value,label in [(lo,f'Minimum\n{lo:+.2f}%'),(mean,f'Mean\n{mean:+.3f}%'),(hi,f'Maximum\n{hi:+.2f}%')]:
        ax.annotate(label,(value,0),(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color='#BDC4CB',linestyle='--',linewidth=.8)
    ax.set_ylim(-.65,.8); ax.set_xlim(-.115,.095); ax.set_yticks([])
    ax.set_xlabel('Relative iteration-time change (%)'); grid(ax,axis='x')
    ax.set_title('Complete failure-free control path: observed mean and range',loc='left',fontweight='bold')
    ax.text(.02,.06,'20 × 1,000 iterations; device completion + all-rank fence included\nObserved range, not a confidence interval; monitoring hook alone < 6 µs/iteration',
            transform=ax.transAxes,fontsize=7.5,color='#46515B')
    save(fig,'control_path_overhead')
    return s


def moc_window():
    source=rows('moc_e2e_aggregate.csv')
    if [r['arm'] for r in source] !=['full_sync_native','pec_sync','pec_2level_async'] or any(int(r['runs'])!=1 or int(r['failed_ranks'])!=1 for r in source):
        raise ValueError('Unexpected MoC controlled-window protocol')
    labels=['Full-sync native','PEC-sync','PEC-2L async']; colors=['#677581',ORANGE,PURPLE]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.5,2.9),layout='constrained')
    for plot,field,title in [(ax,'window_seconds','(a) End-to-end training window'),
                              (bx,'recovery_replay_seconds','(b) Recovery + replay')]:
        values=[positive(r[field]) for r in source]
        bars=plot.bar(np.arange(3),values,color=colors,width=.6)
        plot.bar_label(bars,fmt='%.3f',padding=4,fontsize=8)
        plot.set_xticks(range(3),labels,fontsize=7.5); plot.set_ylabel('Seconds')
        plot.set_ylim(0,max(values)*1.2); grid(plot)
        plot.set_title(title,loc='left',fontweight='bold')
    save(fig,'moc_controlled_restart')
    return dict(runs_per_arm=1,failed_ranks=1,scope='Controlled real worker restart; window includes training/checkpointing/recovery/replay. Separate from layout restoration.')


def cross_model():
    source=rows('cross_model_restoration.csv')
    if [r['model'] for r in source] != ['Qwen3-30B-A3B','DeepSeek-V2-Lite']:
        raise ValueError('Unexpected cross-model configurations')
    full=[positive(r['full_load_seconds']) for r in source]; hybrid=[positive(r['hybrid_seconds']) for r in source]
    fig,ax=plt.subplots(figsize=(6.2,2.9),layout='constrained'); x=np.arange(2)
    for offset,values,color,label in [(-.18,full,ORANGE,'FullLoad'),(.18,hybrid,BLUE,'MoEGambit')]:
        bars=ax.bar(x+offset,values,.34,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f s',padding=4,fontsize=8)
    reductions=[100*(1-h/f) for f,h in zip(full,hybrid)]
    for i,value in enumerate(reductions):
        ax.text(i,max(full[i],hybrid[i])+4.3,f'{value:.1f}% lower',ha='center',fontsize=9,color=BLUE)
    ax.set_xticks(x,[r['model'] for r in source]); ax.set_ylabel('Rank-local restoration latency (s)')
    ax.set_ylim(0,49); grid(ax); ax.legend(frameon=False,loc='upper right')
    ax.set_title('Restoration gains across two model configurations',loc='left',fontweight='bold')
    save(fig,'cross_model_restoration')
    return dict(reduction_percent={r['model']:v for r,v in zip(source,reductions)},scope='Replay excluded; two different configurations, not an isolated architecture ablation')


def main():
    apply_style()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
    report=dict(restoration_ablation=restoration(),burst_quality=burst(),downstream_accuracy=downstream(),
                control_path_overhead=overhead(),moc_controlled_restart=moc_window(),cross_model_restoration=cross_model())
    (OUT/'conclusions_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Six additional conclusion figures checked and generated.')


if __name__=='__main__':
    main()
