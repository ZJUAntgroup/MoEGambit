#!/usr/bin/env python3
"""Plot the user-approved eight-task comparison from the manuscript table."""
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


def main():
    apply_style()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
    report = downstream()
    (OUT / 'downstream_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Downstream task figure checked and generated.')


if __name__ == '__main__':
    main()
