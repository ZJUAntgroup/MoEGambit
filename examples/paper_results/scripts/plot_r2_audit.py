#!/usr/bin/env python3
"""Display reported counts and separately recomputed exact upper limits."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import DATA, OUT
from report_results import audit_aggregate
from plot_style import apply_style, grid

s = json.loads((DATA / 'r2_audit_aggregate.json').read_text())
r = audit_aggregate(); c = s['candidate_decisions']; m = r['metrics']
apply_style()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.8), layout='constrained')
counts = [[c['safe_admitted'], c['safe_rejected']],
          [c['unsafe_admitted'], c['unsafe_rejected']]]
ax.imshow(counts, cmap='Blues', vmin=0, vmax=150)
ax.set_xticks([0, 1], ['Hybrid admitted', 'Restart selected'])
ax.set_yticks([0, 1], ['Candidate safe', 'Candidate unsafe'])
ax.set_title('(a) Candidate decision counts', loc='left')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(counts[i][j]), ha='center', va='center',
                fontsize=14, color='white' if counts[i][j] > 80 else '#243644')
labels = ['Complete policy runs\n3 / 200', 'Admitted candidates\n3 / 132']
rates = [100*m['run_exceedance_rate'], 100*m['admitted_candidate_exceedance_rate']]
bounds = [100*m['run_upper_one_sided_95'], 100*m['admitted_candidate_upper_one_sided_95']]
bx.scatter([0, 1], rates, label='Observed rate', color='#265F84', marker='o', s=40)
bx.scatter([0, 1], bounds, label='One-sided 95% upper', color='#C46B2E', marker='D', s=35)
for x, lo, hi in zip([0, 1], rates, bounds):
    bx.plot([x, x], [lo, hi], color='#AAB4BE', linewidth=1)
    bx.annotate(f'{hi:.2f}%', (x, hi), (7, 5), textcoords='offset points', fontsize=8)
bx.axhline(5, linestyle='--', color='#697586', linewidth=.8)
bx.text(-.35, 5.15, '5% run-risk target', fontsize=7.5, color='#55606F')
bx.set_xticks([0, 1], labels); bx.set_xlim(-.5, 1.55); bx.set_ylim(0, 7.6)
bx.set_ylabel('Exceedance probability (%)'); grid(bx)
bx.set_title('(b) Different risk denominators', loc='left')
bx.legend(frameon=False, loc='upper left', fontsize=7)
fig.savefig(OUT / 'r2_audit.png', dpi=200)
fig.savefig(OUT / 'r2_audit.pdf')
