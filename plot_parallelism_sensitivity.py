import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Data: 4 parallelism configurations, all 64 GPUs (8 nodes x 8 GPUs)
# =============================================================================

configs = [
    # (label, restart_s, moegambit_s, recovery_path)
    # restart = load-checkpoint time; moegambit = post-fault step+1 elapsed avg
    ('TP=1,ETP=1\nPP=8,EP=8\nEDP=1', 36.4, 28.9, 'Hybrid'),       # from RQ1 (unchanged)
    ('TP=2,ETP=2\nPP=4,EP=8\nEDP=1', 28.9, 17.8, 'Hybrid'),       # parallelism1.log
    ('TP=2,ETP=2\nPP=4,EP=4\nEDP=2', 75.3, 21.4, 'Full-Peer'),    # parallelism2.log
    ('TP=1,ETP=1\nPP=8,EP=4\nEDP=2', 82.5, 23.2, 'Full-Peer'),    # parallelism3.log
]

# =============================================================================
# Plot: 1x4 subplots
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.8))

for idx, (ax, (label, restart, moegambit, path)) in enumerate(zip(axes.flat, configs)):
    x = np.arange(2)
    width = 0.5
    colors = ['#d62728', '#1f77b4']
    vals = [restart, moegambit]

    bars = ax.bar(x, vals, width, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)

    # Value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, color='#333333')

    # Speedup ratio annotation
    max_h = max(vals)
    if moegambit < restart:
        ratio = restart / moegambit
        ratio_text = f'{ratio:.1f}\u00d7'
        ratio_color = '#2ca02c'
    else:
        ratio = moegambit / restart
        ratio_text = f'{ratio:.1f}\u00d7\u2193'
        ratio_color = '#d62728'

    ax.annotate(ratio_text,
                xy=(0.5, max_h * 1.08),
                ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color=ratio_color)

    ax.set_xticks(x)
    ax.set_xticklabels(['Restart', 'Gambit'], fontsize=5.5)
    subplot_letter = chr(ord('a') + idx)
    ax.set_title(f'({subplot_letter}) {label}', fontsize=6, fontweight='bold', pad=4)

    # Subtitle: recovery path
    ax.text(0.5, -0.18, f'{path}', transform=ax.transAxes,
            ha='center', fontsize=5, color='#555555')

    if idx in (0, 2):
        ax.set_ylabel('Latency (s)', fontsize=6)
    ax.tick_params(axis='y', labelsize=5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max_h * 1.35)

plt.tight_layout(w_pad=1.0, h_pad=2.0)
plt.savefig('/Users/zds/bsr/parallelism_sensitivity.pdf', dpi=300, bbox_inches='tight')
print('Saved parallelism_sensitivity.pdf')
print()
for label, restart, moegambit, path in configs:
    label_flat = label.replace('\n', ', ')
    if moegambit < restart:
        print(f'  {label_flat}: Restart={restart}s, MoEGambit={moegambit}s, speedup={restart/moegambit:.2f}x ({path})')
    else:
        print(f'  {label_flat}: Restart={restart}s, MoEGambit={moegambit}s, SLOWER={moegambit/restart:.2f}x ({path})')
