import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data from paper (RQ5: Scalability) ---
# Qwen3-30B-A3B, single-failure recovery latency
gpu_labels = ['64 GPUs', '128 GPUs']

# Restart (full checkpoint restart)
restart_total = [36.4, 47.2]  # seconds

# MoEGambit (hybrid + two-phase)
moegambit_total = [28.9, 33.7]  # seconds

# Decomposition estimates (from paper mechanism description):
# Path P (peer pull, dense/router): ~0.8s at 64 GPU, ~1.0s at 128 GPU
# Path C (shard read, expert): the bulk of MoEGambit time
moegambit_pathP = [0.8, 1.0]
moegambit_pathC = [26.1, 30.7]  # total - pathP - two_phase_saving

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))

x = np.arange(len(gpu_labels))
width = 0.32

# Restart bars (single color)
bars_restart = ax.bar(x - width/2, restart_total, width,
                      color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.6,
                      label='Checkpoint Restart')

# MoEGambit bars (stacked: Path P + Path C)
bars_pathP = ax.bar(x + width/2, moegambit_pathP, width,
                    color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.6,
                    label='MoEGambit: Path P (peer)')
bars_pathC = ax.bar(x + width/2, moegambit_pathC, width,
                    bottom=moegambit_pathP,
                    color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6,
                    label='MoEGambit: Path C (shard)')

# Add ratio annotations
for i in range(len(gpu_labels)):
    ratio = restart_total[i] / moegambit_total[i]
    ax.annotate(f'{ratio:.2f}\u00d7',
                xy=(x[i], max(restart_total[i], moegambit_total[i]) + 1.5),
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#333333')

# Add value labels on bars
for bar in bars_restart:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f'{h:.1f}s', ha='center', va='bottom', fontsize=8, color='#333333')

for i, bar in enumerate(bars_pathC):
    total = moegambit_pathP[i] + moegambit_pathC[i]
    ax.text(bar.get_x() + bar.get_width()/2, total + 0.3,
            f'{moegambit_total[i]:.1f}s', ha='center', va='bottom', fontsize=8, color='#333333')

ax.set_ylabel('Recovery Latency (s)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(gpu_labels, fontsize=11)
ax.set_ylim(0, 58)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./scalability.pdf', dpi=300, bbox_inches='tight')
print('Saved scalability.pdf')
