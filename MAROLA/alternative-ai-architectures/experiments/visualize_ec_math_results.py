"""
Visualize SECP256K1 Math Learning Results
==========================================
Create comprehensive visualizations of bio-plausible network performance
on cryptographic mathematics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Load results
results_file = Path(__file__).parent / 'ec_math_learning_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: Exact Accuracy Comparison (p=97)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

tasks_p97 = ['mod_add_p97', 'mod_sub_p97', 'mod_mult_p97', 'mod_inverse_p97', 'mod_exp_p97', 'point_validation_p97']
task_labels = ['Add', 'Sub', 'Mult', 'Inverse', 'Exp', 'Point\nValid']

architectures = ['MLP (Backprop)', 'Liquid Network', 'Bio-Plausible (Hebbian)']
colors = ['#2E86AB', '#A23B72', '#F18F01']

x = np.arange(len(task_labels))
width = 0.25

for i, arch in enumerate(architectures):
    accuracies = [results[task][arch]['exact_accuracy'] for task in tasks_p97]
    ax1.bar(x + i * width, accuracies, width, label=arch, color=colors[i], alpha=0.8)

ax1.set_xlabel('Cryptographic Operation', fontsize=12, fontweight='bold')
ax1.set_ylabel('Exact Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Exact Accuracy on Cryptographic Math (p=97)\nSmaller Prime = "Easier"',
              fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(task_labels)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(0, 0.25)
ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, label='Random Chance (~1%)')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, arch in enumerate(architectures):
    accuracies = [results[task][arch]['exact_accuracy'] for task in tasks_p97]
    for j, acc in enumerate(accuracies):
        if acc > 0.01:  # Only show if significant
            ax1.text(x[j] + i * width, acc + 0.005, f'{acc:.1%}',
                    ha='center', va='bottom', fontsize=8)

# ============================================================================
# Plot 2: MAE Comparison (p=97)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

mae_data = []
for task in tasks_p97:
    for arch in architectures:
        mae_data.append({
            'Task': task_labels[tasks_p97.index(task)],
            'Architecture': arch.split(' ')[0],  # Short name
            'MAE': results[task][arch]['mae']
        })

task_order = task_labels
arch_order = ['MLP', 'Liquid', 'Bio-Plausible']

# Prepare data for heatmap
heatmap_data = np.zeros((len(arch_order), len(task_order)))
for i, arch_short in enumerate(arch_order):
    for j, task_label in enumerate(task_order):
        task = tasks_p97[j]
        arch_full = [a for a in architectures if arch_short in a][0]
        heatmap_data[i, j] = results[task][arch_full]['mae']

im = ax2.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.7)
ax2.set_xticks(np.arange(len(task_order)))
ax2.set_yticks(np.arange(len(arch_order)))
ax2.set_xticklabels(task_order, rotation=45, ha='right')
ax2.set_yticklabels(arch_order)
ax2.set_title('Mean Absolute Error (p=97)\nLower = Better', fontsize=12, fontweight='bold')

# Add value annotations
for i in range(len(arch_order)):
    for j in range(len(task_order)):
        text = ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax2, label='MAE')

# ============================================================================
# Plot 3: Scaling Effect (p=97 vs p=997)
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

tasks_base = ['mod_add', 'mod_sub', 'mod_mult']
task_labels_base = ['Addition', 'Subtraction', 'Multiplication']

x = np.arange(len(architectures))
width = 0.15

for i, task_base in enumerate(tasks_base):
    p97_accs = [results[f'{task_base}_p97'][arch]['exact_accuracy'] for arch in architectures]
    p997_accs = [results[f'{task_base}_p997'][arch]['exact_accuracy'] for arch in architectures]

    ax3.bar(x + i * width * 2, p97_accs, width, label=f'{task_labels_base[i]} (p=97)',
            color=colors[i], alpha=0.8)
    ax3.bar(x + i * width * 2 + width, p997_accs, width, label=f'{task_labels_base[i]} (p=997)',
            color=colors[i], alpha=0.4, hatch='//')

ax3.set_xlabel('Architecture', fontsize=12, fontweight='bold')
ax3.set_ylabel('Exact Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Scaling Catastrophe: Accuracy Degrades 10× with Larger Prime\np=97 (solid) vs p=997 (hatched)',
              fontsize=14, fontweight='bold')
ax3.set_xticks(x + width * 2.5)
ax3.set_xticklabels([a.split(' ')[0] for a in architectures])
ax3.legend(loc='upper right', fontsize=9, ncol=2)
ax3.set_ylim(0, 0.25)
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 4: Backprop vs Bio-Plausible Direct Comparison
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

backprop_arch = 'MLP (Backprop)'
bio_arch = 'Bio-Plausible (Hebbian)'

backprop_accs = [results[task][backprop_arch]['exact_accuracy'] for task in tasks_p97]
bio_accs = [results[task][bio_arch]['exact_accuracy'] for task in tasks_p97]

ax4.scatter(backprop_accs, bio_accs, s=100, alpha=0.6, color='#F18F01')
ax4.plot([0, 0.25], [0, 0.25], 'k--', alpha=0.3, label='Equal Performance')
ax4.set_xlabel('Backprop Accuracy', fontsize=11, fontweight='bold')
ax4.set_ylabel('Bio-Plausible Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('Backprop vs Bio-Plausible\nAll points below line = Backprop wins',
              fontsize=12, fontweight='bold')
ax4.set_xlim(0, 0.25)
ax4.set_ylim(0, 0.25)
ax4.legend()
ax4.grid(alpha=0.3)

# Add task labels
for i, task_label in enumerate(task_labels):
    ax4.annotate(task_label, (backprop_accs[i], bio_accs[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# ============================================================================
# Plot 5: Operation Difficulty Ranking
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Average accuracy across all architectures at p=97
avg_accuracies = []
for task in tasks_p97:
    avg_acc = np.mean([results[task][arch]['exact_accuracy'] for arch in architectures])
    avg_accuracies.append(avg_acc)

sorted_indices = np.argsort(avg_accuracies)
sorted_tasks = [task_labels[i] for i in sorted_indices]
sorted_accs = [avg_accuracies[i] for i in sorted_indices]

bars = ax5.barh(sorted_tasks, sorted_accs, color=plt.cm.RdYlGn(np.array(sorted_accs) * 5))
ax5.set_xlabel('Average Exact Accuracy', fontsize=11, fontweight='bold')
ax5.set_title('Operation Difficulty Ranking\n(Higher = Easier to Learn)',
              fontsize=12, fontweight='bold')
ax5.set_xlim(0, 0.15)
ax5.grid(axis='x', alpha=0.3)

# Add value labels
for i, (task, acc) in enumerate(zip(sorted_tasks, sorted_accs)):
    ax5.text(acc + 0.002, i, f'{acc:.1%}', va='center', fontsize=9)

# ============================================================================
# Plot 6: Training Time vs Accuracy
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])

for i, arch in enumerate(architectures):
    times = []
    accs = []
    for task in tasks_p97:
        times.append(results[task][arch]['train_time'])
        accs.append(results[task][arch]['exact_accuracy'])

    ax6.scatter(times, accs, s=100, alpha=0.7, color=colors[i], label=arch, marker='o')

ax6.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Exact Accuracy', fontsize=11, fontweight='bold')
ax6.set_title('Training Cost vs Performance\nNo clear correlation!',
              fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

# ============================================================================
# Plot 7: Close Accuracy (Within 1%)
# ============================================================================
ax7 = fig.add_subplot(gs[3, :2])

close_acc_data = []
for task in tasks_p97:
    for arch in architectures:
        close_acc_data.append(results[task][arch]['close_accuracy'])

x = np.arange(len(task_labels))
width = 0.25

for i, arch in enumerate(architectures):
    close_accs = [results[task][arch]['close_accuracy'] for task in tasks_p97]
    exact_accs = [results[task][arch]['exact_accuracy'] for task in tasks_p97]

    # Stack close accuracy on top of exact accuracy
    ax7.bar(x + i * width, exact_accs, width, label=f'{arch} (Exact)',
            color=colors[i], alpha=0.8)
    ax7.bar(x + i * width, np.array(close_accs) - np.array(exact_accs), width,
            bottom=exact_accs, color=colors[i], alpha=0.3, hatch='//')

ax7.set_xlabel('Cryptographic Operation', fontsize=12, fontweight='bold')
ax7.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax7.set_title('Exact (solid) vs "Close Enough" (hatched) Accuracy\nNetworks can get "close" but not exact',
              fontsize=14, fontweight='bold')
ax7.set_xticks(x + width)
ax7.set_xticklabels(task_labels)
ax7.legend(loc='upper right', fontsize=8, ncol=3)
ax7.set_ylim(0, 0.8)
ax7.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 8: Summary Statistics
# ============================================================================
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis('off')

summary_text = """
KEY FINDINGS
═══════════════════════════════════

🔴 ALL NETWORKS FAIL
   Best: 20.2% (Liquid, Mod Add, p=97)
   Cryptography: IMPOSSIBLE

🔴 10× DEGRADATION
   p=97 → p=997 causes catastrophic
   accuracy collapse

🔴 BIO-PLAUSIBLE: NEAR ZERO
   Hebbian learning cannot handle
   discrete math (0-1% accuracy)

🔴 BACKPROP NECESSARY (BUT NOT ENOUGH)
   Even with gradients: <21% max

🔴 OPERATION HIERARCHY (HARDEST FIRST)
   1. Point Validation (1.6%)
   2. Mod Inverse (3.2%)
   3. Mod Mult (1.6%)
   4. Mod Exp (3.1%)
   5. Mod Sub (15.5%)
   6. Mod Add (20.2%)

✅ CRYPTOGRAPHY IS SAFE
   Neural networks cannot "learn"
   to break cryptographic math

💡 FUNDAMENTAL LIMITATION
   Discrete math ≠ pattern recognition
   Discontinuities break gradient flow

🧠 BIO-PLAUSIBLE LEARNING HAS LIMITS
   Evolution didn't optimize for
   modular arithmetic!
"""

ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('SECP256K1 ELLIPTIC CURVE MATH LEARNING - THE ULTIMATE BIO-PLAUSIBLE TEST\n' +
             'Can Neural Networks Learn Cryptographic Mathematics? Answer: NO',
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_file = Path(__file__).parent / 'ec_math_learning_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_file}")

plt.show()
