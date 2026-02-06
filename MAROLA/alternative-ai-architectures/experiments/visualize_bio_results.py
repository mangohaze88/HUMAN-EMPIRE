#!/usr/bin/env python3
"""
Visualize bio-plausible modular arithmetic results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Results
results = {
    'methods': ['Random\nBaseline', 'Forward-Forward\n(no backprop)', 'Liquid Network\n(no backprop)', 'Standard NN\n(with backprop)'],
    'accuracies': [14.3, 14.3, 16.3, 100.0],
    'colors': ['gray', 'orange', 'cyan', 'green'],
    'bio_plausible': [False, True, True, False]
}

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Accuracy comparison
bars = ax1.bar(results['methods'], results['accuracies'], color=results['colors'], alpha=0.7, edgecolor='black', linewidth=2)

# Add bio-plausible markers
for i, (bar, bio) in enumerate(zip(bars, results['bio_plausible'])):
    if bio:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                '✓ Bio-Plausible', ha='center', fontsize=9, fontweight='bold', color='darkgreen')

ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Target: 50%')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Modular Arithmetic (p=7): Bio-Plausible vs Backprop', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Method characteristics
characteristics = {
    'Methods': ['Forward-Forward', 'Liquid Network', 'Standard NN'],
    'Accuracy': [14.3, 16.3, 100.0],
    'Bio-Plausible': [10, 10, 0],  # 10 = yes, 0 = no
    'Training Time': [7.2, 23.4, 5.0],
}

x = np.arange(len(characteristics['Methods']))
width = 0.25

ax2.bar(x - width, characteristics['Accuracy'], width, label='Accuracy (%)', color='green', alpha=0.7)
ax2.bar(x, [b*10 for b in [1, 1, 0]], width, label='Bio-Plausible (×10)', color='blue', alpha=0.7)
ax2.bar(x + width, characteristics['Training Time'], width, label='Training Time (s)', color='red', alpha=0.7)

ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
ax2.set_title('Method Characteristics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(characteristics['Methods'])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(__file__), 'bio_plausible_results_visualization.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_file}")

# Create summary table figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Table data
table_data = [
    ['Method', 'Accuracy', 'Training Time', 'Bio-Plausible', 'Backprop', 'Key Feature'],
    ['Random Baseline', '14.3%', '-', 'N/A', 'N/A', 'Chance level'],
    ['Forward-Forward', '14.3%', '7.2s', '✓ YES', '✗ NO', 'Local contrastive learning'],
    ['Liquid Network', '16.3%', '23.4s', '✓ YES', '✗ NO', 'ODE + Hebbian learning'],
    ['Standard NN', '100%', '~5s', '✗ NO', '✓ YES', 'Global error propagation'],
]

# Colors for rows
row_colors = [['lightgray']*6,  # Header
              ['white']*6,        # Random
              ['lightyellow']*6,  # FF
              ['lightcyan']*6,    # LNN
              ['lightgreen']*6]   # Standard

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.12, 0.15, 0.15, 0.12, 0.26],
                cellColours=row_colors)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Bold header
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold')
    cell.set_facecolor('darkgray')

ax.set_title('Bio-Plausible Modular Arithmetic Learning - Complete Results (p=7)',
            fontsize=14, fontweight='bold', pad=20)

# Save table
table_file = os.path.join(os.path.dirname(__file__), 'bio_plausible_results_table.png')
plt.savefig(table_file, dpi=300, bbox_inches='tight')
print(f"Table saved to: {table_file}")

print("\nKey Findings:")
print("1. Bio-plausible methods achieve ~15% accuracy (barely above random)")
print("2. Standard NN with backprop achieves 100% accuracy")
print("3. Trade-off: Biological plausibility vs. performance on symbolic tasks")
print("4. Forward-Forward and Liquid Networks excel on other tasks (vision, time-series)")

plt.show()
