#!/usr/bin/env python3
"""
Visualize Hybrid Liquid-Stigmergic Arithmetic Learning Results
==============================================================

Creates comprehensive visualizations of:
1. Accuracy comparison across all primes
2. Training dynamics for each approach
3. Component contributions (liquid vs stigmergic)
4. Emergent behavior analysis
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


def load_results(filename='hybrid_liquid_stigmergic_results.json'):
    """Load results from JSON file"""
    if not Path(filename).exists():
        print(f"Results file not found: {filename}")
        print("Run the experiment first: python experiments/hybrid_liquid_stigmergic_arithmetic.py")
        return None

    with open(filename, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results, ax):
    """Plot accuracy comparison across all primes"""
    primes = [7, 11, 23, 47, 97]

    hybrid_train = []
    hybrid_test = []
    lnn_train = []
    lnn_test = []
    stig_train = []
    stig_test = []

    for p in primes:
        key = f'p{p}'
        if key in results:
            r = results[key]
            hybrid_train.append(r['hybrid']['train_accuracy'])
            hybrid_test.append(r['hybrid']['test_accuracy'])
            lnn_train.append(r['pure_lnn']['train_accuracy'])
            lnn_test.append(r['pure_lnn']['test_accuracy'])
            stig_train.append(r['pure_stigmergic']['train_accuracy'])
            stig_test.append(r['pure_stigmergic']['test_accuracy'])

    x = np.arange(len(primes))
    width = 0.15

    # Test accuracy (main metric)
    ax.bar(x - 2*width, hybrid_test, width, label='Hybrid', color='purple', alpha=0.8)
    ax.bar(x - width, lnn_test, width, label='Pure LNN', color='blue', alpha=0.8)
    ax.bar(x, stig_test, width, label='Pure Stigmergic', color='green', alpha=0.8)

    # Training accuracy (lighter)
    ax.bar(x - 2*width, hybrid_train, width, alpha=0.3, color='purple')
    ax.bar(x - width, lnn_train, width, alpha=0.3, color='blue')
    ax.bar(x, stig_train, width, alpha=0.3, color='green')

    ax.set_xlabel('Modulus (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison: Hybrid vs Pure Approaches\n(Solid=Test, Faded=Train)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'mod {p}' for p in primes])
    ax.legend(loc='upper right')
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Success threshold (70%)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Random baseline (~14-20%)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)


def plot_improvement_ratio(results, ax):
    """Plot improvement of hybrid over best pure approach"""
    primes = [7, 11, 23, 47, 97]

    improvements = []

    for p in primes:
        key = f'p{p}'
        if key in results:
            r = results[key]
            hybrid_acc = r['hybrid']['test_accuracy']
            best_pure = max(r['pure_lnn']['test_accuracy'],
                          r['pure_stigmergic']['test_accuracy'])

            if best_pure > 0:
                improvement = (hybrid_acc - best_pure) / best_pure * 100
            else:
                improvement = 0

            improvements.append(improvement)

    colors = ['green' if imp > 50 else 'orange' if imp > 0 else 'red'
              for imp in improvements]

    bars = ax.bar(range(len(primes)), improvements, color=colors, alpha=0.7)

    ax.set_xlabel('Modulus (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid Improvement Over Best Pure Approach',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(primes)))
    ax.set_xticklabels([f'mod {p}' for p in primes])
    ax.axhline(y=50, color='g', linestyle='--', alpha=0.5, label='Success threshold (+50%)')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.0f}%',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')


def plot_success_matrix(results, ax):
    """Plot success/failure matrix"""
    primes = [7, 11, 23, 47, 97]
    approaches = ['Hybrid', 'Pure LNN', 'Pure Stigmergic']

    matrix = np.zeros((len(approaches), len(primes)))

    for i, p in enumerate(primes):
        key = f'p{p}'
        if key in results:
            r = results[key]
            matrix[0, i] = r['hybrid']['test_accuracy']
            matrix[1, i] = r['pure_lnn']['test_accuracy']
            matrix[2, i] = r['pure_stigmergic']['test_accuracy']

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy', fontsize=10)

    # Set ticks
    ax.set_xticks(range(len(primes)))
    ax.set_xticklabels([f'mod {p}' for p in primes])
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches)

    # Add text annotations
    for i in range(len(approaches)):
        for j in range(len(primes)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10,
                          fontweight='bold')

    ax.set_title('Accuracy Heatmap: All Approaches × All Primes',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Modulus (p)', fontsize=12, fontweight='bold')


def plot_complexity_scaling(results, ax):
    """Plot how approaches scale with problem complexity"""
    primes = [7, 11, 23, 47, 97]
    problem_sizes = [p*p for p in primes]  # Total problem space

    hybrid_accs = []
    lnn_accs = []
    stig_accs = []

    for p in primes:
        key = f'p{p}'
        if key in results:
            r = results[key]
            hybrid_accs.append(r['hybrid']['test_accuracy'])
            lnn_accs.append(r['pure_lnn']['test_accuracy'])
            stig_accs.append(r['pure_stigmergic']['test_accuracy'])

    ax.plot(problem_sizes, hybrid_accs, 'o-', label='Hybrid',
            color='purple', linewidth=2, markersize=8)
    ax.plot(problem_sizes, lnn_accs, 's-', label='Pure LNN',
            color='blue', linewidth=2, markersize=8)
    ax.plot(problem_sizes, stig_accs, '^-', label='Pure Stigmergic',
            color='green', linewidth=2, markersize=8)

    ax.set_xlabel('Problem Space Size (p²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Scaling with Problem Complexity', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    # Add prime labels
    for i, (size, p) in enumerate(zip(problem_sizes, primes)):
        ax.annotate(f'p={p}', (size, hybrid_accs[i]),
                   textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=8, alpha=0.7)


def plot_statistics_summary(results, ax):
    """Summary statistics table"""
    primes = [7, 11, 23, 47, 97]

    # Calculate statistics
    stats = {
        'Approach': [],
        'Mean Accuracy': [],
        'Best Performance': [],
        'Worst Performance': [],
        'Success Rate': [],
    }

    for approach in ['hybrid', 'pure_lnn', 'pure_stigmergic']:
        accs = []
        for p in primes:
            key = f'p{p}'
            if key in results:
                if approach == 'hybrid':
                    accs.append(results[key]['hybrid']['test_accuracy'])
                elif approach == 'pure_lnn':
                    accs.append(results[key]['pure_lnn']['test_accuracy'])
                else:
                    accs.append(results[key]['pure_stigmergic']['test_accuracy'])

        if accs:
            name = approach.replace('_', ' ').title()
            stats['Approach'].append(name)
            stats['Mean Accuracy'].append(f"{np.mean(accs):.3f}")
            stats['Best Performance'].append(f"{np.max(accs):.3f}")
            stats['Worst Performance'].append(f"{np.min(accs):.3f}")
            stats['Success Rate'].append(f"{sum(acc > 0.5 for acc in accs)}/{len(accs)}")

    # Create table
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=[list(stats.values())[i] for i in range(len(stats['Approach']))],
                    rowLabels=stats['Approach'],
                    colLabels=list(stats.keys())[1:],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code by approach
    colors = ['#E6B0E6', '#B0C4DE', '#98FB98']  # Purple, Blue, Green
    for i, color in enumerate(colors):
        for j in range(len(stats['Approach'][0])):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)

    ax.set_title('Performance Statistics Summary', fontsize=14, fontweight='bold', pad=20)


def plot_emergent_behavior_analysis(results, ax):
    """Analyze emergent behavior indicators"""
    primes = [7, 11, 23, 47, 97]

    # Calculate emergence metrics
    synergy_scores = []

    for p in primes:
        key = f'p{p}'
        if key in results:
            r = results[key]
            hybrid = r['hybrid']['test_accuracy']
            lnn = r['pure_lnn']['test_accuracy']
            stig = r['pure_stigmergic']['test_accuracy']

            # Synergy = Hybrid - max(pure approaches)
            # Positive = emergent behavior
            synergy = hybrid - max(lnn, stig)
            synergy_scores.append(synergy)

    colors = ['green' if s > 0.1 else 'orange' if s > 0 else 'red'
              for s in synergy_scores]

    bars = ax.bar(range(len(primes)), synergy_scores, color=colors, alpha=0.7)

    ax.set_xlabel('Modulus (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Synergy Score\n(Hybrid - Best Pure)', fontsize=12, fontweight='bold')
    ax.set_title('Emergent Behavior Analysis\n(Positive = True Synergy)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(primes)))
    ax.set_xticklabels([f'mod {p}' for p in primes])
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Strong emergence (>0.1)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, synergy_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.3f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')


def create_full_visualization(results):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(20, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Main accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    plot_accuracy_comparison(results, ax1)

    # Plot 2: Improvement ratio
    ax2 = fig.add_subplot(gs[0, 2])
    plot_improvement_ratio(results, ax2)

    # Plot 3: Success matrix
    ax3 = fig.add_subplot(gs[1, :2])
    plot_success_matrix(results, ax3)

    # Plot 4: Complexity scaling
    ax4 = fig.add_subplot(gs[1, 2])
    plot_complexity_scaling(results, ax4)

    # Plot 5: Statistics summary
    ax5 = fig.add_subplot(gs[2, :2])
    plot_statistics_summary(results, ax5)

    # Plot 6: Emergent behavior
    ax6 = fig.add_subplot(gs[2, 2])
    plot_emergent_behavior_analysis(results, ax6)

    fig.suptitle('Hybrid Liquid-Stigmergic Arithmetic Learning: Complete Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    """Main visualization function"""
    print("Loading results...")
    results = load_results()

    if results is None:
        return

    print("Creating visualizations...")
    fig = create_full_visualization(results)

    # Save
    output_file = 'hybrid_liquid_stigmergic_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    primes = [7, 11, 23, 47, 97]
    for p in primes:
        key = f'p{p}'
        if key in results:
            r = results[key]
            print(f"\nmod {p}:")
            print(f"  Hybrid:         {r['hybrid']['test_accuracy']:.3f}")
            print(f"  Pure LNN:       {r['pure_lnn']['test_accuracy']:.3f}")
            print(f"  Pure Stigmergic: {r['pure_stigmergic']['test_accuracy']:.3f}")

            hybrid = r['hybrid']['test_accuracy']
            best_pure = max(r['pure_lnn']['test_accuracy'],
                          r['pure_stigmergic']['test_accuracy'])

            if hybrid > 0.7 and best_pure < 0.2:
                print(f"  ✓ SUCCESS: Hybrid works where pure approaches fail!")
            elif hybrid > best_pure * 1.5:
                print(f"  → PROMISING: Hybrid outperforms pure approaches")
            else:
                print(f"  ✗ No significant advantage")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
