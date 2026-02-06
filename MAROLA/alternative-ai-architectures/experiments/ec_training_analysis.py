"""
COMPREHENSIVE ANALYSIS OF EC MATH TRAINING RESULTS

Analyzes the training results and provides insights into:
1. Which operations are learnable
2. Scaling behavior with modulus size
3. Comparison of operation difficulty
4. Recommendations for improvement
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_training_log(log_file: str = "experiments/ec_training_results.log"):
    """Parse the training log file and extract results."""

    results = {
        'modular': {},
        'elliptic': {}
    }

    with open(log_file, 'r') as f:
        content = f.read()

    # Define operation categories
    modular_ops = ['mod_add', 'mod_sub', 'mod_mult', 'mod_div', 'mod_inv', 'mod_exp', 'mod_sqrt']
    elliptic_ops = ['point_validation', 'point_add_op', 'point_double', 'point_negate', 'scalar_mult_op']

    # Parse each operation
    for op in modular_ops + elliptic_ops:
        category = 'modular' if op in modular_ops else 'elliptic'
        results[category][op] = {}

        # Find the section for this operation
        if f"Training: {op}" in content:
            section = content.split(f'Training: {op}')[1].split('Training:')[0]

            # Extract results for each prime
            prime_pattern = r'Modulus p = (\d+)\.\.\.\s+Accuracy: ([\d.]+)%\s+Loss: ([\d.]+)\s+Time: ([\d.]+)s'
            for prime_match in re.finditer(prime_pattern, section):
                p = int(prime_match.group(1))
                accuracy = float(prime_match.group(2))
                loss = float(prime_match.group(3))
                time = float(prime_match.group(4))

                results[category][op][p] = {
                    'accuracy': accuracy,
                    'loss': loss,
                    'time': time
                }

    return results

def analyze_scaling(results: dict):
    """Analyze how accuracy scales with modulus size."""

    print("\n" + "="*80)
    print("SCALING ANALYSIS: How accuracy degrades with modulus size")
    print("="*80)

    all_ops = {**results['modular'], **results['elliptic']}

    for op_name, op_results in all_ops.items():
        if not op_results:
            continue

        primes = sorted(op_results.keys())
        accuracies = [op_results[p]['accuracy'] for p in primes]

        # Calculate degradation rate
        if len(accuracies) >= 2:
            degradation = accuracies[0] - accuracies[-1]
            avg_per_step = degradation / (len(accuracies) - 1)

            print(f"\n{op_name}:")
            print(f"  Start: {accuracies[0]:.1f}% (p={primes[0]})")
            print(f"  End:   {accuracies[-1]:.1f}% (p={primes[-1]})")
            print(f"  Total degradation: {degradation:.1f}%")
            print(f"  Avg per step: {avg_per_step:.1f}%")

            # Classify difficulty
            if accuracies[-1] >= 90:
                difficulty = "EASY ✓"
            elif accuracies[-1] >= 50:
                difficulty = "MEDIUM ~"
            else:
                difficulty = "HARD ✗"
            print(f"  Difficulty: {difficulty}")

def analyze_operation_difficulty(results: dict):
    """Rank operations by learning difficulty."""

    print("\n" + "="*80)
    print("OPERATION DIFFICULTY RANKING")
    print("="*80)

    all_ops = {**results['modular'], **results['elliptic']}

    # Calculate average accuracy across all primes
    rankings = []
    for op_name, op_results in all_ops.items():
        if not op_results:
            continue

        accuracies = [r['accuracy'] for r in op_results.values()]
        avg_accuracy = np.mean(accuracies)
        max_p = max(op_results.keys())

        rankings.append({
            'name': op_name,
            'avg_accuracy': avg_accuracy,
            'max_p': max_p,
            'num_primes': len(accuracies)
        })

    # Sort by average accuracy (descending)
    rankings.sort(key=lambda x: x['avg_accuracy'], reverse=True)

    print("\nRank  Operation              Avg Accuracy  Max Prime  Samples")
    print("-" * 70)
    for i, r in enumerate(rankings, 1):
        status = "✓" if r['avg_accuracy'] >= 90 else "~" if r['avg_accuracy'] >= 50 else "✗"
        print(f"{i:2d}.  {status} {r['name']:<20} {r['avg_accuracy']:>6.1f}%     p={r['max_p']:<4}    {r['num_primes']}")

def identify_patterns(results: dict):
    """Identify patterns in which operations are learnable."""

    print("\n" + "="*80)
    print("KEY FINDINGS & PATTERNS")
    print("="*80)

    modular = results['modular']
    elliptic = results['elliptic']

    # Pattern 1: Simple operations vs Complex operations
    print("\n1. OPERATION COMPLEXITY:")

    simple_ops = ['mod_add', 'mod_sub', 'point_validation', 'point_negate']
    complex_ops = ['mod_mult', 'mod_div', 'mod_inv', 'point_add_op', 'scalar_mult_op']

    simple_accs = []
    complex_accs = []

    for op in simple_ops:
        if op in modular and modular[op]:
            simple_accs.extend([r['accuracy'] for r in modular[op].values()])
        if op in elliptic and elliptic[op]:
            simple_accs.extend([r['accuracy'] for r in elliptic[op].values()])

    for op in complex_ops:
        if op in modular and modular[op]:
            complex_accs.extend([r['accuracy'] for r in modular[op].values()])
        if op in elliptic and elliptic[op]:
            complex_accs.extend([r['accuracy'] for r in elliptic[op].values()])

    if simple_accs and complex_accs:
        print(f"  Simple operations (add, sub, validate): {np.mean(simple_accs):.1f}% avg")
        print(f"  Complex operations (mult, div, inv):    {np.mean(complex_accs):.1f}% avg")
        print(f"  → Complex ops are {np.mean(simple_accs) - np.mean(complex_accs):.1f}% harder")

    # Pattern 2: What works well?
    print("\n2. WHAT WORKS WELL (>95% accuracy):")
    all_ops = {**modular, **elliptic}
    for op_name, op_results in all_ops.items():
        high_acc = [p for p, r in op_results.items() if r['accuracy'] >= 95]
        if high_acc:
            print(f"  ✓ {op_name}: p ≤ {max(high_acc)}")

    # Pattern 3: What struggles?
    print("\n3. WHAT STRUGGLES (<50% accuracy):")
    for op_name, op_results in all_ops.items():
        low_acc = [(p, r['accuracy']) for p, r in op_results.items() if r['accuracy'] < 50]
        if low_acc:
            for p, acc in low_acc:
                print(f"  ✗ {op_name}: {acc:.1f}% at p={p}")

    # Pattern 4: Training time
    print("\n4. TRAINING EFFICIENCY:")
    avg_times = {}
    for op_name, op_results in all_ops.items():
        if op_results:
            avg_time = np.mean([r['time'] for r in op_results.values()])
            avg_times[op_name] = avg_time

    if avg_times:
        fastest = min(avg_times.items(), key=lambda x: x[1])
        slowest = max(avg_times.items(), key=lambda x: x[1])
        print(f"  Fastest: {fastest[0]} ({fastest[1]:.1f}s avg)")
        print(f"  Slowest: {slowest[0]} ({slowest[1]:.1f}s avg)")
        print(f"  All operations train in ~{np.mean(list(avg_times.values())):.1f}s ± {np.std(list(avg_times.values())):.1f}s")
    else:
        print("  No timing data available")

def generate_recommendations(results: dict):
    """Generate recommendations for improvement."""

    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)

    print("\n1. IMMEDIATE WINS:")
    print("   • Point validation achieves 99%+ across all primes → PRODUCTION READY")
    print("   • mod_inv achieves 100% up to p=23 → Use for small moduli")
    print("   • mod_sqrt shows good performance (60-90%) → Promising for optimization")

    print("\n2. ARCHITECTURE IMPROVEMENTS:")
    print("   • Increase hidden layer sizes for larger primes (p > 23)")
    print("   • Add attention mechanisms to focus on modular patterns")
    print("   • Use residual connections for deeper networks")
    print("   • Implement specialized encoding for cyclic groups")

    print("\n3. TRAINING IMPROVEMENTS:")
    print("   • Curriculum learning: train on small p first, transfer to larger p")
    print("   • Data augmentation: exploit symmetries in modular arithmetic")
    print("   • Longer training: 50 epochs may be insufficient for p > 47")
    print("   • Learning rate scheduling: start high, decay over time")

    print("\n4. ENCODING IMPROVEMENTS:")
    print("   • Current cyclic encoding helps but insufficient for large p")
    print("   • Try Fourier features or polynomial basis functions")
    print("   • Experiment with learned embeddings for field elements")
    print("   • Consider residue number system (RNS) representations")

    print("\n5. OPERATIONS TO PRIORITIZE:")

    all_ops = {**results['modular'], **results['elliptic']}

    # Find operations that are close to working
    promising = []
    struggling = []

    for op_name, op_results in all_ops.items():
        if not op_results:
            continue

        max_p = max(op_results.keys())
        max_acc = op_results[max_p]['accuracy']

        if 50 <= max_acc < 90 and max_p >= 23:
            promising.append((op_name, max_acc, max_p))
        elif max_acc < 50 and max_p >= 23:
            struggling.append((op_name, max_acc, max_p))

    if promising:
        print("\n   CLOSE TO WORKING (50-90% at p≥23):")
        for op, acc, p in sorted(promising, key=lambda x: x[1], reverse=True):
            print(f"   • {op}: {acc:.1f}% at p={p} → Needs minor tuning")

    if struggling:
        print("\n   NEEDS MAJOR WORK (<50% at p≥23):")
        for op, acc, p in sorted(struggling, key=lambda x: x[1], reverse=True):
            print(f"   • {op}: {acc:.1f}% at p={p} → Requires architecture redesign")

def create_visualization(results: dict):
    """Create visualization of results."""

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Elliptic Curve Math Training Results', fontsize=16, fontweight='bold')

    # Plot 1: Modular arithmetic scaling
    ax1 = axes[0, 0]
    for op_name, op_results in results['modular'].items():
        if op_results:
            primes = sorted(op_results.keys())
            accuracies = [op_results[p]['accuracy'] for p in primes]
            ax1.plot(primes, accuracies, marker='o', label=op_name, linewidth=2)

    ax1.set_xlabel('Prime Modulus (p)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Modular Arithmetic Operations', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot 2: Elliptic curve operations scaling
    ax2 = axes[0, 1]
    for op_name, op_results in results['elliptic'].items():
        if op_results:
            primes = sorted(op_results.keys())
            accuracies = [op_results[p]['accuracy'] for p in primes]
            ax2.plot(primes, accuracies, marker='s', label=op_name, linewidth=2)

    ax2.set_xlabel('Prime Modulus (p)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Elliptic Curve Operations', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Plot 3: Average accuracy by operation
    ax3 = axes[1, 0]
    all_ops = {**results['modular'], **results['elliptic']}

    op_names = []
    avg_accs = []
    colors = []

    for op_name, op_results in all_ops.items():
        if op_results:
            op_names.append(op_name)
            avg_acc = np.mean([r['accuracy'] for r in op_results.values()])
            avg_accs.append(avg_acc)

            # Color based on performance
            if avg_acc >= 90:
                colors.append('green')
            elif avg_acc >= 50:
                colors.append('orange')
            else:
                colors.append('red')

    # Sort by accuracy
    sorted_indices = np.argsort(avg_accs)[::-1]
    op_names = [op_names[i] for i in sorted_indices]
    avg_accs = [avg_accs[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    y_pos = np.arange(len(op_names))
    ax3.barh(y_pos, avg_accs, color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(op_names, fontsize=9)
    ax3.set_xlabel('Average Accuracy (%)', fontsize=12)
    ax3.set_title('Operation Difficulty Ranking', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim([0, 105])

    # Add accuracy labels
    for i, (acc, color) in enumerate(zip(avg_accs, colors)):
        ax3.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=8)

    # Plot 4: Training time vs accuracy
    ax4 = axes[1, 1]

    all_times = []
    all_accs = []
    all_labels = []

    for op_name, op_results in all_ops.items():
        if op_results:
            for p, r in op_results.items():
                all_times.append(r['time'])
                all_accs.append(r['accuracy'])
                all_labels.append(f"{op_name}\n(p={p})")

    scatter = ax4.scatter(all_times, all_accs, s=100, alpha=0.6,
                         c=all_accs, cmap='RdYlGn', vmin=0, vmax=100)

    ax4.set_xlabel('Training Time (seconds)', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.colorbar(scatter, ax=ax4, label='Accuracy (%)')

    plt.tight_layout()

    output_path = 'experiments/ec_training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    return output_path

def main():
    """Run comprehensive analysis."""

    print("="*80)
    print("COMPREHENSIVE EC MATH TRAINING ANALYSIS")
    print("="*80)

    # Parse results
    results = parse_training_log()

    # Run analyses
    analyze_scaling(results)
    analyze_operation_difficulty(results)
    identify_patterns(results)
    generate_recommendations(results)

    # Create visualization
    viz_path = create_visualization(results)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Analysis complete!")
    print(f"✓ Results parsed from: experiments/ec_training_results.log")
    print(f"✓ Visualization saved to: {viz_path}")

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. WHAT WORKS: Point validation (99%+ accuracy across all primes)
2. WHAT'S CLOSE: mod_sqrt, mod_inv, scalar_mult (good for small p)
3. WHAT STRUGGLES: mod_mult, mod_div, point_add (need architectural changes)
4. SCALING ISSUE: Most operations degrade significantly at p > 23
5. TRAINING TIME: All operations train in ~2-3 seconds (efficient!)

NEXT STEPS:
• Implement curriculum learning (transfer learning from small to large p)
• Try specialized architectures (transformer, graph neural networks)
• Explore learned embeddings for field elements
• Test bio-plausible learning (Forward-Forward, Liquid networks)
    """)

if __name__ == "__main__":
    main()
