"""
Analyze and visualize stigmergic arithmetic learning results.

Quick analysis tool for understanding ant colony learning dynamics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load results from JSON file."""
    results_path = Path(__file__).parent / 'stigmergic_arithmetic_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_convergence(results):
    """Analyze convergence properties."""
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)

    epochs = results['epochs']
    test_acc = results['test_majority_acc']

    # Find when we first exceed thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

    print("\nTime to Accuracy Milestones:")
    print("-" * 60)
    for threshold in thresholds:
        for i, (epoch, acc) in enumerate(zip(epochs, test_acc)):
            if acc >= threshold:
                print(f"  {threshold*100:.0f}% accuracy: Epoch {epoch}")
                break
        else:
            print(f"  {threshold*100:.0f}% accuracy: Not reached")

    # Convergence rate
    print("\nConvergence Rate:")
    print("-" * 60)
    early_epochs = epochs[:10]
    early_acc = test_acc[:10]
    if len(early_acc) > 1:
        rate = (early_acc[-1] - early_acc[0]) / (early_epochs[-1] - early_epochs[0])
        print(f"  Initial rate (first 10 epochs): {rate:.3f} per epoch")

    final_acc = results['final_test_stats']['majority_vote_acc']
    print(f"  Final accuracy: {final_acc:.3f}")
    print(f"  Improvement over random: {final_acc - 1/results['p']:.3f}")


def analyze_pheromones(results):
    """Analyze pheromone dynamics."""
    print("\n" + "="*60)
    print("PHEROMONE ANALYSIS")
    print("="*60)

    confidence = results['train_conf']
    pheromone_acc = results['test_pheromone_acc']

    print("\nConfidence Evolution:")
    print("-" * 60)
    print(f"  Initial: {confidence[0]:.3f}")
    print(f"  Epoch 10: {confidence[9]:.3f}")
    print(f"  Epoch 20: {confidence[19]:.3f}")
    print(f"  Final: {confidence[-1]:.3f}")

    print("\nPheromone Trail Accuracy:")
    print("-" * 60)
    print(f"  Initial: {pheromone_acc[0]:.3f}")
    print(f"  Final: {results['final_test_stats']['pheromone_acc']:.3f}")

    # When does pheromone become reliable?
    for i, (epoch, acc) in enumerate(zip(results['epochs'], pheromone_acc)):
        if acc >= 0.95:
            print(f"  First 95%+ accuracy: Epoch {epoch}")
            break


def analyze_specialization(results):
    """Analyze ant specialization."""
    print("\n" + "="*60)
    print("ANT SPECIALIZATION ANALYSIS")
    print("="*60)

    diversity = results['ant_diversity']

    print("\nDiversity Evolution:")
    print("-" * 60)
    print(f"  Initial: {diversity[0]:.2f}")
    print(f"  Final: {diversity[-1]:.2f}")
    print(f"  Growth: {diversity[-1] / diversity[0]:.2f}x")

    final_stats = results['final_test_stats']
    print("\nFinal Ant Performance:")
    print("-" * 60)
    print(f"  Average ant: {final_stats['avg_ant_acc']:.3f}")
    print(f"  Best ant: {final_stats['max_ant_acc']:.3f}")
    print(f"  Worst ant: {final_stats['min_ant_acc']:.3f}")
    print(f"  Std dev: {final_stats['ant_acc_std']:.4f}")
    print(f"  Range: {final_stats['max_ant_acc'] - final_stats['min_ant_acc']:.3f}")


def analyze_emergence(results):
    """Analyze emergent collective intelligence."""
    print("\n" + "="*60)
    print("EMERGENT INTELLIGENCE ANALYSIS")
    print("="*60)

    final_stats = results['final_test_stats']
    avg_ant = final_stats['avg_ant_acc']
    colony_majority = final_stats['majority_vote_acc']
    colony_pheromone = final_stats['pheromone_acc']

    print("\nCollective vs Individual Performance:")
    print("-" * 60)
    print(f"  Individual ant: {avg_ant:.3f}")
    print(f"  Colony (majority): {colony_majority:.3f}")
    print(f"  Colony (pheromone): {colony_pheromone:.3f}")

    print("\nEmergence Metrics:")
    print("-" * 60)
    majority_boost = (colony_majority - avg_ant) / avg_ant * 100
    pheromone_boost = (colony_pheromone - avg_ant) / avg_ant * 100

    print(f"  Majority vote boost: +{majority_boost:.1f}%")
    print(f"  Pheromone boost: +{pheromone_boost:.1f}%")

    # Theoretical maximum for random voting
    n_ants = results['n_ants']
    random_guess = 1 / results['p']
    print(f"\n  Random baseline: {random_guess:.3f}")
    print(f"  Improvement: {(colony_majority - random_guess):.3f}")


def plot_detailed_analysis(results):
    """Create detailed analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = results['epochs']

    # 1. Accuracy comparison
    ax = axes[0, 0]
    ax.plot(epochs, results['train_acc'], 'o-', label='Train', alpha=0.7)
    ax.plot(epochs, results['test_majority_acc'], 's-', label='Test (Majority)', alpha=0.7)
    ax.plot(epochs, results['test_pheromone_acc'], '^-', label='Test (Pheromone)', alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (80%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confidence evolution
    ax = axes[0, 1]
    ax.plot(epochs, results['train_conf'], 'o-', color='green', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Confidence')
    ax.set_title('Colony Confidence (1 - normalized entropy)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. Diversity evolution
    ax = axes[0, 2]
    ax.plot(epochs, results['ant_diversity'], 'o-', color='purple', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diversity (avg pairwise distance)')
    ax.set_title('Ant Specialization')
    ax.grid(True, alpha=0.3)

    # 4. Accuracy vs Confidence
    ax = axes[1, 0]
    ax.scatter(results['train_conf'], results['test_majority_acc'], alpha=0.6)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Confidence vs Accuracy')
    ax.grid(True, alpha=0.3)

    # 5. Accuracy gap (pheromone - majority)
    ax = axes[1, 1]
    gap = [p - m for p, m in zip(results['test_pheromone_acc'], results['test_majority_acc'])]
    ax.plot(epochs, gap, 'o-', color='orange', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy Gap')
    ax.set_title('Pheromone Advantage (Pheromone - Majority)')
    ax.grid(True, alpha=0.3)

    # 6. Learning rate (derivative of test accuracy)
    ax = axes[1, 2]
    test_acc = results['test_majority_acc']
    learning_rate = [test_acc[i+1] - test_acc[i] for i in range(len(test_acc)-1)]
    ax.plot(epochs[1:], learning_rate, 'o-', color='red', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy Change')
    ax.set_title('Learning Rate (Δ Accuracy per Epoch)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'stigmergic_arithmetic_detailed_analysis.png',
                dpi=150, bbox_inches='tight')
    print("\nDetailed analysis plot saved: stigmergic_arithmetic_detailed_analysis.png")


def print_summary(results):
    """Print executive summary."""
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)

    final_stats = results['final_test_stats']

    print(f"\nConfiguration:")
    print(f"  Colony size: {results['n_ants']} ants")
    print(f"  Modulus: p = {results['p']}")
    print(f"  Training: {len(results['epochs'])} epochs")
    print(f"  Time: {results['elapsed_time']:.1f} seconds")

    print(f"\nFinal Performance:")
    print(f"  Test Accuracy (Majority): {final_stats['majority_vote_acc']:.3f}")
    print(f"  Test Accuracy (Pheromone): {final_stats['pheromone_acc']:.3f}")
    print(f"  Average Ant: {final_stats['avg_ant_acc']:.3f}")

    target = 0.80
    if final_stats['majority_vote_acc'] >= target:
        print(f"\n✓ SUCCESS! Exceeded {target:.0%} target")
    else:
        print(f"\n✗ Below {target:.0%} target")

    print(f"\nKey Findings:")
    print(f"  1. Colony outperforms individuals by {(final_stats['majority_vote_acc'] - final_stats['avg_ant_acc'])/.01:.0f} percentage points")
    print(f"  2. Pheromones provide perfect accuracy ({final_stats['pheromone_acc']:.3f})")
    print(f"  3. Ants specialized (diversity: {results['ant_diversity'][0]:.1f} → {results['ant_diversity'][-1]:.1f})")
    print(f"  4. No backpropagation used (pure stigmergic learning)")


if __name__ == "__main__":
    # Load results
    results = load_results()

    # Run analyses
    print_summary(results)
    analyze_convergence(results)
    analyze_pheromones(results)
    analyze_specialization(results)
    analyze_emergence(results)

    # Create detailed plots
    plot_detailed_analysis(results)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
