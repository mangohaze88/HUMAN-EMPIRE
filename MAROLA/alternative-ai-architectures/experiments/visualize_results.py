#!/usr/bin/env python3
"""
Visualize MNIST Benchmark Results
==================================
Create a simple ASCII visualization of the benchmark results.
"""

import json
import os

def load_results(filename='mnist_benchmark_results.json'):
    """Load results from JSON file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def create_bar_chart(results):
    """Create ASCII bar chart of test accuracy"""
    print("\n" + "="*80)
    print("TEST ACCURACY COMPARISON (MNIST Digit Classification)")
    print("="*80)
    print()

    max_acc = max(r['test_acc'] for r in results)
    bar_width = 60

    for r in results:
        name = r['name']
        acc = r['test_acc']
        backprop = "BP" if r['uses_backprop'] else "NO-BP"

        # Calculate bar length
        bar_len = int((acc / 100.0) * bar_width)
        bar = "█" * bar_len

        # Color coding
        if acc > 90:
            color = "🟢"
        elif acc > 50:
            color = "🟡"
        elif acc > 20:
            color = "🟠"
        else:
            color = "🔴"

        print(f"{color} {name:<20} [{backprop:>5}] {bar:<{bar_width}} {acc:>5.1f}%")

    print()
    print("-" * 80)
    print("Legend: 🟢 Excellent (>90%)  🟡 Good (>50%)  🟠 Limited (>20%)  🔴 Poor (<20%)")
    print("        BP = Uses Backpropagation, NO-BP = Bio-Plausible (No Backprop)")
    print()

    # Random chance line
    random_chance = 10.0
    random_bar_len = int((random_chance / 100.0) * bar_width)
    random_bar = "·" * random_bar_len
    print(f"   {'Random Chance':<20} {'     '} {random_bar:<{bar_width}} {random_chance:>5.1f}%")
    print()


def create_comparison_table(results):
    """Create comparison table"""
    print("="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    print()
    print(f"{'Metric':<25} {'MLP (Backprop)':<20} {'Best Bio-Plausible':<20}")
    print("-" * 80)

    baseline = [r for r in results if r['uses_backprop']][0]
    bio_plausible = [r for r in results if not r['uses_backprop']]
    best_bio = max(bio_plausible, key=lambda x: x['test_acc']) if bio_plausible else None

    if best_bio:
        print(f"{'Architecture':<25} {baseline['name']:<20} {best_bio['name']:<20}")
        print(f"{'Test Accuracy':<25} {baseline['test_acc']:>6.1f}%{'':13} {best_bio['test_acc']:>6.1f}%")
        print(f"{'Training Time':<25} {baseline['time']:>6.1f}s{'':13} {best_bio['time']:>6.1f}s")
        print(f"{'Parameters':<25} {baseline['params']:>10,}{'':9} {best_bio['params']:>10,}")
        print(f"{'Speed (samples/sec)':<25} {int(5000/baseline['time']):>6,}{'':13} {int(5000/best_bio['time']):>6,}")

        gap = baseline['test_acc'] - best_bio['test_acc']
        slowdown = best_bio['time'] / baseline['time']

        print()
        print("-" * 80)
        print(f"{'Performance Gap':<25} {gap:>6.1f} percentage points")
        print(f"{'Speed Ratio':<25} {slowdown:>6.1f}x slower")
        print(f"{'Parameters Ratio':<25} {best_bio['params']/baseline['params']:>6.2f}x")

    print()


def create_learning_analysis(results):
    """Analyze whether networks learned"""
    print("="*80)
    print("LEARNING ANALYSIS")
    print("="*80)
    print()

    random_chance = 10.0
    learning_threshold = 20.0

    bio_results = [r for r in results if not r['uses_backprop']]

    print(f"Random Chance: {random_chance}%")
    print(f"Learning Threshold (2x random): {learning_threshold}%")
    print()

    for r in bio_results:
        name = r['name']
        acc = r['test_acc']

        if acc > learning_threshold:
            status = "✓ LEARNED"
            analysis = "Significantly above random chance"
        elif acc > random_chance * 1.5:
            status = "~ PARTIAL"
            analysis = "Shows some learning signal"
        elif acc > random_chance * 0.5:
            status = "✗ RANDOM"
            analysis = "At random chance level"
        else:
            status = "✗ BROKEN"
            analysis = "Below random chance (possible implementation issue)"

        print(f"{status:<15} {name:<20} {acc:>5.1f}%  - {analysis}")

    print()
    print("-" * 80)
    print()

    learned = [r for r in bio_results if r['test_acc'] > learning_threshold]
    if learned:
        print(f"✓ {len(learned)}/{len(bio_results)} bio-plausible networks show learning")
    else:
        print(f"✗ 0/{len(bio_results)} bio-plausible networks learned above threshold")
        print("  Possible reasons:")
        print("  1. Hyperparameters not tuned for this task")
        print("  2. These architectures better suited for different tasks")
        print("  3. Need more training epochs or different learning signals")

    print()


def main():
    """Main visualization"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MNIST BENCHMARK RESULTS VISUALIZATION" + " "*26 + "║")
    print("║" + " "*20 + "Bio-Plausible vs Backpropagation" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    print()

    results = load_results()

    create_bar_chart(results)
    create_comparison_table(results)
    create_learning_analysis(results)

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The benchmark demonstrates:")
    print()
    print("1. ✓ Bio-plausible networks CAN run without backpropagation")
    print("2. ✗ Current implementations don't learn MNIST effectively")
    print("3. ⚠  Performance gap of 80+ percentage points shows room for improvement")
    print("4. 💡 These networks may excel at different tasks (temporal, continual, etc.)")
    print()
    print("Next steps: Tune hyperparameters, fix technical issues, try better-suited tasks")
    print()
    print("="*80)
    print()


if __name__ == '__main__':
    main()
