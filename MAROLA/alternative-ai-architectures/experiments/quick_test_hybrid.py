#!/usr/bin/env python3
"""
Quick Test: Hybrid Liquid-Stigmergic Arithmetic
================================================

Fast test on small primes to verify the architecture works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hybrid_liquid_stigmergic_arithmetic import (
    train_hybrid, test_hybrid, benchmark_all_approaches
)
import json


def quick_test():
    """Run quick test on small primes"""

    print("\n" + "="*70)
    print("QUICK TEST: Hybrid Liquid-Stigmergic Arithmetic")
    print("="*70)
    print("\nTesting on mod 7 and mod 11 (fast)")
    print("="*70 + "\n")

    primes = [7, 11]
    all_results = {}

    for p in primes:
        print(f"\n{'='*70}")
        print(f"Testing mod {p}")
        print(f"{'='*70}\n")

        results = benchmark_all_approaches(p, n_train=2000, n_test=500)
        all_results[f'p{p}'] = results

        print(f"\n{'='*70}")
        print(f"RESULTS: mod {p}")
        print(f"{'='*70}")
        print(f"Hybrid:         train={results['hybrid']['train_accuracy']:.3f}, "
              f"test={results['hybrid']['test_accuracy']:.3f}")
        print(f"Pure LNN:       train={results['pure_lnn']['train_accuracy']:.3f}, "
              f"test={results['pure_lnn']['test_accuracy']:.3f}")
        print(f"Pure Stigmergic: train={results['pure_stigmergic']['train_accuracy']:.3f}, "
              f"test={results['pure_stigmergic']['test_accuracy']:.3f}")

        # Analyze results
        hybrid_acc = results['hybrid']['test_accuracy']
        lnn_acc = results['pure_lnn']['test_accuracy']
        stig_acc = results['pure_stigmergic']['test_accuracy']

        print(f"\n{'='*70}")
        print("ANALYSIS:")
        print(f"{'='*70}")

        if hybrid_acc > 0.7 and lnn_acc < 0.2 and stig_acc < 0.2:
            print(f"✓ SUCCESS! Hybrid achieves {hybrid_acc:.1%} where individuals fail!")
        elif hybrid_acc > max(lnn_acc, stig_acc) * 1.5:
            print(f"→ PROMISING! Hybrid {hybrid_acc:.1%} > individual approaches")
        else:
            print(f"✗ NEEDS WORK: Hybrid not significantly better")
            print(f"  Hybrid improvement: {(hybrid_acc - max(lnn_acc, stig_acc)):.3f}")

        # Check for emergent behavior
        best_pure = max(lnn_acc, stig_acc)
        synergy = hybrid_acc - best_pure

        if synergy > 0.1:
            print(f"\n✓ EMERGENT BEHAVIOR DETECTED!")
            print(f"  Synergy score: {synergy:.3f}")
            print(f"  This suggests true hybrid intelligence!")
        elif synergy > 0:
            print(f"\n→ Weak synergy: {synergy:.3f}")
            print(f"  Hybrid has slight advantage")
        else:
            print(f"\n✗ No synergy: {synergy:.3f}")
            print(f"  Hybrid doesn't exceed components")

    # Save results
    output_file = 'quick_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")

    # Summary table
    print("\nQUICK TEST SUMMARY")
    print("="*70)
    print(f"{'Prime':<10} {'Hybrid':<15} {'Pure LNN':<15} {'Pure Stigmergic':<15}")
    print("-"*70)
    for p in primes:
        r = all_results[f'p{p}']
        print(f"mod {p:<6} {r['hybrid']['test_accuracy']:>6.1%}        "
              f"{r['pure_lnn']['test_accuracy']:>6.1%}         "
              f"{r['pure_stigmergic']['test_accuracy']:>6.1%}")
    print("="*70)

    return all_results


if __name__ == '__main__':
    results = quick_test()
