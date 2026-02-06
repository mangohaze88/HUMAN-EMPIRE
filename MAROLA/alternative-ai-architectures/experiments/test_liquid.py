#!/usr/bin/env python3
"""
Quick test script for Liquid Neural Network integration
"""
import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

# Import from current directory
from compare_all import run_liquid

def main():
    print("=" * 60)
    print("TESTING LIQUID NEURAL NETWORK INTEGRATION")
    print("=" * 60)

    # Test CPU version
    print("\n1. Testing CPU implementation...")
    results_cpu = run_liquid(n_steps=100, use_gpu=False)

    print(f"\nCPU Results:")
    print(f"  - Final error: {results_cpu.final_error:.4f}")
    print(f"  - Speed: {results_cpu.steps_per_second:.1f} steps/sec")
    print(f"  - Adaptation rate: {results_cpu.extra_metrics['final_adaptation_rate']:.6f}")
    print(f"  - Time constant: {results_cpu.extra_metrics['mean_time_constant']:.3f}")
    print(f"  - Stability: {results_cpu.extra_metrics['stability_metric']:.3f}")

    # Test GPU version if available
    import torch
    if torch.cuda.is_available():
        print("\n2. Testing GPU implementation...")
        results_gpu = run_liquid(n_steps=100, use_gpu=True)

        print(f"\nGPU Results:")
        print(f"  - Final error: {results_gpu.final_error:.4f}")
        print(f"  - Speed: {results_gpu.steps_per_second:.1f} steps/sec")
        print(f"  - Adaptation rate: {results_gpu.extra_metrics['final_adaptation_rate']:.6f}")
        print(f"  - Time constant: {results_gpu.extra_metrics['mean_time_constant']:.3f}")
        print(f"  - Stability: {results_gpu.extra_metrics['stability_metric']:.3f}")

        speedup = results_gpu.steps_per_second / results_cpu.steps_per_second
        print(f"\n  GPU Speedup: {speedup:.2f}x")
    else:
        print("\n2. GPU not available - skipping GPU test")

    print("\n" + "=" * 60)
    print("✓ LIQUID NEURAL NETWORK INTEGRATION SUCCESSFUL")
    print("=" * 60)

if __name__ == "__main__":
    main()
