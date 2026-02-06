#!/usr/bin/env python3
"""Quick test of Forward-Forward."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the final version
from forward_forward_arithmetic_final import train_optimized_ff

if __name__ == '__main__':
    print("Running quick Forward-Forward test on p=7...")
    print("="*70)

    # Quick test with fewer epochs
    accuracy = train_optimized_ff(
        p=7,
        epochs=500,
        batch_size=64,
        lr=0.6,
        threshold=3.5,
        hidden_dims=[256, 256, 128]
    )

    print(f"\n{'='*70}")
    print(f"QUICK TEST COMPLETE")
    print(f"Final accuracy: {accuracy*100:.2f}%")
    print(f"Target: 90%")
    print(f"Status: {'✓ PASS' if accuracy >= 0.90 else '✗ FAIL'}")
    print(f"{'='*70}")
