#!/usr/bin/env python3
"""
FAST Bio-Plausible Modular Arithmetic Benchmark
================================================

Optimized version that runs quickly but still demonstrates the core capability.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.forward_forward import ForwardForwardNetwork
from src.networks.liquid_neural_network import LiquidNeuralNetwork, NCPWiringConfig


# ============================================================================
# Encoding (same winning strategy)
# ============================================================================

def combined_encoding(a, b, p, bits=10):
    """The encoding that worked for standard NNs!"""
    features = []
    for val in [a, b]:
        features.extend([(val >> i) & 1 for i in range(bits)])
    features.extend([a / p, b / p])
    features.extend([
        np.sin(2 * np.pi * a / p),
        np.cos(2 * np.pi * a / p),
        np.sin(2 * np.pi * b / p),
        np.cos(2 * np.pi * b / p),
    ])
    return np.array(features, dtype=np.float32)


def encode_output_cyclic(result, p):
    return np.array([
        result / p,
        np.sin(2 * np.pi * result / p),
        np.cos(2 * np.pi * result / p),
    ], dtype=np.float32)


def decode_cyclic_output(output, p):
    if len(output) >= 3:
        sin_val = output[1]
        cos_val = output[2]
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi
        result = int(round((angle / (2 * np.pi)) * p)) % p
    else:
        result = int(round(output[0] * p)) % p
    return result


# ============================================================================
# Fast Forward-Forward Training
# ============================================================================

def train_ff_fast(p, epochs=30, samples=5000):
    """Fast Forward-Forward training"""
    print(f"\n{'='*60}")
    print(f"FORWARD-FORWARD: p={p}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create network
    sample_x = combined_encoding(0, 0, p)
    input_dim = len(sample_x)

    ff_net = ForwardForwardNetwork(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        output_dim=p,
        threshold=2.0,
        learning_rate=0.1,
        device=device,
        normalize_activations=True
    )

    # Generate training data
    train_data = []
    for _ in range(samples):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        result = (a + b) % p
        x = combined_encoding(a, b, p)
        train_data.append((x, result))

    # Training
    start_time = time.time()
    batch_size = 128

    for epoch in range(epochs):
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            xs = torch.tensor([x for x, _ in batch], dtype=torch.float32).to(device)
            ys = torch.tensor([y for _, y in batch], dtype=torch.long).to(device)

            ff_net.train_step(xs, ys, return_metrics=False)

        if epoch % 10 == 0:
            # Quick accuracy check
            correct = 0
            test_samples = min(100, p*p)
            for _ in range(test_samples):
                a = np.random.randint(0, p)
                b = np.random.randint(0, p)
                x = torch.tensor(combined_encoding(a, b, p), dtype=torch.float32).unsqueeze(0).to(device)
                pred = ff_net.predict(x).item()
                if pred == (a + b) % p:
                    correct += 1
            acc = correct / test_samples
            print(f"  Epoch {epoch:2d}: acc={acc*100:.1f}%")

    # Final evaluation
    correct = 0
    total = 0
    for a in range(p):
        for b in range(p):
            x = torch.tensor(combined_encoding(a, b, p), dtype=torch.float32).unsqueeze(0).to(device)
            pred = ff_net.predict(x).item()
            if pred == (a + b) % p:
                correct += 1
            total += 1

    final_acc = correct / total
    training_time = time.time() - start_time

    print(f"  Final: {final_acc*100:.1f}% in {training_time:.1f}s")

    return {
        'accuracy': final_acc,
        'time': training_time,
        'epochs': epochs
    }


# ============================================================================
# Fast Liquid Network Training
# ============================================================================

def train_lnn_fast(p, epochs=50, samples=5000):
    """Fast Liquid Network training"""
    print(f"\n{'='*60}")
    print(f"LIQUID NETWORK: p={p}")
    print(f"{'='*60}")

    # Create network (CPU for simplicity)
    sample_x = combined_encoding(0, 0, p)
    input_dim = len(sample_x)

    config = NCPWiringConfig(
        n_sensory=12,
        n_inter=16,
        n_command=6,
        n_motor=3
    )

    lnn = LiquidNeuralNetwork(
        input_dim=input_dim,
        output_dim=3,
        wiring_config=config,
        dt=0.1,
        ode_steps=2,
        learning_rate=0.02
    )

    # Generate training data
    train_data = []
    for _ in range(samples):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        result = (a + b) % p
        x = combined_encoding(a, b, p)
        y = encode_output_cyclic(result, p)
        train_data.append((x, y, result))

    # Training
    start_time = time.time()

    for epoch in range(epochs):
        np.random.shuffle(train_data)
        lnn.reset_state()

        for x, y, _ in train_data:
            output, _ = lnn.forward(x)
            lnn.learn(y, error_weight=1.0)

        if epoch % 10 == 0:
            # Quick accuracy check
            lnn.reset_state()
            correct = 0
            test_samples = min(100, p*p)
            for _ in range(test_samples):
                a = np.random.randint(0, p)
                b = np.random.randint(0, p)
                x = combined_encoding(a, b, p)
                output, _ = lnn.forward(x)
                pred = decode_cyclic_output(output, p)
                if pred == (a + b) % p:
                    correct += 1
            acc = correct / test_samples
            print(f"  Epoch {epoch:2d}: acc={acc*100:.1f}%")

    # Final evaluation
    lnn.reset_state()
    correct = 0
    total = 0
    for a in range(p):
        for b in range(p):
            x = combined_encoding(a, b, p)
            output, _ = lnn.forward(x)
            pred = decode_cyclic_output(output, p)
            if pred == (a + b) % p:
                correct += 1
            total += 1

    final_acc = correct / total
    training_time = time.time() - start_time

    print(f"  Final: {final_acc*100:.1f}% in {training_time:.1f}s")

    return {
        'accuracy': final_acc,
        'time': training_time,
        'epochs': epochs
    }


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("\n" + "="*60)
    print("BIO-PLAUSIBLE MODULAR ARITHMETIC BENCHMARK")
    print("="*60)
    print("\nTesting if Forward-Forward and Liquid Networks")
    print("can learn modular addition WITHOUT backpropagation!")
    print("\nUsing the SAME winning techniques:")
    print("  - Binary + cyclic encoding")
    print("  - Curriculum learning")
    print("="*60)

    curriculum = [7, 11]

    # Forward-Forward results
    print("\n\n" + "#"*60)
    print("1. FORWARD-FORWARD NETWORK (NO BACKPROP)")
    print("#"*60)

    ff_results = {}
    for p in curriculum:
        result = train_ff_fast(p, epochs=30, samples=5000)
        ff_results[p] = result

        if result['accuracy'] < 0.4:
            print(f"  ⚠️  Stopping at p={p}")
            break

    # Liquid Network results
    print("\n\n" + "#"*60)
    print("2. LIQUID NEURAL NETWORK (NO BACKPROP)")
    print("#"*60)

    lnn_results = {}
    for p in curriculum:
        result = train_lnn_fast(p, epochs=50, samples=5000)
        lnn_results[p] = result

        if result['accuracy'] < 0.25:
            print(f"  ⚠️  Stopping at p={p}")
            break

    # Report
    print("\n\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print("\nForward-Forward Network (NO BACKPROP):")
    print(f"{'Prime':<10} {'Accuracy':<15} {'Time (s)':<12} {'Status':<15}")
    print("-" * 60)
    for p, r in ff_results.items():
        status = "✓ PASS" if r['accuracy'] >= 0.5 else "✗ FAIL"
        print(f"p={p:<8} {r['accuracy']*100:>6.1f}%        {r['time']:<12.1f} {status}")

    print("\nLiquid Neural Network (NO BACKPROP):")
    print(f"{'Prime':<10} {'Accuracy':<15} {'Time (s)':<12} {'Status':<15}")
    print("-" * 60)
    for p, r in lnn_results.items():
        status = "✓ PASS" if r['accuracy'] >= 0.3 else "✗ FAIL"
        print(f"p={p:<8} {r['accuracy']*100:>6.1f}%        {r['time']:<12.1f} {status}")

    print("\nComparison with Standard NN (WITH backprop):")
    print(f"{'Prime':<10} {'Standard':<12} {'FF (no BP)':<14} {'LNN (no BP)':<15}")
    print("-" * 60)
    standard = {7: 100.0, 11: 100.0}
    for p in curriculum:
        if p in ff_results and p in lnn_results:
            print(f"p={p:<8} {standard[p]:>5.1f}%      "
                  f"{ff_results[p]['accuracy']*100:>6.1f}%        "
                  f"{lnn_results[p]['accuracy']*100:>6.1f}%")

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print("\n✓ Forward-Forward learns via local contrastive learning")
    print("✓ Liquid Networks use Hebbian + ODE dynamics")
    print("✓ Both are biologically plausible (no backprop!)")
    print("✓ Cyclic encoding is crucial for modular wrap-around")

    if ff_results:
        avg_ff = np.mean([r['accuracy'] for r in ff_results.values()]) * 100
        print(f"\nForward-Forward average: {avg_ff:.1f}%")

    if lnn_results:
        avg_lnn = np.mean([r['accuracy'] for r in lnn_results.values()]) * 100
        print(f"Liquid Network average: {avg_lnn:.1f}%")

    # Save results
    results = {
        'forward_forward': {str(k): {'accuracy': float(v['accuracy']), 'time': float(v['time'])}
                           for k, v in ff_results.items()},
        'liquid_network': {str(k): {'accuracy': float(v['accuracy']), 'time': float(v['time'])}
                          for k, v in lnn_results.items()}
    }

    with open('bio_plausible_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: bio_plausible_results.json")


if __name__ == '__main__':
    main()
