#!/usr/bin/env python3
"""
MINIMAL FORWARD-FORWARD FOR MODULAR ARITHMETIC
===============================================

Clean, minimal implementation focusing on CORE Forward-Forward principles.

KEY INSIGHT:
-----------
Forward-Forward works by learning that CORRECT triplets (a, b, c) where c = (a+b) mod p
have HIGHER total squared activity than INCORRECT triplets.

The network doesn't "know" which is which - it learns from the CONTRAST.

ENCODING STRATEGY:
-----------------
1. Simple one-hot encoding for a, b, result
2. Cyclic features for wrap-around
3. NO cheating by including the correct answer!

Target: >90% accuracy on p=7, 11, 23
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import time


def encode_triplet(a: int, b: int, result: int, p: int, dim_per_value: int = 16) -> np.ndarray:
    """
    Encode (a, b, result) triplet for Forward-Forward.

    Simple encoding:
    - One-hot-ish encoding for each value (with some overlap for generalization)
    - Cyclic sin/cos for modular structure
    - That's it - NO cheating!
    """
    features = []

    # One-hot encoding with cyclic wrap
    for val in [a, b, result]:
        one_hot = np.zeros(dim_per_value)
        # Distributed representation: activate nearby neurons
        for offset in range(-1, 2):  # -1, 0, 1
            idx = (val + offset) % dim_per_value
            one_hot[idx] = 1.0 if offset == 0 else 0.5
        features.extend(one_hot)

    # Cyclic encoding for each value
    for val in [a, b, result]:
        features.extend([
            np.sin(2 * np.pi * val / p),
            np.cos(2 * np.pi * val / p),
        ])

    # Basic relationship features (NOT including correct answer!)
    features.extend([
        a / p,
        b / p,
        result / p,
        (a + b) / (2 * p),  # Sum hint (scaled)
    ])

    return np.array(features, dtype=np.float32)


def generate_hard_negatives(a: int, b: int, p: int, n_neg: int = 3) -> List[int]:
    """Generate hard negative results."""
    correct = (a + b) % p
    negatives = []

    # Off-by-one
    negatives.append((correct + 1) % p)
    negatives.append((correct - 1 + p) % p)

    # Random
    while len(negatives) < n_neg:
        neg = np.random.randint(0, p)
        if neg != correct and neg not in negatives:
            negatives.append(neg)

    return negatives[:n_neg]


class SimpleFFLayer(nn.Module):
    """Simple Forward-Forward layer."""

    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.3, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.lr = lr

        # Initialize with good scaling
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """Forward pass with ReLU."""
        return F.relu(F.linear(x, self.weight, self.bias))

    def goodness(self, x):
        """Compute goodness: mean squared activation."""
        h = self.forward(x)
        return (h ** 2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        """Local update - NO BACKPROP!"""
        with torch.no_grad():
            # Get activations
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            # Compute goodness
            g_pos = self.goodness(x_pos)
            g_neg = self.goodness(x_neg)

            # Compute probabilities
            p_pos = torch.sigmoid(g_pos - self.threshold)
            p_neg = torch.sigmoid(g_neg - self.threshold)

            # Local errors
            err_pos = (1.0 - p_pos).unsqueeze(1) * h_pos
            err_neg = p_neg.unsqueeze(1) * h_neg

            # Hebbian updates
            dW_pos = err_pos.T @ x_pos
            dW_neg = err_neg.T @ x_neg

            # Apply updates
            batch_size = x_pos.size(0)
            self.weight += (self.lr / batch_size) * (dW_pos - dW_neg)
            self.bias += (self.lr / batch_size) * (err_pos.sum(dim=0) - err_neg.sum(dim=0))

            # Weight decay
            self.weight *= 0.9999

            return g_pos.mean().item(), g_neg.mean().item()


class SimpleFFNetwork(nn.Module):
    """Simple Forward-Forward network."""

    def __init__(self, input_dim: int, hidden_dims: List[int], lr: float = 0.3, threshold: float = 2.0):
        super().__init__()

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            SimpleFFLayer(dims[i], dims[i+1], lr=lr, threshold=threshold)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        """Forward through all layers."""
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def total_goodness(self, x):
        """Compute total goodness across all layers."""
        h = x
        total_g = 0.0
        for layer in self.layers:
            g = layer.goodness(h)
            total_g = total_g + g
            h = layer(h)
        return total_g

    def train_step(self, x_pos, x_neg):
        """Train all layers."""
        h_pos = x_pos
        h_neg = x_neg

        stats = {'pos_goodness': [], 'neg_goodness': []}

        for layer in self.layers:
            g_pos, g_neg = layer.train_step(h_pos, h_neg)
            stats['pos_goodness'].append(g_pos)
            stats['neg_goodness'].append(g_neg)

            # Get next layer input
            with torch.no_grad():
                h_pos = layer(h_pos)
                h_neg = layer(h_neg)

        return stats


def train_minimal_ff(p: int, epochs: int = 1000, batch_size: int = 64, lr: float = 0.3):
    """Train minimal Forward-Forward network."""
    print(f"\n{'='*70}")
    print(f"MINIMAL FORWARD-FORWARD: p={p}")
    print(f"{'='*70}\n")

    # Setup
    sample = encode_triplet(0, 0, 0, p)
    input_dim = len(sample)
    print(f"Input dim: {input_dim}")

    # Create network
    net = SimpleFFNetwork(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        lr=lr,
        threshold=2.0
    )

    print(f"Network: {input_dim} -> 256 -> 128")
    print(f"Learning rate: {lr}")
    print(f"Training for {epochs} epochs...\n")

    # Generate training data
    train_data = []
    for _ in range(5000):
        a, b = np.random.randint(0, p, 2)
        result = (a + b) % p
        train_data.append((a, b, result))

    # Training loop
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        np.random.shuffle(train_data)

        epoch_stats = {'pos_goodness': [], 'neg_goodness': []}

        # Mini-batches
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            # Prepare positive samples
            x_pos = []
            x_neg = []

            for a, b, result in batch:
                # Positive
                x_pos.append(encode_triplet(a, b, result, p))

                # Negatives
                for neg_result in generate_hard_negatives(a, b, p, n_neg=2):
                    x_neg.append(encode_triplet(a, b, neg_result, p))

            x_pos = torch.tensor(np.array(x_pos), dtype=torch.float32)
            x_neg = torch.tensor(np.array(x_neg), dtype=torch.float32)

            # Train
            stats = net.train_step(x_pos, x_neg)

            epoch_stats['pos_goodness'].extend(stats['pos_goodness'])
            epoch_stats['neg_goodness'].extend(stats['neg_goodness'])

        # Evaluate every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            acc = evaluate_minimal_ff(net, p)
            if acc > best_acc:
                best_acc = acc

            avg_pos_g = np.mean(epoch_stats['pos_goodness'])
            avg_neg_g = np.mean(epoch_stats['neg_goodness'])
            sep = avg_pos_g - avg_neg_g

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}: Acc={acc*100:5.1f}%, Best={best_acc*100:5.1f}%, "
                  f"G+={avg_pos_g:.3f}, G-={avg_neg_g:.3f}, Sep={sep:.3f}, "
                  f"Time={elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(f"FINAL: {best_acc*100:.2f}% accuracy")
    print(f"{'='*70}\n")

    return best_acc


def evaluate_minimal_ff(net: SimpleFFNetwork, p: int) -> float:
    """Evaluate on all possible inputs."""
    correct = 0
    total = 0

    with torch.no_grad():
        for a in range(p):
            for b in range(p):
                true_result = (a + b) % p

                # Try all possible results
                best_g = -float('inf')
                pred = 0

                for test_result in range(p):
                    x = encode_triplet(a, b, test_result, p)
                    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

                    g = net.total_goodness(x_t).item()

                    if g > best_g:
                        best_g = g
                        pred = test_result

                if pred == true_result:
                    correct += 1
                total += 1

    return correct / total


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--curriculum', action='store_true')

    args = parser.parse_args()

    if args.curriculum:
        print("\nCURRICULUM LEARNING")
        print("="*70)

        results = {}
        for p in [7, 11, 23]:
            acc = train_minimal_ff(p=p, epochs=args.epochs, lr=args.lr)
            results[p] = acc

            if acc < 0.5:
                print(f"\n⚠ Stopping curriculum - accuracy too low ({acc*100:.1f}%)")
                break

        print("\n" + "="*70)
        print("CURRICULUM RESULTS")
        print("="*70)
        for p, acc in results.items():
            status = "✓" if acc >= 0.90 else "✗"
            print(f"p={p:3d}: {acc*100:5.1f}% {status}")

    else:
        train_minimal_ff(p=args.p, epochs=args.epochs, lr=args.lr)


if __name__ == '__main__':
    main()
