#!/usr/bin/env python3
"""
NALU Modular Arithmetic Benchmark
===================================

Tests bio-plausible NALU implementations on modular arithmetic learning.

Compares:
1. Standard NALU (with backpropagation)
2. FF-NALU (Forward-Forward learning, NO backprop)
3. Hebbian-NALU (Three-factor learning, NO backprop)

Task: Learn modular addition (a + b) mod p

Success criteria: >80% accuracy on p=23
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, List, Dict
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.bio_nalu import (
    NALUArithmeticNet,
    create_nalu_network,
    BioNALU,
    NALUConfig
)


# ============================================================================
# DATA GENERATION
# ============================================================================

def to_binary(n: int, bits: int = 10) -> np.ndarray:
    """Convert number to binary representation."""
    return np.array([(n >> i) & 1 for i in range(bits)], dtype=np.float32)


def combined_encoding(a: int, b: int, p: int, bits: int = 10) -> np.ndarray:
    """
    Combined encoding for modular arithmetic.

    Features:
    - Binary encoding (explicit bits)
    - Normalized values (a/p, b/p)
    - Cyclic encoding (sin/cos for wrap-around)
    """
    features = []

    # Binary encoding for both inputs
    for val in [a, b]:
        features.extend([(val >> i) & 1 for i in range(bits)])

    # Normalized values
    features.extend([a / p, b / p])

    # Cyclic encoding (KEY for modular wrap-around!)
    features.extend([
        np.sin(2 * np.pi * a / p),
        np.cos(2 * np.pi * a / p),
        np.sin(2 * np.pi * b / p),
        np.cos(2 * np.pi * b / p),
    ])

    return np.array(features, dtype=np.float32)


def generate_modular_addition_dataset(
    p: int,
    num_samples: int = 5000,
    bits: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset for modular addition.

    Returns:
        X: Input features [num_samples, feature_dim]
        Y: Target labels [num_samples] (values in range [0, p-1])
    """
    X = []
    Y = []

    for _ in range(num_samples):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        result = (a + b) % p

        x = combined_encoding(a, b, p, bits)
        X.append(x)
        Y.append(result)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.long)

    return X, Y


# ============================================================================
# STANDARD NALU (WITH BACKPROPAGATION)
# ============================================================================

class StandardNALU(nn.Module):
    """Standard NALU with backpropagation for baseline comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        epsilon: float = 1e-7,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, device=device)

        # NALU layers
        self.nalu_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.nalu_layers.append(self._create_nalu_layer(hidden_dim, hidden_dim))

        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim, device=device)

    def _create_nalu_layer(self, input_dim: int, output_dim: int):
        """Create a single NALU layer."""
        config = NALUConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.device
        )
        return BioNALU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))

        for nalu in self.nalu_layers:
            h = nalu.forward(h)
            h = F.relu(h)

        out = self.classifier(h)
        return out  # Return logits for cross-entropy


def train_standard_nalu(
    model: StandardNALU,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Train standard NALU with backpropagation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': []
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        train_loss /= len(train_loader)

        # Testing
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)

        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Test Acc={test_acc:.1f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)

    elapsed = time.time() - start_time
    history['time'] = elapsed
    history['best_acc'] = best_acc

    return history


# ============================================================================
# BIO-PLAUSIBLE NALU TRAINING
# ============================================================================

def train_bio_nalu(
    model: NALUArithmeticNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Train bio-plausible NALU (FF or Hebbian)."""
    model = model.to(device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': []
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Bio-plausible training step
            if model.learning_type == 'ff':
                stats = model.train_step_ff(x, y)
                loss = stats['total_loss']
            else:  # hebbian
                stats = model.train_step_hebbian(x, y)
                loss = stats['loss']

            epoch_loss += loss

            # Compute accuracy
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        train_acc = 100.0 * correct / total
        epoch_loss /= len(train_loader)

        # Testing
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)

        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={epoch_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Test Acc={test_acc:.1f}%")

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)

    elapsed = time.time() - start_time
    history['time'] = elapsed
    history['best_acc'] = best_acc

    return history


# ============================================================================
# BENCHMARK
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single experiment."""
    architecture: str
    prime: int
    accuracy: float
    time: float
    bio_plausible: bool
    num_epochs: int
    hidden_dim: int


def run_benchmark(
    primes: List[int] = [7, 23, 97],
    num_samples: int = 5000,
    num_epochs: int = 50,
    hidden_dim: int = 128,
    batch_size: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[BenchmarkResult]:
    """
    Run complete NALU arithmetic benchmark.

    Tests:
    1. Standard NALU (backprop)
    2. FF-NALU (Forward-Forward)
    3. Hebbian-NALU (Three-factor learning)
    """
    results = []

    for p in primes:
        print(f"\n{'='*60}")
        print(f"Testing on MODULAR ADDITION mod {p}")
        print(f"{'='*60}\n")

        # Generate data
        print(f"Generating {num_samples} samples...")
        X_train, Y_train = generate_modular_addition_dataset(p, num_samples)
        X_test, Y_test = generate_modular_addition_dataset(p, num_samples // 5)

        input_dim = X_train.shape[1]
        output_dim = p

        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 1. Standard NALU (WITH backprop)
        print(f"\n[1/3] Training Standard NALU (WITH backpropagation)...")
        model = StandardNALU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=device
        )
        history = train_standard_nalu(
            model, train_loader, test_loader,
            num_epochs=num_epochs, device=device
        )

        results.append(BenchmarkResult(
            architecture="Standard NALU",
            prime=p,
            accuracy=history['best_acc'],
            time=history['time'],
            bio_plausible=False,
            num_epochs=num_epochs,
            hidden_dim=hidden_dim
        ))

        print(f"\n✓ Standard NALU: {history['best_acc']:.1f}% accuracy in {history['time']:.1f}s")

        # 2. FF-NALU (NO backprop)
        print(f"\n[2/3] Training FF-NALU (NO backpropagation)...")
        model_ff = create_nalu_network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_type='ff',
            num_nalu_layers=2,
            learning_rate=0.03,
            threshold=2.0,
            device=device
        )
        history_ff = train_bio_nalu(
            model_ff, train_loader, test_loader,
            num_epochs=num_epochs, device=device
        )

        results.append(BenchmarkResult(
            architecture="FF-NALU",
            prime=p,
            accuracy=history_ff['best_acc'],
            time=history_ff['time'],
            bio_plausible=True,
            num_epochs=num_epochs,
            hidden_dim=hidden_dim
        ))

        print(f"\n✓ FF-NALU: {history_ff['best_acc']:.1f}% accuracy in {history_ff['time']:.1f}s")

        # 3. Hebbian-NALU (NO backprop)
        print(f"\n[3/3] Training Hebbian-NALU (NO backpropagation)...")
        model_hebb = create_nalu_network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_type='hebbian',
            num_nalu_layers=2,
            learning_rate=0.02,
            device=device
        )
        history_hebb = train_bio_nalu(
            model_hebb, train_loader, test_loader,
            num_epochs=num_epochs, device=device
        )

        results.append(BenchmarkResult(
            architecture="Hebbian-NALU",
            prime=p,
            accuracy=history_hebb['best_acc'],
            time=history_hebb['time'],
            bio_plausible=True,
            num_epochs=num_epochs,
            hidden_dim=hidden_dim
        ))

        print(f"\n✓ Hebbian-NALU: {history_hebb['best_acc']:.1f}% accuracy in {history_hebb['time']:.1f}s")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("NALU ARITHMETIC BENCHMARK RESULTS")
    print("="*80 + "\n")

    # Group by architecture
    architectures = ['Standard NALU', 'FF-NALU', 'Hebbian-NALU']
    primes = sorted(list(set(r.prime for r in results)))

    # Print header
    print(f"{'Architecture':<20} | ", end='')
    for p in primes:
        print(f"p={p:<4} | ", end='')
    print(f"Bio-Plausible?")
    print("-" * 80)

    # Print rows
    for arch in architectures:
        arch_results = [r for r in results if r.architecture == arch]
        print(f"{arch:<20} | ", end='')

        for p in primes:
            matching = [r for r in arch_results if r.prime == p]
            if matching:
                acc = matching[0].accuracy
                print(f"{acc:>5.1f}% | ", end='')
            else:
                print(f"  -   | ", end='')

        # Bio-plausible flag
        if arch_results:
            bio = "YES" if arch_results[0].bio_plausible else "NO"
            print(f"{bio:>5}")

    print("-" * 80)
    print()

    # Print success criteria
    print("\nSUCCESS CRITERIA: >80% accuracy on p=23\n")

    for arch in architectures:
        arch_results = [r for r in results if r.architecture == arch and r.prime == 23]
        if arch_results:
            acc = arch_results[0].accuracy
            status = "✓ PASS" if acc > 80 else "✗ FAIL"
            print(f"{arch:<20}: {acc:>5.1f}% {status}")


def save_results(results: List[BenchmarkResult], filepath: str):
    """Save results to JSON."""
    data = {
        'results': [asdict(r) for r in results],
        'summary': {
            'total_experiments': len(results),
            'architectures': list(set(r.architecture for r in results)),
            'primes_tested': sorted(list(set(r.prime for r in results)))
        }
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def plot_results(results: List[BenchmarkResult], save_path: str = None):
    """Plot benchmark results."""
    architectures = ['Standard NALU', 'FF-NALU', 'Hebbian-NALU']
    primes = sorted(list(set(r.prime for r in results)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy comparison
    colors = {'Standard NALU': 'blue', 'FF-NALU': 'green', 'Hebbian-NALU': 'orange'}

    for arch in architectures:
        arch_results = sorted([r for r in results if r.architecture == arch],
                            key=lambda x: x.prime)
        if arch_results:
            primes_arch = [r.prime for r in arch_results]
            accs = [r.accuracy for r in arch_results]

            ax1.plot(primes_arch, accs, 'o-', label=arch,
                    color=colors[arch], linewidth=2, markersize=8)

    ax1.axhline(y=80, color='red', linestyle='--', label='Target (80%)')
    ax1.set_xlabel('Prime (p)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('NALU Modular Arithmetic Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot 2: Time comparison
    for arch in architectures:
        arch_results = sorted([r for r in results if r.architecture == arch],
                            key=lambda x: x.prime)
        if arch_results:
            primes_arch = [r.prime for r in arch_results]
            times = [r.time for r in arch_results]

            ax2.plot(primes_arch, times, 'o-', label=arch,
                    color=colors[arch], linewidth=2, markersize=8)

    ax2.set_xlabel('Prime (p)', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='NALU Arithmetic Benchmark')
    parser.add_argument('--primes', type=int, nargs='+', default=[7, 23],
                       help='List of primes to test (default: [7, 23])')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples (default: 5000)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension (default: 128)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with p=7 only')

    args = parser.parse_args()

    if args.quick:
        primes = [7]
        num_samples = 2000
        num_epochs = 30
    else:
        primes = args.primes
        num_samples = args.samples
        num_epochs = args.epochs

    print("="*80)
    print("NALU ARITHMETIC BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Primes: {primes}")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device: {args.device}")
    print()

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Run benchmark
    results = run_benchmark(
        primes=primes,
        num_samples=num_samples,
        num_epochs=num_epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=device
    )

    # Print results
    print_results_table(results)

    # Save results
    output_dir = os.path.dirname(__file__)
    save_results(results, os.path.join(output_dir, 'nalu_arithmetic_results.json'))

    # Plot results
    plot_results(results, os.path.join(output_dir, 'nalu_arithmetic_results.png'))

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
