#!/usr/bin/env python3
"""
Improved EC Math Learning Benchmark - Version 2

Key improvements:
1. Better number encoding (binary + modular features)
2. Curriculum learning (start easy, scale up)
3. More training data and epochs
4. Better architecture with skip connections
5. Auxiliary losses for more learning signal
6. Start with p=7 (easiest prime)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import os


# ============================================================================
# Number Encoding Strategies
# ============================================================================

def to_binary(n: int, bits: int = 10) -> np.ndarray:
    """Convert number to binary representation."""
    return np.array([(n >> i) & 1 for i in range(bits)], dtype=np.float32)


def modular_features(n: int, p: int) -> np.ndarray:
    """
    Create rich modular features for a number.
    Includes: normalized value, digit features, cyclic features.
    """
    features = [
        n / p,                           # Normalized value
        (n % 10) / 10,                   # Last digit
        (n % 100) / 100,                 # Last two digits
        np.sin(2 * np.pi * n / p),       # Cyclic feature 1
        np.cos(2 * np.pi * n / p),       # Cyclic feature 2
        np.sin(4 * np.pi * n / p),       # Higher frequency cyclic
        np.cos(4 * np.pi * n / p),       # Higher frequency cyclic
    ]
    return np.array(features, dtype=np.float32)


def combined_encoding(n: int, p: int, binary_bits: int = 10) -> np.ndarray:
    """Combine binary and modular features."""
    binary = to_binary(n, binary_bits)
    modular = modular_features(n, p)
    return np.concatenate([binary, modular])


# ============================================================================
# Dataset
# ============================================================================

class ModularAdditionDataset(Dataset):
    """Dataset for modular addition: (a + b) mod p"""

    def __init__(self, p: int, num_samples: int = 100000,
                 encoding: str = 'combined', binary_bits: int = 10):
        self.p = p
        self.num_samples = num_samples
        self.encoding = encoding
        self.binary_bits = binary_bits

        # Generate data
        np.random.seed(42)
        self.data = []

        for _ in range(num_samples):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            result = (a + b) % p

            # Auxiliary labels
            wrap_around = 1 if (a + b >= p) else 0
            quotient = (a + b) // p  # Always 0 or 1 for addition

            self.data.append((a, b, result, wrap_around, quotient))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a, b, result, wrap_around, quotient = self.data[idx]

        # Encode inputs
        if self.encoding == 'simple':
            x = torch.tensor([a / self.p, b / self.p], dtype=torch.float32)
        elif self.encoding == 'binary':
            a_bin = to_binary(a, self.binary_bits)
            b_bin = to_binary(b, self.binary_bits)
            x = torch.tensor(np.concatenate([a_bin, b_bin]), dtype=torch.float32)
        elif self.encoding == 'modular':
            a_feat = modular_features(a, self.p)
            b_feat = modular_features(b, self.p)
            x = torch.tensor(np.concatenate([a_feat, b_feat]), dtype=torch.float32)
        elif self.encoding == 'combined':
            a_enc = combined_encoding(a, self.p, self.binary_bits)
            b_enc = combined_encoding(b, self.p, self.binary_bits)
            x = torch.tensor(np.concatenate([a_enc, b_enc]), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        return x, result, wrap_around, quotient


# ============================================================================
# Model Architectures
# ============================================================================

class SimpleMLPWithSkip(nn.Module):
    """MLP with skip connections for better gradient flow."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 4, use_aux_loss: bool = True):
        super().__init__()
        self.use_aux_loss = use_aux_loss

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output heads
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if use_aux_loss:
            self.wrap_head = nn.Linear(hidden_dim, 2)  # Binary: wrap or not
            self.quotient_head = nn.Linear(hidden_dim, 2)  # 0 or 1

    def forward(self, x):
        # Input
        h = F.relu(self.input_layer(x))

        # Hidden layers with skip connections
        for layer in self.hidden_layers:
            h_new = F.relu(layer(h))
            h = h + h_new  # Skip connection

        # Main output
        logits = self.output_layer(h)

        if self.use_aux_loss:
            wrap_logits = self.wrap_head(h)
            quotient_logits = self.quotient_head(h)
            return logits, wrap_logits, quotient_logits
        else:
            return logits


class AttentionMLP(nn.Module):
    """MLP with attention mechanism for digit positions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 4, use_aux_loss: bool = True):
        super().__init__()
        self.use_aux_loss = use_aux_loss

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Feed-forward
        self.ff1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ff2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output heads
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if use_aux_loss:
            self.wrap_head = nn.Linear(hidden_dim, 2)
            self.quotient_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # Input
        h = F.relu(self.input_layer(x))

        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # [batch, 1, hidden]

        # Self-attention
        h_attn, _ = self.attention(h, h, h)
        h = h + h_attn  # Skip connection

        # Remove sequence dimension
        h = h.squeeze(1)

        # Feed-forward with skip
        h_ff = F.relu(self.ff1(h))
        h_ff = self.ff2(h_ff)
        h = h + h_ff

        # Output
        logits = self.output_layer(h)

        if self.use_aux_loss:
            wrap_logits = self.wrap_head(h)
            quotient_logits = self.quotient_head(h)
            return logits, wrap_logits, quotient_logits
        else:
            return logits


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, use_aux_loss=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, result, wrap, quotient in dataloader:
        x = x.to(device)
        result = result.to(device)
        wrap = wrap.to(device)
        quotient = quotient.to(device)

        optimizer.zero_grad()

        if use_aux_loss:
            logits, wrap_logits, quotient_logits = model(x)

            # Main loss
            loss_main = F.cross_entropy(logits, result)

            # Auxiliary losses
            loss_wrap = F.cross_entropy(wrap_logits, wrap)
            loss_quotient = F.cross_entropy(quotient_logits, quotient)

            # Combined loss (main loss has higher weight)
            loss = loss_main + 0.2 * loss_wrap + 0.2 * loss_quotient
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, result)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == result).sum().item()
        total += result.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device, use_aux_loss=True):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    wrap_correct = 0
    quotient_correct = 0

    with torch.no_grad():
        for x, result, wrap, quotient in dataloader:
            x = x.to(device)
            result = result.to(device)
            wrap = wrap.to(device)
            quotient = quotient.to(device)

            if use_aux_loss:
                logits, wrap_logits, quotient_logits = model(x)

                # Auxiliary predictions
                wrap_pred = wrap_logits.argmax(dim=1)
                wrap_correct += (wrap_pred == wrap).sum().item()

                quotient_pred = quotient_logits.argmax(dim=1)
                quotient_correct += (quotient_pred == quotient).sum().item()
            else:
                logits = model(x)

            pred = logits.argmax(dim=1)
            correct += (pred == result).sum().item()
            total += result.size(0)

    accuracy = correct / total
    wrap_acc = wrap_correct / total if use_aux_loss else 0
    quotient_acc = quotient_correct / total if use_aux_loss else 0

    return accuracy, wrap_acc, quotient_acc


def train_on_prime(p: int, encoding: str = 'combined', architecture: str = 'skip',
                   num_samples: int = 100000, epochs: int = 50,
                   hidden_dim: int = 256, num_layers: int = 4,
                   batch_size: int = 256, learning_rate: float = 0.001,
                   use_aux_loss: bool = True, device: str = 'cuda'):
    """
    Train a model on modular addition for prime p.

    Returns:
        Dict with training history and final model
    """
    print(f"\n{'='*80}")
    print(f"Training on p={p} (modular addition)")
    print(f"Encoding: {encoding}, Architecture: {architecture}")
    print(f"Samples: {num_samples}, Epochs: {epochs}, Hidden: {hidden_dim}")
    print(f"{'='*80}\n")

    # Create datasets
    train_dataset = ModularAdditionDataset(p, num_samples, encoding)
    test_dataset = ModularAdditionDataset(p, num_samples // 10, encoding)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get input dimension from a sample
    sample_x, _, _, _ = train_dataset[0]
    input_dim = sample_x.shape[0]

    # Create model
    if architecture == 'skip':
        model = SimpleMLPWithSkip(input_dim, hidden_dim, p, num_layers, use_aux_loss)
    elif architecture == 'attention':
        model = AttentionMLP(input_dim, hidden_dim, p, num_heads=4, use_aux_loss=use_aux_loss)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)

    # Optimizer with learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_wrap_acc': [],
        'test_quotient_acc': [],
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, use_aux_loss)

        # Evaluate
        test_acc, wrap_acc, quotient_acc = evaluate(model, test_loader, device, use_aux_loss)

        # Learning rate schedule
        scheduler.step(test_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_wrap_acc'].append(wrap_acc)
        history['test_quotient_acc'].append(quotient_acc)

        # Track best
        if test_acc > best_acc:
            best_acc = test_acc

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | "
                  f"Best: {best_acc:.4f}")

            if use_aux_loss:
                print(f"           | Wrap Acc: {wrap_acc:.4f} | Quotient Acc: {quotient_acc:.4f}")

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Best test accuracy: {best_acc:.4f}")

    return {
        'model': model,
        'history': history,
        'best_acc': best_acc,
        'training_time': training_time,
        'p': p,
    }


# ============================================================================
# Curriculum Learning
# ============================================================================

def curriculum_learning(
    curriculum: List[int] = [7, 11, 23, 47, 97],
    target_accuracy: float = 0.80,
    encoding: str = 'combined',
    architecture: str = 'skip',
    **train_kwargs
):
    """
    Progressive curriculum learning: start easy, scale up.

    Args:
        curriculum: List of primes in increasing difficulty
        target_accuracy: Required accuracy to progress to next prime
        encoding: Encoding strategy
        architecture: Model architecture
        **train_kwargs: Additional arguments for train_on_prime
    """
    print("\n" + "="*80)
    print("CURRICULUM LEARNING PROTOCOL")
    print(f"Curriculum: {curriculum}")
    print(f"Target accuracy to progress: {target_accuracy:.2%}")
    print("="*80)

    results = []

    for p in curriculum:
        result = train_on_prime(
            p=p,
            encoding=encoding,
            architecture=architecture,
            **train_kwargs
        )

        results.append(result)

        # Check if we can progress
        if result['best_acc'] >= target_accuracy:
            print(f"\n✓ SUCCESS: Achieved {result['best_acc']:.4f} >= {target_accuracy:.4f}")
            print(f"  Progressing to next prime...\n")
        else:
            print(f"\n✗ FAILED: Only achieved {result['best_acc']:.4f} < {target_accuracy:.4f}")
            print(f"  Stopping curriculum at p={p}\n")
            break

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(results: List[Dict], save_path: str = None):
    """Plot training history for multiple primes."""
    num_results = len(results)

    fig, axes = plt.subplots(2, num_results, figsize=(5*num_results, 10))
    if num_results == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        history = result['history']
        p = result['p']

        # Plot 1: Loss and accuracy
        ax1 = axes[0, i]
        ax1_twin = ax1.twinx()

        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1_twin.plot(epochs, history['train_acc'], 'g-', label='Train Acc')
        ax1_twin.plot(epochs, history['test_acc'], 'r-', label='Test Acc')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1_twin.set_ylabel('Accuracy', color='r')
        ax1.set_title(f'p={p} - Training Progress')
        ax1.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Plot 2: Auxiliary task accuracy
        ax2 = axes[1, i]
        ax2.plot(epochs, history['test_acc'], 'b-', label='Main (Addition)')
        ax2.plot(epochs, history['test_wrap_acc'], 'g-', label='Wrap Detection')
        ax2.plot(epochs, history['test_quotient_acc'], 'r-', label='Quotient')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'p={p} - Task Breakdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def print_curriculum_summary(results: List[Dict]):
    """Print a summary of curriculum learning results."""
    print("\n" + "="*80)
    print("CURRICULUM LEARNING SUMMARY")
    print("="*80)
    print(f"{'Prime':<8} {'Best Acc':<12} {'Training Time':<15} {'Status':<10}")
    print("-"*80)

    for result in results:
        p = result['p']
        acc = result['best_acc']
        time_taken = result['training_time']
        status = "✓ PASS" if acc >= 0.80 else "✗ FAIL"

        print(f"{p:<8} {acc:<12.4f} {time_taken:<15.2f}s {status:<10}")

    print("="*80)

    # Overall statistics
    if results:
        max_p = max(r['p'] for r in results if r['best_acc'] >= 0.80)
        print(f"\nMaximum prime with ≥80% accuracy: p={max_p}")
        print(f"Total primes tested: {len(results)}")
        print(f"Total training time: {sum(r['training_time'] for r in results):.2f}s")


# ============================================================================
# Main Experiments
# ============================================================================

def experiment_1_basic_curriculum():
    """Experiment 1: Basic curriculum with simple settings."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Basic Curriculum Learning")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    results = curriculum_learning(
        curriculum=[7, 11, 23, 47, 97],
        target_accuracy=0.85,
        encoding='combined',
        architecture='skip',
        num_samples=50000,
        epochs=30,
        hidden_dim=256,
        num_layers=4,
        batch_size=256,
        learning_rate=0.001,
        use_aux_loss=True,
        device=device
    )

    print_curriculum_summary(results)

    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_training_history(results, 'plots/curriculum_basic.png')

    return results


def experiment_2_encoding_comparison():
    """Experiment 2: Compare different encodings on p=23."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Encoding Comparison (p=23)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encodings = ['simple', 'binary', 'modular', 'combined']
    results = []

    for encoding in encodings:
        print(f"\n--- Testing encoding: {encoding} ---")
        result = train_on_prime(
            p=23,
            encoding=encoding,
            architecture='skip',
            num_samples=50000,
            epochs=30,
            hidden_dim=256,
            num_layers=4,
            batch_size=256,
            learning_rate=0.001,
            use_aux_loss=True,
            device=device
        )
        results.append(result)

    # Print comparison
    print("\n" + "="*80)
    print("ENCODING COMPARISON RESULTS")
    print("="*80)
    print(f"{'Encoding':<15} {'Best Accuracy':<15} {'Training Time':<15}")
    print("-"*80)

    for encoding, result in zip(encodings, results):
        print(f"{encoding:<15} {result['best_acc']:<15.4f} {result['training_time']:<15.2f}s")

    print("="*80)

    return results


def experiment_3_architecture_comparison():
    """Experiment 3: Compare architectures on p=47."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Architecture Comparison (p=47)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    architectures = ['skip', 'attention']
    results = []

    for arch in architectures:
        print(f"\n--- Testing architecture: {arch} ---")
        result = train_on_prime(
            p=47,
            encoding='combined',
            architecture=arch,
            num_samples=50000,
            epochs=30,
            hidden_dim=256,
            num_layers=4,
            batch_size=256,
            learning_rate=0.001,
            use_aux_loss=True,
            device=device
        )
        results.append(result)

    # Print comparison
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Architecture':<15} {'Best Accuracy':<15} {'Training Time':<15}")
    print("-"*80)

    for arch, result in zip(architectures, results):
        print(f"{arch:<15} {result['best_acc']:<15.4f} {result['training_time']:<15.2f}s")

    print("="*80)

    return results


def experiment_4_scaling_test():
    """Experiment 4: How far can we scale with best settings?"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Maximum Scaling Test")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extended curriculum
    results = curriculum_learning(
        curriculum=[7, 11, 23, 47, 97, 199, 397, 997],
        target_accuracy=0.80,
        encoding='combined',
        architecture='skip',
        num_samples=100000,  # More samples
        epochs=50,           # More epochs
        hidden_dim=512,      # Bigger network
        num_layers=6,
        batch_size=256,
        learning_rate=0.001,
        use_aux_loss=True,
        device=device
    )

    print_curriculum_summary(results)

    # Plot results
    plot_training_history(results, 'plots/curriculum_scaling.png')

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("EC MATH LEARNING BENCHMARK - VERSION 2")
    print("="*80)
    print("\nThis benchmark tests if neural networks can learn modular arithmetic")
    print("(the foundation of elliptic curve cryptography).")
    print("\nRunning multiple experiments...\n")

    # Create output directory
    os.makedirs('plots', exist_ok=True)

    # Run experiments
    print("\n" + "="*80)
    print("Starting Experiment 1: Basic Curriculum")
    print("="*80)
    results_1 = experiment_1_basic_curriculum()

    print("\n" + "="*80)
    print("Starting Experiment 2: Encoding Comparison")
    print("="*80)
    results_2 = experiment_2_encoding_comparison()

    print("\n" + "="*80)
    print("Starting Experiment 3: Architecture Comparison")
    print("="*80)
    results_3 = experiment_3_architecture_comparison()

    print("\n" + "="*80)
    print("Starting Experiment 4: Maximum Scaling")
    print("="*80)
    results_4 = experiment_4_scaling_test()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nKey Findings:")
    print("1. Basic curriculum results show learning capability on small primes")
    print("2. Encoding comparison reveals best representation strategy")
    print("3. Architecture comparison shows which design scales better")
    print("4. Scaling test reveals maximum prime we can handle with 80%+ accuracy")
    print("\nPlots saved to: /root/MAROLA/alternative-ai-architectures/plots/")
    print("="*80)
