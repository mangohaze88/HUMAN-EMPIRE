#!/usr/bin/env python3
"""
Fast Arithmetic LNN - Optimized for Quick Training
==================================================

Simplified but effective architecture for modular arithmetic.
Target: >90% accuracy in <60s training time.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import time


def one_hot_modular(val: int, p: int) -> np.ndarray:
    """Simple one-hot encoding with modular position hints"""
    features = np.zeros(p + 4, dtype=np.float32)
    features[val] = 1.0  # One-hot

    # Add cyclic hints
    angle = 2 * np.pi * val / p
    features[p] = np.sin(angle)
    features[p + 1] = np.cos(angle)
    features[p + 2] = val / p  # Normalized
    features[p + 3] = float(val)  # Raw value (small p)

    return features


class FastArithmeticNet(nn.Module):
    """Fast feedforward network for modular arithmetic"""

    def __init__(self, p: int, hidden_dim: int = 256):
        super().__init__()
        self.p = p

        # Input: two one-hot encoded values
        input_dim = 2 * (p + 4)

        self.net = nn.Sequential(
            # First layer: expand
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            # Second layer: process
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            # Third layer: compress
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output layer
            nn.Linear(hidden_dim // 2, p)
        )

    def encode(self, a: int, b: int) -> torch.Tensor:
        """Encode inputs"""
        a_enc = one_hot_modular(a, self.p)
        b_enc = one_hot_modular(b, self.p)
        features = np.concatenate([a_enc, b_enc])
        return torch.tensor(features, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_fast(
    p: int,
    operation: str = 'add',
    hidden_dim: int = 256,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Fast training loop"""

    print(f"\n{'='*80}")
    print(f"FAST ARITHMETIC NET: {operation.upper()} mod {p}")
    print(f"{'='*80}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Epochs: {n_epochs}")
    print(f"Device: {device}\n")

    # Create network
    net = FastArithmeticNet(p, hidden_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Generate all possible pairs (exhaustive for small p)
    if p <= 50:
        # Use all combinations
        train_data = []
        for a in range(p):
            for b in range(p):
                if operation == 'add':
                    result = (a + b) % p
                elif operation == 'sub':
                    result = (a - b) % p
                elif operation == 'mult':
                    result = (a * b) % p
                train_data.append((a, b, result))

        # Split 80/20
        np.random.shuffle(train_data)
        split = int(0.8 * len(train_data))
        test_data = train_data[split:]
        train_data = train_data[:split]
    else:
        # Sample for large p
        n_train = min(10000, p * p // 2)
        n_test = min(2000, p * p // 10)

        train_data = []
        for _ in range(n_train):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            if operation == 'add':
                result = (a + b) % p
            elif operation == 'sub':
                result = (a - b) % p
            elif operation == 'mult':
                result = (a * b) % p
            train_data.append((a, b, result))

        test_data = []
        for _ in range(n_test):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            if operation == 'add':
                result = (a + b) % p
            elif operation == 'sub':
                result = (a - b) % p
            elif operation == 'mult':
                result = (a * b) % p
            test_data.append((a, b, result))

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}\n")

    # Training
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(n_epochs):
        net.train()
        np.random.shuffle(train_data)

        epoch_loss = 0.0
        epoch_correct = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            x_batch = torch.stack([net.encode(a, b) for a, b, _ in batch]).to(device)
            y_batch = torch.tensor([r for _, _, r in batch], dtype=torch.long, device=device)

            # Forward
            logits = net(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # Stats
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            epoch_correct += (preds == y_batch).sum().item()

        scheduler.step()

        train_acc = epoch_correct / len(train_data)

        # Test
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            net.eval()
            test_correct = 0

            with torch.no_grad():
                for a, b, result in test_data:
                    x = net.encode(a, b).unsqueeze(0).to(device)
                    logits = net(x)
                    pred = torch.argmax(logits, dim=-1).item()
                    if pred == result:
                        test_correct += 1

            test_acc = test_correct / len(test_data)
            best_acc = max(best_acc, test_acc)

            n_batches = max(1, len(train_data) // batch_size)
            print(f"Epoch {epoch:3d}: loss={epoch_loss / n_batches:.4f}, "
                  f"train={train_acc:.4f}, test={test_acc:.4f} "
                  f"{'🌟' if test_acc == best_acc else ''}")

    train_time = time.time() - start_time

    # Final eval
    net.eval()
    final_correct = 0
    with torch.no_grad():
        for a, b, result in test_data:
            x = net.encode(a, b).unsqueeze(0).to(device)
            logits = net(x)
            pred = torch.argmax(logits, dim=-1).item()
            if pred == result:
                final_correct += 1

    final_acc = final_correct / len(test_data)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Final accuracy: {final_acc*100:.2f}%")
    print(f"Best accuracy: {best_acc*100:.2f}%")
    print(f"Training time: {train_time:.2f}s")

    if final_acc >= 0.9:
        print(f"\n🎉 SUCCESS: Achieved >90% accuracy!")

    return {
        'p': p,
        'operation': operation,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
        'train_time': train_time,
    }


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FAST ARITHMETIC NETWORK - QUICK TESTS")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test on small primes
    for p in [7, 11, 23]:
        result = train_fast(p, 'add', hidden_dim=256, n_epochs=100, device=device)
        print(f"\np={p} addition: {result['final_accuracy']*100:.2f}% accuracy\n")

        if result['final_accuracy'] < 0.9:
            print(f"⚠ Warning: Did not reach 90% for p={p}")
