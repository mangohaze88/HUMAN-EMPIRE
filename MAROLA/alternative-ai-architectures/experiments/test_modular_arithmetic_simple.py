#!/usr/bin/env python3
"""
Simple test to verify we can learn modular arithmetic at all.
Start with the simplest possible approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SimpleArithmeticNet(nn.Module):
    """Ultra-simple network for modular arithmetic"""

    def __init__(self, p: int, hidden_dim: int = 128):
        super().__init__()
        self.p = p

        # Input: two one-hot vectors
        input_dim = 2 * p

        # Simple 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p)
        )

    def encode(self, a: int, b: int) -> torch.Tensor:
        """One-hot encoding"""
        x = torch.zeros(2 * self.p, dtype=torch.float32)
        x[a] = 1.0
        x[self.p + b] = 1.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_simple(p: int = 7, n_epochs: int = 500):
    """Test on small prime"""

    print(f"\n{'='*80}")
    print(f"SIMPLE TEST: Modular Addition (mod {p})")
    print(f"{'='*80}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SimpleArithmeticNet(p, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # Generate ALL possible combinations
    all_data = []
    for a in range(p):
        for b in range(p):
            result = (a + b) % p
            all_data.append((a, b, result))

    print(f"Total combinations: {len(all_data)}")

    # Split: 70% train, 30% test
    np.random.shuffle(all_data)
    split = int(0.7 * len(all_data))
    train_data = all_data[:split]
    test_data = all_data[split:]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}\n")

    # Training
    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(n_epochs):
        net.train()
        np.random.shuffle(train_data)

        epoch_correct = 0
        epoch_loss = 0.0

        for a, b, result in train_data:
            x = net.encode(a, b).unsqueeze(0).to(device)
            y = torch.tensor([result], dtype=torch.long, device=device)

            logits = net(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=-1).item()
            if pred == result:
                epoch_correct += 1

        train_acc = epoch_correct / len(train_data)

        # Test
        if epoch % 50 == 0 or epoch == n_epochs - 1:
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
            best_test_acc = max(best_test_acc, test_acc)

            print(f"Epoch {epoch:4d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f} "
                  f"{'🌟 BEST' if test_acc == best_test_acc else ''}")

    train_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"Training time: {train_time:.2f}s")

    if best_test_acc >= 0.9:
        print(f"\n✓ SUCCESS: >90% accuracy!")
    else:
        print(f"\n✗ Failed to reach 90% (got {best_test_acc*100:.2f}%)")

    return best_test_acc


if __name__ == '__main__':
    # Test on progressively larger primes
    for p in [7, 11, 23, 47]:
        acc = test_simple(p, n_epochs=500)
        print(f"\n→ p={p}: {acc*100:.2f}% accuracy\n")

        if acc < 0.9:
            print(f"⚠ Stopping here - need to improve for p={p}")
            break
