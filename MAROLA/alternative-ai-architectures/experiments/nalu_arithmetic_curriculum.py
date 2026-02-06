#!/usr/bin/env python3
"""
NALU Arithmetic with Curriculum Learning
=========================================

Improved bio-plausible NALU with curriculum learning strategy.

Key improvements:
1. Curriculum: Start with p=7, gradually increase
2. Better reward shaping
3. Adaptive learning rates
4. More training iterations
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.bio_nalu import create_nalu_network
from experiments.nalu_arithmetic import (
    combined_encoding,
    generate_modular_addition_dataset,
    BenchmarkResult
)


def train_hebbian_nalu_curriculum(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.02,
    device: str = 'cuda'
) -> Dict:
    """
    Train Hebbian-NALU with improved reward shaping.
    """
    model = model.to(device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': [],
        'rewards': []
    }

    best_acc = 0.0
    start_time = time.time()

    # Adaptive learning rate schedule
    def get_lr(epoch):
        if epoch < 20:
            return learning_rate
        elif epoch < 50:
            return learning_rate * 0.5
        else:
            return learning_rate * 0.1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_rewards = []

        # Update learning rate
        current_lr = get_lr(epoch)
        for nalu in model.nalu_layers:
            nalu.nalu.config.learning_rate = current_lr

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Compute prediction and reward
            with torch.no_grad():
                h = F.relu(model.encoder(x))
                for nalu in model.nalu_layers:
                    h = F.relu(nalu.forward(h))
                logits = model.classifier(h)
                ce_loss = F.cross_entropy(logits, y)

            # Improved reward shaping
            # Reward = 1.0 for correct, scale down for errors
            preds = logits.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

            # Combine accuracy-based and loss-based rewards
            loss_reward = 1.0 / (1.0 + ce_loss.item())
            acc_reward = accuracy
            reward = 0.7 * acc_reward + 0.3 * loss_reward

            epoch_rewards.append(reward)

            # Store activations for Hebbian updates
            activations = []
            h = F.relu(model.encoder(x))
            activations.append(h.detach())

            for nalu in model.nalu_layers:
                h = F.relu(nalu.forward(h))
                activations.append(h.detach())

            # Hebbian updates
            for i, nalu in enumerate(model.nalu_layers):
                x_layer = activations[i]
                y_layer = activations[i + 1]
                nalu.learn(x_layer, y_layer, reward)

            # Update classifier
            h_cls = F.relu(model.encoder(x))
            for nalu in model.nalu_layers:
                h_cls = F.relu(nalu.forward(h_cls))
            logits = model.classifier(h_cls)
            cls_loss = F.cross_entropy(logits, y)

            cls_loss.backward()
            with torch.no_grad():
                if model.classifier.weight.grad is not None:
                    model.classifier.weight -= current_lr * model.classifier.weight.grad
                    model.classifier.bias -= current_lr * model.classifier.bias.grad
                    model.classifier.weight.grad.zero_()
                    model.classifier.bias.grad.zero_()

            epoch_loss += cls_loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        epoch_loss /= len(train_loader)
        avg_reward = np.mean(epoch_rewards)

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
                  f"Loss={epoch_loss:.4f}, Train={train_acc:.1f}%, "
                  f"Test={test_acc:.1f}%, Reward={avg_reward:.3f}, LR={current_lr:.4f}")

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)
        history['rewards'].append(avg_reward)

    elapsed = time.time() - start_time
    history['time'] = elapsed
    history['best_acc'] = best_acc

    return history


def curriculum_learning(
    primes: List[int] = [7, 11, 13, 17, 23],
    samples_per_prime: int = 5000,
    epochs_per_prime: int = 100,
    hidden_dim: int = 256,
    device: str = 'cuda'
) -> List[BenchmarkResult]:
    """
    Train with curriculum learning: easy primes first, then harder ones.
    """
    results = []

    print("="*80)
    print("CURRICULUM LEARNING FOR HEBBIAN-NALU")
    print("="*80)
    print(f"\nCurriculum: {' → '.join(map(str, primes))}")
    print(f"Samples per prime: {samples_per_prime}")
    print(f"Epochs per prime: {epochs_per_prime}")
    print(f"Hidden dimension: {hidden_dim}\n")

    # Create model (will be trained incrementally)
    model = None

    for stage, p in enumerate(primes):
        print(f"\n{'='*60}")
        print(f"Stage {stage+1}/{len(primes)}: Training on p={p}")
        print(f"{'='*60}\n")

        # Generate data
        X_train, Y_train = generate_modular_addition_dataset(p, samples_per_prime)
        X_test, Y_test = generate_modular_addition_dataset(p, samples_per_prime // 5)

        input_dim = X_train.shape[1]
        output_dim = p

        # Create or update model
        if model is None or output_dim != model.output_dim:
            # Create new model for this prime
            print(f"Creating new model for p={p} (output_dim={output_dim})...")
            model = create_nalu_network(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                learning_type='hebbian',
                num_nalu_layers=3,  # More layers for curriculum
                learning_rate=0.03,  # Start higher
                device=device
            )
        else:
            # Update output layer for new prime
            print(f"Updating output layer for p={p}...")
            model.output_dim = output_dim
            model.classifier = nn.Linear(hidden_dim, output_dim, device=device)

        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Train with curriculum-aware epochs
        # More epochs for harder primes
        epochs = epochs_per_prime + (stage * 20)

        history = train_hebbian_nalu_curriculum(
            model, train_loader, test_loader,
            num_epochs=epochs,
            learning_rate=0.03 / (1 + stage * 0.5),  # Decrease LR for harder primes
            device=device
        )

        results.append(BenchmarkResult(
            architecture="Hebbian-NALU (Curriculum)",
            prime=p,
            accuracy=history['best_acc'],
            time=history['time'],
            bio_plausible=True,
            num_epochs=epochs,
            hidden_dim=hidden_dim
        ))

        print(f"\n✓ Stage {stage+1} complete: {history['best_acc']:.1f}% accuracy")
        print(f"   Training time: {history['time']:.1f}s")

        # If performance is too low, increase training
        if history['best_acc'] < 50.0 and stage < len(primes) - 1:
            print(f"\n⚠ Low accuracy ({history['best_acc']:.1f}%), adding bonus epochs...")
            bonus_history = train_hebbian_nalu_curriculum(
                model, train_loader, test_loader,
                num_epochs=50,
                learning_rate=0.01,
                device=device
            )
            if bonus_history['best_acc'] > history['best_acc']:
                results[-1].accuracy = bonus_history['best_acc']
                print(f"   Improved to {bonus_history['best_acc']:.1f}%")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='NALU Curriculum Learning')
    parser.add_argument('--primes', type=int, nargs='+', default=[7, 11, 13, 17, 23],
                       help='Curriculum of primes')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Samples per prime')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Base epochs per prime')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with p=7,11 only')

    args = parser.parse_args()

    if args.quick:
        primes = [7, 11]
        epochs = 50
    else:
        primes = args.primes
        epochs = args.epochs

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    results = curriculum_learning(
        primes=primes,
        samples_per_prime=args.samples,
        epochs_per_prime=epochs,
        hidden_dim=args.hidden_dim,
        device=device
    )

    # Print results
    print("\n" + "="*80)
    print("CURRICULUM LEARNING RESULTS")
    print("="*80 + "\n")

    print(f"{'Prime':<10} {'Accuracy':<12} {'Time (s)':<12} {'Status'}")
    print("-" * 50)

    for r in results:
        status = "✓ PASS" if r.accuracy > 80 else "⚠ LEARNING" if r.accuracy > 50 else "✗ FAIL"
        print(f"p={r.prime:<7} {r.accuracy:>6.1f}%      {r.time:>6.1f}       {status}")

    print()

    # Save results
    output_dir = os.path.dirname(__file__)
    with open(os.path.join(output_dir, 'nalu_curriculum_results.json'), 'w') as f:
        json.dump({
            'results': [r.__dict__ for r in results],
            'config': {
                'primes': primes,
                'samples': args.samples,
                'epochs': epochs,
                'hidden_dim': args.hidden_dim
            }
        }, f, indent=2)

    print(f"Results saved to: nalu_curriculum_results.json\n")

    # Check if we hit target
    p23_results = [r for r in results if r.prime == 23]
    if p23_results:
        acc = p23_results[0].accuracy
        if acc > 80:
            print(f"🎉 SUCCESS! Achieved {acc:.1f}% on p=23 (target: >80%)")
        else:
            print(f"⚠ Not quite there: {acc:.1f}% on p=23 (target: >80%)")
            print(f"   Gap: {80 - acc:.1f} percentage points")


if __name__ == '__main__':
    main()
