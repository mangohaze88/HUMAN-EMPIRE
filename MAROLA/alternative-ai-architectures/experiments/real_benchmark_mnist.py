#!/usr/bin/env python3
"""
REAL BENCHMARK - MNIST Digit Classification
============================================
Test if bio-plausible architectures can learn a REAL task without backpropagation.

This is the TRUTH TEST - can these networks actually learn to recognize handwritten digits?

Architectures tested:
1. Forward-Forward Network (Hinton's local learning)
2. Liquid Neural Network (MIT's continuous-time dynamics)
3. Stigmergic Network (swarm intelligence)
4. CuriosityCore (curiosity-driven learning)
5. MLP with Backprop (baseline for comparison)

Goal: Demonstrate that bio-plausible learning can work on real tasks!
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.forward_forward import ForwardForwardNetwork, create_ff_network
from src.networks.liquid_neural_network import LiquidNeuralNetwork, LiquidNeuralNetworkGPU, NCPWiringConfig
from src.networks.stigmergic_intelligence import StigmergicIntelligenceNetwork
from src.networks.curiosity_core import CuriosityCore

print("Importing MNIST dataset...")
try:
    from torchvision import datasets, transforms
    MNIST_AVAILABLE = True
except ImportError:
    print("WARNING: torchvision not available, will use synthetic data")
    MNIST_AVAILABLE = False


# ============================================================================
# MNIST DATA LOADING
# ============================================================================

def load_mnist_data(data_dir='./data', download=True, num_train=10000, num_test=2000):
    """Load MNIST dataset (or create synthetic data if not available)"""

    if MNIST_AVAILABLE:
        print(f"\nLoading MNIST dataset from {data_dir}...")

        # Download and load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

        # Subsample for faster training
        train_indices = torch.randperm(len(train_dataset))[:num_train]
        test_indices = torch.randperm(len(test_dataset))[:num_test]

        train_data = torch.stack([train_dataset[i][0] for i in train_indices])
        train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])

        test_data = torch.stack([test_dataset[i][0] for i in test_indices])
        test_labels = torch.tensor([test_dataset[i][1] for i in test_indices])

        # Flatten images
        train_data = train_data.view(len(train_data), -1)
        test_data = test_data.view(len(test_data), -1)

        print(f"✓ Loaded {len(train_data)} training samples, {len(test_data)} test samples")
        print(f"  Input dim: {train_data.shape[1]}, Classes: 10")

    else:
        print("\nGenerating synthetic MNIST-like data...")

        # Create synthetic data with 10 patterns
        input_dim = 784
        num_classes = 10

        # Generate prototypes for each digit
        prototypes = []
        for c in range(num_classes):
            np.random.seed(c * 42)
            prototype = np.random.randn(input_dim) * 0.3
            # Add some structure
            prototype[c * 70:(c + 1) * 70] += 2.0
            prototypes.append(prototype)

        # Generate training data
        train_data = []
        train_labels = []
        for c in range(num_classes):
            n_samples = num_train // num_classes
            for _ in range(n_samples):
                # Add noise to prototype
                sample = prototypes[c] + np.random.randn(input_dim) * 0.5
                train_data.append(sample)
                train_labels.append(c)

        # Generate test data
        test_data = []
        test_labels = []
        for c in range(num_classes):
            n_samples = num_test // num_classes
            for _ in range(n_samples):
                sample = prototypes[c] + np.random.randn(input_dim) * 0.6
                test_data.append(sample)
                test_labels.append(c)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        print(f"✓ Generated {len(train_data)} training samples, {len(test_data)} test samples")

    return train_data, train_labels, test_data, test_labels


# ============================================================================
# BASELINE: STANDARD MLP WITH BACKPROP
# ============================================================================

class SimpleMLPBackprop(nn.Module):
    """Standard MLP with backpropagation for comparison"""

    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp_baseline(train_data, train_labels, test_data, test_labels,
                        epochs=10, batch_size=64, device='cpu'):
    """Train standard MLP with backprop (baseline)"""

    print("\n" + "="*70)
    print("BASELINE: Standard MLP with Backpropagation")
    print("="*70)

    model = SimpleMLPBackprop(input_dim=784, hidden_dims=[256, 128], output_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        indices = torch.randperm(len(train_data))

        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = train_data[batch_indices]
            batch_y = train_labels[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += len(batch_y)

        train_acc = 100.0 * correct / total

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            test_predictions = test_outputs.argmax(dim=1)
            test_acc = 100.0 * (test_predictions == test_labels).float().mean().item()

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_data)
        train_predictions = train_outputs.argmax(dim=1)
        final_train_acc = 100.0 * (train_predictions == train_labels).float().mean().item()

        test_outputs = model(test_data)
        test_predictions = test_outputs.argmax(dim=1)
        final_test_acc = 100.0 * (test_predictions == test_labels).float().mean().item()

    return {
        'name': 'MLP (Backprop)',
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'time': train_time,
        'uses_backprop': True,
        'params': sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# BIO-PLAUSIBLE ARCHITECTURE 1: FORWARD-FORWARD
# ============================================================================

def train_forward_forward(train_data, train_labels, test_data, test_labels,
                          epochs=10, batch_size=64, device='cpu'):
    """Train Forward-Forward network (NO backprop!)"""

    print("\n" + "="*70)
    print("BIO-PLAUSIBLE: Forward-Forward Network")
    print("="*70)
    print("KEY: Local learning, contrastive positive/negative samples")
    print("     NO backpropagation!")

    model = create_ff_network(
        input_dim=784,
        hidden_dims=[500, 300],
        output_dim=10,
        use_gpu=(device == 'cuda'),
        threshold=2.0,
        learning_rate=0.03,
        activation='relu',
        negative_strategy='hybrid',
        noise_std=0.4,
    )

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle training data
        indices = torch.randperm(len(train_data))

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = train_data[batch_indices]
            batch_y = train_labels[batch_indices]

            # Forward-Forward training step (NO backprop!)
            metrics = model.train_step(batch_x, batch_y, return_metrics=True)
            epoch_loss += metrics['loss']
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate accuracy every epoch
        with torch.no_grad():
            # Subsample for speed (FF prediction is slow)
            test_sample_indices = torch.randperm(len(test_data))[:500]
            test_sample_x = test_data[test_sample_indices]
            test_sample_y = test_labels[test_sample_indices]

            test_acc = model.compute_accuracy(test_sample_x, test_sample_y)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.3f}, Test Acc (sample)={test_acc*100:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    with torch.no_grad():
        # Train accuracy (sample for speed)
        train_sample_indices = torch.randperm(len(train_data))[:1000]
        train_sample_x = train_data[train_sample_indices]
        train_sample_y = train_labels[train_sample_indices]
        final_train_acc = model.compute_accuracy(train_sample_x, train_sample_y) * 100

        # Test accuracy (full for final result)
        final_test_acc = model.compute_accuracy(test_data, test_labels) * 100

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    return {
        'name': 'Forward-Forward',
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'time': train_time,
        'uses_backprop': False,
        'params': n_params,
    }


# ============================================================================
# BIO-PLAUSIBLE ARCHITECTURE 2: LIQUID NEURAL NETWORK
# ============================================================================

def train_liquid_network(train_data, train_labels, test_data, test_labels,
                         epochs=20, batch_size=1, device='cpu'):
    """Train Liquid Neural Network (continuous-time, NO backprop!)"""

    print("\n" + "="*70)
    print("BIO-PLAUSIBLE: Liquid Neural Network")
    print("="*70)
    print("KEY: Continuous-time dynamics, adaptive time constants")
    print("     Local learning rules, NO backpropagation!")

    # Convert to numpy for CPU version
    train_data_np = train_data.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_data_np = test_data.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Create liquid network with NCP wiring
    config = NCPWiringConfig(
        n_sensory=32,
        n_inter=48,
        n_command=16,
        n_motor=10,
    )

    model = LiquidNeuralNetwork(
        input_dim=784,
        output_dim=10,
        wiring_config=config,
        tau_base=1.0,
        tau_range=3.0,
        dt=0.1,
        ode_steps=3,
        learning_rate=0.01,
        use_cfc=False,
    )

    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_data_np))

        correct = 0
        total = 0

        for i in range(0, len(train_data_np), batch_size):
            batch_indices = indices[i:i+batch_size]

            for idx in batch_indices:
                x = train_data_np[idx]
                y_true = train_labels_np[idx]

                # Reset state for each sample
                model.reset_state()

                # Forward pass
                output, info = model.forward(x)

                # Create one-hot target
                target = np.zeros(10)
                target[y_true] = 1.0

                # Learn (local updates, NO backprop!)
                model.learn(target, error_weight=1.0)

                # Track accuracy
                prediction = np.argmax(output)
                if prediction == y_true:
                    correct += 1
                total += 1

        train_acc = 100.0 * correct / total

        # Evaluate on test set (sample for speed)
        test_correct = 0
        test_total = 0
        test_sample_size = min(500, len(test_data_np))
        test_indices = np.random.choice(len(test_data_np), test_sample_size, replace=False)

        for idx in test_indices:
            model.reset_state()
            x = test_data_np[idx]
            y_true = test_labels_np[idx]

            output, _ = model.forward(x)
            prediction = np.argmax(output)

            if prediction == y_true:
                test_correct += 1
            test_total += 1

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    train_correct = 0
    train_sample_size = min(1000, len(train_data_np))
    train_indices = np.random.choice(len(train_data_np), train_sample_size, replace=False)

    for idx in train_indices:
        model.reset_state()
        output, _ = model.forward(train_data_np[idx])
        if np.argmax(output) == train_labels_np[idx]:
            train_correct += 1

    final_train_acc = 100.0 * train_correct / train_sample_size

    test_correct = 0
    for idx in range(len(test_data_np)):
        model.reset_state()
        output, _ = model.forward(test_data_np[idx], n_steps=5, learn=False)
        if np.argmax(output) == test_labels_np[idx]:
            test_correct += 1

    final_test_acc = 100.0 * test_correct / len(test_data_np)

    # Estimate parameters
    n_params = model.n_neurons * (784 + model.n_neurons + 10)

    return {
        'name': 'Liquid Neural Net',
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'time': train_time,
        'uses_backprop': False,
        'params': n_params,
    }


# ============================================================================
# BIO-PLAUSIBLE ARCHITECTURE 3: STIGMERGIC NETWORK
# ============================================================================

def train_stigmergic_network(train_data, train_labels, test_data, test_labels,
                             epochs=15, batch_size=1, device='cpu'):
    """Train Stigmergic Network (swarm intelligence, NO backprop!)"""

    print("\n" + "="*70)
    print("BIO-PLAUSIBLE: Stigmergic Swarm Network")
    print("="*70)
    print("KEY: Swarm of agents communicating via pheromones")
    print("     Three-factor learning, NO backpropagation!")

    # Convert to numpy
    train_data_np = train_data.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_data_np = test_data.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Create stigmergic network
    model = StigmergicIntelligenceNetwork(
        n_agents=64,
        env_shape=(32, 32),
        feature_dim=16,
        input_dim=784,
        output_dim=10,
        device='cpu',
    )

    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_data_np))

        correct = 0
        total = 0

        for i in range(0, len(train_data_np), batch_size):
            batch_indices = indices[i:i+batch_size]

            for idx in batch_indices:
                x = train_data_np[idx]
                y_true = train_labels_np[idx]

                # Stigmergic forward already includes learning
                # forward(input_data, n_steps=10, learn=True)
                output, info = model.forward(x, n_steps=5, learn=True)

                # Track accuracy
                prediction = np.argmax(output)
                if prediction == y_true:
                    correct += 1
                total += 1

        train_acc = 100.0 * correct / total

        # Test accuracy (sample)
        test_correct = 0
        test_total = 0
        test_sample_size = min(500, len(test_data_np))
        test_indices = np.random.choice(len(test_data_np), test_sample_size, replace=False)

        for idx in test_indices:
            output, _ = model.forward(test_data_np[idx], n_steps=5, learn=False)
            if np.argmax(output) == test_labels_np[idx]:
                test_correct += 1
            test_total += 1

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    train_correct = 0
    train_sample_size = min(1000, len(train_data_np))
    train_indices = np.random.choice(len(train_data_np), train_sample_size, replace=False)

    for idx in train_indices:
        output, _ = model.forward(train_data_np[idx])
        if np.argmax(output) == train_labels_np[idx]:
            train_correct += 1

    final_train_acc = 100.0 * train_correct / train_sample_size

    test_correct = 0
    for idx in range(len(test_data_np)):
        output, _ = model.forward(test_data_np[idx], n_steps=5, learn=False)
        if np.argmax(output) == test_labels_np[idx]:
            test_correct += 1

    final_test_acc = 100.0 * test_correct / len(test_data_np)

    # Estimate parameters
    n_params = model.n_agents * (784 + 10) * 2

    return {
        'name': 'Stigmergic Swarm',
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'time': train_time,
        'uses_backprop': False,
        'params': n_params,
    }


# ============================================================================
# BIO-PLAUSIBLE ARCHITECTURE 4: CURIOSITY CORE
# ============================================================================

def train_curiosity_core(train_data, train_labels, test_data, test_labels,
                         epochs=20, batch_size=1, device='cpu'):
    """Train CuriosityCore (curiosity-driven learning, NO backprop!)"""

    print("\n" + "="*70)
    print("BIO-PLAUSIBLE: CuriosityCore")
    print("="*70)
    print("KEY: Intrinsic motivation, curiosity-driven exploration")
    print("     Local prediction learning, NO backpropagation!")

    # Convert to numpy
    train_data_np = train_data.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_data_np = test_data.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Create curiosity core
    model = CuriosityCore(
        sensory_dim=784,
        hidden_dim=128,
        action_dim=10,
        device='cpu',
    )

    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_data_np))

        correct = 0
        total = 0

        for i in range(0, len(train_data_np), batch_size):
            batch_indices = indices[i:i+batch_size]

            for idx in batch_indices:
                x = train_data_np[idx]
                y_true = train_labels_np[idx]

                # Create reward based on correct classification
                # CuriosityCore uses step() method
                info = model.step(x, external_reward=0.0)

                # Get action as output (action_dim = 10)
                action = info['action']

                # Map action to prediction (use action as logits)
                prediction = np.argmax(action)

                # Provide reward if correct
                reward = 1.0 if prediction == y_true else -0.5
                model.update_action_model(reward)

                if prediction == y_true:
                    correct += 1
                total += 1

        train_acc = 100.0 * correct / total

        # Test accuracy (sample)
        test_correct = 0
        test_total = 0
        test_sample_size = min(500, len(test_data_np))
        test_indices = np.random.choice(len(test_data_np), test_sample_size, replace=False)

        for idx in test_indices:
            info = model.step(test_data_np[idx], external_reward=0.0)
            action = info['action']
            if np.argmax(action) == test_labels_np[idx]:
                test_correct += 1
            test_total += 1

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    train_correct = 0
    train_sample_size = min(1000, len(train_data_np))
    train_indices = np.random.choice(len(train_data_np), train_sample_size, replace=False)

    for idx in train_indices:
        info = model.step(train_data_np[idx], external_reward=0.0)
        if np.argmax(info['action']) == train_labels_np[idx]:
            train_correct += 1

    final_train_acc = 100.0 * train_correct / train_sample_size

    test_correct = 0
    for idx in range(len(test_data_np)):
        info = model.step(test_data_np[idx], external_reward=0.0)
        if np.argmax(info['action']) == test_labels_np[idx]:
            test_correct += 1

    final_test_acc = 100.0 * test_correct / len(test_data_np)

    # Estimate parameters
    n_params = 784 * 128 + 128 * 10

    return {
        'name': 'CuriosityCore',
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'time': train_time,
        'uses_backprop': False,
        'params': n_params,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def print_results_table(results):
    """Print formatted results table"""

    print("\n" + "="*90)
    print("MNIST REAL BENCHMARK - BIO-PLAUSIBLE LEARNING RESULTS")
    print("="*90)
    print()
    print(f"{'Architecture':<20} {'Train Acc':<12} {'Test Acc':<12} {'Time':<10} {'Backprop?':<12} {'Params':<12}")
    print("-" * 90)

    for r in results:
        backprop_str = "YES" if r['uses_backprop'] else "NO"
        print(f"{r['name']:<20} {r['train_acc']:>10.1f}% {r['test_acc']:>10.1f}% {r['time']:>8.1f}s {backprop_str:<12} {r['params']:>10,d}")

    print("="*90)
    print()
    print("KEY FINDINGS:")
    print("-" * 90)

    # Find best bio-plausible
    bio_plausible = [r for r in results if not r['uses_backprop']]
    baseline = [r for r in results if r['uses_backprop']][0]

    if bio_plausible:
        best_bio = max(bio_plausible, key=lambda x: x['test_acc'])

        print(f"✓ Best Bio-Plausible: {best_bio['name']}")
        print(f"  Test Accuracy: {best_bio['test_acc']:.1f}%")
        print(f"  Gap from Backprop: {baseline['test_acc'] - best_bio['test_acc']:.1f}%")
        print()

        # Check if any learned above random chance
        random_chance = 10.0  # 10% for 10 classes
        learned = [r for r in bio_plausible if r['test_acc'] > random_chance * 2]

        if learned:
            print(f"✓ {len(learned)}/{len(bio_plausible)} bio-plausible networks LEARNED (>20% accuracy)")
            print("  This proves bio-plausible learning CAN work on real tasks!")
        else:
            print(f"✗ None of the bio-plausible networks learned effectively")
            print(f"  All are near random chance ({random_chance}%)")

        print()
        print("INTERPRETATION:")
        print("-" * 90)

        if best_bio['test_acc'] > 70:
            print("EXCELLENT: Bio-plausible networks can compete with backprop!")
        elif best_bio['test_acc'] > 50:
            print("GOOD: Bio-plausible networks show significant learning.")
        elif best_bio['test_acc'] > 30:
            print("MODERATE: Bio-plausible networks learn but need improvement.")
        elif best_bio['test_acc'] > 20:
            print("LIMITED: Bio-plausible networks show basic learning ability.")
        else:
            print("POOR: Bio-plausible networks struggle with this task.")

        print()
        print(f"Gap to backprop baseline: {baseline['test_acc'] - best_bio['test_acc']:.1f}%")
        print(f"This shows the current cost of bio-plausibility.")


def main():
    """Run full benchmark"""

    print("\n" + "="*90)
    print(" " * 20 + "MNIST REAL BENCHMARK - BIO-PLAUSIBLE LEARNING")
    print("="*90)
    print("\nGoal: Test if bio-plausible architectures can learn real tasks WITHOUT backprop")
    print("Task: Handwritten digit classification (MNIST)")
    print("Random Chance: 10% (10 classes)")
    print()

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(
        num_train=5000,   # Reduced for faster testing
        num_test=1000,
    )

    results = []

    # Run benchmark for each architecture
    try:
        # 1. Baseline (always run this first)
        result = train_mlp_baseline(train_data, train_labels, test_data, test_labels,
                                     epochs=5, device=device)
        results.append(result)

        # 2. Forward-Forward
        try:
            result = train_forward_forward(train_data, train_labels, test_data, test_labels,
                                           epochs=5, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Forward-Forward failed: {e}")

        # 3. Liquid Neural Network
        try:
            result = train_liquid_network(train_data, train_labels, test_data, test_labels,
                                          epochs=10, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Liquid Neural Network failed: {e}")

        # 4. Stigmergic Network
        try:
            result = train_stigmergic_network(train_data, train_labels, test_data, test_labels,
                                              epochs=10, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Stigmergic Network failed: {e}")

        # 5. CuriosityCore
        try:
            result = train_curiosity_core(train_data, train_labels, test_data, test_labels,
                                          epochs=10, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ CuriosityCore failed: {e}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user!")

    # Print results
    if results:
        print_results_table(results)

        # Save results
        output_file = os.path.join(os.path.dirname(__file__), 'mnist_benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results to display!")


if __name__ == '__main__':
    main()
