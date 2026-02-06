#!/usr/bin/env python3
"""
Quick component test for bio-plausible networks.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.forward_forward import ForwardForwardNetwork
from src.networks.liquid_neural_network import LiquidNeuralNetwork, NCPWiringConfig

print("="*70)
print("TESTING BIO-PLAUSIBLE COMPONENTS")
print("="*70)

# Test encoding
def combined_encoding(a, b, p, bits=10):
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

# Test for p=7
p = 7
x = combined_encoding(3, 4, p)
print(f"\n1. Encoding test for (3+4) mod 7:")
print(f"   Input dimension: {len(x)}")
print(f"   Sample: {x[:5]}...")

# Test Forward-Forward
print(f"\n2. Testing Forward-Forward Network...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

ff_net = ForwardForwardNetwork(
    input_dim=len(x),
    hidden_dims=[64, 64],
    output_dim=p,
    threshold=2.0,
    learning_rate=0.05,
    device=device
)

print(f"   Network created: {len(x)} -> [64, 64] -> {p}")

# Test training step
x_batch = torch.randn(16, len(x)).to(device)
y_batch = torch.randint(0, p, (16,)).to(device)

metrics = ff_net.train_step(x_batch, y_batch)
print(f"   Training step: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

# Test prediction
pred = ff_net.predict(x_batch[:1])
print(f"   Prediction works: {pred.item()}")

print("   ✓ Forward-Forward OK")

# Test Liquid Neural Network (CPU)
print(f"\n3. Testing Liquid Neural Network (CPU)...")

config = NCPWiringConfig(
    n_sensory=8,
    n_inter=12,
    n_command=4,
    n_motor=3  # [normalized, sin, cos]
)

lnn = LiquidNeuralNetwork(
    input_dim=len(x),
    output_dim=3,
    wiring_config=config,
    dt=0.1,
    ode_steps=2,
    learning_rate=0.01
)

print(f"   Network created with {lnn.n_neurons} neurons")

# Test forward pass
output, info = lnn.forward(x)
print(f"   Forward pass: output shape={output.shape}")
print(f"   Adaptation rate: {info['adaptation_rate']:.4f}")

# Test learning
target = np.array([0.5, 0.1, 0.9])
lnn.learn(target)
print(f"   Learning step completed")

print("   ✓ Liquid Network OK")

print("\n" + "="*70)
print("ALL COMPONENTS WORKING!")
print("="*70)
print("\nReady to run full benchmark:")
print("  python experiments/learn_ec_math_bio_plausible.py --quick")
print("  python experiments/learn_ec_math_bio_plausible.py  # Full benchmark")
