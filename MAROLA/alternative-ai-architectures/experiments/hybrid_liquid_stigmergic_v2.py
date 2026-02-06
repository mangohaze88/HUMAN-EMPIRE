#!/usr/bin/env python3
"""
HYBRID LIQUID-STIGMERGIC ARITHMETIC v2.0
========================================

Enhanced version with:
1. Curriculum learning (identity → successor → full addition)
2. Deeper consensus module (non-linear fusion)
3. More sophisticated stigmergic agents
4. Auxiliary learning tasks
5. Better exploration strategies

This version addresses the failures of v1.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import time
from dataclasses import dataclass
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.liquid_neural_network import (
    LiquidNeuralNetwork,
    NCPWiringConfig
)


# ============================================================================
# ENCODING (same as v1)
# ============================================================================

def combined_encoding(a: int, b: int, p: int, bits: int = 10) -> np.ndarray:
    """Combined binary + normalized + cyclic encoding"""
    features = []

    # Binary encoding
    for val in [a, b]:
        features.extend([(val >> i) & 1 for i in range(bits)])

    # Normalized
    features.extend([a / p, b / p])

    # Cyclic (KEY for modular arithmetic!)
    features.extend([
        np.sin(2 * np.pi * a / p),
        np.cos(2 * np.pi * a / p),
        np.sin(2 * np.pi * b / p),
        np.cos(2 * np.pi * b / p),
    ])

    return np.array(features, dtype=np.float32)


# ============================================================================
# ENHANCED COMPONENT 1: LIQUID ENCODER WITH AUXILIARY TASKS
# ============================================================================

class EnhancedLiquidEncoder:
    """
    LNN encoder with auxiliary learning tasks to guide feature learning.

    Auxiliary tasks:
    - Predict if result > p (needs wrap detection)
    - Predict parity of result
    - Predict magnitude range
    """

    def __init__(self, input_dim: int, hidden_dim: int, p: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p

        # Main LNN
        wiring_config = NCPWiringConfig(
            n_sensory=12,  # More sensory neurons
            n_inter=20,    # More interneurons
            n_command=12,  # More command neurons
            n_motor=hidden_dim,
        )

        self.lnn = LiquidNeuralNetwork(
            input_dim=input_dim,
            output_dim=hidden_dim,
            wiring_config=wiring_config,
            tau_base=1.0,
            tau_range=3.0,
            dt=0.1,
            ode_steps=7,  # More ODE steps for richer dynamics
            learning_rate=0.01,
            use_cfc=False,
        )

        # Auxiliary prediction heads (simple linear)
        self.aux_wrap_weights = np.random.randn(hidden_dim) * 0.1  # Predict wrap-around
        self.aux_parity_weights = np.random.randn(hidden_dim) * 0.1  # Predict parity

        print(f"EnhancedLiquidEncoder: {self.lnn.n_neurons} neurons → {hidden_dim}D features")
        print(f"  + Auxiliary tasks: wrap detection, parity prediction")

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features with auxiliary predictions"""
        # Run LNN multiple times
        for _ in range(5):
            output, info = self.lnn.forward(x, return_states=False)

        features = output

        # Auxiliary predictions
        wrap_pred = 1 / (1 + np.exp(-np.dot(self.aux_wrap_weights, features)))
        parity_pred = 1 / (1 + np.exp(-np.dot(self.aux_parity_weights, features)))

        info['wrap_pred'] = wrap_pred
        info['parity_pred'] = parity_pred

        return features, info

    def learn(self, target: np.ndarray, a: int, b: int, correct_result: int):
        """Learn with main target + auxiliary tasks"""
        # Main learning
        self.lnn.learn(target)

        # Auxiliary learning
        # 1. Wrap detection: did (a + b) wrap around?
        wrap_target = 1.0 if (a + b) >= self.p else 0.0
        features, info = self.encode(combined_encoding(a, b, self.p))

        wrap_error = wrap_target - info['wrap_pred']
        self.aux_wrap_weights += 0.01 * wrap_error * features
        self.aux_wrap_weights = np.clip(self.aux_wrap_weights, -5, 5)

        # 2. Parity prediction
        parity_target = float(correct_result % 2)
        parity_error = parity_target - info['parity_pred']
        self.aux_parity_weights += 0.01 * parity_error * features
        self.aux_parity_weights = np.clip(self.aux_parity_weights, -5, 5)

    def reset(self):
        self.lnn.reset_state()


# ============================================================================
# ENHANCED COMPONENT 2: DEEP STIGMERGIC AGENTS
# ============================================================================

class DeepStigmergicAgent:
    """
    More sophisticated agent with non-linear policy.
    """

    def __init__(self, feature_dim: int, p: int):
        self.feature_dim = feature_dim
        self.p = p

        # 2-layer MLP policy
        self.W1 = np.random.randn(feature_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, p) * 0.1
        self.b2 = np.zeros(p)

        self.learning_rate = 0.01

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict distribution over results"""
        # Hidden layer
        h = np.tanh(np.dot(features, self.W1) + self.b1)

        # Output layer
        logits = np.dot(h, self.W2) + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (exp_logits.sum() + 1e-8)

    def learn(self, features: np.ndarray, correct_result: int):
        """Update policy with cross-entropy loss"""
        # Forward pass
        h = np.tanh(np.dot(features, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        probs = np.exp(logits - np.max(logits))
        probs = probs / (probs.sum() + 1e-8)

        # Target: one-hot
        target = np.zeros(self.p)
        target[correct_result] = 1.0

        # Gradient (simplified)
        d_logits = probs - target

        # Backprop to hidden
        d_h = np.dot(d_logits, self.W2.T) * (1 - h**2)

        # Update W2, b2
        self.W2 -= self.learning_rate * np.outer(h, d_logits)
        self.b2 -= self.learning_rate * d_logits

        # Update W1, b1
        self.W1 -= self.learning_rate * np.outer(features, d_h)
        self.b1 -= self.learning_rate * d_h

        # Clip
        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)


class EnhancedStigmergicSearch:
    """Enhanced stigmergic search with deep agents"""

    def __init__(self, p: int, n_agents: int = 100, feature_dim: int = 32):
        self.p = p
        self.n_agents = n_agents
        self.feature_dim = feature_dim

        # Create deep agents
        self.agents = [DeepStigmergicAgent(feature_dim, p) for _ in range(n_agents)]

        # Pheromones
        self.pheromones = np.ones(p) / p
        self.decay_rate = 0.98

        # Exploration schedule (anneal over time)
        self.exploration_rate = 0.5
        self.min_exploration = 0.1

        print(f"EnhancedStigmergicSearch: {n_agents} deep agents exploring [0, {p})")

    def search(self, liquid_features: np.ndarray, explore: bool = True) -> np.ndarray:
        """Aggregate votes from all agents"""
        votes = np.zeros(self.p)

        features = liquid_features[:self.feature_dim]
        if np.linalg.norm(features) > 0:
            features = features / (np.linalg.norm(features) + 1e-8)

        for agent in self.agents:
            # Agent prediction
            agent_probs = agent.predict(features)

            # Exploration: add noise
            if explore and np.random.random() < self.exploration_rate:
                noise = np.random.dirichlet(np.ones(self.p) * 0.1)
                agent_probs = 0.7 * agent_probs + 0.3 * noise

            # Weight by pheromones
            agent_probs *= (1 + self.pheromones)

            votes += agent_probs

        # Normalize
        return votes / (votes.sum() + 1e-8)

    def update_pheromones(self, correct_result: int, was_correct: bool):
        """Update pheromone trails"""
        # Decay
        self.pheromones *= self.decay_rate

        # Reinforce
        if was_correct:
            self.pheromones[correct_result] += 2.0
        else:
            self.pheromones[correct_result] += 0.3

        # Normalize
        self.pheromones = np.clip(self.pheromones, 0.01, 10.0)

    def learn(self, liquid_features: np.ndarray, correct_result: int):
        """Update all agents"""
        features = liquid_features[:self.feature_dim]
        if np.linalg.norm(features) > 0:
            features = features / (np.linalg.norm(features) + 1e-8)

        # Update random subset of agents (faster)
        for agent in np.random.choice(self.agents, size=min(20, self.n_agents), replace=False):
            agent.learn(features, correct_result)

    def anneal_exploration(self):
        """Reduce exploration over time"""
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * 0.99)


# ============================================================================
# ENHANCED COMPONENT 3: DEEP CONSENSUS MODULE
# ============================================================================

class DeepConsensusModule:
    """Non-linear fusion of liquid and stigmergic signals"""

    def __init__(self, liquid_dim: int, p: int):
        self.liquid_dim = liquid_dim
        self.p = p

        # Deep fusion network
        input_dim = liquid_dim + p
        hidden_dim = 128

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 64) * 0.1
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, p) * 0.1
        self.b3 = np.zeros(p)

        self.learning_rate = 0.01

        print(f"DeepConsensusModule: {input_dim}D → {hidden_dim} → 64 → {p}D")

    def forward(self, liquid_features: np.ndarray,
                stigmergic_votes: np.ndarray) -> np.ndarray:
        """Deep fusion forward pass"""
        # Concatenate inputs
        x = np.concatenate([liquid_features[:self.liquid_dim], stigmergic_votes])

        # Layer 1
        h1 = np.tanh(np.dot(x, self.W1) + self.b1)

        # Layer 2
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)

        # Layer 3 (output)
        logits = np.dot(h2, self.W3) + self.b3

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (exp_logits.sum() + 1e-8)

    def learn(self, liquid_features: np.ndarray, stigmergic_votes: np.ndarray,
              correct_result: int):
        """Update with cross-entropy loss"""
        # Forward
        x = np.concatenate([liquid_features[:self.liquid_dim], stigmergic_votes])

        h1 = np.tanh(np.dot(x, self.W1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)
        logits = np.dot(h2, self.W3) + self.b3

        probs = np.exp(logits - np.max(logits))
        probs = probs / (probs.sum() + 1e-8)

        # Target
        target = np.zeros(self.p)
        target[correct_result] = 1.0

        # Backward (simplified)
        d_logits = probs - target

        d_h2 = np.dot(d_logits, self.W3.T) * (1 - h2**2)
        d_h1 = np.dot(d_h2, self.W2.T) * (1 - h1**2)

        # Update weights
        self.W3 -= self.learning_rate * np.outer(h2, d_logits)
        self.b3 -= self.learning_rate * d_logits

        self.W2 -= self.learning_rate * np.outer(h1, d_h2)
        self.b2 -= self.learning_rate * d_h2

        self.W1 -= self.learning_rate * np.outer(x, d_h1)
        self.b1 -= self.learning_rate * d_h1

        # Clip
        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)
        self.W3 = np.clip(self.W3, -5, 5)


# ============================================================================
# ENHANCED HYBRID WITH CURRICULUM LEARNING
# ============================================================================

class EnhancedHybridNetwork:
    """Enhanced hybrid with curriculum learning"""

    def __init__(self, p: int, input_dim: int = 26, liquid_dim: int = 32,
                 n_agents: int = 100):
        self.p = p
        self.input_dim = input_dim
        self.liquid_dim = liquid_dim
        self.n_agents = n_agents

        print(f"\n{'='*70}")
        print(f"ENHANCED HYBRID NETWORK v2.0: mod {p}")
        print(f"{'='*70}")

        self.liquid_encoder = EnhancedLiquidEncoder(input_dim, liquid_dim, p)
        self.stigmergic_search = EnhancedStigmergicSearch(p, n_agents, liquid_dim)
        self.consensus = DeepConsensusModule(liquid_dim, p)

        # Curriculum stage
        self.curriculum_stage = 0  # 0=identity, 1=successor, 2=small, 3=full

    def forward(self, a: int, b: int) -> Tuple[int, Dict[str, Any]]:
        """Forward pass"""
        x = combined_encoding(a, b, self.p)

        liquid_features, lnn_info = self.liquid_encoder.encode(x)
        stigmergic_votes = self.stigmergic_search.search(liquid_features)
        final_probabilities = self.consensus.forward(liquid_features, stigmergic_votes)

        prediction = np.argmax(final_probabilities)
        confidence = final_probabilities[prediction]

        info = {
            'liquid_features': liquid_features,
            'stigmergic_votes': stigmergic_votes,
            'final_probabilities': final_probabilities,
            'confidence': float(confidence),
            'curriculum_stage': self.curriculum_stage,
        }

        return int(prediction), info

    def learn(self, a: int, b: int, correct_result: int,
              liquid_features: np.ndarray, stigmergic_votes: np.ndarray,
              final_probabilities: np.ndarray):
        """Update all components"""
        # LNN target
        target_encoding = np.array([
            correct_result / self.p,
            np.sin(2 * np.pi * correct_result / self.p),
            np.cos(2 * np.pi * correct_result / self.p),
        ])
        target = np.zeros(self.liquid_dim)
        target[:len(target_encoding)] = target_encoding

        self.liquid_encoder.learn(target, a, b, correct_result)

        # Stigmergic
        was_correct = (np.argmax(final_probabilities) == correct_result)
        self.stigmergic_search.update_pheromones(correct_result, was_correct)
        self.stigmergic_search.learn(liquid_features, correct_result)

        # Consensus
        self.consensus.learn(liquid_features, stigmergic_votes, correct_result)

    def reset(self):
        self.liquid_encoder.reset()


def train_with_curriculum(p: int, n_epochs: int = 150):
    """
    Train with curriculum learning:
    1. Stage 0: Identity (a, 0) → a
    2. Stage 1: Successor (a, 1) → (a+1) mod p
    3. Stage 2: Small sums b <= 3
    4. Stage 3: Full problem
    """
    print(f"\n{'='*70}")
    print(f"CURRICULUM TRAINING: mod {p}")
    print(f"{'='*70}\n")

    model = EnhancedHybridNetwork(p=p)

    stages = [
        {'name': 'Identity', 'epochs': 30, 'b_values': [0]},
        {'name': 'Successor', 'epochs': 30, 'b_values': [1]},
        {'name': 'Small sums', 'epochs': 40, 'b_values': list(range(4))},
        {'name': 'Full', 'epochs': 50, 'b_values': list(range(p))},
    ]

    results = {'stages': [], 'final_accuracy': 0.0}

    for stage_idx, stage in enumerate(stages):
        print(f"\n--- Stage {stage_idx}: {stage['name']} ---")
        model.curriculum_stage = stage_idx

        stage_correct = 0
        stage_total = 0

        for epoch in range(stage['epochs']):
            epoch_correct = 0
            epoch_total = 0

            for _ in range(50):  # 50 problems per epoch
                a = np.random.randint(0, p)
                b = np.random.choice(stage['b_values'])
                correct_result = (a + b) % p

                prediction, info = model.forward(a, b)

                if prediction == correct_result:
                    epoch_correct += 1
                epoch_total += 1

                model.learn(a, b, correct_result,
                           info['liquid_features'],
                           info['stigmergic_votes'],
                           info['final_probabilities'])

            # Anneal exploration
            model.stigmergic_search.anneal_exploration()

            stage_correct += epoch_correct
            stage_total += epoch_total

            if epoch % 10 == 0:
                acc = epoch_correct / epoch_total
                print(f"  Epoch {epoch}: accuracy={acc:.3f}, "
                      f"explore={model.stigmergic_search.exploration_rate:.3f}")

        stage_acc = stage_correct / stage_total
        results['stages'].append({
            'name': stage['name'],
            'accuracy': stage_acc
        })
        print(f"Stage {stage_idx} complete: accuracy={stage_acc:.3f}")

    results['final_accuracy'] = results['stages'][-1]['accuracy']
    return model, results


def main():
    """Test enhanced hybrid on small primes"""

    print("\n" + "="*70)
    print("ENHANCED HYBRID v2.0: WITH CURRICULUM & DEEP MODULES")
    print("="*70)

    primes = [7, 11]
    all_results = {}

    for p in primes:
        model, results = train_with_curriculum(p, n_epochs=150)

        # Test
        model.reset()
        test_correct = 0
        for _ in range(500):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            correct_result = (a + b) % p

            prediction, info = model.forward(a, b)
            if prediction == correct_result:
                test_correct += 1

        test_acc = test_correct / 500
        results['test_accuracy'] = test_acc

        all_results[f'p{p}'] = results

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS: mod {p}")
        print(f"{'='*70}")
        for stage in results['stages']:
            print(f"  {stage['name']:<15}: {stage['accuracy']:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")

    # Save
    with open('enhanced_hybrid_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("Enhanced hybrid v2.0 complete!")
    print("Results saved to: enhanced_hybrid_results.json")
    print(f"{'='*70}\n")

    return all_results


if __name__ == '__main__':
    main()
