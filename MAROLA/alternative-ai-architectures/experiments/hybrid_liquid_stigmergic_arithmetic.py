#!/usr/bin/env python3
"""
HYBRID LIQUID-STIGMERGIC ARITHMETIC LEARNING
============================================

Tests the hypothesis: Neither LNN nor Stigmergic alone works for arithmetic,
but COMBINING them might create emergent intelligence!

Architecture:
    Input (a, b, p)
        │
        ▼
    ┌─────────────┐     ┌─────────────────────┐
    │   LIQUID    │────▶│    STIGMERGIC       │
    │   ENCODER   │     │    SEARCH SPACE     │
    │             │     │                     │
    │ ODE dynamics│     │  Ant agents explore │
    │ Time const. │     │  possible results   │
    │ Features    │     │  Pheromone = conf.  │
    └─────────────┘     └─────────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
            ┌───────────────┐
            │   CONSENSUS   │
            │   MODULE      │
            │               │
            │ Combines LNN  │
            │ features with │
            │ ant votes     │
            └───────────────┘
                    │
                    ▼
              Output: result

Key Innovation:
- LNN: Extracts temporal/relational features from (a, b, p)
- Stigmergic: Explores result space [0, p) with ant colony
- Consensus: Combines continuous dynamics with discrete search

Hypothesis: The continuous LNN features guide the discrete search,
while the collective ant intelligence overcomes LNN's arithmetic blindness.
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
    LiquidNeuralNetworkGPU,
    NCPWiringConfig
)


# ============================================================================
# ENCODING (same as successful standard NN experiments)
# ============================================================================

def to_binary(n: int, bits: int = 10) -> np.ndarray:
    """Convert number to binary representation."""
    return np.array([(n >> i) & 1 for i in range(bits)], dtype=np.float32)


def combined_encoding(a: int, b: int, p: int, bits: int = 10) -> np.ndarray:
    """
    The encoding that worked for standard NNs!
    Combines binary, normalized, and cyclic features.
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


# ============================================================================
# COMPONENT 1: LIQUID ENCODER
# ============================================================================

class LiquidEncoder:
    """
    Uses Liquid Neural Network to extract arithmetic-relevant features.

    The LNN's continuous-time dynamics and adaptive time constants help
    discover temporal/relational patterns in the input encoding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, p: int):
        """
        Args:
            input_dim: Size of combined encoding
            hidden_dim: Number of LNN neurons (small! 32-64)
            p: Modulo value (for context)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p

        # Small LNN (liquid magic happens with few neurons!)
        wiring_config = NCPWiringConfig(
            n_sensory=8,
            n_inter=12,
            n_command=8,
            n_motor=hidden_dim,
        )

        self.lnn = LiquidNeuralNetwork(
            input_dim=input_dim,
            output_dim=hidden_dim,
            wiring_config=wiring_config,
            tau_base=1.0,
            tau_range=3.0,
            dt=0.1,
            ode_steps=5,  # Multiple ODE steps to develop rich features
            learning_rate=0.005,
            use_cfc=False,
        )

        print(f"LiquidEncoder: {self.lnn.n_neurons} neurons → {hidden_dim}D features")

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features from input using LNN dynamics.

        Returns:
            features: Rich representation (hidden_dim,)
            info: LNN internal state info
        """
        # Run through LNN multiple times to let dynamics settle
        for _ in range(3):
            output, info = self.lnn.forward(x, return_states=False)

        # Features are the motor neuron outputs
        features = output

        return features, info

    def reset(self):
        """Reset LNN hidden state between problems"""
        self.lnn.reset_state()


# ============================================================================
# COMPONENT 2: STIGMERGIC ARITHMETIC SEARCH
# ============================================================================

class StigmergicArithmeticSearch:
    """
    Ant colony explores possible results [0, p).

    Each ant:
    - Uses liquid features as context
    - Deposits pheromones on guessed results
    - Follows pheromone trails from other ants
    - Learns which guesses correlate with correctness
    """

    def __init__(self, p: int, n_agents: int = 50, feature_dim: int = 32):
        """
        Args:
            p: Modulo value (defines search space [0, p))
            n_agents: Number of ant agents
            feature_dim: Dimension of liquid features
        """
        self.p = p
        self.n_agents = n_agents
        self.feature_dim = feature_dim

        # Pheromone trail: confidence in each possible result
        self.pheromones = np.ones(p) / p  # Start uniform

        # Agent learning: map features → preferred results
        # Each agent has a small weight matrix
        self.agent_weights = np.random.randn(n_agents, feature_dim) * 0.1
        self.agent_biases = np.random.randint(0, p, n_agents)  # Initial preference

        # Decay and learning rates
        self.decay_rate = 0.95  # Pheromone evaporation
        self.learning_rate = 0.1
        self.exploration_rate = 0.3

        print(f"StigmergicSearch: {n_agents} agents exploring [0, {p})")

    def search(self, liquid_features: np.ndarray, a: int, b: int) -> np.ndarray:
        """
        Agents explore result space and vote.

        Returns:
            votes: Probability distribution over [0, p)
        """
        votes = np.zeros(self.p)

        # Normalize features
        features = liquid_features[:self.feature_dim]
        if np.linalg.norm(features) > 0:
            features = features / (np.linalg.norm(features) + 1e-8)

        for agent_idx in range(self.n_agents):
            # Agent uses features + noise to guess
            if np.random.random() < self.exploration_rate:
                # Exploration: random guess
                guess = np.random.randint(0, self.p)
            else:
                # Exploitation: use learned policy
                activation = np.dot(self.agent_weights[agent_idx], features)
                activation += self.agent_biases[agent_idx]

                # Convert to result index
                guess = int(abs(activation) * self.p) % self.p

            # Weight vote by pheromone confidence
            pheromone_strength = self.pheromones[guess]
            votes[guess] += 1.0 + pheromone_strength

        # Normalize to probability distribution
        if votes.sum() > 0:
            votes = votes / votes.sum()
        else:
            votes = np.ones(self.p) / self.p

        return votes

    def update_pheromones(self, correct_result: int, predictions: np.ndarray,
                          was_correct: bool):
        """
        Update pheromone trails based on result.

        Args:
            correct_result: The true (a + b) mod p
            predictions: Probability distribution from search
            was_correct: Whether the consensus was correct
        """
        # Evaporation
        self.pheromones *= self.decay_rate

        # Reinforce correct answer
        if was_correct:
            # Strong reinforcement for correct prediction
            self.pheromones[correct_result] += 2.0
        else:
            # Weak reinforcement to remember correct answer
            self.pheromones[correct_result] += 0.5

        # Penalize confidently wrong predictions
        predicted_result = np.argmax(predictions)
        if predicted_result != correct_result and predictions[predicted_result] > 0.5:
            self.pheromones[predicted_result] *= 0.8

        # Normalize
        self.pheromones = np.clip(self.pheromones, 0.01, 10.0)

    def learn(self, liquid_features: np.ndarray, correct_result: int,
              predictions: np.ndarray):
        """
        Update agent policies using Hebbian-like rule.

        Agents that voted correctly get reinforced.
        """
        features = liquid_features[:self.feature_dim]
        if np.linalg.norm(features) > 0:
            features = features / (np.linalg.norm(features) + 1e-8)

        for agent_idx in range(self.n_agents):
            # Compute what this agent would have guessed
            activation = np.dot(self.agent_weights[agent_idx], features)
            activation += self.agent_biases[agent_idx]
            guess = int(abs(activation) * self.p) % self.p

            # Reward if agent guessed correctly
            if guess == correct_result:
                # Hebbian update: strengthen feature→result connection
                error = correct_result - activation
                self.agent_weights[agent_idx] += self.learning_rate * error * features
                self.agent_biases[agent_idx] = int(
                    0.9 * self.agent_biases[agent_idx] + 0.1 * correct_result
                )

        # Clip weights
        self.agent_weights = np.clip(self.agent_weights, -5, 5)


# ============================================================================
# COMPONENT 3: CONSENSUS MODULE
# ============================================================================

class ConsensusModule:
    """
    Combines LNN features with stigmergic votes to make final decision.

    Uses a simple weighted combination that learns over time.
    """

    def __init__(self, liquid_dim: int, p: int):
        """
        Args:
            liquid_dim: Dimension of liquid features
            p: Modulo value (output space size)
        """
        self.liquid_dim = liquid_dim
        self.p = p

        # Weights for combining signals
        self.W_liquid = np.random.randn(p, liquid_dim) * 0.1
        self.W_stigmergic = np.random.randn(p, p) * 0.1 + np.eye(p) * 0.5
        self.alpha = 0.5  # Balance between LNN and stigmergic (learned)

        self.learning_rate = 0.01

        print(f"ConsensusModule: Combining {liquid_dim}D liquid + {p}D stigmergic")

    def forward(self, liquid_features: np.ndarray,
                stigmergic_votes: np.ndarray) -> np.ndarray:
        """
        Combine features and votes into final prediction.

        Returns:
            probabilities: Distribution over [0, p)
        """
        # Liquid contribution
        liquid_logits = self.W_liquid @ liquid_features[:self.liquid_dim]

        # Stigmergic contribution
        stigmergic_logits = self.W_stigmergic @ stigmergic_votes

        # Combine with learned balance
        combined = self.alpha * liquid_logits + (1 - self.alpha) * stigmergic_logits

        # Softmax to probabilities
        exp_vals = np.exp(combined - np.max(combined))
        probabilities = exp_vals / (exp_vals.sum() + 1e-8)

        return probabilities

    def learn(self, liquid_features: np.ndarray, stigmergic_votes: np.ndarray,
              correct_result: int, prediction: np.ndarray):
        """
        Update consensus weights using local learning rule.

        Simple rule: move weights toward correct result.
        """
        # Target: one-hot for correct result
        target = np.zeros(self.p)
        target[correct_result] = 1.0

        # Error
        error = target - prediction

        # Update liquid weights
        liquid_contrib = self.W_liquid @ liquid_features[:self.liquid_dim]
        stigmergic_contrib = self.W_stigmergic @ stigmergic_votes

        # Local update (simplified gradient)
        self.W_liquid += self.learning_rate * np.outer(error, liquid_features[:self.liquid_dim])
        self.W_stigmergic += self.learning_rate * np.outer(error, stigmergic_votes)

        # Adjust balance based on which component was more useful
        liquid_alignment = np.dot(liquid_contrib, target)
        stigmergic_alignment = np.dot(stigmergic_contrib, target)

        if liquid_alignment > stigmergic_alignment:
            self.alpha = min(0.9, self.alpha + 0.01)
        else:
            self.alpha = max(0.1, self.alpha - 0.01)

        # Clip weights
        self.W_liquid = np.clip(self.W_liquid, -5, 5)
        self.W_stigmergic = np.clip(self.W_stigmergic, -5, 5)


# ============================================================================
# HYBRID ARCHITECTURE
# ============================================================================

class HybridLiquidStigmergicNetwork:
    """
    Complete hybrid architecture for modular arithmetic.

    Pipeline:
        Input → LiquidEncoder → StigmergicSearch → ConsensusModule → Output
    """

    def __init__(self, p: int, input_dim: int = 26, liquid_dim: int = 32,
                 n_agents: int = 50):
        """
        Args:
            p: Modulo value
            input_dim: Size of input encoding
            liquid_dim: LNN feature dimension
            n_agents: Number of stigmergic agents
        """
        self.p = p
        self.input_dim = input_dim
        self.liquid_dim = liquid_dim
        self.n_agents = n_agents

        print(f"\n{'='*70}")
        print(f"HYBRID LIQUID-STIGMERGIC NETWORK: mod {p}")
        print(f"{'='*70}")

        # Components
        self.liquid_encoder = LiquidEncoder(input_dim, liquid_dim, p)
        self.stigmergic_search = StigmergicArithmeticSearch(p, n_agents, liquid_dim)
        self.consensus = ConsensusModule(liquid_dim, p)

        # Stats
        self.history = {
            'accuracy': [],
            'liquid_activity': [],
            'stigmergic_confidence': [],
            'consensus_alpha': [],
        }

    def forward(self, a: int, b: int) -> Tuple[int, Dict[str, Any]]:
        """
        Process one arithmetic problem.

        Args:
            a, b: Operands

        Returns:
            prediction: Predicted (a + b) mod p
            info: Diagnostic information
        """
        # Encode input
        x = combined_encoding(a, b, self.p)

        # Step 1: Extract features with LNN
        liquid_features, lnn_info = self.liquid_encoder.encode(x)

        # Step 2: Stigmergic search
        stigmergic_votes = self.stigmergic_search.search(liquid_features, a, b)

        # Step 3: Consensus
        final_probabilities = self.consensus.forward(liquid_features, stigmergic_votes)

        # Prediction
        prediction = np.argmax(final_probabilities)
        confidence = final_probabilities[prediction]

        # Info
        info = {
            'liquid_features': liquid_features,
            'stigmergic_votes': stigmergic_votes,
            'final_probabilities': final_probabilities,
            'confidence': float(confidence),
            'lnn_activity': lnn_info.get('h_norm', 0.0),
            'lnn_tau_mean': lnn_info.get('mean_time_constant', 0.0),
            'stigmergic_entropy': -np.sum(stigmergic_votes * np.log(stigmergic_votes + 1e-8)),
            'consensus_alpha': self.consensus.alpha,
        }

        return int(prediction), info

    def learn(self, a: int, b: int, correct_result: int,
              liquid_features: np.ndarray, stigmergic_votes: np.ndarray,
              final_probabilities: np.ndarray):
        """
        Update all components after seeing correct answer.

        Uses local learning rules (NO BACKPROP!):
        - LNN: Hebbian update
        - Stigmergic: Pheromone reinforcement + agent policy update
        - Consensus: Local contrastive learning
        """
        # Update LNN (simple target: encode correct result cyclically)
        target_encoding = np.array([
            correct_result / self.p,
            np.sin(2 * np.pi * correct_result / self.p),
            np.cos(2 * np.pi * correct_result / self.p),
        ])
        # Pad to liquid_dim
        target = np.zeros(self.liquid_dim)
        target[:len(target_encoding)] = target_encoding
        self.liquid_encoder.lnn.learn(target)

        # Update stigmergic agents
        was_correct = (np.argmax(final_probabilities) == correct_result)
        self.stigmergic_search.update_pheromones(correct_result, stigmergic_votes,
                                                  was_correct)
        self.stigmergic_search.learn(liquid_features, correct_result, stigmergic_votes)

        # Update consensus
        self.consensus.learn(liquid_features, stigmergic_votes,
                            correct_result, final_probabilities)

    def reset(self):
        """Reset liquid state between problems"""
        self.liquid_encoder.reset()


# ============================================================================
# TRAINING
# ============================================================================

def train_hybrid(p: int, n_epochs: int = 100, n_samples_per_epoch: int = 100):
    """
    Train hybrid network on modular addition.

    Args:
        p: Modulo value
        n_epochs: Number of training epochs
        n_samples_per_epoch: Problems per epoch

    Returns:
        model: Trained hybrid network
        results: Training statistics
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: (a + b) mod {p}")
    print(f"{'='*70}\n")

    model = HybridLiquidStigmergicNetwork(p=p)

    results = {
        'p': p,
        'epoch_accuracy': [],
        'final_accuracy': 0.0,
        'training_time': 0.0,
    }

    start_time = time.time()

    for epoch in range(n_epochs):
        correct = 0
        total = 0

        epoch_info = {
            'liquid_activity': [],
            'stigmergic_confidence': [],
            'consensus_alpha': [],
        }

        for _ in range(n_samples_per_epoch):
            # Generate problem
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            correct_result = (a + b) % p

            # Forward pass
            prediction, info = model.forward(a, b)

            # Learn
            model.learn(a, b, correct_result,
                       info['liquid_features'],
                       info['stigmergic_votes'],
                       info['final_probabilities'])

            # Stats
            if prediction == correct_result:
                correct += 1
            total += 1

            epoch_info['liquid_activity'].append(info['lnn_activity'])
            epoch_info['stigmergic_confidence'].append(info['confidence'])
            epoch_info['consensus_alpha'].append(info['consensus_alpha'])

        accuracy = correct / total
        results['epoch_accuracy'].append(accuracy)

        # Reset LNN state periodically
        if epoch % 10 == 0:
            model.reset()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: accuracy={accuracy:.3f}, "
                  f"liquid={np.mean(epoch_info['liquid_activity']):.3f}, "
                  f"alpha={np.mean(epoch_info['consensus_alpha']):.3f}")

    results['final_accuracy'] = results['epoch_accuracy'][-1]
    results['training_time'] = time.time() - start_time

    return model, results


def test_hybrid(model: HybridLiquidStigmergicNetwork, n_test: int = 1000) -> float:
    """
    Test hybrid network on fresh problems.

    Returns:
        accuracy: Test set accuracy
    """
    model.reset()

    correct = 0
    for _ in range(n_test):
        a = np.random.randint(0, model.p)
        b = np.random.randint(0, model.p)
        correct_result = (a + b) % model.p

        prediction, info = model.forward(a, b)

        if prediction == correct_result:
            correct += 1

    accuracy = correct / n_test
    return accuracy


# ============================================================================
# COMPARISON: Hybrid vs Pure LNN vs Pure Stigmergic
# ============================================================================

def benchmark_all_approaches(p: int, n_train: int = 5000, n_test: int = 1000):
    """
    Compare hybrid against pure approaches.

    Returns:
        results: Dictionary with all results
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: All Approaches on mod {p}")
    print(f"{'='*70}\n")

    results = {'p': p}

    # 1. Hybrid (our approach)
    print("\n--- HYBRID LIQUID-STIGMERGIC ---")
    hybrid_model, hybrid_train_results = train_hybrid(p, n_epochs=50, n_samples_per_epoch=100)
    hybrid_test_acc = test_hybrid(hybrid_model, n_test)
    results['hybrid'] = {
        'train_accuracy': hybrid_train_results['final_accuracy'],
        'test_accuracy': hybrid_test_acc,
        'training_time': hybrid_train_results['training_time'],
    }
    print(f"Hybrid Test Accuracy: {hybrid_test_acc:.3f}")

    # 2. Pure LNN (baseline)
    print("\n--- PURE LIQUID NEURAL NETWORK ---")
    lnn_encoder = LiquidEncoder(26, 32, p)
    # Simple output layer
    lnn_output_weights = np.random.randn(p, 32) * 0.1

    lnn_correct = 0
    for _ in range(n_train):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        correct_result = (a + b) % p

        x = combined_encoding(a, b, p)
        features, _ = lnn_encoder.encode(x)

        logits = lnn_output_weights @ features
        prediction = np.argmax(logits)

        if prediction == correct_result:
            lnn_correct += 1

        # Simple learning
        target = np.zeros(p)
        target[correct_result] = 1.0
        error = target - F.softmax(torch.tensor(logits), dim=0).numpy()
        lnn_output_weights += 0.01 * np.outer(error, features)

    lnn_train_acc = lnn_correct / n_train

    # Test LNN
    lnn_encoder.reset()
    lnn_test_correct = 0
    for _ in range(n_test):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        correct_result = (a + b) % p

        x = combined_encoding(a, b, p)
        features, _ = lnn_encoder.encode(x)

        logits = lnn_output_weights @ features
        prediction = np.argmax(logits)

        if prediction == correct_result:
            lnn_test_correct += 1

    lnn_test_acc = lnn_test_correct / n_test
    results['pure_lnn'] = {
        'train_accuracy': lnn_train_acc,
        'test_accuracy': lnn_test_acc,
    }
    print(f"Pure LNN Test Accuracy: {lnn_test_acc:.3f}")

    # 3. Pure Stigmergic (baseline)
    print("\n--- PURE STIGMERGIC ---")
    stigmergic_search = StigmergicArithmeticSearch(p, n_agents=50, feature_dim=26)

    stig_correct = 0
    for _ in range(n_train):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        correct_result = (a + b) % p

        x = combined_encoding(a, b, p)
        votes = stigmergic_search.search(x, a, b)
        prediction = np.argmax(votes)

        if prediction == correct_result:
            stig_correct += 1

        stigmergic_search.update_pheromones(correct_result, votes,
                                            prediction == correct_result)
        stigmergic_search.learn(x, correct_result, votes)

    stig_train_acc = stig_correct / n_train

    # Test stigmergic
    stig_test_correct = 0
    for _ in range(n_test):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        correct_result = (a + b) % p

        x = combined_encoding(a, b, p)
        votes = stigmergic_search.search(x, a, b)
        prediction = np.argmax(votes)

        if prediction == correct_result:
            stig_test_correct += 1

    stig_test_acc = stig_test_correct / n_test
    results['pure_stigmergic'] = {
        'train_accuracy': stig_train_acc,
        'test_accuracy': stig_test_acc,
    }
    print(f"Pure Stigmergic Test Accuracy: {stig_test_acc:.3f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive benchmark on multiple primes"""

    print("\n" + "="*70)
    print("HYBRID LIQUID-STIGMERGIC ARITHMETIC LEARNING")
    print("="*70)
    print("\nHypothesis: Neither LNN nor Stigmergic alone works for arithmetic,")
    print("but COMBINING them creates emergent intelligence!")
    print("\nTesting on primes: 7, 11, 23, 47, 97")
    print("="*70 + "\n")

    primes = [7, 11, 23, 47, 97]
    all_results = {}

    for p in primes:
        results = benchmark_all_approaches(p, n_train=5000, n_test=1000)
        all_results[f'p{p}'] = results

        print(f"\n{'='*70}")
        print(f"SUMMARY: mod {p}")
        print(f"{'='*70}")
        print(f"Hybrid:         train={results['hybrid']['train_accuracy']:.3f}, "
              f"test={results['hybrid']['test_accuracy']:.3f}")
        print(f"Pure LNN:       train={results['pure_lnn']['train_accuracy']:.3f}, "
              f"test={results['pure_lnn']['test_accuracy']:.3f}")
        print(f"Pure Stigmergic: train={results['pure_stigmergic']['train_accuracy']:.3f}, "
              f"test={results['pure_stigmergic']['test_accuracy']:.3f}")

        # Check success criterion
        hybrid_acc = results['hybrid']['test_accuracy']
        lnn_acc = results['pure_lnn']['test_accuracy']
        stig_acc = results['pure_stigmergic']['test_accuracy']

        if hybrid_acc > 0.7 and lnn_acc < 0.2 and stig_acc < 0.2:
            print(f"\n SUCCESS! Hybrid achieves {hybrid_acc:.1%} where individuals fail!")
        elif hybrid_acc > max(lnn_acc, stig_acc) * 1.5:
            print(f"\n PROMISING! Hybrid {hybrid_acc:.1%} > individual approaches")
        else:
            print(f"\n NEEDS WORK: Hybrid not significantly better")

    # Save results
    output_file = 'hybrid_liquid_stigmergic_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")

    # Final summary table
    print("\nFINAL COMPARISON TABLE")
    print("="*70)
    print(f"{'Prime':<10} {'Hybrid':<15} {'Pure LNN':<15} {'Pure Stigmergic':<15}")
    print("-"*70)
    for p in primes:
        r = all_results[f'p{p}']
        print(f"mod {p:<6} {r['hybrid']['test_accuracy']:>6.1%}        "
              f"{r['pure_lnn']['test_accuracy']:>6.1%}         "
              f"{r['pure_stigmergic']['test_accuracy']:>6.1%}")
    print("="*70)

    return all_results


if __name__ == '__main__':
    results = main()
