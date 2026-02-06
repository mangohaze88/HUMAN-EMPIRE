"""
Stigmergic Arithmetic Learning
================================
Ant colony intelligence for learning modular arithmetic WITHOUT backpropagation.

Concept:
- Each ant "guesses" the result of (a + b) mod p
- Correct ants leave pheromone trails
- Over time, colony converges through collective intelligence
- Pattern recognition emerges from swarm behavior

This is a pure swarm intelligence approach - no neural networks, no gradients.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time
import json


@dataclass
class ArithmeticEnvironment:
    """
    Shared environment storing pheromone trails for arithmetic patterns.

    For each (a, b) input pair, we maintain pheromone levels for each possible result.
    Ants deposit pheromones when they guess correctly, creating a collective memory.
    """
    p: int  # Modulus

    def __post_init__(self):
        # Pheromone grid: (a, b) -> array of pheromone levels for each result
        self.pheromones: Dict[Tuple[int, int], np.ndarray] = {}

        # Global statistics
        self.total_deposits = 0
        self.total_queries = 0

    def get_pheromones(self, a: int, b: int) -> np.ndarray:
        """Get pheromone levels for a given input pair."""
        key = (a % self.p, b % self.p)
        if key not in self.pheromones:
            # Initialize with uniform prior (all results equally likely)
            self.pheromones[key] = np.ones(self.p) / self.p
        return self.pheromones[key]

    def deposit(self, a: int, b: int, result: int, amount: float):
        """Ant deposits pheromone on a specific result."""
        key = (a % self.p, b % self.p)
        pheromones = self.get_pheromones(a, b)
        pheromones[result] += amount
        self.total_deposits += 1

    def evaporate(self, rate: float = 0.95):
        """Evaporate pheromones globally (prevents stagnation)."""
        for key in self.pheromones:
            self.pheromones[key] *= rate
            # Renormalize to maintain probability distribution
            total = np.sum(self.pheromones[key])
            if total > 0:
                self.pheromones[key] /= total

    def get_confidence(self, a: int, b: int) -> float:
        """Get colony's confidence in its answer (entropy-based)."""
        pheromones = self.get_pheromones(a, b)
        # Normalize
        total = np.sum(pheromones)
        if total == 0:
            return 0.0
        probs = pheromones / total
        # Higher entropy = lower confidence
        # Use negative entropy normalized to [0, 1]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(self.p)
        return 1.0 - (entropy / max_entropy)


class ArithmeticAnt:
    """
    Individual ant with internal 'intuition' for arithmetic patterns.

    Each ant combines:
    1. Individual intuition (learned weights)
    2. Collective knowledge (pheromone trails)
    3. Exploration (random noise)
    """

    def __init__(self, env: ArithmeticEnvironment, learning_rate: float = 0.1,
                 exploration_rate: float = 0.1):
        self.env = env
        self.lr = learning_rate
        self.exploration = exploration_rate

        # Feature dimension for encoding inputs
        self.feature_dim = 16

        # Internal "intuition" weights for each possible result
        # Shape: (p, feature_dim)
        self.W = np.random.randn(env.p, self.feature_dim) * 0.1

        # Personal statistics
        self.correct_guesses = 0
        self.total_guesses = 0

    def encode_input(self, a: int, b: int) -> np.ndarray:
        """
        Encode (a, b) as rich feature vector.

        Features include:
        - Normalized values
        - Cyclic encodings (sin/cos for modular structure)
        - Interaction terms
        - Hints about the answer
        """
        p = self.env.p

        # Normalize to [0, 1]
        a_norm = a / p
        b_norm = b / p

        # Cyclic encodings (captures modular structure)
        a_sin = np.sin(2 * np.pi * a / p)
        a_cos = np.cos(2 * np.pi * a / p)
        b_sin = np.sin(2 * np.pi * b / p)
        b_cos = np.cos(2 * np.pi * b / p)

        # Sum hints (helps learn addition)
        sum_norm = (a + b) / (2 * p)
        sum_sin = np.sin(2 * np.pi * (a + b) / p)
        sum_cos = np.cos(2 * np.pi * (a + b) / p)

        # Interaction terms
        ab_product = (a * b) / (p * p)

        # Difference (for diversity)
        diff_norm = abs(a - b) / p

        # Build feature vector
        features = np.array([
            a_norm, b_norm,
            a_sin, a_cos,
            b_sin, b_cos,
            sum_norm, sum_sin, sum_cos,
            ab_product,
            diff_norm,
            1.0,  # Bias
            # Pad to feature_dim
            0.0, 0.0, 0.0, 0.0
        ])[:self.feature_dim]

        return features

    def get_intuition_scores(self, a: int, b: int) -> np.ndarray:
        """Get ant's intuition about each possible result."""
        features = self.encode_input(a, b)
        # Dot product: each result has a weight vector
        scores = self.W @ features
        return scores

    def guess(self, a: int, b: int, use_pheromones: bool = True) -> int:
        """
        Make a guess combining intuition and pheromones.

        Args:
            a, b: Input values
            use_pheromones: If False, use only individual intuition
        """
        # Individual intuition
        intuition_scores = self.get_intuition_scores(a, b)
        intuition = self._softmax(intuition_scores)

        if use_pheromones:
            # Collective knowledge
            pheromones = self.env.get_pheromones(a, b)
            # Normalize pheromones
            total = np.sum(pheromones)
            if total > 0:
                pheromones = pheromones / total
            else:
                pheromones = np.ones(self.env.p) / self.env.p

            # Combine intuition and pheromones
            # Weight: 50% individual, 50% collective
            combined = 0.5 * intuition + 0.5 * pheromones
        else:
            combined = intuition

        # Add exploration noise
        if np.random.rand() < self.exploration:
            # Random exploration
            return np.random.randint(0, self.env.p)
        else:
            # Sample from combined distribution
            return np.random.choice(self.env.p, p=combined)

    def learn(self, a: int, b: int, correct_result: int, my_guess: int):
        """
        Update ant's intuition based on outcome.

        This is Hebbian-like learning:
        - Reinforce correct guesses
        - Weaken incorrect guesses
        """
        features = self.encode_input(a, b)

        if my_guess == correct_result:
            # Reinforce this pattern
            self.W[correct_result] += self.lr * features
            self.correct_guesses += 1
        else:
            # Weaken wrong guess
            self.W[my_guess] -= self.lr * 0.5 * features
            # Also slightly reinforce correct answer
            self.W[correct_result] += self.lr * 0.3 * features

        self.total_guesses += 1

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Stable softmax implementation."""
        x_shifted = (x - np.max(x)) / temperature
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    def get_accuracy(self) -> float:
        """Get ant's personal accuracy."""
        if self.total_guesses == 0:
            return 0.0
        return self.correct_guesses / self.total_guesses


class ArithmeticColony:
    """
    Colony of ants learning arithmetic through stigmergy.

    The colony as a whole learns to compute (a + b) mod p through:
    1. Individual ant learning (intuition)
    2. Pheromone communication (collective memory)
    3. Emergent consensus
    """

    def __init__(self, n_ants: int, p: int, learning_rate: float = 0.1):
        self.p = p
        self.n_ants = n_ants

        # Create shared environment
        self.env = ArithmeticEnvironment(p)

        # Create ant colony
        self.ants = [
            ArithmeticAnt(self.env, learning_rate=learning_rate)
            for _ in range(n_ants)
        ]

        # Training statistics
        self.epoch_accuracies = []
        self.epoch_confidences = []

    def train_step(self, a: int, b: int) -> Dict[str, float]:
        """
        Single training step on one example.

        Returns:
            Dictionary with step statistics
        """
        correct = (a + b) % self.p

        # All ants make guesses
        guesses = [ant.guess(a, b) for ant in self.ants]

        # Count correct guesses
        n_correct = sum(g == correct for g in guesses)
        step_accuracy = n_correct / len(guesses)

        # Update ants and deposit pheromones
        for ant, guess in zip(self.ants, guesses):
            # Ant learns from outcome
            ant.learn(a, b, correct, guess)

            # Correct ants deposit pheromone
            if guess == correct:
                # Deposit amount proportional to confidence
                deposit_amount = 1.0
                self.env.deposit(a, b, correct, deposit_amount)

        # Global pheromone evaporation (slow)
        self.env.evaporate(rate=0.99)

        # Get colony confidence
        confidence = self.env.get_confidence(a, b)

        return {
            'accuracy': step_accuracy,
            'confidence': confidence,
            'n_correct': n_correct
        }

    def train_epoch(self, n_samples: int = 100, verbose: bool = False) -> Dict[str, float]:
        """
        Train for one epoch on random examples.

        Args:
            n_samples: Number of random (a, b) pairs to train on
            verbose: Print progress
        """
        accuracies = []
        confidences = []

        for i in range(n_samples):
            # Random example
            a = np.random.randint(0, self.p)
            b = np.random.randint(0, self.p)

            stats = self.train_step(a, b)
            accuracies.append(stats['accuracy'])
            confidences.append(stats['confidence'])

            if verbose and (i + 1) % 20 == 0:
                avg_acc = np.mean(accuracies[-20:])
                avg_conf = np.mean(confidences[-20:])
                print(f"  Sample {i+1}/{n_samples}: "
                      f"Acc={avg_acc:.3f}, Conf={avg_conf:.3f}")

        epoch_stats = {
            'accuracy': np.mean(accuracies),
            'confidence': np.mean(confidences),
            'accuracy_std': np.std(accuracies),
        }

        self.epoch_accuracies.append(epoch_stats['accuracy'])
        self.epoch_confidences.append(epoch_stats['confidence'])

        return epoch_stats

    def predict(self, a: int, b: int, method: str = 'majority') -> int:
        """
        Make a collective prediction.

        Args:
            method: 'majority' (vote), 'pheromone' (strongest trail),
                   or 'consensus' (require agreement)
        """
        if method == 'majority':
            # Democratic vote
            guesses = [ant.guess(a, b) for ant in self.ants]
            return np.bincount(guesses, minlength=self.p).argmax()

        elif method == 'pheromone':
            # Follow strongest pheromone trail
            pheromones = self.env.get_pheromones(a, b)
            return np.argmax(pheromones)

        elif method == 'consensus':
            # Only predict if ants agree
            guesses = [ant.guess(a, b) for ant in self.ants]
            counts = np.bincount(guesses, minlength=self.p)
            max_votes = np.max(counts)
            if max_votes >= 0.7 * len(guesses):
                return np.argmax(counts)
            else:
                return -1  # No consensus

        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def evaluate(self, n_test: int = 100) -> Dict[str, float]:
        """
        Evaluate colony on random test examples.
        """
        correct_majority = 0
        correct_pheromone = 0
        total = 0

        for _ in range(n_test):
            a = np.random.randint(0, self.p)
            b = np.random.randint(0, self.p)
            expected = (a + b) % self.p

            pred_majority = self.predict(a, b, method='majority')
            pred_pheromone = self.predict(a, b, method='pheromone')

            if pred_majority == expected:
                correct_majority += 1
            if pred_pheromone == expected:
                correct_pheromone += 1
            total += 1

        # Get average ant accuracy
        ant_accuracies = [ant.get_accuracy() for ant in self.ants]

        return {
            'majority_vote_acc': correct_majority / total,
            'pheromone_acc': correct_pheromone / total,
            'avg_ant_acc': np.mean(ant_accuracies),
            'min_ant_acc': np.min(ant_accuracies),
            'max_ant_acc': np.max(ant_accuracies),
            'ant_acc_std': np.std(ant_accuracies),
        }

    def get_ant_diversity(self) -> float:
        """
        Measure diversity of ant weights (specialization).
        Higher = more specialized ants.
        """
        # Flatten all ant weight matrices
        all_weights = np.array([ant.W.flatten() for ant in self.ants])
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(all_weights, metric='euclidean')
        return np.mean(distances)


def run_experiment(n_ants: int, p: int, n_epochs: int = 50,
                   samples_per_epoch: int = 100, verbose: bool = True):
    """
    Run a full stigmergic arithmetic learning experiment.

    Args:
        n_ants: Number of ants in colony
        p: Modulus for arithmetic
        n_epochs: Number of training epochs
        samples_per_epoch: Examples per epoch
        verbose: Print progress
    """
    print(f"\n{'='*60}")
    print(f"Stigmergic Arithmetic Learning Experiment")
    print(f"{'='*60}")
    print(f"Colony size: {n_ants} ants")
    print(f"Modulus: p = {p}")
    print(f"Training: {n_epochs} epochs × {samples_per_epoch} samples")
    print(f"{'='*60}\n")

    # Create colony
    colony = ArithmeticColony(n_ants, p, learning_rate=0.1)

    # Track training
    training_history = {
        'n_ants': n_ants,
        'p': p,
        'epochs': [],
        'train_acc': [],
        'train_conf': [],
        'test_majority_acc': [],
        'test_pheromone_acc': [],
        'ant_diversity': [],
    }

    start_time = time.time()

    # Training loop
    for epoch in range(n_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}")

        # Train
        epoch_stats = colony.train_epoch(samples_per_epoch, verbose=verbose)

        # Evaluate
        test_stats = colony.evaluate(n_test=100)

        # Measure diversity
        diversity = colony.get_ant_diversity()

        # Record
        training_history['epochs'].append(epoch + 1)
        training_history['train_acc'].append(epoch_stats['accuracy'])
        training_history['train_conf'].append(epoch_stats['confidence'])
        training_history['test_majority_acc'].append(test_stats['majority_vote_acc'])
        training_history['test_pheromone_acc'].append(test_stats['pheromone_acc'])
        training_history['ant_diversity'].append(diversity)

        if verbose:
            print(f"  Train Acc: {epoch_stats['accuracy']:.3f} "
                  f"(±{epoch_stats['accuracy_std']:.3f})")
            print(f"  Test Acc (majority): {test_stats['majority_vote_acc']:.3f}")
            print(f"  Test Acc (pheromone): {test_stats['pheromone_acc']:.3f}")
            print(f"  Ant Diversity: {diversity:.2f}")
            print(f"  Avg Ant Accuracy: {test_stats['avg_ant_acc']:.3f} "
                  f"[{test_stats['min_ant_acc']:.3f} - {test_stats['max_ant_acc']:.3f}]")
            print()

    elapsed = time.time() - start_time

    # Final evaluation
    final_stats = colony.evaluate(n_test=500)

    print(f"\n{'='*60}")
    print(f"Training Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Final Test Accuracy (Majority Vote): {final_stats['majority_vote_acc']:.3f}")
    print(f"Final Test Accuracy (Pheromone): {final_stats['pheromone_acc']:.3f}")
    print(f"Average Ant Accuracy: {final_stats['avg_ant_acc']:.3f}")
    print(f"Ant Accuracy Range: [{final_stats['min_ant_acc']:.3f} - "
          f"{final_stats['max_ant_acc']:.3f}]")
    print(f"{'='*60}\n")

    # Store final results
    training_history['final_test_stats'] = final_stats
    training_history['elapsed_time'] = elapsed

    return colony, training_history


def compare_colony_sizes(p: int = 23, n_epochs: int = 40):
    """
    Compare different colony sizes on same problem.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Colony Size Comparison")
    print("="*60)

    colony_sizes = [32, 64, 128]
    results = {}

    for n_ants in colony_sizes:
        print(f"\n{'='*60}")
        print(f"Testing colony size: {n_ants}")
        print(f"{'='*60}")

        colony, history = run_experiment(
            n_ants=n_ants,
            p=p,
            n_epochs=n_epochs,
            samples_per_epoch=100,
            verbose=False
        )

        results[n_ants] = history

        # Print summary
        final_acc = history['final_test_stats']['majority_vote_acc']
        print(f"\nColony size {n_ants}: Final accuracy = {final_acc:.3f}")

    return results


def compare_moduli(n_ants: int = 64, n_epochs: int = 40):
    """
    Compare performance on different moduli.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Modulus Comparison")
    print("="*60)

    moduli = [7, 11, 23, 47]
    results = {}

    for p in moduli:
        print(f"\n{'='*60}")
        print(f"Testing modulus: p = {p}")
        print(f"{'='*60}")

        colony, history = run_experiment(
            n_ants=n_ants,
            p=p,
            n_epochs=n_epochs,
            samples_per_epoch=100,
            verbose=False
        )

        results[p] = history

        # Print summary
        final_acc = history['final_test_stats']['majority_vote_acc']
        print(f"\nModulus p={p}: Final accuracy = {final_acc:.3f}")

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Single experiment demo
    print("\n" + "="*80)
    print("STIGMERGIC ARITHMETIC LEARNING - DEMO")
    print("="*80)

    colony, history = run_experiment(
        n_ants=64,
        p=23,
        n_epochs=50,
        samples_per_epoch=100,
        verbose=True
    )

    # Save results
    with open('/root/MAROLA/alternative-ai-architectures/experiments/stigmergic_arithmetic_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON
        history_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in history.items()
        }
        json.dump(history_serializable, f, indent=2)

    print("\nResults saved to: stigmergic_arithmetic_results.json")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy over time
    ax = axes[0, 0]
    ax.plot(history['epochs'], history['train_acc'], 'o-', label='Train', alpha=0.7)
    ax.plot(history['epochs'], history['test_majority_acc'], 's-', label='Test (Majority)', alpha=0.7)
    ax.plot(history['epochs'], history['test_pheromone_acc'], '^-', label='Test (Pheromone)', alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Curve (p={history["p"]}, {history["n_ants"]} ants)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Confidence over time
    ax = axes[0, 1]
    ax.plot(history['epochs'], history['train_conf'], 'o-', color='green', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Confidence')
    ax.set_title('Colony Confidence (1 - normalized entropy)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Ant diversity
    ax = axes[1, 0]
    ax.plot(history['epochs'], history['ant_diversity'], 'o-', color='purple', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diversity (avg pairwise distance)')
    ax.set_title('Ant Specialization')
    ax.grid(True, alpha=0.3)

    # Plot 4: Final statistics
    ax = axes[1, 1]
    final_stats = history['final_test_stats']
    metrics = ['Majority\nVote', 'Pheromone\nTrail', 'Avg Ant', 'Best Ant']
    values = [
        final_stats['majority_vote_acc'],
        final_stats['pheromone_acc'],
        final_stats['avg_ant_acc'],
        final_stats['max_ant_acc']
    ]
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Test Performance')
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/root/MAROLA/alternative-ai-architectures/experiments/stigmergic_arithmetic_demo.png',
                dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: stigmergic_arithmetic_demo.png")

    # Run comparison experiments
    print("\n\n" + "="*80)
    print("RUNNING COMPARISON EXPERIMENTS")
    print("="*80)

    # Compare colony sizes
    print("\n[1/2] Testing different colony sizes...")
    size_results = compare_colony_sizes(p=23, n_epochs=40)

    # Compare moduli
    print("\n[2/2] Testing different moduli...")
    moduli_results = compare_moduli(n_ants=64, n_epochs=40)

    # Summary visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colony size comparison
    ax = axes[0]
    for n_ants, history in size_results.items():
        ax.plot(history['epochs'], history['test_majority_acc'], 'o-',
                label=f'{n_ants} ants', alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (Majority Vote)')
    ax.set_title(f'Colony Size Comparison (p=23)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Modulus comparison
    ax = axes[1]
    for p, history in moduli_results.items():
        ax.plot(history['epochs'], history['test_majority_acc'], 'o-',
                label=f'p={p}', alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (Majority Vote)')
    ax.set_title(f'Modulus Comparison ({64} ants)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/MAROLA/alternative-ai-architectures/experiments/stigmergic_arithmetic_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nComparison visualization saved to: stigmergic_arithmetic_comparison.png")

    # Print final summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nColony Size Results (p=23):")
    print("-" * 60)
    for n_ants in [32, 64, 128]:
        acc = size_results[n_ants]['final_test_stats']['majority_vote_acc']
        time_taken = size_results[n_ants]['elapsed_time']
        print(f"  {n_ants:3d} ants: {acc:.3f} accuracy ({time_taken:.1f}s)")

    print("\nModulus Results (64 ants):")
    print("-" * 60)
    for p in [7, 11, 23, 47]:
        acc = moduli_results[p]['final_test_stats']['majority_vote_acc']
        time_taken = moduli_results[p]['elapsed_time']
        print(f"  p={p:2d}: {acc:.3f} accuracy ({time_taken:.1f}s)")

    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80)
    target_p = 23
    target_acc = moduli_results[target_p]['final_test_stats']['majority_vote_acc']

    if target_acc >= 0.80:
        print(f"✓ SUCCESS! Achieved {target_acc:.3f} accuracy on p={target_p}")
        print(f"  Target: >0.80 accuracy")
        print(f"  Method: Pure stigmergic learning (NO backpropagation)")
    else:
        print(f"✗ Target not reached: {target_acc:.3f} < 0.80")
        print(f"  Consider: More ants, more epochs, or tuning learning rate")

    print("="*80)
