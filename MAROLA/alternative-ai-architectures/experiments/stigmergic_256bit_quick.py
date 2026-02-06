#!/usr/bin/env python3
"""
Quick test of Stigmergic 256-bit Arithmetic
Optimized for faster training while maintaining accuracy.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time
import json


@dataclass
class DigitEnvironment:
    """Pheromone environment for single-digit addition with carry."""
    base: int = 16

    def __post_init__(self):
        self.n_outputs = self.base * 2  # digit × carry
        self.pheromones: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _key(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return (a % self.base, b % self.base, c % 2)

    def _encode(self, digit: int, carry: int) -> int:
        return digit * 2 + carry

    def _decode(self, idx: int) -> Tuple[int, int]:
        return idx // 2, idx % 2

    def get_pheromones(self, a: int, b: int, c: int) -> np.ndarray:
        key = self._key(a, b, c)
        if key not in self.pheromones:
            self.pheromones[key] = np.ones(self.n_outputs) / self.n_outputs
        return self.pheromones[key]

    def deposit(self, a: int, b: int, c: int, d_out: int, c_out: int, amount: float):
        pheromones = self.get_pheromones(a, b, c)
        pheromones[self._encode(d_out, c_out)] += amount

    def evaporate(self, rate: float = 0.99):
        for key in self.pheromones:
            self.pheromones[key] *= rate
            total = np.sum(self.pheromones[key])
            if total > 0:
                self.pheromones[key] /= total

    def get_best(self, a: int, b: int, c: int) -> Tuple[int, int]:
        return self._decode(np.argmax(self.get_pheromones(a, b, c)))


class DigitAnt:
    """Ant for single-digit arithmetic with efficient Hebbian learning."""

    def __init__(self, env: DigitEnvironment, lr: float = 0.2):
        self.env = env
        self.lr = lr
        self.feature_dim = 16
        self.W = np.random.randn(env.n_outputs, self.feature_dim) * 0.1
        self.correct = 0
        self.total = 0

    def encode(self, a: int, b: int, c: int) -> np.ndarray:
        """Fourier-based encoding for modular arithmetic."""
        base = self.env.base
        s = a + b + c

        return np.array([
            a / base, b / base, float(c),
            np.sin(2 * np.pi * a / base), np.cos(2 * np.pi * a / base),
            np.sin(2 * np.pi * b / base), np.cos(2 * np.pi * b / base),
            np.sin(2 * np.pi * (s % base) / base),
            np.cos(2 * np.pi * (s % base) / base),
            float(s >= base),  # Carry hint
            np.sin(4 * np.pi * a / base), np.cos(4 * np.pi * a / base),
            np.sin(4 * np.pi * b / base), np.cos(4 * np.pi * b / base),
            (a + b) / (2 * base), 1.0
        ], dtype=np.float32)

    def guess(self, a: int, b: int, c: int, use_pher: bool = True) -> Tuple[int, int]:
        features = self.encode(a, b, c)
        scores = self.W @ features
        intuition = np.exp(scores - np.max(scores))
        intuition /= intuition.sum()

        if use_pher:
            pher = self.env.get_pheromones(a, b, c)
            pher = pher / (pher.sum() + 1e-10)
            combined = 0.5 * intuition + 0.5 * pher
        else:
            combined = intuition

        if np.random.rand() < 0.05:  # 5% exploration
            idx = np.random.randint(self.env.n_outputs)
        else:
            idx = np.random.choice(self.env.n_outputs, p=combined)

        return self.env._decode(idx)

    def learn(self, a: int, b: int, c: int, correct_d: int, correct_c: int,
              my_d: int, my_c: int):
        features = self.encode(a, b, c)
        correct_idx = self.env._encode(correct_d, correct_c)
        my_idx = self.env._encode(my_d, my_c)

        if my_idx == correct_idx:
            self.W[correct_idx] += self.lr * features
            self.correct += 1
        else:
            self.W[my_idx] -= self.lr * 0.5 * features
            self.W[correct_idx] += self.lr * 0.3 * features
        self.total += 1


class DigitColony:
    """Colony of ants for single-digit arithmetic."""

    def __init__(self, n_ants: int = 32, base: int = 16):
        self.n_ants = n_ants
        self.base = base
        self.env = DigitEnvironment(base)
        self.ants = [DigitAnt(self.env) for _ in range(n_ants)]

    def train_step(self, a: int, b: int, c: int):
        s = a + b + c
        correct_d, correct_c = s % self.base, s // self.base

        for ant in self.ants:
            my_d, my_c = ant.guess(a, b, c)
            ant.learn(a, b, c, correct_d, correct_c, my_d, my_c)

            if my_d == correct_d and my_c == correct_c:
                self.env.deposit(a, b, c, correct_d, correct_c, 1.0)

        self.env.evaporate(0.995)

    def train_epoch(self, n_samples: int = 256):
        for _ in range(n_samples):
            a = np.random.randint(0, self.base)
            b = np.random.randint(0, self.base)
            c = np.random.randint(0, 2)
            self.train_step(a, b, c)

    def predict(self, a: int, b: int, c: int) -> Tuple[int, int]:
        return self.env.get_best(a, b, c)

    def evaluate(self) -> float:
        correct = 0
        for a in range(self.base):
            for b in range(self.base):
                for c in range(2):
                    pred_d, pred_c = self.predict(a, b, c)
                    s = a + b + c
                    if pred_d == s % self.base and pred_c == s // self.base:
                        correct += 1
        return correct / 512


class Stigmergic256Bit:
    """256-bit arithmetic using stigmergic colonies."""

    def __init__(self, n_ants: int = 32, n_ensembles: int = 3):
        self.n_ants = n_ants
        self.n_ensembles = n_ensembles
        self.base = 16
        self.n_digits = 64
        self.colonies = [DigitColony(n_ants, self.base) for _ in range(n_ensembles)]
        self.trained = False

    def train(self, n_epochs: int = 60, verbose: bool = True):
        start = time.time()

        for epoch in range(n_epochs):
            for colony in self.colonies:
                colony.train_epoch(512)

            if verbose and (epoch + 1) % 10 == 0:
                accs = [c.evaluate() for c in self.colonies]
                print(f"Epoch {epoch+1:3d}: accuracy={np.mean(accs):.4f} "
                      f"(per colony: {', '.join(f'{a:.3f}' for a in accs)})")

        elapsed = time.time() - start
        self.trained = True

        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            for i, c in enumerate(self.colonies):
                print(f"  Colony {i+1} final accuracy: {c.evaluate():.4f}")

        return elapsed

    def _to_digits(self, n: int) -> List[int]:
        digits = []
        for _ in range(self.n_digits):
            digits.append(n % self.base)
            n //= self.base
        return digits

    def _to_int(self, digits: List[int]) -> int:
        result = 0
        for i, d in enumerate(digits):
            result += d * (self.base ** i)
        return result

    def add(self, a: int, b: int, use_ensemble: bool = True) -> int:
        a_digits = self._to_digits(a)
        b_digits = self._to_digits(b)

        if use_ensemble:
            # Get results from all colonies
            results = []
            for colony in self.colonies:
                result_digits = []
                carry = 0
                for i in range(len(a_digits)):
                    d, c = colony.predict(a_digits[i], b_digits[i], carry)
                    result_digits.append(d)
                    carry = c
                if carry:
                    result_digits.append(carry)
                results.append(self._to_int(result_digits))

            # Majority vote
            from collections import Counter
            return Counter(results).most_common(1)[0][0]
        else:
            colony = self.colonies[0]
            result_digits = []
            carry = 0
            for i in range(len(a_digits)):
                d, c = colony.predict(a_digits[i], b_digits[i], carry)
                result_digits.append(d)
                carry = c
            if carry:
                result_digits.append(carry)
            return self._to_int(result_digits)

    def evaluate(self, bits: int, n_samples: int = 500,
                 use_ensemble: bool = True) -> Dict:
        import random
        max_val = (1 << bits) - 1
        correct = 0

        for _ in range(n_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            expected = a + b
            predicted = self.add(a, b, use_ensemble)
            if predicted == expected:
                correct += 1

        return {
            'bits': bits,
            'accuracy': correct / n_samples,
            'correct': correct,
            'total': n_samples
        }


def main():
    print("="*70)
    print("STIGMERGIC 256-BIT ARITHMETIC - QUICK TEST")
    print("Bio-plausible computation at cryptographic scale (NO BACKPROP)")
    print("="*70 + "\n")

    # Create system
    n_ants = 48
    n_ensembles = 5
    n_epochs = 80

    print(f"Configuration:")
    print(f"  Ants per colony: {n_ants}")
    print(f"  Ensemble size: {n_ensembles}")
    print(f"  Training epochs: {n_epochs}")
    print()

    system = Stigmergic256Bit(n_ants=n_ants, n_ensembles=n_ensembles)

    print("Training...")
    elapsed = system.train(n_epochs=n_epochs, verbose=True)

    print("\n" + "="*70)
    print("SCALING TEST")
    print("="*70 + "\n")

    results = {}
    bit_widths = [8, 16, 32, 64, 128, 256]

    for bits in bit_widths:
        print(f"Testing {bits}-bit...")

        result_single = system.evaluate(bits, n_samples=300, use_ensemble=False)
        result_ensemble = system.evaluate(bits, n_samples=300, use_ensemble=True)

        results[f'{bits}bit'] = {
            'single': result_single['accuracy'],
            'ensemble': result_ensemble['accuracy']
        }

        print(f"  Single colony: {result_single['accuracy']:.2%}")
        print(f"  Ensemble ({n_ensembles}): {result_ensemble['accuracy']:.2%}")

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")

    print("Single-digit accuracy (512 cases):")
    for i, c in enumerate(system.colonies):
        print(f"  Colony {i+1}: {c.evaluate():.2%}")

    print(f"\nMulti-digit accuracy (ensemble={n_ensembles}):")
    for bits in bit_widths:
        print(f"  {bits:3d}-bit: {results[f'{bits}bit']['ensemble']:.2%}")

    acc_256 = results['256bit']['ensemble']
    print(f"\n{'='*70}")
    if acc_256 >= 0.95:
        print(f"EXCELLENT: {acc_256:.2%} on 256-bit WITHOUT BACKPROPAGATION!")
    elif acc_256 >= 0.80:
        print(f"GOOD: {acc_256:.2%} on 256-bit WITHOUT BACKPROPAGATION!")
    else:
        print(f"Result: {acc_256:.2%} on 256-bit WITHOUT BACKPROPAGATION")

    print(f"Method: Stigmergic (ant colony) intelligence")
    print(f"Learning: Pure Hebbian (local rules only)")
    print(f"Training time: {elapsed:.1f}s")
    print("="*70)

    # Save results
    output = {
        'config': {'n_ants': n_ants, 'n_ensembles': n_ensembles, 'n_epochs': n_epochs},
        'single_digit': {i: c.evaluate() for i, c in enumerate(system.colonies)},
        'multi_digit': results,
        'elapsed_time': elapsed
    }

    with open('/root/MAROLA/alternative-ai-architectures/experiments/stigmergic_256bit_quick_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved!")
    return system, results


if __name__ == "__main__":
    system, results = main()
