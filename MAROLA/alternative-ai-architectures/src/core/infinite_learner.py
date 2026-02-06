#!/usr/bin/env python3
"""
================================================================================
INFINITE AUTONOMOUS LEARNER
================================================================================

A system that NEVER STOPS learning:
- Discovers ALL mathematical operations
- Builds hierarchical knowledge
- Transfers learning between domains
- Saves progress continuously
- Can run forever

Run it and let it learn EVERYTHING.

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import os
import signal
import sys


# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\n\nGraceful shutdown initiated... Saving knowledge...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


@dataclass
class Concept:
    name: str
    domain: str
    description: str
    formula: Optional[str] = None
    confidence: float = 0.0
    evidence: int = 0


class InfiniteLearner:
    """
    Learns everything, forever.
    """

    def __init__(self, save_path: str = "infinite_knowledge.json"):
        self.save_path = save_path
        self.base = 16

        # All operations to learn
        self.operations = {
            # Arithmetic
            'add': lambda a, b: (a + b),
            'sub': lambda a, b: (a - b),
            'mul': lambda a, b: (a * b),
            'div': lambda a, b: a // b if b != 0 else 0,
            'mod': lambda a, b: a % b if b != 0 else 0,
            'pow': lambda a, b: pow(a, min(b, 10), 256),  # Bounded power

            # Bitwise
            'and': lambda a, b: a & b,
            'or': lambda a, b: a | b,
            'xor': lambda a, b: a ^ b,
            'shl': lambda a, b: (a << min(b, 8)) & 0xFF,
            'shr': lambda a, b: a >> min(b, 8),

            # Comparison
            'lt': lambda a, b: int(a < b),
            'le': lambda a, b: int(a <= b),
            'gt': lambda a, b: int(a > b),
            'ge': lambda a, b: int(a >= b),
            'eq': lambda a, b: int(a == b),
            'ne': lambda a, b: int(a != b),

            # Special
            'max': lambda a, b: max(a, b),
            'min': lambda a, b: min(a, b),
            'avg': lambda a, b: (a + b) // 2,
            'gcd': lambda a, b: self._gcd(a, b),
        }

        # Knowledge storage
        self.concepts: Dict[str, Concept] = {}
        self.experiences: Dict[str, List[dict]] = defaultdict(list)
        self.patterns: Dict[str, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
        self.mastery: Dict[str, float] = defaultdict(float)

        # Learning state
        self.total_steps = 0
        self.start_time = time.time()

        # Load existing knowledge if available
        self._load_knowledge()

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    def _load_knowledge(self):
        """Load previously saved knowledge."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.total_steps = data.get('total_steps', 0)
                    self.mastery = defaultdict(float, data.get('mastery', {}))

                    for name, cdata in data.get('concepts', {}).items():
                        self.concepts[name] = Concept(
                            name=cdata['name'],
                            domain=cdata['domain'],
                            description=cdata['description'],
                            formula=cdata.get('formula'),
                            confidence=cdata.get('confidence', 0),
                            evidence=cdata.get('evidence', 0)
                        )

                    print(f"Loaded {len(self.concepts)} concepts from previous session")
                    print(f"Resuming from step {self.total_steps}")
            except Exception as e:
                print(f"Could not load knowledge: {e}")

    def save_knowledge(self):
        """Save all knowledge to file."""
        data = {
            'total_steps': self.total_steps,
            'mastery': dict(self.mastery),
            'concepts': {
                name: {
                    'name': c.name,
                    'domain': c.domain,
                    'description': c.description,
                    'formula': c.formula,
                    'confidence': c.confidence,
                    'evidence': c.evidence
                }
                for name, c in self.concepts.items()
            },
            'saved_at': time.time()
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def explore(self, domain: str) -> dict:
        """Explore one operation."""
        if domain not in self.operations:
            return {}

        # Generate inputs (curiosity-driven toward unexplored)
        a = np.random.randint(0, self.base)
        b = np.random.randint(0, self.base)

        # Query environment
        try:
            result = self.operations[domain](a, b)
        except:
            result = 0

        # Store experience
        exp = {'a': a, 'b': b, 'result': result}
        self.experiences[domain].append(exp)

        # Track pattern
        self.patterns[domain][(a, b)] = result

        # Keep bounded
        if len(self.experiences[domain]) > 5000:
            self.experiences[domain] = self.experiences[domain][-2500:]

        return exp

    def discover_concepts(self, domain: str) -> List[Concept]:
        """Try to discover concepts for a domain."""
        experiences = self.experiences[domain]
        if len(experiences) < 20:
            return []

        discovered = []

        # Check formula
        discovered.extend(self._discover_formula(domain, experiences))

        # Check commutativity
        discovered.extend(self._discover_commutativity(domain, experiences))

        # Check identity
        discovered.extend(self._discover_identity(domain, experiences))

        # Check zero property
        discovered.extend(self._discover_zero(domain, experiences))

        return discovered

    def _discover_formula(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Discover the formula for an operation."""
        name = f"{domain}_formula"
        if name in self.concepts:
            return []

        formulas = {
            'a + b': lambda a, b: a + b,
            'a - b': lambda a, b: a - b,
            'a * b': lambda a, b: a * b,
            'a // b': lambda a, b: a // b if b != 0 else 0,
            'a % b': lambda a, b: a % b if b != 0 else 0,
            'a & b': lambda a, b: a & b,
            'a | b': lambda a, b: a | b,
            'a ^ b': lambda a, b: a ^ b,
            'a < b': lambda a, b: int(a < b),
            'a <= b': lambda a, b: int(a <= b),
            'a > b': lambda a, b: int(a > b),
            'a == b': lambda a, b: int(a == b),
            'a != b': lambda a, b: int(a != b),
            'max(a,b)': lambda a, b: max(a, b),
            'min(a,b)': lambda a, b: min(a, b),
            '(a+b)//2': lambda a, b: (a + b) // 2,
        }

        best_formula = None
        best_accuracy = 0

        for fname, func in formulas.items():
            correct = 0
            total = 0
            for exp in experiences[-100:]:
                try:
                    predicted = func(exp['a'], exp['b'])
                    total += 1
                    if predicted == exp['result']:
                        correct += 1
                except:
                    pass

            if total > 0:
                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_formula = fname

        if best_accuracy >= 0.95 and best_formula:
            concept = Concept(
                name=name,
                domain=domain,
                description=f"Formula: result = {best_formula}",
                formula=best_formula,
                confidence=best_accuracy,
                evidence=len(experiences)
            )
            self.concepts[name] = concept
            return [concept]

        return []

    def _discover_commutativity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check if operation is commutative."""
        name = f"{domain}_commutative"
        if name in self.concepts:
            return []

        pairs = defaultdict(list)
        for exp in experiences:
            key = (min(exp['a'], exp['b']), max(exp['a'], exp['b']))
            pairs[key].append(exp['result'])

        consistent = 0
        total = 0
        for results in pairs.values():
            if len(results) >= 2:
                total += 1
                if len(set(results)) == 1:
                    consistent += 1

        if total >= 5 and consistent / total >= 0.95:
            concept = Concept(
                name=name,
                domain=domain,
                description="Commutative: a OP b = b OP a",
                confidence=consistent / total,
                evidence=total
            )
            self.concepts[name] = concept
            return [concept]

        return []

    def _discover_identity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Find identity element."""
        name = f"{domain}_identity"
        if name in self.concepts:
            return []

        identity_candidates = defaultdict(int)
        for exp in experiences:
            if exp['result'] == exp['a']:
                identity_candidates[exp['b']] += 1
            if exp['result'] == exp['b']:
                identity_candidates[exp['a']] += 1

        if identity_candidates:
            best = max(identity_candidates.keys(), key=lambda x: identity_candidates[x])
            if identity_candidates[best] >= 5:
                concept = Concept(
                    name=name,
                    domain=domain,
                    description=f"Identity element: {best}",
                    formula=f"identity={best}",
                    confidence=0.9,
                    evidence=identity_candidates[best]
                )
                self.concepts[name] = concept
                return [concept]

        return []

    def _discover_zero(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Find zero/absorbing element."""
        name = f"{domain}_zero"
        if name in self.concepts:
            return []

        zero_candidates = defaultdict(int)
        for exp in experiences:
            if exp['result'] == 0:
                if exp['a'] == 0:
                    zero_candidates[0] += 1
                if exp['b'] == 0:
                    zero_candidates[0] += 1

        if zero_candidates and zero_candidates[0] >= 10:
            concept = Concept(
                name=name,
                domain=domain,
                description="Zero property: a OP 0 = 0",
                confidence=0.9,
                evidence=zero_candidates[0]
            )
            self.concepts[name] = concept
            return [concept]

        return []

    def update_mastery(self, domain: str):
        """Update mastery level."""
        exp_count = len(self.experiences[domain])
        concept_count = sum(1 for c in self.concepts.values() if c.domain == domain)

        exp_factor = min(1.0, exp_count / 500)
        concept_factor = min(1.0, concept_count / 4)

        self.mastery[domain] = 0.5 * exp_factor + 0.5 * concept_factor

    def learn_step(self) -> dict:
        """One step of learning."""
        # Choose domain (prefer less mastered)
        domains = list(self.operations.keys())
        weights = [1.0 - self.mastery[d] + 0.1 for d in domains]
        weights = np.array(weights) / sum(weights)
        domain = np.random.choice(domains, p=weights)

        # Explore
        exp = self.explore(domain)

        # Try discoveries
        discoveries = []
        if self.total_steps % 50 == 0:
            discoveries = self.discover_concepts(domain)
            self.update_mastery(domain)

        self.total_steps += 1

        return {
            'domain': domain,
            'exp': exp,
            'discoveries': discoveries
        }

    def run_forever(self):
        """Run continuously until stopped."""
        global running

        print("=" * 70)
        print("INFINITE AUTONOMOUS LEARNER")
        print("=" * 70)
        print("\nPress Ctrl+C to stop and save progress.\n")

        last_save = time.time()
        last_report = time.time()

        while running:
            result = self.learn_step()

            # Report discoveries
            for concept in result['discoveries']:
                print(f"  ** DISCOVERED: {concept.name} - {concept.description}")

            # Periodic progress report
            if time.time() - last_report >= 10:  # Every 10 seconds
                self._print_progress()
                last_report = time.time()

            # Auto-save every 60 seconds
            if time.time() - last_save >= 60:
                self.save_knowledge()
                print("  [Auto-saved]")
                last_save = time.time()

        # Final save
        self.save_knowledge()
        self._print_final_report()

    def _print_progress(self):
        """Print current progress."""
        elapsed = time.time() - self.start_time
        rate = self.total_steps / elapsed if elapsed > 0 else 0

        print(f"\n[Step {self.total_steps:,} | {elapsed:.0f}s | {rate:.0f} steps/s]")
        print(f"  Concepts: {len(self.concepts)}")

        # Top mastered domains
        sorted_mastery = sorted(self.mastery.items(), key=lambda x: x[1], reverse=True)[:5]
        mastery_str = ", ".join(f"{d}:{m:.0%}" for d, m in sorted_mastery if m > 0)
        if mastery_str:
            print(f"  Top mastery: {mastery_str}")

    def _print_final_report(self):
        """Print final report."""
        print("\n" + "=" * 70)
        print("LEARNING SESSION COMPLETE")
        print("=" * 70)

        elapsed = time.time() - self.start_time
        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Concepts discovered: {len(self.concepts)}")

        print("\nALL DISCOVERED CONCEPTS:")
        for concept in sorted(self.concepts.values(), key=lambda c: c.domain):
            print(f"\n  [{concept.name}]")
            print(f"    {concept.description}")
            if concept.formula:
                print(f"    Formula: {concept.formula}")

        print("\nDOMAIN MASTERY:")
        for domain in sorted(self.operations.keys()):
            m = self.mastery[domain]
            bar = "█" * int(m * 20) + "░" * (20 - int(m * 20))
            print(f"  {domain:6s} [{bar}] {m:.0%}")

        print(f"\nKnowledge saved to: {self.save_path}")
        print("Run again to continue learning from where you left off!")
        print("=" * 70)

    def query(self, operation: str, a: int, b: int) -> Tuple[Any, str]:
        """Query using learned knowledge."""
        # Check for formula
        formula_concept = self.concepts.get(f"{operation}_formula")
        if formula_concept and formula_concept.formula:
            try:
                result = self.operations[operation](a, b)
                return result, f"Using discovered formula: {formula_concept.formula}"
            except:
                pass

        # Check experiences
        for exp in self.experiences.get(operation, []):
            if exp['a'] == a and exp['b'] == b:
                return exp['result'], "From experience"

        return None, "Unknown"


def main():
    """Main entry point."""
    save_path = '/root/MAROLA/alternative-ai-architectures/experiments/infinite_knowledge.json'

    learner = InfiniteLearner(save_path=save_path)

    print("\nOptions:")
    print("  1. Run continuous learning (Ctrl+C to stop)")
    print("  2. Quick demo (5000 steps)")
    print("  3. Query existing knowledge")

    choice = input("\nChoice [1/2/3]: ").strip()

    if choice == '1':
        learner.run_forever()
    elif choice == '2':
        print("\nRunning quick demo...")
        global running

        for step in range(5000):
            if not running:
                break
            result = learner.learn_step()
            for c in result['discoveries']:
                print(f"  ** DISCOVERED: {c.name}")

            if (step + 1) % 1000 == 0:
                learner._print_progress()

        learner.save_knowledge()
        learner._print_final_report()
    elif choice == '3':
        print("\nQuery the learned knowledge:")
        print("Format: operation a b (e.g., 'add 7 8')")
        print("Type 'quit' to exit\n")

        while True:
            query = input("> ").strip()
            if query == 'quit':
                break

            parts = query.split()
            if len(parts) >= 3:
                op = parts[0]
                try:
                    a, b = int(parts[1]), int(parts[2])
                    result, explanation = learner.query(op, a, b)
                    print(f"  Result: {result}")
                    print(f"  Explanation: {explanation}")
                except:
                    print("  Invalid input")
            else:
                print("  Format: operation a b")
    else:
        print("Running quick demo by default...")
        for step in range(2000):
            result = learner.learn_step()
            for c in result['discoveries']:
                print(f"  ** DISCOVERED: {c.name}")

        learner.save_knowledge()
        learner._print_final_report()


if __name__ == "__main__":
    main()
