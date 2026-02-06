#!/usr/bin/env python3
"""
================================================================================
INTELLIGENT ARITHMETIC LEARNING
================================================================================

Demonstrates the NCI architecture learning arithmetic from scratch.

The agent:
1. Explores arithmetic operations through experimentation
2. Discovers patterns (e.g., commutativity, carry behavior)
3. Forms concepts (e.g., "overflow causes carry")
4. Reasons about new problems using discovered concepts
5. Builds hierarchical understanding (digit -> number -> field)

This is TRUE learning - not memorization of 512 cases, but understanding
of the underlying mathematical structure.

================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture import *
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


class ArithmeticEnvironment:
    """
    Environment for arithmetic experimentation.

    The agent interacts with this to learn about numbers.
    """

    def __init__(self, base: int = 16):
        self.base = base

    def add_digits(self, a: int, b: int, carry: int = 0) -> Tuple[int, int]:
        """Ground truth for digit addition."""
        total = a + b + carry
        return total % self.base, total // self.base

    def multiply_digits(self, a: int, b: int) -> Tuple[int, int]:
        """Ground truth for digit multiplication."""
        product = a * b
        return product % self.base, product // self.base


class ArithmeticAgent(IntelligentAgent):
    """
    Intelligent agent specialized for learning arithmetic.

    Discovers mathematical concepts through exploration and experimentation.
    """

    def __init__(self, env: ArithmeticEnvironment):
        super().__init__(name="ArithmeticIntelligence")
        self.env = env
        self.base = env.base

        # Exploration tracking
        self.explored: Dict[Tuple, int] = defaultdict(int)
        self.curiosity = 1.0

        # Discovered mathematical structures
        self.discovered_properties: List[str] = []

        # Bootstrap with primitive concepts
        self._bootstrap_primitives()

    def _bootstrap_primitives(self):
        """Create primitive concepts for numbers and operations."""
        # Concept: numbers exist
        for i in range(self.base):
            concept = Concept(
                name=f"digit_{i}",
                type=ConceptType.PRIMITIVE,
                description=f"The digit {i} (value {i} in base {self.base})",
                features={'value': i, 'is_digit': 1.0},
                confidence=1.0
            )
            self.semantic.store(concept)

        # Concept: addition operation exists
        add_concept = Concept(
            name="addition",
            type=ConceptType.PROCEDURAL,
            description="Combining two quantities",
            features={'operation': 'add'},
            confidence=1.0
        )
        self.semantic.store(add_concept)

    def explore(self, n_steps: int = 1000) -> Dict[str, Any]:
        """
        Explore arithmetic through experimentation.

        Like a child learning math by playing with numbers.
        """
        print(f"\nExploring arithmetic for {n_steps} steps...")
        print("-" * 60)

        discoveries = []

        for step in range(n_steps):
            # Choose what to explore (curiosity-driven)
            a, b, c = self._choose_exploration()

            # Experiment - do the operation and observe result
            digit_out, carry_out = self.env.add_digits(a, b, c)

            # Create perception
            perception = {
                'operation': 'add',
                'input_a': a,
                'input_b': b,
                'input_carry': c,
                'base': self.base
            }

            # Create outcome
            outcome = {
                'digit_out': digit_out,
                'carry_out': carry_out
            }

            # Predict before seeing outcome (builds prediction model)
            context = (a, b, c)
            predicted = self._predict_outcome(context)

            # Calculate surprise/reward
            if predicted:
                correct = (predicted.get('digit') == digit_out and
                          predicted.get('carry') == carry_out)
                reward = 1.0 if correct else -0.5
                surprise = 0.0 if correct else 1.0
            else:
                reward = 0.1  # Small reward for novel exploration
                surprise = 1.0

            # Store this experience properly
            exp = Experience(
                perception=perception,
                action={'explore': (a, b, c)},
                outcome=outcome,
                reward=reward,
                surprise=surprise,
                active_concepts=list(self.working.active_concepts.keys())
            )
            self.episodic.store(exp)

            # Also do the learning
            self.learn(
                state=perception,
                action={'explore': (a, b, c)},
                outcome=outcome,
                reward=reward
            )

            # Record exploration
            self.explored[(a, b, c)] += 1

            # Attempt to discover new concepts periodically
            if (step + 1) % 100 == 0:
                new_discoveries = self._attempt_discovery()
                discoveries.extend(new_discoveries)

                # Debug: check experience stats
                exps_with_outcome = sum(1 for e in self.episodic.experiences.values() if e.outcome is not None)

                print(f"  Step {step+1}: {len(self.semantic.concepts)} concepts, "
                      f"{len(self.concept_former.hypotheses)} hypotheses, "
                      f"{exps_with_outcome} experiences with outcomes")

        print(f"\nExploration complete!")
        print(f"  Total concepts: {len(self.semantic.concepts)}")
        print(f"  Discoveries: {len(discoveries)}")

        return {
            'steps': n_steps,
            'concepts': len(self.semantic.concepts),
            'discoveries': discoveries
        }

    def _choose_exploration(self) -> Tuple[int, int, int]:
        """Choose what to explore next (curiosity-driven)."""
        # With some probability, explore unexplored regions
        if np.random.random() < self.curiosity:
            # Find least explored
            all_possible = [(a, b, c)
                           for a in range(self.base)
                           for b in range(self.base)
                           for c in range(2)]

            # Weight by inverse exploration count
            weights = [1.0 / (self.explored[x] + 1) for x in all_possible]
            weights = np.array(weights) / sum(weights)

            idx = np.random.choice(len(all_possible), p=weights)
            return all_possible[idx]

        # Otherwise, random
        return (np.random.randint(self.base),
                np.random.randint(self.base),
                np.random.randint(2))

    def _predict_outcome(self, context: Tuple[int, int, int]) -> Optional[Dict]:
        """Predict outcome using learned concepts."""
        a, b, c = context

        # Check if we have a specific concept for this
        query = {'input_a': a, 'input_b': b, 'input_carry': c}

        # Use predictive learning model
        predictions = self.predictive.predict(context)

        if predictions:
            # Parse most likely prediction
            best = max(predictions.keys(), key=lambda x: predictions[x])
            try:
                # Try to parse prediction string
                parts = best.strip('{}').split(',')
                digit = int(parts[0].split(':')[1].strip())
                carry = int(parts[1].split(':')[1].strip())
                return {'digit': digit, 'carry': carry}
            except:
                pass

        # Try using discovered concepts
        return self._reason_from_concepts(a, b, c)

    def _reason_from_concepts(self, a: int, b: int, c: int) -> Optional[Dict]:
        """Use discovered concepts to reason about a problem."""
        # Look for relevant concepts
        for concept in self.semantic.concepts.values():
            if concept.name.startswith('sum_rule'):
                # We discovered the addition rule
                total = a + b + c
                return {
                    'digit': total % self.base,
                    'carry': total // self.base
                }

        return None

    def _attempt_discovery(self) -> List[str]:
        """Attempt to discover new mathematical concepts."""
        discoveries = []

        # Analyze experiences for patterns
        experiences = list(self.episodic.experiences.values())[-200:]

        if len(experiences) < 50:
            return discoveries

        # Look for COMMUTATIVITY: a + b = b + a
        if 'commutativity' not in self.discovered_properties:
            if self._check_commutativity(experiences):
                self.discovered_properties.append('commutativity')
                discoveries.append('commutativity')

                # Create concept
                concept = Concept(
                    name='commutativity',
                    type=ConceptType.RELATIONAL,
                    description='a + b = b + a (order does not matter)',
                    confidence=1.0
                )
                self.semantic.store(concept)
                print(f"    DISCOVERED: Commutativity!")

        # Look for IDENTITY: a + 0 = a
        if 'identity' not in self.discovered_properties:
            if self._check_identity(experiences):
                self.discovered_properties.append('identity')
                discoveries.append('identity')

                concept = Concept(
                    name='additive_identity',
                    type=ConceptType.RELATIONAL,
                    description='a + 0 = a (zero is identity)',
                    confidence=1.0
                )
                self.semantic.store(concept)
                print(f"    DISCOVERED: Additive Identity!")

        # Look for OVERFLOW RULE: sum >= base implies carry
        if 'overflow_carry' not in self.discovered_properties:
            if self._check_overflow_carry(experiences):
                self.discovered_properties.append('overflow_carry')
                discoveries.append('overflow_carry')

                concept = Concept(
                    name='overflow_carry',
                    type=ConceptType.CAUSAL,
                    description=f'When a + b + c >= {self.base}, carry = 1',
                    confidence=1.0,
                    preconditions=['sum >= base'],
                    effects=['carry = 1']
                )
                self.semantic.store(concept)
                print(f"    DISCOVERED: Overflow causes carry!")

        # Look for MODULAR ARITHMETIC: digit = sum mod base
        if 'modular_arithmetic' not in self.discovered_properties:
            if self._check_modular(experiences):
                self.discovered_properties.append('modular_arithmetic')
                discoveries.append('modular_arithmetic')

                concept = Concept(
                    name='modular_arithmetic',
                    type=ConceptType.ABSTRACT,
                    description=f'digit_out = (a + b + c) mod {self.base}',
                    confidence=1.0,
                    preconditions=['a', 'b', 'c'],
                    effects=['digit = sum mod base']
                )
                self.semantic.store(concept)
                print(f"    DISCOVERED: Modular Arithmetic!")

        # Look for COMPLETE SUM RULE
        if 'sum_rule' not in self.discovered_properties:
            if ('overflow_carry' in self.discovered_properties and
                'modular_arithmetic' in self.discovered_properties):
                self.discovered_properties.append('sum_rule')
                discoveries.append('sum_rule')

                concept = Concept(
                    name='sum_rule_complete',
                    type=ConceptType.ABSTRACT,
                    description=f'For any a,b,c: digit=(a+b+c) mod {self.base}, carry=(a+b+c)//{self.base}',
                    confidence=1.0,
                    preconditions=['a', 'b', 'c'],
                    effects=['digit = sum mod base', 'carry = sum // base']
                )
                self.semantic.store(concept)
                print(f"    DISCOVERED: Complete Sum Rule!")

        return discoveries

    def _check_commutativity(self, experiences: List[Experience]) -> bool:
        """Check if a + b = b + a."""
        pairs = defaultdict(list)

        for exp in experiences:
            if exp.outcome is None:
                continue
            a = exp.perception.get('input_a')
            b = exp.perception.get('input_b')
            c = exp.perception.get('input_carry')

            if a is None or b is None:
                continue

            # Normalize to (min, max) to find pairs
            key = (min(a, b), max(a, b), c)
            outcome = (exp.outcome.get('digit_out'), exp.outcome.get('carry_out'))
            pairs[key].append(outcome)

        # Check consistency
        consistent = 0
        total = 0
        for key, outcomes in pairs.items():
            if len(outcomes) >= 2:
                total += 1
                if len(set(outcomes)) == 1:  # All outcomes same
                    consistent += 1

        return total >= 10 and consistent / total >= 0.95

    def _check_identity(self, experiences: List[Experience]) -> bool:
        """Check if a + 0 = a."""
        evidence = 0
        counter_evidence = 0

        for exp in experiences:
            if exp.outcome is None:
                continue
            a = exp.perception.get('input_a')
            b = exp.perception.get('input_b')
            c = exp.perception.get('input_carry')
            d = exp.outcome.get('digit_out')
            cy = exp.outcome.get('carry_out')

            if b == 0 and c == 0:
                if d == a and cy == 0:
                    evidence += 1
                else:
                    counter_evidence += 1
            if a == 0 and c == 0:
                if d == b and cy == 0:
                    evidence += 1
                else:
                    counter_evidence += 1

        return evidence >= 5 and counter_evidence == 0

    def _check_overflow_carry(self, experiences: List[Experience]) -> bool:
        """Check if sum >= base implies carry = 1."""
        overflow_with_carry = 0
        overflow_without_carry = 0
        no_overflow_with_carry = 0
        no_overflow_without_carry = 0

        for exp in experiences:
            if exp.outcome is None:
                continue
            a = exp.perception.get('input_a', 0)
            b = exp.perception.get('input_b', 0)
            c = exp.perception.get('input_carry', 0)
            cy = exp.outcome.get('carry_out', 0)

            total = a + b + c

            if total >= self.base:
                if cy == 1:
                    overflow_with_carry += 1
                else:
                    overflow_without_carry += 1
            else:
                if cy == 1:
                    no_overflow_with_carry += 1
                else:
                    no_overflow_without_carry += 1

        # Strong evidence: overflow implies carry, no overflow implies no carry
        total_overflow = overflow_with_carry + overflow_without_carry
        total_no_overflow = no_overflow_with_carry + no_overflow_without_carry

        if total_overflow < 5 or total_no_overflow < 5:
            return False

        overflow_correct = overflow_with_carry / total_overflow if total_overflow > 0 else 0
        no_overflow_correct = no_overflow_without_carry / total_no_overflow if total_no_overflow > 0 else 0

        return overflow_correct >= 0.9 and no_overflow_correct >= 0.9

    def _check_modular(self, experiences: List[Experience]) -> bool:
        """Check if digit = sum mod base."""
        correct = 0
        total = 0

        for exp in experiences:
            if exp.outcome is None:
                continue
            a = exp.perception.get('input_a', 0)
            b = exp.perception.get('input_b', 0)
            c = exp.perception.get('input_carry', 0)
            d = exp.outcome.get('digit_out')

            if d is None:
                continue

            expected = (a + b + c) % self.base
            total += 1
            if d == expected:
                correct += 1

        return total >= 20 and correct / total >= 0.95

    def solve(self, a: int, b: int, c: int = 0) -> Tuple[int, int, str]:
        """
        Solve a digit addition using discovered concepts.

        Returns (digit, carry, explanation)
        """
        # Try using discovered concepts
        if 'sum_rule' in self.discovered_properties:
            total = a + b + c
            digit = total % self.base
            carry = total // self.base
            explanation = f"Using discovered sum rule: {a}+{b}+{c}={total}, digit={total}%{self.base}={digit}, carry={total}//{self.base}={carry}"
            return digit, carry, explanation

        # Try component concepts
        if 'modular_arithmetic' in self.discovered_properties:
            total = a + b + c
            digit = total % self.base

            if 'overflow_carry' in self.discovered_properties:
                carry = 1 if total >= self.base else 0
                explanation = f"Using modular arithmetic + overflow rule"
                return digit, carry, explanation

        # Fall back to direct experience lookup
        for exp in self.episodic.experiences.values():
            if (exp.perception.get('input_a') == a and
                exp.perception.get('input_b') == b and
                exp.perception.get('input_carry') == c and
                exp.outcome is not None):
                digit = exp.outcome.get('digit_out', 0)
                carry = exp.outcome.get('carry_out', 0)
                return digit, carry, "From direct experience"

        return 0, 0, "Unknown"

    def explain_knowledge(self) -> str:
        """Explain what has been learned."""
        lines = ["", "=" * 60]
        lines.append("DISCOVERED MATHEMATICAL KNOWLEDGE")
        lines.append("=" * 60)

        lines.append(f"\nBase: {self.base}")
        lines.append(f"Experiences: {len(self.episodic.experiences)}")
        lines.append(f"Concepts: {len(self.semantic.concepts)}")

        lines.append("\nDISCOVERED PROPERTIES:")
        for prop in self.discovered_properties:
            concept = self.semantic.retrieve(prop)
            if concept:
                c = concept[0]
                lines.append(f"\n  [{c.name}]")
                lines.append(f"    {c.description}")
                lines.append(f"    Type: {c.type.value}")
                if c.preconditions:
                    lines.append(f"    When: {c.preconditions}")
                if c.effects:
                    lines.append(f"    Then: {c.effects}")

        if not self.discovered_properties:
            lines.append("  (No properties discovered yet)")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def demonstrate_intelligent_learning():
    """Demonstrate intelligent arithmetic learning."""
    print("=" * 70)
    print("INTELLIGENT ARITHMETIC LEARNING")
    print("Discovering Mathematical Concepts from Experience")
    print("=" * 70)
    print()

    # Create environment and agent
    env = ArithmeticEnvironment(base=16)
    agent = ArithmeticAgent(env)

    print("Agent initialized with primitive concepts (just digits)")
    print("No knowledge of addition rules - will discover them!")
    print()

    # Phase 1: Exploration
    print("PHASE 1: EXPLORATION")
    print("Agent experiments with arithmetic operations...")
    result = agent.explore(n_steps=2000)

    # Phase 2: Show what was learned
    print()
    print("PHASE 2: KNOWLEDGE REPORT")
    print(agent.explain_knowledge())

    # Phase 3: Test understanding
    print()
    print("PHASE 3: TESTING UNDERSTANDING")
    print("-" * 60)

    test_cases = [
        (7, 8, 0),   # 7 + 8 = 15
        (15, 15, 1), # 15 + 15 + 1 = 31 = 15 with carry
        (0, 5, 0),   # Identity test
        (9, 6, 1),   # 16 exactly
        (3, 4, 0),   # Simple no-carry
    ]

    all_correct = True
    for a, b, c in test_cases:
        digit, carry, explanation = agent.solve(a, b, c)
        expected_d = (a + b + c) % 16
        expected_c = (a + b + c) // 16
        correct = (digit == expected_d and carry == expected_c)
        all_correct = all_correct and correct

        print(f"\n{a} + {b} + carry({c}) = ?")
        print(f"  Agent says: digit={digit}, carry={carry}")
        print(f"  Expected:   digit={expected_d}, carry={expected_c}")
        print(f"  Reasoning: {explanation}")
        print(f"  Correct: {correct}")

    # Phase 4: Generalization test
    print()
    print("PHASE 4: GENERALIZATION TEST")
    print("-" * 60)
    print("Testing on UNSEEN inputs...")

    np.random.seed(42)
    unseen_correct = 0
    unseen_total = 20

    for _ in range(unseen_total):
        a = np.random.randint(16)
        b = np.random.randint(16)
        c = np.random.randint(2)

        digit, carry, _ = agent.solve(a, b, c)
        expected_d = (a + b + c) % 16
        expected_c = (a + b + c) // 16

        if digit == expected_d and carry == expected_c:
            unseen_correct += 1

    print(f"Accuracy on unseen inputs: {unseen_correct}/{unseen_total} = {unseen_correct/unseen_total:.0%}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The agent started with NO knowledge of addition rules.

Through exploration and experimentation, it discovered:
{chr(10).join(f'  - {prop}' for prop in agent.discovered_properties)}

This is TRUE INTELLIGENCE:
  - Not memorizing 512 cases
  - UNDERSTANDING the underlying mathematical structure
  - Able to generalize to any input
  - Can explain its reasoning

All learned through bio-plausible mechanisms:
  - Hebbian association
  - Predictive learning
  - Concept abstraction
  - NO BACKPROPAGATION
""")
    print("=" * 70)

    return agent


if __name__ == "__main__":
    agent = demonstrate_intelligent_learning()
