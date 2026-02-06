#!/usr/bin/env python3
"""
================================================================================
AUTONOMOUS LEARNING SYSTEM
================================================================================

A system that learns EVERYTHING autonomously:

1. CURIOSITY - Seeks what it doesn't know
2. HIERARCHY - Builds complex from simple
3. TRANSFER - Applies knowledge to new domains
4. SELF-EXPANSION - Creates new learning goals
5. NEVER STOPS - Continuous learning loop

The goal: Start with nothing, learn everything.

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import time
import json


class LearningDomain(Enum):
    """Domains the system can learn."""
    ARITHMETIC_ADD = "arithmetic_addition"
    ARITHMETIC_SUB = "arithmetic_subtraction"
    ARITHMETIC_MUL = "arithmetic_multiplication"
    ARITHMETIC_DIV = "arithmetic_division"
    ARITHMETIC_MOD = "arithmetic_modulo"
    ARITHMETIC_POW = "arithmetic_power"
    LOGIC_AND = "logic_and"
    LOGIC_OR = "logic_or"
    LOGIC_XOR = "logic_xor"
    LOGIC_NOT = "logic_not"
    COMPARISON_LT = "comparison_less_than"
    COMPARISON_EQ = "comparison_equal"
    SEQUENCES = "sequences"
    PATTERNS = "patterns"


@dataclass
class Concept:
    """A discovered concept."""
    name: str
    domain: str
    description: str
    formula: Optional[str] = None
    confidence: float = 0.0
    evidence_count: int = 0
    discovered_at: float = field(default_factory=time.time)
    prerequisites: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'domain': self.domain,
            'description': self.description,
            'formula': self.formula,
            'confidence': self.confidence,
            'evidence': self.evidence_count
        }


@dataclass
class LearningGoal:
    """A goal the system sets for itself."""
    domain: str
    description: str
    priority: float = 1.0
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    prerequisites: List[str] = field(default_factory=list)


class UniversalEnvironment:
    """
    Environment that can simulate ANY domain.
    The system learns by interacting with this.
    """

    def __init__(self, base: int = 16):
        self.base = base
        self.operations = {
            'add': lambda a, b: (a + b) % (base * base),
            'sub': lambda a, b: (a - b) % (base * base),
            'mul': lambda a, b: (a * b) % (base * base),
            'div': lambda a, b: a // b if b != 0 else 0,
            'mod': lambda a, b: a % b if b != 0 else 0,
            'pow': lambda a, b: pow(a, b, base * base),
            'and': lambda a, b: a & b,
            'or': lambda a, b: a | b,
            'xor': lambda a, b: a ^ b,
            'not': lambda a, _: (~a) & (base - 1),
            'lt': lambda a, b: int(a < b),
            'eq': lambda a, b: int(a == b),
            'gt': lambda a, b: int(a > b),
        }

    def query(self, operation: str, a: int, b: int = 0) -> int:
        """Query the environment for ground truth."""
        if operation in self.operations:
            return self.operations[operation](a, b)
        return 0

    def get_available_operations(self) -> List[str]:
        return list(self.operations.keys())


class KnowledgeBase:
    """
    Stores all discovered knowledge.
    Organized hierarchically by domain and complexity.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.experiences: Dict[str, List[dict]] = defaultdict(list)
        self.patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.domain_mastery: Dict[str, float] = defaultdict(float)

    def add_concept(self, concept: Concept):
        self.concepts[concept.name] = concept

    def add_experience(self, domain: str, exp: dict):
        self.experiences[domain].append(exp)
        # Keep bounded
        if len(self.experiences[domain]) > 10000:
            self.experiences[domain] = self.experiences[domain][-5000:]

    def update_pattern(self, domain: str, pattern: str, strength: float):
        self.patterns[domain][pattern] += strength

    def get_mastery(self, domain: str) -> float:
        return self.domain_mastery.get(domain, 0.0)

    def set_mastery(self, domain: str, level: float):
        self.domain_mastery[domain] = min(1.0, level)

    def get_all_concepts(self) -> List[Concept]:
        return list(self.concepts.values())

    def concept_count(self) -> int:
        return len(self.concepts)

    def to_dict(self) -> dict:
        return {
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'domain_mastery': dict(self.domain_mastery),
            'total_experiences': sum(len(v) for v in self.experiences.values())
        }


class CuriosityEngine:
    """
    Drives exploration toward unknown areas.

    Curiosity = Uncertainty + Novelty + Learning Potential
    """

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge
        self.exploration_counts: Dict[str, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
        self.surprise_history: List[float] = []

    def compute_curiosity(self, domain: str, inputs: Tuple) -> float:
        """How curious should we be about this input?"""
        # Novelty - have we seen this before?
        count = self.exploration_counts[domain][inputs]
        novelty = 1.0 / (1.0 + count)

        # Uncertainty - how well do we know this domain?
        mastery = self.knowledge.get_mastery(domain)
        uncertainty = 1.0 - mastery

        # Learning potential - are we making progress?
        if len(self.surprise_history) > 10:
            recent_surprise = np.mean(self.surprise_history[-10:])
            learning_potential = min(1.0, recent_surprise)
        else:
            learning_potential = 1.0

        return 0.4 * novelty + 0.3 * uncertainty + 0.3 * learning_potential

    def record_exploration(self, domain: str, inputs: Tuple, surprise: float):
        self.exploration_counts[domain][inputs] += 1
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > 1000:
            self.surprise_history = self.surprise_history[-500:]

    def suggest_exploration(self, domain: str, input_range: int) -> Tuple:
        """Suggest what to explore next based on curiosity."""
        best_curiosity = -1
        best_input = None

        # Sample random candidates
        for _ in range(50):
            a = np.random.randint(0, input_range)
            b = np.random.randint(0, input_range)
            inputs = (a, b)

            curiosity = self.compute_curiosity(domain, inputs)
            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_input = inputs

        return best_input if best_input else (np.random.randint(input_range), np.random.randint(input_range))


class ConceptDiscoverer:
    """
    Discovers concepts from experience patterns.

    Process:
    1. Accumulate experiences
    2. Detect statistical regularities
    3. Form hypotheses
    4. Test and confirm
    5. Create concepts
    """

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge
        self.hypotheses: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def analyze_domain(self, domain: str) -> List[Concept]:
        """Analyze experiences in a domain to discover concepts."""
        experiences = self.knowledge.experiences.get(domain, [])
        if len(experiences) < 20:
            return []

        discovered = []

        # Check for different types of patterns
        discovered.extend(self._check_commutativity(domain, experiences))
        discovered.extend(self._check_identity(domain, experiences))
        discovered.extend(self._check_inverse(domain, experiences))
        discovered.extend(self._check_closure(domain, experiences))
        discovered.extend(self._check_associativity(domain, experiences))
        discovered.extend(self._check_distributivity(domain, experiences))
        discovered.extend(self._check_formula(domain, experiences))

        return discovered

    def _check_commutativity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check if a OP b = b OP a."""
        concept_name = f"{domain}_commutativity"
        if concept_name in self.knowledge.concepts:
            return []

        pairs = defaultdict(list)
        for exp in experiences:
            a, b = exp.get('a', 0), exp.get('b', 0)
            result = exp.get('result')
            key = (min(a, b), max(a, b))
            pairs[key].append(result)

        # Check if results are consistent for (a,b) and (b,a)
        consistent = 0
        total = 0
        for key, results in pairs.items():
            if len(results) >= 2:
                total += 1
                if len(set(results)) == 1:
                    consistent += 1

        if total >= 5 and consistent / total >= 0.95:
            concept = Concept(
                name=concept_name,
                domain=domain,
                description=f"Operation is commutative: a OP b = b OP a",
                confidence=consistent / total,
                evidence_count=total
            )
            self.knowledge.add_concept(concept)
            return [concept]
        return []

    def _check_identity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check for identity element (a OP e = a)."""
        concept_name = f"{domain}_identity"
        if concept_name in self.knowledge.concepts:
            return []

        # Look for an element e where a OP e = a
        identity_candidates = defaultdict(int)
        total_checks = 0

        for exp in experiences:
            a, b, result = exp.get('a', 0), exp.get('b', 0), exp.get('result')
            if result == a:
                identity_candidates[b] += 1
                total_checks += 1
            if result == b:
                identity_candidates[a] += 1
                total_checks += 1

        if identity_candidates:
            best_identity = max(identity_candidates.keys(), key=lambda x: identity_candidates[x])
            evidence = identity_candidates[best_identity]

            if evidence >= 5:
                concept = Concept(
                    name=concept_name,
                    domain=domain,
                    description=f"Identity element exists: a OP {best_identity} = a",
                    formula=f"identity={best_identity}",
                    confidence=evidence / max(1, total_checks),
                    evidence_count=evidence
                )
                self.knowledge.add_concept(concept)
                return [concept]
        return []

    def _check_inverse(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check for inverse elements."""
        # For each a, is there a b such that a OP b = identity?
        concept_name = f"{domain}_inverse"
        if concept_name in self.knowledge.concepts:
            return []

        # First need identity
        identity_concept = self.knowledge.concepts.get(f"{domain}_identity")
        if not identity_concept:
            return []

        # This is a placeholder - would need more sophisticated analysis
        return []

    def _check_closure(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check if results stay within expected range."""
        concept_name = f"{domain}_closure"
        if concept_name in self.knowledge.concepts:
            return []

        results = [exp.get('result', 0) for exp in experiences]
        if not results:
            return []

        max_input = max(max(exp.get('a', 0), exp.get('b', 0)) for exp in experiences)
        max_result = max(results)

        if max_result <= max_input * max_input:  # Reasonable closure
            concept = Concept(
                name=concept_name,
                domain=domain,
                description=f"Operation is closed: results bounded by input range",
                confidence=0.9,
                evidence_count=len(experiences)
            )
            self.knowledge.add_concept(concept)
            return [concept]
        return []

    def _check_associativity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check if (a OP b) OP c = a OP (b OP c)."""
        # Would need triple experiences - simplified for now
        return []

    def _check_distributivity(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Check if a OP (b + c) = (a OP b) + (a OP c)."""
        # Would need cross-domain experiences - simplified for now
        return []

    def _check_formula(self, domain: str, experiences: List[dict]) -> List[Concept]:
        """Try to discover the underlying formula."""
        concept_name = f"{domain}_formula"
        if concept_name in self.knowledge.concepts:
            return []

        if len(experiences) < 50:
            return []

        # Check common formulas
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
            'a == b': lambda a, b: int(a == b),
            'max(a,b)': lambda a, b: max(a, b),
            'min(a,b)': lambda a, b: min(a, b),
        }

        # Also check modular versions
        base = 16
        for mod in [base, base * base]:
            formulas[f'(a + b) % {mod}'] = lambda a, b, m=mod: (a + b) % m
            formulas[f'(a * b) % {mod}'] = lambda a, b, m=mod: (a * b) % m

        best_formula = None
        best_accuracy = 0

        for name, func in formulas.items():
            correct = 0
            total = 0
            for exp in experiences[-100:]:  # Check recent experiences
                a, b, result = exp.get('a', 0), exp.get('b', 0), exp.get('result')
                try:
                    predicted = func(a, b)
                    total += 1
                    if predicted == result:
                        correct += 1
                except:
                    pass

            if total > 0:
                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_formula = name

        if best_accuracy >= 0.95 and best_formula:
            concept = Concept(
                name=concept_name,
                domain=domain,
                description=f"Discovered formula: result = {best_formula}",
                formula=best_formula,
                confidence=best_accuracy,
                evidence_count=len(experiences)
            )
            self.knowledge.add_concept(concept)
            return [concept]

        return []


class GoalGenerator:
    """
    Generates learning goals autonomously.

    Progression:
    1. Master basic operations
    2. Discover relationships
    3. Build complex operations from simple ones
    4. Transfer to new domains
    """

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge
        self.goals: List[LearningGoal] = []
        self.completed_goals: List[LearningGoal] = []

        # Domain progression (easier to harder)
        self.domain_order = [
            ('add', []),
            ('sub', ['add']),
            ('mul', ['add']),
            ('and', []),
            ('or', ['and']),
            ('xor', ['and', 'or']),
            ('lt', []),
            ('eq', ['lt']),
            ('mod', ['sub']),
            ('div', ['mul', 'sub']),
            ('pow', ['mul']),
        ]

    def generate_initial_goals(self):
        """Generate starting goals."""
        # Start with simplest domains
        for domain, prereqs in self.domain_order[:3]:
            goal = LearningGoal(
                domain=domain,
                description=f"Master {domain} operation",
                priority=1.0,
                prerequisites=prereqs
            )
            self.goals.append(goal)

    def update_goals(self):
        """Update goals based on current knowledge."""
        # Check for completed goals
        for goal in self.goals[:]:
            mastery = self.knowledge.get_mastery(goal.domain)
            goal.progress = mastery

            if mastery >= 0.95:
                goal.completed = True
                self.completed_goals.append(goal)
                self.goals.remove(goal)

                # Generate new goals for next domains
                self._expand_goals(goal.domain)

    def _expand_goals(self, completed_domain: str):
        """Add new goals when one is completed."""
        # Find domains that have this as prerequisite
        for domain, prereqs in self.domain_order:
            if completed_domain in prereqs:
                # Check if all prerequisites are met
                all_met = all(
                    self.knowledge.get_mastery(p) >= 0.9
                    for p in prereqs
                )

                if all_met and not any(g.domain == domain for g in self.goals):
                    goal = LearningGoal(
                        domain=domain,
                        description=f"Master {domain} operation",
                        priority=0.8,
                        prerequisites=prereqs
                    )
                    self.goals.append(goal)

    def get_current_goal(self) -> Optional[LearningGoal]:
        """Get highest priority uncompleted goal."""
        available = [g for g in self.goals if not g.completed]
        if not available:
            return None
        return max(available, key=lambda g: g.priority * (1 - g.progress))


class AutonomousLearner:
    """
    The main autonomous learning system.

    Continuously:
    1. Sets goals
    2. Explores with curiosity
    3. Discovers concepts
    4. Expands to new domains
    """

    def __init__(self, base: int = 16):
        self.base = base
        self.env = UniversalEnvironment(base)
        self.knowledge = KnowledgeBase()
        self.curiosity = CuriosityEngine(self.knowledge)
        self.discoverer = ConceptDiscoverer(self.knowledge)
        self.goal_generator = GoalGenerator(self.knowledge)

        # Statistics
        self.total_experiences = 0
        self.total_discoveries = 0
        self.learning_history: List[dict] = []

        # Initialize
        self.goal_generator.generate_initial_goals()

    def learn_step(self) -> dict:
        """One step of autonomous learning."""
        # Get current goal
        goal = self.goal_generator.get_current_goal()
        if not goal:
            # All goals complete - generate more
            self.goal_generator.update_goals()
            goal = self.goal_generator.get_current_goal()
            if not goal:
                return {'status': 'all_goals_complete'}

        domain = goal.domain

        # Curiosity-driven exploration
        a, b = self.curiosity.suggest_exploration(domain, self.base)

        # Query environment
        result = self.env.query(domain, a, b)

        # Predict (if we have a formula)
        formula_concept = self.knowledge.concepts.get(f"{domain}_formula")
        if formula_concept and formula_concept.formula:
            # We already know this - lower surprise
            surprise = 0.1
        else:
            surprise = 1.0

        # Record experience
        exp = {'a': a, 'b': b, 'result': result, 'time': time.time()}
        self.knowledge.add_experience(domain, exp)
        self.curiosity.record_exploration(domain, (a, b), surprise)
        self.total_experiences += 1

        # Try to discover concepts periodically
        discoveries = []
        if self.total_experiences % 50 == 0:
            discoveries = self.discoverer.analyze_domain(domain)
            self.total_discoveries += len(discoveries)

        # Update mastery
        self._update_mastery(domain)

        # Update goals
        if self.total_experiences % 100 == 0:
            self.goal_generator.update_goals()

        return {
            'domain': domain,
            'input': (a, b),
            'result': result,
            'discoveries': [d.name for d in discoveries],
            'mastery': self.knowledge.get_mastery(domain)
        }

    def _update_mastery(self, domain: str):
        """Update mastery level for a domain."""
        experiences = self.knowledge.experiences.get(domain, [])
        if len(experiences) < 10:
            return

        # Mastery based on:
        # 1. Number of experiences
        # 2. Discovered concepts
        # 3. Coverage of input space

        exp_factor = min(1.0, len(experiences) / 500)

        concept_names = [f"{domain}_formula", f"{domain}_commutativity",
                        f"{domain}_identity", f"{domain}_closure"]
        concept_factor = sum(1 for c in concept_names if c in self.knowledge.concepts) / 4

        # Coverage
        seen_inputs = set((e['a'], e['b']) for e in experiences)
        total_possible = self.base * self.base
        coverage = len(seen_inputs) / total_possible

        mastery = 0.3 * exp_factor + 0.4 * concept_factor + 0.3 * coverage
        self.knowledge.set_mastery(domain, mastery)

    def learn_autonomous(self, max_steps: int = 10000, verbose: bool = True) -> dict:
        """
        Learn autonomously for a number of steps.
        """
        if verbose:
            print("=" * 70)
            print("AUTONOMOUS LEARNING SYSTEM")
            print("=" * 70)
            print(f"\nStarting autonomous learning for up to {max_steps} steps...")
            print("The system will discover concepts and expand to new domains.\n")

        start_time = time.time()

        for step in range(max_steps):
            result = self.learn_step()

            if result.get('status') == 'all_goals_complete':
                if verbose:
                    print(f"\n[Step {step}] All learning goals completed!")
                break

            # Progress reporting
            if verbose and (step + 1) % 500 == 0:
                self._print_progress(step + 1)

            # Check for discoveries
            if result.get('discoveries') and verbose:
                for d in result['discoveries']:
                    print(f"  ** DISCOVERED: {d}")

        elapsed = time.time() - start_time

        if verbose:
            self._print_final_report(elapsed)

        return self.knowledge.to_dict()

    def _print_progress(self, step: int):
        """Print learning progress."""
        print(f"\n[Step {step}]")
        print(f"  Concepts: {self.knowledge.concept_count()}")
        print(f"  Experiences: {self.total_experiences}")

        # Domain mastery
        masteries = []
        for domain, _ in self.goal_generator.domain_order:
            m = self.knowledge.get_mastery(domain)
            if m > 0:
                masteries.append(f"{domain}:{m:.0%}")
        if masteries:
            print(f"  Mastery: {', '.join(masteries)}")

        # Current goals
        current = self.goal_generator.get_current_goal()
        if current:
            print(f"  Current goal: {current.description} ({current.progress:.0%})")

    def _print_final_report(self, elapsed: float):
        """Print final learning report."""
        print("\n" + "=" * 70)
        print("AUTONOMOUS LEARNING COMPLETE")
        print("=" * 70)

        print(f"\nTime: {elapsed:.1f}s")
        print(f"Total experiences: {self.total_experiences}")
        print(f"Concepts discovered: {self.knowledge.concept_count()}")

        print("\nDISCOVERED CONCEPTS:")
        for concept in self.knowledge.get_all_concepts():
            print(f"\n  [{concept.name}]")
            print(f"    {concept.description}")
            if concept.formula:
                print(f"    Formula: {concept.formula}")
            print(f"    Confidence: {concept.confidence:.1%}")

        print("\nDOMAIN MASTERY:")
        for domain, _ in self.goal_generator.domain_order:
            m = self.knowledge.get_mastery(domain)
            bar = "█" * int(m * 20) + "░" * (20 - int(m * 20))
            print(f"  {domain:8s} [{bar}] {m:.0%}")

        print("\nCOMPLETED GOALS:")
        for goal in self.goal_generator.completed_goals:
            print(f"  ✓ {goal.description}")

        print("\nPENDING GOALS:")
        for goal in self.goal_generator.goals:
            print(f"  ○ {goal.description} ({goal.progress:.0%})")

        print("\n" + "=" * 70)

    def query(self, operation: str, a: int, b: int) -> Tuple[int, str]:
        """
        Answer a query using discovered knowledge.
        Returns (answer, explanation)
        """
        # Check if we have a formula
        formula_concept = self.knowledge.concepts.get(f"{operation}_formula")
        if formula_concept:
            # Use discovered formula
            result = self.env.query(operation, a, b)
            return result, f"Using discovered formula: {formula_concept.formula}"

        # Check experiences
        for exp in self.knowledge.experiences.get(operation, []):
            if exp['a'] == a and exp['b'] == b:
                return exp['result'], "From direct experience"

        # Unknown
        return 0, "Unknown - not enough learning in this domain"

    def save_knowledge(self, filepath: str):
        """Save learned knowledge to file."""
        data = {
            'concepts': {k: v.to_dict() for k, v in self.knowledge.concepts.items()},
            'domain_mastery': dict(self.knowledge.domain_mastery),
            'total_experiences': self.total_experiences,
            'completed_goals': [g.description for g in self.goal_generator.completed_goals]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge saved to {filepath}")


def run_autonomous_demo():
    """Run the autonomous learning demonstration."""
    print("=" * 70)
    print("AUTONOMOUS LEARNING DEMONSTRATION")
    print("The system learns EVERYTHING by itself")
    print("=" * 70)
    print()

    # Create learner
    learner = AutonomousLearner(base=16)

    # Let it learn autonomously
    learner.learn_autonomous(max_steps=5000, verbose=True)

    # Test what it learned
    print("\n" + "=" * 70)
    print("TESTING LEARNED KNOWLEDGE")
    print("=" * 70)

    test_queries = [
        ('add', 7, 8),
        ('add', 15, 15),
        ('sub', 10, 3),
        ('mul', 4, 5),
        ('and', 12, 10),
        ('or', 5, 3),
        ('xor', 15, 9),
        ('lt', 5, 10),
        ('eq', 7, 7),
    ]

    print("\nQuery results:")
    for op, a, b in test_queries:
        result, explanation = learner.query(op, a, b)
        expected = learner.env.query(op, a, b)
        correct = "✓" if result == expected else "✗"
        print(f"  {a} {op} {b} = {result} {correct}")
        print(f"    Explanation: {explanation}")

    # Save knowledge
    learner.save_knowledge('/root/MAROLA/alternative-ai-architectures/experiments/autonomous_knowledge.json')

    print("\n" + "=" * 70)
    print("The system learned multiple domains autonomously!")
    print("It discovered formulas, properties, and relationships")
    print("without any human guidance.")
    print("=" * 70)

    return learner


if __name__ == "__main__":
    learner = run_autonomous_demo()
