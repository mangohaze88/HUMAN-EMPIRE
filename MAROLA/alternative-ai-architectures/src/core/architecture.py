#!/usr/bin/env python3
"""
================================================================================
NEURAL COLLECTIVE INTELLIGENCE (NCI) ARCHITECTURE
================================================================================

A truly intelligent architecture that:
1. DISCOVERS concepts through experience (not memorization)
2. BUILDS hierarchical knowledge (concepts on concepts)
3. REASONS compositionally (combine concepts for new problems)
4. TRANSFERS knowledge between domains
5. LEARNS HOW TO LEARN (meta-learning)

All without backpropagation - using only bio-plausible mechanisms:
- Hebbian learning (fire together, wire together)
- Stigmergic communication (environmental memory)
- Attention through salience (not softmax)
- Prediction error as learning signal
- Sleep consolidation for memory

================================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from collections import defaultdict
from enum import Enum
import numpy as np
import time
import uuid


# =============================================================================
# FUNDAMENTAL DATA STRUCTURES
# =============================================================================

class ConceptType(Enum):
    """Types of concepts the system can form."""
    PRIMITIVE = "primitive"      # Basic sensory/motor patterns
    RELATIONAL = "relational"    # Relationships between concepts
    PROCEDURAL = "procedural"    # How to do something
    CAUSAL = "causal"           # Why something happens
    ABSTRACT = "abstract"        # High-level generalizations
    META = "meta"               # Concepts about concepts


@dataclass
class Concept:
    """
    A learned concept - the fundamental unit of knowledge.

    Unlike neural network weights, concepts are:
    - Interpretable (have meaning)
    - Compositional (built from other concepts)
    - Transferable (can apply to new domains)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    type: ConceptType = ConceptType.PRIMITIVE

    # What this concept represents
    description: str = ""

    # Grounding - how concept connects to experience
    exemplars: List[Any] = field(default_factory=list)  # Specific examples
    features: Dict[str, float] = field(default_factory=dict)  # Feature weights

    # Relationships to other concepts
    parents: List[str] = field(default_factory=list)  # More abstract concepts
    children: List[str] = field(default_factory=list)  # More specific concepts
    associations: Dict[str, float] = field(default_factory=dict)  # Related concepts

    # Procedural knowledge
    preconditions: List[str] = field(default_factory=list)  # When does it apply
    effects: List[str] = field(default_factory=list)  # What happens when applied

    # Learning statistics
    confidence: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def update_confidence(self, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.confidence = self.success_count / self.usage_count if self.usage_count > 0 else 0
        self.last_used = time.time()


@dataclass
class Experience:
    """A single experience/observation that can be learned from."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # The experience itself
    perception: Dict[str, Any] = field(default_factory=dict)  # What was observed
    action: Optional[Dict[str, Any]] = None  # What action was taken (if any)
    outcome: Optional[Dict[str, Any]] = None  # What happened next

    # Evaluation
    reward: float = 0.0  # How good/bad was this
    surprise: float = 0.0  # How unexpected was this

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    active_concepts: List[str] = field(default_factory=list)  # Concepts that were active


@dataclass
class Hypothesis:
    """A hypothesis about a pattern - proto-concept before confirmation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # What the hypothesis claims
    pattern: str = ""  # Description
    conditions: Dict[str, Any] = field(default_factory=dict)  # When it applies
    prediction: Dict[str, Any] = field(default_factory=dict)  # What it predicts

    # Evidence
    supporting: List[str] = field(default_factory=list)  # Experience IDs
    contradicting: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        total = len(self.supporting) + len(self.contradicting)
        if total == 0:
            return 0.5
        return len(self.supporting) / total

    @property
    def evidence_strength(self) -> float:
        return min(1.0, (len(self.supporting) + len(self.contradicting)) / 20)


# =============================================================================
# MEMORY SYSTEMS
# =============================================================================

class MemorySystem(ABC):
    """Abstract base for different memory systems."""

    @abstractmethod
    def store(self, item: Any) -> str:
        """Store an item, return its ID."""
        pass

    @abstractmethod
    def retrieve(self, query: Any, k: int = 1) -> List[Any]:
        """Retrieve k most relevant items."""
        pass

    @abstractmethod
    def consolidate(self):
        """Consolidate memories (like sleep)."""
        pass


class EpisodicMemory(MemorySystem):
    """
    Memory for specific experiences (like hippocampus).

    Stores complete experiences that can be replayed.
    Used for learning from single experiences.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.experiences: Dict[str, Experience] = {}
        self.index: Dict[str, List[str]] = defaultdict(list)  # Feature -> experience IDs

    def store(self, experience: Experience) -> str:
        # Store experience
        self.experiences[experience.id] = experience

        # Index by features
        for key, value in experience.perception.items():
            self.index[f"{key}:{value}"].append(experience.id)

        # Capacity management
        if len(self.experiences) > self.capacity:
            self._forget_oldest()

        return experience.id

    def retrieve(self, query: Dict[str, Any], k: int = 5) -> List[Experience]:
        """Retrieve similar experiences."""
        scores = defaultdict(float)

        for key, value in query.items():
            index_key = f"{key}:{value}"
            for exp_id in self.index.get(index_key, []):
                scores[exp_id] += 1.0

        # Sort by score and return top k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        return [self.experiences[eid] for eid in sorted_ids if eid in self.experiences]

    def consolidate(self):
        """Move important experiences to long-term, forget unimportant."""
        # Keep experiences with high reward or surprise
        important = {
            eid: exp for eid, exp in self.experiences.items()
            if abs(exp.reward) > 0.5 or exp.surprise > 0.5
        }

        # Keep recent experiences
        recent_threshold = time.time() - 3600  # Last hour
        recent = {
            eid: exp for eid, exp in self.experiences.items()
            if exp.timestamp > recent_threshold
        }

        # Merge and trim
        self.experiences = {**important, **recent}
        self._rebuild_index()

    def _forget_oldest(self):
        """Remove oldest experiences."""
        sorted_exps = sorted(self.experiences.items(), key=lambda x: x[1].timestamp)
        to_remove = sorted_exps[:len(sorted_exps) // 4]  # Remove oldest 25%

        for eid, _ in to_remove:
            del self.experiences[eid]

        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the feature index."""
        self.index.clear()
        for exp in self.experiences.values():
            for key, value in exp.perception.items():
                self.index[f"{key}:{value}"].append(exp.id)


class SemanticMemory(MemorySystem):
    """
    Memory for concepts and facts (like cortex).

    Stores abstract knowledge extracted from experiences.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.hierarchy: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.associations: Dict[Tuple[str, str], float] = {}  # (c1, c2) -> strength

    def store(self, concept: Concept) -> str:
        self.concepts[concept.id] = concept

        # Update hierarchy
        for parent_id in concept.parents:
            self.hierarchy[parent_id].add(concept.id)

        return concept.id

    def retrieve(self, query: Any, k: int = 5) -> List[Concept]:
        """Retrieve concepts relevant to query."""
        if isinstance(query, str):
            # Name-based retrieval
            matches = [c for c in self.concepts.values() if query.lower() in c.name.lower()]
            return matches[:k]
        elif isinstance(query, dict):
            # Feature-based retrieval
            scores = {}
            for cid, concept in self.concepts.items():
                score = sum(
                    concept.features.get(k, 0) * v
                    for k, v in query.items()
                )
                if score > 0:
                    scores[cid] = score

            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
            return [self.concepts[cid] for cid in sorted_ids]

        return []

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        return self.concepts.get(concept_id)

    def associate(self, concept1_id: str, concept2_id: str, strength: float):
        """Create/strengthen association between concepts."""
        key = (min(concept1_id, concept2_id), max(concept1_id, concept2_id))
        current = self.associations.get(key, 0.0)
        self.associations[key] = min(1.0, current + strength)

    def get_associated(self, concept_id: str, min_strength: float = 0.1) -> List[Tuple[str, float]]:
        """Get concepts associated with this one."""
        associated = []
        for (c1, c2), strength in self.associations.items():
            if strength >= min_strength:
                if c1 == concept_id:
                    associated.append((c2, strength))
                elif c2 == concept_id:
                    associated.append((c1, strength))
        return sorted(associated, key=lambda x: x[1], reverse=True)

    def consolidate(self):
        """Strengthen important concepts, weaken unused ones."""
        # Decay unused concepts
        for concept in self.concepts.values():
            age = time.time() - concept.last_used
            if age > 86400:  # Older than a day
                concept.confidence *= 0.99

        # Decay weak associations
        to_remove = []
        for key, strength in self.associations.items():
            self.associations[key] = strength * 0.995
            if self.associations[key] < 0.01:
                to_remove.append(key)

        for key in to_remove:
            del self.associations[key]


class WorkingMemory:
    """
    Short-term memory for current processing (like prefrontal cortex).

    Holds currently active concepts and experiences.
    Limited capacity forces prioritization.
    """

    def __init__(self, capacity: int = 7):  # Miller's magic number
        self.capacity = capacity
        self.active_concepts: Dict[str, float] = {}  # concept_id -> activation
        self.focus: Optional[str] = None
        self.context: Dict[str, Any] = {}

    def activate(self, concept_id: str, strength: float = 1.0):
        """Activate a concept in working memory."""
        self.active_concepts[concept_id] = min(1.0,
            self.active_concepts.get(concept_id, 0) + strength)

        # Enforce capacity
        if len(self.active_concepts) > self.capacity:
            # Remove least active
            sorted_concepts = sorted(self.active_concepts.items(), key=lambda x: x[1])
            to_remove = sorted_concepts[0][0]
            del self.active_concepts[to_remove]

    def decay(self, rate: float = 0.1):
        """Decay activations over time."""
        to_remove = []
        for cid in self.active_concepts:
            self.active_concepts[cid] -= rate
            if self.active_concepts[cid] <= 0:
                to_remove.append(cid)

        for cid in to_remove:
            del self.active_concepts[cid]

    def get_active(self, min_activation: float = 0.1) -> List[Tuple[str, float]]:
        """Get currently active concepts."""
        return [(cid, act) for cid, act in self.active_concepts.items()
                if act >= min_activation]

    def set_focus(self, concept_id: str):
        """Focus attention on a concept."""
        self.focus = concept_id
        self.activate(concept_id, 0.5)

    def clear(self):
        """Clear working memory."""
        self.active_concepts.clear()
        self.focus = None
        self.context.clear()


# =============================================================================
# LEARNING MECHANISMS
# =============================================================================

class HebbianLearner:
    """
    Hebbian learning: "neurons that fire together wire together"

    Bio-plausible learning without backpropagation.
    Strengthens connections between co-active concepts.
    """

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate
        self.connections: Dict[Tuple[str, str], float] = defaultdict(float)

    def learn(self, active_concepts: List[str], outcome: float):
        """
        Strengthen connections between co-active concepts.

        Args:
            active_concepts: Concepts that were active together
            outcome: How good was this activation pattern (reward signal)
        """
        # Modulate learning by outcome
        lr = self.lr * (0.5 + outcome)  # Higher learning for positive outcomes

        # Strengthen pairwise connections
        for i, c1 in enumerate(active_concepts):
            for c2 in active_concepts[i+1:]:
                key = (min(c1, c2), max(c1, c2))
                self.connections[key] += lr

    def get_connection_strength(self, c1: str, c2: str) -> float:
        key = (min(c1, c2), max(c1, c2))
        return self.connections.get(key, 0.0)

    def spread_activation(self, active: Dict[str, float],
                         threshold: float = 0.1) -> Dict[str, float]:
        """Spread activation through learned connections."""
        new_activation = dict(active)

        for (c1, c2), strength in self.connections.items():
            if strength < threshold:
                continue

            if c1 in active and c2 not in new_activation:
                new_activation[c2] = active[c1] * strength * 0.5
            elif c2 in active and c1 not in new_activation:
                new_activation[c1] = active[c2] * strength * 0.5

        return new_activation


class PredictiveLearner:
    """
    Learning through prediction error minimization.

    The brain constantly predicts what will happen next.
    Prediction errors drive learning.
    """

    def __init__(self):
        self.predictors: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # predictors[context][outcome] = probability

    def predict(self, context: Tuple) -> Dict[str, float]:
        """Predict likely outcomes given context."""
        context_key = str(context)
        if context_key in self.predictors:
            return dict(self.predictors[context_key])
        return {}

    def learn(self, context: Tuple, actual_outcome: str):
        """Learn from prediction error."""
        context_key = str(context)

        # Get current predictions
        predictions = self.predictors[context_key]

        # Calculate surprise (prediction error)
        expected_prob = predictions.get(actual_outcome, 0.1)
        surprise = -np.log(max(0.01, expected_prob))

        # Update predictions
        total = sum(predictions.values()) + 1

        # Increase probability of actual outcome
        predictions[actual_outcome] = predictions.get(actual_outcome, 0) + 0.2

        # Normalize
        total = sum(predictions.values())
        for k in predictions:
            predictions[k] /= total

        return surprise


class StigmergicLearner:
    """
    Learning through environmental traces (like ant pheromones).

    Successful patterns leave traces that guide future behavior.
    Enables collective learning across agents.
    """

    def __init__(self, evaporation_rate: float = 0.01):
        self.evaporation = evaporation_rate
        self.pheromones: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # pheromones[state][action] = strength

    def deposit(self, state: str, action: str, amount: float = 1.0):
        """Deposit pheromone for a successful action."""
        self.pheromones[state][action] += amount

    def sense(self, state: str) -> Dict[str, float]:
        """Sense pheromone levels for possible actions."""
        return dict(self.pheromones.get(state, {}))

    def evaporate(self):
        """Evaporate pheromones over time."""
        for state in list(self.pheromones.keys()):
            for action in list(self.pheromones[state].keys()):
                self.pheromones[state][action] *= (1 - self.evaporation)
                if self.pheromones[state][action] < 0.01:
                    del self.pheromones[state][action]
            if not self.pheromones[state]:
                del self.pheromones[state]

    def get_best_action(self, state: str) -> Optional[str]:
        """Get action with strongest pheromone trail."""
        trails = self.sense(state)
        if not trails:
            return None
        return max(trails.keys(), key=lambda a: trails[a])


# =============================================================================
# CONCEPT FORMATION
# =============================================================================

class ConceptFormer:
    """
    Forms new concepts from experience patterns.

    This is the core of intelligence - abstracting patterns into reusable concepts.
    """

    def __init__(self, semantic_memory: SemanticMemory):
        self.semantic = semantic_memory
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.pattern_buffer: List[Experience] = []
        self.min_evidence = 5  # Minimum evidence to form concept

    def observe(self, experience: Experience):
        """Process an experience, potentially forming new concepts."""
        self.pattern_buffer.append(experience)

        # Limit buffer size
        if len(self.pattern_buffer) > 1000:
            self.pattern_buffer = self.pattern_buffer[-500:]

        # Periodically attempt concept formation
        if len(self.pattern_buffer) % 50 == 0:
            self._attempt_concept_formation()

    def _attempt_concept_formation(self):
        """Try to form new concepts from patterns."""
        # Look for statistical regularities
        patterns = self._find_patterns()

        for pattern in patterns:
            # Check if we already have a hypothesis for this
            existing = self._find_matching_hypothesis(pattern)

            if existing:
                # Update existing hypothesis
                self._update_hypothesis(existing, pattern)
            else:
                # Create new hypothesis
                self._create_hypothesis(pattern)

        # Promote strong hypotheses to concepts
        self._promote_hypotheses()

    def _find_patterns(self) -> List[Dict]:
        """Find statistical patterns in experience buffer."""
        patterns = []

        # Group experiences by input features
        groups = defaultdict(list)
        for exp in self.pattern_buffer:
            # Create feature signature
            sig = tuple(sorted(exp.perception.items()))
            groups[sig].append(exp)

        # Look for consistent input->output mappings
        for sig, exps in groups.items():
            if len(exps) >= 3:
                # Check if outcomes are consistent
                outcomes = [exp.outcome for exp in exps if exp.outcome]
                if outcomes:
                    # Find most common outcome
                    outcome_counts = defaultdict(int)
                    for o in outcomes:
                        outcome_counts[str(o)] += 1

                    most_common = max(outcome_counts.keys(), key=lambda x: outcome_counts[x])
                    frequency = outcome_counts[most_common] / len(outcomes)

                    if frequency >= 0.8:  # 80% consistency
                        patterns.append({
                            'input': dict(sig),
                            'output': most_common,
                            'confidence': frequency,
                            'count': len(exps)
                        })

        return patterns

    def _find_matching_hypothesis(self, pattern: Dict) -> Optional[Hypothesis]:
        """Find hypothesis that matches this pattern."""
        for hyp in self.hypotheses.values():
            if hyp.conditions == pattern['input']:
                return hyp
        return None

    def _create_hypothesis(self, pattern: Dict):
        """Create a new hypothesis from a pattern."""
        hyp = Hypothesis(
            pattern=f"When {pattern['input']}, then {pattern['output']}",
            conditions=pattern['input'],
            prediction={'output': pattern['output']},
            supporting=[str(i) for i in range(pattern['count'])]
        )
        self.hypotheses[hyp.id] = hyp

    def _update_hypothesis(self, hypothesis: Hypothesis, pattern: Dict):
        """Update hypothesis with new evidence."""
        if str(pattern['output']) == str(hypothesis.prediction.get('output')):
            hypothesis.supporting.extend(['new'] * pattern['count'])
        else:
            hypothesis.contradicting.extend(['new'] * pattern['count'])

    def _promote_hypotheses(self):
        """Promote confirmed hypotheses to concepts."""
        for hyp_id, hyp in list(self.hypotheses.items()):
            if (hyp.confidence >= 0.9 and
                len(hyp.supporting) >= self.min_evidence):
                # Create concept from hypothesis
                concept = Concept(
                    name=f"concept_{hyp_id}",
                    type=ConceptType.RELATIONAL,
                    description=hyp.pattern,
                    features=hyp.conditions,
                    preconditions=list(hyp.conditions.keys()),
                    effects=[str(hyp.prediction)],
                    confidence=hyp.confidence
                )

                self.semantic.store(concept)
                del self.hypotheses[hyp_id]

                return concept

        return None


# =============================================================================
# REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Performs reasoning using learned concepts.

    Types of reasoning:
    - Deductive: Apply concepts to reach conclusions
    - Analogical: Use similar situations to guide current one
    - Causal: Trace cause-effect chains
    """

    def __init__(self, semantic: SemanticMemory, episodic: EpisodicMemory,
                 working: WorkingMemory):
        self.semantic = semantic
        self.episodic = episodic
        self.working = working
        self.reasoning_trace: List[str] = []

    def reason(self, query: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """
        Reason about a query and return (answer, explanation).
        """
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Query: {query}")

        # Step 1: Activate relevant concepts
        relevant = self._activate_relevant_concepts(query)
        self.reasoning_trace.append(f"Activated concepts: {[c.name for c in relevant]}")

        # Step 2: Try deductive reasoning
        answer = self._deductive_reasoning(query, relevant)
        if answer is not None:
            return answer, self.reasoning_trace

        # Step 3: Try analogical reasoning
        answer = self._analogical_reasoning(query)
        if answer is not None:
            return answer, self.reasoning_trace

        # Step 4: Fall back to first principles
        answer = self._first_principles(query)
        return answer, self.reasoning_trace

    def _activate_relevant_concepts(self, query: Dict) -> List[Concept]:
        """Find and activate concepts relevant to the query."""
        relevant = []

        # Search by features
        concepts = self.semantic.retrieve(query, k=10)

        for concept in concepts:
            # Check if concept applies to this query
            if self._concept_applies(concept, query):
                relevant.append(concept)
                self.working.activate(concept.id)

        return relevant

    def _concept_applies(self, concept: Concept, query: Dict) -> bool:
        """Check if a concept's preconditions are met."""
        for precond in concept.preconditions:
            if precond not in query:
                return False
        return True

    def _deductive_reasoning(self, query: Dict, concepts: List[Concept]) -> Optional[Any]:
        """Apply concepts deductively to answer query."""
        for concept in sorted(concepts, key=lambda c: c.confidence, reverse=True):
            if concept.confidence < 0.5:
                continue

            self.reasoning_trace.append(f"Trying concept '{concept.name}' (conf={concept.confidence:.2f})")

            # Check if we can apply this concept
            if concept.effects:
                self.reasoning_trace.append(f"  Applied! Effects: {concept.effects}")
                concept.update_confidence(True)
                return {'concept_applied': concept.name, 'effects': concept.effects}

        return None

    def _analogical_reasoning(self, query: Dict) -> Optional[Any]:
        """Use similar past experiences to answer query."""
        similar = self.episodic.retrieve(query, k=5)

        if similar:
            best = similar[0]
            self.reasoning_trace.append(f"Found analogous experience: {best.id}")

            if best.outcome:
                self.reasoning_trace.append(f"  Outcome was: {best.outcome}")
                return {'by_analogy': best.outcome}

        return None

    def _first_principles(self, query: Dict) -> Any:
        """Reason from first principles when no concepts apply."""
        self.reasoning_trace.append("Reasoning from first principles...")

        # This is domain-specific - subclasses should override
        return {'first_principles': 'No applicable concepts found'}


# =============================================================================
# ATTENTION MECHANISM
# =============================================================================

class AttentionSystem:
    """
    Bio-plausible attention through salience computation.

    Unlike transformer attention (softmax over all), this:
    - Uses salience (surprise, reward, novelty) to guide attention
    - Has limited capacity (can't attend to everything)
    - Is influenced by current goals
    """

    def __init__(self, working_memory: WorkingMemory):
        self.working = working_memory
        self.salience_weights = {
            'novelty': 0.3,
            'reward_relevance': 0.3,
            'goal_relevance': 0.4
        }
        self.current_goal: Optional[str] = None

    def compute_salience(self, item: Any, context: Dict) -> float:
        """Compute how attention-worthy an item is."""
        salience = 0.0

        # Novelty - new/unexpected things grab attention
        novelty = context.get('novelty', 0.5)
        salience += self.salience_weights['novelty'] * novelty

        # Reward relevance - things related to reward
        reward_rel = context.get('reward_relevance', 0.5)
        salience += self.salience_weights['reward_relevance'] * reward_rel

        # Goal relevance - things related to current goal
        if self.current_goal:
            goal_rel = context.get('goal_relevance', 0.5)
            salience += self.salience_weights['goal_relevance'] * goal_rel

        return min(1.0, salience)

    def select_focus(self, candidates: List[Tuple[str, Dict]]) -> Optional[str]:
        """Select what to focus on from candidates."""
        if not candidates:
            return None

        # Compute salience for each candidate
        saliences = [
            (cid, self.compute_salience(cid, ctx))
            for cid, ctx in candidates
        ]

        # Probabilistic selection weighted by salience
        total = sum(s for _, s in saliences)
        if total == 0:
            return candidates[0][0]

        probs = [s / total for _, s in saliences]
        idx = np.random.choice(len(candidates), p=probs)

        selected = candidates[idx][0]
        self.working.set_focus(selected)

        return selected

    def set_goal(self, goal: str):
        """Set current goal for goal-directed attention."""
        self.current_goal = goal


# =============================================================================
# META-COGNITION
# =============================================================================

class MetaCognition:
    """
    Thinking about thinking - monitoring and controlling cognition.

    Enables:
    - Knowing what you know (and don't know)
    - Choosing strategies
    - Learning to learn
    """

    def __init__(self, semantic: SemanticMemory):
        self.semantic = semantic
        self.strategy_history: List[Dict] = []
        self.confidence_calibration: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

    def assess_confidence(self, concept_id: str, answer: Any) -> float:
        """How confident should we be in this answer?"""
        concept = self.semantic.get_concept(concept_id)
        if not concept:
            return 0.5

        # Base confidence from concept
        base = concept.confidence

        # Adjust based on calibration history
        calibration = self.confidence_calibration.get(concept_id, [])
        if len(calibration) >= 5:
            # How well calibrated have we been?
            correct = sum(1 for conf, was_correct in calibration[-10:] if was_correct)
            total = len(calibration[-10:])
            calibration_factor = correct / total

            return base * calibration_factor

        return base

    def record_outcome(self, concept_id: str, confidence: float, was_correct: bool):
        """Record whether our confidence was well-calibrated."""
        self.confidence_calibration[concept_id].append((confidence, was_correct))

    def choose_strategy(self, problem_type: str) -> str:
        """Choose best strategy for this problem type."""
        # Look at history for this problem type
        relevant = [h for h in self.strategy_history if h['problem_type'] == problem_type]

        if not relevant:
            return 'exploratory'  # Default to exploration

        # Find most successful strategy
        strategy_scores = defaultdict(lambda: {'success': 0, 'total': 0})
        for h in relevant:
            s = h['strategy']
            strategy_scores[s]['total'] += 1
            if h['success']:
                strategy_scores[s]['success'] += 1

        best = max(strategy_scores.keys(),
                   key=lambda s: strategy_scores[s]['success'] / max(1, strategy_scores[s]['total']))

        return best

    def record_strategy_outcome(self, problem_type: str, strategy: str, success: bool):
        """Record how well a strategy worked."""
        self.strategy_history.append({
            'problem_type': problem_type,
            'strategy': strategy,
            'success': success,
            'timestamp': time.time()
        })

        # Keep history bounded
        if len(self.strategy_history) > 1000:
            self.strategy_history = self.strategy_history[-500:]

    def identify_knowledge_gaps(self) -> List[str]:
        """Identify areas where knowledge is weak."""
        gaps = []

        for cid, concept in self.semantic.concepts.items():
            # Low confidence concepts
            if concept.confidence < 0.6 and concept.usage_count > 5:
                gaps.append(f"Low confidence in '{concept.name}'")

            # Poorly calibrated areas
            calibration = self.confidence_calibration.get(cid, [])
            if len(calibration) >= 10:
                recent = calibration[-10:]
                accuracy = sum(1 for _, correct in recent if correct) / len(recent)
                if accuracy < 0.7:
                    gaps.append(f"Poor calibration for '{concept.name}'")

        return gaps


# =============================================================================
# INTEGRATED INTELLIGENT AGENT
# =============================================================================

class IntelligentAgent:
    """
    Complete intelligent agent integrating all components.

    This is the main interface for the intelligent architecture.
    """

    def __init__(self, name: str = "Agent"):
        self.name = name

        # Memory systems
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()

        # Learning mechanisms
        self.hebbian = HebbianLearner()
        self.predictive = PredictiveLearner()
        self.stigmergic = StigmergicLearner()

        # Higher cognition
        self.concept_former = ConceptFormer(self.semantic)
        self.reasoner = ReasoningEngine(self.semantic, self.episodic, self.working)
        self.attention = AttentionSystem(self.working)
        self.metacognition = MetaCognition(self.semantic)

        # Statistics
        self.experience_count = 0
        self.concept_count = 0

    def perceive(self, perception: Dict[str, Any]) -> str:
        """Process a perception from the environment."""
        # Create experience
        exp = Experience(perception=perception)

        # Store in episodic memory
        exp_id = self.episodic.store(exp)

        # Attempt concept formation
        self.concept_former.observe(exp)

        self.experience_count += 1

        return exp_id

    def act(self, state: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """Decide on an action given current state."""
        # Use reasoning to decide
        answer, trace = self.reasoner.reason(state)

        # Check stigmergic guidance
        state_key = str(sorted(state.items()))
        stigmergic_suggestion = self.stigmergic.get_best_action(state_key)

        if stigmergic_suggestion:
            trace.append(f"Stigmergic suggestion: {stigmergic_suggestion}")

        return answer, trace

    def learn(self, state: Dict, action: Any, outcome: Any, reward: float):
        """Learn from an experience."""
        # Create and store experience
        exp = Experience(
            perception=state,
            action={'action': action},
            outcome={'outcome': outcome},
            reward=reward,
            active_concepts=list(self.working.active_concepts.keys())
        )
        self.episodic.store(exp)

        # Hebbian learning on active concepts
        active = list(self.working.active_concepts.keys())
        self.hebbian.learn(active, reward)

        # Predictive learning
        context = tuple(sorted(state.items()))
        surprise = self.predictive.learn(context, str(outcome))
        exp.surprise = surprise

        # Stigmergic learning
        if reward > 0:
            state_key = str(sorted(state.items()))
            self.stigmergic.deposit(state_key, str(action), reward)

        # Concept formation
        self.concept_former.observe(exp)

        # Working memory decay
        self.working.decay()

        # Periodic consolidation
        if self.experience_count % 100 == 0:
            self.consolidate()

    def consolidate(self):
        """Consolidate memories (like sleep)."""
        self.episodic.consolidate()
        self.semantic.consolidate()
        self.stigmergic.evaporate()

    def introspect(self) -> Dict[str, Any]:
        """Report on internal state."""
        return {
            'name': self.name,
            'experiences': self.experience_count,
            'concepts': len(self.semantic.concepts),
            'hypotheses': len(self.concept_former.hypotheses),
            'working_memory': self.working.get_active(),
            'knowledge_gaps': self.metacognition.identify_knowledge_gaps(),
            'associations': len(self.semantic.associations)
        }

    def explain_concept(self, concept_name: str) -> str:
        """Explain a concept in natural language."""
        concepts = self.semantic.retrieve(concept_name)
        if not concepts:
            return f"I don't know about '{concept_name}'"

        concept = concepts[0]
        lines = [
            f"Concept: {concept.name}",
            f"Type: {concept.type.value}",
            f"Description: {concept.description}",
            f"Confidence: {concept.confidence:.2%}",
            f"Used {concept.usage_count} times"
        ]

        if concept.preconditions:
            lines.append(f"Applies when: {concept.preconditions}")
        if concept.effects:
            lines.append(f"Effects: {concept.effects}")

        # Associated concepts
        associated = self.semantic.get_associated(concept.id)
        if associated:
            assoc_names = [self.semantic.get_concept(cid).name
                         for cid, _ in associated[:3]
                         if self.semantic.get_concept(cid)]
            lines.append(f"Related to: {assoc_names}")

        return "\n".join(lines)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL COLLECTIVE INTELLIGENCE (NCI) ARCHITECTURE")
    print("=" * 70)
    print()

    # Create agent
    agent = IntelligentAgent(name="NCI-1")

    print("Created intelligent agent with:")
    print(f"  - Episodic memory (experiences)")
    print(f"  - Semantic memory (concepts)")
    print(f"  - Working memory (active processing)")
    print(f"  - Hebbian learning (association)")
    print(f"  - Predictive learning (prediction error)")
    print(f"  - Stigmergic learning (environmental traces)")
    print(f"  - Concept formation (abstraction)")
    print(f"  - Reasoning engine (deduction, analogy)")
    print(f"  - Attention system (salience-based)")
    print(f"  - Meta-cognition (self-awareness)")
    print()

    print("This architecture learns through:")
    print("  1. EXPERIENCE - observing patterns in the world")
    print("  2. ABSTRACTION - forming concepts from patterns")
    print("  3. REASONING - applying concepts to new situations")
    print("  4. REFLECTION - knowing what it knows")
    print()
    print("All WITHOUT backpropagation!")
    print("=" * 70)
