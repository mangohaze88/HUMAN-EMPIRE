# Advanced Stigmergic Intelligence: Technical Documentation

## Mission: 1000x Performance Improvement

**Current Performance**: 0.074 error with three-factor learning
**Target**: <0.01 error
**Required Improvement**: ~7.4x minimum, aiming for 1000x overall capability

---

## Implemented Advanced Mechanisms

### 1. Ant Colony Optimization (ACO) for Credit Assignment

**Biological Inspiration**: Real ant colonies use pheromone trails to find optimal paths. Successful paths are reinforced, failed paths evaporate.

**Implementation**:
- **Path Quality Tracking**: Each pheromone deposit has an associated quality metric (1.0 + reward)
- **Quality-Based Evaporation**: High-quality paths evaporate slower than low-quality paths
  - `evap_rate = base_rate × (2.0 - quality_normalized)`
  - Successful patterns persist longer in the environment
- **ACO Deposit Formula**: `deposit_amount = amount × path_quality`
- **Adaptive Learning**: The environment "learns" which patterns lead to success

**Why It Helps**:
- Better credit assignment: successful patterns are reinforced automatically
- Failed explorations disappear faster, reducing noise
- Emergent optimization: the best solutions naturally dominate the field

**Key Code**:
```python
# ACO-style evaporation based on path quality
quality_normalized = torch.sigmoid(self.path_quality[ch] - 1.0)
evap_rate = self.base_evaporation * (2.0 - quality_normalized)
self.high_res[ch] *= (1 - evap_rate)
```

---

### 2. Multi-Colony Competition

**Biological Inspiration**: Different ant colonies compete for resources. The most successful colonies expand, unsuccessful ones shrink.

**Implementation**:
- **4 Colonies** with different specializations:
  - **Explorer**: High exploration rate, fast movement
  - **Exploiter**: Low exploration, high learning rate, stays near successful regions
  - **Coordinator**: Balanced, wider attention span
  - **Generalist**: Default balanced parameters

- **Colony-Specific Channels**: Each colony has its own pheromone channel for communication
- **Performance Tracking**: Each colony's reward is tracked separately
- **Solution Propagation**: Best colony's solutions (genomes) migrate to other colonies
  - Every 100 steps, best colony sends "immigrants" to replace worst 10% of other colonies

**Why It Helps**:
- **Specialization**: Different strategies explore different parts of the solution space
- **Diversity**: Multiple approaches prevent premature convergence
- **Best-of-Breed**: Successful strategies automatically propagate
- **Robust**: If one colony fails, others continue

**Key Code**:
```python
# Multi-colony competition: share best solutions
best_colony_idx = np.argmax(colony_rewards)
best_genome = self.colonies[best_colony_idx].best_genome

# Propagate to other colonies (immigration)
for colony in other_colonies:
    replace_worst_10_percent_with(best_genome.mutate(0.15))
```

---

### 3. Stigmergic Attention

**Biological Inspiration**: Animals attend to salient stimuli. Attention focuses processing on important information.

**Implementation**:
- **Dynamic Attention Weights**: Each agent computes attention over sensory channels
  - `attention = softmax(|sensory| / (magnitude + ε))`
  - High-variance, high-magnitude channels get more attention

- **Genome-Controlled Width**: Each agent has an evolved `attention_width` parameter
  - Wide attention: sees broader patterns
  - Narrow attention: focuses on specific details

- **Attended Sensory Processing**:
  - `attended_sensory = sensory × attention × genome.attention_width`
  - Amplifies important signals, suppresses noise

**Why It Helps**:
- **Computational Efficiency**: Focus processing on important channels
- **Noise Reduction**: Ignore irrelevant pheromone variations
- **Adaptive**: Attention evolves to match task requirements
- **Sparse Activation**: Better use of neural capacity

**Key Code**:
```python
def compute_attention(self, sensory: torch.Tensor) -> torch.Tensor:
    variance = torch.var(sensory)
    magnitude = torch.mean(torch.abs(sensory))
    attention = torch.softmax(torch.abs(sensory) / (magnitude + 1e-8), dim=0)
    attended = sensory * attention * self.genome.attention_width
    return attended
```

---

### 4. Hierarchical Pheromone Fields (Multi-Scale Processing)

**Biological Inspiration**: Visual cortex processes images at multiple scales simultaneously. V1 detects edges, V2 detects shapes, V4 detects objects.

**Implementation**:
- **Three Resolution Levels**:
  - **High Resolution**: Full 128×128 grid (fine details)
  - **Medium Resolution**: 64×64 grid (intermediate patterns)
  - **Low Resolution**: 32×32 grid (global context)

- **Simultaneous Reading**: Agents read all three scales and concatenate:
  - `sensory = [high_res, med_res, low_res]`
  - Total features = n_channels × 3

- **Automatic Downsampling**: Average pooling maintains spatial relationships
  - `med_res = avg_pool2d(high_res, kernel_size=2)`
  - `low_res = avg_pool2d(high_res, kernel_size=4)`

**Why It Helps**:
- **Multi-Scale Features**: Capture both fine details and global patterns
- **Better Generalization**: Coarse features are more robust
- **Hierarchical Processing**: Like deep neural networks, but stigmergic
- **Receptive Fields**: Each agent effectively has large receptive field

**Key Code**:
```python
# Multi-resolution fields
self.high_res = torch.zeros(n_channels, 128, 128)
self.med_res = torch.zeros(n_channels, 64, 64)
self.low_res = torch.zeros(n_channels, 32, 32)

# Update via average pooling
self.med_res = F.avg_pool2d(self.high_res, kernel_size=2, stride=2)
self.low_res = F.avg_pool2d(self.high_res, kernel_size=4, stride=4)
```

---

### 5. Temporal Pheromone Memory (Hippocampal Replay)

**Biological Inspiration**: Hippocampus stores episodic memories and "replays" successful experiences during rest. This consolidates learning.

**Implementation**:
- **Memory Buffer**: Stores up to 200 successful pheromone patterns
  - Each pattern is a full snapshot of all pheromone channels
  - Stored with associated reward value

- **Priority Storage**: Only stores patterns that improve best-error-ever
  - Worst patterns are replaced when buffer is full

- **Weighted Replay**:
  - Sample patterns proportional to their reward
  - Better patterns are replayed more often

- **Blend with Current State**:
  - 10% chance of replay every step
  - `new_state = 0.75 × current + 0.25 × replayed_pattern`
  - Gentle guidance, not override

**Why It Helps**:
- **Memory Consolidation**: Successful patterns are reinforced over time
- **Avoid Catastrophic Forgetting**: Past successes are not lost
- **Pattern Completion**: Partial patterns can be completed from memory
- **Implicit Regularization**: Learned patterns act as priors

**Key Code**:
```python
# Store successful pattern
if final_error < self.best_error:
    pattern_snapshot = self.pheromone_field.high_res.clone()
    self.temporal_memory.store(pattern_snapshot, reward=1.0/(error + 1e-8))

# Replay with probability 0.1
if np.random.random() < 0.1:
    replayed = self.temporal_memory.replay(n_samples=3)
    for pattern in replayed:
        self.pheromone_field.high_res = 0.75 * current + 0.25 * pattern
```

---

### 6. Evolutionary Agent Optimization

**Biological Inspiration**: Natural selection optimizes organisms over generations. Successful individuals reproduce, unsuccessful ones die.

**Implementation**:
- **Agent Genome**: 6 evolvable hyperparameters per agent
  - `learning_rate`: How fast weights update (0.001 - 0.1)
  - `exploration_rate`: How much random movement (0.0 - 1.0)
  - `trace_decay`: Eligibility trace decay rate (0.5 - 0.99)
  - `movement_speed`: How fast agent moves (0.001 - 0.1)
  - `attention_width`: Attention modulation strength (1.0 - 10.0)
  - `deposit_strength`: Pheromone deposit scaling (0.1 - 5.0)

- **Fitness Tracking**: Each agent accumulates fitness = (1 - error) × reward

- **Evolution Every 50 Steps**:
  1. **Selection**: Keep top 20% performers
  2. **Crossover**: Offspring = average of two random top performers
  3. **Mutation**: Add Gaussian noise to each gene (σ = 0.1)
  4. **Population**: Maintain constant colony size

- **Best Genome Tracking**: Each colony tracks its best-performing genome

**Why It Helps**:
- **Automatic Hyperparameter Tuning**: Each agent finds its optimal parameters
- **Adaptation**: Parameters evolve to match task requirements
- **Diversity**: Different genomes explore different strategies
- **No Manual Tuning**: Self-optimizing system

**Key Code**:
```python
def evolve(self, top_k: int = 10):
    # Select top performers
    _, top_indices = torch.topk(self.fitness, top_k)

    # Create new generation
    for i in range(self.size):
        if i < top_k:
            keep_elite()
        else:
            parent1, parent2 = random_sample(top_indices)
            child_genome = crossover(parent1, parent2).mutate(0.1)
            child_weights = (weights1 + weights2) / 2 + noise
```

---

### 7. Direct Spatial Frequency Input Encoding

**Biological Inspiration**: Retinal ganglion cells and early visual cortex encode images using spatial frequencies (edges, textures). Gabor filters model V1 receptive fields.

**Implementation**:
- **Gabor-Like Filters**: Multiple frequency and orientation filters
  - Frequencies: [1, 2, 4, 8] cycles per image
  - Orientations: [0°, 45°, 90°, 135°]
  - Total: 16 spatial frequency encoders

- **Filter Formula**: Gabor filter = Gaussian × Cosine wave
  ```
  gabor = exp(-(x² + y²) / 2) × cos(freq × (x·cos(θ) + y·sin(θ)))
  ```

- **Input Injection**:
  1. Reshape input to 2D spatial pattern (e.g., 8×8)
  2. Convolve with each Gabor filter
  3. Deposit encoded patterns at environment center
  4. Different frequencies in different pheromone subregions

**Why It Helps**:
- **Rich Encoding**: Captures edges, textures, and structures
- **Distributed Representation**: Input is spread across pheromone field
- **Biological Plausibility**: How real visual systems encode images
- **Better Readout**: Output network can extract multi-scale features

**Key Code**:
```python
# Create Gabor filters
for freq in [1, 2, 4, 8]:
    for theta in [0, π/4, π/2, 3π/4]:
        gabor = exp(-(x² + y²) / 2) × cos(freq × (x·cos(θ) + y·sin(θ)))
        encoders.append(gabor)

# Encode input
for encoder in encoders:
    encoded = input_pattern × encoder
    deposit_at_center(encoded)
```

---

## System Integration

### How All Mechanisms Work Together

1. **Input Encoding** (Mechanism 7):
   - Input is encoded as spatial frequencies
   - Deposited across multiple pheromone channels at environment center

2. **Multi-Scale Reading** (Mechanism 4):
   - Agents read pheromone field at 3 resolutions simultaneously
   - Get both local details and global context

3. **Attention** (Mechanism 3):
   - Each agent applies evolved attention to focus on important channels
   - Reduces noise, improves signal

4. **Multi-Colony Processing** (Mechanism 2):
   - 4 specialized colonies process information in parallel
   - Explorers find new patterns, exploiters refine them, coordinators integrate

5. **ACO Credit Assignment** (Mechanism 1):
   - Successful agent actions reinforce pheromone trails
   - Failed actions evaporate faster
   - Best paths automatically dominate

6. **Temporal Memory** (Mechanism 5):
   - Successful pheromone patterns stored in memory
   - Periodically replayed to guide learning
   - Prevents forgetting, accelerates convergence

7. **Evolution** (Mechanism 6):
   - Agent hyperparameters evolve every 50 steps
   - Best strategies reproduce and spread
   - System self-optimizes over time

8. **Competition & Migration**:
   - Best colony's solutions propagate to other colonies every 100 steps
   - Maintains diversity while sharing successful strategies

---

## Expected Performance Gains

| Mechanism | Expected Improvement | Justification |
|-----------|---------------------|---------------|
| ACO Credit Assignment | 1.5-2x | Better reinforcement of successful patterns |
| Multi-Colony Competition | 1.3-1.5x | Parallel exploration + diversity |
| Stigmergic Attention | 1.2-1.4x | Noise reduction + focus |
| Hierarchical Fields | 1.5-2x | Multi-scale features (like deep CNNs) |
| Temporal Memory | 1.3-1.6x | Consolidation + replay |
| Evolutionary Optimization | 1.5-2x | Automatic hyperparameter tuning |
| Spatial Frequency Encoding | 1.3-1.5x | Rich, distributed input representation |
| **Combined (multiplicative)** | **~7-15x** | Synergistic effects |

**Target**: 0.074 → <0.01 = ~7.4x improvement minimum
**Conservative Estimate**: 7-15x = achievable
**Optimistic Estimate**: 20-50x with tuning

---

## Hyperparameter Summary

### Network Configuration
- `n_colonies`: 4 (explorer, exploiter, coordinator, generalist)
- `agents_per_colony`: 256 (total 1024 agents)
- `env_shape`: (128, 128) (high resolution for fine patterns)
- `feature_dim`: 32 (agent internal state dimension)
- `n_pheromones`: 12 channels

### Evolution Parameters
- `evolution_interval`: 50 steps (evolve colonies)
- `migration_interval`: 100 steps (best colony → others)
- `top_k_selection`: 20% (elite retention)
- `mutation_rate`: 0.1 (genome mutation strength)

### ACO Parameters
- `base_evaporation`: 0.05 (moderate decay)
- `quality_modulation`: [0, 2] (adaptive evaporation)
- `alpha`: 1.0 (pheromone importance)
- `beta`: 2.0 (heuristic importance)

### Memory Parameters
- `memory_capacity`: 200 patterns
- `replay_probability`: 0.1 (10% of steps)
- `replay_blend`: 0.25 (25% replayed, 75% current)
- `storage_threshold`: best_error improvement

### Spatial Frequency Encoding
- `frequencies`: [1, 2, 4, 8] cycles
- `orientations`: [0°, 45°, 90°, 135°]
- `filter_type`: Gabor (Gaussian × Cosine)

---

## Comparison with Baseline

| Feature | Baseline (Three-Factor) | Advanced (7 Mechanisms) |
|---------|------------------------|------------------------|
| **Agents** | 1024 uniform | 1024 in 4 specialized colonies |
| **Pheromone Fields** | Single resolution | 3 resolutions (hierarchical) |
| **Credit Assignment** | Simple reward signal | ACO path quality + evaporation |
| **Learning** | Fixed hyperparameters | Evolved per-agent hyperparameters |
| **Attention** | None | Stigmergic attention per agent |
| **Memory** | None | Temporal replay of 200 patterns |
| **Input Encoding** | Raw spatial | Spatial frequency (Gabor filters) |
| **Output Network** | 2-layer MLP | 3-layer MLP with dropout |
| **Optimization** | Manual hyperparams | Self-optimizing via evolution |

---

## Testing Protocol

### Test 1: Single Input Learning
- Train on fixed random input for 1000 steps
- Measure: initial error, final error, improvement %
- Success: final error < 0.01

### Test 2: Generalization
- Train on 5 different input patterns (200 steps each)
- Measure: average final error across all inputs
- Success: avg error < 0.02

### Test 3: Convergence Speed
- Measure steps to reach error < 0.05
- Compare baseline vs advanced
- Success: >2x speedup

### Test 4: Memory & Evolution
- Track temporal memory size over time
- Track best genome performance
- Verify: memory grows, genomes improve

---

## Future Enhancements

### Potential Further Improvements (for >100x total)

1. **Meta-Learning Colony Dynamics**
   - Learn when to trigger evolution
   - Learn migration rates between colonies

2. **Adaptive Field Resolution**
   - Dynamically change grid resolution based on task
   - High-res for fine details, low-res for coarse patterns

3. **Hierarchical Agent Organization**
   - "Supervisor" agents coordinate groups of "worker" agents
   - Multi-level stigmergic communication

4. **Predictive Memory**
   - Predict future pheromone states
   - Pre-emptive pattern activation

5. **Neuromodulation**
   - Global signals modulate all agents simultaneously
   - Implement dopamine/serotonin-like broadcasts

6. **Morphological Computation**
   - Agent "bodies" with different shapes/sizes
   - Physical constraints as computational benefits

---

## Usage Example

```python
from networks.stigmergic_intelligence_advanced import AdvancedStigmergicNetwork

# Create network
net = AdvancedStigmergicNetwork(
    n_colonies=4,
    agents_per_colony=256,
    env_shape=(128, 128),
    feature_dim=32,
    input_dim=64,
    output_dim=32,
    device='cuda'
)

# Training loop
x = torch.randn(64, device='cuda')
for i in range(1000):
    output, info = net.forward(x, n_steps=10, learn=True)

    if i % 100 == 0:
        print(f"Step {i}: error={info['task_error']:.6f}, "
              f"best_ever={info['best_error_ever']:.6f}, "
              f"memory={info['memory_size']}")

print(f"Final error: {info['best_error_ever']:.6f}")
```

---

## References

### ACO & Swarm Intelligence
- Dorigo & Stützle (2004): "Ant Colony Optimization"
- Bonabeau et al. (1999): "Swarm Intelligence: From Natural to Artificial Systems"

### Stigmergy
- Theraulaz & Bonabeau (1999): "A Brief History of Stigmergy"
- Marsh & Onof (2008): "Stigmergic epistemology, stigmergic cognition"

### Hierarchical Processing
- Marr (1982): "Vision" - Multi-scale visual processing
- Hubel & Wiesel: Receptive fields in visual cortex

### Hippocampal Replay
- Wilson & McNaughton (1994): "Reactivation of hippocampal ensemble memories during sleep"
- Foster & Wilson (2006): "Reverse replay of behavioural sequences in hippocampal place cells during the awake state"

### Evolutionary Algorithms
- De Jong (2006): "Evolutionary Computation: A Unified Approach"
- Eiben & Smith (2015): "Introduction to Evolutionary Computing"

### Spatial Frequency Encoding
- Daugman (1985): "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters"
- Jones & Palmer (1987): "An evaluation of the two-dimensional Gabor filter model of simple receptive fields in cat striate cortex"

---

## License

MIT License - Free to use, modify, and distribute.

## Contact

For questions or contributions, see main repository README.
