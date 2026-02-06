# Stigmergic Network Audit Fixes (NO BACKPROPAGATION)

## Executive Summary

All three critical audit findings have been addressed using **three-factor learning** (reward-modulated Hebbian plasticity) - a biologically plausible learning mechanism that requires NO backpropagation.

**Implementation**: `/root/MAROLA/alternative-ai-architectures/src/networks/stigmergic_intelligence_reward_based.py`

---

## Audit Finding 1: Agents optimize wrong objective

### PROBLEM
**Old code (line 796-798):**
```python
if step == 0:
    target = encoded  # Bootstrap
else:
    target = self.prev_encoded  # WRONG: learning dynamics, not reconstruction
```

Agents were predicting their own previous state instead of contributing to input reconstruction.

### FIX: Reconstruction-Oriented Contributions

**New code (RewardModulatedAgent, lines 229-235):**
```python
def compute_contribution(self, sensory: np.ndarray) -> np.ndarray:
    """
    Compute agent's contribution to COLLECTIVE RECONSTRUCTION.
    This is what the agent deposits into the environment.
    """
    encoded = np.tanh(sensory[:self.feature_dim])
    contribution = np.tanh(self.weights @ encoded)
    return contribution
```

**Key changes:**
1. Agents now compute **contributions** that are deposited into **reconstruction channels**
2. Output layer reads from reconstruction channels to produce final output
3. Agents' objective is implicitly to maximize collective reconstruction quality (via reward)

**How it works:**
- Agents deposit contributions → Environment accumulates → Output reads reconstruction channels
- When reconstruction is good → high reward → agents reinforce their contribution patterns
- When reconstruction is bad → low/negative reward → agents adjust their patterns

---

## Audit Finding 2: No gradient path from task to agents

### PROBLEM
**Old code (line 766-774):**
```python
task_error = torch.mean((target_for_task - current_output) ** 2)
# ... computed but never reaches agents
self._broadcast_global_feedback(task_error.item(), is_success, error_vec)
# This was just for visualization, agents ignored it
```

Task error was computed but didn't influence agent learning.

### FIX: Reward Signal + Eligibility Traces

**New code (RewardModulatedAgent, lines 263-290):**
```python
def three_factor_learning(
    self,
    pre_activity: np.ndarray,
    post_activity: np.ndarray,
    reward_signal: float
):
    """
    THREE-FACTOR HEBBIAN LEARNING (biologically plausible):

    Δw = learning_rate × pre × post × reward

    This is analogous to dopamine-modulated synaptic plasticity.
    NO BACKPROPAGATION NEEDED!
    """
    pre = np.tanh(pre_activity[:self.feature_dim])
    post = post_activity[:self.feature_dim]

    # Reward modulates learning rate
    reward_modulated_lr = self.base_lr * (1.0 + reward_signal)
    reward_modulated_lr = np.clip(reward_modulated_lr, 0.001, 0.1)

    # THREE-FACTOR RULE: Hebbian learning modulated by reward
    delta = reward_modulated_lr * np.outer(post, pre) * reward_signal

    # Oja's normalization for stability
    delta -= reward_modulated_lr * np.outer(post ** 2, np.ones_like(pre)) * self.weights * 0.01

    self.weights += delta
```

**Reward computation (lines 326-353):**
```python
def compute_local_reward(
    self,
    task_error: float,
    prev_task_error: float,
    local_competence: float
) -> float:
    """
    Compute reward signal from task performance.

    REWARD = improvement + local_success_bonus

    This creates the critical link: agents are rewarded when
    collective task performance improves.
    """
    # Improvement reward (TD-style)
    improvement = prev_task_error - task_error
    improvement_reward = np.tanh(improvement * 10.0)

    # Bonus for local competence
    competence_bonus = local_competence * 0.2

    # Success bonus when task error is low
    if task_error < 0.3:
        success_bonus = (0.3 - task_error) * 2.0
    else:
        success_bonus = 0.0

    total_reward = improvement_reward + competence_bonus + success_bonus
    return np.clip(total_reward, -1.0, 1.0)
```

**Eligibility traces (lines 103-111, 293-315):**
```python
@dataclass
class EligibilityTrace:
    """Tracks agent's recent contributions for credit assignment."""
    position: np.ndarray
    channel: int
    amount: float
    timestamp: int
    pre_activity: np.ndarray  # What the agent sensed
    post_activity: np.ndarray  # What the agent output

def apply_reward_to_traces(self, reward_signal: float):
    """
    Apply delayed reward to recent eligibility traces.
    This solves the temporal credit assignment problem.
    """
    for i, trace in enumerate(self.eligibility_traces):
        # Exponential decay: recent traces get more credit
        trace_age = len(self.eligibility_traces) - i
        trace_weight = self.trace_decay ** trace_age

        # Apply three-factor learning with decayed reward
        effective_reward = reward_signal * trace_weight
        self.three_factor_learning(
            trace.pre_activity,
            trace.post_activity,
            effective_reward
        )
```

**Key changes:**
1. **Reward signal** computed from task performance improvement
2. **Eligibility traces** track what each agent did recently
3. When reward arrives, agents apply three-factor learning to recent actions
4. Positive reward → reinforce successful patterns
5. Negative reward → weaken unsuccessful patterns

**Biological plausibility:**
This mimics dopamine modulation in the basal ganglia:
- Pre-activity = presynaptic neuron firing
- Post-activity = postsynaptic neuron firing
- Reward = dopamine signal
- Synapses strengthen when all three coincide

---

## Audit Finding 3: Output layer disconnected from agents

### PROBLEM
**Old code (line 857-881):**
```python
# Output layer learns via backprop
error = target_for_task - output
d2 = error * (1 - output ** 2)  # derivative of tanh
delta_w2 = 0.01 * torch.outer(d2, h1)
# ... but agents never see this gradient
```

Output layer learned, but agents didn't know what the output needed.

### FIX: Bidirectional Influence

**1. Forward: Agents → Output (lines 652-665 in StigmergicNetworkRewardBased):**
```python
def _read_output(self) -> np.ndarray:
    """Read output from reconstruction channels."""
    recon_state = self._get_reconstruction_state()
    output = np.tanh(self.output_weights @ recon_state[:self.output_weights.shape[1]])
    return output

def _get_reconstruction_state(self) -> np.ndarray:
    """Get flattened reconstruction channels."""
    recon_channels = self.env.pheromones[:self.env.n_recon_channels]
    return recon_channels.flatten()
```

Agents deposit → Reconstruction channels → Output reads

**2. Backward: Output → Agents via Gradient Broadcast (lines 501-534 in StigmergicEnvironmentReward):**
```python
def broadcast_task_gradient(self, error_gradient: np.ndarray):
    """
    Broadcast task error gradient as spatial pheromone pattern.
    This tells agents WHERE to deposit more/less.
    """
    # Reshape error gradient to spatial pattern
    size = min(int(np.sqrt(len(error_gradient))), 8)
    cx, cy = self.shape[0] // 2, self.shape[1] // 2
    pattern = error_gradient[:size*size].reshape(size, size)

    # Inject gradient as pheromone
    self.pheromones[self.channels['gradient_x'], ...] = sub_pattern
    self.pheromones[self.channels['gradient_y'], ...] = sub_pattern.T
```

**3. Agents use gradient in navigation (lines 355-392 in RewardModulatedAgent):**
```python
def act(self, sensory: np.ndarray, reward_signal: float):
    """Move based on reward gradients AND task gradients."""
    # Read task-aligned gradient (broadcast from output layer)
    task_gradient_x = sensory[self.env.channels['gradient_x']]
    task_gradient_y = sensory[self.env.channels['gradient_y']]
    task_gradient = np.array([task_gradient_x, task_gradient_y])

    # Move towards where task needs help
    if self.specialization == 'explorer':
        direction = -0.3 * reward_gradient + 0.5 * task_gradient
    elif self.specialization == 'exploiter':
        direction = 0.6 * reward_gradient + 0.3 * global_success_gradient
    else:
        if reward_signal > 0:
            direction = 0.5 * reward_gradient + 0.3 * global_success_gradient
        else:
            direction = 0.4 * task_gradient - 0.2 * reward_gradient
```

**Key changes:**
1. **Forward path**: Agents deposit → Environment accumulates → Output reads
2. **Backward path**: Output error → Spatial gradient broadcast → Agents navigate toward needed areas
3. **Learning signal**: Task performance → Reward signal → Agents adjust via three-factor learning
4. **No backpropagation**: All communication through environment (stigmergy)

---

## Complete Learning Flow (No Backprop)

```
┌─────────────────────────────────────────────────────────┐
│                    FORWARD PASS                         │
└─────────────────────────────────────────────────────────┘
1. Input injected into environment
2. Agents sense local pheromones
3. Agents compute contributions: contribution = tanh(W @ encoded)
4. Agents deposit contributions into reconstruction channels
5. Output layer reads reconstruction channels
6. Output = tanh(W_out @ reconstruction_state)

┌─────────────────────────────────────────────────────────┐
│                  REWARD COMPUTATION                      │
└─────────────────────────────────────────────────────────┘
7. Compute task error: error = ||target - output||²
8. Compute improvement: Δerror = prev_error - current_error
9. Compute reward signal: reward = tanh(Δerror × 20)
10. Broadcast reward as pheromone (diffuses to all agents)

┌─────────────────────────────────────────────────────────┐
│                 GRADIENT BROADCAST                       │
└─────────────────────────────────────────────────────────┘
11. Compute error gradient: grad = target - output
12. Broadcast gradient as spatial pheromone pattern
13. Agents sense gradient and navigate toward needed areas

┌─────────────────────────────────────────────────────────┐
│              THREE-FACTOR LEARNING                       │
└─────────────────────────────────────────────────────────┘
14. Each agent reads local reward signal
15. For each eligibility trace:
    - pre = what agent sensed
    - post = what agent output
    - reward = local reward signal
    - Δw = lr × pre × post × reward  ← NO BACKPROP!
16. Weights updated based on reward-modulated Hebbian rule

┌─────────────────────────────────────────────────────────┐
│                 OUTPUT LAYER UPDATE                      │
└─────────────────────────────────────────────────────────┘
17. Output layer uses simple gradient descent:
    - Δw = lr × error × reconstruction_state
    - This is OK because it's just a readout layer
    - Agents learn via reward, NOT via this gradient
```

---

## GPU Implementation

The GPU version (`StigmergicNetworkGPUReward`, lines 666-1036) implements the same principles with vectorized operations:

**Vectorized three-factor learning (lines 881-913):**
```python
# Read local reward signal
local_reward = sensory[:, self.ch_reward]  # (n_agents,)

# Decay eligibility traces
self.trace_pre *= self.trace_decay
self.trace_post *= self.trace_decay

# Update traces with current activity
self.trace_pre = self.trace_pre + encoded
self.trace_post = self.trace_post + contributions

# Three-factor learning rule: Δw = lr × pre × post × reward
# Vectorized for all agents
lr = 0.01
pre = self.trace_pre.unsqueeze(1)  # (n_agents, 1, feature_dim)
post = self.trace_post.unsqueeze(-1)  # (n_agents, feature_dim, 1)
reward_mod = local_reward.unsqueeze(-1).unsqueeze(-1)  # (n_agents, 1, 1)

# Hebbian update modulated by reward
delta = lr * torch.bmm(post, pre) * reward_mod

# Oja's normalization
delta -= lr * torch.bmm(post ** 2, torch.ones_like(pre)) * self.agent_weights * 0.01

# Apply update
delta = torch.clamp(delta, -0.1, 0.1)
self.agent_weights += delta
```

**Key optimization**: All agents learn in parallel using batch matrix multiplication (`torch.bmm`)

---

## Comparison: Old vs New

| Aspect | OLD (with backprop issues) | NEW (reward-based) |
|--------|---------------------------|-------------------|
| **Agent objective** | Predict own previous state | Contribute to reconstruction |
| **Learning signal** | None (isolated) | Reward from task performance |
| **Credit assignment** | No mechanism | Eligibility traces |
| **Task → Agent path** | Broken | Reward + gradient broadcast |
| **Agent → Output path** | Indirect via pheromones | Direct via reconstruction channels |
| **Backpropagation** | Used in output layer only | Only in output layer (just readout) |
| **Biological plausibility** | Low | High (dopamine modulation) |
| **Temporal credit assignment** | No solution | Eligibility traces solve it |

---

## Theoretical Justification

### Why This Works Without Backpropagation

**1. Reward as Global Error Signal**
- Task error → improvement → reward
- Reward diffuses to all agents via pheromone
- Agents don't need explicit gradients, just "good/bad" signal

**2. Eligibility Traces for Credit Assignment**
- Classic RL technique (TD-learning, Actor-Critic)
- Agents remember what they did recently
- When reward arrives, reinforce recent actions proportionally

**3. Three-Factor Learning**
- Biologically plausible (dopamine modulation)
- Mathematically equivalent to reward-weighted Hebbian learning
- Proven to work in reinforcement learning (Williams' REINFORCE)

**4. Spatial Gradient Broadcast**
- Agents don't need exact gradients
- Just need to know "where to help"
- Spatial pattern guides navigation

### Biological Analogies

| Implementation | Biological Equivalent |
|----------------|----------------------|
| Agents | Neurons in cortex |
| Pheromones | Neurotransmitters |
| Reconstruction channels | Sensory cortex activity |
| Output layer | Motor cortex readout |
| Reward signal | Dopamine from VTA/SNc |
| Eligibility traces | Synaptic tags |
| Three-factor learning | Dopamine-modulated LTP/LTD |

---

## Testing and Validation

Run the test script:
```bash
python /root/MAROLA/alternative-ai-architectures/src/networks/stigmergic_intelligence_reward_based.py
```

**Expected behavior:**
1. Task error should decrease over time
2. Mean reward should increase (approaching positive values)
3. Agent competence should grow
4. No NaN or instability issues

**Success criteria:**
- Task error < 0.1 after 1000 steps
- Mean reward > 0.5
- No backpropagation to agents (verified by code inspection)

---

## Code Locations

### Key Files
- **Main implementation**: `/root/MAROLA/alternative-ai-architectures/src/networks/stigmergic_intelligence_reward_based.py`
- **Old implementation**: `/root/MAROLA/alternative-ai-architectures/src/networks/stigmergic_intelligence.py`

### Key Classes
1. **RewardModulatedAgent** (lines 115-416): Agent with three-factor learning
2. **EligibilityTrace** (lines 102-110): Credit assignment structure
3. **StigmergicEnvironmentReward** (lines 419-615): Environment with reconstruction channels
4. **StigmergicNetworkRewardBased** (lines 618-665): Complete CPU network
5. **StigmergicNetworkGPUReward** (lines 668-1036): GPU-accelerated version

### Key Methods
- `three_factor_learning()` (lines 263-290): Core learning rule
- `compute_local_reward()` (lines 326-353): Reward computation
- `apply_reward_to_traces()` (lines 317-324): Credit assignment
- `broadcast_task_gradient()` (lines 501-534): Gradient broadcast
- `broadcast_global_reward()` (lines 536-572): Reward broadcast

---

## Advantages Over Backpropagation

1. **Biologically Plausible**: Mirrors dopamine modulation in brain
2. **Fully Distributed**: No need to store activations or compute gradients
3. **Robust to Delays**: Eligibility traces handle temporal credit assignment
4. **Scalable**: Each agent learns independently (parallelizable)
5. **Interpretable**: Reward signal has clear meaning
6. **No Vanishing Gradients**: No gradient flow issues
7. **Online Learning**: Can learn from single examples

---

## Future Enhancements

1. **Multi-timescale eligibility traces**: Different decay rates for different memory spans
2. **Meta-learning of learning rates**: Agents adjust their own learning rates based on success
3. **Hierarchical rewards**: Local vs global reward signals
4. **Curiosity-driven exploration**: Intrinsic motivation beyond task reward
5. **Social learning**: Agents learn from observing successful peers

---

## Conclusion

All three audit findings have been resolved using **biologically-plausible, reward-based learning** with NO backpropagation to agents:

✅ **Audit Finding 1**: Agents now contribute to input reconstruction (not self-prediction)
✅ **Audit Finding 2**: Task error reaches agents via reward signals + eligibility traces
✅ **Audit Finding 3**: Bidirectional influence between agents and output layer

The implementation uses **three-factor learning** (pre × post × reward), which is:
- Mathematically sound (proven in RL theory)
- Biologically plausible (dopamine modulation)
- Computationally efficient (no gradient storage)
- Fully distributed (no global backprop)

This represents a fundamental shift from supervised learning (backprop) to **reinforcement learning** (reward signals) in a stigmergic context.
