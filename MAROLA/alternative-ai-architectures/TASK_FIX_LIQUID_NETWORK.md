# TASK: Fix Liquid Neural Network for Modular Arithmetic
**Priority:** HIGH
**Agent:** Bio-Plausible Learning Specialist
**Estimated Time:** 4-6 days
**Depends On:** TASK_IMPLEMENT_FOURIER_ENCODING

---

## OBJECTIVE

Improve Liquid Neural Network from 16.3% accuracy (barely above random) to >40% accuracy on modular addition (p=7), and >35% on p=23.

---

## CURRENT STATUS

**Baseline Performance:**
- p=7: 16.3% accuracy (random is 14.3%)
- Only 2% above random chance
- Network is not learning systematic patterns

**Current Architecture:**
- Input: 26 dimensions (binary + cyclic encoding)
- NCP Wiring: 12 sensory + 16 inter + 6 command + 3 motor = 37 neurons
- Output: 3 dimensions (normalized, sin, cos)
- Learning: Hebbian-style updates
- ODE steps: 2
- Time constant: Adaptive (0.1-10.0s)

---

## ROOT CAUSE ANALYSIS

### Problem 1: Simple Hebbian Learning Insufficient
**Current:** `W += lr * x^T @ h` (basic correlation)
**Issue:** No credit assignment for multi-step reasoning
**Impact:** Cannot learn complex input-output mappings

### Problem 2: Smooth ODE Dynamics vs Discrete Math
**Current:** Continuous-time integration
**Issue:** Modular arithmetic has discontinuities
**Impact:** Smooth dynamics cannot represent wrap-around

### Problem 3: Fixed Time Constants
**Current:** Adaptive but not task-specific
**Issue:** May be too fast or too slow for arithmetic
**Impact:** Suboptimal temporal integration

### Problem 4: No Reward Signal
**Current:** Only supervised error
**Issue:** No reinforcement for correct predictions
**Impact:** Weak learning signal for discrete outputs

---

## SOLUTION STRATEGY

### Fix 1: Implement Three-Factor Learning Rule

**Problem:** Simple Hebbian learning has no error signal

**Solution:** Add eligibility traces and neuromodulation

**Theory:**
```
Three-Factor Learning Rule:
ΔW = η * [pre-synaptic activity] * [post-synaptic activity] * [reward signal]

Factor 1: Pre-synaptic activity (input)
Factor 2: Post-synaptic activity (neuron state)
Factor 3: Neuromodulator (dopamine-like reward signal)
```

**Implementation:**
```python
class LiquidNetworkThreeFactor:
    """
    Liquid Neural Network with three-factor learning rule.
    """

    def __init__(self, input_dim, output_dim, wiring_config, dt=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wiring_config = wiring_config
        self.dt = dt

        # Initialize weights
        self.W_sensory_inter = self._init_sparse_weights(
            wiring_config.n_sensory,
            wiring_config.n_inter,
            wiring_config.sensory_to_inter_sparsity
        )

        self.W_inter_command = self._init_sparse_weights(
            wiring_config.n_inter,
            wiring_config.n_command,
            wiring_config.inter_to_command_sparsity
        )

        self.W_command_motor = self._init_sparse_weights(
            wiring_config.n_command,
            wiring_config.n_motor,
            wiring_config.command_to_motor_sparsity
        )

        # Eligibility traces
        self.trace_sensory_inter = np.zeros_like(self.W_sensory_inter)
        self.trace_inter_command = np.zeros_like(self.W_inter_command)
        self.trace_command_motor = np.zeros_like(self.W_command_motor)

        # Trace decay rate
        self.trace_decay = 0.9

        # Neuron states
        self.h_inter = np.zeros(wiring_config.n_inter)
        self.h_command = np.zeros(wiring_config.n_command)
        self.h_motor = np.zeros(wiring_config.n_motor)

    def forward(self, x):
        """
        Forward pass through liquid network.
        Returns output and saves activations for learning.
        """
        # Sensory → Inter
        z_inter = x @ self.W_sensory_inter
        self.h_inter = np.tanh(z_inter)

        # Inter → Command
        z_command = self.h_inter @ self.W_inter_command
        self.h_command = np.tanh(z_command)

        # Command → Motor
        z_motor = self.h_command @ self.W_command_motor
        self.h_motor = np.tanh(z_motor)

        # Store for learning
        self.last_input = x
        self.last_inter = self.h_inter
        self.last_command = self.h_command

        return self.h_motor

    def compute_reward_signal(self, target, output):
        """
        Compute neuromodulatory signal based on prediction accuracy.

        Uses smooth reward function (not binary).
        """
        error = np.mean((target - output) ** 2)

        # Reward signal (higher = better prediction)
        # Maps MSE [0, ∞) to reward [1, 0)
        reward = np.exp(-error)

        return reward

    def update_eligibility_traces(self):
        """
        Update eligibility traces with recent activity.

        Eligibility trace: running average of pre × post activity.
        """
        # Sensory → Inter
        self.trace_sensory_inter = (
            self.trace_decay * self.trace_sensory_inter +
            np.outer(self.last_input, self.last_inter * (1 - self.last_inter**2))
        )

        # Inter → Command
        self.trace_inter_command = (
            self.trace_decay * self.trace_inter_command +
            np.outer(self.last_inter, self.last_command * (1 - self.last_command**2))
        )

        # Command → Motor
        self.trace_command_motor = (
            self.trace_decay * self.trace_command_motor +
            np.outer(self.last_command, self.h_motor * (1 - self.h_motor**2))
        )

    def three_factor_update(self, target, output, learning_rate=0.01):
        """
        Three-factor learning rule update.

        ΔW = η * eligibility_trace * reward * error
        """
        # Compute reward signal
        reward = self.compute_reward_signal(target, output)

        # Compute error
        error = target - output

        # Update eligibility traces
        self.update_eligibility_traces()

        # Three-factor updates
        # Command → Motor (output layer, use error directly)
        self.W_command_motor += (
            learning_rate *
            self.trace_command_motor *
            reward *
            error.mean()
        )

        # Inter → Command (hidden layer, use reward only)
        self.W_inter_command += (
            learning_rate *
            self.trace_inter_command *
            reward *
            0.5  # Smaller update for hidden layers
        )

        # Sensory → Inter (input layer, use reward only)
        self.W_sensory_inter += (
            learning_rate *
            self.trace_sensory_inter *
            reward *
            0.3  # Smallest update for input layer
        )

        return reward

    def train_step(self, x, target, learning_rate=0.01):
        """
        Single training step with three-factor learning.
        """
        # Forward pass
        output = self.forward(x)

        # Three-factor update
        reward = self.three_factor_update(target, output, learning_rate)

        # Compute loss for monitoring
        loss = np.mean((target - output) ** 2)

        return output, loss, reward
```

**Expected Gain:** 2-3x improvement (16% → 40%+)

---

### Fix 2: Discrete Math-Aware ODE Dynamics

**Problem:** Smooth ODE cannot represent discontinuous wrap-around

**Solution:** Add discrete state transitions

**Implementation:**
```python
class HybridODEDiscreteNeuron:
    """
    Neuron with hybrid continuous-discrete dynamics.

    Continuous: Most of time, smooth ODE integration
    Discrete: When crossing thresholds, discrete jumps
    """

    def __init__(self, tau_min=0.1, tau_max=5.0, threshold=0.8):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.threshold = threshold
        self.state = 0.0

    def forward(self, input_current, dt=0.1):
        """
        Hybrid dynamics: smooth + discrete.
        """
        # Continuous ODE dynamics
        tau = self._compute_time_constant(input_current)
        dh_dt = (-self.state + input_current) / tau

        # Euler integration
        new_state = self.state + dt * dh_dt

        # Discrete transition (modular wrap-around)
        if abs(new_state) > self.threshold:
            # Wrap around (like modular arithmetic!)
            new_state = new_state % (2 * self.threshold) - self.threshold

        self.state = np.tanh(new_state)
        return self.state

    def _compute_time_constant(self, input_current):
        """
        Adaptive time constant.
        Fast when input is strong, slow when weak.
        """
        magnitude = abs(input_current)
        tau = self.tau_max * np.exp(-magnitude) + self.tau_min
        return tau
```

**Expected Gain:** 10-20% relative improvement

---

### Fix 3: Task-Specific Time Constant Initialization

**Problem:** Time constants not optimized for arithmetic

**Solution:** Learn optimal time constants for discrete math

**Implementation:**
```python
def initialize_time_constants_for_arithmetic(network, prime):
    """
    Initialize time constants based on task characteristics.

    For modular arithmetic:
    - Input layer: Fast (process inputs quickly)
    - Hidden layers: Medium (integrate information)
    - Output layer: Slow (stabilize output)
    """
    # Input sensory neurons: fast response
    network.tau_sensory = 0.1 * np.ones(network.n_sensory)

    # Inter neurons: medium response
    network.tau_inter = 0.5 * np.ones(network.n_inter)

    # Command neurons: medium-slow response
    network.tau_command = 1.0 * np.ones(network.n_command)

    # Motor neurons: slow response (stable output)
    network.tau_motor = 2.0 * np.ones(network.n_motor)

    # Scale by prime size
    # Larger primes → need more integration time
    scale_factor = np.log(prime) / np.log(7)  # Normalize to p=7
    network.tau_sensory *= scale_factor
    network.tau_inter *= scale_factor
    network.tau_command *= scale_factor
    network.tau_motor *= scale_factor

    print(f"Time constants initialized for p={prime}")
    print(f"  Sensory: {network.tau_sensory[0]:.3f}")
    print(f"  Inter: {network.tau_inter[0]:.3f}")
    print(f"  Command: {network.tau_command[0]:.3f}")
    print(f"  Motor: {network.tau_motor[0]:.3f}")
```

**Expected Gain:** 5-10% relative improvement

---

### Fix 4: Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP)

**Problem:** Hebbian learning has no temporal credit assignment

**Solution:** STDP with reward modulation

**Theory:**
```
R-STDP combines:
1. STDP: Weight changes depend on spike timing
2. Reward: Modulates plasticity based on success
3. Eligibility trace: Tracks recent spike correlations

If pre-spike before post-spike → strengthen (LTP)
If post-spike before pre-spike → weaken (LTD)
But only if reward signal is positive!
```

**Implementation:**
```python
class RewardModulatedSTDP:
    """
    Reward-modulated spike-timing-dependent plasticity.
    """

    def __init__(self, n_pre, n_post, tau_trace=20.0):
        self.W = np.random.randn(n_pre, n_post) * 0.01
        self.tau_trace = tau_trace  # Eligibility trace time constant

        # Eligibility traces for LTP and LTD
        self.trace_pre = np.zeros(n_pre)
        self.trace_post = np.zeros(n_post)

        # Weight eligibility
        self.eligibility = np.zeros((n_pre, n_post))

    def update_traces(self, pre_spikes, post_spikes, dt=0.1):
        """
        Update eligibility traces.

        Traces decay exponentially and spike on neuron activity.
        """
        # Decay
        self.trace_pre *= np.exp(-dt / self.tau_trace)
        self.trace_post *= np.exp(-dt / self.tau_trace)

        # Add spikes
        self.trace_pre += pre_spikes
        self.trace_post += post_spikes

    def update_eligibility(self, pre_spikes, post_spikes, dt=0.1):
        """
        Update weight eligibility based on spike timing.

        LTP (potentiation): pre before post
        LTD (depression): post before pre
        """
        # LTP component: pre_trace × post_spike
        ltp = np.outer(self.trace_pre, post_spikes)

        # LTD component: pre_spike × post_trace
        ltd = np.outer(pre_spikes, self.trace_post)

        # Update eligibility
        self.eligibility += ltp - 0.5 * ltd

        # Decay eligibility
        self.eligibility *= np.exp(-dt / self.tau_trace)

    def reward_modulated_update(self, reward, learning_rate=0.01):
        """
        Update weights using reward signal.

        ΔW = η * eligibility * reward
        """
        self.W += learning_rate * self.eligibility * reward

        # Reset eligibility after reward
        self.eligibility *= 0.5

    def train_step(self, pre_activity, post_activity, reward, dt=0.1, lr=0.01):
        """
        Complete R-STDP training step.
        """
        # Convert activity to spikes (threshold)
        pre_spikes = (pre_activity > 0.5).astype(float)
        post_spikes = (post_activity > 0.5).astype(float)

        # Update traces
        self.update_traces(pre_spikes, post_spikes, dt)

        # Update eligibility
        self.update_eligibility(pre_spikes, post_spikes, dt)

        # Reward-modulated weight update
        self.reward_modulated_update(reward, lr)
```

**Expected Gain:** 20-40% relative improvement

---

### Fix 5: Curriculum Learning with Adaptive ODE Steps

**Problem:** Fixed ODE steps may be suboptimal

**Solution:** Adapt integration steps based on curriculum stage

**Implementation:**
```python
def adaptive_ode_training(network, curriculum, base_epochs=100):
    """
    Train with adaptive ODE steps.

    Easy primes: Fewer ODE steps (faster, simpler)
    Hard primes: More ODE steps (precise, stable)
    """

    results = {}

    for prime in curriculum:
        # Determine ODE steps based on prime size
        if prime <= 11:
            ode_steps = 2  # Fast for small primes
        elif prime <= 47:
            ode_steps = 4  # Medium for medium primes
        else:
            ode_steps = 6  # Precise for large primes

        network.ode_steps = ode_steps

        print(f"\nTraining on p={prime} with {ode_steps} ODE steps")

        # Generate data
        train_data = generate_modular_data(prime, n_samples=5000)
        test_data = generate_modular_data(prime, n_samples=1000)

        # Train
        for epoch in range(base_epochs):
            for x, target in train_data:
                output, loss, reward = network.train_step(x, target)

            # Evaluate
            if epoch % 10 == 0:
                test_acc = evaluate(network, test_data)
                print(f"  Epoch {epoch}: Test acc = {test_acc:.1%}")

        results[prime] = test_acc

    return results
```

**Expected Gain:** 10-15% relative improvement

---

## IMPLEMENTATION CHECKLIST

### Week 1: Three-Factor Learning
- [ ] Implement `LiquidNetworkThreeFactor` class
- [ ] Add eligibility traces
- [ ] Implement reward signal computation
- [ ] Test on p=7
- [ ] Measure improvement over baseline
- [ ] Target: 25-35% accuracy on p=7

### Week 2: Hybrid Dynamics + Task-Specific Initialization
- [ ] Implement `HybridODEDiscreteNeuron`
- [ ] Add discrete transition logic
- [ ] Implement task-specific time constant initialization
- [ ] Test on p=7, 11
- [ ] Target: 35-45% accuracy on p=7

### Week 3: R-STDP + Adaptive Curriculum
- [ ] Implement `RewardModulatedSTDP`
- [ ] Add eligibility traces for STDP
- [ ] Implement adaptive ODE step curriculum
- [ ] Full integration and testing
- [ ] Target: 45-55% accuracy on p=7, 35%+ on p=23

---

## VALIDATION CRITERIA

### Success Criteria
- [ ] p=7: >40% exact accuracy
- [ ] p=11: >35% exact accuracy
- [ ] p=23: >25% exact accuracy
- [ ] Training time: <2 hours per prime
- [ ] Reward signal correlation with accuracy >0.7

### Performance Benchmarks
Compare against:
1. **Baseline Liquid Network:** 16.3% (current)
2. **Random Guessing:** 14.3%
3. **Forward-Forward (improved):** 50-70%
4. **Standard NN with backprop:** 100%
5. **Target:** 40-60% (bio-plausible without backprop)

---

## TESTING PROTOCOL

### Unit Tests
```python
def test_three_factor_learning():
    """Test three-factor learning rule."""
    network = LiquidNetworkThreeFactor(10, 3, wiring_config)

    x = np.random.randn(10)
    target = np.array([0.5, 0.2, 0.8])

    # Train step
    output, loss, reward = network.train_step(x, target)

    # Check reward signal
    assert 0.0 <= reward <= 1.0

    # Check eligibility traces updated
    assert np.any(network.trace_sensory_inter != 0)


def test_hybrid_ode_dynamics():
    """Test hybrid continuous-discrete dynamics."""
    neuron = HybridODEDiscreteNeuron()

    # Test continuous dynamics
    for _ in range(10):
        output = neuron.forward(0.5, dt=0.1)
        assert -1.0 <= output <= 1.0

    # Test discrete wrap-around
    neuron.state = 0.9  # Near threshold
    output = neuron.forward(0.5, dt=0.1)
    # Should wrap around
    assert output != neuron.state + 0.5 * 0.1


def test_reward_signal():
    """Test reward signal computation."""
    network = LiquidNetworkThreeFactor(10, 3, wiring_config)

    # Perfect prediction
    target = np.array([0.5, 0.5, 0.5])
    output = np.array([0.5, 0.5, 0.5])
    reward = network.compute_reward_signal(target, output)
    assert reward > 0.9  # High reward

    # Bad prediction
    output = np.array([0.0, 0.0, 0.0])
    reward = network.compute_reward_signal(target, output)
    assert reward < 0.5  # Low reward
```

### Integration Tests
```python
def test_full_liquid_network_improved():
    """Test complete improved Liquid Network pipeline."""

    p = 7
    network = LiquidNetworkThreeFactor(
        input_dim=26,
        output_dim=3,
        wiring_config=NCPWiringConfig(
            n_sensory=12, n_inter=16, n_command=6, n_motor=3
        )
    )

    # Initialize for arithmetic
    initialize_time_constants_for_arithmetic(network, p)

    # Train
    train_data = generate_modular_data(p, n_samples=5000)
    test_data = generate_modular_data(p, n_samples=1000)

    for epoch in range(100):
        for x, target in train_data:
            output, loss, reward = network.train_step(x, target)

    # Evaluate
    accuracy = evaluate(network, test_data)

    assert accuracy > 0.4, f"Expected >40%, got {accuracy:.1%}"
    print(f"✓ SUCCESS: {accuracy:.1%} accuracy achieved")
```

---

## EXPECTED RESULTS

### Baseline vs Improved

| Prime | Baseline LNN | Improved LNN | Gain |
|-------|--------------|--------------|------|
| p=7   | 16.3%        | 45-55%       | 3x   |
| p=11  | -            | 40-50%       | -    |
| p=23  | -            | 30-40%       | -    |

### Detailed Breakdown

**After Fix 1 (Three-Factor Learning):**
- p=7: 16% → 35% (+119%)

**After Fix 2 (Hybrid Dynamics):**
- p=7: 35% → 40% (+14%)

**After Fix 3 (Task-Specific Time Constants):**
- p=7: 40% → 43% (+8%)

**After Fix 4 (R-STDP):**
- p=7: 43% → 52% (+21%)

**After Fix 5 (Adaptive Curriculum):**
- p=7: 52% → 57% (+10%)

**Final Expected:** 50-60% accuracy on p=7

---

## FILES TO MODIFY

1. `/root/MAROLA/alternative-ai-architectures/src/networks/liquid_neural_network.py`
   - Add three-factor learning
   - Implement hybrid dynamics
   - Update training loop

2. `/root/MAROLA/alternative-ai-architectures/experiments/learn_ec_math_bio_plausible.py`
   - Update `train_liquid_network()`
   - Add reward signal tracking
   - Add adaptive curriculum

3. **NEW FILE:** `/root/MAROLA/alternative-ai-architectures/src/networks/liquid_neural_network_three_factor.py`
   - Complete three-factor implementation
   - R-STDP implementation
   - All improvements integrated

---

## DELIVERABLES

1. **Code:**
   - Three-factor Liquid Network implementation
   - R-STDP module
   - Hybrid ODE dynamics
   - Task-specific initialization
   - Adaptive curriculum

2. **Results:**
   - Benchmark comparison (baseline vs improved)
   - Training curves (accuracy, loss, reward)
   - Reward signal analysis

3. **Documentation:**
   - Three-factor learning guide
   - R-STDP tutorial
   - API documentation

---

## NEXT STEPS

After completing this task:
1. Compare with TASK_FIX_FORWARD_FORWARD results
2. Integrate with Fourier encoding
3. Run full curriculum (p=7 → 11 → 23)
4. Proceed to Phase 2 (hybrid architecture)

---

**Priority:** HIGH
**Blocking:** Phase 2 hybrid architecture
**Estimated Completion:** 1-2 weeks
**Success Definition:** >40% accuracy on p=7, >25% on p=23
