# TASK: Fix Forward-Forward Network for Modular Arithmetic
**Priority:** HIGH
**Agent:** Bio-Plausible Learning Specialist
**Estimated Time:** 3-5 days
**Depends On:** TASK_IMPLEMENT_FOURIER_ENCODING

---

## OBJECTIVE

Improve Forward-Forward network from 14.3% accuracy (random baseline) to >50% accuracy on modular addition (p=7), and >70% on p=23.

---

## CURRENT STATUS

**Baseline Performance:**
- p=7: 14.3% accuracy (random is 14.3% = 1/7)
- Only slightly above random chance
- Network is not learning meaningful patterns

**Current Architecture:**
- Input: 26 dimensions (binary + cyclic encoding)
- Hidden: [128, 128]
- Output: 7 classes
- Learning: Local contrastive (positive/negative samples)
- Threshold: 2.0
- Learning rate: 0.1

---

## ROOT CAUSE ANALYSIS

### Problem 1: Weak Negative Samples
**Current:** Random label corruption
**Issue:** Negatives are too easy to distinguish
**Impact:** Network memorizes, doesn't learn structure

### Problem 2: Insufficient Goodness Separation
**Current:** Threshold = 2.0
**Issue:** Positive and negative activations overlap
**Impact:** No clear learning signal

### Problem 3: No Layer Coordination
**Current:** Each layer learns independently
**Issue:** Cannot form compositional representations
**Impact:** Limited capacity for complex operations

### Problem 4: Fixed Epoch Count
**Current:** 30-50 epochs
**Issue:** Need more time to discover structure
**Impact:** Stopped before grokking can occur

---

## SOLUTION STRATEGY

### Fix 1: Hard Negative Sample Generation

**Replace:** Random label corruption
**With:** Structured hard negatives

**Implementation:**
```python
def generate_hard_negatives(a: int, b: int, correct_result: int, p: int, strategy: str = "near_miss"):
    """
    Generate hard negative samples for Forward-Forward training.

    Strategies:
    - near_miss: Results that are off by ±1 or ±2
    - boundary: Wrap-around cases
    - symmetric: Exploit commutativity confusion
    """
    if strategy == "near_miss":
        # Results close to correct answer
        offsets = [-2, -1, 1, 2]
        negatives = [(correct_result + offset) % p for offset in offsets]

    elif strategy == "boundary":
        # Wrap-around confusion cases
        negatives = [
            (correct_result + 1) % p,  # Just past wrap
            (correct_result - 1) % p,  # Just before wrap
            (p - 1 - correct_result) % p,  # Symmetric wrap
        ]

    elif strategy == "symmetric":
        # Exploit commutativity
        # (a+b) often confused with (b+a) in early training
        negatives = [
            (a + a) % p,  # Double first input
            (b + b) % p,  # Double second input
            (a * b) % p,  # Wrong operation
        ]

    # Remove correct answer if it appears
    negatives = [n for n in negatives if n != correct_result]

    # Return multiple negatives for contrastive learning
    return negatives[:3]  # Top 3 hard negatives


def embed_sample_with_label(x: torch.Tensor, label: int, p: int, embedding_dim: int = 10):
    """
    Embed label information into input.

    Uses one-hot encoding of label concatenated with input features.
    This allows the network to learn to discriminate labeled inputs.
    """
    batch_size = x.shape[0]

    # One-hot encode label
    label_onehot = torch.zeros(batch_size, p)
    label_onehot[:, label] = 1.0

    # Concatenate with input
    return torch.cat([x, label_onehot], dim=1)
```

**Expected Gain:** 2-3x improvement (14% → 35%)

---

### Fix 2: Adaptive Threshold and Goodness Measurement

**Problem:** Fixed threshold doesn't adapt to data distribution

**Solution:**
```python
class AdaptiveForwardForwardLayer:
    """
    Forward-Forward layer with adaptive threshold.
    """

    def __init__(self, input_dim, output_dim, initial_threshold=2.0):
        self.weights = torch.randn(input_dim, output_dim) * 0.01
        self.bias = torch.zeros(output_dim)

        # Adaptive threshold
        self.threshold = initial_threshold
        self.pos_goodness_history = []
        self.neg_goodness_history = []

    def compute_goodness(self, activations):
        """
        Compute goodness score.
        Higher = more positive-like.
        """
        # Sum of squared activations (energy-based)
        return torch.sum(activations ** 2, dim=1)

    def forward(self, x):
        """Forward pass."""
        activations = torch.relu(x @ self.weights + self.bias)
        goodness = self.compute_goodness(activations)
        return activations, goodness

    def train_step(self, x_pos, x_neg):
        """
        Train on positive and negative samples.
        """
        # Forward pass
        act_pos, good_pos = self.forward(x_pos)
        act_neg, good_neg = self.forward(x_neg)

        # Track goodness statistics
        self.pos_goodness_history.append(good_pos.mean().item())
        self.neg_goodness_history.append(good_neg.mean().item())

        # Adapt threshold (running mean of midpoint)
        if len(self.pos_goodness_history) > 10:
            avg_pos = np.mean(self.pos_goodness_history[-100:])
            avg_neg = np.mean(self.neg_goodness_history[-100:])
            self.threshold = (avg_pos + avg_neg) / 2.0

        # Compute loss
        # Positive samples should be ABOVE threshold
        loss_pos = torch.relu(self.threshold - good_pos).mean()

        # Negative samples should be BELOW threshold
        loss_neg = torch.relu(good_neg - self.threshold).mean()

        # Total loss
        loss = loss_pos + loss_neg

        # Update weights (LOCAL UPDATE - no backprop through layers!)
        with torch.no_grad():
            # Positive direction: increase goodness
            grad_pos = x_pos.T @ (act_pos * (1.0 - (good_pos > self.threshold).float()[:, None]))

            # Negative direction: decrease goodness
            grad_neg = x_neg.T @ (act_neg * ((good_neg > self.threshold).float()[:, None]))

            # Update
            learning_rate = 0.03
            self.weights += learning_rate * (grad_pos - grad_neg) / x_pos.shape[0]

        return loss.item()
```

**Expected Gain:** 20-30% relative improvement

---

### Fix 3: Layer-Wise Curriculum Learning

**Problem:** Deep layers never see meaningful signals

**Solution:**
```python
def train_forward_forward_curriculum(model, data, epochs=1000):
    """
    Train Forward-Forward with layer-wise curriculum.

    Phase 1: Train layer 1 only
    Phase 2: Freeze layer 1, train layer 2
    Phase 3: Fine-tune all layers together
    """

    # Phase 1: Train first layer (epochs 0-300)
    print("Phase 1: Training layer 1...")
    for epoch in range(300):
        for x_pos, x_neg in data:
            model.layers[0].train_step(x_pos, x_neg)

    # Phase 2: Train second layer (epochs 300-600)
    print("Phase 2: Training layer 2...")
    for epoch in range(300):
        for x_pos, x_neg in data:
            # Forward through layer 1 (frozen)
            with torch.no_grad():
                h1_pos, _ = model.layers[0].forward(x_pos)
                h1_neg, _ = model.layers[0].forward(x_neg)

            # Train layer 2
            model.layers[1].train_step(h1_pos, h1_neg)

    # Phase 3: Fine-tune all layers (epochs 600-1000)
    print("Phase 3: Fine-tuning all layers...")
    for epoch in range(400):
        for x_pos, x_neg in data:
            # Train all layers
            for layer in model.layers:
                h_pos, _ = layer.forward(x_pos)
                h_neg, _ = layer.forward(x_neg)
                layer.train_step(x_pos, x_neg)

                # Use output as input for next layer
                x_pos = h_pos
                x_neg = h_neg
```

**Expected Gain:** 30-50% relative improvement

---

### Fix 4: Extended Training with Grokking

**Problem:** 30-50 epochs is not enough

**Solution:**
```python
def train_until_grokking(model, train_data, test_data, max_epochs=5000):
    """
    Train Forward-Forward until grokking occurs.

    Grokking indicators:
    - Sudden jump in test accuracy
    - Training accuracy already high
    """

    best_test_acc = 0.0
    patience = 0
    max_patience = 500  # Very long patience

    train_accs = []
    test_accs = []

    for epoch in range(max_epochs):
        # Training
        for x_pos, x_neg in train_data:
            model.train_step(x_pos, x_neg)

        # Evaluation (every 10 epochs)
        if epoch % 10 == 0:
            train_acc = evaluate(model, train_data)
            test_acc = evaluate(model, test_data)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f"Epoch {epoch}: Train={train_acc:.1%}, Test={test_acc:.1%}")

            # Check for grokking
            if len(test_accs) > 5:
                # Sudden jump detection
                recent_improvement = test_acc - np.mean(test_accs[-5:-1])
                if recent_improvement > 0.2:  # 20% jump
                    print(f"🎉 GROKKING detected at epoch {epoch}!")
                    print(f"   Test accuracy jumped by {recent_improvement:.1%}")

            # Check for convergence
            if test_acc > 0.95:
                print(f"✓ Converged at epoch {epoch}")
                break

            # Early stopping with long patience
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1

            if patience > max_patience:
                print(f"Stopping at epoch {epoch} (no improvement)")
                break

    return model, train_accs, test_accs
```

**Expected Gain:** Enable generalization (may unlock 50%+ accuracy)

---

### Fix 5: Auxiliary Contrastive Tasks

**Problem:** Single task provides weak signal

**Solution:**
```python
def create_auxiliary_tasks(a, b, result, p):
    """
    Generate auxiliary contrastive tasks to strengthen learning.

    Tasks:
    1. Parity prediction (even/odd result)
    2. Magnitude prediction (result > p/2)
    3. Boundary detection (result near 0 or p)
    4. Commutativity verification (a+b == b+a)
    """
    auxiliary_labels = {}

    # Task 1: Parity
    auxiliary_labels['parity'] = result % 2

    # Task 2: Magnitude
    auxiliary_labels['magnitude'] = int(result > p // 2)

    # Task 3: Boundary
    boundary_threshold = p // 10
    auxiliary_labels['boundary'] = int(
        result < boundary_threshold or result > (p - boundary_threshold)
    )

    # Task 4: Commutativity (always True for addition)
    auxiliary_labels['commutative'] = 1

    return auxiliary_labels


def train_with_auxiliary_tasks(model, data, epochs=1000):
    """
    Train Forward-Forward with multi-task learning.
    """
    for epoch in range(epochs):
        for x, labels in data:
            # Main task: modular addition
            main_loss = model.train_step_main(x, labels['result'])

            # Auxiliary tasks
            aux_losses = []
            for task_name, task_label in labels['auxiliary'].items():
                aux_loss = model.train_step_auxiliary(x, task_label, task_name)
                aux_losses.append(aux_loss)

            # Combined learning signal
            total_loss = main_loss + 0.3 * np.mean(aux_losses)
```

**Expected Gain:** 10-20% relative improvement

---

## IMPLEMENTATION CHECKLIST

### Week 1: Hard Negatives + Adaptive Threshold
- [ ] Implement `generate_hard_negatives()` function
- [ ] Test negative generation strategies (near_miss, boundary, symmetric)
- [ ] Implement `AdaptiveForwardForwardLayer`
- [ ] Test threshold adaptation on p=7
- [ ] Measure accuracy improvement
- [ ] Target: 30-40% accuracy on p=7

### Week 2: Layer-Wise Curriculum + Extended Training
- [ ] Implement layer-wise curriculum
- [ ] Test phase-by-phase training
- [ ] Implement extended training loop (5000 epochs)
- [ ] Add grokking detection
- [ ] Measure accuracy improvement
- [ ] Target: 50-60% accuracy on p=7

### Week 3: Auxiliary Tasks + Integration
- [ ] Implement auxiliary task generation
- [ ] Add multi-task training loop
- [ ] Integrate all improvements
- [ ] Full curriculum test (p=7, 11, 23)
- [ ] Target: 70%+ accuracy on p=23

---

## VALIDATION CRITERIA

### Success Criteria
- [ ] p=7: >50% exact accuracy
- [ ] p=11: >45% exact accuracy
- [ ] p=23: >30% exact accuracy
- [ ] Training time: <1 hour per prime
- [ ] Grokking observed (sudden accuracy jump)

### Performance Benchmarks
Compare against:
1. **Baseline Forward-Forward:** 14.3% (current)
2. **Random Guessing:** 14.3% (1/7 for p=7)
3. **Standard NN with backprop:** 100%
4. **Target:** 50-70% (bio-plausible without backprop)

---

## TESTING PROTOCOL

### Unit Tests
```python
def test_hard_negatives():
    """Test hard negative generation."""
    a, b = 3, 5
    result = (a + b) % 7  # = 1

    negatives = generate_hard_negatives(a, b, result, p=7, strategy="near_miss")

    assert result not in negatives
    assert all(0 <= n < 7 for n in negatives)
    assert len(negatives) == 3


def test_adaptive_threshold():
    """Test threshold adaptation."""
    layer = AdaptiveForwardForwardLayer(10, 20)

    # Simulate training
    for _ in range(100):
        x_pos = torch.randn(32, 10) + 1.0  # Positive samples
        x_neg = torch.randn(32, 10) - 1.0  # Negative samples
        layer.train_step(x_pos, x_neg)

    # Threshold should be between positive and negative goodness
    avg_pos = np.mean(layer.pos_goodness_history[-10:])
    avg_neg = np.mean(layer.neg_goodness_history[-10:])

    assert avg_neg < layer.threshold < avg_pos


def test_grokking_detection():
    """Test grokking detection logic."""
    test_accs = [0.15, 0.16, 0.14, 0.15, 0.16, 0.45]  # Sudden jump

    recent_improvement = test_accs[-1] - np.mean(test_accs[-5:-1])
    assert recent_improvement > 0.2  # 20% jump detected
```

### Integration Tests
```python
def test_full_forward_forward_improved():
    """Test complete improved Forward-Forward pipeline."""

    # Setup
    p = 7
    model = ImprovedForwardForwardNetwork(
        input_dim=26,
        hidden_dims=[128, 128],
        output_dim=p,
        use_hard_negatives=True,
        use_adaptive_threshold=True,
        use_auxiliary_tasks=True
    )

    # Generate data
    train_data = generate_modular_addition_data(p, n_samples=5000)
    test_data = generate_modular_addition_data(p, n_samples=1000)

    # Train
    model, history = train_until_grokking(
        model, train_data, test_data, max_epochs=2000
    )

    # Validate
    final_accuracy = history['test_accs'][-1]
    assert final_accuracy > 0.5, f"Expected >50%, got {final_accuracy:.1%}"

    print(f"✓ SUCCESS: {final_accuracy:.1%} accuracy achieved")
```

---

## EXPECTED RESULTS

### Baseline vs Improved

| Prime | Baseline FF | Improved FF | Gain |
|-------|-------------|-------------|------|
| p=7   | 14.3%       | 50-60%      | 3-4x |
| p=11  | -           | 45-55%      | -    |
| p=23  | -           | 30-40%      | -    |

### Detailed Breakdown

**After Fix 1 (Hard Negatives):**
- p=7: 14% → 35% (+150%)

**After Fix 2 (Adaptive Threshold):**
- p=7: 35% → 42% (+20%)

**After Fix 3 (Layer Curriculum):**
- p=7: 42% → 52% (+24%)

**After Fix 4 (Extended Training):**
- p=7: 52% → 58% (+12%)

**After Fix 5 (Auxiliary Tasks):**
- p=7: 58% → 65% (+12%)

**Final Expected:** 60-70% accuracy on p=7

---

## FILES TO MODIFY

1. `/root/MAROLA/alternative-ai-architectures/src/networks/forward_forward.py`
   - Add `generate_hard_negatives()`
   - Implement `AdaptiveForwardForwardLayer`
   - Update training loop

2. `/root/MAROLA/alternative-ai-architectures/experiments/learn_ec_math_bio_plausible.py`
   - Update `train_forward_forward()`
   - Add curriculum learning
   - Add grokking detection

3. **NEW FILE:** `/root/MAROLA/alternative-ai-architectures/src/networks/forward_forward_improved.py`
   - Complete improved implementation
   - All fixes integrated

---

## DELIVERABLES

1. **Code:**
   - Improved Forward-Forward implementation
   - Hard negative generation
   - Adaptive threshold mechanism
   - Layer-wise curriculum
   - Auxiliary task system

2. **Results:**
   - Benchmark comparison (baseline vs improved)
   - Training curves (accuracy, loss, threshold)
   - Grokking detection logs

3. **Documentation:**
   - Implementation guide
   - API documentation
   - Usage examples

---

## NEXT STEPS

After completing this task:
1. Integrate with Fourier encoding (from TASK_IMPLEMENT_FOURIER_ENCODING)
2. Run full curriculum (p=7 → 11 → 23 → 47)
3. Compare with TASK_FIX_LIQUID_NETWORK results
4. Proceed to Phase 2 (NALU integration)

---

**Priority:** HIGH
**Blocking:** Phase 2 tasks
**Estimated Completion:** 1-2 weeks
**Success Definition:** >50% accuracy on p=7, >30% on p=23
