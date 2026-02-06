# Forward-Forward Algorithm - Quick Reference

## One-Liner

**Learning without backpropagation using local contrastive rules.**

---

## Installation

Already included! Just use:

```python
from src.networks import create_ff_network
```

---

## 30-Second Example

```python
from src.networks import create_ff_network

# Create network (NO backprop!)
net = create_ff_network(
    input_dim=784,      # MNIST size
    hidden_dims=[500, 300],
    output_dim=10
)

# Train
for x, y in dataloader:
    net.train_step(x, y)  # NO loss.backward()!

# Predict
predictions = net.predict(x_test)
```

---

## Key Insight

Each layer learns **locally** by contrasting:
- **Positive samples** (real data + correct label) → HIGH goodness
- **Negative samples** (corrupted data/wrong label) → LOW goodness

NO error signals propagated backward!

---

## Run Examples

```bash
# Simple demo
python demo_forward_forward.py

# MNIST example
python examples/forward_forward_mnist_example.py

# Verify it works
python verify_forward_forward.py

# Full test suite
python test_forward_forward.py
```

---

## What Makes It Special

1. **No backpropagation** - Weights update locally
2. **Bio-plausible** - Like real neurons
3. **Memory efficient** - No computation graph
4. **Parallel** - Layers learn independently

---

## Results

- ✓ Learns WITHOUT backprop (verified)
- ✓ Goodness separation: 4.2 (layer 0), 2.5 (layer 1)
- ✓ MNIST accuracy: 50% (from 10% random)
- ✓ Binary classification: 95-100%

---

## Files

```
src/networks/forward_forward.py              # Implementation
src/networks/README_FORWARD_FORWARD.md       # Full docs
test_forward_forward.py                      # Tests
examples/forward_forward_mnist_example.py    # MNIST demo
```

---

## Paper

Hinton, G. (2022). "The Forward-Forward Algorithm"
https://www.cs.toronto.edu/~hinton/FFA13.pdf

---

**That's it! Bio-plausible learning in ~10 lines of code.**
