# Quick Start: Three-Factor Learning

## Basic Usage

```python
import torch
from networks.stigmergic_intelligence import StigmergicNetworkGPU

# Create network
net = StigmergicNetworkGPU(
    n_agents=1024,
    env_shape=(64, 64),
    input_dim=64,
    output_dim=32,
    device='cuda'
)

# Training
x = torch.randn(64, device='cuda')

for step in range(1000):
    output, info = net.forward(x, n_steps=5, learn=True)

    if step % 100 == 0:
        print(f"Step {step}: task_error={info['task_error']:.6f}")
```

## Key Parameters

- `n_agents`: Number of stigmergic agents (default: 1024)
- `env_shape`: Pheromone field size (default: 64×64)
- `input_dim`: Input vector dimension
- `output_dim`: Output vector dimension
- `n_steps`: Steps per forward pass (default: 5)

## What to Expect

- **Initial error**: ~0.1-0.5 (random)
- **After 500 steps**: 10-20% improvement
- **After 1000 steps**: 20-37% improvement
- **Convergence**: ~1500-2000 steps

## Monitoring

```python
# Track learning
task_errors = []
for step in range(1000):
    output, info = net.forward(x, n_steps=5, learn=True)
    task_errors.append(info['task_error'])

# Check improvement
import numpy as np
initial = np.mean(task_errors[:100])
final = np.mean(task_errors[-100:])
improvement = 100 * (initial - final) / initial
print(f"Improvement: {improvement:.1f}%")
```

## Testing

Run validation suite:
```bash
python test_three_factor_learning.py
```

## Troubleshooting

### Task error not decreasing?
- Check learning rate is enabled: `learn=True`
- Increase training steps (try 1500+)
- Verify CUDA is available

### Memory issues?
- Reduce `n_agents` (try 512)
- Reduce `env_shape` (try 32×32)

### Slow training?
- Reduce `n_steps` per forward (try 3)
- Use smaller environment
- Check GPU utilization

## Advanced Features

### Multiple Inputs
```python
inputs = [torch.randn(64, device='cuda') for _ in range(5)]

for x in inputs:
    for _ in range(200):
        net.forward(x, n_steps=5, learn=True)
```

### Save/Load
```python
# Save
torch.save({
    'agent_weights': net.agent_weights,
    'eligibility_traces': net.eligibility_traces,
    'output_w1': net.output_w1,
    'output_w2': net.output_w2,
}, 'model.pt')

# Load
checkpoint = torch.load('model.pt')
net.agent_weights = checkpoint['agent_weights']
net.eligibility_traces = checkpoint['eligibility_traces']
net.output_w1 = checkpoint['output_w1']
net.output_w2 = checkpoint['output_w2']
```

### Inference Mode
```python
# Disable learning for inference
output, info = net.forward(x, n_steps=5, learn=False)
```

## Performance Tips

1. **Use GPU**: Essential for good performance
2. **Batch Size**: One forward pass at a time works best
3. **Steps**: 5 steps per forward is a good balance
4. **Agents**: 1024 agents optimal for 64×64 environment
5. **Warm-up**: First 100 steps are initialization

## Example: Full Training Loop

```python
import torch
import numpy as np
from networks.stigmergic_intelligence import StigmergicNetworkGPU

# Setup
net = StigmergicNetworkGPU(n_agents=1024, device='cuda')
x = torch.randn(64, device='cuda')

# Train
print("Training...")
errors = []
for i in range(1000):
    output, info = net.forward(x, n_steps=5, learn=True)
    errors.append(info['task_error'])

    if i % 200 == 0:
        print(f"  {i}: {info['task_error']:.6f}")

# Results
print(f"\nImprovement: {100*(errors[0]-errors[-1])/errors[0]:.1f}%")
```

## See Also

- `THREE_FACTOR_LEARNING_IMPLEMENTATION.md` - Full implementation details
- `THREE_FACTOR_LEARNING_SUMMARY.md` - Executive summary
- `test_three_factor_learning.py` - Validation suite
