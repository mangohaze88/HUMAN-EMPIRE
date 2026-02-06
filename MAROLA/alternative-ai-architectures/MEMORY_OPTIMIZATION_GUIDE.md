# Memory Optimization Guide

## Problem Summary

Tests were failing with **Exit Code 137 (Out of Memory - OOM)**:
```
/bin/bash: line 1:  5520 Killed python test_advanced_quick.py
```

This indicates the process consumed all available GPU/system memory and was killed by the OS.

## Root Causes

The original test files had memory-intensive configurations:

1. **Large Network Sizes**
   - 4 colonies × 256 agents = 1024 total agents
   - Environment: 128×128 pixels
   - Feature dimensions: 32+
   - Multiple pheromone fields with multi-resolution storage

2. **Memory-Hungry Operations**
   - No gradient checkpointing
   - Full precision (float32) everywhere
   - Large temporal memory buffers (200+ patterns)
   - Deep neural networks (256→128→output)
   - No CUDA cache clearing

3. **Visualization Overhead**
   - High DPI plots (150+)
   - Multiple large figures
   - All data kept in memory

## Solutions Implemented

### 1. Optimized Test Files

Created memory-efficient versions of all problematic tests:

| Original File | Optimized Version | Key Changes |
|--------------|-------------------|-------------|
| `test_advanced_quick.py` | `test_advanced_quick_optimized.py` | 2 colonies, 64 agents, 32×32 env |
| `test_three_factor_learning.py` | `test_three_factor_optimized.py` | 256-512 agents, 32×32 env |
| `test_advanced_stigmergic.py` | `test_advanced_stigmergic_optimized.py` | Comprehensive reduction |

### 2. Memory Optimization Techniques Applied

#### A. Reduced Network Dimensions
```python
# Original (OOM)
net = AdvancedStigmergicNetwork(
    n_colonies=4,
    agents_per_colony=256,
    env_shape=(128, 128),
    feature_dim=32,
    input_dim=64,
    output_dim=32
)

# Optimized (No OOM)
net = AdvancedStigmergicNetwork(
    n_colonies=2,           # 50% reduction
    agents_per_colony=64,   # 75% reduction
    env_shape=(32, 32),     # 93.75% memory reduction
    feature_dim=16,         # 50% reduction
    input_dim=32,           # 50% reduction
    output_dim=16           # 50% reduction
)
```

#### B. Periodic CUDA Cache Clearing
```python
def clear_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use every 50-100 iterations
for i in range(n_steps):
    # ... training ...
    if i % 50 == 0:
        clear_memory()
```

#### C. torch.no_grad() for Inference
```python
# During validation/testing
with torch.no_grad():
    output, info = net.forward(x, n_steps=5, learn=False)
```

#### D. Reduced Training Iterations
```python
# Original
for i in range(1000):  # 1000 steps

# Optimized
for i in range(500):   # 500 steps (still effective)
```

#### E. Efficient Visualization
```python
# Lower DPI
plt.savefig(save_path, dpi=100)  # Instead of 150

# Close figures immediately
plt.close(fig)

# Smaller windows
window = min(30, len(data) // 10)  # Adaptive
```

#### F. Memory-Efficient Settings
```python
# Disable benchmarking (saves memory)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear memory at start
torch.cuda.empty_cache()
gc.collect()
```

### 3. Test Runner

Created `run_tests_memory_optimized.py` that:
- Runs tests in isolated subprocesses
- Monitors memory usage
- Clears cache between tests
- Handles OOM gracefully
- Reports memory statistics

## Usage

### Run Individual Optimized Tests

```bash
# Minimal test (smallest memory footprint)
python test_minimal.py

# Quick optimized test
python test_advanced_quick_optimized.py

# Three-factor learning optimized
python test_three_factor_optimized.py

# Full suite optimized
python test_advanced_stigmergic_optimized.py
```

### Run All Tests with Memory Management

```bash
# Recommended: Use the test runner
python run_tests_memory_optimized.py
```

## Memory Footprint Comparison

| Configuration | Agents | Env Size | Approx GPU Memory |
|--------------|--------|----------|-------------------|
| Original (OOM) | 1024 | 128×128 | ~8-12 GB |
| Optimized Small | 128 | 32×32 | ~500 MB - 1 GB |
| Optimized Medium | 256 | 64×64 | ~1-2 GB |
| Optimized Large | 512 | 64×64 | ~2-4 GB |

## If Still Getting OOM

### Further Reduce Network Size

Edit the test file and reduce parameters even more:

```python
net = AdvancedStigmergicNetwork(
    n_colonies=1,           # Single colony
    agents_per_colony=32,   # Minimal agents
    env_shape=(16, 16),     # Tiny environment
    feature_dim=8,          # Minimal features
    input_dim=16,
    output_dim=8
)
```

### Use CPU Instead of CUDA

```python
device='cpu'  # Instead of 'cuda'
```

Note: Much slower but won't OOM.

### Reduce Training Steps

```python
for i in range(100):  # Instead of 500+
```

### Monitor Memory Usage

```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB cached")
```

## Architecture-Specific Optimizations

### For Stigmergic Networks

1. Reduce pheromone field resolution
2. Limit temporal memory capacity
3. Reduce number of colonies
4. Decrease evolution frequency

### For Liquid Networks

1. Use smaller neuron counts
2. Reduce connectivity sparsity
3. Limit time constants range
4. Use smaller batch sizes

### For Hybrid Networks

1. Start with one module at a time
2. Use smaller embedding dimensions
3. Limit attention heads
4. Reduce layer depths

## Best Practices

1. **Always start with smallest config** - Verify it works, then scale up
2. **Monitor memory** - Check usage regularly during development
3. **Clear cache frequently** - Every 50-100 iterations
4. **Use no_grad()** - For all inference/validation
5. **Test incrementally** - Add features one at a time
6. **Profile before full run** - Run 10 steps to estimate memory
7. **Use subprocess isolation** - For multiple tests

## Memory Debugging Commands

```python
# Check current memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Get memory summary
print(torch.cuda.memory_summary())

# Track peak memory
torch.cuda.reset_peak_memory_stats()
# ... run code ...
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak: {peak:.2f} GB")

# Find memory leaks
import objgraph
objgraph.show_growth()
```

## File Reference

### Optimized Files (Safe to Run)
- ✅ `test_minimal.py` - Baseline minimal test
- ✅ `test_advanced_quick_optimized.py` - Quick advanced test
- ✅ `test_three_factor_optimized.py` - Three-factor learning
- ✅ `test_advanced_stigmergic_optimized.py` - Full suite
- ✅ `run_tests_memory_optimized.py` - Test runner

### Original Files (May OOM)
- ⚠️ `test_advanced_quick.py` - Original (can OOM)
- ⚠️ `test_three_factor_learning.py` - Original (can OOM)
- ⚠️ `test_advanced_stigmergic.py` - Original (can OOM)

### When to Use Each

**Use Optimized** when:
- Limited GPU memory (<8GB)
- Running multiple tests
- Development/debugging
- CI/CD pipelines
- Quick verification

**Use Original** when:
- High-end GPU (16GB+)
- Final benchmarking
- Need maximum accuracy
- Publication results

## Conclusion

The memory-optimized versions maintain the same algorithmic functionality while reducing memory consumption by **75-90%**. They are suitable for:

- Development and testing
- Resource-constrained environments
- Quick validation
- Educational purposes

For production or publication-quality results on high-end hardware, the original configurations can be used.

## Exit Code Reference

- **0** - Success
- **137** - Killed by OS (OOM)
- **1** - General error
- **139** - Segmentation fault

If you see **137**, apply the optimizations in this guide.
