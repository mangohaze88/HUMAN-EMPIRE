# Quick Fix for OOM (Exit Code 137)

## Problem
```
/bin/bash: line 1:  5520 Killed python test_advanced_quick.py
```
Exit code 137 = Out of Memory

## Instant Solution

### Option 1: Use Optimized Tests (Recommended)
```bash
# Instead of the original tests, use these:
python test_advanced_quick_optimized.py
python test_three_factor_optimized.py
python test_advanced_stigmergic_optimized.py

# Or run all at once:
python run_tests_memory_optimized.py
```

### Option 2: Quick Fix Original Files

Add this to the top of any test file that's OOMing:

```python
import gc
import torch

# Clear memory first
torch.cuda.empty_cache()
gc.collect()

# Add this function
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Then in your training loop, every ~50 steps:
for i in range(steps):
    # ... your code ...
    if i % 50 == 0:
        clear_memory()
```

### Option 3: Reduce Network Size

Find these lines in the test and reduce the numbers:

```python
# BEFORE (OOM)
net = AdvancedStigmergicNetwork(
    n_colonies=4,
    agents_per_colony=256,
    env_shape=(128, 128),
    feature_dim=32,
)

# AFTER (No OOM)
net = AdvancedStigmergicNetwork(
    n_colonies=2,           # Halve this
    agents_per_colony=64,   # Quarter this
    env_shape=(32, 32),     # Quarter each dimension
    feature_dim=16,         # Halve this
)
```

## What Changed in Optimized Files

1. **Smaller Networks**
   - Fewer agents (64-128 instead of 256-1024)
   - Smaller environments (32×32 instead of 128×128)
   - Reduced feature dimensions

2. **Memory Management**
   - Periodic CUDA cache clearing
   - Garbage collection
   - torch.no_grad() for inference

3. **Reduced Iterations**
   - 200-500 steps instead of 1000+
   - Fewer test inputs
   - Smaller batch sizes

## Memory Usage Comparison

| Test Type | Original | Optimized | Memory Saved |
|-----------|----------|-----------|--------------|
| Quick Test | ~4-6 GB | ~0.5-1 GB | 80-85% |
| Three-Factor | ~6-8 GB | ~1-2 GB | 75-80% |
| Full Suite | ~8-12 GB | ~2-4 GB | 70-75% |

## Files Overview

### ✅ Safe to Run (Won't OOM)
- `test_minimal.py` - Smallest test
- `test_advanced_quick_optimized.py` - Quick test (optimized)
- `test_three_factor_optimized.py` - Learning test (optimized)
- `test_advanced_stigmergic_optimized.py` - Full suite (optimized)
- `run_tests_memory_optimized.py` - Run all tests safely

### ⚠️ May OOM on Low Memory Systems
- `test_advanced_quick.py` - Original quick test
- `test_three_factor_learning.py` - Original learning test
- `test_advanced_stigmergic.py` - Original full suite

## Quick Test

To verify the fix works:

```bash
# Run the smallest test first
python test_minimal.py

# If that works, try the optimized quick test
python test_advanced_quick_optimized.py

# If that works, run all optimized tests
python run_tests_memory_optimized.py
```

## Still Getting OOM?

Reduce parameters even more in the optimized files:

```python
# Ultra-minimal config
net = AdvancedStigmergicNetwork(
    n_colonies=1,        # Just one colony
    agents_per_colony=32, # Minimal agents
    env_shape=(16, 16),  # Tiny environment
    feature_dim=8,       # Minimal features
    input_dim=16,
    output_dim=8
)
```

Or switch to CPU (slower but won't OOM):
```python
device='cpu'  # Instead of 'cuda'
```

## Prevention Tips

1. Always start with `test_minimal.py` to verify setup
2. Use optimized versions for development
3. Only use original versions on high-end GPUs (16GB+)
4. Monitor memory with `nvidia-smi` or `torch.cuda.memory_allocated()`
5. Clear cache periodically in long-running tests

## More Details

See `MEMORY_OPTIMIZATION_GUIDE.md` for comprehensive information.
