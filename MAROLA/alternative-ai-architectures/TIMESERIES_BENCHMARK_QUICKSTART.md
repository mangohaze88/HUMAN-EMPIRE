# Time Series Benchmark - Quick Start

## What We Built

A **REAL** time series benchmark testing bio-plausible architectures on actual challenging data (NOT toy sine waves).

## Run It

```bash
cd /root/MAROLA/alternative-ai-architectures
python experiments/real_benchmark_timeseries.py
```

Runtime: ~30-60 seconds

## What It Tests

### 3 Real-World Tasks:
1. **Mackey-Glass** - Classic chaotic time series (deterministic chaos)
2. **Lorenz Attractor** - 3D butterfly attractor (nonlinear dynamics)
3. **Stock-like** - Random walk with trends (noisy, unpredictable)

### 5 Architectures:
1. **Liquid Neural Net** - Continuous-time dynamics (NO backprop)
2. **Forward-Forward** - Hinton's bio-plausible algorithm (NO backprop)
3. **CuriosityCore** - Curiosity-driven learning (NO backprop)
4. **LSTM** - Baseline with backpropagation
5. **Simple AR** - Linear regression baseline

## Results at a Glance

### Mackey-Glass (Chaotic)
- **Winner:** LSTM (MSE: 0.086) ✓ Backprop
- **Bio-plausible best:** Liquid Net (MSE: 0.703) - 8x worse
- **Verdict:** Bio-plausible struggles with deterministic chaos

### Lorenz Attractor (Nonlinear)
- **Winner:** Simple AR (MSE: 0.018) - Linear structure!
- **Bio-plausible best:** Liquid Net (MSE: 0.966) - 55x worse
- **Verdict:** Even linear methods beat complex bio-plausible

### Stock-like (Noisy)
- **Winner:** LSTM (MSE: 0.418) ✓ Backprop
- **Bio-plausible best:** Liquid Net (MSE: 0.421) - **Competitive!**
- **Verdict:** Bio-plausible shines on unpredictable data ✓

## Key Takeaway

**Bio-plausible methods trade performance for biological realism:**
- 8-55x worse on deterministic chaos
- Competitive on noisy/random data
- No backpropagation (local learning only)
- Tiny networks (37 neurons vs larger LSTMs)
- Good for: online learning, edge devices, noisy environments
- Not good for: complex deterministic patterns requiring global optimization

## The Big Question Answered

**Do Liquid Networks "shine" on temporal tasks as claimed?**

**Answer:** ⚠️ **Partially**
- ✓ Continuous-time dynamics work
- ✓ Tiny networks (37 neurons) effective
- ✓ Competitive on noisy data
- ✗ 8x worse than LSTM on chaos
- ✗ Struggle with deterministic patterns

**Their real strength:** Noisy, online, resource-constrained scenarios. NOT a replacement for backprop on complex deterministic time series.

## Files

- **experiments/real_benchmark_timeseries.py** - Full benchmark code
- **REAL_TIMESERIES_BENCHMARK_RESULTS.md** - Detailed analysis
- **TIMESERIES_BENCHMARK_QUICKSTART.md** - This file

## Quick Comparison Table

| Method | MSE (Mackey-Glass) | Backprop? | Speed | Bio-plausible? |
|--------|-------------------|-----------|-------|----------------|
| LSTM | **0.086** ⭐ | YES | Medium | NO |
| Simple AR | 0.138 | NO | **Fast** ⚡ | NO |
| Liquid Net | 0.703 | NO | Fast | **YES** ✓ |
| CuriosityCore | 1.027 | NO | Slow | **YES** ✓ |
| Forward-Forward | 6.407 ❌ | NO | Slowest | **YES** ✓ |

## The Reality Check

This benchmark proves that:

1. **Backprop still dominates** for complex temporal prediction
2. **Bio-plausibility costs performance** (8-55x worse on chaos)
3. **Liquid Networks have niche advantages** (noise, online, efficiency)
4. **Simple methods often underrated** (AR beats complex networks on Lorenz)
5. **Task characteristics matter** (chaos vs noise, deterministic vs random)

## Next Steps

Want to improve bio-plausible methods?
- Add global error signals (hybrid approach)
- Use eligibility traces (better credit assignment)
- Increase ODE integration steps
- Meta-learning for temporal patterns
- Hierarchical time scales

Want to use Liquid Networks in production?
- Best for: Edge devices, online learning, noisy sensors
- Not for: High-accuracy chaos prediction, offline batch training

---

**Bottom line:** We now have a REAL benchmark showing that bio-plausible learning works but has fundamental limitations compared to backpropagation. The gap is substantial (8-55x) on complex tasks but closes on noisy data. Choose your architecture based on your constraints (bio-plausibility, resources, accuracy requirements).
