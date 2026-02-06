# Arithmetic Learning Implementation Checklist

## Project Goal
Fix arithmetic learning in bio-plausible neural networks
**Current:** 0-20% accuracy on modular arithmetic
**Target:** 80-95% accuracy

---

## Phase 1: Quick Wins (Priority 1 - Do First)

### Task 1.1: Implement Fourier Number Encoding
**Estimated time:** 2-3 hours
**Expected improvement:** 20% → 60% accuracy

- [ ] Create `FourierNumberEncoder` class
  - [ ] `__init__(self, prime, n_frequencies=10)`
  - [ ] `encode(self, number)` - convert int to Fourier features
  - [ ] `encode_pair(self, a, b)` - encode two numbers
  - [ ] `decode(self, features)` - convert features back to int

- [ ] Add unit tests
  - [ ] Test encode/decode round-trip (should match for all 0 to p-1)
  - [ ] Test encoding is periodic (encode(0) ≈ encode(prime))
  - [ ] Test feature dimension is 2*n_frequencies

- [ ] Integrate into data generation
  - [ ] Modify `ECMathDataGenerator.generate_modular_addition()`
  - [ ] Update input dimensions (now 2*2*n_frequencies)
  - [ ] Update output dimensions (now 2*n_frequencies)

- [ ] Run baseline test
  - [ ] Train on modular addition, p=97
  - [ ] Compare accuracy: old encoding vs Fourier
  - [ ] Document results

**Success criteria:** >60% accuracy on modular addition (p=97)

---

### Task 1.2: Enable Grokking Optimization
**Estimated time:** 1-2 hours
**Expected improvement:** 60% → 85% accuracy

- [ ] Modify training hyperparameters
  - [ ] Add weight_decay=1.0 to optimizer
  - [ ] Increase epochs from 100 to 5000
  - [ ] Change batch_size to full dataset (not mini-batch)

- [ ] Implement IPR tracking
  - [ ] Add `compute_inverse_participation_ratio()` function
  - [ ] Track IPR every 100 epochs
  - [ ] Plot IPR vs epoch (should spike at grokking)

- [ ] Monitor grokking behavior
  - [ ] Track train accuracy (should hit 100% early)
  - [ ] Track test accuracy (should spike suddenly)
  - [ ] Identify grokking epoch (when IPR jumps)

- [ ] Visualize results
  - [ ] Plot loss over time
  - [ ] Plot accuracy over time (both train and test)
  - [ ] Plot IPR over time
  - [ ] Mark grokking epoch on plots

**Success criteria:**
- Clear grokking observed (sudden test accuracy jump)
- IPR increases sharply at grokking
- Final accuracy >85%

---

### Task 1.3: Implement Curriculum Learning
**Estimated time:** 3-4 hours
**Expected improvement:** More stable training, 85% → 95% accuracy

- [ ] Create `ModularArithmeticCurriculum` class
  - [ ] Generate list of primes from 11 to target
  - [ ] Track current curriculum stage
  - [ ] Implement `should_advance(accuracy)` logic
  - [ ] Implement `advance_stage()` method

- [ ] Modify training pipeline
  - [ ] Loop over curriculum stages
  - [ ] Train on current prime
  - [ ] Evaluate accuracy
  - [ ] Advance when accuracy >80%

- [ ] Add transfer learning
  - [ ] Keep model weights between stages
  - [ ] Use previous learning as initialization
  - [ ] Track improvement from transfer

- [ ] Comprehensive evaluation
  - [ ] Test on all curriculum stages
  - [ ] Compare curriculum vs direct training
  - [ ] Plot accuracy vs prime size

**Success criteria:**
- Successfully train on primes 11 → 97 → 997
- Each stage achieves >80% before advancing
- Final accuracy >90% on p=997

---

## Phase 2: Architecture Improvements (Priority 2)

### Task 2.1: Implement iNALU Module
**Estimated time:** 4-6 hours
**Expected improvement:** 85% → 95% accuracy

- [ ] Create `iNALU` class
  - [ ] NAC component (addition/subtraction)
  - [ ] Multiplicative component (handles negatives)
  - [ ] Input-independent gating
  - [ ] Proper weight initialization

- [ ] Add to architecture options
  - [ ] Create `train_inalu()` function
  - [ ] Add to benchmark comparison
  - [ ] Test on all operations

- [ ] Analyze learned weights
  - [ ] Visualize weight matrices
  - [ ] Check sparsity (should be mostly -1, 0, +1)
  - [ ] Verify interpretability

- [ ] Performance comparison
  - [ ] Compare iNALU vs MLP vs Liquid
  - [ ] Test extrapolation (train on small, test on large)
  - [ ] Document advantages/limitations

**Success criteria:**
- iNALU achieves >95% on modular addition
- Learned weights are sparse and interpretable
- Outperforms standard MLP by >30%

---

### Task 2.2: Hybrid Bio-Plausible + NALU Architecture
**Estimated time:** 6-8 hours
**Expected improvement:** Bio-plausible viable (50-70% accuracy)

- [ ] Design hybrid architecture
  - [ ] Bio-plausible layers for feature learning
  - [ ] NALU module for arithmetic reasoning
  - [ ] Clear separation of unsupervised/supervised

- [ ] Implement training procedure
  - [ ] Phase 1: Train bio layers (unsupervised)
  - [ ] Phase 2: Train NALU (supervised, bio frozen)
  - [ ] Optional: Fine-tune end-to-end

- [ ] Test bio-plausible options
  - [ ] Forward-Forward + NALU
  - [ ] Hebbian + NALU
  - [ ] Curiosity-driven + NALU

- [ ] Comprehensive evaluation
  - [ ] Compare hybrid vs pure bio-plausible
  - [ ] Compare hybrid vs pure supervised
  - [ ] Analyze learned representations

**Success criteria:**
- Hybrid achieves >50% accuracy (vs 0-2% pure bio)
- Bio-plausible layers learn useful features
- Clear improvement over either component alone

---

### Task 2.3: Position Coupling for Multi-Digit Operations
**Estimated time:** 5-7 hours
**Expected improvement:** Enable length generalization

- [ ] Implement position-coupled embeddings
  - [ ] Digit significance instead of sequence position
  - [ ] e.g., "123" → positions [2, 1, 0] not [0, 1, 2]

- [ ] Create multi-digit dataset
  - [ ] Generate 2-digit addition problems
  - [ ] Generate 3-digit addition problems
  - [ ] Generate 5-digit addition problems

- [ ] Test length generalization
  - [ ] Train on 2-digit
  - [ ] Test on 3, 4, 5-digit
  - [ ] Measure accuracy vs length
  - [ ] Compare with standard positional encoding

**Success criteria:**
- Train on 2-digit, achieve >80% on 5-digit
- Position coupling outperforms standard encoding
- Clear length extrapolation demonstrated

---

## Phase 3: Advanced Techniques (Priority 3)

### Task 3.1: Chain-of-Thought for Complex Operations
**Estimated time:** 8-10 hours
**Expected improvement:** Complex operations become learnable

- [ ] Implement CoT data generation
  - [ ] Modular inverse with step-by-step solution
  - [ ] Modular exponentiation with steps
  - [ ] Point validation with reasoning

- [ ] Train sequence model
  - [ ] Input: operation description
  - [ ] Output: step-by-step solution + answer
  - [ ] Loss on both intermediate steps and final answer

- [ ] Evaluate on complex operations
  - [ ] Modular inverse accuracy
  - [ ] Modular exponentiation accuracy
  - [ ] Compare with/without CoT

**Success criteria:**
- Modular inverse: >40% accuracy (vs <5% baseline)
- Steps are interpretable
- Clear improvement from CoT

---

### Task 3.2: Reverse Digit Order Processing
**Estimated time:** 3-4 hours
**Expected improvement:** 20-30% on carry-heavy operations

- [ ] Implement little-endian encoding
  - [ ] Reverse number strings before encoding
  - [ ] Process least-significant first

- [ ] Test on multi-digit addition
  - [ ] Compare big-endian vs little-endian
  - [ ] Measure accuracy on problems with carries

- [ ] Analyze carry propagation
  - [ ] Visualize attention patterns
  - [ ] Verify right-to-left processing learned

**Success criteria:**
- Little-endian outperforms big-endian by >20%
- Clear carry propagation pattern observed

---

### Task 3.3: Comprehensive Benchmark Suite
**Estimated time:** 4-6 hours

- [ ] Run all techniques on all operations
  - [ ] 6 operations (add, sub, mul, inv, exp, point)
  - [ ] 3 primes (97, 997, 7919)
  - [ ] 5 architectures (MLP, Liquid, iNALU, Bio, Hybrid)

- [ ] Generate comparison visualizations
  - [ ] Accuracy heatmap (operation × architecture)
  - [ ] Scaling curves (accuracy vs prime size)
  - [ ] Training dynamics (loss/accuracy over time)

- [ ] Statistical analysis
  - [ ] Significance tests
  - [ ] Error analysis
  - [ ] Failure mode identification

- [ ] Write results document
  - [ ] Summary statistics
  - [ ] Key findings
  - [ ] Recommendations

**Success criteria:**
- Complete benchmark data for all combinations
- Clear visualizations
- Documented insights

---

## Phase 4: Bio-Plausible Deep Dive (Research Phase)

### Task 4.1: Fourier-Based Hebbian Learning
**Estimated time:** 10-15 hours

- [ ] Implement Fourier Hebbian rule
  - [ ] Weight updates based on phase alignment
  - [ ] Encourage periodic weight patterns

- [ ] Test emergence of periodicity
  - [ ] Train without supervision
  - [ ] Measure IPR over time
  - [ ] Compare with supervised grokking

**Success criteria:**
- Unsupervised emergence of Fourier features
- Clear periodicity in learned weights

---

### Task 4.2: Neo-Hebbian (R-STDP) for Arithmetic
**Estimated time:** 10-15 hours

- [ ] Implement reward-modulated STDP
  - [ ] Accuracy as reward signal
  - [ ] Spike-timing dependent plasticity
  - [ ] Three-factor learning rule

- [ ] Test on modular operations
  - [ ] Compare with supervised learning
  - [ ] Analyze sample efficiency

**Success criteria:**
- R-STDP achieves >30% accuracy
- More bio-plausible than backprop

---

### Task 4.3: Forward-Forward with Arithmetic Bias
**Estimated time:** 8-12 hours

- [ ] Modify Forward-Forward for arithmetic
  - [ ] Add NALU-like constraints
  - [ ] Positive/negative discrimination
  - [ ] Test on arithmetic tasks

**Success criteria:**
- FF with bias outperforms vanilla FF by >20%

---

## Validation Checklist

Before considering each phase complete:

### Phase 1 Validation
- [ ] Modular addition (p=97) achieves >80% accuracy
- [ ] Grokking clearly observed (IPR spike + accuracy jump)
- [ ] Curriculum learning works (successful progression)
- [ ] Results documented with plots

### Phase 2 Validation
- [ ] iNALU achieves >95% accuracy
- [ ] Hybrid bio-plausible achieves >50% accuracy
- [ ] Position coupling enables length generalization
- [ ] All architectures compared fairly

### Phase 3 Validation
- [ ] CoT improves complex operations by >30%
- [ ] Reverse order processing validated
- [ ] Complete benchmark data collected
- [ ] Results paper draft written

### Phase 4 Validation
- [ ] At least one bio-plausible technique achieves >30%
- [ ] Clear comparison with supervised methods
- [ ] Scientific insights documented
- [ ] Research contribution identified

---

## Testing Protocol

For each implementation:

1. **Unit tests:**
   - [ ] Test individual functions
   - [ ] Verify input/output shapes
   - [ ] Check edge cases

2. **Integration tests:**
   - [ ] Test full training pipeline
   - [ ] Verify end-to-end accuracy
   - [ ] Check for NaN/inf issues

3. **Comparison tests:**
   - [ ] Compare with baseline
   - [ ] Measure improvement
   - [ ] Statistical significance

4. **Documentation:**
   - [ ] Code comments
   - [ ] Docstrings
   - [ ] Usage examples
   - [ ] Results summary

---

## Success Metrics

### Minimum Viable Success (Phase 1)
- Modular addition (p=97): **>80% accuracy**
- Training time: **<10 minutes**
- Clear grokking behavior observed
- Code is clean and documented

### Good Success (Phase 1+2)
- Modular addition (p=997): **>60% accuracy**
- iNALU achieves: **>95% accuracy**
- Hybrid bio-plausible: **>50% accuracy**
- Multiple operations working

### Excellent Success (Phase 1+2+3)
- All 6 operations: **>40% average accuracy**
- Length generalization working
- Complete benchmark comparison
- Research paper quality results

### Research Contribution (Phase 1+2+3+4)
- Bio-plausible learning: **>30% accuracy**
- Novel insights into neural arithmetic
- Publishable results
- Code released open-source

---

## Timeline Estimate

### Aggressive (Full-time, 1 week)
- Day 1: Phase 1 (Tasks 1.1, 1.2, 1.3)
- Day 2-3: Phase 2 (Tasks 2.1, 2.2)
- Day 4-5: Phase 3 (Tasks 3.1, 3.2, 3.3)
- Day 6-7: Phase 4 (Start research)

### Moderate (Part-time, 1 month)
- Week 1: Phase 1
- Week 2-3: Phase 2
- Week 4: Phase 3
- Beyond: Phase 4 (research phase)

### Realistic (Side project, 2-3 months)
- Month 1: Phase 1
- Month 2: Phase 2
- Month 3: Phase 3
- Ongoing: Phase 4

---

## Resources Needed

### Computational
- [ ] CPU: Sufficient for p=97, 997
- [ ] GPU: Recommended for p=7919+
- [ ] RAM: 8GB minimum, 16GB recommended
- [ ] Storage: ~5GB for datasets and results

### Code Dependencies
- [ ] PyTorch >= 1.10
- [ ] NumPy >= 1.21
- [ ] Matplotlib >= 3.4
- [ ] Seaborn (for visualizations)

### References
- [ ] ARITHMETIC_LEARNING_RESEARCH_REPORT.md (full research)
- [ ] ARITHMETIC_FIX_QUICKSTART.md (quick start)
- [ ] 50+ cited papers (in research report)

---

## Daily Progress Tracking

Use this template for daily updates:

```
Date: YYYY-MM-DD
Tasks completed:
- [ ] Task X.X: Description
- [ ] Task X.X: Description

Results:
- Accuracy achieved: XX%
- Time spent: X hours
- Issues encountered: ...

Next steps:
- [ ] Continue with Task X.X
- [ ] Debug issue Y
- [ ] Run experiment Z

Notes:
...
```

---

## Final Deliverables

When all phases complete:

- [ ] Working code (clean, documented, tested)
- [ ] Comprehensive results (JSON + visualizations)
- [ ] Research report (findings + insights)
- [ ] README with usage instructions
- [ ] Examples demonstrating all techniques
- [ ] Comparison with baseline (before/after)

---

## Questions to Answer

Track answers as you progress:

1. **What is the minimum accuracy improvement from Fourier encoding?**
   - Baseline: ____%
   - With Fourier: ____%
   - Improvement: ____x

2. **At what epoch does grokking typically occur?**
   - Observed range: epoch ____ to ____
   - Average: epoch ____

3. **How does accuracy scale with prime size?**
   - p=97: ____%
   - p=997: ____%
   - p=7919: ____%

4. **What is the best bio-plausible architecture?**
   - Pure bio: ____% accuracy
   - Hybrid: ____% accuracy
   - Best approach: ____________

5. **Which operations are learnable vs impossible?**
   - Modular addition: ____% ✓/✗
   - Modular subtraction: ____% ✓/✗
   - Modular multiplication: ____% ✓/✗
   - Modular inverse: ____% ✓/✗
   - Modular exponentiation: ____% ✓/✗
   - Point validation: ____% ✓/✗

---

**Start with Phase 1, Task 1.1. Everything else builds on that foundation.**

Good luck! 🚀
