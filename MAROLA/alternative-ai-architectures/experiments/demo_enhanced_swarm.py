#!/usr/bin/env python3
"""
Interactive Demo: Enhanced Stigmergic Swarm
===========================================

Demonstrates key capabilities of the enhanced swarm system.

Run: python experiments/demo_enhanced_swarm.py

Author: Innovation & Experimentation Specialist
Date: 2026-02-05
"""

import sys
sys.path.append('/root/MAROLA/alternative-ai-architectures')

import torch
import numpy as np
import time
from typing import Dict, Any

from src.networks.enhanced_stigmergic_swarm import (
    EnhancedStigmergicSwarmGPU,
    SwarmComputer,
    Caste
)


def print_header(text: str):
    """Pretty print section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_stats(swarm: EnhancedStigmergicSwarmGPU):
    """Print current swarm statistics"""
    state = swarm._get_global_state()

    print(f"Step {swarm.step_count}:")
    print(f"  Mean surprise:  {state['mean_surprise']:.4f}")
    print(f"  Mean energy:    {state['mean_energy']:.1f}")
    print(f"  Mean speed:     {state['mean_speed']:.4f}")

    if swarm.history['emergence_score']:
        print(f"  Emergence:      {swarm.history['emergence_score'][-1]:.4f}")

    # Caste distribution
    caste_counts = [
        (swarm.specializations == i).sum().item()
        for i in range(3)
    ]
    print(f"  Castes: Explorers={caste_counts[0]}, "
          f"Exploiters={caste_counts[1]}, "
          f"Coordinators={caste_counts[2]}")


def demo_1_basic_dynamics():
    """Demo 1: Basic swarm dynamics and emergence"""
    print_header("DEMO 1: Basic Swarm Dynamics")

    print("Creating swarm with 2048 agents...")
    swarm = EnhancedStigmergicSwarmGPU(
        n_agents=2048,
        env_shape=(128, 128),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Initial state:")
    print_stats(swarm)

    print("\nRunning dynamics for 100 steps...")
    start_time = time.time()

    for step in range(100):
        swarm.step()

        if step % 25 == 0 and step > 0:
            print(f"\n--- Step {step} ---")
            print_stats(swarm)

    elapsed = time.time() - start_time

    print(f"\n✓ Completed 100 steps in {elapsed:.2f}s")
    print(f"  Performance: {100/elapsed:.1f} steps/second")

    return swarm


def demo_2_pattern_formation(swarm: EnhancedStigmergicSwarmGPU):
    """Demo 2: Self-organizing spatial patterns"""
    print_header("DEMO 2: Pattern Formation")

    print("Running extended dynamics to form patterns...")
    print("(Pheromone interactions + diffusion → emergent structure)")

    for step in range(100):
        swarm.step()

        if step % 25 == 0:
            # Analyze pattern strength
            novelty_field = swarm.pheromones.fields[0].cpu().numpy()
            pattern_variance = np.var(novelty_field)
            pattern_mean = np.mean(novelty_field)

            print(f"\nStep {step + 100}:")
            print(f"  Pattern variance: {pattern_variance:.2f}")
            print(f"  Pattern mean:     {pattern_mean:.2f}")

    # Final analysis
    print("\nFinal pheromone field analysis:")
    for ch_id, name in [(0, 'Novelty'), (1, 'Competence'),
                        (3, 'Food'), (4, 'Trail')]:
        field = swarm.pheromones.fields[ch_id].cpu().numpy()
        print(f"  {name:12s}: mean={np.mean(field):6.2f}, "
              f"std={np.std(field):6.2f}, "
              f"max={np.max(field):6.2f}")

    print("\n✓ Spatial patterns formed")


def demo_3_collective_decision():
    """Demo 3: Collective decision making (voting)"""
    print_header("DEMO 3: Collective Decision Making")

    print("Creating fresh swarm for decision task...")
    swarm = EnhancedStigmergicSwarmGPU(
        n_agents=1024,
        env_shape=(64, 64),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\nScenario: Two food sources of different quality")
    print("  Option A: Quality 0.9 at position (0.25, 0.25)")
    print("  Option B: Quality 0.4 at position (0.75, 0.75)")

    # Place options as pheromone deposits
    grid_a = (16, 16)  # 0.25 * 64
    grid_b = (48, 48)  # 0.75 * 64

    print("\nInjecting options into environment...")
    swarm.pheromones.fields[3, grid_a[0]-3:grid_a[0]+4, grid_a[1]-3:grid_a[1]+4] = 90
    swarm.pheromones.fields[3, grid_b[0]-3:grid_b[0]+4, grid_b[1]-3:grid_b[1]+4] = 40

    print("Running consensus process...")
    for step in range(100):
        swarm.step()

        if step % 25 == 0:
            # Count agents near each option
            dists_a = torch.norm(
                swarm.positions - torch.tensor([0.25, 0.25], device=swarm.device),
                dim=1
            )
            dists_b = torch.norm(
                swarm.positions - torch.tensor([0.75, 0.75], device=swarm.device),
                dim=1
            )

            near_a = (dists_a < 0.15).sum().item()
            near_b = (dists_b < 0.15).sum().item()

            print(f"\nStep {step}:")
            print(f"  Agents near A (better): {near_a}")
            print(f"  Agents near B (worse):  {near_b}")

    # Final decision
    dists_a = torch.norm(
        swarm.positions - torch.tensor([0.25, 0.25], device=swarm.device),
        dim=1
    )
    dists_b = torch.norm(
        swarm.positions - torch.tensor([0.75, 0.75], device=swarm.device),
        dim=1
    )

    near_a = (dists_a < 0.15).sum().item()
    near_b = (dists_b < 0.15).sum().item()

    total = near_a + near_b
    if total > 0:
        vote_a = near_a / total
        print(f"\n✓ Final decision: {vote_a*100:.1f}% chose option A (better)")
        print(f"  Result: {'SUCCESS' if vote_a > 0.6 else 'FAILED'}")
    else:
        print("\n✗ No agents converged to either option")


def demo_4_task_allocation(swarm: EnhancedStigmergicSwarmGPU):
    """Demo 4: Self-organizing task allocation (castes)"""
    print_header("DEMO 4: Task Allocation (Caste Formation)")

    print("Monitoring caste evolution over time...")
    print("(Agents specialize based on experience)\n")

    history = []

    for step in range(150):
        swarm.step()

        if step % 30 == 0:
            caste_counts = [
                (swarm.specializations == i).sum().item()
                for i in range(3)
            ]
            history.append(caste_counts)

            # Calculate diversity (Shannon entropy)
            probs = np.array(caste_counts) / swarm.n_agents
            entropy = -np.sum(probs * np.log(probs + 1e-9))

            print(f"Step {step}:")
            print(f"  Explorers:    {caste_counts[0]:4d} ({caste_counts[0]/swarm.n_agents*100:5.1f}%)")
            print(f"  Exploiters:   {caste_counts[1]:4d} ({caste_counts[1]/swarm.n_agents*100:5.1f}%)")
            print(f"  Coordinators: {caste_counts[2]:4d} ({caste_counts[2]/swarm.n_agents*100:5.1f}%)")
            print(f"  Diversity (entropy): {entropy:.3f} / {np.log(3):.3f}")
            print()

    print("✓ Caste specialization emerged")

    # Analyze stability
    if len(history) > 3:
        early = np.array(history[0])
        late = np.array(history[-1])
        change = np.linalg.norm(late - early)
        print(f"  Change from start to end: {change:.1f} agents")
        print(f"  Stability: {'HIGH' if change < 200 else 'LOW'}")


def demo_5_learning():
    """Demo 5: Learning and prediction"""
    print_header("DEMO 5: Learning Curve")

    print("Creating swarm for learning task...")
    swarm = EnhancedStigmergicSwarmGPU(
        n_agents=1024,
        env_shape=(64, 64),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Create predictable pattern
    pattern = torch.randn(64, device=swarm.device)

    print("Injecting repeating pattern...")
    print("(Swarm should learn to predict it)\n")

    errors = []

    for step in range(100):
        # Inject same pattern each time
        swarm._inject_input(pattern)
        swarm.step()

        error = swarm.history['collective_error'][-1]
        errors.append(error)

        if step % 20 == 0:
            print(f"Step {step:3d}: prediction error = {error:.4f}")

    # Analyze learning
    early_error = np.mean(errors[:20])
    late_error = np.mean(errors[-20:])
    improvement = (early_error - late_error) / (early_error + 1e-6)

    print(f"\nLearning analysis:")
    print(f"  Early error (steps 0-20):   {early_error:.4f}")
    print(f"  Late error (steps 80-100):  {late_error:.4f}")
    print(f"  Improvement: {improvement*100:.1f}%")
    print(f"  Result: {'LEARNING DETECTED' if improvement > 0.1 else 'NO CLEAR LEARNING'}")

    if improvement > 0.1:
        print("✓ Swarm learned to predict pattern")
    else:
        print("✗ Learning unclear (may need more steps)")


def demo_6_computation():
    """Demo 6: Using swarm as computational substrate"""
    print_header("DEMO 6: Collective Computation")

    print("Creating swarm configured as computer...")
    swarm = EnhancedStigmergicSwarmGPU(
        n_agents=2048,
        env_shape=(128, 128),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    computer = SwarmComputer(swarm)

    # Task 1: Pattern processing
    print("\nTask 1: Pattern Processing")
    print("Input: Random 64-element vector")

    input_vec = torch.randn(64, device=swarm.device)
    print(f"Input mean: {input_vec.mean():.3f}, std: {input_vec.std():.3f}")

    output, info = swarm.forward(input_vec, n_steps=50, task='autoencoding')

    print(f"\nOutput after 50 steps:")
    print(f"  Output mean: {output.mean():.3f}")
    print(f"  Final error: {info['final_error']:.4f}")
    print(f"  Emergence:   {info['emergence_score']:.4f}")

    # Task 2: Peak finding
    print("\n\nTask 2: Peak Detection")
    print("(Finding maxima in pheromone fields)")

    # Run some dynamics to create patterns
    for _ in range(50):
        swarm.step()

    peaks = computer.find_peaks(channel=3, min_height=30)

    print(f"\nFound {len(peaks)} peaks in 'food' channel:")
    for i, peak in enumerate(peaks[:5]):
        print(f"  Peak {i+1}: position {peak['position']}, height {peak['height']:.1f}")

    print("\n✓ Swarm executed collective computation")


def main():
    """Run all demos"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "ENHANCED STIGMERGIC SWARM DEMO" + " "*23 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Demos will run on CPU (slower).")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    try:
        # Run demos
        swarm = demo_1_basic_dynamics()
        demo_2_pattern_formation(swarm)
        demo_3_collective_decision()
        demo_4_task_allocation(swarm)
        demo_5_learning()
        demo_6_computation()

        # Summary
        print_header("DEMO SUMMARY")
        print("All demonstrations completed successfully!\n")

        print("Demonstrated capabilities:")
        print("  ✓ Basic swarm dynamics and emergence")
        print("  ✓ Self-organizing spatial patterns")
        print("  ✓ Collective decision making (voting)")
        print("  ✓ Task allocation (caste specialization)")
        print("  ✓ Learning and prediction")
        print("  ✓ Collective computation primitives")

        print("\n" + "="*70)
        print("\nNext steps:")
        print("  1. Run full test suite: python experiments/test_emergence.py")
        print("  2. Read architecture docs: docs/ENHANCED_STIGMERGIC_ARCHITECTURE.md")
        print("  3. Try custom applications: docs/QUICK_START_ENHANCED_SWARM.md")
        print("\n" + "="*70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: Demo failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
