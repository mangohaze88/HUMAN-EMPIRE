#!/usr/bin/env python3
"""Quick test for Stigmergic Network with Global Feedback"""

import numpy as np
import torch
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks.stigmergic_intelligence import StigmergicNetworkGPU

def test_stigmergic(n_steps: int = 5000):
    print("=" * 60)
    print("STIGMERGIC NETWORK - Global Feedback Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    net = StigmergicNetworkGPU(
        n_agents=1024, env_shape=(64, 64),
        input_dim=64, output_dim=32
    )

    print(f"Config: {net.n_agents} agents, {net.n_pheromones} pheromone channels")
    print(f"Global feedback channels: error={net.ch_global_error}, success={net.ch_global_success}")
    print()

    task_errors = []
    agent_errors = []
    competences = []

    start = time.time()

    for i in range(n_steps):
        x = torch.randn(64, device='cuda')
        output, info = net.forward(x, n_steps=5)

        task_errors.append(info['task_error'])
        agent_errors.append(info['collective_error'])
        competences.append(info['mean_competence'])

        if i % 500 == 0:
            # Calculate recent averages
            recent_task = np.mean(task_errors[-100:]) if len(task_errors) >= 100 else np.mean(task_errors)
            recent_agent = np.mean(agent_errors[-100:]) if len(agent_errors) >= 100 else np.mean(agent_errors)

            print(f"Step {i:5d}: task_err={recent_task:.4f}, agent_err={recent_agent:.4f}, "
                  f"competence={info['mean_competence']:.4f}")

    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s ({n_steps/elapsed:.1f} steps/sec)")
    print()

    # Show progression in windows
    windows = 5
    window_size = n_steps // windows
    print("Task error progression:")
    for w in range(windows):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size
        avg = np.mean(task_errors[start_idx:end_idx])
        print(f"  Steps {start_idx:5d}-{end_idx:5d}: {avg:.4f}")

    print()
    print(f"Initial avg (first 200): {np.mean(task_errors[:200]):.4f}")
    print(f"Final avg (last 200):    {np.mean(task_errors[-200:]):.4f}")
    print(f"Improvement: {(1 - np.mean(task_errors[-200:])/np.mean(task_errors[:200]))*100:.1f}%")

    return {
        'task_errors': task_errors,
        'agent_errors': agent_errors,
        'final_task_error': np.mean(task_errors[-200:]),
        'improvement': (1 - np.mean(task_errors[-200:])/np.mean(task_errors[:200]))*100
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()

    results = test_stigmergic(args.steps)
