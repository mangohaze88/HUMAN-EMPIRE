#!/usr/bin/env python3
"""
Compare All Alternative AI Architectures
=========================================
Run all five architectures on the same task and compare:
- Learning speed
- Final performance
- Energy efficiency (compute time)
- Memory usage
- Emergent behaviors
"""

import numpy as np
import torch
import time
import sys
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks import (
    ThermodynamicNetwork, ThermodynamicNetworkGPU,
    HolographicCompositionalNetwork, HolographicNetworkGPU,
    StigmergicIntelligenceNetwork, StigmergicNetworkGPU,
    MetabolicNetwork, MetabolicNetworkGPU,
    CuriosityCore, CuriosityCoreGPU,
    LiquidNeuralNetwork, LiquidNeuralNetworkGPU
)
from environments import CuriosityWorld, PredictionWorld


@dataclass
class ExperimentResults:
    """Results from running one architecture"""
    name: str
    learning_curve: List[float] = field(default_factory=list)
    final_error: float = 0.0
    total_time: float = 0.0
    steps_per_second: float = 0.0
    memory_mb: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


def run_thermodynamic(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test Thermodynamic Network"""
    print("\n=== THERMODYNAMIC NETWORK ===")
    results = ExperimentResults(name="Thermodynamic")

    if use_gpu and torch.cuda.is_available():
        net = ThermodynamicNetworkGPU(input_dim=64, hidden_dims=[128, 64], output_dim=32)
        device = 'cuda'
    else:
        net = ThermodynamicNetwork(input_dim=64, hidden_dims=[64, 32], output_dim=16)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    errors = []

    for step in range(n_steps):
        obs = world.step()
        target = world.get_ground_truth_next()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            output, info = net.forward(obs_t, relaxation_steps=20)
            error = torch.mean((output[:32] - torch.tensor(target[:32], device='cuda')) ** 2).item()
        else:
            output, info = net.forward(obs, relaxation_steps=20)
            error = np.mean((output - target[:len(output)]) ** 2)

        errors.append(error)

        if step % 200 == 0:
            print(f"  Step {step}: error={error:.4f}, energy={info.get('final_energy', 0):.2f}")

    results.total_time = time.time() - start_time
    results.learning_curve = errors
    results.final_error = np.mean(errors[-100:])
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {'final_energy': info.get('final_energy', 0)}

    print(f"  Final: error={results.final_error:.4f}, {results.steps_per_second:.1f} steps/sec")
    return results


def run_holographic(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test Holographic Network"""
    print("\n=== HOLOGRAPHIC NETWORK ===")
    results = ExperimentResults(name="Holographic")

    if use_gpu and torch.cuda.is_available():
        net = HolographicNetworkGPU(input_dim=64, holo_dim=8192, n_layers=3)
        device = 'cuda'
    else:
        net = HolographicCompositionalNetwork(input_dim=64, holo_dim=2048, n_layers=3)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    errors = []

    for step in range(n_steps):
        obs = world.step()
        target = world.get_ground_truth_next()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            output, info = net.forward(obs_t)
            error = torch.mean((output[:64] - torch.tensor(target[:64], device='cuda')) ** 2).item()
        else:
            output, info = net.forward(obs)
            error = np.mean((output[:len(target)] - target[:len(output)]) ** 2)

        errors.append(info.get('prediction_error', error))

        if step % 200 == 0:
            print(f"  Step {step}: error={errors[-1]:.4f}, capacity={info.get('capacity_used', 0):.4f}")

    results.total_time = time.time() - start_time
    results.learning_curve = errors
    results.final_error = np.mean(errors[-100:])
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {'capacity_used': info.get('capacity_used', 0)}

    print(f"  Final: error={results.final_error:.4f}, {results.steps_per_second:.1f} steps/sec")
    return results


def run_stigmergic(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test Stigmergic Intelligence with Global Feedback"""
    print("\n=== STIGMERGIC INTELLIGENCE (with Global Feedback) ===")
    results = ExperimentResults(name="Stigmergic")

    if use_gpu and torch.cuda.is_available():
        # Use new parameters for global feedback version
        net = StigmergicNetworkGPU(
            n_agents=1024, env_shape=(64, 64),
            input_dim=64, output_dim=32
        )
        device = 'cuda'
    else:
        net = StigmergicIntelligenceNetwork(n_agents=64, env_shape=(32, 32), input_dim=64, output_dim=32)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    task_errors = []  # Track actual task error, not just agent error
    agent_errors = []

    for step in range(n_steps):
        obs = world.step()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            output, info = net.forward(obs_t, n_steps=5)
        else:
            output, info = net.forward(obs, n_steps=5)

        # Track both task error and agent error
        task_error = info.get('task_error', info['collective_error'])
        task_errors.append(task_error)
        agent_errors.append(info['collective_error'])

        if step % 200 == 0:
            competence = info.get('mean_competence', info.get('collective_competence', 0))
            print(f"  Step {step}: task_error={task_error:.4f}, agent_error={info['collective_error']:.4f}, "
                  f"competence={competence:.4f}")

    results.total_time = time.time() - start_time
    results.learning_curve = task_errors  # Use task error for learning curve
    results.final_error = np.mean(task_errors[-100:])  # Use task error for final metric
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {
        'collective_competence': info.get('collective_competence', 0),
        'final_agent_error': np.mean(agent_errors[-100:]),
        'final_task_error': results.final_error
    }

    print(f"  Final: task_error={results.final_error:.4f}, agent_error={np.mean(agent_errors[-100:]):.4f}, "
          f"{results.steps_per_second:.1f} steps/sec")
    return results


def run_metabolic(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test Metabolic Network"""
    print("\n=== METABOLIC NETWORK ===")
    results = ExperimentResults(name="Metabolic")

    if use_gpu and torch.cuda.is_available():
        net = MetabolicNetworkGPU(input_dim=64, hidden_dims=[256, 128], output_dim=64)
        device = 'cuda'
    else:
        net = MetabolicNetwork(input_dim=64, hidden_dims=[64, 32], output_dim=32)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    errors = []

    for step in range(n_steps):
        obs = world.step()
        target = world.get_ground_truth_next()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            target_t = torch.tensor(target, dtype=torch.float32, device='cuda')
            output, info = net.forward(obs_t, target=target_t)
        else:
            output, info = net.forward(obs, target=target[:32])

        errors.append(info['prediction_error'])

        if step % 200 == 0:
            alive = info.get('alive_counts', [info.get('total_alive', 0)])
            sparsity = info.get('mean_sparsity', 0)
            print(f"  Step {step}: error={errors[-1]:.4f}, alive={alive}, sparsity={sparsity:.2f}")

    results.total_time = time.time() - start_time
    results.learning_curve = errors
    results.final_error = np.mean(errors[-100:])
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {
        'total_alive': info.get('total_alive', sum(info.get('alive_counts', []))),
        'mean_sparsity': info.get('mean_sparsity', 0)
    }

    print(f"  Final: error={results.final_error:.4f}, {results.steps_per_second:.1f} steps/sec")
    return results


def run_curiosity_core(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test CuriosityCore"""
    print("\n=== CURIOSITY CORE ===")
    results = ExperimentResults(name="CuriosityCore")

    if use_gpu and torch.cuda.is_available():
        net = CuriosityCoreGPU(sensory_dim=64, hidden_dim=64, action_dim=32)
        device = 'cuda'
    else:
        net = CuriosityCore(sensory_dim=64, hidden_dim=32, action_dim=16)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    errors = []
    curiosities = []

    for step in range(n_steps):
        obs = world.step()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            info = net.step(obs_t)
        else:
            info = net.step(obs)

        errors.append(info['world_error'])
        curiosities.append(info['curiosity'])

        if step % 200 == 0:
            print(f"  Step {step}: world_error={info['world_error']:.4f}, "
                  f"curiosity={info['curiosity']:.4f}, energy={info['energy']:.1f}")

    results.total_time = time.time() - start_time
    results.learning_curve = errors
    results.final_error = np.mean(errors[-100:])
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {
        'final_curiosity': np.mean(curiosities[-100:]),
        'final_energy': info['energy']
    }

    # Get self-knowledge if available
    if hasattr(net, 'get_self_knowledge'):
        results.extra_metrics['self_knowledge'] = net.get_self_knowledge()

    print(f"  Final: error={results.final_error:.4f}, {results.steps_per_second:.1f} steps/sec")
    return results


def run_liquid(n_steps: int = 1000, use_gpu: bool = True) -> ExperimentResults:
    """Test Liquid Neural Network"""
    print("\n=== LIQUID NEURAL NETWORK ===")
    results = ExperimentResults(name="Liquid")

    if use_gpu and torch.cuda.is_available():
        net = LiquidNeuralNetworkGPU(input_dim=64, output_dim=32, dt=0.1, ode_steps=5, learning_rate=0.01)
        device = 'cuda'
    else:
        net = LiquidNeuralNetwork(input_dim=64, output_dim=32, dt=0.1, ode_steps=5)
        device = 'cpu'

    world = PredictionWorld(dim=64, complexity=3)

    start_time = time.time()
    errors = []

    for step in range(n_steps):
        obs = world.step()
        target = world.get_ground_truth_next()

        if device == 'cuda':
            obs_t = torch.tensor(obs, dtype=torch.float32, device='cuda')
            target_t = torch.tensor(target[:32], dtype=torch.float32, device='cuda')
            output, info = net.forward(obs_t, target=target_t)
            error = torch.mean((output - target_t) ** 2).item()
        else:
            output, info = net.forward(obs)
            error = np.mean((output - target[:len(output)]) ** 2)

        errors.append(error)

        if step % 200 == 0:
            tau = info.get('mean_time_constant', info.get('mean_tau', 0))
            print(f"  Step {step}: error={error:.4f}, tau={tau:.3f}")

    results.total_time = time.time() - start_time
    results.learning_curve = errors
    results.final_error = np.mean(errors[-100:])
    results.steps_per_second = n_steps / results.total_time
    results.extra_metrics = {
        'mean_time_constant': info.get('mean_time_constant', info.get('mean_tau', 0)),
    }

    print(f"  Final: error={results.final_error:.4f}, {results.steps_per_second:.1f} steps/sec")
    return results


def run_all_experiments(n_steps: int = 1000, use_gpu: bool = True) -> Dict[str, ExperimentResults]:
    """Run all architectures and compare"""
    print("=" * 60)
    print("COMPARING ALL ALTERNATIVE AI ARCHITECTURES")
    print("=" * 60)

    device_str = "GPU (CUDA)" if use_gpu and torch.cuda.is_available() else "CPU"
    print(f"Device: {device_str}")
    print(f"Steps: {n_steps}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    # Run each architecture
    results['thermodynamic'] = run_thermodynamic(n_steps, use_gpu)
    results['holographic'] = run_holographic(n_steps, use_gpu)
    results['stigmergic'] = run_stigmergic(n_steps, use_gpu)
    results['metabolic'] = run_metabolic(n_steps, use_gpu)
    results['curiosity_core'] = run_curiosity_core(n_steps, use_gpu)
    results['liquid'] = run_liquid(n_steps, use_gpu)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<20} {'Final Error':<15} {'Steps/sec':<15} {'Time (s)':<10}")
    print("-" * 60)

    for name, r in sorted(results.items(), key=lambda x: x[1].final_error):
        print(f"{r.name:<20} {r.final_error:<15.4f} {r.steps_per_second:<15.1f} {r.total_time:<10.2f}")

    # Winner
    best = min(results.values(), key=lambda r: r.final_error)
    fastest = max(results.values(), key=lambda r: r.steps_per_second)

    print("\n" + "-" * 60)
    print(f"Best accuracy: {best.name} (error={best.final_error:.4f})")
    print(f"Fastest: {fastest.name} ({fastest.steps_per_second:.1f} steps/sec)")

    return results


def plot_results(results: Dict[str, ExperimentResults], save_path: str = None):
    """Plot learning curves (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Learning curves
        ax1 = axes[0, 0]
        for name, r in results.items():
            # Smooth learning curve
            window = 50
            smoothed = np.convolve(r.learning_curve, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label=r.name)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Prediction Error')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.set_yscale('log')

        # Final errors bar chart
        ax2 = axes[0, 1]
        names = [r.name for r in results.values()]
        final_errors = [r.final_error for r in results.values()]
        bars = ax2.bar(names, final_errors)
        ax2.set_ylabel('Final Error')
        ax2.set_title('Final Prediction Error')
        ax2.tick_params(axis='x', rotation=45)

        # Speed comparison
        ax3 = axes[1, 0]
        speeds = [r.steps_per_second for r in results.values()]
        ax3.bar(names, speeds, color='green')
        ax3.set_ylabel('Steps/Second')
        ax3.set_title('Computational Speed')
        ax3.tick_params(axis='x', rotation=45)

        # Extra metrics radar (if available)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Extra Metrics\n(see console output)',
                ha='center', va='center', fontsize=12)
        ax4.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()

    except ImportError:
        print("\nNote: Install matplotlib for visualization (pip install matplotlib)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare Alternative AI Architectures')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps per experiment')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--plot', type=str, default=None, help='Save plot to file')

    args = parser.parse_args()

    results = run_all_experiments(
        n_steps=args.steps,
        use_gpu=not args.cpu
    )

    if args.plot:
        plot_results(results, args.plot)

    # Save results to JSON
    results_dict = {
        name: {
            'name': r.name,
            'final_error': r.final_error,
            'total_time': r.total_time,
            'steps_per_second': r.steps_per_second,
            'extra_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in r.extra_metrics.items()}
        }
        for name, r in results.items()
    }

    results_path = os.path.join(os.path.dirname(__file__), 'results_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")
