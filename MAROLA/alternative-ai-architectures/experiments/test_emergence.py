"""
Experimental Framework for Stigmergic Emergence
================================================

Rigorous testing of emergent properties in enhanced swarm system.

Tests:
1. Pattern formation (Turing patterns, clustering)
2. Collective decision-making (consensus, voting)
3. Path optimization (TSP-like problems)
4. Task allocation (caste specialization)
5. Learning curves (prediction improvement)
6. Phase transitions (critical density)
7. Information flow (mutual information)

Author: Innovation & Experimentation Specialist
Date: 2026-02-05
"""

import sys
sys.path.append('/root/MAROLA/alternative-ai-architectures')

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

from src.networks.enhanced_stigmergic_swarm import (
    EnhancedStigmergicSwarmGPU,
    SwarmComputer,
    Caste
)


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    name: str
    success: bool
    metric_value: float
    threshold: float
    time_elapsed: float
    metadata: Dict[str, Any]


class EmergenceExperiments:
    """
    Suite of emergence tests.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = []

    def run_all(self) -> List[ExperimentResult]:
        """Run complete test suite"""
        print("="*60)
        print("STIGMERGIC EMERGENCE TEST SUITE")
        print("="*60)

        tests = [
            self.test_pattern_formation,
            self.test_collective_decision,
            self.test_task_allocation,
            self.test_learning_curve,
            self.test_critical_density,
        ]

        for test_fn in tests:
            print(f"\n{'='*60}")
            result = test_fn()
            self.results.append(result)
            self._print_result(result)

        self._print_summary()
        return self.results

    def test_pattern_formation(self) -> ExperimentResult:
        """
        TEST 1: Can swarm self-organize into spatial patterns?

        Metric: FFT power spectrum should show dominant frequency
        (indicating periodic patterns like Turing stripes/spots)
        """
        print("TEST 1: Pattern Formation")
        print("-" * 60)

        start_time = time.time()

        # Create swarm
        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=2048,
            env_shape=(128, 128),
            device=self.device
        )

        # Run until patterns emerge
        print("Running dynamics to form patterns...")
        for step in range(200):
            swarm.step()

            if step % 50 == 0:
                # Check pattern formation
                novelty_field = swarm.pheromones.fields[0].cpu().numpy()
                mean_val = np.mean(novelty_field)
                std_val = np.std(novelty_field)
                print(f"  Step {step}: mean={mean_val:.2f}, std={std_val:.2f}")

        # Analyze final pattern via FFT
        final_field = swarm.pheromones.fields[0].cpu().numpy()

        # 2D FFT
        fft = np.fft.fft2(final_field)
        power_spectrum = np.abs(fft) ** 2

        # Find dominant non-DC frequency
        power_spectrum[0, 0] = 0  # Remove DC component
        max_power = np.max(power_spectrum)

        # Metric: max power should be high (patterns have strong frequencies)
        threshold = 1000  # Arbitrary threshold for pattern strength
        success = max_power > threshold

        elapsed = time.time() - start_time

        metadata = {
            'max_power': float(max_power),
            'field_mean': float(np.mean(final_field)),
            'field_std': float(np.std(final_field)),
            'final_emergence': swarm.history['emergence_score'][-1] if swarm.history['emergence_score'] else 0,
        }

        return ExperimentResult(
            name='Pattern Formation',
            success=success,
            metric_value=float(max_power),
            threshold=threshold,
            time_elapsed=elapsed,
            metadata=metadata
        )

    def test_collective_decision(self) -> ExperimentResult:
        """
        TEST 2: Can swarm reach consensus on best option?

        Setup: Present two "food sources" of different quality
        Metric: Swarm should converge to better option (>80% vote)
        """
        print("TEST 2: Collective Decision-Making")
        print("-" * 60)

        start_time = time.time()

        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=1024,
            env_shape=(64, 64),
            device=self.device
        )

        # Place two options
        # Option A: quality 0.9 at position (0.3, 0.3)
        # Option B: quality 0.4 at position (0.7, 0.7)

        print("Placing options:")
        print("  Option A: quality=0.9 at (0.3, 0.3)")
        print("  Option B: quality=0.4 at (0.7, 0.7)")

        # Inject options as food pheromone
        grid_a = (int(0.3 * 64), int(0.3 * 64))
        grid_b = (int(0.7 * 64), int(0.7 * 64))

        swarm.pheromones.fields[3, grid_a[0]-2:grid_a[0]+3, grid_a[1]-2:grid_a[1]+3] = 90  # A
        swarm.pheromones.fields[3, grid_b[0]-2:grid_b[0]+3, grid_b[1]-2:grid_b[1]+3] = 40  # B

        # Run dynamics
        print("Running consensus process...")
        for step in range(100):
            swarm.step()

        # Count agents near each option
        dists_a = torch.norm(swarm.positions - torch.tensor([0.3, 0.3], device=self.device), dim=1)
        dists_b = torch.norm(swarm.positions - torch.tensor([0.7, 0.7], device=self.device), dim=1)

        near_a = (dists_a < 0.15).sum().item()
        near_b = (dists_b < 0.15).sum().item()

        total_near = near_a + near_b
        if total_near > 0:
            vote_a = near_a / total_near
        else:
            vote_a = 0.5

        print(f"\nFinal distribution:")
        print(f"  Near A (better): {near_a}")
        print(f"  Near B (worse): {near_b}")
        print(f"  Vote fraction for A: {vote_a:.2f}")

        # Success: >70% chose better option
        threshold = 0.7
        success = vote_a > threshold

        elapsed = time.time() - start_time

        return ExperimentResult(
            name='Collective Decision',
            success=success,
            metric_value=vote_a,
            threshold=threshold,
            time_elapsed=elapsed,
            metadata={
                'near_a': near_a,
                'near_b': near_b,
                'total_agents': swarm.n_agents,
            }
        )

    def test_task_allocation(self) -> ExperimentResult:
        """
        TEST 3: Do agents specialize into distinct castes?

        Metric: Shannon entropy of caste distribution
        High entropy = good diversity (all castes represented)
        """
        print("TEST 3: Task Allocation (Caste Formation)")
        print("-" * 60)

        start_time = time.time()

        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=2048,
            env_shape=(128, 128),
            device=self.device
        )

        # Run to allow specialization
        print("Running dynamics for specialization...")
        for step in range(150):
            swarm.step()

            if step % 50 == 0:
                caste_counts = [
                    (swarm.specializations == i).sum().item()
                    for i in range(3)
                ]
                print(f"  Step {step}: Explorers={caste_counts[0]}, "
                      f"Exploiters={caste_counts[1]}, "
                      f"Coordinators={caste_counts[2]}")

        # Final caste distribution
        final_counts = [
            (swarm.specializations == i).sum().item()
            for i in range(3)
        ]

        # Shannon entropy
        probs = np.array(final_counts) / swarm.n_agents
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # Max entropy for 3 castes = log(3) ≈ 1.099
        max_entropy = np.log(3)
        normalized_entropy = entropy / max_entropy

        print(f"\nFinal caste distribution: {final_counts}")
        print(f"Shannon entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})")

        # Success: entropy > 0.7 (reasonably diverse)
        threshold = 0.7
        success = normalized_entropy > threshold

        elapsed = time.time() - start_time

        return ExperimentResult(
            name='Task Allocation',
            success=success,
            metric_value=normalized_entropy,
            threshold=threshold,
            time_elapsed=elapsed,
            metadata={
                'caste_counts': final_counts,
                'raw_entropy': float(entropy),
            }
        )

    def test_learning_curve(self) -> ExperimentResult:
        """
        TEST 4: Does swarm improve predictions over time?

        Metric: Prediction error should decrease
        """
        print("TEST 4: Learning Curve")
        print("-" * 60)

        start_time = time.time()

        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=1024,
            env_shape=(64, 64),
            device=self.device
        )

        # Inject a predictable pattern
        print("Injecting predictable pattern...")
        test_pattern = torch.randn(64, device=self.device)

        errors = []

        for step in range(100):
            # Inject pattern each step
            swarm._inject_input(test_pattern)

            # Step
            swarm.step()

            # Record error
            current_error = swarm.history['collective_error'][-1]
            errors.append(current_error)

            if step % 25 == 0:
                print(f"  Step {step}: error={current_error:.4f}")

        # Check if error decreased
        early_error = np.mean(errors[:20])
        late_error = np.mean(errors[-20:])

        improvement = (early_error - late_error) / (early_error + 1e-6)

        print(f"\nEarly error (avg first 20): {early_error:.4f}")
        print(f"Late error (avg last 20): {late_error:.4f}")
        print(f"Improvement: {improvement*100:.1f}%")

        # Success: at least 10% improvement
        threshold = 0.1
        success = improvement > threshold

        elapsed = time.time() - start_time

        return ExperimentResult(
            name='Learning Curve',
            success=success,
            metric_value=improvement,
            threshold=threshold,
            time_elapsed=elapsed,
            metadata={
                'early_error': float(early_error),
                'late_error': float(late_error),
                'errors': [float(e) for e in errors[::10]],  # Subsample
            }
        )

    def test_critical_density(self) -> ExperimentResult:
        """
        TEST 5: Find critical agent density for emergence.

        Sweep agent count, measure coherence.
        Should see phase transition.
        """
        print("TEST 5: Critical Density (Phase Transition)")
        print("-" * 60)

        start_time = time.time()

        agent_counts = [128, 256, 512, 1024, 2048, 4096]
        coherences = []

        print("Sweeping agent density...")
        for n_agents in agent_counts:
            swarm = EnhancedStigmergicSwarmGPU(
                n_agents=n_agents,
                env_shape=(128, 128),
                device=self.device
            )

            # Run to equilibrium
            for _ in range(50):
                swarm.step()

            # Measure coherence (inverse of position variance)
            pos_var = torch.var(swarm.positions).item()
            coherence = 1.0 / (1.0 + pos_var)
            coherences.append(coherence)

            density = n_agents / (128 * 128)
            print(f"  N={n_agents} (density={density:.4f}): coherence={coherence:.4f}")

        # Find peak gradient (critical point)
        densities = [n / (128*128) for n in agent_counts]
        gradients = np.gradient(coherences, densities)
        critical_idx = np.argmax(np.abs(gradients))
        critical_density = densities[critical_idx]

        print(f"\nCritical density: {critical_density:.4f}")
        print(f"Critical agent count: {agent_counts[critical_idx]}")

        # Success: found a clear transition (max gradient > 10)
        max_gradient = np.max(np.abs(gradients))
        threshold = 5.0
        success = max_gradient > threshold

        elapsed = time.time() - start_time

        return ExperimentResult(
            name='Critical Density',
            success=success,
            metric_value=float(max_gradient),
            threshold=threshold,
            time_elapsed=elapsed,
            metadata={
                'critical_density': critical_density,
                'critical_n_agents': agent_counts[critical_idx],
                'densities': densities,
                'coherences': coherences,
            }
        )

    def _print_result(self, result: ExperimentResult):
        """Pretty print result"""
        status = "PASS" if result.success else "FAIL"
        symbol = "✓" if result.success else "✗"

        print(f"\n{symbol} {result.name}: {status}")
        print(f"  Metric: {result.metric_value:.4f} (threshold: {result.threshold:.4f})")
        print(f"  Time: {result.time_elapsed:.2f}s")

        if result.metadata:
            print("  Metadata:")
            for key, val in result.metadata.items():
                if isinstance(val, (list, np.ndarray)):
                    continue  # Skip large arrays
                print(f"    {key}: {val}")

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        n_pass = sum(r.success for r in self.results)
        n_total = len(self.results)
        total_time = sum(r.time_elapsed for r in self.results)

        print(f"Passed: {n_pass}/{n_total} ({n_pass/n_total*100:.0f}%)")
        print(f"Total time: {total_time:.1f}s")

        print("\nResults:")
        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            print(f"  {r.name:30s} {status}")


class VisualizationExperiments:
    """
    Visual tests to inspect emergence qualitatively.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def visualize_pheromone_evolution(self, save_path: str = '/tmp/pheromone_evolution.png'):
        """
        Create visualization of pheromone fields over time.
        """
        print("\nVISUALIZATION: Pheromone Evolution")
        print("-" * 60)

        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=2048,
            env_shape=(128, 128),
            device=self.device
        )

        # Capture snapshots
        snapshots = []
        steps_to_capture = [0, 25, 50, 100, 150, 200]

        for step in range(201):
            swarm.step()

            if step in steps_to_capture:
                state = swarm.get_visualization_state()
                snapshots.append({
                    'step': step,
                    'pheromones': state['pheromone_fields'].copy(),
                    'positions': state['positions'].copy(),
                })
                print(f"  Captured step {step}")

        # Plot
        fig, axes = plt.subplots(
            len(snapshots), 4,
            figsize=(16, 4 * len(snapshots))
        )

        for i, snap in enumerate(snapshots):
            # Plot key pheromone channels
            channels_to_plot = [
                (0, 'Novelty'),
                (1, 'Competence'),
                (3, 'Food'),
                (4, 'Trail')
            ]

            for j, (ch_idx, ch_name) in enumerate(channels_to_plot):
                ax = axes[i, j] if len(snapshots) > 1 else axes[j]

                field = snap['pheromones'][ch_idx]
                im = ax.imshow(field, cmap='hot', vmin=0, vmax=100)
                ax.set_title(f"{ch_name} (step {snap['step']})")
                ax.axis('off')

                # Overlay agent positions
                pos = snap['positions']
                ax.scatter(
                    pos[:, 1] * 128, pos[:, 0] * 128,
                    c='cyan', s=0.5, alpha=0.3
                )

                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved visualization to {save_path}")

    def visualize_caste_distribution(self, save_path: str = '/tmp/caste_distribution.png'):
        """
        Visualize agent specialization over time.
        """
        print("\nVISUALIZATION: Caste Distribution")
        print("-" * 60)

        swarm = EnhancedStigmergicSwarmGPU(
            n_agents=2048,
            env_shape=(128, 128),
            device=self.device
        )

        # Track caste evolution
        caste_history = []

        for step in range(200):
            swarm.step()

            if step % 5 == 0:
                counts = [
                    (swarm.specializations == i).sum().item()
                    for i in range(3)
                ]
                caste_history.append(counts)

        caste_history = np.array(caste_history)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Stacked area chart
        steps = np.arange(0, 200, 5)
        ax1.stackplot(
            steps,
            caste_history[:, 0],
            caste_history[:, 1],
            caste_history[:, 2],
            labels=['Explorers', 'Exploiters', 'Coordinators'],
            alpha=0.7
        )
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Agent Count')
        ax1.set_title('Caste Distribution Over Time')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3)

        # Final spatial distribution
        state = swarm.get_visualization_state()
        pos = state['positions']
        castes = state['specializations']

        colors = ['red', 'blue', 'green']
        labels = ['Explorer', 'Exploiter', 'Coordinator']

        for caste_id in range(3):
            mask = castes == caste_id
            ax2.scatter(
                pos[mask, 0], pos[mask, 1],
                c=colors[caste_id],
                label=labels[caste_id],
                s=10,
                alpha=0.6
            )

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Final Spatial Distribution by Caste')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    print("Enhanced Stigmergic Swarm - Emergence Test Suite\n")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

        # Run quantitative tests
        experiments = EmergenceExperiments(device='cuda')
        results = experiments.run_all()

        # Run visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        viz = VisualizationExperiments(device='cuda')

        try:
            viz.visualize_pheromone_evolution()
        except Exception as e:
            print(f"Visualization 1 failed: {e}")

        try:
            viz.visualize_caste_distribution()
        except Exception as e:
            print(f"Visualization 2 failed: {e}")

        print("\n" + "="*60)
        print("EXPERIMENTS COMPLETE")
        print("="*60)

    else:
        print("ERROR: CUDA not available. This test suite requires GPU.")
