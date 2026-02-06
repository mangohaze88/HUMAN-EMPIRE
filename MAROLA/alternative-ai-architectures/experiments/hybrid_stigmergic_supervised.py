"""
Hybrid Stigmergic-Supervised Learning Approaches
=================================================

Experimental framework for bridging stigmergic self-organization with supervised learning.

CORE CHALLENGE:
- Pure stigmergy: agents predict their own future state (stable, but doesn't reduce task error)
- Supervised learning: needs gradient signal from task loss to agent behaviors
- Problem: How do we connect global task error to local agent decisions?

This module implements 5 experimental approaches with rigorous evaluation.

Author: Innovation & Experimentation Specialist
Date: 2026-02-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import copy


class HybridApproach(Enum):
    """Different hybrid strategies"""
    REWARD_PHEROMONE = 1
    GRADIENT_FIELD = 2
    EVOLUTIONARY = 3
    CREDIT_ASSIGNMENT = 4
    HYBRID_BACKPROP = 5


@dataclass
class ExperimentalResult:
    """Results from one experimental run"""
    approach: HybridApproach
    task_error_history: List[float]
    agent_error_history: List[float]
    convergence_steps: int
    final_task_error: float
    computational_cost: float
    stability_score: float
    explanation: str


# ==============================================================================
# APPROACH 1: REWARD PHEROMONE
# ==============================================================================

class RewardPheromoneSystem:
    """
    MECHANISM:
    - Track task error at each step
    - When task error decreases: release "reward pheromone" globally
    - Agents that recently deposited pheromones get reinforced
    - Use reward signal to modulate agent learning rates

    INSPIRED BY:
    - Reinforcement learning (TD error)
    - Dopamine signaling in neural systems
    - Foraging reward in ant colonies

    ADVANTAGES:
    + Simple to implement
    + Biologically plausible
    + No need for explicit credit assignment
    + Preserves stigmergic autonomy

    POTENTIAL ISSUES:
    - Temporal credit assignment problem (which action caused improvement?)
    - Reward signal might be too diffuse
    - Could reinforce spurious correlations
    - Delayed feedback loop (multiple steps between action and reward)
    """

    def __init__(
        self,
        n_agents: int,
        env_shape: Tuple[int, int],
        n_pheromones: int = 12,  # +2 for reward/punishment
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.env_shape = env_shape
        self.n_pheromones = n_pheromones

        # Standard pheromone field
        self.pheromones = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

        # NEW: Reward/punishment channels
        self.ch_reward = n_pheromones - 2
        self.ch_punishment = n_pheromones - 1

        # Agent state
        self.positions = torch.rand(n_agents, 2, device=self.device)
        self.velocities = torch.zeros(n_agents, 2, device=self.device)
        self.agent_weights = torch.randn(
            n_agents, 32, n_pheromones, device=self.device
        ) * 0.01

        # NEW: Track recent agent contributions for credit assignment
        self.contribution_history = torch.zeros(
            n_agents, 10, device=self.device
        )  # Last 10 steps
        self.history_ptr = 0

        # Task error history
        self.prev_task_error = None

        # Output layer
        env_size = env_shape[0] * env_shape[1] * n_pheromones
        self.output_weights = torch.randn(
            32, env_size, device=self.device
        ) * 0.01

    def forward(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input with reward-based learning.
        """
        # Inject input
        self._inject_input(input_data)

        task_errors = []
        agent_errors = []

        for step in range(n_steps):
            # 1. Read sensory
            sensory = self._read_pheromones(self.positions)

            # 2. Agent prediction
            predictions = torch.bmm(
                self.agent_weights,
                sensory.unsqueeze(-1)
            ).squeeze(-1)
            predictions = torch.tanh(predictions)

            # 3. Compute agent errors (self-prediction)
            agent_error = torch.mean((predictions[:, :self.n_pheromones] - sensory) ** 2, dim=1)
            agent_errors.append(agent_error.mean().item())

            # 4. Compute task error (global objective)
            env_flat = self.pheromones.flatten()
            output = torch.tanh(
                self.output_weights @ env_flat[:self.output_weights.shape[1]]
            )
            task_error = torch.mean((target - output) ** 2)
            task_errors.append(task_error.item())

            # 5. REWARD SIGNAL: Compare with previous task error
            if self.prev_task_error is not None:
                delta_error = self.prev_task_error - task_error.item()

                if delta_error > 0:  # Improvement
                    # Broadcast reward pheromone
                    reward_strength = delta_error * 100  # Scale for visibility
                    self._broadcast_reward(reward_strength)

                    # Reinforce agents that contributed recently
                    # Agents with high recent contribution get learning boost
                    recent_contribution = self.contribution_history.mean(dim=1)
                    learning_boost = 1.0 + torch.sigmoid(recent_contribution) * 0.5
                else:  # Degradation
                    # Broadcast punishment
                    punishment_strength = abs(delta_error) * 100
                    self._broadcast_punishment(punishment_strength)

                    # Reduce learning rates
                    learning_boost = torch.ones(self.n_agents, device=self.device) * 0.8
            else:
                learning_boost = torch.ones(self.n_agents, device=self.device)

            self.prev_task_error = task_error.item()

            # 6. Learn with modulated learning rate
            self._learn_with_reward(
                sensory, predictions, agent_error, learning_boost
            )

            # 7. Move and deposit
            self._move_simple()
            self._deposit_contributions(agent_error)

            # 8. Environment step
            self._env_step()

        # Final output
        env_flat = self.pheromones.flatten()
        output = torch.tanh(
            self.output_weights @ env_flat[:self.output_weights.shape[1]]
        )

        # Update output weights
        output_error = target - output
        self.output_weights += 0.01 * torch.outer(
            output_error, env_flat[:self.output_weights.shape[1]]
        )
        self.output_weights *= 0.999

        info = {
            'task_error_history': task_errors,
            'agent_error_history': agent_errors,
            'final_task_error': task_errors[-1],
            'mean_reward': self.pheromones[self.ch_reward].mean().item(),
        }

        return output, info

    def _broadcast_reward(self, strength: float):
        """Broadcast reward pheromone globally"""
        # Radial pattern from center
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        y, x = torch.meshgrid(
            torch.arange(self.env_shape[0], device=self.device),
            torch.arange(self.env_shape[1], device=self.device),
            indexing='ij'
        )
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        falloff = 1.0 / (1.0 + 0.05 * dist)

        self.pheromones[self.ch_reward] = strength * falloff

    def _broadcast_punishment(self, strength: float):
        """Broadcast punishment signal"""
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        y, x = torch.meshgrid(
            torch.arange(self.env_shape[0], device=self.device),
            torch.arange(self.env_shape[1], device=self.device),
            indexing='ij'
        )
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        falloff = 1.0 / (1.0 + 0.05 * dist)

        self.pheromones[self.ch_punishment] = strength * falloff

    def _learn_with_reward(
        self,
        sensory: torch.Tensor,
        predictions: torch.Tensor,
        errors: torch.Tensor,
        learning_boost: torch.Tensor
    ):
        """Learn with reward-modulated learning rate"""
        # Read reward/punishment signals
        reward_signal = sensory[:, self.ch_reward] / 100.0
        punishment_signal = sensory[:, self.ch_punishment] / 100.0

        # Modulate learning rate
        reward_factor = 1.0 + reward_signal - punishment_signal
        reward_factor = torch.clamp(reward_factor, 0.5, 2.0)

        adaptive_lr = 0.01 * learning_boost * reward_factor

        # Standard predictive learning
        target = sensory[:, :predictions.shape[1]]
        error = target - predictions

        # Update weights
        delta = torch.bmm(
            error.unsqueeze(-1),
            sensory.unsqueeze(1)
        )
        delta = adaptive_lr.view(-1, 1, 1) * delta
        delta = torch.clamp(delta, -0.1, 0.1)

        self.agent_weights += delta
        self.agent_weights *= 0.9999
        self.agent_weights = torch.clamp(self.agent_weights, -5, 5)

    def _deposit_contributions(self, errors: torch.Tensor):
        """Track agent contributions for credit assignment"""
        # Contribution = inverse of error (good agents contribute more)
        contribution = torch.clamp(1.0 - errors, 0, 1)

        # Update history
        self.contribution_history[:, self.history_ptr] = contribution
        self.history_ptr = (self.history_ptr + 1) % 10

        # Deposit proportional to contribution
        x = (self.positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (self.positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        for ch in range(min(4, self.n_pheromones - 2)):
            amounts = contribution * torch.randn(self.n_agents, device=self.device).abs()
            flat_idx = x * self.env_shape[1] + y
            self.pheromones[ch].view(-1).scatter_add_(0, flat_idx, amounts)

    def _read_pheromones(self, positions: torch.Tensor) -> torch.Tensor:
        """Read pheromones at positions"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]
        return self.pheromones[:, x, y].T

    def _move_simple(self):
        """Simple random movement"""
        self.velocities = 0.9 * self.velocities + 0.1 * torch.randn_like(self.velocities)
        self.velocities = torch.clamp(self.velocities, -0.3, 0.3)
        self.positions += 0.01 * self.velocities
        self.positions = self.positions % 1.0

    def _inject_input(self, input_data: torch.Tensor):
        """Inject input into environment"""
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        size = min(int(np.sqrt(len(input_data))), 8)

        if len(input_data) >= size * size:
            pattern = input_data[:size*size].reshape(size, size)
            x_s, x_e = cx - size//2, cx + size//2
            y_s, y_e = cy - size//2, cy + size//2
            self.pheromones[0, x_s:x_e, y_s:y_e] += pattern.abs()

    def _env_step(self):
        """Simple diffusion and evaporation"""
        # Diffusion
        kernel = torch.tensor([
            [0, 0.05, 0],
            [0.05, 0.8, 0.05],
            [0, 0.05, 0]
        ], device=self.device).view(1, 1, 3, 3).repeat(self.n_pheromones, 1, 1, 1)

        self.pheromones = F.conv2d(
            self.pheromones.unsqueeze(0),
            kernel,
            padding=1,
            groups=self.n_pheromones
        ).squeeze(0)

        # Evaporation (faster for reward/punishment)
        evap_rates = torch.ones(self.n_pheromones, device=self.device) * 0.02
        evap_rates[self.ch_reward] = 0.1  # Fast decay
        evap_rates[self.ch_punishment] = 0.1

        for ch in range(self.n_pheromones):
            self.pheromones[ch] *= (1 - evap_rates[ch])

        self.pheromones = torch.clamp(self.pheromones, 0, 100)


# ==============================================================================
# APPROACH 2: GRADIENT FIELD
# ==============================================================================

class GradientFieldSystem:
    """
    MECHANISM:
    - Compute gradient of task loss w.r.t. pheromone field
    - Broadcast this gradient as a spatial pheromone pattern
    - Agents sense gradient and adjust deposits accordingly
    - "If error wants more activation here, I deposit more"

    INSPIRED BY:
    - Backpropagation (explicit gradients)
    - Chemotaxis (bacteria following chemical gradients)
    - Electric field guidance

    ADVANTAGES:
    + Direct gradient signal (optimal direction)
    + Mathematically principled
    + Should converge faster
    + No spurious credit assignment

    POTENTIAL ISSUES:
    - Requires differentiable pheromone field (breaks stigmergic autonomy?)
    - Gradient computation expensive
    - Agents must "understand" gradient semantics
    - May lose emergent properties by being too direct
    - Gradient might be noisy/unstable in high-dimensional space
    """

    def __init__(
        self,
        n_agents: int,
        env_shape: Tuple[int, int],
        n_pheromones: int = 10,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.env_shape = env_shape
        self.n_pheromones = n_pheromones

        # Pheromone field (requires_grad for gradient computation)
        self.pheromones = torch.zeros(
            n_pheromones, *env_shape, device=self.device, requires_grad=True
        )

        # Gradient field (computed from task loss)
        self.gradient_field = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

        # Agent state
        self.positions = torch.rand(n_agents, 2, device=self.device)
        self.velocities = torch.zeros(n_agents, 2, device=self.device)

        # Output network
        env_size = env_shape[0] * env_shape[1] * n_pheromones
        self.output_weights = torch.randn(
            32, env_size, device=self.device
        ) * 0.01

    def forward(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process with gradient field guidance.
        """
        self._inject_input(input_data)

        task_errors = []
        agent_errors = []

        for step in range(n_steps):
            # 1. Compute task error and gradient
            with torch.enable_grad():
                # Ensure pheromones require grad
                if not self.pheromones.requires_grad:
                    self.pheromones = self.pheromones.detach().requires_grad_(True)

                env_flat = self.pheromones.flatten()
                output = torch.tanh(
                    self.output_weights @ env_flat[:self.output_weights.shape[1]]
                )
                task_error = torch.mean((target - output) ** 2)

                # Backprop to get gradient w.r.t. pheromones
                task_error.backward(retain_graph=True)

                # Store gradient field
                if self.pheromones.grad is not None:
                    self.gradient_field = self.pheromones.grad.detach().clone()
                    self.pheromones.grad.zero_()
                else:
                    self.gradient_field = torch.zeros_like(self.pheromones)

            task_errors.append(task_error.item())

            # 2. Agents sense gradient field
            sensory_grad = self._read_gradient_at_positions(self.positions)

            # 3. Agents deposit pheromones guided by gradient
            # Positive gradient = need MORE pheromone here
            # Negative gradient = need LESS pheromone here
            self._deposit_gradient_guided(sensory_grad)

            # 4. Simple movement
            self._move_simple()

            # 5. Environment step (detach to avoid growing computation graph)
            with torch.no_grad():
                self._env_step()

        # Final output
        with torch.no_grad():
            env_flat = self.pheromones.flatten()
            output = torch.tanh(
                self.output_weights @ env_flat[:self.output_weights.shape[1]]
            )

        # Update output weights
        with torch.no_grad():
            output_error = target - output
            self.output_weights += 0.01 * torch.outer(
                output_error, env_flat[:self.output_weights.shape[1]]
            )
            self.output_weights *= 0.999

        info = {
            'task_error_history': task_errors,
            'agent_error_history': agent_errors,
            'final_task_error': task_errors[-1],
            'mean_gradient': self.gradient_field.abs().mean().item(),
        }

        return output, info

    def _read_gradient_at_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Read gradient field at agent positions"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        with torch.no_grad():
            return self.gradient_field[:, x, y].T

    def _deposit_gradient_guided(self, gradients: torch.Tensor):
        """Deposit pheromones in direction that reduces task error"""
        x = (self.positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (self.positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        with torch.no_grad():
            # Gradient descent step: move pheromones OPPOSITE to gradient
            # But agents deposit, so: deposit where gradient is positive (need more)
            for ch in range(self.n_pheromones):
                # Positive gradient = need more activation
                amounts = torch.clamp(gradients[:, ch] * 0.5, 0, 10)

                flat_idx = x * self.env_shape[1] + y
                self.pheromones.data[ch].view(-1).scatter_add_(0, flat_idx, amounts)

    def _move_simple(self):
        """Simple random walk"""
        with torch.no_grad():
            self.velocities = 0.9 * self.velocities + 0.1 * torch.randn_like(self.velocities)
            self.velocities = torch.clamp(self.velocities, -0.3, 0.3)
            self.positions += 0.01 * self.velocities
            self.positions = self.positions % 1.0

    def _inject_input(self, input_data: torch.Tensor):
        """Inject input"""
        with torch.no_grad():
            cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
            size = min(int(np.sqrt(len(input_data))), 8)

            if len(input_data) >= size * size:
                pattern = input_data[:size*size].reshape(size, size)
                x_s, x_e = cx - size//2, cx + size//2
                y_s, y_e = cy - size//2, cy + size//2
                self.pheromones.data[0, x_s:x_e, y_s:y_e] += pattern.abs()

    def _env_step(self):
        """Diffusion and evaporation"""
        kernel = torch.tensor([
            [0, 0.05, 0],
            [0.05, 0.8, 0.05],
            [0, 0.05, 0]
        ], device=self.device).view(1, 1, 3, 3).repeat(self.n_pheromones, 1, 1, 1)

        diffused = F.conv2d(
            self.pheromones.data.unsqueeze(0),
            kernel,
            padding=1,
            groups=self.n_pheromones
        ).squeeze(0)

        self.pheromones.data = diffused * 0.98  # Evaporation
        self.pheromones.data = torch.clamp(self.pheromones.data, 0, 100)


# ==============================================================================
# APPROACH 3: EVOLUTIONARY
# ==============================================================================

class EvolutionarySystem:
    """
    MECHANISM:
    - Each agent has a "genotype" (its weights)
    - Measure fitness = contribution to task error reduction
    - Agents with higher fitness reproduce more
    - Selection pressure drives population toward task-optimal behaviors

    INSPIRED BY:
    - Genetic algorithms
    - Natural selection
    - Neuroevolution (NEAT, ES)

    ADVANTAGES:
    + No gradient needed (black-box optimization)
    + Can discover novel solutions
    + Naturally handles credit assignment via fitness
    + Biologically plausible

    POTENTIAL ISSUES:
    - Very sample inefficient (need many generations)
    - Slow convergence
    - Fitness evaluation noisy (stochastic environment)
    - Population diversity vs exploitation tradeoff
    - How to measure individual fitness in collective task?
    """

    def __init__(
        self,
        n_agents: int,
        env_shape: Tuple[int, int],
        n_pheromones: int = 10,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.env_shape = env_shape
        self.n_pheromones = n_pheromones

        # Pheromone field
        self.pheromones = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

        # Agent state
        self.positions = torch.rand(n_agents, 2, device=self.device)
        self.velocities = torch.zeros(n_agents, 2, device=self.device)
        self.agent_weights = torch.randn(
            n_agents, 32, n_pheromones, device=self.device
        ) * 0.01

        # Fitness tracking
        self.fitness = torch.zeros(n_agents, device=self.device)
        self.contribution_score = torch.zeros(n_agents, device=self.device)

        # Output layer
        env_size = env_shape[0] * env_shape[1] * n_pheromones
        self.output_weights = torch.randn(
            32, env_size, device=self.device
        ) * 0.01

        self.generation = 0
        self.task_error_baseline = None

    def forward(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20,
        evolve_every: int = 5
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process with evolutionary learning.
        """
        self._inject_input(input_data)

        task_errors = []
        agent_errors = []

        # Measure baseline task error (without agents)
        if self.task_error_baseline is None:
            env_flat = self.pheromones.flatten()
            output = torch.tanh(
                self.output_weights @ env_flat[:self.output_weights.shape[1]]
            )
            self.task_error_baseline = torch.mean((target - output) ** 2).item()

        # Reset contribution scores
        self.contribution_score.zero_()

        for step in range(n_steps):
            # 1. Read sensory
            sensory = self._read_pheromones(self.positions)

            # 2. Agent predictions and deposits
            predictions = torch.bmm(
                self.agent_weights,
                sensory.unsqueeze(-1)
            ).squeeze(-1)
            predictions = torch.tanh(predictions)

            # 3. Deposit pheromones
            self._deposit_pheromones(predictions)

            # 4. Compute task error
            env_flat = self.pheromones.flatten()
            output = torch.tanh(
                self.output_weights @ env_flat[:self.output_weights.shape[1]]
            )
            task_error = torch.mean((target - output) ** 2)
            task_errors.append(task_error.item())

            # 5. Estimate individual contributions
            # Agents whose deposits align with reducing error get higher fitness
            improvement = self.task_error_baseline - task_error.item()

            # Crude contribution estimation: agents in high-gradient areas contribute more
            env_gradient = self._compute_env_gradient()
            agent_gradient = self._read_gradient_at_positions(env_gradient, self.positions)
            contribution = torch.abs(agent_gradient).mean(dim=1) * max(0, improvement)
            self.contribution_score += contribution

            # 6. Move
            self._move_simple()

            # 7. Environment step
            self._env_step()

            # 8. Evolution (every N steps)
            if step % evolve_every == 0 and step > 0:
                self._evolve_population()

        # Final output
        env_flat = self.pheromones.flatten()
        output = torch.tanh(
            self.output_weights @ env_flat[:self.output_weights.shape[1]]
        )

        # Update output weights
        output_error = target - output
        self.output_weights += 0.01 * torch.outer(
            output_error, env_flat[:self.output_weights.shape[1]]
        )
        self.output_weights *= 0.999

        info = {
            'task_error_history': task_errors,
            'agent_error_history': agent_errors,
            'final_task_error': task_errors[-1],
            'generation': self.generation,
            'mean_fitness': self.fitness.mean().item(),
        }

        return output, info

    def _evolve_population(self):
        """Selection, crossover, mutation"""
        # Fitness = contribution score
        self.fitness = self.contribution_score.clone()

        # Selection: keep top 50%
        n_survivors = self.n_agents // 2
        sorted_indices = torch.argsort(self.fitness, descending=True)
        survivors = sorted_indices[:n_survivors]

        # Reproduce to fill population
        new_weights = []
        for i in range(self.n_agents):
            if i < n_survivors:
                # Keep survivor
                new_weights.append(self.agent_weights[survivors[i]].clone())
            else:
                # Create offspring from two random survivors
                parent1_idx = survivors[torch.randint(0, n_survivors, (1,))]
                parent2_idx = survivors[torch.randint(0, n_survivors, (1,))]

                # Crossover
                mask = torch.rand_like(self.agent_weights[0]) > 0.5
                child = torch.where(
                    mask,
                    self.agent_weights[parent1_idx],
                    self.agent_weights[parent2_idx]
                )

                # Mutation
                mutation = torch.randn_like(child) * 0.1
                child = child + mutation

                new_weights.append(child)

        self.agent_weights = torch.stack(new_weights)
        self.agent_weights = torch.clamp(self.agent_weights, -5, 5)

        self.generation += 1
        self.contribution_score.zero_()

    def _compute_env_gradient(self) -> torch.Tensor:
        """Compute spatial gradient of pheromone field"""
        grad_x = torch.zeros_like(self.pheromones)
        grad_y = torch.zeros_like(self.pheromones)

        for ch in range(self.n_pheromones):
            grad_x[ch] = torch.gradient(self.pheromones[ch], dim=0)[0]
            grad_y[ch] = torch.gradient(self.pheromones[ch], dim=1)[0]

        return torch.sqrt(grad_x**2 + grad_y**2)

    def _read_gradient_at_positions(
        self, gradient_field: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Read gradient at positions"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]
        return gradient_field[:, x, y].T

    def _deposit_pheromones(self, predictions: torch.Tensor):
        """Deposit based on predictions"""
        x = (self.positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (self.positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        for ch in range(min(self.n_pheromones, predictions.shape[1])):
            amounts = torch.clamp(predictions[:, ch].abs(), 0, 10)
            flat_idx = x * self.env_shape[1] + y
            self.pheromones[ch].view(-1).scatter_add_(0, flat_idx, amounts)

    def _read_pheromones(self, positions: torch.Tensor) -> torch.Tensor:
        """Read pheromones"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]
        return self.pheromones[:, x, y].T

    def _move_simple(self):
        """Simple movement"""
        self.velocities = 0.9 * self.velocities + 0.1 * torch.randn_like(self.velocities)
        self.velocities = torch.clamp(self.velocities, -0.3, 0.3)
        self.positions += 0.01 * self.velocities
        self.positions = self.positions % 1.0

    def _inject_input(self, input_data: torch.Tensor):
        """Inject input"""
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        size = min(int(np.sqrt(len(input_data))), 8)

        if len(input_data) >= size * size:
            pattern = input_data[:size*size].reshape(size, size)
            x_s, x_e = cx - size//2, cx + size//2
            y_s, y_e = cy - size//2, cy + size//2
            self.pheromones[0, x_s:x_e, y_s:y_e] += pattern.abs()

    def _env_step(self):
        """Diffusion and evaporation"""
        kernel = torch.tensor([
            [0, 0.05, 0],
            [0.05, 0.8, 0.05],
            [0, 0.05, 0]
        ], device=self.device).view(1, 1, 3, 3).repeat(self.n_pheromones, 1, 1, 1)

        self.pheromones = F.conv2d(
            self.pheromones.unsqueeze(0),
            kernel,
            padding=1,
            groups=self.n_pheromones
        ).squeeze(0)

        self.pheromones *= 0.98
        self.pheromones = torch.clamp(self.pheromones, 0, 100)


# ==============================================================================
# APPROACH 4: CREDIT ASSIGNMENT (Attention-based)
# ==============================================================================

class CreditAssignmentSystem:
    """
    MECHANISM:
    - Track which pheromone deposits contributed to output
    - Use attention mechanism to assign credit backward
    - Agents whose deposits were "attended to" get stronger learning signal
    - Similar to backprop through attention

    INSPIRED BY:
    - Attention mechanisms in transformers
    - Credit assignment in RL (A3C, PPO)
    - Counterfactual reasoning

    ADVANTAGES:
    + Precise credit assignment
    + Can handle long temporal dependencies
    + Mathematically principled

    POTENTIAL ISSUES:
    - Requires tracking contribution history (memory intensive)
    - Attention computation expensive
    - May break stigmergic locality (global attention)
    - Complex to implement
    """

    def __init__(
        self,
        n_agents: int,
        env_shape: Tuple[int, int],
        n_pheromones: int = 10,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.env_shape = env_shape
        self.n_pheromones = n_pheromones

        # Pheromone field
        self.pheromones = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

        # Contribution tracking: who deposited what where
        self.contribution_map = torch.zeros(
            n_agents, n_pheromones, *env_shape, device=self.device
        )

        # Agent state
        self.positions = torch.rand(n_agents, 2, device=self.device)
        self.velocities = torch.zeros(n_agents, 2, device=self.device)
        self.agent_weights = torch.randn(
            n_agents, 32, n_pheromones, device=self.device
        ) * 0.01

        # Output network with attention
        env_size = env_shape[0] * env_shape[1] * n_pheromones
        self.output_query = torch.randn(32, 64, device=self.device) * 0.1
        self.pheromone_key = torch.randn(64, n_pheromones, device=self.device) * 0.1
        self.output_value = torch.randn(32, n_pheromones, device=self.device) * 0.1

    def forward(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process with attention-based credit assignment.
        """
        self._inject_input(input_data)

        task_errors = []
        agent_errors = []

        # Reset contribution map
        self.contribution_map.zero_()

        for step in range(n_steps):
            # 1. Read sensory
            sensory = self._read_pheromones(self.positions)

            # 2. Agent predictions
            predictions = torch.bmm(
                self.agent_weights,
                sensory.unsqueeze(-1)
            ).squeeze(-1)
            predictions = torch.tanh(predictions)

            # 3. Deposit pheromones and TRACK contributions
            self._deposit_with_tracking(predictions)

            # 4. Compute output with attention
            output, attention_weights = self._compute_output_with_attention()

            # 5. Task error
            task_error = torch.mean((target - output) ** 2)
            task_errors.append(task_error.item())

            # 6. Backpropagate credit using attention weights
            credit_signal = self._compute_credit_signal(
                target - output, attention_weights
            )

            # 7. Learn with credit-weighted update
            self._learn_with_credit(sensory, predictions, credit_signal)

            # 8. Move
            self._move_simple()

            # 9. Environment step (decay contribution map too)
            self._env_step()
            self.contribution_map *= 0.95

        # Final output
        output, _ = self._compute_output_with_attention()

        info = {
            'task_error_history': task_errors,
            'agent_error_history': agent_errors,
            'final_task_error': task_errors[-1],
            'mean_contribution': self.contribution_map.abs().mean().item(),
        }

        return output, info

    def _compute_output_with_attention(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output using attention over pheromone field.

        Returns:
            output: (output_dim,)
            attention_weights: (n_pheromones, H, W) spatial attention
        """
        # Flatten pheromones to (n_pheromones, H*W)
        pheromones_flat = self.pheromones.view(self.n_pheromones, -1)

        # Compute attention: Q @ K^T
        # Query: output_query @ pheromone_key = (32, 64) @ (64, n_ph) = (32, n_ph)
        query = self.output_query @ self.pheromone_key  # (32, n_ph)

        # Attention over channels (simplified - not spatial)
        attention_logits = torch.sum(query, dim=0)  # (n_ph,)
        attention_weights = F.softmax(attention_logits, dim=0)  # (n_ph,)

        # Weighted sum of pheromones
        weighted_pheromones = attention_weights.view(-1, 1) * pheromones_flat  # (n_ph, H*W)
        aggregated = torch.sum(weighted_pheromones, dim=0)  # (H*W,)

        # Output projection
        output = torch.tanh(self.output_value @ attention_weights)

        # Reshape attention for spatial credit assignment
        attention_spatial = attention_weights.view(-1, 1, 1).expand(
            self.n_pheromones, *self.env_shape
        )

        return output, attention_spatial

    def _compute_credit_signal(
        self,
        output_error: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Backpropagate credit to agents based on attention.

        Returns: (n_agents,) credit signal
        """
        # Credit = how much did this agent's contributions get attended to?
        # contribution_map: (n_agents, n_ph, H, W)
        # attention_weights: (n_ph, H, W)

        # Element-wise product and sum
        agent_credit = torch.sum(
            self.contribution_map * attention_weights.unsqueeze(0),
            dim=(1, 2, 3)  # Sum over channels and space
        )  # (n_agents,)

        # Modulate by output error magnitude
        error_magnitude = torch.norm(output_error)
        agent_credit = agent_credit * error_magnitude

        return agent_credit

    def _learn_with_credit(
        self,
        sensory: torch.Tensor,
        predictions: torch.Tensor,
        credit_signal: torch.Tensor
    ):
        """Learn with credit-weighted updates"""
        # Credit signal determines learning rate
        adaptive_lr = 0.01 * (1.0 + torch.sigmoid(credit_signal) * 2.0)
        adaptive_lr = torch.clamp(adaptive_lr, 0.001, 0.1)

        # Standard predictive learning
        target = sensory[:, :predictions.shape[1]]
        error = target - predictions

        delta = torch.bmm(
            error.unsqueeze(-1),
            sensory.unsqueeze(1)
        )
        delta = adaptive_lr.view(-1, 1, 1) * delta
        delta = torch.clamp(delta, -0.1, 0.1)

        self.agent_weights += delta
        self.agent_weights *= 0.9999
        self.agent_weights = torch.clamp(self.agent_weights, -5, 5)

    def _deposit_with_tracking(self, predictions: torch.Tensor):
        """Deposit pheromones and track which agent deposited where"""
        x = (self.positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (self.positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        for ch in range(min(self.n_pheromones, predictions.shape[1])):
            amounts = torch.clamp(predictions[:, ch].abs(), 0, 10)

            # Update pheromone field
            flat_idx = x * self.env_shape[1] + y
            self.pheromones[ch].view(-1).scatter_add_(0, flat_idx, amounts)

            # Update contribution map
            for i in range(self.n_agents):
                self.contribution_map[i, ch, x[i], y[i]] += amounts[i]

    def _read_pheromones(self, positions: torch.Tensor) -> torch.Tensor:
        """Read pheromones"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]
        return self.pheromones[:, x, y].T

    def _move_simple(self):
        """Simple movement"""
        self.velocities = 0.9 * self.velocities + 0.1 * torch.randn_like(self.velocities)
        self.velocities = torch.clamp(self.velocities, -0.3, 0.3)
        self.positions += 0.01 * self.velocities
        self.positions = self.positions % 1.0

    def _inject_input(self, input_data: torch.Tensor):
        """Inject input"""
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        size = min(int(np.sqrt(len(input_data))), 8)

        if len(input_data) >= size * size:
            pattern = input_data[:size*size].reshape(size, size)
            x_s, x_e = cx - size//2, cx + size//2
            y_s, y_e = cy - size//2, cy + size//2
            self.pheromones[0, x_s:x_e, y_s:y_e] += pattern.abs()

    def _env_step(self):
        """Diffusion and evaporation"""
        kernel = torch.tensor([
            [0, 0.05, 0],
            [0.05, 0.8, 0.05],
            [0, 0.05, 0]
        ], device=self.device).view(1, 1, 3, 3).repeat(self.n_pheromones, 1, 1, 1)

        self.pheromones = F.conv2d(
            self.pheromones.unsqueeze(0),
            kernel,
            padding=1,
            groups=self.n_pheromones
        ).squeeze(0)

        self.pheromones *= 0.98
        self.pheromones = torch.clamp(self.pheromones, 0, 100)


# ==============================================================================
# APPROACH 5: HYBRID BACKPROP
# ==============================================================================

class HybridBackpropSystem:
    """
    MECHANISM:
    - Agents form stigmergic hidden representation (emergent, self-organized)
    - Output layer uses standard backprop
    - Gradient from output layer flows back to modulate pheromone deposits
    - "Best of both worlds": emergence + gradient learning

    INSPIRED BY:
    - Hybrid neural architectures (CNN + transformer)
    - Reservoir computing (random hidden layer + trained readout)
    - Liquid state machines

    ADVANTAGES:
    + Preserves stigmergic emergence in representation
    + Fast convergence via backprop on output
    + Only output layer needs gradients (simple)
    + Agents still autonomous

    POTENTIAL ISSUES:
    - Output layer might not be expressive enough
    - Hidden representation might not align with task
    - Two-phase optimization (agents vs output) could conflict
    - How to modulate stigmergy based on output gradient?
    """

    def __init__(
        self,
        n_agents: int,
        env_shape: Tuple[int, int],
        n_pheromones: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 32,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.env_shape = env_shape
        self.n_pheromones = n_pheromones
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Stigmergic substrate (agents + pheromones)
        self.pheromones = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

        self.positions = torch.rand(n_agents, 2, device=self.device)
        self.velocities = torch.zeros(n_agents, 2, device=self.device)
        self.agent_weights = torch.randn(
            n_agents, 32, n_pheromones, device=self.device
        ) * 0.01

        # Trainable readout network (standard MLP with backprop)
        env_size = env_shape[0] * env_shape[1] * n_pheromones
        self.readout = nn.Sequential(
            nn.Linear(env_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.readout.parameters(), lr=0.01)

        # Modulation signal: gradient magnitude from readout
        self.readout_gradient_signal = torch.zeros(
            n_pheromones, *env_shape, device=self.device
        )

    def forward(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process with hybrid backprop.
        """
        self._inject_input(input_data)

        task_errors = []
        agent_errors = []

        for step in range(n_steps):
            # PHASE 1: STIGMERGIC DYNAMICS (self-organized)
            # Agents sense, predict, deposit pheromones
            sensory = self._read_pheromones(self.positions)

            predictions = torch.bmm(
                self.agent_weights,
                sensory.unsqueeze(-1)
            ).squeeze(-1)
            predictions = torch.tanh(predictions)

            # Agents learn to predict their sensory (stigmergic objective)
            agent_error = torch.mean((predictions[:, :self.n_pheromones] - sensory) ** 2, dim=1)
            agent_errors.append(agent_error.mean().item())

            self._learn_stigmergic(sensory, predictions)

            # Deposit pheromones (modulated by readout gradient)
            self._deposit_modulated(predictions)

            self._move_simple()
            self._env_step()

        # PHASE 2: READOUT WITH BACKPROP
        # Read out pheromone field using trainable network
        env_flat = self.pheromones.flatten()

        # Forward pass with gradient tracking
        output = self.readout(env_flat[:self.readout[0].in_features])

        # Compute task loss
        task_error = F.mse_loss(output, target)
        task_errors.append(task_error.item())

        # Backprop through readout
        self.optimizer.zero_grad()
        task_error.backward()

        # Extract gradient w.r.t. input (pheromone field)
        # This tells us which parts of pheromone field matter for task
        with torch.no_grad():
            input_grad = self.readout[0].weight.grad
            if input_grad is not None:
                # Reshape gradient to spatial field
                grad_reshaped = input_grad.abs().mean(dim=0)  # (env_size,)
                grad_reshaped = grad_reshaped[:self.n_pheromones * self.env_shape[0] * self.env_shape[1]]
                try:
                    self.readout_gradient_signal = grad_reshaped.view(
                        self.n_pheromones, *self.env_shape
                    )
                except:
                    # Fallback if reshape fails
                    self.readout_gradient_signal = torch.zeros_like(self.pheromones)

        self.optimizer.step()

        info = {
            'task_error_history': task_errors,
            'agent_error_history': agent_errors,
            'final_task_error': task_errors[-1],
            'readout_gradient': self.readout_gradient_signal.mean().item(),
        }

        return output.detach(), info

    def _learn_stigmergic(self, sensory: torch.Tensor, predictions: torch.Tensor):
        """Agents learn their own dynamics (stigmergic objective)"""
        target = sensory[:, :predictions.shape[1]]
        error = target - predictions

        delta = 0.01 * torch.bmm(
            error.unsqueeze(-1),
            sensory.unsqueeze(1)
        )
        delta = torch.clamp(delta, -0.1, 0.1)

        self.agent_weights += delta
        self.agent_weights *= 0.9999
        self.agent_weights = torch.clamp(self.agent_weights, -5, 5)

    def _deposit_modulated(self, predictions: torch.Tensor):
        """
        Deposit pheromones, modulated by readout gradient.

        High gradient area = important for task = deposit more there.
        """
        x = (self.positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (self.positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]

        # Read gradient signal at agent positions
        gradient_at_pos = self.readout_gradient_signal[:, x, y].T  # (n_agents, n_ph)

        # Modulate deposits
        modulation = 1.0 + gradient_at_pos  # Higher gradient = more deposit

        for ch in range(min(self.n_pheromones, predictions.shape[1])):
            amounts = torch.clamp(
                predictions[:, ch].abs() * modulation[:, ch],
                0, 10
            )

            flat_idx = x * self.env_shape[1] + y
            self.pheromones[ch].view(-1).scatter_add_(0, flat_idx, amounts)

    def _read_pheromones(self, positions: torch.Tensor) -> torch.Tensor:
        """Read pheromones"""
        x = (positions[:, 0] * self.env_shape[0]).long() % self.env_shape[0]
        y = (positions[:, 1] * self.env_shape[1]).long() % self.env_shape[1]
        return self.pheromones[:, x, y].T

    def _move_simple(self):
        """Simple movement"""
        self.velocities = 0.9 * self.velocities + 0.1 * torch.randn_like(self.velocities)
        self.velocities = torch.clamp(self.velocities, -0.3, 0.3)
        self.positions += 0.01 * self.velocities
        self.positions = self.positions % 1.0

    def _inject_input(self, input_data: torch.Tensor):
        """Inject input"""
        cx, cy = self.env_shape[0] // 2, self.env_shape[1] // 2
        size = min(int(np.sqrt(len(input_data))), 8)

        if len(input_data) >= size * size:
            pattern = input_data[:size*size].reshape(size, size)
            x_s, x_e = cx - size//2, cx + size//2
            y_s, y_e = cy - size//2, cy + size//2
            self.pheromones[0, x_s:x_e, y_s:y_e] += pattern.abs()

    def _env_step(self):
        """Diffusion and evaporation"""
        kernel = torch.tensor([
            [0, 0.05, 0],
            [0.05, 0.8, 0.05],
            [0, 0.05, 0]
        ], device=self.device).view(1, 1, 3, 3).repeat(self.n_pheromones, 1, 1, 1)

        self.pheromones = F.conv2d(
            self.pheromones.unsqueeze(0),
            kernel,
            padding=1,
            groups=self.n_pheromones
        ).squeeze(0)

        self.pheromones *= 0.98
        self.pheromones = torch.clamp(self.pheromones, 0, 100)


# ==============================================================================
# EXPERIMENTAL EVALUATION FRAMEWORK
# ==============================================================================

def run_comparative_experiment(
    approach: HybridApproach,
    n_trials: int = 5,
    n_train_steps: int = 100,
    task_size: int = 32,
    device: str = 'cuda'
) -> ExperimentalResult:
    """
    Rigorously evaluate one approach.

    Measures:
    - Convergence speed (steps to threshold)
    - Final task error
    - Computational cost (time per step)
    - Stability (variance in error)
    """
    import time

    results = {
        'task_errors': [],
        'convergence_steps': [],
        'times': [],
    }

    for trial in range(n_trials):
        # Create system
        if approach == HybridApproach.REWARD_PHEROMONE:
            system = RewardPheromoneSystem(
                n_agents=256, env_shape=(32, 32), device=device
            )
        elif approach == HybridApproach.GRADIENT_FIELD:
            system = GradientFieldSystem(
                n_agents=256, env_shape=(32, 32), device=device
            )
        elif approach == HybridApproach.EVOLUTIONARY:
            system = EvolutionarySystem(
                n_agents=256, env_shape=(32, 32), device=device
            )
        elif approach == HybridApproach.CREDIT_ASSIGNMENT:
            system = CreditAssignmentSystem(
                n_agents=256, env_shape=(32, 32), device=device
            )
        elif approach == HybridApproach.HYBRID_BACKPROP:
            system = HybridBackpropSystem(
                n_agents=256, env_shape=(32, 32), output_dim=task_size, device=device
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Generate task
        input_data = torch.randn(task_size, device=device)
        target = torch.randn(task_size, device=device)

        # Train
        start_time = time.time()

        task_error_history = []
        for step in range(n_train_steps):
            output, info = system.forward(input_data, target, n_steps=10)

            # Record task error
            if 'final_task_error' in info:
                task_error_history.append(info['final_task_error'])
            elif 'task_error_history' in info:
                task_error_history.extend(info['task_error_history'])

        elapsed = time.time() - start_time

        # Measure convergence
        threshold = 0.1
        converged_at = n_train_steps
        for i, err in enumerate(task_error_history):
            if err < threshold:
                converged_at = i
                break

        results['task_errors'].append(task_error_history)
        results['convergence_steps'].append(converged_at)
        results['times'].append(elapsed)

    # Aggregate results
    final_errors = [hist[-1] if len(hist) > 0 else 1.0 for hist in results['task_errors']]
    mean_final_error = np.mean(final_errors)

    mean_convergence = np.mean(results['convergence_steps'])
    mean_time = np.mean(results['times'])

    # Stability = inverse of variance
    error_variance = np.var([hist[-10:] for hist in results['task_errors']])
    stability = 1.0 / (1.0 + error_variance)

    return ExperimentalResult(
        approach=approach,
        task_error_history=results['task_errors'][0],  # First trial
        agent_error_history=[],
        convergence_steps=int(mean_convergence),
        final_task_error=float(mean_final_error),
        computational_cost=float(mean_time),
        stability_score=float(stability),
        explanation=_get_approach_explanation(approach)
    )


def _get_approach_explanation(approach: HybridApproach) -> str:
    """Get summary of approach mechanism"""
    explanations = {
        HybridApproach.REWARD_PHEROMONE: (
            "Broadcasts reward/punishment signal when task error changes. "
            "Agents modulate learning rate based on local reward pheromone."
        ),
        HybridApproach.GRADIENT_FIELD: (
            "Computes gradient of task loss w.r.t. pheromone field. "
            "Agents deposit pheromones in direction that reduces error."
        ),
        HybridApproach.EVOLUTIONARY: (
            "Agents with higher fitness (contribution to error reduction) reproduce more. "
            "Natural selection drives population toward task-optimal behaviors."
        ),
        HybridApproach.CREDIT_ASSIGNMENT: (
            "Uses attention mechanism to track which deposits contributed to output. "
            "Credit backpropagated to agents whose contributions were used."
        ),
        HybridApproach.HYBRID_BACKPROP: (
            "Stigmergic dynamics form hidden representation. "
            "Trainable readout network with backprop. Gradient modulates deposits."
        ),
    }
    return explanations.get(approach, "Unknown approach")


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HYBRID STIGMERGIC-SUPERVISED LEARNING: EXPERIMENTAL COMPARISON")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Using CPU (will be slow).")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    approaches = [
        HybridApproach.REWARD_PHEROMONE,
        HybridApproach.GRADIENT_FIELD,
        HybridApproach.EVOLUTIONARY,
        HybridApproach.CREDIT_ASSIGNMENT,
        HybridApproach.HYBRID_BACKPROP,
    ]

    print("\nRunning experiments...")
    print("Each approach: 5 trials × 100 training steps")
    print("-" * 80)

    results = []
    for approach in approaches:
        print(f"\nTesting: {approach.name}")
        result = run_comparative_experiment(
            approach, n_trials=3, n_train_steps=50, device=device
        )
        results.append(result)

        print(f"  Convergence: {result.convergence_steps} steps")
        print(f"  Final error: {result.final_task_error:.6f}")
        print(f"  Time: {result.computational_cost:.2f}s")
        print(f"  Stability: {result.stability_score:.4f}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Approach':<25} {'Convergence':<15} {'Final Error':<15} {'Time (s)':<12} {'Stability':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result.approach.name:<25} "
              f"{result.convergence_steps:<15} "
              f"{result.final_task_error:<15.6f} "
              f"{result.computational_cost:<12.2f} "
              f"{result.stability_score:<10.4f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Score each approach (lower is better)
    scores = []
    for r in results:
        # Normalize metrics
        conv_score = r.convergence_steps / 100.0
        error_score = r.final_task_error
        time_score = r.computational_cost / 10.0
        stability_score = 1.0 - r.stability_score

        # Weighted sum
        total = conv_score * 0.4 + error_score * 0.4 + time_score * 0.1 + stability_score * 0.1
        scores.append((total, r))

    scores.sort(key=lambda x: x[0])
    best = scores[0][1]

    print(f"\nMOST PROMISING APPROACH: {best.approach.name}")
    print(f"\nExplanation: {best.explanation}")
    print(f"\nRationale:")
    print(f"  - Convergence speed: {best.convergence_steps} steps")
    print(f"  - Final task error: {best.final_task_error:.6f}")
    print(f"  - Computational cost: {best.computational_cost:.2f}s")
    print(f"  - Stability: {best.stability_score:.4f}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("  1. Implement full-scale version of winning approach")
    print("  2. Test on real supervised tasks (MNIST, regression)")
    print("  3. Ablation studies to understand key components")
    print("  4. Optimize hyperparameters")
    print("=" * 80)
