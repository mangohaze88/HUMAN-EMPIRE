#!/usr/bin/env python3
"""
REAL TIME SERIES BENCHMARK - Bio-Plausible Architectures
=========================================================

Tests bio-plausible learning on ACTUAL temporal patterns:
1. Mackey-Glass chaotic time series (classic benchmark)
2. Lorenz attractor (3D chaos)
3. Stock-like random walk with trends

Architectures tested:
- Liquid Neural Network (designed for time series!)
- Forward-Forward Network
- CuriosityCore
- LSTM baseline (with backprop)
- Simple AR model

NO BACKPROPAGATION for bio-plausible methods!
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.networks.liquid_neural_network import LiquidNeuralNetwork, NCPWiringConfig
from src.networks.forward_forward import ForwardForwardNetwork
from src.networks.curiosity_core import CuriosityCore

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, LSTM baseline will be skipped")


# ============================================================================
# BENCHMARK DATA GENERATORS
# ============================================================================

def generate_mackey_glass(
    n_samples: int = 5000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    initial_history: float = 1.2
) -> np.ndarray:
    """
    Generate Mackey-Glass chaotic time series.

    Equation: dx/dt = β*x(t-τ)/(1+x(t-τ)^n) - γ*x(t)

    This is a CLASSIC benchmark for time series prediction.
    It's chaotic but deterministic - perfect for testing learning.

    Args:
        n_samples: Number of samples to generate
        tau: Time delay (17 = chaotic regime)
        beta: Parameter β (default 0.2)
        gamma: Parameter γ (default 0.1)
        n: Parameter n (default 10)
        dt: Time step
        initial_history: Initial condition

    Returns:
        Time series array of shape (n_samples,)
    """
    # Initialize with history
    history_length = int(tau / dt) + 1
    x = np.ones(n_samples + history_length) * initial_history

    # Add small perturbations to initial history
    x[:history_length] += np.random.randn(history_length) * 0.01

    # Integrate using Euler method
    for t in range(history_length, len(x)):
        x_tau = x[t - int(tau / dt)]
        dx_dt = (beta * x_tau) / (1 + x_tau ** n) - gamma * x[t-1]
        x[t] = x[t-1] + dt * dx_dt

    # Return only the requested samples (discard history)
    return x[history_length:history_length + n_samples]


def generate_lorenz(
    n_samples: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Generate Lorenz attractor time series.

    Equations:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    Famous chaotic system with beautiful butterfly attractor.

    Args:
        n_samples: Number of samples
        dt: Time step (0.01 recommended for stability)
        sigma, rho, beta: Lorenz parameters (standard: 10, 28, 8/3)
        initial_state: Initial (x, y, z)

    Returns:
        Time series array of shape (n_samples, 3)
    """
    trajectory = np.zeros((n_samples, 3))
    trajectory[0] = initial_state

    for t in range(1, n_samples):
        x, y, z = trajectory[t-1]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        trajectory[t, 0] = x + dt * dx
        trajectory[t, 1] = y + dt * dy
        trajectory[t, 2] = z + dt * dz

    return trajectory


def generate_stock_like(
    n_samples: int = 5000,
    trend_strength: float = 0.0002,
    volatility: float = 0.02,
    initial_price: float = 100.0
) -> np.ndarray:
    """
    Generate stock-like price series with trends and noise.

    Realistic financial time series:
    - Random walk base
    - Trend component
    - Heteroscedastic volatility
    - Mean reversion

    Args:
        n_samples: Number of samples
        trend_strength: Strength of upward trend
        volatility: Base volatility
        initial_price: Starting price

    Returns:
        Price series of shape (n_samples,)
    """
    prices = np.zeros(n_samples)
    prices[0] = initial_price

    # Generate trends that change over time
    n_trend_periods = 10
    trend_changes = np.random.choice(n_samples, n_trend_periods, replace=False)
    trend_changes = np.sort(trend_changes)
    current_trend = trend_strength

    for t in range(1, n_samples):
        # Check for trend change
        if t in trend_changes:
            current_trend = np.random.randn() * trend_strength * 2

        # Volatility clustering (GARCH-like)
        if t > 20:
            recent_returns = np.diff(prices[t-20:t])
            current_volatility = volatility * (1 + 0.5 * np.std(recent_returns))
        else:
            current_volatility = volatility

        # Random walk with trend
        return_pct = current_trend + np.random.randn() * current_volatility

        # Mean reversion component
        mean_price = np.mean(prices[max(0, t-100):t])
        reversion = -0.001 * (prices[t-1] - mean_price)

        prices[t] = prices[t-1] * (1 + return_pct + reversion)

    return prices


# ============================================================================
# TIME SERIES PREDICTION WRAPPER
# ============================================================================

class TimeSeriesPredictor:
    """Base class for time series prediction"""

    def __init__(self, window_size: int = 10, horizon: int = 1):
        self.window_size = window_size
        self.horizon = horizon

    def train(self, data: np.ndarray, n_epochs: int = 1):
        """Train on time series data"""
        raise NotImplementedError

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict next values given window"""
        raise NotImplementedError

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data"""
        raise NotImplementedError


class LiquidPredictor(TimeSeriesPredictor):
    """Liquid Neural Network for time series"""

    def __init__(self, window_size: int = 10, horizon: int = 1, verbose: bool = False):
        super().__init__(window_size, horizon)
        self.verbose = verbose

        # Create small LNN optimized for time series
        config = NCPWiringConfig(
            n_sensory=8,
            n_inter=16,
            n_command=8,
            n_motor=horizon,
            sensory_to_inter_sparsity=0.6,
            inter_recurrent=True,
            command_recurrent=True
        )

        self.lnn = LiquidNeuralNetwork(
            input_dim=window_size,
            output_dim=horizon,
            wiring_config=config,
            dt=0.1,
            ode_steps=2,  # More ODE steps for better temporal dynamics
            learning_rate=0.02,
            use_cfc=False  # Use ODE for training
        )

    def train(self, data: np.ndarray, n_epochs: int = 1):
        """Train with local learning rule (NO BACKPROP!)"""
        n_samples = len(data) - self.window_size - self.horizon

        for epoch in range(n_epochs):
            epoch_errors = []

            # Reset state at start of each epoch
            self.lnn.reset_state()

            for i in range(n_samples):
                # Create window and target
                window = data[i:i + self.window_size]
                target = data[i + self.window_size:i + self.window_size + self.horizon]

                # Forward pass (continuous-time dynamics!)
                output, info = self.lnn.forward(window)

                # Compute error
                error = np.mean((output - target) ** 2)
                epoch_errors.append(error)

                # Local learning update (NO BACKPROP!)
                self.lnn.learn(target, error_weight=1.0)

            if self.verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: MSE = {np.mean(epoch_errors):.6f}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict next values"""
        output, _ = self.lnn.forward(window)
        return output

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data"""
        n_samples = len(data) - self.window_size - self.horizon
        predictions = []
        targets = []

        self.lnn.reset_state()

        for i in range(n_samples):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size:i + self.window_size + self.horizon]

            pred = self.predict(window)
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # Correlation
        if self.horizon == 1:
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        else:
            correlation = np.mean([
                np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                for i in range(self.horizon)
            ])

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'rmse': np.sqrt(mse)
        }


class ForwardForwardPredictor(TimeSeriesPredictor):
    """Forward-Forward Network for time series (experimental)"""

    def __init__(self, window_size: int = 10, horizon: int = 1, verbose: bool = False):
        super().__init__(window_size, horizon)
        self.verbose = verbose

        # Note: FF is designed for classification, adapting for regression
        # We'll discretize the output space
        self.n_bins = 20
        self.output_classes = horizon * self.n_bins

        # Not available if torch not installed
        if not TORCH_AVAILABLE:
            self.ff_net = None
            return

        try:
            self.ff_net = ForwardForwardNetwork(
                input_dim=window_size,
                hidden_dims=[64, 32],
                output_dim=self.output_classes,
                threshold=1.5,
                learning_rate=0.03,
                activation='relu',
                negative_strategy='hybrid',
                device='cpu'
            )
        except Exception as e:
            print(f"Warning: Could not create FF network: {e}")
            self.ff_net = None

        # Store data range for normalization
        self.data_min = None
        self.data_max = None

    def _discretize(self, values: np.ndarray) -> int:
        """Convert continuous value to discrete class"""
        if self.data_min is None:
            return 0
        normalized = (values[0] - self.data_min) / (self.data_max - self.data_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        return int(normalized * (self.n_bins - 1))

    def _undiscretize(self, class_idx: int) -> float:
        """Convert class back to continuous value"""
        if self.data_min is None:
            return 0.0
        normalized = class_idx / (self.n_bins - 1)
        return self.data_min + normalized * (self.data_max - self.data_min)

    def train(self, data: np.ndarray, n_epochs: int = 1):
        """Train with Forward-Forward (NO BACKPROP!)"""
        if self.ff_net is None:
            return

        self.data_min = np.min(data)
        self.data_max = np.max(data)

        n_samples = len(data) - self.window_size - self.horizon

        for epoch in range(n_epochs):
            epoch_losses = []

            for i in range(n_samples):
                window = data[i:i + self.window_size]
                target_val = data[i + self.window_size:i + self.window_size + self.horizon]

                # Convert to tensors
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor([self._discretize(target_val)], dtype=torch.long)

                # Forward-Forward training step (NO BACKPROP!)
                metrics = self.ff_net.train_step(x, y)
                epoch_losses.append(metrics['loss'])

            if self.verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: Loss = {np.mean(epoch_losses):.6f}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict next values"""
        if self.ff_net is None:
            return np.zeros(self.horizon)

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        # Get prediction
        pred_class = self.ff_net.predict(x).item()
        pred_value = self._undiscretize(pred_class % self.n_bins)

        return np.array([pred_value] * self.horizon)

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data"""
        if self.ff_net is None:
            return {'mse': float('inf'), 'mae': float('inf'), 'correlation': 0.0, 'rmse': float('inf')}

        n_samples = len(data) - self.window_size - self.horizon
        predictions = []
        targets = []

        for i in range(n_samples):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size:i + self.window_size + self.horizon]

            pred = self.predict(window)
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # Correlation
        if self.horizon == 1:
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        else:
            correlation = np.mean([
                np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                for i in range(self.horizon)
            ])

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'rmse': np.sqrt(mse)
        }


class CuriosityPredictor(TimeSeriesPredictor):
    """CuriosityCore for time series (world model prediction)"""

    def __init__(self, window_size: int = 10, horizon: int = 1, verbose: bool = False):
        super().__init__(window_size, horizon)
        self.verbose = verbose

        self.core = CuriosityCore(
            sensory_dim=window_size,
            hidden_dim=32,
            action_dim=horizon,
            device='cpu'
        )

    def train(self, data: np.ndarray, n_epochs: int = 1):
        """Train world model (NO BACKPROP!)"""
        n_samples = len(data) - self.window_size - self.horizon

        for epoch in range(n_epochs):
            epoch_errors = []

            for i in range(n_samples):
                window = data[i:i + self.window_size]
                target = data[i + self.window_size:i + self.window_size + self.horizon]

                # Step curiosity core
                result = self.core.step(window, external_reward=0.0)

                # Use world model prediction error
                epoch_errors.append(result['world_error'])

            if self.verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: World Error = {np.mean(epoch_errors):.6f}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict using action model"""
        result = self.core.step(window, external_reward=0.0)
        # Use action as prediction (adaptive output)
        return result['action'][:self.horizon]

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data"""
        n_samples = len(data) - self.window_size - self.horizon
        predictions = []
        targets = []

        for i in range(n_samples):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size:i + self.window_size + self.horizon]

            pred = self.predict(window)
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # Correlation
        if self.horizon == 1:
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        else:
            correlation = np.mean([
                np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                for i in range(self.horizon)
            ])

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'rmse': np.sqrt(mse)
        }


class SimpleARPredictor(TimeSeriesPredictor):
    """Simple Autoregressive baseline"""

    def __init__(self, window_size: int = 10, horizon: int = 1, verbose: bool = False):
        super().__init__(window_size, horizon)
        self.verbose = verbose
        self.weights = np.zeros(window_size)
        self.bias = 0.0

    def train(self, data: np.ndarray, n_epochs: int = 1):
        """Train with simple least squares"""
        n_samples = len(data) - self.window_size - self.horizon

        X = []
        y = []

        for i in range(n_samples):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size]  # Only predict 1 step
            X.append(window)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # Least squares solution
        X_with_bias = np.column_stack([X, np.ones(len(X))])
        coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

        self.weights = coeffs[:-1]
        self.bias = coeffs[-1]

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Linear prediction"""
        pred = np.dot(self.weights, window) + self.bias
        return np.array([pred] * self.horizon)

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data"""
        n_samples = len(data) - self.window_size - self.horizon
        predictions = []
        targets = []

        for i in range(n_samples):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size:i + self.window_size + self.horizon]

            pred = self.predict(window)
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # Correlation
        if self.horizon == 1:
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        else:
            correlation = np.mean([
                np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                for i in range(self.horizon)
            ])

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'rmse': np.sqrt(mse)
        }


if TORCH_AVAILABLE:
    class LSTMPredictor(TimeSeriesPredictor):
        """LSTM baseline with backpropagation"""

        def __init__(self, window_size: int = 10, horizon: int = 1, verbose: bool = False):
            super().__init__(window_size, horizon)
            self.verbose = verbose

            # Simple LSTM
            self.model = nn.LSTM(
                input_size=1,
                hidden_size=32,
                num_layers=1,
                batch_first=True
            )
            self.fc = nn.Linear(32, horizon)
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.fc.parameters()),
                lr=0.001
            )
            self.criterion = nn.MSELoss()

        def train(self, data: np.ndarray, n_epochs: int = 1):
            """Train with backpropagation"""
            n_samples = len(data) - self.window_size - self.horizon

            for epoch in range(n_epochs):
                epoch_losses = []

                for i in range(n_samples):
                    window = data[i:i + self.window_size]
                    target = data[i + self.window_size:i + self.window_size + self.horizon]

                    # Convert to tensors
                    x = torch.tensor(window, dtype=torch.float32).view(1, -1, 1)
                    y = torch.tensor(target, dtype=torch.float32).view(1, -1)

                    # Forward pass
                    self.optimizer.zero_grad()
                    lstm_out, _ = self.model(x)
                    pred = self.fc(lstm_out[:, -1, :])

                    # Backpropagation
                    loss = self.criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()

                    epoch_losses.append(loss.item())

                if self.verbose and epoch % max(1, n_epochs // 10) == 0:
                    print(f"    Epoch {epoch+1}/{n_epochs}: Loss = {np.mean(epoch_losses):.6f}")

        def predict(self, window: np.ndarray) -> np.ndarray:
            """Predict next values"""
            x = torch.tensor(window, dtype=torch.float32).view(1, -1, 1)

            with torch.no_grad():
                lstm_out, _ = self.model(x)
                pred = self.fc(lstm_out[:, -1, :])

            return pred.numpy().flatten()

        def evaluate(self, data: np.ndarray) -> Dict[str, float]:
            """Evaluate on test data"""
            n_samples = len(data) - self.window_size - self.horizon
            predictions = []
            targets = []

            for i in range(n_samples):
                window = data[i:i + self.window_size]
                target = data[i + self.window_size:i + self.window_size + self.horizon]

                pred = self.predict(window)
                predictions.append(pred)
                targets.append(target)

            predictions = np.array(predictions)
            targets = np.array(targets)

            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))

            # Correlation
            if self.horizon == 1:
                correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            else:
                correlation = np.mean([
                    np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                    for i in range(self.horizon)
                ])

            return {
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'rmse': np.sqrt(mse)
            }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(
    task_name: str,
    data: np.ndarray,
    window_size: int = 10,
    horizon: int = 1,
    train_size: float = 0.7,
    n_epochs: int = 5,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run benchmark on a time series task.

    Args:
        task_name: Name of the task
        data: Time series data
        window_size: Input window size
        horizon: Prediction horizon
        train_size: Fraction of data for training
        n_epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Dictionary of results per architecture
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Task: {task_name}")
        print(f"{'='*70}")
        print(f"Data: {len(data)} samples")
        print(f"Window size: {window_size}, Horizon: {horizon}")

    # Split data
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    if verbose:
        print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # Normalize data
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data_norm = (train_data - mean) / (std + 1e-8)
    test_data_norm = (test_data - mean) / (std + 1e-8)

    results = {}

    # Test each architecture
    architectures = [
        ('Simple AR', SimpleARPredictor(window_size, horizon, verbose=False)),
        ('Liquid Neural Net', LiquidPredictor(window_size, horizon, verbose=False)),
        ('CuriosityCore', CuriosityPredictor(window_size, horizon, verbose=False)),
    ]

    # Add Forward-Forward if available
    if TORCH_AVAILABLE:
        architectures.append(
            ('Forward-Forward', ForwardForwardPredictor(window_size, horizon, verbose=False))
        )
        architectures.append(
            ('LSTM (baseline)', LSTMPredictor(window_size, horizon, verbose=False))
        )

    for name, predictor in architectures:
        if verbose:
            print(f"\n--- {name} ---")

        try:
            # Train
            start_time = time.time()
            if verbose:
                print(f"  Training...")
            predictor.train(train_data_norm, n_epochs=n_epochs)
            train_time = time.time() - start_time

            # Evaluate
            if verbose:
                print(f"  Evaluating...")
            start_time = time.time()
            metrics = predictor.evaluate(test_data_norm)
            test_time = time.time() - start_time

            metrics['train_time'] = train_time
            metrics['test_time'] = test_time

            # Check if uses backprop
            uses_backprop = name in ['LSTM (baseline)']
            metrics['backprop'] = uses_backprop

            results[name] = metrics

            if verbose:
                print(f"  Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
                print(f"  MSE: {metrics['mse']:.6f}, Correlation: {metrics['correlation']:.4f}")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[name] = {
                'mse': float('inf'),
                'mae': float('inf'),
                'correlation': 0.0,
                'rmse': float('inf'),
                'train_time': 0.0,
                'test_time': 0.0,
                'backprop': False
            }

    return results


def print_results_table(all_results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print formatted results table"""
    print(f"\n{'='*90}")
    print("TIME SERIES BENCHMARK - BIO-PLAUSIBLE LEARNING")
    print(f"{'='*90}\n")

    for task_name, results in all_results.items():
        print(f"\nTask: {task_name}")
        print("-" * 90)
        print(f"{'Architecture':<25} {'Train MSE':>12} {'Test MSE':>12} {'Correlation':>12} {'Backprop?':>10}")
        print("-" * 90)

        # Sort by test MSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])

        for name, metrics in sorted_results:
            backprop_str = "YES" if metrics.get('backprop', False) else "NO"
            print(f"{name:<25} {metrics.get('mse', 0):<12.6f} {metrics['mse']:<12.6f} "
                  f"{metrics['correlation']:>12.4f} {backprop_str:>10}")

        print()


def main():
    """Run full benchmark suite"""
    print("\n" + "="*90)
    print("REAL-WORLD TIME SERIES BENCHMARK")
    print("Testing Bio-Plausible Architectures on Actual Temporal Data")
    print("="*90)

    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    all_results = {}

    # Parameters - REDUCED FOR FASTER BENCHMARK
    window_size = 10
    horizon = 5
    n_epochs = 3  # Reduced from 10

    # ========================================================================
    # 1. MACKEY-GLASS (Classic Benchmark)
    # ========================================================================

    print("\n[1/3] Generating Mackey-Glass time series...")
    mg_data = generate_mackey_glass(n_samples=1000, tau=17)  # Reduced from 3000

    results_mg = run_benchmark(
        task_name=f"Mackey-Glass Prediction (horizon={horizon})",
        data=mg_data,
        window_size=window_size,
        horizon=horizon,
        n_epochs=n_epochs,
        verbose=True
    )
    all_results["Mackey-Glass"] = results_mg

    # ========================================================================
    # 2. LORENZ ATTRACTOR
    # ========================================================================

    print("\n[2/3] Generating Lorenz attractor...")
    lorenz_data = generate_lorenz(n_samples=1000, dt=0.01)  # Reduced from 3000
    # Use just x-coordinate for simplicity
    lorenz_x = lorenz_data[:, 0]

    results_lorenz = run_benchmark(
        task_name=f"Lorenz Attractor X (horizon={horizon})",
        data=lorenz_x,
        window_size=window_size,
        horizon=horizon,
        n_epochs=n_epochs,
        verbose=True
    )
    all_results["Lorenz"] = results_lorenz

    # ========================================================================
    # 3. STOCK-LIKE DATA
    # ========================================================================

    print("\n[3/3] Generating stock-like price series...")
    stock_data = generate_stock_like(n_samples=1000, trend_strength=0.0003, volatility=0.02)  # Reduced from 3000
    # Use log returns for stationarity
    log_returns = np.diff(np.log(stock_data))

    results_stock = run_benchmark(
        task_name=f"Stock-like Returns (horizon={horizon})",
        data=log_returns,
        window_size=window_size,
        horizon=horizon,
        n_epochs=n_epochs,
        verbose=True
    )
    all_results["Stock-like"] = results_stock

    # ========================================================================
    # PRINT SUMMARY TABLE
    # ========================================================================

    print_results_table(all_results)

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print("\n" + "="*90)
    print("ANALYSIS")
    print("="*90)

    # Check if Liquid Networks shine on temporal tasks
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")

        liquid_mse = results.get('Liquid Neural Net', {}).get('mse', float('inf'))

        # Compare with baselines
        ar_mse = results.get('Simple AR', {}).get('mse', float('inf'))

        if liquid_mse < ar_mse:
            improvement = ((ar_mse - liquid_mse) / ar_mse) * 100
            print(f"  ✓ Liquid Network beats Simple AR by {improvement:.1f}%")
        else:
            print(f"  ✗ Liquid Network does not beat Simple AR")

        # Compare with LSTM if available
        if 'LSTM (baseline)' in results:
            lstm_mse = results['LSTM (baseline)']['mse']
            if liquid_mse < lstm_mse * 1.5:  # Within 50% of LSTM
                print(f"  ✓ Liquid Network competitive with LSTM (no backprop!)")
            else:
                ratio = liquid_mse / lstm_mse
                print(f"  → Liquid Network {ratio:.2f}x worse than LSTM")

        # Check correlation
        liquid_corr = results.get('Liquid Neural Net', {}).get('correlation', 0)
        if liquid_corr > 0.8:
            print(f"  ✓ High correlation: {liquid_corr:.3f} (good temporal learning)")
        elif liquid_corr > 0.5:
            print(f"  ~ Medium correlation: {liquid_corr:.3f}")
        else:
            print(f"  ✗ Low correlation: {liquid_corr:.3f}")

    print("\n" + "="*90)
    print("BENCHMARK COMPLETE!")
    print("="*90)
    print("\nKey Findings:")
    print("- Liquid Networks are DESIGNED for time series (continuous-time dynamics)")
    print("- NO BACKPROPAGATION used for bio-plausible methods")
    print("- Compare performance on REAL chaotic/temporal data (not toy sine waves)")
    print("- Check if claimed advantages (temporal learning, small networks) hold up")
    print("\nThis is a REAL benchmark on actual challenging time series tasks!")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
