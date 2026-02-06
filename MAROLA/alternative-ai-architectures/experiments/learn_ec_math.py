"""
SECP256K1 ELLIPTIC CURVE MATH LEARNING - THE ULTIMATE BIO-PLAUSIBLE TEST
=========================================================================

Can bio-plausible neural networks learn actual cryptographic mathematics?
This is the ULTIMATE challenge - learning discrete modular arithmetic and
elliptic curve operations WITHOUT backpropagation.

The discontinuity of modular wrap-around and the complexity of point addition
represent a fundamentally different challenge from continuous function approximation.

Tasks (increasing difficulty):
-------------------------------
Level 1: Basic Modular Arithmetic (p=97, 997, 7919)
  - Modular Addition: (a + b) mod p
  - Modular Subtraction: (a - b) mod p
  - Modular Multiplication: (a * b) mod p

Level 2: Field Operations
  - Modular Inverse: a^(-1) mod p (hardest)
  - Modular Exponentiation: a^e mod p

Level 3: Elliptic Curve Point Operations
  - Point Validation: Is (x,y) on curve y^2 = x^3 + 7 (mod p)?
  - Point Addition: P1 + P2 = P3 (if time permits)

Key Challenge: Can networks learn the "wrap-around" discontinuity of modular arithmetic?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import time
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Note: Forward-Forward requires adaptation for regression tasks, so we'll use
# a custom bio-plausible implementation based on Hebbian learning


# ============================================================================
# DATA GENERATION: Cryptographic Math Operations
# ============================================================================

class ECMathDataGenerator:
    """Generate training data for elliptic curve mathematics."""

    def __init__(self, prime: int = 997):
        """
        Args:
            prime: The modulus prime number (start small, scale up)
        """
        self.p = prime

    def generate_mod_add(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate modular addition data: (a + b) mod p

        Returns:
            inputs: [n_samples, 2] normalized to [0, 1]
            targets: [n_samples, 1] normalized to [0, 1]
        """
        a = np.random.randint(0, self.p, n_samples)
        b = np.random.randint(0, self.p, n_samples)
        result = (a + b) % self.p

        # Normalize to [0, 1] for neural networks
        inputs = np.stack([a / self.p, b / self.p], axis=1).astype(np.float32)
        targets = (result / self.p).astype(np.float32).reshape(-1, 1)

        return inputs, targets

    def generate_mod_sub(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate modular subtraction: (a - b) mod p"""
        a = np.random.randint(0, self.p, n_samples)
        b = np.random.randint(0, self.p, n_samples)
        result = (a - b) % self.p

        inputs = np.stack([a / self.p, b / self.p], axis=1).astype(np.float32)
        targets = (result / self.p).astype(np.float32).reshape(-1, 1)

        return inputs, targets

    def generate_mod_mult(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate modular multiplication: (a * b) mod p

        HARDER - More complex wrap-around behavior than addition.
        """
        a = np.random.randint(0, self.p, n_samples)
        b = np.random.randint(0, self.p, n_samples)
        result = (a * b) % self.p

        inputs = np.stack([a / self.p, b / self.p], axis=1).astype(np.float32)
        targets = (result / self.p).astype(np.float32).reshape(-1, 1)

        return inputs, targets

    def generate_mod_inverse(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate modular inverse: a^(-1) mod p

        HARDEST arithmetic operation - requires Extended Euclidean Algorithm.
        Only defined for gcd(a, p) = 1.
        """
        # Generate valid inputs (coprime to p)
        valid_a = []
        while len(valid_a) < n_samples:
            a = np.random.randint(1, self.p)  # Start from 1
            if np.gcd(a, self.p) == 1:
                valid_a.append(a)

        a = np.array(valid_a)
        # Compute modular inverse using Fermat's little theorem: a^(p-2) mod p
        result = np.array([pow(int(ai), self.p - 2, self.p) for ai in a])

        inputs = (a / self.p).astype(np.float32).reshape(-1, 1)
        targets = (result / self.p).astype(np.float32).reshape(-1, 1)

        return inputs, targets

    def generate_mod_exp(self, n_samples: int, max_exp: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate modular exponentiation: a^e mod p"""
        a = np.random.randint(1, self.p, n_samples)
        e = np.random.randint(1, max_exp + 1, n_samples)
        result = np.array([pow(int(ai), int(ei), self.p) for ai, ei in zip(a, e)])

        inputs = np.stack([a / self.p, e / max_exp], axis=1).astype(np.float32)
        targets = (result / self.p).astype(np.float32).reshape(-1, 1)

        return inputs, targets

    def generate_point_validation(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point validation data: Is (x, y) on curve y^2 = x^3 + 7 (mod p)?

        Returns binary classification: 1 if on curve, 0 otherwise.
        """
        inputs = []
        targets = []

        # Generate half valid points, half invalid
        for i in range(n_samples):
            if i < n_samples // 2:
                # Generate valid point on curve
                x = np.random.randint(0, self.p)
                y_squared = (pow(x, 3, self.p) + 7) % self.p

                # Check if y_squared is a quadratic residue
                y = self._tonelli_shanks(y_squared, self.p)
                if y is not None:
                    inputs.append([x / self.p, y / self.p])
                    targets.append(1.0)
                else:
                    # If not a QR, generate random invalid point
                    y = np.random.randint(0, self.p)
                    inputs.append([x / self.p, y / self.p])
                    targets.append(0.0)
            else:
                # Generate invalid point
                x = np.random.randint(0, self.p)
                y = np.random.randint(0, self.p)

                # Verify it's actually invalid
                lhs = (y * y) % self.p
                rhs = (pow(x, 3, self.p) + 7) % self.p

                if lhs == rhs:
                    # Accidentally valid, mark as such
                    targets.append(1.0)
                else:
                    targets.append(0.0)

                inputs.append([x / self.p, y / self.p])

        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

        return inputs, targets

    def _tonelli_shanks(self, n: int, p: int) -> int:
        """
        Compute square root mod p using Tonelli-Shanks algorithm.
        Returns None if n is not a quadratic residue.
        """
        # Check if n is a quadratic residue
        if pow(n, (p - 1) // 2, p) != 1:
            return None

        # Simple case
        if p % 4 == 3:
            return pow(n, (p + 1) // 4, p)

        # Find Q and S such that p - 1 = Q * 2^S
        Q = p - 1
        S = 0
        while Q % 2 == 0:
            Q //= 2
            S += 1

        # Find a quadratic non-residue z
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        # Initialize
        M = S
        c = pow(z, Q, p)
        t = pow(n, Q, p)
        R = pow(n, (Q + 1) // 2, p)

        while True:
            if t == 0:
                return 0
            if t == 1:
                return R

            # Find lowest i such that t^(2^i) = 1
            i = 1
            temp = (t * t) % p
            while temp != 1:
                temp = (temp * temp) % p
                i += 1

            # Update
            b = pow(c, 1 << (M - i - 1), p)
            M = i
            c = (b * b) % p
            t = (t * c) % p
            R = (R * b) % p


# ============================================================================
# BASELINE MLP WITH BACKPROPAGATION (for comparison)
# ============================================================================

class MLPBaseline(nn.Module):
    """Standard MLP with backpropagation - the comparison baseline."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# LIQUID NEURAL NETWORK ADAPTER (simplified)
# ============================================================================

class SimplifiedLiquidNetwork(nn.Module):
    """Simplified Liquid Neural Network for regression tasks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Liquid cell parameters
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.5)  # Time constants
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # State
        self.state = None

    def forward(self, x, steps: int = 10):
        batch_size = x.size(0)

        # Initialize state
        if self.state is None or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Input encoding
        u = self.input_proj(x)

        # Evolve liquid state
        for _ in range(steps):
            # Continuous-time dynamics (Euler integration)
            activation = torch.tanh(self.W @ self.state.T + self.b.unsqueeze(1)).T
            dstate = (-self.state + activation + u) / (self.tau + 1e-6)
            self.state = self.state + 0.1 * dstate  # dt = 0.1

        # Output
        return self.output_proj(self.state)

    def reset_state(self):
        """Reset hidden state."""
        self.state = None


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray, prime: int) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted values (normalized [0, 1])
        targets: True values (normalized [0, 1])
        prime: Modulus prime for denormalization

    Returns:
        Dictionary of metrics
    """
    # Denormalize to integer space
    pred_int = np.round(predictions * prime).astype(int) % prime
    target_int = np.round(targets * prime).astype(int) % prime

    # Exact accuracy: predicted value exactly matches target
    exact_matches = (pred_int == target_int).astype(float)
    exact_acc = np.mean(exact_matches)

    # Mean Absolute Error (in normalized space)
    mae = np.mean(np.abs(predictions - targets))

    # "Close enough" accuracy: within 1% of prime
    threshold = max(1, int(0.01 * prime))  # At least 1
    close_matches = (np.abs(pred_int - target_int) <= threshold).astype(float)
    close_acc = np.mean(close_matches)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return {
        'exact_accuracy': exact_acc,
        'mae': mae,
        'close_accuracy': close_acc,
        'rmse': rmse
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_mlp_baseline(
    model: MLPBaseline,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Train MLP with backpropagation."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Convert to tensors
    train_inputs_t = torch.FloatTensor(train_inputs).to(device)
    train_targets_t = torch.FloatTensor(train_targets).to(device)
    test_inputs_t = torch.FloatTensor(test_inputs).to(device)
    test_targets_t = torch.FloatTensor(test_targets).to(device)

    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Mini-batch training
        indices = np.random.permutation(len(train_inputs))
        for i in range(0, len(train_inputs), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_inputs = train_inputs_t[batch_idx]
            batch_targets = train_targets_t[batch_idx]

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / (len(train_inputs) // batch_size))

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_pred = model(test_inputs_t)
            test_loss = criterion(test_pred, test_targets_t).item()
            test_losses.append(test_loss)

    train_time = time.time() - start_time

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(test_inputs_t).cpu().numpy()

    return {
        'predictions': test_pred,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_time': train_time
    }


def train_liquid_network(
    model: SimplifiedLiquidNetwork,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Train Liquid Neural Network."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Convert to tensors
    train_inputs_t = torch.FloatTensor(train_inputs).to(device)
    train_targets_t = torch.FloatTensor(train_targets).to(device)
    test_inputs_t = torch.FloatTensor(test_inputs).to(device)
    test_targets_t = torch.FloatTensor(test_targets).to(device)

    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        model.reset_state()
        epoch_loss = 0.0

        # Mini-batch training
        indices = np.random.permutation(len(train_inputs))
        for i in range(0, len(train_inputs), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_inputs = train_inputs_t[batch_idx]
            batch_targets = train_targets_t[batch_idx]

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Reset state between batches
            model.reset_state()

        train_losses.append(epoch_loss / (len(train_inputs) // batch_size))

        # Evaluate on test set
        model.eval()
        model.reset_state()
        with torch.no_grad():
            test_pred = model(test_inputs_t)
            test_loss = criterion(test_pred, test_targets_t).item()
            test_losses.append(test_loss)

    train_time = time.time() - start_time

    # Final evaluation
    model.eval()
    model.reset_state()
    with torch.no_grad():
        test_pred = model(test_inputs_t).cpu().numpy()

    return {
        'predictions': test_pred,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_time': train_time
    }


def train_bio_plausible_network(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    epochs: int = 100,
    hidden_size: int = 128
) -> Dict[str, Any]:
    """
    Train a bio-plausible network using Hebbian learning + local error signals.

    NO backpropagation - only local learning rules!

    Uses:
    - Hebbian learning: neurons that fire together, wire together
    - Local error signals: each layer only sees its own prediction error
    - Anti-Hebbian for error correction
    """

    input_dim = train_inputs.shape[1]
    output_dim = train_targets.shape[1]

    # Initialize weights with small random values
    # Layer 1: input -> hidden
    W1 = np.random.randn(input_dim, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)

    # Layer 2: hidden -> output
    W2 = np.random.randn(hidden_size, output_dim) * 0.01
    b2 = np.zeros(output_dim)

    # Learning rates
    lr_hebbian = 0.01  # Hebbian learning rate
    lr_error = 0.02    # Error-driven learning rate

    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle training data
        indices = np.random.permutation(len(train_inputs))

        for idx in indices:
            x = train_inputs[idx:idx+1]
            target = train_targets[idx:idx+1]

            # === FORWARD PASS ===
            # Layer 1
            z1 = x @ W1 + b1
            h1 = np.tanh(z1)  # Hidden activations

            # Layer 2
            z2 = h1 @ W2 + b2
            pred = np.tanh(z2)  # Output prediction

            # Compute error
            error_out = target - pred
            loss = np.mean(error_out ** 2)
            epoch_loss += loss

            # === LOCAL LEARNING (NO BACKPROP!) ===

            # Output layer update (local error signal)
            # Anti-Hebbian: reduce connections that produce error
            # Hebbian: strengthen connections that reduce error
            # Derivative of tanh: 1 - tanh^2
            grad_out = error_out * (1 - pred ** 2)

            # Update W2 and b2 using LOCAL information only
            dW2 = lr_error * (h1.T @ grad_out)
            db2 = lr_error * np.sum(grad_out, axis=0)

            W2 += dW2
            b2 += db2

            # Hidden layer update (even more local)
            # Use output error as modulatory signal, but don't backpropagate it
            # Instead, use local Hebbian learning modulated by prediction accuracy

            # Hebbian component: strengthen active connections
            # When prediction is good (low error), reinforce current patterns
            success_signal = 1.0 / (1.0 + np.abs(error_out.mean()))  # Higher when error is low
            hebbian_update = lr_hebbian * success_signal * (x.T @ h1)

            # Anti-Hebbian component: weaken when error is high
            error_signal = np.abs(error_out.mean())  # Higher when error is high
            anti_hebbian_update = -lr_hebbian * error_signal * (x.T @ (h1 * np.sign(h1)))

            # Combine updates
            dW1 = hebbian_update + anti_hebbian_update
            db1 = lr_hebbian * (success_signal * np.sum(h1, axis=0) -
                                 error_signal * np.sum(np.abs(h1), axis=0))

            W1 += dW1
            b1 += db1

            # Light weight decay (regularization)
            W1 *= 0.9999
            W2 *= 0.9999

        train_losses.append(epoch_loss / len(train_inputs))

        # Evaluate on test set
        test_loss = 0.0
        test_predictions = []

        for i in range(len(test_inputs)):
            x = test_inputs[i:i+1]
            target = test_targets[i:i+1]

            # Forward pass
            h1 = np.tanh(x @ W1 + b1)
            pred = np.tanh(h1 @ W2 + b2)

            test_predictions.append(pred)

            error = target - pred
            loss = np.mean(error ** 2)
            test_loss += loss

        test_losses.append(test_loss / len(test_inputs))

    train_time = time.time() - start_time

    # Final predictions
    test_pred = np.vstack(test_predictions)

    return {
        'predictions': test_pred,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_time': train_time
    }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_task_benchmark(
    task_name: str,
    prime: int,
    n_train: int = 5000,
    n_test: int = 1000,
    epochs: int = 50,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Run benchmark for a specific cryptographic task.

    Args:
        task_name: Name of the task (e.g., 'mod_add', 'mod_mult')
        prime: Modulus prime
        n_train: Number of training samples
        n_test: Number of test samples
        epochs: Training epochs
        device: Device to use

    Returns:
        Dictionary of results for all architectures
    """

    print(f"\n{'='*70}")
    print(f"TASK: {task_name.upper()} (p={prime})")
    print(f"{'='*70}\n")

    # Generate data
    generator = ECMathDataGenerator(prime=prime)

    if task_name == 'mod_add':
        train_inputs, train_targets = generator.generate_mod_add(n_train)
        test_inputs, test_targets = generator.generate_mod_add(n_test)
        input_dim = 2
        output_dim = 1
    elif task_name == 'mod_sub':
        train_inputs, train_targets = generator.generate_mod_sub(n_train)
        test_inputs, test_targets = generator.generate_mod_sub(n_test)
        input_dim = 2
        output_dim = 1
    elif task_name == 'mod_mult':
        train_inputs, train_targets = generator.generate_mod_mult(n_train)
        test_inputs, test_targets = generator.generate_mod_mult(n_test)
        input_dim = 2
        output_dim = 1
    elif task_name == 'mod_inverse':
        train_inputs, train_targets = generator.generate_mod_inverse(n_train)
        test_inputs, test_targets = generator.generate_mod_inverse(n_test)
        input_dim = 1
        output_dim = 1
    elif task_name == 'mod_exp':
        train_inputs, train_targets = generator.generate_mod_exp(n_train)
        test_inputs, test_targets = generator.generate_mod_exp(n_test)
        input_dim = 2
        output_dim = 1
    elif task_name == 'point_validation':
        train_inputs, train_targets = generator.generate_point_validation(n_train)
        test_inputs, test_targets = generator.generate_point_validation(n_test)
        input_dim = 2
        output_dim = 1
    else:
        raise ValueError(f"Unknown task: {task_name}")

    results = {}

    # ========================================
    # 1. MLP Baseline (with backprop)
    # ========================================
    print(f"Training MLP Baseline...")
    mlp = MLPBaseline(input_dim=input_dim, hidden_dims=[128, 128, 64], output_dim=output_dim)
    mlp_results = train_mlp_baseline(
        mlp, train_inputs, train_targets, test_inputs, test_targets,
        epochs=epochs, device=device
    )
    mlp_metrics = compute_metrics(mlp_results['predictions'], test_targets, prime)
    mlp_metrics['train_time'] = mlp_results['train_time']
    mlp_metrics['backprop'] = True
    results['MLP (Backprop)'] = mlp_metrics
    print(f"  Exact Acc: {mlp_metrics['exact_accuracy']:.3f}, MAE: {mlp_metrics['mae']:.4f}")

    # ========================================
    # 2. Liquid Neural Network
    # ========================================
    print(f"Training Liquid Neural Network...")
    liquid = SimplifiedLiquidNetwork(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
    liquid_results = train_liquid_network(
        liquid, train_inputs, train_targets, test_inputs, test_targets,
        epochs=epochs, device=device
    )
    liquid_metrics = compute_metrics(liquid_results['predictions'], test_targets, prime)
    liquid_metrics['train_time'] = liquid_results['train_time']
    liquid_metrics['backprop'] = True  # Uses backprop but different dynamics
    results['Liquid Network'] = liquid_metrics
    print(f"  Exact Acc: {liquid_metrics['exact_accuracy']:.3f}, MAE: {liquid_metrics['mae']:.4f}")

    # ========================================
    # 3. Bio-Plausible Network (Hebbian + Local Error)
    # ========================================
    print(f"Training Bio-Plausible Network (Hebbian + Local Error)...")
    bio_results = train_bio_plausible_network(
        train_inputs, train_targets, test_inputs, test_targets,
        epochs=epochs, hidden_size=128
    )
    bio_metrics = compute_metrics(bio_results['predictions'], test_targets, prime)
    bio_metrics['train_time'] = bio_results['train_time']
    bio_metrics['backprop'] = False
    results['Bio-Plausible (Hebbian)'] = bio_metrics
    print(f"  Exact Acc: {bio_metrics['exact_accuracy']:.3f}, MAE: {bio_metrics['mae']:.4f}")

    # ========================================
    # 4. Forward-Forward Network (if available)
    # ========================================
    # Note: Forward-Forward is designed for classification, not regression
    # We'll skip it for now as it requires significant adaptation

    return results


def run_full_benchmark(device: str = 'cuda') -> Dict[str, Any]:
    """Run the complete benchmark suite across all tasks and scales."""

    print("\n" + "="*70)
    print("SECP256K1 MATH LEARNING - BIO-PLAUSIBLE NETWORKS")
    print("="*70)
    print("\nTesting whether bio-plausible networks can learn cryptographic math")
    print("Key Challenge: Discrete modular arithmetic with wrap-around discontinuity")
    print("="*70 + "\n")

    all_results = {}

    # ========================================
    # LEVEL 1: Basic Modular Arithmetic
    # ========================================
    print("\n" + "="*70)
    print("LEVEL 1: BASIC MODULAR ARITHMETIC")
    print("="*70)

    # Test with increasing prime sizes
    primes = [97, 997]  # Start small, scale up
    tasks_l1 = ['mod_add', 'mod_sub', 'mod_mult']

    for prime in primes:
        for task in tasks_l1:
            key = f"{task}_p{prime}"
            all_results[key] = run_task_benchmark(
                task_name=task,
                prime=prime,
                n_train=5000,
                n_test=1000,
                epochs=50,
                device=device
            )

    # ========================================
    # LEVEL 2: Field Operations
    # ========================================
    print("\n" + "="*70)
    print("LEVEL 2: FIELD OPERATIONS (HARDER)")
    print("="*70)

    tasks_l2 = ['mod_inverse', 'mod_exp']

    for prime in [97, 997]:
        for task in tasks_l2:
            key = f"{task}_p{prime}"
            all_results[key] = run_task_benchmark(
                task_name=task,
                prime=prime,
                n_train=5000,
                n_test=1000,
                epochs=75,  # More epochs for harder tasks
                device=device
            )

    # ========================================
    # LEVEL 3: Elliptic Curve Operations
    # ========================================
    print("\n" + "="*70)
    print("LEVEL 3: ELLIPTIC CURVE POINT OPERATIONS")
    print("="*70)

    for prime in [97, 997]:
        key = f"point_validation_p{prime}"
        all_results[key] = run_task_benchmark(
            task_name='point_validation',
            prime=prime,
            n_train=5000,
            n_test=1000,
            epochs=50,
            device=device
        )

    return all_results


def print_summary_report(results: Dict[str, Any]):
    """Print comprehensive summary report."""

    print("\n" + "="*70)
    print("FINAL SUMMARY REPORT")
    print("="*70 + "\n")

    # Group by task type
    task_groups = {
        'Level 1: Modular Addition': [],
        'Level 1: Modular Subtraction': [],
        'Level 1: Modular Multiplication': [],
        'Level 2: Modular Inverse': [],
        'Level 2: Modular Exponentiation': [],
        'Level 3: Point Validation': []
    }

    for key, result in results.items():
        if 'mod_add' in key:
            task_groups['Level 1: Modular Addition'].append((key, result))
        elif 'mod_sub' in key:
            task_groups['Level 1: Modular Subtraction'].append((key, result))
        elif 'mod_mult' in key:
            task_groups['Level 1: Modular Multiplication'].append((key, result))
        elif 'mod_inverse' in key:
            task_groups['Level 2: Modular Inverse'].append((key, result))
        elif 'mod_exp' in key:
            task_groups['Level 2: Modular Exponentiation'].append((key, result))
        elif 'point_validation' in key:
            task_groups['Level 3: Point Validation'].append((key, result))

    # Print each group
    for group_name, group_results in task_groups.items():
        if not group_results:
            continue

        print(f"\n{group_name}")
        print("-" * 70)

        for key, result in group_results:
            print(f"\n{key}:")
            print(f"{'Architecture':<25} {'Exact Acc':<12} {'MAE':<12} {'Close Acc':<12} {'Backprop?':<10}")
            print("-" * 70)

            for arch_name, metrics in result.items():
                backprop = "YES" if metrics['backprop'] else "NO"
                print(f"{arch_name:<25} {metrics['exact_accuracy']:>10.3f}  {metrics['mae']:>10.4f}  "
                      f"{metrics['close_accuracy']:>10.3f}  {backprop:<10}")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70 + "\n")

    print("1. CAN BIO-PLAUSIBLE NETWORKS LEARN MODULAR ARITHMETIC?")
    print("   - CuriosityCore uses NO backpropagation")
    print("   - Compare its accuracy to MLP baseline")
    print("   - The wrap-around discontinuity is the key challenge\n")

    print("2. DOES ACCURACY DEGRADE WITH LARGER PRIMES?")
    print("   - Compare p=97 vs p=997 results")
    print("   - Larger primes = more complex modular space\n")

    print("3. WHICH OPERATIONS ARE HARDEST?")
    print("   - Addition: Simple wrap-around")
    print("   - Multiplication: Complex wrap-around patterns")
    print("   - Inverse: Requires Extended Euclidean Algorithm logic\n")

    print("4. CAN NETWORKS LEARN ELLIPTIC CURVE GEOMETRY?")
    print("   - Point validation tests understanding of y^2 = x^3 + 7")
    print("   - This is the foundation for ECDSA signatures\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EC Math Learning Benchmark')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Run benchmark
    if args.quick:
        # Quick test with small prime
        print("\nQUICK TEST MODE\n")
        results = run_task_benchmark(
            task_name='mod_add',
            prime=97,
            n_train=1000,
            n_test=200,
            epochs=20,
            device=args.device
        )
        print("\nQuick test complete!")
    else:
        # Full benchmark
        results = run_full_benchmark(device=args.device)

        # Print summary
        print_summary_report(results)

        # Save results
        output_dir = Path(__file__).parent
        output_file = output_dir / 'ec_math_learning_results.json'

        # Convert to JSON-serializable format
        results_serializable = {}
        for key, result in results.items():
            results_serializable[key] = {
                arch: {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                       for k, v in metrics.items()}
                for arch, metrics in result.items()
            }

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n Results saved to: {output_file}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
