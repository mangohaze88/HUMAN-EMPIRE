"""
COMPREHENSIVE ELLIPTIC CURVE MATH TRAINING SYSTEM

Trains neural networks on ALL operations needed for EC cryptography:
1. Modular Arithmetic (7 operations)
2. Elliptic Curve Point Operations (5 operations)
3. Key Generation (1 operation)

Tests BOTH standard backprop and bio-plausible learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Callable
import time
from dataclasses import dataclass

# ============================================================================
# ELLIPTIC CURVE MATH PRIMITIVES
# ============================================================================

def mod_inverse(a: int, p: int) -> int:
    """Extended Euclidean algorithm for modular inverse."""
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        ratio = high // low
        nm, new = hm - lm * ratio, high - low * ratio
        lm, low, hm, high = nm, new, lm, low
    return lm % p

def tonelli_shanks(n: int, p: int) -> int:
    """Modular square root using Tonelli-Shanks algorithm."""
    if pow(n, (p - 1) // 2, p) != 1:
        return None  # No square root exists

    # Find Q and S such that p - 1 = Q * 2^S
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    # Find a quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)

    while True:
        if t == 0:
            return 0
        if t == 1:
            return R

        # Find the least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1 and i < M:
            temp = (temp * temp) % p
            i += 1

        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p

def point_add(P1: Tuple[int, int], P2: Tuple[int, int], p: int, a: int = 0) -> Tuple[int, int]:
    """Add two points on elliptic curve y² = x³ + ax + b (mod p)."""
    if P1 is None:
        return P2
    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    if x1 == x2:
        if y1 == y2:
            # Point doubling
            s = (3 * x1 * x1 + a) * mod_inverse(2 * y1, p) % p
        else:
            # Points are inverses
            return None
    else:
        # Point addition
        s = (y2 - y1) * mod_inverse(x2 - x1, p) % p

    x3 = (s * s - x1 - x2) % p
    y3 = (s * (x1 - x3) - y1) % p

    return (x3, y3)

def scalar_mult(k: int, P: Tuple[int, int], p: int, a: int = 0) -> Tuple[int, int]:
    """Scalar multiplication using double-and-add."""
    if k == 0:
        return None
    if k == 1:
        return P

    result = None
    addend = P

    while k:
        if k & 1:
            result = point_add(result, addend, p, a)
        addend = point_add(addend, addend, p, a)
        k >>= 1

    return result

def is_on_curve(x: int, y: int, p: int, a: int = 0, b: int = 7) -> bool:
    """Check if point (x, y) is on curve y² = x³ + ax + b (mod p)."""
    return (y * y) % p == (x * x * x + a * x + b) % p

def find_curve_point(x: int, p: int, a: int = 0, b: int = 7) -> Tuple[int, int]:
    """Find a point on the curve with given x coordinate."""
    y_squared = (x * x * x + a * x + b) % p
    y = tonelli_shanks(y_squared, p)
    if y is None:
        return None
    return (x, y)

def generate_curve_points(n: int, p: int, a: int = 0, b: int = 7) -> List[Tuple[int, int]]:
    """Generate n random points on the curve."""
    points = []
    attempts = 0
    max_attempts = n * 10

    while len(points) < n and attempts < max_attempts:
        x = np.random.randint(0, p)
        point = find_curve_point(x, p, a, b)
        if point is not None:
            points.append(point)
        attempts += 1

    return points

# ============================================================================
# DATA GENERATION FOR EACH OPERATION
# ============================================================================

class OperationGenerator:
    """Generates training data for EC operations."""

    @staticmethod
    def mod_add(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(0, p, n_samples)
        b = np.random.randint(0, p, n_samples)
        result = (a + b) % p
        return np.stack([a, b], axis=1), result

    @staticmethod
    def mod_sub(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(0, p, n_samples)
        b = np.random.randint(0, p, n_samples)
        result = (a - b) % p
        return np.stack([a, b], axis=1), result

    @staticmethod
    def mod_mult(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(0, p, n_samples)
        b = np.random.randint(0, p, n_samples)
        result = (a * b) % p
        return np.stack([a, b], axis=1), result

    @staticmethod
    def mod_div(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(0, p, n_samples)
        b = np.random.randint(1, p, n_samples)  # Avoid division by zero
        result = np.array([(a[i] * mod_inverse(int(b[i]), p)) % p for i in range(n_samples)])
        return np.stack([a, b], axis=1), result

    @staticmethod
    def mod_inv(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(1, p, n_samples)  # Exclude 0
        result = np.array([mod_inverse(int(x), p) for x in a])
        return a.reshape(-1, 1), result

    @staticmethod
    def mod_exp(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        a = np.random.randint(0, p, n_samples)
        e = np.random.randint(0, min(p, 20), n_samples)  # Keep exponents reasonable
        result = np.array([pow(int(a[i]), int(e[i]), p) for i in range(n_samples)])
        return np.stack([a, e], axis=1), result

    @staticmethod
    def mod_sqrt(n_samples: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        # Only generate quadratic residues
        results = []
        inputs = []
        while len(results) < n_samples:
            a = np.random.randint(0, p)
            sqrt = tonelli_shanks(a, p)
            if sqrt is not None:
                inputs.append(a)
                results.append(sqrt)
        return np.array(inputs).reshape(-1, 1), np.array(results)

    @staticmethod
    def point_validation(n_samples: int, p: int, a: int = 0, b: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        # Half valid, half invalid points
        n_valid = n_samples // 2
        n_invalid = n_samples - n_valid

        valid_points = generate_curve_points(n_valid, p, a, b)

        # Generate invalid points
        invalid_points = []
        for _ in range(n_invalid):
            x = np.random.randint(0, p)
            y = np.random.randint(0, p)
            if not is_on_curve(x, y, p, a, b):
                invalid_points.append((x, y))

        # Pad if needed
        while len(invalid_points) < n_invalid:
            x, y = np.random.randint(0, p, 2)
            invalid_points.append((x, y))

        all_points = valid_points + invalid_points
        labels = np.array([1] * n_valid + [0] * n_invalid)

        # Shuffle
        indices = np.random.permutation(n_samples)
        points_array = np.array(all_points)[indices]
        labels = labels[indices]

        return points_array, labels

    @staticmethod
    def point_add_op(n_samples: int, p: int, a: int = 0, b: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        points = generate_curve_points(n_samples * 2, p, a, b)
        if len(points) < n_samples * 2:
            raise ValueError(f"Could not generate enough points for p={p}")

        P1 = points[:n_samples]
        P2 = points[n_samples:2*n_samples]

        results = []
        inputs = []
        for p1, p2 in zip(P1, P2):
            result = point_add(p1, p2, p, a)
            if result is not None:  # Skip point at infinity
                results.append(result)
                inputs.append([p1[0], p1[1], p2[0], p2[1]])

        return np.array(inputs), np.array(results)

    @staticmethod
    def point_double(n_samples: int, p: int, a: int = 0, b: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        points = generate_curve_points(n_samples, p, a, b)

        results = []
        inputs = []
        for point in points:
            result = point_add(point, point, p, a)
            if result is not None:
                results.append(result)
                inputs.append(point)

        return np.array(inputs), np.array(results)

    @staticmethod
    def point_negate(n_samples: int, p: int, a: int = 0, b: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        points = generate_curve_points(n_samples, p, a, b)

        inputs = np.array(points)
        results = np.array([(x, (-y) % p) for x, y in points])

        return inputs, results

    @staticmethod
    def scalar_mult_op(n_samples: int, p: int, a: int = 0, b: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        # Generate a base point
        base_points = generate_curve_points(min(10, n_samples), p, a, b)
        if len(base_points) == 0:
            raise ValueError(f"Could not generate base points for p={p}")

        G = base_points[0]

        # Generate scalars and compute results
        k = np.random.randint(1, min(p, 100), n_samples)  # Keep scalars reasonable

        results = []
        inputs = []
        for scalar in k:
            result = scalar_mult(scalar, G, p, a)
            if result is not None:
                results.append(result)
                inputs.append([scalar, G[0], G[1]])

        return np.array(inputs), np.array(results)

# ============================================================================
# ENCODING STRATEGIES
# ============================================================================

def combined_encoding(values: np.ndarray, p: int, bits: int = 10) -> np.ndarray:
    """
    Combined encoding with binary, normalized, and cyclic features.
    Critical for capturing modular arithmetic wrap-around.
    """
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)

    batch_size, n_values = values.shape
    features = []

    for i in range(n_values):
        vals = values[:, i]

        # Binary representation
        binary = np.zeros((batch_size, bits))
        for b in range(bits):
            binary[:, b] = (vals.astype(int) >> b) & 1
        features.append(binary)

        # Normalized value
        normalized = (vals / p).reshape(-1, 1)
        features.append(normalized)

        # Cyclic encoding (KEY for modular wrap-around!)
        sin_val = np.sin(2 * np.pi * vals / p).reshape(-1, 1)
        cos_val = np.cos(2 * np.pi * vals / p).reshape(-1, 1)
        features.append(sin_val)
        features.append(cos_val)

    return np.concatenate(features, axis=1).astype(np.float32)

def decode_output(output: np.ndarray, p: int) -> np.ndarray:
    """Decode network output to integer in range [0, p)."""
    return np.clip(np.round(output * p), 0, p - 1).astype(int)

# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class StandardNN(nn.Module):
    """Standard feedforward network with backpropagation."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ForwardForwardNN(nn.Module):
    """Bio-plausible Forward-Forward learning."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)
        self.threshold = 2.0

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.output_layer(h)

    def train_layer(self, layer_idx: int, pos_x: torch.Tensor, neg_x: torch.Tensor,
                    optimizer: torch.optim.Optimizer, epochs: int = 1):
        """Train a single layer with positive and negative samples."""
        layer = self.layers[layer_idx]

        for _ in range(epochs):
            # Forward pass for positive samples
            pos_h = pos_x
            for i in range(layer_idx + 1):
                pos_h = torch.relu(self.layers[i](pos_h))

            # Forward pass for negative samples
            neg_h = neg_x
            for i in range(layer_idx + 1):
                neg_h = torch.relu(self.layers[i](neg_h))

            # Compute goodness (squared activity)
            pos_goodness = (pos_h ** 2).sum(dim=1).mean()
            neg_goodness = (neg_h ** 2).sum(dim=1).mean()

            # Loss: maximize positive goodness, minimize negative goodness
            loss = -pos_goodness + neg_goodness

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

# ============================================================================
# TRAINING FRAMEWORK
# ============================================================================

@dataclass
class TrainingResult:
    accuracy: float
    loss: float
    time: float
    samples: int

class ECMathTrainer:
    """Comprehensive training system for all EC operations."""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results: Dict[str, Dict[int, TrainingResult]] = {}

    def train_operation(self,
                       model: nn.Module,
                       generator: Callable,
                       p: int,
                       n_samples: int = 10000,
                       epochs: int = 50,
                       batch_size: int = 256,
                       lr: float = 0.001,
                       model_type: str = 'standard') -> TrainingResult:
        """Train model on a specific operation."""

        start_time = time.time()

        try:
            # Generate data
            inputs, targets = generator(n_samples, p)

            # Handle different output dimensions
            if len(targets.shape) == 1:
                output_size = 1
                targets_encoded = (targets / p).reshape(-1, 1)
            else:
                output_size = targets.shape[1]
                targets_encoded = targets / p

            # Encode inputs
            inputs_encoded = combined_encoding(inputs, p)

            # Convert to tensors
            X = torch.FloatTensor(inputs_encoded).to(self.device)
            y = torch.FloatTensor(targets_encoded).to(self.device)

            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # Evaluation
            model.eval()
            with torch.no_grad():
                predictions = model(X).cpu().numpy()
                predictions_decoded = decode_output(predictions, p)

                if output_size == 1:
                    accuracy = np.mean(predictions_decoded.flatten() == targets.flatten())
                else:
                    accuracy = np.mean(np.all(predictions_decoded == targets, axis=1))

            elapsed = time.time() - start_time

            return TrainingResult(
                accuracy=float(accuracy),
                loss=float(total_loss / (len(X) / batch_size)),
                time=elapsed,
                samples=n_samples
            )

        except Exception as e:
            print(f"    Error: {e}")
            return TrainingResult(accuracy=0.0, loss=float('inf'), time=0.0, samples=0)

    def run_curriculum(self,
                      operation_name: str,
                      generator: Callable,
                      primes: List[int],
                      model_type: str = 'standard',
                      **kwargs) -> Dict[int, TrainingResult]:
        """Run curriculum learning across multiple prime moduli."""

        print(f"\n{'='*80}")
        print(f"Training: {operation_name} ({model_type})")
        print(f"{'='*80}")

        results = {}

        for p in primes:
            print(f"\n  Modulus p = {p}...")

            # Create fresh model for each modulus
            input_size = self._get_input_size(operation_name, p)
            output_size = self._get_output_size(operation_name)

            if model_type == 'standard':
                model = StandardNN(input_size, output_size).to(self.device)
            else:
                model = ForwardForwardNN(input_size, output_size).to(self.device)

            # Train
            result = self.train_operation(model, generator, p, model_type=model_type, **kwargs)
            results[p] = result

            print(f"    Accuracy: {result.accuracy*100:.1f}%  Loss: {result.loss:.4f}  Time: {result.time:.1f}s")

            # Early stopping if stuck
            if result.accuracy < 0.5 and p > min(primes):
                print(f"    ⚠ Stuck at p={p}, stopping curriculum")
                break

        return results

    def _get_input_size(self, operation_name: str, p: int, bits: int = 10) -> int:
        """Calculate input size based on operation type."""
        features_per_value = bits + 1 + 2  # binary + normalized + cyclic(sin, cos)

        if operation_name in ['mod_inv', 'mod_sqrt']:
            return features_per_value * 1  # Single input
        elif operation_name in ['mod_add', 'mod_sub', 'mod_mult', 'mod_div', 'mod_exp', 'point_validation']:
            return features_per_value * 2  # Two inputs
        elif operation_name in ['point_double', 'point_negate']:
            return features_per_value * 2  # Point (x, y)
        elif operation_name in ['point_add_op']:
            return features_per_value * 4  # Four inputs (two points)
        elif operation_name in ['scalar_mult_op']:
            return features_per_value * 3  # Scalar + point
        else:
            return features_per_value * 2  # Default

    def _get_output_size(self, operation_name: str) -> int:
        """Calculate output size based on operation type."""
        if operation_name == 'point_validation':
            return 1  # Binary classification
        elif operation_name in ['point_add_op', 'point_double', 'point_negate', 'scalar_mult_op']:
            return 2  # Point coordinates (x, y)
        else:
            return 1  # Single value

    def print_summary(self, all_results: Dict[str, Dict[str, Dict[int, TrainingResult]]]):
        """Print comprehensive summary of all results."""

        print("\n" + "="*80)
        print("COMPLETE EC MATH TRAINING RESULTS")
        print("="*80)

        # Modular Arithmetic Section
        print("\nMODULAR ARITHMETIC")
        print("-"*80)

        mod_ops = ['mod_add', 'mod_sub', 'mod_mult', 'mod_div', 'mod_inv', 'mod_exp', 'mod_sqrt']
        self._print_operation_table(all_results, mod_ops)

        # Elliptic Curve Operations Section
        print("\n\nELLIPTIC CURVE OPERATIONS")
        print("-"*80)

        ec_ops = ['point_validation', 'point_add_op', 'point_double', 'point_negate', 'scalar_mult_op']
        self._print_operation_table(all_results, ec_ops)

        print("\n" + "="*80)

    def _print_operation_table(self, all_results: Dict, operations: List[str]):
        """Print results table for a set of operations."""

        # Get all primes that were tested
        all_primes = set()
        for model_type in all_results.values():
            for op_results in model_type.values():
                all_primes.update(op_results.keys())
        primes = sorted(all_primes)

        # Print header
        header = f"{'Operation':<20}"
        for p in primes:
            header += f"  p={p:<5}"
        print(header)
        print("-"*80)

        # Print each operation
        for op in operations:
            for model_name, model_results in all_results.items():
                if op in model_results:
                    row = f"{op:<20}"
                    for p in primes:
                        if p in model_results[op]:
                            acc = model_results[op][p].accuracy * 100
                            if acc >= 95:
                                row += f"  {acc:>5.1f}% "
                            elif acc >= 50:
                                row += f"  {acc:>5.1f}% "
                            else:
                                row += f"  {acc:>5.1f}% "
                        else:
                            row += "  ---    "
                    row += f" [{model_name}]"
                    print(row)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive training on all EC operations."""

    print("="*80)
    print("COMPREHENSIVE ELLIPTIC CURVE MATH TRAINING")
    print("="*80)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")

    # Initialize trainer
    trainer = ECMathTrainer()

    # Curriculum: Start with small primes, scale up
    primes = [7, 11, 23, 47, 97]

    # All operations to train
    operations = [
        # Modular Arithmetic (Level 1)
        ('mod_add', OperationGenerator.mod_add),
        ('mod_sub', OperationGenerator.mod_sub),
        ('mod_mult', OperationGenerator.mod_mult),
        ('mod_div', OperationGenerator.mod_div),
        ('mod_inv', OperationGenerator.mod_inv),
        ('mod_exp', OperationGenerator.mod_exp),
        ('mod_sqrt', OperationGenerator.mod_sqrt),

        # Elliptic Curve Operations (Level 2)
        ('point_validation', OperationGenerator.point_validation),
        ('point_add_op', OperationGenerator.point_add_op),
        ('point_double', OperationGenerator.point_double),
        ('point_negate', OperationGenerator.point_negate),
        ('scalar_mult_op', OperationGenerator.scalar_mult_op),
    ]

    # Train with both standard and bio-plausible models
    all_results = {
        'Standard (Backprop)': {},
        # 'Forward-Forward': {},  # Uncomment to test bio-plausible
    }

    # Run training for standard model
    print("\n" + "="*80)
    print("PHASE 1: Standard Neural Network (Backpropagation)")
    print("="*80)

    for op_name, generator in operations:
        results = trainer.run_curriculum(
            op_name,
            generator,
            primes,
            model_type='standard',
            n_samples=10000,
            epochs=50,
            batch_size=256,
            lr=0.001
        )
        all_results['Standard (Backprop)'][op_name] = results

    # Print comprehensive summary
    trainer.print_summary(all_results)

    # Save detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        for op_name, op_results in model_results.items():
            print(f"\n  {op_name}:")
            for p, result in op_results.items():
                print(f"    p={p:3d}: accuracy={result.accuracy*100:5.1f}% "
                      f"loss={result.loss:8.4f} time={result.time:6.1f}s")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

if __name__ == "__main__":
    main()
