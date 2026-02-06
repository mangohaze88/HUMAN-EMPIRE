"""
Proof of Concept: Scaling Neural Networks to Large Bit-Widths
Demonstrates digit-by-digit processing with Fourier features

Tests progression: 16-bit → 32-bit → 64-bit → (256-bit)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import time
import json
from collections import defaultdict


# ============================================================================
# Fourier Encoding for Digits
# ============================================================================

class FourierDigitEncoder:
    """
    Encode digits using Fourier features for better learning.
    Proven effective for modular arithmetic (grokking research).
    """

    def __init__(self, base: int, n_frequencies: int):
        """
        Args:
            base: Numerical base (16 for hex, 10 for decimal, 2 for binary)
            n_frequencies: Number of Fourier frequency components
        """
        self.base = base
        self.n_frequencies = n_frequencies
        self.feature_dim = 2 * n_frequencies  # sin and cos for each frequency

    def encode(self, digit: int) -> np.ndarray:
        """
        Encode digit as Fourier features.

        Args:
            digit: Integer in range [0, base)

        Returns:
            Fourier features of shape (2 * n_frequencies,)
        """
        features = []
        for k in range(1, self.n_frequencies + 1):
            angle = 2 * np.pi * k * digit / self.base
            features.extend([np.sin(angle), np.cos(angle)])
        return np.array(features, dtype=np.float32)

    def decode(self, features: np.ndarray) -> float:
        """
        Decode Fourier features back to digit.

        Uses phase of fundamental frequency.

        Args:
            features: Fourier features

        Returns:
            Decoded digit (may need rounding)
        """
        sin_1 = features[0]
        cos_1 = features[1]
        angle = np.arctan2(sin_1, cos_1)
        if angle < 0:
            angle += 2 * np.pi
        digit = self.base * angle / (2 * np.pi)
        return digit

    def encode_batch(self, digits: List[int]) -> np.ndarray:
        """Encode batch of digits"""
        return np.array([self.encode(d) for d in digits])

    def decode_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Decode batch of features"""
        return np.array([self.decode(f) for f in features_batch])


# ============================================================================
# Single Digit Processor Network
# ============================================================================

class DigitProcessorNetwork(nn.Module):
    """
    Neural network that processes single digit addition with carry.

    Input: digit_a (Fourier), digit_b (Fourier), carry_in (Fourier)
    Output: digit_result (Fourier), carry_out (Fourier)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Output in [-1, 1] for Fourier features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# Scalable Arithmetic Network
# ============================================================================

class ScalableArithmeticNetwork:
    """
    Main network that scales from single digits to arbitrary bit-widths.
    Uses digit-by-digit processing with Fourier features.
    """

    def __init__(self, base: int = 16, n_frequencies: int = 8, device: str = 'cpu'):
        """
        Args:
            base: Numerical base (16 for hexadecimal)
            n_frequencies: Number of Fourier frequencies
            device: 'cpu' or 'cuda'
        """
        self.base = base
        self.n_frequencies = n_frequencies
        self.device = device

        # Fourier encoder for digits
        self.encoder = FourierDigitEncoder(base, n_frequencies)

        # Neural network for digit processing
        input_dim = 3 * 2 * n_frequencies  # 3 inputs × Fourier features
        output_dim = 2 * 2 * n_frequencies  # 2 outputs × Fourier features
        hidden_dim = 128

        self.digit_network = DigitProcessorNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)

    def forward_digit(
        self,
        digit_a: int,
        digit_b: int,
        carry_in: int
    ) -> Tuple[int, int]:
        """
        Process a single digit addition with carry.

        Args:
            digit_a: First digit (0 to base-1)
            digit_b: Second digit (0 to base-1)
            carry_in: Incoming carry (0 or 1)

        Returns:
            (digit_result, carry_out)
        """
        # Encode inputs as Fourier features
        features_a = self.encoder.encode(digit_a)
        features_b = self.encoder.encode(digit_b)
        features_carry = self.encoder.encode(carry_in)

        # Concatenate features
        features = np.concatenate([features_a, features_b, features_carry])
        features_tensor = torch.from_numpy(features).float().to(self.device)

        # Forward through network
        self.digit_network.eval()
        with torch.no_grad():
            output_features = self.digit_network(features_tensor).cpu().numpy()

        # Split output into digit and carry
        split_idx = 2 * self.n_frequencies
        digit_features = output_features[:split_idx]
        carry_features = output_features[split_idx:]

        # Decode outputs
        digit_result = self.encoder.decode(digit_features)
        carry_out = self.encoder.decode(carry_features)

        # Round and clamp to valid ranges
        digit_result = int(round(np.clip(digit_result, 0, self.base - 1)))
        carry_out = int(round(np.clip(carry_out, 0, 1)))

        return digit_result, carry_out

    def int_to_digits(self, n: int, n_digits: int) -> List[int]:
        """Convert integer to digit list (least significant first)"""
        digits = []
        for _ in range(n_digits):
            digits.append(n % self.base)
            n //= self.base
        return digits

    def digits_to_int(self, digits: List[int]) -> int:
        """Convert digit list to integer"""
        result = 0
        for i, digit in enumerate(digits):
            result += digit * (self.base ** i)
        return result

    def add_multidigit(
        self,
        a_digits: List[int],
        b_digits: List[int]
    ) -> List[int]:
        """
        Add two multi-digit numbers.

        Args:
            a_digits: Digits of first number (least significant first)
            b_digits: Digits of second number (least significant first)

        Returns:
            Result digits (least significant first)
        """
        n_digits = max(len(a_digits), len(b_digits))

        # Pad to same length
        a_digits = a_digits + [0] * (n_digits - len(a_digits))
        b_digits = b_digits + [0] * (n_digits - len(b_digits))

        result_digits = []
        carry = 0

        # Process from least significant to most significant
        for i in range(n_digits):
            digit_result, carry = self.forward_digit(
                a_digits[i],
                b_digits[i],
                carry
            )
            result_digits.append(digit_result)

        # Add final carry if present
        if carry > 0:
            result_digits.append(carry)

        return result_digits

    def add_numbers(self, a: int, b: int, n_digits: int) -> int:
        """
        Add two numbers with specified bit-width.

        Args:
            a: First number
            b: Second number
            n_digits: Number of digits (hex) to use

        Returns:
            Sum
        """
        a_digits = self.int_to_digits(a, n_digits)
        b_digits = self.int_to_digits(b, n_digits)

        result_digits = self.add_multidigit(a_digits, b_digits)

        return self.digits_to_int(result_digits)


# ============================================================================
# Training Functions
# ============================================================================

def generate_single_digit_data(base: int = 16) -> Tuple[List, List]:
    """
    Generate ALL possible single-digit additions with carry.

    For base 16: 16 × 16 × 2 = 512 total cases (fully enumerable!)

    Returns:
        X_train: List of (digit_a, digit_b, carry_in) tuples
        y_train: List of (digit_out, carry_out) tuples
    """
    X_train, y_train = [], []

    for digit_a in range(base):
        for digit_b in range(base):
            for carry_in in range(2):
                # Compute ground truth
                sum_value = digit_a + digit_b + carry_in
                digit_out = sum_value % base
                carry_out = sum_value // base

                X_train.append((digit_a, digit_b, carry_in))
                y_train.append((digit_out, carry_out))

    return X_train, y_train


def prepare_training_data(
    X_train: List,
    y_train: List,
    encoder: FourierDigitEncoder,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training data with Fourier encoding.

    Args:
        X_train: Input tuples (digit_a, digit_b, carry_in)
        y_train: Output tuples (digit_out, carry_out)
        encoder: Fourier encoder
        device: Device to put tensors on

    Returns:
        (input_features, output_features) as tensors
    """
    input_features = []
    output_features = []

    for (digit_a, digit_b, carry_in), (digit_out, carry_out) in zip(X_train, y_train):
        # Encode inputs
        feat_a = encoder.encode(digit_a)
        feat_b = encoder.encode(digit_b)
        feat_c = encoder.encode(carry_in)
        input_feat = np.concatenate([feat_a, feat_b, feat_c])

        # Encode outputs
        feat_out = encoder.encode(digit_out)
        feat_carry = encoder.encode(carry_out)
        output_feat = np.concatenate([feat_out, feat_carry])

        input_features.append(input_feat)
        output_features.append(output_feat)

    # Convert to tensors
    X_tensor = torch.from_numpy(np.array(input_features)).float().to(device)
    y_tensor = torch.from_numpy(np.array(output_features)).float().to(device)

    return X_tensor, y_tensor


def train_digit_network(
    network: DigitProcessorNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 5000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1.0,
    verbose: bool = True
) -> Dict:
    """
    Train the digit processor network with grokking optimization.

    Args:
        network: Network to train
        X_train: Training inputs
        y_train: Training targets
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay (critical for grokking)
        verbose: Print progress

    Returns:
        Training history
    """
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    loss_fn = nn.MSELoss()

    history = {
        'loss': [],
        'epoch_times': []
    }

    if verbose:
        print(f"Training digit processor network...")
        print(f"  Total samples: {len(X_train)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch: Full batch (grokking mode)")
        print()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Full batch training (standard for grokking)
        network.train()
        optimizer.zero_grad()

        outputs = network(X_train)
        loss = loss_fn(outputs, y_train)

        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Log progress
        if epoch % 100 == 0:
            history['loss'].append(loss.item())

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, "
                      f"Time={epoch_time*1000:.1f}ms")

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Final loss: {loss.item():.6f}")
        avg_time = np.mean(history['epoch_times'])
        print(f"  Average epoch time: {avg_time*1000:.1f}ms")

    return history


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_single_digit_accuracy(
    network: ScalableArithmeticNetwork,
    base: int = 16
) -> Dict:
    """
    Evaluate accuracy on all single-digit cases.

    Args:
        network: Trained network
        base: Numerical base

    Returns:
        Accuracy metrics
    """
    X_test, y_test = generate_single_digit_data(base)

    correct_digit = 0
    correct_carry = 0
    correct_both = 0
    total = len(X_test)

    for (digit_a, digit_b, carry_in), (expected_digit, expected_carry) in zip(X_test, y_test):
        pred_digit, pred_carry = network.forward_digit(digit_a, digit_b, carry_in)

        if pred_digit == expected_digit:
            correct_digit += 1
        if pred_carry == expected_carry:
            correct_carry += 1
        if pred_digit == expected_digit and pred_carry == expected_carry:
            correct_both += 1

    return {
        'digit_accuracy': correct_digit / total,
        'carry_accuracy': correct_carry / total,
        'exact_match': correct_both / total,
        'total_cases': total
    }


def evaluate_multidigit_accuracy(
    network: ScalableArithmeticNetwork,
    bits: int,
    n_samples: int = 1000
) -> Dict:
    """
    Evaluate accuracy on multi-digit additions.

    Args:
        network: Trained network
        bits: Bit-width to test
        n_samples: Number of random test cases

    Returns:
        Accuracy metrics
    """
    n_digits = (bits + 3) // 4  # Hex digits needed
    max_value = (1 << bits) - 1

    correct = 0
    errors = []

    for _ in range(n_samples):
        # Use Python's random for large numbers
        import random
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)

        expected = a + b
        predicted = network.add_numbers(a, b, n_digits)

        if predicted == expected:
            correct += 1
        else:
            error = abs(predicted - expected)
            errors.append(error)

    accuracy = correct / n_samples

    # Calculate error statistics
    if errors:
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = max(errors)
    else:
        mean_error = median_error = max_error = 0

    return {
        'bits': bits,
        'accuracy': accuracy,
        'correct': correct,
        'total': n_samples,
        'mean_error': float(mean_error),
        'median_error': float(median_error),
        'max_error': float(max_error)
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_scaling_experiment(device: str = 'cpu', quick_mode: bool = False):
    """
    Run complete scaling experiment from 16-bit to 256-bit.

    Args:
        device: 'cpu' or 'cuda'
        quick_mode: Use fewer epochs for faster testing
    """
    print("=" * 80)
    print("PROOF OF CONCEPT: Scaling Neural Networks to Large Bit-Widths")
    print("=" * 80)
    print()

    # Configuration
    base = 16  # Hexadecimal
    n_frequencies = 8
    epochs = 1000 if quick_mode else 5000

    print(f"Configuration:")
    print(f"  Base: {base} (hexadecimal)")
    print(f"  Fourier frequencies: {n_frequencies}")
    print(f"  Training epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"  Quick mode: {quick_mode}")
    print()

    # Step 1: Create network
    print("Step 1: Creating network...")
    network = ScalableArithmeticNetwork(
        base=base,
        n_frequencies=n_frequencies,
        device=device
    )
    print("  Network created!")
    print()

    # Step 2: Generate training data
    print("Step 2: Generating training data...")
    X_train, y_train = generate_single_digit_data(base)
    print(f"  Generated {len(X_train)} training examples")
    print(f"  Coverage: 100% of {base}×{base}×2 = {len(X_train)} cases")
    print()

    # Step 3: Prepare data with Fourier encoding
    print("Step 3: Encoding with Fourier features...")
    X_tensor, y_tensor = prepare_training_data(
        X_train, y_train,
        network.encoder,
        device=device
    )
    print(f"  Input shape: {X_tensor.shape}")
    print(f"  Output shape: {y_tensor.shape}")
    print()

    # Step 4: Train network
    print("Step 4: Training network...")
    print()
    history = train_digit_network(
        network.digit_network,
        X_tensor,
        y_tensor,
        epochs=epochs,
        learning_rate=1e-3,
        weight_decay=1.0,
        verbose=True
    )
    print()

    # Step 5: Evaluate single-digit accuracy
    print("Step 5: Evaluating single-digit accuracy...")
    single_digit_results = evaluate_single_digit_accuracy(network, base)
    print(f"  Digit accuracy: {single_digit_results['digit_accuracy']:.2%}")
    print(f"  Carry accuracy: {single_digit_results['carry_accuracy']:.2%}")
    print(f"  Exact match: {single_digit_results['exact_match']:.2%}")
    print()

    # Step 6: Test scaling to multiple bit-widths
    print("Step 6: Testing scaling to larger bit-widths...")
    print()

    bit_widths = [16, 32, 64]
    if not quick_mode:
        bit_widths.extend([128, 256])

    results = {
        'config': {
            'base': base,
            'n_frequencies': n_frequencies,
            'epochs': epochs,
            'device': device
        },
        'single_digit': single_digit_results,
        'multi_digit': {}
    }

    for bits in bit_widths:
        print(f"Testing {bits}-bit addition...")

        n_samples = 100 if bits >= 128 else 1000

        eval_results = evaluate_multidigit_accuracy(
            network,
            bits=bits,
            n_samples=n_samples
        )

        results['multi_digit'][f'{bits}bit'] = eval_results

        print(f"  Accuracy: {eval_results['accuracy']:.2%}")
        print(f"  Correct: {eval_results['correct']}/{eval_results['total']}")

        if eval_results['mean_error'] > 0:
            print(f"  Mean error: {eval_results['mean_error']:.2e}")
            print(f"  Median error: {eval_results['median_error']:.2e}")

        # Calculate theoretical error accumulation
        n_digits = (bits + 3) // 4
        single_digit_acc = single_digit_results['exact_match']
        theoretical_acc = single_digit_acc ** n_digits
        print(f"  Theoretical (independent errors): {theoretical_acc:.2%}")

        print()

    # Step 7: Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Single-Digit Performance:")
    print(f"  Exact match accuracy: {single_digit_results['exact_match']:.2%}")
    print()
    print("Multi-Digit Performance:")
    for bits in bit_widths:
        acc = results['multi_digit'][f'{bits}bit']['accuracy']
        print(f"  {bits}-bit: {acc:.2%}")
    print()

    # Save results
    output_file = '/root/MAROLA/alternative-ai-architectures/experiments/scaling_poc_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Analysis
    print("Analysis:")
    print()

    if single_digit_results['exact_match'] > 0.95:
        print("✓ Single-digit accuracy is EXCELLENT (>95%)")
        print("  → Grokking was successful!")
    elif single_digit_results['exact_match'] > 0.80:
        print("~ Single-digit accuracy is GOOD (>80%)")
        print("  → May need more training for grokking")
    else:
        print("✗ Single-digit accuracy is LOW (<80%)")
        print("  → Training failed, try adjusting hyperparameters")

    print()

    if bit_widths[-1] >= 64:
        acc_64 = results['multi_digit']['64bit']['accuracy']
        if acc_64 > 0.90:
            print("✓ 64-bit accuracy is EXCELLENT (>90%)")
            print("  → Scaling is working well!")
        elif acc_64 > 0.70:
            print("~ 64-bit accuracy is MODERATE (>70%)")
            print("  → Some error accumulation, but usable")
        else:
            print("✗ 64-bit accuracy is LOW (<70%)")
            print("  → Error accumulation is too high")

    print()
    print("Next Steps:")
    print("  1. Train ensemble of networks for error correction")
    print("  2. Implement verification checks")
    print("  3. Test on 256-bit numbers")
    print("  4. Extend to multiplication and other operations")
    print()

    return results


if __name__ == "__main__":
    import sys

    # Parse arguments
    quick_mode = '--quick' in sys.argv
    use_cuda = '--cuda' in sys.argv and torch.cuda.is_available()

    device = 'cuda' if use_cuda else 'cpu'

    # Run experiment
    results = run_scaling_experiment(device=device, quick_mode=quick_mode)

    print("=" * 80)
    print("Experiment complete!")
    print("=" * 80)
