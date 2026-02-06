# Quick Reference - Elliptic Curve Operations

## 30-Second Start

```python
from src.crypto.ec_operations import create_toy_curve

curve = create_toy_curve(bits=8)
P = curve.generate_point()
Q = curve.generate_point()
R = curve.point_add(P, Q)
print(f"{P} + {Q} = {R}")
```

## Common Tasks

### Create a Curve
```python
from src.crypto.ec_operations import create_toy_curve, EllipticCurve

# Quick toy curve
curve = create_toy_curve(bits=8)  # or 16, 32

# Custom curve
curve = EllipticCurve(p=251, a=0, b=7)
```

### Point Operations
```python
# Generate points
P = curve.generate_point()
Q = curve.generate_point()

# Operations
R = curve.point_add(P, Q)           # P + Q
P2 = curve.point_double(P)          # 2P
neg_P = curve.point_negate(P)       # -P
kP = curve.scalar_mult(5, P)        # 5*P
diff = curve.point_subtract(P, Q)   # P - Q

# Validation
is_valid = curve.is_on_curve(x, y)
are_equal = curve.is_equal(P, Q)
```

### Generate Training Data
```python
from src.crypto.ec_operations import generate_training_data

# Point addition dataset
X, y = generate_training_data(
    curve,
    operation='point_addition',  # or 'point_validation', 'scalar_mult', etc.
    n_samples=1000,
    random_seed=42
)
```

### Encode for Neural Networks
```python
from src.crypto.ec_operations import encode_point, batch_encode_points

# Single point
encoded = encode_point(P, curve.p, 'normalized')

# Batch
points = [curve.generate_point() for _ in range(100)]
encoded_batch = batch_encode_points(points, curve.p, 'normalized')
```

### Complete Training Pipeline
```python
# 1. Create curve
curve = create_toy_curve(bits=8)

# 2. Generate data
X_raw, y_raw = generate_training_data(curve, 'point_addition', n_samples=10000)

# 3. Encode inputs (pairs of points)
X = []
for P1, P2 in X_raw:
    enc1 = encode_point(P1, curve.p, 'normalized')
    enc2 = encode_point(P2, curve.p, 'normalized')
    X.append(np.concatenate([enc1, enc2]))
X = np.array(X)  # Shape: (10000, 6)

# 4. Encode outputs
y = batch_encode_points(y_raw, curve.p, 'normalized')  # Shape: (10000, 3)

# 5. Train model
model.fit(X, y)

# 6. Decode predictions
preds = model.predict(X_test)
decoded = batch_decode_points(preds, curve.p, 'normalized')
```

## Available Operations

| Operation | Description | Input | Output |
|-----------|-------------|-------|--------|
| `point_validation` | Valid/invalid classification | (x, y) | 0 or 1 |
| `point_addition` | Learn P1 + P2 | (P1, P2) | P3 |
| `point_doubling` | Learn 2P | P | 2P |
| `point_negation` | Learn -P | P | -P |
| `scalar_mult` | Learn k*P (large k) | (k, P) | kP |
| `scalar_mult_small` | Learn k*P (k≤20) | (k, P) | kP |

## Encoding Schemes

| Scheme | Size | Range | Best For |
|--------|------|-------|----------|
| `'normalized'` | 3 | [0, 1] | Neural networks |
| `'binary'` | 2*bits+1 | {0, 1} | Exact representation |
| `'hex'` | 3 | [0, p] | Small primes |

## Curve Sizes

| Bits | Prime | Speed | Use Case |
|------|-------|-------|----------|
| 8 | 251 | Very fast | Quick testing |
| 16 | 65521 | Fast | Training |
| 32 | 4294967291 | Moderate | Realistic patterns |
| 256 | secp256k1 | Slow | Production validation |

## Files

```
src/crypto/ec_operations.py           # Core module
tests/test_ec_operations.py           # Tests (38 tests)
examples/ec_operations_usage.py       # Examples (7 examples)
docs/EC_OPERATIONS_DOCUMENTATION.md   # Full docs
```

## Run Tests
```bash
python tests/test_ec_operations.py
```

## Run Examples
```bash
python examples/ec_operations_usage.py
```

## Quick Verification
```python
from src.crypto.ec_operations import verify_curve_operations

curve = create_toy_curve(bits=8)
success = verify_curve_operations(curve)  # Should print ✓ for all tests
```

## Common Patterns

### Pattern 1: Classification Task (Point Validation)
```python
X, y = generate_training_data(curve, 'point_validation', n_samples=10000)
X_enc = batch_encode_points(X, curve.p, 'normalized')  # Shape: (10000, 3)
y = np.array(y)  # Shape: (10000,) - binary labels
# Train binary classifier
```

### Pattern 2: Regression Task (Point Addition)
```python
X, y = generate_training_data(curve, 'point_addition', n_samples=10000)
# Encode pairs of points
X_enc = np.array([
    np.concatenate([
        encode_point(P1, curve.p, 'normalized'),
        encode_point(P2, curve.p, 'normalized')
    ])
    for P1, P2 in X
])  # Shape: (10000, 6)
y_enc = batch_encode_points(y, curve.p, 'normalized')  # Shape: (10000, 3)
# Train regression model
```

### Pattern 3: Scalar Multiplication Task
```python
X, y = generate_training_data(curve, 'scalar_mult_small', n_samples=10000)
# Encode scalar + point
X_enc = np.array([
    np.concatenate([
        [k / 20],  # Normalize scalar
        encode_point(P, curve.p, 'normalized')
    ])
    for k, P in X
])  # Shape: (10000, 4)
y_enc = batch_encode_points(y, curve.p, 'normalized')  # Shape: (10000, 3)
# Train model
```

## Performance Tips

1. **Use small curves for training:** 8-bit or 16-bit
2. **Use windowed scalar mult for large k:** `scalar_mult_windowed()`
3. **Cache generator:** `G = curve.get_generator()`
4. **Batch encode datasets:** Use `batch_encode_points()`
5. **Set random seed:** For reproducible datasets

## Troubleshooting

**Issue:** Point not on curve
```python
# Verify
assert curve.is_on_curve(P[0], P[1])
```

**Issue:** Encoding/decoding mismatch
```python
# Use same encoding type
encoded = encode_point(P, curve.p, 'normalized')
decoded = decode_point(encoded, curve.p, 'normalized')  # Same type!
```

**Issue:** Slow performance
```python
# Use smaller curve
curve = create_toy_curve(bits=8)  # Much faster than 32-bit
```

## Key Formulas

**Point Addition (different points):**
```
λ = (y₂ - y₁) / (x₂ - x₁)
x₃ = λ² - x₁ - x₂
y₃ = λ(x₁ - x₃) - y₁
```

**Point Doubling:**
```
λ = (3x² + a) / (2y)
x₃ = λ² - 2x
y₃ = λ(x - x₃) - y
```

**Point Negation:**
```
-P = (x, -y mod p)
```

## Group Properties Verified

✓ Closure: P + Q on curve
✓ Associativity: (P+Q)+R = P+(Q+R)
✓ Commutativity: P+Q = Q+P
✓ Identity: P+O = P
✓ Inverse: P+(-P) = O
✓ Distributivity: k(P+Q) = kP+kQ

## Example Output

```python
>>> curve = create_toy_curve(bits=8)
>>> P = (84, 241)
>>> Q = (95, 79)
>>> curve.point_add(P, Q)
(13, 14)
>>> curve.scalar_mult(5, P)
(69, 31)
```

---

**Ready to use!** For full details, see `docs/EC_OPERATIONS_DOCUMENTATION.md`
