# TASK: Scale to 256-bit Elliptic Curve Arithmetic
**Priority:** CRITICAL (Long-term)
**Agent:** Scaling Expert + Crypto Operations Specialist
**Estimated Time:** 8-12 weeks
**Depends On:** All Phase 1-4 tasks

---

## OBJECTIVE

Scale modular arithmetic from small primes (p=7-997) to cryptographic-scale 256-bit numbers used in secp256k1 elliptic curve.

**Final Goal:** 100% accuracy on private key → public key derivation.

---

## THE SCALE CHALLENGE

### Current vs Target

**Current Scale:**
- p=7: 7 unique values
- p=97: 97 unique values
- p=997: 997 unique values

**Target Scale:**
- secp256k1 prime: 2^256 - 2^32 - 977
- ~10^77 unique values
- 115,792,089,237,316,195,423,570,985,008,687,907,853,269,984,665,640,564,039,457,584,007,908,834,671,663

**Challenge:** 10^74 times larger than p=997!

### Why This Is Hard

1. **Sample Complexity Explosion**
   - Cannot sample all combinations
   - Need compositional generalization
   - Must learn algorithmic structure

2. **Representation Challenge**
   - Cannot use one-hot encoding
   - Cannot use simple normalization
   - Need hierarchical representation

3. **Memory Constraints**
   - Full attention: O(n^2) memory
   - 256-bit number: 256 tokens
   - Attention: 65,536 elements per layer!

4. **Training Time**
   - Slow forward passes
   - Many epochs needed
   - Potential for 10+ hours per experiment

---

## SOLUTION STRATEGY

### Strategy 1: Hierarchical Decomposition

**Key Insight:** Break 256-bit number into smaller chunks.

**Approach:**
```
256-bit number
    ↓ decompose
64-bit + 64-bit + 64-bit + 64-bit
    ↓ process independently
Learn operations on 64-bit chunks
    ↓ compose
Combine with carry/borrow propagation
    ↓ result
256-bit result
```

**Implementation:**
```python
class HierarchicalModularArithmetic:
    """
    Hierarchical processing for large numbers.

    Decomposes 256-bit operations into 64-bit sub-operations.
    """

    def __init__(self, chunk_bits=64):
        self.chunk_bits = chunk_bits
        self.n_chunks = 256 // chunk_bits  # = 4

        # Sub-network for each chunk
        self.chunk_networks = [
            ImprovedNALU(input_dim=40, output_dim=20)
            for _ in range(self.n_chunks)
        ]

        # Carry/borrow propagation network
        self.carry_network = nn.Sequential(
            nn.Linear(20 * self.n_chunks, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_chunks)  # Carry for each chunk
        )

    def decompose(self, num_256bit):
        """Decompose 256-bit number into 4×64-bit chunks."""
        chunks = []
        for i in range(self.n_chunks):
            shift = i * self.chunk_bits
            mask = (1 << self.chunk_bits) - 1
            chunk = (num_256bit >> shift) & mask
            chunks.append(chunk)
        return chunks

    def compose(self, chunks):
        """Compose 4×64-bit chunks into 256-bit number."""
        result = 0
        for i, chunk in enumerate(chunks):
            result |= (chunk << (i * self.chunk_bits))
        return result

    def forward_addition(self, a_256bit, b_256bit, prime_256bit):
        """
        Hierarchical modular addition.
        """
        # Decompose
        a_chunks = self.decompose(a_256bit)
        b_chunks = self.decompose(b_256bit)

        # Process each chunk
        chunk_results = []
        for i in range(self.n_chunks):
            # Encode chunks
            a_enc = fourier_encode(a_chunks[i], 2**self.chunk_bits)
            b_enc = fourier_encode(b_chunks[i], 2**self.chunk_bits)
            input_enc = np.concatenate([a_enc, b_enc])

            # Forward through sub-network
            chunk_result = self.chunk_networks[i](
                torch.tensor(input_enc, dtype=torch.float32)
            )
            chunk_results.append(chunk_result)

        # Propagate carries
        combined = torch.cat(chunk_results, dim=0)
        carries = self.carry_network(combined)

        # Apply carries and compose
        final_chunks = []
        for i, chunk_result in enumerate(chunk_results):
            # Decode chunk
            chunk_val = fourier_decode(chunk_result.detach().numpy(), 2**self.chunk_bits)

            # Add carry from previous chunk
            if i > 0:
                chunk_val += carries[i-1].item()

            # Wrap around chunk size
            chunk_val = chunk_val % (2**self.chunk_bits)

            final_chunks.append(int(chunk_val))

        # Compose result
        result = self.compose(final_chunks)

        # Final modular reduction
        result = result % prime_256bit

        return result
```

**Expected Gain:** Makes 256-bit tractable (30-50% accuracy)

---

### Strategy 2: Algorithmic Learning with Chain-of-Thought

**Key Insight:** Teach explicit algorithms, not just input-output mapping.

**Approach:**
```
Input: a, b (256-bit)
Step 1: Decompose into limbs
Step 2: Add limbs left-to-right
Step 3: Propagate carries
Step 4: Apply modular reduction
Step 5: Compose result
Output: (a + b) mod p
```

**Training Data Format:**
```python
def generate_cot_256bit_addition(a, b, p):
    """
    Generate chain-of-thought training data for 256-bit addition.
    """
    steps = []

    # Step 1: Decomposition
    a_limbs = decompose_256bit(a)
    b_limbs = decompose_256bit(b)
    steps.append({
        "operation": "decompose",
        "description": f"Decompose {hex(a)} into limbs: {[hex(x) for x in a_limbs]}"
    })

    # Step 2: Limb addition
    result_limbs = []
    carry = 0
    for i in range(len(a_limbs)):
        limb_sum = a_limbs[i] + b_limbs[i] + carry
        result_limb = limb_sum % (2**64)
        carry = limb_sum >> 64

        steps.append({
            "operation": "add_limb",
            "description": f"Limb {i}: {hex(a_limbs[i])} + {hex(b_limbs[i])} + carry {carry} = {hex(result_limb)}"
        })

        result_limbs.append(result_limb)

    # Step 3: Compose
    result = compose_256bit(result_limbs)
    steps.append({
        "operation": "compose",
        "description": f"Compose limbs into {hex(result)}"
    })

    # Step 4: Modular reduction
    final_result = result % p
    steps.append({
        "operation": "mod",
        "description": f"Apply modulo: {hex(result)} mod {hex(p)} = {hex(final_result)}"
    })

    # Format as training example
    input_text = f"Compute ({hex(a)} + {hex(b)}) mod {hex(p)}"
    output_text = "Let's solve step by step:\n"
    for step in steps:
        output_text += f"{step['description']}\n"
    output_text += f"Answer: {hex(final_result)}"

    return input_text, output_text
```

**Expected Gain:** Enable learning of algorithmic structure (50-70% accuracy)

---

### Strategy 3: Learned Number Embeddings

**Key Insight:** Learn how to represent large numbers efficiently.

**Implementation:**
```python
class Learned256BitEmbedding(nn.Module):
    """
    Learn to embed 256-bit numbers in dense vector space.
    """

    def __init__(self, embed_dim=128):
        super().__init__()

        # Embed each limb
        self.limb_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),  # Binary representation
                nn.ReLU(),
                nn.Linear(64, embed_dim)
            )
            for _ in range(4)  # 4 limbs
        ])

        # Positional encoding for limb position
        self.position_embed = nn.Embedding(4, embed_dim)

        # Combine limbs
        self.combiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8),
            num_layers=2
        )

    def forward(self, num_256bit):
        """
        Embed 256-bit number.
        """
        # Decompose
        limbs = decompose_256bit_to_binary(num_256bit)  # 4 × 64 bits

        # Embed each limb
        limb_embeds = []
        for i, limb in enumerate(limbs):
            limb_tensor = torch.tensor(limb, dtype=torch.float32)
            limb_embed = self.limb_embeddings[i](limb_tensor)

            # Add positional encoding
            pos_embed = self.position_embed(torch.tensor([i]))
            limb_embed = limb_embed + pos_embed

            limb_embeds.append(limb_embed)

        # Combine with transformer
        combined = torch.stack(limb_embeds)  # [4, embed_dim]
        final_embed = self.combiner(combined)  # [4, embed_dim]

        # Pool
        pooled = final_embed.mean(dim=0)  # [embed_dim]

        return pooled
```

**Expected Gain:** Efficient representation (enables training)

---

### Strategy 4: Montgomery Form for Modular Arithmetic

**Key Insight:** Convert to Montgomery form for faster modular reduction.

**Montgomery Form:**
```
Instead of computing: (a * b) mod N
Compute in Montgomery form: (aR * bR) * R^(-1) mod N

Where R = 2^256 (convenient for 256-bit)

Advantages:
- Modular reduction becomes bit shifts
- Faster for repeated operations
- Used in real crypto libraries
```

**Implementation:**
```python
class MontgomeryModularArithmetic:
    """
    Modular arithmetic in Montgomery form.
    """

    def __init__(self, modulus_256bit):
        self.N = modulus_256bit
        self.R = 2**256
        self.R_inv = self.mod_inverse(self.R, self.N)
        self.N_prime = self.compute_N_prime()

    def to_montgomery(self, x):
        """Convert to Montgomery form: xR mod N"""
        return (x * self.R) % self.N

    def from_montgomery(self, xR):
        """Convert from Montgomery form: xR * R^(-1) mod N"""
        return (xR * self.R_inv) % self.N

    def montgomery_mul(self, aR, bR):
        """
        Montgomery multiplication: (aR * bR) * R^(-1) mod N
        """
        # Standard multiplication
        t = aR * bR

        # Montgomery reduction (faster than regular mod)
        m = ((t % self.R) * self.N_prime) % self.R
        t_reduced = (t + m * self.N) >> 256  # Bit shift instead of division!

        if t_reduced >= self.N:
            t_reduced -= self.N

        return t_reduced

    def train_montgomery_network(self, network):
        """
        Train network to learn Montgomery multiplication.
        """
        # Generate training data
        for _ in range(10000):
            a = random.randint(0, self.N-1)
            b = random.randint(0, self.N-1)

            # Convert to Montgomery
            aR = self.to_montgomery(a)
            bR = self.to_montgomery(b)

            # Montgomery multiplication
            result_R = self.montgomery_mul(aR, bR)

            # Train network
            input_enc = encode([aR, bR])
            target_enc = encode([result_R])

            network.train_step(input_enc, target_enc)
```

**Expected Gain:** 2-3x faster training and inference

---

## PHASED SCALING ROADMAP

### Phase 1: 32-bit Operations (Week 1-2)

**Goal:** Establish hierarchical processing pipeline

**Tasks:**
- [ ] Implement 32-bit decomposition (2×16-bit chunks)
- [ ] Train chunk networks on 16-bit operations
- [ ] Implement carry propagation
- [ ] Test on 32-bit addition
- [ ] Target: >70% accuracy

### Phase 2: 64-bit Operations (Week 3-4)

**Goal:** Scale to practical integer size

**Tasks:**
- [ ] Extend to 64-bit (4×16-bit or 2×32-bit chunks)
- [ ] Optimize carry propagation
- [ ] Add chain-of-thought supervision
- [ ] Test on 64-bit add/sub/mul
- [ ] Target: >60% accuracy

### Phase 3: 128-bit Operations (Week 5-7)

**Goal:** Approach cryptographic scale

**Tasks:**
- [ ] Implement 128-bit processing (2×64-bit chunks)
- [ ] Add learned number embeddings
- [ ] Test Montgomery form
- [ ] Curriculum from 64-bit → 128-bit
- [ ] Target: >50% accuracy

### Phase 4: 256-bit Operations (Week 8-10)

**Goal:** Full cryptographic scale

**Tasks:**
- [ ] Implement 256-bit processing (4×64-bit chunks)
- [ ] Full hierarchical pipeline
- [ ] Montgomery form integration
- [ ] Chain-of-thought for all operations
- [ ] Target: >40% accuracy

### Phase 5: Elliptic Curve Operations (Week 11-12)

**Goal:** Point addition and scalar multiplication

**Tasks:**
- [ ] Implement point addition using 256-bit ops
- [ ] Implement point doubling
- [ ] Implement scalar multiplication (double-and-add)
- [ ] Test on secp256k1 curve
- [ ] Target: >30% accuracy → 95-100% with refinement

---

## VERIFICATION STRATEGY

### Test Against Reference Implementations

```python
def verify_against_openssl(our_implementation):
    """
    Verify our 256-bit arithmetic against OpenSSL.
    """
    import subprocess
    import secrets

    test_cases = 1000
    failures = []

    for i in range(test_cases):
        # Generate random 256-bit numbers
        a = secrets.randbits(256)
        b = secrets.randbits(256)
        p = SECP256K1_PRIME

        # Our implementation
        our_result = our_implementation.mod_add(a, b, p)

        # OpenSSL verification
        openssl_result = verify_with_openssl(a, b, p)

        if our_result != openssl_result:
            failures.append({
                'a': hex(a),
                'b': hex(b),
                'our': hex(our_result),
                'openssl': hex(openssl_result)
            })

    accuracy = (test_cases - len(failures)) / test_cases
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Failures: {len(failures)}/{test_cases}")

    return accuracy, failures
```

### Known Test Vectors

```python
# secp256k1 known values
KNOWN_TESTS = [
    {
        'privkey': 0x0000000000000000000000000000000000000000000000000000000000000001,
        'pubkey_x': 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        'pubkey_y': 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    },
    {
        'privkey': 0x0000000000000000000000000000000000000000000000000000000000000002,
        'pubkey_x': 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5,
        'pubkey_y': 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A
    }
]

def test_known_vectors(model):
    """Test against known secp256k1 test vectors."""
    correct = 0
    for test in KNOWN_TESTS:
        pubkey = model.privkey_to_pubkey(test['privkey'])

        if (pubkey[0] == test['pubkey_x'] and
            pubkey[1] == test['pubkey_y']):
            correct += 1
            print(f"✓ Test passed: privkey={hex(test['privkey'])}")
        else:
            print(f"✗ Test failed: privkey={hex(test['privkey'])}")
            print(f"  Expected: ({hex(test['pubkey_x'])}, {hex(test['pubkey_y'])})")
            print(f"  Got: ({hex(pubkey[0])}, {hex(pubkey[1])})")

    accuracy = correct / len(KNOWN_TESTS)
    return accuracy
```

---

## EXPECTED RESULTS

### Accuracy Progression

| Phase | Bit Size | Operations | Target Acc | Time |
|-------|----------|------------|------------|------|
| 1 | 32-bit | add, sub | 70% | 2w |
| 2 | 64-bit | add, sub, mul | 60% | 2w |
| 3 | 128-bit | add, sub, mul | 50% | 3w |
| 4 | 256-bit | all mod ops | 40% | 2w |
| 5 | 256-bit | EC point ops | 30-100% | 2w |

### Final Target

**Private Key → Public Key (secp256k1):**
- Using hierarchical processing: 40-60% accuracy
- Using Montgomery form: 60-80% accuracy
- Using chain-of-thought: 70-90% accuracy
- **With all techniques + refinement: 95-100% accuracy**

---

## DELIVERABLES

1. **Code:**
   - Hierarchical 256-bit arithmetic
   - Montgomery form implementation
   - Chain-of-thought data generation
   - Learned number embeddings
   - Full EC point operations

2. **Results:**
   - Benchmark on all scales (32, 64, 128, 256-bit)
   - Verification against OpenSSL
   - Known test vector results
   - Accuracy vs scale analysis

3. **Documentation:**
   - 256-bit arithmetic guide
   - Montgomery form tutorial
   - EC operations documentation
   - Verification protocol

---

## SUCCESS CRITERIA

- [ ] 32-bit operations: >70% accuracy
- [ ] 64-bit operations: >60% accuracy
- [ ] 128-bit operations: >50% accuracy
- [ ] 256-bit modular add: >40% accuracy
- [ ] EC point addition: >30% accuracy
- [ ] Known test vectors: 95-100% pass rate
- [ ] Privkey→Pubkey: 95-100% accuracy (final goal)

---

**Priority:** CRITICAL
**Blocking:** Final integration
**Estimated Completion:** 8-12 weeks
**Success Definition:** 95-100% accuracy on secp256k1 privkey→pubkey
