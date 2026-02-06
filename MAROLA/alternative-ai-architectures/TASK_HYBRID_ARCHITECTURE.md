# TASK: Design and Implement Hybrid Bio-Plausible Architecture
**Priority:** HIGH
**Agent:** Architecture Designer
**Estimated Time:** 1-2 weeks
**Depends On:** TASK_IMPLEMENT_NALU, TASK_FIX_FORWARD_FORWARD, TASK_FIX_LIQUID_NETWORK

---

## OBJECTIVE

Design hybrid architecture combining bio-plausible feature learning with specialized arithmetic modules to achieve 70-85% accuracy on modular arithmetic (p=97).

**Key Insight:** Bio-plausible methods excel at feature learning, specialized modules excel at arithmetic. Combine both!

---

## ARCHITECTURE DESIGN

### Hybrid Model Structure

```
Input (Fourier Features)
        ↓
Bio-Plausible Feature Extractor
(Forward-Forward or Liquid Network)
   - Unsupervised learning
   - Learn general representations
   - No backpropagation
        ↓
Arithmetic Reasoning Module
(NALU or iNALU)
   - Supervised learning
   - Specialized for arithmetic
   - Explicit arithmetic operations
        ↓
Output (Result)
```

### Design Principles

1. **Separation of Concerns:**
   - Feature learning: Bio-plausible (unsupervised)
   - Arithmetic: Specialized modules (supervised)

2. **Gradual Supervision:**
   - Phase 1: Train features unsupervised
   - Phase 2: Freeze features, train arithmetic
   - Phase 3: Optional fine-tuning

3. **Modular Testing:**
   - Can test each component independently
   - Can swap bio-plausible methods (FF vs LNN)
   - Can swap arithmetic modules (NALU vs iNALU)

---

## IMPLEMENTATION

### Architecture 1: Forward-Forward + iNALU

```python
class ForwardForwardNALUHybrid(nn.Module):
    """
    Hybrid: Forward-Forward feature learning + iNALU arithmetic.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, prime):
        super().__init__()

        self.prime = prime

        # Bio-plausible feature extractor (Forward-Forward)
        self.ff_layer1 = ForwardForwardLayer(
            input_dim, hidden_dim,
            threshold=2.0,
            learning_rate=0.03
        )

        self.ff_layer2 = ForwardForwardLayer(
            hidden_dim, hidden_dim,
            threshold=2.0,
            learning_rate=0.03
        )

        # Arithmetic reasoning module (iNALU)
        self.nalu = iNALU(hidden_dim, output_dim)

    def extract_features(self, x):
        """
        Extract features using Forward-Forward layers.
        No labels needed (unsupervised).
        """
        h1, _ = self.ff_layer1.forward(x)
        h2, _ = self.ff_layer2.forward(h1)
        return h2

    def forward(self, x):
        """Full forward pass."""
        features = self.extract_features(x)
        output = self.nalu(features)
        return output

    def train_features_unsupervised(self, data_loader, epochs=50):
        """
        Phase 1: Train Forward-Forward layers unsupervised.
        """
        print("Phase 1: Training bio-plausible feature extractor...")

        for epoch in range(epochs):
            for x_batch in data_loader:
                # Generate positive samples (real data)
                x_pos = x_batch

                # Generate negative samples (corrupted data)
                x_neg = x_batch + 0.1 * torch.randn_like(x_batch)

                # Train layer 1
                self.ff_layer1.train_step(x_pos, x_neg)

                # Forward through layer 1
                with torch.no_grad():
                    h1_pos, _ = self.ff_layer1.forward(x_pos)
                    h1_neg, _ = self.ff_layer1.forward(x_neg)

                # Train layer 2
                self.ff_layer2.train_step(h1_pos, h1_neg)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{epochs}")

        print("Phase 1 complete: Features learned!")

    def train_arithmetic_supervised(self, data_loader, epochs=100, lr=1e-3):
        """
        Phase 2: Train NALU module supervised (freeze features).
        """
        print("Phase 2: Training arithmetic module...")

        optimizer = torch.optim.Adam(self.nalu.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for x_batch, y_batch in data_loader:
                # Extract features (frozen)
                with torch.no_grad():
                    features = self.extract_features(x_batch)

                # Train NALU
                optimizer.zero_grad()
                output = self.nalu(features)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = epoch_loss / len(data_loader)
                print(f"  Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")

        print("Phase 2 complete: Arithmetic learned!")

    def train_hybrid(self, unsupervised_data, supervised_data,
                     phase1_epochs=50, phase2_epochs=100):
        """
        Complete two-phase training.
        """
        # Phase 1: Unsupervised feature learning
        self.train_features_unsupervised(unsupervised_data, phase1_epochs)

        # Phase 2: Supervised arithmetic learning
        self.train_arithmetic_supervised(supervised_data, phase2_epochs)
```

**Expected Accuracy:** 70-80% on p=97

---

### Architecture 2: Liquid Network + ModularNALU

```python
class LiquidNetworkNALUHybrid(nn.Module):
    """
    Hybrid: Liquid Network feature learning + ModularNALU arithmetic.
    """

    def __init__(self, input_dim, output_dim, wiring_config, prime):
        super().__init__()

        self.prime = prime

        # Bio-plausible feature extractor (Liquid Network)
        self.liquid = LiquidNeuralNetworkThreeFactor(
            input_dim=input_dim,
            output_dim=wiring_config.n_motor,
            wiring_config=wiring_config,
            dt=0.1,
            ode_steps=3
        )

        # Arithmetic reasoning module (ModularNALU)
        self.nalu = ModularNALU(
            input_dim=wiring_config.n_motor,
            output_dim=output_dim,
            prime=prime
        )

    def extract_features(self, x):
        """
        Extract features using Liquid Network.
        """
        features = self.liquid.forward(x)
        return features

    def forward(self, x):
        """Full forward pass."""
        features = self.extract_features(x)
        output = self.nalu(features)
        return output

    def train_features_unsupervised(self, data_loader, epochs=100):
        """
        Phase 1: Train Liquid Network with reward-based learning.
        """
        print("Phase 1: Training liquid feature extractor...")

        for epoch in range(epochs):
            self.liquid.reset_state()

            for x_batch in data_loader:
                # Auto-encoding task (unsupervised)
                target = x_batch  # Reconstruct input

                # Forward
                output = self.liquid.forward(x_batch)

                # Compute reward (reconstruction quality)
                error = np.mean((target - output) ** 2)
                reward = np.exp(-error)

                # Three-factor learning update
                self.liquid.three_factor_update(target, output, learning_rate=0.01)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{epochs}")

        print("Phase 1 complete: Liquid features learned!")

    def train_arithmetic_supervised(self, data_loader, epochs=100, lr=1e-3):
        """
        Phase 2: Train ModularNALU supervised (freeze features).
        """
        print("Phase 2: Training modular arithmetic module...")

        optimizer = torch.optim.Adam(self.nalu.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for x_batch, y_batch in data_loader:
                # Extract features (frozen)
                with torch.no_grad():
                    features = torch.tensor(
                        self.extract_features(x_batch.numpy()),
                        dtype=torch.float32
                    )

                # Train NALU
                optimizer.zero_grad()
                output = self.nalu(features)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = epoch_loss / len(data_loader)
                print(f"  Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")

        print("Phase 2 complete: Modular arithmetic learned!")
```

**Expected Accuracy:** 65-75% on p=97

---

### Architecture 3: Dual-Stream Hybrid

```python
class DualStreamHybrid(nn.Module):
    """
    Dual-stream: Bio-plausible AND standard backprop in parallel.
    Combines outputs with learned gating.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, prime):
        super().__init__()

        # Stream 1: Bio-plausible (Forward-Forward)
        self.bio_stream = ForwardForwardNALUHybrid(
            input_dim, hidden_dim, output_dim, prime
        )

        # Stream 2: Standard backprop (iNALU only)
        self.standard_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            iNALU(hidden_dim, output_dim)
        )

        # Gating network (learns which stream to trust)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """Forward through both streams and combine."""
        # Stream outputs
        out_bio = self.bio_stream(x)
        out_standard = self.standard_stream(x)

        # Gating weights
        gate_weights = self.gate(x)  # [batch, 2]

        # Weighted combination
        output = (
            gate_weights[:, 0:1] * out_bio +
            gate_weights[:, 1:2] * out_standard
        )

        return output, gate_weights

    def train_dual_stream(self, data_loader, epochs=100):
        """
        Train both streams and gating network.
        """
        # Phase 1: Train bio-plausible stream unsupervised
        unsupervised_data = [x for x, _ in data_loader]
        self.bio_stream.train_features_unsupervised(
            unsupervised_data, epochs=50
        )

        # Phase 2: Train both streams + gate supervised
        optimizer = torch.optim.Adam([
            {'params': self.bio_stream.nalu.parameters()},
            {'params': self.standard_stream.parameters()},
            {'params': self.gate.parameters()}
        ], lr=1e-3)

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()

                output, gate_weights = self.forward(x_batch)
                loss = criterion(output, y_batch)

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                # Check which stream is preferred
                avg_bio_weight = gate_weights[:, 0].mean().item()
                print(f"Epoch {epoch}: Bio-stream weight = {avg_bio_weight:.2f}")
```

**Expected Accuracy:** 80-90% on p=97 (best of both worlds!)

---

## TRAINING PROTOCOL

### Two-Phase Training

```python
def train_hybrid_model(model, prime, data_config):
    """
    Complete training protocol for hybrid models.
    """

    print("="*70)
    print("HYBRID MODEL TRAINING")
    print("="*70)

    # Generate data
    train_data_unsupervised = generate_unsupervised_data(prime, 10000)
    train_data_supervised = generate_supervised_data(prime, 5000)
    test_data = generate_test_data(prime, 1000)

    # Phase 1: Unsupervised feature learning
    print("\nPHASE 1: Bio-plausible feature learning (unsupervised)")
    model.train_features_unsupervised(
        train_data_unsupervised,
        epochs=50
    )

    # Evaluate feature quality
    feature_quality = evaluate_feature_representation(
        model, test_data
    )
    print(f"Feature quality: {feature_quality:.2f}")

    # Phase 2: Supervised arithmetic learning
    print("\nPHASE 2: Arithmetic module training (supervised)")
    model.train_arithmetic_supervised(
        train_data_supervised,
        epochs=100
    )

    # Final evaluation
    print("\nFINAL EVALUATION:")
    test_accuracy = evaluate_accuracy(model, test_data)
    print(f"Test accuracy: {test_accuracy*100:.1f}%")

    return model, test_accuracy
```

### Curriculum Learning for Hybrid

```python
def curriculum_train_hybrid(model_class, curriculum=[7, 11, 23, 47, 97]):
    """
    Train hybrid model with curriculum learning.
    """

    results = {}

    for i, prime in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"CURRICULUM STAGE {i+1}: p={prime}")
        print(f"{'='*70}")

        # Create model
        if i == 0:
            # Fresh model for first prime
            model = model_class(input_dim=40, hidden_dim=128, output_dim=20, prime=prime)
        else:
            # Transfer features from previous prime
            print(f"Transferring features from p={curriculum[i-1]}")
            model = transfer_features(model, prime)

        # Train
        model, accuracy = train_hybrid_model(model, prime, data_config)

        results[prime] = accuracy

        # Check if should continue
        if accuracy < 0.7:
            print(f"\n⚠️  Accuracy {accuracy*100:.1f}% below 70% - stopping")
            break

    return results
```

---

## EVALUATION METRICS

### Feature Quality Metrics

```python
def evaluate_feature_representation(model, test_data):
    """
    Evaluate quality of learned features.

    Metrics:
    1. Separation: How well features separate classes
    2. Clustering: How well same-class features cluster
    3. Dimensionality: Effective dimensionality of features
    """

    # Extract features for all test samples
    features = []
    labels = []

    for x, y in test_data:
        with torch.no_grad():
            f = model.extract_features(x)
            features.append(f.numpy())
            labels.append(y.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    # Metric 1: Silhouette score (separation)
    from sklearn.metrics import silhouette_score
    separation = silhouette_score(features, labels)

    # Metric 2: Clustering coefficient
    from sklearn.cluster import KMeans
    n_classes = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_classes)
    kmeans.fit(features)
    clustering_acc = np.mean(kmeans.labels_ == labels)

    # Metric 3: Effective dimensionality (PCA)
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(features)
    effective_dim = np.sum(pca.explained_variance_ratio_ > 0.01)

    print(f"\nFeature Quality Metrics:")
    print(f"  Separation: {separation:.3f}")
    print(f"  Clustering: {clustering_acc:.3f}")
    print(f"  Effective dim: {effective_dim}/{features.shape[1]}")

    # Combined score
    quality = (separation + clustering_acc) / 2
    return quality
```

### Arithmetic Module Analysis

```python
def analyze_arithmetic_module(model, test_data):
    """
    Analyze learned arithmetic operations.

    For NALU: Check if weights are sparse (-1, 0, +1).
    """

    # Extract NALU weights
    nalu_weights = model.nalu.W.detach().numpy()

    # Check sparsity
    sparse_mask = np.abs(nalu_weights) < 0.1
    sparsity = np.mean(sparse_mask)

    # Check concentration at {-1, 0, +1}
    discrete_mask = (
        (np.abs(nalu_weights) < 0.1) |
        (np.abs(nalu_weights - 1.0) < 0.2) |
        (np.abs(nalu_weights + 1.0) < 0.2)
    )
    discreteness = np.mean(discrete_mask)

    print(f"\nArithmetic Module Analysis:")
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Discreteness: {discreteness:.1%}")

    # Visualize weights
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(nalu_weights.flatten(), bins=50)
    plt.title("Weight Distribution")
    plt.xlabel("Weight value")

    plt.subplot(1, 2, 2)
    plt.imshow(nalu_weights, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Weight Matrix")
    plt.savefig("nalu_weights_analysis.png")
```

---

## EXPECTED RESULTS

### Performance Comparison

| Architecture | p=7 | p=23 | p=97 | Training Time |
|-------------|-----|------|------|---------------|
| Pure Forward-Forward | 65% | 40% | 25% | 1h |
| Pure Liquid Network | 55% | 35% | 20% | 2h |
| Pure iNALU (backprop) | 95% | 95% | 95% | 30min |
| FF + iNALU Hybrid | 80% | 70% | 75% | 1.5h |
| LNN + NALU Hybrid | 75% | 65% | 70% | 2.5h |
| Dual-Stream | 90% | 85% | 85% | 2h |

### Key Insights

1. **Hybrid > Pure Bio-Plausible:** 2-3x improvement
2. **Hybrid approaches backprop:** 70-85% of backprop performance
3. **Trade-off:** Biological plausibility vs accuracy
4. **Best:** Dual-stream (combines both paradigms)

---

## DELIVERABLES

1. **Code:**
   - Three hybrid architectures implemented
   - Two-phase training protocol
   - Feature quality evaluation
   - Transfer learning support

2. **Results:**
   - Benchmark on p=7, 23, 97
   - Feature quality analysis
   - NALU weight visualization
   - Comparison with pure approaches

3. **Documentation:**
   - Architecture design document
   - Training protocol guide
   - API documentation
   - Usage examples

---

## SUCCESS CRITERIA

- [ ] FF + iNALU: >75% on p=97
- [ ] LNN + NALU: >70% on p=97
- [ ] Dual-stream: >85% on p=97
- [ ] Feature quality: >0.7
- [ ] NALU sparsity: >60%

---

**Priority:** HIGH
**Blocking:** Phase 3 (hierarchical processing)
**Estimated Completion:** 1-2 weeks
**Success Definition:** >75% accuracy on p=97 with bio-plausible features
