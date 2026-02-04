"""
MULTI-LAYER PERCEPTRON (MLP) — Paradigm: LEARNED FEATURES

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Stack differentiable transformations. Let gradient descent learn the features.

Each layer: z = activation(W @ x + b)

The network learns to transform inputs into increasingly useful representations
for the task.

===============================================================
UNIVERSAL APPROXIMATION THEOREM
===============================================================

A neural network with ONE hidden layer and enough neurons can approximate
ANY continuous function to arbitrary precision.

But "enough neurons" can be exponentially large. Deep networks are often
more parameter-efficient than wide-shallow networks.

===============================================================
BACKPROPAGATION = CHAIN RULE
===============================================================

Forward pass: compute activations layer by layer
Backward pass: compute gradients layer by layer (chain rule)

For loss L and intermediate activation z:
    ∂L/∂W = ∂L/∂z × ∂z/∂W

This is just the chain rule applied recursively through the network.

===============================================================
ACTIVATION FUNCTIONS — THE KEY TO NONLINEARITY
===============================================================

WITHOUT nonlinear activations, a deep network collapses to a single linear transform!

    W₂(W₁x + b₁) + b₂ = W₂W₁x + W₂b₁ + b₂ = W'x + b'

COMMON ACTIVATIONS:
    Sigmoid:  σ(x) = 1/(1+e^(-x))    range (0,1), vanishing gradient issue
    Tanh:     tanh(x)                 range (-1,1), centered, vanishing gradient
    ReLU:     max(0, x)               simple, no saturation, but "dead neurons"
    LeakyReLU: max(αx, x)             fixes dead neuron problem

ReLU is now the default — simple, fast, works well.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Hierarchical feature composition (layer structure)
2. Smooth functions (gradient-based optimization)
3. Local connectivity patterns (if using weight constraints)

WHAT IT CAN DO:
    ✓ Learn ANY continuous function (universal approximation)
    ✓ Automatically learn useful features
    ✓ Scale to massive datasets with SGD

WHAT IT CAN'T DO:
    ✗ Guarantee finding global optimum (non-convex)
    ✗ Sample efficiently (needs lots of data)
    ✗ Interpretable (black box)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
plot_decision_boundary = datasets_module.plot_decision_boundary
accuracy = datasets_module.accuracy


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

class Activation:
    """Base class for activation functions."""
    def forward(self, x):
        raise NotImplementedError
    def backward(self, x):
        raise NotImplementedError


class ReLU(Activation):
    """
    ReLU: f(x) = max(0, x)

    Derivative: f'(x) = 1 if x > 0 else 0

    PROS: Simple, no saturation, sparse activations
    CONS: "Dead neurons" — once a neuron outputs 0, gradient is 0 forever
    """
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid: f(x) = 1 / (1 + e^(-x))

    Derivative: f'(x) = f(x)(1 - f(x))

    PROS: Bounded (0,1), interpretable as probability
    CONS: Vanishing gradient when |x| is large, not zero-centered
    """
    def forward(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    """
    Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Derivative: f'(x) = 1 - f(x)²

    PROS: Zero-centered (unlike sigmoid)
    CONS: Still has vanishing gradient in tails
    """
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """No activation (identity function)."""
    def forward(self, x):
        return x
    def backward(self, x):
        return np.ones_like(x)


ACTIVATIONS = {
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'linear': Linear()
}


# ============================================================
# NEURAL NETWORK
# ============================================================

class MLP:
    """
    Multi-Layer Perceptron with flexible architecture.

    ARCHITECTURE:
        Input → [Linear → Activation] × L → Output

    TRAINING:
        Forward: compute predictions
        Backward: compute gradients via backprop
        Update: gradient descent step
    """

    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid',
                 lr=0.01, n_epochs=1000, batch_size=32, l2_reg=0.0,
                 init='xavier', random_state=None):
        """
        Parameters:
        -----------
        layer_sizes : list of ints, e.g., [2, 64, 32, 1]
                      First = input dim, Last = output dim
        activation : activation for hidden layers
        output_activation : activation for output layer
        lr : learning rate
        n_epochs : training epochs
        batch_size : mini-batch size
        l2_reg : L2 regularization strength
        init : weight initialization ('xavier', 'he', 'random')
        """
        self.layer_sizes = layer_sizes
        self.activation = ACTIVATIONS[activation]
        self.output_activation = ACTIVATIONS[output_activation]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.init = init
        self.random_state = random_state

        self.weights = []
        self.biases = []
        self.loss_history = []

    def _init_weights(self):
        """
        Initialize weights.

        XAVIER: For tanh/sigmoid — Var(W) = 1/fan_in
        HE: For ReLU — Var(W) = 2/fan_in

        Proper initialization prevents vanishing/exploding gradients at start.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            if self.init == 'xavier':
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif self.init == 'he':
                std = np.sqrt(2.0 / fan_in)
            else:
                std = 0.01

            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X):
        """
        FORWARD PASS

        Store activations for backprop:
            z[l] = a[l-1] @ W[l] + b[l]  (linear)
            a[l] = activation(z[l])       (nonlinear)
        """
        self.z_cache = []  # Pre-activation values
        self.a_cache = [X]  # Post-activation values (input is a[0])

        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)

            # Use output activation for last layer
            if i == len(self.weights) - 1:
                a = self.output_activation.forward(z)
            else:
                a = self.activation.forward(z)

            self.a_cache.append(a)

        return a

    def _backward(self, y):
        """
        BACKWARD PASS (Backpropagation)

        THE CHAIN RULE IN ACTION:
            ∂L/∂W[l] = ∂L/∂z[l] × ∂z[l]/∂W[l]
                     = δ[l] × a[l-1]ᵀ

        where δ[l] = ∂L/∂z[l] propagates backward:
            δ[L] = ∂L/∂a[L] × ∂a[L]/∂z[L]  (output layer)
            δ[l] = (W[l+1] δ[l+1]) ⊙ ∂a[l]/∂z[l]  (hidden layers)
        """
        m = y.shape[0]
        n_layers = len(self.weights)

        # Gradients to compute
        dW = [None] * n_layers
        db = [None] * n_layers

        # Output layer gradient
        # For binary cross-entropy + sigmoid: ∂L/∂z = a - y
        a_out = self.a_cache[-1]
        z_out = self.z_cache[-1]

        # δ = ∂L/∂z for output layer
        if isinstance(self.output_activation, Sigmoid):
            # BCE + sigmoid simplifies to (a - y)
            delta = a_out - y.reshape(-1, 1)
        else:
            # General case
            delta = (a_out - y.reshape(-1, 1)) * self.output_activation.backward(z_out)

        # Backpropagate
        for l in reversed(range(n_layers)):
            a_prev = self.a_cache[l]

            # Weight gradient: ∂L/∂W = aᵀ × δ
            dW[l] = (a_prev.T @ delta) / m + self.l2_reg * self.weights[l]

            # Bias gradient: ∂L/∂b = mean(δ)
            db[l] = np.mean(delta, axis=0, keepdims=True)

            # Propagate gradient to previous layer
            if l > 0:
                delta = (delta @ self.weights[l].T) * self.activation.backward(self.z_cache[l-1])

        return dW, db

    def _update_weights(self, dW, db):
        """Gradient descent update."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dW[i]
            self.biases[i] -= self.lr * db[i]

    def _compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = y_true.reshape(-1, 1)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2 = 0.5 * self.l2_reg * sum(np.sum(W**2) for W in self.weights)
        return bce + l2

    def fit(self, X, y):
        """Train the network."""
        self._init_weights()
        n_samples = X.shape[0]
        self.loss_history = []

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward
                y_pred = self._forward(X_batch)

                # Backward
                dW, db = self._backward(y_batch)

                # Update
                self._update_weights(dW, db)

            # Track loss
            y_pred_all = self._forward(X)
            loss = self._compute_loss(y_pred_all, y)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probabilities."""
        return self._forward(X).flatten()

    def predict(self, X):
        """Return class predictions."""
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component?
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    # -------- Experiment 1: NO ACTIVATION (Linear Network) --------
    print("\n1. WITHOUT ACTIVATION FUNCTIONS")
    print("-" * 40)

    # With ReLU
    mlp_relu = MLP([2, 64, 32, 1], activation='relu', lr=0.1, n_epochs=500, random_state=42)
    mlp_relu.fit(X_train, y_train)
    acc_relu = accuracy(y_test, mlp_relu.predict(X_test))

    # Without activation (linear)
    mlp_linear = MLP([2, 64, 32, 1], activation='linear', output_activation='sigmoid',
                    lr=0.1, n_epochs=500, random_state=42)
    mlp_linear.fit(X_train, y_train)
    acc_linear = accuracy(y_test, mlp_linear.predict(X_test))

    print(f"With ReLU activation:    accuracy={acc_relu:.3f}")
    print(f"Without activation:      accuracy={acc_linear:.3f}")
    print("→ WITHOUT NONLINEARITY, DEEP NETWORK = LINEAR MODEL!")

    # -------- Experiment 2: Network Depth --------
    print("\n2. EFFECT OF DEPTH")
    print("-" * 40)

    architectures = [
        [2, 32, 1],           # 1 hidden layer
        [2, 32, 32, 1],       # 2 hidden layers
        [2, 32, 32, 32, 1],   # 3 hidden layers
        [2, 32, 32, 32, 32, 1],  # 4 hidden layers
    ]

    for arch in architectures:
        mlp = MLP(arch, activation='relu', lr=0.1, n_epochs=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy(y_test, mlp.predict(X_test))
        print(f"Layers={len(arch)-1}  architecture={arch}  accuracy={acc:.3f}")
    print("→ Depth helps, but diminishing returns (also harder to optimize)")

    # -------- Experiment 3: Network Width --------
    print("\n3. EFFECT OF WIDTH")
    print("-" * 40)

    for width in [4, 16, 64, 128, 256]:
        mlp = MLP([2, width, 1], activation='relu', lr=0.1, n_epochs=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy(y_test, mlp.predict(X_test))
        print(f"Width={width:<4}  accuracy={acc:.3f}")
    print("→ Wider networks have more capacity (can fit more complex functions)")

    # -------- Experiment 4: Activation Functions --------
    print("\n4. ACTIVATION FUNCTION COMPARISON")
    print("-" * 40)

    for act in ['relu', 'sigmoid', 'tanh']:
        mlp = MLP([2, 64, 32, 1], activation=act, lr=0.1, n_epochs=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy(y_test, mlp.predict(X_test))
        final_loss = mlp.loss_history[-1]
        print(f"activation={act:<8}  accuracy={acc:.3f}  final_loss={final_loss:.4f}")
    print("→ ReLU usually trains faster, sigmoid/tanh can have vanishing gradients")

    # -------- Experiment 5: Learning Rate --------
    print("\n5. LEARNING RATE SWEEP")
    print("-" * 40)

    for lr in [0.001, 0.01, 0.1, 0.5, 1.0]:
        mlp = MLP([2, 64, 1], activation='relu', lr=lr, n_epochs=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy(y_test, mlp.predict(X_test))
        final_loss = mlp.loss_history[-1] if mlp.loss_history else float('inf')
        print(f"lr={lr:<5}  accuracy={acc:.3f}  final_loss={final_loss:.4f}")
    print("→ Too low: slow. Too high: unstable. Need to tune.")

    # -------- Experiment 6: Initialization --------
    print("\n6. WEIGHT INITIALIZATION")
    print("-" * 40)

    for init in ['random', 'xavier', 'he']:
        mlp = MLP([2, 64, 32, 1], activation='relu', lr=0.1, n_epochs=500,
                 init=init, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy(y_test, mlp.predict(X_test))
        print(f"init={init:<8}  accuracy={acc:.3f}")
    print("→ Proper initialization (Xavier/He) prevents vanishing/exploding gradients")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate MLP on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: MLP on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        input_dim = X_tr.shape[1]
        mlp = MLP([input_dim, 64, 32, 1], activation='relu', lr=0.1,
                 n_epochs=500, random_state=42)
        mlp.fit(X_tr, y_tr)
        y_pred = mlp.predict(X_te)
        acc = accuracy(y_te, y_pred)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    datasets = get_2d_datasets()
    plot_datasets = {k: v for k, v in datasets.items() if k != 'clustered'}

    n = len(plot_datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        mlp = MLP([2, 64, 32, 1], activation='relu', lr=0.1, n_epochs=500, random_state=42)
        mlp.fit(X_tr, y_tr)
        acc = accuracy(y_te, mlp.predict(X_te))

        plot_decision_boundary(mlp.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('MLP: Decision Boundaries\n'
                 '(Universal approximator — can learn any smooth boundary)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_activation_effect():
    """Show how removing activations collapses the network."""
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['circles']
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # With ReLU
    mlp_relu = MLP([2, 64, 32, 1], activation='relu', lr=0.1, n_epochs=500, random_state=42)
    mlp_relu.fit(X_train, y_train)
    acc_relu = accuracy(y_test, mlp_relu.predict(X_test))
    plot_decision_boundary(mlp_relu.predict, X, y, ax=axes[0],
                          title=f'With ReLU (acc={acc_relu:.2f})')

    # With linear (no activation)
    mlp_linear = MLP([2, 64, 32, 1], activation='linear', output_activation='sigmoid',
                    lr=0.1, n_epochs=500, random_state=42)
    mlp_linear.fit(X_train, y_train)
    acc_linear = accuracy(y_test, mlp_linear.predict(X_test))
    plot_decision_boundary(mlp_linear.predict, X, y, ax=axes[1],
                          title=f'No Activation (acc={acc_linear:.2f})')

    # Compare logistic regression
    from importlib import import_module
    logreg_module = import_module('02_logistic_regression')
    logreg = logreg_module.LogisticRegression(lr=0.1, n_iters=1000)
    logreg.fit(X_train, y_train)
    acc_logreg = accuracy(y_test, logreg.predict(X_test))
    plot_decision_boundary(logreg.predict, X, y, ax=axes[2],
                          title=f'Logistic Regression (acc={acc_logreg:.2f})')

    plt.suptitle('CRITICAL: Without Activation, Deep Network = Linear Model\n'
                 'The nonlinearity is ESSENTIAL!',
                 fontsize=12)
    plt.tight_layout()
    return fig


def visualize_hidden_space_transformation():
    """
    THE KEY VISUALIZATION: Show how MLP transforms data layer by layer.

    This is THE fundamental intuition:
        "The network UNTANGLES the data until it becomes linearly separable"

    For XOR/Circles:
        - Input space: Classes are interleaved, NOT linearly separable
        - Hidden space: Network stretches/folds space to SEPARATE classes
        - Output: A simple line can now divide them
    """
    np.random.seed(42)

    # Use XOR - the classic non-linearly separable problem
    n_points = 100
    X = np.random.randn(n_points, 2) * 0.5
    X[:25] += np.array([0, 0])
    X[25:50] += np.array([2, 2])
    X[50:75] += np.array([2, 0])
    X[75:] += np.array([0, 2])
    y = np.array([0]*25 + [0]*25 + [1]*25 + [1]*25)

    # Train MLP with architecture that allows visualization
    # [2, 3, 2, 1] - we can visualize the 2D hidden layers!
    mlp = MLP([2, 8, 2, 1], activation='relu', lr=0.1, n_epochs=1000, random_state=42)
    mlp.fit(X, y)

    # Get activations at each layer
    def get_layer_activations(mlp, X):
        """Extract activations at each layer."""
        activations = [X]  # Input
        a = X
        for i in range(len(mlp.weights)):
            z = a @ mlp.weights[i] + mlp.biases[i]
            if i == len(mlp.weights) - 1:
                a = mlp.output_activation.forward(z)
            else:
                a = mlp.activation.forward(z)
            activations.append(a)
        return activations

    activations = get_layer_activations(mlp, X)

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Row 1: The transformation story
    # Plot 1: Input space
    ax1 = fig.add_subplot(2, 4, 1)
    colors = ['blue' if yi == 0 else 'red' for yi in y]
    ax1.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, edgecolors='k', s=50)
    ax1.set_title('INPUT SPACE\n(NOT linearly separable)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    # Draw a line showing linear separation is impossible
    ax1.text(0.5, -0.15, '❌ No line can separate!', transform=ax1.transAxes,
             ha='center', fontsize=9, color='red')
    ax1.set_aspect('equal')

    # Plot 2: After first hidden layer (8D projected to 2D via PCA)
    ax2 = fig.add_subplot(2, 4, 2)
    h1 = activations[1]  # Shape: (n, 8)
    # Use PCA to project to 2D for visualization
    h1_centered = h1 - h1.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(h1_centered, full_matrices=False)
        h1_2d = h1_centered @ Vt[:2].T
    except:
        h1_2d = h1[:, :2]  # Fallback: just take first 2 dimensions
    ax2.scatter(h1_2d[:, 0], h1_2d[:, 1], c=colors, alpha=0.7, edgecolors='k', s=50)
    ax2.set_title('AFTER LAYER 1\n(8D → 2D via PCA)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.text(0.5, -0.15, 'Starting to separate...', transform=ax2.transAxes,
             ha='center', fontsize=9, color='orange')

    # Plot 3: After second hidden layer (2D - can visualize directly!)
    ax3 = fig.add_subplot(2, 4, 3)
    h2 = activations[2]  # Shape: (n, 2)
    ax3.scatter(h2[:, 0], h2[:, 1], c=colors, alpha=0.7, edgecolors='k', s=50)
    ax3.set_title('AFTER LAYER 2\n(2D hidden space)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('h₁')
    ax3.set_ylabel('h₂')

    # Try to draw a separating line if possible
    # Simple heuristic: find the midpoint between class means
    class0_mean = h2[np.array(y) == 0].mean(axis=0)
    class1_mean = h2[np.array(y) == 1].mean(axis=0)

    ax3.text(0.5, -0.15, '✓ NOW linearly separable!', transform=ax3.transAxes,
             ha='center', fontsize=9, color='green', fontweight='bold')

    # Plot 4: Output (1D)
    ax4 = fig.add_subplot(2, 4, 4)
    output = activations[3].flatten()
    ax4.scatter(output, np.zeros_like(output), c=colors, alpha=0.7, edgecolors='k', s=50)
    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision threshold')
    ax4.set_title('OUTPUT\n(1D probability)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('P(class=1)')
    ax4.set_yticks([])
    ax4.set_xlim(-0.1, 1.1)
    ax4.legend(fontsize=8)
    ax4.text(0.5, -0.15, '✓ Simple threshold works!', transform=ax4.transAxes,
             ha='center', fontsize=9, color='green', fontweight='bold')

    # Row 2: Same story with CIRCLES dataset (more dramatic)
    datasets = get_2d_datasets()
    X_c, _, y_c, _ = datasets['circles']

    mlp_c = MLP([2, 16, 2, 1], activation='relu', lr=0.1, n_epochs=500, random_state=42)
    mlp_c.fit(X_c, y_c)
    activations_c = get_layer_activations(mlp_c, X_c)
    colors_c = ['blue' if yi == 0 else 'red' for yi in y_c]

    # Input
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.scatter(X_c[:, 0], X_c[:, 1], c=colors_c, alpha=0.6, edgecolors='k', s=30)
    ax5.set_title('CIRCLES: Input\n(Concentric rings)', fontsize=10)
    ax5.set_aspect('equal')

    # After layer 1 (PCA projection)
    ax6 = fig.add_subplot(2, 4, 6)
    h1_c = activations_c[1]
    h1_c_centered = h1_c - h1_c.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(h1_c_centered, full_matrices=False)
        h1_c_2d = h1_c_centered @ Vt[:2].T
    except:
        h1_c_2d = h1_c[:, :2]
    ax6.scatter(h1_c_2d[:, 0], h1_c_2d[:, 1], c=colors_c, alpha=0.6, edgecolors='k', s=30)
    ax6.set_title('After Layer 1\n(16D → 2D PCA)', fontsize=10)

    # After layer 2 (2D)
    ax7 = fig.add_subplot(2, 4, 7)
    h2_c = activations_c[2]
    ax7.scatter(h2_c[:, 0], h2_c[:, 1], c=colors_c, alpha=0.6, edgecolors='k', s=30)
    ax7.set_title('After Layer 2\n(Network "unrolls" the circles!)', fontsize=10)

    # Output
    ax8 = fig.add_subplot(2, 4, 8)
    output_c = activations_c[3].flatten()
    ax8.scatter(output_c, np.zeros_like(output_c), c=colors_c, alpha=0.6, edgecolors='k', s=30)
    ax8.axvline(x=0.5, color='green', linestyle='--', linewidth=2)
    ax8.set_title('Output\n(Clean separation)', fontsize=10)
    ax8.set_xlim(-0.1, 1.1)
    ax8.set_yticks([])

    plt.suptitle('THE KEY INSIGHT: MLP Transforms Space Until Data Becomes Linearly Separable\n'
                 'Each layer bends, stretches, and folds the space to "untangle" the classes',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_gradient_flow():
    """
    Visualize gradient magnitudes through layers during training.

    Shows:
    1. Gradient magnitude per layer (are they vanishing?)
    2. Comparison: deep vs shallow networks
    3. Effect of activation function on gradient flow
    """
    np.random.seed(42)
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ============ Row 1: Gradient magnitude per layer ============

    # Train networks and track gradient magnitudes
    def train_and_track_gradients(layer_sizes, X, y, activation='relu', n_epochs=200):
        """Train and return gradient history."""
        mlp = MLP(layer_sizes, activation=activation, lr=0.1, n_epochs=1, random_state=42)
        mlp._init_weights()

        gradient_history = {i: [] for i in range(len(layer_sizes) - 1)}
        loss_history = []

        for epoch in range(n_epochs):
            # Forward
            y_pred = mlp._forward(X)
            loss = mlp._compute_loss(y_pred, y)
            loss_history.append(loss)

            # Backward - capture gradients
            m = y.shape[0]
            n_layers = len(mlp.weights)
            a_out = mlp.a_cache[-1]
            z_out = mlp.z_cache[-1]

            if isinstance(mlp.output_activation, Sigmoid):
                delta = a_out - y.reshape(-1, 1)
            else:
                delta = (a_out - y.reshape(-1, 1)) * mlp.output_activation.backward(z_out)

            for l in reversed(range(n_layers)):
                a_prev = mlp.a_cache[l]
                dW = (a_prev.T @ delta) / m
                gradient_history[l].append(np.linalg.norm(dW))

                if l > 0:
                    delta = (delta @ mlp.weights[l].T) * mlp.activation.backward(mlp.z_cache[l-1])

            # Update
            mlp._forward(X)
            dW_list, db_list = mlp._backward(y)
            mlp._update_weights(dW_list, db_list)

        return gradient_history, loss_history

    # Shallow network
    grad_shallow, loss_shallow = train_and_track_gradients([2, 32, 1], X_train, y_train)

    # Deep network
    grad_deep, loss_deep = train_and_track_gradients([2, 16, 16, 16, 16, 1], X_train, y_train)

    # Plot 1: Shallow network gradients
    ax1 = axes[0, 0]
    for layer_idx, grads in grad_shallow.items():
        ax1.plot(grads, label=f'Layer {layer_idx}', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('||∂L/∂W||')
    ax1.set_title('SHALLOW Network [2,32,1]\nGradients stay healthy', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Deep network gradients
    ax2 = axes[0, 1]
    for layer_idx, grads in grad_deep.items():
        ax2.plot(grads, label=f'Layer {layer_idx}', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('||∂L/∂W||')
    ax2.set_title('DEEP Network [2,16,16,16,16,1]\nEarly layers get smaller gradients', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final gradient magnitude by layer
    ax3 = axes[0, 2]
    # Average gradient magnitude over last 50 epochs
    shallow_final = [np.mean(grads[-50:]) for grads in grad_shallow.values()]
    deep_final = [np.mean(grads[-50:]) for grads in grad_deep.values()]

    x_shallow = np.arange(len(shallow_final))
    x_deep = np.arange(len(deep_final))

    ax3.bar(x_shallow - 0.2, shallow_final, width=0.4, label='Shallow', color='steelblue')
    ax3.bar(x_deep + 0.2, deep_final, width=0.4, label='Deep', color='coral')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Avg ||∂L/∂W|| (last 50 epochs)')
    ax3.set_title('Gradient Magnitude by Layer\n(Deep nets: earlier layers get less gradient)', fontsize=10)
    ax3.legend()
    ax3.set_yscale('log')

    # ============ Row 2: Effect of activation functions ============

    # ReLU vs Sigmoid vs Tanh
    grad_relu, _ = train_and_track_gradients([2, 16, 16, 16, 1], X_train, y_train, activation='relu')
    grad_sigmoid, _ = train_and_track_gradients([2, 16, 16, 16, 1], X_train, y_train, activation='sigmoid')
    grad_tanh, _ = train_and_track_gradients([2, 16, 16, 16, 1], X_train, y_train, activation='tanh')

    # Plot 4: ReLU gradients
    ax4 = axes[1, 0]
    for layer_idx, grads in grad_relu.items():
        ax4.plot(grads, label=f'Layer {layer_idx}', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('||∂L/∂W||')
    ax4.set_title('ReLU Activation\n(Gradients flow well)', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Sigmoid gradients
    ax5 = axes[1, 1]
    for layer_idx, grads in grad_sigmoid.items():
        ax5.plot(grads, label=f'Layer {layer_idx}', alpha=0.8)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('||∂L/∂W||')
    ax5.set_title('Sigmoid Activation\n(Gradients VANISH in early layers!)', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Comparison of first layer gradient
    ax6 = axes[1, 2]
    ax6.plot(grad_relu[0], label='ReLU', color='green', alpha=0.8)
    ax6.plot(grad_sigmoid[0], label='Sigmoid', color='red', alpha=0.8)
    ax6.plot(grad_tanh[0], label='Tanh', color='blue', alpha=0.8)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('||∂L/∂W|| for Layer 0')
    ax6.set_title('FIRST Layer Gradient Comparison\n(Sigmoid causes vanishing gradients)', fontsize=10)
    ax6.legend()
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('GRADIENT FLOW: Why Deep Networks Can Be Hard to Train\n'
                 'Gradients must flow backward through ALL layers — they can vanish or explode',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_learning_dynamics():
    """
    Visualize how the decision boundary evolves during training.

    Shows:
    1. Decision boundary at different epochs
    2. Loss curve alongside
    3. The "learning trajectory" in weight space
    """
    np.random.seed(42)
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    fig = plt.figure(figsize=(16, 8))

    # Train and capture snapshots
    epochs_to_capture = [1, 5, 20, 50, 100, 200, 500]
    snapshots = {}

    mlp = MLP([2, 32, 16, 1], activation='relu', lr=0.1, n_epochs=1, random_state=42)
    mlp._init_weights()

    all_losses = []

    for epoch in range(1, 501):
        # Train one epoch
        n_samples = X_train.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        batch_size = 32
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            y_pred = mlp._forward(X_batch)
            dW, db = mlp._backward(y_batch)
            mlp._update_weights(dW, db)

        # Track loss
        y_pred_all = mlp._forward(X_train)
        loss = mlp._compute_loss(y_pred_all, y_train)
        all_losses.append(loss)

        # Capture snapshot
        if epoch in epochs_to_capture:
            # Deep copy weights
            snapshots[epoch] = {
                'weights': [w.copy() for w in mlp.weights],
                'biases': [b.copy() for b in mlp.biases],
                'loss': loss,
                'acc': accuracy(y_test, mlp.predict(X_test))
            }

    # Plot decision boundaries at different epochs
    n_snapshots = len(epochs_to_capture)

    for idx, epoch in enumerate(epochs_to_capture):
        ax = fig.add_subplot(2, 4, idx + 1)

        # Restore weights
        snapshot = snapshots[epoch]
        mlp.weights = snapshot['weights']
        mlp.biases = snapshot['biases']

        # Plot decision boundary
        plot_decision_boundary(mlp.predict, X, y, ax=ax,
                              title=f'Epoch {epoch}\nloss={snapshot["loss"]:.3f}, acc={snapshot["acc"]:.2f}')

    # Plot loss curve
    ax_loss = fig.add_subplot(2, 4, 8)
    ax_loss.plot(all_losses, 'b-', linewidth=1.5)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss\n(Learning is gradual refinement)')
    ax_loss.grid(True, alpha=0.3)

    # Mark captured epochs
    for epoch in epochs_to_capture:
        if epoch <= len(all_losses):
            ax_loss.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
            ax_loss.scatter([epoch], [all_losses[epoch-1]], color='red', s=50, zorder=5)

    plt.suptitle('LEARNING DYNAMICS: How the Decision Boundary Evolves\n'
                 'The network starts random, then gradually learns the pattern',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("MULTI-LAYER PERCEPTRON — Paradigm: LEARNED FEATURES")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Stack of [Linear → Activation] layers.
    Learns features automatically through backpropagation.

UNIVERSAL APPROXIMATION:
    One hidden layer can approximate ANY continuous function
    (given enough neurons).

CRITICAL INSIGHT:
    WITHOUT NONLINEAR ACTIVATIONS, A DEEP NETWORK
    COLLAPSES TO A SINGLE LINEAR TRANSFORMATION!

    W₂(W₁x + b₁) + b₂ = W₂W₁x + (W₂b₁ + b₂) = W'x + b'

KEY COMPONENTS:
    - Depth: more layers = more abstraction levels
    - Width: more neurons = more capacity
    - Activation: ReLU is default, essential for nonlinearity
    - Initialization: Xavier/He prevent vanishing gradients
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize
    print("\nGenerating visualizations...")

    # 1. Decision boundaries (existing)
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/12_mlp_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved decision boundaries to: {save_path1}")
    plt.close(fig1)

    # 2. Activation effect (existing)
    fig2 = visualize_activation_effect()
    save_path2 = '/Users/sid47/ML Algorithms/12_mlp_activation.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved activation effect to: {save_path2}")
    plt.close(fig2)

    # 3. Hidden space transformation (NEW - THE KEY INSIGHT!)
    fig3 = visualize_hidden_space_transformation()
    save_path3 = '/Users/sid47/ML Algorithms/12_mlp_hidden_space.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved hidden space transformation to: {save_path3}")
    plt.close(fig3)

    # 4. Gradient flow (NEW)
    fig4 = visualize_gradient_flow()
    save_path4 = '/Users/sid47/ML Algorithms/12_mlp_gradients.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight')
    print(f"Saved gradient flow to: {save_path4}")
    plt.close(fig4)

    # 5. Learning dynamics (NEW)
    fig5 = visualize_learning_dynamics()
    save_path5 = '/Users/sid47/ML Algorithms/12_mlp_learning.png'
    fig5.savefig(save_path5, dpi=150, bbox_inches='tight')
    print(f"Saved learning dynamics to: {save_path5}")
    plt.close(fig5)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What MLP Reveals")
    print("="*60)
    print("""
1. ACTIVATION FUNCTIONS ARE ESSENTIAL — without them, deep = linear
2. Backprop = just the chain rule applied recursively
3. Depth adds abstraction levels, width adds capacity
4. Initialization matters — Xavier/He prevent gradient issues
5. Learning rate needs tuning — too high diverges, too low stalls

===============================================================
THE KEY INSIGHT (see 12_mlp_hidden_space.png):
===============================================================

    The network TRANSFORMS the input space layer by layer
    until the data becomes LINEARLY SEPARABLE.

    XOR/Circles in input space → NOT separable
    After hidden layers → Classes are UNTANGLED
    Final layer → Simple threshold works!

    This is what "learning features" means:
    The network learns a COORDINATE SYSTEM where
    the problem becomes trivial.

VISUALIZATIONS GENERATED:
    1. 12_mlp_boundaries.png    — Decision boundaries (universal approximation)
    2. 12_mlp_activation.png    — Why activation functions matter
    3. 12_mlp_hidden_space.png  — THE KEY: Space transformation layer by layer
    4. 12_mlp_gradients.png     — Gradient flow (vanishing gradient problem)
    5. 12_mlp_learning.png      — How the boundary evolves during training

NEXT: CNN — exploit spatial structure through convolution
    """)
