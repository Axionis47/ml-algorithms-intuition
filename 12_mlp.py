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
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/12_mlp_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_activation_effect()
    save_path2 = '/Users/sid47/ML Algorithms/12_mlp_activation.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved activation effect to: {save_path2}")
    plt.close(fig2)

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

KEY ARCHITECTURAL INSIGHT:
    The power of neural networks comes from COMPOSING
    nonlinear functions. Each layer transforms the
    representation to make the task easier for the next layer.

NEXT: CNN — exploit spatial structure through convolution
    """)
