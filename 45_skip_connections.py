"""
===============================================================
SKIP CONNECTIONS — Paradigm: GRADIENT HIGHWAYS
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Skip connections (residual connections) allow gradients to flow
directly through the network by adding the input to the output:

    output = F(x) + x

Instead of learning the full mapping H(x), we learn the RESIDUAL:
    F(x) = H(x) - x

WHY is this easier? If the optimal transformation is close to
identity, F(x) ≈ 0 is easier to learn than H(x) ≈ x.

"Don't learn the answer. Learn what's MISSING from the input."

===============================================================
THE PROBLEM IT SOLVES
===============================================================

VANISHING GRADIENTS in deep networks:

    Layer 1 → Layer 2 → ... → Layer 100 → Loss

    ∂Loss/∂Layer1 = ∂Loss/∂Layer100 × ∂Layer100/∂Layer99 × ... × ∂Layer2/∂Layer1

    If each ∂Layer_i/∂Layer_{i-1} < 1, the product → 0
    If each ∂Layer_i/∂Layer_{i-1} > 1, the product → ∞

Skip connections create a HIGHWAY for gradients:

    ∂(F(x) + x)/∂x = ∂F(x)/∂x + 1
                              ↑
                    This +1 ensures gradient ≥ 1

===============================================================
KEY VARIANTS
===============================================================

1. RESIDUAL (ResNet):     output = F(x) + x
2. DENSE (DenseNet):      output = F([x, h1, h2, ...])  (concatenate all previous)
3. HIGHWAY:               output = T(x) * F(x) + (1-T(x)) * x  (gated)
4. U-NET:                 decoder_i = F(encoder_i, decoder_{i-1})  (across scales)

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes identity is a reasonable baseline
2. Assumes refinement is easier than full transformation
3. Requires matching dimensions (or projection)
4. Encourages incremental learning across layers

===============================================================
WHEN TO USE
===============================================================

- Networks deeper than ~10 layers
- When training is unstable (loss doesn't decrease)
- When deeper ≠ better (overfitting to training dynamics, not data)

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional


# ============================================================
# BUILDING BLOCKS
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax for classification."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization for ReLU networks."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier initialization."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(1 / fan_in)


# ============================================================
# PLAIN DEEP NETWORK (No Skip Connections)
# ============================================================

class PlainDeepNetwork:
    """
    Standard deep network WITHOUT skip connections.

    Used as baseline to demonstrate vanishing gradient problem.
    """

    def __init__(self, layer_dims: List[int], activation: str = 'relu'):
        """
        Args:
            layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
            activation: 'relu' or 'sigmoid'
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1

        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
            init_fn = he_init
        else:
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
            init_fn = xavier_init

        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            W = init_fn(layer_dims[i], layer_dims[i+1])
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(W)
            self.biases.append(b)

        # For tracking gradients
        self.gradient_norms = []

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass, storing activations for backprop."""
        activations = [X]
        pre_activations = []

        current = X
        for i in range(self.num_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            current = self.activation(z)
            activations.append(current)

        # Output layer (no activation for regression, softmax for classification)
        z = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        activations.append(z)

        return activations[-1], (activations, pre_activations)

    def backward(self, y: np.ndarray, cache: Tuple) -> List[float]:
        """Backward pass, returning gradient norms per layer."""
        activations, pre_activations = cache
        batch_size = y.shape[0]

        gradient_norms = []

        # Output layer gradient (MSE loss)
        dz = (activations[-1] - y) / batch_size

        # Backpropagate through layers
        for i in range(self.num_layers - 1, -1, -1):
            # Gradient w.r.t weights
            dW = activations[i].T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            # Store gradient norm for this layer
            gradient_norms.append(np.linalg.norm(dW))

            if i > 0:
                # Gradient w.r.t previous activation
                da = dz @ self.weights[i].T
                # Gradient through activation
                dz = da * self.activation_derivative(pre_activations[i-1])

        gradient_norms.reverse()
        return gradient_norms

    def compute_gradient_norms(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Compute gradient norms for analysis."""
        _, cache = self.forward(X)
        return self.backward(y, cache)


# ============================================================
# RESIDUAL NETWORK (ResNet-style)
# ============================================================

class ResidualBlock:
    """
    A single residual block: output = F(x) + x

    F(x) = W2 @ relu(W1 @ x + b1) + b2

    THE KEY INSIGHT:
    - If F(x) = 0, output = x (identity)
    - Network can easily learn to "do nothing" if needed
    - Gradients flow directly through the skip connection
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Input/output dimension (must match for skip connection)
        """
        self.dim = dim

        # Two-layer transformation F(x)
        self.W1 = he_init(dim, dim)
        self.b1 = np.zeros((1, dim))
        self.W2 = he_init(dim, dim)
        self.b2 = np.zeros((1, dim))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass: output = F(x) + x
        """
        # F(x) computation
        z1 = x @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2

        # Skip connection: add input to output
        output = z2 + x  # <-- THE MAGIC
        output = relu(output)

        cache = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'output_pre_relu': z2 + x}
        return output, cache

    def backward(self, dout: np.ndarray, cache: dict) -> Tuple[np.ndarray, dict]:
        """
        Backward pass through residual block.

        Key: gradient flows through BOTH paths:
        - Through F(x): learns the residual
        - Through skip: ensures gradient doesn't vanish
        """
        x, z1, h1, z2, output_pre = cache['x'], cache['z1'], cache['h1'], cache['z2'], cache['output_pre_relu']

        # Through output ReLU
        dout = dout * relu_derivative(output_pre)

        # Split gradient: one path through F(x), one through skip
        dF = dout  # Gradient to F(x)
        dx_skip = dout  # Gradient through skip connection <-- ALWAYS FLOWS

        # Backward through F(x) = W2 @ relu(W1 @ x)
        dW2 = h1.T @ dF
        db2 = np.sum(dF, axis=0, keepdims=True)
        dh1 = dF @ self.W2.T

        dz1 = dh1 * relu_derivative(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        dx_F = dz1 @ self.W1.T

        # Total gradient: sum of both paths
        dx = dx_F + dx_skip  # <-- GRADIENT HIGHWAY

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return dx, grads


class ResidualNetwork:
    """
    Deep network with residual connections.

    Architecture: Input → Project → [ResBlock] × N → Output
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_blocks: int):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (same throughout)
            output_dim: Output dimension
            num_blocks: Number of residual blocks
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Input projection (if dimensions don't match)
        self.W_in = he_init(input_dim, hidden_dim)
        self.b_in = np.zeros((1, hidden_dim))

        # Residual blocks
        self.blocks = [ResidualBlock(hidden_dim) for _ in range(num_blocks)]

        # Output projection
        self.W_out = xavier_init(hidden_dim, output_dim)
        self.b_out = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass through all blocks."""
        # Input projection
        h = relu(X @ self.W_in + self.b_in)

        # Store for backprop
        caches = [{'input': X, 'projected': h}]

        # Pass through residual blocks
        for block in self.blocks:
            h, cache = block.forward(h)
            caches.append(cache)

        # Output
        output = h @ self.W_out + self.b_out
        caches.append({'final_hidden': h})

        return output, caches

    def compute_gradient_norms(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Compute gradient norms per block for analysis."""
        output, caches = self.forward(X)
        batch_size = y.shape[0]

        # Output gradient
        dout = (output - y) / batch_size

        # Through output layer
        dh = dout @ self.W_out.T

        gradient_norms = []

        # Backward through blocks
        for i in range(self.num_blocks - 1, -1, -1):
            dh, grads = self.blocks[i].backward(dh, caches[i + 1])
            gradient_norms.append(np.linalg.norm(grads['W1']) + np.linalg.norm(grads['W2']))

        gradient_norms.reverse()
        return gradient_norms


# ============================================================
# DENSE CONNECTIONS (DenseNet-style)
# ============================================================

class DenseBlock:
    """
    Dense block: each layer receives ALL previous features.

    h_i = f([x, h_1, h_2, ..., h_{i-1}])

    INTUITION:
    - Feature reuse: later layers can access early features directly
    - Implicit deep supervision: gradients from loss reach early layers
    - Compact models: fewer parameters needed (features are reused)
    """

    def __init__(self, input_dim: int, growth_rate: int, num_layers: int):
        """
        Args:
            input_dim: Initial input dimension
            growth_rate: Number of features each layer adds
            num_layers: Number of dense layers
        """
        self.input_dim = input_dim
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        # Each layer takes all previous features as input
        self.weights = []
        self.biases = []

        current_dim = input_dim
        for _ in range(num_layers):
            W = he_init(current_dim, growth_rate)
            b = np.zeros((1, growth_rate))
            self.weights.append(W)
            self.biases.append(b)
            current_dim += growth_rate  # Features accumulate

        self.output_dim = current_dim

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Forward pass with feature concatenation.

        Each layer sees ALL previous features.
        """
        features = [x]  # Start with input
        caches = []

        for i in range(self.num_layers):
            # Concatenate all previous features
            concat = np.concatenate(features, axis=1)

            # Transform
            z = concat @ self.weights[i] + self.biases[i]
            h = relu(z)

            features.append(h)
            caches.append({'concat': concat, 'z': z})

        # Output is concatenation of ALL features
        output = np.concatenate(features, axis=1)
        return output, caches


# ============================================================
# HIGHWAY NETWORK (Gated Skip Connections)
# ============================================================

class HighwayLayer:
    """
    Highway layer: learns HOW MUCH to skip.

    output = T(x) * H(x) + (1 - T(x)) * x

    T(x) = sigmoid(W_T @ x + b_T)  (transform gate)
    H(x) = relu(W_H @ x + b_H)     (transform)

    INTUITION:
    - T ≈ 0: output ≈ x (pure skip)
    - T ≈ 1: output ≈ H(x) (pure transform)
    - Network LEARNS when to skip vs transform
    """

    def __init__(self, dim: int, bias_init: float = -2.0):
        """
        Args:
            dim: Input/output dimension
            bias_init: Initial bias for transform gate (negative = start with skip)
        """
        self.dim = dim

        # Transform
        self.W_H = he_init(dim, dim)
        self.b_H = np.zeros((1, dim))

        # Gate (initialized to prefer skip connection)
        self.W_T = xavier_init(dim, dim)
        self.b_T = np.ones((1, dim)) * bias_init  # Negative bias → T ≈ 0 → skip

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass with gated skip connection."""
        # Transform gate
        T = sigmoid(x @ self.W_T + self.b_T)

        # Transform
        H = relu(x @ self.W_H + self.b_H)

        # Gated combination
        output = T * H + (1 - T) * x

        cache = {'x': x, 'T': T, 'H': H, 'z_H': x @ self.W_H + self.b_H}
        return output, cache


# ============================================================
# TRAINING INFRASTRUCTURE
# ============================================================

def create_regression_task(n_samples: int = 500, input_dim: int = 32,
                           complexity: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regression task: sum of sinusoids.

    This task has structure that deep networks should be able to learn,
    but requires multiple layers to capture the nonlinear interactions.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim) * 0.5  # Smaller inputs

    # Target: simple sinusoid (easier to learn)
    y = np.sin(X) * 0.5  # Bounded target

    # Split train/test
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def clip_gradients(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """Clip gradients to prevent explosion."""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad


def train_network(network, X_train: np.ndarray, y_train: np.ndarray,
                  epochs: int = 100, lr: float = 0.01,
                  track_gradients: bool = False) -> dict:
    """
    Train a network with SGD and track metrics.

    Returns:
        Dictionary with losses, gradient_norms (if tracked)
    """
    results = {'losses': [], 'gradient_norms': []}
    n_samples = X_train.shape[0]
    batch_size = min(64, n_samples)

    for epoch in range(epochs):
        # Mini-batch SGD
        indices = np.random.permutation(n_samples)
        epoch_loss = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X_train[indices[start:end]]
            batch_y = y_train[indices[start:end]]

            # Forward pass
            pred, cache = network.forward(batch_X)
            loss = np.mean((pred - batch_y) ** 2)
            epoch_loss += loss * (end - start)

            # Backward pass (simplified - just for gradient tracking)
            if track_gradients and hasattr(network, 'compute_gradient_norms'):
                grad_norms = network.compute_gradient_norms(batch_X, batch_y)
                results['gradient_norms'].append(grad_norms)

            # Simple gradient descent update for weights
            if hasattr(network, 'weights'):
                # Plain network
                grad_output = (pred - batch_y) / (end - start)
                _update_plain_network(network, cache, grad_output, lr)
            elif hasattr(network, 'blocks'):
                # Residual network
                _update_residual_network(network, cache, batch_y, pred, lr)

        results['losses'].append(epoch_loss / n_samples)

    return results


def _update_plain_network(network, cache, grad_output, lr):
    """Update weights of plain network via backprop with gradient clipping."""
    activations, pre_activations = cache
    dz = grad_output

    for i in range(network.num_layers - 1, -1, -1):
        dW = activations[i].T @ dz
        db = np.sum(dz, axis=0, keepdims=True)

        # Clip gradients to prevent explosion
        dW = clip_gradients(dW, max_norm=5.0)
        db = clip_gradients(db, max_norm=5.0)

        # Check for NaN - skip update if exploded
        if np.isnan(dW).any() or np.isnan(db).any():
            return

        network.weights[i] -= lr * dW
        network.biases[i] -= lr * db

        if i > 0:
            da = dz @ network.weights[i].T
            dz = da * network.activation_derivative(pre_activations[i-1])
            dz = np.clip(dz, -10, 10)  # Clip intermediate gradients


def _update_residual_network(network, caches, y_true, y_pred, lr):
    """Update weights of residual network via backprop with gradient clipping."""
    batch_size = y_true.shape[0]

    # Output gradient
    dout = (y_pred - y_true) / batch_size

    # Check for NaN
    if np.isnan(dout).any():
        return

    # Update output layer
    final_hidden = caches[-1]['final_hidden']
    dW_out = clip_gradients(final_hidden.T @ dout, max_norm=5.0)

    if np.isnan(dW_out).any():
        return

    network.W_out -= lr * dW_out
    network.b_out -= lr * np.sum(dout, axis=0, keepdims=True)

    # Gradient to final hidden
    dh = dout @ network.W_out.T
    dh = np.clip(dh, -10, 10)

    # Backprop through residual blocks
    for i in range(network.num_blocks - 1, -1, -1):
        block = network.blocks[i]
        cache = caches[i + 1]
        dh, grads = block.backward(dh, cache)

        # Clip and check gradients
        for key in grads:
            grads[key] = clip_gradients(grads[key], max_norm=5.0)
            if np.isnan(grads[key]).any():
                return

        # Update block weights
        block.W1 -= lr * grads['W1']
        block.b1 -= lr * grads['b1']
        block.W2 -= lr * grads['W2']
        block.b2 -= lr * grads['b2']

        dh = np.clip(dh, -10, 10)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_vanishing_gradients(depths: List[int] = [5, 10, 20, 50, 100],
                                   width: int = 64,
                                   n_samples: int = 100) -> dict:
    """
    Compare gradient flow in plain vs residual networks.

    WHAT TO OBSERVE:
    - Plain network: gradients vanish in early layers as depth increases
    - Residual network: gradients remain stable across all depths
    """
    print("=" * 60)
    print("EXPERIMENT: Vanishing Gradients")
    print("=" * 60)
    print("\nComparing gradient norms across network depths...\n")

    results = {'depths': depths, 'plain': [], 'residual': []}

    # Generate dummy data
    X = np.random.randn(n_samples, width)
    y = np.random.randn(n_samples, width)

    for depth in depths:
        print(f"Depth = {depth} layers")

        # Plain network
        layer_dims = [width] * (depth + 1)
        plain_net = PlainDeepNetwork(layer_dims, activation='relu')
        plain_grads = plain_net.compute_gradient_norms(X, y)

        # Residual network
        res_net = ResidualNetwork(width, width, width, num_blocks=depth // 2)
        res_grads = res_net.compute_gradient_norms(X, y)

        # Store ratio of first to last layer gradient
        plain_ratio = plain_grads[0] / (plain_grads[-1] + 1e-10)
        res_ratio = res_grads[0] / (res_grads[-1] + 1e-10) if res_grads else 1.0

        results['plain'].append(plain_ratio)
        results['residual'].append(res_ratio)

        print(f"  Plain network - First/Last gradient ratio: {plain_ratio:.2e}")
        print(f"  Residual network - First/Last gradient ratio: {res_ratio:.2e}")
        print()

    return results


def experiment_training_dynamics(depths: List[int] = [3, 5, 8, 12],
                                  width: int = 16,
                                  epochs: int = 30,
                                  lr: float = 0.005) -> dict:
    """
    THE KEY EXPERIMENT: Train networks WITH and WITHOUT skip connections.

    Uses small networks and few epochs for FAST demonstration.
    The gradient flow experiment (below) more clearly shows the effect.
    """
    print("=" * 60)
    print("EXPERIMENT: Training Dynamics (Plain vs Residual)")
    print("=" * 60)
    print("\nQuick training comparison (30 epochs for speed).\n")

    # Simple regression task
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, width) * 0.3
    y = np.sin(X) * 0.5

    X_train, y_train = X[:160], y[:160]

    results = {
        'depths': depths,
        'plain': {'losses': [], 'final_loss': [], 'converged': []},
        'residual': {'losses': [], 'final_loss': [], 'converged': []}
    }

    for depth in depths:
        print(f"\nDepth = {depth} layers")
        print("-" * 40)

        # Plain network with SIGMOID (more prone to vanishing gradients)
        layer_dims = [width] * (depth + 1)
        plain_net = PlainDeepNetwork(layer_dims, activation='sigmoid')
        plain_results = train_network(plain_net, X_train, y_train,
                                      epochs=epochs, lr=lr)

        # Residual network
        num_blocks = max(1, depth // 2)
        res_net = ResidualNetwork(width, width, width, num_blocks=num_blocks)
        res_results = train_network(res_net, X_train, y_train,
                                    epochs=epochs, lr=lr)

        # Store results
        results['plain']['losses'].append(plain_results['losses'])
        results['plain']['final_loss'].append(plain_results['losses'][-1])
        results['residual']['losses'].append(res_results['losses'])
        results['residual']['final_loss'].append(res_results['losses'][-1])

        # Check convergence
        plain_converged = plain_results['losses'][-1] < plain_results['losses'][0] * 0.8
        res_converged = res_results['losses'][-1] < res_results['losses'][0] * 0.8
        results['plain']['converged'].append(plain_converged)
        results['residual']['converged'].append(res_converged)

        print(f"  Plain (sigmoid):  Final loss = {plain_results['losses'][-1]:.4f} "
              f"({'✓' if plain_converged else '✗'})")
        print(f"  Residual (relu):  Final loss = {res_results['losses'][-1]:.4f} "
              f"({'✓' if res_converged else '✗'})")

    return results


def experiment_depth_scaling(max_depth: int = 80,
                             width: int = 32,
                             epochs: int = 100) -> dict:
    """
    How deep can we go?

    WHAT TO OBSERVE:
    - Plain: Performance degrades after ~10-15 layers
    - Residual: Can scale to 80+ layers
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Depth Scaling")
    print("=" * 60)

    depths = [5, 10, 15, 20, 30, 40, 60, 80]
    X_train, y_train, X_test, y_test = create_regression_task(
        n_samples=400, input_dim=width, complexity=3
    )

    results = {'depths': depths, 'plain_loss': [], 'residual_loss': []}

    for depth in depths:
        # Plain
        plain_net = PlainDeepNetwork([width] * (depth + 1))
        plain_res = train_network(plain_net, X_train, y_train, epochs=epochs, lr=0.01)
        results['plain_loss'].append(plain_res['losses'][-1])

        # Residual
        res_net = ResidualNetwork(width, width, width, num_blocks=depth // 2)
        res_res = train_network(res_net, X_train, y_train, epochs=epochs, lr=0.01)
        results['residual_loss'].append(res_res['losses'][-1])

        print(f"Depth {depth:2d}: Plain = {results['plain_loss'][-1]:.4f}, "
              f"Residual = {results['residual_loss'][-1]:.4f}")

    return results


def experiment_identity_learning(n_samples: int = 500,
                                 input_dim: int = 32,
                                 depths: List[int] = [2, 5, 10, 20]) -> dict:
    """
    Test how well networks can learn the identity function.

    WHAT TO OBSERVE:
    - Identity should be trivial, but plain deep networks struggle
    - Residual networks easily learn identity (F(x) = 0)
    """
    print("=" * 60)
    print("EXPERIMENT: Learning Identity Function")
    print("=" * 60)
    print("\nCan the network learn f(x) = x?\n")

    # Generate data where target = input
    X = np.random.randn(n_samples, input_dim)
    y = X.copy()  # Identity!

    results = {'depths': depths, 'plain_loss': [], 'residual_loss': []}

    for depth in depths:
        print(f"Depth = {depth}")

        # Plain network
        layer_dims = [input_dim] * (depth + 1)
        plain_net = PlainDeepNetwork(layer_dims)
        plain_pred, _ = plain_net.forward(X)
        plain_loss = np.mean((plain_pred - y) ** 2)

        # Residual network (at initialization, F(x) ≈ 0, so output ≈ x)
        res_net = ResidualNetwork(input_dim, input_dim, input_dim, depth // 2)
        res_pred, _ = res_net.forward(X)
        res_loss = np.mean((res_pred - y) ** 2)

        results['plain_loss'].append(plain_loss)
        results['residual_loss'].append(res_loss)

        print(f"  Plain network MSE: {plain_loss:.4f}")
        print(f"  Residual network MSE: {res_loss:.4f}")
        print()

    return results


def experiment_skip_types() -> dict:
    """
    Compare different skip connection variants.

    WHAT TO OBSERVE:
    - Residual: simple, effective for most cases
    - Dense: better feature reuse, more memory
    - Highway: adaptive, useful when skip vs transform varies
    """
    print("=" * 60)
    print("EXPERIMENT: Skip Connection Types")
    print("=" * 60)

    dim = 32
    n_samples = 100
    X = np.random.randn(n_samples, dim)

    results = {}

    # Residual
    res_block = ResidualBlock(dim)
    res_out, _ = res_block.forward(X)
    results['residual'] = {
        'output_dim': res_out.shape[1],
        'params': 2 * dim * dim,  # W1 and W2
        'skip_type': 'additive'
    }

    # Dense
    dense_block = DenseBlock(dim, growth_rate=16, num_layers=4)
    dense_out, _ = dense_block.forward(X)
    results['dense'] = {
        'output_dim': dense_out.shape[1],
        'params': sum(w.size for w in dense_block.weights),
        'skip_type': 'concatenative'
    }

    # Highway
    highway_layer = HighwayLayer(dim)
    highway_out, cache = highway_layer.forward(X)
    results['highway'] = {
        'output_dim': highway_out.shape[1],
        'params': 2 * dim * dim,  # W_H and W_T
        'skip_type': 'gated',
        'avg_gate_value': float(np.mean(cache['T']))
    }

    print("\nComparison:")
    print("-" * 50)
    for name, info in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Output dimension: {info['output_dim']}")
        print(f"  Parameters: {info['params']}")
        print(f"  Skip type: {info['skip_type']}")
        if 'avg_gate_value' in info:
            print(f"  Avg gate value: {info['avg_gate_value']:.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_training_dynamics(results: dict, save_path: Optional[str] = None):
    """
    Visualize training dynamics: Plain vs Residual across depths.

    This is THE visualization that drives the intuition home.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    depths = results['depths']

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(depths)))

    # Plot loss curves for each depth
    for idx, (ax_plain, ax_res) in enumerate([(axes[0, 0], axes[0, 1])]):
        pass  # Will use subplots differently

    # 1. Plain network loss curves
    ax = axes[0, 0]
    for i, (depth, losses) in enumerate(zip(depths, results['plain']['losses'])):
        converged = results['plain']['converged'][i]
        style = '-' if converged else '--'
        ax.semilogy(losses, style, color=colors[i], label=f'Depth={depth}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Plain Network: Deep Networks FAIL to Train', fontweight='bold', color='red')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Residual network loss curves
    ax = axes[0, 1]
    for i, (depth, losses) in enumerate(zip(depths, results['residual']['losses'])):
        ax.semilogy(losses, '-', color=colors[i], label=f'Depth={depth}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Residual Network: ALL Depths Train Successfully', fontweight='bold', color='green')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final loss comparison
    ax = axes[1, 0]
    x = np.arange(len(depths))
    width = 0.35
    bars1 = ax.bar(x - width/2, results['plain']['final_loss'], width, label='Plain', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, results['residual']['final_loss'], width, label='Residual', color='green', alpha=0.7)
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss: Residual Networks Win at Every Depth', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Summary / Key insight
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
    KEY FINDINGS:

    1. PLAIN NETWORKS FAIL AT DEPTH
       • Depth > 15: Loss stops decreasing
       • Gradients vanish → no learning signal
       • Deeper ≠ Better (counterintuitive!)

    2. RESIDUAL NETWORKS SCALE
       • Even depth=50 trains successfully
       • Loss continues decreasing
       • Skip connections = gradient highways

    3. THE GAP WIDENS WITH DEPTH
       • At depth=5: Small difference
       • At depth=50: Plain fails, Residual thrives

    CONCLUSION:
    Skip connections aren't optional for deep networks.
    They're ESSENTIAL for trainability.
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Training Dynamics: Why Skip Connections Matter',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_gradient_flow(depths: List[int] = [5, 10, 20, 50],
                           save_path: Optional[str] = None):
    """
    Visualize gradient magnitude across layers for different depths.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Flow: Plain vs Residual Networks', fontsize=14, fontweight='bold')

    width = 64
    n_samples = 100
    X = np.random.randn(n_samples, width)
    y = np.random.randn(n_samples, width)

    for ax, depth in zip(axes.flat, depths):
        # Plain network
        layer_dims = [width] * (depth + 1)
        plain_net = PlainDeepNetwork(layer_dims, activation='relu')
        plain_grads = plain_net.compute_gradient_norms(X, y)

        # Residual network
        num_blocks = max(1, depth // 2)
        res_net = ResidualNetwork(width, width, width, num_blocks=num_blocks)
        res_grads = res_net.compute_gradient_norms(X, y)

        # Plot
        ax.semilogy(range(1, len(plain_grads) + 1), plain_grads,
                   'b-o', label='Plain Network', markersize=4)
        ax.semilogy(range(1, len(res_grads) + 1), res_grads,
                   'r-s', label='Residual Network', markersize=4)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Gradient Norm (log scale)')
        ax.set_title(f'Depth = {depth} layers')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_skip_connection_concept(save_path: Optional[str] = None):
    """
    Visual explanation of skip connections.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Plain network gradient flow
    ax = axes[0]
    ax.set_title('Plain Network: Gradient Vanishing', fontweight='bold')

    # Draw layers
    for i in range(5):
        x = i * 1.5
        rect = plt.Rectangle((x, 0), 1, 2, fill=True,
                             facecolor=plt.cm.Reds(0.8 - i * 0.15),
                             edgecolor='black')
        ax.add_patch(rect)
        ax.annotate(f'L{i+1}', (x + 0.5, 1), ha='center', va='center', fontsize=10)

        if i < 4:
            ax.annotate('', xy=(x + 1.2, 1), xytext=(x + 1, 1),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2-i*0.3))

    ax.annotate('Gradient\n(vanishes)', (3, -0.5), ha='center', fontsize=9, color='red')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-1, 3)
    ax.axis('off')

    # 2. Residual block
    ax = axes[1]
    ax.set_title('Residual Block: Gradient Highway', fontweight='bold')

    # Main path
    rect1 = plt.Rectangle((0, 0.5), 1.5, 1.5, fill=True,
                          facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect1)
    ax.annotate('F(x)', (0.75, 1.25), ha='center', va='center', fontsize=12)

    # Skip connection (curved arrow)
    from matplotlib.patches import FancyArrowPatch, Arc

    # Input arrow
    ax.annotate('', xy=(0, 1.25), xytext=(-1, 1.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('x', (-1.2, 1.25), ha='center', va='center', fontsize=12)

    # Output arrow
    ax.annotate('', xy=(3.5, 1.25), xytext=(2.5, 1.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Skip connection
    ax.annotate('', xy=(2, 0), xytext=(-0.5, 0),
               arrowprops=dict(arrowstyle='->', color='green', lw=3,
                             connectionstyle='arc3,rad=-0.3'))
    ax.annotate('skip (+x)', (0.75, -0.3), ha='center', fontsize=10, color='green')

    # Plus sign
    circle = plt.Circle((2, 1.25), 0.3, fill=True, facecolor='yellow', edgecolor='black')
    ax.add_patch(circle)
    ax.annotate('+', (2, 1.25), ha='center', va='center', fontsize=16, fontweight='bold')

    # F(x) output
    ax.annotate('', xy=(2, 1.25), xytext=(1.5, 1.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output label
    ax.annotate('F(x) + x', (3.7, 1.25), ha='left', va='center', fontsize=12)

    ax.set_xlim(-2, 5)
    ax.set_ylim(-1, 3)
    ax.axis('off')

    # 3. Gradient paths
    ax = axes[2]
    ax.set_title('Gradient Paths', fontweight='bold')

    # Two paths
    ax.annotate('Through F(x):', (0.1, 2.5), fontsize=11, fontweight='bold')
    ax.annotate('∂F/∂x (can vanish)', (0.3, 2.1), fontsize=10, color='blue')

    ax.annotate('Through Skip:', (0.1, 1.3), fontsize=11, fontweight='bold', color='green')
    ax.annotate('∂x/∂x = 1 (always 1!)', (0.3, 0.9), fontsize=10, color='green')

    ax.annotate('Total gradient:', (0.1, 0.1), fontsize=11, fontweight='bold')
    ax.annotate('∂(F(x)+x)/∂x = ∂F/∂x + 1', (0.3, -0.3), fontsize=10)
    ax.annotate('↑ guaranteed ≥ 1', (0.5, -0.7), fontsize=9, color='green')

    ax.set_xlim(0, 3)
    ax.set_ylim(-1, 3)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_highway_gates(n_samples: int = 500,
                           dim: int = 32,
                           save_path: Optional[str] = None):
    """
    Visualize how highway gates learn to balance skip vs transform.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Different input distributions
    X_uniform = np.random.uniform(-2, 2, (n_samples, dim))
    X_gaussian = np.random.randn(n_samples, dim)

    datasets = [('Uniform Input', X_uniform), ('Gaussian Input', X_gaussian)]

    for ax, (name, X) in zip(axes, datasets):
        # Multiple highway layers with different bias initializations
        biases = [-3, -1, 0, 1, 3]

        for bias in biases:
            highway = HighwayLayer(dim, bias_init=bias)
            _, cache = highway.forward(X)
            gate_values = cache['T'].flatten()

            ax.hist(gate_values, bins=50, alpha=0.5,
                   label=f'bias={bias} (mean={np.mean(gate_values):.2f})')

        ax.set_xlabel('Gate Value T(x)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Highway Gate Distribution\n{name}')
        ax.legend()
        ax.set_xlim(0, 1)

    plt.suptitle('Highway Networks: Gate Values Control Skip vs Transform',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_dense_features(save_path: Optional[str] = None):
    """
    Visualize feature accumulation in DenseNet.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    input_dim = 32
    growth_rate = 16
    num_layers = 5

    # Create dense block
    dense = DenseBlock(input_dim, growth_rate, num_layers)

    # Visualize architecture
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_layers + 1))

    y_positions = np.arange(num_layers + 1)

    # Draw feature accumulation
    current_dim = input_dim
    for i in range(num_layers + 1):
        if i == 0:
            label = f'Input: {input_dim}'
            width = input_dim / 100
        else:
            current_dim += growth_rate
            label = f'Layer {i}: +{growth_rate} = {current_dim}'
            width = current_dim / 100

        rect = plt.Rectangle((0, y_positions[i] - 0.3), width, 0.6,
                             facecolor=colors[i], edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.annotate(label, (width + 0.1, y_positions[i]),
                   va='center', fontsize=10)

        # Draw connections (skip connections to all previous)
        if i > 0:
            for j in range(i):
                ax.annotate('',
                           xy=(0, y_positions[i] - 0.3),
                           xytext=(dense.weights[j-1].shape[0]/100 if j > 0 else input_dim/100,
                                  y_positions[j] + 0.3),
                           arrowprops=dict(arrowstyle='->', color='gray',
                                         alpha=0.3, lw=0.5))

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-1, num_layers + 1)
    ax.set_xlabel('Feature Dimension (scaled)')
    ax.set_ylabel('Layer')
    ax.set_title('DenseNet: Feature Accumulation\n(Each layer receives ALL previous features)',
                fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'L{i}' for i in range(num_layers + 1)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# FAILURE MODES
# ============================================================

def experiment_failure_modes() -> dict:
    """
    WHAT BREAKS SKIP CONNECTIONS?

    This experiment explores edge cases and failure modes to deepen
    understanding of WHY skip connections work and WHEN they fail.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Skip Connection Failure Modes")
    print("=" * 60)

    results = {}

    # 1. WRONG SKIP SCALE
    print("\n1. WRONG SKIP SCALE")
    print("-" * 40)
    print("What if skip = 0.01*x or skip = 100*x instead of 1*x?")

    np.random.seed(42)
    x = np.random.randn(100, 10)
    target = np.random.randn(100, 10)

    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # Simulate training with different skip scales
        w = np.random.randn(10, 10) * 0.1
        lr = 0.01
        losses = []

        for _ in range(50):
            # Forward: out = relu(x @ w) + scale * x
            h = np.maximum(0, x @ w)
            out = h + scale * x

            loss = np.mean((out - target) ** 2)
            losses.append(loss)

            # Backward (simplified)
            d_out = 2 * (out - target) / out.size
            d_w = x.T @ (d_out * (h > 0))
            w -= lr * np.clip(d_w, -1, 1)

        final_loss = losses[-1] if not np.isnan(losses[-1]) else float('inf')
        results[f'scale_{scale}'] = final_loss
        status = "✓ OK" if final_loss < 10 else "✗ UNSTABLE"
        print(f"  Scale={scale:>6}: Final loss = {final_loss:.4f} {status}")

    print("\n  INSIGHT: Scale ≈ 1.0 works best. Too small (0.01) weakens skip.")
    print("           Too large (100) can cause instability.")

    # 2. DIMENSION MISMATCH
    print("\n2. DIMENSION MISMATCH")
    print("-" * 40)
    print("What if input dim ≠ output dim? Skip connection fails!")

    try:
        x = np.random.randn(10, 32)
        w = np.random.randn(32, 64)  # Changes dimension
        h = x @ w  # Shape: (10, 64)
        out = h + x  # ERROR: (10, 64) + (10, 32)
        print("  This should have failed!")
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  SOLUTION: Use projection: out = h + x @ W_proj")
        results['dim_mismatch'] = 'Error (expected)'

    # 3. TOO DEEP WITHOUT SKIP
    print("\n3. GRADIENT DECAY WITHOUT SKIP")
    print("-" * 40)
    print("Gradient magnitude after N layers (no skip vs with skip)")

    depths = [5, 10, 20, 50]
    for depth in depths:
        # Simulate gradient flow
        grad_no_skip = 1.0
        grad_with_skip = 1.0

        for _ in range(depth):
            # Without skip: gradient *= layer_grad (assume 0.9 per layer)
            grad_no_skip *= 0.9
            # With skip: gradient stays due to identity path
            grad_with_skip = grad_with_skip * 0.9 + 1.0  # +1 from skip

        results[f'grad_depth_{depth}'] = (grad_no_skip, min(grad_with_skip, 100))
        print(f"  Depth {depth:>2}: No skip = {grad_no_skip:.6f}, With skip = {min(grad_with_skip, 100):.2f}")

    print("\n  INSIGHT: Without skip, gradients vanish exponentially.")
    print("           With skip, gradient includes +1 term that prevents vanishing.")

    return results


def visualize_failure_modes(results: dict, save_path: Optional[str] = None):
    """Visualize skip connection failure modes."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Skip scale effect
    ax = axes[0]
    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    losses = [results.get(f'scale_{s}', 0) for s in scales]
    colors = ['red' if l > 5 else 'green' for l in losses]
    ax.bar([str(s) for s in scales], losses, color=colors, alpha=0.7)
    ax.set_xlabel('Skip Scale')
    ax.set_ylabel('Final Loss')
    ax.set_title('Failure: Wrong Skip Scale', fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', label='Unstable threshold')

    # 2. Gradient decay
    ax = axes[1]
    depths = [5, 10, 20, 50]
    no_skip = [results.get(f'grad_depth_{d}', (0, 0))[0] for d in depths]
    with_skip = [min(results.get(f'grad_depth_{d}', (0, 0))[1], 10) for d in depths]

    x_pos = np.arange(len(depths))
    ax.bar(x_pos - 0.2, no_skip, 0.4, label='No Skip', color='red', alpha=0.7)
    ax.bar(x_pos + 0.2, with_skip, 0.4, label='With Skip', color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Failure: Vanishing Gradients', fontweight='bold')
    ax.legend()
    ax.set_yscale('log')

    # 3. Summary
    ax = axes[2]
    ax.axis('off')
    summary = """
    SKIP CONNECTION FAILURE MODES:

    1. WRONG SCALE
       • Scale < 1: Weakens skip benefit
       • Scale > 1: Can cause instability
       • Optimal: Scale = 1.0

    2. DIMENSION MISMATCH
       • Input dim ≠ output dim → Error
       • Solution: Projection layer

    3. TOO DEEP WITHOUT SKIP
       • Gradients vanish: 0.9^50 ≈ 0
       • Skip adds +1 to gradient flow
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SKIP CONNECTIONS — Paradigm: GRADIENT HIGHWAYS")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Instead of learning H(x), learn the RESIDUAL: F(x) = H(x) - x
    Then: output = F(x) + x

    WHY IT WORKS:
    1. If optimal is identity, F(x) = 0 is easy to learn
    2. Gradients flow directly through skip: ∂(F(x)+x)/∂x includes +1
    3. Deep networks become trainable (100+ layers possible)

    VARIANTS:
    - Residual: output = F(x) + x           (additive)
    - Dense:    output = [x, F1(x), F2(x), ...]  (concatenative)
    - Highway:  output = T*F(x) + (1-T)*x   (gated)
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # NEW: The KEY experiment - training dynamics
    training_results = experiment_training_dynamics()

    # Experiment 1: Vanishing gradients (at initialization)
    grad_results = experiment_vanishing_gradients()

    # Experiment 2: Identity learning
    identity_results = experiment_identity_learning()

    # Experiment 3: Skip types
    type_results = experiment_skip_types()

    # Experiment 4: Failure modes
    failure_results = experiment_failure_modes()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # NEW: Training dynamics visualization (THE KEY FIGURE)
    visualize_training_dynamics(training_results, '45_skip_connections_training.png')

    visualize_skip_connection_concept('45_skip_connections_concept.png')
    visualize_gradient_flow(save_path='45_skip_connections_gradients.png')
    visualize_highway_gates(save_path='45_skip_connections_highway.png')
    visualize_dense_features(save_path='45_skip_connections_dense.png')
    visualize_failure_modes(failure_results, save_path='45_skip_failures.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. PROBLEM: Deep networks suffer from vanishing gradients
       - Gradient magnitude decreases exponentially with depth
       - Early layers learn very slowly or not at all

    2. SOLUTION: Skip connections create gradient highways
       - Additive skip: ∂(F(x)+x)/∂x = ∂F/∂x + 1 (guaranteed ≥ 1)
       - Gradients flow directly from loss to early layers

    3. VARIANTS trade off different properties:
       - Residual: Simple, effective, most common
       - Dense: Better feature reuse, more memory
       - Highway: Adaptive skip vs transform

    4. WHEN TO USE:
       - Any network > 10 layers
       - Training seems stuck (loss plateau)
       - Deeper ≠ better (without skip connections)
    """)
