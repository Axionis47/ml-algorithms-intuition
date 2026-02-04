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

    # Experiment 1: Vanishing gradients
    grad_results = experiment_vanishing_gradients()

    # Experiment 2: Identity learning
    identity_results = experiment_identity_learning()

    # Experiment 3: Skip types
    type_results = experiment_skip_types()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_skip_connection_concept('45_skip_connections_concept.png')
    visualize_gradient_flow(save_path='45_skip_connections_gradients.png')
    visualize_highway_gates(save_path='45_skip_connections_highway.png')
    visualize_dense_features(save_path='45_skip_connections_dense.png')

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
