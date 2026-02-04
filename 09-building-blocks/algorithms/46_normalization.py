"""
===============================================================
NORMALIZATION — Paradigm: DISTRIBUTION CONTROL
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Normalization keeps activations in a "healthy" range during training.

The general formula:
    y = γ × (x - μ) / √(σ² + ε) + β

Where:
    μ, σ² = mean and variance (computed over some dimension)
    γ, β  = learned scale and shift (restore representational power)
    ε     = small constant for numerical stability

"Center, scale, then let the network learn the optimal distribution."

===============================================================
THE PROBLEM IT SOLVES: INTERNAL COVARIATE SHIFT
===============================================================

As training progresses:
    Layer 1 weights change → Layer 2 inputs change → Layer 2 must re-adapt

Each layer sees a SHIFTING input distribution.
This slows training and requires careful initialization/learning rates.

Normalization STABILIZES the input distribution to each layer.

===============================================================
KEY VARIANTS (differ in WHAT dimensions are normalized)
===============================================================

Given input X with shape (Batch, Channels, Height, Width):

1. BATCH NORM:     normalize over (Batch, H, W)     — per channel
2. LAYER NORM:     normalize over (Channels, H, W)  — per sample
3. INSTANCE NORM:  normalize over (H, W)            — per sample, per channel
4. GROUP NORM:     normalize over (Group, H, W)     — per sample, per group

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes normalized distributions are easier to learn from
2. Batch Norm: Assumes batch statistics approximate population
3. Layer Norm: Assumes features should have similar magnitudes
4. Instance Norm: Style should be normalized (good for style transfer)

===============================================================
WHEN TO USE WHICH
===============================================================

- Batch Norm:    CNNs with large batches (batch ≥ 32)
- Layer Norm:    Transformers, RNNs, small batches
- Instance Norm: Style transfer, generative models
- Group Norm:    CNNs with small batches (batch < 16)

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


# ============================================================
# BATCH NORMALIZATION
# ============================================================

class BatchNorm1D:
    """
    Batch Normalization for fully-connected layers.

    Normalizes over the BATCH dimension.

    Input shape: (batch_size, features)
    Statistics computed over: batch dimension (axis=0)

    THE KEY INSIGHT:
    Each feature is normalized to μ=0, σ=1 across the batch.
    Then γ and β let the network learn the optimal distribution.

    TRAINING vs INFERENCE:
    - Training: Use batch statistics
    - Inference: Use running (population) statistics
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        """
        Args:
            num_features: Number of features (C in [B, C])
            momentum: For running statistics update
            eps: Numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones((1, num_features))   # Scale
        self.beta = np.zeros((1, num_features))   # Shift

        # Running statistics (for inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        self.training = True
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Training: Use batch statistics, update running stats
        Inference: Use running statistics
        """
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

            # Cache for backward
            self.cache = (x, x_norm, batch_mean, batch_var)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through batch normalization.

        The gradient is more complex than you'd expect because
        μ and σ are functions of the entire batch!
        """
        x, x_norm, mean, var = self.cache
        N = x.shape[0]
        std = np.sqrt(var + self.eps)

        # Gradients of learnable parameters
        self.dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradient of normalized x
        dx_norm = dout * self.gamma

        # Gradient of variance
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=0, keepdims=True)

        # Gradient of mean
        dmean = np.sum(dx_norm * -1 / std, axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        # Gradient of input
        dx = dx_norm / std + dvar * 2 * (x - mean) / N + dmean / N

        return dx


class BatchNorm2D:
    """
    Batch Normalization for convolutional layers.

    Input shape: (batch_size, channels, height, width)
    Statistics computed over: (batch, height, width) — i.e., per channel

    Each channel is normalized independently.
    """

    def __init__(self, num_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Shape: (1, C, 1, 1) for broadcasting
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))

        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))

        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 4D input."""
        if self.training:
            # Statistics over (N, H, W), keeping C
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            self.cache = (x, x_norm, batch_mean, batch_var)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


# ============================================================
# LAYER NORMALIZATION
# ============================================================

class LayerNorm:
    """
    Layer Normalization — normalizes over FEATURES (not batch).

    Input shape: (batch_size, features) or (batch, seq, features)
    Statistics computed over: last dimension (features)

    KEY DIFFERENCE FROM BATCH NORM:
    - Each sample is normalized INDEPENDENTLY
    - No dependence on batch size
    - Same behavior in training and inference

    WHY USE IT?
    - Works with any batch size (even batch=1)
    - Essential for Transformers and RNNs
    - No running statistics needed
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Args:
            normalized_shape: Size of last dimension(s) to normalize over
            eps: Numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Normalize each sample independently over feature dimension.
        """
        # Statistics over last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta

        self.cache = (x, x_norm, mean, var)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through layer normalization."""
        x, x_norm, mean, var = self.cache
        D = x.shape[-1]  # Feature dimension
        std = np.sqrt(var + self.eps)

        self.dgamma = np.sum(dout * x_norm, axis=tuple(range(len(x.shape)-1)))
        self.dbeta = np.sum(dout, axis=tuple(range(len(x.shape)-1)))

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / std, axis=-1, keepdims=True)

        dx = dx_norm / std + dvar * 2 * (x - mean) / D + dmean / D
        return dx


# ============================================================
# INSTANCE NORMALIZATION
# ============================================================

class InstanceNorm2D:
    """
    Instance Normalization — normalizes over SPATIAL dimensions only.

    Input shape: (batch_size, channels, height, width)
    Statistics computed over: (height, width) — per sample, per channel

    KEY INSIGHT FOR STYLE TRANSFER:
    Mean and variance of feature maps encode "style".
    Normalizing them removes style, allowing style injection.

    "Normalize away the style of each instance."
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_channels: Number of channels
            eps: Numerical stability
            affine: Whether to learn γ and β
        """
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = np.ones((1, num_channels, 1, 1))
            self.beta = np.zeros((1, num_channels, 1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Each (sample, channel) is normalized independently.
        """
        # Statistics over spatial dimensions only
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            return self.gamma * x_norm + self.beta
        return x_norm


# ============================================================
# GROUP NORMALIZATION
# ============================================================

class GroupNorm:
    """
    Group Normalization — normalizes over GROUPS of channels.

    Input shape: (batch_size, channels, height, width)
    Statistics computed over: (group_channels, height, width)

    INTUITION:
    - Channels within a group are normalized together
    - Like Layer Norm for CNNs, but respects channel structure
    - Works well with small batches (unlike Batch Norm)

    RELATIONSHIP TO OTHERS:
    - num_groups = num_channels → Instance Norm
    - num_groups = 1 → Layer Norm
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        """
        Args:
            num_groups: Number of groups to divide channels into
            num_channels: Total number of channels (must be divisible by num_groups)
            eps: Numerical stability
        """
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.eps = eps

        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Reshape to expose groups, normalize, reshape back.
        """
        N, C, H, W = x.shape

        # Reshape: (N, C, H, W) → (N, G, C//G, H, W)
        x_grouped = x.reshape(N, self.num_groups, self.channels_per_group, H, W)

        # Statistics over (C//G, H, W) — i.e., within each group
        mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)

        x_norm = (x_grouped - mean) / np.sqrt(var + self.eps)

        # Reshape back
        x_norm = x_norm.reshape(N, C, H, W)

        return self.gamma * x_norm + self.beta


# ============================================================
# RMS NORMALIZATION (used in modern LLMs)
# ============================================================

class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Used in LLaMA, Gemma, and other modern LLMs.

    KEY DIFFERENCE FROM LAYER NORM:
    - No mean centering (only scale normalization)
    - Slightly faster and often works just as well

    Formula:
        y = x / RMS(x) * γ
        RMS(x) = √(mean(x²))
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using RMS normalization."""
        # RMS = sqrt(mean(x^2))
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.gamma


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_normalization_effect(n_layers: int = 10,
                                    hidden_dim: int = 64,
                                    n_samples: int = 100) -> dict:
    """
    Show how normalization affects activation distributions.

    WHAT TO OBSERVE:
    - Without normalization: activations explode or vanish
    - With normalization: activations stay in reasonable range
    """
    print("=" * 60)
    print("EXPERIMENT: Normalization Effect on Activations")
    print("=" * 60)

    results = {'layers': list(range(n_layers)),
               'no_norm': {'mean': [], 'std': []},
               'batch_norm': {'mean': [], 'std': []},
               'layer_norm': {'mean': [], 'std': []}}

    # Input
    X = np.random.randn(n_samples, hidden_dim)

    # Initialize weights (without careful init)
    weights = [np.random.randn(hidden_dim, hidden_dim) * 0.5 for _ in range(n_layers)]

    # Track activations for each normalization type

    # 1. No normalization
    h = X.copy()
    for i, W in enumerate(weights):
        h = np.maximum(0, h @ W)  # ReLU
        results['no_norm']['mean'].append(np.mean(np.abs(h)))
        results['no_norm']['std'].append(np.std(h))

    # 2. Batch normalization
    h = X.copy()
    bn_layers = [BatchNorm1D(hidden_dim) for _ in range(n_layers)]
    for i, (W, bn) in enumerate(zip(weights, bn_layers)):
        h = h @ W
        h = bn.forward(h)
        h = np.maximum(0, h)  # ReLU
        results['batch_norm']['mean'].append(np.mean(np.abs(h)))
        results['batch_norm']['std'].append(np.std(h))

    # 3. Layer normalization
    h = X.copy()
    ln_layers = [LayerNorm(hidden_dim) for _ in range(n_layers)]
    for i, (W, ln) in enumerate(zip(weights, ln_layers)):
        h = h @ W
        h = ln.forward(h)
        h = np.maximum(0, h)  # ReLU
        results['layer_norm']['mean'].append(np.mean(np.abs(h)))
        results['layer_norm']['std'].append(np.std(h))

    print("\nActivation statistics at each layer:")
    print("-" * 60)
    print(f"{'Layer':<8} {'No Norm (std)':<15} {'BatchNorm (std)':<15} {'LayerNorm (std)':<15}")
    print("-" * 60)
    for i in range(n_layers):
        print(f"{i:<8} {results['no_norm']['std'][i]:<15.4f} "
              f"{results['batch_norm']['std'][i]:<15.4f} "
              f"{results['layer_norm']['std'][i]:<15.4f}")

    return results


def experiment_batch_size_sensitivity(batch_sizes: List[int] = [2, 4, 8, 16, 32, 64, 128],
                                       hidden_dim: int = 64) -> dict:
    """
    Show how Batch Norm depends on batch size.

    WHAT TO OBSERVE:
    - Small batches: Batch statistics are noisy
    - Layer/Group Norm: Independent of batch size
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Batch Size Sensitivity")
    print("=" * 60)

    results = {'batch_sizes': batch_sizes,
               'bn_variance': [],
               'ln_variance': [],
               'gn_variance': []}

    # Fixed "true" statistics
    true_mean = 2.0
    true_std = 0.5

    for bs in batch_sizes:
        # Generate data with known statistics
        X = np.random.randn(bs, hidden_dim) * true_std + true_mean

        # Batch Norm: statistics depend on batch
        bn = BatchNorm1D(hidden_dim)
        bn_out = bn.forward(X)
        bn_batch_mean = np.mean(bn_out)
        bn_batch_std = np.std(bn_out)

        # Layer Norm: statistics per sample
        ln = LayerNorm(hidden_dim)
        ln_out = ln.forward(X)
        ln_batch_mean = np.mean(ln_out)
        ln_batch_std = np.std(ln_out)

        # For 2D simulation, reshape
        X_2d = X.reshape(bs, hidden_dim, 1, 1).repeat(4, axis=2).repeat(4, axis=3)
        gn = GroupNorm(num_groups=8, num_channels=hidden_dim)
        gn_out = gn.forward(X_2d)
        gn_batch_mean = np.mean(gn_out)
        gn_batch_std = np.std(gn_out)

        # Variance of statistics across multiple runs
        bn_vars = []
        ln_vars = []
        for _ in range(10):
            X_trial = np.random.randn(bs, hidden_dim) * true_std + true_mean
            bn_vars.append(np.mean(bn.forward(X_trial)))
            ln_vars.append(np.mean(ln.forward(X_trial)))

        results['bn_variance'].append(np.var(bn_vars))
        results['ln_variance'].append(np.var(ln_vars))

    print("\nVariance of output mean across trials (lower = more stable):")
    print("-" * 50)
    print(f"{'Batch Size':<12} {'BatchNorm':<15} {'LayerNorm':<15}")
    print("-" * 50)
    for i, bs in enumerate(batch_sizes):
        print(f"{bs:<12} {results['bn_variance'][i]:<15.6f} {results['ln_variance'][i]:<15.6f}")

    return results


def experiment_norm_locations(n_layers: int = 5,
                              hidden_dim: int = 32,
                              n_samples: int = 100) -> dict:
    """
    Compare Pre-Norm vs Post-Norm configurations.

    Post-Norm (original):  x → Layer → Norm → Activation
    Pre-Norm (modern):     x → Norm → Layer → Activation

    WHAT TO OBSERVE:
    - Pre-Norm often trains more stably
    - Post-Norm may have better final performance
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Pre-Norm vs Post-Norm")
    print("=" * 60)

    X = np.random.randn(n_samples, hidden_dim)
    weights = [np.random.randn(hidden_dim, hidden_dim) * 0.3 for _ in range(n_layers)]

    results = {'post_norm': {'activations': []},
               'pre_norm': {'activations': []}}

    # Post-Norm: x → W → LN → ReLU
    h = X.copy()
    for W in weights:
        h = h @ W
        ln = LayerNorm(hidden_dim)
        h = ln.forward(h)
        h = np.maximum(0, h)
        results['post_norm']['activations'].append(np.std(h))

    # Pre-Norm: x → LN → W → ReLU
    h = X.copy()
    for W in weights:
        ln = LayerNorm(hidden_dim)
        h = ln.forward(h)
        h = h @ W
        h = np.maximum(0, h)
        results['pre_norm']['activations'].append(np.std(h))

    print("\nActivation std at each layer:")
    print("-" * 40)
    print(f"{'Layer':<8} {'Post-Norm':<15} {'Pre-Norm':<15}")
    print("-" * 40)
    for i in range(n_layers):
        print(f"{i:<8} {results['post_norm']['activations'][i]:<15.4f} "
              f"{results['pre_norm']['activations'][i]:<15.4f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_normalization_types(save_path: Optional[str] = None):
    """
    Visual explanation of different normalization types.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Normalization Types: Which Dimensions Are Normalized?',
                fontsize=14, fontweight='bold')

    # Create sample tensor visualization
    # Shape: (Batch=4, Channels=6, Height, Width) simplified to (B, C, HW)

    def draw_norm_pattern(ax, title, norm_dims, description):
        """Draw which dimensions are normalized together."""
        B, C = 4, 6

        # Draw grid
        for b in range(B):
            for c in range(C):
                color = 'lightgray'
                ax.add_patch(plt.Rectangle((c, B-1-b), 0.9, 0.9,
                                          facecolor=color, edgecolor='black'))

        # Highlight normalized groups
        colors = plt.cm.Set3(np.linspace(0, 1, 12))

        if norm_dims == 'batch':  # Batch Norm: per channel
            for c in range(C):
                for b in range(B):
                    ax.add_patch(plt.Rectangle((c, B-1-b), 0.9, 0.9,
                                              facecolor=colors[c], edgecolor='black', alpha=0.7))
        elif norm_dims == 'layer':  # Layer Norm: per sample
            for b in range(B):
                for c in range(C):
                    ax.add_patch(plt.Rectangle((c, B-1-b), 0.9, 0.9,
                                              facecolor=colors[b], edgecolor='black', alpha=0.7))
        elif norm_dims == 'instance':  # Instance Norm: per sample, per channel
            idx = 0
            for b in range(B):
                for c in range(C):
                    ax.add_patch(plt.Rectangle((c, B-1-b), 0.9, 0.9,
                                              facecolor=colors[idx % 12], edgecolor='black', alpha=0.7))
                    idx += 1
        elif norm_dims == 'group':  # Group Norm: per sample, per group
            groups = 2
            for b in range(B):
                for c in range(C):
                    group_idx = c // (C // groups)
                    ax.add_patch(plt.Rectangle((c, B-1-b), 0.9, 0.9,
                                              facecolor=colors[b * groups + group_idx],
                                              edgecolor='black', alpha=0.7))

        ax.set_xlim(-0.5, C + 0.5)
        ax.set_ylim(-0.5, B + 0.5)
        ax.set_xlabel('Channels (C)')
        ax.set_ylabel('Batch (N)')
        ax.set_title(f'{title}\n{description}', fontweight='bold')
        ax.set_xticks(np.arange(C) + 0.45)
        ax.set_xticklabels([f'C{i}' for i in range(C)])
        ax.set_yticks(np.arange(B) + 0.45)
        ax.set_yticklabels([f'N{B-1-i}' for i in range(B)])

    draw_norm_pattern(axes[0, 0], 'Batch Normalization', 'batch',
                     'Same color = normalized together\n(per channel, across batch)')
    draw_norm_pattern(axes[0, 1], 'Layer Normalization', 'layer',
                     'Same color = normalized together\n(per sample, across channels)')
    draw_norm_pattern(axes[1, 0], 'Instance Normalization', 'instance',
                     'Same color = normalized together\n(per sample, per channel)')
    draw_norm_pattern(axes[1, 1], 'Group Normalization (G=2)', 'group',
                     'Same color = normalized together\n(per sample, per group)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_activation_distributions(save_path: Optional[str] = None):
    """
    Visualize how normalization affects activation distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    hidden_dim = 64
    n_samples = 1000

    # Simulate activations after several layers
    X = np.random.randn(n_samples, hidden_dim)
    W1 = np.random.randn(hidden_dim, hidden_dim) * 0.8
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.8
    W3 = np.random.randn(hidden_dim, hidden_dim) * 0.8

    # 1. No normalization
    ax = axes[0, 0]
    h = np.maximum(0, X @ W1)
    h = np.maximum(0, h @ W2)
    h = np.maximum(0, h @ W3)
    ax.hist(h.flatten(), bins=50, density=True, alpha=0.7, color='red')
    ax.set_title(f'No Normalization\nμ={np.mean(h):.2f}, σ={np.std(h):.2f}', fontweight='bold')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')

    # 2. Batch Normalization
    ax = axes[0, 1]
    bn1, bn2, bn3 = BatchNorm1D(hidden_dim), BatchNorm1D(hidden_dim), BatchNorm1D(hidden_dim)
    h = np.maximum(0, bn1.forward(X @ W1))
    h = np.maximum(0, bn2.forward(h @ W2))
    h = np.maximum(0, bn3.forward(h @ W3))
    ax.hist(h.flatten(), bins=50, density=True, alpha=0.7, color='blue')
    ax.set_title(f'Batch Normalization\nμ={np.mean(h):.2f}, σ={np.std(h):.2f}', fontweight='bold')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')

    # 3. Layer Normalization
    ax = axes[1, 0]
    ln1, ln2, ln3 = LayerNorm(hidden_dim), LayerNorm(hidden_dim), LayerNorm(hidden_dim)
    h = np.maximum(0, ln1.forward(X @ W1))
    h = np.maximum(0, ln2.forward(h @ W2))
    h = np.maximum(0, ln3.forward(h @ W3))
    ax.hist(h.flatten(), bins=50, density=True, alpha=0.7, color='green')
    ax.set_title(f'Layer Normalization\nμ={np.mean(h):.2f}, σ={np.std(h):.2f}', fontweight='bold')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')

    # 4. Comparison of means across layers
    ax = axes[1, 1]
    n_layers = 20
    no_norm_means = []
    bn_means = []
    ln_means = []

    h_none = X.copy()
    h_bn = X.copy()
    h_ln = X.copy()

    for i in range(n_layers):
        W = np.random.randn(hidden_dim, hidden_dim) * 0.5

        h_none = np.maximum(0, h_none @ W)
        no_norm_means.append(np.mean(np.abs(h_none)))

        bn = BatchNorm1D(hidden_dim)
        h_bn = np.maximum(0, bn.forward(h_bn @ W))
        bn_means.append(np.mean(np.abs(h_bn)))

        ln = LayerNorm(hidden_dim)
        h_ln = np.maximum(0, ln.forward(h_ln @ W))
        ln_means.append(np.mean(np.abs(h_ln)))

    ax.semilogy(range(n_layers), no_norm_means, 'r-', label='No Norm', linewidth=2)
    ax.semilogy(range(n_layers), bn_means, 'b-', label='Batch Norm', linewidth=2)
    ax.semilogy(range(n_layers), ln_means, 'g-', label='Layer Norm', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean |activation| (log scale)')
    ax.set_title('Activation Magnitude Across Layers', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Effect of Normalization on Activation Distributions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_batch_size_effect(save_path: Optional[str] = None):
    """
    Visualize how batch size affects Batch Norm statistics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hidden_dim = 64
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    n_trials = 50

    # True population parameters
    true_mean = 3.0
    true_std = 1.5

    bn_mean_vars = []
    bn_std_vars = []

    for bs in batch_sizes:
        means = []
        stds = []
        for _ in range(n_trials):
            X = np.random.randn(bs, hidden_dim) * true_std + true_mean
            batch_mean = np.mean(X)
            batch_std = np.std(X)
            means.append(batch_mean)
            stds.append(batch_std)

        bn_mean_vars.append(np.var(means))
        bn_std_vars.append(np.var(stds))

    # Plot variance of batch statistics
    ax = axes[0]
    ax.loglog(batch_sizes, bn_mean_vars, 'b-o', label='Variance of batch mean', linewidth=2)
    ax.loglog(batch_sizes, bn_std_vars, 'r-s', label='Variance of batch std', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Variance (log scale)')
    ax.set_title('Batch Statistics Variance vs Batch Size\n(Lower = more stable)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show actual distributions for small vs large batch
    ax = axes[1]
    small_batch = 4
    large_batch = 128

    means_small = [np.mean(np.random.randn(small_batch, hidden_dim) * true_std + true_mean)
                   for _ in range(200)]
    means_large = [np.mean(np.random.randn(large_batch, hidden_dim) * true_std + true_mean)
                   for _ in range(200)]

    ax.hist(means_small, bins=30, alpha=0.6, label=f'Batch size = {small_batch}', density=True)
    ax.hist(means_large, bins=30, alpha=0.6, label=f'Batch size = {large_batch}', density=True)
    ax.axvline(true_mean, color='black', linestyle='--', label=f'True mean = {true_mean}')
    ax.set_xlabel('Batch Mean')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Batch Means\n(Larger batch = tighter distribution)', fontweight='bold')
    ax.legend()

    plt.suptitle('Why Batch Normalization Needs Large Batches', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_norm_comparison_table(save_path: Optional[str] = None):
    """
    Create a comparison table of normalization methods.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    headers = ['Method', 'Normalize Over', 'Best For', 'Batch Dependent?', 'Running Stats?']
    data = [
        ['Batch Norm', '(N, H, W)', 'CNNs, large batches', 'Yes', 'Yes'],
        ['Layer Norm', '(C, H, W)', 'Transformers, RNNs', 'No', 'No'],
        ['Instance Norm', '(H, W)', 'Style transfer', 'No', 'No'],
        ['Group Norm', '(G, H, W)', 'CNNs, small batches', 'No', 'No'],
        ['RMS Norm', '(C)', 'Modern LLMs', 'No', 'No'],
    ]

    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.15, 0.25, 0.15, 0.15]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax.set_title('Normalization Methods Comparison\n', fontsize=14, fontweight='bold')

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
    print("NORMALIZATION — Paradigm: DISTRIBUTION CONTROL")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Keep activations in a "healthy" range by normalizing to μ=0, σ=1,
    then let γ and β learn the optimal distribution.

    THE FORMULA:
        y = γ × (x - μ) / √(σ² + ε) + β

    KEY VARIANTS:
    ┌─────────────────┬────────────────────┬─────────────────────┐
    │ Method          │ Normalize Over     │ Best For            │
    ├─────────────────┼────────────────────┼─────────────────────┤
    │ Batch Norm      │ Batch, H, W        │ CNNs (large batch)  │
    │ Layer Norm      │ Channels, H, W     │ Transformers, RNNs  │
    │ Instance Norm   │ H, W only          │ Style transfer      │
    │ Group Norm      │ Group, H, W        │ CNNs (small batch)  │
    │ RMS Norm        │ Features (no mean) │ Modern LLMs         │
    └─────────────────┴────────────────────┴─────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Normalization effect
    norm_results = experiment_normalization_effect()

    # Experiment 2: Batch size sensitivity
    batch_results = experiment_batch_size_sensitivity()

    # Experiment 3: Pre-norm vs post-norm
    location_results = experiment_norm_locations()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_normalization_types('46_normalization_types.png')
    visualize_activation_distributions('46_normalization_distributions.png')
    visualize_batch_size_effect('46_normalization_batch_size.png')
    visualize_norm_comparison_table('46_normalization_comparison.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. PROBLEM: Internal covariate shift
       - Each layer sees shifting input distribution
       - Requires careful initialization and small learning rates

    2. SOLUTION: Normalize activations
       - Center to μ=0, scale to σ=1
       - Learn optimal γ and β

    3. WHICH TO USE:
       - Batch Norm: CNNs with batch ≥ 32
       - Layer Norm: Transformers, RNNs, any batch size
       - Instance Norm: Style transfer (removes style info)
       - Group Norm: CNNs with batch < 16
       - RMS Norm: Modern LLMs (simpler, faster)

    4. TRAINING vs INFERENCE:
       - Batch Norm: Running stats at inference
       - Others: Same behavior (no running stats)
    """)
