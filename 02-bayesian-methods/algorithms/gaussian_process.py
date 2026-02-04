"""
GAUSSIAN PROCESSES — Paradigm: DISTRIBUTION OVER FUNCTIONS

===============================================================
WHAT IT IS (THE MIND-BENDING IDEA)
===============================================================

Instead of finding ONE function f(x), maintain a DISTRIBUTION over
all possible functions, then condition on observed data.

Before seeing data: prior distribution p(f)
After seeing data: posterior distribution p(f | data)

At each new point x*, you get:
    - Mean prediction μ(x*)
    - Uncertainty σ(x*)

This is FULL BAYESIAN INFERENCE over functions.

===============================================================
THE ONE EQUATION (Posterior Predictive)
===============================================================

Given training data (X, y), predict at new point x*:

    μ(x*) = K(x*, X) [K(X, X) + σ²I]⁻¹ y
    σ²(x*) = K(x*, x*) - K(x*, X) [K(X, X) + σ²I]⁻¹ K(X, x*)

where K is the kernel (covariance) function.

INTUITION:
    - K(x*, X): how similar is x* to training points?
    - [K(X,X) + σ²I]⁻¹ y: optimal weights for combining training outputs
    - σ²(x*): high when x* is far from training data (high uncertainty)

===============================================================
THE KERNEL IS THE INDUCTIVE BIAS
===============================================================

The kernel K(x, x') encodes your beliefs about the function:

RBF (Radial Basis Function / Squared Exponential):
    K(x, x') = exp(-||x - x'||² / 2ℓ²)
    - Infinitely differentiable (very smooth functions)
    - ℓ (lengthscale) controls how quickly correlation decays

LINEAR:
    K(x, x') = x · x'
    - Only linear functions
    - Equivalent to Bayesian linear regression

MATERN:
    K(x, x') = ... (involves Bessel functions)
    - Parameter ν controls roughness
    - ν=0.5: Ornstein-Uhlenbeck (continuous but not differentiable)
    - ν=∞: RBF

PERIODIC:
    K(x, x') = exp(-2 sin²(π|x-x'|/p) / ℓ²)
    - For periodic functions with period p

You can ADD and MULTIPLY kernels to create new ones!
    K₁ + K₂: either pattern
    K₁ × K₂: both patterns must be present

===============================================================
INDUCTIVE BIAS
===============================================================

1. Function smoothness is controlled by kernel choice
2. Uncertainty increases away from training data
3. Noise level σ² affects how much we trust the data

WHAT IT CAN DO:
    ✓ Uncertainty quantification (crucial for decision-making)
    ✓ Nonlinear patterns with appropriate kernel
    ✓ Exact interpolation (with σ²=0)
    ✓ Bayesian model selection (optimize kernel hyperparameters)

WHAT IT CAN'T DO:
    ✗ Scale beyond ~10k points (O(n³) complexity)
    ✗ Handle very high-dimensional input (kernel design hard)
    ✗ Discrete outputs directly (need approximations for classification)

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
# KERNEL FUNCTIONS
# ============================================================

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    RBF (Radial Basis Function) Kernel — the most common choice.

    K(x, x') = σ² exp(-||x - x'||² / 2ℓ²)

    INTUITION:
        - Points close together (||x - x'|| small) have K ≈ σ²
        - Points far apart have K ≈ 0
        - lengthscale ℓ controls how quickly correlation decays
            - Small ℓ: wiggly functions (fast decay)
            - Large ℓ: smooth functions (slow decay)
        - variance σ² controls vertical scale of functions

    Also called Squared Exponential or Gaussian kernel.
    """
    # Compute squared Euclidean distances
    # ||x - x'||² = ||x||² + ||x'||² - 2x·x'
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1)                  # (n2,)
    cross = X1 @ X2.T                               # (n1, n2)
    sq_dist = sq1 + sq2 - 2 * cross                # (n1, n2)
    sq_dist = np.maximum(sq_dist, 0)  # Numerical stability

    return variance * np.exp(-sq_dist / (2 * lengthscale ** 2))


def linear_kernel(X1, X2, variance=1.0, offset=0.0):
    """
    Linear Kernel — for linear functions.

    K(x, x') = σ² (x · x' + c)

    INTUITION:
        GP with linear kernel is equivalent to Bayesian linear regression.
        No nonlinearity, but still get uncertainty.
    """
    return variance * (X1 @ X2.T + offset)


def matern_kernel(X1, X2, lengthscale=1.0, variance=1.0, nu=1.5):
    """
    Matérn Kernel — controls roughness precisely.

    ν parameter:
        - ν = 0.5: Exponential kernel (continuous but not differentiable)
        - ν = 1.5: Once differentiable
        - ν = 2.5: Twice differentiable
        - ν → ∞: RBF kernel (infinitely differentiable)

    More realistic for many real-world processes than RBF.
    """
    from scipy.spatial.distance import cdist
    from scipy.special import kv, gamma

    dist = cdist(X1, X2, metric='euclidean')
    dist = np.maximum(dist, 1e-10)  # Avoid division by zero

    scaled_dist = np.sqrt(2 * nu) * dist / lengthscale

    if nu == 0.5:
        # Exponential kernel
        K = variance * np.exp(-dist / lengthscale)
    elif nu == 1.5:
        K = variance * (1 + scaled_dist) * np.exp(-scaled_dist)
    elif nu == 2.5:
        K = variance * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
    else:
        # General case
        K = variance * (2**(1-nu) / gamma(nu)) * (scaled_dist**nu) * kv(nu, scaled_dist)
        K = np.where(dist < 1e-10, variance, K)

    return K


# ============================================================
# GAUSSIAN PROCESS REGRESSION
# ============================================================

class GaussianProcessRegressor:
    """
    Gaussian Process for Regression.

    Given training data (X, y), predicts at new points with uncertainty.
    """

    def __init__(self, kernel='rbf', lengthscale=1.0, variance=1.0, noise=1e-6):
        """
        Parameters:
        -----------
        kernel : 'rbf', 'linear', or 'matern'
        lengthscale : for RBF/Matern, controls smoothness
        variance : signal variance (vertical scale)
        noise : observation noise σ² (jitter for numerical stability)
        """
        self.kernel_name = kernel
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise

        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Precomputed inverse for prediction

    def _kernel(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel_name == 'rbf':
            return rbf_kernel(X1, X2, self.lengthscale, self.variance)
        elif self.kernel_name == 'linear':
            return linear_kernel(X1, X2, self.variance)
        elif self.kernel_name == 'matern':
            return matern_kernel(X1, X2, self.lengthscale, self.variance)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def fit(self, X, y):
        """
        "Fit" = store data and precompute kernel inverse.

        The key computation: [K(X, X) + σ²I]⁻¹

        This is O(n³) — the main bottleneck of GPs.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Compute kernel matrix
        K = self._kernel(X, X)

        # Add noise to diagonal (regularization + observation noise)
        K += self.noise * np.eye(len(X))

        # Compute inverse (or use Cholesky for stability)
        # Using Cholesky: K = LL', then solve via two triangular systems
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T,
                                         np.linalg.solve(self.L, y))
        except np.linalg.LinAlgError:
            # Fallback to direct inverse if Cholesky fails
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y
            self.L = None

        return self

    def predict(self, X, return_std=False):
        """
        Predict mean and optionally standard deviation.

        μ(x*) = K(x*, X) α    where α = [K + σ²I]⁻¹ y
        σ²(x*) = K(x*, x*) - K(x*, X) [K + σ²I]⁻¹ K(X, x*)
        """
        # Kernel between test and train points
        K_star = self._kernel(X, self.X_train)  # (n_test, n_train)

        # Mean prediction
        mean = K_star @ self.alpha

        if return_std:
            # Variance prediction
            K_ss = self._kernel(X, X)  # (n_test, n_test)

            if self.L is not None:
                v = np.linalg.solve(self.L, K_star.T)
                var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            else:
                var = np.diag(K_ss) - np.diag(K_star @ self.K_inv @ K_star.T)

            # Numerical stability
            var = np.maximum(var, 0)
            std = np.sqrt(var)

            return mean, std

        return mean


class GaussianProcessClassifier:
    """
    Gaussian Process for Classification.

    Uses Laplace approximation: approximate posterior with Gaussian
    around the MAP estimate.

    Simpler approach: use GP regression on {-1, +1} labels,
    then threshold. Works surprisingly well in practice.
    """

    def __init__(self, kernel='rbf', lengthscale=1.0, variance=1.0, noise=0.1):
        self.gp = GaussianProcessRegressor(kernel, lengthscale, variance, noise)

    def fit(self, X, y):
        # Convert to {-1, +1}
        y_transformed = 2 * y - 1
        self.gp.fit(X, y_transformed)
        return self

    def predict(self, X):
        mean = self.gp.predict(X)
        return (mean > 0).astype(int)

    def predict_proba(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        # Probit approximation: integrate Gaussian through sigmoid
        # Simplified: just use the mean through sigmoid
        prob = 1 / (1 + np.exp(-mean / np.sqrt(1 + std**2)))
        return np.column_stack([1 - prob, prob])


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

    # Generate 1D regression data for visualization
    X_train = np.random.uniform(-3, 3, 20).reshape(-1, 1)
    y_train = np.sin(X_train.squeeze()) + np.random.randn(20) * 0.1
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    y_true = np.sin(X_test.squeeze())

    # -------- Experiment 1: Lengthscale Effect --------
    print("\n1. LENGTHSCALE EFFECT (RBF Kernel)")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    lengthscales = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    for idx, ls in enumerate(lengthscales):
        ax = axes[idx // 3, idx % 3]

        gp = GaussianProcessRegressor(kernel='rbf', lengthscale=ls, noise=0.01)
        gp.fit(X_train, y_train)
        mean, std = gp.predict(X_test, return_std=True)

        # Plot
        ax.fill_between(X_test.squeeze(), mean - 2*std, mean + 2*std,
                       alpha=0.3, label='±2σ')
        ax.plot(X_test, mean, 'b-', label='Mean')
        ax.plot(X_test, y_true, 'g--', alpha=0.5, label='True')
        ax.scatter(X_train, y_train, c='r', s=30, zorder=5, label='Data')
        ax.set_title(f'lengthscale = {ls}')
        ax.set_ylim(-2, 2)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('GP: Effect of Lengthscale\n'
                 'Small ℓ = wiggly, Large ℓ = smooth', fontsize=12)
    plt.tight_layout()
    fig.savefig('/Users/sid47/ML Algorithms/05_gp_lengthscale.png', dpi=100)
    plt.close(fig)
    print("Saved lengthscale experiment to 05_gp_lengthscale.png")

    for ls in [0.1, 0.5, 1.0, 2.0]:
        gp = GaussianProcessRegressor(kernel='rbf', lengthscale=ls, noise=0.01)
        gp.fit(X_train, y_train)
        mean = gp.predict(X_test)
        mse = np.mean((mean - y_true) ** 2)
        print(f"lengthscale={ls:<4}  MSE={mse:.4f}")
    print("→ Small ℓ overfits (wiggly), large ℓ underfits (oversmooth)")

    # -------- Experiment 2: Kernel Comparison --------
    print("\n2. KERNEL COMPARISON")
    print("-" * 40)

    kernels = ['rbf', 'linear', 'matern']
    for kernel in kernels:
        gp = GaussianProcessRegressor(kernel=kernel, lengthscale=1.0, noise=0.01)
        gp.fit(X_train, y_train)
        mean = gp.predict(X_test)
        mse = np.mean((mean - y_true) ** 2)
        print(f"kernel={kernel:<8}  MSE={mse:.4f}")
    print("→ RBF and Matern capture the sine wave, Linear cannot")

    # -------- Experiment 3: Noise Level --------
    print("\n3. NOISE LEVEL EFFECT")
    print("-" * 40)
    for noise in [1e-6, 0.01, 0.1, 1.0]:
        gp = GaussianProcessRegressor(kernel='rbf', lengthscale=0.5, noise=noise)
        gp.fit(X_train, y_train)
        mean = gp.predict(X_test)
        mse = np.mean((mean - y_true) ** 2)
        print(f"noise={noise:<8}  MSE={mse:.4f}")
    print("→ More noise = less data trust = smoother fit")

    # -------- Experiment 4: Uncertainty Away from Data --------
    print("\n4. UNCERTAINTY QUANTIFICATION")
    print("-" * 40)

    gp = GaussianProcessRegressor(kernel='rbf', lengthscale=1.0, noise=0.01)
    gp.fit(X_train, y_train)

    # Check uncertainty at different distances from training data
    test_points = np.array([[-3], [0], [4]])
    means, stds = gp.predict(test_points, return_std=True)

    print("Distance from data vs Uncertainty:")
    for i, x in enumerate(test_points.squeeze()):
        min_dist = np.min(np.abs(X_train.squeeze() - x))
        print(f"  x={x:>4}, min_dist_to_train={min_dist:.2f}, uncertainty={stds[i]:.4f}")
    print("→ Uncertainty increases away from training data!")


def benchmark_classification():
    """Benchmark GP classifier on challenge datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: GP Classifier on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    # Note: GP is slow, use smaller training set for speed
    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential', 'high_dim']:  # Skip slow/high-dim
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        # Use subset for speed
        n_train = min(200, len(X_tr))
        X_tr_sub = X_tr[:n_train]
        y_tr_sub = y_tr[:n_train]

        gp = GaussianProcessClassifier(kernel='rbf', lengthscale=1.0, noise=0.1)
        gp.fit(X_tr_sub, y_tr_sub)
        y_pred = gp.predict(X_te)
        acc = accuracy(y_te, y_pred)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    print("\n" + "="*60)
    print("VISUALIZING DECISION BOUNDARIES")
    print("="*60)

    datasets = get_2d_datasets()
    plot_datasets = {k: v for k, v in datasets.items()
                     if k not in ['clustered', 'dist_shift']}

    n = len(plot_datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        # Use subset for speed
        n_train = min(100, len(X_tr))
        gp = GaussianProcessClassifier(kernel='rbf', lengthscale=1.0, noise=0.1)
        gp.fit(X_tr[:n_train], y_tr[:n_train])
        acc = accuracy(y_te, gp.predict(X_te))

        plot_decision_boundary(gp.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('GAUSSIAN PROCESS: Decision Boundaries\n'
                 '(RBF kernel enables smooth nonlinear boundaries)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_uncertainty():
    """Visualize GP uncertainty on a 2D classification problem."""
    print("\n" + "="*60)
    print("VISUALIZING UNCERTAINTY")
    print("="*60)

    datasets = get_2d_datasets()
    X_train, _, y_train, _ = datasets['moons']

    # Use subset
    n = 50
    gp = GaussianProcessClassifier(kernel='rbf', lengthscale=0.5, noise=0.1)
    gp.fit(X_train[:n], y_train[:n])

    # Create grid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions and uncertainty
    probs = gp.predict_proba(X_grid)
    prob_class1 = probs[:, 1].reshape(xx.shape)

    # Entropy as uncertainty measure
    eps = 1e-10
    entropy = -prob_class1 * np.log(prob_class1 + eps) - \
              (1 - prob_class1) * np.log(1 - prob_class1 + eps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot probability
    im1 = axes[0].contourf(xx, yy, prob_class1, levels=20, cmap='RdYlBu')
    axes[0].scatter(X_train[:n, 0], X_train[:n, 1], c=y_train[:n],
                   cmap='RdYlBu', edgecolors='k', s=50)
    axes[0].set_title('P(class=1)')
    plt.colorbar(im1, ax=axes[0])

    # Plot uncertainty
    im2 = axes[1].contourf(xx, yy, entropy, levels=20, cmap='viridis')
    axes[1].scatter(X_train[:n, 0], X_train[:n, 1], c='red',
                   edgecolors='k', s=50)
    axes[1].set_title('Uncertainty (Entropy)')
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle('GP Classification: Probability and Uncertainty\n'
                 'High uncertainty (yellow) in ambiguous/sparse regions')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GAUSSIAN PROCESSES — Distribution Over Functions")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    A distribution over FUNCTIONS, not parameters.
    "What's the probability this function generated my data?"

THE KEY INSIGHT:
    You get UNCERTAINTY for free. At each point, you know
    how confident the model is.

THE KERNEL IS EVERYTHING:
    - RBF: smooth functions
    - Linear: linear functions
    - Matérn: controlled roughness
    - Lengthscale: how quickly correlation decays

STRENGTHS:
    ✓ Built-in uncertainty quantification
    ✓ Nonparametric (can fit any smooth function)
    ✓ Kernel encodes prior beliefs explicitly

WEAKNESSES:
    ✗ O(n³) training complexity — doesn't scale
    ✗ Requires kernel engineering
    ✗ Classification requires approximations
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_classification()

    # Visualize decision boundaries
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/05_gp_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path1}")
    plt.close(fig1)

    # Visualize uncertainty
    fig2 = visualize_uncertainty()
    save_path2 = '/Users/sid47/ML Algorithms/05_gp_uncertainty.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved uncertainty plot to: {save_path2}")
    plt.close(fig2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Gaussian Processes Reveal")
    print("="*60)
    print("""
1. You can have uncertainty over FUNCTIONS, not just parameters
2. The kernel encodes ALL your prior beliefs about the function
3. Lengthscale controls smoothness (small = wiggly, large = smooth)
4. Uncertainty increases away from training data
5. O(n³) makes GPs impractical for large datasets

KEY INSIGHT:
    GPs are the "Bayesian" way to do nonparametric regression.
    You get not just predictions, but confidence bounds.

WHEN TO USE:
    - Small datasets (< 10k points)
    - Uncertainty matters (Bayesian optimization, active learning)
    - Prior knowledge can be encoded in kernel

NEXT: Bayesian Linear Regression — uncertainty for linear models
    """)
