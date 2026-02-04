"""
SUPPORT VECTOR MACHINE — Paradigm: MARGIN

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Find the hyperplane with the MAXIMUM MARGIN to the nearest points.

Given a separating hyperplane w'x + b = 0:
    - Margin = 2 / ||w||
    - Points with w'x + b = ±1 are ON the margin (support vectors)

OBJECTIVE:
    min ||w||²/2  (maximize margin)
    s.t. yᵢ(w'xᵢ + b) ≥ 1  (all points on correct side of margin)

WHY MARGIN?
    - Large margin = robust to perturbations
    - Small margin = precarious, easily flipped
    - Margin is related to generalization error (VC theory)

===============================================================
SUPPORT VECTORS — THE CRITICAL INSIGHT
===============================================================

ONLY points ON the margin boundary determine the solution!

These points with yᵢ(w'xᵢ + b) = 1 are called SUPPORT VECTORS.

If you remove any non-support vector, the solution doesn't change.
The model only "remembers" the critical boundary points.

This is fundamentally different from other models that use all data.

===============================================================
THE KERNEL TRICK — NONLINEAR MAGIC
===============================================================

What if data isn't linearly separable in input space?

SOLUTION: Map to higher-dimensional space where it IS separable.

The KERNEL TRICK: Compute dot products in feature space
WITHOUT explicitly computing the features!

K(x, x') = φ(x)' φ(x')

COMMON KERNELS:
    Linear:     K(x, x') = x' x'
    Polynomial: K(x, x') = (γ x'x' + r)^d
    RBF:        K(x, x') = exp(-γ ||x - x'||²)
    Sigmoid:    K(x, x') = tanh(γ x'x' + r)

RBF kernel implicitly maps to INFINITE dimensional space!

===============================================================
SOFT MARGIN (C parameter)
===============================================================

What if data isn't perfectly separable even in feature space?

Allow some points to violate the margin, but penalize them:

    min ||w||²/2 + C Σᵢ ξᵢ
    s.t. yᵢ(w'xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

C controls the tradeoff:
    - Large C: penalize violations heavily → try to classify all correctly
    - Small C: allow more violations → larger margin, more regularization

===============================================================
INDUCTIVE BIAS
===============================================================

1. Large margin → good generalization (geometric)
2. Kernel choice → what similarities matter (RBF = local, linear = global)
3. Only support vectors matter (sparse solution)

WHAT IT CAN DO:
    ✓ Nonlinear boundaries via kernels
    ✓ Works in high dimensions (kernel computes in input space)
    ✓ Sparse solution (efficient at test time)
    ✓ Strong theoretical guarantees (VC dimension)

WHAT IT CAN'T DO:
    ✗ Native probability estimates (need Platt scaling)
    ✗ Efficient training on huge datasets (O(n²) to O(n³))
    ✗ Automatic kernel selection

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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

def linear_kernel(X1, X2):
    """Linear kernel: K(x, x') = x' x'"""
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=1.0):
    """
    RBF (Gaussian) Kernel: K(x, x') = exp(-γ ||x - x'||²)

    This implicitly maps to INFINITE dimensional space!

    γ (gamma) controls the kernel width:
        - Large γ: narrow kernel, points must be very close to be similar
        - Small γ: wide kernel, distant points still have some similarity
    """
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1)
    sq_dist = sq1 + sq2 - 2 * (X1 @ X2.T)
    sq_dist = np.maximum(sq_dist, 0)
    return np.exp(-gamma * sq_dist)


def polynomial_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """Polynomial kernel: K(x, x') = (γ x'x' + r)^d"""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


# ============================================================
# SVM CLASSIFIER (Simplified SMO-like approach)
# ============================================================

class SVM:
    """
    Support Vector Machine Classifier.

    This implementation uses a simplified optimization approach.
    Real SVMs use SMO (Sequential Minimal Optimization) for efficiency.
    """

    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, degree=3, tol=1e-3, max_iter=1000):
        """
        Parameters:
        -----------
        C : Regularization parameter (larger = less regularization)
        kernel : 'linear', 'rbf', or 'poly'
        gamma : Kernel coefficient for RBF/poly
        degree : Degree for polynomial kernel
        tol : Tolerance for stopping criterion
        max_iter : Maximum iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter

        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None

    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        elif self.kernel == 'poly':
            return polynomial_kernel(X1, X2, self.degree, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """
        Train SVM using a simplified gradient-based approach.

        The DUAL problem:
            max Σᵢ αᵢ - 0.5 Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
            s.t. 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0

        Real SVMs use SMO for this. We use scipy.optimize for simplicity.
        """
        self.X_train = X.copy()
        # Convert labels to {-1, +1}
        self.y_train = (2 * y - 1).astype(float)
        n_samples = len(y)

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Solve dual problem using scipy
        # We minimize the negative of the dual objective
        def objective(alpha):
            # -Σαᵢ + 0.5 Σᵢⱼ αᵢαⱼyᵢyⱼKᵢⱼ
            return -np.sum(alpha) + 0.5 * np.sum(
                (alpha * self.y_train).reshape(-1, 1) *
                (alpha * self.y_train).reshape(1, -1) * K
            )

        def gradient(alpha):
            return -np.ones(n_samples) + (
                (alpha * self.y_train).reshape(-1, 1) * self.y_train * K
            ).sum(axis=1)

        # Constraints: 0 ≤ α ≤ C, Σαᵢyᵢ = 0
        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, self.y_train)}
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Initialize
        alpha0 = np.zeros(n_samples)

        # Optimize
        result = minimize(objective, alpha0, method='SLSQP', jac=gradient,
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': self.max_iter, 'ftol': self.tol})

        self.alpha = result.x

        # Find support vectors (α > threshold)
        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = self.y_train[sv_mask]
        self.support_vector_alphas = self.alpha[sv_mask]

        # Compute bias b using support vectors
        # For support vectors with 0 < α < C:
        # yᵢ(Σⱼ αⱼyⱼK(xⱼ,xᵢ) + b) = 1
        # → b = yᵢ - Σⱼ αⱼyⱼK(xⱼ,xᵢ)
        sv_on_margin = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        if np.sum(sv_on_margin) > 0:
            K_sv = self._compute_kernel(X[sv_on_margin], X)
            self.b = np.mean(
                self.y_train[sv_on_margin] -
                np.sum(self.alpha * self.y_train * K_sv, axis=1)
            )
        else:
            self.b = 0

        return self

    def decision_function(self, X):
        """
        Compute signed distance to hyperplane.

        f(x) = Σᵢ αᵢyᵢK(xᵢ, x) + b
        """
        K = self._compute_kernel(self.X_train, X)
        return np.sum(self.alpha[:, np.newaxis] * self.y_train[:, np.newaxis] * K, axis=0) + self.b

    def predict(self, X):
        """Predict class labels."""
        return (self.decision_function(X) >= 0).astype(int)


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

    # -------- Experiment 1: Linear vs Nonlinear Kernels --------
    print("\n1. KERNEL COMPARISON")
    print("-" * 40)

    for name in ['linear', 'circles', 'xor', 'moons']:
        X_train, X_test, y_train, y_test = datasets[name]

        results = {}
        for kernel in ['linear', 'rbf', 'poly']:
            svm = SVM(C=1.0, kernel=kernel, gamma=1.0)
            svm.fit(X_train, y_train)
            acc = accuracy(y_test, svm.predict(X_test))
            results[kernel] = acc

        print(f"{name:<10} linear={results['linear']:.2f}  rbf={results['rbf']:.2f}  poly={results['poly']:.2f}")

    print("→ Linear kernel fails on nonlinear data")
    print("→ RBF kernel handles most nonlinear patterns")

    # -------- Experiment 2: Effect of C (Regularization) --------
    print("\n2. EFFECT OF C (Soft Margin)")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['moons']

    print("C controls margin/violation tradeoff:")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svm = SVM(C=C, kernel='rbf', gamma=1.0)
        svm.fit(X_train, y_train)
        n_sv = len(svm.support_vectors)
        acc = accuracy(y_test, svm.predict(X_test))
        print(f"C={C:<6} n_support_vectors={n_sv:<4} accuracy={acc:.3f}")

    print("→ Small C: more support vectors, smoother boundary")
    print("→ Large C: fewer support vectors, more complex boundary")

    # -------- Experiment 3: Effect of Gamma (RBF) --------
    print("\n3. EFFECT OF GAMMA (RBF Kernel Width)")
    print("-" * 40)

    print("Gamma controls kernel width:")
    for gamma in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svm = SVM(C=1.0, kernel='rbf', gamma=gamma)
        svm.fit(X_train, y_train)
        acc = accuracy(y_test, svm.predict(X_test))
        print(f"gamma={gamma:<6} accuracy={acc:.3f}")

    print("→ Small γ: smooth, underfit")
    print("→ Large γ: wiggly, overfit (each point becomes its own island)")

    # -------- Experiment 4: Support Vectors --------
    print("\n4. SUPPORT VECTORS — THE CRITICAL POINTS")
    print("-" * 40)

    svm = SVM(C=1.0, kernel='rbf', gamma=1.0)
    svm.fit(X_train, y_train)

    n_total = len(X_train)
    n_sv = len(svm.support_vectors)

    print(f"Total training points: {n_total}")
    print(f"Support vectors:       {n_sv} ({100*n_sv/n_total:.1f}%)")
    print("→ Only support vectors determine the decision boundary!")
    print("   Removing other points wouldn't change the model.")

    # -------- Experiment 5: Decision Boundary Visualization --------
    print("\n5. VISUALIZING KERNEL EFFECT ON CIRCLES")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['circles']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    for i, kernel in enumerate(['linear', 'rbf', 'poly']):
        svm = SVM(C=1.0, kernel=kernel, gamma=1.0, degree=2)
        svm.fit(X_train, y_train)
        acc = accuracy(y_test, svm.predict(X_test))

        plot_decision_boundary(svm.predict, X, y, ax=axes[i],
                              title=f'{kernel} kernel (acc={acc:.2f})')

        # Mark support vectors
        if len(svm.support_vectors) > 0:
            axes[i].scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                          s=100, facecolors='none', edgecolors='k', linewidths=2,
                          label='Support Vectors')

    plt.suptitle('SVM: Kernel Comparison on Circles\n'
                 'Linear fails, RBF succeeds. Black circles = support vectors.',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig('/Users/sid47/ML Algorithms/08_svm_kernels.png', dpi=100)
    plt.close(fig)
    print("Saved kernel comparison to 08_svm_kernels.png")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate SVM on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: SVM on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential', 'high_dim']:  # Skip slow ones
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        svm = SVM(C=1.0, kernel='rbf', gamma=1.0)
        svm.fit(X_tr, y_tr)
        y_pred = svm.predict(X_te)
        acc = accuracy(y_te, y_pred)
        n_sv = len(svm.support_vectors)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}  (n_sv={n_sv})")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    print("\n" + "="*60)
    print("VISUALIZING DECISION BOUNDARIES")
    print("="*60)

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

        svm = SVM(C=1.0, kernel='rbf', gamma=1.0)
        svm.fit(X_tr, y_tr)
        acc = accuracy(y_te, svm.predict(X_te))

        plot_decision_boundary(svm.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('SVM (RBF Kernel): Decision Boundaries\n'
                 '(Kernel trick enables nonlinear boundaries)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("SUPPORT VECTOR MACHINE — Paradigm: MARGIN")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Find the hyperplane with MAXIMUM MARGIN to nearest points.
    Only "support vectors" on the margin boundary matter.

THE KERNEL TRICK:
    Compute dot products in high-D feature space without
    explicitly computing the features!
    RBF kernel = infinite dimensional space.

KEY PARAMETERS:
    C: margin vs misclassification tradeoff
    γ (gamma): kernel width (RBF)

INDUCTIVE BIAS:
    - Maximum margin = good generalization
    - Kernel choice defines similarity measure
    - Sparse solution (only support vectors matter)
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize
    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/08_svm_boundaries.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path}")
    plt.close(fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What SVM Reveals")
    print("="*60)
    print("""
1. MARGIN is a fundamentally different objective than loss minimization
2. Only SUPPORT VECTORS determine the boundary (sparse solution)
3. KERNEL TRICK enables nonlinear without explicit feature computation
4. C controls regularization (small C = large margin, simple)
5. γ controls RBF width (small = smooth, large = wiggly)

KEY INSIGHT:
    SVM shows that the RIGHT INDUCTIVE BIAS (margin) can be more
    important than model complexity. A linear SVM with good margin
    often beats a complex model with poor generalization.

NEXT: Random Forest — reduce tree variance by averaging
    """)
