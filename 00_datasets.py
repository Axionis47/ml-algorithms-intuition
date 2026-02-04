"""
SHARED DATASETS — The Challenge Suite

Every model benchmarks against these same datasets.
Each dataset is designed to expose specific capabilities and failures.

WHY THESE DATASETS?
    You can't understand what a model does by watching it succeed.
    You understand it by watching WHERE it fails and WHY.
    These datasets create controlled failure modes.

USAGE:
    from 00_datasets import get_all_datasets, plot_decision_boundary

    datasets = get_all_datasets()
    for name, (X_train, X_test, y_train, y_test) in datasets.items():
        # train and evaluate your model
        ...
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable
import os

# ============================================================
# DATASET GENERATORS
# ============================================================

def make_linear_blobs(n_samples=500, noise=0.1, random_state=42):
    """
    WHAT: Two linearly separable Gaussian blobs.
    TESTS: Baseline — everyone should get ~100%.
    WHO WINS: Everyone.
    WHO FAILS: Nobody (if you fail here, your implementation is broken).
    """
    np.random.seed(random_state)
    n = n_samples // 2

    # Class 0: centered at (-2, -2)
    X0 = np.random.randn(n, 2) * noise + np.array([-2, -2])
    # Class 1: centered at (2, 2)
    X1 = np.random.randn(n, 2) * noise + np.array([2, 2])

    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)

    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def make_circles(n_samples=500, noise=0.05, random_state=42):
    """
    WHAT: Concentric circles — inner circle is class 0, outer is class 1.
    TESTS: Nonlinear decision boundary.
    WHO WINS: Trees, SVMs with RBF kernel, neural networks.
    WHO FAILS: Linear models (logistic regression, linear SVM) — they can only
               draw a straight line, but the boundary is a circle.
    """
    np.random.seed(random_state)
    n = n_samples // 2

    # Inner circle (class 0) — radius 1
    theta0 = np.random.rand(n) * 2 * np.pi
    r0 = 1 + np.random.randn(n) * noise
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])

    # Outer circle (class 1) — radius 3
    theta1 = np.random.rand(n) * 2 * np.pi
    r1 = 3 + np.random.randn(n) * noise
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def make_xor(n_samples=500, noise=0.2, random_state=42):
    """
    WHAT: Four clusters in XOR pattern — (0,0), (1,1) are class 0; (0,1), (1,0) are class 1.
    TESTS: Feature interaction — neither feature alone predicts the class.
    WHO WINS: Trees (natural at axis-aligned XOR), neural networks (learn the interaction).
    WHO FAILS: Any single-layer linear model — XOR is the classic example of
               what a single perceptron cannot learn.
    """
    np.random.seed(random_state)
    n = n_samples // 4

    # Class 0: (0,0) and (1,1)
    X00 = np.random.randn(n, 2) * noise + np.array([0, 0])
    X11 = np.random.randn(n, 2) * noise + np.array([1, 1])

    # Class 1: (0,1) and (1,0)
    X01 = np.random.randn(n, 2) * noise + np.array([0, 1])
    X10 = np.random.randn(n, 2) * noise + np.array([1, 0])

    X = np.vstack([X00, X11, X01, X10])
    y = np.array([0]*n + [0]*n + [1]*n + [1]*n)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_spiral(n_samples=500, noise=0.2, random_state=42):
    """
    WHAT: Two interleaved spirals.
    TESTS: Complex nonlinear manifold — boundary winds through the space.
    WHO WINS: Deep neural networks, Gaussian processes with appropriate kernel.
    WHO FAILS: Shallow models, trees (need very deep trees that overfit).

    This is one of the hardest 2D classification problems.
    """
    np.random.seed(random_state)
    n = n_samples // 2

    # Spiral 1 (class 0)
    theta0 = np.linspace(0, 4*np.pi, n)
    r0 = theta0 / (4*np.pi) * 5
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    X0 += np.random.randn(n, 2) * noise

    # Spiral 2 (class 1) — rotated 180 degrees
    theta1 = np.linspace(0, 4*np.pi, n)
    r1 = theta1 / (4*np.pi) * 5
    X1 = np.column_stack([r1 * np.cos(theta1 + np.pi), r1 * np.sin(theta1 + np.pi)])
    X1 += np.random.randn(n, 2) * noise

    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_moons(n_samples=500, noise=0.15, random_state=42):
    """
    WHAT: Two interleaved half-moons.
    TESTS: Curved nonlinear boundary + noise tolerance.
    WHO WINS: Most nonlinear models.
    WHO FAILS: Linear models, and models that overfit to noise.
    """
    np.random.seed(random_state)
    n = n_samples // 2

    # Moon 1 (class 0) — upper semicircle
    theta0 = np.linspace(0, np.pi, n)
    X0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    X0 += np.random.randn(n, 2) * noise

    # Moon 2 (class 1) — lower semicircle, shifted
    theta1 = np.linspace(0, np.pi, n)
    X1 = np.column_stack([1 - np.cos(theta1), 0.5 - np.sin(theta1)])
    X1 += np.random.randn(n, 2) * noise

    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_high_dim_sparse(n_samples=500, n_features=100, n_informative=5, random_state=42):
    """
    WHAT: 100 features, but only 5 actually matter for classification.
    TESTS: Curse of dimensionality, feature selection, regularization.
    WHO WINS: Models with built-in feature selection (trees, L1-regularized models).
    WHO FAILS: KNN (distances become meaningless in high-D), unregularized models
               (overfit to noise features).

    CURSE OF DIMENSIONALITY:
        In high dimensions, all points become equidistant. The concept of
        "nearest neighbor" breaks down. Models that rely on local structure
        (KNN, kernel methods) struggle.
    """
    np.random.seed(random_state)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Only first n_informative features determine the class
    # Linear combination of informative features
    true_weights = np.random.randn(n_informative)
    signal = X[:, :n_informative] @ true_weights

    # Threshold to create classes
    y = (signal > 0).astype(int)

    # Shuffle (features are already in order of importance, don't reveal this)
    feature_order = np.random.permutation(n_features)
    X = X[:, feature_order]

    return X, y


def make_imbalanced(n_samples=500, ratio=0.05, random_state=42):
    """
    WHAT: Two classes with 95/5 imbalance.
    TESTS: How models handle rare classes.
    WHO WINS: Models with class weighting, probabilistic models.
    WHO FAILS: Models that optimize accuracy (predicting majority class gets 95%),
               models without probability calibration.

    REAL-WORLD ANALOG: Fraud detection, disease diagnosis — the interesting
    class is rare but critically important.
    """
    np.random.seed(random_state)

    n_minority = int(n_samples * ratio)
    n_majority = n_samples - n_minority

    # Majority class (0) — large blob
    X0 = np.random.randn(n_majority, 2) * 1.5 + np.array([0, 0])

    # Minority class (1) — small blob, slightly overlapping
    X1 = np.random.randn(n_minority, 2) * 0.5 + np.array([2, 2])

    X = np.vstack([X0, X1])
    y = np.array([0]*n_majority + [1]*n_minority)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_sequential(n_samples=500, seq_length=20, random_state=42):
    """
    WHAT: Time series classification — pattern in the sequence determines class.
    TESTS: Temporal dependency modeling.
    WHO WINS: RNNs, LSTMs, Transformers.
    WHO FAILS: Everything else — standard classifiers see features as independent,
               but the pattern is in the ORDER.

    PATTERN:
        Class 0: Sequence trends upward then downward (peak in middle)
        Class 1: Sequence trends downward then upward (trough in middle)
    """
    np.random.seed(random_state)

    X = []
    y = []

    for _ in range(n_samples // 2):
        # Class 0: peak pattern
        t = np.linspace(0, 2*np.pi, seq_length)
        seq = np.sin(t) + np.random.randn(seq_length) * 0.2
        X.append(seq)
        y.append(0)

        # Class 1: trough pattern
        seq = -np.sin(t) + np.random.randn(seq_length) * 0.2
        X.append(seq)
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_clustered(n_samples=500, n_clusters=5, random_state=42):
    """
    WHAT: Data with natural cluster structure, but cluster labels are unknown.
    TESTS: Unsupervised structure discovery.
    WHO WINS: Generative models (GMM), clustering algorithms.
    WHO FAILS: Discriminative classifiers (they need labels).

    This is for testing unsupervised methods — the 'y' returned is the true
    cluster assignment for evaluation, but models shouldn't see it during training.
    """
    np.random.seed(random_state)

    n_per_cluster = n_samples // n_clusters
    X = []
    y = []

    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randn(2) * 5
        # Random cluster spread
        spread = np.random.rand() * 0.5 + 0.3

        cluster_points = np.random.randn(n_per_cluster, 2) * spread + center
        X.append(cluster_points)
        y.extend([i] * n_per_cluster)

    X = np.vstack(X)
    y = np.array(y)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_distribution_shift(n_samples=500, random_state=42):
    """
    WHAT: Training and test data come from different distributions.
    TESTS: Robustness to distribution shift, uncertainty quantification.
    WHO WINS: Conformal prediction (provides valid coverage even under shift),
              Bayesian methods (express uncertainty in OOD regions).
    WHO FAILS: Standard point-estimate models (confidently wrong on shifted data).

    TRAINING: Both classes centered near origin.
    TEST: Classes shifted outward — model sees points unlike training data.
    """
    np.random.seed(random_state)
    n = n_samples // 2

    # TRAINING distribution: classes close together
    X_train_0 = np.random.randn(n, 2) * 0.5 + np.array([-1, 0])
    X_train_1 = np.random.randn(n, 2) * 0.5 + np.array([1, 0])
    X_train = np.vstack([X_train_0, X_train_1])
    y_train = np.array([0]*n + [1]*n)

    # TEST distribution: classes shifted outward (covariate shift)
    X_test_0 = np.random.randn(n, 2) * 0.5 + np.array([-3, 0])
    X_test_1 = np.random.randn(n, 2) * 0.5 + np.array([3, 0])
    X_test = np.vstack([X_test_0, X_test_1])
    y_test = np.array([0]*n + [1]*n)

    # Shuffle
    idx_train = np.random.permutation(len(y_train))
    idx_test = np.random.permutation(len(y_test))

    return X_train[idx_train], y_train[idx_train], X_test[idx_test], y_test[idx_test]


# ============================================================
# TRAIN/TEST SPLIT UTILITY
# ============================================================

def train_test_split(X, y, test_ratio=0.2, random_state=42):
    """Simple train/test split."""
    np.random.seed(random_state)
    n = len(y)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# GET ALL DATASETS
# ============================================================

def get_all_datasets(n_samples=500, test_ratio=0.2, random_state=42) -> Dict[str, Tuple]:
    """
    Returns all challenge datasets as a dictionary.

    Each entry is: (X_train, X_test, y_train, y_test)

    For distribution_shift, the split is built-in (train and test are different distributions).
    For sequential, the data is 3D (n_samples, seq_length, 1).
    """
    datasets = {}

    # Standard classification datasets
    standard_makers = [
        ('linear', make_linear_blobs),
        ('circles', make_circles),
        ('xor', make_xor),
        ('spiral', make_spiral),
        ('moons', make_moons),
        ('imbalanced', make_imbalanced),
    ]

    for name, maker in standard_makers:
        X, y = maker(n_samples=n_samples, random_state=random_state)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio, random_state)
        datasets[name] = (X_tr, X_te, y_tr, y_te)

    # High-dim needs more samples to be meaningful
    X, y = make_high_dim_sparse(n_samples=n_samples, random_state=random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio, random_state)
    datasets['high_dim'] = (X_tr, X_te, y_tr, y_te)

    # Sequential data
    X, y = make_sequential(n_samples=n_samples, random_state=random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio, random_state)
    datasets['sequential'] = (X_tr, X_te, y_tr, y_te)

    # Clustered (for unsupervised)
    X, y = make_clustered(n_samples=n_samples, random_state=random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio, random_state)
    datasets['clustered'] = (X_tr, X_te, y_tr, y_te)

    # Distribution shift (has its own train/test split)
    X_tr, y_tr, X_te, y_te = make_distribution_shift(n_samples=n_samples, random_state=random_state)
    datasets['dist_shift'] = (X_tr, X_te, y_tr, y_te)

    return datasets


def get_2d_datasets(n_samples=500, test_ratio=0.2, random_state=42) -> Dict[str, Tuple]:
    """
    Returns only the 2D datasets (for decision boundary visualization).
    Excludes high_dim and sequential.
    """
    all_ds = get_all_datasets(n_samples, test_ratio, random_state)
    return {k: v for k, v in all_ds.items() if k not in ['high_dim', 'sequential']}


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

def plot_dataset(X, y, ax=None, title='', alpha=0.6, s=20):
    """Plot a single 2D dataset."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], label=f'Class {c}',
                   alpha=alpha, s=s, edgecolors='none')

    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize=8)
    return ax


def plot_decision_boundary(model_predict: Callable, X, y, ax=None, title='',
                          resolution=100, alpha=0.3):
    """
    Plot decision boundary for a classifier.

    model_predict: function that takes X and returns predictions
    X, y: data to overlay as scatter
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    # Predict on mesh
    Z = model_predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=alpha, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

    # Overlay data
    plot_dataset(X, y, ax=ax, title=title, alpha=0.8, s=15)

    return ax


def plot_all_datasets(datasets: Dict = None, figsize=(16, 12)):
    """Visualize all 2D datasets in a grid."""
    if datasets is None:
        datasets = get_2d_datasets()

    # Remove non-2D datasets for plotting
    plot_datasets = {k: v for k, v in datasets.items()
                     if k not in ['high_dim', 'sequential']}

    n = len(plot_datasets)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        # Combine train and test for visualization
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])
        plot_dataset(X, y, ax=axes[i], title=name)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig, axes


# ============================================================
# EVALUATION UTILITIES
# ============================================================

def accuracy(y_true, y_pred):
    """Simple accuracy metric."""
    return np.mean(y_true == y_pred)


def evaluate_model(model, datasets: Dict, model_name: str = 'Model'):
    """
    Evaluate a model on all applicable datasets.

    model: object with fit(X, y) and predict(X) methods
    Returns: dict of {dataset_name: accuracy}
    """
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        # Skip sequential unless model handles it
        if name == 'sequential' and X_tr.ndim == 2:
            # For sequential, X is (n_samples, seq_length)
            # Most standard classifiers can't handle this
            continue

        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc = accuracy(y_te, y_pred)
            results[name] = acc
        except Exception as e:
            results[name] = f'Error: {str(e)[:20]}'

    return results


def print_results_table(results: Dict[str, Dict[str, float]], dataset_order: list = None):
    """
    Print comparison table of multiple models across datasets.

    results: {model_name: {dataset_name: accuracy}}
    """
    if not results:
        return

    # Get all datasets
    all_datasets = set()
    for model_results in results.values():
        all_datasets.update(model_results.keys())

    if dataset_order:
        datasets = [d for d in dataset_order if d in all_datasets]
    else:
        datasets = sorted(all_datasets)

    # Header
    header = f"{'Model':<20}" + "".join(f"{d:<12}" for d in datasets)
    print(header)
    print("-" * len(header))

    # Rows
    for model_name, model_results in results.items():
        row = f"{model_name:<20}"
        for d in datasets:
            val = model_results.get(d, '-')
            if isinstance(val, float):
                row += f"{val:<12.3f}"
            else:
                row += f"{str(val):<12}"
        print(row)


# ============================================================
# DEMO
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CHALLENGE DATASETS — The Benchmark Suite")
    print("=" * 60)

    datasets = get_all_datasets()

    print("\nDataset shapes:")
    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        print(f"  {name:<15} train: {X_tr.shape}, test: {X_te.shape}, "
              f"classes: {np.unique(y_tr)}")

    # Visualize 2D datasets
    print("\nPlotting 2D datasets...")
    fig, axes = plot_all_datasets()

    save_path = '/Users/sid47/ML Algorithms/00_datasets.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)  # Close instead of show for non-blocking execution
