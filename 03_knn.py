"""
K-NEAREST NEIGHBORS — Paradigm: MEMORY

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

DON'T MODEL ANYTHING. Just memorize the data.

At prediction time:
    1. Find the k training points closest to x
    2. Return majority class among those k neighbors

The "model" IS the training data. Zero parameters learned.

===============================================================
THE ONE EQUATION
===============================================================

    ŷ(x) = mode({y_i : x_i ∈ N_k(x)})

where N_k(x) is the set of k nearest neighbors of x.

With distance d(x, x'):
    - Euclidean: sqrt(sum((x - x')²))
    - Manhattan: sum(|x - x'|)
    - Minkowski: (sum(|x - x'|^p))^(1/p)

===============================================================
INDUCTIVE BIAS
===============================================================

1. SMOOTHNESS: Points close in feature space have similar labels
2. LOCALITY: Only nearby points matter for prediction
3. DISTANCE METRIC: The metric defines "similarity"
   - Euclidean assumes features are equally important
   - Assumes features are on comparable scales
   - Implicitly assumes isotropic neighborhoods

WHAT IT CAN DO:
    ✓ Nonlinear boundaries (can fit ANY boundary if k=1 and enough data)
    ✓ No training time (just store data)
    ✓ Works with any distance metric

WHAT IT CAN'T DO:
    ✗ Handle high dimensions (curse of dimensionality)
    ✗ Handle irrelevant features (they corrupt distances)
    ✗ Fast prediction (must scan all data)
    ✗ Handle class imbalance well (majority dominates)

===============================================================
CURSE OF DIMENSIONALITY — THE CRITICAL FAILURE MODE
===============================================================

In high dimensions, a counterintuitive thing happens:
ALL POINTS BECOME EQUIDISTANT.

Consider a unit hypercube [0,1]^d:
- To capture 10% of data in d=1: need edge length 0.1
- To capture 10% of data in d=10: need edge length 0.1^(1/10) ≈ 0.79
- To capture 10% of data in d=100: need edge length 0.1^(1/100) ≈ 0.977

In 100 dimensions, your "local" neighborhood spans 97.7% of the space!
The concept of "nearest" becomes meaningless.

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


class KNN:
    """
    K-Nearest Neighbors Classifier

    Zero training. Memorize. Compute distances at prediction time.
    """

    def __init__(self, k=5, distance='euclidean', weighted=False):
        """
        Parameters:
        -----------
        k : number of neighbors to consider
        distance : 'euclidean', 'manhattan', or 'minkowski'
        weighted : if True, closer neighbors have more influence
        """
        self.k = k
        self.distance = distance
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "Fitting" = just storing the data.
        This is the laziest possible learning algorithm.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes = np.unique(y)
        return self

    def _compute_distances(self, X):
        """
        Compute distance from each test point to all training points.

        Returns: (n_test, n_train) matrix of distances
        """
        if self.distance == 'euclidean':
            # ||x - y||² = ||x||² + ||y||² - 2x·y
            # This trick avoids explicit loop
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n_test, 1)
            train_sq = np.sum(self.X_train ** 2, axis=1)   # (n_train,)
            cross = X @ self.X_train.T                      # (n_test, n_train)
            distances = np.sqrt(np.maximum(X_sq + train_sq - 2 * cross, 0))

        elif self.distance == 'manhattan':
            # L1 distance: sum of absolute differences
            # Need to do this with broadcasting
            distances = np.zeros((X.shape[0], self.X_train.shape[0]))
            for i, x in enumerate(X):
                distances[i] = np.sum(np.abs(self.X_train - x), axis=1)

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return distances

    def predict(self, X):
        """
        For each test point:
        1. Compute distances to all training points
        2. Find k nearest
        3. Vote (optionally weighted by 1/distance)
        """
        distances = self._compute_distances(X)
        predictions = []

        for i in range(X.shape[0]):
            # Get k nearest neighbor indices
            k_indices = np.argsort(distances[i])[:self.k]
            k_labels = self.y_train[k_indices]

            if self.weighted:
                # Weight by 1/distance (closer = more weight)
                k_distances = distances[i, k_indices]
                # Avoid division by zero
                k_distances = np.maximum(k_distances, 1e-10)
                weights = 1 / k_distances

                # Weighted vote
                class_votes = {}
                for c in self.classes:
                    class_votes[c] = np.sum(weights[k_labels == c])
                prediction = max(class_votes, key=class_votes.get)
            else:
                # Unweighted majority vote
                prediction = np.bincount(k_labels.astype(int)).argmax()

            predictions.append(prediction)

        return np.array(predictions)


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

    # Use moons dataset — nonlinear but learnable
    X_train, X_test, y_train, y_test = datasets['moons']

    # -------- Experiment 1: Effect of k --------
    print("\n1. EFFECT OF k (number of neighbors)")
    print("-" * 40)
    for k in [1, 3, 5, 10, 20, 50, 100]:
        model = KNN(k=k)
        model.fit(X_train, y_train)
        acc_train = accuracy(y_train, model.predict(X_train))
        acc_test = accuracy(y_test, model.predict(X_test))
        print(f"k={k:<3}  train_acc={acc_train:.3f}  test_acc={acc_test:.3f}")
    print("→ k=1 overfits (perfect train, lower test)")
    print("→ Large k underfit (too much averaging)")
    print("→ Optimal k is somewhere in between")

    # -------- Experiment 2: Distance Metrics --------
    print("\n2. DISTANCE METRIC COMPARISON")
    print("-" * 40)
    for dist in ['euclidean', 'manhattan']:
        model = KNN(k=5, distance=dist)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        print(f"distance={dist:<12} accuracy={acc:.3f}")
    print("→ Euclidean is more sensitive to outliers (squared)")
    print("→ Manhattan is more robust")

    # -------- Experiment 3: Weighted vs Unweighted --------
    print("\n3. WEIGHTED vs UNWEIGHTED VOTING")
    print("-" * 40)
    for weighted in [False, True]:
        model = KNN(k=10, weighted=weighted)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        print(f"weighted={str(weighted):<6} accuracy={acc:.3f}")
    print("→ Weighting helps when some neighbors are much closer")

    # -------- Experiment 4: Feature Scaling --------
    print("\n4. FEATURE SCALING EFFECT")
    print("-" * 40)
    # Create data with different scales
    X_train_scaled = X_train.copy()
    X_train_scaled[:, 0] *= 100  # Scale first feature by 100x
    X_test_scaled = X_test.copy()
    X_test_scaled[:, 0] *= 100

    model_original = KNN(k=5)
    model_original.fit(X_train, y_train)
    acc_original = accuracy(y_test, model_original.predict(X_test))

    model_scaled = KNN(k=5)
    model_scaled.fit(X_train_scaled, y_train)
    acc_scaled = accuracy(y_test, model_scaled.predict(X_test_scaled))

    print(f"Original features:        accuracy={acc_original:.3f}")
    print(f"First feature scaled 100x: accuracy={acc_scaled:.3f}")
    print("→ Unequal scales make one feature dominate distance!")

    # -------- Experiment 5: CURSE OF DIMENSIONALITY --------
    print("\n5. CURSE OF DIMENSIONALITY")
    print("-" * 40)
    X_high, X_test_high, y_high, y_test_high = datasets['high_dim']

    # Compare distance statistics in low-D vs high-D
    X_low = X_train[:100]  # Take 100 points from 2D moons

    # Low-D distances
    from scipy.spatial.distance import pdist
    dist_low = pdist(X_low, metric='euclidean')
    print(f"Low-D (2 features):")
    print(f"  Distance range: [{dist_low.min():.3f}, {dist_low.max():.3f}]")
    print(f"  Distance std:   {dist_low.std():.3f}")
    print(f"  Ratio max/min:  {dist_low.max()/dist_low.min():.1f}x")

    # High-D distances
    dist_high = pdist(X_high[:100], metric='euclidean')
    print(f"\nHigh-D (100 features):")
    print(f"  Distance range: [{dist_high.min():.3f}, {dist_high.max():.3f}]")
    print(f"  Distance std:   {dist_high.std():.3f}")
    print(f"  Ratio max/min:  {dist_high.max()/dist_high.min():.1f}x")

    # Accuracy comparison
    model_low = KNN(k=5)
    model_low.fit(X_train, y_train)
    acc_low = accuracy(y_test, model_low.predict(X_test))

    model_high = KNN(k=5)
    model_high.fit(X_high, y_high)
    acc_high = accuracy(y_test_high, model_high.predict(X_test_high))

    print(f"\nAccuracy on moons (2D):    {acc_low:.3f}")
    print(f"Accuracy on high_dim (100D): {acc_high:.3f}")
    print("→ In high-D, all points are nearly equidistant!")
    print("   'Nearest neighbor' becomes meaningless.")

    # -------- Experiment 6: Effect of Noise Features --------
    print("\n6. EFFECT OF IRRELEVANT FEATURES")
    print("-" * 40)
    for n_noise in [0, 2, 5, 10, 20]:
        X_noisy = np.c_[X_train, np.random.randn(X_train.shape[0], n_noise)]
        X_test_noisy = np.c_[X_test, np.random.randn(X_test.shape[0], n_noise)]

        model = KNN(k=5)
        model.fit(X_noisy, y_train)
        acc = accuracy(y_test, model.predict(X_test_noisy))
        print(f"n_noise_features={n_noise:<2}  accuracy={acc:.3f}")
    print("→ Noise features corrupt distances, hurting KNN badly")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate KNN on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: KNN on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = KNN(k=5)
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
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
    plot_datasets = {k: v for k, v in datasets.items() if k != 'clustered'}

    n = len(plot_datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        model = KNN(k=5)
        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('KNN (k=5): Decision Boundaries\n'
                 '(Can fit ANY boundary — just memorizes local structure)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_k_effect():
    """Visualize how k affects decision boundary."""
    print("\n" + "="*60)
    print("EFFECT OF k ON DECISION BOUNDARY")
    print("="*60)

    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    k_values = [1, 3, 5, 10, 30, 100]

    for i, k in enumerate(k_values):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'k={k} (acc={acc:.2f})')

    plt.suptitle('KNN: Effect of k on Decision Boundary (Moons Dataset)\n'
                 'k=1 overfits (jagged), large k underfits (oversmoothed)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_with_linear():
    """Compare KNN vs Linear models."""
    print("\n" + "="*60)
    print("COMPARISON: KNN vs Linear Models")
    print("="*60)

    from importlib import import_module
    logistic_module = import_module('02_logistic_regression')
    LogisticRegression = logistic_module.LogisticRegression

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'Logistic':<10} {'KNN(k=5)':<10} {'Winner':<10}")
    print("-" * 45)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        # Logistic
        log_model = LogisticRegression(lr=0.1, n_iters=1000)
        log_model.fit(X_tr, y_tr)
        acc_log = accuracy(y_te, log_model.predict(X_te))

        # KNN
        knn_model = KNN(k=5)
        knn_model.fit(X_tr, y_tr)
        acc_knn = accuracy(y_te, knn_model.predict(X_te))

        winner = 'KNN' if acc_knn > acc_log else 'Logistic' if acc_log > acc_knn else 'Tie'
        print(f"{name:<15} {acc_log:<10.3f} {acc_knn:<10.3f} {winner}")

    print("\n→ KNN wins on nonlinear data (circles, xor, spiral, moons)")
    print("→ Logistic wins on high-dim (curse of dimensionality kills KNN)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("K-NEAREST NEIGHBORS — Paradigm: MEMORY")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Memorize all training data. Predict by looking up similar points.
    "What did my neighbors say?"

THE RADICAL INSIGHT:
    No parameters to learn. No training. The data IS the model.

INDUCTIVE BIAS:
    - Points close together have similar labels (smoothness)
    - Distance metric defines "similar"
    - All features are equally important (unless you weight)

STRENGTHS:
    ✓ Can learn ANY boundary (nonparametric)
    ✓ No training time
    ✓ Simple, interpretable

WEAKNESSES:
    ✗ Slow prediction (O(n) distance computations)
    ✗ Dies in high dimensions (curse of dimensionality)
    ✗ Sensitive to irrelevant features
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Compare with linear
    compare_with_linear()

    # Visualize decision boundaries
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/03_knn_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path1}")
    plt.close(fig1)

    # Visualize k effect
    fig2 = visualize_k_effect()
    save_path2 = '/Users/sid47/ML Algorithms/03_knn_k_effect.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved k effect to: {save_path2}")
    plt.close(fig2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What KNN Reveals")
    print("="*60)
    print("""
1. You don't NEED to model. Memorization can work.
2. Nonlinear boundaries emerge naturally from local voting
3. The distance metric IS the inductive bias
4. CURSE OF DIMENSIONALITY: In high-D, all points equidistant
5. k controls bias-variance: k=1 overfits, large k underfits

KEY TRADEOFF:
    KNN trades training time for prediction time.
    Linear models: slow train, fast predict
    KNN: instant "train", slow predict

WHEN TO USE:
    - Low dimensions
    - All features are relevant
    - You have compute budget for test time

NEXT: Naive Bayes (Probabilistic paradigm) — model the data distribution
    """)
