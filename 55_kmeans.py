"""
K-MEANS CLUSTERING — Paradigm: CENTROID PARTITIONING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Partition data into K clusters by finding K centroids that
minimize within-cluster variance:

    argmin  Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

"Find K points (centroids) such that each data point is close
 to its nearest centroid."

THE ALGORITHM (Lloyd's Algorithm):
    1. Initialize K centroids (randomly or smartly)
    2. ASSIGN: Each point → nearest centroid
    3. UPDATE: Each centroid → mean of its points
    4. Repeat 2-3 until convergence

===============================================================
THE KEY INSIGHT: VORONOI PARTITIONING
===============================================================

K-means creates a VORONOI TESSELLATION of the space:
- Each centroid "owns" the region closer to it than any other
- Boundaries are perpendicular bisectors between centroids
- Every point gets assigned to exactly one cluster

The algorithm alternates between:
    - "Given centroids, find best assignments" (easy: nearest neighbor)
    - "Given assignments, find best centroids" (easy: mean)

Each step DECREASES the objective. Convergence guaranteed!

===============================================================
INDUCTIVE BIAS — What K-Means Assumes
===============================================================

1. SPHERICAL CLUSTERS: K-means uses Euclidean distance →
   assumes clusters are roughly spherical (isotropic)

2. EQUAL VARIANCE: All clusters have similar spread.
   Elongated or varying-size clusters are problematic.

3. EQUAL SIZE: Tends to produce clusters of similar size.
   Can split large natural clusters.

4. K IS KNOWN: Must specify number of clusters beforehand.
   Choosing K is a separate problem (elbow method, silhouette).

5. CONVEX CLUSTERS: Cannot find clusters that wrap around
   each other (spirals, moons, etc.)

===============================================================
K-MEANS++ INITIALIZATION
===============================================================

Bad initialization → bad local minimum!

K-Means++ fixes this:
    1. Choose first centroid uniformly at random
    2. For each subsequent centroid:
       - Compute D(x) = distance to nearest existing centroid
       - Choose new centroid with probability ∝ D(x)²
       - This spreads centroids apart!

K-Means++ gives O(log k) approximation guarantee.

===============================================================
FAILURE MODES
===============================================================

1. ELONGATED CLUSTERS: K-means splits them into spherical pieces
   → Use GMM with full covariance instead

2. DENSITY VARIATIONS: Dense and sparse clusters problematic
   → Use DBSCAN (density-based)

3. NON-CONVEX SHAPES: Spirals, moons, interlocking rings
   → Use Spectral Clustering or DBSCAN

4. OUTLIERS: Single outlier can pull centroid away
   → Use K-medoids (uses actual points as centers)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
make_clustered = datasets_module.make_clustered
make_circles = datasets_module.make_circles
make_moons = datasets_module.make_moons


class KMeans:
    """
    K-Means Clustering — Classic Lloyd's Algorithm.

    Paradigm: CENTROID PARTITIONING
    - Hard assignment: each point belongs to exactly one cluster
    - Spherical assumption: uses Euclidean distance
    - Iterative refinement: assign-update until convergence
    """

    def __init__(self, n_clusters=3, init='kmeans++', max_iter=300,
                 tol=1e-4, n_init=10, random_state=None):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters K
        init : str
            'random': random initialization
            'kmeans++': smart initialization (spread out)
        max_iter : int
            Maximum iterations per run
        tol : float
            Convergence tolerance (centroid movement)
        n_init : int
            Number of runs with different initializations
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        # Attributes set after fit
        self.cluster_centers_ = None  # Centroids (K × d)
        self.labels_ = None           # Cluster assignments (n_samples,)
        self.inertia_ = None          # Within-cluster sum of squares
        self.n_iter_ = None           # Iterations until convergence

    def _init_centroids_random(self, X):
        """Initialize centroids by randomly selecting K data points."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centroids_kmeans_plus_plus(self, X):
        """
        K-Means++ initialization: spread centroids apart.

        1. Choose first centroid uniformly at random
        2. For each next centroid:
           - Compute D(x)² = squared distance to nearest centroid
           - Sample new centroid with probability ∝ D(x)²
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # First centroid: random
        centroids[0] = X[np.random.randint(n_samples)]

        for k in range(1, self.n_clusters):
            # Compute squared distances to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                # Distance to each existing centroid
                dists_to_centroids = np.sum((X[i] - centroids[:k])**2, axis=1)
                distances[i] = np.min(dists_to_centroids)

            # Sample proportional to D²
            probabilities = distances / distances.sum()
            new_idx = np.random.choice(n_samples, p=probabilities)
            centroids[k] = X[new_idx]

        return centroids

    def _assign_clusters(self, X, centroids):
        """
        E-step equivalent: assign each point to nearest centroid.

        Returns cluster labels and within-cluster sum of squares.
        """
        n_samples = X.shape[0]

        # Compute distances to all centroids (n_samples × K)
        # ||x - c||² = ||x||² + ||c||² - 2x·c
        X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
        C_sq = np.sum(centroids**2, axis=1)         # (K,)
        XC = X @ centroids.T                         # (n, K)

        distances_sq = X_sq + C_sq - 2 * XC  # Broadcasting: (n, K)

        # Assign to nearest
        labels = np.argmin(distances_sq, axis=1)

        # Compute inertia (within-cluster sum of squares)
        inertia = np.sum(np.min(distances_sq, axis=1))

        return labels, inertia

    def _update_centroids(self, X, labels):
        """
        M-step equivalent: update centroids to cluster means.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize randomly
                centroids[k] = X[np.random.randint(X.shape[0])]

        return centroids

    def _single_run(self, X):
        """Single K-means run with one initialization."""
        # Initialize
        if self.init == 'kmeans++':
            centroids = self._init_centroids_kmeans_plus_plus(X)
        else:
            centroids = self._init_centroids_random(X)

        for iteration in range(self.max_iter):
            # Assign
            labels, inertia = self._assign_clusters(X, centroids)

            # Update
            new_centroids = self._update_centroids(X, labels)

            # Check convergence (centroid movement)
            shift = np.sum((new_centroids - centroids)**2)
            centroids = new_centroids

            if shift < self.tol:
                break

        return centroids, labels, inertia, iteration + 1

    def fit(self, X):
        """
        Fit K-Means clustering.

        Runs n_init times and keeps the best result (lowest inertia).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_inertia = np.inf

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(X)

            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centroids
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter

        return self

    def predict(self, X):
        """Assign new points to nearest centroid."""
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """Transform X to cluster-distance space."""
        # Return distances to each centroid
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(self.cluster_centers_**2, axis=1)
        XC = X @ self.cluster_centers_.T
        return np.sqrt(np.maximum(X_sq + C_sq - 2 * XC, 0))


def clustering_accuracy(y_true, y_pred, n_clusters):
    """
    Compute clustering accuracy with optimal label permutation.

    Clusters have arbitrary labels, so we find the permutation
    that maximizes accuracy.
    """
    best_acc = 0
    for perm in permutations(range(n_clusters)):
        y_mapped = np.array([perm[y] if y < len(perm) else -1 for y in y_pred])
        acc = np.mean(y_mapped == y_true)
        best_acc = max(best_acc, acc)
    return best_acc


def silhouette_score(X, labels):
    """
    Compute silhouette score — how well-separated are clusters?

    For each point:
        a = mean distance to points in same cluster
        b = mean distance to points in nearest other cluster
        s = (b - a) / max(a, b)

    Score in [-1, 1]: higher = better separated
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0.0  # Can't compute silhouette with 1 cluster

    silhouettes = np.zeros(n_samples)

    for i in range(n_samples):
        # a: mean distance to same-cluster points
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        if np.sum(same_cluster) > 0:
            a = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i])**2, axis=1)))
        else:
            a = 0

        # b: mean distance to nearest other cluster
        b = np.inf
        for k in unique_labels:
            if k != labels[i]:
                other_cluster = labels == k
                if np.sum(other_cluster) > 0:
                    mean_dist = np.mean(np.sqrt(np.sum((X[other_cluster] - X[i])**2, axis=1)))
                    b = min(b, mean_dist)

        if b == np.inf:
            b = 0

        silhouettes[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return np.mean(silhouettes)


def elbow_analysis(X, k_range=range(1, 11)):
    """
    Elbow method for choosing K.

    Plot inertia vs K — look for "elbow" where improvement slows.
    """
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    return list(k_range), inertias


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X, y_true = make_clustered(n_samples=600, n_clusters=5)

    # -------- Experiment 1: Initialization Method --------
    print("\n1. EFFECT OF INITIALIZATION METHOD")
    print("-" * 40)
    print("K-Means++ vs Random initialization:")

    for init_method in ['random', 'kmeans++']:
        inertias = []
        for seed in range(20):
            km = KMeans(n_clusters=5, init=init_method, n_init=1, random_state=seed)
            km.fit(X)
            inertias.append(km.inertia_)
        print(f"  {init_method:<10} inertia: mean={np.mean(inertias):.1f}, "
              f"std={np.std(inertias):.1f}, range=[{min(inertias):.1f}, {max(inertias):.1f}]")
    print("-> K-Means++ gives more consistent results (lower variance)")
    print("-> Random can get stuck in bad local minima")

    # -------- Experiment 2: Number of Clusters --------
    print("\n2. EFFECT OF NUMBER OF CLUSTERS (K)")
    print("-" * 40)
    print("True K=5. What happens with different K?")

    for k in [2, 3, 4, 5, 6, 7, 8]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        print(f"  K={k}  inertia={km.inertia_:>8.1f}  silhouette={sil:.3f}")
    print("-> Inertia always decreases (more clusters = less within-cluster variance)")
    print("-> Silhouette peaks near true K")

    # -------- Experiment 3: Number of Initializations --------
    print("\n3. EFFECT OF N_INIT (Number of Random Starts)")
    print("-" * 40)

    for n_init in [1, 3, 5, 10, 20]:
        # Run multiple trials
        best_inertias = []
        for trial in range(10):
            km = KMeans(n_clusters=5, n_init=n_init, random_state=trial*100)
            km.fit(X)
            best_inertias.append(km.inertia_)
        print(f"  n_init={n_init:<3}  mean_best_inertia={np.mean(best_inertias):.1f}  "
              f"std={np.std(best_inertias):.1f}")
    print("-> More initializations = better chance of finding global minimum")
    print("-> Diminishing returns after n_init~10")

    # -------- Experiment 4: Cluster Shape Sensitivity --------
    print("\n4. CLUSTER SHAPE SENSITIVITY (K-Means' Achilles Heel)")
    print("-" * 40)

    # Create elongated clusters
    np.random.seed(42)
    X_elongated = np.vstack([
        np.random.randn(100, 2) @ [[3, 0], [0, 0.3]] + [0, 0],   # Horizontal ellipse
        np.random.randn(100, 2) @ [[0.3, 0], [0, 3]] + [5, 0],   # Vertical ellipse
    ])
    y_elongated = np.array([0]*100 + [1]*100)

    km = KMeans(n_clusters=2, random_state=42)
    km.fit(X_elongated)
    acc = clustering_accuracy(y_elongated, km.labels_, 2)
    print(f"  Elongated clusters: accuracy={acc:.3f}")

    # Create concentric circles
    X_circles, y_circles = make_circles(n_samples=300)

    km.fit(X_circles)
    acc = clustering_accuracy(y_circles, km.labels_, 2)
    print(f"  Concentric circles: accuracy={acc:.3f}")

    # Create moons
    X_moons, y_moons = make_moons(n_samples=300)

    km.fit(X_moons)
    acc = clustering_accuracy(y_moons, km.labels_, 2)
    print(f"  Half-moons:         accuracy={acc:.3f}")

    print("-> K-Means fails on non-spherical clusters!")
    print("-> Elongated: splits the ellipse in half")
    print("-> Non-convex (circles, moons): cannot separate")

    # -------- Experiment 5: Convergence Speed --------
    print("\n5. CONVERGENCE BEHAVIOR")
    print("-" * 40)

    # Track iterations for different K
    for k in [2, 5, 10, 20]:
        n_iters = []
        for seed in range(20):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed)
            km.fit(X)
            n_iters.append(km.n_iter_)
        print(f"  K={k:<3}  iterations: mean={np.mean(n_iters):.1f}, "
              f"max={max(n_iters)}")
    print("-> More clusters = more iterations to converge")
    print("-> K-Means is generally fast (linear in n)")


def visualize_kmeans():
    """Visualize K-Means clustering and its limitations."""
    print("\n" + "="*60)
    print("K-MEANS VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Different K values on clustered data
    np.random.seed(42)
    X, y_true = make_clustered(n_samples=500, n_clusters=4)

    for i, k in enumerate([2, 3, 4, 5]):
        ax = fig.add_subplot(3, 4, i+1)
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)

        ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', alpha=0.6, s=20)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                  c='red', marker='X', s=200, edgecolors='black', linewidth=2)

        sil = silhouette_score(X, km.labels_)
        ax.set_title(f'K={k}, sil={sil:.2f}')
        ax.set_aspect('equal')

    # Row 2: K-Means failures (non-spherical clusters)
    X_elongated = np.vstack([
        np.random.randn(150, 2) @ [[3, 0], [0, 0.3]] + [0, 0],
        np.random.randn(150, 2) @ [[3, 0], [0, 0.3]] + [0, 3],
    ])
    y_elongated = np.array([0]*150 + [1]*150)

    X_sizes = np.vstack([
        np.random.randn(200, 2) * 2 + [0, 0],
        np.random.randn(50, 2) * 0.3 + [5, 0],
    ])
    y_sizes = np.array([0]*200 + [1]*50)

    X_circles_vis, y_circles_vis = make_circles(n_samples=300)
    X_moons_vis, y_moons_vis = make_moons(n_samples=300)

    datasets = [
        ("Elongated", X_elongated, y_elongated),
        ("Circles", X_circles_vis, y_circles_vis),
        ("Moons", X_moons_vis, y_moons_vis),
        ("Sizes", X_sizes, y_sizes),
    ]

    for i, (name, X_local, y_true_local) in enumerate(datasets):

        ax = fig.add_subplot(3, 4, 5+i)
        km = KMeans(n_clusters=2, random_state=42)
        km.fit(X_local)

        acc = clustering_accuracy(y_true_local, km.labels_, 2)

        # Plot with true colors (outline) and predicted colors (fill)
        scatter = ax.scatter(X_local[:, 0], X_local[:, 1], c=km.labels_,
                           cmap='coolwarm', alpha=0.6, s=30,
                           edgecolors=plt.cm.Set1(y_true_local), linewidth=1)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                  c='black', marker='X', s=200, edgecolors='white', linewidth=2)

        ax.set_title(f'{name}\nacc={acc:.2f}')
        ax.set_aspect('equal')

    # Row 3: Elbow and Silhouette analysis
    X_analysis, _ = make_clustered(n_samples=500, n_clusters=4)

    # Elbow plot
    ax = fig.add_subplot(3, 4, 9)
    k_range, inertias = elbow_analysis(X_analysis)
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=4, color='r', linestyle='--', label='True K=4')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Silhouette plot
    ax = fig.add_subplot(3, 4, 10)
    silhouettes = [silhouette_score(X_analysis, KMeans(n_clusters=k, random_state=42).fit_predict(X_analysis))
                   for k in range(2, 11)]
    ax.plot(range(2, 11), silhouettes, 'go-', linewidth=2, markersize=8)
    ax.axvline(x=4, color='r', linestyle='--', label='True K=4')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # K-Means++ vs Random
    ax = fig.add_subplot(3, 4, 11)
    random_inertias = []
    kpp_inertias = []
    for seed in range(50):
        km_rand = KMeans(n_clusters=4, init='random', n_init=1, random_state=seed)
        km_kpp = KMeans(n_clusters=4, init='kmeans++', n_init=1, random_state=seed)
        km_rand.fit(X_analysis)
        km_kpp.fit(X_analysis)
        random_inertias.append(km_rand.inertia_)
        kpp_inertias.append(km_kpp.inertia_)

    ax.hist(random_inertias, bins=15, alpha=0.5, label='Random', color='blue')
    ax.hist(kpp_inertias, bins=15, alpha=0.5, label='K-Means++', color='green')
    ax.set_xlabel('Inertia')
    ax.set_ylabel('Count')
    ax.set_title('Initialization Comparison\n(50 runs each)')
    ax.legend()

    # Convergence over iterations
    ax = fig.add_subplot(3, 4, 12)
    km = KMeans(n_clusters=4, max_iter=1, n_init=1, random_state=42)
    inertia_history = []
    for _ in range(20):
        km.fit(X_analysis)
        inertia_history.append(km.inertia_)
        km.max_iter += 1
    ax.plot(range(1, 21), inertia_history, 'b-', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Inertia')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)

    plt.suptitle('K-MEANS CLUSTERING\n'
                 'Row 1: Effect of K | Row 2: Failure modes (fill=predicted, edge=true) | '
                 'Row 3: Analysis tools',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_kmeans_algorithm():
    """Visualize K-Means algorithm step by step."""
    print("\n" + "="*60)
    print("K-MEANS ALGORITHM VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    X, _ = make_clustered(n_samples=200, n_clusters=3)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Manual step-by-step
    np.random.seed(123)
    indices = np.random.choice(len(X), 3, replace=False)
    centroids = X[indices].copy()

    steps = [0, 1, 2, 3, 5, 10, 15, 20]

    for idx, step in enumerate(steps):
        ax = axes[idx // 4, idx % 4]

        # Run K-means for 'step' iterations
        if step == 0:
            # Just show initial centroids
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=20)
            ax.scatter(centroids[:, 0], centroids[:, 1],
                      c=['red', 'green', 'blue'], marker='X', s=300,
                      edgecolors='black', linewidth=2)
            ax.set_title(f'Step {step}: Initialize')
        else:
            # Run K-means
            km = KMeans(n_clusters=3, max_iter=step, n_init=1, random_state=123)
            km.fit(X)

            colors = ['red', 'green', 'blue']
            for k in range(3):
                mask = km.labels_ == k
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.5, s=20)

            ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                      c=colors, marker='X', s=300, edgecolors='black', linewidth=2)

            ax.set_title(f'Step {step}: inertia={km.inertia_:.0f}')

        ax.set_aspect('equal')
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    plt.suptitle('K-MEANS ALGORITHM: Step-by-Step Convergence\n'
                 'X marks = centroids, colors = cluster assignments',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_clustering():
    """Benchmark K-Means on various datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: K-Means Clustering")
    print("="*60)

    results = {}

    # Test on clustered data with different K
    for true_k in [3, 4, 5]:
        X, y_true = make_clustered(n_samples=600, n_clusters=true_k, random_state=42)

        km = KMeans(n_clusters=true_k, random_state=42)
        km.fit(X)

        acc = clustering_accuracy(y_true, km.labels_, true_k)
        sil = silhouette_score(X, km.labels_)

        results[f'clustered_K{true_k}'] = {'accuracy': acc, 'silhouette': sil}
        print(f"Clustered (K={true_k}): accuracy={acc:.3f}, silhouette={sil:.3f}")

    # Test on challenging datasets
    X_circles, y_circles = make_circles(n_samples=300)
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(X_circles)
    acc = clustering_accuracy(y_circles, km.labels_, 2)
    results['circles'] = {'accuracy': acc}
    print(f"Circles:            accuracy={acc:.3f} (expected: ~0.5 - K-means fails)")

    X_moons, y_moons = make_moons(n_samples=300)
    km.fit(X_moons)
    acc = clustering_accuracy(y_moons, km.labels_, 2)
    results['moons'] = {'accuracy': acc}
    print(f"Moons:              accuracy={acc:.3f} (expected: ~0.5 - K-means fails)")

    return results


if __name__ == '__main__':
    print("="*60)
    print("K-MEANS CLUSTERING — Paradigm: CENTROID PARTITIONING")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Partition data into K clusters by minimizing within-cluster variance:

    argmin  Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

    Each point belongs to the cluster with nearest centroid.

THE ALGORITHM (Lloyd's):
    1. Initialize K centroids
    2. ASSIGN each point to nearest centroid
    3. UPDATE centroids to cluster means
    4. Repeat until convergence

K-MEANS++ INITIALIZATION:
    Spread initial centroids apart:
    - First: random
    - Each next: sample with probability ∝ D(x)²

INDUCTIVE BIAS (What K-Means Assumes):
    - Spherical clusters (isotropic, equal variance)
    - Clusters of similar size
    - Convex cluster shapes
    - K is known beforehand

FAILURE MODES:
    - Elongated clusters → splits them
    - Non-convex (moons, circles) → cannot separate
    - Outliers → pull centroids away
    """)

    ablation_experiments()
    results = benchmark_clustering()

    print("\nGenerating visualizations...")

    fig1 = visualize_kmeans()
    save_path1 = '/Users/sid47/ML Algorithms/55_kmeans.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_kmeans_algorithm()
    save_path2 = '/Users/sid47/ML Algorithms/55_kmeans_algorithm.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "="*60)
    print("SUMMARY: What K-Means Reveals")
    print("="*60)
    print("""
1. K-Means creates VORONOI partitions — each centroid owns a region
2. Objective ALWAYS decreases — convergence guaranteed to LOCAL minimum
3. K-Means++ initialization prevents bad local minima
4. SPHERICAL ASSUMPTION: fails on elongated/non-convex clusters
5. Must choose K beforehand — use elbow/silhouette methods
6. Fast and simple — O(n·K·d·iter), usually converges quickly

WHEN TO USE:
    - Roughly spherical clusters
    - When you know K (or can estimate it)
    - As initialization for more complex methods
    - Fast baseline for clustering

WHEN TO AVOID:
    - Non-convex clusters (use DBSCAN or Spectral)
    - Unknown number of clusters (use DBSCAN)
    - Elliptical clusters (use GMM)

NEXT: Hierarchical Clustering — no need to specify K, builds dendrogram
    """)
