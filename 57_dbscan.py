"""
DBSCAN — Paradigm: DENSITY-BASED CLUSTERING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Clusters are DENSE REGIONS separated by SPARSE REGIONS.

Key insight: Don't count clusters, FIND them by following density.

Two parameters define "dense":
    ε (eps): Neighborhood radius
    MinPts: Minimum points to be "dense"

Point classification:
    CORE POINT: Has ≥ MinPts neighbors within ε
    BORDER POINT: Within ε of a core point, but not core itself
    NOISE POINT: Neither core nor border → outlier!

===============================================================
THE ALGORITHM
===============================================================

1. For each unvisited point p:
   a. Find all neighbors within ε
   b. If |neighbors| < MinPts → mark as NOISE (for now)
   c. If |neighbors| ≥ MinPts → start a new cluster:
      - Add p and all neighbors to cluster
      - For each neighbor that's a core point:
        - Recursively add ITS neighbors
      - Continue until no more core points to expand

2. Border points may be reassigned from noise to a cluster

DENSITY-REACHABILITY:
    - Point q is "directly density-reachable" from p if:
      p is core AND d(p,q) ≤ ε
    - Point q is "density-reachable" from p if there's a chain
      of directly density-reachable points from p to q
    - A cluster is a maximal set of density-connected points

===============================================================
THE KEY INSIGHT: NO K REQUIRED
===============================================================

Unlike K-means or hierarchical, DBSCAN doesn't need K!

The number of clusters EMERGES from the data:
- Dense regions → clusters
- Sparse regions → boundaries between clusters
- Isolated points → noise/outliers

This is powerful when you don't know how many clusters exist.

===============================================================
INDUCTIVE BIAS
===============================================================

1. DENSITY-BASED: Clusters are dense regions
   (fails if clusters have varying densities)

2. ARBITRARY SHAPES: Can find non-convex clusters
   (moons, spirals, rings — no problem!)

3. OUTLIER DETECTION: Naturally identifies noise points
   (K-means forces everything into a cluster)

4. PARAMETER SENSITIVITY:
   - ε too small → everything is noise
   - ε too large → everything in one cluster
   - MinPts too large → few core points
   - MinPts too small → noise becomes clusters

5. STRUGGLES WITH VARYING DENSITY:
   ε is global — can't handle clusters of different densities

===============================================================
CHOOSING PARAMETERS (Practical Guidance)
===============================================================

MinPts:
    Rule of thumb: MinPts ≥ d + 1 (d = dimensions)
    Or: MinPts = 2 × d
    For noisy data: increase MinPts

ε (eps):
    Use the k-distance graph:
    1. For each point, compute distance to k-th nearest neighbor
    2. Sort these distances
    3. Plot them — look for "elbow"
    4. ε ≈ y-value at elbow

===============================================================
COMPLEXITY
===============================================================

Without spatial index: O(n²) — compute all pairwise distances
With spatial index (KD-tree, Ball tree): O(n log n) average

DBSCAN is often faster than hierarchical in practice.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from collections import deque
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
make_clustered = datasets_module.make_clustered
make_moons = datasets_module.make_moons
make_circles = datasets_module.make_circles
make_spiral = datasets_module.make_spiral


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Paradigm: DENSITY-BASED
    - No K required — number of clusters emerges
    - Finds arbitrary-shaped clusters
    - Identifies outliers/noise
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Parameters:
        -----------
        eps : float
            The maximum distance between two samples for one to be
            considered as in the neighborhood of the other (ε).
        min_samples : int
            The number of samples in a neighborhood for a point
            to be considered as a core point (MinPts).
        """
        self.eps = eps
        self.min_samples = min_samples

        # Attributes set after fit
        self.labels_ = None           # Cluster labels (-1 = noise)
        self.core_sample_indices_ = None  # Indices of core points
        self.components_ = None       # Core points themselves
        self.n_clusters_ = None       # Number of clusters found

    def _compute_distance_matrix(self, X):
        """Compute pairwise Euclidean distance matrix."""
        n = X.shape[0]
        X_sq = np.sum(X**2, axis=1)
        D_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * X @ X.T
        return np.sqrt(np.maximum(D_sq, 0))

    def _get_neighbors(self, D, point_idx):
        """Get indices of all points within eps of point_idx."""
        return np.where(D[point_idx] <= self.eps)[0]

    def fit(self, X):
        """
        Perform DBSCAN clustering.

        Algorithm:
        1. Compute distance matrix (or use spatial index)
        2. Identify core points (≥ min_samples neighbors)
        3. Build clusters by expanding from core points
        4. Assign border points to nearest core's cluster
        5. Everything else is noise
        """
        n_samples = X.shape[0]

        # Initialize all points as unvisited noise
        self.labels_ = np.full(n_samples, -1, dtype=int)

        # Compute distance matrix
        D = self._compute_distance_matrix(X)

        # Find neighbors for each point
        neighbors = [self._get_neighbors(D, i) for i in range(n_samples)]

        # Identify core points
        core_mask = np.array([len(neighbors[i]) >= self.min_samples
                             for i in range(n_samples)])
        self.core_sample_indices_ = np.where(core_mask)[0]
        self.components_ = X[self.core_sample_indices_]

        # Track visited points
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True

            point_neighbors = neighbors[point_idx]

            # Not a core point — leave as noise (for now, may become border)
            if len(point_neighbors) < self.min_samples:
                continue

            # Start a new cluster
            self.labels_[point_idx] = cluster_id

            # BFS to expand cluster
            # Use deque for efficient popleft
            seed_set = deque([n for n in point_neighbors if n != point_idx])

            while seed_set:
                current = seed_set.popleft()

                if self.labels_[current] == -1:
                    # Was noise, now border point
                    self.labels_[current] = cluster_id

                if visited[current]:
                    continue

                visited[current] = True
                self.labels_[current] = cluster_id

                current_neighbors = neighbors[current]

                # If current is a core point, expand
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if not visited[neighbor]:
                            seed_set.append(neighbor)

            cluster_id += 1

        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def k_distance_plot(X, k=5):
    """
    Generate k-distance plot for choosing epsilon.

    For each point, compute distance to k-th nearest neighbor.
    Sort and plot — look for elbow.
    """
    n_samples = X.shape[0]

    # Compute distance matrix
    X_sq = np.sum(X**2, axis=1)
    D = np.sqrt(np.maximum(X_sq[:, np.newaxis] + X_sq - 2 * X @ X.T, 0))

    # Get k-th nearest neighbor distance for each point
    # (k+1 because point itself is at distance 0)
    k_distances = np.sort(D, axis=1)[:, k]

    # Sort these distances
    k_distances_sorted = np.sort(k_distances)[::-1]

    return k_distances_sorted


def clustering_accuracy(y_true, y_pred, ignore_noise=True):
    """
    Compute clustering accuracy with optimal label permutation.

    If ignore_noise, exclude noise points (-1) from accuracy calculation.
    """
    if ignore_noise:
        mask = y_pred != -1
        if np.sum(mask) == 0:
            return 0.0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred

    # Get unique predicted labels (excluding -1)
    unique_pred = [l for l in np.unique(y_pred_filtered) if l != -1]
    unique_true = np.unique(y_true_filtered)

    if len(unique_pred) == 0:
        return 0.0

    # Try all permutations (limited to prevent explosion)
    n_labels = max(len(unique_pred), len(unique_true))
    if n_labels > 8:
        # For many labels, use greedy matching instead
        best_acc = 0
        remaining_true = list(range(n_labels))
        mapping = {}
        for pred_label in unique_pred:
            best_match_acc = 0
            best_match = 0
            for true_label in remaining_true:
                match_acc = np.mean((y_pred_filtered == pred_label) &
                                   (y_true_filtered == true_label))
                if match_acc > best_match_acc:
                    best_match_acc = match_acc
                    best_match = true_label
            mapping[pred_label] = best_match
            if best_match in remaining_true:
                remaining_true.remove(best_match)

        y_mapped = np.array([mapping.get(y, -1) for y in y_pred_filtered])
        return np.mean(y_mapped == y_true_filtered)

    best_acc = 0
    for perm in permutations(range(n_labels)):
        y_mapped = np.array([perm[y] if 0 <= y < len(perm) else -1
                           for y in y_pred_filtered])
        acc = np.mean(y_mapped == y_true_filtered)
        best_acc = max(best_acc, acc)

    return best_acc


def silhouette_score(X, labels):
    """Compute silhouette score, ignoring noise points."""
    # Filter out noise
    mask = labels != -1
    if np.sum(mask) < 2:
        return 0.0

    X_filtered = X[mask]
    labels_filtered = labels[mask]

    n_samples = X_filtered.shape[0]
    unique_labels = np.unique(labels_filtered)

    if len(unique_labels) < 2:
        return 0.0

    silhouettes = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster = labels_filtered == labels_filtered[i]
        same_cluster[i] = False
        if np.sum(same_cluster) > 0:
            a = np.mean(np.sqrt(np.sum((X_filtered[same_cluster] - X_filtered[i])**2, axis=1)))
        else:
            a = 0

        b = np.inf
        for k in unique_labels:
            if k != labels_filtered[i]:
                other_cluster = labels_filtered == k
                if np.sum(other_cluster) > 0:
                    mean_dist = np.mean(np.sqrt(np.sum((X_filtered[other_cluster] - X_filtered[i])**2, axis=1)))
                    b = min(b, mean_dist)

        if b == np.inf:
            b = 0

        silhouettes[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return np.mean(silhouettes)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X, y_true = make_clustered(n_samples=300, n_clusters=4)

    # -------- Experiment 1: Effect of Epsilon --------
    print("\n1. EFFECT OF EPSILON (ε)")
    print("-" * 40)
    print("Too small → everything is noise. Too large → one big cluster.")

    for eps in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(X)
        n_clusters = dbscan.n_clusters_
        n_noise = np.sum(dbscan.labels_ == -1)
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={n_noise}")
    print("-> Small eps: many small clusters + lots of noise")
    print("-> Large eps: few large clusters, less noise")
    print("-> Need to find the sweet spot!")

    # -------- Experiment 2: Effect of MinPts --------
    print("\n2. EFFECT OF MIN_SAMPLES (MinPts)")
    print("-" * 40)

    for min_samples in [2, 3, 5, 10, 15, 20]:
        dbscan = DBSCAN(eps=0.8, min_samples=min_samples)
        dbscan.fit(X)
        n_clusters = dbscan.n_clusters_
        n_noise = np.sum(dbscan.labels_ == -1)
        n_core = len(dbscan.core_sample_indices_)
        print(f"  min_samples={min_samples:<3}  clusters={n_clusters}  "
              f"noise={n_noise}  core_points={n_core}")
    print("-> Higher MinPts = stricter definition of 'dense'")
    print("-> Fewer core points, more noise")

    # -------- Experiment 3: Non-Convex Shapes (DBSCAN's Strength) --------
    print("\n3. DBSCAN ON NON-CONVEX SHAPES")
    print("-" * 40)
    print("Where K-means fails, DBSCAN shines!")

    # Moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.08)
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X_moons)
    acc = clustering_accuracy(y_moons, dbscan.labels_)
    n_noise = np.sum(dbscan.labels_ == -1)
    print(f"  Two Moons:  clusters={dbscan.n_clusters_}  acc={acc:.3f}  noise={n_noise}")

    # Circles
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05)
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X_circles)
    acc = clustering_accuracy(y_circles, dbscan.labels_)
    n_noise = np.sum(dbscan.labels_ == -1)
    print(f"  Circles:    clusters={dbscan.n_clusters_}  acc={acc:.3f}  noise={n_noise}")

    # Spiral
    X_spiral, y_spiral = make_spiral(n_samples=400, noise=0.3)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_spiral)
    acc = clustering_accuracy(y_spiral, dbscan.labels_)
    n_noise = np.sum(dbscan.labels_ == -1)
    print(f"  Spiral:     clusters={dbscan.n_clusters_}  acc={acc:.3f}  noise={n_noise}")

    print("-> DBSCAN finds non-convex shapes that K-means cannot!")

    # -------- Experiment 4: Noise Detection --------
    print("\n4. NOISE/OUTLIER DETECTION")
    print("-" * 40)

    # Add outliers to clustered data
    X_outliers = np.vstack([X, np.random.uniform(-10, 10, (20, 2))])
    y_outliers = np.concatenate([y_true, np.full(20, -1)])  # Mark as outliers

    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan.fit(X_outliers)

    # Check how many true outliers were detected as noise
    true_outliers = y_outliers == -1
    detected_noise = dbscan.labels_ == -1

    outliers_correct = np.sum(true_outliers & detected_noise)
    outliers_total = np.sum(true_outliers)
    false_noise = np.sum(~true_outliers & detected_noise)

    print(f"  True outliers detected as noise: {outliers_correct}/{outliers_total}")
    print(f"  Non-outliers marked as noise: {false_noise}")
    print("-> DBSCAN naturally detects outliers!")

    # -------- Experiment 5: Varying Density (DBSCAN's Weakness) --------
    print("\n5. VARYING DENSITY (DBSCAN's Weakness)")
    print("-" * 40)

    # Create clusters with different densities
    np.random.seed(42)
    X_dense = np.random.randn(100, 2) * 0.3 + [0, 0]
    X_sparse = np.random.randn(100, 2) * 1.5 + [5, 0]
    X_varying = np.vstack([X_dense, X_sparse])
    y_varying = np.array([0]*100 + [1]*100)

    for eps in [0.3, 0.5, 0.8, 1.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(X_varying)
        n_clusters = dbscan.n_clusters_
        n_noise = np.sum(dbscan.labels_ == -1)
        acc = clustering_accuracy(y_varying, dbscan.labels_)
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={n_noise}  acc={acc:.3f}")

    print("-> No single eps works for both clusters!")
    print("-> Dense cluster needs small eps, sparse needs large eps")
    print("-> Consider OPTICS or HDBSCAN for varying density")


def visualize_dbscan():
    """Visualize DBSCAN clustering."""
    print("\n" + "="*60)
    print("DBSCAN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Effect of epsilon
    np.random.seed(42)
    X, y_true = make_clustered(n_samples=300, n_clusters=3)

    eps_values = [0.3, 0.5, 0.8, 1.2]
    for i, eps in enumerate(eps_values):
        ax = fig.add_subplot(3, 4, i+1)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(X)

        # Color by cluster, noise in gray
        colors = dbscan.labels_.copy().astype(float)
        colors[colors == -1] = -0.5  # Noise gets special color

        scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis',
                           alpha=0.7, s=30)

        # Mark core points
        if len(dbscan.core_sample_indices_) > 0:
            ax.scatter(X[dbscan.core_sample_indices_, 0],
                      X[dbscan.core_sample_indices_, 1],
                      facecolors='none', edgecolors='red', s=80, linewidth=1)

        n_noise = np.sum(dbscan.labels_ == -1)
        ax.set_title(f'eps={eps}, clusters={dbscan.n_clusters_}\nnoise={n_noise}')
        ax.set_aspect('equal')

    # Row 2: DBSCAN on non-convex shapes
    datasets = [
        ("Two Moons", make_moons(n_samples=300, noise=0.08), 0.2),
        ("Circles", make_circles(n_samples=300, noise=0.05), 0.2),
        ("Spiral", make_spiral(n_samples=400, noise=0.3), 0.5),
        ("Clustered", (X, y_true), 0.8),
    ]

    for i, (name, (X_data, y_data), eps) in enumerate(datasets):
        ax = fig.add_subplot(3, 4, 5+i)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(X_data)

        colors = dbscan.labels_.copy().astype(float)
        colors[colors == -1] = -0.5

        ax.scatter(X_data[:, 0], X_data[:, 1], c=colors, cmap='viridis',
                  alpha=0.7, s=30)

        acc = clustering_accuracy(y_data, dbscan.labels_)
        ax.set_title(f'{name}\nacc={acc:.2f}, clusters={dbscan.n_clusters_}')
        ax.set_aspect('equal')

    # Row 3: K-distance plot and point types
    ax = fig.add_subplot(3, 4, 9)
    k_dist = k_distance_plot(X, k=5)
    ax.plot(range(len(k_dist)), k_dist, 'b-', linewidth=2)
    ax.axhline(y=0.8, color='r', linestyle='--', label='eps=0.8')
    ax.set_xlabel('Points (sorted)')
    ax.set_ylabel('5-th NN Distance')
    ax.set_title('K-Distance Plot\n(elbow suggests eps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show point types
    ax = fig.add_subplot(3, 4, 10)
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan.fit(X)

    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    noise_mask = dbscan.labels_ == -1
    border_mask = ~core_mask & ~noise_mask

    ax.scatter(X[core_mask, 0], X[core_mask, 1], c='green', s=50,
              label=f'Core ({np.sum(core_mask)})', alpha=0.7)
    ax.scatter(X[border_mask, 0], X[border_mask, 1], c='yellow', s=50,
              label=f'Border ({np.sum(border_mask)})', alpha=0.7, edgecolors='black')
    ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', s=50,
              label=f'Noise ({np.sum(noise_mask)})', alpha=0.7, marker='x')
    ax.set_title('Point Types')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # Varying density problem
    ax = fig.add_subplot(3, 4, 11)
    np.random.seed(42)
    X_dense = np.random.randn(100, 2) * 0.3 + [0, 0]
    X_sparse = np.random.randn(100, 2) * 1.5 + [5, 0]
    X_varying = np.vstack([X_dense, X_sparse])
    y_varying = np.array([0]*100 + [1]*100)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_varying)

    colors = dbscan.labels_.copy().astype(float)
    colors[colors == -1] = -0.5
    ax.scatter(X_varying[:, 0], X_varying[:, 1], c=colors, cmap='viridis',
              alpha=0.7, s=30)
    ax.set_title(f'Varying Density Problem\nclusters={dbscan.n_clusters_}')
    ax.set_aspect('equal')

    # Noise detection
    ax = fig.add_subplot(3, 4, 12)
    X_with_outliers = np.vstack([X, np.random.uniform(-5, 5, (15, 2))])

    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan.fit(X_with_outliers)

    colors = dbscan.labels_.copy().astype(float)
    colors[colors == -1] = -0.5

    ax.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=colors,
              cmap='viridis', alpha=0.7, s=30)
    ax.scatter(X_with_outliers[dbscan.labels_ == -1, 0],
              X_with_outliers[dbscan.labels_ == -1, 1],
              facecolors='none', edgecolors='red', s=100, linewidth=2)

    n_noise = np.sum(dbscan.labels_ == -1)
    ax.set_title(f'Outlier Detection\n{n_noise} noise points (red circles)')
    ax.set_aspect('equal')

    plt.suptitle('DBSCAN: Density-Based Spatial Clustering\n'
                 'Row 1: Effect of eps | Row 2: Non-convex shapes | Row 3: Analysis',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_dbscan_algorithm():
    """Visualize how DBSCAN works step by step."""
    print("\n" + "="*60)
    print("DBSCAN ALGORITHM VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    X, _ = make_clustered(n_samples=100, n_clusters=3)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Compute neighborhoods
    eps = 0.8
    min_samples = 5

    X_sq = np.sum(X**2, axis=1)
    D = np.sqrt(np.maximum(X_sq[:, np.newaxis] + X_sq - 2 * X @ X.T, 0))

    # Classify points
    neighbors = [np.where(D[i] <= eps)[0] for i in range(len(X))]
    core_mask = np.array([len(n) >= min_samples for n in neighbors])
    border_mask = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if not core_mask[i]:
            # Check if border (within eps of any core)
            for j in neighbors[i]:
                if core_mask[j]:
                    border_mask[i] = True
                    break
    noise_mask = ~core_mask & ~border_mask

    # Step visualizations
    steps = [
        "1. Data Points",
        "2. ε-neighborhoods",
        "3. Core Points",
        "4. Border & Noise",
        "5. Start Cluster 1",
        "6. Expand Cluster 1",
        "7. Cluster 2",
        "8. Final Clusters"
    ]

    # Run actual DBSCAN for final result
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    for idx, (ax, step_name) in enumerate(zip(axes.flat, steps)):
        ax.set_title(step_name, fontsize=10)

        if idx == 0:
            # Raw data
            ax.scatter(X[:, 0], X[:, 1], c='gray', s=30, alpha=0.7)

        elif idx == 1:
            # Show neighborhoods for a few points
            ax.scatter(X[:, 0], X[:, 1], c='gray', s=30, alpha=0.5)
            sample_points = [10, 50, 80]
            colors = ['red', 'green', 'blue']
            for p, c in zip(sample_points, colors):
                circle = plt.Circle((X[p, 0], X[p, 1]), eps,
                                   fill=False, color=c, linewidth=2)
                ax.add_patch(circle)
                ax.scatter(X[p, 0], X[p, 1], c=c, s=100, zorder=5)

        elif idx == 2:
            # Core points
            ax.scatter(X[~core_mask, 0], X[~core_mask, 1], c='gray', s=30, alpha=0.5)
            ax.scatter(X[core_mask, 0], X[core_mask, 1], c='green', s=50,
                      label=f'Core ({np.sum(core_mask)})')
            ax.legend(fontsize=8)

        elif idx == 3:
            # Core, border, noise
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', s=30,
                      marker='x', label=f'Noise ({np.sum(noise_mask)})')
            ax.scatter(X[border_mask, 0], X[border_mask, 1], c='orange', s=30,
                      label=f'Border ({np.sum(border_mask)})')
            ax.scatter(X[core_mask, 0], X[core_mask, 1], c='green', s=50,
                      label=f'Core ({np.sum(core_mask)})')
            ax.legend(fontsize=8)

        elif idx == 4:
            # Start first cluster
            labels = dbscan.labels_.copy()
            cluster0 = labels == 0
            ax.scatter(X[~cluster0, 0], X[~cluster0, 1], c='gray', s=30, alpha=0.3)
            ax.scatter(X[cluster0, 0], X[cluster0, 1], c='blue', s=50)

        elif idx == 5:
            # Show expansion arrows
            labels = dbscan.labels_.copy()
            cluster0 = labels == 0
            ax.scatter(X[~cluster0, 0], X[~cluster0, 1], c='gray', s=30, alpha=0.3)
            ax.scatter(X[cluster0, 0], X[cluster0, 1], c='blue', s=50)
            # Draw some expansion arrows
            cluster0_core = cluster0 & core_mask
            core_indices = np.where(cluster0_core)[0]
            if len(core_indices) > 1:
                for i in range(min(5, len(core_indices)-1)):
                    ax.annotate('', xy=X[core_indices[i+1]],
                               xytext=X[core_indices[i]],
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))

        elif idx == 6:
            # Two clusters
            labels = dbscan.labels_.copy()
            colors = labels.copy().astype(float)
            colors[colors == -1] = -0.5
            mask01 = (labels == 0) | (labels == 1)
            ax.scatter(X[~mask01, 0], X[~mask01, 1], c='gray', s=30, alpha=0.3)
            ax.scatter(X[mask01, 0], X[mask01, 1], c=labels[mask01], cmap='tab10', s=50)

        else:
            # Final result
            colors = dbscan.labels_.copy().astype(float)
            colors[colors == -1] = -0.5
            ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', s=50)
            ax.scatter(X[dbscan.labels_ == -1, 0], X[dbscan.labels_ == -1, 1],
                      facecolors='none', edgecolors='red', s=80, linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    plt.suptitle(f'DBSCAN Algorithm: eps={eps}, min_samples={min_samples}\n'
                 'Core points define dense regions, clusters grow by connectivity',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_dbscan():
    """Benchmark DBSCAN on various datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: DBSCAN Clustering")
    print("="*60)

    results = {}

    # Test on non-convex shapes (DBSCAN's strength)
    datasets = [
        ("moons", make_moons(n_samples=300, noise=0.08), 0.2, 5),
        ("circles", make_circles(n_samples=300, noise=0.05), 0.2, 5),
        ("spiral", make_spiral(n_samples=400, noise=0.3), 0.5, 5),
        ("clustered", make_clustered(n_samples=300, n_clusters=4), 0.8, 5),
    ]

    for name, (X, y_true), eps, min_samples in datasets:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)

        acc = clustering_accuracy(y_true, dbscan.labels_)
        n_noise = np.sum(dbscan.labels_ == -1)
        noise_frac = n_noise / len(X)

        results[name] = {
            'accuracy': acc,
            'n_clusters': dbscan.n_clusters_,
            'noise_fraction': noise_frac
        }
        print(f"{name:<12}: acc={acc:.3f}, clusters={dbscan.n_clusters_}, "
              f"noise={noise_frac:.1%}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("DBSCAN — Paradigm: DENSITY-BASED CLUSTERING")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Clusters are DENSE REGIONS separated by SPARSE REGIONS.

    Two parameters:
    - eps (ε): Neighborhood radius
    - min_samples: Minimum points to be "dense"

    Point types:
    - CORE: ≥ min_samples neighbors within ε
    - BORDER: Within ε of core, but not core itself
    - NOISE: Neither core nor border → OUTLIER

THE ALGORITHM:
    1. Find core points
    2. Grow clusters by connecting core points within ε
    3. Add border points to nearest cluster
    4. Everything else is noise

KEY ADVANTAGES:
    - No K required — clusters emerge from data
    - Finds ARBITRARY SHAPES (non-convex!)
    - Natural OUTLIER DETECTION

INDUCTIVE BIAS:
    - Clusters are dense regions
    - ε is global (struggles with varying density)
    - Assumes density is uniform within clusters

PARAMETER SELECTION:
    - min_samples ≈ 2 × dimensions
    - eps: use k-distance plot (look for elbow)
    """)

    ablation_experiments()
    results = benchmark_dbscan()

    print("\nGenerating visualizations...")

    fig1 = visualize_dbscan()
    save_path1 = '/Users/sid47/ML Algorithms/57_dbscan.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_dbscan_algorithm()
    save_path2 = '/Users/sid47/ML Algorithms/57_dbscan_algorithm.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "="*60)
    print("SUMMARY: What DBSCAN Reveals")
    print("="*60)
    print("""
1. DENSITY-BASED: Clusters are where points are packed together
2. NO K REQUIRED: Number of clusters emerges from ε and min_samples
3. ARBITRARY SHAPES: Can find moons, spirals, rings (K-means cannot)
4. OUTLIER DETECTION: Noise points identified naturally
5. VARYING DENSITY: Main weakness — global ε can't handle it

WHEN TO USE:
    - Unknown number of clusters
    - Non-convex cluster shapes
    - Need outlier detection
    - Clusters have similar densities

WHEN TO AVOID:
    - Varying density clusters (use HDBSCAN/OPTICS)
    - High-dimensional data (density becomes ill-defined)
    - Very large datasets (O(n²) without index)

COMPARISON:
    K-Means: Spherical clusters, needs K, no outliers
    Hierarchical: Any K, but expensive, greedy merges
    DBSCAN: Arbitrary shapes, auto K, handles outliers
    Spectral: Non-convex, uses graph, needs K

NEXT: Spectral Clustering — uses graph Laplacian, bridges to GNNs
    """)
