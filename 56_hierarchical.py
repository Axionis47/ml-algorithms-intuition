"""
HIERARCHICAL CLUSTERING — Paradigm: DENDROGRAM (Tree of Merges)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Build a HIERARCHY of clusters, from individual points to one big cluster.

TWO APPROACHES:

AGGLOMERATIVE (Bottom-Up) — Most common:
    1. Start: each point is its own cluster
    2. Merge: combine the two "closest" clusters
    3. Repeat until one cluster remains
    4. Result: a DENDROGRAM (tree showing merge history)

DIVISIVE (Top-Down) — Less common:
    1. Start: all points in one cluster
    2. Split: divide clusters recursively
    3. Result: same dendrogram, built differently

The dendrogram lets you choose ANY number of clusters by
cutting at different heights!

===============================================================
THE KEY INSIGHT: LINKAGE CRITERIA
===============================================================

"Distance between clusters" is ambiguous for multi-point clusters.
LINKAGE defines how to measure cluster-to-cluster distance:

SINGLE LINKAGE (Minimum):
    d(A, B) = min { d(a, b) : a ∈ A, b ∈ B }
    "Distance to nearest point in other cluster"
    → Can find non-convex clusters (chaining)
    → Sensitive to noise (single point can bridge clusters)

COMPLETE LINKAGE (Maximum):
    d(A, B) = max { d(a, b) : a ∈ A, b ∈ B }
    "Distance to farthest point in other cluster"
    → Produces compact clusters
    → Sensitive to outliers

AVERAGE LINKAGE (UPGMA):
    d(A, B) = (1/|A||B|) Σ_{a,b} d(a, b)
    "Average of all pairwise distances"
    → Compromise between single and complete
    → Most commonly used

WARD'S METHOD:
    Minimize increase in total within-cluster variance
    d(A, B) = Δ variance when merging A and B
    → Produces spherical, similarly-sized clusters
    → Most similar to K-means

===============================================================
DENDROGRAM: Reading the Tree
===============================================================

    Height
       |
       |      ___________
       |     |           |
       |    _|_        __|__
       |   |   |      |     |
       |   1   2      3    _|_
       |                  |   |
       |                  4   5
       |_____________________|_________

- Y-axis: Distance at which clusters merge
- Cut horizontally → get clusters at that granularity
- Tall vertical lines = well-separated clusters
- Short vertical lines = clusters are close (uncertain split)

===============================================================
INDUCTIVE BIAS
===============================================================

1. HIERARCHY EXISTS: Assumes data has nested cluster structure
   (may not be true for all data)

2. GREEDY: Merges are permanent — can't undo a bad merge
   (unlike K-means which can reassign)

3. LINKAGE-DEPENDENT:
   - Single: elongated, non-convex OK
   - Complete/Ward: spherical, compact

4. NO CLUSTER CENTERS: Unlike K-means, no representative points
   (centroids emerge only if you compute them post-hoc)

===============================================================
TIME & SPACE COMPLEXITY
===============================================================

Naive: O(n³) time, O(n²) space (distance matrix)
Optimized: O(n² log n) time possible for some linkages

For large datasets, hierarchical clustering is expensive!
Consider: Mini-batch K-means → hierarchical on centroids

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
make_moons = datasets_module.make_moons
make_circles = datasets_module.make_circles


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering.

    Paradigm: DENDROGRAM — build a tree of cluster merges
    - No need to specify K beforehand
    - Cut dendrogram at any level to get clusters
    - Different linkages → different cluster shapes
    """

    def __init__(self, n_clusters=None, linkage='ward', distance_threshold=None):
        """
        Parameters:
        -----------
        n_clusters : int or None
            Number of clusters to find. If None, must specify distance_threshold.
        linkage : str
            'single', 'complete', 'average', or 'ward'
        distance_threshold : float or None
            Cut dendrogram at this height. If None, use n_clusters.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

        # Attributes set after fit
        self.labels_ = None
        self.n_leaves_ = None
        self.children_ = None      # Merge history: (i, j) merged at step k
        self.distances_ = None     # Distance at each merge
        self.dendrogram_data_ = None

    def _compute_distance_matrix(self, X):
        """Compute pairwise Euclidean distance matrix."""
        n = X.shape[0]
        # ||a - b||² = ||a||² + ||b||² - 2a·b
        X_sq = np.sum(X**2, axis=1)
        D_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * X @ X.T
        return np.sqrt(np.maximum(D_sq, 0))

    def _linkage_distance(self, D, cluster_i, cluster_j, sizes):
        """
        Compute linkage distance between two clusters.

        D: current distance matrix (between cluster representatives)
        cluster_i, cluster_j: cluster indices
        sizes: dict mapping cluster index to size
        """
        # For efficiency, we update distances incrementally
        # This method is called when we need fresh computation
        pass  # Implemented inline in fit() for clarity

    def fit(self, X):
        """
        Fit hierarchical clustering using agglomerative algorithm.

        Algorithm:
        1. Initialize each point as a cluster
        2. Find two closest clusters (by linkage)
        3. Merge them, record the merge
        4. Update distance matrix
        5. Repeat until one cluster or threshold reached
        """
        n_samples = X.shape[0]
        self.n_leaves_ = n_samples

        # Initialize: each point is a cluster
        # We'll track active clusters and their members
        # cluster_id → set of original point indices
        clusters = {i: {i} for i in range(n_samples)}
        active = set(range(n_samples))

        # Compute initial distance matrix
        D = self._compute_distance_matrix(X)

        # For Ward's method, we also need cluster means
        if self.linkage == 'ward':
            centroids = {i: X[i].copy() for i in range(n_samples)}

        # Storage for dendrogram
        self.children_ = []
        self.distances_ = []

        # Keep merging until done
        next_cluster_id = n_samples  # New clusters get IDs >= n_samples

        while len(active) > 1:
            # Find the two closest clusters
            min_dist = np.inf
            merge_i, merge_j = None, None

            active_list = sorted(active)
            for idx_i, i in enumerate(active_list):
                for j in active_list[idx_i + 1:]:
                    # Compute linkage distance
                    members_i = clusters[i]
                    members_j = clusters[j]

                    if self.linkage == 'single':
                        # Minimum distance between any pair
                        dist = min(D[a, b] for a in members_i for b in members_j)

                    elif self.linkage == 'complete':
                        # Maximum distance between any pair
                        dist = max(D[a, b] for a in members_i for b in members_j)

                    elif self.linkage == 'average':
                        # Average of all pairwise distances
                        total = sum(D[a, b] for a in members_i for b in members_j)
                        dist = total / (len(members_i) * len(members_j))

                    elif self.linkage == 'ward':
                        # Increase in variance from merging
                        # Ward distance = sqrt(2 * n_i * n_j / (n_i + n_j) * ||c_i - c_j||²)
                        n_i, n_j = len(members_i), len(members_j)
                        c_i, c_j = centroids[i], centroids[j]
                        dist = np.sqrt(2 * n_i * n_j / (n_i + n_j) * np.sum((c_i - c_j)**2))

                    else:
                        raise ValueError(f"Unknown linkage: {self.linkage}")

                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Record the merge
            self.children_.append((merge_i, merge_j))
            self.distances_.append(min_dist)

            # Create new cluster
            new_members = clusters[merge_i] | clusters[merge_j]
            clusters[next_cluster_id] = new_members

            # Update centroid for Ward
            if self.linkage == 'ward':
                n_i = len(clusters[merge_i])
                n_j = len(clusters[merge_j])
                centroids[next_cluster_id] = (n_i * centroids[merge_i] + n_j * centroids[merge_j]) / (n_i + n_j)

            # Remove old clusters, add new one
            active.remove(merge_i)
            active.remove(merge_j)
            active.add(next_cluster_id)

            next_cluster_id += 1

            # Check stopping condition
            if self.distance_threshold is not None and min_dist > self.distance_threshold:
                # Stop before this merge (don't include it)
                self.children_.pop()
                self.distances_.pop()
                active.remove(next_cluster_id - 1)
                active.add(merge_i)
                active.add(merge_j)
                break

            if self.n_clusters is not None and len(active) == self.n_clusters:
                break

        # Extract cluster labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, cluster_id in enumerate(active):
            for point_idx in clusters[cluster_id]:
                self.labels_[point_idx] = label

        # Store dendrogram data for visualization
        self._build_dendrogram_data(n_samples)

        return self

    def _build_dendrogram_data(self, n_samples):
        """Build data structure for dendrogram plotting."""
        # scipy-compatible format: each row is [idx1, idx2, dist, n_points]
        self.dendrogram_data_ = []

        # Track cluster sizes
        sizes = {i: 1 for i in range(n_samples)}

        for (i, j), dist in zip(self.children_, self.distances_):
            new_id = n_samples + len(self.dendrogram_data_)
            size = sizes.get(i, 1) + sizes.get(j, 1)
            sizes[new_id] = size
            self.dendrogram_data_.append([i, j, dist, size])

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def plot_dendrogram(hc, ax=None, truncate_mode=None, p=30, leaf_rotation=90,
                    leaf_font_size=8, show_labels=True):
    """
    Plot dendrogram from hierarchical clustering results.

    Simple implementation without scipy.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if hc.dendrogram_data_ is None or len(hc.dendrogram_data_) == 0:
        ax.text(0.5, 0.5, 'No dendrogram data', ha='center', va='center')
        return ax

    n_leaves = hc.n_leaves_
    data = np.array(hc.dendrogram_data_)

    # Compute leaf positions (x-coordinate for each original point)
    # We'll place leaves based on merge order
    leaf_positions = {}

    # Start with leaves at positions 0, 1, 2, ...
    # We need to figure out leaf order from the merges

    # Simple approach: place leaves in order, clusters at mean of their leaves
    def get_x_position(node_id, positions):
        if node_id < n_leaves:
            if node_id not in positions:
                positions[node_id] = len([k for k in positions if k < n_leaves])
            return positions[node_id]
        else:
            # Internal node: get children
            merge_idx = node_id - n_leaves
            if merge_idx >= len(data):
                return 0
            left, right = int(data[merge_idx, 0]), int(data[merge_idx, 1])
            left_x = get_x_position(left, positions)
            right_x = get_x_position(right, positions)
            return (left_x + right_x) / 2

    # Build positions
    positions = {}
    for i in range(n_leaves):
        positions[i] = i

    # Reorder based on dendrogram structure for better visualization
    def get_leaves(node_id):
        if node_id < n_leaves:
            return [node_id]
        merge_idx = node_id - n_leaves
        if merge_idx >= len(data):
            return []
        left, right = int(data[merge_idx, 0]), int(data[merge_idx, 1])
        return get_leaves(left) + get_leaves(right)

    # Get leaf order from root
    if len(data) > 0:
        root_id = n_leaves + len(data) - 1
        leaf_order = get_leaves(root_id)
        positions = {leaf: i for i, leaf in enumerate(leaf_order)}

    # Draw the dendrogram
    def draw_node(node_id, positions, ax):
        if node_id < n_leaves:
            # Leaf node
            x = positions.get(node_id, node_id)
            return x, 0

        merge_idx = node_id - n_leaves
        if merge_idx >= len(data):
            return 0, 0

        left, right = int(data[merge_idx, 0]), int(data[merge_idx, 1])
        height = data[merge_idx, 2]

        # Recursively draw children
        left_x, left_y = draw_node(left, positions, ax)
        right_x, right_y = draw_node(right, positions, ax)

        # Draw horizontal line connecting children
        ax.plot([left_x, left_x], [left_y, height], 'b-', linewidth=1)
        ax.plot([right_x, right_x], [right_y, height], 'b-', linewidth=1)
        ax.plot([left_x, right_x], [height, height], 'b-', linewidth=1)

        return (left_x + right_x) / 2, height

    if len(data) > 0:
        root_id = n_leaves + len(data) - 1
        draw_node(root_id, positions, ax)

    # Draw leaf labels
    if show_labels and n_leaves <= 50:
        for leaf, x in positions.items():
            ax.text(x, -0.02 * max(data[:, 2]), str(leaf),
                   ha='center', va='top', fontsize=leaf_font_size, rotation=leaf_rotation)

    ax.set_ylabel('Distance')
    ax.set_xlabel('Sample Index')
    ax.set_xlim(-0.5, n_leaves - 0.5)

    return ax


def clustering_accuracy(y_true, y_pred, n_clusters):
    """Compute clustering accuracy with optimal label permutation."""
    best_acc = 0
    for perm in permutations(range(n_clusters)):
        y_mapped = np.array([perm[y] if y < len(perm) else -1 for y in y_pred])
        acc = np.mean(y_mapped == y_true)
        best_acc = max(best_acc, acc)
    return best_acc


def silhouette_score(X, labels):
    """Compute silhouette score."""
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0

    silhouettes = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        if np.sum(same_cluster) > 0:
            a = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i])**2, axis=1)))
        else:
            a = 0

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


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X, y_true = make_clustered(n_samples=300, n_clusters=4)

    # -------- Experiment 1: Linkage Methods --------
    print("\n1. EFFECT OF LINKAGE METHOD")
    print("-" * 40)
    print("Same data, different linkage → different clusters:")

    for linkage in ['single', 'complete', 'average', 'ward']:
        hc = HierarchicalClustering(n_clusters=4, linkage=linkage)
        hc.fit(X)
        acc = clustering_accuracy(y_true, hc.labels_, 4)
        sil = silhouette_score(X, hc.labels_)
        print(f"  {linkage:<10}  accuracy={acc:.3f}  silhouette={sil:.3f}")
    print("-> Ward typically best for spherical clusters (like K-means)")
    print("-> Single can find elongated clusters but prone to chaining")

    # -------- Experiment 2: Linkage on Non-Convex Shapes --------
    print("\n2. LINKAGE ON NON-CONVEX SHAPES")
    print("-" * 40)

    X_moons, y_moons = make_moons(n_samples=200, noise=0.1)

    for linkage in ['single', 'complete', 'average', 'ward']:
        hc = HierarchicalClustering(n_clusters=2, linkage=linkage)
        hc.fit(X_moons)
        acc = clustering_accuracy(y_moons, hc.labels_, 2)
        print(f"  Moons + {linkage:<10}  accuracy={acc:.3f}")
    print("-> Single linkage can find non-convex shapes!")
    print("-> Complete/Ward fail (assume convex clusters)")

    # -------- Experiment 3: Number of Clusters --------
    print("\n3. EFFECT OF NUMBER OF CLUSTERS")
    print("-" * 40)
    print("Cutting dendrogram at different levels:")

    for n_clusters in [2, 3, 4, 5, 6]:
        hc = HierarchicalClustering(n_clusters=n_clusters, linkage='ward')
        hc.fit(X)
        sil = silhouette_score(X, hc.labels_)
        print(f"  K={n_clusters}  silhouette={sil:.3f}")
    print("-> Same dendrogram, different cuts → different K")
    print("-> No need to recompute! (advantage over K-means)")

    # -------- Experiment 4: Distance Threshold vs N_Clusters --------
    print("\n4. DISTANCE THRESHOLD STOPPING")
    print("-" * 40)

    hc = HierarchicalClustering(n_clusters=4, linkage='ward')
    hc.fit(X)

    if len(hc.distances_) > 0:
        print("Merge distances in dendrogram:")
        for i, d in enumerate(hc.distances_[-10:]):  # Last 10 merges
            print(f"  Merge {len(hc.distances_)-10+i+1}: distance={d:.3f}")

        # Use a threshold
        threshold = np.median(hc.distances_)
        hc_thresh = HierarchicalClustering(distance_threshold=threshold, linkage='ward')
        hc_thresh.fit(X)
        n_found = len(np.unique(hc_thresh.labels_))
        print(f"\nThreshold={threshold:.2f} → {n_found} clusters found")
    print("-> Can stop based on distance instead of K")

    # -------- Experiment 5: Sensitivity to Outliers --------
    print("\n5. SENSITIVITY TO OUTLIERS")
    print("-" * 40)

    # Add outliers
    X_outliers = np.vstack([X, [[10, 10], [10, -10], [-10, 10]]])
    y_outliers = np.concatenate([y_true, [4, 4, 4]])  # Outlier class

    for linkage in ['single', 'complete', 'ward']:
        hc = HierarchicalClustering(n_clusters=4, linkage=linkage)
        hc.fit(X_outliers)
        # Check if outliers are in their own cluster or absorbed
        outlier_labels = hc.labels_[-3:]
        unique_outlier = len(np.unique(outlier_labels))
        print(f"  {linkage:<10}  outlier labels: {outlier_labels}  (unique={unique_outlier})")
    print("-> Single: outliers may be chained to main clusters")
    print("-> Complete: outliers more likely separate")


def visualize_hierarchical():
    """Visualize hierarchical clustering."""
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING VISUALIZATION")
    print("="*60)

    np.random.seed(42)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Different linkages on same data
    X, y_true = make_clustered(n_samples=150, n_clusters=3)

    linkages = ['single', 'complete', 'average', 'ward']
    for i, linkage in enumerate(linkages):
        ax = fig.add_subplot(3, 4, i+1)

        hc = HierarchicalClustering(n_clusters=3, linkage=linkage)
        hc.fit(X)

        scatter = ax.scatter(X[:, 0], X[:, 1], c=hc.labels_, cmap='viridis',
                           alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

        acc = clustering_accuracy(y_true, hc.labels_, 3)
        ax.set_title(f'{linkage.capitalize()}\nacc={acc:.2f}')
        ax.set_aspect('equal')

    # Row 2: Non-convex shapes (where single linkage shines)
    datasets = [
        ("Two Moons", make_moons(n_samples=200, noise=0.08)),
        ("Circles", make_circles(n_samples=200, noise=0.04)),
    ]

    for i, (name, (X_data, y_data)) in enumerate(datasets):
        # Single linkage
        ax = fig.add_subplot(3, 4, 5 + i*2)
        hc_single = HierarchicalClustering(n_clusters=2, linkage='single')
        hc_single.fit(X_data)
        ax.scatter(X_data[:, 0], X_data[:, 1], c=hc_single.labels_,
                  cmap='coolwarm', alpha=0.7, s=30)
        acc = clustering_accuracy(y_data, hc_single.labels_, 2)
        ax.set_title(f'{name} - Single\nacc={acc:.2f}')
        ax.set_aspect('equal')

        # Ward linkage
        ax = fig.add_subplot(3, 4, 6 + i*2)
        hc_ward = HierarchicalClustering(n_clusters=2, linkage='ward')
        hc_ward.fit(X_data)
        ax.scatter(X_data[:, 0], X_data[:, 1], c=hc_ward.labels_,
                  cmap='coolwarm', alpha=0.7, s=30)
        acc = clustering_accuracy(y_data, hc_ward.labels_, 2)
        ax.set_title(f'{name} - Ward\nacc={acc:.2f}')
        ax.set_aspect('equal')

    # Row 3: Dendrograms
    X_small, _ = make_clustered(n_samples=30, n_clusters=3)

    for i, linkage in enumerate(['single', 'ward']):
        ax = fig.add_subplot(3, 4, 9 + i*2)

        hc = HierarchicalClustering(n_clusters=3, linkage=linkage)
        hc.fit(X_small)

        plot_dendrogram(hc, ax=ax, show_labels=True, leaf_font_size=6)
        ax.set_title(f'Dendrogram ({linkage})')

        # Show clustering result
        ax = fig.add_subplot(3, 4, 10 + i*2)
        ax.scatter(X_small[:, 0], X_small[:, 1], c=hc.labels_,
                  cmap='viridis', s=50, edgecolors='black')
        for idx in range(len(X_small)):
            ax.annotate(str(idx), (X_small[idx, 0], X_small[idx, 1]),
                       fontsize=7, ha='center', va='bottom')
        ax.set_title(f'Clusters ({linkage})')
        ax.set_aspect('equal')

    plt.suptitle('HIERARCHICAL CLUSTERING\n'
                 'Row 1: Linkage comparison | Row 2: Non-convex shapes | Row 3: Dendrograms',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_dendrogram_cuts():
    """Visualize how cutting the dendrogram gives different numbers of clusters."""
    print("\n" + "="*60)
    print("DENDROGRAM CUTTING VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    X, y_true = make_clustered(n_samples=100, n_clusters=4)

    fig = plt.figure(figsize=(16, 6))

    # Build full dendrogram once
    hc_full = HierarchicalClustering(n_clusters=1, linkage='ward')
    hc_full.fit(X)

    # Plot dendrogram with cut lines
    ax1 = fig.add_subplot(1, 4, 1)
    plot_dendrogram(hc_full, ax=ax1, show_labels=False)

    # Draw horizontal cut lines at different heights
    if len(hc_full.distances_) > 0:
        cuts = [hc_full.distances_[-1] * 0.3,  # Many clusters
                hc_full.distances_[-1] * 0.5,  # Medium
                hc_full.distances_[-1] * 0.7]  # Few clusters
        colors = ['green', 'orange', 'red']

        for cut, color in zip(cuts, colors):
            ax1.axhline(y=cut, color=color, linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_title('Dendrogram with Cut Lines')

    # Show different clusterings
    for i, n_clusters in enumerate([6, 4, 2]):
        ax = fig.add_subplot(1, 4, i+2)

        hc = HierarchicalClustering(n_clusters=n_clusters, linkage='ward')
        hc.fit(X)

        scatter = ax.scatter(X[:, 0], X[:, 1], c=hc.labels_, cmap='tab10',
                           alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        ax.set_title(f'Cut at K={n_clusters}')
        ax.set_aspect('equal')

    plt.suptitle('CUTTING THE DENDROGRAM AT DIFFERENT HEIGHTS\n'
                 'Same hierarchy, different granularity — no recomputation needed!',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_hierarchical():
    """Benchmark hierarchical clustering."""
    print("\n" + "="*60)
    print("BENCHMARK: Hierarchical Clustering")
    print("="*60)

    results = {}

    # Test on spherical clusters
    for n_clusters in [3, 4, 5]:
        X, y_true = make_clustered(n_samples=300, n_clusters=n_clusters, random_state=42)

        for linkage in ['ward', 'average']:
            hc = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage)
            hc.fit(X)

            acc = clustering_accuracy(y_true, hc.labels_, n_clusters)
            sil = silhouette_score(X, hc.labels_)

            key = f'clustered_K{n_clusters}_{linkage}'
            results[key] = {'accuracy': acc, 'silhouette': sil}
            print(f"{key}: acc={acc:.3f}, sil={sil:.3f}")

    # Test on non-convex shapes
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1)

    for linkage in ['single', 'ward']:
        hc = HierarchicalClustering(n_clusters=2, linkage=linkage)
        hc.fit(X_moons)
        acc = clustering_accuracy(y_moons, hc.labels_, 2)
        print(f"Moons ({linkage}): acc={acc:.3f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("HIERARCHICAL CLUSTERING — Paradigm: DENDROGRAM")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Build a TREE of cluster merges from individual points to one big cluster.

    Agglomerative (bottom-up):
    1. Each point starts as its own cluster
    2. Repeatedly merge the two closest clusters
    3. Result: DENDROGRAM showing merge history

THE KEY INSIGHT (Linkage):
    "Distance between clusters" depends on LINKAGE:
    - Single:   min distance (can find non-convex, but chains)
    - Complete: max distance (compact clusters)
    - Average:  mean distance (compromise)
    - Ward:     minimize variance increase (spherical, like K-means)

THE DENDROGRAM:
    Cut horizontally → get K clusters at that granularity
    One hierarchy → any K without recomputing!

INDUCTIVE BIAS:
    - Hierarchy exists (nested cluster structure)
    - Greedy (merges are permanent)
    - Linkage determines cluster shape expectations

COMPLEXITY:
    O(n²) space (distance matrix)
    O(n³) time naive, O(n² log n) optimized
    """)

    ablation_experiments()
    results = benchmark_hierarchical()

    print("\nGenerating visualizations...")

    fig1 = visualize_hierarchical()
    save_path1 = '/Users/sid47/ML Algorithms/56_hierarchical.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_dendrogram_cuts()
    save_path2 = '/Users/sid47/ML Algorithms/56_hierarchical_dendrogram.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "="*60)
    print("SUMMARY: What Hierarchical Clustering Reveals")
    print("="*60)
    print("""
1. DENDROGRAM captures full hierarchy — cut at any level for K clusters
2. LINKAGE is crucial:
   - Ward → spherical (best for most cases)
   - Single → non-convex (but sensitive to noise)
3. NO K REQUIRED beforehand — explore different granularities
4. GREEDY — cannot undo bad merges
5. EXPENSIVE — O(n²) space, O(n³) time for large data

WHEN TO USE:
    - Don't know K beforehand
    - Want to explore cluster hierarchy
    - Data has nested structure
    - Small-medium datasets (< 10K points)

WHEN TO AVOID:
    - Large datasets (use K-means or mini-batch)
    - Need fast updates (hierarchical is batch-only)
    - Spherical clusters with known K (K-means is faster)

NEXT: DBSCAN — density-based clustering, finds arbitrary shapes, auto K
    """)
