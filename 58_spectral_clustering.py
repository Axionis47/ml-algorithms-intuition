"""
SPECTRAL CLUSTERING — Paradigm: GRAPH LAPLACIAN

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Transform the clustering problem into a GRAPH PARTITIONING problem:

1. Build a SIMILARITY GRAPH from data points
2. Compute the GRAPH LAPLACIAN
3. Find eigenvectors of the Laplacian
4. Cluster in the EIGENSPACE (using K-means)

Key insight: The graph Laplacian's eigenvectors encode cluster structure!
Cutting the graph optimally ≈ clustering in eigenspace.

===============================================================
THE SIMILARITY GRAPH
===============================================================

Given data X, construct a weighted graph where:
    - Nodes = data points
    - Edges = similarities between points

Common constructions:

1. ε-NEIGHBORHOOD GRAPH:
   Connect points if d(i,j) < ε
   Edges are unweighted (0 or 1)

2. K-NEAREST NEIGHBORS GRAPH:
   Connect each point to its k nearest neighbors
   Can be mutual (both directions) or not

3. FULLY CONNECTED (RBF/Gaussian):
   W_ij = exp(-||x_i - x_j||² / (2σ²))
   Every pair connected, weighted by similarity
   This is the most common choice

===============================================================
THE GRAPH LAPLACIAN
===============================================================

Given weight matrix W:
    D = diagonal matrix with D_ii = Σ_j W_ij  (degree matrix)
    L = D - W                                  (unnormalized Laplacian)

Properties of L:
    - Symmetric, positive semi-definite
    - Smallest eigenvalue is 0 (eigenvector = 1)
    - Number of 0 eigenvalues = number of connected components
    - EIGENGAP: Large gap after k-th eigenvalue suggests k clusters

NORMALIZED LAPLACIAN (often better):
    L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2)
    L_rw  = D^(-1) L = I - D^(-1) W

===============================================================
WHY EIGENVECTORS? (The Key Insight)
===============================================================

Graph cut problem: Find partition that minimizes edges cut.

MIN-CUT: Minimize total weight of cut edges
    → Trivial solution: put one point in its own cluster

NORMALIZED CUT (NCut): Minimize cut relative to cluster size
    NCut(A, B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)

Theorem (Shi & Malik, 2000):
    Relaxing the discrete optimization → eigenvalue problem!
    Second eigenvector of L_rw approximates optimal NCut

INTUITION:
    - First eigenvector: constant (trivial)
    - Second eigenvector: splits graph into two "natural" parts
    - k eigenvectors: k-way partition

This is why spectral methods find NON-CONVEX clusters!
K-means in eigenspace ≠ K-means in original space.

===============================================================
THE ALGORITHM
===============================================================

1. Construct similarity graph W (e.g., RBF kernel)
2. Compute normalized Laplacian L_sym or L_rw
3. Compute first k eigenvectors of L (smallest eigenvalues)
4. Form matrix U ∈ R^(n × k) from eigenvectors as columns
5. Normalize rows of U (for L_sym variant)
6. Cluster rows of U using K-means
7. Assign point i to cluster of row i

===============================================================
INDUCTIVE BIAS
===============================================================

1. GRAPH STRUCTURE: Assumes meaningful similarity graph exists
   (σ in RBF kernel is crucial!)

2. CLUSTER = CONNECTED COMPONENT-ISH: Good clusters are
   "nearly disconnected" subgraphs

3. NORMALIZED vs UNNORMALIZED:
   - Normalized: handles varying cluster sizes better
   - Unnormalized: simpler, but biased toward equal sizes

4. K STILL REQUIRED: Like K-means, need to specify number of clusters

5. EIGENGAP: Can help choose k (but not always clear)

===============================================================
CONNECTION TO GRAPH NEURAL NETWORKS
===============================================================

Spectral clustering is the ANCESTOR of spectral GNNs!

- Graph Laplacian eigenvectors = Fourier basis on graphs
- GCN uses: H' = σ(D^(-1/2) A D^(-1/2) H W)
  This is essentially the normalized adjacency ≈ I - L_sym
- Graph convolution = filtering in spectral domain

Spectral clustering → Spectral graph convolution → Modern GNNs

Understanding spectral clustering helps understand GCNs!

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
make_spiral = datasets_module.make_spiral


class SpectralClustering:
    """
    Spectral Clustering using Graph Laplacian.

    Paradigm: GRAPH LAPLACIAN
    - Transform clustering into graph partitioning
    - Eigenvectors encode cluster structure
    - Finds non-convex clusters via eigenspace
    """

    def __init__(self, n_clusters=2, affinity='rbf', gamma=1.0,
                 n_neighbors=10, assign_labels='kmeans',
                 laplacian='normalized', random_state=None):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to find
        affinity : str
            'rbf': Gaussian/RBF kernel (fully connected)
            'nearest_neighbors': k-NN graph
        gamma : float
            RBF kernel parameter: exp(-gamma * ||x-y||²)
            Smaller gamma = wider kernel = more connections
        n_neighbors : int
            Number of neighbors for k-NN affinity
        assign_labels : str
            'kmeans': cluster eigenvectors with K-means
            'discretize': direct discretization (faster but less accurate)
        laplacian : str
            'normalized': L_sym = I - D^(-1/2) W D^(-1/2)
            'unnormalized': L = D - W
        random_state : int or None
            Random seed for K-means
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.laplacian = laplacian
        self.random_state = random_state

        # Attributes set after fit
        self.labels_ = None
        self.affinity_matrix_ = None
        self.embedding_ = None  # Eigenvector coordinates
        self.eigenvalues_ = None

    def _compute_rbf_affinity(self, X):
        """
        Compute RBF (Gaussian) affinity matrix.

        W_ij = exp(-gamma * ||x_i - x_j||²)
        """
        # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 x_i·x_j
        X_sq = np.sum(X**2, axis=1)
        D_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * X @ X.T

        W = np.exp(-self.gamma * D_sq)
        np.fill_diagonal(W, 0)  # No self-loops
        return W

    def _compute_knn_affinity(self, X):
        """
        Compute k-nearest neighbors affinity matrix.

        Symmetric: W_ij = 1 if i in kNN(j) OR j in kNN(i)
        """
        n_samples = X.shape[0]

        # Compute distances
        X_sq = np.sum(X**2, axis=1)
        D = np.sqrt(np.maximum(X_sq[:, np.newaxis] + X_sq - 2 * X @ X.T, 0))

        # Find k nearest neighbors for each point
        W = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Get indices of k nearest (excluding self)
            distances = D[i].copy()
            distances[i] = np.inf  # Exclude self
            knn_indices = np.argsort(distances)[:self.n_neighbors]

            for j in knn_indices:
                # Symmetric: both directions
                W[i, j] = 1
                W[j, i] = 1

        return W

    def _compute_laplacian(self, W):
        """
        Compute graph Laplacian.

        Unnormalized: L = D - W
        Normalized (symmetric): L_sym = I - D^(-1/2) W D^(-1/2)
        """
        # Degree matrix
        d = np.sum(W, axis=1)

        if self.laplacian == 'unnormalized':
            D = np.diag(d)
            L = D - W

        else:  # normalized
            # Handle zero degrees
            d_inv_sqrt = np.zeros_like(d)
            nonzero = d > 0
            d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])

            D_inv_sqrt = np.diag(d_inv_sqrt)

            # L_sym = I - D^(-1/2) W D^(-1/2)
            L = np.eye(len(d)) - D_inv_sqrt @ W @ D_inv_sqrt

        return L

    def _kmeans_simple(self, X, n_clusters, max_iter=100):
        """Simple K-means for clustering eigenvectors."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # Initialize centroids randomly
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[indices].copy()

        for _ in range(max_iter):
            # Assign to nearest centroid
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            C_sq = np.sum(centroids**2, axis=1)
            distances_sq = X_sq + C_sq - 2 * X @ centroids.T
            labels = np.argmin(distances_sq, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(n_samples)]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels

    def fit(self, X):
        """
        Perform spectral clustering.

        Algorithm:
        1. Compute affinity matrix W
        2. Compute Laplacian L
        3. Find k smallest eigenvectors of L
        4. Cluster rows of eigenvector matrix
        """
        n_samples = X.shape[0]

        # 1. Compute affinity matrix
        if self.affinity == 'rbf':
            W = self._compute_rbf_affinity(X)
        else:
            W = self._compute_knn_affinity(X)

        self.affinity_matrix_ = W

        # 2. Compute Laplacian
        L = self._compute_laplacian(W)

        # 3. Compute eigenvectors
        # We want the k smallest eigenvalues (closest to 0)
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues

        # Take first k eigenvectors (smallest eigenvalues)
        U = eigenvectors[:, :self.n_clusters]

        # For normalized Laplacian, normalize rows
        if self.laplacian == 'normalized':
            row_norms = np.sqrt(np.sum(U**2, axis=1, keepdims=True))
            row_norms = np.maximum(row_norms, 1e-10)  # Avoid division by zero
            U = U / row_norms

        self.embedding_ = U

        # 4. Cluster in eigenspace
        if self.assign_labels == 'kmeans':
            self.labels_ = self._kmeans_simple(U, self.n_clusters)
        else:
            # Simple discretization: assign based on max component
            self.labels_ = np.argmax(np.abs(U), axis=1)

        return self

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def clustering_accuracy(y_true, y_pred, n_clusters=None):
    """Compute clustering accuracy with optimal label permutation."""
    if n_clusters is None:
        n_clusters = max(len(np.unique(y_true)), len(np.unique(y_pred)))

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

    if len(unique_labels) < 2:
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

    # -------- Experiment 1: Effect of Gamma (RBF kernel parameter) --------
    print("\n1. EFFECT OF GAMMA (RBF Kernel Width)")
    print("-" * 40)
    print("gamma = 1/(2σ²): controls neighborhood size")

    X_moons, y_moons = make_moons(n_samples=200, noise=0.1)

    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sc = SpectralClustering(n_clusters=2, gamma=gamma, random_state=42)
        sc.fit(X_moons)
        acc = clustering_accuracy(y_moons, sc.labels_, 2)
        print(f"  gamma={gamma:<5}  accuracy={acc:.3f}")
    print("-> Too small gamma: everything connected → one cluster")
    print("-> Too large gamma: nothing connected → noise")
    print("-> Sweet spot depends on data scale")

    # -------- Experiment 2: Affinity Type --------
    print("\n2. AFFINITY TYPE: RBF vs K-NN")
    print("-" * 40)

    X, y_true = make_clustered(n_samples=300, n_clusters=3)

    sc_rbf = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42)
    sc_rbf.fit(X)
    acc_rbf = clustering_accuracy(y_true, sc_rbf.labels_, 3)

    sc_knn = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
                                n_neighbors=10, random_state=42)
    sc_knn.fit(X)
    acc_knn = clustering_accuracy(y_true, sc_knn.labels_, 3)

    print(f"  RBF (gamma=1.0):     accuracy={acc_rbf:.3f}")
    print(f"  k-NN (k=10):         accuracy={acc_knn:.3f}")
    print("-> RBF: smoother, depends on gamma")
    print("-> k-NN: local connectivity, less sensitive to scale")

    # -------- Experiment 3: Normalized vs Unnormalized Laplacian --------
    print("\n3. LAPLACIAN TYPE")
    print("-" * 40)

    # Create clusters of different sizes
    np.random.seed(42)
    X_unequal = np.vstack([
        np.random.randn(150, 2) * 0.5 + [0, 0],
        np.random.randn(50, 2) * 0.5 + [3, 0],
    ])
    y_unequal = np.array([0]*150 + [1]*50)

    for laplacian in ['normalized', 'unnormalized']:
        sc = SpectralClustering(n_clusters=2, gamma=1.0, laplacian=laplacian,
                               random_state=42)
        sc.fit(X_unequal)
        acc = clustering_accuracy(y_unequal, sc.labels_, 2)
        print(f"  {laplacian:<15}  accuracy={acc:.3f}")
    print("-> Normalized handles unequal cluster sizes better")
    print("-> Unnormalized can be biased toward equal cuts")

    # -------- Experiment 4: Non-Convex Shapes (Spectral's Strength) --------
    print("\n4. SPECTRAL ON NON-CONVEX SHAPES")
    print("-" * 40)
    print("Where K-means fails, spectral succeeds!")

    datasets = [
        ("Moons", make_moons(n_samples=300, noise=0.08), 0.5),
        ("Circles", make_circles(n_samples=300, noise=0.05), 0.5),
        ("Spiral", make_spiral(n_samples=400, noise=0.3), 0.2),
    ]

    for name, (X_data, y_data), gamma in datasets:
        # Spectral clustering
        sc = SpectralClustering(n_clusters=2, gamma=gamma, random_state=42)
        sc.fit(X_data)
        acc_spectral = clustering_accuracy(y_data, sc.labels_, 2)

        # Compare to K-means baseline
        from scipy.cluster.vq import kmeans2
        try:
            _, labels_km = kmeans2(X_data, 2, seed=42)
            acc_kmeans = clustering_accuracy(y_data, labels_km, 2)
        except:
            acc_kmeans = 0.5

        print(f"  {name:<10}  Spectral={acc_spectral:.3f}  K-means={acc_kmeans:.3f}")

    print("-> Spectral finds non-convex clusters!")
    print("-> This is because eigenspace transforms the geometry")

    # -------- Experiment 5: Eigengap Analysis --------
    print("\n5. EIGENGAP ANALYSIS (Choosing K)")
    print("-" * 40)

    X, y_true = make_clustered(n_samples=300, n_clusters=4)

    sc = SpectralClustering(n_clusters=4, gamma=0.5, random_state=42)
    sc.fit(X)

    print("First 8 eigenvalues:")
    for i, ev in enumerate(sc.eigenvalues_[:8]):
        gap = sc.eigenvalues_[i+1] - ev if i < 7 else 0
        marker = " <-- gap" if gap > 0.1 else ""
        print(f"  λ_{i} = {ev:.4f}  (gap to next: {gap:.4f}){marker}")
    print("-> Large gap after k eigenvalues suggests k clusters")
    print("-> Look for 'elbow' in eigenvalue plot")


def visualize_spectral():
    """Visualize spectral clustering."""
    print("\n" + "="*60)
    print("SPECTRAL CLUSTERING VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Non-convex shapes comparison
    datasets = [
        ("Two Moons", make_moons(n_samples=300, noise=0.08), 0.5),
        ("Circles", make_circles(n_samples=300, noise=0.05), 0.5),
        ("Spiral", make_spiral(n_samples=400, noise=0.3), 0.2),
    ]

    for i, (name, (X_data, y_data), gamma) in enumerate(datasets):
        # Spectral result
        ax = fig.add_subplot(3, 4, i+1)
        sc = SpectralClustering(n_clusters=2, gamma=gamma, random_state=42)
        sc.fit(X_data)
        ax.scatter(X_data[:, 0], X_data[:, 1], c=sc.labels_, cmap='coolwarm',
                  alpha=0.7, s=30)
        acc = clustering_accuracy(y_data, sc.labels_, 2)
        ax.set_title(f'{name} - Spectral\nacc={acc:.2f}')
        ax.set_aspect('equal')

    # Effect of gamma
    ax = fig.add_subplot(3, 4, 4)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1)
    accs = []
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for gamma in gammas:
        sc = SpectralClustering(n_clusters=2, gamma=gamma, random_state=42)
        sc.fit(X_moons)
        accs.append(clustering_accuracy(y_moons, sc.labels_, 2))
    ax.plot(gammas, accs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Gamma\n(Two Moons)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Row 2: Eigenspace visualization
    X_moons, y_moons = make_moons(n_samples=200, noise=0.08)

    sc = SpectralClustering(n_clusters=2, gamma=0.5, random_state=42)
    sc.fit(X_moons)

    # Original space
    ax = fig.add_subplot(3, 4, 5)
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='coolwarm',
              alpha=0.7, s=30)
    ax.set_title('Original Space\n(moons are tangled)')
    ax.set_aspect('equal')

    # Eigenspace (first two eigenvectors)
    ax = fig.add_subplot(3, 4, 6)
    ax.scatter(sc.embedding_[:, 0], sc.embedding_[:, 1], c=y_moons,
              cmap='coolwarm', alpha=0.7, s=30)
    ax.set_title('Eigenspace\n(moons are separated!)')
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')
    ax.set_aspect('equal')

    # Affinity matrix
    ax = fig.add_subplot(3, 4, 7)
    # Sort by cluster for visualization
    sort_idx = np.argsort(sc.labels_)
    W_sorted = sc.affinity_matrix_[sort_idx][:, sort_idx]
    im = ax.imshow(W_sorted, cmap='viridis', aspect='auto')
    ax.set_title('Affinity Matrix\n(sorted by cluster)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Eigenvalues
    ax = fig.add_subplot(3, 4, 8)
    ax.bar(range(min(10, len(sc.eigenvalues_))), sc.eigenvalues_[:10],
           color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectrum\n(gap suggests K=2)')

    # Row 3: Comparison with K-means
    np.random.seed(42)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05)

    # K-means on circles
    ax = fig.add_subplot(3, 4, 9)
    from scipy.cluster.vq import kmeans2
    try:
        _, labels_km = kmeans2(X_circles, 2, seed=42)
    except:
        labels_km = np.zeros(len(X_circles))
    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=labels_km, cmap='coolwarm',
              alpha=0.7, s=30)
    acc_km = clustering_accuracy(y_circles, labels_km, 2)
    ax.set_title(f'K-Means on Circles\nacc={acc_km:.2f} (fails!)')
    ax.set_aspect('equal')

    # Spectral on circles
    ax = fig.add_subplot(3, 4, 10)
    sc = SpectralClustering(n_clusters=2, gamma=0.5, random_state=42)
    sc.fit(X_circles)
    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=sc.labels_, cmap='coolwarm',
              alpha=0.7, s=30)
    acc_sc = clustering_accuracy(y_circles, sc.labels_, 2)
    ax.set_title(f'Spectral on Circles\nacc={acc_sc:.2f} (succeeds!)')
    ax.set_aspect('equal')

    # Clustered data - both should work
    X_clustered, y_clustered = make_clustered(n_samples=300, n_clusters=3)

    ax = fig.add_subplot(3, 4, 11)
    try:
        _, labels_km = kmeans2(X_clustered, 3, seed=42)
    except:
        labels_km = np.zeros(len(X_clustered))
    ax.scatter(X_clustered[:, 0], X_clustered[:, 1], c=labels_km, cmap='viridis',
              alpha=0.7, s=30)
    acc_km = clustering_accuracy(y_clustered, labels_km, 3)
    ax.set_title(f'K-Means on Spherical\nacc={acc_km:.2f}')
    ax.set_aspect('equal')

    ax = fig.add_subplot(3, 4, 12)
    sc = SpectralClustering(n_clusters=3, gamma=0.5, random_state=42)
    sc.fit(X_clustered)
    ax.scatter(X_clustered[:, 0], X_clustered[:, 1], c=sc.labels_, cmap='viridis',
              alpha=0.7, s=30)
    acc_sc = clustering_accuracy(y_clustered, sc.labels_, 3)
    ax.set_title(f'Spectral on Spherical\nacc={acc_sc:.2f}')
    ax.set_aspect('equal')

    plt.suptitle('SPECTRAL CLUSTERING: Graph Laplacian Approach\n'
                 'Row 1: Non-convex shapes | Row 2: Eigenspace transform | Row 3: vs K-means',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_laplacian_eigenvectors():
    """Visualize how Laplacian eigenvectors encode cluster structure."""
    print("\n" + "="*60)
    print("LAPLACIAN EIGENVECTORS VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 8))

    # Create data with clear cluster structure
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2) * 0.5 + [0, 0],
        np.random.randn(50, 2) * 0.5 + [3, 0],
        np.random.randn(50, 2) * 0.5 + [1.5, 3],
    ])
    y_true = np.array([0]*50 + [1]*50 + [2]*50)

    sc = SpectralClustering(n_clusters=3, gamma=0.5, random_state=42)
    sc.fit(X)

    # Plot 1: Original data
    ax = fig.add_subplot(2, 4, 1)
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=30)
    ax.set_title('Original Data\n(3 clusters)')
    ax.set_aspect('equal')

    # Plot 2: Affinity matrix
    ax = fig.add_subplot(2, 4, 2)
    sort_idx = np.argsort(y_true)
    W_sorted = sc.affinity_matrix_[sort_idx][:, sort_idx]
    ax.imshow(W_sorted, cmap='viridis', aspect='auto')
    ax.set_title('Affinity Matrix\n(block structure)')

    # Plot 3-5: Individual eigenvectors
    for i in range(3):
        ax = fig.add_subplot(2, 4, 3+i)
        ax.scatter(X[:, 0], X[:, 1], c=sc.embedding_[:, i], cmap='coolwarm',
                  alpha=0.7, s=30)
        ax.set_title(f'Eigenvector {i+1}\nλ_{i}={sc.eigenvalues_[i]:.3f}')
        ax.set_aspect('equal')

    # Plot 6: Eigenspace (2D projection)
    ax = fig.add_subplot(2, 4, 6)
    ax.scatter(sc.embedding_[:, 0], sc.embedding_[:, 1], c=y_true,
              cmap='viridis', alpha=0.7, s=30)
    ax.set_title('Eigenspace (v1, v2)\nClusters become separable')
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')

    # Plot 7: Eigenvalue spectrum
    ax = fig.add_subplot(2, 4, 7)
    ax.bar(range(min(10, len(sc.eigenvalues_))), sc.eigenvalues_[:10],
           color='steelblue')
    # Highlight gap
    if len(sc.eigenvalues_) > 3:
        gap = sc.eigenvalues_[3] - sc.eigenvalues_[2]
        ax.annotate(f'Gap={gap:.2f}', xy=(2.5, sc.eigenvalues_[2] + gap/2),
                   fontsize=10, ha='center')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalues\n(gap at K=3)')

    # Plot 8: Final clustering
    ax = fig.add_subplot(2, 4, 8)
    ax.scatter(X[:, 0], X[:, 1], c=sc.labels_, cmap='viridis', alpha=0.7, s=30)
    acc = clustering_accuracy(y_true, sc.labels_, 3)
    ax.set_title(f'Spectral Clustering\nacc={acc:.2f}')
    ax.set_aspect('equal')

    plt.suptitle('HOW SPECTRAL CLUSTERING WORKS\n'
                 'Eigenvectors of Graph Laplacian encode cluster membership',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_spectral():
    """Benchmark spectral clustering."""
    print("\n" + "="*60)
    print("BENCHMARK: Spectral Clustering")
    print("="*60)

    results = {}

    # Test on various datasets
    datasets = [
        ("moons", make_moons(n_samples=300, noise=0.08), 2, 0.5),
        ("circles", make_circles(n_samples=300, noise=0.05), 2, 0.5),
        ("spiral", make_spiral(n_samples=400, noise=0.3), 2, 0.2),
        ("clustered_3", make_clustered(n_samples=300, n_clusters=3), 3, 0.5),
        ("clustered_4", make_clustered(n_samples=300, n_clusters=4), 4, 0.5),
    ]

    for name, (X, y_true), n_clusters, gamma in datasets:
        sc = SpectralClustering(n_clusters=n_clusters, gamma=gamma, random_state=42)
        sc.fit(X)

        acc = clustering_accuracy(y_true, sc.labels_, n_clusters)
        sil = silhouette_score(X, sc.labels_)

        results[name] = {'accuracy': acc, 'silhouette': sil}
        print(f"{name:<15}: acc={acc:.3f}, sil={sil:.3f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("SPECTRAL CLUSTERING — Paradigm: GRAPH LAPLACIAN")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Transform clustering into GRAPH PARTITIONING:

    1. Build similarity graph (RBF kernel, k-NN)
    2. Compute Graph Laplacian: L = D - W
    3. Find eigenvectors of L (smallest eigenvalues)
    4. Cluster in EIGENSPACE using K-means

THE KEY INSIGHT:
    Laplacian eigenvectors encode cluster structure!
    - First eigenvector: constant (trivial)
    - Second eigenvector: optimal 2-way partition
    - k eigenvectors: k-way partition

    The eigenspace transformation makes NON-CONVEX clusters
    become LINEARLY SEPARABLE!

CONNECTION TO GNNs:
    Spectral clustering → Spectral graph convolution → GCN
    - Graph Laplacian eigenvectors = Fourier basis on graphs
    - GCN uses normalized adjacency (related to Laplacian)
    - Understanding spectral methods helps understand GNNs!

INDUCTIVE BIAS:
    - Clusters = nearly disconnected subgraphs
    - Gamma (RBF) or k (k-NN) defines connectivity
    - Still requires K (but eigengap can help choose)
    """)

    ablation_experiments()
    results = benchmark_spectral()

    print("\nGenerating visualizations...")

    fig1 = visualize_spectral()
    save_path1 = '/Users/sid47/ML Algorithms/58_spectral_clustering.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_laplacian_eigenvectors()
    save_path2 = '/Users/sid47/ML Algorithms/58_spectral_eigenvectors.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "="*60)
    print("SUMMARY: What Spectral Clustering Reveals")
    print("="*60)
    print("""
1. GRAPH PERSPECTIVE: Clustering = graph partitioning
2. LAPLACIAN EIGENVECTORS: Encode cluster structure beautifully
3. NON-CONVEX SHAPES: Eigenspace makes them separable
4. GAMMA IS CRUCIAL: Controls graph connectivity (like DBSCAN's eps)
5. BRIDGE TO GNNs: Same math underlies Graph Neural Networks!

WHEN TO USE:
    - Non-convex cluster shapes
    - When graph structure is natural
    - Medium-sized datasets (eigendecomposition is O(n³))
    - When you understand the similarity scale (gamma)

WHEN TO AVOID:
    - Very large datasets (eigendecomposition doesn't scale)
    - When K is truly unknown (use DBSCAN)
    - Very high-dimensional data (similarity becomes noisy)

COMPARISON SUMMARY:
    K-Means:      Fast, spherical clusters, needs K
    Hierarchical: Any K post-hoc, expensive, greedy
    DBSCAN:       Auto K, arbitrary shapes, density-based
    Spectral:     Non-convex, graph-based, needs K, elegant math

THE CLUSTERING TOOLKIT:
    - Spherical + known K → K-Means (fast baseline)
    - Unknown K + outliers → DBSCAN
    - Non-convex + known K → Spectral or DBSCAN
    - Hierarchical structure → Hierarchical clustering
    - Very large data → Mini-batch K-means

CONNECTION TO GRAPH LEARNING (35-44):
    Spectral clustering concepts directly connect to:
    - GCN (36_gcn.py): Uses normalized adjacency
    - Graph Transformer (44_graph_transformer.py): Laplacian PE
    - Graph fundamentals (35_graph_fundamentals.py): Laplacian definition

    Learning spectral clustering IS learning graph neural network foundations!
    """)
