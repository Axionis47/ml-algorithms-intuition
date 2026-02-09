"""
UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP) — Paradigm: TOPOLOGICAL EMBEDDING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Build a GRAPH of neighborhoods in high-D. Optimize a low-D layout
to have a SIMILAR graph. Preserves both local AND some global structure.

Think of it like this:
    1. In high-D: connect each point to its k nearest neighbors
       (with fuzzy/soft edges — closer = stronger connection)
    2. In low-D: place points so that the connection pattern
       is as similar as possible

UMAP vs t-SNE:
    t-SNE: Match PROBABILITIES (KL divergence)
    UMAP: Match FUZZY GRAPHS (cross-entropy)

    UMAP is faster, preserves more global structure,
    and has stronger mathematical foundations (topology).

===============================================================
THE MATHEMATICS
===============================================================

HIGH-DIMENSIONAL GRAPH:
    For each point x_i, find k nearest neighbors.

    Edge weight (fuzzy membership):
    μ(x_i, x_j) = exp(-(d(x_i, x_j) - ρ_i) / σ_i)

    where:
    ρ_i = distance to nearest neighbor (local scale)
    σ_i = found via binary search to match log₂(k) target

    Symmetrize: μ_sym = μ + μ^T - μ ⊙ μ^T
    (probabilistic OR: at least one direction considers them neighbors)

LOW-DIMENSIONAL EMBEDDING:
    ν(y_i, y_j) = (1 + a × ||y_i - y_j||^(2b))^(-1)

    a, b are derived from min_dist parameter
    (approximate: a ≈ 1, b ≈ 1 for most settings)

LOSS (Fuzzy Set Cross-Entropy):
    L = Σ_ij [μ_ij log(μ_ij / ν_ij) + (1 - μ_ij) log((1 - μ_ij) / (1 - ν_ij))]

    First term: ATTRACTIVE — pull connected points together
    Second term: REPULSIVE — push disconnected points apart

OPTIMIZATION: SGD with negative sampling
    - Sample positive edges from graph
    - Sample negative edges uniformly
    - Much faster than computing full gradient

===============================================================
INDUCTIVE BIAS
===============================================================

1. MANIFOLD ASSUMPTION — Data lies on a low-D manifold
   - Local distances are meaningful
   - Global distances are approximated by graph geodesics

2. LOCAL CONNECTIVITY — Each point MUST have at least one neighbor
   - ρ_i ensures this (distance to nearest neighbor)
   - Even isolated points get connected

3. TOPOLOGICAL — Preserves connectivity, not exact distances
   - Connected components → clusters
   - Loops → preserved in embedding

4. FASTER THAN t-SNE — SGD with negative sampling
   - O(n log n) vs t-SNE's O(n²)
   - 10-100x faster in practice

5. BETTER GLOBAL STRUCTURE than t-SNE
   - Cross-entropy loss preserves both attraction AND repulsion
   - KL divergence (t-SNE) mostly cares about attraction

WHEN UMAP EXCELS:
    ✓ Large datasets (scales better than t-SNE)
    ✓ When global structure matters
    ✓ Interactive exploration (fast enough to iterate)
    ✓ As preprocessing for clustering

WHEN UMAP STRUGGLES:
    ✗ Very noisy data (noise becomes structure)
    ✗ When exact distances matter (use MDS)
    ✗ Non-manifold data (discrete, tabular with mixed types)

===============================================================
CONNECTION TO OTHER ALGORITHMS
===============================================================

UMAP ↔ GRAPH LEARNING (35-44):
    UMAP builds a k-NN graph — same as GNN input!
    Spectral initialization uses Laplacian eigenvectors
    Understanding UMAP = understanding graph-based ML

UMAP ↔ SPECTRAL CLUSTERING (58):
    Spectral clustering: eigenvectors of Laplacian → cluster
    UMAP: eigenvectors for initialization → optimize with SGD
    Both use the same graph Laplacian!

UMAP ↔ t-SNE (60):
    Both: nonlinear dimensionality reduction
    t-SNE: KL(P||Q), Q uses Student-t
    UMAP: Cross-entropy, faster optimization

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


# ============================================================
# UMAP IMPLEMENTATION
# ============================================================

class UMAP:
    """
    Uniform Manifold Approximation and Projection — TOPOLOGICAL EMBEDDING.

    Build a fuzzy graph in high-D, optimize low-D to match it.

    Parameters:
    -----------
    n_components : int
        Output dimensionality (usually 2).
    n_neighbors : int
        Number of nearest neighbors (controls local vs global).
    min_dist : float
        Minimum distance in embedding (controls cluster tightness).
    n_epochs : int
        Number of optimization epochs.
    learning_rate : float
        SGD learning rate.
    random_state : int or None
        Random seed.
    init : str
        Initialization: 'spectral' or 'random'.
    negative_sample_rate : int
        Number of negative samples per positive edge.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1,
                 n_epochs=200, learning_rate=1.0, random_state=None,
                 init='spectral', negative_sample_rate=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.init = init
        self.negative_sample_rate = negative_sample_rate

        # Compute a, b from min_dist
        self._a, self._b = self._find_ab_params(min_dist)

    def _find_ab_params(self, min_dist, spread=1.0):
        """
        Find a, b parameters for the low-D kernel:
            ν(d) = (1 + a × d^(2b))^(-1)

        These are fit so that ν(min_dist) ≈ 1 and ν(spread) ≈ 0.

        For simplicity, we use approximate values.
        """
        if min_dist < 0.001:
            a, b = 1.0, 1.0
        else:
            # Simple approximation that works well
            b = 1.0
            a = 1.0 / (min_dist ** (2 * b)) if min_dist > 0 else 1.0
        return a, b

    def _compute_knn(self, X):
        """
        Find k-nearest neighbors using brute force.

        Returns:
            knn_indices: (n, k) indices of nearest neighbors
            knn_distances: (n, k) distances to nearest neighbors
        """
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)

        # Pairwise distances
        dists = np.sqrt(np.maximum(
            np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T,
            0
        ))

        knn_indices = np.zeros((n, k), dtype=int)
        knn_distances = np.zeros((n, k))

        for i in range(n):
            d = dists[i].copy()
            d[i] = np.inf  # exclude self
            idx = np.argsort(d)[:k]
            knn_indices[i] = idx
            knn_distances[i] = d[idx]

        return knn_indices, knn_distances

    def _compute_membership_strengths(self, knn_indices, knn_distances):
        """
        Compute fuzzy membership strengths (edge weights).

        For each point i:
            ρ_i = distance to nearest neighbor
            σ_i = found via binary search so that Σ_j exp(-(d_ij - ρ_i)/σ_i) = log₂(k)

        Edge weight: μ_ij = exp(-(d_ij - ρ_i) / σ_i)
        """
        n, k = knn_indices.shape
        target = np.log2(k)

        # ρ_i: distance to nearest neighbor
        rho = knn_distances[:, 0].copy()

        # σ_i: binary search
        sigmas = np.zeros(n)
        for i in range(n):
            lo, hi = 0.001, 100.0
            for _ in range(64):  # Binary search iterations
                mid = (lo + hi) / 2
                val = np.sum(np.exp(-(np.maximum(knn_distances[i] - rho[i], 0)) / mid))
                if val > target:
                    hi = mid
                else:
                    lo = mid
            sigmas[i] = (lo + hi) / 2

        # Build sparse graph as dense matrix (for small n)
        graph = np.zeros((n, n))
        for i in range(n):
            for j_idx in range(k):
                j = knn_indices[i, j_idx]
                d = max(knn_distances[i, j_idx] - rho[i], 0)
                graph[i, j] = np.exp(-d / (sigmas[i] + 1e-10))

        # Symmetrize: probabilistic OR
        # μ_sym = μ + μ^T - μ ⊙ μ^T
        graph_sym = graph + graph.T - graph * graph.T

        return graph_sym

    def _spectral_init(self, graph):
        """
        Initialize embedding using spectral layout.

        Use eigenvectors of the graph Laplacian:
            L = D - W
            Smallest non-zero eigenvectors give initial layout.

        This is the SAME as spectral clustering (58_spectral_clustering.py)!
        """
        n = graph.shape[0]

        # Normalized Laplacian
        D = np.diag(np.sum(graph, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(graph, axis=1) + 1e-10))
        L_norm = np.eye(n) - D_inv_sqrt @ graph @ D_inv_sqrt

        # Eigenvectors (smallest non-trivial)
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Take eigenvectors 1 to n_components (skip constant eigenvector 0)
        # Guard: ensure we have enough eigenvectors
        max_components = min(self.n_components, n - 1)
        init = eigenvectors[:, 1:max_components + 1]
        # Pad with random if we don't have enough eigenvectors
        if max_components < self.n_components:
            padding = np.random.randn(n, self.n_components - max_components) * 0.01
            init = np.column_stack([init, padding])

        # Scale to reasonable range
        init = init / (np.std(init) + 1e-10) * 0.01

        return init

    def _low_dim_kernel(self, dist_sq):
        """
        Low-dimensional kernel: ν(d) = (1 + a × d^(2b))^(-1)
        """
        return 1.0 / (1.0 + self._a * np.power(np.maximum(dist_sq, 1e-10), self._b))

    def _optimize_embedding(self, graph, Y):
        """
        Optimize embedding via SGD with edge sampling + negative sampling.

        For each edge (i, j) with weight μ_ij > 0:
            Attractive: Move y_i, y_j closer
                grad_attract ∝ -2ab d^(2b-2) (y_i - y_j) / (1 + a d^(2b))

            Negative sampling: Sample random k, push y_i, y_k apart
                grad_repel ∝ 2b (y_i - y_k) / (d² (1 + a d^(2b)))
        """
        n = Y.shape[0]

        # Get all positive edges with weights
        rows, cols = np.where(graph > 0)
        weights = graph[rows, cols]

        # Normalize weights to sample probabilities
        edge_probs = weights / weights.sum()
        n_edges = len(rows)

        # Number of edge samples per epoch
        samples_per_epoch = max(n_edges, n * self.n_neighbors)

        alpha = self.learning_rate

        for epoch in range(self.n_epochs):
            # Learning rate decay
            lr = alpha * (1.0 - epoch / self.n_epochs)
            lr = max(lr, 0.001)

            # Sample positive edges
            edge_indices = np.random.choice(n_edges, size=min(samples_per_epoch, n_edges),
                                           p=edge_probs)

            for idx in edge_indices:
                i, j = rows[idx], cols[idx]

                # Compute distance
                diff = Y[i] - Y[j]
                dist_sq = np.sum(diff ** 2)

                # Attractive force
                if dist_sq > 0:
                    grad_coeff = -2.0 * self._a * self._b * \
                                 np.power(dist_sq, self._b - 1) / \
                                 (1.0 + self._a * np.power(dist_sq, self._b))
                    grad = grad_coeff * diff
                    grad = np.clip(grad, -4, 4)
                    Y[i] += lr * grad
                    Y[j] -= lr * grad

                # Negative sampling
                for _ in range(self.negative_sample_rate):
                    k = np.random.randint(n)
                    if k == i:
                        continue

                    diff_neg = Y[i] - Y[k]
                    dist_sq_neg = np.sum(diff_neg ** 2)

                    if dist_sq_neg > 0:
                        grad_coeff_neg = 2.0 * self._b / \
                                         ((0.001 + dist_sq_neg) *
                                          (1.0 + self._a * np.power(max(dist_sq_neg, 1e-10), self._b)))
                        grad_neg = grad_coeff_neg * diff_neg
                        grad_neg = np.clip(grad_neg, -4, 4)
                        Y[i] += lr * grad_neg

        return Y

    def fit_transform(self, X):
        """
        Compute UMAP embedding.

        THE ALGORITHM:
            1. Find k-nearest neighbors
            2. Compute fuzzy membership strengths (edge weights)
            3. Initialize embedding (spectral or random)
            4. Optimize via SGD with negative sampling
            5. Return embedding

        Args:
            X: Data, shape (n_samples, n_features)

        Returns:
            Y: Embedding, shape (n_samples, n_components)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = X.shape[0]
        print(f"   Step 1: Finding {self.n_neighbors}-nearest neighbors...")
        knn_indices, knn_distances = self._compute_knn(X)

        print(f"   Step 2: Computing fuzzy membership strengths...")
        graph = self._compute_membership_strengths(knn_indices, knn_distances)

        print(f"   Step 3: Initializing embedding ({self.init})...")
        if self.init == 'spectral':
            try:
                Y = self._spectral_init(graph)
            except Exception:
                Y = np.random.randn(n, self.n_components) * 0.01
        else:
            Y = np.random.randn(n, self.n_components) * 0.01

        print(f"   Step 4: Optimizing ({self.n_epochs} epochs)...")
        Y = self._optimize_embedding(graph, Y)

        return Y


# ============================================================
# DATASETS
# ============================================================

def make_high_dim_clusters(n_samples=300, n_features=50, n_clusters=5, random_state=42):
    """High-dimensional clusters."""
    np.random.seed(random_state)
    n_per = n_samples // n_clusters
    X, labels = [], []
    for c in range(n_clusters):
        center = np.zeros(n_features)
        center[c * 3:(c + 1) * 3] = 5.0
        X.append(np.random.randn(n_per, n_features) * 0.8 + center)
        labels.append(np.full(n_per, c))
    return np.vstack(X), np.concatenate(labels)


def make_swiss_roll(n_samples=500, random_state=42):
    """Swiss roll in 3D."""
    np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 30 * np.random.rand(n_samples)
    z = t * np.sin(t)
    return np.column_stack([x, y, z]), t


def make_two_moons_hd(n_samples=300, n_features=20, noise=0.1, random_state=42):
    """Two moons in high dimensions."""
    np.random.seed(random_state)
    n = n_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, n)
    X_2d = np.vstack([
        np.column_stack([np.cos(theta1), np.sin(theta1)]),
        np.column_stack([1 - np.cos(theta2), -np.sin(theta2) + 0.5])
    ]) + np.random.randn(n_samples, 2) * noise
    W = np.random.randn(2, n_features)
    X = X_2d @ W + np.random.randn(n_samples, n_features) * 0.1
    labels = np.array([0]*n + [1]*n)
    return X, labels


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each UMAP parameter?
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X, labels = make_high_dim_clusters(150, n_features=20, n_clusters=4)

    # -------- Experiment 1: n_neighbors --------
    print("\n1. EFFECT OF n_neighbors")
    print("-" * 40)
    print("n_neighbors controls local vs global structure")

    for nn in [5, 15, 50]:
        umap = UMAP(n_neighbors=nn, n_epochs=100, random_state=42, init='random')
        Y = umap.fit_transform(X)
        spread = Y.std()
        print(f"   n_neighbors={nn:<4} spread={spread:.3f}")

    print("→ Few neighbors: preserves fine local structure")
    print("→ Many neighbors: preserves more global organization")

    # -------- Experiment 2: min_dist --------
    print("\n2. EFFECT OF min_dist")
    print("-" * 40)
    print("min_dist controls cluster tightness")

    for md in [0.0, 0.1, 0.5, 1.0]:
        umap = UMAP(n_neighbors=15, min_dist=md, n_epochs=100,
                    random_state=42, init='random')
        Y = umap.fit_transform(X)
        spread = Y.std()
        print(f"   min_dist={md:<4} spread={spread:.3f}")

    print("→ min_dist=0: tight, densely packed clusters")
    print("→ min_dist=1: spread out, preserves more continuous structure")

    # -------- Experiment 3: Initialization --------
    print("\n3. EFFECT OF INITIALIZATION")
    print("-" * 40)

    for init_method in ['random', 'spectral']:
        umap = UMAP(n_neighbors=15, n_epochs=100, random_state=42,
                    init=init_method)
        Y = umap.fit_transform(X)
        spread = Y.std()
        print(f"   init={init_method:<10} spread={spread:.3f}")

    print("→ Spectral: uses graph Laplacian for initial layout")
    print("→ Random: starts from noise, needs more epochs")
    print("→ Spectral is better for preserving global structure")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_umap_neighbors():
    """
    THE KEY ABLATION: Effect of n_neighbors.
    """
    np.random.seed(42)
    X, labels = make_high_dim_clusters(200, n_features=20, n_clusters=5)

    neighbor_values = [5, 15, 50, 100]
    fig, axes = plt.subplots(1, len(neighbor_values), figsize=(16, 4))

    colors = plt.cm.tab10(labels / max(labels.max(), 1))

    for i, nn in enumerate(neighbor_values):
        umap = UMAP(n_neighbors=nn, n_epochs=150, min_dist=0.1,
                    random_state=42, init='random')
        Y = umap.fit_transform(X)

        axes[i].scatter(Y[:, 0], Y[:, 1], c=colors, s=20, alpha=0.7)
        axes[i].set_title(f'n_neighbors = {nn}', fontsize=11)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle('UMAP: Effect of n_neighbors\n'
                 'Few neighbors = local structure | '
                 'Many neighbors = global structure',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_umap_mindist():
    """
    Effect of min_dist on cluster tightness.
    """
    np.random.seed(42)
    X, labels = make_high_dim_clusters(200, n_features=20, n_clusters=5)

    mindist_values = [0.0, 0.1, 0.5, 1.0]
    fig, axes = plt.subplots(1, len(mindist_values), figsize=(16, 4))

    colors = plt.cm.tab10(labels / max(labels.max(), 1))

    for i, md in enumerate(mindist_values):
        umap = UMAP(n_neighbors=15, n_epochs=150, min_dist=md,
                    random_state=42, init='random')
        Y = umap.fit_transform(X)

        axes[i].scatter(Y[:, 0], Y[:, 1], c=colors, s=20, alpha=0.7)
        axes[i].set_title(f'min_dist = {md}', fontsize=11)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle('UMAP: Effect of min_dist\n'
                 'Small = tight clusters | Large = spread out, continuous',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_umap_vs_tsne():
    """
    UMAP vs t-SNE: Direct comparison on same data.
    """
    np.random.seed(42)

    from importlib import import_module
    tsne_module = import_module('60_tsne')
    TSNE = tsne_module.TSNE

    pca_module = import_module('59_pca')
    PCA = pca_module.PCA

    # Datasets
    X_clust, y_clust = make_high_dim_clusters(200, n_features=20, n_clusters=5)
    X_moons, y_moons = make_two_moons_hd(200, n_features=20)

    datasets = [
        ('Clusters (20D)', X_clust, y_clust),
        ('Moons (20D)', X_moons, y_moons),
    ]

    fig, axes = plt.subplots(len(datasets), 3, figsize=(14, 4 * len(datasets)))

    for row, (name, X, y) in enumerate(datasets):
        # PCA
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(X)
        axes[row, 0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, s=15, alpha=0.7, cmap='Spectral')
        axes[row, 0].set_title(f'PCA — {name}', fontsize=10)
        axes[row, 0].set_aspect('equal')
        axes[row, 0].grid(True, alpha=0.2)

        # t-SNE
        tsne = TSNE(perplexity=30, n_iter=800, learning_rate=200, random_state=42)
        Z_tsne = tsne.fit_transform(X)
        axes[row, 1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y, s=15, alpha=0.7, cmap='Spectral')
        axes[row, 1].set_title(f't-SNE — {name}', fontsize=10)
        axes[row, 1].set_aspect('equal')
        axes[row, 1].grid(True, alpha=0.2)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # UMAP
        umap = UMAP(n_neighbors=15, n_epochs=200, min_dist=0.1, random_state=42, init='random')
        Z_umap = umap.fit_transform(X)
        axes[row, 2].scatter(Z_umap[:, 0], Z_umap[:, 1], c=y, s=15, alpha=0.7, cmap='Spectral')
        axes[row, 2].set_title(f'UMAP — {name}', fontsize=10)
        axes[row, 2].set_aspect('equal')
        axes[row, 2].grid(True, alpha=0.2)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    plt.suptitle('PCA vs t-SNE vs UMAP: Three Paradigms of Dimensionality Reduction\n'
                 'Linear (PCA) → Local Nonlinear (t-SNE) → Topological (UMAP)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP)")
    print("Paradigm: TOPOLOGICAL EMBEDDING")
    print("="*60)

    print("""
WHAT THIS ALGORITHM IS:
    Build a fuzzy k-NN graph in high-D.
    Optimize a low-D layout to match the graph.
    Preserves both local AND some global structure.

THE KEY EQUATIONS:
    High-D: μ(x_i, x_j) = exp(-(d_ij - ρ_i) / σ_i)
    Low-D:  ν(y_i, y_j) = (1 + a||y_i - y_j||^(2b))^(-1)
    Loss:   Cross-entropy between μ and ν

INDUCTIVE BIAS:
    - MANIFOLD: data lies on low-D manifold
    - LOCAL CONNECTIVITY: each point has at least one neighbor
    - TOPOLOGICAL: preserves connectivity, not distances
    - FASTER than t-SNE (SGD with negative sampling)

KEY PARAMETERS:
    - n_neighbors: local vs global structure (like perplexity)
    - min_dist: cluster tightness
    """)

    # Run ablations
    ablation_experiments()

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_umap_neighbors()
    save_path1 = '/Users/sid47/ML Algorithms/61_umap_neighbors.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_umap_mindist()
    save_path2 = '/Users/sid47/ML Algorithms/61_umap_mindist.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_umap_vs_tsne()
    save_path3 = '/Users/sid47/ML Algorithms/61_umap_vs_tsne.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What UMAP Reveals")
    print("="*60)
    print("""
1. UMAP builds a FUZZY GRAPH in high-D, matches it in low-D
   → Preserves topology (connectivity), not exact distances

2. n_neighbors controls LOCAL vs GLOBAL structure
   → Like perplexity in t-SNE, but with graph interpretation
   → Few neighbors: fine local detail
   → Many neighbors: global organization

3. min_dist controls CLUSTER TIGHTNESS
   → 0.0: dense packed points
   → 1.0: spread out, more continuous

4. UMAP preserves MORE GLOBAL STRUCTURE than t-SNE
   → Cross-entropy loss handles both attraction AND repulsion
   → t-SNE's KL divergence mostly cares about attraction

5. UMAP is FASTER than t-SNE
   → SGD with negative sampling vs full gradient
   → Scales to millions of points

CONNECTIONS:
    → 35_graph_fundamentals: UMAP builds a k-NN graph
    → 58_spectral_clustering: spectral init uses same Laplacian
    → 60_tsne: same goal, different math

NEXT: 62_dimred_arena.py — All three methods head-to-head
    """)
