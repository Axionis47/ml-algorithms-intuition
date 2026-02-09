"""
t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE) — Paradigm: PROBABILITY MATCHING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Convert DISTANCES to PROBABILITIES. Then make low-D match high-D.

High-dimensional: "How likely is point j to be point i's neighbor?"
    → Use Gaussian: p(j|i) ∝ exp(-||x_i - x_j||² / 2σ²)

Low-dimensional: "Same question, but in 2D"
    → Use Student-t: q(j|i) ∝ (1 + ||y_i - y_j||²)^(-1)

Objective: Make Q look like P
    → Minimize KL(P || Q) via gradient descent

THE CROWDING PROBLEM (why Student-t, not Gaussian):
    In high-D, a point has room for MANY equidistant neighbors.
    In 2D, there's less room — moderate-distance points get crushed.
    Student-t has HEAVY TAILS → pushes moderate-distance points apart.
    This is THE key insight of t-SNE (the "t" in t-SNE).

===============================================================
THE MATHEMATICS
===============================================================

HIGH-DIMENSIONAL AFFINITIES:
    p(j|i) = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i} exp(-||x_i - x_k||² / 2σ_i²)

    Symmetrize: p_ij = (p(j|i) + p(i|j)) / 2n

    σ_i is set per-point to achieve target PERPLEXITY:
    Perp(P_i) = 2^{H(P_i)} where H = -Σ p log₂ p

LOW-DIMENSIONAL AFFINITIES (Student-t with df=1 = Cauchy):
    q_ij = (1 + ||y_i - y_j||²)^(-1) / Σ_{k≠l} (1 + ||y_k - y_l||²)^(-1)

OBJECTIVE:
    KL(P || Q) = Σ_ij p_ij log(p_ij / q_ij)

GRADIENT:
    ∂KL/∂y_i = 4 Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||²)^(-1)

    Attractive: p_ij > q_ij → pull y_i toward y_j (neighbors stay near)
    Repulsive:  p_ij < q_ij → push y_i from y_j (non-neighbors pushed apart)

EARLY EXAGGERATION:
    Multiply P by 4-12 for first ~250 iterations
    WHY: Forces clusters to form quickly, then refine

===============================================================
INDUCTIVE BIAS
===============================================================

1. NON-PARAMETRIC — No function from high-D to low-D
   - Cannot transform new points (must rerun on full dataset)
   - Unlike PCA which gives you a projection matrix

2. PERPLEXITY ≈ Effective number of neighbors
   - Low perplexity → local structure (small clusters)
   - High perplexity → global structure (big blobs)
   - Typical: 5-50 (paper recommends 5-50)

3. STOCHASTIC — Different runs give different layouts
   - Random initialization + stochastic gradient descent
   - Cluster POSITIONS are arbitrary (only relative distances matter)

4. DISTANCES LIE — t-SNE distances don't represent true distances
   - Inter-cluster distances are meaningless
   - Only local neighborhoods are preserved
   - DO NOT interpret cluster sizes or gaps literally

5. GLOBAL STRUCTURE is sacrificed for LOCAL structure
   - Good at showing clusters
   - Bad at showing overall manifold shape

WHAT t-SNE CAN DO:
    ✓ Reveal cluster structure in high-D data
    ✓ Handle nonlinear manifolds (moons, swiss roll)
    ✓ Beautiful visualizations for exploration
    ✓ Discover unexpected groupings

WHAT t-SNE CAN'T DO:
    ✗ Preserve global structure reliably
    ✗ Give meaningful inter-cluster distances
    ✗ Transform new points (non-parametric)
    ✗ Work well with very high perplexity
    ✗ Be deterministic (different runs, different results)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


# ============================================================
# t-SNE IMPLEMENTATION
# ============================================================

class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding — PROBABILITY MATCHING.

    Convert pairwise distances to probabilities, then match them
    in low-dimensional space using KL divergence.

    Parameters:
    -----------
    n_components : int
        Output dimensionality (usually 2).
    perplexity : float
        Effective number of neighbors (5-50 typical).
    learning_rate : float
        Step size for gradient descent.
    n_iter : int
        Number of optimization iterations.
    early_exaggeration : float
        Multiply P by this for first 250 iterations.
    random_state : int or None
        Random seed.
    momentum_init : float
        Momentum for first 250 iterations.
    momentum_final : float
        Momentum after 250 iterations.
    """

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, random_state=None,
                 momentum_init=0.5, momentum_final=0.8):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final

        # Diagnostics
        self.kl_divergence_ = None
        self.kl_history_ = []
        self.embedding_history_ = []

    def _compute_pairwise_distances(self, X):
        """Compute pairwise squared Euclidean distances."""
        sum_X = np.sum(X ** 2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        np.fill_diagonal(D, 0)
        D = np.maximum(D, 0)  # Numerical stability
        return D

    def _binary_search_perplexity(self, distances_i, target_perplexity, tol=1e-5, max_iter=50):
        """
        Binary search for σ_i that gives target perplexity.

        Perplexity = 2^{H(P_i)}
        H(P_i) = -Σ p(j|i) log₂ p(j|i)

        We search for σ_i (or equivalently β_i = 1/(2σ_i²)).
        """
        target_entropy = np.log2(target_perplexity)

        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0  # β = 1/(2σ²)

        for iteration in range(max_iter):
            # Compute conditional probabilities
            exp_D = np.exp(-distances_i * beta)
            sum_exp_D = np.sum(exp_D) + 1e-10

            P_i = exp_D / sum_exp_D

            # Compute entropy H(P_i)
            P_i_safe = np.maximum(P_i, 1e-12)
            entropy = -np.sum(P_i * np.log2(P_i_safe))

            # Check convergence
            entropy_diff = entropy - target_entropy
            if np.abs(entropy_diff) < tol:
                break

            # Binary search update
            if entropy_diff > 0:
                # Entropy too high → σ too large → increase β
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                # Entropy too low → σ too small → decrease β
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

        return P_i, beta

    def _compute_pairwise_affinities(self, X):
        """
        Compute symmetric pairwise affinities P.

        For each point i:
            1. Binary search for σ_i to match target perplexity
            2. Compute conditional p(j|i)

        Then symmetrize: p_ij = (p(j|i) + p(i|j)) / 2n
        """
        n = X.shape[0]
        D = self._compute_pairwise_distances(X)

        P = np.zeros((n, n))

        for i in range(n):
            # Distances from point i to all others
            distances_i = D[i].copy()
            distances_i[i] = np.inf  # Exclude self

            # Find σ_i via binary search
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            P_i, _ = self._binary_search_perplexity(
                distances_i[mask], self.perplexity
            )

            # Fill in P
            P[i, mask] = P_i

        # Symmetrize
        P = (P + P.T) / (2 * n)

        # Ensure minimum probability (numerical stability)
        P = np.maximum(P, 1e-12)

        return P

    def _compute_low_dim_affinities(self, Y):
        """
        Compute low-dimensional affinities Q using Student-t.

        q_ij = (1 + ||y_i - y_j||²)^(-1) / Z
        where Z = Σ_{k≠l} (1 + ||y_k - y_l||²)^(-1)

        Student-t (df=1 = Cauchy) has heavier tails than Gaussian.
        This solves the CROWDING PROBLEM.
        """
        D = self._compute_pairwise_distances(Y)

        # Student-t kernel
        numerator = 1.0 / (1.0 + D)
        np.fill_diagonal(numerator, 0)

        # Normalize
        Q = numerator / (np.sum(numerator) + 1e-10)
        Q = np.maximum(Q, 1e-12)

        return Q, numerator

    def _kl_divergence(self, P, Q):
        """KL(P || Q) = Σ p_ij log(p_ij / q_ij)"""
        return np.sum(P * np.log(P / (Q + 1e-12) + 1e-12))

    def _gradient(self, P, Q, Y, numerator):
        """
        Compute gradient of KL divergence.

        ∂KL/∂y_i = 4 Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||²)^(-1)

        Positive terms (p > q): ATTRACTIVE (pull neighbors closer)
        Negative terms (p < q): REPULSIVE (push non-neighbors apart)
        """
        n = Y.shape[0]
        PQ_diff = P - Q  # (n, n)

        # (p_ij - q_ij) × (1 + ||y_i - y_j||²)^(-1)
        # The numerator already contains the Student-t kernel values
        grad_coeff = PQ_diff * numerator  # (n, n)

        # ∂KL/∂y_i = 4 Σ_j coeff_ij × (y_i - y_j)
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y  # (n, d)
            grad[i] = 4 * np.sum(grad_coeff[i, :, np.newaxis] * diff, axis=0)

        return grad

    def fit_transform(self, X):
        """
        Compute t-SNE embedding.

        THE ALGORITHM:
            1. Compute pairwise affinities P in high-D (with perplexity)
            2. Initialize Y randomly in low-D
            3. For each iteration:
               a. Compute Q in low-D (Student-t)
               b. Compute gradient of KL(P||Q)
               c. Update Y with momentum
            4. Return Y

        Args:
            X: High-dimensional data, shape (n_samples, n_features)

        Returns:
            Y: Low-dimensional embedding, shape (n_samples, n_components)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = X.shape[0]

        if self.perplexity >= n:
            raise ValueError(f"Perplexity ({self.perplexity}) must be less than n_samples ({n}). "
                             f"Try perplexity <= {n // 3} for best results.")

        # Step 1: Compute pairwise affinities
        print(f"   Computing pairwise affinities (perplexity={self.perplexity})...")
        P = self._compute_pairwise_affinities(X)

        # Step 2: Initialize embedding
        Y = np.random.randn(n, self.n_components) * 0.01
        velocity = np.zeros_like(Y)

        # Step 3: Optimize
        self.kl_history_ = []
        self.embedding_history_ = [Y.copy()]

        for iteration in range(self.n_iter):
            # Early exaggeration
            if iteration < 250:
                P_used = P * self.early_exaggeration
                momentum = self.momentum_init
            else:
                P_used = P
                momentum = self.momentum_final

            # Compute Q and gradient
            Q, numerator = self._compute_low_dim_affinities(Y)
            grad = self._gradient(P_used, Q, Y, numerator)

            # Update with momentum
            velocity = momentum * velocity - self.learning_rate * grad
            Y = Y + velocity

            # Center (prevent drift)
            Y = Y - np.mean(Y, axis=0)

            # Record
            kl = self._kl_divergence(P_used, Q)
            self.kl_history_.append(kl)

            # Save snapshots for visualization
            if iteration in [0, 5, 10, 25, 50, 100, 250, 500, 750, 999] or \
               iteration % 100 == 0:
                self.embedding_history_.append(Y.copy())

            if (iteration + 1) % 250 == 0:
                print(f"   Iteration {iteration+1}/{self.n_iter}, KL={kl:.4f}")

        self.kl_divergence_ = self.kl_history_[-1]
        return Y


# ============================================================
# DATASETS
# ============================================================

def make_high_dim_clusters(n_samples=300, n_features=50, n_clusters=5, random_state=42):
    """High-dimensional clusters for t-SNE visualization."""
    np.random.seed(random_state)
    n_per = n_samples // n_clusters
    X, labels = [], []
    for c in range(n_clusters):
        center = np.zeros(n_features)
        center[c * 3:(c + 1) * 3] = 5.0  # Each cluster informative in 3 dims
        X.append(np.random.randn(n_per, n_features) * 0.8 + center)
        labels.append(np.full(n_per, c))
    return np.vstack(X), np.concatenate(labels)


def make_swiss_roll_2d(n_samples=500, random_state=42):
    """Swiss roll in 3D → project to show structure."""
    np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 30 * np.random.rand(n_samples)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    return X, t  # color by t


def make_two_moons_hd(n_samples=300, n_features=20, noise=0.1, random_state=42):
    """Two moons embedded in high dimensions."""
    np.random.seed(random_state)
    n = n_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, n)
    X_2d = np.vstack([
        np.column_stack([np.cos(theta1), np.sin(theta1)]),
        np.column_stack([1 - np.cos(theta2), -np.sin(theta2) + 0.5])
    ]) + np.random.randn(n_samples, 2) * noise

    # Embed in higher dimensions with random rotation
    W = np.random.randn(2, n_features)
    X = X_2d @ W + np.random.randn(n_samples, n_features) * 0.1
    labels = np.array([0]*n + [1]*n)
    return X, labels


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component of t-SNE?
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # -------- Experiment 1: Perplexity Effect --------
    print("\n1. EFFECT OF PERPLEXITY")
    print("-" * 40)
    print("Perplexity ≈ number of effective neighbors")

    np.random.seed(42)
    X, labels = make_high_dim_clusters(200, n_features=20, n_clusters=4)

    for perp in [5, 15, 30, 50]:
        tsne = TSNE(perplexity=perp, n_iter=500, learning_rate=200,
                    random_state=42)
        Y = tsne.fit_transform(X)
        print(f"   perplexity={perp:<4} final_KL={tsne.kl_divergence_:.4f}")

    print("→ Low perplexity: tight local clusters, may fragment")
    print("→ High perplexity: broader neighborhoods, merges small clusters")
    print("→ Rule of thumb: perplexity = 5-50, try multiple values!")

    # -------- Experiment 2: Stochasticity --------
    print("\n2. STOCHASTICITY — Different runs, different results")
    print("-" * 40)

    for seed in [0, 1, 2]:
        tsne = TSNE(perplexity=30, n_iter=500, learning_rate=200,
                    random_state=seed)
        Y = tsne.fit_transform(X)
        print(f"   seed={seed}: Y_mean=({Y[:, 0].mean():.2f}, {Y[:, 1].mean():.2f})  "
              f"Y_std=({Y[:, 0].std():.2f}, {Y[:, 1].std():.2f})  "
              f"KL={tsne.kl_divergence_:.4f}")

    print("→ Cluster POSITIONS change between runs")
    print("→ But cluster STRUCTURE (who's neighbors) is preserved")
    print("→ t-SNE is for exploration, not definitive coordinates!")

    # -------- Experiment 3: Early Exaggeration --------
    print("\n3. EFFECT OF EARLY EXAGGERATION")
    print("-" * 40)

    for exag in [1.0, 4.0, 12.0, 50.0]:
        tsne = TSNE(perplexity=30, n_iter=500, early_exaggeration=exag,
                    learning_rate=200, random_state=42)
        Y = tsne.fit_transform(X)
        print(f"   exaggeration={exag:<5} KL={tsne.kl_divergence_:.4f}")

    print("→ Exaggeration=1: clusters form slowly")
    print("→ Exaggeration=12: standard — forces quick cluster formation")
    print("→ Too high: clusters may not merge properly")

    # -------- Experiment 4: Learning Rate --------
    print("\n4. EFFECT OF LEARNING RATE")
    print("-" * 40)

    for lr in [10, 100, 200, 1000]:
        tsne = TSNE(perplexity=30, n_iter=500, learning_rate=lr,
                    random_state=42)
        Y = tsne.fit_transform(X)
        Y_spread = Y.std()
        print(f"   lr={lr:<6} KL={tsne.kl_divergence_:.4f}  "
              f"spread={Y_spread:.2f}")

    print("→ Too small: slow convergence, clusters don't separate")
    print("→ Too large: unstable, may blow up")
    print("→ Rule of thumb: lr = n/early_exag ≈ 200")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_tsne_perplexity():
    """
    THE KEY ABLATION: Effect of perplexity on t-SNE embedding.
    """
    np.random.seed(42)
    X, labels = make_high_dim_clusters(200, n_features=20, n_clusters=5)

    perplexities = [5, 15, 30, 50]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(16, 4))

    colors = plt.cm.tab10(labels / max(labels.max(), 1))

    for i, perp in enumerate(perplexities):
        tsne = TSNE(perplexity=perp, n_iter=800, learning_rate=200,
                    random_state=42)
        Y = tsne.fit_transform(X)

        axes[i].scatter(Y[:, 0], Y[:, 1], c=colors, s=20, alpha=0.7)
        axes[i].set_title(f'Perplexity = {perp}\nKL = {tsne.kl_divergence_:.3f}',
                         fontsize=11)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle('t-SNE: Effect of Perplexity\n'
                 'Perplexity ≈ effective number of neighbors | '
                 'Low = local structure, High = global structure',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_tsne_evolution():
    """
    How t-SNE evolves: from noise to clusters over iterations.
    Shows early exaggeration phase → refinement.
    """
    np.random.seed(42)
    X, labels = make_high_dim_clusters(200, n_features=20, n_clusters=4)

    tsne = TSNE(perplexity=30, n_iter=1000, learning_rate=200,
                random_state=42)
    Y = tsne.fit_transform(X)

    # Select snapshots
    snapshot_iters = [0, 1, 2, 3, 4, 5, 6, 7]
    n_show = min(len(tsne.embedding_history_), 8)
    snapshots = tsne.embedding_history_[:n_show]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(labels / max(labels.max(), 1))
    iter_labels = ['Init', 'Iter 5', 'Iter 10', 'Iter 25',
                   'Iter 50', 'Iter 100', 'Iter 250', 'Iter 500']

    for i in range(min(n_show, 8)):
        Y_snap = snapshots[i]
        axes[i].scatter(Y_snap[:, 0], Y_snap[:, 1], c=colors, s=15, alpha=0.7)
        label = iter_labels[i] if i < len(iter_labels) else f'Step {i}'
        exag_note = ' [EXAGGERATED]' if i < 5 else ''
        axes[i].set_title(f'{label}{exag_note}', fontsize=10)
        axes[i].set_aspect('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(n_show, 8):
        axes[j].axis('off')

    plt.suptitle('t-SNE EVOLUTION: Noise → Clusters\n'
                 'Early exaggeration (first 250 iter) forces clusters apart, '
                 'then refinement',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_tsne_vs_pca():
    """
    t-SNE vs PCA: Show where t-SNE wins (nonlinear data).
    """
    np.random.seed(42)

    # Import PCA from our own file
    from importlib import import_module
    pca_module = import_module('59_pca')
    PCA = pca_module.PCA

    # Create datasets
    datasets = {}

    # High-dim clusters
    X_clust, y_clust = make_high_dim_clusters(200, n_features=20, n_clusters=4)
    datasets['High-D Clusters'] = (X_clust, y_clust)

    # Moons in high-D
    X_moons, y_moons = make_two_moons_hd(200, n_features=20)
    datasets['Moons (20D)'] = (X_moons, y_moons)

    # Swiss roll
    X_swiss, y_swiss = make_swiss_roll_2d(200)
    datasets['Swiss Roll (3D)'] = (X_swiss, y_swiss)

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 4 * len(datasets)))

    for row, (name, (X, y)) in enumerate(datasets.items()):
        # PCA
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(X)

        ax = axes[row, 0]
        scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, s=15, alpha=0.7,
                            cmap='Spectral')
        ax.set_title(f'PCA — {name}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # t-SNE
        tsne = TSNE(perplexity=30, n_iter=800, learning_rate=200,
                    random_state=42)
        Z_tsne = tsne.fit_transform(X)

        ax = axes[row, 1]
        scatter = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y, s=15, alpha=0.7,
                            cmap='Spectral')
        ax.set_title(f't-SNE — {name}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('PCA vs t-SNE: Linear vs Nonlinear Dimensionality Reduction\n'
                 'PCA preserves variance (global), t-SNE preserves neighborhoods (local)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
    print("Paradigm: PROBABILITY MATCHING")
    print("="*60)

    print("""
WHAT THIS ALGORITHM IS:
    Convert distances to probabilities in high-D and low-D.
    Minimize KL divergence between them.
    Nearby points stay near; far points get pushed apart.

THE KEY EQUATIONS:
    High-D: p_ij ∝ exp(-||x_i - x_j||² / 2σ²)  (Gaussian)
    Low-D:  q_ij ∝ (1 + ||y_i - y_j||²)^(-1)    (Student-t)
    Loss:   KL(P || Q) = Σ p_ij log(p_ij / q_ij)

INDUCTIVE BIAS:
    - NON-PARAMETRIC: no projection matrix
    - PERPLEXITY controls neighborhood size
    - STOCHASTIC: different runs ≠ same layout
    - DISTANCES LIE: only neighborhoods matter
    - LOCAL over GLOBAL structure

EXPECT IT TO SUCCEED ON:
    - High-D cluster discovery
    - Nonlinear manifolds (moons, swiss roll)

EXPECT IT TO MISLEAD ON:
    - Cluster sizes and inter-cluster distances
    - Global arrangement of clusters
    """)

    # Run ablations
    ablation_experiments()

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_tsne_perplexity()
    save_path1 = '/Users/sid47/ML Algorithms/60_tsne_perplexity.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_tsne_evolution()
    save_path2 = '/Users/sid47/ML Algorithms/60_tsne_evolution.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_tsne_vs_pca()
    save_path3 = '/Users/sid47/ML Algorithms/60_tsne_vs_pca.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What t-SNE Reveals")
    print("="*60)
    print("""
1. t-SNE converts DISTANCES to PROBABILITIES, then matches them
   → High-D uses Gaussian, low-D uses Student-t (heavy tails)

2. The CROWDING PROBLEM is why Student-t matters
   → In 2D, moderate-distance points get crushed together
   → Heavy tails push them apart, creating visible clusters

3. PERPLEXITY is the most important hyperparameter
   → Low (5): tight local clusters, may fragment
   → High (50): broader structure, may merge small groups
   → ALWAYS try multiple values!

4. t-SNE is STOCHASTIC — different runs give different layouts
   → Cluster positions are arbitrary
   → Only relative neighborhoods are meaningful
   → Do NOT interpret distances or cluster sizes literally!

5. t-SNE wins over PCA on NONLINEAR data
   → Moons, swiss roll, high-D clusters with nonlinear boundaries
   → PCA can only find linear subspaces

CONNECTIONS:
    → 18_vae: Both use KL divergence (but very differently)
    → 59_pca: PCA = global linear; t-SNE = local nonlinear
    → 04_naive_bayes: Both work with probability distributions

NEXT: 61_umap.py — Faster, preserves more global structure
      (The modern alternative to t-SNE)
    """)
