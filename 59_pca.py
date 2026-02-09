"""
PRINCIPAL COMPONENT ANALYSIS (PCA) — Paradigm: LINEAR PROJECTION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Find the directions where data VARIES MOST. Project onto them.

Think of it like this:
    You have data scattered in 3D. You want to flatten it to 2D.
    Which 2D plane captures the most information?
    → The plane that preserves the most SPREAD (variance).

THE ALGORITHM:
    1. Center data (subtract mean)
    2. Compute covariance matrix C = (1/n) X^T X
    3. Find eigenvectors of C (principal components)
    4. Project data onto top-k eigenvectors

WHY EIGENVECTORS?
    Eigenvectors of the covariance matrix point in the directions
    of maximum variance. The eigenvalue = variance in that direction.
    Largest eigenvalue → most important direction.

===============================================================
THE MATHEMATICS
===============================================================

COVARIANCE MATRIX:
    C = (1/(n-1)) X^T X  (after centering, unbiased estimate)

EIGENDECOMPOSITION:
    C v_i = λ_i v_i
    v_i = i-th principal component (direction)
    λ_i = variance along that direction

PROJECTION (dimensionality reduction):
    Z = X W_k    (W_k = matrix of top-k eigenvectors)
    Z has shape (n_samples, k) — reduced from d dimensions to k

RECONSTRUCTION:
    X̂ = Z W_k^T   (project back to original space)
    Reconstruction error = Σ_{i>k} λ_i  (lost variance)

EXPLAINED VARIANCE RATIO:
    ratio_i = λ_i / Σ_j λ_j
    "How much of total variance does this component explain?"

ALTERNATIVE: SVD
    X = U Σ V^T
    Principal components = columns of V
    Projections = U Σ (scaled left singular vectors)
    Same result, numerically more stable

===============================================================
INDUCTIVE BIAS — What PCA Assumes
===============================================================

1. LINEAR — Can only find linear subspaces
   - Cannot discover curved manifolds (moons, spirals)
   - For nonlinear: use t-SNE, UMAP, or kernel PCA

2. VARIANCE = IMPORTANCE — High variance directions matter most
   - If the signal is in a LOW variance direction, PCA will miss it
   - Adversarial example: class boundary orthogonal to max variance

3. ORTHOGONAL — Principal components are perpendicular
   - Forces uncorrelated projections
   - May miss correlated but informative combinations

4. GLOBAL — Same linear transform applied everywhere
   - Cannot adapt to local structure
   - Every point is transformed identically

5. SCALE-SENSITIVE — Features with larger scales dominate
   - ALWAYS standardize before PCA (unless scales are meaningful)
   - Without standardization: temperature (°C) dominates over pH

WHAT PCA CAN DO:
    ✓ Reduce dimensionality efficiently
    ✓ Remove correlated redundancy
    ✓ Denoise (reconstruct with fewer components)
    ✓ Visualize high-dimensional data (project to 2D/3D)
    ✓ Speed up downstream algorithms

WHAT PCA CAN'T DO:
    ✗ Find nonlinear structure (moons, spirals)
    ✗ Preserve local neighborhoods (use t-SNE/UMAP)
    ✗ Handle discrete/categorical features
    ✗ Guarantee downstream task performance

===============================================================
CONNECTION TO OTHER ALGORITHMS
===============================================================

PCA ↔ SPECTRAL CLUSTERING (58):
    Both solve eigendecomposition of symmetric matrices.
    Spectral: eigenvectors of graph Laplacian → cluster structure
    PCA: eigenvectors of covariance matrix → variance directions

PCA ↔ VAE (18):
    PCA = linear encoder/decoder
    VAE = nonlinear encoder/decoder + probabilistic
    Linear VAE with unit variance → recovers PCA

PCA ↔ LINEAR REGRESSION (01):
    Both project data onto linear subspaces
    LR: project y onto span of X (supervised)
    PCA: project X onto directions of max variance (unsupervised)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


# ============================================================
# PCA IMPLEMENTATION
# ============================================================

class PCA:
    """
    Principal Component Analysis — LINEAR PROJECTION.

    Find the directions of maximum variance and project data onto them.

    Parameters:
    -----------
    n_components : int
        Number of principal components to keep.
    random_state : int or None
        Random seed (for reproducibility, though PCA is deterministic).
    """

    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        # Fitted attributes (set after fit)
        self.components_ = None        # (n_components, n_features) — eigenvectors
        self.eigenvalues_ = None       # (n_features,) — all eigenvalues
        self.explained_variance_ = None  # (n_components,) — variance per component
        self.explained_variance_ratio_ = None  # fraction of total variance
        self.mean_ = None              # (n_features,) — data mean
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X):
        """
        Fit PCA on data X.

        Steps:
            1. Center data (subtract mean)
            2. Compute covariance matrix
            3. Eigendecomposition
            4. Sort by eigenvalue (descending)
            5. Keep top n_components

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            self (for chaining: pca.fit(X).transform(X))
        """
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)

        if self.n_samples_ < 2:
            raise ValueError("PCA requires at least 2 samples (n_samples=1 → division by zero in covariance)")

        # Step 1: Center data
        X_centered = X - self.mean_

        # Step 2: Covariance matrix
        # C = (1/(n-1)) X^T X  (using n-1 for unbiased estimate)
        cov_matrix = (X_centered.T @ X_centered) / (self.n_samples_ - 1)

        # Step 3: Eigendecomposition
        # np.linalg.eigh is for symmetric matrices (covariance is symmetric)
        # Returns eigenvalues in ASCENDING order
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort descending (reverse)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

        # Step 5: Store results
        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors[:, :self.n_components].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        if total_variance < 1e-10:
            # All-constant data → no meaningful variance
            self.explained_variance_ratio_ = np.zeros(self.n_components)
            self.total_explained_variance_ratio_ = np.zeros(len(eigenvalues))
        else:
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
            self.total_explained_variance_ratio_ = eigenvalues / total_variance

        return self

    def transform(self, X):
        """
        Project data onto principal components.

        Z = (X - mean) @ W_k
        where W_k = matrix of top-k eigenvectors

        Args:
            X: Data, shape (n_samples, n_features)

        Returns:
            Z: Projected data, shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """
        Reconstruct data from reduced representation.

        X̂ = Z @ W_k^T + mean
        This is LOSSY — information in dropped components is gone.

        Args:
            Z: Reduced data, shape (n_samples, n_components)

        Returns:
            X_hat: Reconstructed data, shape (n_samples, n_features)
        """
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        """
        Compute mean squared reconstruction error.

        Error = (1/n) Σ ||x_i - x̂_i||²
        This equals the sum of dropped eigenvalues.
        """
        Z = self.transform(X)
        X_hat = self.inverse_transform(Z)
        return np.mean(np.sum((X - X_hat) ** 2, axis=1))


# ============================================================
# PCA VIA SVD (alternative, more numerically stable)
# ============================================================

class PCA_SVD:
    """
    PCA via Singular Value Decomposition.

    Instead of eigendecomposition of X^T X:
        X = U Σ V^T
        Principal components = rows of V^T (right singular vectors)
        σ_i² / (n-1) = eigenvalue = variance along i-th component

    Advantages over eigendecomposition:
        - Numerically more stable
        - Works even when n_samples < n_features
        - Avoids explicitly forming the covariance matrix
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None

    def fit(self, X):
        n, d = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X = U Σ V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Principal components (rows of Vt)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        # Eigenvalues = σ² / (n-1)
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n - 1)
        total_var = np.sum(S ** 2) / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


# ============================================================
# DATASETS FOR PCA
# ============================================================

def make_high_dim_clusters(n_samples=500, n_features=50, n_informative=5,
                           n_clusters=3, random_state=42):
    """
    High-dimensional data where only a few dimensions carry information.
    PCA should find those dimensions.
    """
    np.random.seed(random_state)
    n_per_cluster = n_samples // n_clusters

    X = np.random.randn(n_samples, n_features) * 0.5  # noise in all dims

    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_clusters):
        start = i * n_per_cluster
        end = start + n_per_cluster
        labels[start:end] = i
        # Put signal in first n_informative dimensions
        center = np.zeros(n_informative)
        center[i % n_informative] = 3.0
        X[start:end, :n_informative] += center

    return X, labels


def make_correlated_features(n_samples=500, n_features=10, n_latent=2, random_state=42):
    """
    Data generated from a low-rank model: X = Z @ W + noise
    PCA should recover the latent dimensions.
    """
    np.random.seed(random_state)
    Z = np.random.randn(n_samples, n_latent)  # latent factors
    W = np.random.randn(n_latent, n_features)  # mixing matrix
    noise = np.random.randn(n_samples, n_features) * 0.3

    X = Z @ W + noise
    return X, Z


def make_2d_rotated(n_samples=300, angle=30, random_state=42):
    """
    Elongated 2D Gaussian — PCA should find the axis of elongation.
    """
    np.random.seed(random_state)
    # Elongated along x-axis
    X = np.column_stack([
        np.random.randn(n_samples) * 3,  # high variance
        np.random.randn(n_samples) * 0.5  # low variance
    ])
    # Rotate
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X = X @ R.T
    return X


def make_moons_2d(n_samples=500, noise=0.1, random_state=42):
    """Two half-moons — PCA will fail (nonlinear structure)."""
    np.random.seed(random_state)
    n = n_samples // 2

    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, n)

    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    X2 = np.column_stack([1 - np.cos(theta2), -np.sin(theta2) + 0.5])

    X = np.vstack([X1, X2]) + np.random.randn(n_samples, 2) * noise
    labels = np.array([0]*n + [1]*n)
    return X, labels


def make_circles_2d(n_samples=500, noise=0.05, random_state=42):
    """Concentric circles — PCA will fail (radial structure)."""
    np.random.seed(random_state)
    n = n_samples // 2

    theta = np.random.rand(n) * 2 * np.pi
    X_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    X_inner = 0.5 * np.column_stack([np.cos(theta), np.sin(theta)])

    X = np.vstack([X_outer, X_inner]) + np.random.randn(n_samples, 2) * noise
    labels = np.array([0]*n + [1]*n)
    return X, labels


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component of PCA?

    These experiments build intuition about what PCA does and doesn't do.
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # -------- Experiment 1: Number of Components --------
    print("\n1. EFFECT OF NUMBER OF COMPONENTS")
    print("-" * 40)
    print("How much variance does each component capture?")

    np.random.seed(42)
    X, _ = make_correlated_features(n_samples=500, n_features=10, n_latent=2)

    pca = PCA(n_components=10).fit(X)
    cumulative = np.cumsum(pca.total_explained_variance_ratio_)

    for i in range(min(10, len(pca.eigenvalues_))):
        bar = "█" * int(pca.total_explained_variance_ratio_[i] * 50)
        print(f"   PC{i+1}: {pca.total_explained_variance_ratio_[i]:.3f} "
              f"(cumulative: {cumulative[i]:.3f}) {bar}")

    print("→ First 2 components capture most variance (data has 2 latent dims)")
    print("→ The 'elbow' tells you how many components to keep")

    # -------- Experiment 2: PCA on Linear vs Nonlinear --------
    print("\n2. PCA ON LINEAR vs NONLINEAR DATA")
    print("-" * 40)
    print("PCA assumes linear structure. What happens on curved data?")

    # Linear data
    X_linear = make_2d_rotated(300)
    pca_lin = PCA(n_components=1).fit(X_linear)
    Z_lin = pca_lin.transform(X_linear)
    err_lin = pca_lin.reconstruction_error(X_linear)

    # Nonlinear data
    X_moons, y_moons = make_moons_2d(300)
    pca_moon = PCA(n_components=1).fit(X_moons)
    Z_moon = pca_moon.transform(X_moons)
    err_moon = pca_moon.reconstruction_error(X_moons)

    print(f"   Rotated Gaussian (linear): recon error = {err_lin:.4f}")
    print(f"   Moons (nonlinear):         recon error = {err_moon:.4f}")
    print(f"   Variance explained (linear): {pca_lin.explained_variance_ratio_[0]:.3f}")
    print(f"   Variance explained (moons):  {pca_moon.explained_variance_ratio_[0]:.3f}")
    print("→ PCA preserves variance but LOSES the class structure on moons")
    print("→ Variance ≠ useful information for nonlinear data!")

    # -------- Experiment 3: Reconstruction Error vs Components --------
    print("\n3. RECONSTRUCTION ERROR vs NUMBER OF COMPONENTS")
    print("-" * 40)

    X_hd, _ = make_high_dim_clusters(500, n_features=50, n_informative=5)

    errors = []
    for k in range(1, 20):
        pca_k = PCA(n_components=k).fit(X_hd)
        err = pca_k.reconstruction_error(X_hd)
        errors.append(err)
        if k <= 10:
            print(f"   k={k:<3}  error={err:.4f}")

    print("→ Error drops sharply for first ~5 components (the informative ones)")
    print("→ After that, each component only removes a tiny bit of noise")

    # -------- Experiment 4: Effect of Feature Scaling --------
    print("\n4. EFFECT OF FEATURE SCALING")
    print("-" * 40)
    print("What happens when features have different scales?")

    np.random.seed(42)
    X_scaled = np.random.randn(300, 5)
    X_scaled[:, 0] *= 100   # Feature 0 has huge variance
    X_scaled[:, 1] *= 0.01  # Feature 1 has tiny variance

    pca_raw = PCA(n_components=2).fit(X_scaled)
    print("   WITHOUT standardization:")
    for i in range(2):
        print(f"     PC{i+1} variance ratio: {pca_raw.explained_variance_ratio_[i]:.4f}")
    print(f"     PC1 loading (feature 0): {abs(pca_raw.components_[0, 0]):.4f}")
    print("     → PC1 is dominated by feature 0 (just because it's big)")

    # Standardize
    X_std = (X_scaled - X_scaled.mean(axis=0)) / (X_scaled.std(axis=0) + 1e-10)
    pca_std = PCA(n_components=2).fit(X_std)
    print("   WITH standardization:")
    for i in range(2):
        print(f"     PC{i+1} variance ratio: {pca_std.explained_variance_ratio_[i]:.4f}")
    print("     → Now all features contribute equally")
    print("→ ALWAYS standardize unless feature scales are meaningful!")

    # -------- Experiment 5: PCA as Denoising --------
    print("\n5. PCA AS DENOISING")
    print("-" * 40)

    np.random.seed(42)
    X_clean, Z_true = make_correlated_features(300, n_features=10, n_latent=2)
    X_clean_no_noise = Z_true @ np.random.randn(2, 10)  # noiseless

    noise_levels = [0.0, 0.5, 1.0, 2.0]
    for noise in noise_levels:
        X_noisy = X_clean + np.random.randn(*X_clean.shape) * noise
        pca_denoise = PCA(n_components=2).fit(X_noisy)
        X_denoised = pca_denoise.inverse_transform(pca_denoise.transform(X_noisy))
        err_noisy = np.mean(np.sum((X_clean - X_noisy) ** 2, axis=1))
        err_denoised = np.mean(np.sum((X_clean - X_denoised) ** 2, axis=1))
        improvement = (err_noisy - err_denoised) / (err_noisy + 1e-10) * 100
        print(f"   noise={noise:.1f}  error_noisy={err_noisy:.2f}  "
              f"error_denoised={err_denoised:.2f}  improvement={improvement:.1f}%")

    print("→ PCA removes noise by dropping noisy components")
    print("→ Works because signal is low-rank, noise is full-rank")

    # -------- Experiment 6: Eigendecomposition vs SVD --------
    print("\n6. EIGENDECOMPOSITION vs SVD")
    print("-" * 40)

    X_test, _ = make_correlated_features(200, n_features=10, n_latent=3)

    pca_eig = PCA(n_components=3).fit(X_test)
    pca_svd = PCA_SVD(n_components=3).fit(X_test)

    Z_eig = pca_eig.transform(X_test)
    Z_svd = pca_svd.transform(X_test)

    # Components should be the same (up to sign)
    for i in range(3):
        # Align signs: use the element with largest magnitude for robust sign detection
        # (avoids np.sign(0) = 0 problem when first element is near zero)
        ref_idx = np.argmax(np.abs(pca_eig.components_[i]))
        sign = np.sign(pca_eig.components_[i, ref_idx]) * np.sign(pca_svd.components_[i, ref_idx])
        if sign == 0:
            sign = 1  # fallback if both are exactly zero (degenerate case)
        diff = np.max(np.abs(pca_eig.components_[i] - sign * pca_svd.components_[i]))
        print(f"   PC{i+1} max difference: {diff:.2e}")
        print(f"     Variance (eig): {pca_eig.explained_variance_ratio_[i]:.6f}")
        print(f"     Variance (SVD): {pca_svd.explained_variance_ratio_[i]:.6f}")

    print("→ Both methods give IDENTICAL results (up to numerical precision)")
    print("→ SVD is preferred in practice (more stable, no covariance matrix)")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_pca_projection():
    """
    THE KEY INSIGHT: What PCA projection actually looks like.

    4-panel visualization:
    1. Original 2D data with principal component arrows
    2. Data projected onto PC1 (1D reduction)
    3. Data projected onto PC2
    4. Reconstruction from 1 component (information lost)
    """
    np.random.seed(42)
    X = make_2d_rotated(300, angle=35)

    pca = PCA(n_components=2).fit(X)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Original data + PC arrows
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, c='steelblue')

    # Draw PC arrows from mean
    mean = pca.mean_
    for i, (component, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        scale = np.sqrt(var) * 2
        color = 'red' if i == 0 else 'orange'
        ax.annotate('', xy=mean + component * scale, xytext=mean,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3))
        ax.text(mean[0] + component[0] * scale * 1.15,
                mean[1] + component[1] * scale * 1.15,
                f'PC{i+1}\n(λ={var:.2f})', fontsize=10, color=color,
                fontweight='bold', ha='center')

    ax.set_title('Original Data + Principal Components\n'
                 'Arrows = directions of maximum variance', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: Project onto PC1 (show as 1D)
    ax = axes[0, 1]
    Z1 = (X - mean) @ pca.components_[0]  # scalar projection onto PC1
    ax.scatter(Z1, np.zeros_like(Z1), s=10, alpha=0.5, c='steelblue')
    ax.set_title(f'Projected onto PC1 (1D)\n'
                 f'{pca.explained_variance_ratio_[0]:.1%} of variance preserved',
                 fontsize=11)
    ax.set_xlabel('PC1 score')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    # Panel 3: Show projection lines
    ax = axes[1, 0]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.3, c='steelblue')

    # Reconstruct from PC1 only
    pca1 = PCA(n_components=1).fit(X)
    Z = pca1.transform(X)
    X_proj = pca1.inverse_transform(Z)

    # Draw lines from original to projected
    for i in range(0, len(X), 5):  # Every 5th point to avoid clutter
        ax.plot([X[i, 0], X_proj[i, 0]], [X[i, 1], X_proj[i, 1]],
                'r-', alpha=0.3, linewidth=0.5)

    ax.scatter(X_proj[:, 0], X_proj[:, 1], s=10, alpha=0.5, c='red',
               label='Projected (PC1 only)')
    ax.set_title('Projection = Nearest Point on PC1 Line\n'
                 'Red lines = information lost', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 4: Reconstruction error
    ax = axes[1, 1]
    errors = []
    for k in [1, 2]:
        pca_k = PCA(n_components=k).fit(X)
        err = pca_k.reconstruction_error(X)
        errors.append(err)

    ax.bar(['1 component', '2 components'], errors, color=['coral', 'green'], alpha=0.7)
    ax.set_ylabel('Mean Squared Reconstruction Error')
    ax.set_title('Reconstruction Error\n'
                 '2 components = perfect (no information lost)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('PCA: THE KEY INSIGHT — Linear Projection onto Maximum Variance Directions\n'
                 'Project data onto eigenvectors of the covariance matrix',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_pca_variance():
    """
    Scree plot + cumulative variance explained.
    On high-dimensional data — shows where to cut.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Dataset 1: High-dim with 5 informative features
    X1, _ = make_high_dim_clusters(500, n_features=50, n_informative=5)
    pca1 = PCA(n_components=50).fit(X1)

    # Dataset 2: Correlated features with 2 latent dimensions
    X2, _ = make_correlated_features(500, n_features=20, n_latent=2)
    pca2 = PCA(n_components=20).fit(X2)

    # Dataset 3: Random noise (no structure)
    X3 = np.random.randn(500, 20)
    pca3 = PCA(n_components=20).fit(X3)

    datasets = [
        (pca1, '50D, 5 informative', 50),
        (pca2, '20D, 2 latent dims', 20),
        (pca3, '20D, pure noise', 20)
    ]

    for ax, (pca, title, d) in zip(axes, datasets):
        n_show = min(15, d)
        x = np.arange(1, n_show + 1)

        ratios = pca.total_explained_variance_ratio_[:n_show]
        cumulative = np.cumsum(ratios)

        # Bar plot for individual
        ax.bar(x, ratios, alpha=0.6, color='steelblue', label='Individual')

        # Line for cumulative
        ax2 = ax.twinx()
        ax2.plot(x, cumulative, 'ro-', linewidth=2, markersize=5, label='Cumulative')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=9)

        # Mark the elbow
        if title != '20D, pure noise':
            diffs = np.diff(ratios)
            if len(diffs) > 1:
                # Elbow = where the RATE OF CHANGE drops most sharply
                # i.e., largest drop in consecutive differences
                second_diffs = np.diff(diffs)
                elbow = np.argmax(np.abs(second_diffs)) + 1
                ax.axvline(x=elbow + 0.5, color='red', linestyle='--', alpha=0.5)
                ax.text(elbow + 0.7, max(ratios) * 0.8, f'elbow\nat k={elbow}',
                        fontsize=9, color='red')

        ax.set_xlabel('Component')
        ax.set_ylabel('Explained Variance Ratio', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2)

    plt.suptitle('SCREE PLOTS: How Many Components to Keep?\n'
                 'Look for the "elbow" — where eigenvalues drop off sharply',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_pca_failure():
    """
    Where PCA FAILS: nonlinear data.
    Side by side: PCA on linear vs moons vs circles.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Datasets
    X_linear = make_2d_rotated(300, angle=40)
    y_linear = np.zeros(300)

    X_moons, y_moons = make_moons_2d(300)
    X_circles, y_circles = make_circles_2d(300)

    datasets = [
        (X_linear, y_linear, 'Rotated Gaussian\n(LINEAR)'),
        (X_moons, y_moons, 'Moons\n(NONLINEAR)'),
        (X_circles, y_circles, 'Circles\n(NONLINEAR)')
    ]

    for col, (X, y, name) in enumerate(datasets):
        # Top: Original data colored by class
        ax = axes[0, col]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=15, alpha=0.6,
                            cmap='RdYlBu')
        ax.set_title(f'{name}\nOriginal Data', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Bottom: PCA projection to 1D and back
        ax = axes[1, col]
        pca = PCA(n_components=1).fit(X)
        Z = pca.transform(X)
        X_recon = pca.inverse_transform(Z)

        ax.scatter(X_recon[:, 0], X_recon[:, 1], c=y, s=15, alpha=0.6,
                  cmap='RdYlBu')

        var_explained = pca.explained_variance_ratio_[0]

        # Check if classes are separable in 1D
        if len(np.unique(y)) == 2:
            Z_0 = Z[y == 0]
            Z_1 = Z[y == 1]
            overlap = max(0, min(Z_0.max(), Z_1.max()) - max(Z_0.min(), Z_1.min()))
            total_range = max(Z.max() - Z.min(), 1e-10)
            sep = 1 - overlap / total_range
            ax.set_title(f'PCA → 1D → Reconstruct\n'
                        f'Var: {var_explained:.1%} | Sep: {sep:.1%}', fontsize=10)
        else:
            ax.set_title(f'PCA → 1D → Reconstruct\n'
                        f'Var explained: {var_explained:.1%}', fontsize=10)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('WHERE PCA FAILS: Linear vs Nonlinear Data\n'
                 'PCA preserves VARIANCE but loses STRUCTURE on curved data',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_pca_denoising():
    """
    PCA as denoising: reconstruct with fewer components to remove noise.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Generate clean low-rank data
    n_samples, n_features, n_latent = 300, 10, 2
    Z_true = np.random.randn(n_samples, n_latent)
    W_true = np.random.randn(n_latent, n_features)
    X_clean = Z_true @ W_true

    # Add noise
    noise_level = 1.5
    X_noisy = X_clean + np.random.randn(n_samples, n_features) * noise_level

    # Top row: Reconstruction with different k
    k_values = [1, 2, 5, 10]
    for col, k in enumerate(k_values):
        ax = axes[0, col]
        pca = PCA(n_components=k).fit(X_noisy)
        X_recon = pca.inverse_transform(pca.transform(X_noisy))

        # Error vs clean signal
        err = np.mean(np.sum((X_clean - X_recon) ** 2, axis=1))

        # Visualize first 2 dims
        ax.scatter(X_clean[:, 0], X_clean[:, 1], s=10, alpha=0.3, c='green',
                  label='Clean')
        ax.scatter(X_recon[:, 0], X_recon[:, 1], s=10, alpha=0.3, c='blue',
                  label='Reconstructed')
        ax.set_title(f'k = {k} components\nError: {err:.2f}', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Bottom row: Progressive denoising visualization
    ax = axes[1, 0]
    ax.scatter(X_clean[:, 0], X_clean[:, 1], s=10, alpha=0.5, c='green')
    ax.set_title('Clean Signal', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s=10, alpha=0.5, c='red')
    ax.set_title(f'Noisy (σ={noise_level})', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    pca_opt = PCA(n_components=2).fit(X_noisy)
    X_denoised = pca_opt.inverse_transform(pca_opt.transform(X_noisy))
    ax.scatter(X_denoised[:, 0], X_denoised[:, 1], s=10, alpha=0.5, c='blue')
    err_denoised = np.mean(np.sum((X_clean - X_denoised) ** 2, axis=1))
    ax.set_title(f'PCA Denoised (k=2)\nError: {err_denoised:.2f}', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error curve
    ax = axes[1, 3]
    ks = range(1, min(n_features + 1, 15))
    errors_clean = []
    errors_noisy = []
    for k in ks:
        pca_k = PCA(n_components=k).fit(X_noisy)
        X_rec = pca_k.inverse_transform(pca_k.transform(X_noisy))
        errors_clean.append(np.mean(np.sum((X_clean - X_rec) ** 2, axis=1)))
        errors_noisy.append(np.mean(np.sum((X_noisy - X_rec) ** 2, axis=1)))

    ax.plot(list(ks), errors_clean, 'go-', linewidth=2, label='vs Clean Signal')
    ax.plot(list(ks), errors_noisy, 'ro-', linewidth=2, label='vs Noisy Input')
    ax.axvline(x=n_latent, color='blue', linestyle='--', alpha=0.5, label=f'True rank={n_latent}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('MSE')
    ax.set_title('Optimal k = true rank', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('PCA AS DENOISING: Reconstruct with Fewer Components\n'
                 'Signal lives in k dimensions, noise lives in all dimensions → '
                 'dropping high components removes noise',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("Paradigm: LINEAR PROJECTION")
    print("="*60)

    print("""
WHAT THIS ALGORITHM IS:
    Find directions of maximum variance, project data onto them.
    Dimensionality reduction = keeping only the important directions.

THE KEY EQUATION:
    C v_i = λ_i v_i  (eigenvectors of covariance matrix)
    Z = X @ W_k      (project onto top-k eigenvectors)

INDUCTIVE BIAS:
    - LINEAR: can only find linear subspaces
    - VARIANCE = IMPORTANCE: assumes high variance = informative
    - ORTHOGONAL: components are perpendicular
    - GLOBAL: same transform for all points
    - SCALE-SENSITIVE: standardize first!

EXPECT IT TO FAIL ON:
    - Moons (nonlinear boundary)
    - Circles (radial structure)
    - Any data where structure ≠ variance
    """)

    # Run ablations
    ablation_experiments()

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_pca_projection()
    save_path1 = '/Users/sid47/ML Algorithms/59_pca_projection.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_pca_variance()
    save_path2 = '/Users/sid47/ML Algorithms/59_pca_variance.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_pca_failure()
    save_path3 = '/Users/sid47/ML Algorithms/59_pca_failure.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    fig4 = visualize_pca_denoising()
    save_path4 = '/Users/sid47/ML Algorithms/59_pca_denoising.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What PCA Reveals")
    print("="*60)
    print("""
1. PCA finds directions of MAXIMUM VARIANCE
   → Eigenvectors of the covariance matrix

2. Scree plot shows how many components matter
   → Look for the "elbow" (sharp drop-off)

3. PCA FAILS on nonlinear data (moons, circles)
   → Variance ≠ structure for curved manifolds
   → Need t-SNE or UMAP for nonlinear

4. Feature scaling is CRITICAL
   → Without standardization, large-scale features dominate
   → Always standardize unless scales are meaningful

5. PCA can DENOISE by dropping noisy components
   → Signal is low-rank, noise is full-rank
   → Optimal k = intrinsic dimensionality

6. Eigendecomposition vs SVD give IDENTICAL results
   → SVD is numerically more stable
   → Both are O(min(n,d)³) complexity

CONNECTIONS:
    → 58_spectral_clustering: eigendecomposition of Laplacian
    → 18_vae: PCA = linear VAE
    → 01_linear_regression: both are linear projections

NEXT: 60_tsne.py — Nonlinear dimensionality reduction
      (What PCA can't do, t-SNE can — but with trade-offs)
    """)
