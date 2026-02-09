"""
DIMENSIONALITY REDUCTION ARENA — Paradigm: COMPARISON

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Head-to-head comparison of PCA, t-SNE, and UMAP.

THREE PARADIGMS OF SEEING:
    PCA:   LINEAR — project onto max variance directions
    t-SNE: PROBABILITY MATCHING — make low-D probabilities match high-D
    UMAP:  TOPOLOGICAL — preserve fuzzy graph structure

Each makes different assumptions, sees different things.
This arena shows WHERE each method wins and fails.

| Property        | PCA      | t-SNE     | UMAP      |
|-----------------|----------|-----------|-----------|
| Type            | Linear   | Nonlinear | Nonlinear |
| Deterministic?  | Yes      | No        | No        |
| Preserves       | Variance | Local nbr | Local+Global |
| New points?     | Yes      | No        | Limited   |
| Speed           | O(nd²)   | O(n²)    | O(n^1.8)  |
| Global struct   | Yes      | Poor      | Good      |
| Nonlinear struct| No       | Yes       | Yes       |

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
pca_module = import_module('59_pca')
tsne_module = import_module('60_tsne')
umap_module = import_module('61_umap')

PCA = pca_module.PCA
TSNE = tsne_module.TSNE
UMAP = umap_module.UMAP


# ============================================================
# DATASETS
# ============================================================

def make_gaussian_mixture(n_samples=300, n_features=50, n_clusters=6, random_state=42):
    """Well-separated Gaussian clusters in high-D."""
    np.random.seed(random_state)
    n_per = n_samples // n_clusters
    X, labels = [], []
    for c in range(n_clusters):
        center = np.zeros(n_features)
        center[c * 2:(c + 1) * 2] = 4.0
        X.append(np.random.randn(n_per, n_features) * 0.7 + center)
        labels.append(np.full(n_per, c))
    return np.vstack(X), np.concatenate(labels)


def make_moons_hd(n_samples=300, n_features=20, random_state=42):
    """Two moons embedded in high-D."""
    np.random.seed(random_state)
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)
    X_2d = np.vstack([
        np.column_stack([np.cos(theta), np.sin(theta)]),
        np.column_stack([1 - np.cos(theta), -np.sin(theta) + 0.5])
    ]) + np.random.randn(n_samples, 2) * 0.1
    W = np.random.randn(2, n_features)
    X = X_2d @ W + np.random.randn(n_samples, n_features) * 0.1
    labels = np.array([0]*n + [1]*n)
    return X, labels


def make_concentric_spheres(n_samples=300, n_features=20, random_state=42):
    """Concentric spheres — radial structure."""
    np.random.seed(random_state)
    n = n_samples // 2
    # Outer sphere
    X_outer = np.random.randn(n, n_features)
    X_outer = X_outer / np.linalg.norm(X_outer, axis=1, keepdims=True) * 3
    # Inner sphere
    X_inner = np.random.randn(n, n_features)
    X_inner = X_inner / np.linalg.norm(X_inner, axis=1, keepdims=True) * 1
    X = np.vstack([X_outer, X_inner]) + np.random.randn(n_samples, n_features) * 0.2
    labels = np.array([0]*n + [1]*n)
    return X, labels


def make_swiss_roll(n_samples=300, random_state=42):
    """Swiss roll in 3D."""
    np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 30 * np.random.rand(n_samples)
    z = t * np.sin(t)
    return np.column_stack([x, y, z]), t


# ============================================================
# ARENA COMPARISON
# ============================================================

def run_arena():
    """
    Run all three methods on all datasets and visualize.
    """
    np.random.seed(42)

    datasets = {
        'Gaussians\n(50D, 6 clusters)': make_gaussian_mixture(200, 50, 6),
        'Moons\n(20D)': make_moons_hd(200, 20),
        'Spheres\n(20D)': make_concentric_spheres(200, 20),
        'Swiss Roll\n(3D)': make_swiss_roll(200),
    }

    methods = {
        'PCA': lambda X: PCA(n_components=2).fit_transform(X),
        't-SNE': lambda X: TSNE(perplexity=30, n_iter=600, learning_rate=200,
                                random_state=42).fit_transform(X),
        'UMAP': lambda X: UMAP(n_neighbors=15, n_epochs=150, min_dist=0.1,
                               random_state=42, init='random').fit_transform(X),
    }

    n_datasets = len(datasets)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_datasets, n_methods, figsize=(4 * n_methods, 4 * n_datasets))

    for row, (ds_name, (X, y)) in enumerate(datasets.items()):
        for col, (method_name, method_fn) in enumerate(methods.items()):
            ax = axes[row, col]

            t0 = time.time()
            Z = method_fn(X)
            elapsed = time.time() - t0

            ax.scatter(Z[:, 0], Z[:, 1], c=y, s=15, alpha=0.7, cmap='Spectral')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f'{method_name}\n({elapsed:.2f}s)', fontsize=12,
                            fontweight='bold')
            else:
                ax.set_title(f'({elapsed:.2f}s)', fontsize=9)

            if col == 0:
                ax.set_ylabel(ds_name, fontsize=11, fontweight='bold')

    plt.suptitle('DIMENSIONALITY REDUCTION ARENA\n'
                 'PCA (linear) vs t-SNE (local) vs UMAP (topological)\n'
                 'Color = true labels | Each method reveals different structure',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def run_properties_comparison():
    """
    Compare properties: speed, determinism, global structure.
    """
    np.random.seed(42)
    X, labels = make_gaussian_mixture(200, 50, 6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Speed comparison
    ax = axes[0]
    sizes = [50, 100, 200]
    times = {'PCA': [], 't-SNE': [], 'UMAP': []}

    for n in sizes:
        X_sub = X[:n]

        t0 = time.time()
        PCA(2).fit_transform(X_sub)
        times['PCA'].append(time.time() - t0)

        t0 = time.time()
        TSNE(perplexity=min(20, n//3), n_iter=300, random_state=42).fit_transform(X_sub)
        times['t-SNE'].append(time.time() - t0)

        t0 = time.time()
        UMAP(n_neighbors=min(15, n//3), n_epochs=100, random_state=42,
             init='random').fit_transform(X_sub)
        times['UMAP'].append(time.time() - t0)

    x_pos = np.arange(len(sizes))
    width = 0.25
    ax.bar(x_pos - width, times['PCA'], width, label='PCA', color='steelblue', alpha=0.7)
    ax.bar(x_pos, times['t-SNE'], width, label='t-SNE', color='coral', alpha=0.7)
    ax.bar(x_pos + width, times['UMAP'], width, label='UMAP', color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'n={n}' for n in sizes])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Speed Comparison', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 2: Determinism test (run 3 times)
    ax = axes[1]
    X_test = X[:100]
    for run in range(3):
        Z_pca = PCA(2).fit_transform(X_test)
        ax.scatter(Z_pca[:, 0] + run * 0.01, Z_pca[:, 1],
                  s=5, alpha=0.5, color='steelblue', label='PCA' if run == 0 else '')

        Z_umap = UMAP(n_neighbors=15, n_epochs=100, random_state=run,
                      init='random').fit_transform(X_test)
        ax.scatter(Z_umap[:, 0] + 10 + run * 0.5, Z_umap[:, 1],
                  s=5, alpha=0.5, color='green', label='UMAP' if run == 0 else '')

    ax.set_title('Determinism: 3 Runs\nPCA = identical, UMAP = varies', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 3: Properties table
    ax = axes[2]
    ax.axis('off')
    table_data = [
        ['Property', 'PCA', 't-SNE', 'UMAP'],
        ['Type', 'Linear', 'Nonlinear', 'Nonlinear'],
        ['Deterministic', '✓', '✗', '✗'],
        ['Global Structure', '✓', '✗', '~✓'],
        ['Local Structure', '✗', '✓', '✓'],
        ['New Points', '✓', '✗', '~'],
        ['Speed', 'Fast', 'Slow', 'Medium'],
        ['Key Param', 'n_comp', 'perplexity', 'n_neighbors'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color header
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Properties Comparison', fontsize=11, pad=20)

    plt.suptitle('DIMENSIONALITY REDUCTION: Properties & Trade-offs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("DIMENSIONALITY REDUCTION ARENA")
    print("PCA vs t-SNE vs UMAP")
    print("="*60)

    print("""
THREE PARADIGMS OF SEEING DATA:

    PCA:   Find max variance directions (LINEAR)
           Best for: preprocessing, denoising, fast overview
           Fails on: nonlinear manifolds

    t-SNE: Match neighborhood probabilities (PROBABILITY)
           Best for: revealing clusters, beautiful visualizations
           Fails on: global structure, reproducibility

    UMAP:  Match fuzzy graph topology (TOPOLOGICAL)
           Best for: large data, global+local structure
           Fails on: very noisy data, exact distances

WHEN TO USE WHAT:
    → Quick exploration → PCA first (always!)
    → Publication visualization → t-SNE (prettiest)
    → Large dataset + clustering → UMAP (fastest nonlinear)
    → Preprocessing for ML → PCA (invertible, fast)
    """)

    print("\n" + "="*60)
    print("RUNNING ARENA COMPARISON")
    print("="*60)

    fig1 = run_arena()
    save_path1 = '/Users/sid47/ML Algorithms/62_dimred_arena.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = run_properties_comparison()
    save_path2 = '/Users/sid47/ML Algorithms/62_dimred_tradeoffs.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("""
VERDICT:
    1. ALWAYS start with PCA for a quick sanity check
    2. Use t-SNE for EXPLORATION (but try multiple perplexities)
    3. Use UMAP for PRODUCTION (faster, more stable)
    4. Never trust cluster SIZES or DISTANCES from t-SNE/UMAP
    5. PCA + UMAP is a powerful combo (PCA to 50D, then UMAP to 2D)
    """)
