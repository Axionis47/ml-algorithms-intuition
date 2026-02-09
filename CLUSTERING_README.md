# Clustering Algorithms — Intuition-First Implementation

## Overview

This module implements four fundamental clustering algorithms from scratch, following the repository's philosophy of **intuition-first learning**. Each algorithm represents a different paradigm for discovering structure in unlabeled data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE CLUSTERING PARADIGMS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  55_kmeans.py        │ CENTROID PARTITIONING  │ Minimize distance to K     │
│                      │                         │ representative points      │
├─────────────────────────────────────────────────────────────────────────────┤
│  56_hierarchical.py  │ DENDROGRAM             │ Build tree of merges,      │
│                      │                         │ cut at any level           │
├─────────────────────────────────────────────────────────────────────────────┤
│  57_dbscan.py        │ DENSITY-BASED          │ Clusters = dense regions   │
│                      │                         │ separated by sparse        │
├─────────────────────────────────────────────────────────────────────────────┤
│  58_spectral.py      │ GRAPH LAPLACIAN        │ Cluster in eigenspace of   │
│                      │                         │ similarity graph           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Comparison

| Algorithm      | K Required? | Non-Convex? | Outliers? | Complexity | Best For |
|---------------|-------------|-------------|-----------|------------|----------|
| **K-Means**   | Yes         | No          | No        | O(nKd)     | Fast baseline, spherical clusters |
| **Hierarchical** | No (cut) | Single: Yes | No        | O(n³)      | Exploring hierarchy, unknown K |
| **DBSCAN**    | No (auto)   | Yes         | Yes       | O(n²)      | Arbitrary shapes, outlier detection |
| **Spectral**  | Yes         | Yes         | No        | O(n³)      | Non-convex with known K |

---

## 55. K-Means — Centroid Partitioning

### The Core Idea
Find K points (centroids) such that each data point is close to its nearest centroid:

```
argmin  Σₖ Σ_{x∈Cₖ} ||x - μₖ||²
```

### Algorithm (Lloyd's)
```
1. Initialize K centroids (K-Means++ for smart init)
2. ASSIGN: Each point → nearest centroid
3. UPDATE: Each centroid → mean of assigned points
4. Repeat until convergence
```

### Key Findings from Experiments

**Initialization Matters:**
```
Random init:    mean inertia=3253.7, std=4664.0
K-Means++ init: mean inertia=487.2,  std=279.5
```
→ K-Means++ dramatically reduces variance and avoids bad local minima

**Choosing K:**
- Silhouette score peaks at true K
- Elbow method: look for bend in inertia plot

**Failure Modes:**
```
Elongated clusters: accuracy=0.900 (splits them!)
Concentric circles: accuracy=0.527 (fails)
Half-moons:         accuracy=0.740 (fails)
```

### When to Use
✓ Roughly spherical clusters
✓ Known K (or can estimate)
✓ Need fast baseline
✗ Non-convex shapes
✗ Unknown number of clusters

---

## 56. Hierarchical Clustering — Dendrogram

### The Core Idea
Build a **tree of merges** from individual points to one big cluster. Cut at any level to get K clusters — no recomputation needed!

### Linkage Methods
```
SINGLE:   d(A,B) = min{d(a,b)}  → finds non-convex, but chains
COMPLETE: d(A,B) = max{d(a,b)}  → compact clusters
AVERAGE:  d(A,B) = mean{d(a,b)} → compromise
WARD:     minimize variance     → spherical (like K-means)
```

### Key Findings from Experiments

**Linkage Comparison on Moons (Non-Convex):**
```
Single:   accuracy=0.995  ← Winner for non-convex!
Complete: accuracy=0.855
Average:  accuracy=0.825
Ward:     accuracy=0.825
```

**Advantage: Explore Different K**
```
Same dendrogram, different cuts:
K=2: silhouette=0.530
K=3: silhouette=0.724
K=4: silhouette=0.811 ← Peak
K=5: silhouette=0.685
```

### When to Use
✓ Don't know K beforehand
✓ Want to explore cluster hierarchy
✓ Small-medium datasets (< 10K points)
✗ Large datasets (O(n³) time)
✗ Need fast updates

---

## 57. DBSCAN — Density-Based Clustering

### The Core Idea
Clusters are **dense regions** separated by **sparse regions**. No K required!

### Parameters
- **ε (eps):** Neighborhood radius
- **MinPts:** Minimum points to be "dense"

### Point Types
```
CORE:   ≥ MinPts neighbors within ε → cluster seeds
BORDER: Within ε of core, but not core itself
NOISE:  Neither → OUTLIER (labeled -1)
```

### Key Findings from Experiments

**Effect of Epsilon:**
```
eps=0.1: clusters=0,  noise=300 (everything is noise!)
eps=0.5: clusters=5,  noise=33
eps=0.8: clusters=4,  noise=5   ← Sweet spot
eps=2.0: clusters=3,  noise=0   (merged too much)
```

**Non-Convex Shapes (Where K-Means Fails):**
```
Two Moons: clusters=2, accuracy=1.000, noise=0  ← Perfect!
```

**Natural Outlier Detection:**
```
True outliers detected as noise: 15/20 (75%)
```

**Weakness — Varying Density:**
```
Dense cluster (σ=0.3) + Sparse cluster (σ=1.5):
No single eps works for both!
→ Use HDBSCAN or OPTICS instead
```

### When to Use
✓ Unknown number of clusters
✓ Non-convex cluster shapes
✓ Need outlier detection
✓ Clusters have similar densities
✗ Varying density clusters
✗ High-dimensional data

---

## 58. Spectral Clustering — Graph Laplacian

### The Core Idea
Transform clustering into **graph partitioning**:

```
1. Build similarity graph W (RBF kernel)
2. Compute Laplacian: L = D - W
3. Find k smallest eigenvectors of L
4. Cluster in EIGENSPACE using K-means
```

### Why It Works
The graph Laplacian's eigenvectors **encode cluster structure**:
- First eigenvector: constant (trivial)
- Second eigenvector: optimal 2-way partition
- k eigenvectors: k-way partition

**Non-convex clusters become linearly separable in eigenspace!**

### Key Findings from Experiments

**Eigengap Analysis (Choosing K):**
```
λ_0 = 0.0000
λ_1 = 0.0000
λ_2 = 0.0000
λ_3 = 0.0007  ← Gap after 4 eigenvalues
λ_4 = 0.6393     suggests K=4
```

**Gamma (RBF bandwidth) Effect:**
```
gamma=0.1:  accuracy=0.740 (too connected)
gamma=5.0:  accuracy=0.840
gamma=10.0: accuracy=0.890 (sweet spot)
```

### Connection to Graph Neural Networks
```
Spectral Clustering → Spectral Graph Convolution → GCN

- Laplacian eigenvectors = Fourier basis on graphs
- GCN uses: H' = σ(D^(-1/2) A D^(-1/2) H W)
- This IS the normalized adjacency ≈ I - L_sym

Understanding spectral clustering = understanding GNNs!
```

### When to Use
✓ Non-convex cluster shapes
✓ When graph structure is natural
✓ Medium-sized datasets
✗ Very large datasets (eigendecomposition is O(n³))
✗ Unknown K (use DBSCAN instead)

---

## Decision Flowchart

```
                        START
                          │
                          ▼
                  ┌───────────────┐
                  │ Do you know K? │
                  └───────┬───────┘
                         │
            ┌────────────┴────────────┐
            │ YES                     │ NO
            ▼                         ▼
    ┌───────────────┐         ┌───────────────┐
    │ Are clusters  │         │ Need outlier  │
    │  spherical?   │         │  detection?   │
    └───────┬───────┘         └───────┬───────┘
            │                         │
     ┌──────┴──────┐           ┌──────┴──────┐
     │ YES         │ NO        │ YES         │ NO
     ▼             ▼           ▼             ▼
  K-MEANS      SPECTRAL     DBSCAN    HIERARCHICAL
  (fast!)     (eigenspace)  (density)  (dendrogram)
```

---

## Generated Visualizations

Each algorithm generates detailed visualizations:

| File | Description |
|------|-------------|
| `55_kmeans.png` | Effect of K, failure modes, elbow/silhouette analysis |
| `55_kmeans_algorithm.png` | Step-by-step convergence animation |
| `56_hierarchical.png` | Linkage comparison, non-convex shapes |
| `56_hierarchical_dendrogram.png` | Dendrogram cutting at different levels |
| `57_dbscan.png` | Effect of ε, point types, outlier detection |
| `57_dbscan_algorithm.png` | Core/border/noise classification steps |
| `58_spectral_clustering.png` | Eigenspace transformation, vs K-means |
| `58_spectral_eigenvectors.png` | How eigenvectors encode cluster structure |

---

## Running the Experiments

```bash
# Run individual algorithms
python3 55_kmeans.py
python3 56_hierarchical.py
python3 57_dbscan.py
python3 58_spectral_clustering.py

# Each produces:
# 1. Detailed ablation experiments (printed)
# 2. Benchmark results on standard datasets
# 3. Visualizations saved as PNG files
```

---

## Key Takeaways

### 1. **No Universal Best Algorithm**
Each paradigm has its strengths:
- K-Means: Speed
- Hierarchical: Flexibility (any K post-hoc)
- DBSCAN: Automatic K + outliers
- Spectral: Mathematical elegance + non-convex

### 2. **Inductive Bias Matters**
- K-Means assumes spherical clusters
- DBSCAN assumes uniform density
- Spectral assumes graph connectivity reflects clusters

### 3. **Parameters Are Critical**
- K-Means: number of clusters K
- DBSCAN: ε and MinPts
- Spectral: gamma (RBF bandwidth)

### 4. **Spectral Methods Bridge to GNNs**
Understanding spectral clustering provides the foundation for:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Graph Transformers

---

## Files

```
55_kmeans.py              # K-Means with K-Means++ initialization
56_hierarchical.py        # Agglomerative clustering with dendrograms
57_dbscan.py              # Density-based clustering with noise detection
58_spectral_clustering.py # Graph Laplacian spectral methods
```

Each file follows the repository pattern:
- Extensive docstring with intuition
- Clean NumPy implementation
- `ablation_experiments()` function
- `benchmark_*()` function
- `visualize_*()` functions
- Detailed summary in main block
