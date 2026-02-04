"""
Graph Pooling — Hierarchical Graph Representations
===================================================

Paradigm: COARSEN GRAPHS FOR GRAPH-LEVEL TASKS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

For GRAPH-LEVEL tasks (classify entire graph), we need to:
1. Aggregate node features into a single graph representation
2. Create HIERARCHICAL representations

SIMPLE POOLING (Readout):
    h_G = READOUT({h_v : v ∈ G})

    Options:
    - SUM: Σ_v h_v (captures size, sensitive to outliers)
    - MEAN: (1/|V|) Σ_v h_v (size-invariant)
    - MAX: max_v h_v (captures salient features)

HIERARCHICAL POOLING:
    Graph → Coarsen → Smaller Graph → ... → Final
    "Build a pyramid of increasingly abstract representations"

===============================================================
WHY HIERARCHICAL?
===============================================================

1. MULTI-SCALE: Capture local AND global patterns
2. EFFICIENCY: Reduce computation for large graphs
3. INTERPRETABILITY: See what structures are preserved

ANALOGY to CNNs:
- CNN: Stride/Pooling reduces spatial resolution
- Graph: Pooling reduces number of nodes

===============================================================
DIFFPOOL — Differentiable Pooling
===============================================================

Learn SOFT cluster assignments end-to-end!

S^(l) = softmax(GNN_pool(A, X))  (n × k assignment matrix)

New features:  X^(l+1) = S^T X^(l)       (k × d)
New adjacency: A^(l+1) = S^T A^(l) S     (k × k)

WHERE:
- S_ij = probability that node i belongs to cluster j
- Each layer reduces n nodes to k clusters

DIFFERENTIABLE: Gradients flow through soft assignments!

===============================================================
TOP-K POOLING — Select Top Nodes
===============================================================

Learn a SCORE for each node, keep top-k.

y = sigmoid(X p / ||p||)     (node scores)
idx = top-k(y)               (select indices)
X' = X[idx] ⊙ y[idx]         (gate by score)

WHERE:
- p is a learnable projection vector
- Nodes with high scores are kept
- Low-score nodes are dropped

===============================================================
SAGPOOL — Self-Attention Graph Pooling
===============================================================

Use GNN to compute attention scores:

z = GNN(A, X)                (node representations)
y = tanh(z @ W)              (attention scores)
idx = top-k(y)               (select top-k nodes)
X' = X[idx] ⊙ y[idx]

More expressive than TopK (uses structure for scoring).

===============================================================
GRAPH U-NET — Encode-Decode with Skip Connections
===============================================================

Like image U-Net but for graphs:

ENCODER: Pool → Reduce resolution
DECODER: Unpool → Restore resolution
SKIP: Connect encoder to decoder at each level

Useful for node-level tasks with hierarchical context.

===============================================================
INDUCTIVE BIAS
===============================================================

1. HIERARCHY: Graphs have multi-scale structure
2. SOFT ASSIGNMENT: Nodes belong probabilistically to clusters
3. LEARNED: Pooling is task-dependent
4. PERMUTATION INVARIANCE: Output doesn't depend on node order

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
graph_fund = import_module('35_graph_fundamentals')
Graph = graph_fund.Graph
karate_club = graph_fund.karate_club
create_community_graph = graph_fund.create_community_graph
spring_layout = graph_fund.spring_layout


class GlobalPooling:
    """
    Simple global pooling (readout) operations.

    Aggregate all node features into graph feature.
    """

    def __init__(self, pooling_type='mean'):
        """
        pooling_type: 'sum', 'mean', 'max', or 'attention'
        """
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Learnable attention vector
            self.attention = None

    def forward(self, H, mask=None):
        """
        H: Node features (n × d)
        mask: Optional node mask

        Returns: Graph feature (d,)
        """
        if mask is not None:
            H = H[mask]

        if self.pooling_type == 'sum':
            return np.sum(H, axis=0)

        elif self.pooling_type == 'mean':
            return np.mean(H, axis=0)

        elif self.pooling_type == 'max':
            return np.max(H, axis=0)

        elif self.pooling_type == 'attention':
            # Learnable attention pooling
            if self.attention is None or self.attention.shape[0] != H.shape[1]:
                self.attention = np.random.randn(H.shape[1]) * 0.1

            # Attention scores
            scores = H @ self.attention
            weights = np.exp(scores - np.max(scores))
            weights = weights / (np.sum(weights) + 1e-10)

            return np.sum(H * weights.reshape(-1, 1), axis=0)


class TopKPooling:
    """
    Top-K Graph Pooling.

    Select top-k nodes based on learned projection scores.
    """

    def __init__(self, in_features, ratio=0.5):
        """
        in_features: Node feature dimension
        ratio: Fraction of nodes to keep (0 < ratio ≤ 1)
        """
        self.in_features = in_features
        self.ratio = ratio

        # Learnable projection for scoring
        self.p = np.random.randn(in_features) * 0.1

    def forward(self, adj, H):
        """
        adj: Adjacency matrix (n × n)
        H: Node features (n × d)

        Returns: (new_adj, new_H, selected_indices, scores)
        """
        n = H.shape[0]
        k = max(1, int(n * self.ratio))

        # Compute scores
        scores = H @ self.p / (np.linalg.norm(self.p) + 1e-10)
        scores = np.tanh(scores)  # Normalize to [-1, 1]

        # Select top-k
        idx = np.argsort(scores)[-k:]
        idx = np.sort(idx)  # Keep original order

        # Gate features by score
        new_H = H[idx] * scores[idx].reshape(-1, 1)

        # Subset adjacency
        new_adj = adj[np.ix_(idx, idx)]

        return new_adj, new_H, idx, scores


class DiffPool:
    """
    Differentiable Pooling.

    Learn soft cluster assignments using GNN.
    """

    def __init__(self, in_features, n_clusters, hidden_dim=32):
        """
        in_features: Input feature dimension
        n_clusters: Number of clusters (output nodes)
        """
        self.in_features = in_features
        self.n_clusters = n_clusters

        # GNN for computing cluster assignments
        scale = np.sqrt(2.0 / in_features)
        self.W_assign = np.random.randn(in_features, hidden_dim) * scale
        self.W_assign2 = np.random.randn(hidden_dim, n_clusters) * 0.1

        # GNN for computing node embeddings
        self.W_embed = np.random.randn(in_features, hidden_dim) * scale

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, adj, H):
        """
        adj: Adjacency matrix (n × n)
        H: Node features (n × d)

        Returns: (new_adj, new_H, assignment_matrix)
        """
        # Add self-loops and normalize
        adj_norm = adj + np.eye(adj.shape[0])
        D = np.diag(1.0 / (np.sqrt(np.sum(adj_norm, axis=1)) + 1e-10))
        adj_norm = D @ adj_norm @ D

        # Compute cluster assignments: S = softmax(GNN(A, X))
        Z_assign = adj_norm @ H @ self.W_assign
        Z_assign = self.relu(Z_assign)
        Z_assign = adj_norm @ Z_assign @ self.W_assign2

        # Softmax over clusters
        Z_max = np.max(Z_assign, axis=1, keepdims=True)
        exp_Z = np.exp(Z_assign - Z_max)
        S = exp_Z / (np.sum(exp_Z, axis=1, keepdims=True) + 1e-10)  # (n, k)

        # Compute node embeddings
        Z_embed = adj_norm @ H @ self.W_embed
        Z_embed = self.relu(Z_embed)  # (n, hidden)

        # Pool features: X' = S^T Z
        new_H = S.T @ Z_embed  # (k, hidden)

        # Pool adjacency: A' = S^T A S
        new_adj = S.T @ adj @ S  # (k, k)

        return new_adj, new_H, S


class SAGPool:
    """
    Self-Attention Graph Pooling.

    Use GNN to compute attention scores for node selection.
    """

    def __init__(self, in_features, ratio=0.5):
        """
        in_features: Node feature dimension
        ratio: Fraction of nodes to keep
        """
        self.in_features = in_features
        self.ratio = ratio

        # GNN parameters for attention
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, 1) * scale

    def forward(self, adj, H):
        """
        adj: Adjacency matrix (n × n)
        H: Node features (n × d)

        Returns: (new_adj, new_H, selected_indices, scores)
        """
        n = H.shape[0]
        k = max(1, int(n * self.ratio))

        # Add self-loops and normalize
        adj_norm = adj + np.eye(n)
        D = np.diag(1.0 / (np.sqrt(np.sum(adj_norm, axis=1)) + 1e-10))
        adj_norm = D @ adj_norm @ D

        # GNN to compute attention scores
        Z = adj_norm @ H @ self.W  # (n, 1)
        scores = np.tanh(Z).squeeze()

        # Select top-k
        idx = np.argsort(scores)[-k:]
        idx = np.sort(idx)

        # Gate features by score
        new_H = H[idx] * scores[idx].reshape(-1, 1)

        # Subset adjacency
        new_adj = adj[np.ix_(idx, idx)]

        return new_adj, new_H, idx, scores


class GraphClassifier:
    """
    Graph Classification model with pooling.

    GNN → Pool → GNN → Pool → ... → Readout → Classifier
    """

    def __init__(self, n_features, hidden_dim, n_classes,
                 pooling_type='diffpool', pool_ratio=0.5):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.pooling_type = pooling_type

        # First GNN layer
        scale = np.sqrt(2.0 / n_features)
        self.W1 = np.random.randn(n_features, hidden_dim) * scale

        # Pooling layer
        if pooling_type == 'topk':
            self.pool = TopKPooling(hidden_dim, ratio=pool_ratio)
        elif pooling_type == 'sagpool':
            self.pool = SAGPool(hidden_dim, ratio=pool_ratio)
        elif pooling_type == 'diffpool':
            self.pool = DiffPool(hidden_dim, n_clusters=max(2, int(10 * pool_ratio)))
        else:
            self.pool = None

        # Second GNN layer
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)

        # Readout
        self.readout = GlobalPooling('mean')

        # Classifier
        self.W_out = np.random.randn(hidden_dim, n_classes) * 0.1
        self.b_out = np.zeros(n_classes)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, graph):
        """Forward pass."""
        adj = graph.adj
        H = graph.X

        # Add self-loops and normalize
        adj_norm = adj + np.eye(adj.shape[0])
        D = np.diag(1.0 / (np.sqrt(np.sum(adj_norm, axis=1)) + 1e-10))
        adj_norm = D @ adj_norm @ D

        # First GNN
        H = self.relu(adj_norm @ H @ self.W1)

        # Pooling
        if self.pool is not None:
            if self.pooling_type == 'diffpool':
                adj, H, S = self.pool.forward(adj, H)
            else:
                adj, H, idx, scores = self.pool.forward(adj, H)

            # Renormalize
            if adj.shape[0] > 0:
                adj_norm = adj + np.eye(adj.shape[0])
                D = np.diag(1.0 / (np.sqrt(np.sum(adj_norm, axis=1)) + 1e-10))
                adj_norm = D @ adj_norm @ D
            else:
                adj_norm = adj

        # Second GNN
        if H.shape[0] > 0:
            H = self.relu(adj_norm @ H @ self.W2)

        # Readout
        h_graph = self.readout.forward(H)

        # Classifier
        logits = h_graph @ self.W_out + self.b_out

        # Softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        return probs


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_graph_pooling():
    """
    Comprehensive graph pooling visualization:
    1. Global pooling comparison
    2. TopK pooling
    3. DiffPool clusters
    4. SAGPool attention
    5. Hierarchical pooling
    6. Summary
    """
    print("\n" + "="*60)
    print("GRAPH POOLING VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    graph, labels = karate_club()
    graph.X = np.random.randn(graph.n_nodes, 16)

    # ============ Plot 1: Global Pooling Comparison ============
    ax1 = fig.add_subplot(2, 3, 1)

    pooling_types = ['sum', 'mean', 'max']
    H = graph.X

    features = []
    for ptype in pooling_types:
        pool = GlobalPooling(ptype)
        h_g = pool.forward(H)
        features.append(np.mean(np.abs(h_g)))  # Average absolute value

    x = np.arange(len(pooling_types))
    ax1.bar(x, features, color=['steelblue', 'coral', 'mediumseagreen'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['SUM\n(size-sensitive)', 'MEAN\n(size-invariant)', 'MAX\n(salient)'])
    ax1.set_ylabel('Avg |Feature Value|')
    ax1.set_title('Global Pooling Methods\nDifferent aggregation semantics')

    # ============ Plot 2: TopK Pooling ============
    ax2 = fig.add_subplot(2, 3, 2)

    topk = TopKPooling(in_features=16, ratio=0.5)
    new_adj, new_H, idx, scores = topk.forward(graph.adj, graph.X)

    pos = spring_layout(graph)

    # Draw all edges faintly
    for i, j in graph.get_edge_list():
        ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    # Draw nodes with score-based colors
    scatter = ax2.scatter(pos[:, 0], pos[:, 1], c=scores, cmap='RdYlGn',
                         s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax2, label='Score')

    # Highlight selected nodes
    ax2.scatter(pos[idx, 0], pos[idx, 1], c='none', s=200,
               edgecolors='blue', linewidths=2)

    ax2.set_title(f'TopK Pooling (ratio=0.5)\nBlue circles = selected nodes')
    ax2.axis('off')

    # ============ Plot 3: DiffPool Clusters ============
    ax3 = fig.add_subplot(2, 3, 3)

    diffpool = DiffPool(in_features=16, n_clusters=4, hidden_dim=16)
    new_adj, new_H, S = diffpool.forward(graph.adj, graph.X)

    # Color nodes by cluster assignment
    cluster_assign = np.argmax(S, axis=1)
    colors = plt.cm.Set1(cluster_assign / max(cluster_assign.max(), 1))

    for i, j in graph.get_edge_list():
        ax3.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    ax3.scatter(pos[:, 0], pos[:, 1], c=colors, s=100, edgecolors='black')

    ax3.set_title(f'DiffPool Clusters (k={4})\nSoft cluster assignments')
    ax3.axis('off')

    # ============ Plot 4: SAGPool Attention ============
    ax4 = fig.add_subplot(2, 3, 4)

    sagpool = SAGPool(in_features=16, ratio=0.5)
    new_adj, new_H, idx, scores = sagpool.forward(graph.adj, graph.X)

    for i, j in graph.get_edge_list():
        ax4.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    scatter = ax4.scatter(pos[:, 0], pos[:, 1], c=scores, cmap='viridis',
                         s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax4, label='Attention Score')

    ax4.scatter(pos[idx, 0], pos[idx, 1], c='none', s=200,
               edgecolors='red', linewidths=2)

    ax4.set_title('SAGPool (GNN-based attention)\nRed circles = selected nodes')
    ax4.axis('off')

    # ============ Plot 5: Hierarchical Visualization ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Show pooling hierarchy
    levels = []

    # Level 0: original
    levels.append(('Original', graph.n_nodes, graph.X.shape))

    # Level 1: TopK pool
    adj1, H1, _, _ = TopKPooling(16, ratio=0.5).forward(graph.adj, graph.X)
    levels.append(('After TopK', adj1.shape[0], H1.shape))

    # Level 2: Another TopK
    if adj1.shape[0] > 1:
        adj2, H2, _, _ = TopKPooling(H1.shape[1], ratio=0.5).forward(adj1, H1)
        levels.append(('After 2nd TopK', adj2.shape[0], H2.shape))

    y_positions = np.arange(len(levels))
    sizes = [l[1] for l in levels]

    ax5.barh(y_positions, sizes, color='steelblue', edgecolor='black')
    ax5.set_yticks(y_positions)
    ax5.set_yticklabels([l[0] for l in levels])
    ax5.set_xlabel('Number of Nodes')
    ax5.set_title('Hierarchical Pooling\nProgressively coarsen graph')

    for i, (name, n_nodes, shape) in enumerate(levels):
        ax5.text(n_nodes + 0.5, i, f'({n_nodes} nodes)', va='center')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Graph Pooling
    ══════════════════════════════

    THE KEY IDEA:
    Aggregate nodes for graph-level tasks!

    GLOBAL POOLING (Readout):
    h_G = AGG({h_v : v ∈ G})

    ┌────────────────────────────┐
    │ SUM: Size-sensitive        │
    │ MEAN: Size-invariant       │
    │ MAX: Salient features      │
    └────────────────────────────┘

    HIERARCHICAL POOLING:
    ┌────────────────────────────┐
    │ TopK: Select high-score    │
    │       nodes (projection)   │
    ├────────────────────────────┤
    │ DiffPool: Learn soft       │
    │          cluster assign    │
    ├────────────────────────────┤
    │ SAGPool: GNN-based         │
    │         attention scores   │
    └────────────────────────────┘

    BENEFITS:
    ✓ Multi-scale representation
    ✓ Learnable/adaptive
    ✓ End-to-end differentiable
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Graph Pooling — Hierarchical Graph Representations\n'
                 'Coarsen graphs for graph-level tasks',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for graph pooling."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Global pooling type
    print("\n1. GLOBAL POOLING TYPES")
    print("-" * 40)

    for ptype in ['sum', 'mean', 'max', 'attention']:
        results = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            pool = GlobalPooling(ptype)
            h_g = pool.forward(graph.X)
            results.append(np.linalg.norm(h_g))

        print(f"{ptype:<12}  norm={np.mean(results):.3f} ± {np.std(results):.3f}")

    print("→ SUM captures more information but is scale-dependent")

    # 2. Pool ratio effect
    print("\n2. POOL RATIO EFFECT (TopK)")
    print("-" * 40)

    for ratio in [0.25, 0.5, 0.75, 1.0]:
        results = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            topk = TopKPooling(16, ratio=ratio)
            new_adj, new_H, idx, scores = topk.forward(graph.adj, graph.X)
            results.append(new_adj.shape[0])

        print(f"ratio={ratio:.2f}  kept_nodes={np.mean(results):.1f}")

    print("→ Trade-off: more nodes = more info, fewer = more abstract")

    # 3. DiffPool clusters
    print("\n3. DIFFPOOL NUMBER OF CLUSTERS")
    print("-" * 40)

    for n_clusters in [2, 4, 8, 16]:
        results = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            diffpool = DiffPool(16, n_clusters=n_clusters, hidden_dim=16)
            new_adj, new_H, S = diffpool.forward(graph.adj, graph.X)

            # Measure cluster quality (entropy of assignments)
            entropy = -np.mean(np.sum(S * np.log(S + 1e-10), axis=1))
            results.append(entropy)

        print(f"k={n_clusters:<3}  assignment_entropy={np.mean(results):.3f}")

    print("→ More clusters = more specific assignments")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Graph Pooling — Hierarchical Graph Representations")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_graph_pooling()
    save_path = '/Users/sid47/ML Algorithms/41_graph_pooling.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Graph Pooling: Aggregate node features for graph-level tasks
2. Global Pooling: SUM, MEAN, MAX, ATTENTION
3. TopK: Select top-scoring nodes by learned projection
4. DiffPool: Learn soft cluster assignments (differentiable)
5. SAGPool: Use GNN for attention-based node selection
6. Hierarchical: Multi-level coarsening for multi-scale
7. All are differentiable → end-to-end training
    """)
