"""
GRAPH FUNDAMENTALS — Relational Data Structures
================================================

Paradigm: RELATIONAL DATA

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Data with RELATIONSHIPS between entities.

GRAPH G = (V, E)
- V: nodes (entities) — people, molecules, words
- E: edges (relationships) — friendships, bonds, connections

REPRESENTATIONS:
1. Adjacency Matrix A ∈ {0,1}^(n×n)
   A[i,j] = 1 if edge from i to j

2. Edge List: [(i,j), (k,l), ...]

3. Adjacency List: {i: [j,k], j: [i,l], ...}

NODE FEATURES:
X ∈ R^(n × d) — feature vector for each node

===============================================================
GRAPH TASKS
===============================================================

1. NODE CLASSIFICATION
   - Predict label per node
   - Example: Is this user a bot?

2. LINK PREDICTION
   - Predict if edge exists
   - Example: Will these users become friends?

3. GRAPH CLASSIFICATION
   - Predict label per graph
   - Example: Is this molecule toxic?

4. NODE EMBEDDING
   - Learn vector representation per node
   - Example: Similar nodes → similar vectors

===============================================================
WHY GRAPHS NEED SPECIAL TREATMENT
===============================================================

Graphs have NO FIXED SIZE, NO FIXED ORDER!

PROBLEMS WITH STANDARD ML:
1. Can't flatten to vector (loses structure)
2. Can't use CNN (no grid)
3. Can't use RNN (no sequence)
4. Need PERMUTATION INVARIANT/EQUIVARIANT operations

THE KEY INSIGHT:
Aggregate information from NEIGHBORS!

h_v = f(x_v, AGGREGATE({x_u : u ∈ N(v)}))

Where N(v) = neighbors of node v

===============================================================
IMPORTANT GRAPH PROPERTIES
===============================================================

1. DEGREE d(v) = number of edges at node v

2. DEGREE MATRIX D = diag(d(v₁), d(v₂), ..., d(vₙ))

3. GRAPH LAPLACIAN L = D - A
   - Encodes graph structure
   - Eigenvalues = "frequencies" on graph
   - Eigenvectors = basis for graph signals

4. NORMALIZED LAPLACIAN L_norm = I - D^(-1/2) A D^(-1/2)
   - Eigenvalues in [0, 2]
   - Better numerical properties

===============================================================
INDUCTIVE BIAS OF GRAPHS
===============================================================

1. LOCAL STRUCTURE MATTERS
   - Connected nodes are related
   - Information flows along edges

2. PERMUTATION INVARIANCE (for graph tasks)
   - Reordering nodes shouldn't change output

3. PERMUTATION EQUIVARIANCE (for node tasks)
   - Reordering nodes reorders outputs correspondingly

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# GRAPH DATA STRUCTURE
# ============================================================

class Graph:
    """
    Basic graph data structure with utilities.

    Supports:
    - Adjacency matrix representation
    - Node features
    - Both directed and undirected graphs
    """

    def __init__(self, n_nodes=None, adj_matrix=None, edge_list=None,
                 node_features=None, directed=False):
        """
        Initialize graph from adjacency matrix or edge list.

        Parameters:
        - n_nodes: Number of nodes (required if using edge_list)
        - adj_matrix: (n, n) binary adjacency matrix
        - edge_list: List of (i, j) tuples
        - node_features: (n, d) node feature matrix
        - directed: Whether edges are directed
        """
        self.directed = directed

        if adj_matrix is not None:
            self.adj = adj_matrix.copy()
            self.n_nodes = adj_matrix.shape[0]
        elif edge_list is not None:
            self.n_nodes = n_nodes
            self.adj = np.zeros((n_nodes, n_nodes))
            for i, j in edge_list:
                self.adj[i, j] = 1
                if not directed:
                    self.adj[j, i] = 1
        else:
            self.n_nodes = n_nodes
            self.adj = np.zeros((n_nodes, n_nodes))

        if node_features is not None:
            self.X = node_features.copy()
        else:
            self.X = np.eye(self.n_nodes)  # One-hot by default

        self._compute_properties()

    def _compute_properties(self):
        """Compute graph properties."""
        # Degree
        self.degrees = np.sum(self.adj, axis=1)

        # Degree matrix
        self.D = np.diag(self.degrees)

        # Graph Laplacian: L = D - A
        self.L = self.D - self.adj

        # Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(self.degrees, 1e-10)))
        self.L_norm = np.eye(self.n_nodes) - D_inv_sqrt @ self.adj @ D_inv_sqrt

        # Normalized adjacency (for GCN): A_hat = D^(-1/2) A D^(-1/2)
        self.A_norm = D_inv_sqrt @ self.adj @ D_inv_sqrt

    def add_self_loops(self):
        """Add self-loops to the graph (A = A + I)."""
        self.adj = self.adj + np.eye(self.n_nodes)
        self._compute_properties()

    def get_neighbors(self, node):
        """Get list of neighbors for a node."""
        return np.where(self.adj[node] > 0)[0].tolist()

    def get_edge_list(self):
        """Return edge list representation."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adj[i, j] > 0:
                    if self.directed or i <= j:
                        edges.append((i, j))
        return edges

    def subgraph(self, node_indices):
        """Extract subgraph with given nodes."""
        mask = np.array(node_indices)
        sub_adj = self.adj[np.ix_(mask, mask)]
        sub_features = self.X[mask] if self.X is not None else None
        return Graph(adj_matrix=sub_adj, node_features=sub_features,
                    directed=self.directed)

    def spectral_decomposition(self):
        """
        Compute eigendecomposition of the Laplacian.

        THE GRAPH FOURIER BASIS:
        L = U Λ U^T

        Where:
        - U: eigenvectors (graph Fourier basis)
        - Λ: eigenvalues (graph frequencies)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.L_norm)
        return eigenvalues, eigenvectors


# ============================================================
# CLASSIC GRAPH DATASETS
# ============================================================

def karate_club():
    """
    Zachary's Karate Club — Classic small graph.

    34 members of a karate club, edges = friendships.
    After a dispute, club split into two groups.
    Task: Predict which group each member joined.

    This is THE classic graph for demos!
    """
    # Adjacency list (0-indexed)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10),
        (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ]

    n_nodes = 34

    # Ground truth: which group each member joined
    # Node 0 = instructor (Mr. Hi), Node 33 = administrator (Officer)
    labels = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,  # 0-9
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,  # 10-19
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,  # 20-29
        1, 1, 1, 1                      # 30-33
    ])

    graph = Graph(n_nodes=n_nodes, edge_list=edges)

    return graph, labels


def create_random_graph(n_nodes=100, edge_prob=0.1, n_features=16, n_classes=3):
    """
    Create random Erdos-Renyi graph with random features and labels.

    Useful for testing scalability.
    """
    # Random adjacency
    adj = (np.random.rand(n_nodes, n_nodes) < edge_prob).astype(float)
    adj = np.triu(adj, 1)  # Upper triangular
    adj = adj + adj.T  # Symmetric (undirected)

    # Random features
    features = np.random.randn(n_nodes, n_features)

    # Random labels (clustered by features for some structure)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    graph = Graph(adj_matrix=adj, node_features=features)

    return graph, labels


def create_community_graph(n_communities=3, nodes_per_community=20,
                          p_in=0.3, p_out=0.01, feature_dim=8):
    """
    Create graph with community structure (Stochastic Block Model).

    Nodes within same community connected with probability p_in.
    Nodes in different communities connected with probability p_out.

    This tests whether GNNs can discover community structure!
    """
    n_nodes = n_communities * nodes_per_community

    # Create adjacency matrix
    adj = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            comm_i = i // nodes_per_community
            comm_j = j // nodes_per_community

            if comm_i == comm_j:
                p = p_in
            else:
                p = p_out

            if np.random.rand() < p:
                adj[i, j] = 1
                adj[j, i] = 1

    # Create features: community-specific + noise
    features = np.zeros((n_nodes, feature_dim))
    for i in range(n_nodes):
        comm = i // nodes_per_community
        # Community-specific signal in first few dimensions
        features[i, :n_communities] = 0.1
        features[i, comm] = 1.0
        # Add noise
        features[i] += np.random.randn(feature_dim) * 0.3

    # Labels = community membership
    labels = np.array([i // nodes_per_community for i in range(n_nodes)])

    graph = Graph(adj_matrix=adj, node_features=features)

    return graph, labels


def create_tree_graph(depth=4, branching_factor=2):
    """
    Create a tree graph.

    Useful for testing message passing depth.
    At depth d, information from leaves reaches root only after d layers!
    """
    n_nodes = sum(branching_factor**i for i in range(depth + 1))

    edges = []
    node_id = 0

    for d in range(depth):
        nodes_at_depth = branching_factor**d
        for _ in range(nodes_at_depth):
            parent = node_id
            node_id += 1
            for child in range(parent * branching_factor + 1,
                              parent * branching_factor + branching_factor + 1):
                if child < n_nodes:
                    edges.append((parent, child))

    # Labels: depth in tree
    labels = np.zeros(n_nodes, dtype=int)
    node_id = 0
    for d in range(depth + 1):
        nodes_at_depth = branching_factor**d
        for _ in range(nodes_at_depth):
            labels[node_id] = d
            node_id += 1
            if node_id >= n_nodes:
                break

    graph = Graph(n_nodes=n_nodes, edge_list=edges)

    return graph, labels


def create_molecular_graph():
    """
    Create simple molecule-like graphs.

    Returns a set of small graphs representing molecules.
    """
    molecules = []

    # Ethane-like: C-C
    mol1_edges = [(0, 1)]
    mol1 = Graph(n_nodes=2, edge_list=mol1_edges,
                node_features=np.array([[1, 0], [1, 0]]))  # Both carbon
    molecules.append((mol1, 0))

    # Water-like: H-O-H
    mol2_edges = [(0, 1), (1, 2)]
    mol2 = Graph(n_nodes=3, edge_list=mol2_edges,
                node_features=np.array([[0, 1], [1, 0], [0, 1]]))  # H, O, H
    molecules.append((mol2, 1))

    # Triangle (cycle)
    mol3_edges = [(0, 1), (1, 2), (2, 0)]
    mol3 = Graph(n_nodes=3, edge_list=mol3_edges,
                node_features=np.array([[1, 0], [1, 0], [1, 0]]))
    molecules.append((mol3, 2))

    # Linear chain
    mol4_edges = [(0, 1), (1, 2), (2, 3)]
    mol4 = Graph(n_nodes=4, edge_list=mol4_edges,
                node_features=np.array([[1, 0], [0, 1], [0, 1], [1, 0]]))
    molecules.append((mol4, 0))

    return molecules


# ============================================================
# SIMPLE NODE EMBEDDINGS
# ============================================================

def random_walk_embedding(graph, walk_length=20, n_walks=10, embedding_dim=16):
    """
    Simple random walk node embedding (DeepWalk-style).

    THE IDEA:
    - Do random walks from each node
    - Nodes that appear in same walks are similar
    - Learn embeddings where similar nodes are close

    This is a simplified version - real DeepWalk uses skip-gram.
    """
    n_nodes = graph.n_nodes
    co_occurrence = np.zeros((n_nodes, n_nodes))

    # Perform random walks
    for start_node in range(n_nodes):
        for _ in range(n_walks):
            walk = [start_node]
            current = start_node

            for _ in range(walk_length):
                neighbors = graph.get_neighbors(current)
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(current)

            # Count co-occurrences in window
            for i, node_i in enumerate(walk):
                for j, node_j in enumerate(walk):
                    if i != j and abs(i - j) <= 5:  # Window size 5
                        co_occurrence[node_i, node_j] += 1

    # Simple embedding via SVD of co-occurrence
    # (Real DeepWalk would use skip-gram)
    co_occurrence = np.log1p(co_occurrence)  # Log transform
    U, S, Vt = np.linalg.svd(co_occurrence, full_matrices=False)

    embeddings = U[:, :embedding_dim] * np.sqrt(S[:embedding_dim])

    return embeddings


def spectral_embedding(graph, embedding_dim=16):
    """
    Spectral node embedding using Laplacian eigenvectors.

    THE IDEA:
    - Eigenvectors of Laplacian are "graph Fourier basis"
    - Smallest eigenvectors capture global structure
    - Use them as node embeddings

    Spectral clustering uses this!
    """
    eigenvalues, eigenvectors = graph.spectral_decomposition()

    # Skip first eigenvector (constant), take next embedding_dim
    embeddings = eigenvectors[:, 1:embedding_dim+1]

    return embeddings


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_graph_fundamentals():
    """
    Create comprehensive graph fundamentals visualization:
    1. Karate club graph
    2. Community structure
    3. Adjacency matrix vs Laplacian
    4. Spectral embedding
    5. Random walk embedding
    6. Message passing concept
    """
    print("\n" + "="*60)
    print("GRAPH FUNDAMENTALS VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: Karate Club Graph ============
    ax1 = fig.add_subplot(2, 3, 1)

    graph, labels = karate_club()

    # Simple force-directed layout (spring embedding)
    pos = spring_layout(graph)

    # Draw edges
    for i, j in graph.get_edge_list():
        ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.3, linewidth=0.5)

    # Draw nodes
    colors = ['red' if l == 0 else 'blue' for l in labels]
    ax1.scatter(pos[:, 0], pos[:, 1], c=colors, s=100, zorder=5)

    # Label key nodes
    ax1.annotate('Mr. Hi\n(Instructor)', pos[0], fontsize=8)
    ax1.annotate('Officer\n(Admin)', pos[33], fontsize=8)

    ax1.set_title("Zachary's Karate Club\n34 nodes, 2 communities")
    ax1.axis('off')

    # ============ Plot 2: Community Graph ============
    ax2 = fig.add_subplot(2, 3, 2)

    comm_graph, comm_labels = create_community_graph(
        n_communities=3, nodes_per_community=15, p_in=0.4, p_out=0.02
    )
    pos2 = spring_layout(comm_graph)

    for i, j in comm_graph.get_edge_list():
        ax2.plot([pos2[i, 0], pos2[j, 0]], [pos2[i, 1], pos2[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    scatter = ax2.scatter(pos2[:, 0], pos2[:, 1], c=comm_labels,
                         cmap='viridis', s=50, zorder=5)
    ax2.set_title('Community Graph (SBM)\nClear cluster structure')
    ax2.axis('off')

    # ============ Plot 3: Adjacency vs Laplacian ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Use small subset of karate club
    small_graph = graph.subgraph(list(range(10)))

    # Show both matrices side by side
    combined = np.zeros((10, 21))
    combined[:, :10] = small_graph.adj
    combined[:, 11:] = small_graph.L

    im = ax3.imshow(combined, cmap='RdBu_r', aspect='auto')
    ax3.axvline(10, color='black', linewidth=2)

    ax3.set_title('Adjacency A (left) vs Laplacian L (right)\nL = D - A captures structure')
    ax3.set_xlabel('← A | L →')
    ax3.set_yticks([])

    # ============ Plot 4: Spectral Embedding ============
    ax4 = fig.add_subplot(2, 3, 4)

    spec_emb = spectral_embedding(graph, embedding_dim=2)

    ax4.scatter(spec_emb[:, 0], spec_emb[:, 1], c=labels,
               cmap='coolwarm', s=50)
    ax4.set_xlabel('Eigenvector 1')
    ax4.set_ylabel('Eigenvector 2')
    ax4.set_title('Spectral Embedding (Laplacian eigenvectors)\nNaturally separates communities!')

    # ============ Plot 5: Random Walk Embedding ============
    ax5 = fig.add_subplot(2, 3, 5)

    rw_emb = random_walk_embedding(graph, walk_length=20, n_walks=10, embedding_dim=2)

    ax5.scatter(rw_emb[:, 0], rw_emb[:, 1], c=labels,
               cmap='coolwarm', s=50)
    ax5.set_xlabel('Dimension 1')
    ax5.set_ylabel('Dimension 2')
    ax5.set_title('Random Walk Embedding (DeepWalk-style)\nSimilar nodes → similar walks')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    GRAPH FUNDAMENTALS
    ══════════════════

    GRAPH = (V, E)
    • V: nodes (entities)
    • E: edges (relations)

    REPRESENTATIONS:
    • Adjacency matrix A
    • Degree matrix D
    • Laplacian L = D - A

    KEY PROPERTIES:
    • No fixed size/order
    • Need permutation invariance
    • Aggregate from neighbors

    TASKS:
    ┌────────────────────────┐
    │ Node classification    │
    │ Link prediction        │
    │ Graph classification   │
    │ Node embedding         │
    └────────────────────────┘

    MESSAGE PASSING:
    h_v = f(x_v, AGG(neighbors))
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('GRAPH FUNDAMENTALS — Relational Data\n'
                 'Nodes + Edges → Structure → Learning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def spring_layout(graph, iterations=50):
    """
    Simple force-directed graph layout.

    Nodes repel each other, edges attract.
    """
    n = graph.n_nodes
    pos = np.random.randn(n, 2)

    for _ in range(iterations):
        # Repulsion between all pairs
        disp = np.zeros_like(pos)
        for i in range(n):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist = np.maximum(dist, 0.01)
            disp[i] = np.sum(diff / (dist**2), axis=0)

        # Attraction along edges
        for i, j in graph.get_edge_list():
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            force = diff * dist * 0.01
            disp[i] += force
            disp[j] -= force

        # Update positions
        pos += 0.1 * disp

        # Center
        pos -= pos.mean(axis=0)

    return pos


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run experiments to understand graph properties."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # 1. Karate club properties
    print("\n1. KARATE CLUB GRAPH PROPERTIES")
    print("-" * 40)

    graph, labels = karate_club()
    print(f"Nodes: {graph.n_nodes}")
    print(f"Edges: {len(graph.get_edge_list())}")
    print(f"Average degree: {np.mean(graph.degrees):.2f}")
    print(f"Max degree: {np.max(graph.degrees)} (node {np.argmax(graph.degrees)})")
    print(f"Min degree: {np.min(graph.degrees)}")

    # Spectral gap (difference between first two non-zero eigenvalues)
    eigenvalues, _ = graph.spectral_decomposition()
    spectral_gap = eigenvalues[1]  # First non-zero eigenvalue
    print(f"Spectral gap: {spectral_gap:.4f}")
    print("→ Spectral gap indicates community structure")

    # 2. Community structure detection
    print("\n2. COMMUNITY STRUCTURE (Spectral Clustering)")
    print("-" * 40)

    spec_emb = spectral_embedding(graph, embedding_dim=2)

    # Simple clustering on spectral embedding
    labels_pred = (spec_emb[:, 0] > np.median(spec_emb[:, 0])).astype(int)
    accuracy = max(np.mean(labels_pred == labels), np.mean(labels_pred != labels))
    print(f"Spectral clustering accuracy: {accuracy:.3f}")
    print("→ Laplacian eigenvectors reveal communities!")

    # 3. Effect of p_in/p_out ratio in SBM
    print("\n3. COMMUNITY DETECTABILITY (SBM)")
    print("-" * 40)

    for p_ratio in [2, 5, 10, 20, 50]:
        p_in = 0.3
        p_out = p_in / p_ratio

        graph, labels = create_community_graph(
            n_communities=2, nodes_per_community=50,
            p_in=p_in, p_out=p_out
        )

        spec_emb = spectral_embedding(graph, embedding_dim=1)
        labels_pred = (spec_emb[:, 0] > np.median(spec_emb[:, 0])).astype(int)
        accuracy = max(np.mean(labels_pred == labels), np.mean(labels_pred != labels))

        print(f"p_in/p_out={p_ratio:>3}  accuracy={accuracy:.3f}")

    print("→ Higher ratio = clearer communities = easier detection")

    # 4. Embedding comparison
    print("\n4. EMBEDDING METHODS COMPARISON")
    print("-" * 40)

    graph, labels = karate_club()

    # Spectral embedding
    spec_emb = spectral_embedding(graph, embedding_dim=8)

    # Random walk embedding
    rw_emb = random_walk_embedding(graph, embedding_dim=8)

    # Evaluate by cluster quality
    def evaluate_embedding(emb, labels):
        # Simple metric: average distance between same-class vs different-class
        n = len(labels)
        same_dist, diff_dist = [], []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(emb[i] - emb[j])
                if labels[i] == labels[j]:
                    same_dist.append(d)
                else:
                    diff_dist.append(d)
        return np.mean(diff_dist) / (np.mean(same_dist) + 1e-6)

    spec_quality = evaluate_embedding(spec_emb, labels)
    rw_quality = evaluate_embedding(rw_emb, labels)

    print(f"Spectral embedding quality (ratio): {spec_quality:.3f}")
    print(f"Random walk embedding quality (ratio): {rw_quality:.3f}")
    print("→ Higher ratio = better separation")

    # 5. Scalability
    print("\n5. SCALABILITY (Graph size vs embedding time)")
    print("-" * 40)

    import time

    for n in [50, 100, 200, 500]:
        graph, _ = create_community_graph(
            n_communities=3, nodes_per_community=n//3,
            p_in=0.2, p_out=0.02
        )

        start = time.time()
        _ = spectral_embedding(graph, embedding_dim=8)
        spec_time = time.time() - start

        start = time.time()
        _ = random_walk_embedding(graph, walk_length=10, n_walks=5, embedding_dim=8)
        rw_time = time.time() - start

        print(f"n={n:>4}  spectral={spec_time:.3f}s  random_walk={rw_time:.3f}s")

    print("→ Spectral: O(n³) eigendecomp, Random walk: O(n × walks × length)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GRAPH FUNDAMENTALS — Relational Data")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_graph_fundamentals()
    save_path = '/Users/sid47/ML Algorithms/35_graph_fundamentals.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Graphs: nodes + edges = relational structure
2. Adjacency matrix A, Laplacian L = D - A
3. Tasks: node/link/graph classification, embedding
4. Key challenge: permutation invariance
5. Spectral: eigendecomp reveals structure
6. Random walks: co-occurrence → similarity
    """)
