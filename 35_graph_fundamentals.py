"""
GRAPH FUNDAMENTALS — Paradigm: RELATIONAL DATA

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Data with RELATIONSHIPS between entities.

GRAPH G = (V, E)
- V: nodes (entities) — people, molecules, words, data points
- E: edges (relationships) — friendships, bonds, citations

Node features: X ∈ R^(n × d) — what each entity "looks like"
Adjacency matrix: A ∈ {0,1}^(n × n) — who is connected to whom

===============================================================
WHEN TO USE GRAPHS (The Decision Tree)
===============================================================

Ask yourself:

1. Do your entities have RELATIONSHIPS that matter?
   NO  → Standard ML (MLP, logistic regression, etc.)
   YES ↓

2. Is the structure IRREGULAR (not a grid or sequence)?
   NO  → Grid → CNN, Sequence → RNN/Transformer
   YES ↓

3. Does information FLOW along connections?
   NO  → Maybe just use features, ignore structure
   YES → USE A GRAPH!

EXAMPLES:
- Social network: users + friendships → GRAPH ✓
- Image pixels: regular grid → CNN (not graph)
- Sentence words: sequential → Transformer
- Molecular atoms: irregular bonds → GRAPH ✓
- Citation papers: who-cites-whom → GRAPH ✓
- Point cloud: k-NN on coordinates → GRAPH ✓

===============================================================
HOW TO CONVERT DATA TO GRAPHS (The Thought Process)
===============================================================

STEP 1: WHAT ARE YOUR ENTITIES? → These become NODES
    - Users in a social network
    - Atoms in a molecule
    - Papers in a citation network
    - Data points in a dataset (yes, ANY tabular data!)

STEP 2: WHAT ARE YOUR RELATIONSHIPS? → These become EDGES

    NATURAL edges (already exist in your data):
        - Social connections, chemical bonds, hyperlinks, citations

    CONSTRUCTED edges (you create them):
        - k-NN: connect each point to its k nearest neighbors
        - ε-ball: connect points within distance ε
        - RBF kernel: weighted edges W_ij = exp(-||x_i - x_j||²/2σ²)
        - Correlation: connect features above threshold

STEP 3: WHAT ARE YOUR FEATURES? → These become NODE FEATURES X
    - Raw attributes of each entity
    - One-hot encoding (if no features, use identity matrix)
    - Learned embeddings

STEP 4: WHAT TYPE OF GRAPH?
    - Directed vs Undirected
    - Weighted vs Unweighted
    - Homogeneous vs Heterogeneous (multiple node/edge types)
    - Static vs Dynamic (changes over time)

===============================================================
GRAPH REPRESENTATIONS
===============================================================

1. ADJACENCY MATRIX A ∈ {0,1}^(n×n)
   A[i,j] = 1 if edge from i to j
   Dense: O(n²) space, fast matrix operations

2. EDGE LIST: [(i,j), (k,l), ...]
   Sparse: O(|E|) space, good for large sparse graphs

3. ADJACENCY LIST: {i: [j,k], j: [i,l], ...}
   Fast neighbor lookup: O(degree) per query

KEY MATRICES:
- Degree matrix D: D_ii = Σ_j A_ij (diagonal)
- Graph Laplacian: L = D - A
- Normalized adjacency: Ã = D^(-1/2) A D^(-1/2)
- With self-loops: Ã = D̃^(-1/2)(A+I)D̃^(-1/2)

===============================================================
GRAPH TASKS
===============================================================

1. NODE CLASSIFICATION — predict label per node
   "Is this user a bot?" "What topic is this paper?"

2. LINK PREDICTION — predict if edge should exist
   "Will these users become friends?" "Is this drug-target pair valid?"

3. GRAPH CLASSIFICATION — predict label per graph
   "Is this molecule toxic?" "What protein family?"

===============================================================
TRAIN/TEST ON GRAPHS (It's Different!)
===============================================================

Standard ML: split DATA into train/test → model sees only train data.

On graphs, it's more subtle because nodes are CONNECTED:

1. TRANSDUCTIVE NODE CLASSIFICATION (most common)
   - ALL nodes are in ONE graph
   - We MASK labels: only train node labels used for loss
   - Test nodes' FEATURES are visible during training!
   - The graph structure connects train ↔ test nodes
   - This is SEMI-SUPERVISED: structure propagates labels

   Why this works: if your friend is a bot, you might be too.
   Message passing uses ALL neighbors, including unlabeled ones.

2. INDUCTIVE NODE CLASSIFICATION
   - Train on graph A, test on DIFFERENT graph B
   - Model must learn a GENERAL aggregation function
   - Can't memorize node-specific patterns
   - GraphSAGE (37) designed for this

3. LINK PREDICTION
   - Split EDGES into train/test
   - Train: use observed edges, predict held-out edges
   - Need NEGATIVE SAMPLING: sample non-edges as negatives

4. GRAPH CLASSIFICATION
   - Split GRAPHS into train/test sets
   - Each graph is an independent sample
   - Just like standard ML splitting

===============================================================
WHY GRAPHS NEED SPECIAL ARCHITECTURE
===============================================================

Graphs have NO FIXED SIZE, NO FIXED ORDER!

PROBLEMS WITH STANDARD ML:
1. Can't flatten to vector (loses structure)
2. Can't use CNN (no grid)
3. Can't use RNN (no sequence order)
4. Need PERMUTATION INVARIANT/EQUIVARIANT operations

THE KEY INSIGHT: Aggregate information from neighbors
    h_v = f(x_v, AGGREGATE({x_u : u ∈ N(v)}))

This is the foundation of ALL Graph Neural Networks.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


# ============================================================
# GRAPH DATA STRUCTURE
# ============================================================

class Graph:
    """
    Graph data structure for Graph Neural Networks.

    Stores:
    - Adjacency matrix A (n × n)
    - Node features X (n × d)
    - Edge list [(i, j), ...]
    """

    def __init__(self, n_nodes, node_features=None, directed=False):
        """
        Parameters:
        -----------
        n_nodes : int
            Number of nodes
        node_features : ndarray (n_nodes, d) or None
            Node feature matrix. If None, uses one-hot identity.
        directed : bool
            If False, edges are added in both directions.
        """
        self.n_nodes = n_nodes
        self.directed = directed
        self.A = np.zeros((n_nodes, n_nodes))  # Adjacency matrix
        self.edge_list = []

        if node_features is not None:
            self.X = node_features.copy()
        else:
            self.X = np.eye(n_nodes)  # One-hot if no features

    def add_edge(self, i, j, weight=1.0):
        """Add edge between nodes i and j."""
        self.A[i, j] = weight
        self.edge_list.append((i, j))
        if not self.directed:
            self.A[j, i] = weight
            self.edge_list.append((j, i))

    def neighbors(self, node):
        """Get neighbors of a node."""
        return np.where(self.A[node] > 0)[0]

    def degree_matrix(self):
        """Diagonal degree matrix D where D_ii = sum of row i."""
        return np.diag(np.sum(self.A, axis=1))

    def degrees(self):
        """Degree of each node."""
        return np.sum(self.A, axis=1)

    def laplacian(self):
        """Graph Laplacian L = D - A."""
        return self.degree_matrix() - self.A

    def normalized_adjacency(self, self_loops=True):
        """
        Normalized adjacency: Ã = D̃^(-1/2)(A+I)D̃^(-1/2)

        This is what GCN uses. Self-loops ensure a node
        retains its own features during message passing.
        """
        A = self.A.copy()
        if self_loops:
            A = A + np.eye(self.n_nodes)

        d = np.sum(A, axis=1)
        d_inv_sqrt = np.zeros_like(d)
        nonzero = d > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)

        return D_inv_sqrt @ A @ D_inv_sqrt

    def n_edges(self):
        """Number of edges (undirected = count once)."""
        if self.directed:
            return int(np.sum(self.A > 0))
        else:
            return int(np.sum(self.A > 0)) // 2

    def copy(self):
        """Create a copy of this graph."""
        g = Graph(self.n_nodes, self.X.copy(), self.directed)
        g.A = self.A.copy()
        g.edge_list = list(self.edge_list)
        return g


# ============================================================
# GRAPH DATASETS
# ============================================================

def karate_club():
    """
    Zachary's Karate Club — the classic small graph dataset.

    34 members of a karate club, split into 2 factions
    after a dispute between instructor (node 0) and
    president (node 33).

    Returns: (Graph, labels)
    """
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
        (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),
        (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
        (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
        (3,7),(3,12),(3,13),
        (4,6),(4,10),
        (5,6),(5,10),(5,16),
        (6,16),
        (8,30),(8,32),(8,33),
        (9,33),
        (13,33),
        (14,32),(14,33),
        (15,32),(15,33),
        (18,32),(18,33),
        (19,33),
        (20,32),(20,33),
        (22,32),(22,33),
        (23,25),(23,27),(23,29),(23,32),(23,33),
        (24,25),(24,27),(24,31),
        (25,31),
        (26,29),(26,33),
        (27,33),
        (28,31),(28,33),
        (29,32),(29,33),
        (30,32),(30,33),
        (31,32),(31,33),
        (32,33),
    ]

    n_nodes = 34
    graph = Graph(n_nodes)

    for i, j in edges:
        graph.add_edge(i, j)

    # Ground truth: two factions
    labels = np.array([
        0,0,0,0,0,0,0,0,1,1,
        0,0,0,0,1,1,0,0,1,0,
        1,0,1,1,1,1,1,1,1,1,
        1,1,1,1
    ])

    return graph, labels


def create_community_graph(n_communities=3, nodes_per_community=20,
                           p_in=0.3, p_out=0.02, feature_dim=16,
                           random_state=42):
    """
    Stochastic Block Model — communities with dense internal connections.

    Parameters:
    -----------
    n_communities : int
        Number of communities
    nodes_per_community : int
        Nodes per community
    p_in : float
        Probability of edge within community (high = dense)
    p_out : float
        Probability of edge between communities (low = separated)
    feature_dim : int
        Dimension of node features (correlated with community label)
    random_state : int
        Random seed

    Returns: (Graph, labels)
    """
    np.random.seed(random_state)
    n_nodes = n_communities * nodes_per_community

    # Create node features correlated with community
    prototypes = np.random.randn(n_communities, feature_dim) * 2
    X = np.zeros((n_nodes, feature_dim))
    labels = np.zeros(n_nodes, dtype=int)

    for c in range(n_communities):
        start = c * nodes_per_community
        end = start + nodes_per_community
        labels[start:end] = c
        X[start:end] = prototypes[c] + np.random.randn(nodes_per_community, feature_dim) * 0.5

    graph = Graph(n_nodes, X)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if labels[i] == labels[j]:
                if np.random.rand() < p_in:
                    graph.add_edge(i, j)
            else:
                if np.random.rand() < p_out:
                    graph.add_edge(i, j)

    return graph, labels


def create_citation_network(n_papers=100, n_classes=3, feature_dim=16,
                            p_cite_same=0.15, p_cite_diff=0.02,
                            random_state=42):
    """
    Synthetic citation network — papers cite similar papers more.

    Node features are "bag of words" style: correlated with topic.

    Returns: (Graph, labels)
    """
    np.random.seed(random_state)

    labels = np.random.randint(0, n_classes, n_papers)

    # Feature vectors: topic-correlated
    X = np.random.rand(n_papers, feature_dim) * 0.3
    for c in range(n_classes):
        mask = labels == c
        k = feature_dim // n_classes
        X[mask, c*k:(c+1)*k] += np.random.rand(np.sum(mask), k) * 2

    graph = Graph(n_papers, X, directed=False)

    for i in range(n_papers):
        for j in range(i + 1, n_papers):
            p = p_cite_same if labels[i] == labels[j] else p_cite_diff
            if np.random.rand() < p:
                graph.add_edge(i, j)

    return graph, labels


def create_molecular_dataset(n_molecules=60, n_classes=3, random_state=42):
    """
    Synthetic molecular graphs — small graphs with structural patterns.

    Class 0: CHAIN molecules (linear paths)
    Class 1: RING molecules (cycles)
    Class 2: STAR molecules (hub-and-spoke)

    Returns: list of (Graph, label) pairs
    """
    np.random.seed(random_state)
    molecules = []
    per_class = n_molecules // n_classes

    for c in range(n_classes):
        for _ in range(per_class):
            if c == 0:
                # CHAIN: linear path
                n = np.random.randint(4, 9)
                X = np.random.randn(n, 4) * 0.5
                X[:, 0] += 1.0
                g = Graph(n, X)
                for i in range(n - 1):
                    g.add_edge(i, i + 1)

            elif c == 1:
                # RING: cycle
                n = np.random.randint(4, 8)
                X = np.random.randn(n, 4) * 0.5
                X[:, 1] += 1.0
                g = Graph(n, X)
                for i in range(n):
                    g.add_edge(i, (i + 1) % n)

            else:
                # STAR: hub-and-spoke
                branches = np.random.randint(3, 7)
                n = 1 + branches
                X = np.random.randn(n, 4) * 0.5
                X[:, 2] += 1.0
                X[0, 3] += 2.0
                g = Graph(n, X)
                for i in range(1, n):
                    g.add_edge(0, i)

            molecules.append((g, c))

    np.random.shuffle(molecules)
    return molecules


def create_tree_graph(depth=3, branching=2, feature_dim=4, random_state=42):
    """
    Tree graph — for testing message passing range.

    Root is labeled 1, leaves are labeled 0.
    A GNN needs 'depth' layers to propagate root info to leaves.

    Returns: (Graph, labels)
    """
    np.random.seed(random_state)

    n_nodes = sum(branching**d for d in range(depth + 1))
    X = np.random.randn(n_nodes, feature_dim) * 0.5
    graph = Graph(n_nodes, X)
    labels = np.zeros(n_nodes, dtype=int)
    labels[0] = 1

    # Build tree level by level
    current_idx = 0
    for d in range(depth):
        n_at_level = branching ** d
        next_start = sum(branching**dd for dd in range(d + 1))
        for i in range(n_at_level):
            parent = current_idx + i
            for b in range(branching):
                child = next_start + (i * branching + b)
                if child < n_nodes:
                    graph.add_edge(parent, child)
        current_idx = next_start

    return graph, labels


# ============================================================
# DATA-TO-GRAPH CONVERSION
# ============================================================

def tabular_to_graph(X, method='knn', k=5, sigma=1.0, epsilon=None):
    """
    Convert tabular data to a graph.

    THIS IS THE KEY FUNCTION for understanding graph learning!

    Given n data points in R^d, create a graph where:
    - Each data point becomes a node
    - Edges connect "similar" points
    - Node features = original data point features

    Parameters:
    -----------
    X : ndarray (n, d)
        Data matrix
    method : str
        'knn' — connect each point to k nearest neighbors
        'rbf' — weighted edges using RBF kernel
        'epsilon' — connect points within distance epsilon
    k : int
        Number of neighbors for k-NN
    sigma : float
        RBF kernel bandwidth
    epsilon : float or None
        Distance threshold for epsilon-ball method

    Returns: Graph
    """
    n = X.shape[0]

    # Compute pairwise distances
    X_sq = np.sum(X**2, axis=1)
    D = np.sqrt(np.maximum(X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T, 0))

    graph = Graph(n, X)

    if method == 'knn':
        for i in range(n):
            dists = D[i].copy()
            dists[i] = np.inf
            knn = np.argsort(dists)[:k]
            for j in knn:
                graph.A[i, j] = 1
                graph.A[j, i] = 1

    elif method == 'rbf':
        W = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(W, 0)
        graph.A = W

    elif method == 'epsilon':
        if epsilon is None:
            epsilon = np.median(D[D > 0])
        graph.A = (D < epsilon).astype(float)
        np.fill_diagonal(graph.A, 0)

    return graph


# ============================================================
# TRAIN/TEST SPLITTING UTILITIES
# ============================================================

def create_transductive_split(n_nodes, labels, train_ratio=0.1,
                              val_ratio=0.1, random_state=42):
    """
    Create transductive train/val/test masks for node classification.

    TRANSDUCTIVE: All nodes are in the same graph.
    We only hide LABELS, not nodes or edges.

    Returns: (train_mask, val_mask, test_mask) as boolean arrays
    """
    np.random.seed(random_state)

    n_classes = len(np.unique(labels))
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        n_train = max(1, int(len(class_idx) * train_ratio))
        n_val = max(1, int(len(class_idx) * val_ratio))

        train_mask[class_idx[:n_train]] = True
        val_mask[class_idx[n_train:n_train + n_val]] = True
        test_mask[class_idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def create_link_split(graph, test_ratio=0.2, random_state=42):
    """
    Split edges for link prediction.

    Returns:
    - train_graph: Graph with test edges removed
    - test_edges_pos: held-out edges
    - test_edges_neg: sampled non-edges
    """
    np.random.seed(random_state)

    edges = []
    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            if graph.A[i, j] > 0:
                edges.append((i, j))

    edges = np.array(edges)
    n_test = max(1, int(len(edges) * test_ratio))

    perm = np.random.permutation(len(edges))
    test_idx = perm[:n_test]

    test_edges_pos = [tuple(edges[i]) for i in test_idx]

    train_graph = graph.copy()
    for i, j in test_edges_pos:
        train_graph.A[i, j] = 0
        train_graph.A[j, i] = 0

    test_edges_neg = []
    while len(test_edges_neg) < n_test:
        i = np.random.randint(graph.n_nodes)
        j = np.random.randint(graph.n_nodes)
        if i != j and graph.A[i, j] == 0 and (i, j) not in test_edges_neg:
            test_edges_neg.append((i, j))

    return train_graph, test_edges_pos, test_edges_neg


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

def spring_layout(graph, n_iter=50, seed=42):
    """
    Simple spring layout for graph visualization.

    Attractive force between connected nodes,
    repulsive force between all nodes.
    """
    np.random.seed(seed)
    n = graph.n_nodes
    pos = np.random.randn(n, 2) * 0.5

    k = 1.0 / np.sqrt(n)

    for _ in range(n_iter):
        forces = np.zeros((n, 2))

        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = max(np.linalg.norm(diff), 0.01)
                force = k**2 / dist * diff / dist
                forces[i] += force
                forces[j] -= force

        for i in range(n):
            for j in graph.neighbors(i):
                if j > i:
                    diff = pos[j] - pos[i]
                    dist = max(np.linalg.norm(diff), 0.01)
                    force = dist / k * diff / dist
                    forces[i] += force
                    forces[j] -= force

        pos += 0.1 * forces

    pos -= pos.mean(axis=0)
    return pos


def draw_graph(graph, labels=None, pos=None, ax=None, title='',
               node_size=80, edge_alpha=0.3, cmap='tab10',
               highlight_mask=None):
    """Draw a graph with nodes, edges, and optional coloring."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if pos is None:
        pos = spring_layout(graph)

    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            if graph.A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                       'gray', alpha=edge_alpha, linewidth=0.5)

    if labels is not None:
        colors = labels
    else:
        colors = np.zeros(graph.n_nodes)

    if highlight_mask is not None:
        faded = ~highlight_mask
        ax.scatter(pos[faded, 0], pos[faded, 1], c=colors[faded],
                  cmap=cmap, s=node_size, alpha=0.15, edgecolors='gray',
                  linewidths=0.5, zorder=2)
        ax.scatter(pos[highlight_mask, 0], pos[highlight_mask, 1],
                  c=colors[highlight_mask], cmap=cmap, s=node_size,
                  alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, cmap=cmap,
                  s=node_size, alpha=0.8, edgecolors='black',
                  linewidths=0.5, zorder=2)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')


def spectral_embedding(graph, dim=2):
    """
    Embed graph nodes using Laplacian eigenvectors.

    Connection to 58_spectral_clustering.py:
    The smallest eigenvectors of the Laplacian encode
    the graph's cluster structure.
    """
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    embedding = eigenvectors[:, idx[1:dim+1]]
    return embedding


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # -------- Experiment 1: Data-to-Graph Conversion --------
    print("\n1. DATA-TO-GRAPH CONVERSION")
    print("-" * 40)
    print("How does k in k-NN affect the graph?")

    n = 100
    t = np.linspace(0, np.pi, n // 2)
    X_top = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n//2, 2) * 0.1
    X_bot = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + np.random.randn(n//2, 2) * 0.1
    X_moons = np.vstack([X_top, X_bot])

    for k in [3, 5, 10, 20, 40]:
        g = tabular_to_graph(X_moons, method='knn', k=k)
        n_edges = int(np.sum(g.A > 0)) // 2
        avg_degree = np.mean(g.degrees())

        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(g.neighbors(node))
        connected = len(visited) == g.n_nodes

        print(f"  k={k:<4}  edges={n_edges:<6}  avg_degree={avg_degree:.1f}"
              f"  connected={connected}")

    print("-> Small k: sparse, may disconnect")
    print("-> Large k: dense, loses local structure")
    print("-> Sweet spot: k=5-10 for most datasets")

    # -------- Experiment 2: Community Graph Density --------
    print("\n2. COMMUNITY GRAPH DENSITY")
    print("-" * 40)
    print("When are communities detectable?")

    for p_in in [0.1, 0.2, 0.3, 0.5]:
        for p_out in [0.01, 0.05, 0.1]:
            g, labels = create_community_graph(
                n_communities=2, nodes_per_community=20,
                p_in=p_in, p_out=p_out, random_state=42
            )
            n_intra = 0
            n_total = 0
            for i in range(g.n_nodes):
                for j in range(i+1, g.n_nodes):
                    if g.A[i,j] > 0:
                        n_total += 1
                        if labels[i] == labels[j]:
                            n_intra += 1
            frac = n_intra / max(n_total, 1)
            ratio = p_in / max(p_out, 0.001)
            print(f"  p_in={p_in:.1f}  p_out={p_out:.2f}"
                  f"  ratio={ratio:>5.1f}  intra_frac={frac:.2f}")

    print("-> High p_in/p_out ratio → clear communities")
    print("-> When ratio ≈ 1 → random graph, no structure")

    # -------- Experiment 3: Train/Test Split Types --------
    print("\n3. TRAIN/TEST SPLITTING ON GRAPHS")
    print("-" * 40)

    g, labels = create_community_graph(
        n_communities=3, nodes_per_community=20,
        p_in=0.3, p_out=0.02, random_state=42
    )

    train_mask, val_mask, test_mask = create_transductive_split(
        g.n_nodes, labels, train_ratio=0.1, val_ratio=0.1
    )

    print(f"  Total nodes: {g.n_nodes}")
    print(f"  Train: {np.sum(train_mask)} ({np.mean(train_mask)*100:.0f}%)")
    print(f"  Val:   {np.sum(val_mask)} ({np.mean(val_mask)*100:.0f}%)")
    print(f"  Test:  {np.sum(test_mask)} ({np.mean(test_mask)*100:.0f}%)")
    print(f"  Total edges: {g.n_edges()}")

    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        class_counts = [np.sum(labels[mask] == c) for c in range(3)]
        print(f"  {name} class distribution: {class_counts}")

    print("\n  TRANSDUCTIVE KEY INSIGHT:")
    print("  Train nodes can see test nodes' FEATURES (not labels)")
    print("  Message passing uses ALL neighbors, even unlabeled ones")

    train_g, pos_edges, neg_edges = create_link_split(g, test_ratio=0.2)
    print(f"\n  Link prediction split:")
    print(f"    Train edges: {train_g.n_edges()}")
    print(f"    Test positive edges: {len(pos_edges)}")
    print(f"    Test negative edges: {len(neg_edges)}")

    # -------- Experiment 4: Feature Correlation --------
    print("\n4. FEATURE CORRELATION WITH LABELS")
    print("-" * 40)

    for feat_dim in [2, 8, 16, 32]:
        g, labels = create_community_graph(
            n_communities=3, nodes_per_community=20,
            feature_dim=feat_dim, random_state=42
        )
        sim_in, sim_out, n_in, n_out = 0, 0, 0, 0
        for i in range(min(g.n_nodes, 30)):
            for j in range(i+1, min(g.n_nodes, 30)):
                norm_i = np.linalg.norm(g.X[i])
                norm_j = np.linalg.norm(g.X[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos = (g.X[i] @ g.X[j]) / (norm_i * norm_j)
                else:
                    cos = 0.0
                if labels[i] == labels[j]:
                    sim_in += cos
                    n_in += 1
                else:
                    sim_out += cos
                    n_out += 1
        avg_in = sim_in / max(n_in, 1)
        avg_out = sim_out / max(n_out, 1)
        print(f"  feat_dim={feat_dim:<4}  intra_sim={avg_in:.3f}"
              f"  inter_sim={avg_out:.3f}  gap={avg_in - avg_out:.3f}")

    print("-> Higher feature dim → clearer feature signal")
    print("-> GNNs can exploit BOTH feature similarity AND graph structure")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_data_to_graph():
    """
    THE KEY VISUALIZATION: How tabular data becomes a graph.
    Shows: scatter plot → k-NN graph → adjacency matrix → eigenspace
    """
    print("\nGenerating: Data-to-Graph conversion visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    np.random.seed(42)
    n = 80
    t = np.linspace(0, np.pi, n // 2)
    X_top = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n//2, 2) * 0.08
    X_bot = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + np.random.randn(n//2, 2) * 0.08
    X = np.vstack([X_top, X_bot])
    y = np.array([0]*(n//2) + [1]*(n//2))

    # Panel 1: Raw scatter plot
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=40, alpha=0.8,
              edgecolors='black', linewidths=0.5)
    ax.set_title('Step 1: Raw Data\n(tabular: each row = point)')
    ax.set_aspect('equal')

    # Panel 2: k-NN graph (k=5)
    ax = axes[0, 1]
    g5 = tabular_to_graph(X, method='knn', k=5)
    for i in range(n):
        for j in range(i+1, n):
            if g5.A[i,j] > 0:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]],
                       'gray', alpha=0.15, linewidth=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=40, alpha=0.8,
              edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_title('Step 2: Build k-NN Graph\n(k=5, each point → 5 neighbors)')
    ax.set_aspect('equal')

    # Panel 3: Adjacency matrix
    ax = axes[0, 2]
    sort_idx = np.argsort(y)
    A_sorted = g5.A[sort_idx][:, sort_idx]
    ax.imshow(A_sorted, cmap='Blues', aspect='auto')
    ax.set_title('Step 3: Adjacency Matrix\n(block structure = classes)')
    ax.set_xlabel('Node (sorted by class)')
    ax.set_ylabel('Node')

    # Panel 4: Eigenspace
    ax = axes[0, 3]
    emb = spectral_embedding(g5, dim=2)
    ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='coolwarm', s=40, alpha=0.8,
              edgecolors='black', linewidths=0.5)
    ax.set_title('Step 4: Eigenspace\n(classes become separable!)')
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')

    # Row 2: Different k values
    for idx, k in enumerate([3, 5, 10, 30]):
        ax = axes[1, idx]
        g = tabular_to_graph(X, method='knn', k=k)
        for i in range(n):
            for j in range(i+1, n):
                if g.A[i,j] > 0:
                    ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]],
                           'gray', alpha=0.1, linewidth=0.5)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, alpha=0.8,
                  edgecolors='black', linewidths=0.3, zorder=3)
        n_edges = int(np.sum(g.A > 0)) // 2
        ax.set_title(f'k={k} ({n_edges} edges)\n'
                     f'{"Too sparse" if k == 3 else "Good" if k in [5,10] else "Too dense"}')
        ax.set_aspect('equal')

    plt.suptitle('HOW TO CONVERT DATA TO GRAPHS\n'
                 'Row 1: The pipeline | Row 2: Effect of k in k-NN',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_graph_tasks():
    """Show the three main graph learning tasks."""
    print("\nGenerating: Graph tasks visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    np.random.seed(42)

    # Task 1: Node Classification
    ax = axes[0]
    g, labels = create_community_graph(
        n_communities=3, nodes_per_community=10,
        p_in=0.4, p_out=0.03, random_state=42
    )
    pos = spring_layout(g, seed=42)
    draw_graph(g, labels, pos, ax,
              title='NODE CLASSIFICATION\n"What class is each node?"',
              cmap='Set1')

    # Task 2: Link Prediction
    ax = axes[1]
    g2, labels2 = create_community_graph(
        n_communities=2, nodes_per_community=12,
        p_in=0.3, p_out=0.05, random_state=42
    )
    pos2 = spring_layout(g2, seed=42)
    for i in range(g2.n_nodes):
        for j in range(i+1, g2.n_nodes):
            if g2.A[i,j] > 0:
                ax.plot([pos2[i,0], pos2[j,0]], [pos2[i,1], pos2[j,1]],
                       'gray', alpha=0.3, linewidth=0.5)
    ax.scatter(pos2[:, 0], pos2[:, 1], c=labels2, cmap='Set1',
              s=80, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)
    for i, j in [(0, 5), (3, 8), (10, 20)]:
        if i < g2.n_nodes and j < g2.n_nodes:
            ax.plot([pos2[i,0], pos2[j,0]], [pos2[i,1], pos2[j,1]],
                   'r--', alpha=0.8, linewidth=2, zorder=2)
    ax.set_title('LINK PREDICTION\n"Should this edge exist?"')
    ax.set_aspect('equal')
    ax.axis('off')

    # Task 3: Graph Classification
    ax = axes[2]
    molecules = create_molecular_dataset(n_molecules=6, n_classes=3, random_state=42)
    class_names = ['Chain', 'Ring', 'Star']

    for idx, (mol, label) in enumerate(molecules[:6]):
        row = idx // 3
        col = idx % 3
        offset_x = col * 3
        offset_y = -row * 3

        mol_pos = spring_layout(mol, seed=idx)
        mol_pos = mol_pos * 0.8
        mol_pos[:, 0] += offset_x
        mol_pos[:, 1] += offset_y

        for i in range(mol.n_nodes):
            for j in range(i+1, mol.n_nodes):
                if mol.A[i,j] > 0:
                    ax.plot([mol_pos[i,0], mol_pos[j,0]],
                           [mol_pos[i,1], mol_pos[j,1]],
                           'gray', alpha=0.5, linewidth=1)

        colors_list = ['#e74c3c', '#2ecc71', '#3498db']
        ax.scatter(mol_pos[:, 0], mol_pos[:, 1], c=colors_list[label],
                  s=50, alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)
        ax.text(offset_x, offset_y - 1.2, class_names[label],
               ha='center', fontsize=8, color=colors_list[label])

    ax.set_title('GRAPH CLASSIFICATION\n"What type of graph?"')
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('THE THREE GRAPH LEARNING TASKS',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_train_test_split():
    """Visualize transductive train/test splitting."""
    print("\nGenerating: Train/test split visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    g, labels = create_community_graph(
        n_communities=3, nodes_per_community=15,
        p_in=0.35, p_out=0.03, random_state=42
    )
    pos = spring_layout(g, seed=42)
    train_mask, val_mask, test_mask = create_transductive_split(
        g.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    # Panel 1: All labels visible
    draw_graph(g, labels, pos, axes[0],
              title='Ground Truth\n(all labels visible)', cmap='Set1')

    # Panel 2: Transductive — only train labels visible
    ax = axes[1]
    for i in range(g.n_nodes):
        for j in range(i+1, g.n_nodes):
            if g.A[i,j] > 0:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                       'gray', alpha=0.2, linewidth=0.5)

    ax.scatter(pos[train_mask, 0], pos[train_mask, 1],
              c=labels[train_mask], cmap='Set1', s=120,
              alpha=0.9, edgecolors='black', linewidths=1.5, zorder=3,
              marker='s', label='Train (labeled)')

    unlabeled = ~train_mask
    ax.scatter(pos[unlabeled, 0], pos[unlabeled, 1],
              c='lightgray', s=60, alpha=0.6, edgecolors='gray',
              linewidths=0.5, zorder=2, label='Unlabeled')

    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f'Transductive Split\n({np.sum(train_mask)} labeled, '
                 f'{np.sum(unlabeled)} unlabeled)')
    ax.set_aspect('equal')
    ax.axis('off')

    # Panel 3: What the model "sees" — message passing view
    ax = axes[2]
    for i in range(g.n_nodes):
        for j in range(i+1, g.n_nodes):
            if g.A[i,j] > 0:
                if train_mask[i] and train_mask[j]:
                    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                           'green', alpha=0.4, linewidth=1)
                elif train_mask[i] or train_mask[j]:
                    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                           'orange', alpha=0.3, linewidth=0.8)
                else:
                    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                           'gray', alpha=0.15, linewidth=0.5)

    ax.scatter(pos[train_mask, 0], pos[train_mask, 1],
              c=labels[train_mask], cmap='Set1', s=120,
              alpha=0.9, edgecolors='black', linewidths=1.5, zorder=3,
              marker='s')
    ax.scatter(pos[unlabeled, 0], pos[unlabeled, 1],
              c='lightgray', s=60, alpha=0.6, edgecolors='gray',
              linewidths=0.5, zorder=2)

    ax.set_title('Message Passing View\nGreen=train-train, Orange=train-test\n'
                 'Labels flow through structure!')
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('TRANSDUCTIVE TRAIN/TEST: Structure connects labeled and unlabeled',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_fundamentals():
    """Benchmark graph statistics across datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Graph Dataset Statistics")
    print("="*60)

    datasets = {
        'karate_club': karate_club(),
        'community_2': create_community_graph(2, 20, 0.3, 0.02),
        'community_3': create_community_graph(3, 20, 0.3, 0.02),
        'community_4': create_community_graph(4, 15, 0.3, 0.02),
        'citation': create_citation_network(100, 3, 16),
    }

    print(f"\n{'Dataset':<15} {'Nodes':<8} {'Edges':<8} {'Avg Deg':<10}"
          f" {'Classes':<9} {'Feat Dim':<9}")
    print("-" * 60)

    for name, (g, labels) in datasets.items():
        n_classes = len(np.unique(labels))
        avg_deg = np.mean(g.degrees())
        feat_dim = g.X.shape[1]
        print(f"{name:<15} {g.n_nodes:<8} {g.n_edges():<8} {avg_deg:<10.1f}"
              f" {n_classes:<9} {feat_dim:<9}")

    molecules = create_molecular_dataset(60, 3)
    sizes = [m[0].n_nodes for m in molecules]
    print(f"\n{'mol_dataset':<15} {len(molecules)} graphs, "
          f"avg size={np.mean(sizes):.1f}, min={min(sizes)}, max={max(sizes)}")

    return datasets


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GRAPH FUNDAMENTALS — Paradigm: RELATIONAL DATA")
    print("="*60)

    print("""
THE THOUGHT PROCESS FOR GRAPH LEARNING:

    1. ENTITIES → Nodes (users, atoms, papers, data points)
    2. RELATIONSHIPS → Edges (natural or constructed)
    3. FEATURES → Node feature matrix X

    The key question: "Does the STRUCTURE contain information
    that helps with the task?"

    If YES → Graph Neural Network
    If NO  → Standard ML (just use features)

TRAIN/TEST ON GRAPHS — IT'S DIFFERENT:

    TRANSDUCTIVE (most common):
        All nodes in ONE graph, but labels are MASKED.
        Train nodes: labeled → compute loss
        Test nodes: unlabeled → predict
        Message passing uses ALL nodes (even unlabeled!)

    INDUCTIVE:
        Train on graph A, test on graph B.
        Model must learn GENERAL aggregation.

    GRAPH CLASSIFICATION:
        Each graph is one sample.
        Standard train/test split on graphs.
    """)

    ablation_experiments()
    benchmark_fundamentals()

    print("\nGenerating visualizations...")

    fig1 = visualize_data_to_graph()
    save_path1 = '/Users/sid47/ML Algorithms/35_graph_fundamentals.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_graph_tasks()
    save_path2 = '/Users/sid47/ML Algorithms/35_graph_tasks.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_train_test_split()
    save_path3 = '/Users/sid47/ML Algorithms/35_train_test.png'
    fig3.savefig(save_path3, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    print("\n" + "="*60)
    print("SUMMARY: What Graph Fundamentals Reveals")
    print("="*60)
    print("""
1. GRAPHS = ENTITIES + RELATIONSHIPS
   Any data with meaningful connections can be a graph.

2. DATA-TO-GRAPH CONVERSION
   Even tabular data can become a graph (k-NN, RBF).
   The key: does structure help the task?

3. TRAIN/TEST IS DIFFERENT
   Transductive: labels masked, structure shared
   Inductive: separate graphs
   This is why GNNs are semi-supervised!

4. THREE TASKS
   Node classification → per-node labels
   Link prediction → predict missing edges
   Graph classification → per-graph labels

5. WHY SPECIAL ARCHITECTURE?
   No grid → can't use CNN
   No sequence → can't use RNN
   Variable size → need permutation invariance
   Solution: AGGREGATE from neighbors (→ GNNs)

NEXT: 36_gcn.py — The first Graph Neural Network!
    """)
