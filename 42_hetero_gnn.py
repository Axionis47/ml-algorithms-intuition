"""
HETEROGENEOUS GRAPH NEURAL NETWORK (R-GCN) — Paradigm: MULTIPLE NODE/EDGE TYPES

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Real-world graphs are rarely homogeneous. They contain DIFFERENT
types of nodes and DIFFERENT types of edges (relations):

    - Knowledge graphs: (entity, relation, entity)
      e.g., (Einstein, born_in, Ulm), (Einstein, field, Physics)

    - Academic networks: papers, authors, venues
      edges: writes, cites, published_in

    - Social networks: users, posts, pages
      edges: follows, likes, shares

HETEROGENEOUS GRAPH:
    G = (V, E, tau_v, tau_e)
    tau_v: node type function (maps node -> type)
    tau_e: edge type function (maps edge -> relation)

===============================================================
R-GCN: RELATIONAL GRAPH CONVOLUTIONAL NETWORK
===============================================================

Key insight: Different relations need DIFFERENT transformations.

Standard GCN: H' = sigma(A_hat H W)  -- ONE weight matrix W
R-GCN:        h_i' = sigma( SUM_r SUM_{j in N_r(i)} (1/c_{i,r}) W_r h_j + W_0 h_i )

Where:
    r: relation type index
    N_r(i): neighbors of i via relation r
    W_r: weight matrix SPECIFIC to relation r
    W_0: self-loop weight matrix
    c_{i,r}: normalization constant (e.g., |N_r(i)|)

WHAT THIS DOES:
    For each relation type, apply a DIFFERENT linear transform.
    Then SUM the contributions from all relations.
    This lets the model learn that "cites" and "writes" edges
    carry fundamentally different types of information.

===============================================================
THE PARAMETER EXPLOSION PROBLEM
===============================================================

If we have R relations, each W_r is (d_in x d_out).
Total parameters per layer: R x d_in x d_out

With many relations (knowledge graphs have 100s!), this EXPLODES.

SOLUTION: BASIS DECOMPOSITION

    W_r = SUM_b=1..B  a_rb * B_b

Where:
    B_b: shared basis matrices (B of them, each d_in x d_out)
    a_rb: scalar coefficients (R x B matrix)

Parameters: B x d_in x d_out + R x B
Much fewer when B << R!

Trade-off:
    B = R: Full per-relation weights (most expressive)
    B = 1: All relations share ONE transform (least expressive)
    B in between: Controlled expressiveness

===============================================================
KNOWLEDGE GRAPHS (ENTITY-RELATION-ENTITY)
===============================================================

Knowledge graphs are the canonical use case for R-GCN:
    - Nodes = entities (people, places, concepts)
    - Edges = typed relations (born_in, capital_of, part_of)
    - Task: Link prediction or entity classification

Example triples:
    (Paris, capital_of, France)
    (France, continent, Europe)
    (Eiffel_Tower, located_in, Paris)

R-GCN can learn that:
    "capital_of" transforms city features -> country features
    "located_in" transforms landmark features -> city features

===============================================================
WHEN TO USE HETERO GNNs vs REGULAR GNNs
===============================================================

Use REGULAR GCN/GAT/GIN when:
    - All nodes are same type (molecules, social network of users)
    - All edges mean the same thing (bonded, connected)
    - Simpler model is sufficient

Use HETEROGENEOUS (R-GCN) when:
    - Multiple node types (papers + authors + venues)
    - Multiple edge/relation types (cites, writes, published_in)
    - Relations carry DIFFERENT semantics
    - Knowledge graph tasks

===============================================================
INDUCTIVE BIAS
===============================================================

1. RELATION-SPECIFIC TRANSFORMS: Different edge types = different info
2. SHARED AGGREGATION: Same mechanism across all relation types
3. LOCAL + TYPED: Each layer aggregates 1-hop typed neighbors
4. HOMOPHILY per relation: nodes connected by same relation are similar

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
graph_module = import_module('35_graph_fundamentals')
Graph = graph_module.Graph
karate_club = graph_module.karate_club
create_community_graph = graph_module.create_community_graph
create_citation_network = graph_module.create_citation_network
create_transductive_split = graph_module.create_transductive_split
spring_layout = graph_module.spring_layout
draw_graph = graph_module.draw_graph

gcn_module = import_module('36_gcn')
softmax = gcn_module.softmax
cross_entropy_loss = gcn_module.cross_entropy_loss


# ============================================================
# HETEROGENEOUS GRAPH DATA STRUCTURE
# ============================================================

class HeterogeneousGraph:
    """
    Extension of Graph with typed nodes and typed edges (relations).

    Stores:
    - Adjacency matrix A (n x n) -- overall connectivity
    - Node features X (n x d)
    - Node types: array of type IDs per node
    - Relation-specific adjacency matrices
    - Typed edge list
    """

    def __init__(self, n_nodes, node_types, node_features=None, directed=False):
        """
        Parameters:
        -----------
        n_nodes : int
            Number of nodes
        node_types : ndarray of int
            Type ID for each node (e.g., 0=paper, 1=author)
        node_features : ndarray (n_nodes, d) or None
            Node feature matrix. If None, uses one-hot identity.
        directed : bool
            If False, typed edges are added in both directions.
        """
        self.n_nodes = n_nodes
        self.directed = directed
        self.A = np.zeros((n_nodes, n_nodes))
        self.edge_list = []
        self.node_types = np.array(node_types, dtype=int)

        if node_features is not None:
            self.X = node_features.copy()
        else:
            self.X = np.eye(n_nodes)

        # Relation-specific storage
        self._relation_edges = {}  # rel_type -> list of (i, j)
        self._relation_adj = {}    # rel_type -> adjacency matrix (cached)
        self._n_relations = 0

    def add_typed_edge(self, i, j, rel_type):
        """
        Add an edge between nodes i and j with a specific relation type.

        Parameters:
        -----------
        i, j : int
            Source and target node indices
        rel_type : int
            Relation type ID
        """
        # Update overall adjacency
        self.A[i, j] = 1.0
        self.edge_list.append((i, j))

        if not self.directed:
            self.A[j, i] = 1.0
            self.edge_list.append((j, i))

        # Update relation-specific storage
        if rel_type not in self._relation_edges:
            self._relation_edges[rel_type] = []
        self._relation_edges[rel_type].append((i, j))
        if not self.directed:
            self._relation_edges[rel_type].append((j, i))

        # Invalidate cached adjacency matrices
        self._relation_adj = {}

        # Track number of relations
        self._n_relations = max(self._n_relations, rel_type + 1)

    def get_relation_adjacency(self, rel_type):
        """
        Get the adjacency matrix for a specific relation type.

        Returns:
        --------
        A_r : ndarray (n_nodes, n_nodes)
            Adjacency matrix where A_r[i,j] = 1 if edge (i,j) has rel_type.
        """
        if rel_type in self._relation_adj:
            return self._relation_adj[rel_type]

        A_r = np.zeros((self.n_nodes, self.n_nodes))
        if rel_type in self._relation_edges:
            for i, j in self._relation_edges[rel_type]:
                A_r[i, j] = 1.0

        self._relation_adj[rel_type] = A_r
        return A_r

    @property
    def n_relations(self):
        """Number of distinct relation types."""
        return self._n_relations

    def neighbors(self, node):
        """Get neighbors of a node (all relations combined)."""
        return np.where(self.A[node] > 0)[0]

    def degrees(self):
        """Degree of each node (all relations combined)."""
        return np.sum(self.A, axis=1)

    def relation_degrees(self, rel_type):
        """Degree of each node for a specific relation."""
        A_r = self.get_relation_adjacency(rel_type)
        return np.sum(A_r, axis=1)

    def get_node_type_mask(self, node_type):
        """Boolean mask for nodes of a specific type."""
        return self.node_types == node_type

    @property
    def n_node_types(self):
        """Number of distinct node types."""
        return len(np.unique(self.node_types))


# ============================================================
# DATASET: ACADEMIC GRAPH
# ============================================================

def create_academic_graph(n_papers=40, n_authors=20, n_topics=3,
                          feature_dim=16, random_state=42):
    """
    Create a heterogeneous academic graph with papers and authors.

    Node types:
        0 = Paper (with topic-correlated features)
        1 = Author (with research-area features)

    Relation types:
        0 = "cites"  (paper <-> paper)
        1 = "writes" (author <-> paper)

    Labels: paper topic categories (only for paper nodes).
    Author nodes get label -1 (unlabeled).

    Parameters:
    -----------
    n_papers : int
        Number of paper nodes
    n_authors : int
        Number of author nodes
    n_topics : int
        Number of paper topic categories
    feature_dim : int
        Feature vector dimension
    random_state : int
        Random seed

    Returns:
    --------
    hetero_graph : HeterogeneousGraph
    labels : ndarray
        Labels for ALL nodes. Papers: 0..n_topics-1. Authors: -1.
    """
    np.random.seed(random_state)

    n_nodes = n_papers + n_authors
    node_types = np.array([0] * n_papers + [1] * n_authors, dtype=int)

    # Create node features
    X = np.random.randn(n_nodes, feature_dim) * 0.3

    # Paper features: correlated with topic
    paper_labels = np.random.randint(0, n_topics, n_papers)
    topic_prototypes = np.random.randn(n_topics, feature_dim) * 1.5
    for i in range(n_papers):
        X[i] += topic_prototypes[paper_labels[i]]

    # Author features: correlated with primary research area
    author_areas = np.random.randint(0, n_topics, n_authors)
    for i in range(n_authors):
        X[n_papers + i] += topic_prototypes[author_areas[i]] * 0.8

    # Labels: paper topics, authors get -1 (no classification label)
    labels = np.full(n_nodes, -1, dtype=int)
    labels[:n_papers] = paper_labels

    # Build heterogeneous graph
    hetero_graph = HeterogeneousGraph(n_nodes, node_types, X, directed=False)

    # Relation 0: "cites" (paper <-> paper)
    # Papers in same topic cite each other more often
    for i in range(n_papers):
        for j in range(i + 1, n_papers):
            if paper_labels[i] == paper_labels[j]:
                if np.random.rand() < 0.20:
                    hetero_graph.add_typed_edge(i, j, rel_type=0)
            else:
                if np.random.rand() < 0.03:
                    hetero_graph.add_typed_edge(i, j, rel_type=0)

    # Relation 1: "writes" (author <-> paper)
    # Each author writes 1-4 papers, preferring their research area
    for a in range(n_authors):
        n_written = np.random.randint(1, 5)
        same_area = np.where(paper_labels == author_areas[a])[0]
        diff_area = np.where(paper_labels != author_areas[a])[0]

        targets = []
        for _ in range(n_written):
            if len(same_area) > 0 and np.random.rand() < 0.7:
                p = np.random.choice(same_area)
            elif len(diff_area) > 0:
                p = np.random.choice(diff_area)
            else:
                p = np.random.randint(n_papers)
            targets.append(p)

        for p in set(targets):
            hetero_graph.add_typed_edge(n_papers + a, p, rel_type=1)

    return hetero_graph, labels


# ============================================================
# R-GCN IMPLEMENTATION
# ============================================================

def _normalize_relation_adjacency(A_r):
    """
    Row-normalize a relation adjacency matrix: D_r^(-1) A_r.

    For each node, divides by its in-degree for that relation.
    Prevents high-degree nodes from dominating.
    """
    d = np.sum(A_r, axis=1)
    d_inv = np.zeros_like(d)
    nonzero = d > 0
    d_inv[nonzero] = 1.0 / d[nonzero]
    return np.diag(d_inv) @ A_r


class RGCN:
    """
    Relational Graph Convolutional Network with full backpropagation.

    Paradigm: MULTIPLE NODE/EDGE TYPES

    Each R-GCN layer computes:
        h_i' = sigma( SUM_r  A_r_norm @ H @ W_r  +  H @ W_0  +  bias )

    Where:
        W_r: per-relation weight matrix (or basis-decomposed)
        W_0: self-loop weight matrix
        A_r_norm: row-normalized adjacency for relation r

    Basis decomposition (when n_bases > 0):
        W_r = SUM_b  a_rb * B_b
        Reduces parameters from R*d_in*d_out to B*d_in*d_out + R*B

    Full analytical backpropagation through all components.
    """

    def __init__(self, n_features, n_hidden, n_classes, n_relations,
                 n_layers=2, n_bases=0, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, random_state=None):
        """
        Parameters:
        -----------
        n_features : int
            Input feature dimension
        n_hidden : int
            Hidden layer dimension
        n_classes : int
            Number of output classes
        n_relations : int
            Number of relation types
        n_layers : int
            Number of R-GCN layers (2-3 recommended)
        n_bases : int
            Number of basis matrices for decomposition.
            0 = full per-relation weights (no decomposition).
            >0 = basis decomposition with n_bases shared basis matrices.
        dropout : float
            Dropout rate
        lr : float
            Learning rate
        weight_decay : float
            L2 regularization strength
        random_state : int or None
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.n_bases = n_bases
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for all layers.

        Per layer stores:
            - W_self: self-loop weights (d_in x d_out)
            - bias: bias vector (d_out,)
            - If n_bases == 0: W_rel[r] for each relation (R matrices, each d_in x d_out)
            - If n_bases > 0: bases (n_bases x d_in x d_out) + coeffs (n_relations x n_bases)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        dims = ([self.n_features]
                + [self.n_hidden] * (self.n_layers - 1)
                + [self.n_classes])

        self.W_self = []     # Self-loop weights per layer
        self.biases = []     # Biases per layer
        self.W_rel = []      # Per-relation weights (when n_bases == 0)
        self.bases = []      # Basis matrices (when n_bases > 0)
        self.coeffs = []     # Coefficients (when n_bases > 0)

        for l in range(self.n_layers):
            d_in, d_out = dims[l], dims[l + 1]
            std = np.sqrt(2.0 / (d_in + d_out))

            # Self-loop weight
            self.W_self.append(np.random.randn(d_in, d_out) * std)

            # Bias
            self.biases.append(np.zeros(d_out))

            if self.n_bases == 0:
                # Full per-relation weights
                layer_W_rel = []
                for r in range(self.n_relations):
                    layer_W_rel.append(np.random.randn(d_in, d_out) * std)
                self.W_rel.append(layer_W_rel)
                self.bases.append(None)
                self.coeffs.append(None)
            else:
                # Basis decomposition
                B = np.random.randn(self.n_bases, d_in, d_out) * std
                self.bases.append(B)
                a = np.random.randn(self.n_relations, self.n_bases) * 0.1
                self.coeffs.append(a)
                self.W_rel.append(None)

    def _get_relation_weight(self, layer, rel):
        """
        Get W_r for a given layer and relation.

        If n_bases == 0: returns stored W_rel[layer][rel]
        If n_bases > 0:  computes W_r = SUM_b a_rb * B_b
        """
        if self.n_bases == 0:
            return self.W_rel[layer][rel]
        else:
            # W_r = sum_b a_rb * B_b
            a_r = self.coeffs[layer][rel]  # (n_bases,)
            W_r = np.tensordot(a_r, self.bases[layer], axes=([0], [0]))
            return W_r  # (d_in, d_out)

    def forward(self, hetero_graph, training=True):
        """
        Forward pass through all R-GCN layers.

        For each layer l:
            Z = SUM_r (A_r_norm @ H @ W_r) + H @ W_0 + bias
            H = ReLU(Z) + dropout  (hidden layers)
            H = softmax(Z)         (output layer)

        Parameters:
        -----------
        hetero_graph : HeterogeneousGraph
        training : bool

        Returns:
        --------
        output : ndarray (n_nodes, n_classes) -- class probabilities
        cache : dict -- stored values for backpropagation
        """
        # Precompute normalized relation adjacencies
        A_r_norms = []
        for r in range(self.n_relations):
            A_r = hetero_graph.get_relation_adjacency(r)
            A_r_norms.append(_normalize_relation_adjacency(A_r))

        H = hetero_graph.X.copy()
        cache = {
            'H': [H],
            'Z': [],
            'dropout_masks': [],
            'A_r_norms': A_r_norms,
        }

        for l in range(self.n_layers):
            # Self-loop contribution: H @ W_0
            Z = H @ self.W_self[l]

            # Relation contributions: SUM_r A_r_norm @ H @ W_r
            for r in range(self.n_relations):
                W_r = self._get_relation_weight(l, r)
                Z = Z + A_r_norms[r] @ H @ W_r

            # Add bias
            Z = Z + self.biases[l]
            cache['Z'].append(Z)

            if l < self.n_layers - 1:
                # Hidden layer: ReLU + dropout
                H = np.maximum(Z, 0)

                if training and self.dropout > 0:
                    mask = (np.random.rand(*H.shape) > self.dropout).astype(float)
                    H = H * mask / (1 - self.dropout + 1e-10)
                    cache['dropout_masks'].append(mask)
                else:
                    cache['dropout_masks'].append(np.ones_like(H))
            else:
                # Output layer: softmax
                H = softmax(Z)

            cache['H'].append(H)

        return H, cache

    def backward(self, hetero_graph, labels, mask, cache):
        """
        Full analytical backpropagation through R-GCN layers.

        Computes gradients for:
        - W_self[l]: self-loop weights
        - W_rel[l][r] or bases[l] + coeffs[l]: per-relation weights
        - biases[l]: biases

        Updates all parameters with gradient descent.
        """
        n = len(labels)
        probs = cache['H'][-1]
        A_r_norms = cache['A_r_norms']

        # Gradient of cross-entropy + softmax at output
        dZ = probs.copy()
        valid = labels >= 0  # Only labeled nodes
        combined_mask = mask & valid
        n_train = max(np.sum(combined_mask), 1)

        dZ[np.arange(n), np.clip(labels, 0, self.n_classes - 1)] -= 1
        mask_float = combined_mask.astype(float)
        dZ = dZ * mask_float[:, None] / n_train

        for l in range(self.n_layers - 1, -1, -1):
            H_prev = cache['H'][l]

            # Gradient for bias
            g_bias = np.sum(dZ, axis=0)

            # Gradient for self-loop weight: (H_prev)^T @ dZ
            g_W_self = H_prev.T @ dZ + self.weight_decay * self.W_self[l]

            # Gradient for relation weights
            if self.n_bases == 0:
                # Full per-relation weights
                g_W_rel = []
                for r in range(self.n_relations):
                    # Z += A_r_norm @ H @ W_r
                    # dL/dW_r = (A_r_norm @ H_prev)^T @ dZ
                    AH = A_r_norms[r] @ H_prev
                    g_Wr = AH.T @ dZ + self.weight_decay * self.W_rel[l][r]
                    g_W_rel.append(g_Wr)
            else:
                # Basis decomposition gradients
                g_bases = np.zeros_like(self.bases[l])
                g_coeffs = np.zeros_like(self.coeffs[l])

                for r in range(self.n_relations):
                    AH = A_r_norms[r] @ H_prev
                    # g_Wr: gradient w.r.t. effective W_r for this relation
                    g_Wr = AH.T @ dZ

                    # W_r = sum_b a_rb * B_b
                    # dL/d(a_rb) = sum of element-wise (g_Wr * B_b)
                    for b in range(self.n_bases):
                        g_coeffs[r, b] = np.sum(g_Wr * self.bases[l][b])

                    # dL/d(B_b) += a_rb * g_Wr
                    a_r = self.coeffs[l][r]
                    for b in range(self.n_bases):
                        g_bases[b] += a_r[b] * g_Wr

                # Add weight decay to bases
                g_bases += self.weight_decay * self.bases[l]

            # Backprop dZ to previous layer (if not first layer)
            if l > 0:
                # dH from self-loop: dZ @ W_self^T
                dH = dZ @ self.W_self[l].T

                # dH from each relation: A_r_norm^T @ (dZ @ W_r^T)
                for r in range(self.n_relations):
                    W_r = self._get_relation_weight(l, r)
                    dH += A_r_norms[r].T @ (dZ @ W_r.T)

                # Backprop through dropout
                dH = dH * cache['dropout_masks'][l - 1] / (1 - self.dropout + 1e-10)

                # Backprop through ReLU
                dZ = dH * (cache['Z'][l - 1] > 0).astype(float)

            # Update parameters
            self.W_self[l] -= self.lr * g_W_self
            self.biases[l] -= self.lr * g_bias

            if self.n_bases == 0:
                for r in range(self.n_relations):
                    self.W_rel[l][r] -= self.lr * g_W_rel[r]
            else:
                self.bases[l] -= self.lr * g_bases
                self.coeffs[l] -= self.lr * g_coeffs

    def fit(self, hetero_graph, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train R-GCN on a heterogeneous graph with transductive split.

        Parameters:
        -----------
        hetero_graph : HeterogeneousGraph
        labels : ndarray
            Node labels (-1 for unlabeled nodes)
        train_mask : boolean array
            Which nodes are in training set
        n_epochs : int
        verbose : bool

        Returns:
        --------
        loss_history : list of float
        """
        loss_history = []

        for epoch in range(n_epochs):
            # Forward pass
            probs, cache = self.forward(hetero_graph, training=True)

            # Loss on labeled train nodes
            valid = labels >= 0
            combined_mask = train_mask & valid
            if np.sum(combined_mask) == 0:
                break

            loss = cross_entropy_loss(
                probs, np.clip(labels, 0, self.n_classes - 1), combined_mask
            )
            loss_history.append(loss)

            # Backward pass
            self.backward(hetero_graph, labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                train_pred = np.argmax(probs, axis=1)
                train_acc = np.mean(train_pred[combined_mask] == labels[combined_mask])
                print(f"  Epoch {epoch+1:>4}: loss={loss:.4f}, "
                      f"train_acc={train_acc:.3f}")

        return loss_history

    def predict(self, hetero_graph):
        """Predict node labels."""
        probs, _ = self.forward(hetero_graph, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, hetero_graph):
        """Predict node class probabilities."""
        probs, _ = self.forward(hetero_graph, training=False)
        return probs

    def get_embeddings(self, hetero_graph, layer=-2):
        """Get node embeddings from a specific layer."""
        _, cache = self.forward(hetero_graph, training=False)
        return cache['H'][layer]

    def count_parameters(self):
        """Count total number of trainable parameters."""
        total = 0
        for l in range(self.n_layers):
            total += self.W_self[l].size
            total += self.biases[l].size
            if self.n_bases == 0:
                for r in range(self.n_relations):
                    total += self.W_rel[l][r].size
            else:
                total += self.bases[l].size
                total += self.coeffs[l].size
        return total


# ============================================================
# HELPER: Convert HeterogeneousGraph to homogeneous for GCN
# ============================================================

def _hetero_to_homogeneous(hetero_graph):
    """
    Convert a HeterogeneousGraph to a plain Graph.
    Drops all relation type information (treats all edges the same).
    Used for comparison: standard GCN on heterogeneous data.
    """
    g = Graph(hetero_graph.n_nodes, hetero_graph.X.copy(), hetero_graph.directed)
    g.A = hetero_graph.A.copy()
    g.edge_list = list(hetero_graph.edge_list)
    return g


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    np.random.seed(42)

    # Create dataset
    hetero_graph, labels = create_academic_graph(
        n_papers=50, n_authors=25, n_topics=3, feature_dim=16,
        random_state=42
    )
    n_classes = 3

    # Only classify paper nodes (labels >= 0)
    paper_mask = labels >= 0
    train_mask, val_mask, test_mask = create_transductive_split(
        hetero_graph.n_nodes, np.clip(labels, 0, n_classes - 1),
        train_ratio=0.2, val_ratio=0.1, random_state=42
    )
    # Restrict to paper nodes
    train_mask = train_mask & paper_mask
    test_mask = test_mask & paper_mask

    print(f"\n  Graph: {hetero_graph.n_nodes} nodes "
          f"({np.sum(paper_mask)} papers, {np.sum(~paper_mask)} authors)")
    print(f"  Relations: {hetero_graph.n_relations} (cites, writes)")
    print(f"  Train: {np.sum(train_mask)} papers, "
          f"Test: {np.sum(test_mask)} papers")

    # -------- Experiment 1: Shared vs Per-Relation Weights --------
    print("\n1. SHARED vs PER-RELATION WEIGHTS")
    print("-" * 40)
    print("Does relation-specific transformation help?")

    GCN = gcn_module.GCN

    homo_graph = _hetero_to_homogeneous(hetero_graph)
    gcn_shared = GCN(homo_graph.X.shape[1], 16, n_classes, n_layers=2,
                     lr=0.01, dropout=0.3, random_state=42)
    gcn_shared.fit(homo_graph, np.clip(labels, 0, n_classes - 1),
                   train_mask, n_epochs=200, verbose=False)
    preds_shared = gcn_shared.predict(homo_graph)
    acc_shared = np.mean(preds_shared[test_mask] == labels[test_mask])

    rgcn_full = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                     n_relations=hetero_graph.n_relations,
                     n_layers=2, n_bases=0, dropout=0.3, lr=0.01,
                     random_state=42)
    rgcn_full.fit(hetero_graph, labels, train_mask,
                  n_epochs=200, verbose=False)
    preds_rgcn = rgcn_full.predict(hetero_graph)
    acc_rgcn = np.mean(preds_rgcn[test_mask] == labels[test_mask])

    params_gcn = sum(w.size for w in gcn_shared.weights)
    params_rgcn = rgcn_full.count_parameters()

    print(f"  GCN (shared weights):  test_acc={acc_shared:.3f}  "
          f"params={params_gcn}")
    print(f"  R-GCN (per-relation):  test_acc={acc_rgcn:.3f}  "
          f"params={params_rgcn}")
    print("-> Per-relation weights capture relation-specific semantics")

    # -------- Experiment 2: Number of Basis Matrices --------
    print("\n2. NUMBER OF BASIS MATRICES (Decomposition)")
    print("-" * 40)
    print("Trade-off: expressiveness vs parameter efficiency")

    for n_bases in [0, 1, 2, 4]:
        rgcn = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                     n_relations=hetero_graph.n_relations,
                     n_layers=2, n_bases=n_bases, dropout=0.3, lr=0.01,
                     random_state=42)
        rgcn.fit(hetero_graph, labels, train_mask,
                 n_epochs=200, verbose=False)
        preds = rgcn.predict(hetero_graph)
        acc = np.mean(preds[test_mask] == labels[test_mask])
        n_params = rgcn.count_parameters()

        label = "full" if n_bases == 0 else f"{n_bases} bases"
        print(f"  {label:<12}  test_acc={acc:.3f}  params={n_params}")

    print("-> n_bases=0 (full) is most expressive but most parameters")
    print("-> Small n_bases reduces parameters with modest accuracy cost")
    print("-> Critical when #relations is large (knowledge graphs)")

    # -------- Experiment 3: R-GCN vs GCN across settings --------
    print("\n3. R-GCN vs STANDARD GCN ON HETEROGENEOUS DATA")
    print("-" * 40)
    print("Across different numbers of topics:")

    for n_topics in [2, 3, 4]:
        hg, lb = create_academic_graph(
            n_papers=50, n_authors=25, n_topics=n_topics,
            feature_dim=16, random_state=42
        )
        pm = lb >= 0
        tm, _, tsm = create_transductive_split(
            hg.n_nodes, np.clip(lb, 0, n_topics - 1),
            train_ratio=0.2, val_ratio=0.1, random_state=42
        )
        tm = tm & pm
        tsm = tsm & pm

        # GCN
        hg_homo = _hetero_to_homogeneous(hg)
        gcn = GCN(hg_homo.X.shape[1], 16, n_topics, n_layers=2,
                  lr=0.01, dropout=0.3, random_state=42)
        gcn.fit(hg_homo, np.clip(lb, 0, n_topics - 1), tm,
                n_epochs=200, verbose=False)
        acc_gcn = np.mean(gcn.predict(hg_homo)[tsm] == lb[tsm])

        # R-GCN
        rgcn = RGCN(hg.X.shape[1], 16, n_topics,
                     n_relations=hg.n_relations,
                     n_layers=2, n_bases=0, dropout=0.3, lr=0.01,
                     random_state=42)
        rgcn.fit(hg, lb, tm, n_epochs=200, verbose=False)
        acc_rg = np.mean(rgcn.predict(hg)[tsm] == lb[tsm])

        print(f"  topics={n_topics}  GCN={acc_gcn:.3f}  R-GCN={acc_rg:.3f}  "
              f"diff={acc_rg - acc_gcn:+.3f}")

    print("-> R-GCN benefits from typed message passing")

    # -------- Experiment 4: Number of Layers --------
    print("\n4. NUMBER OF R-GCN LAYERS")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4]:
        rgcn = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                     n_relations=hetero_graph.n_relations,
                     n_layers=n_layers, n_bases=0, dropout=0.3, lr=0.01,
                     random_state=42)
        rgcn.fit(hetero_graph, labels, train_mask,
                 n_epochs=200, verbose=False)
        preds = rgcn.predict(hetero_graph)
        acc = np.mean(preds[test_mask] == labels[test_mask])

        emb = rgcn.get_embeddings(hetero_graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms
        cos_sim = emb_norm @ emb_norm.T
        avg_sim = ((np.sum(cos_sim) - hetero_graph.n_nodes)
                   / (hetero_graph.n_nodes * (hetero_graph.n_nodes - 1)))

        print(f"  layers={n_layers}  test_acc={acc:.3f}  "
              f"avg_cos_sim={avg_sim:.3f}")

    print("-> 2 layers usually optimal (same over-smoothing as GCN)")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    """Benchmark R-GCN on heterogeneous graph datasets."""
    print("\n" + "=" * 60)
    print("BENCHMARK: R-GCN on Heterogeneous Graphs")
    print("=" * 60)

    GCN = gcn_module.GCN
    results = {}

    configs = [
        ("academic_small", 30, 15, 3),
        ("academic_medium", 50, 25, 3),
        ("academic_large", 80, 40, 4),
    ]

    print(f"\n{'Dataset':<20} {'Nodes':<8} {'Rels':<6} "
          f"{'R-GCN Acc':<12} {'GCN Acc':<12}")
    print("-" * 60)

    for name, n_papers, n_authors, n_topics in configs:
        hg, lb = create_academic_graph(
            n_papers=n_papers, n_authors=n_authors, n_topics=n_topics,
            feature_dim=16, random_state=42
        )
        pm = lb >= 0
        tm, _, tsm = create_transductive_split(
            hg.n_nodes, np.clip(lb, 0, n_topics - 1),
            train_ratio=0.2, val_ratio=0.1, random_state=42
        )
        tm = tm & pm
        tsm = tsm & pm

        if np.sum(tsm) == 0:
            continue

        # R-GCN
        rgcn = RGCN(hg.X.shape[1], 16, n_topics,
                     n_relations=hg.n_relations,
                     n_layers=2, n_bases=0, dropout=0.3, lr=0.01,
                     random_state=42)
        rgcn.fit(hg, lb, tm, n_epochs=200, verbose=False)
        acc_rgcn = np.mean(rgcn.predict(hg)[tsm] == lb[tsm])

        # GCN baseline
        hg_homo = _hetero_to_homogeneous(hg)
        gcn = GCN(hg_homo.X.shape[1], 16, n_topics, n_layers=2,
                  lr=0.01, dropout=0.3, random_state=42)
        gcn.fit(hg_homo, np.clip(lb, 0, n_topics - 1), tm,
                n_epochs=200, verbose=False)
        acc_gcn = np.mean(gcn.predict(hg_homo)[tsm] == lb[tsm])

        results[name] = {'rgcn': acc_rgcn, 'gcn': acc_gcn}
        print(f"{name:<20} {hg.n_nodes:<8} {hg.n_relations:<6} "
              f"{acc_rgcn:<12.3f} {acc_gcn:<12.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_hetero_graph():
    """
    Main visualization: 2x3 figure showing heterogeneous GNN concepts.

    Panel 1: Heterogeneous graph with typed nodes and edges
    Panel 2: R-GCN predictions on paper nodes
    Panel 3: Per-relation weight magnitude visualization
    Panel 4: Basis decomposition comparison (bar chart)
    Panel 5: Training loss curve
    Panel 6: Summary text

    Saves as 42_hetero_gnn.png
    """
    print("\nGenerating: Heterogeneous GNN visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    np.random.seed(42)

    # Create dataset
    hetero_graph, labels = create_academic_graph(
        n_papers=40, n_authors=20, n_topics=3, feature_dim=16,
        random_state=42
    )
    n_classes = 3
    paper_mask = labels >= 0

    train_mask, val_mask, test_mask = create_transductive_split(
        hetero_graph.n_nodes, np.clip(labels, 0, n_classes - 1),
        train_ratio=0.2, val_ratio=0.1, random_state=42
    )
    train_mask = train_mask & paper_mask
    test_mask = test_mask & paper_mask

    # Train R-GCN
    rgcn = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                n_relations=hetero_graph.n_relations,
                n_layers=2, n_bases=0, dropout=0.3, lr=0.01,
                random_state=42)
    loss_history = rgcn.fit(hetero_graph, labels, train_mask,
                            n_epochs=250, verbose=False)

    # Layout: spring layout on overall graph
    temp_graph = Graph(hetero_graph.n_nodes, hetero_graph.X,
                       hetero_graph.directed)
    temp_graph.A = hetero_graph.A.copy()
    temp_graph.edge_list = list(hetero_graph.edge_list)
    pos = spring_layout(temp_graph, n_iter=80, seed=42)

    # ---- Panel 1: Heterogeneous graph with typed nodes/edges ----
    ax = axes[0, 0]
    rel_colors = ['#3498db', '#e74c3c']
    rel_names = ['cites', 'writes']
    for r in range(hetero_graph.n_relations):
        A_r = hetero_graph.get_relation_adjacency(r)
        for i in range(hetero_graph.n_nodes):
            for j in range(i + 1, hetero_graph.n_nodes):
                if A_r[i, j] > 0 or A_r[j, i] > 0:
                    ax.plot([pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            color=rel_colors[r], alpha=0.25,
                            linewidth=0.8)

    node_type_colors = ['#2ecc71', '#f39c12']
    node_type_names = ['Paper', 'Author']
    node_type_markers = ['o', 's']
    for t in range(hetero_graph.n_node_types):
        mask_t = hetero_graph.get_node_type_mask(t)
        ax.scatter(pos[mask_t, 0], pos[mask_t, 1],
                   c=node_type_colors[t], s=60, alpha=0.85,
                   edgecolors='black', linewidths=0.5,
                   marker=node_type_markers[t], zorder=3,
                   label=node_type_names[t])

    for r in range(hetero_graph.n_relations):
        ax.plot([], [], color=rel_colors[r], linewidth=2,
                label=f'{rel_names[r]} edge')

    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.set_title('Heterogeneous Graph\n(different node & edge types)')
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- Panel 2: R-GCN predictions on paper nodes ----
    ax = axes[0, 1]
    for i in range(hetero_graph.n_nodes):
        for j in range(i + 1, hetero_graph.n_nodes):
            if hetero_graph.A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        'gray', alpha=0.15, linewidth=0.5)

    preds = rgcn.predict(hetero_graph)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])

    # Author nodes (gray, no label)
    author_mask = ~paper_mask
    ax.scatter(pos[author_mask, 0], pos[author_mask, 1],
               c='lightgray', s=30, alpha=0.4, edgecolors='gray',
               linewidths=0.3, marker='s', zorder=2)

    # Paper nodes colored by prediction
    cmap = plt.cm.Set1
    for c in range(n_classes):
        c_mask = paper_mask & (preds == c)
        ax.scatter(pos[c_mask, 0], pos[c_mask, 1],
                   c=[cmap(c)], s=70, alpha=0.85,
                   edgecolors='black', linewidths=0.5,
                   zorder=3, label=f'Topic {c}')

    # Mark incorrect predictions
    incorrect = paper_mask & (preds != np.clip(labels, 0, n_classes - 1))
    if np.any(incorrect):
        ax.scatter(pos[incorrect, 0], pos[incorrect, 1],
                   facecolors='none', edgecolors='red', s=150,
                   linewidths=2, zorder=4, label='Wrong')

    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.set_title(f'R-GCN Predictions\ntest_acc={test_acc:.3f}')
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- Panel 3: Per-relation weight visualization ----
    ax = axes[0, 2]
    # Show weight matrices for layer 0 side by side
    d_in = rgcn.W_self[0].shape[0]
    d_out = rgcn.W_self[0].shape[1]
    n_show = min(d_in, 16)
    d_show = min(d_out, 16)

    gap = 1  # pixel gap between matrices
    total_w = d_show * 3 + gap * 2
    combined = np.full((n_show, total_w), np.nan)
    combined[:, :d_show] = rgcn.W_self[0][:n_show, :d_show]
    off1 = d_show + gap
    combined[:, off1:off1 + d_show] = (
        rgcn._get_relation_weight(0, 0)[:n_show, :d_show]
    )
    off2 = off1 + d_show + gap
    combined[:, off2:off2 + d_show] = (
        rgcn._get_relation_weight(0, 1)[:n_show, :d_show]
    )

    vmax = np.nanmax(np.abs(combined))
    masked = np.ma.array(combined, mask=np.isnan(combined))
    im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')

    ax.set_xticks([d_show // 2, off1 + d_show // 2, off2 + d_show // 2])
    ax.set_xticklabels(['W_self', 'W_cites', 'W_writes'], fontsize=8)
    ax.set_ylabel('Input features')
    ax.set_title('Per-Relation Weight Matrices\n'
                 '(Layer 0: each relation has its own W)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ---- Panel 4: Basis decomposition comparison ----
    ax = axes[1, 0]
    basis_configs = [0, 1, 2, 4]
    accs = []
    params = []
    config_labels = []

    for nb in basis_configs:
        rgcn_test = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                         n_relations=hetero_graph.n_relations,
                         n_layers=2, n_bases=nb, dropout=0.3, lr=0.01,
                         random_state=42)
        rgcn_test.fit(hetero_graph, labels, train_mask,
                      n_epochs=200, verbose=False)
        pred_test = rgcn_test.predict(hetero_graph)
        acc = np.mean(pred_test[test_mask] == labels[test_mask])
        accs.append(acc)
        params.append(rgcn_test.count_parameters())
        config_labels.append('Full' if nb == 0 else f'{nb} bases')

    x_pos = np.arange(len(basis_configs))
    bar_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = ax.bar(x_pos, accs, color=bar_colors,
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, (bar, p) in enumerate(zip(bars, params)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{p} params', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Basis Decomposition\n(accuracy vs parameter count)')
    ax.set_ylim(0, max(accs) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Panel 5: Training loss curve ----
    ax = axes[1, 1]
    ax.plot(loss_history, 'b-', linewidth=1, alpha=0.8, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('R-GCN Training Curve')
    ax.grid(True, alpha=0.3)

    # Also track test accuracy during training
    rgcn_track = RGCN(hetero_graph.X.shape[1], 16, n_classes,
                      n_relations=hetero_graph.n_relations,
                      n_layers=2, n_bases=0, dropout=0.3, lr=0.01,
                      random_state=42)
    track_epochs = list(range(0, 250, 10))
    track_accs = []
    for epoch in range(250):
        probs_t, cache_t = rgcn_track.forward(hetero_graph, training=True)
        rgcn_track.backward(hetero_graph, labels, train_mask, cache_t)
        if epoch in track_epochs:
            preds_t = np.argmax(probs_t, axis=1)
            acc_t = np.mean(preds_t[test_mask] == labels[test_mask])
            track_accs.append(acc_t)

    ax2 = ax.twinx()
    ax2.plot(track_epochs, track_accs, 'r--', linewidth=1.5,
             alpha=0.7, label='Test Acc')
    ax2.set_ylabel('Test Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='center right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)

    # ---- Panel 6: Summary text ----
    ax = axes[1, 2]
    ax.axis('off')

    n_papers_count = np.sum(paper_mask)
    n_authors_count = np.sum(~paper_mask)
    n_cites = int(np.sum(hetero_graph.get_relation_adjacency(0))) // 2
    n_writes = int(np.sum(hetero_graph.get_relation_adjacency(1))) // 2

    summary_text = (
        "R-GCN: Relational Graph Convolutional Network\n"
        "=" * 46 + "\n\n"
        f"Graph Statistics:\n"
        f"  Nodes: {hetero_graph.n_nodes} "
        f"({n_papers_count} papers, {n_authors_count} authors)\n"
        f"  Edges: {n_cites} cites, {n_writes} writes\n"
        f"  Relations: {hetero_graph.n_relations}\n"
        f"  Features: {hetero_graph.X.shape[1]}D\n"
        f"  Classes: {n_classes} paper topics\n\n"
        f"R-GCN (full) test accuracy: {test_acc:.3f}\n"
        f"R-GCN parameters: {rgcn.count_parameters()}\n\n"
        "Key Insight:\n"
        "  Different edge types need different\n"
        "  weight matrices. 'cites' and 'writes'\n"
        "  carry fundamentally different info.\n\n"
        "  W_r transforms features differently\n"
        "  for each relation type r.\n\n"
        "Basis Decomposition:\n"
        "  W_r = SUM_b a_rb * B_b\n"
        "  Reduces params when R is large."
    )

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('HETEROGENEOUS GNN (R-GCN): Per-Relation Message Passing\n'
                 'Different edge types get different weight matrices',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("HETEROGENEOUS GNN (R-GCN)")
    print("Paradigm: MULTIPLE NODE/EDGE TYPES")
    print("=" * 60)

    print("""
WHAT THIS MODEL IS:
    R-GCN extends GCN to heterogeneous graphs with TYPED edges.

    Standard GCN layer:
        H' = sigma(A_hat @ H @ W)       -- ONE weight matrix W

    R-GCN layer:
        H' = sigma(SUM_r A_r_norm @ H @ W_r + H @ W_0 + bias)

    Where W_r is a DIFFERENT weight matrix for each relation r.

    This lets the model learn that different edge types
    (e.g., "cites" vs "writes") carry different semantics.

THE PARAMETER EXPLOSION PROBLEM:
    R relations x d_in x d_out parameters per layer!
    Solution: BASIS DECOMPOSITION
        W_r = SUM_b a_rb * B_b
    Share B basis matrices, only learn R x B coefficients.

KEY CONCEPTS:
    1. Heterogeneous graph = multiple node/edge types
    2. Per-relation weights = type-specific message passing
    3. Basis decomposition = parameter-efficient with many relations
    4. Knowledge graphs are the canonical application
    """)

    ablation_experiments()
    results = benchmark_on_datasets()

    print("\nGenerating visualizations...")

    fig = visualize_hetero_graph()
    save_path = '/Users/sid47/ML Algorithms/42_hetero_gnn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY: What Heterogeneous GNNs Reveal")
    print("=" * 60)
    print("""
1. NOT ALL EDGES ARE EQUAL
   "cites" and "writes" carry different information.
   R-GCN learns SEPARATE transformations per relation.

2. PARAMETER EXPLOSION IS REAL
   R x d_in x d_out parameters per layer with R relations.
   Knowledge graphs have 100s of relations!
   Basis decomposition: W_r = SUM a_rb * B_b solves this.

3. HETEROGENEOUS > HOMOGENEOUS (on typed data)
   When relations have different semantics,
   ignoring types (standard GCN) loses information.

4. SAME OVER-SMOOTHING ISSUE
   2-3 layers optimal, just like GCN.
   Relation-typing doesn't solve over-smoothing.

5. DESIGN CHOICES MATTER
   - Full vs basis decomposition: accuracy-efficiency trade-off
   - Number of bases: B=1 (shared) to B=R (full)
   - Self-loop weight W_0: retains node identity

CONNECTION TO OTHER FILES:
    36_gcn.py: R-GCN generalizes GCN to typed edges
    38_gat.py: Attention could also be per-relation (HAN)
    43_temporal_gnn.py: Time as a special "relation type"
    44_graph_transformer.py: Full attention with edge features

WHEN TO USE R-GCN:
    - Knowledge graphs (entity-relation-entity triples)
    - Academic networks (papers, authors, venues)
    - Biological networks (proteins, drugs, diseases)
    - Any graph with MEANINGFUL edge type distinctions
    """)
