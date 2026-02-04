"""
Heterogeneous GNN — Multiple Node and Edge Types
=================================================

Paradigm: GRAPHS WITH DIFFERENT TYPES OF ENTITIES AND RELATIONS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Real-world graphs often have DIFFERENT types of nodes and edges!

HOMOGENEOUS GRAPH:
    One type of node, one type of edge.
    Example: Social network (users → friends_with → users)

HETEROGENEOUS GRAPH:
    Multiple node types, multiple edge types.
    Example: Citation network
        - Nodes: Papers, Authors, Venues
        - Edges: (Paper, cites, Paper), (Author, writes, Paper),
                (Paper, published_in, Venue)

FORMAL DEFINITION:
    G = (V, E, τ_v, τ_e)
    - V: nodes
    - E: edges
    - τ_v: V → T_V (node type function)
    - τ_e: E → T_E (edge type function)

===============================================================
WHY HETEROGENEOUS?
===============================================================

1. MORE EXPRESSIVE: Different relations have different semantics
2. REALISTIC: Most real graphs are heterogeneous
3. RICH FEATURES: Each type can have different features

EXAMPLES:
- Knowledge graphs: (entity, relation, entity)
- E-commerce: (user, buys, product), (product, belongs_to, category)
- Academic: (paper, cites, paper), (author, writes, paper)
- Social: (user, follows, user), (user, posts, content)

===============================================================
RELATIONAL GCN (R-GCN)
===============================================================

Different transformation for each relation type!

h_v' = σ(Σ_r Σ_{u∈N_r(v)} (1/c_{v,r}) W_r h_u + W_0 h_v)

WHERE:
- r indexes relation types
- N_r(v) = neighbors of v via relation r
- W_r = weight matrix specific to relation r
- c_{v,r} = normalization (degree in relation r)
- W_0 = self-loop transformation

===============================================================
PARAMETER EXPLOSION PROBLEM
===============================================================

Problem: |relations| × d × d parameters per layer!
If 100 relations and d=256, that's 6.5M params per layer!

SOLUTIONS:

1. BASIS DECOMPOSITION
   W_r = Σ_b a_rb B_b
   Share B bases across relations, learn coefficients a_rb.

2. BLOCK DIAGONAL
   W_r = diag(Q_r^1, ..., Q_r^B)
   Each relation uses block-diagonal weights.

===============================================================
HETEROGENEOUS ATTENTION (HAN)
===============================================================

Use METAPATHS to define semantic relationships!

METAPATH: A path with specific node types
    Example: Author → Paper → Author (co-author)
    Example: Paper → Author → Paper (same author)

HAN APPROACH:
1. For each metapath, apply GAT-style attention
2. Aggregate across metapaths with attention

This captures different SEMANTIC views of the graph.

===============================================================
HETEROGENEOUS GRAPH TRANSFORMER (HGT)
===============================================================

Use Transformer attention with type-specific projections.

Q = W^Q_{τ(v)} h_v     (type-specific query)
K = W^K_{τ(u)} h_u     (type-specific key)
V = W^V_{τ(u)} h_u     (type-specific value)

Attention also depends on EDGE TYPE:
    α(v,e,u) = softmax(Q_v^T W^A_{φ(e)} K_u / √d)

===============================================================
INDUCTIVE BIAS
===============================================================

1. TYPED ENTITIES: Different types need different processing
2. RELATIONAL: Edge type matters for message passing
3. SEMANTIC: Metapaths capture higher-order structure
4. SHARED STRUCTURE: Similar relations can share parameters

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class HeterogeneousGraph:
    """
    Heterogeneous graph with multiple node and edge types.
    """

    def __init__(self):
        self.nodes = {}  # type → list of node ids
        self.edges = {}  # (src_type, edge_type, dst_type) → list of (src, dst)
        self.node_features = {}  # node_id → features
        self.node_types = {}  # node_id → type
        self.n_nodes = 0

    def add_node(self, node_type, features=None):
        """Add a node of given type."""
        node_id = self.n_nodes
        self.n_nodes += 1

        if node_type not in self.nodes:
            self.nodes[node_type] = []
        self.nodes[node_type].append(node_id)

        self.node_types[node_id] = node_type

        if features is not None:
            self.node_features[node_id] = features

        return node_id

    def add_edge(self, src, dst, edge_type):
        """Add an edge with given type."""
        src_type = self.node_types[src]
        dst_type = self.node_types[dst]
        key = (src_type, edge_type, dst_type)

        if key not in self.edges:
            self.edges[key] = []
        self.edges[key].append((src, dst))

    def get_edge_types(self):
        """Get all edge type tuples."""
        return list(self.edges.keys())

    def get_node_types(self):
        """Get all node types."""
        return list(self.nodes.keys())

    def get_adjacency(self, edge_type_tuple):
        """Get adjacency matrix for specific edge type."""
        adj = np.zeros((self.n_nodes, self.n_nodes))
        if edge_type_tuple in self.edges:
            for src, dst in self.edges[edge_type_tuple]:
                adj[src, dst] = 1
        return adj

    def get_features_matrix(self, feature_dim=16):
        """Get feature matrix for all nodes."""
        X = np.zeros((self.n_nodes, feature_dim))
        for node_id, feat in self.node_features.items():
            X[node_id] = feat[:feature_dim]
        return X


class RGCNLayer:
    """
    Relational GCN Layer.

    h_v' = σ(Σ_r Σ_{u∈N_r(v)} (1/c_{v,r}) W_r h_u + W_0 h_v)

    Different weight matrix for each relation type.
    """

    def __init__(self, in_features, out_features, num_relations,
                 num_bases=None, activation='relu'):
        """
        Parameters:
        - in_features: Input feature dimension
        - out_features: Output feature dimension
        - num_relations: Number of relation types
        - num_bases: Number of basis matrices (for parameter sharing)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases or num_relations
        self.activation = activation

        scale = np.sqrt(2.0 / in_features)

        if num_bases and num_bases < num_relations:
            # Basis decomposition: W_r = Σ_b a_rb B_b
            self.bases = [np.random.randn(in_features, out_features) * scale
                         for _ in range(num_bases)]
            self.coefficients = np.random.randn(num_relations, num_bases) * 0.1
            self.use_basis = True
        else:
            # Direct weight matrices
            self.W = [np.random.randn(in_features, out_features) * scale
                     for _ in range(num_relations)]
            self.use_basis = False

        # Self-loop transformation
        self.W_0 = np.random.randn(in_features, out_features) * scale

    def get_weight(self, relation_idx):
        """Get weight matrix for given relation."""
        if self.use_basis:
            # Compute W_r = Σ_b a_rb B_b
            W = np.zeros((self.in_features, self.out_features))
            for b in range(self.num_bases):
                W += self.coefficients[relation_idx, b] * self.bases[b]
            return W
        else:
            return self.W[relation_idx]

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, H, adjacencies):
        """
        Forward pass.

        H: Node features (n × in_features)
        adjacencies: List of adjacency matrices, one per relation

        Returns: Updated features (n × out_features)
        """
        n = H.shape[0]

        # Aggregate from all relations
        output = np.zeros((n, self.out_features))

        for r, adj in enumerate(adjacencies):
            W_r = self.get_weight(r)

            # Normalize by degree
            degree = np.sum(adj, axis=1, keepdims=True) + 1e-10
            adj_norm = adj / degree

            # Message passing for this relation
            output += adj_norm @ H @ W_r

        # Self-loop
        output += H @ self.W_0

        # Activation
        if self.activation == 'relu':
            output = self.relu(output)

        return output


class RGCN:
    """
    Relational Graph Convolutional Network.

    For heterogeneous graphs with multiple edge types.
    """

    def __init__(self, n_features, hidden_dims, n_classes, n_relations, num_bases=None):
        self.layers = []

        dims = [n_features] + hidden_dims

        for i in range(len(dims) - 1):
            layer = RGCNLayer(dims[i], dims[i+1], n_relations,
                             num_bases=num_bases, activation='relu')
            self.layers.append(layer)

        # Output layer
        self.W_out = np.random.randn(hidden_dims[-1], n_classes) * 0.1
        self.b_out = np.zeros(n_classes)

    def forward(self, H, adjacencies):
        """Forward pass through all layers."""
        for layer in self.layers:
            H = layer.forward(H, adjacencies)

        # Output
        logits = H @ self.W_out + self.b_out

        # Softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs


class HAN:
    """
    Heterogeneous Attention Network.

    Uses metapaths to define semantic relationships.
    """

    def __init__(self, n_features, hidden_dim, n_classes, n_metapaths):
        self.n_metapaths = n_metapaths
        self.hidden_dim = hidden_dim

        # Per-metapath transformation
        scale = np.sqrt(2.0 / n_features)
        self.W_meta = [np.random.randn(n_features, hidden_dim) * scale
                      for _ in range(n_metapaths)]

        # Metapath attention
        self.a_meta = np.random.randn(n_metapaths, hidden_dim) * 0.1
        self.q = np.random.randn(hidden_dim) * 0.1

        # Output
        self.W_out = np.random.randn(hidden_dim, n_classes) * 0.1

    def forward(self, H, metapath_adjs):
        """
        Forward pass.

        H: Node features (n × in_features)
        metapath_adjs: List of adjacency matrices (one per metapath)

        Returns: Class probabilities
        """
        n = H.shape[0]

        # Process each metapath
        metapath_outputs = []

        for m, adj in enumerate(metapath_adjs):
            # Transform
            h_m = H @ self.W_meta[m]

            # Normalize adjacency
            degree = np.sum(adj, axis=1, keepdims=True) + 1e-10
            adj_norm = adj / degree

            # Aggregate
            h_m = adj_norm @ h_m
            metapath_outputs.append(h_m)

        # Stack: (n, n_metapaths, hidden)
        Z = np.stack(metapath_outputs, axis=1)

        # Metapath-level attention
        # Score each metapath for each node
        scores = np.zeros((n, self.n_metapaths))
        for m in range(self.n_metapaths):
            # Attention score: project to scalar per node
            # Z[:, m] is (n, hidden_dim), a_meta[m] is (hidden_dim,)
            scores[:, m] = np.tanh(Z[:, m] @ self.a_meta[m])  # (n,)

        # Softmax over metapaths
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

        # Weighted combination
        H_out = np.sum(Z * weights[:, :, np.newaxis], axis=1)

        # Output
        logits = H_out @ self.W_out
        logits_max = np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits - logits_max) / np.sum(np.exp(logits - logits_max), axis=1, keepdims=True)

        return probs, weights


def create_academic_graph():
    """
    Create example academic heterogeneous graph.

    Node types: Paper, Author, Venue
    Edge types: (Paper, cites, Paper), (Author, writes, Paper), (Paper, published_in, Venue)
    """
    g = HeterogeneousGraph()

    # Add papers
    n_papers = 20
    for i in range(n_papers):
        features = np.random.randn(16)
        g.add_node('paper', features)

    # Add authors
    n_authors = 10
    for i in range(n_authors):
        features = np.random.randn(16)
        g.add_node('author', features)

    # Add venues
    n_venues = 3
    for i in range(n_venues):
        features = np.random.randn(16)
        g.add_node('venue', features)

    # Add citation edges (paper → paper)
    for i in range(n_papers):
        n_citations = np.random.randint(1, 4)
        cited = np.random.choice(n_papers, n_citations, replace=False)
        for j in cited:
            if i != j:
                g.add_edge(i, j, 'cites')

    # Add authorship edges (author → paper)
    for a in range(n_papers, n_papers + n_authors):
        n_papers_authored = np.random.randint(1, 5)
        papers = np.random.choice(n_papers, n_papers_authored, replace=False)
        for p in papers:
            g.add_edge(a, p, 'writes')

    # Add venue edges (paper → venue)
    for p in range(n_papers):
        venue = np.random.randint(0, n_venues)
        g.add_edge(p, n_papers + n_authors + venue, 'published_in')

    return g


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_hetero_gnn():
    """
    Comprehensive heterogeneous GNN visualization.
    """
    print("\n" + "="*60)
    print("HETEROGENEOUS GNN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: Heterogeneous Graph Structure ============
    ax1 = fig.add_subplot(2, 3, 1)

    g = create_academic_graph()

    # Create positions
    n_papers = len(g.nodes.get('paper', []))
    n_authors = len(g.nodes.get('author', []))
    n_venues = len(g.nodes.get('venue', []))

    pos = {}
    # Papers in middle
    for i, p in enumerate(g.nodes.get('paper', [])):
        angle = 2 * np.pi * i / n_papers
        pos[p] = (2 * np.cos(angle), 2 * np.sin(angle))

    # Authors on outer ring
    for i, a in enumerate(g.nodes.get('author', [])):
        angle = 2 * np.pi * i / n_authors
        pos[a] = (3.5 * np.cos(angle), 3.5 * np.sin(angle))

    # Venues at center
    for i, v in enumerate(g.nodes.get('venue', [])):
        angle = 2 * np.pi * i / n_venues
        pos[v] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))

    # Draw edges by type
    edge_colors = {
        ('paper', 'cites', 'paper'): 'blue',
        ('author', 'writes', 'paper'): 'green',
        ('paper', 'published_in', 'venue'): 'red'
    }

    for edge_type, edges in g.edges.items():
        color = edge_colors.get(edge_type, 'gray')
        for src, dst in edges:
            ax1.plot([pos[src][0], pos[dst][0]], [pos[src][1], pos[dst][1]],
                    color=color, alpha=0.3, linewidth=0.5)

    # Draw nodes by type
    node_colors = {'paper': 'lightblue', 'author': 'lightgreen', 'venue': 'salmon'}
    node_markers = {'paper': 'o', 'author': 's', 'venue': '^'}

    for node_type, nodes in g.nodes.items():
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]
        ax1.scatter(x, y, c=node_colors[node_type], s=100,
                   marker=node_markers[node_type], edgecolors='black',
                   label=node_type, zorder=5)

    ax1.legend(loc='upper right')
    ax1.set_title('Heterogeneous Academic Graph\nPapers, Authors, Venues')
    ax1.axis('off')

    # ============ Plot 2: R-GCN Architecture ============
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.axis('off')

    rgcn_text = """
    R-GCN: Relational GCN
    ═════════════════════════════════

    Different weights for each relation!

    h_v' = σ(Σ_r Σ_{u∈N_r(v)} (1/c) W_r h_u + W_0 h_v)

    PARAMETER SHARING:
    ┌────────────────────────────┐
    │ Basis Decomposition:       │
    │   W_r = Σ_b a_rb B_b       │
    │   Share B bases            │
    ├────────────────────────────┤
    │ Block Diagonal:            │
    │   W_r = diag(Q_r^1, ...)   │
    │   Reduce parameters        │
    └────────────────────────────┘

    Example edge types:
    • Paper ──cites──> Paper
    • Author ──writes──> Paper
    • Paper ──in──> Venue
    """

    ax2.text(0.05, 0.95, rgcn_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('R-GCN Architecture')

    # ============ Plot 3: Basis Decomposition Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Compare different numbers of bases
    n_relations = 5
    n_features = 16
    bases_values = [1, 2, 5, 10]

    params_counts = []
    for num_bases in bases_values:
        if num_bases < n_relations:
            # Bases + coefficients
            params = num_bases * n_features * n_features + n_relations * num_bases
        else:
            # Full weights
            params = n_relations * n_features * n_features
        params_counts.append(params)

    x = np.arange(len(bases_values))
    ax3.bar(x, params_counts, color='steelblue', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{b} bases' for b in bases_values])
    ax3.set_ylabel('Number of Parameters')
    ax3.set_title('Basis Decomposition\nReduces parameters')
    ax3.axhline(y=n_relations * n_features * n_features, color='red',
               linestyle='--', label='Full weights')
    ax3.legend()

    # ============ Plot 4: HAN Metapath Attention ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Simple demo with random metapaths
    n_nodes = 10
    n_metapaths = 3
    H = np.random.randn(n_nodes, 16)

    metapath_adjs = [np.random.rand(n_nodes, n_nodes) > 0.7
                    for _ in range(n_metapaths)]

    han = HAN(n_features=16, hidden_dim=8, n_classes=2, n_metapaths=n_metapaths)
    _, weights = han.forward(H, metapath_adjs)

    # Show metapath weights for each node
    im = ax4.imshow(weights, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Metapath')
    ax4.set_ylabel('Node')
    ax4.set_xticks(range(n_metapaths))
    ax4.set_xticklabels(['APA', 'APVPA', 'APPA'])
    ax4.set_title('HAN Metapath Attention\nDifferent nodes weight metapaths differently')
    plt.colorbar(im, ax=ax4, label='Attention Weight')

    # ============ Plot 5: Edge Type Importance ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Simulate importance of different edge types
    edge_types = ['cites', 'writes', 'published_in']
    importances = [0.5, 0.3, 0.2]  # Example

    ax5.barh(range(len(edge_types)), importances, color=['blue', 'green', 'red'])
    ax5.set_yticks(range(len(edge_types)))
    ax5.set_yticklabels(edge_types)
    ax5.set_xlabel('Importance')
    ax5.set_title('Edge Type Importance\n(Example: Paper classification)')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Heterogeneous GNN
    ══════════════════════════════

    THE KEY IDEA:
    Different types need different processing!

    G = (V, E, τ_v, τ_e)
    Multiple node types, edge types

    APPROACHES:
    ┌────────────────────────────┐
    │ R-GCN: Per-relation W_r    │
    │   Basis decomposition      │
    │   for parameter sharing    │
    ├────────────────────────────┤
    │ HAN: Metapath attention    │
    │   Different semantic views │
    │   Learned importance       │
    ├────────────────────────────┤
    │ HGT: Type-specific Q,K,V   │
    │   Transformer + types      │
    └────────────────────────────┘

    APPLICATIONS:
    • Knowledge graphs
    • Citation networks
    • Recommender systems
    • Social networks
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))

    plt.suptitle('Heterogeneous GNN — Multiple Node and Edge Types\n'
                 'Different relations need different transformations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for heterogeneous GNN."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Number of basis matrices
    print("\n1. BASIS DECOMPOSITION EFFECT")
    print("-" * 40)

    n_relations = 5
    n_features = 16

    for num_bases in [1, 2, 3, 5]:
        # Create R-GCN layer
        layer = RGCNLayer(n_features, n_features, n_relations, num_bases=num_bases)

        # Count parameters
        if layer.use_basis:
            n_params = sum(b.size for b in layer.bases) + layer.coefficients.size
        else:
            n_params = sum(w.size for w in layer.W)
        n_params += layer.W_0.size

        print(f"num_bases={num_bases}  params={n_params}")

    print("→ Fewer bases = fewer parameters, but less expressive")

    # 2. Number of relations
    print("\n2. SCALING WITH NUMBER OF RELATIONS")
    print("-" * 40)

    for n_rels in [2, 5, 10, 20]:
        # Without basis decomposition
        layer_full = RGCNLayer(16, 16, n_rels, num_bases=None)
        params_full = sum(w.size for w in layer_full.W) + layer_full.W_0.size

        # With basis decomposition (only if num_bases < n_rels)
        num_bases = min(3, n_rels - 1) if n_rels > 3 else 2
        layer_basis = RGCNLayer(16, 16, n_rels, num_bases=num_bases)
        if layer_basis.use_basis:
            params_basis = sum(b.size for b in layer_basis.bases) + layer_basis.coefficients.size + layer_basis.W_0.size
        else:
            params_basis = params_full

        print(f"n_relations={n_rels:<3}  full={params_full:<6}  basis({num_bases})={params_basis}")

    print("→ Basis decomposition scales better with more relations")

    # 3. HAN metapath attention
    print("\n3. HAN METAPATH LEARNING")
    print("-" * 40)

    for n_metapaths in [1, 2, 3, 5]:
        n_nodes = 20
        H = np.random.randn(n_nodes, 16)
        metapath_adjs = [np.random.rand(n_nodes, n_nodes) > 0.7
                        for _ in range(n_metapaths)]

        han = HAN(n_features=16, hidden_dim=8, n_classes=2, n_metapaths=n_metapaths)
        _, weights = han.forward(H, metapath_adjs)

        # Measure weight entropy (higher = more uniform)
        entropy = -np.mean(np.sum(weights * np.log(weights + 1e-10), axis=1))
        print(f"n_metapaths={n_metapaths}  avg_entropy={entropy:.3f}")

    print("→ More metapaths = more distributed attention")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Heterogeneous GNN — Multiple Node and Edge Types")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_hetero_gnn()
    save_path = '/Users/sid47/ML Algorithms/42_hetero_gnn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Heterogeneous Graphs: Multiple node types, multiple edge types
2. R-GCN: Different weight matrix W_r for each relation
3. Basis decomposition: W_r = Σ_b a_rb B_b (parameter sharing)
4. HAN: Metapath attention (semantic-level aggregation)
5. HGT: Type-specific Q, K, V transformations
6. Applications: Knowledge graphs, citations, recommendations
    """)
