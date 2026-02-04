"""
GraphSAGE — Inductive Graph Learning
=====================================

Paradigm: SAMPLE AND AGGREGATE

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

LEARN to aggregate neighbor features (don't use fixed weights per node)!

THE GRAPHSAGE FRAMEWORK:
    h_v^(k) = σ(W × CONCAT(h_v^(k-1), AGG({h_u^(k-1) : u ∈ N(v)})))

BREAKDOWN:
1. h_v^(k-1): Node v's features at layer k-1
2. AGG: Aggregate function over neighbors
3. CONCAT: Combine own features with aggregated neighbors
4. W: Learnable weight matrix
5. σ: Nonlinearity (ReLU)

===============================================================
WHY GRAPHSAGE > GCN?
===============================================================

1. INDUCTIVE: Can generalize to UNSEEN nodes/graphs!
   - GCN learns weights for the SPECIFIC graph
   - GraphSAGE learns an AGGREGATION FUNCTION
   - Apply to new graphs, new nodes!

2. SAMPLING: Don't need full neighborhood!
   - Sample k neighbors uniformly
   - Makes computation tractable for large graphs
   - Mini-batch training possible

3. FLEXIBLE AGGREGATORS:
   - Mean: Simple, effective
   - Max-Pooling: Captures salient features
   - LSTM: Learns complex combinations

===============================================================
AGGREGATORS
===============================================================

1. MEAN AGGREGATOR
   AGG_mean = mean({h_u : u ∈ N(v)})

   Simple average of neighbor features.
   Similar to GCN (but with sampling).

2. MAX-POOLING AGGREGATOR
   AGG_pool = max({σ(W_pool h_u + b) : u ∈ N(v)})

   Element-wise max after transformation.
   Captures most "activated" feature per dimension.

3. LSTM AGGREGATOR
   AGG_lstm = LSTM({h_u : u ∈ N(v)})

   Process neighbors as sequence.
   Problem: neighbors have no natural order!
   Solution: Random permutation (works surprisingly well)

===============================================================
SAMPLING STRATEGY
===============================================================

THE CHALLENGE:
Full neighborhood can be huge (degree explosion)
Layer 1: k neighbors
Layer 2: k² neighbors
Layer L: k^L neighbors

SOLUTION: Sample fixed number at each layer

S(v) = sample(N(v), size=k)

Typical: k=10-25 neighbors per layer

TRADE-OFF:
- Larger sample = more accurate, slower
- Smaller sample = faster, more variance

===============================================================
TRAINING MODES
===============================================================

1. SUPERVISED
   - Given labeled nodes, minimize classification loss
   - Standard cross-entropy

2. UNSUPERVISED (Random Walk)
   Loss = -log(σ(z_u · z_v)) - Σ E[log(σ(-z_u · z_w))]

   Maximize similarity for:
   - Nodes in same random walks (positive)
   Minimize similarity for:
   - Random node pairs (negative)

   Similar to word2vec skip-gram!

===============================================================
INDUCTIVE BIAS
===============================================================

1. AGGREGATION FUNCTION
   - Same function everywhere
   - Generalizes to new graphs

2. LOCALITY
   - Still aggregates from local neighborhood
   - But LEARNS how to aggregate

3. SAMPLING
   - Approximates full aggregation
   - Trade accuracy for scalability

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


class MeanAggregator:
    """
    MEAN AGGREGATOR

    AGG_mean(v) = mean({h_u : u ∈ S(v)})

    Simple and effective. Similar to GCN.
    """

    def __init__(self):
        self.name = 'mean'

    def aggregate(self, neighbor_features):
        """
        neighbor_features: list of feature vectors from neighbors
        """
        if len(neighbor_features) == 0:
            return np.zeros_like(neighbor_features[0]) if neighbor_features else None
        return np.mean(neighbor_features, axis=0)


class PoolingAggregator:
    """
    MAX-POOLING AGGREGATOR

    AGG_pool(v) = max({σ(W h_u + b) : u ∈ S(v)})

    First transform, then take element-wise maximum.
    Captures most "activated" features.
    """

    def __init__(self, input_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = input_dim

        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W = np.random.randn(input_dim, hidden_dim) * scale
        self.b = np.zeros(hidden_dim)
        self.name = 'pool'

    def aggregate(self, neighbor_features):
        if len(neighbor_features) == 0:
            return np.zeros(self.W.shape[1])

        # Transform each neighbor
        transformed = []
        for h in neighbor_features:
            z = h @ self.W + self.b
            transformed.append(np.maximum(0, z))  # ReLU

        # Element-wise max
        return np.max(transformed, axis=0)


class GraphSAGELayer:
    """
    Single GraphSAGE Layer

    h_v^(k) = σ(W · CONCAT(h_v^(k-1), AGG({h_u^(k-1)})))
    """

    def __init__(self, input_dim, output_dim, aggregator_type='mean',
                 sample_size=10, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sample_size = sample_size
        self.activation = activation

        # Aggregator
        if aggregator_type == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator_type == 'pool':
            self.aggregator = PoolingAggregator(input_dim)
        else:
            self.aggregator = MeanAggregator()

        # Weight for concatenated features: [self || neighbors]
        # Input: 2 * input_dim (self + aggregated neighbors)
        concat_dim = 2 * input_dim

        scale = np.sqrt(2.0 / (concat_dim + output_dim))
        self.W = np.random.randn(concat_dim, output_dim) * scale
        self.b = np.zeros(output_dim)

        self.cache = {}

    def sample_neighbors(self, graph, node, current_features):
        """Sample neighbors for a node."""
        neighbors = graph.get_neighbors(node)

        if len(neighbors) == 0:
            # No neighbors: use self
            return [current_features[node]]

        if len(neighbors) <= self.sample_size:
            sampled = neighbors
        else:
            sampled = np.random.choice(neighbors, self.sample_size, replace=False)

        return [current_features[n] for n in sampled]

    def forward(self, graph, features):
        """
        Forward pass for all nodes.

        graph: Graph object
        features: (n_nodes, input_dim) current node features

        Returns: (n_nodes, output_dim) new node features
        """
        n_nodes = graph.n_nodes
        output = np.zeros((n_nodes, self.output_dim))

        all_concatenated = []

        for v in range(n_nodes):
            # Sample and aggregate neighbors
            neighbor_features = self.sample_neighbors(graph, v, features)
            aggregated = self.aggregator.aggregate(neighbor_features)

            if aggregated is None:
                aggregated = np.zeros(self.input_dim)

            # Concatenate self and aggregated
            concat = np.concatenate([features[v], aggregated])
            all_concatenated.append(concat)

            # Transform
            z = concat @ self.W + self.b

            # Activation
            if self.activation == 'relu':
                output[v] = np.maximum(0, z)
            else:
                output[v] = z

        self.cache = {
            'features': features,
            'concatenated': np.array(all_concatenated),
            'output': output
        }

        # L2 normalize (important for GraphSAGE)
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        output = output / (norms + 1e-10)

        return output

    def backward(self, d_output, learning_rate=0.01):
        """Backward pass (simplified)."""
        concat = self.cache['concatenated']
        output = self.cache['output']

        # Gradient through L2 norm (simplified: ignore for now)
        d_z = d_output

        # Gradient through activation
        if self.activation == 'relu':
            d_z = d_z * (output > 0)

        # Gradient for W and b
        dW = concat.T @ d_z
        db = np.sum(d_z, axis=0)

        # Update
        self.W -= learning_rate * dW / len(d_output)
        self.b -= learning_rate * db / len(d_output)

        # Gradient for input (for further backprop)
        d_concat = d_z @ self.W.T

        # Split into self and aggregated gradients
        d_features = d_concat[:, :self.input_dim]

        return d_features


class GraphSAGE:
    """
    GraphSAGE: Sample and Aggregate

    Inductive node embedding through learned aggregation.
    """

    def __init__(self, n_features, hidden_dims, n_classes,
                 aggregator_type='mean', sample_sizes=None):
        """
        Parameters:
        - n_features: Input feature dimension
        - hidden_dims: List of hidden dimensions
        - n_classes: Number of output classes
        - aggregator_type: 'mean' or 'pool'
        - sample_sizes: Number of neighbors to sample at each layer
        """
        self.layers = []

        if sample_sizes is None:
            sample_sizes = [10] * (len(hidden_dims) + 1)

        # Build layers
        dims = [n_features] + hidden_dims + [n_classes]
        for i in range(len(dims) - 1):
            activation = 'none' if i == len(dims) - 2 else 'relu'
            layer = GraphSAGELayer(
                dims[i], dims[i+1],
                aggregator_type=aggregator_type,
                sample_size=sample_sizes[i],
                activation=activation
            )
            self.layers.append(layer)

    def forward(self, graph):
        """Forward pass through all layers."""
        h = graph.X
        for layer in self.layers:
            h = layer.forward(graph, h)

        # Softmax for output
        exp_h = np.exp(h - np.max(h, axis=1, keepdims=True))
        probs = exp_h / np.sum(exp_h, axis=1, keepdims=True)

        return probs

    def compute_loss(self, probs, labels, mask):
        """Cross-entropy loss."""
        eps = 1e-10
        loss = -np.mean(np.log(probs[mask, labels[mask]] + eps))
        return loss

    def fit(self, graph, labels, train_mask, val_mask=None,
            epochs=200, learning_rate=0.01, verbose=True):
        """Train GraphSAGE."""
        train_losses = []
        val_accs = []

        for epoch in range(epochs):
            # Forward pass
            probs = self.forward(graph)
            loss = self.compute_loss(probs, labels, train_mask)
            train_losses.append(loss)

            # Backward pass (simplified)
            n = graph.n_nodes
            Y_one_hot = np.zeros((n, probs.shape[1]))
            Y_one_hot[np.arange(n), labels] = 1

            d_probs = (probs - Y_one_hot) / np.sum(train_mask)
            d_probs[~train_mask] = 0

            d_h = d_probs
            for layer in reversed(self.layers):
                d_h = layer.backward(d_h, learning_rate)

            # Validation
            if val_mask is not None:
                pred = np.argmax(probs[val_mask], axis=1)
                acc = np.mean(pred == labels[val_mask])
                val_accs.append(acc)

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"Epoch {epoch+1}: loss={loss:.4f}"
                if val_mask is not None:
                    msg += f", val_acc={acc:.3f}"
                print(msg)

        return train_losses, val_accs

    def predict(self, graph):
        """Predict labels."""
        probs = self.forward(graph)
        return np.argmax(probs, axis=1)


# ============================================================
# INDUCTIVE TESTING
# ============================================================

def test_inductive(model, train_graph, test_graph, train_labels, test_labels):
    """
    Test INDUCTIVE capability: Train on one graph, test on another!

    This is what makes GraphSAGE special.
    GCN can't do this easily.
    """
    # Predict on new graph
    pred = model.predict(test_graph)
    accuracy = np.mean(pred == test_labels)
    return accuracy


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_graphsage():
    """
    Create comprehensive GraphSAGE visualization:
    1. Node classification
    2. Aggregator comparison
    3. Sample size effect
    4. INDUCTIVE capability
    5. Embedding space
    6. Summary
    """
    print("\n" + "="*60)
    print("GRAPHSAGE VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    graph, labels = karate_club()
    graph.X = np.random.randn(graph.n_nodes, 16)

    # ============ Plot 1: GraphSAGE Classification ============
    ax1 = fig.add_subplot(2, 3, 1)

    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 30, 32, 33]] = True

    model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2,
                      aggregator_type='mean')
    losses, _ = model.fit(graph, labels, train_mask, epochs=200, verbose=False)

    pred = model.predict(graph)
    acc = np.mean(pred == labels)

    pos = spring_layout(graph)
    for i, j in graph.get_edge_list():
        ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    colors = ['red' if l == 0 else 'blue' for l in labels]
    for i in range(graph.n_nodes):
        marker = 'o' if pred[i] == labels[i] else 'X'
        ax1.scatter(pos[i, 0], pos[i, 1], c=colors[i], s=100,
                   marker=marker, edgecolors='black', zorder=5)

    ax1.set_title(f'GraphSAGE Classification\nAccuracy: {acc:.0%}')
    ax1.axis('off')

    # ============ Plot 2: Aggregator Comparison ============
    ax2 = fig.add_subplot(2, 3, 2)

    aggregators = ['mean', 'pool']
    n_runs = 5

    results = {}
    for agg in aggregators:
        accs = []
        for _ in range(n_runs):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2,
                              aggregator_type=agg)
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            pred = model.predict(graph)
            accs.append(np.mean(pred == labels))
        results[agg] = accs

    x = np.arange(len(aggregators))
    means = [np.mean(results[a]) for a in aggregators]
    stds = [np.std(results[a]) for a in aggregators]

    ax2.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Mean', 'Max-Pool'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Aggregator Comparison\nBoth work well on this task')
    ax2.set_ylim(0, 1.1)

    # ============ Plot 3: Sample Size Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    sample_sizes = [1, 3, 5, 10, 20]
    mean_accs = []
    std_accs = []

    for sample_size in sample_sizes:
        accs = []
        for _ in range(n_runs):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2,
                              sample_sizes=[sample_size, sample_size])
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))
        mean_accs.append(np.mean(accs))
        std_accs.append(np.std(accs))

    ax3.errorbar(sample_sizes, mean_accs, yerr=std_accs, marker='o',
                capsize=5, linewidth=2, markersize=8)
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Effect of Sample Size\nMore neighbors = better (but slower)')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: INDUCTIVE LEARNING ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Train on one graph, test on another!
    train_graph, train_labels = create_community_graph(
        n_communities=2, nodes_per_community=30, p_in=0.3, p_out=0.02
    )
    test_graph, test_labels = create_community_graph(
        n_communities=2, nodes_per_community=30, p_in=0.3, p_out=0.02
    )

    # Full training on train graph
    train_mask_full = np.ones(train_graph.n_nodes, dtype=bool)

    model = GraphSAGE(n_features=8, hidden_dims=[8], n_classes=2)
    model.fit(train_graph, train_labels, train_mask_full, epochs=200, verbose=False)

    # Test on DIFFERENT graph!
    train_acc = np.mean(model.predict(train_graph) == train_labels)
    test_acc = np.mean(model.predict(test_graph) == test_labels)

    x = np.arange(2)
    ax4.bar(x, [train_acc, test_acc], color=['steelblue', 'coral'])
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Train Graph', 'NEW Graph'])
    ax4.set_ylabel('Accuracy')
    ax4.set_title('INDUCTIVE LEARNING\nGeneralizes to unseen graph!')
    ax4.set_ylim(0, 1.1)

    # Add annotation
    ax4.annotate('Same model\ndifferent graph!', xy=(1, test_acc),
                xytext=(1.3, test_acc - 0.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # ============ Plot 5: Learning Curve ============
    ax5 = fig.add_subplot(2, 3, 5)

    graph, labels = karate_club()
    graph.X = np.random.randn(graph.n_nodes, 16)

    model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2)
    losses, _ = model.fit(graph, labels, train_mask, epochs=300, verbose=False)

    ax5.plot(losses, 'b-', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Cross-Entropy Loss')
    ax5.set_title('GraphSAGE Training\nLoss decreases smoothly')
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    GraphSAGE — Sample and Aggregate
    ═════════════════════════════════

    THE KEY IDEA:
    LEARN the aggregation function!

    h_v = σ(W·CONCAT(h_v, AGG(neighbors)))

    AGGREGATORS:
    • Mean: Simple average
    • Pool: Max after transform
    • LSTM: Sequence over neighbors

    ADVANTAGES:
    ┌─────────────────────────┐
    │ ✓ INDUCTIVE             │
    │   Works on new graphs!  │
    ├─────────────────────────┤
    │ ✓ SCALABLE              │
    │   Sample neighbors      │
    ├─────────────────────────┤
    │ ✓ MINI-BATCH            │
    │   Train on subgraphs    │
    └─────────────────────────┘

    vs GCN:
    GCN: Fixed aggregation (degree-weighted)
    GraphSAGE: LEARNED aggregation
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('GraphSAGE — Inductive Node Representation Learning\n'
                 'Sample neighbors, learn to aggregate, generalize to new graphs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for GraphSAGE."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Aggregator comparison
    print("\n1. AGGREGATOR TYPE")
    print("-" * 40)

    for agg in ['mean', 'pool']:
        accs = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            train_mask = np.zeros(graph.n_nodes, dtype=bool)
            train_mask[[0, 1, 2, 30, 32, 33]] = True

            model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2,
                              aggregator_type=agg)
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        print(f"{agg:<10}  accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}")

    print("→ Mean aggregator often sufficient")

    # 2. Sample size
    print("\n2. SAMPLE SIZE")
    print("-" * 40)

    for sample_size in [1, 5, 10, 20]:
        accs = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            train_mask = np.zeros(graph.n_nodes, dtype=bool)
            train_mask[[0, 1, 2, 30, 32, 33]] = True

            model = GraphSAGE(n_features=16, hidden_dims=[16], n_classes=2,
                              sample_sizes=[sample_size, sample_size])
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        print(f"sample_size={sample_size:<3}  accuracy={np.mean(accs):.3f}")

    print("→ Sampling ~10 neighbors typically sufficient")

    # 3. Number of layers
    print("\n3. NUMBER OF LAYERS")
    print("-" * 40)

    for n_layers in [1, 2, 3]:
        hidden_dims = [16] * (n_layers - 1) if n_layers > 1 else []
        accs = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            train_mask = np.zeros(graph.n_nodes, dtype=bool)
            train_mask[[0, 1, 2, 30, 32, 33]] = True

            model = GraphSAGE(n_features=16, hidden_dims=hidden_dims, n_classes=2)
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        print(f"layers={n_layers}  accuracy={np.mean(accs):.3f}")

    print("→ 2 layers usually optimal (like GCN)")

    # 4. INDUCTIVE TEST
    print("\n4. INDUCTIVE GENERALIZATION")
    print("-" * 40)

    inductive_accs = []
    transductive_accs = []

    for _ in range(5):
        # Create two different graphs
        train_graph, train_labels = create_community_graph(
            n_communities=2, nodes_per_community=40, p_in=0.3, p_out=0.02
        )
        test_graph, test_labels = create_community_graph(
            n_communities=2, nodes_per_community=40, p_in=0.3, p_out=0.02
        )

        train_mask = np.ones(train_graph.n_nodes, dtype=bool)

        model = GraphSAGE(n_features=8, hidden_dims=[8], n_classes=2)
        model.fit(train_graph, train_labels, train_mask, epochs=200, verbose=False)

        transductive_accs.append(np.mean(model.predict(train_graph) == train_labels))
        inductive_accs.append(np.mean(model.predict(test_graph) == test_labels))

    print(f"Transductive (train graph): {np.mean(transductive_accs):.3f}")
    print(f"Inductive (new graph):      {np.mean(inductive_accs):.3f}")
    print("→ GraphSAGE generalizes to unseen graphs!")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GRAPHSAGE — Inductive Graph Learning")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_graphsage()
    save_path = '/Users/sid47/ML Algorithms/37_graphsage.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. GraphSAGE: Learn to aggregate neighbor features
2. h_v = σ(W·CONCAT(h_v, AGG(neighbors)))
3. INDUCTIVE: Generalizes to unseen nodes/graphs!
4. SAMPLING: Makes large graphs tractable
5. Aggregators: Mean, Max-Pool, LSTM
6. Key insight: Learn the aggregation, not fixed weights!
    """)
