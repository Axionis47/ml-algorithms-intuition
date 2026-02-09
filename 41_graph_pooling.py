"""
Graph Pooling — Paradigm: HIERARCHICAL GRAPHS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

How to go from NODE representations to GRAPH representation?

GNNs produce node embeddings H in R^(n x d).
But for GRAPH CLASSIFICATION, we need ONE vector per graph.

This file addresses: Given a set of graphs, each with a label,
how do we classify entire graphs?

===============================================================
GLOBAL POOLING (Simple Approaches)
===============================================================

Given node embeddings H = [h_1, h_2, ..., h_n]:

1. SUM:  h_G = sum_v h_v
   - Size-sensitive (larger graphs = larger vectors)
   - Preserves total information
   - Used in GIN (provably most expressive)

2. MEAN: h_G = (1/n) * sum_v h_v
   - Size-invariant (same scale regardless of graph size)
   - Loses count information
   - Good when graph size varies widely

3. MAX:  h_G = max_v h_v  (element-wise)
   - Captures most salient feature per dimension
   - Loses everything but the maximum
   - Good for detecting specific motifs

===============================================================
HIERARCHICAL POOLING
===============================================================

Learn to COARSEN the graph:
    Graph -> Cluster -> Smaller Graph -> ... -> Final embedding

TopK Pooling:
    1. Score each node: y = sigma(X @ p)
    2. Keep top-k scored nodes: idx = top-k(y)
    3. Gate features by score: X' = X[idx] * y[idx]
    4. Create subgraph from selected nodes

DiffPool (Differential Pooling):
    1. Learn soft cluster assignments: S = softmax(GNN(A, X))
    2. Cluster features: X' = S^T @ X
    3. Cluster adjacency: A' = S^T @ A @ S
    4. End-to-end differentiable!

===============================================================
WHEN TO USE WHAT
===============================================================

- Small graphs, simple task -> Global SUM/MEAN
- Variable-size graphs -> Global MEAN
- Need expressiveness -> Global SUM (GIN-style)
- Multi-scale structure -> Hierarchical (TopK/DiffPool)
- Large graphs -> Hierarchical (reduces computation)

===============================================================
GRAPH CLASSIFICATION PIPELINE
===============================================================

The full pipeline:
    1. Node features X
    2. GNN layers: X -> H (node embeddings)
    3. POOLING: H -> h_G (graph embedding)
    4. MLP classifier: h_G -> class probabilities

===============================================================
INDUCTIVE BIAS
===============================================================

1. PERMUTATION INVARIANCE: Pooling must not depend on node order
   SUM, MEAN, MAX are all permutation invariant
2. SIZE INVARIANCE: Should handle different graph sizes
   MEAN is naturally size-invariant, SUM is not
3. INFORMATION PRESERVATION: Don't lose too much
   SUM > MEAN > MAX (in terms of information retained)

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
create_transductive_split = graph_module.create_transductive_split
spring_layout = graph_module.spring_layout
draw_graph = graph_module.draw_graph

gcn_module = import_module('36_gcn')
softmax = gcn_module.softmax
cross_entropy_loss = gcn_module.cross_entropy_loss
compute_normalized_adjacency = gcn_module.compute_normalized_adjacency


# ============================================================
# DATASET: Molecular-like graph classification
# ============================================================

def create_graph_classification_dataset(n_graphs=60, n_classes=3, random_state=42):
    """
    Create a dataset of small graphs for graph classification.

    Three classes of graphs with different structural properties:
    - Class 0: "Chain" graphs (path-like, linear)
    - Class 1: "Ring" graphs (cycle-like, closed loops)
    - Class 2: "Star" graphs (hub-and-spoke)

    Returns: list of (Graph, label) pairs
    """
    rng = np.random.RandomState(random_state)
    dataset = []

    graphs_per_class = n_graphs // n_classes

    for cls in range(n_classes):
        for i in range(graphs_per_class):
            if cls == 0:
                # Chain graphs: 5-10 nodes in a path
                n = rng.randint(5, 11)
                g = Graph(n)
                for j in range(n - 1):
                    g.add_edge(j, j + 1)
                # Add a few random edges for variety
                for _ in range(rng.randint(0, 3)):
                    a, b = rng.randint(0, n, size=2)
                    if a != b:
                        g.add_edge(a, b)

            elif cls == 1:
                # Ring graphs: 5-10 nodes in a cycle
                n = rng.randint(5, 11)
                g = Graph(n)
                for j in range(n):
                    g.add_edge(j, (j + 1) % n)
                # Add cross-ring edges
                for _ in range(rng.randint(0, 2)):
                    a, b = rng.randint(0, n, size=2)
                    if a != b and abs(a - b) > 1:
                        g.add_edge(a, b)

            else:
                # Star graphs: 1 hub + 4-9 spokes
                n_spokes = rng.randint(4, 10)
                n = n_spokes + 1
                g = Graph(n)
                for j in range(1, n):
                    g.add_edge(0, j)
                # Add a few spoke-to-spoke edges
                for _ in range(rng.randint(0, 3)):
                    a = rng.randint(1, n)
                    b = rng.randint(1, n)
                    if a != b:
                        g.add_edge(a, b)

            # Node features based on structural properties
            feat_dim = 8
            X = np.zeros((n, feat_dim))
            degs = g.degrees()
            X[:, 0] = degs / max(np.max(degs), 1)  # normalized degree
            X[:, 1] = 1.0 / (degs + 1)  # inverse degree
            X[:, 2] = (degs == 1).astype(float)  # leaf indicator
            X[:, 3] = (degs >= 3).astype(float)  # hub indicator
            X[:, 4] = np.arange(n) / n  # position encoding
            X[:, 5:] = rng.randn(n, feat_dim - 5) * 0.1  # noise
            g.X = X

            dataset.append((g, cls))

    # Shuffle
    indices = rng.permutation(len(dataset))
    dataset = [dataset[i] for i in indices]

    return dataset


# ============================================================
# GNN LAYER (shared by all pooling methods)
# ============================================================

class GNNLayer:
    """Simple GCN-style layer for graph classification pipeline."""

    def __init__(self, d_in, d_out, random_state=None):
        rng = np.random.RandomState(random_state)
        std = np.sqrt(2.0 / (d_in + d_out))
        self.W = rng.randn(d_in, d_out) * std
        self.b = np.zeros(d_out)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, A_hat, H):
        """H' = ReLU(A_hat @ H @ W + b)"""
        Z = A_hat @ H @ self.W + self.b
        H_out = np.maximum(Z, 0)
        return H_out, Z


# ============================================================
# GLOBAL POOLING
# ============================================================

class GlobalPooling:
    """Global pooling: aggregate all node features into one graph vector."""

    def __init__(self, method='sum'):
        """method: 'sum', 'mean', or 'max'"""
        self.method = method

    def forward(self, H):
        """
        H: (n_nodes, d) -> (d,) graph-level vector
        """
        if self.method == 'sum':
            return np.sum(H, axis=0)
        elif self.method == 'mean':
            return np.mean(H, axis=0)
        elif self.method == 'max':
            return np.max(H, axis=0)
        else:
            return np.sum(H, axis=0)

    def backward(self, dG, n_nodes, H=None):
        """
        Backprop through pooling.
        dG: (d,) gradient from classifier
        Returns: dH (n_nodes, d)
        """
        if self.method == 'sum':
            return np.tile(dG, (n_nodes, 1))
        elif self.method == 'mean':
            return np.tile(dG / n_nodes, (n_nodes, 1))
        elif self.method == 'max':
            # Gradient flows to the max elements
            dH = np.zeros_like(H)
            max_idx = np.argmax(H, axis=0)
            for d in range(H.shape[1]):
                dH[max_idx[d], d] = dG[d]
            return dH
        return np.tile(dG, (n_nodes, 1))


# ============================================================
# TOP-K POOLING
# ============================================================

class TopKPooling:
    """
    TopK Pooling: select top-k scored nodes.

    Score each node with a learnable projection,
    keep the top-k, gate features by score.
    """

    def __init__(self, d_in, ratio=0.5, random_state=None):
        """
        d_in: feature dimension
        ratio: fraction of nodes to keep
        """
        rng = np.random.RandomState(random_state)
        self.p = rng.randn(d_in) * 0.1  # scoring vector
        self.ratio = ratio
        self.d_in = d_in

    def forward(self, H, A):
        """
        H: (n, d) node features
        A: (n, n) adjacency matrix

        Returns: (H_pooled, A_pooled, idx, scores)
        """
        n = H.shape[0]
        k = max(1, int(n * self.ratio))

        # Score nodes
        scores = H @ self.p  # (n,)
        scores = np.tanh(scores)

        # Select top-k
        idx = np.argsort(scores)[-k:]
        idx = np.sort(idx)

        # Gate features by score
        H_pooled = H[idx] * scores[idx, None]

        # Create subgraph adjacency
        A_pooled = A[np.ix_(idx, idx)]

        return H_pooled, A_pooled, idx, scores


# ============================================================
# GRAPH CLASSIFIER (End-to-End)
# ============================================================

class GraphClassifier:
    """
    End-to-end graph classification:
    GNN layers -> Pooling -> MLP classifier

    Supports global pooling (sum/mean/max) and TopK pooling.
    Full backpropagation for training.
    """

    def __init__(self, n_features, n_hidden, n_classes, n_gnn_layers=2,
                 pool_method='sum', pool_ratio=0.5, lr=0.01,
                 dropout=0.3, random_state=None):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_gnn_layers = n_gnn_layers
        self.pool_method = pool_method
        self.lr = lr
        self.dropout = dropout

        rng = np.random.RandomState(random_state)

        # GNN layers
        self.gnn_layers = []
        for l in range(n_gnn_layers):
            d_in = n_features if l == 0 else n_hidden
            layer = GNNLayer(d_in, n_hidden, random_state=rng.randint(10000))
            self.gnn_layers.append(layer)

        # Pooling
        if pool_method == 'topk':
            self.pooling = TopKPooling(n_hidden, ratio=pool_ratio,
                                       random_state=rng.randint(10000))
        else:
            self.pooling = GlobalPooling(method=pool_method)

        # Classifier MLP: graph_embedding -> classes
        readout_dim = n_hidden * n_gnn_layers  # concat all layer outputs
        std = np.sqrt(2.0 / (readout_dim + n_hidden))
        self.W_mlp1 = rng.randn(readout_dim, n_hidden) * std
        self.b_mlp1 = np.zeros(n_hidden)
        std2 = np.sqrt(2.0 / (n_hidden + n_classes))
        self.W_mlp2 = rng.randn(n_hidden, n_classes) * std2
        self.b_mlp2 = np.zeros(n_classes)

    def forward_single(self, graph, training=True):
        """Forward pass for a single graph. Returns: (probs, cache)"""
        A = graph.A.astype(float)
        A_hat = compute_normalized_adjacency(A)
        H = graph.X.copy()

        cache = {
            'H_layers': [H.copy()],
            'Z_layers': [],
            'A_hat': A_hat,
        }

        all_H = []
        for l, layer in enumerate(self.gnn_layers):
            H_out, Z = layer.forward(A_hat, H)
            if training and self.dropout > 0:
                mask = (np.random.rand(*H_out.shape) > self.dropout).astype(float)
                H_out = H_out * mask / (1 - self.dropout + 1e-10)
            cache['H_layers'].append(H_out.copy())
            cache['Z_layers'].append(Z.copy())
            all_H.append(H_out.copy())
            H = H_out

        # Graph-level readout: concatenate pooled embeddings from all layers
        if self.pool_method == 'topk':
            H_pooled, A_pooled, idx, scores = self.pooling.forward(H, A)
            graph_embedding = np.sum(H_pooled, axis=0)
            graph_embedding = np.tile(graph_embedding, self.n_gnn_layers)[:self.n_hidden * self.n_gnn_layers]
            cache['topk'] = (idx, scores, H)
        else:
            pooled_layers = []
            for layer_H in all_H:
                pooled_layers.append(self.pooling.forward(layer_H))
            graph_embedding = np.concatenate(pooled_layers)

        cache['graph_embedding'] = graph_embedding
        cache['all_H'] = all_H

        # MLP classifier
        z1 = graph_embedding @ self.W_mlp1 + self.b_mlp1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W_mlp2 + self.b_mlp2

        exp_z = np.exp(z2 - np.max(z2))
        probs = exp_z / (np.sum(exp_z) + 1e-10)

        cache['z1'] = z1
        cache['h1'] = h1
        cache['z2'] = z2
        cache['probs'] = probs

        return probs, cache

    def backward_single(self, label, cache):
        """Backprop through a single graph."""
        probs = cache['probs']

        # Cross-entropy + softmax gradient
        dZ2 = probs.copy()
        dZ2[label] -= 1

        h1 = cache['h1']
        gW2 = np.outer(h1, dZ2)
        gb2 = dZ2

        dH1 = dZ2 @ self.W_mlp2.T
        dZ1 = dH1 * (cache['z1'] > 0).astype(float)

        graph_emb = cache['graph_embedding']
        gW1 = np.outer(graph_emb, dZ1)
        gb1 = dZ1

        dGraphEmb = dZ1 @ self.W_mlp1.T

        # Split gradient back to per-layer pooled embeddings
        all_H = cache['all_H']
        dH_per_layer = []
        offset = 0
        for layer_H in all_H:
            d = layer_H.shape[1]
            dPooled = dGraphEmb[offset:offset + d]
            dH = self.pooling.backward(dPooled, layer_H.shape[0], layer_H)
            dH_per_layer.append(dH)
            offset += d

        # Backprop through GNN layers
        A_hat = cache['A_hat']
        for l in range(self.n_gnn_layers - 1, -1, -1):
            dH = dH_per_layer[l]
            Z = cache['Z_layers'][l]
            H_prev = cache['H_layers'][l]

            dZ = dH * (Z > 0).astype(float)
            AH = A_hat @ H_prev
            gW = AH.T @ dZ
            gb = np.sum(dZ, axis=0)

            self.gnn_layers[l].W -= self.lr * gW
            self.gnn_layers[l].b -= self.lr * gb

        self.W_mlp2 -= self.lr * gW2
        self.b_mlp2 -= self.lr * gb2
        self.W_mlp1 -= self.lr * gW1
        self.b_mlp1 -= self.lr * gb1

    def fit(self, dataset, n_epochs=100, train_ratio=0.8, verbose=True):
        """Train on a dataset of (Graph, label) pairs."""
        n = len(dataset)
        n_train = int(n * train_ratio)
        train_data = dataset[:n_train]
        val_data = dataset[n_train:]

        loss_history = []
        train_accs = []
        val_accs = []

        for epoch in range(n_epochs):
            idx = np.random.permutation(len(train_data))
            epoch_loss = 0.0
            correct = 0

            for i in idx:
                graph, label = train_data[i]
                probs, cache = self.forward_single(graph, training=True)
                loss = -np.log(probs[label] + 1e-10)
                epoch_loss += loss
                if np.argmax(probs) == label:
                    correct += 1
                self.backward_single(label, cache)

            epoch_loss /= len(train_data)
            train_acc = correct / len(train_data)
            loss_history.append(epoch_loss)
            train_accs.append(train_acc)

            if len(val_data) > 0:
                val_correct = sum(1 for g, l in val_data
                                  if np.argmax(self.forward_single(g, False)[0]) == l)
                val_acc = val_correct / len(val_data)
            else:
                val_acc = 0.0
            val_accs.append(val_acc)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:>4}: loss={epoch_loss:.4f}, "
                      f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        return loss_history, train_accs, val_accs

    def evaluate(self, dataset):
        """Evaluate accuracy on a dataset."""
        correct = sum(1 for g, l in dataset
                      if np.argmax(self.forward_single(g, False)[0]) == l)
        return correct / len(dataset)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    dataset = create_graph_classification_dataset(60, 3, random_state=42)
    n_features = dataset[0][0].X.shape[1]
    n_classes = 3

    # -------- Experiment 1: Pooling method comparison --------
    print("\n1. POOLING METHOD COMPARISON")
    print("-" * 40)

    pool_results = {}
    for pool in ['sum', 'mean', 'max']:
        accs = []
        for trial in range(3):
            model = GraphClassifier(
                n_features, 16, n_classes, n_gnn_layers=2,
                pool_method=pool, lr=0.005, dropout=0.2,
                random_state=42 + trial
            )
            _, _, va = model.fit(dataset, n_epochs=80, train_ratio=0.8, verbose=False)
            accs.append(va[-1] if va else 0.0)
        mean_acc = np.mean(accs)
        pool_results[pool] = mean_acc
        print(f"  {pool:<6}  val_acc={mean_acc:.3f} +/- {np.std(accs):.3f}")

    print("-> SUM: preserves count info (best for distinguishing structures)")
    print("-> MEAN: size-invariant (good when graph sizes vary)")
    print("-> MAX: captures salient features only")

    # -------- Experiment 2: Number of GNN layers --------
    print("\n2. NUMBER OF GNN LAYERS")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4]:
        model = GraphClassifier(
            n_features, 16, n_classes, n_gnn_layers=n_layers,
            pool_method='sum', lr=0.005, dropout=0.2, random_state=42
        )
        _, _, va = model.fit(dataset, n_epochs=80, train_ratio=0.8, verbose=False)
        print(f"  layers={n_layers}  val_acc={va[-1]:.3f}")

    print("-> 2-3 layers captures enough structure for small graphs")

    # -------- Experiment 3: Hidden dimension --------
    print("\n3. HIDDEN DIMENSION")
    print("-" * 40)

    for n_hidden in [8, 16, 32, 64]:
        model = GraphClassifier(
            n_features, n_hidden, n_classes, n_gnn_layers=2,
            pool_method='sum', lr=0.005, dropout=0.2, random_state=42
        )
        _, _, va = model.fit(dataset, n_epochs=80, train_ratio=0.8, verbose=False)
        print(f"  hidden={n_hidden:<4}  val_acc={va[-1]:.3f}")

    print("-> 16-32 sufficient for simple structure recognition")

    return pool_results


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    print("\n" + "="*60)
    print("BENCHMARK: Graph Classification")
    print("="*60)

    results = {}
    for n_graphs, label in [(60, 'small_60'), (120, 'medium_120')]:
        dataset = create_graph_classification_dataset(n_graphs, 3, random_state=42)
        n_features = dataset[0][0].X.shape[1]

        model = GraphClassifier(
            n_features, 16, 3, n_gnn_layers=2,
            pool_method='sum', lr=0.005, dropout=0.2, random_state=42
        )
        losses, ta, va = model.fit(dataset, n_epochs=100, train_ratio=0.8, verbose=False)
        results[label] = {'train_acc': ta[-1], 'val_acc': va[-1]}
        print(f"\n{label}: train_acc={ta[-1]:.3f}, val_acc={va[-1]:.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_pooling():
    print("\nGenerating: Graph pooling visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    np.random.seed(42)

    dataset = create_graph_classification_dataset(60, 3, random_state=42)
    n_features = dataset[0][0].X.shape[1]
    n_classes = 3

    # Panel 1: Example graphs from each class
    ax = axes[0, 0]
    class_names = ['Chain', 'Ring', 'Star']
    class_colors = ['#e74c3c', '#3498db', '#2ecc71']

    examples = {}
    for g, label in dataset:
        if label not in examples:
            examples[label] = g
        if len(examples) == n_classes:
            break

    for cls in range(n_classes):
        g = examples[cls]
        pos = spring_layout(g, seed=cls * 10 + 1)
        offset_x = cls * 3.0
        pos[:, 0] += offset_x
        pos[:, 1] *= 0.8

        for i, j in g.edge_list:
            if i < j:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                       color=class_colors[cls], alpha=0.5, linewidth=1)
        ax.scatter(pos[:, 0], pos[:, 1], c=class_colors[cls], s=50,
                  edgecolors='black', linewidths=0.5, zorder=3)
        ax.text(offset_x + 0.5, -1.5, class_names[cls],
               ha='center', fontsize=10, fontweight='bold',
               color=class_colors[cls])

    ax.set_title('Three Graph Classes\nChain / Ring / Star')
    ax.set_aspect('equal')
    ax.axis('off')

    # Panel 2: Pooling comparison
    ax = axes[0, 1]
    pool_accs = {}
    for pool in ['sum', 'mean', 'max']:
        model = GraphClassifier(
            n_features, 16, n_classes, n_gnn_layers=2,
            pool_method=pool, lr=0.005, dropout=0.2, random_state=42
        )
        _, _, va = model.fit(dataset, n_epochs=80, train_ratio=0.8, verbose=False)
        pool_accs[pool] = va[-1] if va else 0.0

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(pool_accs.keys(), pool_accs.values(), color=colors,
                  edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars, pool_accs.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Global Pooling Comparison\nSUM vs MEAN vs MAX')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Training curve
    ax = axes[0, 2]
    model = GraphClassifier(
        n_features, 16, n_classes, n_gnn_layers=2,
        pool_method='sum', lr=0.005, dropout=0.2, random_state=42
    )
    losses, ta, va = model.fit(dataset, n_epochs=100, train_ratio=0.8, verbose=False)
    ax.plot(losses, 'b-', linewidth=1, alpha=0.7, label='Loss')
    ax2 = ax.twinx()
    ax2.plot(ta, 'g-', linewidth=1, alpha=0.7, label='Train')
    ax2.plot(va, 'r-', linewidth=1, alpha=0.7, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='blue')
    ax2.set_ylabel('Accuracy', color='red')
    ax.set_title('Training Curve\n(SUM pooling)')
    ax2.legend(loc='center right', fontsize=8)

    # Panel 4: Pipeline diagram
    ax = axes[1, 0]
    ax.axis('off')
    diagram = (
        "GRAPH CLASSIFICATION PIPELINE\n"
        "-------------------------------\n\n"
        "  Input Graph\n"
        "      |\n"
        "  GNN Layer 1: H = s(A_hat X W_1)\n"
        "      |            \\\n"
        "  GNN Layer 2: H = s(A_hat H W_2)\n"
        "      |              \\\n"
        "  POOL: sum/mean/max  POOL\n"
        "      |                |\n"
        "  CONCAT readouts from all layers\n"
        "      |\n"
        "  MLP: [h_G] -> classes\n"
        "      |\n"
        "  softmax -> prediction"
    )
    ax.text(0.05, 0.5, diagram, fontsize=10, fontfamily='monospace',
           va='center', ha='left',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Pipeline Overview')

    # Panel 5: Graph size distribution
    ax = axes[1, 1]
    sizes = [g.n_nodes for g, _ in dataset]
    labels_data = [l for _, l in dataset]
    for cls in range(n_classes):
        cls_sizes = [s for s, l in zip(sizes, labels_data) if l == cls]
        ax.hist(cls_sizes, bins=range(4, 13), alpha=0.5,
               label=class_names[cls], color=class_colors[cls])
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Count')
    ax.set_title('Graph Size Distribution\nby Class')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        "GRAPH POOLING SUMMARY\n"
        "------------------------\n\n"
        "GLOBAL POOLING:\n"
        "  SUM:  Preserves total info\n"
        "        Size-sensitive\n"
        "  MEAN: Size-invariant\n"
        "        Loses count info\n"
        "  MAX:  Captures salient features\n"
        "        Loses all but max\n\n"
        "MULTI-LAYER READOUT:\n"
        "  Concatenate pooled embeddings\n"
        "  from ALL GNN layers.\n"
        "  Each layer captures different\n"
        "  structural information.\n\n"
        "KEY INSIGHT:\n"
        "  SUM pooling + GIN layers\n"
        "  = maximally expressive\n"
        "  graph classification."
    )
    ax.text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
           va='center', ha='left',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.suptitle('Graph Pooling: From Node Embeddings to Graph Classification\n'
                 'GNN + Pooling + MLP = End-to-End Graph Classifier',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Graph Pooling -- Paradigm: HIERARCHICAL GRAPHS")
    print("="*60)

    print("""
WHAT THIS MODULE IS:
    Going from NODE embeddings to GRAPH embeddings.

    The full pipeline for graph classification:
        1. GNN layers: node features -> node embeddings
        2. POOLING: node embeddings -> graph embedding
        3. MLP: graph embedding -> class prediction

    GLOBAL POOLING:
        SUM:  h_G = sum(h_v)        -- size-sensitive, preserves info
        MEAN: h_G = mean(h_v)       -- size-invariant
        MAX:  h_G = max(h_v)        -- captures salient features

    GIN-STYLE READOUT:
        Concatenate pooled embeddings from ALL GNN layers.
        Each layer captures different structural info.
        h_G = CONCAT(POOL(H^1), POOL(H^2), ..., POOL(H^K))

KEY INSIGHT:
    The choice of pooling determines what graph-level info is preserved.
    SUM > MEAN > MAX in terms of information (like aggregation in GIN).
    """)

    pool_results = ablation_experiments()
    results = benchmark_on_datasets()

    print("\nGenerating visualizations...")

    fig = visualize_pooling()
    save_path = '/Users/sid47/ML Algorithms/41_graph_pooling.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY: What Graph Pooling Reveals")
    print("="*60)
    print("""
1. POOLING = NODE-TO-GRAPH AGGREGATION
   Just like message passing aggregates neighbors to nodes,
   pooling aggregates nodes to graphs.

2. SUM PRESERVES THE MOST
   SUM pooling + SUM aggregation (GIN) =
   maximally expressive graph classification.

3. MULTI-LAYER READOUT IS KEY
   Concatenating pooled embeddings from ALL layers
   captures multi-scale structural information.

4. GRAPH CLASSIFICATION = DIFFERENT PARADIGM
   Each graph is a SAMPLE (like an image in CNN).
   Standard train/test split on the set of graphs.

CONNECTION TO OTHER FILES:
    36_gcn.py: Node-level GNN (provides the layers)
    39_gin.py: Most expressive node aggregation (SUM)
    40_mpnn.py: Unified message passing framework
    42_hetero_gnn.py: Different node/edge types

NEXT: 42_hetero_gnn.py -- What if nodes and edges have different TYPES?
    """)
