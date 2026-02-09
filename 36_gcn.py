"""
GCN — Graph Convolutional Network

Paradigm: SPECTRAL CONVOLUTION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

The first practical Graph Neural Network (Kipf & Welling, 2017).

Each layer SMOOTHS features across the graph:

    H' = σ(Ã H W)

Where:
    H  : node features (n × d_in)
    W  : learnable weight matrix (d_in × d_out)
    Ã  : normalized adjacency with self-loops
    σ  : activation (ReLU)

What this does per node:

    h_v' = σ( Σ_{u ∈ N(v)∪{v}} (1/√(d_u × d_v)) × h_u × W )

    "Average your neighbors' features (weighted by degree),
     transform with W, apply nonlinearity."

===============================================================
WHERE IT COMES FROM (Spectral Theory)
===============================================================

1. GRAPH LAPLACIAN: L = D - A
   Eigenvalues = graph frequencies
   Eigenvectors = Fourier basis for graphs

2. SPECTRAL CONVOLUTION:
   g * x = U g(Λ) U^T x
   Filtering in the spectral domain (like FFT for graphs)

3. SIMPLIFICATION (Kipf & Welling):
   First-order Chebyshev approximation →
   H' = σ(Ã H W)

   Connection to 58_spectral_clustering.py:
   GCN propagation ≈ Laplacian smoothing ≈ low-pass filter

===============================================================
THE NORMALIZED ADJACENCY
===============================================================

Ã = D̃^(-1/2) (A + I) D̃^(-1/2)

Why A + I? Self-loops: node retains its own features.
Why D̃^(-1/2)...D̃^(-1/2)? Symmetric normalization:
    - Prevents high-degree nodes from dominating
    - Row sums ≤ 1 → numerically stable

===============================================================
OVER-SMOOTHING PROBLEM
===============================================================

After k GCN layers, each node sees its k-hop neighborhood.

Too many layers → all nodes converge to same representation!
    Layer 1: 1-hop (direct neighbors)
    Layer 2: 2-hop (friends of friends)
    Layer 4: most of graph visible
    Layer 8: everything looks the same

This is because GCN is a LOW-PASS FILTER:
    Repeated smoothing → constant signal

PRACTICAL RULE: 2-3 layers is usually optimal.

===============================================================
INDUCTIVE BIAS
===============================================================

1. HOMOPHILY: Connected nodes should have similar labels
   (friends have same interests, cited papers share topics)

2. SMOOTHNESS: Node features should vary slowly across edges

3. LOCAL: Each layer aggregates 1-hop, k layers = k-hop

4. DEGREE NORMALIZATION: High-degree nodes don't dominate

WHEN IT WORKS:
    ✓ Homophilic graphs (same-class nodes cluster together)
    ✓ Semi-supervised (few labels + structure)
    ✓ Medium-size graphs (< 10k nodes)

WHEN IT FAILS:
    ✗ Heterophilic graphs (neighbors have DIFFERENT labels)
    ✗ Very deep (over-smoothing)
    ✗ Inductive (can't generalize to new graphs easily)

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


# ============================================================
# GCN IMPLEMENTATION
# ============================================================

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-10)


def cross_entropy_loss(probs, y, mask=None):
    """Cross-entropy loss with optional mask."""
    n = len(y)
    log_probs = np.log(probs[np.arange(n), y] + 1e-10)
    if mask is not None:
        return -np.mean(log_probs[mask])
    return -np.mean(log_probs)


def compute_normalized_adjacency(A, self_loops=True):
    """
    Compute Ã = D̃^(-1/2)(A+I)D̃^(-1/2).

    This is the core operation of GCN.
    """
    A_hat = A.copy()
    if self_loops:
        A_hat = A_hat + np.eye(A.shape[0])

    d = np.sum(A_hat, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCN:
    """
    Graph Convolutional Network with full backpropagation.

    Paradigm: SPECTRAL CONVOLUTION
    - Each layer: H' = σ(Ã H W)
    - Multi-layer → multi-hop aggregation
    - Full gradient-based training
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=2,
                 dropout=0.5, lr=0.01, weight_decay=5e-4,
                 random_state=None):
        """
        Parameters:
        -----------
        n_features : int
            Input feature dimension
        n_hidden : int
            Hidden layer dimension
        n_classes : int
            Number of output classes
        n_layers : int
            Number of GCN layers (2-3 recommended)
        dropout : float
            Dropout rate (applied to hidden layers)
        lr : float
            Learning rate
        weight_decay : float
            L2 regularization strength
        random_state : int or None
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all layers."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        # Layer dimensions: features → hidden → ... → hidden → classes
        dims = [self.n_features] + [self.n_hidden] * (self.n_layers - 1) + [self.n_classes]

        for i in range(len(dims) - 1):
            # Xavier initialization
            std = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * std
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X, A_hat, training=True):
        """
        Forward pass through all GCN layers.

        GCN layer: H' = σ(Ã H W + b)

        Returns: (output_probs, cache_for_backprop)
        """
        H = X.copy()
        cache = {'H': [H], 'Z': [], 'dropout_masks': []}

        for l in range(self.n_layers):
            # Graph convolution: Ã H W + b
            Z = A_hat @ H @ self.weights[l] + self.biases[l]
            cache['Z'].append(Z)

            if l < self.n_layers - 1:
                # Hidden layer: ReLU + dropout
                H = np.maximum(Z, 0)  # ReLU

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

    def backward(self, A_hat, y, mask, cache):
        """
        Full backpropagation through all GCN layers.

        Chain rule through:
        softmax → linear → (ReLU → dropout → linear) × (n_layers - 1)
        """
        n = len(y)
        probs = cache['H'][-1]  # softmax output

        # Gradient of cross-entropy + softmax
        dZ = probs.copy()
        dZ[np.arange(n), y] -= 1

        # Mask: only backprop through train nodes
        mask_float = mask.astype(float)
        n_train = np.sum(mask)
        dZ = dZ * mask_float[:, None] / n_train

        grad_W = []
        grad_b = []

        for l in range(self.n_layers - 1, -1, -1):
            H_prev = cache['H'][l]

            # Gradient for W_l: dL/dW = H_prev^T @ (Ã^T @ dZ)
            dZ_prop = A_hat.T @ dZ
            gW = H_prev.T @ dZ_prop + self.weight_decay * self.weights[l]
            gb = np.sum(dZ_prop, axis=0)

            grad_W.insert(0, gW)
            grad_b.insert(0, gb)

            if l > 0:
                # Backprop through linear: dH = dZ @ W^T, then through Ã
                dH = dZ_prop @ self.weights[l].T

                # Backprop through dropout
                dH = dH * cache['dropout_masks'][l-1] / (1 - self.dropout + 1e-10)

                # Backprop through ReLU
                dZ = dH * (cache['Z'][l-1] > 0).astype(float)

        # Update weights
        for l in range(self.n_layers):
            self.weights[l] -= self.lr * grad_W[l]
            self.biases[l] -= self.lr * grad_b[l]

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train GCN on a graph with transductive split.

        Parameters:
        -----------
        graph : Graph
            Input graph
        labels : ndarray
            Node labels (all nodes, but only train_mask used for loss)
        train_mask : boolean array
            Which nodes are in training set
        n_epochs : int
        verbose : bool

        Returns: loss_history
        """
        A_hat = compute_normalized_adjacency(graph.A)
        X = graph.X

        loss_history = []

        for epoch in range(n_epochs):
            # Forward
            probs, cache = self.forward(X, A_hat, training=True)

            # Loss (only on train nodes)
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

            # Backward
            self.backward(A_hat, labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                train_pred = np.argmax(probs, axis=1)
                train_acc = np.mean(train_pred[train_mask] == labels[train_mask])
                print(f"  Epoch {epoch+1:>4}: loss={loss:.4f}, train_acc={train_acc:.3f}")

        return loss_history

    def predict(self, graph):
        """Predict node labels."""
        A_hat = compute_normalized_adjacency(graph.A)
        probs, _ = self.forward(graph.X, A_hat, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, graph):
        """Predict node class probabilities."""
        A_hat = compute_normalized_adjacency(graph.A)
        probs, _ = self.forward(graph.X, A_hat, training=False)
        return probs

    def get_embeddings(self, graph, layer=-2):
        """Get node embeddings from a specific layer."""
        A_hat = compute_normalized_adjacency(graph.A)
        _, cache = self.forward(graph.X, A_hat, training=False)
        return cache['H'][layer]


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # Use community graph
    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=30,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=42
    )
    n_classes = 3
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    # -------- Experiment 1: Number of Layers (Over-smoothing) --------
    print("\n1. NUMBER OF LAYERS (Over-Smoothing)")
    print("-" * 40)
    print("More layers = larger receptive field, but too many → over-smoothing")

    for n_layers in [1, 2, 3, 4, 6, 8]:
        gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=n_layers,
                  lr=0.01, dropout=0.5, random_state=42)
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = gcn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        # Measure over-smoothing: cosine similarity of embeddings
        emb = gcn.get_embeddings(graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms
        cos_sim = emb_norm @ emb_norm.T
        avg_sim = (np.sum(cos_sim) - graph.n_nodes) / (graph.n_nodes * (graph.n_nodes - 1))

        print(f"  layers={n_layers}  test_acc={test_acc:.3f}"
              f"  avg_cos_sim={avg_sim:.3f}")

    print("-> 2-3 layers optimal")
    print("-> Deep → all embeddings converge (high cosine similarity)")
    print("-> This IS over-smoothing: nodes become indistinguishable")

    # -------- Experiment 2: Self-Loops --------
    print("\n2. SELF-LOOPS (A+I vs A)")
    print("-" * 40)

    for self_loops in [True, False]:
        gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
                  lr=0.01, random_state=42)
        A_hat = compute_normalized_adjacency(graph.A, self_loops=self_loops)
        X = graph.X

        for epoch in range(200):
            probs, cache = gcn.forward(X, A_hat, training=True)
            gcn.backward(A_hat, labels, train_mask, cache)

        probs, _ = gcn.forward(X, A_hat, training=False)
        preds = np.argmax(probs, axis=1)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        label = "A+I (self-loops)" if self_loops else "A (no self-loops)"
        print(f"  {label:<25}  test_acc={test_acc:.3f}")

    print("-> Self-loops preserve node's own features")
    print("-> Without: node identity washed out by neighbors")

    # -------- Experiment 3: Hidden Dimension --------
    print("\n3. HIDDEN DIMENSION")
    print("-" * 40)

    for hidden in [4, 8, 16, 32, 64]:
        gcn = GCN(graph.X.shape[1], hidden, n_classes, n_layers=2,
                  lr=0.01, random_state=42)
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = gcn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        n_params = sum(w.size for w in gcn.weights)
        print(f"  hidden={hidden:<4}  test_acc={test_acc:.3f}"
              f"  params={n_params}")

    print("-> 16-32 usually sufficient")
    print("-> Diminishing returns above 32 for small graphs")

    # -------- Experiment 4: Learning Rate --------
    print("\n4. LEARNING RATE")
    print("-" * 40)

    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
                  lr=lr, random_state=42)
        losses = gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = gcn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        final_loss = losses[-1] if losses else float('inf')
        print(f"  lr={lr:<6}  test_acc={test_acc:.3f}"
              f"  final_loss={final_loss:.4f}")

    print("-> 0.01 is a good default")
    print("-> Too high → unstable, too low → slow convergence")

    # -------- Experiment 5: Dropout --------
    print("\n5. DROPOUT")
    print("-" * 40)

    for dropout in [0.0, 0.2, 0.5, 0.8]:
        gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
                  lr=0.01, dropout=dropout, random_state=42)
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = gcn.predict(graph)
        train_acc = np.mean(preds[train_mask] == labels[train_mask])
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        gap = train_acc - test_acc
        print(f"  dropout={dropout:.1f}  train={train_acc:.3f}"
              f"  test={test_acc:.3f}  gap={gap:.3f}")

    print("-> 0.5 typically best for GCNs")
    print("-> No dropout → overfitting (large train-test gap)")
    print("-> Too much dropout → underfitting")

    # -------- Experiment 6: Features vs Structure --------
    print("\n6. FEATURES vs STRUCTURE")
    print("-" * 40)
    print("What matters more: node features or graph structure?")

    # With features + structure
    gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
              lr=0.01, random_state=42)
    gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
    preds = gcn.predict(graph)
    acc_both = np.mean(preds[test_mask] == labels[test_mask])

    # Features only (no graph structure → identity adjacency)
    graph_no_edges = graph.copy()
    graph_no_edges.A = np.eye(graph.n_nodes)
    gcn2 = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
               lr=0.01, random_state=42)
    gcn2.fit(graph_no_edges, labels, train_mask, n_epochs=200, verbose=False)
    preds2 = gcn2.predict(graph_no_edges)
    acc_feat = np.mean(preds2[test_mask] == labels[test_mask])

    # Structure only (no features → one-hot identity)
    graph_no_feat = graph.copy()
    graph_no_feat.X = np.eye(graph.n_nodes)
    gcn3 = GCN(graph.n_nodes, 16, n_classes, n_layers=2,
               lr=0.01, random_state=42)
    gcn3.fit(graph_no_feat, labels, train_mask, n_epochs=200, verbose=False)
    preds3 = gcn3.predict(graph_no_feat)
    acc_struct = np.mean(preds3[test_mask] == labels[test_mask])

    print(f"  Features + Structure:  {acc_both:.3f}")
    print(f"  Features only (MLP):   {acc_feat:.3f}")
    print(f"  Structure only:        {acc_struct:.3f}")
    print("-> GCN benefits from BOTH features and structure")
    print("-> Structure alone can propagate few labels to many nodes")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_gcn():
    """Main GCN visualization."""
    print("\nGenerating: GCN visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Use karate club
    graph, labels = karate_club()
    n_classes = 2
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )
    pos = spring_layout(graph, seed=42)

    # Train GCN
    gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
              lr=0.01, dropout=0.3, random_state=42)
    losses = gcn.fit(graph, labels, train_mask, n_epochs=300, verbose=False)

    # Panel 1: Ground truth
    draw_graph(graph, labels, pos, axes[0, 0],
              title='Ground Truth\n(Karate Club)', cmap='coolwarm')

    # Panel 2: GCN predictions
    preds = gcn.predict(graph)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])
    draw_graph(graph, preds, pos, axes[0, 1],
              title=f'GCN Predictions\ntest_acc={test_acc:.3f}', cmap='coolwarm')

    # Panel 3: Training curve
    ax = axes[0, 2]
    ax.plot(losses, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Panel 4: Learned embeddings (2D)
    ax = axes[1, 0]
    emb = gcn.get_embeddings(graph)
    if emb.shape[1] > 2:
        # Simple PCA
        emb_centered = emb - emb.mean(axis=0)
        U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
        emb_2d = emb_centered @ Vt[:2].T
    else:
        emb_2d = emb
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='coolwarm',
              s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.set_title('Learned Embeddings\n(classes separated in embedding space)')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

    # Panel 5: Train vs test accuracy
    ax = axes[1, 1]
    graph2, labels2 = create_community_graph(3, 25, 0.3, 0.02, random_state=42)
    tm2, vm2, tsm2 = create_transductive_split(graph2.n_nodes, labels2, 0.15, 0.1)
    gcn2 = GCN(graph2.X.shape[1], 16, 3, n_layers=2, lr=0.01, random_state=42)

    train_accs, test_accs = [], []
    for epoch in range(200):
        A_hat = compute_normalized_adjacency(graph2.A)
        probs, cache = gcn2.forward(graph2.X, A_hat, training=True)
        gcn2.backward(A_hat, labels2, tm2, cache)

        probs_eval, _ = gcn2.forward(graph2.X, A_hat, training=False)
        preds2 = np.argmax(probs_eval, axis=1)
        train_accs.append(np.mean(preds2[tm2] == labels2[tm2]))
        test_accs.append(np.mean(preds2[tsm2] == labels2[tsm2]))

    ax.plot(train_accs, 'b-', label='Train', alpha=0.7)
    ax.plot(test_accs, 'r-', label='Test', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy\n(Community Graph)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: Confusion-style: correct vs incorrect
    ax = axes[1, 2]
    preds_final = gcn.predict(graph)
    correct = preds_final == labels
    colors_correct = np.where(correct, 0, 1)
    for i in range(graph.n_nodes):
        for j in range(i+1, graph.n_nodes):
            if graph.A[i,j] > 0:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                       'gray', alpha=0.2, linewidth=0.5)
    ax.scatter(pos[correct, 0], pos[correct, 1], c='green',
              s=80, alpha=0.8, edgecolors='black', linewidths=0.5,
              zorder=3, label='Correct')
    ax.scatter(pos[~correct, 0], pos[~correct, 1], c='red',
              s=100, alpha=0.9, edgecolors='black', linewidths=1,
              zorder=4, marker='X', label='Wrong')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f'Prediction Map\n{np.sum(correct)}/{graph.n_nodes} correct')
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('GCN: Graph Convolutional Network\n'
                 'Row 1: predictions & loss | Row 2: embeddings & accuracy',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_oversmoothing():
    """Visualize the over-smoothing problem."""
    print("\nGenerating: Over-smoothing visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    graph, labels = create_community_graph(
        3, 20, 0.3, 0.02, feature_dim=16, random_state=42
    )
    train_mask, _, test_mask = create_transductive_split(
        graph.n_nodes, labels, 0.15, 0.1
    )

    layer_counts = [1, 2, 4, 8]

    for idx, n_layers in enumerate(layer_counts):
        gcn = GCN(graph.X.shape[1], 16, 3, n_layers=n_layers,
                  lr=0.01, dropout=0.3, random_state=42)
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)

        emb = gcn.get_embeddings(graph)
        preds = gcn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        # Row 1: Cosine similarity heatmap
        ax = axes[0, idx]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms
        cos_sim = emb_norm @ emb_norm.T

        sort_idx = np.argsort(labels)
        cos_sorted = cos_sim[sort_idx][:, sort_idx]

        im = ax.imshow(cos_sorted, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'{n_layers} layer{"s" if n_layers > 1 else ""}\n'
                     f'test_acc={test_acc:.3f}')
        if idx == 0:
            ax.set_ylabel('Node (sorted by class)')

        # Row 2: Embedding PCA
        ax = axes[1, idx]
        emb_centered = emb - emb.mean(axis=0)
        if emb_centered.shape[1] >= 2:
            U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
            emb_2d = emb_centered @ Vt[:2].T
        else:
            emb_2d = np.column_stack([emb_centered[:, 0],
                                       np.zeros(emb_centered.shape[0])])
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='Set1',
                  s=40, alpha=0.7, edgecolors='black', linewidths=0.3)
        ax.set_title(f'Embedding Space ({n_layers}L)')
        if idx == 0:
            ax.set_ylabel('PC 2')
        ax.set_xlabel('PC 1')

    plt.colorbar(im, ax=axes[0, -1], fraction=0.046, label='Cosine Similarity')

    plt.suptitle('OVER-SMOOTHING: More layers → all nodes converge\n'
                 'Row 1: Node similarity (blue=different, red=same) | '
                 'Row 2: Embedding space',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def benchmark_on_datasets():
    """Benchmark GCN across standard graph datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: GCN on Graph Datasets")
    print("="*60)

    results = {}

    datasets = {
        'karate_club': (karate_club, 2),
        'community_2': (lambda: create_community_graph(2, 25, 0.3, 0.02), 2),
        'community_3': (lambda: create_community_graph(3, 20, 0.3, 0.02), 3),
        'community_4': (lambda: create_community_graph(4, 15, 0.3, 0.02), 4),
        'citation': (lambda: create_citation_network(100, 3, 16), 3),
    }

    print(f"\n{'Dataset':<15} {'Train Acc':<12} {'Test Acc':<12} {'Nodes':<8}")
    print("-" * 50)

    for name, (dataset_fn, n_classes) in datasets.items():
        graph, labels = dataset_fn()
        train_mask, val_mask, test_mask = create_transductive_split(
            graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
        )

        gcn = GCN(graph.X.shape[1], 16, n_classes, n_layers=2,
                  lr=0.01, dropout=0.5, random_state=42)
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)

        preds = gcn.predict(graph)
        train_acc = np.mean(preds[train_mask] == labels[train_mask])
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name:<15} {train_acc:<12.3f} {test_acc:<12.3f} {graph.n_nodes:<8}")

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GCN — Paradigm: SPECTRAL CONVOLUTION")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Each GCN layer: H' = σ(Ã H W)

    Ã = D̃^(-1/2)(A+I)D̃^(-1/2)  (normalized adjacency + self-loops)

    WHAT EACH LAYER DOES:
    1. Add self-loops: A + I
    2. Normalize: symmetric degree normalization
    3. Aggregate: multiply by Ã (smooth features across graph)
    4. Transform: multiply by learnable W
    5. Nonlinearity: ReLU (hidden), softmax (output)

KEY HYPERPARAMETERS:
    - n_layers: 2-3 (more → over-smoothing)
    - hidden_dim: 16-32 (larger → more expressive but slower)
    - dropout: 0.5 (important for regularization)
    - lr: 0.01 (standard Adam/SGD range)

INDUCTIVE BIAS:
    - HOMOPHILY: neighbors should be similar
    - LOCAL SMOOTHING: features averaged over neighbors
    - k layers = k-hop receptive field
    """)

    ablation_experiments()
    results = benchmark_on_datasets()

    print("\nGenerating visualizations...")

    fig1 = visualize_gcn()
    save_path1 = '/Users/sid47/ML Algorithms/36_gcn.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_oversmoothing()
    save_path2 = '/Users/sid47/ML Algorithms/36_gcn_oversmoothing.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "="*60)
    print("SUMMARY: What GCN Reveals")
    print("="*60)
    print("""
1. GCN = SMOOTHING
   Each layer averages features across neighbors.
   This is a low-pass filter on the graph signal.

2. OVER-SMOOTHING IS REAL
   2-3 layers optimal. More → all nodes converge.
   Node embeddings become indistinguishable.

3. SELF-LOOPS MATTER
   Without A+I, node's own features get washed out.

4. SEMI-SUPERVISED POWER
   10-15% labeled nodes can classify the rest!
   Structure propagates labels through the graph.

5. FEATURES + STRUCTURE > EITHER ALONE
   GCN combines feature similarity and graph structure.

CONNECTION TO OTHER FILES:
    58_spectral_clustering.py: GCN's Ã is related to Laplacian smoothing
    37_graphsage.py: Learns aggregation inductively (unlike GCN)
    38_gat.py: Learns attention weights (unlike GCN's fixed degree weights)
    39_gin.py: Uses SUM (more expressive than GCN's normalized mean)

NEXT: 37_graphsage.py — What if you need to generalize to NEW graphs?
    """)
