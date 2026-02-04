"""
GCN — Graph Convolutional Network
==================================

Paradigm: SPECTRAL CONVOLUTION ON GRAPHS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Convolution on graphs via the GRAPH LAPLACIAN.

THE GCN LAYER:
    H' = σ(Ã H W)

Where:
- H: node features (n × d_in)
- W: learnable weights (d_in × d_out)
- Ã: normalized adjacency + self-loop
- σ: activation function

Ã = D̃^(-1/2) (A + I) D̃^(-1/2)

WHERE THIS COMES FROM (Spectral Theory):

1. GRAPH LAPLACIAN: L = D - A
   - Eigenvalues = graph frequencies
   - Eigenvectors = Fourier basis for graphs

2. SPECTRAL CONVOLUTION:
   g * x = U g(Λ) U^T x
   (Filter in spectral domain, like FFT)

3. SIMPLIFICATION (Kipf & Welling 2017):
   Use first-order approximation → H' = σ(Ã H W)

===============================================================
WHAT GCN DOES INTUITIVELY
===============================================================

Each node AGGREGATES features from its neighbors!

h_v' = σ(Σ_{u ∈ N(v) ∪ {v}} 1/√(d_u × d_v) × h_u × W)

1. Take your own features
2. Take neighbors' features
3. Weight by degree normalization
4. Transform with W
5. Apply nonlinearity

This is SMOOTHING — connected nodes become more similar!

===============================================================
WHY SELF-LOOPS?
===============================================================

Without self-loop: node's own features can be washed out
With self-loop: node retains its own information

A + I = adjacency with self-connections

===============================================================
OVER-SMOOTHING PROBLEM
===============================================================

Deep GCN → all nodes converge to SAME representation!

WHY?
- Repeated smoothing = information diffusion
- After k layers: node sees k-hop neighborhood
- Too deep = see entire graph = everything looks the same

SOLUTIONS:
- Residual connections
- Jumping knowledge
- DropEdge
- Different normalization

===============================================================
INDUCTIVE BIAS
===============================================================

1. LOCALITY
   - Only immediate neighbors affect each layer
   - Multi-hop requires multiple layers

2. HOMOPHILY ASSUMPTION
   - Connected nodes are similar
   - GCN smooths features → enforces homophily

3. DEGREE NORMALIZATION
   - High-degree nodes don't dominate
   - Low-degree nodes aren't drowned out

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

# Import graph utilities
from importlib import import_module
graph_fund = import_module('35_graph_fundamentals')
Graph = graph_fund.Graph
karate_club = graph_fund.karate_club
create_community_graph = graph_fund.create_community_graph
spring_layout = graph_fund.spring_layout


class GCNLayer:
    """
    Single Graph Convolutional Layer.

    THE FORWARD PASS:
    H' = σ(Ã H W + b)

    Where Ã = D̃^(-1/2) (A + I) D̃^(-1/2)
    """

    def __init__(self, in_features, out_features, activation='relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        # For backprop
        self.cache = {}

    def forward(self, A_norm, H):
        """
        Forward pass.

        A_norm: Normalized adjacency (n × n)
        H: Node features (n × d_in)

        Returns: H' (n × d_out)
        """
        # Aggregate: ÃH
        AH = A_norm @ H

        # Transform: ÃHW + b
        Z = AH @ self.W + self.b

        # Activation
        if self.activation == 'relu':
            H_out = np.maximum(0, Z)
        elif self.activation == 'softmax':
            # Softmax for final layer
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            H_out = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            H_out = Z

        # Cache for backprop
        self.cache = {
            'A_norm': A_norm,
            'H': H,
            'AH': AH,
            'Z': Z,
            'H_out': H_out
        }

        return H_out

    def backward(self, dH_out, learning_rate=0.01):
        """
        Backward pass.

        dH_out: Gradient w.r.t. output (n × d_out)

        Returns: dH (gradient w.r.t. input)
        """
        A_norm = self.cache['A_norm']
        H = self.cache['H']
        AH = self.cache['AH']
        Z = self.cache['Z']

        # Gradient through activation
        if self.activation == 'relu':
            dZ = dH_out * (Z > 0)
        elif self.activation == 'softmax':
            # For softmax + cross-entropy, dZ = H_out - Y (passed directly)
            dZ = dH_out
        else:
            dZ = dH_out

        # Gradients
        dW = AH.T @ dZ
        db = np.sum(dZ, axis=0)

        # Gradient w.r.t. input
        dAH = dZ @ self.W.T
        dH = A_norm.T @ dAH  # A_norm is symmetric

        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dH


class GCN:
    """
    Graph Convolutional Network for node classification.

    Architecture:
    Input (n × d) → GCNLayer → ReLU → GCNLayer → ... → Softmax (n × c)
    """

    def __init__(self, n_features, hidden_dims, n_classes, dropout=0.5):
        """
        Parameters:
        - n_features: Input feature dimension
        - hidden_dims: List of hidden layer dimensions
        - n_classes: Number of output classes
        - dropout: Dropout rate (not implemented in this simple version)
        """
        self.dropout = dropout
        self.layers = []

        # Build layers
        dims = [n_features] + hidden_dims + [n_classes]
        for i in range(len(dims) - 1):
            activation = 'softmax' if i == len(dims) - 2 else 'relu'
            layer = GCNLayer(dims[i], dims[i+1], activation=activation)
            self.layers.append(layer)

    def forward(self, A_norm, X):
        """
        Forward pass through all layers.

        A_norm: Normalized adjacency
        X: Node features

        Returns: Class probabilities (n × c)
        """
        H = X
        for layer in self.layers:
            H = layer.forward(A_norm, H)
        return H

    def compute_loss(self, Y_pred, Y_true, mask=None):
        """
        Cross-entropy loss for node classification.

        mask: Boolean array indicating which nodes to compute loss on
              (for semi-supervised setting)
        """
        if mask is None:
            mask = np.ones(len(Y_true), dtype=bool)

        # Get predictions for masked nodes
        Y_pred_masked = Y_pred[mask]
        Y_true_masked = Y_true[mask]

        # Cross-entropy: -sum(y * log(p))
        eps = 1e-10
        loss = -np.mean(np.log(Y_pred_masked[np.arange(len(Y_true_masked)), Y_true_masked] + eps))

        return loss

    def backward(self, Y_pred, Y_true, mask=None, learning_rate=0.01):
        """
        Backward pass (backpropagation).

        For softmax + cross-entropy: dZ = Y_pred - Y_one_hot
        """
        n = len(Y_true)
        if mask is None:
            mask = np.ones(n, dtype=bool)

        # dZ for cross-entropy with softmax
        dZ = Y_pred.copy()
        Y_one_hot = np.zeros_like(Y_pred)
        Y_one_hot[np.arange(n), Y_true] = 1

        dZ[mask] = (Y_pred[mask] - Y_one_hot[mask]) / np.sum(mask)
        dZ[~mask] = 0  # No gradient for unlabeled nodes

        # Backprop through layers
        dH = dZ
        for layer in reversed(self.layers):
            dH = layer.backward(dH, learning_rate)

    def fit(self, graph, labels, train_mask, val_mask=None,
            epochs=200, learning_rate=0.01, verbose=True):
        """
        Train the GCN.

        graph: Graph object
        labels: Node labels (n,)
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        """
        # Compute normalized adjacency with self-loops
        A = graph.adj + np.eye(graph.n_nodes)  # A + I
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        X = graph.X

        train_losses = []
        val_accs = []

        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(A_norm, X)

            # Compute loss (only on training nodes)
            loss = self.compute_loss(Y_pred, labels, train_mask)
            train_losses.append(loss)

            # Backward pass
            self.backward(Y_pred, labels, train_mask, learning_rate)

            # Validation accuracy
            if val_mask is not None:
                val_pred = np.argmax(Y_pred[val_mask], axis=1)
                val_acc = np.mean(val_pred == labels[val_mask])
                val_accs.append(val_acc)

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"Epoch {epoch+1}: loss={loss:.4f}"
                if val_mask is not None:
                    msg += f", val_acc={val_acc:.3f}"
                print(msg)

        return train_losses, val_accs

    def predict(self, graph):
        """Predict labels for all nodes."""
        A = graph.adj + np.eye(graph.n_nodes)
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        Y_pred = self.forward(A_norm, graph.X)
        return np.argmax(Y_pred, axis=1)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_gcn():
    """
    Create comprehensive GCN visualization:
    1. Before/after GCN smoothing
    2. Learning curve
    3. Layer-wise representations
    4. Over-smoothing demonstration
    5. Feature propagation
    6. Summary
    """
    print("\n" + "="*60)
    print("GCN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # Get karate club
    graph, labels = karate_club()

    # Add random features
    graph.X = np.random.randn(graph.n_nodes, 16)

    # ============ Plot 1: Node Classification ============
    ax1 = fig.add_subplot(2, 3, 1)

    # Train GCN
    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 30, 32, 33]] = True  # 6 labeled nodes

    gcn = GCN(n_features=16, hidden_dims=[16], n_classes=2)
    losses, _ = gcn.fit(graph, labels, train_mask, epochs=200, verbose=False)

    # Predict all
    pred = gcn.predict(graph)
    acc = np.mean(pred == labels)

    # Visualize
    pos = spring_layout(graph)
    for i, j in graph.get_edge_list():
        ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    # Correct predictions: circles, Wrong: X
    colors = ['red' if l == 0 else 'blue' for l in labels]
    for i in range(graph.n_nodes):
        marker = 'o' if pred[i] == labels[i] else 'X'
        ax1.scatter(pos[i, 0], pos[i, 1], c=colors[i], s=100,
                   marker=marker, edgecolors='black', zorder=5)

    ax1.set_title(f'GCN Node Classification\n6 labeled → {acc:.0%} accuracy')
    ax1.axis('off')

    # ============ Plot 2: Learning Curve ============
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('GCN Training Loss\nConverges smoothly')
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Feature Smoothing Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Show how features smooth over layers
    graph2, labels2 = create_community_graph(n_communities=2, nodes_per_community=20,
                                             p_in=0.3, p_out=0.02, feature_dim=2)

    # Initial features
    X0 = graph2.X.copy()

    # After 1 GCN layer (just propagation, no learning)
    A = graph2.adj + np.eye(graph2.n_nodes)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    X1 = A_norm @ X0  # One propagation step

    # Plot both
    ax3.scatter(X0[:, 0], X0[:, 1], c=labels2, cmap='coolwarm',
               alpha=0.5, marker='o', label='Before GCN')
    ax3.scatter(X1[:, 0], X1[:, 1], c=labels2, cmap='coolwarm',
               alpha=0.8, marker='s', label='After 1 GCN layer')

    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_title('GCN Smoothing Effect\nConnected nodes become similar')
    ax3.legend(fontsize=9)

    # ============ Plot 4: Over-smoothing ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Track feature variance over layers
    n_layers_range = range(1, 15)
    variances = []

    for n_layers in n_layers_range:
        X = graph2.X.copy()
        for _ in range(n_layers):
            X = A_norm @ X

        # Variance of features
        var = np.mean(np.var(X, axis=0))
        variances.append(var)

    ax4.plot(list(n_layers_range), variances, 'b-o', linewidth=2)
    ax4.set_xlabel('Number of GCN Layers')
    ax4.set_ylabel('Feature Variance')
    ax4.set_title('OVER-SMOOTHING\nDeep GCN → all nodes same!')
    ax4.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Semi-supervised Performance ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Vary number of labeled nodes
    label_counts = [2, 4, 6, 10, 15, 20]
    accuracies = []

    for n_labels in label_counts:
        # Select n_labels nodes per class
        train_mask = np.zeros(graph.n_nodes, dtype=bool)
        for c in [0, 1]:
            class_indices = np.where(labels == c)[0]
            selected = np.random.choice(class_indices, min(n_labels//2, len(class_indices)), replace=False)
            train_mask[selected] = True

        gcn = GCN(n_features=16, hidden_dims=[16], n_classes=2)
        gcn.fit(graph, labels, train_mask, epochs=200, verbose=False)
        pred = gcn.predict(graph)
        acc = np.mean(pred == labels)
        accuracies.append(acc)

    ax5.plot(label_counts, accuracies, 'g-o', linewidth=2, markersize=8)
    ax5.set_xlabel('Number of Labeled Nodes')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Semi-Supervised GCN\nFew labels → good accuracy!')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.5, 1.05)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    GCN — Graph Convolutional Network
    ═══════════════════════════════════

    THE LAYER:
    H' = σ(Ã H W)

    Ã = D̃^(-1/2)(A + I)D̃^(-1/2)

    WHAT IT DOES:
    1. Add self-loops (A + I)
    2. Normalize by degree
    3. Aggregate neighbor features
    4. Transform with W
    5. Apply activation

    KEY PROPERTIES:
    ✓ Locality: 1-hop per layer
    ✓ Weight sharing: same W everywhere
    ✓ Smoothing: neighbors become similar

    OVER-SMOOTHING:
    ✗ Deep GCN fails!
    ✗ All nodes → same representation
    → Keep GCN shallow (2-3 layers)

    SOLUTIONS:
    • Residual connections
    • Skip connections
    • DropEdge
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('GCN — Graph Convolutional Network\n'
                 'Aggregate neighbors → Learn representations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for GCN."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    graph, labels = karate_club()
    graph.X = np.random.randn(graph.n_nodes, 16)

    # 1. Number of layers
    print("\n1. EFFECT OF GCN DEPTH")
    print("-" * 40)

    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 30, 32, 33]] = True

    for n_layers in [1, 2, 3, 4, 5]:
        hidden_dims = [16] * (n_layers - 1)
        gcn = GCN(n_features=16, hidden_dims=hidden_dims, n_classes=2)
        gcn.fit(graph, labels, train_mask, epochs=200, verbose=False)
        pred = gcn.predict(graph)
        acc = np.mean(pred == labels)
        print(f"layers={n_layers}  accuracy={acc:.3f}")

    print("→ 2-3 layers usually optimal")
    print("→ Deeper = over-smoothing!")

    # 2. Hidden dimension
    print("\n2. EFFECT OF HIDDEN DIMENSION")
    print("-" * 40)

    for hidden_dim in [4, 8, 16, 32, 64]:
        gcn = GCN(n_features=16, hidden_dims=[hidden_dim], n_classes=2)
        gcn.fit(graph, labels, train_mask, epochs=200, verbose=False)
        pred = gcn.predict(graph)
        acc = np.mean(pred == labels)
        n_params = 16*hidden_dim + hidden_dim + hidden_dim*2 + 2
        print(f"hidden={hidden_dim:<3}  params={n_params:<5}  accuracy={acc:.3f}")

    print("→ Moderate dimension sufficient")

    # 3. Learning rate
    print("\n3. EFFECT OF LEARNING RATE")
    print("-" * 40)

    for lr in [0.001, 0.01, 0.05, 0.1, 0.5]:
        gcn = GCN(n_features=16, hidden_dims=[16], n_classes=2)
        gcn.fit(graph, labels, train_mask, epochs=200, learning_rate=lr, verbose=False)
        pred = gcn.predict(graph)
        acc = np.mean(pred == labels)
        print(f"lr={lr:.3f}  accuracy={acc:.3f}")

    print("→ lr=0.01-0.1 typically works well")

    # 4. With vs without self-loops
    print("\n4. WITH vs WITHOUT SELF-LOOPS")
    print("-" * 40)

    # Without self-loops
    A_no_self = graph.adj.copy()
    D = np.diag(np.sum(A_no_self, axis=1) + 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_norm_no_self = D_inv_sqrt @ A_no_self @ D_inv_sqrt

    # With self-loops
    A_self = graph.adj + np.eye(graph.n_nodes)
    D = np.diag(np.sum(A_self, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_norm_self = D_inv_sqrt @ A_self @ D_inv_sqrt

    # Test both
    for name, A_n in [('No self-loops', A_norm_no_self), ('With self-loops', A_norm_self)]:
        # Manual forward pass
        X = graph.X.copy()
        W1 = np.random.randn(16, 16) * 0.1
        W2 = np.random.randn(16, 2) * 0.1

        for _ in range(200):
            H1 = np.maximum(0, A_n @ X @ W1)
            H2 = A_n @ H1 @ W2
            exp_H2 = np.exp(H2 - np.max(H2, axis=1, keepdims=True))
            Y_pred = exp_H2 / np.sum(exp_H2, axis=1, keepdims=True)

            # Simple gradient step on training nodes
            Y_true = np.zeros_like(Y_pred)
            Y_true[np.arange(graph.n_nodes), labels] = 1
            dH2 = (Y_pred - Y_true) / np.sum(train_mask)
            dH2[~train_mask] = 0

        pred = np.argmax(Y_pred, axis=1)
        acc = np.mean(pred == labels)
        print(f"{name:<20} accuracy={acc:.3f}")

    print("→ Self-loops important for preserving node features")

    # 5. Comparison with simple baselines
    print("\n5. GCN vs SIMPLE BASELINES")
    print("-" * 40)

    # Label propagation
    Y_one_hot = np.zeros((graph.n_nodes, 2))
    Y_one_hot[np.arange(graph.n_nodes), labels] = 1
    Y_one_hot[~train_mask] = 0

    A_norm = A_norm_self
    Y_prop = Y_one_hot.copy()
    for _ in range(10):
        Y_prop = A_norm @ Y_prop
        Y_prop[train_mask] = Y_one_hot[train_mask]

    lp_pred = np.argmax(Y_prop, axis=1)
    lp_acc = np.mean(lp_pred == labels)

    # MLP (no graph)
    mlp_acc = 0.5  # Random baseline (MLP would need more implementation)

    # GCN
    gcn = GCN(n_features=16, hidden_dims=[16], n_classes=2)
    gcn.fit(graph, labels, train_mask, epochs=200, verbose=False)
    gcn_acc = np.mean(gcn.predict(graph) == labels)

    print(f"Label Propagation:  {lp_acc:.3f}")
    print(f"Random Baseline:    {mlp_acc:.3f}")
    print(f"GCN:                {gcn_acc:.3f}")
    print("→ GCN combines feature learning + structure")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("GCN — Graph Convolutional Network")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_gcn()
    save_path = '/Users/sid47/ML Algorithms/36_gcn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. GCN: Convolve on graphs via normalized adjacency
2. H' = σ(Ã H W) where Ã = D̃^(-1/2)(A+I)D̃^(-1/2)
3. Aggregates neighbor features (smoothing)
4. Self-loops important for preserving node info
5. Over-smoothing: deep GCN fails!
6. Semi-supervised: few labels → good performance
    """)
