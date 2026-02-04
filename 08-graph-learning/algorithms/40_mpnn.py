"""
MPNN — Message Passing Neural Network
======================================

Paradigm: UNIFIED FRAMEWORK FOR GRAPH NEURAL NETWORKS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A GENERAL framework that encompasses GCN, GraphSAGE, GAT, GIN, etc.

THE MESSAGE PASSING PARADIGM:
    m_v^(t+1) = Σ_{u∈N(v)} M_t(h_v^t, h_u^t, e_vu)   (message)
    h_v^(t+1) = U_t(h_v^t, m_v^(t+1))                 (update)

TWO KEY FUNCTIONS:
1. MESSAGE M_t: How to compute message from neighbor u to v
2. UPDATE U_t: How to combine self-features with aggregated messages

===============================================================
INSTANTIATIONS
===============================================================

GCN:
    M = h_u / √(d_v × d_u)  (degree-normalized)
    U = σ(W × Σ messages)

GraphSAGE:
    M = h_u  (raw features)
    U = σ(W × [h_v || AGG(messages)])

GAT:
    M = α_vu × W h_u  (attention-weighted)
    U = σ(Σ messages)

GIN:
    M = h_u  (raw features)
    U = MLP((1+ε)h_v + Σ messages)

===============================================================
EDGE FEATURES
===============================================================

Many real graphs have EDGE attributes:
- Molecular bonds: single, double, triple
- Social networks: relationship type
- Transportation: distance, time

MPNN naturally incorporates edge features:
    M_t(h_v, h_u, e_vu)
    ↑
    Edge features!

===============================================================
READOUT FOR GRAPH-LEVEL TASKS
===============================================================

For graph classification, need to aggregate node features:

    h_G = READOUT({h_v : v ∈ G})

READOUT options:
1. SUM: Σ_v h_v (size-sensitive)
2. MEAN: (1/|V|) Σ_v h_v (size-invariant)
3. ATTENTION: Weighted sum
4. SET2SET: Recurrent aggregation

===============================================================
INDUCTIVE BIAS
===============================================================

1. LOCALITY: Aggregate from neighbors only
2. PERMUTATION INVARIANCE: Order of nodes doesn't matter
3. PARAMETER SHARING: Same function for all nodes
4. COMPOSITIONALITY: Stack layers for larger receptive field

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


class MPNNLayer:
    """
    General Message Passing Layer.

    m_v = AGG({M(h_v, h_u, e_vu) : u ∈ N(v)})  (message)
    h_v' = U(h_v, m_v)                          (update)

    Supports various message and update functions.
    """

    def __init__(self, in_features, out_features,
                 message_type='mlp', update_type='gru',
                 edge_features=None, activation='relu'):
        """
        Parameters:
        - in_features: Input node feature dimension
        - out_features: Output node feature dimension
        - message_type: 'linear', 'mlp', 'attention', 'edge_conditioned'
        - update_type: 'replace', 'residual', 'gru'
        - edge_features: Edge feature dimension (if any)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.message_type = message_type
        self.update_type = update_type
        self.edge_dim = edge_features
        self.activation_type = activation

        # Initialize message function parameters
        self._init_message_params()

        # Initialize update function parameters
        self._init_update_params()

        self.cache = {}

    def _init_message_params(self):
        """Initialize message function parameters."""
        scale = np.sqrt(2.0 / self.in_features)

        if self.message_type == 'linear':
            # M(h_u) = W h_u
            self.W_msg = np.random.randn(self.in_features, self.out_features) * scale

        elif self.message_type == 'mlp':
            # M(h_v, h_u) = MLP([h_v || h_u])
            self.W_msg1 = np.random.randn(2 * self.in_features, self.out_features) * scale
            self.b_msg1 = np.zeros(self.out_features)
            self.W_msg2 = np.random.randn(self.out_features, self.out_features) * 0.1
            self.b_msg2 = np.zeros(self.out_features)

        elif self.message_type == 'attention':
            # Like GAT
            self.W_msg = np.random.randn(self.in_features, self.out_features) * scale
            self.a = np.random.randn(2 * self.out_features, 1) * 0.1

        elif self.message_type == 'edge_conditioned':
            # M(h_v, h_u, e) = W_e h_u where W_e depends on edge
            self.W_msg = np.random.randn(self.in_features, self.out_features) * scale
            if self.edge_dim:
                self.W_edge = np.random.randn(self.edge_dim, self.out_features) * 0.1

    def _init_update_params(self):
        """Initialize update function parameters."""
        scale = np.sqrt(2.0 / self.out_features)

        if self.update_type == 'replace':
            # h' = m (just use message)
            pass

        elif self.update_type == 'residual':
            # h' = h + W m
            self.W_upd = np.random.randn(self.out_features, self.out_features) * scale

        elif self.update_type == 'gru':
            # GRU-like update
            # z = σ(W_z [h || m])  (update gate)
            # r = σ(W_r [h || m])  (reset gate)
            # h' = (1-z) * h + z * tanh(W_h [r*h || m])

            self.W_z = np.random.randn(2 * self.out_features, self.out_features) * scale
            self.W_r = np.random.randn(2 * self.out_features, self.out_features) * scale
            self.W_h = np.random.randn(2 * self.out_features, self.out_features) * scale

    def activation(self, x):
        """Apply activation function."""
        if self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'elu':
            return np.where(x > 0, x, np.exp(x) - 1)
        else:
            return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)

    def compute_messages(self, adj, H, edge_features=None):
        """
        Compute messages from all neighbors.

        Returns: messages (n × out_features)
        """
        n = H.shape[0]

        if self.message_type == 'linear':
            # Simple linear transformation
            transformed = H @ self.W_msg  # (n, out_features)
            # Sum over neighbors
            messages = adj @ transformed

        elif self.message_type == 'mlp':
            # MLP on concatenated features
            messages = np.zeros((n, self.out_features))

            for v in range(n):
                neighbors = np.where(adj[v] > 0)[0]
                if len(neighbors) == 0:
                    continue

                # Concatenate v's features with each neighbor's
                h_v = np.tile(H[v], (len(neighbors), 1))
                h_u = H[neighbors]
                concat = np.concatenate([h_v, h_u], axis=1)

                # MLP
                msg = self.activation(concat @ self.W_msg1 + self.b_msg1)
                msg = msg @ self.W_msg2 + self.b_msg2

                # Sum messages
                messages[v] = np.sum(msg, axis=0)

        elif self.message_type == 'attention':
            # Attention-weighted messages
            Wh = H @ self.W_msg  # (n, out_features)

            # Compute attention scores
            a_src = self.a[:self.out_features]
            a_dst = self.a[self.out_features:]

            e_src = Wh @ a_src
            e_dst = Wh @ a_dst
            e = self.leaky_relu(e_src + e_dst.T)

            # Mask non-neighbors
            e = np.where(adj > 0, e, -1e9)

            # Softmax
            e_max = np.max(e, axis=1, keepdims=True)
            exp_e = np.exp(e - e_max)
            exp_e = np.where(adj > 0, exp_e, 0)
            alpha = exp_e / (np.sum(exp_e, axis=1, keepdims=True) + 1e-10)

            # Weighted sum
            messages = alpha @ Wh

            self.cache['attention'] = alpha

        elif self.message_type == 'edge_conditioned':
            # Edge-conditioned messages
            transformed = H @ self.W_msg

            if edge_features is not None and self.edge_dim:
                # Modulate by edge features
                # (Simplified: add edge contribution)
                edge_contrib = np.zeros((n, n, self.out_features))
                for i in range(n):
                    for j in range(n):
                        if adj[i, j] > 0:
                            edge_contrib[i, j] = edge_features[i, j] @ self.W_edge

                messages = np.zeros((n, self.out_features))
                for v in range(n):
                    neighbors = np.where(adj[v] > 0)[0]
                    for u in neighbors:
                        messages[v] += transformed[u] + edge_contrib[v, u]
            else:
                messages = adj @ transformed

        return messages

    def update_nodes(self, H, messages):
        """
        Update node features based on messages.

        H: Current features (n × in_features)
        messages: Aggregated messages (n × out_features)

        Returns: Updated features (n × out_features)
        """
        # Project H if dimensions don't match
        if H.shape[1] != self.out_features:
            # Simple projection
            W_proj = np.random.randn(H.shape[1], self.out_features) * 0.1
            H = H @ W_proj

        if self.update_type == 'replace':
            return self.activation(messages)

        elif self.update_type == 'residual':
            return H + self.activation(messages @ self.W_upd)

        elif self.update_type == 'gru':
            # GRU-style update
            concat = np.concatenate([H, messages], axis=1)

            z = self.sigmoid(concat @ self.W_z)  # Update gate
            r = self.sigmoid(concat @ self.W_r)  # Reset gate

            concat_reset = np.concatenate([r * H, messages], axis=1)
            h_tilde = np.tanh(concat_reset @ self.W_h)

            return (1 - z) * H + z * h_tilde

    def forward(self, adj, H, edge_features=None):
        """
        Forward pass.

        adj: Adjacency matrix (n × n)
        H: Node features (n × in_features)
        edge_features: Optional edge features (n × n × edge_dim)

        Returns: Updated features (n × out_features)
        """
        # Add self-loops
        adj_with_self = adj + np.eye(adj.shape[0])

        # Compute messages
        messages = self.compute_messages(adj_with_self, H, edge_features)

        # Update nodes
        H_new = self.update_nodes(H, messages)

        return H_new


class MPNN:
    """
    Message Passing Neural Network for node classification.

    Configurable message and update functions.
    """

    def __init__(self, n_features, hidden_dims, n_classes,
                 message_type='mlp', update_type='gru'):
        """
        Parameters:
        - n_features: Input feature dimension
        - hidden_dims: List of hidden dimensions
        - n_classes: Number of output classes
        - message_type: Type of message function
        - update_type: Type of update function
        """
        self.layers = []

        dims = [n_features] + hidden_dims

        for i in range(len(dims) - 1):
            layer = MPNNLayer(
                dims[i], dims[i+1],
                message_type=message_type,
                update_type=update_type,
                activation='relu'
            )
            self.layers.append(layer)

        # Output layer
        self.W_out = np.random.randn(hidden_dims[-1], n_classes) * 0.1
        self.b_out = np.zeros(n_classes)

    def forward(self, graph):
        """Forward pass through all layers."""
        H = graph.X

        for layer in self.layers:
            H = layer.forward(graph.adj, H)

        # Output
        logits = H @ self.W_out + self.b_out

        # Softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def fit(self, graph, labels, train_mask, epochs=200, lr=0.01, verbose=True):
        """Train the model."""
        losses = []

        for epoch in range(epochs):
            probs = self.forward(graph)

            # Cross-entropy loss
            eps = 1e-10
            loss = -np.mean(np.log(probs[train_mask, labels[train_mask]] + eps))
            losses.append(loss)

            # Simplified gradient update
            n = graph.n_nodes
            Y_one_hot = np.zeros((n, probs.shape[1]))
            Y_one_hot[np.arange(n), labels] = 1

            d_logits = (probs - Y_one_hot) / np.sum(train_mask)
            d_logits[~train_mask] = 0

            # Update output layer
            H_last = self.layers[-1].cache.get('H', graph.X)
            if H_last.shape[1] != self.W_out.shape[0]:
                H_last = graph.X
            dW = H_last.T @ d_logits
            self.W_out -= lr * dW / n

            if verbose and (epoch + 1) % 50 == 0:
                pred = np.argmax(probs, axis=1)
                acc = np.mean(pred[train_mask] == labels[train_mask])
                print(f"Epoch {epoch+1}: loss={loss:.4f}, train_acc={acc:.3f}")

        return losses

    def predict(self, graph):
        """Predict labels."""
        probs = self.forward(graph)
        return np.argmax(probs, axis=1)


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def compare_message_functions():
    """Compare different message function types."""
    print("\n" + "="*60)
    print("COMPARING MESSAGE FUNCTIONS")
    print("="*60)

    np.random.seed(42)

    message_types = ['linear', 'mlp', 'attention']
    results = {}

    for msg_type in message_types:
        accs = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            train_mask = np.zeros(graph.n_nodes, dtype=bool)
            train_mask[[0, 1, 2, 30, 32, 33]] = True

            model = MPNN(
                n_features=16, hidden_dims=[16], n_classes=2,
                message_type=msg_type, update_type='residual'
            )
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        results[msg_type] = accs
        print(f"{msg_type:<12}  accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}")

    return results


def compare_update_functions():
    """Compare different update function types."""
    print("\n" + "="*60)
    print("COMPARING UPDATE FUNCTIONS")
    print("="*60)

    np.random.seed(42)

    update_types = ['replace', 'residual', 'gru']
    results = {}

    for upd_type in update_types:
        accs = []
        for _ in range(5):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            train_mask = np.zeros(graph.n_nodes, dtype=bool)
            train_mask[[0, 1, 2, 30, 32, 33]] = True

            model = MPNN(
                n_features=16, hidden_dims=[16], n_classes=2,
                message_type='mlp', update_type=upd_type
            )
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        results[upd_type] = accs
        print(f"{upd_type:<12}  accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}")

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_mpnn():
    """
    Comprehensive MPNN visualization:
    1. Framework diagram
    2. Message function comparison
    3. Update function comparison
    4. Unified view of GNNs
    5. Edge features effect
    6. Summary
    """
    print("\n" + "="*60)
    print("MPNN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    graph, labels = karate_club()
    graph.X = np.random.randn(graph.n_nodes, 16)

    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 30, 32, 33]] = True

    # ============ Plot 1: MPNN Classification ============
    ax1 = fig.add_subplot(2, 3, 1)

    model = MPNN(n_features=16, hidden_dims=[16], n_classes=2,
                 message_type='mlp', update_type='gru')
    losses = model.fit(graph, labels, train_mask, epochs=300, verbose=False)

    pred = model.predict(graph)
    acc = np.mean(pred == labels)

    pos = spring_layout(graph)

    # Draw edges
    for i, j in graph.get_edge_list():
        ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', alpha=0.3, linewidth=0.5)

    colors = ['red' if l == 0 else 'blue' for l in labels]
    for i in range(graph.n_nodes):
        marker = 'o' if pred[i] == labels[i] else 'X'
        ax1.scatter(pos[i, 0], pos[i, 1], c=colors[i], s=100,
                   marker=marker, edgecolors='black', zorder=5)

    ax1.set_title(f'MPNN Classification\nAccuracy: {acc:.0%}\nMLP message + GRU update')
    ax1.axis('off')

    # ============ Plot 2: Message Function Comparison ============
    ax2 = fig.add_subplot(2, 3, 2)

    message_types = ['linear', 'mlp', 'attention']
    msg_accs = []

    for msg_type in message_types:
        accs = []
        for _ in range(3):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            model = MPNN(n_features=16, hidden_dims=[16], n_classes=2,
                        message_type=msg_type, update_type='residual')
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))
        msg_accs.append(accs)

    means = [np.mean(a) for a in msg_accs]
    stds = [np.std(a) for a in msg_accs]

    x = np.arange(len(message_types))
    ax2.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'mediumseagreen'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Linear\n(GCN-like)', 'MLP\n(expressive)', 'Attention\n(GAT-like)'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Message Function Comparison\nHow to compute messages?')
    ax2.set_ylim(0, 1.1)

    # ============ Plot 3: Update Function Comparison ============
    ax3 = fig.add_subplot(2, 3, 3)

    update_types = ['replace', 'residual', 'gru']
    upd_accs = []

    for upd_type in update_types:
        accs = []
        for _ in range(3):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)
            model = MPNN(n_features=16, hidden_dims=[16], n_classes=2,
                        message_type='mlp', update_type=upd_type)
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))
        upd_accs.append(accs)

    means = [np.mean(a) for a in upd_accs]
    stds = [np.std(a) for a in upd_accs]

    x = np.arange(len(update_types))
    ax3.bar(x, means, yerr=stds, capsize=5, color=['lightcoral', 'lightskyblue', 'lightgreen'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Replace\n(h=m)', 'Residual\n(h+m)', 'GRU\n(gated)'])
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Update Function Comparison\nHow to combine h and m?')
    ax3.set_ylim(0, 1.1)

    # ============ Plot 4: GNN Unification ============
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')

    gnn_text = """
    MPNN Unifies All GNNs!
    ═══════════════════════════════════════

    GENERAL FORM:
    m_v = AGG({M(h_v, h_u, e) : u ∈ N(v)})
    h_v' = U(h_v, m_v)

    ┌──────────────┬────────────────────────┐
    │ GNN          │ MPNN Instantiation     │
    ├──────────────┼────────────────────────┤
    │ GCN          │ M = h_u/√(d_v×d_u)     │
    │              │ U = σ(W × Σ m)         │
    ├──────────────┼────────────────────────┤
    │ GraphSAGE    │ M = h_u                │
    │              │ U = σ(W × [h || AGG])  │
    ├──────────────┼────────────────────────┤
    │ GAT          │ M = α_vu × W h_u       │
    │              │ U = σ(Σ m)             │
    ├──────────────┼────────────────────────┤
    │ GIN          │ M = h_u                │
    │              │ U = MLP((1+ε)h + Σm)   │
    └──────────────┴────────────────────────┘

    All are special cases of MPNN!
    """

    ax4.text(0.05, 0.95, gnn_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============ Plot 5: Learning Curves ============
    ax5 = fig.add_subplot(2, 3, 5)

    configs = [
        ('linear', 'replace', 'r--'),
        ('mlp', 'residual', 'b-'),
        ('attention', 'gru', 'g-.')
    ]

    for msg_type, upd_type, style in configs:
        graph, labels = karate_club()
        graph.X = np.random.randn(graph.n_nodes, 16)

        model = MPNN(n_features=16, hidden_dims=[16], n_classes=2,
                    message_type=msg_type, update_type=upd_type)
        losses = model.fit(graph, labels, train_mask, epochs=200, verbose=False)

        ax5.plot(losses, style, linewidth=2,
                label=f'{msg_type}/{upd_type}')

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Learning Curves\nDifferent MPNN configurations')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    MPNN — Message Passing Neural Network
    ══════════════════════════════════════

    THE KEY IDEA:
    A unified framework for GNNs!

    m_v = AGG({M(h_v, h_u, e) : u ∈ N(v)})
    h_v' = U(h_v, m_v)

    MESSAGE FUNCTIONS M:
    • Linear (GCN-style)
    • MLP (expressive)
    • Attention (GAT-style)
    • Edge-conditioned

    UPDATE FUNCTIONS U:
    • Replace (h' = m)
    • Residual (h' = h + m)
    • GRU (gated recurrent)

    BENEFITS:
    ✓ Unifies all GNN architectures
    ✓ Flexible design space
    ✓ Natural edge features
    ✓ Clear abstraction
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('MPNN — Message Passing Neural Network\n'
                 'A unified framework for graph neural networks',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for MPNN."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    train_mask = np.zeros(34, dtype=bool)
    train_mask[[0, 1, 2, 30, 32, 33]] = True

    # 1. Message function type
    print("\n1. MESSAGE FUNCTION TYPE")
    print("-" * 40)
    compare_message_functions()
    print("→ MLP most expressive, attention adaptive")

    # 2. Update function type
    print("\n2. UPDATE FUNCTION TYPE")
    print("-" * 40)
    compare_update_functions()
    print("→ GRU preserves information best")

    # 3. Number of layers
    print("\n3. NUMBER OF LAYERS")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4]:
        accs = []
        for _ in range(3):
            graph, labels = karate_club()
            graph.X = np.random.randn(graph.n_nodes, 16)

            model = MPNN(n_features=16, hidden_dims=[16]*n_layers, n_classes=2,
                        message_type='mlp', update_type='residual')
            model.fit(graph, labels, train_mask, epochs=200, verbose=False)
            accs.append(np.mean(model.predict(graph) == labels))

        print(f"n_layers={n_layers}  accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}")

    print("→ 2-3 layers typically sufficient")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("MPNN — Message Passing Neural Network")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_mpnn()
    save_path = '/Users/sid47/ML Algorithms/40_mpnn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. MPNN: Unified framework for all GNNs
2. Two key functions: MESSAGE M and UPDATE U
3. m_v = AGG({M(h_v, h_u, e) : u ∈ N(v)})
4. h_v' = U(h_v, m_v)
5. Instantiates GCN, GraphSAGE, GAT, GIN, etc.
6. Naturally incorporates edge features
7. Flexible design space for experimentation
    """)
