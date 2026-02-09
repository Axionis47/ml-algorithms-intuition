"""
GAT -- Graph Attention Network

Paradigm: ATTENTION ON GRAPHS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Not all neighbors are equally important. LEARN attention weights!

THE GAT LAYER:
    h_i' = sigma(SUM_j alpha_ij W h_j)

Where attention alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))

BREAKDOWN:
1. W h_i: Transform node i's features (linear projection)
2. W h_j: Transform node j's features (same W)
3. [Wh_i || Wh_j]: Concatenate both projections
4. a^T [...]: Learned attention mechanism (a is a vector)
5. LeakyReLU: Nonlinearity (negative slope 0.2)
6. softmax over j in N(i): Normalize across neighbors
7. Result: Weighted combination of neighbor features

EFFICIENT DECOMPOSITION:
Instead of computing a^T [Wh_i || Wh_j] for every pair:
    a = [a_l || a_r]  (split attention vector)
    e_ij = a_l^T Wh_i + a_r^T Wh_j   (additive decomposition)

This avoids materializing the n x n x 2d concatenation tensor!

===============================================================
MULTI-HEAD ATTENTION
===============================================================

Like Transformers: use multiple independent attention heads!

HIDDEN LAYERS (concatenate):
    h_i' = ||_{k=1}^K sigma(SUM_j alpha_ij^k W^k h_j)

OUTPUT LAYER (average):
    h_i' = (1/K) SUM_k sigma(SUM_j alpha_ij^k W^k h_j)

WHY MULTI-HEAD?
- Different heads can capture different relationships
- Stabilizes the learning process (reduces variance)
- More expressive without proportional parameter increase

===============================================================
GAT vs GCN
===============================================================

GCN:
- Fixed weights based on degree: 1/sqrt(d_i * d_j)
- All neighbors contribute equally (after normalization)
- Simple, fast, fewer parameters

GAT:
- LEARNED weights based on FEATURES
- Different neighbors have different importance
- More expressive, more parameters
- Weights are INTERPRETABLE (can visualize attention)

WHEN ATTENTION HELPS:
- Neighbor importance varies (not all edges equal)
- Relationships are asymmetric (directed or heterogeneous)
- You want interpretability (attention as explanation)
- Enough data to learn meaningful attention patterns

WHEN GCN SUFFICES:
- Homogeneous graphs (all edges similar)
- Small datasets (attention may overfit)
- Computational budget is tight

===============================================================
ATTENTION VISUALIZATION
===============================================================

A powerful benefit: attention weights are INTERPRETABLE!

We can see:
- Which nodes the model focuses on
- What relationships it has learned
- How information flows through the graph
- Whether attention is concentrated or diffuse

HIGH ENTROPY attention = uniform (like GCN)
LOW ENTROPY attention = concentrated (GAT's strength)

===============================================================
INDUCTIVE BIAS
===============================================================

1. FEATURE-BASED ATTENTION
   - Attention depends on node features, not just structure
   - Richer than degree-based GCN weighting

2. LOCAL ATTENTION
   - Only attend to neighbors (masked softmax)
   - Sparse attention pattern (unlike Transformer)

3. SOFT ATTENTION
   - Differentiable (end-to-end training)
   - All neighbors contribute (just with different weights)

4. SHARED MECHANISM
   - Same attention parameters a for all node pairs
   - Inductive: can generalize to new nodes/graphs

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
GCN = gcn_module.GCN


# ============================================================
# GAT IMPLEMENTATION
# ============================================================

class GAT:
    """
    Graph Attention Network with full backpropagation.

    Paradigm: ATTENTION ON GRAPHS

    Each layer computes:
        For each head k:
            Wh = X @ W^k                           (project features)
            e_ij = LeakyReLU(a_l^k . Wh_i + a_r^k . Wh_j)  (attention logits)
            alpha_ij = softmax_j(e_ij)  over N(i)   (attention weights)
            h_i'^k = SUM_j alpha_ij^k * Wh_j        (aggregate)

        Hidden layers: h' = CONCAT(head_1, ..., head_K) then ReLU
        Output layer:  h' = MEAN(head_1, ..., head_K) then softmax

    Full gradient-based training with backprop through attention.
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=2,
                 n_heads=4, dropout=0.5, lr=0.01, random_state=None):
        """
        Parameters:
        -----------
        n_features : int
            Input feature dimension
        n_hidden : int
            Hidden dimension PER HEAD (total hidden = n_hidden * n_heads for concat layers)
        n_classes : int
            Number of output classes
        n_layers : int
            Number of GAT layers (2-3 recommended)
        n_heads : int
            Number of attention heads per layer
        dropout : float
            Dropout rate for features and attention
        lr : float
            Learning rate
        random_state : int or None
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.random_state = random_state

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for W and a per head per layer."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Layer dimensions:
        # Layer 0: n_features -> n_hidden (per head), n_heads heads, concat -> n_hidden * n_heads
        # Layer 1..n_layers-2: n_hidden * n_heads -> n_hidden (per head), concat
        # Layer n_layers-1: n_hidden * n_heads -> n_classes (per head), AVERAGE heads

        self.W = []   # W[layer][head]: (in_dim, out_dim)
        self.a_l = [] # a_l[layer][head]: (out_dim, 1) -- left attention
        self.a_r = [] # a_r[layer][head]: (out_dim, 1) -- right attention

        for layer in range(self.n_layers):
            # Input dimension for this layer
            if layer == 0:
                in_dim = self.n_features
            else:
                in_dim = self.n_hidden * self.n_heads

            # Output dimension for this layer
            is_last = (layer == self.n_layers - 1)
            if is_last:
                out_dim = self.n_classes
                n_heads_this = self.n_heads  # still use multi-head, but average
            else:
                out_dim = self.n_hidden
                n_heads_this = self.n_heads

            W_layer = []
            a_l_layer = []
            a_r_layer = []

            for h in range(n_heads_this):
                # Xavier initialization
                std = np.sqrt(2.0 / (in_dim + out_dim))
                W_layer.append(np.random.randn(in_dim, out_dim) * std)

                # Attention vectors -- small init
                a_l_layer.append(np.random.randn(out_dim, 1) * 0.1)
                a_r_layer.append(np.random.randn(out_dim, 1) * 0.1)

            self.W.append(W_layer)
            self.a_l.append(a_l_layer)
            self.a_r.append(a_r_layer)

    def _leaky_relu(self, x, alpha=0.2):
        """LeakyReLU activation."""
        return np.where(x > 0, x, alpha * x)

    def _leaky_relu_grad(self, x, alpha=0.2):
        """Gradient of LeakyReLU."""
        return np.where(x > 0, 1.0, alpha)

    def forward(self, graph, training=True):
        """
        Forward pass through all GAT layers.

        Returns: (output_probs, cache)

        Cache stores all intermediates needed for backpropagation:
        - A_self: adjacency with self-loops
        - H_input[layer]: input features to each layer
        - Wh[layer][head]: projected features
        - e_raw[layer][head]: raw attention logits (before LeakyReLU)
        - e_lrelu[layer][head]: after LeakyReLU
        - alpha[layer][head]: attention weights after masked softmax
        - head_out[layer][head]: per-head output (alpha @ Wh)
        - H_pre_act[layer]: concatenated/averaged multi-head output
        - dropout_masks[layer]: dropout masks for features
        - attn_dropout_masks[layer][head]: dropout masks for attention
        """
        A = graph.A
        n = graph.n_nodes
        A_self = A + np.eye(n)

        H = graph.X.copy()

        cache = {
            'A_self': A_self,
            'H_input': [],
            'Wh': [],
            'e_raw': [],
            'e_lrelu': [],
            'alpha': [],
            'head_out': [],
            'H_pre_act': [],
            'dropout_masks': [],
            'attn_dropout_masks': [],
        }

        for layer in range(self.n_layers):
            is_last = (layer == self.n_layers - 1)
            n_heads_this = self.n_heads

            cache['H_input'].append(H.copy())

            # Feature dropout (not on last layer)
            if training and self.dropout > 0 and not is_last:
                drop_mask = (np.random.rand(*H.shape) > self.dropout).astype(float)
                H_dropped = H * drop_mask / (1 - self.dropout + 1e-10)
                cache['dropout_masks'].append(drop_mask)
            else:
                H_dropped = H.copy()
                cache['dropout_masks'].append(np.ones_like(H))

            Wh_heads = []
            e_raw_heads = []
            e_lrelu_heads = []
            alpha_heads = []
            head_out_list = []
            attn_drop_heads = []

            for h in range(n_heads_this):
                W = self.W[layer][h]
                al = self.a_l[layer][h]
                ar = self.a_r[layer][h]

                # Project: Wh = H @ W  (n x out_dim)
                Wh = H_dropped @ W
                Wh_heads.append(Wh)

                # Attention logits: e_ij = a_l^T Wh_i + a_r^T Wh_j
                e_l = Wh @ al  # (n, 1)
                e_r = Wh @ ar  # (n, 1)
                e_raw = e_l + e_r.T  # (n, n) via broadcasting
                e_raw_heads.append(e_raw)

                # LeakyReLU
                e_lr = self._leaky_relu(e_raw)
                e_lrelu_heads.append(e_lr)

                # Masked softmax over neighbors (+ self-loops)
                e_masked = np.where(A_self > 0, e_lr, -1e9)
                e_max = np.max(e_masked, axis=1, keepdims=True)
                exp_e = np.exp(e_masked - e_max)
                exp_e = np.where(A_self > 0, exp_e, 0.0)
                alpha = exp_e / (np.sum(exp_e, axis=1, keepdims=True) + 1e-10)
                alpha_heads.append(alpha)

                # Attention dropout
                if training and self.dropout > 0:
                    attn_mask = (np.random.rand(*alpha.shape) > self.dropout).astype(float)
                    alpha_d = alpha * attn_mask
                    alpha_sum = np.sum(alpha_d, axis=1, keepdims=True) + 1e-10
                    alpha_d = alpha_d / alpha_sum
                    attn_drop_heads.append(attn_mask)
                else:
                    alpha_d = alpha
                    attn_drop_heads.append(np.ones_like(alpha))

                # Aggregate: h_i' = SUM_j alpha_ij * Wh_j
                h_head = alpha_d @ Wh
                head_out_list.append(h_head)

            cache['Wh'].append(Wh_heads)
            cache['e_raw'].append(e_raw_heads)
            cache['e_lrelu'].append(e_lrelu_heads)
            cache['alpha'].append(alpha_heads)
            cache['head_out'].append(head_out_list)
            cache['attn_dropout_masks'].append(attn_drop_heads)

            # Combine heads
            if is_last:
                # Output layer: AVERAGE heads
                H_combined = np.mean(head_out_list, axis=0)
            else:
                # Hidden layer: CONCATENATE heads
                H_combined = np.concatenate(head_out_list, axis=1)

            cache['H_pre_act'].append(H_combined.copy())

            # Activation
            if is_last:
                # Softmax for classification
                e_out = np.exp(H_combined - np.max(H_combined, axis=1, keepdims=True))
                H = e_out / (np.sum(e_out, axis=1, keepdims=True) + 1e-10)
            else:
                # ReLU for hidden layers
                H = np.maximum(H_combined, 0)

        # Cache for get_attention_weights
        self._last_cache = cache

        return H, cache

    def backward(self, graph, y, mask, cache):
        """
        Full backpropagation through all GAT layers.

        Computes gradients through:
        1. Softmax + cross-entropy (output layer)
        2. Attention weights (alpha) via masked softmax
        3. LeakyReLU on attention logits
        4. Projection weights W and attention vectors a_l, a_r

        Updates all parameters using gradient descent.
        """
        n = graph.n_nodes
        A_self = cache['A_self']
        n_train = max(np.sum(mask), 1)

        # --- Output layer gradient: cross-entropy + softmax ---
        H_pre_act_last = cache['H_pre_act'][-1]
        # Recompute softmax for numerical consistency
        e_out = np.exp(H_pre_act_last - np.max(H_pre_act_last, axis=1, keepdims=True))
        probs = e_out / (np.sum(e_out, axis=1, keepdims=True) + 1e-10)

        # dL/dZ (softmax + cross-entropy combined)
        dZ = probs.copy()
        dZ[np.arange(n), y] -= 1
        mask_float = mask.astype(float)
        dZ = dZ * mask_float[:, None] / n_train

        # Backprop layer by layer (reverse order)
        for layer in range(self.n_layers - 1, -1, -1):
            is_last = (layer == self.n_layers - 1)
            n_heads_this = self.n_heads

            H_in = cache['H_input'][layer]
            drop_mask = cache['dropout_masks'][layer]
            if self.dropout > 0 and not is_last:
                H_dropped = H_in * drop_mask / (1 - self.dropout + 1e-10)
            else:
                H_dropped = H_in.copy()

            if not is_last:
                # Backprop through ReLU
                relu_mask = (cache['H_pre_act'][layer] > 0).astype(float)
                dZ = dZ * relu_mask

            # Split gradient into per-head gradients
            if is_last:
                # Average: dL/d(head_k) = dZ / n_heads
                dH_heads = [dZ / n_heads_this for _ in range(n_heads_this)]
            else:
                # Concatenate: split dZ along feature axis
                out_dim = self.n_hidden
                dH_heads = []
                for h in range(n_heads_this):
                    dH_heads.append(dZ[:, h * out_dim : (h + 1) * out_dim])

            # Accumulate gradient for input (backprop to previous layer)
            if layer > 0:
                dH_prev = np.zeros_like(H_in)

            for h in range(n_heads_this):
                W = self.W[layer][h]
                al = self.a_l[layer][h]
                ar = self.a_r[layer][h]
                Wh = cache['Wh'][layer][h]
                alpha = cache['alpha'][layer][h]
                attn_mask = cache['attn_dropout_masks'][layer][h]
                e_raw = cache['e_raw'][layer][h]

                # Reconstruct alpha after dropout
                alpha_d = alpha * attn_mask
                alpha_sum = np.sum(alpha_d, axis=1, keepdims=True) + 1e-10
                alpha_d = alpha_d / alpha_sum

                dH_head = dH_heads[h]  # (n, out_dim)

                # --- Gradient through h_head = alpha_d @ Wh ---
                d_alpha_d = dH_head @ Wh.T       # dL/d(alpha_d): (n x n)
                dWh_agg = alpha_d.T @ dH_head     # dL/d(Wh) from aggregation

                # --- Gradient through attention dropout + renormalization ---
                d_alpha_raw = d_alpha_d * attn_mask / alpha_sum
                row_dot = np.sum(d_alpha_raw * alpha_d, axis=1, keepdims=True)
                d_alpha_raw = d_alpha_raw - alpha_d * row_dot

                # --- Gradient through masked softmax (Jacobian) ---
                d_e_lrelu = alpha * d_alpha_raw
                row_sums = np.sum(d_e_lrelu, axis=1, keepdims=True)
                d_e_lrelu = d_e_lrelu - alpha * row_sums
                d_e_lrelu = np.where(A_self > 0, d_e_lrelu, 0.0)

                # --- Gradient through LeakyReLU ---
                d_e_raw = d_e_lrelu * self._leaky_relu_grad(e_raw)

                # --- Gradient through attention logits ---
                # e_ij = a_l^T Wh_i + a_r^T Wh_j
                d_e_row_sum = np.sum(d_e_raw, axis=1, keepdims=True)  # (n, 1)
                d_e_col_sum = np.sum(d_e_raw, axis=0, keepdims=True).T  # (n, 1)

                d_al = Wh.T @ d_e_row_sum  # (out_dim, 1)
                d_ar = Wh.T @ d_e_col_sum  # (out_dim, 1)

                # dWh from attention path
                dWh_attn = d_e_row_sum @ al.T + d_e_col_sum @ ar.T

                # Total dWh = aggregation path + attention path
                dWh = dWh_agg + dWh_attn

                # --- Gradient for W: Wh = H_dropped @ W ---
                dW = H_dropped.T @ dWh

                # --- Gradient for input (backprop to previous layer) ---
                if layer > 0:
                    dH_drop = dWh @ W.T
                    if self.dropout > 0 and not is_last:
                        dH_in = dH_drop * drop_mask / (1 - self.dropout + 1e-10)
                    else:
                        dH_in = dH_drop
                    dH_prev += dH_in

                # --- Update parameters ---
                self.W[layer][h] -= self.lr * dW
                self.a_l[layer][h] -= self.lr * d_al
                self.a_r[layer][h] -= self.lr * d_ar

            # Pass gradient to previous layer
            if layer > 0:
                dZ = dH_prev

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train GAT on a graph with transductive split.

        Parameters:
        -----------
        graph : Graph
            Input graph (uses graph.A and graph.X)
        labels : ndarray
            Node labels (all nodes, but only train_mask used for loss)
        train_mask : boolean array
            Which nodes are in training set
        n_epochs : int
            Number of training epochs
        verbose : bool
            Print progress every 50 epochs

        Returns: loss_history (list of floats)
        """
        loss_history = []

        for epoch in range(n_epochs):
            # Forward pass
            probs, cache = self.forward(graph, training=True)

            # Compute loss
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

            # Backward pass (updates parameters)
            self.backward(graph, labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                preds = np.argmax(probs, axis=1)
                train_acc = np.mean(preds[train_mask] == labels[train_mask])
                print(f"  Epoch {epoch+1:>4}: loss={loss:.4f}, train_acc={train_acc:.3f}")

        return loss_history

    def predict(self, graph):
        """Predict node labels."""
        probs, _ = self.forward(graph, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, graph):
        """Predict node class probabilities."""
        probs, _ = self.forward(graph, training=False)
        return probs

    def get_embeddings(self, graph):
        """Get penultimate layer node embeddings."""
        _, cache = self.forward(graph, training=False)
        # Penultimate = H_pre_act of the second-to-last layer with ReLU applied
        if self.n_layers >= 2:
            return np.maximum(cache['H_pre_act'][-2], 0)
        else:
            return cache['H_input'][0]

    def get_attention_weights(self, layer=0):
        """
        Get attention matrices per head for a given layer.

        Must call forward() or predict() first to populate the cache.

        Returns: list of (n x n) attention matrices, one per head
        """
        if not hasattr(self, '_last_cache'):
            return None
        if layer >= len(self._last_cache.get('alpha', [])):
            return None
        return self._last_cache['alpha'][layer]


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    Ablation experiments for GAT:
    1. Number of attention heads
    2. GAT vs GCN comparison
    3. Attention dropout rates
    4. Attention entropy analysis
    5. Number of layers (over-smoothing)
    """
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    np.random.seed(42)

    # Setup: community graph
    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=25,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=42
    )
    n_classes = 3
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    # -------- 1. Number of Attention Heads --------
    print("\n1. NUMBER OF ATTENTION HEADS")
    print("-" * 40)
    print("More heads = more stable, more perspectives")

    for n_heads in [1, 2, 4, 8]:
        accs = []
        for run in range(3):
            model = GAT(graph.X.shape[1], 8, n_classes, n_layers=2,
                        n_heads=n_heads, dropout=0.3, lr=0.01,
                        random_state=42 + run)
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))

        print(f"  n_heads={n_heads}  test_acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    print("-> Multi-head attention improves stability")
    print("-> 4 heads is a good default (balance of expressiveness and parameters)")

    # -------- 2. GAT vs GCN --------
    print("\n2. GAT vs GCN COMPARISON")
    print("-" * 40)

    datasets_for_comparison = {
        'community_3': (graph, labels, train_mask, test_mask, n_classes),
    }
    # Add karate club
    g_kc, l_kc = karate_club()
    tm_kc, _, tsm_kc = create_transductive_split(
        g_kc.n_nodes, l_kc, train_ratio=0.15, val_ratio=0.1
    )
    datasets_for_comparison['karate_club'] = (g_kc, l_kc, tm_kc, tsm_kc, 2)

    for dname, (g, l, tm, tsm, nc) in datasets_for_comparison.items():
        # GAT
        gat_accs = []
        for run in range(3):
            model = GAT(g.X.shape[1], 8, nc, n_layers=2, n_heads=4,
                        dropout=0.3, lr=0.01, random_state=42 + run)
            model.fit(g, l, tm, n_epochs=200, verbose=False)
            preds = model.predict(g)
            gat_accs.append(np.mean(preds[tsm] == l[tsm]))

        # GCN
        gcn_accs = []
        for run in range(3):
            gcn = GCN(g.X.shape[1], 16, nc, n_layers=2, dropout=0.3,
                      lr=0.01, random_state=42 + run)
            gcn.fit(g, l, tm, n_epochs=200, verbose=False)
            preds = gcn.predict(g)
            gcn_accs.append(np.mean(preds[tsm] == l[tsm]))

        print(f"  {dname}:")
        print(f"    GAT: {np.mean(gat_accs):.3f} +/- {np.std(gat_accs):.3f}")
        print(f"    GCN: {np.mean(gcn_accs):.3f} +/- {np.std(gcn_accs):.3f}")

    print("-> GAT learns adaptive weights; GCN uses fixed degree-based weights")
    print("-> Advantage depends on whether attention helps for the task")

    # -------- 3. Attention Dropout --------
    print("\n3. ATTENTION DROPOUT")
    print("-" * 40)

    for dropout in [0.0, 0.3, 0.6]:
        accs = []
        for run in range(3):
            model = GAT(graph.X.shape[1], 8, n_classes, n_layers=2,
                        n_heads=4, dropout=dropout, lr=0.01,
                        random_state=42 + run)
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))

        print(f"  dropout={dropout:.1f}  test_acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    print("-> Moderate dropout (0.3-0.5) helps regularization")
    print("-> Too much dropout destabilizes attention learning")

    # -------- 4. Attention Entropy Analysis --------
    print("\n4. ATTENTION ENTROPY ANALYSIS")
    print("-" * 40)
    print("High entropy = uniform attention (like GCN)")
    print("Low entropy = concentrated attention (GAT's strength)")

    model = GAT(graph.X.shape[1], 8, n_classes, n_layers=2,
                n_heads=4, dropout=0.0, lr=0.01, random_state=42)
    model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
    _ = model.predict(graph)  # populate cache

    attn_weights = model.get_attention_weights(layer=0)
    if attn_weights is not None:
        for h_idx, attn in enumerate(attn_weights[:4]):
            entropies = []
            for i in range(graph.n_nodes):
                neighbors_mask = (graph.A + np.eye(graph.n_nodes))[i] > 0
                attn_row = attn[i, neighbors_mask]
                attn_row = attn_row[attn_row > 1e-10]
                if len(attn_row) > 0:
                    entropy = -np.sum(attn_row * np.log(attn_row + 1e-10))
                    entropies.append(entropy)
            deg = graph.degrees()
            max_entropy = np.mean(np.log(deg + 1 + 1e-10))
            print(f"  Head {h_idx}: avg_entropy={np.mean(entropies):.3f}"
                  f"  max_possible~={max_entropy:.3f}"
                  f"  ratio={np.mean(entropies)/max_entropy:.2f}")

    print("-> Ratio < 1 means attention is NOT uniform")
    print("-> Different heads may have different entropy patterns")

    # -------- 5. Number of Layers (Over-smoothing) --------
    print("\n5. NUMBER OF LAYERS (Over-Smoothing)")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4, 6]:
        accs = []
        for run in range(3):
            model = GAT(graph.X.shape[1], 8, n_classes, n_layers=n_layers,
                        n_heads=4, dropout=0.3, lr=0.01,
                        random_state=42 + run)
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))

        # Measure embedding similarity
        emb = model.get_embeddings(graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms
        cos_sim = emb_norm @ emb_norm.T
        avg_sim = (np.sum(cos_sim) - graph.n_nodes) / (graph.n_nodes * (graph.n_nodes - 1))

        print(f"  layers={n_layers}  test_acc={np.mean(accs):.3f}"
              f"  avg_cos_sim={avg_sim:.3f}")

    print("-> 2-3 layers optimal (same as GCN)")
    print("-> Attention does NOT prevent over-smoothing")
    print("-> Deep GAT still has all nodes converging")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    """Benchmark GAT across standard graph datasets."""
    print("\n" + "=" * 60)
    print("BENCHMARK: GAT on Graph Datasets")
    print("=" * 60)

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
        g, l = dataset_fn()
        train_mask, val_mask, test_mask = create_transductive_split(
            g.n_nodes, l, train_ratio=0.15, val_ratio=0.1
        )

        model = GAT(
            g.X.shape[1], 8, n_classes, n_layers=2,
            n_heads=4, dropout=0.5, lr=0.01, random_state=42
        )
        model.fit(g, l, train_mask, n_epochs=200, verbose=False)

        preds = model.predict(g)
        train_acc = np.mean(preds[train_mask] == l[train_mask])
        test_acc = np.mean(preds[test_mask] == l[test_mask])

        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name:<15} {train_acc:<12.3f} {test_acc:<12.3f} {g.n_nodes:<8}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_gat():
    """
    Main GAT visualization: 2x3 grid.

    Panel 1: Ground truth graph
    Panel 2: GAT predictions with attention-weighted edges
    Panel 3: Attention heatmap (subset of nodes)
    Panel 4: Multi-head attention comparison
    Panel 5: Training curve
    Panel 6: GAT vs GCN bar chart

    Returns: fig
    Saves: 38_gat.png
    """
    print("\nGenerating: GAT visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    np.random.seed(42)

    # Setup
    graph, labels = karate_club()
    n_classes = 2
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )
    pos = spring_layout(graph, seed=42)

    # Train GAT
    model = GAT(graph.X.shape[1], 8, n_classes, n_layers=2,
                n_heads=4, dropout=0.3, lr=0.01, random_state=42)
    losses = model.fit(graph, labels, train_mask, n_epochs=300, verbose=False)

    preds = model.predict(graph)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])

    # Populate attention cache
    _ = model.forward(graph, training=False)
    attn_weights = model.get_attention_weights(layer=0)

    # ============ Panel 1: Ground Truth ============
    ax = axes[0, 0]
    draw_graph(graph, labels, pos, ax,
               title='Ground Truth\n(Karate Club)', cmap='coolwarm')

    # ============ Panel 2: GAT Predictions + Attention Edges ============
    ax = axes[0, 1]

    # Draw edges with attention-based width
    if attn_weights is not None:
        avg_attn = np.mean(attn_weights, axis=0)  # average over heads
        for i in range(graph.n_nodes):
            for j in range(i + 1, graph.n_nodes):
                if graph.A[i, j] > 0:
                    w = (avg_attn[i, j] + avg_attn[j, i]) / 2
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            'gray', alpha=min(0.9, w * 8), linewidth=max(0.3, w * 6))
    else:
        for i in range(graph.n_nodes):
            for j in range(i + 1, graph.n_nodes):
                if graph.A[i, j] > 0:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            'gray', alpha=0.2, linewidth=0.5)

    # Color by prediction, mark errors
    for i in range(graph.n_nodes):
        c = 'C0' if preds[i] == 0 else 'C3'
        marker = 'o' if preds[i] == labels[i] else 'X'
        ms = 60 if preds[i] == labels[i] else 100
        ax.scatter(pos[i, 0], pos[i, 1], c=c, s=ms, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_title(f'GAT Predictions\ntest_acc={test_acc:.3f}\n'
                 f'Edge width = attention weight')
    ax.set_aspect('equal')
    ax.axis('off')

    # ============ Panel 3: Attention Heatmap ============
    ax = axes[0, 2]

    if attn_weights is not None:
        avg_attn = np.mean(attn_weights, axis=0)
        subset = list(range(min(15, graph.n_nodes)))
        attn_sub = avg_attn[np.ix_(subset, subset)]

        im = ax.imshow(attn_sub, cmap='viridis', aspect='auto')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        ax.set_title('Attention Weights\n(first 15 nodes, avg over heads)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No attention\nweights available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Attention Heatmap')

    # ============ Panel 4: Multi-Head Comparison ============
    ax = axes[1, 0]

    if attn_weights is not None and len(attn_weights) >= 2:
        # Pick a high-degree node
        degrees = graph.degrees()
        node = int(np.argmax(degrees))
        nbrs = list(graph.neighbors(node)) + [node]
        nbrs = sorted(nbrs)[:12]  # limit to 12 for readability
        n_nbrs = len(nbrs)

        x_pos = np.arange(n_nbrs)
        width = 0.8 / min(4, len(attn_weights))
        colors = plt.cm.Set2(np.linspace(0, 0.8, min(4, len(attn_weights))))

        for h_idx in range(min(4, len(attn_weights))):
            weights = attn_weights[h_idx][node, nbrs]
            ax.bar(x_pos + h_idx * width, weights, width,
                   label=f'Head {h_idx+1}', color=colors[h_idx])

        ax.set_xticks(x_pos + width * (min(4, len(attn_weights)) - 1) / 2)
        ax.set_xticklabels([f'{n}' for n in nbrs], fontsize=7, rotation=45)
        ax.set_xlabel('Neighbor Node')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Multi-Head Attention for Node {node}\n'
                     f'(degree={int(degrees[node])})')
        ax.legend(fontsize=7, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Multi-head comparison\nnot available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Multi-Head Attention')

    # ============ Panel 5: Training Curve ============
    ax = axes[1, 1]
    ax.plot(losses, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('GAT Training Curve')
    ax.grid(True, alpha=0.3)

    # ============ Panel 6: GAT vs GCN Bar Chart ============
    ax = axes[1, 2]

    n_runs = 5
    gat_accs = []
    gcn_accs = []

    for run in range(n_runs):
        # GAT
        g, l = karate_club()
        tm, _, tsm = create_transductive_split(g.n_nodes, l, 0.15, 0.1)
        m = GAT(g.X.shape[1], 8, 2, n_layers=2, n_heads=4,
                dropout=0.3, lr=0.01, random_state=42 + run)
        m.fit(g, l, tm, n_epochs=200, verbose=False)
        gat_accs.append(np.mean(m.predict(g)[tsm] == l[tsm]))

        # GCN
        g2, l2 = karate_club()
        gcn = GCN(g2.X.shape[1], 16, 2, n_layers=2, dropout=0.3,
                  lr=0.01, random_state=42 + run)
        gcn.fit(g2, l2, tm, n_epochs=200, verbose=False)
        gcn_accs.append(np.mean(gcn.predict(g2)[tsm] == l2[tsm]))

    x = np.arange(2)
    means = [np.mean(gcn_accs), np.mean(gat_accs)]
    stds = [np.std(gcn_accs), np.std(gat_accs)]

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=['steelblue', 'coral'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['GCN', 'GAT'])
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'GAT vs GCN (Karate Club)\n{n_runs} runs each')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('GAT -- Graph Attention Network\n'
                 'Learn to attend to important neighbors',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = '/Users/sid47/ML Algorithms/38_gat.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def visualize_attention_patterns():
    """
    Attention pattern analysis: 1x3 grid.

    Panel 1: Attention entropy per node (colored by degree)
    Panel 2: Attention distribution for hub vs leaf node
    Panel 3: Attention vs graph structure (degree correlation)

    Returns: fig
    Saves: 38_gat_attention.png
    """
    print("\nGenerating: Attention pattern visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42)

    # Use community graph for richer structure
    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=20,
        p_in=0.3, p_out=0.02, feature_dim=16, random_state=42
    )
    n_classes = 3
    train_mask, _, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    model = GAT(graph.X.shape[1], 8, n_classes, n_layers=2,
                n_heads=4, dropout=0.0, lr=0.01, random_state=42)
    model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
    _ = model.forward(graph, training=False)
    attn_weights = model.get_attention_weights(layer=0)

    degrees = graph.degrees()
    A_self = graph.A + np.eye(graph.n_nodes)

    # ============ Panel 1: Attention Entropy per Node ============
    ax = axes[0]

    if attn_weights is not None:
        avg_attn = np.mean(attn_weights, axis=0)

        entropies = []
        for i in range(graph.n_nodes):
            nbr_mask = A_self[i] > 0
            row = avg_attn[i, nbr_mask]
            row = row[row > 1e-10]
            if len(row) > 0:
                ent = -np.sum(row * np.log(row + 1e-10))
            else:
                ent = 0.0
            entropies.append(ent)

        entropies = np.array(entropies)

        sc = ax.scatter(degrees, entropies, c=labels, cmap='Set1',
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Attention Entropy')
        ax.set_title('Attention Entropy per Node\n'
                     'Higher = more uniform attention')
        ax.grid(True, alpha=0.3)

        # Add max entropy line (log(degree+1) for self-loop)
        deg_range = np.linspace(1, max(degrees) + 1, 50)
        ax.plot(deg_range - 1, np.log(deg_range), 'k--', alpha=0.5,
                label='Max entropy = log(deg+1)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No attention data', ha='center', va='center',
                transform=ax.transAxes)

    # ============ Panel 2: Hub vs Leaf Attention ============
    ax = axes[1]

    if attn_weights is not None:
        # Find hub (highest degree) and leaf (lowest degree) nodes
        hub_node = int(np.argmax(degrees))
        leaf_candidates = np.where(degrees > 0)[0]
        leaf_node = int(leaf_candidates[np.argmin(degrees[leaf_candidates])])

        avg_attn = np.mean(attn_weights, axis=0)

        for node, name, color in [(hub_node, 'Hub', 'coral'),
                                   (leaf_node, 'Leaf', 'steelblue')]:
            nbr_mask = A_self[node] > 0
            weights = avg_attn[node, nbr_mask]
            n_nbrs = len(weights)
            uniform = np.ones(n_nbrs) / n_nbrs

            x_pos = np.arange(n_nbrs)
            width = 0.35

            if node == hub_node:
                bars = ax.bar(x_pos[:min(15, n_nbrs)] - width/2,
                              weights[:min(15, n_nbrs)], width,
                              label=f'{name} (deg={int(degrees[node])})',
                              color=color, alpha=0.7)
                ax.bar(x_pos[:min(15, n_nbrs)] - width/2,
                       uniform[:min(15, n_nbrs)], width,
                       fill=False, edgecolor=color, linestyle='--', linewidth=1)
            else:
                bars = ax.bar(x_pos[:min(15, n_nbrs)] + width/2,
                              weights[:min(15, n_nbrs)], width,
                              label=f'{name} (deg={int(degrees[node])})',
                              color=color, alpha=0.7)
                ax.bar(x_pos[:min(15, n_nbrs)] + width/2,
                       uniform[:min(15, n_nbrs)], width,
                       fill=False, edgecolor=color, linestyle='--', linewidth=1)

        ax.set_xlabel('Neighbor Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Hub vs Leaf Node Attention\n'
                     'Dashed = uniform (what GCN does)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No attention data', ha='center', va='center',
                transform=ax.transAxes)

    # ============ Panel 3: Attention vs Structure ============
    ax = axes[2]

    if attn_weights is not None:
        avg_attn = np.mean(attn_weights, axis=0)

        # For each edge, compare attention weight to GCN-style weight
        gcn_weights = []
        gat_weights = []

        for i in range(graph.n_nodes):
            for j in range(graph.n_nodes):
                if A_self[i, j] > 0 and i != j:
                    # GCN weight: 1/sqrt(d_i * d_j) (approximate)
                    gcn_w = 1.0 / np.sqrt((degrees[i] + 1) * (degrees[j] + 1))
                    gcn_weights.append(gcn_w)
                    gat_weights.append(avg_attn[i, j])

        gcn_weights = np.array(gcn_weights)
        gat_weights = np.array(gat_weights)

        ax.scatter(gcn_weights, gat_weights, s=10, alpha=0.3, c='steelblue')
        ax.set_xlabel('GCN Weight (degree-based)')
        ax.set_ylabel('GAT Weight (learned)')
        ax.set_title('GAT vs GCN Weights per Edge\n'
                     'Off-diagonal = GAT deviates from GCN')

        # Add diagonal line
        lims = [0, max(np.max(gcn_weights), np.max(gat_weights)) * 1.1]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x (perfect agreement)')
        ax.set_xlim(0, lims[1])
        ax.set_ylim(0, lims[1])

        # Correlation
        corr = np.corrcoef(gcn_weights, gat_weights)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=10, va='top')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No attention data', ha='center', va='center',
                transform=ax.transAxes)

    plt.suptitle('GAT Attention Patterns -- What Does Attention Learn?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = '/Users/sid47/ML Algorithms/38_gat_attention.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GAT -- Graph Attention Network")
    print("Paradigm: ATTENTION ON GRAPHS")
    print("=" * 60)

    print("""
WHAT THIS MODEL IS:

    Each GAT layer computes (per head k):

        Wh = X @ W^k                              (project features)
        e_ij = LeakyReLU(a_l^k . Wh_i + a_r^k . Wh_j)   (attention logits)
        alpha_ij = softmax_j(e_ij) over neighbors  (attention weights)
        h_i' = SUM_j alpha_ij * Wh_j               (weighted aggregation)

    Hidden layers: h' = CONCAT(head_1, ..., head_K) then ReLU
    Output layer:  h' = MEAN(head_1, ..., head_K) then softmax

KEY DIFFERENCE FROM GCN:
    GCN: weights are FIXED by node degree (1/sqrt(d_i * d_j))
    GAT: weights are LEARNED from node FEATURES (attention)

    This means:
    - Different neighbors contribute differently
    - Weights adapt to the specific input
    - Attention is interpretable (we can visualize it)

KEY HYPERPARAMETERS:
    - n_heads: 4 (more heads = more stable, more expressive)
    - n_hidden: 8 per head (total = n_hidden * n_heads)
    - n_layers: 2-3 (still limited by over-smoothing)
    - dropout: 0.3-0.6 (applied to features AND attention)
    """)

    # Run ablation experiments
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Generate visualizations
    print("\nGenerating visualizations...")
    fig1 = visualize_gat()
    plt.close(fig1)

    fig2 = visualize_attention_patterns()
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("SUMMARY: What GAT Reveals")
    print("=" * 60)
    print("""
1. ATTENTION = LEARNED NEIGHBOR WEIGHTING
   alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
   Not all neighbors are treated equally.

2. MULTI-HEAD ATTENTION
   Multiple attention heads capture different relationships.
   Concatenate for hidden layers, average for output.

3. GAT vs GCN
   GCN: fixed weights from graph structure
   GAT: adaptive weights from node features
   GAT is more expressive but has more parameters.

4. ATTENTION IS INTERPRETABLE
   We can visualize which neighbors each node attends to.
   High entropy = uniform (GCN-like), low entropy = selective.

5. OVER-SMOOTHING PERSISTS
   Even with attention, deep GAT still over-smooths.
   2-3 layers remains the practical limit.

6. WHEN TO USE GAT
   - Heterogeneous neighbor importance
   - When interpretability matters
   - Enough training data for attention to learn

CONNECTION TO OTHER FILES:
    36_gcn.py:             Fixed degree-based weights (GAT generalizes this)
    37_graphsage.py:       Sampling + learned aggregation (orthogonal to attention)
    39_gin.py:             Maximum expressiveness via SUM aggregation
    15_transformer.py:     Full self-attention (GAT = sparse, local attention)
    44_graph_transformer.py: Full attention on graphs (GAT only neighbors)

NEXT: 39_gin.py -- How expressive can GNNs be? (WL test)
    """)
