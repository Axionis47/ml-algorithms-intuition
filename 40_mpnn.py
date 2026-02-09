"""
MPNN — Message Passing Neural Network

Paradigm: UNIFIED FRAMEWORK FOR GRAPH NEURAL NETWORKS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A GENERAL framework that encompasses GCN, GraphSAGE, GAT, GIN,
and virtually every other GNN.

Every GNN does three things per layer:

    1. MESSAGE:   m_ij = M(h_i, h_j, e_ij)
    2. AGGREGATE: m_i  = AGG({m_ij : j in N(i)})
    3. UPDATE:    h_i' = U(h_i, m_i)

That's it. ALL GNNs are message passing.

===============================================================
HOW GCN, GAT, GIN ARE ALL MPNN
===============================================================

| Model     | MESSAGE M(h_i,h_j)        | AGGREGATE | UPDATE U(h_i,m_i) |
|-----------|---------------------------|-----------|-------------------|
| GCN       | h_j / sqrt(d_i*d_j)      | SUM       | sigma(W * m_i)    |
| GraphSAGE | h_j                       | MEAN/POOL | sigma(W*[h_i||m]) |
| GAT       | alpha_ij * W*h_j          | SUM       | sigma(m_i)        |
| GIN       | h_j                       | SUM       | MLP((1+e)h_i+m_i) |

The difference is just the CHOICE of M, AGG, and U!

===============================================================
EDGE FEATURES
===============================================================

Many graphs have edge attributes (bond type, distance, etc.)

MPNN naturally incorporates them:
    m_ij = M(h_i, h_j, e_ij)

Examples:
- Bond type in molecules -> different message per bond
- Distance in point clouds -> distance-weighted messages
- Relationship type in knowledge graphs -> per-relation transform

===============================================================
GATED UPDATES
===============================================================

Instead of simple linear update, use GRU-style gating:

    r = sigma(W_r [h_i || m_i])          (reset gate)
    z = sigma(W_z [h_i || m_i])          (update gate)
    h_tilde = tanh(W_h [r*h_i || m_i])   (candidate)
    h_i' = (1-z)*h_i + z*h_tilde         (gated update)

WHY? Prevents information being washed out over many layers.
Like LSTM for sequences, GRU helps preserve important features.

===============================================================
WHY UNIFICATION MATTERS
===============================================================

Once you see GNNs as message passing, you can:

1. DESIGN new GNNs by choosing M, AGG, U
2. ANALYZE expressiveness of different choices
3. COMBINE ideas (e.g., attention messages + GRU update)
4. UNDERSTAND limitations (1-WL bound applies to ALL MPNN)

===============================================================
INDUCTIVE BIAS
===============================================================

1. LOCAL: Each layer sees 1-hop neighborhood
2. PERMUTATION EQUIVARIANT: Output invariant to node ordering
3. STRUCTURE-AWARE: Uses graph topology for aggregation
4. EXPRESSIVENESS BOUNDED: By 1-WL isomorphism test

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
compute_normalized_adjacency = gcn_module.compute_normalized_adjacency


# ============================================================
# MPNN IMPLEMENTATION (Vectorized with analytical backprop)
# ============================================================

class MPNN:
    """
    Message Passing Neural Network — Unified GNN Framework.

    Configurable message, aggregation, and update functions
    that can reproduce GCN, GraphSAGE, GIN as special cases.

    All operations are VECTORIZED using adjacency matrix multiplication
    for efficiency. Training uses analytical backpropagation.

    Paradigm: MESSAGE -> AGGREGATE -> UPDATE
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=2,
                 message_type='simple', aggregate_type='sum',
                 update_type='concat', dropout=0.5, lr=0.01,
                 random_state=None):
        """
        Parameters:
        -----------
        n_features : int — Input feature dimension
        n_hidden : int — Hidden layer dimension
        n_classes : int — Number of output classes
        n_layers : int — Number of message passing layers
        message_type : str
            'simple' — M(h_j) = h_j (like GIN)
            'linear' — M(h_j) = W_m h_j (like GCN)
        aggregate_type : str
            'sum', 'mean', 'max'
        update_type : str
            'concat' — U(h,m) = sigma(W[h||m]) (like GraphSAGE)
            'add'    — U(h,m) = sigma(W(h+m)) (like GIN)
            'gru'    — U(h,m) = GRU(h, m) (gated)
        dropout : float
        lr : float
        random_state : int or None
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.message_type = message_type
        self.aggregate_type = aggregate_type
        self.update_type = update_type
        self.dropout = dropout
        self.lr = lr
        self.random_state = random_state

        self._init_weights()

    def _init_weights(self):
        """Initialize all learnable parameters."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.W_msg = []    # message transform weights per layer
        self.W_upd = []    # update weights per layer
        self.b_upd = []    # update biases per layer

        # GRU params (only if gru update)
        self.W_r = []
        self.b_r = []
        self.W_z = []
        self.b_z = []
        self.W_h = []
        self.b_h = []

        for l in range(self.n_layers):
            d_in = self.n_features if l == 0 else self.n_hidden

            # Message weights (for linear message type)
            if self.message_type == 'linear':
                std = np.sqrt(2.0 / (d_in + self.n_hidden))
                self.W_msg.append(np.random.randn(d_in, self.n_hidden) * std)
            else:
                self.W_msg.append(None)

            # Message output dim
            msg_dim = self.n_hidden if self.message_type == 'linear' else d_in

            # Update weights
            if self.update_type == 'concat':
                upd_in = d_in + msg_dim
                std = np.sqrt(2.0 / (upd_in + self.n_hidden))
                self.W_upd.append(np.random.randn(upd_in, self.n_hidden) * std)
                self.b_upd.append(np.zeros(self.n_hidden))
            elif self.update_type == 'add':
                # Needs d_in == msg_dim for add; if not, project message
                upd_in = max(d_in, msg_dim)
                std = np.sqrt(2.0 / (upd_in + self.n_hidden))
                self.W_upd.append(np.random.randn(upd_in, self.n_hidden) * std)
                self.b_upd.append(np.zeros(self.n_hidden))
            elif self.update_type == 'gru':
                total_in = d_in + msg_dim
                self.W_r.append(np.random.randn(total_in, self.n_hidden) * 0.1)
                self.b_r.append(np.zeros(self.n_hidden))
                self.W_z.append(np.random.randn(total_in, self.n_hidden) * 0.1)
                self.b_z.append(np.zeros(self.n_hidden))
                self.W_h.append(np.random.randn(total_in, self.n_hidden) * 0.1)
                self.b_h.append(np.zeros(self.n_hidden))
                self.W_upd.append(None)
                self.b_upd.append(None)

        # Output layer
        self.W_out = np.random.randn(self.n_hidden, self.n_classes) * np.sqrt(2.0 / self.n_hidden)
        self.b_out = np.zeros(self.n_classes)

    def _get_agg_matrix(self, A):
        """Get the aggregation matrix based on aggregate_type."""
        if self.aggregate_type == 'sum':
            return A.copy()
        elif self.aggregate_type == 'mean':
            deg = np.sum(A, axis=1, keepdims=True)
            deg = np.maximum(deg, 1)
            return A / deg
        else:
            return A.copy()  # max handled separately

    def forward(self, graph, training=True):
        """
        Vectorized forward pass: MESSAGE -> AGGREGATE -> UPDATE.

        Uses adjacency matrix multiplication for message passing.

        Returns: (probs, cache)
        """
        n = graph.n_nodes
        A = graph.A.astype(float)
        H = graph.X.copy()

        cache = {
            'H': [H.copy()],
            'messages': [],
            'aggregated': [],
            'pre_act': [],
            'dropout_masks': [],
            'gru_cache': [],
        }

        for l in range(self.n_layers):
            d_in = H.shape[1]

            # ---- STEP 1: MESSAGE ----
            if self.message_type == 'linear':
                messages = H @ self.W_msg[l]  # (n, n_hidden)
            else:
                messages = H  # (n, d_in)

            cache['messages'].append(messages.copy())

            # ---- STEP 2: AGGREGATE ----
            if self.aggregate_type == 'max':
                # Max aggregation: for each node, max over neighbor messages
                m_agg = np.zeros((n, messages.shape[1]))
                for i in range(n):
                    neighbors = np.where(A[i] > 0)[0]
                    if len(neighbors) > 0:
                        m_agg[i] = np.max(messages[neighbors], axis=0)
            else:
                # Sum or mean via matrix multiply
                agg_matrix = self._get_agg_matrix(A)
                m_agg = agg_matrix @ messages

            cache['aggregated'].append(m_agg.copy())

            # ---- STEP 3: UPDATE ----
            if self.update_type == 'concat':
                combined = np.concatenate([H, m_agg], axis=1)
                Z = combined @ self.W_upd[l] + self.b_upd[l]
                cache['pre_act'].append(Z.copy())
                H_new = np.maximum(Z, 0)  # ReLU

            elif self.update_type == 'add':
                # Pad if dims don't match
                if H.shape[1] != m_agg.shape[1]:
                    if H.shape[1] < m_agg.shape[1]:
                        H_pad = np.zeros((n, m_agg.shape[1]))
                        H_pad[:, :H.shape[1]] = H
                    else:
                        H_pad = H
                        m_pad = np.zeros((n, H.shape[1]))
                        m_pad[:, :m_agg.shape[1]] = m_agg
                        m_agg = m_pad
                else:
                    H_pad = H
                combined = H_pad + m_agg
                Z = combined @ self.W_upd[l] + self.b_upd[l]
                cache['pre_act'].append(Z.copy())
                H_new = np.maximum(Z, 0)

            elif self.update_type == 'gru':
                combined = np.concatenate([H, m_agg], axis=1)
                total_in = self.W_r[l].shape[0]
                if combined.shape[1] < total_in:
                    pad = np.zeros((n, total_in - combined.shape[1]))
                    combined = np.concatenate([combined, pad], axis=1)
                elif combined.shape[1] > total_in:
                    combined = combined[:, :total_in]

                r = 1.0 / (1 + np.exp(-(combined @ self.W_r[l] + self.b_r[l])))
                z = 1.0 / (1 + np.exp(-(combined @ self.W_z[l] + self.b_z[l])))

                h_prev = H[:, :self.n_hidden] if H.shape[1] >= self.n_hidden else np.pad(H, ((0,0),(0,self.n_hidden - H.shape[1])))
                h_reset = r * h_prev
                combined_h = np.concatenate([h_reset, m_agg], axis=1)
                if combined_h.shape[1] < total_in:
                    pad = np.zeros((n, total_in - combined_h.shape[1]))
                    combined_h = np.concatenate([combined_h, pad], axis=1)
                elif combined_h.shape[1] > total_in:
                    combined_h = combined_h[:, :total_in]

                h_candidate = np.tanh(combined_h @ self.W_h[l] + self.b_h[l])
                H_new = (1 - z) * h_prev + z * h_candidate

                cache['gru_cache'].append({
                    'r': r, 'z': z, 'h_prev': h_prev,
                    'h_candidate': h_candidate,
                    'combined': combined, 'combined_h': combined_h
                })
                cache['pre_act'].append(None)

            # Dropout
            if training and self.dropout > 0:
                mask = (np.random.rand(*H_new.shape) > self.dropout).astype(float)
                H_new = H_new * mask / (1 - self.dropout + 1e-10)
                cache['dropout_masks'].append(mask)
            else:
                cache['dropout_masks'].append(np.ones_like(H_new))

            H = H_new
            cache['H'].append(H.copy())

        # Output layer
        logits = H @ self.W_out + self.b_out
        probs = softmax(logits)
        cache['logits'] = logits
        cache['probs'] = probs

        return probs, cache

    def backward(self, graph, labels, train_mask, cache):
        """Analytical backpropagation through all MPNN layers."""
        n = graph.n_nodes
        A = graph.A.astype(float)
        probs = cache['probs']

        # Gradient of cross-entropy + softmax
        dLogits = probs.copy()
        dLogits[np.arange(n), labels] -= 1
        mask_float = train_mask.astype(float)
        n_train = max(np.sum(train_mask), 1)
        dLogits = dLogits * mask_float[:, None] / n_train

        # Output layer gradients
        H_last = cache['H'][-1]
        gW_out = H_last.T @ dLogits
        gb_out = np.sum(dLogits, axis=0)

        # Gradient w.r.t. last hidden
        dH = dLogits @ self.W_out.T

        # Update output layer
        self.W_out -= self.lr * gW_out
        self.b_out -= self.lr * gb_out

        # Backprop through layers (reverse order)
        for l in range(self.n_layers - 1, -1, -1):
            # Dropout
            dH = dH * cache['dropout_masks'][l] / (1 - self.dropout + 1e-10)

            H_prev = cache['H'][l]
            messages = cache['messages'][l]
            m_agg = cache['aggregated'][l]

            if self.update_type == 'concat':
                # Backprop through ReLU
                Z = cache['pre_act'][l]
                dZ = dH * (Z > 0).astype(float)

                # Gradient w.r.t. W_upd, b_upd
                combined = np.concatenate([H_prev, m_agg], axis=1)
                gW = combined.T @ dZ
                gb = np.sum(dZ, axis=0)

                # Gradient w.r.t. combined
                dCombined = dZ @ self.W_upd[l].T
                d_in = H_prev.shape[1]
                dH_prev = dCombined[:, :d_in]
                dM_agg = dCombined[:, d_in:]

                self.W_upd[l] -= self.lr * gW
                self.b_upd[l] -= self.lr * gb

            elif self.update_type == 'add':
                Z = cache['pre_act'][l]
                dZ = dH * (Z > 0).astype(float)

                d_in = H_prev.shape[1]
                msg_dim = m_agg.shape[1]
                upd_dim = max(d_in, msg_dim)

                if d_in != msg_dim:
                    if d_in < msg_dim:
                        H_pad = np.zeros((n, msg_dim))
                        H_pad[:, :d_in] = H_prev
                    else:
                        H_pad = H_prev
                        m_pad = np.zeros((n, d_in))
                        m_pad[:, :msg_dim] = m_agg
                        m_agg = m_pad
                else:
                    H_pad = H_prev

                combined = H_pad + m_agg
                gW = combined.T @ dZ
                gb = np.sum(dZ, axis=0)

                dCombined = dZ @ self.W_upd[l].T
                dH_prev = dCombined[:, :d_in]
                dM_agg = dCombined[:, :msg_dim]

                self.W_upd[l] -= self.lr * gW
                self.b_upd[l] -= self.lr * gb

            elif self.update_type == 'gru':
                gc = cache['gru_cache'][l]
                r, z = gc['r'], gc['z']
                h_prev, h_candidate = gc['h_prev'], gc['h_candidate']
                combined_input = gc['combined']
                combined_h_input = gc['combined_h']
                total_in = self.W_r[l].shape[0]

                d_in = H_prev.shape[1]
                msg_dim = m_agg.shape[1]

                # dH_new = (1-z)*h_prev + z*h_candidate
                dH_prev_gru = dH * (1 - z)  # (n, n_hidden)
                dH_cand = dH * z
                dZ_gate = dH * (h_candidate - h_prev)

                # h_candidate = tanh(combined_h @ W_h + b_h)
                d_tanh = dH_cand * (1 - h_candidate**2)
                gW_h = combined_h_input.T @ d_tanh
                gb_h = np.sum(d_tanh, axis=0)
                dCombined_h = d_tanh @ self.W_h[l].T  # (n, total_in)

                # z = sigmoid(combined @ W_z + b_z)
                dZ_pre = dZ_gate * z * (1 - z)
                gW_z = combined_input.T @ dZ_pre
                gb_z = np.sum(dZ_pre, axis=0)
                dCombined_from_z = dZ_pre @ self.W_z[l].T  # (n, total_in)

                # combined_h = [r*h_prev(n_hidden), m_agg(msg_dim), pad...]
                # Extract gradients for r*h_prev and m_agg from dCombined_h
                dR_h = dCombined_h[:, :self.n_hidden]  # gradient through h_reset
                # m_agg starts at n_hidden in combined_h
                m_start = self.n_hidden
                m_end = min(m_start + msg_dim, total_in)
                dM_from_h = np.zeros((n, msg_dim))
                if m_end > m_start:
                    dM_from_h[:, :m_end-m_start] = dCombined_h[:, m_start:m_end]

                # h_reset = r * h_prev
                dR = dR_h * h_prev  # gradient for r
                dH_prev_from_r = dR_h * r  # gradient for h_prev through reset

                # r = sigmoid(combined @ W_r + b_r)
                dR_pre = dR * r * (1 - r)
                gW_r = combined_input.T @ dR_pre
                gb_r = np.sum(dR_pre, axis=0)
                dCombined_from_r = dR_pre @ self.W_r[l].T  # (n, total_in)

                # combined = [H(d_in), m_agg(msg_dim), pad...]
                dCombined = dCombined_from_z + dCombined_from_r
                dH_prev = dCombined[:, :d_in]
                # Add gradient from h_prev (which was H[:, :n_hidden] padded)
                dH_prev[:, :min(d_in, self.n_hidden)] += dH_prev_gru[:, :min(d_in, self.n_hidden)]
                dH_prev[:, :min(d_in, self.n_hidden)] += dH_prev_from_r[:, :min(d_in, self.n_hidden)]

                m_comb_end = min(d_in + msg_dim, total_in)
                dM_agg = np.zeros((n, msg_dim))
                actual_msg = m_comb_end - d_in
                if actual_msg > 0:
                    dM_agg[:, :actual_msg] = dCombined[:, d_in:m_comb_end]
                dM_agg += dM_from_h

                self.W_r[l] -= self.lr * gW_r
                self.b_r[l] -= self.lr * gb_r
                self.W_z[l] -= self.lr * gW_z
                self.b_z[l] -= self.lr * gb_z
                self.W_h[l] -= self.lr * gW_h
                self.b_h[l] -= self.lr * gb_h

            # Backprop through aggregation
            if self.aggregate_type == 'sum':
                dMessages = A.T @ dM_agg
            elif self.aggregate_type == 'mean':
                deg = np.sum(A, axis=1, keepdims=True)
                deg = np.maximum(deg, 1)
                agg_matrix = A / deg
                dMessages = agg_matrix.T @ dM_agg
            elif self.aggregate_type == 'max':
                # Approximate: gradient flows to the max elements
                dMessages = np.zeros_like(messages)
                for i in range(n):
                    neighbors = np.where(A[i] > 0)[0]
                    if len(neighbors) > 0:
                        max_idx = np.argmax(messages[neighbors], axis=0)
                        for d in range(messages.shape[1]):
                            dMessages[neighbors[max_idx[d]], d] += dM_agg[i, d]

            # Backprop through message function
            if self.message_type == 'linear':
                gW_msg = H_prev.T @ dMessages
                dH_from_msg = dMessages @ self.W_msg[l].T
                self.W_msg[l] -= self.lr * gW_msg
            else:
                dH_from_msg = dMessages

            # Combine gradients flowing to H_prev
            dH = dH_prev + dH_from_msg

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=True):
        """Train MPNN with analytical backpropagation."""
        loss_history = []

        for epoch in range(n_epochs):
            probs, cache = self.forward(graph, training=True)
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

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
        """Predict class probabilities."""
        probs, _ = self.forward(graph, training=False)
        return probs

    def get_embeddings(self, graph):
        """Get node embeddings from last hidden layer."""
        _, cache = self.forward(graph, training=False)
        return cache['H'][-2]


def mpnn_as_gcn_config():
    """MPNN configuration that mimics GCN."""
    return {'message_type': 'linear', 'aggregate_type': 'sum', 'update_type': 'add'}


def mpnn_as_graphsage_config():
    """MPNN configuration that mimics GraphSAGE (mean)."""
    return {'message_type': 'simple', 'aggregate_type': 'mean', 'update_type': 'concat'}


def mpnn_as_gin_config():
    """MPNN configuration that mimics GIN."""
    return {'message_type': 'simple', 'aggregate_type': 'sum', 'update_type': 'add'}


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    graph, labels = create_community_graph(
        n_communities=2, nodes_per_community=20,
        p_in=0.3, p_out=0.03, feature_dim=8, random_state=42
    )
    n_classes = 2
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.2, val_ratio=0.1
    )

    # -------- Experiment 1: MPNN as different architectures --------
    print("\n1. MPNN CONFIGURED AS DIFFERENT ARCHITECTURES")
    print("-" * 40)
    print("Same framework, different message/aggregate/update choices")

    configs = {
        'GCN-style':      {'message_type': 'linear', 'aggregate_type': 'sum', 'update_type': 'add'},
        'SAGE-style':     {'message_type': 'simple', 'aggregate_type': 'mean', 'update_type': 'concat'},
        'GIN-style':      {'message_type': 'simple', 'aggregate_type': 'sum', 'update_type': 'add'},
        'Linear+concat':  {'message_type': 'linear', 'aggregate_type': 'sum', 'update_type': 'concat'},
    }

    for name, cfg in configs.items():
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=2,
                    lr=0.01, dropout=0.3, random_state=42, **cfg)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = mpnn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        print(f"  {name:<15}  msg={cfg['message_type']:<8}"
              f"  agg={cfg['aggregate_type']:<5}"
              f"  upd={cfg['update_type']:<7}"
              f"  test_acc={test_acc:.3f}")

    print("-> Different M/AGG/U choices = different GNN variants")
    print("-> All are instances of the same message passing framework")

    # -------- Experiment 2: Aggregation comparison --------
    print("\n2. AGGREGATION FUNCTION COMPARISON")
    print("-" * 40)

    for agg in ['sum', 'mean', 'max']:
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=2,
                    message_type='simple', aggregate_type=agg,
                    update_type='concat', lr=0.01, dropout=0.3, random_state=42)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = mpnn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        print(f"  {agg:<5}  test_acc={test_acc:.3f}")

    print("-> SUM: counts matter (injective for multisets)")
    print("-> MEAN: size-invariant (ignores multiplicity)")
    print("-> MAX: captures salient features (ignores count)")

    # -------- Experiment 3: Update function comparison --------
    print("\n3. UPDATE FUNCTION COMPARISON")
    print("-" * 40)

    for upd in ['concat', 'add', 'gru']:
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=2,
                    message_type='simple', aggregate_type='sum',
                    update_type=upd, lr=0.01, dropout=0.3, random_state=42)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = mpnn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        print(f"  {upd:<7}  test_acc={test_acc:.3f}")

    print("-> concat: preserves self and neighbor info separately")
    print("-> add: simpler, fewer params")
    print("-> gru: gated update, helps with deep networks")

    # -------- Experiment 4: Message function comparison --------
    print("\n4. MESSAGE FUNCTION COMPARISON")
    print("-" * 40)

    for msg in ['simple', 'linear']:
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=2,
                    message_type=msg, aggregate_type='sum',
                    update_type='concat', lr=0.01, dropout=0.3, random_state=42)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = mpnn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        print(f"  {msg:<8}  test_acc={test_acc:.3f}")

    print("-> simple: pass neighbor features directly")
    print("-> linear: transform before passing (more expressive)")

    # -------- Experiment 5: Depth comparison --------
    print("\n5. NUMBER OF LAYERS")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4]:
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=n_layers,
                    message_type='simple', aggregate_type='sum',
                    update_type='concat', lr=0.01, dropout=0.3, random_state=42)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = mpnn.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])
        emb = mpnn.get_embeddings(graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_n = emb / norms
        sim = emb_n @ emb_n.T
        avg_sim = (np.sum(sim) - graph.n_nodes) / (graph.n_nodes * (graph.n_nodes - 1))
        print(f"  layers={n_layers}  test_acc={test_acc:.3f}  avg_cos_sim={avg_sim:.3f}")

    print("-> 2 layers typically best")
    print("-> More layers -> over-smoothing (cosine similarity increases)")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    """Benchmark MPNN configurations across standard graph datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: MPNN on Graph Datasets")
    print("="*60)

    results = {}
    datasets = {
        'karate_club': (karate_club, 2),
        'community_2': (lambda: create_community_graph(2, 25, 0.3, 0.02, feature_dim=16), 2),
        'community_3': (lambda: create_community_graph(3, 20, 0.3, 0.02, feature_dim=16), 3),
        'citation': (lambda: create_citation_network(80, 3, 16), 3),
    }

    print(f"\n{'Dataset':<15} {'Config':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 55)

    for name, (dataset_fn, n_classes) in datasets.items():
        graph, labels = dataset_fn()
        train_mask, _, test_mask = create_transductive_split(
            graph.n_nodes, labels, 0.2, 0.1
        )

        # Use the best default config (SAGE-style)
        mpnn = MPNN(graph.X.shape[1], 16, n_classes, n_layers=2,
                    message_type='simple', aggregate_type='mean',
                    update_type='concat', lr=0.01, dropout=0.3, random_state=42)
        mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)

        preds = mpnn.predict(graph)
        train_acc = np.mean(preds[train_mask] == labels[train_mask])
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name:<15} {'SAGE-style':<15} {train_acc:<12.3f} {test_acc:<12.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_mpnn_framework():
    """Visualize the MPNN framework and unification."""
    print("\nGenerating: MPNN framework visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Use community graph
    graph, labels = create_community_graph(
        3, 20, 0.25, 0.02, feature_dim=16, random_state=42
    )
    train_mask, _, test_mask = create_transductive_split(
        graph.n_nodes, labels, 0.2, 0.1
    )
    pos = spring_layout(graph, seed=42)

    # Panel 1: Ground truth
    draw_graph(graph, labels, pos, axes[0, 0],
              title='Ground Truth\n(3 communities)', cmap='Set1')

    # Panel 2: MPNN prediction
    mpnn = MPNN(graph.X.shape[1], 16, 3, n_layers=2,
                message_type='simple', aggregate_type='sum',
                update_type='concat', lr=0.01, dropout=0.3, random_state=42)
    losses = mpnn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
    preds = mpnn.predict(graph)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])
    draw_graph(graph, preds, pos, axes[0, 1],
              title=f'MPNN Predictions\ntest_acc={test_acc:.3f}', cmap='Set1')

    # Panel 3: Architecture comparison bar chart
    ax = axes[0, 2]
    configs = {
        'GCN\nstyle': mpnn_as_gcn_config(),
        'SAGE\nstyle': mpnn_as_graphsage_config(),
        'GIN\nstyle': mpnn_as_gin_config(),
    }
    accs = {}
    for name, cfg in configs.items():
        m = MPNN(graph.X.shape[1], 16, 3, n_layers=2,
                 lr=0.01, dropout=0.3, random_state=42, **cfg)
        m.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        p = m.predict(graph)
        accs[name] = np.mean(p[test_mask] == labels[test_mask])

    colors = ['steelblue', 'coral', '#2ecc71']
    bars = ax.bar(accs.keys(), accs.values(), color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('MPNN as Different GNNs\n(same framework, different config)')
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Aggregation comparison
    ax = axes[1, 0]
    agg_accs = {}
    for agg in ['sum', 'mean', 'max']:
        m = MPNN(graph.X.shape[1], 16, 3, n_layers=2,
                 message_type='simple', aggregate_type=agg,
                 update_type='concat', lr=0.01, dropout=0.3, random_state=42)
        m.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        p = m.predict(graph)
        agg_accs[agg] = np.mean(p[test_mask] == labels[test_mask])

    bars = ax.bar(agg_accs.keys(), agg_accs.values(),
                  color=['#2ecc71', '#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Aggregation Function\nSUM vs MEAN vs MAX')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: Training curve
    ax = axes[1, 1]
    ax.plot(losses, 'b-', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MPNN Training Loss')
    ax.grid(True, alpha=0.3)

    # Panel 6: The unification table as text
    ax = axes[1, 2]
    ax.axis('off')
    table_text = (
        "MPNN UNIFICATION TABLE\n"
        "---------------------------------------\n"
        "Model      MSG        AGG    UPD\n"
        "---------------------------------------\n"
        "GCN        W*h_j      SUM    s(W*sum)\n"
        "GraphSAGE  h_j        MEAN   s(W[h||m])\n"
        "GAT        a*W*h_j    SUM    s(sum)\n"
        "GIN        h_j        SUM    MLP(h+sum)\n"
        "---------------------------------------\n\n"
        "ALL GNNs = MESSAGE PASSING!\n"
        "Choose M, AGG, U -> get a GNN\n\n"
        "EXPRESSIVENESS:\n"
        "  All MPNN bounded by 1-WL test\n"
        "  SUM > MEAN > MAX (injectivity)\n"
    )
    ax.text(0.1, 0.5, table_text, fontsize=10, fontfamily='monospace',
           va='center', ha='left',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('MPNN: Message Passing Neural Network\n'
                 'ALL GNNs are instances of the Message -> Aggregate -> Update framework',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("MPNN -- Paradigm: MESSAGE PASSING (Unified Framework)")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    A GENERAL framework for ALL Graph Neural Networks.

    Every GNN does three things per layer:
        1. MESSAGE:   m_ij = M(h_i, h_j, e_ij)
        2. AGGREGATE: m_i  = AGG({m_ij : j in N(i)})
        3. UPDATE:    h_i' = U(h_i, m_i)

    CHOOSE M, AGG, U -> GET A SPECIFIC GNN:
        GCN       = linear message  + SUM + add
        GraphSAGE = simple message  + MEAN + concat
        GAT       = attention msg   + SUM + direct
        GIN       = simple message  + SUM + MLP

KEY INSIGHT:
    All GNNs are MESSAGE PASSING.
    The difference is just the choice of M, AGG, U.

    Once you understand this, you can:
    - DESIGN new GNNs by mixing M, AGG, U
    - ANALYZE why one GNN works better than another
    - COMBINE ideas (e.g., attention + GRU update)
    """)

    ablation_experiments()
    results = benchmark_on_datasets()

    print("\nGenerating visualizations...")

    fig1 = visualize_mpnn_framework()
    save_path1 = '/Users/sid47/ML Algorithms/40_mpnn.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    print("\n" + "="*60)
    print("SUMMARY: What MPNN Reveals")
    print("="*60)
    print("""
1. ALL GNNs = MESSAGE PASSING
   GCN, GraphSAGE, GAT, GIN are all special cases
   of the same MESSAGE -> AGGREGATE -> UPDATE framework.

2. AGGREGATION MATTERS
   SUM: counts matter (most expressive, see GIN)
   MEAN: size-invariant (good for varying degree)
   MAX: captures salient features

3. UPDATE FUNCTION
   concat: preserves self vs neighbor info
   add: simpler but mixes signals
   GRU: gated, helps with deep networks

4. EDGE FEATURES
   MPNN naturally handles edge attributes
   via edge-conditioned message function.

5. EXPRESSIVENESS BOUND
   ALL MPNN variants bounded by 1-WL test.
   No message passing GNN can distinguish graphs
   that the WL test cannot.

CONNECTION TO OTHER FILES:
    36_gcn.py: MPNN with linear message + SUM + add
    37_graphsage.py: MPNN with simple message + MEAN + concat
    38_gat.py: MPNN with attention message + SUM
    39_gin.py: MPNN with simple message + SUM + MLP (maximally expressive)
    41_graph_pooling.py: Goes from node to GRAPH representation

NEXT: 41_graph_pooling.py -- How to classify entire GRAPHS?
    """)
