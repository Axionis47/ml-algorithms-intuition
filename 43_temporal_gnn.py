"""
TEMPORAL GNN — Paradigm: DYNAMIC GRAPHS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Real-world graphs CHANGE over time:
- Social networks: friendships form and dissolve
- Financial networks: transactions happen at specific times
- Traffic networks: congestion patterns shift hourly
- Citation networks: new papers appear, citing older ones

STATIC GRAPH:   G = (V, E)          -- fixed snapshot
TEMPORAL GRAPH: G(t) = (V(t), E(t)) -- evolves over time

The question: How do we learn on graphs that change?

===============================================================
TWO APPROACHES
===============================================================

1. DISCRETE TIME (Snapshots)
   G_1, G_2, ..., G_T

   Process each snapshot with a GNN (spatial aggregation),
   then combine across time (temporal aggregation).

   h_v^(t) = GNN(G_t, X_t)       -- spatial: aggregate neighbors
   z_v^(t) = RNN(h_v^(1:t))      -- temporal: aggregate history

   PROS: Simple, reuses existing GNN architectures
   CONS: Loses fine-grained temporal info between snapshots

2. CONTINUOUS TIME (Events)
   Events: (u, v, t, type)

   Each interaction updates node memory via a GRU.
   Time encoding (sinusoidal) captures "how long ago" something
   happened, analogous to positional encoding in Transformers.

   PROS: Fine-grained, captures exact timing
   CONS: Sequential processing, harder to parallelize

===============================================================
TIME ENCODING
===============================================================

How to represent time difference dt?

Sinusoidal (like Transformer positional encoding):
    Phi(dt) = [cos(w_1 * dt), sin(w_1 * dt),
               cos(w_2 * dt), sin(w_2 * dt), ...]

This gives a smooth, continuous, fixed-dimensional representation
of any time gap. Different frequencies capture different scales.

===============================================================
NODE MEMORY (for continuous-time models)
===============================================================

Each node maintains a memory vector s_i that is updated
whenever the node participates in an interaction:

    message:  m_i(t) = f(s_i, s_j, Phi(dt), e_ij)
    update:   s_i(t) = GRU(s_i(t-), m_i(t))

The memory captures a node's interaction history in a
compressed, fixed-size vector.

===============================================================
INDUCTIVE BIAS
===============================================================

1. TEMPORAL LOCALITY: Recent events matter more than old ones
2. TEMPORAL PATTERNS: Behaviors repeat or evolve smoothly
3. SPATIAL + TEMPORAL: Both graph structure and time matter
4. STATIONARITY: The dynamics do not change abruptly
   (violation: concept drift, regime changes)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
graph_module = import_module('35_graph_fundamentals')
Graph = graph_module.Graph
create_community_graph = graph_module.create_community_graph
create_transductive_split = graph_module.create_transductive_split
spring_layout = graph_module.spring_layout
draw_graph = graph_module.draw_graph

gcn_module = import_module('36_gcn')
softmax = gcn_module.softmax
cross_entropy_loss = gcn_module.cross_entropy_loss
GCN = gcn_module.GCN


# ============================================================
# TIME ENCODING
# ============================================================

def time_encoding(delta_t, dim):
    """
    Sinusoidal time encoding, analogous to positional encoding
    in Transformers (15_transformer.py).

    Maps a scalar time difference to a dim-dimensional vector
    using cosines and sines at different frequencies.

    Parameters:
    -----------
    delta_t : float or ndarray
        Time difference(s) to encode
    dim : int
        Encoding dimension (must be even)

    Returns:
    --------
    encoding : ndarray of shape (..., dim)
    """
    delta_t = np.atleast_1d(np.asarray(delta_t, dtype=float))
    half = dim // 2
    # Frequencies span multiple scales: from fast to slow oscillation
    freqs = 1.0 / (10000 ** (np.arange(half) / max(half, 1)))
    # Outer product: (n_times, half)
    angles = np.outer(delta_t, freqs)
    encoding = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    if encoding.shape[0] == 1:
        encoding = encoding.squeeze(0)
    return encoding


# ============================================================
# TEMPORAL COMMUNITY GRAPH GENERATOR
# ============================================================

def create_temporal_community_graph(n_nodes=40, n_communities=3,
                                     n_snapshots=5, p_in=0.3, p_out=0.03,
                                     migration_rate=0.08, feature_dim=16,
                                     random_state=42):
    """
    Create a sequence of graph snapshots with evolving community structure.

    Communities gradually evolve: at each snapshot, some nodes may
    migrate to a neighboring community. Edge structure is regenerated
    each snapshot according to the current community assignments.

    Parameters:
    -----------
    n_nodes : int
        Total number of nodes (shared across all snapshots)
    n_communities : int
        Number of communities
    n_snapshots : int
        Number of temporal snapshots
    p_in : float
        Intra-community edge probability
    p_out : float
        Inter-community edge probability
    migration_rate : float
        Fraction of nodes that switch communities per snapshot
    feature_dim : int
        Node feature dimension
    random_state : int

    Returns:
    --------
    snapshots : list of (Graph, labels) tuples, one per snapshot
    """
    rng = np.random.RandomState(random_state)

    # Initial community assignments
    labels = np.array([i % n_communities for i in range(n_nodes)])
    rng.shuffle(labels)

    # Community prototypes for features
    prototypes = rng.randn(n_communities, feature_dim) * 2.0

    snapshots = []
    for t in range(n_snapshots):
        # Migrate some nodes to neighboring community
        if t > 0:
            n_migrate = max(1, int(n_nodes * migration_rate))
            migrants = rng.choice(n_nodes, n_migrate, replace=False)
            for m in migrants:
                # Move to a random different community
                old = labels[m]
                candidates = [c for c in range(n_communities) if c != old]
                labels[m] = rng.choice(candidates)

        # Generate node features correlated with current community
        X = np.zeros((n_nodes, feature_dim))
        for c in range(n_communities):
            mask = labels == c
            n_c = np.sum(mask)
            if n_c > 0:
                X[mask] = prototypes[c] + rng.randn(n_c, feature_dim) * 0.5
        # Add small temporal drift to features
        X += rng.randn(n_nodes, feature_dim) * 0.1 * (t + 1)

        # Build graph for this snapshot
        graph = Graph(n_nodes, X)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                p = p_in if labels[i] == labels[j] else p_out
                if rng.rand() < p:
                    graph.add_edge(i, j)

        snapshots.append((graph, labels.copy()))

    return snapshots


# ============================================================
# HELPER: NORMALIZED ADJACENCY FORWARD PASS
# ============================================================

def _gcn_layer_forward(A, H, W, b, apply_relu=True):
    """
    Single GCN layer forward pass: H' = sigma(A_hat @ H @ W + b)

    Parameters:
    -----------
    A : ndarray (n, n) — raw adjacency matrix
    H : ndarray (n, d_in) — input node features
    W : ndarray (d_in, d_out) — weight matrix
    b : ndarray (d_out,) — bias
    apply_relu : bool — whether to apply ReLU

    Returns:
    --------
    H_out : ndarray (n, d_out)
    A_hat : ndarray (n, n) — normalized adjacency (for reuse)
    Z : ndarray (n, d_out) — pre-activation (for backprop)
    """
    n = A.shape[0]
    A_tilde = A + np.eye(n)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    Z = A_hat @ H @ W + b
    if apply_relu:
        H_out = np.maximum(Z, 0)
    else:
        H_out = Z
    return H_out, A_hat, Z


# ============================================================
# SNAPSHOT GNN (Primary Model)
# ============================================================

class SnapshotGNN:
    """
    Snapshot-based Temporal Graph Neural Network.

    Architecture:
        For each snapshot t:
            H_t = GCN_forward(G_t, X_t)   -- 2-layer GCN (shared weights)
        Temporal aggregation:
            Z = sum_t alpha_t * H_t        -- learned temporal attention
        Output:
            Y = softmax(Z @ W_out + b_out) -- node classification

    Training uses full analytical backpropagation through:
        softmax -> output linear -> temporal attention -> GCN layers

    Parameters:
    -----------
    n_features : int — input feature dimension
    n_hidden : int — hidden dimension for GCN layers
    n_classes : int — number of output classes
    n_snapshots : int — number of temporal snapshots
    lr : float — learning rate
    random_state : int or None
    """

    def __init__(self, n_features, n_hidden, n_classes, n_snapshots,
                 lr=0.01, random_state=None):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_snapshots = n_snapshots
        self.lr = lr
        self.random_state = random_state
        self._init_weights()

    def _init_weights(self):
        """Initialize GCN weights (shared across snapshots) + temporal + output."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # GCN layer 1: features -> hidden
        std1 = np.sqrt(2.0 / (self.n_features + self.n_hidden))
        self.W1 = np.random.randn(self.n_features, self.n_hidden) * std1
        self.b1 = np.zeros(self.n_hidden)

        # GCN layer 2: hidden -> hidden
        std2 = np.sqrt(2.0 / (self.n_hidden + self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, self.n_hidden) * std2
        self.b2 = np.zeros(self.n_hidden)

        # Temporal attention: one learnable weight per snapshot
        # Initialized to uniform (1/T), so it starts as simple averaging
        self.temporal_logits = np.zeros(self.n_snapshots)

        # Output layer: hidden -> classes
        std_out = np.sqrt(2.0 / (self.n_hidden + self.n_classes))
        self.W_out = np.random.randn(self.n_hidden, self.n_classes) * std_out
        self.b_out = np.zeros(self.n_classes)

    def _temporal_attention(self):
        """Softmax over temporal logits to get snapshot weights."""
        e = np.exp(self.temporal_logits - np.max(self.temporal_logits))
        return e / (np.sum(e) + 1e-10)

    def forward(self, snapshots, training=True):
        """
        Forward pass over a list of (Graph, labels) snapshots.

        Parameters:
        -----------
        snapshots : list of (Graph, labels) tuples
        training : bool

        Returns:
        --------
        probs : ndarray (n_nodes, n_classes) — softmax output
        cache : dict — cached values for backprop
        """
        n_nodes = snapshots[0][0].n_nodes
        T = len(snapshots)

        # Temporal attention weights
        alpha = self._temporal_attention()

        cache = {
            'snapshots': snapshots,
            'alpha': alpha,
            'H0_list': [],   # input features per snapshot
            'A_hat_list': [],
            'Z1_list': [],   # pre-ReLU layer 1
            'H1_list': [],   # post-ReLU layer 1
            'Z2_list': [],   # output of GCN layer 2 (no activation)
        }

        # Process each snapshot with shared GCN weights
        H_all = np.zeros((T, n_nodes, self.n_hidden))

        for t in range(T):
            graph_t = snapshots[t][0]
            X_t = graph_t.X
            A_t = graph_t.A

            cache['H0_list'].append(X_t)

            # GCN layer 1: ReLU(A_hat @ X @ W1 + b1)
            H1, A_hat, Z1 = _gcn_layer_forward(A_t, X_t, self.W1, self.b1,
                                                  apply_relu=True)
            cache['A_hat_list'].append(A_hat)
            cache['Z1_list'].append(Z1)
            cache['H1_list'].append(H1)

            # GCN layer 2: A_hat @ H1 @ W2 + b2 (no activation — linear output)
            Z2 = A_hat @ H1 @ self.W2 + self.b2
            cache['Z2_list'].append(Z2)

            H_all[t] = Z2

        # Temporal aggregation: Z = sum_t alpha_t * H_t
        Z_temporal = np.zeros((n_nodes, self.n_hidden))
        for t in range(T):
            Z_temporal += alpha[t] * H_all[t]
        cache['H_all'] = H_all
        cache['Z_temporal'] = Z_temporal

        # Output: softmax(Z_temporal @ W_out + b_out)
        logits = Z_temporal @ self.W_out + self.b_out
        probs = softmax(logits)
        cache['logits'] = logits
        cache['probs'] = probs

        return probs, cache

    def backward(self, labels, train_mask, cache):
        """
        Backpropagation through temporal attention + GCN layers.

        Gradient flow:
            dL/dprobs -> dL/dlogits -> dL/dZ_temporal
            -> dL/dalpha (temporal attention grad)
            -> dL/dH_all[t] (per-snapshot)
            -> dL/dW2, dL/db2, dL/dW1, dL/db1 (GCN weight grads)
        """
        n = len(labels)
        n_train = np.sum(train_mask)
        probs = cache['probs']
        alpha = cache['alpha']
        H_all = cache['H_all']
        T = len(cache['snapshots'])

        # dL/dlogits (cross-entropy + softmax gradient)
        dlogits = probs.copy()
        dlogits[np.arange(n), labels] -= 1.0
        mask_float = train_mask.astype(float)
        dlogits = dlogits * mask_float[:, None] / max(n_train, 1)

        # dL/dW_out, dL/db_out
        Z_temporal = cache['Z_temporal']
        dW_out = Z_temporal.T @ dlogits
        db_out = np.sum(dlogits, axis=0)

        # dL/dZ_temporal
        dZ_temporal = dlogits @ self.W_out.T

        # dL/dalpha_t (temporal attention logit gradients)
        dalpha = np.zeros(T)
        for t in range(T):
            # dL/dalpha_t = sum over nodes/dims of dZ_temporal * H_all[t]
            dalpha[t] = np.sum(dZ_temporal * H_all[t])

        # Softmax backprop for temporal logits
        # d(softmax)/d(logit_i) = alpha_i * (delta_ij - alpha_j)
        dtemporal_logits = np.zeros(T)
        for t in range(T):
            for s in range(T):
                if t == s:
                    dtemporal_logits[t] += dalpha[s] * alpha[s] * (1.0 - alpha[s])
                else:
                    dtemporal_logits[t] -= dalpha[s] * alpha[s] * alpha[t]

        # Per-snapshot gradients: dL/dH_all[t] = alpha_t * dZ_temporal
        # Accumulate GCN weight gradients across snapshots
        dW2_accum = np.zeros_like(self.W2)
        db2_accum = np.zeros_like(self.b2)
        dW1_accum = np.zeros_like(self.W1)
        db1_accum = np.zeros_like(self.b1)

        for t in range(T):
            A_hat = cache['A_hat_list'][t]
            H0 = cache['H0_list'][t]
            Z1 = cache['Z1_list'][t]
            H1 = cache['H1_list'][t]

            # dL/dZ2_t = alpha_t * dZ_temporal
            dZ2 = alpha[t] * dZ_temporal

            # Backprop through GCN layer 2: Z2 = A_hat @ H1 @ W2 + b2
            dZ2_prop = A_hat.T @ dZ2
            dW2_accum += H1.T @ dZ2_prop
            db2_accum += np.sum(dZ2_prop, axis=0)

            # dL/dH1 = dZ2_prop @ W2.T
            dH1 = dZ2_prop @ self.W2.T

            # Backprop through ReLU
            dZ1 = dH1 * (Z1 > 0).astype(float)

            # Backprop through GCN layer 1: Z1 = A_hat @ H0 @ W1 + b1
            dZ1_prop = A_hat.T @ dZ1
            dW1_accum += H0.T @ dZ1_prop
            db1_accum += np.sum(dZ1_prop, axis=0)

        # Update weights
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
        self.W2 -= self.lr * dW2_accum
        self.b2 -= self.lr * db2_accum
        self.W1 -= self.lr * dW1_accum
        self.b1 -= self.lr * db1_accum
        self.temporal_logits -= self.lr * dtemporal_logits

    def fit(self, snapshots, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train the Snapshot GNN.

        Parameters:
        -----------
        snapshots : list of (Graph, labels) tuples
        labels : ndarray — ground truth labels (for the FINAL snapshot)
        train_mask : boolean array
        n_epochs : int
        verbose : bool

        Returns:
        --------
        loss_history : list of float
        """
        loss_history = []
        for epoch in range(n_epochs):
            probs, cache = self.forward(snapshots, training=True)
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)
            self.backward(labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                preds = np.argmax(probs, axis=1)
                train_acc = np.mean(preds[train_mask] == labels[train_mask])
                print(f"  Epoch {epoch+1:>4}: loss={loss:.4f}, "
                      f"train_acc={train_acc:.3f}")

        return loss_history

    def predict(self, snapshots):
        """Predict node labels from a sequence of snapshots."""
        probs, _ = self.forward(snapshots, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, snapshots):
        """Predict node class probabilities."""
        probs, _ = self.forward(snapshots, training=False)
        return probs

    def get_temporal_weights(self):
        """Return the learned temporal attention weights."""
        return self._temporal_attention()


# ============================================================
# TEMPORAL GNN (Continuous-Time Model)
# ============================================================

class TemporalGNN:
    """
    Simplified continuous-time Temporal Graph Network.

    Each node has a memory vector that is updated whenever the
    node participates in an interaction. Uses time encoding to
    capture temporal patterns.

    Used for temporal link prediction: given a history of events,
    predict whether a future edge will exist.

    Parameters:
    -----------
    n_nodes : int
    n_features : int — node feature dimension (unused, for API compat)
    memory_dim : int — dimension of node memory
    time_dim : int — dimension of time encoding
    lr : float — learning rate
    random_state : int or None
    """

    def __init__(self, n_nodes, n_features=16, memory_dim=16,
                 time_dim=8, lr=0.01, random_state=None):
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.lr = lr

        if random_state is not None:
            np.random.seed(random_state)

        # Node memory
        self.memory = np.zeros((n_nodes, memory_dim))
        self.last_time = np.zeros(n_nodes)

        # Message function: concat(mem_src, mem_dst, time_enc) -> message
        msg_input = 2 * memory_dim + time_dim
        std_msg = np.sqrt(2.0 / (msg_input + memory_dim))
        self.W_msg = np.random.randn(msg_input, memory_dim) * std_msg
        self.b_msg = np.zeros(memory_dim)

        # GRU-style update gate
        gru_input = 2 * memory_dim
        std_gru = np.sqrt(2.0 / (gru_input + memory_dim))
        self.W_z = np.random.randn(gru_input, memory_dim) * std_gru
        self.b_z = np.zeros(memory_dim)
        self.W_r = np.random.randn(gru_input, memory_dim) * std_gru
        self.b_r = np.zeros(memory_dim)
        self.W_h = np.random.randn(gru_input, memory_dim) * std_gru
        self.b_h = np.zeros(memory_dim)

        # Link predictor: concat(mem_src, mem_dst) -> scalar
        std_link = np.sqrt(2.0 / (2 * memory_dim + 1))
        self.W_link = np.random.randn(2 * memory_dim, 1) * std_link
        self.b_link = np.zeros(1)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def process_event(self, src, dst, t):
        """
        Process a single interaction event and update node memories.

        Parameters:
        -----------
        src : int — source node
        dst : int — destination node
        t : float — event time
        """
        dt_src = t - self.last_time[src]
        dt_dst = t - self.last_time[dst]
        te_src = time_encoding(dt_src, self.time_dim)
        te_dst = time_encoding(dt_dst, self.time_dim)

        # Compute messages
        msg_input_src = np.concatenate([self.memory[src], self.memory[dst],
                                         te_src])
        msg_input_dst = np.concatenate([self.memory[dst], self.memory[src],
                                         te_dst])
        msg_src = np.tanh(msg_input_src @ self.W_msg + self.b_msg)
        msg_dst = np.tanh(msg_input_dst @ self.W_msg + self.b_msg)

        # GRU update for source
        self.memory[src] = self._gru_update(self.memory[src], msg_src)
        # GRU update for destination
        self.memory[dst] = self._gru_update(self.memory[dst], msg_dst)

        self.last_time[src] = t
        self.last_time[dst] = t

    def _gru_update(self, mem, msg):
        """GRU-style memory update."""
        concat = np.concatenate([mem, msg])
        z = self._sigmoid(concat @ self.W_z + self.b_z)
        r = self._sigmoid(concat @ self.W_r + self.b_r)
        concat_reset = np.concatenate([r * mem, msg])
        h_tilde = np.tanh(concat_reset @ self.W_h + self.b_h)
        return (1 - z) * mem + z * h_tilde

    def predict_link(self, src, dst):
        """
        Predict probability of edge (src, dst) based on current memory.

        Returns:
        --------
        prob : float in [0, 1]
        """
        concat = np.concatenate([self.memory[src], self.memory[dst]])
        logit = concat @ self.W_link + self.b_link
        return float(self._sigmoid(logit))

    def get_embeddings(self):
        """Return current node memory as embeddings."""
        return self.memory.copy()

    def reset(self):
        """Reset all node memories and timestamps."""
        self.memory = np.zeros((self.n_nodes, self.memory_dim))
        self.last_time = np.zeros(self.n_nodes)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    # -------- Experiment 1: Lookback Window --------
    print("\n1. LOOKBACK WINDOW (number of snapshots used)")
    print("-" * 40)
    print("How many past snapshots should the model see?")

    snapshots = create_temporal_community_graph(
        n_nodes=40, n_communities=3, n_snapshots=5,
        p_in=0.3, p_out=0.03, migration_rate=0.08, random_state=42
    )
    final_labels = snapshots[-1][1]
    n_classes = len(np.unique(final_labels))
    train_mask, _, test_mask = create_transductive_split(
        40, final_labels, train_ratio=0.3, val_ratio=0.1, random_state=42
    )

    for lookback in [1, 2, 3, 5]:
        # Use last 'lookback' snapshots
        subset = snapshots[-lookback:]
        model = SnapshotGNN(
            n_features=snapshots[0][0].X.shape[1],
            n_hidden=16, n_classes=n_classes,
            n_snapshots=lookback, lr=0.01, random_state=42
        )
        model.fit(subset, final_labels, train_mask, n_epochs=200,
                  verbose=False)
        preds = model.predict(subset)
        test_acc = np.mean(preds[test_mask] == final_labels[test_mask])
        train_acc = np.mean(preds[train_mask] == final_labels[train_mask])
        weights = model.get_temporal_weights()
        weight_str = ", ".join(f"{w:.2f}" for w in weights)
        print(f"  lookback={lookback}  train_acc={train_acc:.3f}  "
              f"test_acc={test_acc:.3f}  weights=[{weight_str}]")

    print("-> More snapshots give more temporal context")
    print("-> Temporal weights show which snapshots matter most")
    print("-> Recent snapshots usually get higher weight")

    # -------- Experiment 2: With vs Without Temporal Encoding --------
    print("\n2. SNAPSHOT GNN vs STATIC GCN (last snapshot only)")
    print("-" * 40)
    print("Does temporal information help?")

    # Temporal: SnapshotGNN with all snapshots
    model_temporal = SnapshotGNN(
        n_features=snapshots[0][0].X.shape[1],
        n_hidden=16, n_classes=n_classes,
        n_snapshots=len(snapshots), lr=0.01, random_state=42
    )
    model_temporal.fit(snapshots, final_labels, train_mask,
                       n_epochs=200, verbose=False)
    preds_temporal = model_temporal.predict(snapshots)
    acc_temporal = np.mean(preds_temporal[test_mask] == final_labels[test_mask])

    # Static: GCN on last snapshot only
    last_graph = snapshots[-1][0]
    gcn_static = GCN(
        last_graph.X.shape[1], 16, n_classes, n_layers=2,
        dropout=0.3, lr=0.01, random_state=42
    )
    gcn_static.fit(last_graph, final_labels, train_mask,
                   n_epochs=200, verbose=False)
    preds_static = gcn_static.predict(last_graph)
    acc_static = np.mean(preds_static[test_mask] == final_labels[test_mask])

    # Static: GCN on first snapshot only
    first_graph = snapshots[0][0]
    first_labels = snapshots[0][1]
    gcn_first = GCN(
        first_graph.X.shape[1], 16, n_classes, n_layers=2,
        dropout=0.3, lr=0.01, random_state=42
    )
    gcn_first.fit(first_graph, final_labels, train_mask,
                  n_epochs=200, verbose=False)
    preds_first = gcn_first.predict(first_graph)
    acc_first = np.mean(preds_first[test_mask] == final_labels[test_mask])

    print(f"  SnapshotGNN (all snapshots): {acc_temporal:.3f}")
    print(f"  Static GCN (last snapshot):  {acc_static:.3f}")
    print(f"  Static GCN (first snapshot): {acc_first:.3f}")
    print("-> Temporal model sees community evolution")
    print("-> Static model misses historical structure")
    print("-> First snapshot may have stale community info")

    # -------- Experiment 3: Number of Training Snapshots --------
    print("\n3. NUMBER OF TRAINING SNAPSHOTS")
    print("-" * 40)
    print("How does increasing temporal data help?")

    bigger_snapshots = create_temporal_community_graph(
        n_nodes=40, n_communities=3, n_snapshots=8,
        p_in=0.3, p_out=0.03, migration_rate=0.08, random_state=42
    )
    final_labels_big = bigger_snapshots[-1][1]
    train_mask_big, _, test_mask_big = create_transductive_split(
        40, final_labels_big, train_ratio=0.3, val_ratio=0.1, random_state=42
    )
    n_classes_big = len(np.unique(final_labels_big))

    for n_snap in [1, 2, 4, 6, 8]:
        subset = bigger_snapshots[-n_snap:]
        model = SnapshotGNN(
            n_features=bigger_snapshots[0][0].X.shape[1],
            n_hidden=16, n_classes=n_classes_big,
            n_snapshots=n_snap, lr=0.01, random_state=42
        )
        model.fit(subset, final_labels_big, train_mask_big,
                  n_epochs=200, verbose=False)
        preds = model.predict(subset)
        test_acc = np.mean(preds[test_mask_big] == final_labels_big[test_mask_big])
        print(f"  n_snapshots={n_snap}  test_acc={test_acc:.3f}")

    print("-> More snapshots generally helps, then saturates")
    print("-> Very old snapshots may have irrelevant structure")

    # -------- Experiment 4: Community Migration Rate --------
    print("\n4. COMMUNITY MIGRATION RATE")
    print("-" * 40)
    print("How does the rate of change affect temporal models?")

    for rate in [0.0, 0.05, 0.10, 0.20, 0.40]:
        snaps = create_temporal_community_graph(
            n_nodes=40, n_communities=3, n_snapshots=5,
            p_in=0.3, p_out=0.03, migration_rate=rate, random_state=42
        )
        fl = snaps[-1][1]
        n_cl = len(np.unique(fl))
        tm, _, tsm = create_transductive_split(
            40, fl, train_ratio=0.3, val_ratio=0.1, random_state=42
        )
        # Compare temporal vs static
        model_t = SnapshotGNN(
            n_features=snaps[0][0].X.shape[1],
            n_hidden=16, n_classes=n_cl,
            n_snapshots=len(snaps), lr=0.01, random_state=42
        )
        model_t.fit(snaps, fl, tm, n_epochs=200, verbose=False)
        acc_t = np.mean(model_t.predict(snaps)[tsm] == fl[tsm])

        gcn_s = GCN(
            snaps[-1][0].X.shape[1], 16, n_cl, n_layers=2,
            dropout=0.3, lr=0.01, random_state=42
        )
        gcn_s.fit(snaps[-1][0], fl, tm, n_epochs=200, verbose=False)
        acc_s = np.mean(gcn_s.predict(snaps[-1][0])[tsm] == fl[tsm])

        gap = acc_t - acc_s
        print(f"  migration={rate:.2f}  temporal={acc_t:.3f}  "
              f"static={acc_s:.3f}  gap={gap:+.3f}")

    print("-> Low migration: static is nearly as good (graph barely changes)")
    print("-> High migration: temporal model adapts better")
    print("-> Very high migration: both struggle (too much noise)")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    """Benchmark temporal GNN on community prediction tasks."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Temporal GNN on Dynamic Community Prediction")
    print("=" * 60)

    configs = [
        ("3-comm, 5 snap", 40, 3, 5, 0.30, 0.03, 0.08),
        ("3-comm, 3 snap", 40, 3, 3, 0.30, 0.03, 0.08),
        ("2-comm, 5 snap", 40, 2, 5, 0.35, 0.03, 0.10),
        ("4-comm, 5 snap", 40, 4, 5, 0.30, 0.03, 0.06),
        ("3-comm, dense",  40, 3, 5, 0.40, 0.05, 0.08),
    ]

    print(f"\n{'Config':<20} {'Snap-GNN':<12} {'GCN(last)':<12} "
          f"{'GCN(first)':<12} {'Gain':<8}")
    print("-" * 65)

    results = {}
    for name, n_n, n_c, n_s, pi, po, mr in configs:
        snaps = create_temporal_community_graph(
            n_nodes=n_n, n_communities=n_c, n_snapshots=n_s,
            p_in=pi, p_out=po, migration_rate=mr, random_state=42
        )
        fl = snaps[-1][1]
        n_cl = len(np.unique(fl))
        tm, _, tsm = create_transductive_split(
            n_n, fl, train_ratio=0.3, val_ratio=0.1, random_state=42
        )

        # SnapshotGNN
        sgnn = SnapshotGNN(
            snaps[0][0].X.shape[1], 16, n_cl, len(snaps),
            lr=0.01, random_state=42
        )
        sgnn.fit(snaps, fl, tm, n_epochs=200, verbose=False)
        acc_sgnn = np.mean(sgnn.predict(snaps)[tsm] == fl[tsm])

        # GCN last
        gcn_l = GCN(
            snaps[-1][0].X.shape[1], 16, n_cl, n_layers=2,
            dropout=0.3, lr=0.01, random_state=42
        )
        gcn_l.fit(snaps[-1][0], fl, tm, n_epochs=200, verbose=False)
        acc_last = np.mean(gcn_l.predict(snaps[-1][0])[tsm] == fl[tsm])

        # GCN first
        gcn_f = GCN(
            snaps[0][0].X.shape[1], 16, n_cl, n_layers=2,
            dropout=0.3, lr=0.01, random_state=42
        )
        gcn_f.fit(snaps[0][0], fl, tm, n_epochs=200, verbose=False)
        acc_first = np.mean(gcn_f.predict(snaps[0][0])[tsm] == fl[tsm])

        gain = acc_sgnn - acc_last
        print(f"{name:<20} {acc_sgnn:<12.3f} {acc_last:<12.3f} "
              f"{acc_first:<12.3f} {gain:+.3f}")

        results[name] = {
            'snapshot_gnn': acc_sgnn,
            'gcn_last': acc_last,
            'gcn_first': acc_first,
        }

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_temporal_graph():
    """
    Comprehensive temporal GNN visualization (2x3 grid):

    Row 1: Three snapshots showing graph evolution (community migration)
    Row 2: Snapshot GNN vs static GCN, training curve, lookback effect
    """
    print("\nGenerating: Temporal GNN visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    np.random.seed(42)

    # Create temporal community graph with visible evolution
    snapshots = create_temporal_community_graph(
        n_nodes=35, n_communities=3, n_snapshots=5,
        p_in=0.30, p_out=0.03, migration_rate=0.10,
        feature_dim=16, random_state=42
    )
    final_labels = snapshots[-1][1]
    n_classes = len(np.unique(final_labels))
    n_nodes = 35
    train_mask, _, test_mask = create_transductive_split(
        n_nodes, final_labels, train_ratio=0.3, val_ratio=0.1,
        random_state=42
    )

    # Use a fixed layout based on the first snapshot for consistency
    pos = spring_layout(snapshots[0][0], n_iter=80, seed=42)

    # ---- Panel 1-3: Three snapshots showing evolution ----
    snapshot_indices = [0, 2, 4]
    for panel_idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[0, panel_idx]
        graph_t, labels_t = snapshots[snap_idx]

        # Draw edges
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if graph_t.A[i, j] > 0:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            'gray', alpha=0.2, linewidth=0.5)

        # Draw nodes colored by community
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], c=labels_t, cmap='Set1',
            s=80, alpha=0.85, edgecolors='black', linewidths=0.5, zorder=3,
            vmin=0, vmax=n_classes - 1
        )

        # Highlight nodes that changed community since first snapshot
        if snap_idx > 0:
            first_labels = snapshots[0][1]
            changed = labels_t != first_labels
            if np.any(changed):
                ax.scatter(
                    pos[changed, 0], pos[changed, 1],
                    facecolors='none', edgecolors='gold', s=160,
                    linewidths=2.0, zorder=4, label='Migrated'
                )
                ax.legend(loc='lower right', fontsize=7)

        n_edges = int(np.sum(graph_t.A > 0)) // 2
        ax.set_title(f'Snapshot t={snap_idx + 1}\n'
                      f'{n_edges} edges, '
                      f'{np.sum(labels_t != snapshots[0][1]) if snap_idx > 0 else 0} '
                      f'migrated nodes')
        ax.set_aspect('equal')
        ax.axis('off')

    # ---- Panel 4: Snapshot GNN vs Static GCN accuracy ----
    ax = axes[1, 0]

    # Train SnapshotGNN
    sgnn = SnapshotGNN(
        n_features=snapshots[0][0].X.shape[1],
        n_hidden=16, n_classes=n_classes,
        n_snapshots=len(snapshots), lr=0.01, random_state=42
    )
    sgnn.fit(snapshots, final_labels, train_mask, n_epochs=200, verbose=False)
    preds_sgnn = sgnn.predict(snapshots)
    acc_sgnn = np.mean(preds_sgnn[test_mask] == final_labels[test_mask])

    # Train static GCN on last snapshot
    gcn_last = GCN(
        snapshots[-1][0].X.shape[1], 16, n_classes, n_layers=2,
        dropout=0.3, lr=0.01, random_state=42
    )
    gcn_last.fit(snapshots[-1][0], final_labels, train_mask,
                 n_epochs=200, verbose=False)
    preds_gcn = gcn_last.predict(snapshots[-1][0])
    acc_gcn = np.mean(preds_gcn[test_mask] == final_labels[test_mask])

    # Train static GCN on first snapshot
    gcn_first = GCN(
        snapshots[0][0].X.shape[1], 16, n_classes, n_layers=2,
        dropout=0.3, lr=0.01, random_state=42
    )
    gcn_first.fit(snapshots[0][0], final_labels, train_mask,
                  n_epochs=200, verbose=False)
    preds_gcn_f = gcn_first.predict(snapshots[0][0])
    acc_gcn_f = np.mean(preds_gcn_f[test_mask] == final_labels[test_mask])

    bars = ax.bar(
        ['SnapshotGNN\n(all snapshots)', 'GCN\n(last snap)', 'GCN\n(first snap)'],
        [acc_sgnn, acc_gcn, acc_gcn_f],
        color=['#2ecc71', '#3498db', '#e67e22'],
        edgecolor='black', linewidth=0.5
    )
    for bar, val in zip(bars, [acc_sgnn, acc_gcn, acc_gcn_f]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Temporal vs Static Models')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # ---- Panel 5: Training curve ----
    ax = axes[1, 1]

    # Re-train with full loss tracking
    sgnn2 = SnapshotGNN(
        n_features=snapshots[0][0].X.shape[1],
        n_hidden=16, n_classes=n_classes,
        n_snapshots=len(snapshots), lr=0.01, random_state=42
    )
    losses = sgnn2.fit(snapshots, final_labels, train_mask,
                       n_epochs=200, verbose=False)

    # Also track GCN loss
    gcn2 = GCN(
        snapshots[-1][0].X.shape[1], 16, n_classes, n_layers=2,
        dropout=0.3, lr=0.01, random_state=42
    )
    gcn_losses = gcn2.fit(snapshots[-1][0], final_labels, train_mask,
                          n_epochs=200, verbose=False)

    ax.plot(losses, color='#2ecc71', linewidth=1.5, label='SnapshotGNN')
    ax.plot(gcn_losses, color='#3498db', linewidth=1.5, label='GCN (last)',
            linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: Lookback window effect ----
    ax = axes[1, 2]

    lookbacks = [1, 2, 3, 4, 5]
    accs_lookback = []
    for lb in lookbacks:
        subset = snapshots[-lb:]
        m = SnapshotGNN(
            n_features=snapshots[0][0].X.shape[1],
            n_hidden=16, n_classes=n_classes,
            n_snapshots=lb, lr=0.01, random_state=42
        )
        m.fit(subset, final_labels, train_mask, n_epochs=200, verbose=False)
        preds_lb = m.predict(subset)
        acc_lb = np.mean(preds_lb[test_mask] == final_labels[test_mask])
        accs_lookback.append(acc_lb)

    ax.plot(lookbacks, accs_lookback, 'o-', color='#9b59b6', linewidth=2,
            markersize=8)
    ax.set_xlabel('Lookback Window (number of snapshots)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Effect of Lookback Window')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lookbacks)

    plt.suptitle('TEMPORAL GNN -- Dynamic Graphs Over Time\n'
                 'Row 1: Graph evolution (gold rings = migrated nodes) | '
                 'Row 2: Model comparison & analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TEMPORAL GNN -- Paradigm: DYNAMIC GRAPHS")
    print("=" * 60)

    print("""
WHAT THIS MODEL IS:
    Graphs that CHANGE over time: nodes migrate between
    communities, edges appear and disappear, features drift.

    SNAPSHOT GNN (primary model):
        For each snapshot t:
            H_t = GCN(G_t, X_t)           -- spatial aggregation
        Temporal aggregation:
            Z = sum_t alpha_t * H_t        -- learned attention over time
        Output:
            Y = softmax(Z @ W_out + b_out) -- node classification

    TEMPORAL GNN (continuous-time):
        Each interaction (u, v, t) updates node memories
        via GRU with time encoding.
        Used for temporal link prediction.

KEY CONCEPTS:
    1. TIME ENCODING: Phi(dt) = [cos(w*dt), sin(w*dt), ...]
       Maps time gaps to continuous features (like pos. encoding)

    2. TEMPORAL ATTENTION: Learned weights over snapshots
       Recent snapshots usually matter more

    3. NODE MEMORY: Persistent state updated at each interaction
       Captures a node's history in a fixed-size vector

INDUCTIVE BIAS:
    - Temporal locality: recent events matter more
    - Spatial + temporal: both structure and timing matter
    - Smooth evolution: communities change gradually
    """)

    # Run ablation experiments
    ablation_experiments()

    # Benchmark
    benchmark_on_datasets()

    # Visualization
    print("\nGenerating visualizations...")
    fig = visualize_temporal_graph()
    save_path = '/Users/sid47/ML Algorithms/43_temporal_gnn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY: What Temporal GNN Reveals")
    print("=" * 60)
    print("""
1. GRAPHS CHANGE OVER TIME
   Communities evolve, nodes migrate, edges rewire.
   A single snapshot misses this temporal structure.

2. SNAPSHOT GNN CAPTURES TEMPORAL PATTERNS
   Shared GCN weights process each snapshot spatially.
   Learned temporal attention weights combine them.
   Recent snapshots typically receive more attention.

3. TEMPORAL > STATIC (when graphs evolve)
   SnapshotGNN outperforms static GCN because it sees
   the full trajectory, not just a single moment.

4. LOOKBACK WINDOW MATTERS
   Too few snapshots = missing history.
   Too many = noise from irrelevant past.
   Sweet spot depends on rate of change.

5. CONTINUOUS-TIME MODELS (TGN)
   Node memory + time encoding + GRU updates.
   Better for fine-grained event-level modeling.
   Snapshot approach is simpler and often sufficient.

CONNECTION TO OTHER FILES:
    36_gcn.py: GCN is the spatial backbone at each snapshot
    14_rnn_lstm.py: RNN/LSTM for sequential memory
    15_transformer.py: Time encoding ~ positional encoding
    35_graph_fundamentals.py: Graph class and utilities

NEXT: 44_graph_transformer.py -- Full attention on graphs!
    """)
