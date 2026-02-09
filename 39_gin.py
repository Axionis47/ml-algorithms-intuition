"""
GIN -- Graph Isomorphism Network
=================================

Paradigm: MAXIMALLY EXPRESSIVE GNN

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Make GNNs as powerful as the WEISFEILER-LEHMAN (WL) graph
isomorphism test -- the gold standard for telling graphs apart!

THE WL TEST (Iterative Label Refinement):
1. Give each node an initial label (e.g., its degree)
2. At each iteration:
       new_label = HASH(own_label, SORTED(neighbor_labels))
3. After k iterations, compare label HISTOGRAMS
4. Different histograms => graphs are NON-ISOMORPHIC

THE KEY THEOREM (Xu et al., 2019):
Among all message-passing GNNs, GIN is AS POWERFUL as the
1-WL test. No MPNN can be more powerful.

THE GIN LAYER:
    h_v^(k) = MLP((1 + eps) * h_v^(k-1) + SUM_{u in N(v)} h_u^(k-1))

    "Weight yourself by (1+eps), add SUM of neighbor features,
     then pass through MLP."

===============================================================
WHY SUM AGGREGATION? (The Crucial Insight)
===============================================================

We need an INJECTIVE multiset function -- one that maps
different multisets of neighbor features to different outputs.

Consider three neighborhoods with features:

  Neighborhood A: {1, 1, 1}    (three neighbors, all feature=1)
  Neighborhood B: {1, 2}       (two neighbors, features 1 and 2)
  Neighborhood C: {1}          (one neighbor, feature=1)

SUM aggregation:
  A -> 3,  B -> 3,  C -> 1
  Distinguishes A from C! (and B gets same as A -- not perfect
  alone, but MLP can fix this when combined with self-features)

MEAN aggregation:
  A -> 1,  B -> 1.5,  C -> 1
  A and C are IDENTICAL! Lost the count information.
  {1,1,1} and {1} look the same. FATAL for expressiveness.

MAX aggregation:
  A -> 1,  B -> 2,  C -> 1
  A and C are IDENTICAL! Lost everything except the maximum.
  {1,1,1} and {1} look the same. FATAL for expressiveness.

SUM is the ONLY standard aggregation that is INJECTIVE over
multisets (up to the expressive power of the subsequent MLP).

FORMAL RESULT (Theorem 3 in Xu et al.):
With a sufficiently expressive MLP, SUM aggregation can
approximate any function over multisets. MEAN and MAX cannot.

===============================================================
THE (1 + eps) TRICK
===============================================================

    h_v = MLP((1 + eps) * h_v + SUM h_u)

WHY (1 + eps)?
Without it, the node's OWN features are weighted the same as
each neighbor's features. This means:
    - A node with feature 3, no neighbors
    - A node with feature 0, one neighbor with feature 3
Would be INDISTINGUISHABLE!

The (1+eps) factor gives the node's own features a DIFFERENT
weight from the aggregated neighbor features, ensuring the
function is injective over (node, multiset-of-neighbors) pairs.

eps can be:
- Fixed at 0: Works fine in practice (self-loop effectively)
- Learnable: Let the network decide the weighting
- Fixed nonzero: Different emphasis on self vs neighbors

===============================================================
GRAPH-LEVEL READOUT
===============================================================

For graph classification, aggregate node embeddings to a
single graph-level representation.

SIMPLE: h_G = SUM_v h_v^(K)    (only final layer)

BETTER (GIN paper): Concatenate across ALL layers:
    h_G = CONCAT(SUM h^(0), SUM h^(1), ..., SUM h^(K))

WHY concatenate layers? Each GIN layer captures a different
level of structural information:
    Layer 1: immediate neighborhood
    Layer 2: 2-hop patterns
    Layer K: k-hop substructure

Concatenating preserves ALL levels of information.

===============================================================
GIN vs OTHER GNNs (Expressiveness Hierarchy)
===============================================================

GCN (36): Mean aggregation + degree normalization
    Less expressive: loses count information
    But: more robust to noise, well-regularized

GraphSAGE (37): Various aggregators (mean, pool, LSTM)
    Practical but not provably maximal
    Inductive: works on unseen graphs

GAT (38): Learned attention weights
    Adaptive aggregation, but still bounded by WL
    Attention is a weighted mean -- still not injective

GIN: Sum + MLP
    PROVABLY as powerful as 1-WL test
    Most expressive among ALL message-passing GNNs

===============================================================
THE WL LIMITATION (What GIN CANNOT Do)
===============================================================

The 1-WL test (and thus ALL MPNNs including GIN) CANNOT
distinguish some non-isomorphic graphs:

Example: All REGULAR graphs with same degree distribution
    - 6-node cycle vs two 3-node triangles
    - Both: 6 nodes, all degree 2
    - WL gives SAME labeling!

This is a FUNDAMENTAL limitation. Solutions:
    - Higher-order k-WL tests (k-GNNs)
    - Random node initialization features
    - Subgraph counting features
    - Graph Transformers (44_graph_transformer.py)

===============================================================
INDUCTIVE BIAS
===============================================================

1. PERMUTATION INVARIANCE (graph) / EQUIVARIANCE (node)
   Output doesn't depend on arbitrary node ordering

2. INJECTIVE AGGREGATION (SUM)
   Different neighborhoods => different outputs
   Preserves multiset structure

3. MLP EXPRESSIVENESS
   Universal approximation after aggregation
   Necessary for achieving WL-level power

4. MESSAGE PASSING LOCALITY
   Each layer sees 1-hop; k layers = k-hop
   Same over-smoothing risk as other GNNs

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
create_molecular_dataset = graph_module.create_molecular_dataset
create_transductive_split = graph_module.create_transductive_split
spring_layout = graph_module.spring_layout
draw_graph = graph_module.draw_graph

gcn_module = import_module('36_gcn')
softmax = gcn_module.softmax
cross_entropy_loss = gcn_module.cross_entropy_loss
GCN = gcn_module.GCN


# ============================================================
# GIN IMPLEMENTATION
# ============================================================

class GIN:
    """
    Graph Isomorphism Network -- Maximally Expressive GNN.

    As powerful as the 1-WL graph isomorphism test.

    Each GIN layer:
        h_v^(k) = MLP((1+eps)*h_v^(k-1) + SUM_{u in N(v)} h_u^(k-1))

    The MLP is a 2-layer network: Linear -> ReLU -> Linear

    Parameters
    ----------
    n_features : int
        Input feature dimension
    n_hidden : int
        Hidden layer dimension for each GIN layer's MLP
    n_classes : int
        Number of output classes
    n_layers : int
        Number of GIN layers (each with its own MLP)
    epsilon : float
        Initial value of eps (weights self-features)
    learn_epsilon : bool
        If True, eps is updated during training
    dropout : float
        Dropout rate applied after each GIN layer
    lr : float
        Learning rate for gradient descent
    random_state : int or None
        Random seed for reproducibility
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=3,
                 epsilon=0.0, learn_epsilon=False, dropout=0.5,
                 lr=0.01, random_state=None):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.learn_epsilon = learn_epsilon
        self.dropout = dropout
        self.lr = lr
        self.random_state = random_state

        # Per-layer epsilon values
        self.epsilons = [float(epsilon)] * n_layers

        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for each GIN layer + output layer.

        Each GIN layer has a 2-layer MLP:
            W1: (in_dim, hidden_dim), b1: (hidden_dim,)
            W2: (hidden_dim, hidden_dim), b2: (hidden_dim,)

        Output layer:
            W_out: (hidden_dim, n_classes), b_out: (n_classes,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # MLP parameters per GIN layer: (W1, b1, W2, b2)
        self.mlp_params = []

        dims_in = [self.n_features] + [self.n_hidden] * (self.n_layers - 1)

        for l in range(self.n_layers):
            d_in = dims_in[l]
            d_hid = self.n_hidden

            # Xavier initialization
            std1 = np.sqrt(2.0 / (d_in + d_hid))
            W1 = np.random.randn(d_in, d_hid) * std1
            b1 = np.zeros(d_hid)

            std2 = np.sqrt(2.0 / (d_hid + d_hid))
            W2 = np.random.randn(d_hid, d_hid) * std2
            b2 = np.zeros(d_hid)

            self.mlp_params.append((W1, b1, W2, b2))

        # Output layer
        std_out = np.sqrt(2.0 / (self.n_hidden + self.n_classes))
        self.W_out = np.random.randn(self.n_hidden, self.n_classes) * std_out
        self.b_out = np.zeros(self.n_classes)

    def forward(self, graph, training=True):
        """
        Full forward pass through all GIN layers.

        For each GIN layer l:
            1. aggregate = A @ H          (sum of neighbor features)
            2. combined = (1+eps)*H + aggregate
            3. z1 = combined @ W1 + b1    (MLP layer 1)
            4. h1 = ReLU(z1)
            5. z2 = h1 @ W2 + b2          (MLP layer 2)
            6. H_next = ReLU(z2)           (output activation)
            7. Apply dropout (if training)

        Final: logits = H_final @ W_out + b_out -> softmax

        Parameters
        ----------
        graph : Graph
            Input graph with .A and .X
        training : bool
            If True, apply dropout

        Returns
        -------
        probs : ndarray (n_nodes, n_classes)
            Class probabilities
        cache : dict
            Intermediate values for backpropagation
        """
        A = graph.A
        H = graph.X.copy()

        cache = {
            'H_inputs': [],       # H before each layer
            'aggregated': [],     # A @ H for each layer
            'combined': [],       # (1+eps)*H + aggregate
            'z1': [],             # combined @ W1 + b1
            'h1': [],             # ReLU(z1)
            'z2': [],             # h1 @ W2 + b2
            'H_outputs': [],      # ReLU(z2) (before dropout)
            'dropout_masks': [],  # dropout mask per layer
            'A': A,
        }

        for l in range(self.n_layers):
            W1, b1, W2, b2 = self.mlp_params[l]
            eps = self.epsilons[l]

            cache['H_inputs'].append(H.copy())

            # Step 1: Sum aggregation via adjacency matrix
            aggregate = A @ H

            # Step 2: Combine self and neighbors
            combined = (1.0 + eps) * H + aggregate

            cache['aggregated'].append(aggregate)
            cache['combined'].append(combined)

            # Step 3-4: MLP layer 1 + ReLU
            z1 = combined @ W1 + b1
            h1 = np.maximum(z1, 0.0)

            cache['z1'].append(z1)
            cache['h1'].append(h1)

            # Step 5-6: MLP layer 2 + ReLU
            z2 = h1 @ W2 + b2
            H_out = np.maximum(z2, 0.0)

            cache['z2'].append(z2)
            cache['H_outputs'].append(H_out.copy())

            # Step 7: Dropout
            if training and self.dropout > 0:
                mask = (np.random.rand(*H_out.shape) > self.dropout).astype(float)
                H = H_out * mask / (1.0 - self.dropout + 1e-10)
                cache['dropout_masks'].append(mask)
            else:
                H = H_out.copy()
                cache['dropout_masks'].append(np.ones_like(H_out))

        # Output layer: linear -> softmax
        logits = H @ self.W_out + self.b_out
        probs = softmax(logits)

        cache['H_final'] = H
        cache['logits'] = logits
        cache['probs'] = probs

        return probs, cache

    def backward(self, graph, y, mask, cache):
        """
        Full backpropagation through the entire GIN.

        Gradient flow (reverse order):
            softmax+CE -> output linear -> dropout -> ReLU -> MLP_l2 ->
            ReLU -> MLP_l1 -> aggregation -> previous layer

        Parameters
        ----------
        graph : Graph
            Input graph
        y : ndarray (n_nodes,)
            True labels
        mask : boolean ndarray (n_nodes,)
            Train mask
        cache : dict
            Forward pass cache
        """
        n = len(y)
        n_train = max(np.sum(mask), 1)
        A = cache['A']
        probs = cache['probs']

        # ---- Gradient of cross-entropy + softmax ----
        # dL/d_logits = (probs - one_hot) / n_train, masked
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits = dlogits * mask.astype(float)[:, None] / n_train

        # ---- Gradient through output layer ----
        H_final = cache['H_final']  # (n, hidden)
        dW_out = H_final.T @ dlogits
        db_out = np.sum(dlogits, axis=0)

        # dL/dH_final
        dH = dlogits @ self.W_out.T  # (n, hidden)

        # ---- Backprop through GIN layers (reverse) ----
        grad_mlp = []  # store (dW1, db1, dW2, db2) per layer

        for l in range(self.n_layers - 1, -1, -1):
            W1, b1, W2, b2 = self.mlp_params[l]

            # Through dropout
            dmask = cache['dropout_masks'][l]
            dH_pre_drop = dH * dmask / (1.0 - self.dropout + 1e-10)

            # Through ReLU after MLP layer 2
            z2 = cache['z2'][l]
            dz2 = dH_pre_drop * (z2 > 0).astype(float)

            # Gradients for W2, b2
            h1 = cache['h1'][l]
            dW2 = h1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            # Through MLP layer 2 linear
            dh1 = dz2 @ W2.T

            # Through ReLU after MLP layer 1
            z1 = cache['z1'][l]
            dz1 = dh1 * (z1 > 0).astype(float)

            # Gradients for W1, b1
            combined = cache['combined'][l]
            dW1 = combined.T @ dz1
            db1 = np.sum(dz1, axis=0)

            grad_mlp.insert(0, (dW1, db1, dW2, db2))

            # Through MLP layer 1 linear -> d_combined
            d_combined = dz1 @ W1.T

            # Through aggregation: combined = (1+eps)*H + A @ H
            eps = self.epsilons[l]
            H_in = cache['H_inputs'][l]

            # d_combined/dH_in = (1+eps)*I  (from self)
            # d_combined/dH_in += A^T       (from neighbor sum, transpose for backprop)
            dH = (1.0 + eps) * d_combined + A.T @ d_combined

            # Gradient for epsilon (if learnable)
            if self.learn_epsilon:
                # d_loss/d_eps = sum of (d_combined * H_in) over all elements
                d_eps = np.sum(d_combined * H_in)
                self.epsilons[l] -= self.lr * d_eps

        # ---- Gradient clipping (prevent explosions in deep nets) ----
        max_norm = 5.0
        all_grads = [dW_out, db_out]
        for dW1, db1, dW2, db2 in grad_mlp:
            all_grads.extend([dW1, db1, dW2, db2])
        total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads
                                 if np.all(np.isfinite(g))))
        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm
            dW_out = dW_out * scale
            db_out = db_out * scale
            grad_mlp = [
                (dW1 * scale, db1 * scale, dW2 * scale, db2 * scale)
                for dW1, db1, dW2, db2 in grad_mlp
            ]

        # ---- Update all weights ----
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out

        for l in range(self.n_layers):
            dW1, db1, dW2, db2 = grad_mlp[l]
            W1, b1, W2, b2 = self.mlp_params[l]
            W1 -= self.lr * dW1
            b1 -= self.lr * db1
            W2 -= self.lr * dW2
            b2 -= self.lr * db2
            self.mlp_params[l] = (W1, b1, W2, b2)

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train GIN on a graph with transductive node classification.

        Parameters
        ----------
        graph : Graph
        labels : ndarray (n_nodes,)
        train_mask : boolean ndarray (n_nodes,)
        n_epochs : int
        verbose : bool

        Returns
        -------
        loss_history : list of float
        """
        loss_history = []

        for epoch in range(n_epochs):
            # Forward
            probs, cache = self.forward(graph, training=True)

            # Compute loss
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

            # Backward
            self.backward(graph, labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                preds = np.argmax(probs, axis=1)
                train_acc = np.mean(preds[train_mask] == labels[train_mask])
                print(f"  Epoch {epoch+1:>4}: loss={loss:.4f}, "
                      f"train_acc={train_acc:.3f}")

        return loss_history

    def predict(self, graph):
        """Predict node class labels."""
        probs, _ = self.forward(graph, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, graph):
        """Predict node class probabilities."""
        probs, _ = self.forward(graph, training=False)
        return probs

    def get_embeddings(self, graph):
        """Get penultimate layer embeddings (before output layer)."""
        _, cache = self.forward(graph, training=False)
        return cache['H_final']


# ============================================================
# GIN WITH DIFFERENT AGGREGATIONS (for ablation)
# ============================================================

class GINAblation:
    """
    GIN variant that supports SUM, MEAN, or MAX aggregation.

    Used to demonstrate WHY sum aggregation is essential
    for maximal expressiveness.
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=3,
                 aggregation='sum', epsilon=0.0, dropout=0.5,
                 lr=0.01, random_state=None):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aggregation = aggregation  # 'sum', 'mean', or 'max'
        self.epsilon = epsilon
        self.dropout = dropout
        self.lr = lr
        self.random_state = random_state
        self._init_weights()

    def _init_weights(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.mlp_params = []
        dims_in = [self.n_features] + [self.n_hidden] * (self.n_layers - 1)

        for l in range(self.n_layers):
            d_in = dims_in[l]
            d_hid = self.n_hidden
            std1 = np.sqrt(2.0 / (d_in + d_hid))
            W1 = np.random.randn(d_in, d_hid) * std1
            b1 = np.zeros(d_hid)
            std2 = np.sqrt(2.0 / (d_hid + d_hid))
            W2 = np.random.randn(d_hid, d_hid) * std2
            b2 = np.zeros(d_hid)
            self.mlp_params.append((W1, b1, W2, b2))

        std_out = np.sqrt(2.0 / (self.n_hidden + self.n_classes))
        self.W_out = np.random.randn(self.n_hidden, self.n_classes) * std_out
        self.b_out = np.zeros(self.n_classes)

    def _aggregate(self, A, H):
        """Aggregate neighbor features using chosen method."""
        if self.aggregation == 'sum':
            return A @ H
        elif self.aggregation == 'mean':
            deg = np.sum(A, axis=1, keepdims=True)
            deg = np.maximum(deg, 1.0)
            return (A @ H) / deg
        elif self.aggregation == 'max':
            n = H.shape[0]
            d = H.shape[1]
            result = np.full((n, d), -np.inf)
            for i in range(n):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    result[i] = np.max(H[neighbors], axis=0)
                else:
                    result[i] = np.zeros(d)
            return result
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(self, graph, training=True):
        A = graph.A
        H = graph.X.copy()

        for l in range(self.n_layers):
            W1, b1, W2, b2 = self.mlp_params[l]

            aggregate = self._aggregate(A, H)
            combined = (1.0 + self.epsilon) * H + aggregate

            z1 = combined @ W1 + b1
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ W2 + b2
            H_out = np.maximum(z2, 0.0)

            if training and self.dropout > 0:
                mask = (np.random.rand(*H_out.shape) > self.dropout).astype(float)
                H = H_out * mask / (1.0 - self.dropout + 1e-10)
            else:
                H = H_out

        logits = H @ self.W_out + self.b_out
        probs = softmax(logits)
        return probs, H

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=False):
        """Simplified training (output layer gradient only, for ablation speed)."""
        loss_history = []

        for epoch in range(n_epochs):
            probs, H = self.forward(graph, training=True)
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

            # Gradient for output layer
            n = len(labels)
            n_train = max(np.sum(train_mask), 1)
            dlogits = probs.copy()
            dlogits[np.arange(n), labels] -= 1.0
            dlogits = dlogits * train_mask.astype(float)[:, None] / n_train

            dW_out = H.T @ dlogits
            db_out = np.sum(dlogits, axis=0)

            self.W_out -= self.lr * dW_out
            self.b_out -= self.lr * db_out

        return loss_history

    def predict(self, graph):
        probs, _ = self.forward(graph, training=False)
        return np.argmax(probs, axis=1)


# ============================================================
# WEISFEILER-LEHMAN ISOMORPHISM TEST
# ============================================================

def wl_test(graph1, graph2, iterations=5):
    """
    Weisfeiler-Lehman graph isomorphism test.

    The WL test iteratively refines node labels:
        1. Initialize: label = degree
        2. Each iteration: new_label = hash(own_label, sorted(neighbor_labels))
        3. Compare label histograms of both graphs

    If histograms differ at ANY iteration, graphs are NON-ISOMORPHIC.
    If histograms match after all iterations, graphs MIGHT be isomorphic
    (WL test is necessary but not sufficient).

    Parameters
    ----------
    graph1, graph2 : Graph
    iterations : int

    Returns
    -------
    distinguishable : bool
        True if WL test can distinguish the graphs (definitely non-isomorphic)
    history : list of (hist1, hist2) per iteration
    """
    def get_labels_and_hist(graph, labels):
        """One WL refinement step."""
        new_labels = []
        for v in range(graph.n_nodes):
            nbr_labels = sorted([labels[u] for u in graph.neighbors(v)])
            # Create a hashable representation
            new_labels.append(hash((labels[v], tuple(nbr_labels))))
        # Relabel to consecutive integers
        unique = sorted(set(new_labels))
        label_map = {l: i for i, l in enumerate(unique)}
        new_labels = [label_map[l] for l in new_labels]
        # Histogram
        hist = {}
        for l in new_labels:
            hist[l] = hist.get(l, 0) + 1
        return new_labels, hist

    # Initialize labels from degree
    labels1 = list(graph1.degrees().astype(int))
    labels2 = list(graph2.degrees().astype(int))

    history = []

    for it in range(iterations):
        labels1, hist1 = get_labels_and_hist(graph1, labels1)
        labels2, hist2 = get_labels_and_hist(graph2, labels2)

        # Normalize histograms to comparable form
        # Use sorted (label, count) tuples for comparison
        sig1 = tuple(sorted(hist1.values()))
        sig2 = tuple(sorted(hist2.values()))

        history.append((hist1, hist2))

        if sig1 != sig2:
            return True, history  # Distinguishable!

    return False, history  # Cannot distinguish


def wl_test_with_labels(graph1, graph2, iterations=5):
    """
    WL test that also returns the per-iteration label arrays
    for visualization purposes.

    Returns
    -------
    distinguishable : bool
    label_history : list of (labels1, labels2) per iteration
    """
    labels1 = list(graph1.degrees().astype(int))
    labels2 = list(graph2.degrees().astype(int))

    label_history = [(list(labels1), list(labels2))]

    for it in range(iterations):
        new1, new2 = [], []
        for v in range(graph1.n_nodes):
            nbr = sorted([labels1[u] for u in graph1.neighbors(v)])
            new1.append(hash((labels1[v], tuple(nbr))))
        for v in range(graph2.n_nodes):
            nbr = sorted([labels2[u] for u in graph2.neighbors(v)])
            new2.append(hash((labels2[v], tuple(nbr))))

        # Relabel to consecutive integers (globally across both graphs)
        all_labels = sorted(set(new1 + new2))
        lmap = {l: i for i, l in enumerate(all_labels)}
        labels1 = [lmap[l] for l in new1]
        labels2 = [lmap[l] for l in new2]

        label_history.append((list(labels1), list(labels2)))

        hist1 = {}
        for l in labels1:
            hist1[l] = hist1.get(l, 0) + 1
        hist2 = {}
        for l in labels2:
            hist2[l] = hist2.get(l, 0) + 1

        if tuple(sorted(hist1.values())) != tuple(sorted(hist2.values())):
            return True, label_history

    return False, label_history


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run comprehensive ablation experiments for GIN."""

    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    np.random.seed(42)

    # ---- Experiment 1: SUM vs MEAN vs MAX Multiset Test ----
    print("\n1. SUM vs MEAN vs MAX -- MULTISET EXPRESSIVENESS")
    print("-" * 50)
    print("Can the aggregation DISTINGUISH different neighbor sets?\n")

    # These test cases are chosen to highlight WHERE each aggregation
    # fails. The key insight: MEAN loses count, MAX loses everything
    # but the maximum. SUM preserves the full multiset information
    # (when followed by an MLP).
    test_cases = [
        # MEAN fails (same mean, different multisets):
        ("MEAN fail: {1,1,1} vs {1}", [1, 1, 1], [1]),
        ("MEAN fail: {2,2} vs {1,1,2,2}", [2, 2], [1, 1, 2, 2]),
        # MAX fails (same max, different multisets):
        ("MAX fail: {1,2,3} vs {3}", [1, 2, 3], [3]),
        ("MAX fail: {0,0,5} vs {5}", [0, 0, 5], [5]),
        # Both MEAN and MAX fail:
        ("Both fail: {1,1} vs {1}", [1, 1], [1]),
        # SUM correctly distinguishes ALL of the above
    ]

    print(f"  {'Test Case':<32} {'SUM':<15} {'MEAN':<15} {'MAX':<15}")
    print("  " + "-" * 77)

    sum_wins, mean_wins, max_wins = 0, 0, 0
    for name, a, b in test_cases:
        s1, s2 = sum(a), sum(b)
        m1, m2 = np.mean(a), np.mean(b)
        x1, x2 = max(a), max(b)

        s_diff = "DIFF" if abs(s1 - s2) > 1e-8 else "SAME"
        m_diff = "DIFF" if abs(m1 - m2) > 1e-8 else "SAME"
        x_diff = "DIFF" if abs(x1 - x2) > 1e-8 else "SAME"

        if s_diff == "DIFF":
            sum_wins += 1
        if m_diff == "DIFF":
            mean_wins += 1
        if x_diff == "DIFF":
            max_wins += 1

        print(f"  {name:<32} {s1}vs{s2} {s_diff:<6}  "
              f"{m1:.1f}vs{m2:.1f} {m_diff:<6}  "
              f"{x1}vs{x2} {x_diff:<6}")

    print(f"\n  Distinctions: SUM={sum_wins}/{len(test_cases)}  "
          f"MEAN={mean_wins}/{len(test_cases)}  "
          f"MAX={max_wins}/{len(test_cases)}")
    print("  => SUM is most expressive (INJECTIVE over multisets)")

    # ---- Experiment 1b: SUM vs MEAN vs MAX on Node Classification ----
    print("\n  Node classification accuracy with each aggregation:")
    print("  " + "-" * 50)

    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=25,
        p_in=0.3, p_out=0.02, feature_dim=16, random_state=42
    )
    train_mask, _, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    for agg in ['sum', 'mean', 'max']:
        accs = []
        for trial in range(3):
            model = GINAblation(
                n_features=16, n_hidden=16, n_classes=3, n_layers=2,
                aggregation=agg, dropout=0.3, lr=0.01,
                random_state=42 + trial
            )
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))
        print(f"    {agg.upper():<6}  test_acc={np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}")

    print("  => SUM typically matches or exceeds MEAN/MAX")

    # ---- Experiment 2: Learnable eps vs Fixed eps ----
    print("\n2. LEARNABLE eps vs FIXED eps=0")
    print("-" * 50)

    graph, labels = karate_club()
    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 3, 30, 31, 32, 33]] = True

    for learn_eps in [False, True]:
        accs = []
        for trial in range(5):
            model = GIN(
                n_features=graph.X.shape[1], n_hidden=16, n_classes=2,
                n_layers=2, epsilon=0.0, learn_epsilon=learn_eps,
                dropout=0.3, lr=0.01, random_state=42 + trial
            )
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds == labels))

        label = "Learnable eps" if learn_eps else "Fixed eps=0  "
        print(f"  {label}  accuracy={np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}")
        if learn_eps:
            print(f"    Final eps values: "
                  f"{[f'{e:.3f}' for e in model.epsilons]}")

    print("  => eps=0 often works well; learnable eps can help")

    # ---- Experiment 3: Number of Layers ----
    print("\n3. NUMBER OF GIN LAYERS (Over-Smoothing Check)")
    print("-" * 50)

    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=25,
        p_in=0.3, p_out=0.02, feature_dim=16, random_state=42
    )
    train_mask, _, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    for n_layers in [1, 2, 3, 4, 5]:
        accs = []
        for trial in range(3):
            model = GIN(
                n_features=16, n_hidden=16, n_classes=3,
                n_layers=n_layers, dropout=0.3, lr=0.01,
                random_state=42 + trial
            )
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))

        # Check over-smoothing via embedding similarity
        emb = model.get_embeddings(graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_n = emb / norms
        cos_sim = emb_n @ emb_n.T
        avg_sim = (np.sum(cos_sim) - graph.n_nodes) / (
            graph.n_nodes * (graph.n_nodes - 1))

        print(f"  layers={n_layers}  test_acc={np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}  avg_cos_sim={avg_sim:.3f}")

    print("  => 2-3 layers optimal; deep GIN over-smooths like GCN")

    # ---- Experiment 4: Hidden Dimension ----
    print("\n4. HIDDEN DIMENSION SWEEP")
    print("-" * 50)

    for hidden in [8, 16, 32, 64]:
        accs = []
        for trial in range(3):
            model = GIN(
                n_features=16, n_hidden=hidden, n_classes=3,
                n_layers=2, dropout=0.3, lr=0.01,
                random_state=42 + trial
            )
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            accs.append(np.mean(preds[test_mask] == labels[test_mask]))

        n_params = sum(
            w.size for params in model.mlp_params
            for w in params
        ) + model.W_out.size + model.b_out.size
        print(f"  hidden={hidden:<4}  test_acc={np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}  params={n_params}")

    print("  => 16-32 usually sufficient for small graphs")

    # ---- Experiment 5: WL Test Demo ----
    print("\n5. WEISFEILER-LEHMAN TEST DEMO")
    print("-" * 50)

    # Case A: Non-isomorphic graphs WL CAN distinguish
    # Triangle vs path of length 3
    g1 = Graph(3)
    g1.add_edge(0, 1)
    g1.add_edge(1, 2)
    g1.add_edge(0, 2)

    g2 = Graph(3)
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)

    dist_a, hist_a = wl_test(g1, g2, iterations=3)
    print(f"  Triangle vs Path(3 nodes):")
    print(f"    Distinguishable: {dist_a}")
    print(f"    (Triangle: all degree 2; Path: degrees 1,2,1)")

    # Case B: Non-isomorphic graphs WL CANNOT distinguish
    # Two non-isomorphic 3-regular graphs on 6 nodes
    # Graph 1: 6-cycle (C6)
    g3 = Graph(6)
    for i in range(6):
        g3.add_edge(i, (i + 1) % 6)
    # Add cross edges to make 3-regular: 0-3, 1-4, 2-5
    g3.add_edge(0, 3)
    g3.add_edge(1, 4)
    g3.add_edge(2, 5)

    # Graph 2: Two triangles connected (K3,3-like)
    g4 = Graph(6)
    # Triangle 1: 0-1-2
    g4.add_edge(0, 1)
    g4.add_edge(1, 2)
    g4.add_edge(0, 2)
    # Triangle 2: 3-4-5
    g4.add_edge(3, 4)
    g4.add_edge(4, 5)
    g4.add_edge(3, 5)
    # Cross connections to make 3-regular: 0-3, 1-4, 2-5
    g4.add_edge(0, 3)
    g4.add_edge(1, 4)
    g4.add_edge(2, 5)

    # Check that both are 3-regular
    deg3 = g3.degrees()
    deg4 = g4.degrees()

    # WL test
    dist_b, hist_b = wl_test(g3, g4, iterations=5)
    print(f"\n  3-regular graph A vs 3-regular graph B (6 nodes each):")
    print(f"    Graph A degrees: {deg3}")
    print(f"    Graph B degrees: {deg4}")
    print(f"    Distinguishable by WL: {dist_b}")
    if not dist_b:
        print("    => WL (and GIN) CANNOT distinguish these!")
        print("    This is the FUNDAMENTAL LIMIT of message-passing GNNs.")
    else:
        print("    => WL CAN distinguish these specific structures.")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_on_datasets():
    """Benchmark GIN on multiple graph datasets."""

    print("\n" + "=" * 60)
    print("BENCHMARK: GIN on Graph Datasets")
    print("=" * 60)

    results = {}

    datasets = {
        'karate_club': (karate_club, 2),
        'community_2': (lambda: create_community_graph(
            2, 25, 0.3, 0.02, random_state=42), 2),
        'community_3': (lambda: create_community_graph(
            3, 20, 0.3, 0.02, random_state=42), 3),
        'community_4': (lambda: create_community_graph(
            4, 15, 0.3, 0.02, random_state=42), 4),
        'citation': (lambda: create_citation_network(
            100, 3, 16, random_state=42), 3),
    }

    print(f"\n{'Dataset':<15} {'GIN Train':<12} {'GIN Test':<12} "
          f"{'GCN Test':<12} {'Nodes':<8}")
    print("-" * 60)

    for name, (dataset_fn, n_classes) in datasets.items():
        graph, labels = dataset_fn()
        train_mask, val_mask, test_mask = create_transductive_split(
            graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
        )

        # GIN
        gin = GIN(
            n_features=graph.X.shape[1], n_hidden=16, n_classes=n_classes,
            n_layers=2, dropout=0.3, lr=0.01, random_state=42
        )
        gin.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        gin_preds = gin.predict(graph)
        gin_train = np.mean(gin_preds[train_mask] == labels[train_mask])
        gin_test = np.mean(gin_preds[test_mask] == labels[test_mask])

        # GCN for comparison
        gcn = GCN(
            n_features=graph.X.shape[1], n_hidden=16, n_classes=n_classes,
            n_layers=2, dropout=0.3, lr=0.01, random_state=42
        )
        gcn.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        gcn_preds = gcn.predict(graph)
        gcn_test = np.mean(gcn_preds[test_mask] == labels[test_mask])

        results[name] = {
            'gin_train': gin_train, 'gin_test': gin_test,
            'gcn_test': gcn_test, 'n_nodes': graph.n_nodes
        }

        print(f"{name:<15} {gin_train:<12.3f} {gin_test:<12.3f} "
              f"{gcn_test:<12.3f} {graph.n_nodes:<8}")

    print("\n  GIN uses SUM aggregation (injective) vs GCN's normalized mean.")
    print("  On homophilic graphs, both perform similarly.")
    print("  GIN's advantage is theoretical: provably maximal expressiveness.")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_gin():
    """
    Create comprehensive 2x3 GIN visualization.

    Panel 1: GIN predictions on karate club graph
    Panel 2: SUM vs MEAN vs MAX expressiveness (bar chart)
    Panel 3: Layer depth vs accuracy
    Panel 4: Training loss curve
    Panel 5: GIN vs GCN comparison
    Panel 6: Summary text

    Returns fig, saves as 39_gin.png.
    """
    print("\nGenerating: GIN visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    np.random.seed(42)

    # Load karate club
    graph, labels = karate_club()
    n_classes = 2
    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 3, 30, 31, 32, 33]] = True
    test_mask = ~train_mask
    pos = spring_layout(graph, seed=42)

    # ============ Panel 1: GIN Node Classification ============
    ax = axes[0, 0]

    gin = GIN(
        n_features=graph.X.shape[1], n_hidden=16, n_classes=2,
        n_layers=2, dropout=0.3, lr=0.01, random_state=42
    )
    losses = gin.fit(graph, labels, train_mask, n_epochs=300, verbose=False)
    preds = gin.predict(graph)
    acc = np.mean(preds == labels)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])

    # Draw edges
    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            if graph.A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        'gray', alpha=0.2, linewidth=0.5)

    # Draw nodes: correct=circle, wrong=X
    correct = preds == labels
    colors_map = {0: '#e74c3c', 1: '#3498db'}
    for i in range(graph.n_nodes):
        c = colors_map[labels[i]]
        marker = 'o' if correct[i] else 'X'
        size = 80 if correct[i] else 120
        ax.scatter(pos[i, 0], pos[i, 1], c=c, s=size,
                   marker=marker, edgecolors='black', linewidths=0.5,
                   zorder=5)

    ax.set_title(f'GIN Node Classification\n'
                 f'Accuracy: {acc:.0%} (test: {test_acc:.0%})')
    ax.axis('off')

    # ============ Panel 2: SUM vs MEAN vs MAX ============
    ax = axes[0, 1]

    # Run multiset distinguishability test -- same cases as ablation
    test_cases = [
        ("{1,1,1} vs {1}", [1, 1, 1], [1]),
        ("{2,2} vs {1,1,2,2}", [2, 2], [1, 1, 2, 2]),
        ("{1,2,3} vs {3}", [1, 2, 3], [3]),
        ("{0,0,5} vs {5}", [0, 0, 5], [5]),
        ("{1,1} vs {1}", [1, 1], [1]),
    ]

    sum_count, mean_count, max_count = 0, 0, 0
    for _, a, b in test_cases:
        if abs(sum(a) - sum(b)) > 1e-8:
            sum_count += 1
        if abs(np.mean(a) - np.mean(b)) > 1e-8:
            mean_count += 1
        if abs(max(a) - max(b)) > 1e-8:
            max_count += 1

    agg_names = ['SUM', 'MEAN', 'MAX']
    counts = [sum_count, mean_count, max_count]
    colors = ['#2ecc71', '#e67e22', '#e74c3c']

    bars = ax.bar(agg_names, counts, color=colors, edgecolor='black',
                  linewidth=0.5)
    ax.set_ylabel('Multisets Distinguished')
    ax.set_title(f'Aggregation Expressiveness\n'
                 f'({len(test_cases)} multiset pair tests)')
    ax.set_ylim(0, len(test_cases) + 0.5)

    # Annotations
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{count}/{len(test_cases)}', ha='center', fontsize=10,
                fontweight='bold')

    ax.text(0, -0.8, 'Injective', ha='center', fontsize=8, color='#2ecc71',
            fontweight='bold')
    ax.text(1, -0.8, 'Loses count', ha='center', fontsize=8,
            color='#e67e22')
    ax.text(2, -0.8, 'Loses all\nbut max', ha='center', fontsize=8,
            color='#e74c3c')

    # ============ Panel 3: Layer Depth vs Accuracy ============
    ax = axes[0, 2]

    comm_graph, comm_labels = create_community_graph(
        n_communities=3, nodes_per_community=25,
        p_in=0.3, p_out=0.02, feature_dim=16, random_state=42
    )
    tm, _, tsm = create_transductive_split(
        comm_graph.n_nodes, comm_labels, train_ratio=0.15, val_ratio=0.1
    )

    layer_counts = [1, 2, 3, 4, 5]
    layer_accs = []
    layer_sims = []

    for nl in layer_counts:
        accs_trial = []
        for trial in range(3):
            m = GIN(
                n_features=16, n_hidden=16, n_classes=3,
                n_layers=nl, dropout=0.3, lr=0.01,
                random_state=42 + trial
            )
            m.fit(comm_graph, comm_labels, tm, n_epochs=200, verbose=False)
            p = m.predict(comm_graph)
            accs_trial.append(np.mean(p[tsm] == comm_labels[tsm]))

        layer_accs.append(np.mean(accs_trial))

        # Over-smoothing measure
        emb = m.get_embeddings(comm_graph)
        norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        emb_n = emb / norms
        cs = emb_n @ emb_n.T
        avg_s = (np.sum(cs) - comm_graph.n_nodes) / (
            comm_graph.n_nodes * (comm_graph.n_nodes - 1))
        layer_sims.append(avg_s)

    ax.plot(layer_counts, layer_accs, 'bo-', linewidth=2, markersize=8,
            label='Test Accuracy')
    ax.set_xlabel('Number of GIN Layers')
    ax.set_ylabel('Test Accuracy', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    ax2 = ax.twinx()
    ax2.plot(layer_counts, layer_sims, 'r^--', linewidth=1.5, markersize=7,
             label='Avg Cos Sim', alpha=0.7)
    ax2.set_ylabel('Avg Cosine Similarity', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_title('Layer Depth: Accuracy vs Over-Smoothing')
    ax.set_xticks(layer_counts)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right',
              fontsize=8)
    ax.grid(True, alpha=0.3)

    # ============ Panel 4: Training Loss Curve ============
    ax = axes[1, 0]

    gin2 = GIN(
        n_features=graph.X.shape[1], n_hidden=16, n_classes=2,
        n_layers=2, dropout=0.3, lr=0.01, random_state=42
    )
    losses2 = gin2.fit(graph, labels, train_mask, n_epochs=300, verbose=False)

    ax.plot(losses2, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('GIN Training Loss\n(Karate Club)')
    ax.grid(True, alpha=0.3)

    # Mark convergence
    min_loss = min(losses2)
    min_epoch = losses2.index(min_loss)
    ax.axhline(y=min_loss, color='red', linestyle='--', alpha=0.4,
               label=f'Min loss: {min_loss:.3f}')
    ax.legend(fontsize=8)

    # ============ Panel 5: GIN vs GCN Comparison ============
    ax = axes[1, 1]

    dataset_names = ['karate', 'comm_3', 'citation']
    dataset_fns = [
        karate_club,
        lambda: create_community_graph(3, 20, 0.3, 0.02, random_state=42),
        lambda: create_citation_network(100, 3, 16, random_state=42),
    ]
    dataset_classes = [2, 3, 3]

    gin_accs, gcn_accs = [], []

    for ds_fn, nc in zip(dataset_fns, dataset_classes):
        g, l = ds_fn()
        tmk, _, tsmk = create_transductive_split(
            g.n_nodes, l, train_ratio=0.15, val_ratio=0.1
        )

        # GIN
        gin_m = GIN(
            n_features=g.X.shape[1], n_hidden=16, n_classes=nc,
            n_layers=2, dropout=0.3, lr=0.01, random_state=42
        )
        gin_m.fit(g, l, tmk, n_epochs=200, verbose=False)
        gin_p = gin_m.predict(g)
        gin_accs.append(np.mean(gin_p[tsmk] == l[tsmk]))

        # GCN
        gcn_m = GCN(
            n_features=g.X.shape[1], n_hidden=16, n_classes=nc,
            n_layers=2, dropout=0.3, lr=0.01, random_state=42
        )
        gcn_m.fit(g, l, tmk, n_epochs=200, verbose=False)
        gcn_p = gcn_m.predict(g)
        gcn_accs.append(np.mean(gcn_p[tsmk] == l[tsmk]))

    x = np.arange(len(dataset_names))
    width = 0.35
    ax.bar(x - width / 2, gin_accs, width, label='GIN (SUM)',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, gcn_accs, width, label='GCN (Mean)',
           color='#3498db', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('GIN vs GCN\n(SUM vs normalized MEAN aggregation)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    # Add value labels
    for i, (ga, ca) in enumerate(zip(gin_accs, gcn_accs)):
        ax.text(i - width / 2, ga + 0.02, f'{ga:.2f}', ha='center',
                fontsize=8)
        ax.text(i + width / 2, ca + 0.02, f'{ca:.2f}', ha='center',
                fontsize=8)

    # ============ Panel 6: Summary Text ============
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
  GIN -- Graph Isomorphism Network

  THE KEY IDEA:
  Be as powerful as the WL test!

  h_v = MLP((1+eps)*h_v + SUM h_u)

  WHY SUM?
  +------------------------------+
  | SUM: Injective!              |
  | {1,1,1}->3, {1}->1    DIFF  |
  +------------------------------+
  | MEAN: Loses count            |
  | {1,1,1}->1, {1}->1    SAME  |
  +------------------------------+
  | MAX: Loses all but max       |
  | {1,2,3}->3, {3}->3    SAME  |
  +------------------------------+

  THEOREM (Xu et al., 2019):
  GIN is as powerful as 1-WL test
  (provably maximal for MPNNs)

  LIMITATION:
  WL can't distinguish ALL graphs
  (e.g., same-degree regular graphs)

  ARCHITECTURE:
  SUM agg -> 2-layer MLP -> ReLU
  (per GIN layer, stacked K deep)
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))

    plt.suptitle('GIN -- Graph Isomorphism Network\n'
                 'Maximally expressive message-passing GNN '
                 '(as powerful as 1-WL test)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def visualize_wl_test():
    """
    Visualize the Weisfeiler-Lehman test.

    Panel 1: Two non-isomorphic graphs that WL CAN distinguish
             (triangle vs path) -- show label refinement
    Panel 2: Two graphs that WL CANNOT distinguish
             (regular graphs with same degree sequence)

    Returns fig, saves as 39_gin_wl.png.
    """
    print("\nGenerating: WL test visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ============ Panel 1: WL CAN distinguish ============
    ax = axes[0]

    # Graph A: Triangle (K3) -- 4 nodes, triangle + pendant
    g1 = Graph(4)
    g1.add_edge(0, 1)
    g1.add_edge(1, 2)
    g1.add_edge(0, 2)
    g1.add_edge(2, 3)

    # Graph B: Path (P4)
    g2 = Graph(4)
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)
    g2.add_edge(2, 3)

    # Run WL with label tracking
    dist, label_hist = wl_test_with_labels(g1, g2, iterations=3)

    # Positions for side-by-side display
    # Graph A on left, Graph B on right
    pos1 = np.array([[0, 1], [1, 1], [0.5, 0], [1.5, 0]])
    pos2 = np.array([[3, 1], [4, 1], [5, 1], [6, 1]])

    # Draw edges
    for i in range(g1.n_nodes):
        for j in range(i + 1, g1.n_nodes):
            if g1.A[i, j] > 0:
                ax.plot([pos1[i, 0], pos1[j, 0]], [pos1[i, 1], pos1[j, 1]],
                        'k-', linewidth=2, alpha=0.5)
    for i in range(g2.n_nodes):
        for j in range(i + 1, g2.n_nodes):
            if g2.A[i, j] > 0:
                ax.plot([pos2[i, 0], pos2[j, 0]], [pos2[i, 1], pos2[j, 1]],
                        'k-', linewidth=2, alpha=0.5)

    # Color by initial labels (degree)
    cmap = plt.cm.Set1
    init_labels1 = label_hist[0][0]
    init_labels2 = label_hist[0][1]

    # After 1 iteration
    final_labels1 = label_hist[1][0]
    final_labels2 = label_hist[1][1]
    all_labels = sorted(set(final_labels1 + final_labels2))
    n_colors = len(all_labels)
    color_map = {l: cmap(i / max(n_colors, 1)) for i, l in
                 enumerate(all_labels)}

    for i in range(g1.n_nodes):
        c = color_map.get(final_labels1[i], 'gray')
        ax.scatter(pos1[i, 0], pos1[i, 1], c=[c], s=300,
                   edgecolors='black', linewidths=2, zorder=5)
        ax.text(pos1[i, 0], pos1[i, 1], str(final_labels1[i]),
                ha='center', va='center', fontsize=12, fontweight='bold',
                zorder=6)

    for i in range(g2.n_nodes):
        c = color_map.get(final_labels2[i], 'gray')
        ax.scatter(pos2[i, 0], pos2[i, 1], c=[c], s=300,
                   edgecolors='black', linewidths=2, zorder=5)
        ax.text(pos2[i, 0], pos2[i, 1], str(final_labels2[i]),
                ha='center', va='center', fontsize=12, fontweight='bold',
                zorder=6)

    ax.text(0.75, -0.8, 'Graph A\n(triangle + pendant)', ha='center',
            fontsize=10, fontweight='bold')
    ax.text(4.5, -0.8, 'Graph B\n(path P4)', ha='center',
            fontsize=10, fontweight='bold')

    # Show histograms
    hist1 = {}
    for l in final_labels1:
        hist1[l] = hist1.get(l, 0) + 1
    hist2 = {}
    for l in final_labels2:
        hist2[l] = hist2.get(l, 0) + 1

    ax.text(3.25, 1.8,
            f'Label histograms (after 1 WL iteration):\n'
            f'  A: {dict(sorted(hist1.items()))}\n'
            f'  B: {dict(sorted(hist2.items()))}\n'
            f'  => {"DIFFERENT! Non-isomorphic." if dist else "Same"}',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_title('WL Test CAN Distinguish These Graphs\n'
                 '(different degree distributions lead to different labels)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ============ Panel 2: WL CANNOT distinguish ============
    ax = axes[1]

    # Create two 3-regular graphs on 6 nodes
    # Graph C: Prism graph (triangular prism)
    g3 = Graph(6)
    # Top triangle
    g3.add_edge(0, 1)
    g3.add_edge(1, 2)
    g3.add_edge(0, 2)
    # Bottom triangle
    g3.add_edge(3, 4)
    g3.add_edge(4, 5)
    g3.add_edge(3, 5)
    # Vertical edges
    g3.add_edge(0, 3)
    g3.add_edge(1, 4)
    g3.add_edge(2, 5)

    # Graph D: Octahedron minus matching (also 3-regular, 6 nodes)
    # K_{3,3} -- complete bipartite
    g4 = Graph(6)
    for i in [0, 1, 2]:
        for j in [3, 4, 5]:
            g4.add_edge(i, j)

    dist2, label_hist2 = wl_test_with_labels(g3, g4, iterations=5)

    # Positions
    pos3 = np.array([
        [0, 1.2], [1, 1.2], [0.5, 0.3],
        [0, -0.8], [1, -0.8], [0.5, -1.7]
    ])
    pos4 = np.array([
        [3.5, 1.0], [4.5, 1.0], [5.5, 1.0],
        [3.5, -1.0], [4.5, -1.0], [5.5, -1.0]
    ])

    # Draw edges
    for i in range(g3.n_nodes):
        for j in range(i + 1, g3.n_nodes):
            if g3.A[i, j] > 0:
                ax.plot([pos3[i, 0], pos3[j, 0]], [pos3[i, 1], pos3[j, 1]],
                        'k-', linewidth=2, alpha=0.5)
    for i in range(g4.n_nodes):
        for j in range(i + 1, g4.n_nodes):
            if g4.A[i, j] > 0:
                ax.plot([pos4[i, 0], pos4[j, 0]], [pos4[i, 1], pos4[j, 1]],
                        'k-', linewidth=2, alpha=0.5)

    # All nodes same color (WL assigns same labels)
    for i in range(g3.n_nodes):
        ax.scatter(pos3[i, 0], pos3[i, 1], c='#f39c12', s=300,
                   edgecolors='black', linewidths=2, zorder=5)
        deg = int(g3.degrees()[i])
        ax.text(pos3[i, 0], pos3[i, 1], str(deg), ha='center',
                va='center', fontsize=12, fontweight='bold', zorder=6)

    for i in range(g4.n_nodes):
        ax.scatter(pos4[i, 0], pos4[i, 1], c='#f39c12', s=300,
                   edgecolors='black', linewidths=2, zorder=5)
        deg = int(g4.degrees()[i])
        ax.text(pos4[i, 0], pos4[i, 1], str(deg), ha='center',
                va='center', fontsize=12, fontweight='bold', zorder=6)

    ax.text(0.5, -2.5, 'Graph C: Prism\n(3-regular)', ha='center',
            fontsize=10, fontweight='bold')
    ax.text(4.5, -2.5, 'Graph D: K_{3,3}\n(3-regular)', ha='center',
            fontsize=10, fontweight='bold')

    result_text = (
        'WL test result:\n'
        '  All nodes degree 3 => same initial label\n'
        '  After refinement: all neighbors also degree 3\n'
        f'  Distinguishable: {dist2}\n'
    )
    if not dist2:
        result_text += '  => WL (and GIN) CANNOT tell these apart!\n'
        result_text += '  This is the fundamental MPNN limitation.'
    else:
        result_text += '  => WL can distinguish these structures.'

    ax.text(2.75, 2.0, result_text, fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))

    ax.set_title('WL Test CANNOT Distinguish Regular Graphs\n'
                 '(same degree => same labels at every iteration)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-3.2, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('Weisfeiler-Lehman Graph Isomorphism Test\n'
                 'The theoretical foundation (and limitation) of GIN',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GIN -- Graph Isomorphism Network")
    print("Paradigm: MAXIMALLY EXPRESSIVE GNN")
    print("=" * 60)

    print("""
WHAT THIS MODEL IS:
    Each GIN layer:
        h_v = MLP((1+eps)*h_v + SUM_{u in N(v)} h_u)

    WHERE:
        SUM  = neighbor aggregation (INJECTIVE over multisets)
        MLP  = 2-layer neural network (universal approximator)
        eps  = self-weight parameter (fixed or learnable)

    GRAPH-LEVEL READOUT:
        h_G = CONCAT(SUM h^(0), SUM h^(1), ..., SUM h^(K))

KEY RESULT (Xu et al., 2019):
    GIN is as powerful as the 1-WL graph isomorphism test.
    No message-passing GNN can be more expressive.

WHY SUM (not MEAN or MAX)?
    SUM is INJECTIVE over multisets:
        {1,1,1} -> 3, {1} -> 1  (DIFFERENT)
    MEAN loses count:
        {1,1,1} -> 1, {1} -> 1  (SAME -- bad!)
    MAX loses everything but max:
        {1,2,3} -> 3, {3} -> 3  (SAME -- bad!)

INDUCTIVE BIAS:
    1. Permutation invariance (graph) / equivariance (node)
    2. Injective aggregation (SUM preserves multiset)
    3. MLP for universal approximation
    4. Local message passing: k layers = k-hop receptive field
    """)

    # Run ablation experiments
    ablation_experiments()

    # Run benchmark
    results = benchmark_on_datasets()

    # Generate visualizations
    print("\nGenerating visualizations...")

    fig1 = visualize_gin()
    save_path1 = '/Users/sid47/ML Algorithms/39_gin.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_wl_test()
    save_path2 = '/Users/sid47/ML Algorithms/39_gin_wl.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("SUMMARY: What GIN Reveals")
    print("=" * 60)
    print("""
1. GIN IS PROVABLY MAXIMAL
   As powerful as 1-WL test among ALL message-passing GNNs.
   No MPNN can do better.

2. SUM AGGREGATION IS THE KEY
   SUM is injective over multisets.
   MEAN/MAX lose information => strictly less expressive.

3. THE (1+eps) TRICK
   Distinguishes node's own features from neighbor sum.
   eps=0 works fine in practice; learnable eps can help.

4. MLP AFTER AGGREGATION
   Universal approximation is necessary for injectivity.
   Simple linear layers would NOT achieve WL power.

5. SAME LIMITATIONS AS ALL MPNNs
   Cannot distinguish regular graphs with same degree.
   This is the fundamental 1-WL ceiling.
   Solutions: higher-order GNNs, graph transformers.

6. PRACTICAL TRADE-OFFS
   GIN is most expressive but not always most accurate.
   GCN's smoothing can be more robust on noisy/homophilic graphs.
   Choose based on whether you need fine structural discrimination.

CONNECTION TO OTHER FILES:
   36_gcn.py:  Normalized mean aggregation (less expressive)
   37_graphsage.py: Various aggregators (not provably maximal)
   38_gat.py:  Attention-weighted mean (adaptive but still bounded)
   40_mpnn.py: Unified framework that encompasses GIN
   44_graph_transformer.py: Full attention can go beyond WL
    """)
