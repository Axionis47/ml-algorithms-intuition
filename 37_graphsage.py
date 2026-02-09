"""
GraphSAGE — Paradigm: INDUCTIVE AGGREGATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

GraphSAGE = Graph SAmple and aggreGatE (Hamilton et al., 2017)

Instead of fixed graph convolution like GCN, LEARN an aggregation
function from SAMPLED neighbors.

For each node v, at each layer k:

    1. SAMPLE: Pick k neighbors of v (not all!)
    2. AGGREGATE: Combine neighbor features
       h_N(v)^(k) = AGGREGATE({h_u^(k-1) : u in SAMPLE(N(v))})
    3. CONCAT: Combine self with aggregated neighbors
       h_v^(k) = sigma(W^(k) . CONCAT(h_v^(k-1), h_N(v)^(k)))
    4. NORMALIZE: L2-normalize the embedding
       h_v^(k) = h_v^(k) / ||h_v^(k)||

The CONCAT is the key difference from GCN:
    GCN: smooths self INTO neighbors (weighted average)
    GraphSAGE: EXPLICITLY separates self from neighbors

===============================================================
WHY GRAPHSAGE > GCN
===============================================================

1. INDUCTIVE: GCN learns weights tied to graph structure.
   GraphSAGE learns a GENERAL aggregation function.
   -> Can process UNSEEN nodes and UNSEEN graphs!

2. SCALABLE: GCN uses ALL neighbors (full A_hat multiplication).
   GraphSAGE SAMPLES k neighbors per node per layer.
   -> Constant computation per node, regardless of degree.

3. FLEXIBLE: GCN uses fixed degree-weighted mean.
   GraphSAGE supports multiple aggregator types.
   -> Mean, pool, max -- each with different strengths.

===============================================================
AGGREGATORS
===============================================================

1. MEAN: Simple average of neighbor features
   h_N(v) = mean({h_u : u in S(N(v))})
   Equivalent to GCN's normalized mean (but with sampling).

2. POOL: Apply MLP, then element-wise max-pool
   h_N(v) = max({sigma(W_pool h_u + b_pool) : u in S(N(v))})
   More expressive: nonlinear transformation before aggregation.

3. MAX: Element-wise maximum of neighbor features
   h_N(v) = max({h_u : u in S(N(v))})
   Captures the most salient feature per dimension.

===============================================================
SAMPLING: WHY NOT USE ALL NEIGHBORS?
===============================================================

Full neighborhood aggregation has problems:
1. HIGH-DEGREE NODES: Some nodes have 1000+ neighbors
   -> Computation and memory explodes
2. MINI-BATCH TRAINING: Can't fit full neighborhoods in batch
   -> Need bounded computation per node
3. REGULARIZATION: Sampling adds stochasticity
   -> Acts like dropout, prevents overfitting

Typical sample sizes: 10-25 per layer
With 2 layers: at most 25 * 25 = 625 nodes per ego-graph

===============================================================
INDUCTIVE BIAS
===============================================================

1. LEARNED AGGREGATION: Generalizes to new structures
2. CONCAT(self, neighbors): Self-info never lost (unlike GCN)
3. SAMPLING: Bounded computation, stochastic regularization
4. HOMOPHILY: Still assumes neighbors share information
5. LOCAL: k layers = k-hop, same as GCN

WHEN IT WORKS:
    Inductive: train on one graph, test on another
    Large graphs: sampling makes it scalable
    Dynamic graphs: new nodes can be classified immediately

WHEN IT FAILS:
    Very heterophilic graphs (neighbors are different classes)
    Very small sample sizes lose too much information

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
GCN = gcn_module.GCN


# ============================================================
# GRAPHSAGE IMPLEMENTATION
# ============================================================

class GraphSAGE:
    """
    GraphSAGE: Graph SAmple and aggreGatE.

    Paradigm: INDUCTIVE AGGREGATION

    Key difference from GCN:
        GCN:       H' = sigma( A_hat H W )   (fixed graph convolution)
        GraphSAGE: h_v = sigma( W . CONCAT(h_v, AGG({h_u : u in S(N(v))})) )

    The aggregation function is LEARNED, not fixed.
    Neighbors are SAMPLED, not all used.
    Self-features are CONCATENATED, not smoothed in.
    """

    def __init__(self, n_features, n_hidden, n_classes, n_layers=2,
                 aggregator='mean', sample_size=10, dropout=0.5,
                 lr=0.01, random_state=None):
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
            Number of GraphSAGE layers (2 recommended)
        aggregator : str
            'mean', 'pool', or 'max'
        sample_size : int
            Number of neighbors to sample per node per layer
        dropout : float
            Dropout rate
        lr : float
            Learning rate
        random_state : int or None
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aggregator = aggregator
        self.sample_size = sample_size
        self.dropout = dropout
        self.lr = lr
        self.random_state = random_state

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all layers."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        # Pool aggregator has extra MLP weights per layer
        self.pool_weights = []
        self.pool_biases = []

        # Layer dimensions:
        # Layer 0: CONCAT(self=n_features, agg=n_features) -> n_hidden
        # Layer 1..L-2: CONCAT(self=n_hidden, agg=n_hidden) -> n_hidden
        # Layer L-1: CONCAT(self=n_hidden, agg=n_hidden) -> n_classes

        for l in range(self.n_layers):
            if l == 0:
                d_in = self.n_features
            else:
                d_in = self.n_hidden

            if l == self.n_layers - 1:
                d_out = self.n_classes
            else:
                d_out = self.n_hidden

            # CONCAT(self, aggregated) -> 2 * d_in input dimension
            concat_dim = 2 * d_in
            std = np.sqrt(2.0 / (concat_dim + d_out))
            W = np.random.randn(concat_dim, d_out) * std
            b = np.zeros(d_out)
            self.weights.append(W)
            self.biases.append(b)

            # Pool aggregator MLP: d_in -> d_in
            if self.aggregator == 'pool':
                std_pool = np.sqrt(2.0 / (d_in + d_in))
                W_pool = np.random.randn(d_in, d_in) * std_pool
                b_pool = np.zeros(d_in)
                self.pool_weights.append(W_pool)
                self.pool_biases.append(b_pool)

    def _sample_neighbors(self, graph, node, k):
        """
        Sample k neighbors for a node.

        If the node has fewer than k neighbors, use all of them.
        If the node has no neighbors, return [node] (self-loop fallback).

        Parameters:
        -----------
        graph : Graph
        node : int
        k : int - number of neighbors to sample

        Returns: array of sampled neighbor indices
        """
        neighbors = graph.neighbors(node)

        if len(neighbors) == 0:
            # Isolated node: use self as fallback
            return np.array([node])

        if k >= len(neighbors):
            # Use all neighbors (no sampling needed)
            return neighbors

        # Sample k neighbors without replacement
        return np.random.choice(neighbors, size=k, replace=False)

    def _aggregate_mean(self, neighbor_features):
        """
        Mean aggregator: simple average of neighbor features.

        Parameters:
        -----------
        neighbor_features : ndarray (k, d) -- features of sampled neighbors

        Returns: ndarray (d,) -- aggregated feature vector
        """
        return np.mean(neighbor_features, axis=0)

    def _aggregate_pool(self, neighbor_features, W_pool, b_pool):
        """
        Pool aggregator: MLP followed by element-wise max-pooling.

        h_N(v) = max(sigma(W_pool * h_u + b_pool) for u in neighbors)

        Parameters:
        -----------
        neighbor_features : ndarray (k, d)
        W_pool : ndarray (d, d) -- pool MLP weights
        b_pool : ndarray (d,) -- pool MLP bias

        Returns: ndarray (d,) -- aggregated feature vector
        """
        # Apply MLP: sigma(W_pool * h + b_pool)
        transformed = neighbor_features @ W_pool + b_pool
        transformed = np.maximum(transformed, 0)  # ReLU

        # Element-wise max pooling
        return np.max(transformed, axis=0)

    def _aggregate_max(self, neighbor_features):
        """
        Max aggregator: element-wise maximum of neighbor features.

        Parameters:
        -----------
        neighbor_features : ndarray (k, d)

        Returns: ndarray (d,) -- aggregated feature vector
        """
        return np.max(neighbor_features, axis=0)

    def forward(self, graph, X=None, training=True):
        """
        Forward pass through all GraphSAGE layers.

        For each layer:
            1. Sample neighbors for each node
            2. Aggregate neighbor features
            3. Concat(self, aggregated)
            4. Linear transform + activation
            5. L2 normalize (hidden layers)

        Parameters:
        -----------
        graph : Graph
        X : ndarray (n, d) or None -- node features (uses graph.X if None)
        training : bool -- if True, apply dropout

        Returns: (output_probs, cache)
        """
        if X is None:
            X = graph.X

        n = graph.n_nodes
        H = X.copy()

        cache = {
            'H': [H.copy()],
            'Z': [],
            'concat': [],
            'agg': [],
            'sampled_neighbors': [],
            'neighbor_features': [],
            'pool_pre_relu': [],
            'dropout_masks': [],
            'H_pre_norm': [],
        }

        for l in range(self.n_layers):
            d_in = H.shape[1]
            aggregated = np.zeros((n, d_in))
            sampled = []
            neigh_feats_layer = []
            pool_pre_relu_layer = []

            for v in range(n):
                # Step 1: Sample neighbors
                neighbors_v = self._sample_neighbors(graph, v, self.sample_size)
                sampled.append(neighbors_v)

                # Step 2: Get neighbor features
                neigh_feats = H[neighbors_v]
                neigh_feats_layer.append(neigh_feats)

                # Step 3: Aggregate
                if self.aggregator == 'mean':
                    aggregated[v] = self._aggregate_mean(neigh_feats)
                elif self.aggregator == 'pool':
                    pre_relu = neigh_feats @ self.pool_weights[l] + self.pool_biases[l]
                    pool_pre_relu_layer.append(pre_relu)
                    aggregated[v] = self._aggregate_pool(
                        neigh_feats, self.pool_weights[l], self.pool_biases[l]
                    )
                elif self.aggregator == 'max':
                    aggregated[v] = self._aggregate_max(neigh_feats)

            cache['sampled_neighbors'].append(sampled)
            cache['neighbor_features'].append(neigh_feats_layer)
            cache['pool_pre_relu'].append(pool_pre_relu_layer)
            cache['agg'].append(aggregated.copy())

            # Step 4: Concatenate self with aggregated
            concat_features = np.concatenate([H, aggregated], axis=1)
            cache['concat'].append(concat_features.copy())

            # Step 5: Linear transform
            Z = concat_features @ self.weights[l] + self.biases[l]
            cache['Z'].append(Z.copy())

            # Step 6: Activation
            if l < self.n_layers - 1:
                # Hidden layer: ReLU
                H = np.maximum(Z, 0)

                # Dropout
                if training and self.dropout > 0:
                    mask = (np.random.rand(*H.shape) > self.dropout).astype(float)
                    H = H * mask / (1 - self.dropout + 1e-10)
                    cache['dropout_masks'].append(mask)
                else:
                    cache['dropout_masks'].append(np.ones_like(H))

                cache['H_pre_norm'].append(H.copy())

                # L2 normalize (stabilizes training)
                norms = np.linalg.norm(H, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                H = H / norms
            else:
                # Output layer: softmax
                H = softmax(Z)
                cache['H_pre_norm'].append(None)

            cache['H'].append(H.copy())

        return H, cache

    def backward(self, graph, y, mask, cache):
        """
        Full backpropagation through all GraphSAGE layers.

        Chain rule through:
        softmax -> linear -> concat -> (aggregate + self)
        -> relu -> dropout -> L2norm -> linear -> ...

        Parameters:
        -----------
        graph : Graph
        y : ndarray -- true labels
        mask : boolean array -- train mask
        cache : dict -- from forward pass
        """
        n = len(y)
        probs = cache['H'][-1]

        # Gradient of cross-entropy + softmax
        dZ = probs.copy()
        dZ[np.arange(n), y] -= 1

        # Mask: only backprop through train nodes
        mask_float = mask.astype(float)
        n_train = max(np.sum(mask), 1)
        dZ = dZ * mask_float[:, None] / n_train

        for l in range(self.n_layers - 1, -1, -1):
            concat_features = cache['concat'][l]
            H_prev = cache['H'][l]
            d_in = H_prev.shape[1]

            # Gradient w.r.t. weights and biases
            gW = concat_features.T @ dZ
            gb = np.sum(dZ, axis=0)

            # Gradient w.r.t. concat_features
            dConcat = dZ @ self.weights[l].T

            # Split gradient into self and aggregated parts
            dSelf = dConcat[:, :d_in]
            dAgg = dConcat[:, d_in:]

            # Gradient through aggregation -> to previous layer features
            dH_prev = dSelf.copy()

            if self.aggregator == 'mean':
                for v in range(n):
                    neighbors_v = cache['sampled_neighbors'][l][v]
                    k = len(neighbors_v)
                    if k > 0:
                        for nb in neighbors_v:
                            dH_prev[nb] += dAgg[v] / k

            elif self.aggregator == 'pool':
                dW_pool = np.zeros_like(self.pool_weights[l])
                db_pool = np.zeros_like(self.pool_biases[l])

                for v in range(n):
                    neighbors_v = cache['sampled_neighbors'][l][v]
                    neigh_feats = cache['neighbor_features'][l][v]
                    pre_relu = cache['pool_pre_relu'][l][v]

                    if len(neighbors_v) == 0:
                        continue

                    post_relu = np.maximum(pre_relu, 0)
                    max_idx = np.argmax(post_relu, axis=0)

                    d_post_relu = np.zeros_like(post_relu)
                    for dim in range(d_post_relu.shape[1]):
                        d_post_relu[max_idx[dim], dim] = dAgg[v, dim]

                    d_pre_relu = d_post_relu * (pre_relu > 0).astype(float)

                    dW_pool += neigh_feats.T @ d_pre_relu
                    db_pool += np.sum(d_pre_relu, axis=0)

                    d_neigh = d_pre_relu @ self.pool_weights[l].T
                    for idx, nb in enumerate(neighbors_v):
                        dH_prev[nb] += d_neigh[idx]

                self.pool_weights[l] -= self.lr * dW_pool
                self.pool_biases[l] -= self.lr * db_pool

            elif self.aggregator == 'max':
                for v in range(n):
                    neighbors_v = cache['sampled_neighbors'][l][v]
                    neigh_feats = cache['neighbor_features'][l][v]

                    if len(neighbors_v) == 0:
                        continue

                    max_idx = np.argmax(neigh_feats, axis=0)
                    for dim in range(neigh_feats.shape[1]):
                        nb_idx = max_idx[dim]
                        if nb_idx < len(neighbors_v):
                            dH_prev[neighbors_v[nb_idx]] += dAgg[v, dim]

            # Update weights for this layer
            self.weights[l] -= self.lr * gW
            self.biases[l] -= self.lr * gb

            # Prepare gradient for previous layer
            if l > 0:
                # Backprop through L2 normalization
                H_pre_norm = cache['H_pre_norm'][l - 1]
                if H_pre_norm is not None:
                    norms = np.linalg.norm(H_pre_norm, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-8)
                    H_normed = H_pre_norm / norms

                    dot_products = np.sum(dH_prev * H_normed, axis=1, keepdims=True)
                    dH_pre_norm = (dH_prev - H_normed * dot_products) / norms
                else:
                    dH_pre_norm = dH_prev

                # Backprop through dropout
                dH_pre_norm = dH_pre_norm * cache['dropout_masks'][l - 1] / (1 - self.dropout + 1e-10)

                # Backprop through ReLU
                dZ = dH_pre_norm * (cache['Z'][l - 1] > 0).astype(float)

    def fit(self, graph, labels, train_mask, n_epochs=200, verbose=True):
        """
        Train GraphSAGE with transductive split.

        Parameters:
        -----------
        graph : Graph
        labels : ndarray -- node labels
        train_mask : boolean array
        n_epochs : int
        verbose : bool

        Returns: loss_history
        """
        loss_history = []

        for epoch in range(n_epochs):
            # Forward pass
            probs, cache = self.forward(graph, training=True)

            # Loss
            loss = cross_entropy_loss(probs, labels, train_mask)
            loss_history.append(loss)

            # Backward pass
            self.backward(graph, labels, train_mask, cache)

            if verbose and (epoch + 1) % 50 == 0:
                train_pred = np.argmax(probs, axis=1)
                train_acc = np.mean(train_pred[train_mask] == labels[train_mask])
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
        """Get learned node representations from second-to-last layer."""
        _, cache = self.forward(graph, training=False)
        return cache['H'][-2]


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    # -------- Experiment 1: Aggregator Comparison --------
    print("\n1. AGGREGATOR COMPARISON (mean vs pool vs max)")
    print("-" * 40)
    print("Which aggregation strategy works best?")

    graph, labels = create_community_graph(
        n_communities=3, nodes_per_community=25,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=42
    )
    n_classes = 3
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )

    agg_results = {}
    for agg in ['mean', 'pool', 'max']:
        accs = []
        for trial in range(3):
            model = GraphSAGE(
                graph.X.shape[1], 16, n_classes, n_layers=2,
                aggregator=agg, sample_size=10, dropout=0.3,
                lr=0.01, random_state=42 + trial
            )
            model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
            preds = model.predict(graph)
            test_acc = np.mean(preds[test_mask] == labels[test_mask])
            accs.append(test_acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        agg_results[agg] = mean_acc
        print(f"  {agg:<6}  test_acc={mean_acc:.3f} +/- {std_acc:.3f}")

    print("-> Mean: simplest, usually competitive")
    print("-> Pool: most expressive (MLP + max), captures nonlinear patterns")
    print("-> Max: captures salient features, ignores quantity")

    # -------- Experiment 2: Sample Size --------
    print("\n2. SAMPLE SIZE (neighbors per node)")
    print("-" * 40)
    print("How many neighbors do we need to sample?")

    for sample_size in [3, 5, 10, 25, 100]:
        label = f"k={sample_size}" if sample_size < 100 else "k=ALL"
        model = GraphSAGE(
            graph.X.shape[1], 16, n_classes, n_layers=2,
            aggregator='mean', sample_size=sample_size, dropout=0.3,
            lr=0.01, random_state=42
        )
        model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = model.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        avg_deg = np.mean(graph.degrees())
        coverage = min(sample_size / max(avg_deg, 1), 1.0)

        print(f"  {label:<8}  test_acc={test_acc:.3f}"
              f"  coverage={coverage:.1%} of avg_degree={avg_deg:.1f}")

    print("-> Small sample: faster but noisier")
    print("-> k=10-25 usually sufficient")
    print("-> Sampling adds regularization (like dropout)")

    # -------- Experiment 3: INDUCTIVE Generalization --------
    print("\n3. INDUCTIVE GENERALIZATION")
    print("-" * 40)
    print("Train on one graph, test on a DIFFERENT graph!")
    print("This is where GraphSAGE shines vs GCN.")

    # Train graph: 3 communities, 20 nodes each
    train_graph, train_labels = create_community_graph(
        n_communities=3, nodes_per_community=20,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=42
    )
    train_mask_ind = np.ones(train_graph.n_nodes, dtype=bool)

    # Test graph: 3 communities, 30 nodes each (DIFFERENT graph!)
    test_graph, test_labels = create_community_graph(
        n_communities=3, nodes_per_community=30,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=99
    )

    # GraphSAGE: train on train_graph, test on test_graph
    sage = GraphSAGE(
        train_graph.X.shape[1], 16, n_classes, n_layers=2,
        aggregator='mean', sample_size=10, dropout=0.3,
        lr=0.01, random_state=42
    )
    sage.fit(train_graph, train_labels, train_mask_ind, n_epochs=300, verbose=False)

    sage_train_preds = sage.predict(train_graph)
    sage_train_acc = np.mean(sage_train_preds == train_labels)
    sage_test_preds = sage.predict(test_graph)
    sage_test_acc = np.mean(sage_test_preds == test_labels)

    print(f"\n  GraphSAGE:")
    print(f"    Train graph acc: {sage_train_acc:.3f} ({train_graph.n_nodes} nodes)")
    print(f"    Test graph acc:  {sage_test_acc:.3f} ({test_graph.n_nodes} nodes, UNSEEN!)")

    # GCN: train on train_graph, test on test_graph
    gcn = GCN(
        train_graph.X.shape[1], 16, n_classes, n_layers=2,
        lr=0.01, dropout=0.3, random_state=42
    )
    gcn.fit(train_graph, train_labels, train_mask_ind, n_epochs=300, verbose=False)

    gcn_train_preds = gcn.predict(train_graph)
    gcn_train_acc = np.mean(gcn_train_preds == train_labels)
    gcn_test_preds = gcn.predict(test_graph)
    gcn_test_acc = np.mean(gcn_test_preds == test_labels)

    print(f"\n  GCN:")
    print(f"    Train graph acc: {gcn_train_acc:.3f} ({train_graph.n_nodes} nodes)")
    print(f"    Test graph acc:  {gcn_test_acc:.3f} ({test_graph.n_nodes} nodes, UNSEEN!)")

    print(f"\n  Difference on unseen graph: GraphSAGE={sage_test_acc:.3f} vs GCN={gcn_test_acc:.3f}")
    print("-> GraphSAGE learns GENERAL aggregation -> transfers to new graphs")
    print("-> GCN learns graph-specific weights -> may not transfer well")

    # -------- Experiment 4: Number of Layers --------
    print("\n4. NUMBER OF LAYERS (Over-smoothing check)")
    print("-" * 40)

    for n_layers in [1, 2, 3, 4]:
        model = GraphSAGE(
            graph.X.shape[1], 16, n_classes, n_layers=n_layers,
            aggregator='mean', sample_size=10, dropout=0.3,
            lr=0.01, random_state=42
        )
        model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)
        preds = model.predict(graph)
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        # Measure embedding diversity (over-smoothing indicator)
        emb = model.get_embeddings(graph)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms
        cos_sim = emb_norm @ emb_norm.T
        avg_sim = (np.sum(cos_sim) - graph.n_nodes) / (graph.n_nodes * (graph.n_nodes - 1))

        print(f"  layers={n_layers}  test_acc={test_acc:.3f}"
              f"  avg_cos_sim={avg_sim:.3f}")

    print("-> 2 layers is typically best (same as GCN)")
    print("-> GraphSAGE's CONCAT helps: self-info preserved better")
    print("-> Still susceptible to over-smoothing, but less than GCN")

    return agg_results


# ============================================================
# BENCHMARK ON DATASETS
# ============================================================

def benchmark_on_datasets():
    """Benchmark GraphSAGE across standard graph datasets."""
    print("\n" + "=" * 60)
    print("BENCHMARK: GraphSAGE on Graph Datasets")
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
        graph, labels = dataset_fn()
        train_mask, val_mask, test_mask = create_transductive_split(
            graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
        )

        model = GraphSAGE(
            graph.X.shape[1], 16, n_classes, n_layers=2,
            aggregator='mean', sample_size=10, dropout=0.5,
            lr=0.01, random_state=42
        )
        model.fit(graph, labels, train_mask, n_epochs=200, verbose=False)

        preds = model.predict(graph)
        train_acc = np.mean(preds[train_mask] == labels[train_mask])
        test_acc = np.mean(preds[test_mask] == labels[test_mask])

        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name:<15} {train_acc:<12.3f} {test_acc:<12.3f} {graph.n_nodes:<8}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_graphsage():
    """
    Main GraphSAGE visualization: 2x3 grid.

    Panel 1: Ground truth (karate club)
    Panel 2: GraphSAGE predictions
    Panel 3: Sampling visualization
    Panel 4: Aggregator comparison bar chart
    Panel 5: Training curve
    Panel 6: Correct vs incorrect predictions
    """
    print("\nGenerating: GraphSAGE visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Use karate club
    graph, labels = karate_club()
    n_classes = 2
    train_mask, val_mask, test_mask = create_transductive_split(
        graph.n_nodes, labels, train_ratio=0.15, val_ratio=0.1
    )
    pos = spring_layout(graph, seed=42)

    # Train GraphSAGE
    sage = GraphSAGE(
        graph.X.shape[1], 16, n_classes, n_layers=2,
        aggregator='mean', sample_size=10, dropout=0.3,
        lr=0.01, random_state=42
    )
    losses = sage.fit(graph, labels, train_mask, n_epochs=300, verbose=False)

    # --- Panel 1: Ground truth ---
    draw_graph(graph, labels, pos, axes[0, 0],
               title='Ground Truth\n(Karate Club)', cmap='coolwarm')

    # --- Panel 2: GraphSAGE predictions ---
    preds = sage.predict(graph)
    test_acc = np.mean(preds[test_mask] == labels[test_mask])
    draw_graph(graph, preds, pos, axes[0, 1],
               title=f'GraphSAGE Predictions\ntest_acc={test_acc:.3f}', cmap='coolwarm')

    # --- Panel 3: Sampling visualization ---
    ax = axes[0, 2]
    center_node = 0  # Node 0 (hub in karate club)
    np.random.seed(42)
    sampled_neighbors = sage._sample_neighbors(graph, center_node, 10)
    all_neighbors = graph.neighbors(center_node)

    # Draw all edges faded
    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            if graph.A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        'gray', alpha=0.1, linewidth=0.5)

    # Highlight edges to sampled neighbors
    for nb in sampled_neighbors:
        ax.plot([pos[center_node, 0], pos[nb, 0]],
                [pos[center_node, 1], pos[nb, 1]],
                'red', alpha=0.7, linewidth=2, zorder=4)

    # Highlight edges to un-sampled neighbors
    unsampled = set(all_neighbors.tolist()) - set(sampled_neighbors.tolist())
    for nb in unsampled:
        ax.plot([pos[center_node, 0], pos[nb, 0]],
                [pos[center_node, 1], pos[nb, 1]],
                'orange', alpha=0.4, linewidth=1, linestyle='--', zorder=3)

    # Draw all nodes faded
    ax.scatter(pos[:, 0], pos[:, 1], c='lightgray', s=40, alpha=0.4,
               edgecolors='gray', linewidths=0.3, zorder=2)

    # Highlight sampled neighbors
    ax.scatter(pos[sampled_neighbors, 0], pos[sampled_neighbors, 1],
               c='red', s=80, alpha=0.8, edgecolors='black', linewidths=0.5,
               zorder=5, label=f'Sampled ({len(sampled_neighbors)})')

    # Highlight un-sampled neighbors
    if len(unsampled) > 0:
        unsampled_arr = np.array(list(unsampled))
        ax.scatter(pos[unsampled_arr, 0], pos[unsampled_arr, 1],
                   c='orange', s=60, alpha=0.6, edgecolors='black', linewidths=0.3,
                   zorder=5, label=f'Not sampled ({len(unsampled)})')

    # Center node
    ax.scatter([pos[center_node, 0]], [pos[center_node, 1]],
               c='blue', s=200, alpha=0.9, edgecolors='black', linewidths=1.5,
               zorder=6, marker='*', label=f'Node {center_node}')

    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(f'Neighbor Sampling\nNode {center_node}: {len(all_neighbors)} neighbors,'
                 f' sample {len(sampled_neighbors)}')
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Panel 4: Aggregator comparison bar chart ---
    ax = axes[1, 0]
    cg, cl = create_community_graph(3, 20, 0.25, 0.02, 16, random_state=42)
    tm, vm, tsm = create_transductive_split(cg.n_nodes, cl, 0.15, 0.1)

    agg_accs = {}
    for agg_name in ['mean', 'pool', 'max']:
        model = GraphSAGE(
            cg.X.shape[1], 16, 3, n_layers=2,
            aggregator=agg_name, sample_size=10, dropout=0.3,
            lr=0.01, random_state=42
        )
        model.fit(cg, cl, tm, n_epochs=200, verbose=False)
        p = model.predict(cg)
        agg_accs[agg_name] = np.mean(p[tsm] == cl[tsm])

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(list(agg_accs.keys()), list(agg_accs.values()),
                  color=colors, edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars, agg_accs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Aggregator Comparison\n(Community Graph)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 5: Training curve ---
    ax = axes[1, 1]
    ax.plot(losses, 'b-', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss\n(Karate Club)')
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Correct vs incorrect predictions ---
    ax = axes[1, 2]
    correct = preds == labels
    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            if graph.A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
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

    plt.suptitle('GraphSAGE: Sample + Aggregate + Concat\n'
                 'Row 1: predictions & sampling | Row 2: aggregators & training',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_inductive():
    """
    Inductive learning visualization: 1x3 grid.

    Panel 1: Train graph with labels
    Panel 2: Test graph (DIFFERENT graph, same structure type)
    Panel 3: Comparison table: GCN vs GraphSAGE
    """
    print("\nGenerating: Inductive learning visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Create train and test graphs with DIFFERENT random seeds
    train_graph, train_labels = create_community_graph(
        n_communities=3, nodes_per_community=20,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=42
    )
    test_graph, test_labels = create_community_graph(
        n_communities=3, nodes_per_community=30,
        p_in=0.25, p_out=0.02, feature_dim=16, random_state=99
    )

    train_mask = np.ones(train_graph.n_nodes, dtype=bool)
    n_classes = 3

    # Train both models on train_graph
    sage = GraphSAGE(
        train_graph.X.shape[1], 16, n_classes, n_layers=2,
        aggregator='mean', sample_size=10, dropout=0.3,
        lr=0.01, random_state=42
    )
    sage.fit(train_graph, train_labels, train_mask, n_epochs=300, verbose=False)

    gcn = GCN(
        train_graph.X.shape[1], 16, n_classes, n_layers=2,
        lr=0.01, dropout=0.3, random_state=42
    )
    gcn.fit(train_graph, train_labels, train_mask, n_epochs=300, verbose=False)

    # Evaluate on both graphs
    sage_train_preds = sage.predict(train_graph)
    sage_test_preds = sage.predict(test_graph)
    gcn_train_preds = gcn.predict(train_graph)
    gcn_test_preds = gcn.predict(test_graph)

    sage_train_acc = np.mean(sage_train_preds == train_labels)
    sage_test_acc = np.mean(sage_test_preds == test_labels)
    gcn_train_acc = np.mean(gcn_train_preds == train_labels)
    gcn_test_acc = np.mean(gcn_test_preds == test_labels)

    # --- Panel 1: Train graph ---
    train_pos = spring_layout(train_graph, seed=42)
    draw_graph(train_graph, train_labels, train_pos, axes[0],
               title=f'TRAIN Graph\n{train_graph.n_nodes} nodes, 3 communities',
               cmap='Set1')

    # --- Panel 2: Test graph with GraphSAGE predictions ---
    test_pos = spring_layout(test_graph, seed=99)
    ax = axes[1]
    for i in range(test_graph.n_nodes):
        for j in range(i + 1, test_graph.n_nodes):
            if test_graph.A[i, j] > 0:
                ax.plot([test_pos[i, 0], test_pos[j, 0]],
                        [test_pos[i, 1], test_pos[j, 1]],
                        'gray', alpha=0.2, linewidth=0.5)

    correct_sage = sage_test_preds == test_labels
    ax.scatter(test_pos[correct_sage, 0], test_pos[correct_sage, 1],
               c=test_labels[correct_sage], cmap='Set1', s=60, alpha=0.8,
               edgecolors='black', linewidths=0.5, zorder=3,
               vmin=0, vmax=2)
    if np.sum(~correct_sage) > 0:
        ax.scatter(test_pos[~correct_sage, 0], test_pos[~correct_sage, 1],
                   c='red', s=80, alpha=0.9, edgecolors='black', linewidths=1,
                   zorder=4, marker='X')
    ax.set_title(f'TEST Graph (UNSEEN!)\n{test_graph.n_nodes} nodes, '
                 f'GraphSAGE acc={sage_test_acc:.3f}')
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Panel 3: Comparison table ---
    ax = axes[2]
    ax.axis('off')

    table_data = [
        ['', 'Train Graph', 'Test Graph\n(UNSEEN)'],
        ['GraphSAGE', f'{sage_train_acc:.3f}', f'{sage_test_acc:.3f}'],
        ['GCN', f'{gcn_train_acc:.3f}', f'{gcn_test_acc:.3f}'],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # Style the header row
    for j in range(3):
        table[0, j].set_facecolor('#d4e6f1')
        table[0, j].set_text_props(fontweight='bold')

    # Style the model name column
    for i in range(1, 3):
        table[i, 0].set_facecolor('#fdebd0')
        table[i, 0].set_text_props(fontweight='bold')

    # Highlight better test accuracy
    if sage_test_acc >= gcn_test_acc:
        table[1, 2].set_facecolor('#d5f5e3')
    else:
        table[2, 2].set_facecolor('#d5f5e3')

    ax.set_title('INDUCTIVE Comparison\nTrain on 60 nodes, test on 90 nodes\n'
                 '(different graph, same community structure)',
                 fontsize=11, fontweight='bold')

    ax.text(0.5, 0.08,
            'GraphSAGE learns GENERAL aggregation functions\n'
            'that transfer to unseen graphs.\n'
            'GCN learns weights tied to specific graph structure.',
            ha='center', va='center', fontsize=9,
            style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('INDUCTIVE LEARNING: Train on one graph, test on ANOTHER',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GraphSAGE -- Paradigm: INDUCTIVE AGGREGATION")
    print("=" * 60)

    print("""
WHAT THIS MODEL IS:

    GraphSAGE = Graph SAmple and aggreGatE

    For each node v, at each layer:
        1. SAMPLE k neighbors: S(N(v))
        2. AGGREGATE: h_N(v) = AGG({h_u : u in S(N(v))})
        3. CONCAT:    [h_v || h_N(v)]
        4. TRANSFORM: h_v' = sigma(W . [h_v || h_N(v)])

    KEY DIFFERENCES FROM GCN:
        GCN:       H' = sigma(A_hat H W)    (fixed aggregation)
        GraphSAGE: h_v = W . CONCAT(h_v, AGG(neighbors))

    1. CONCAT separates self from neighbors (self never lost)
    2. SAMPLING bounds computation (scalable to large graphs)
    3. LEARNED aggregation (generalizes to new structures)

KEY HYPERPARAMETERS:
    - aggregator: 'mean' (simple), 'pool' (expressive), 'max'
    - sample_size: 10-25 (trade-off: accuracy vs speed)
    - n_layers: 2 (same over-smoothing concern as GCN)
    - hidden_dim: 16-32

INDUCTIVE BIAS:
    - HOMOPHILY: neighbors share information
    - AGGREGATION GENERALIZES: same function works on new graphs
    - BOUNDED NEIGHBORHOOD: sampling limits receptive field
    """)

    agg_results = ablation_experiments()
    results = benchmark_on_datasets()

    print("\nGenerating visualizations...")

    fig1 = visualize_graphsage()
    save_path1 = '/Users/sid47/ML Algorithms/37_graphsage.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_inductive()
    save_path2 = '/Users/sid47/ML Algorithms/37_graphsage_inductive.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("SUMMARY: What GraphSAGE Reveals")
    print("=" * 60)
    print("""
1. INDUCTIVE > TRANSDUCTIVE
   GraphSAGE learns GENERAL aggregation, not graph-specific weights.
   Train on graph A, deploy on graph B. GCN cannot do this.

2. CONCAT IS KEY
   By explicitly separating self from neighbors:
   h_v' = W . [h_v || AGG(neighbors)]
   The node's own features are NEVER lost.
   GCN's smoothing can wash out node identity.

3. SAMPLING = REGULARIZATION + SCALABILITY
   Sampling k neighbors per node:
   - Bounds computation (O(k^L) instead of O(n))
   - Acts as stochastic regularization (like dropout)
   - Makes mini-batch training possible

4. AGGREGATOR MATTERS (but not as much as you'd think)
   Mean: simple, competitive baseline
   Pool: MLP + max, most parameters, most expressive
   Max: captures salient features, lightweight
   In practice, mean is often good enough.

5. STILL LIMITED BY HOMOPHILY
   Like GCN, GraphSAGE assumes neighbors share information.
   For heterophilic graphs (neighbors differ), both struggle.

CONNECTION TO OTHER FILES:
    36_gcn.py: Fixed spectral convolution (GraphSAGE's predecessor)
    38_gat.py: Attention-weighted aggregation (learned importance)
    39_gin.py: SUM aggregation for maximum expressiveness
    40_mpnn.py: Unified framework that encompasses all of these

NEXT: 38_gat.py -- What if different neighbors matter differently?
    """)
