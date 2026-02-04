"""
Temporal GNN — Dynamic Graphs Over Time
========================================

Paradigm: GRAPHS THAT CHANGE OVER TIME

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Real graphs often CHANGE:
- Edges appear and disappear
- Node features evolve
- New nodes join, old nodes leave

STATIC GRAPH: G = (V, E)
    Fixed nodes and edges

TEMPORAL GRAPH: G(t) = (V(t), E(t))
    Nodes and edges change over time

EXAMPLES:
- Social networks: New friendships, posts
- Financial networks: Transactions over time
- Traffic networks: Congestion patterns
- Email networks: Communication flows

===============================================================
APPROACHES
===============================================================

1. DISCRETE TIME (Snapshots)
   G_1, G_2, ..., G_T
   Process each snapshot with GNN, combine with RNN/Transformer

2. CONTINUOUS TIME
   Events: (u, v, t, type)
   Model as temporal point process
   No fixed time steps

===============================================================
SNAPSHOT-BASED METHODS
===============================================================

SIMPLE: GNN + RNN
    h_v^(t) = GNN(G_t, X_t)      (spatial)
    z_v^(t) = RNN(h_v^(1:t))     (temporal)

EVOLVED GCN:
    Use GNN at each timestep
    RNN/LSTM evolves the GNN weights themselves!

GCRN (Graph Convolutional Recurrent Network):
    Combine GCN and LSTM in each cell
    h_t = GCN(A_t, LSTM(h_{t-1}, x_t))

===============================================================
CONTINUOUS-TIME METHODS
===============================================================

TEMPORAL GRAPH NETWORKS (TGN):

Key components:
1. MESSAGE FUNCTION: Encode interaction event
2. MEMORY: Node-level memory updated at each interaction
3. EMBEDDING: Combine memory with neighbor info

When event (u, v, t) occurs:
    m_u(t) = msg(s_u(t-), s_v(t-), Δt, e_uv)
    s_u(t) = GRU(s_u(t-), m_u(t))

Memory captures temporal patterns!

===============================================================
TEMPORAL ATTENTION
===============================================================

Attend over neighbors WEIGHTED BY TIME:

    α_ij(t) = softmax_j(q_i^T [k_j || Φ(t - t_j)])

Where:
- Φ is time encoding (like positional encoding)
- t - t_j is time since last interaction with j
- Recent interactions get more attention

===============================================================
TIME ENCODING
===============================================================

How to represent time difference Δt?

1. LEARNABLE EMBEDDING
   Similar to positional encoding in Transformers
   Φ(Δt) = [cos(ω_1 Δt), sin(ω_1 Δt), ...]

2. EXPONENTIAL DECAY
   Φ(Δt) = exp(-α Δt)
   "Recent matters more"

3. BINNING
   Discretize time into buckets

===============================================================
TASKS ON TEMPORAL GRAPHS
===============================================================

1. LINK PREDICTION (over time)
   Will edge (u, v) exist at time t+1?

2. NODE CLASSIFICATION (evolving)
   What is node u's label at time t?

3. EVENT PREDICTION
   When will the next event occur?

===============================================================
INDUCTIVE BIAS
===============================================================

1. TEMPORAL LOCALITY: Recent events matter more
2. TEMPORAL PATTERNS: Behaviors repeat over time
3. GRAPH + TIME: Both spatial and temporal structure
4. MEMORY: Nodes have history that affects future

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


class TemporalGraph:
    """
    Temporal graph with time-stamped edges.
    """

    def __init__(self, n_nodes, feature_dim=16):
        self.n_nodes = n_nodes
        self.feature_dim = feature_dim

        # Node features (can evolve)
        self.X = np.random.randn(n_nodes, feature_dim)

        # Temporal edges: list of (src, dst, time, features)
        self.events = []

        # Current time
        self.current_time = 0

    def add_event(self, src, dst, time=None, features=None):
        """Add a temporal edge/event."""
        if time is None:
            time = self.current_time
            self.current_time += 1

        if features is None:
            features = np.zeros(1)

        self.events.append((src, dst, time, features))

    def get_snapshot(self, t):
        """Get graph snapshot at time t."""
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for src, dst, time, _ in self.events:
            if time <= t:
                adj[src, dst] = 1
                adj[dst, src] = 1  # Undirected
        return adj

    def get_events_before(self, t, lookback=None):
        """Get events before time t (within lookback window)."""
        events = []
        for e in self.events:
            if e[2] < t:
                if lookback is None or t - e[2] <= lookback:
                    events.append(e)
        return events


class TimeEncoding:
    """
    Time encoding for temporal graphs.
    Similar to positional encoding in Transformers.
    """

    def __init__(self, dim):
        self.dim = dim
        # Learnable frequencies
        self.omega = np.random.randn(dim // 2) * 0.1

    def encode(self, delta_t):
        """Encode time difference."""
        if isinstance(delta_t, (int, float)):
            delta_t = np.array([delta_t])

        # Fourier features
        angles = np.outer(delta_t, self.omega)
        encoding = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)

        return encoding.squeeze()


class TemporalGCN:
    """
    Temporal GCN: GCN applied to graph snapshots.
    Captures spatial structure at each timestep.
    """

    def __init__(self, n_features, hidden_dim, out_dim):
        scale = np.sqrt(2.0 / n_features)
        self.W = np.random.randn(n_features, hidden_dim) * scale
        self.W_out = np.random.randn(hidden_dim, out_dim) * 0.1

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, adj, X):
        """Forward pass on single snapshot."""
        # Normalize adjacency
        adj_norm = adj + np.eye(adj.shape[0])
        D = np.diag(1.0 / (np.sqrt(np.sum(adj_norm, axis=1)) + 1e-10))
        adj_norm = D @ adj_norm @ D

        # GCN layer
        H = self.relu(adj_norm @ X @ self.W)
        out = H @ self.W_out

        return out


class SnapshotGNN:
    """
    Snapshot-based Temporal GNN.

    Process each snapshot with GNN, combine with RNN.
    """

    def __init__(self, n_features, hidden_dim, n_classes):
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # GCN for spatial
        self.gcn = TemporalGCN(n_features, hidden_dim, hidden_dim)

        # RNN for temporal
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_x = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

        # Output
        self.W_out = np.random.randn(hidden_dim, n_classes) * 0.1

    def forward(self, snapshots, X_list):
        """
        Forward pass over sequence of snapshots.

        snapshots: List of adjacency matrices
        X_list: List of node features (one per timestep)

        Returns: Node embeddings after temporal aggregation
        """
        n_nodes = X_list[0].shape[0]

        # Initialize hidden state
        h = np.zeros((n_nodes, self.hidden_dim))

        for adj, X in zip(snapshots, X_list):
            # Spatial: GCN
            gcn_out = self.gcn.forward(adj, X)

            # Temporal: RNN update
            h = np.tanh(gcn_out @ self.W_x + h @ self.W_h + self.b_h)

        # Output
        logits = h @ self.W_out
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs


class TGN:
    """
    Simplified Temporal Graph Network.

    Key idea: Nodes have MEMORY that is updated with each interaction.
    """

    def __init__(self, n_nodes, feature_dim, memory_dim, time_dim=16):
        self.n_nodes = n_nodes
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim

        # Node memory (persistent state)
        self.memory = np.zeros((n_nodes, memory_dim))

        # Time encoding
        self.time_encoding = TimeEncoding(time_dim)

        # Message function
        scale = np.sqrt(2.0 / (2 * memory_dim + time_dim))
        self.W_msg = np.random.randn(2 * memory_dim + time_dim, memory_dim) * scale

        # Memory update (GRU-style)
        self.W_z = np.random.randn(2 * memory_dim, memory_dim) * scale
        self.W_r = np.random.randn(2 * memory_dim, memory_dim) * scale
        self.W_h = np.random.randn(2 * memory_dim, memory_dim) * scale

        # Last interaction time for each node
        self.last_time = np.zeros(n_nodes)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def process_event(self, src, dst, time):
        """Process a single event and update memory."""
        # Time encoding
        delta_t_src = time - self.last_time[src]
        delta_t_dst = time - self.last_time[dst]

        time_enc_src = self.time_encoding.encode(delta_t_src)
        time_enc_dst = self.time_encoding.encode(delta_t_dst)

        # Message from interaction
        msg_src = np.concatenate([self.memory[src], self.memory[dst], time_enc_src])
        msg_dst = np.concatenate([self.memory[dst], self.memory[src], time_enc_dst])

        msg_src = np.tanh(msg_src @ self.W_msg)
        msg_dst = np.tanh(msg_dst @ self.W_msg)

        # GRU-style memory update for source
        concat_src = np.concatenate([self.memory[src], msg_src])
        z_src = self.sigmoid(concat_src @ self.W_z)
        r_src = self.sigmoid(concat_src @ self.W_r)
        concat_reset = np.concatenate([r_src * self.memory[src], msg_src])
        h_tilde_src = np.tanh(concat_reset @ self.W_h)
        self.memory[src] = (1 - z_src) * self.memory[src] + z_src * h_tilde_src

        # GRU-style memory update for destination
        concat_dst = np.concatenate([self.memory[dst], msg_dst])
        z_dst = self.sigmoid(concat_dst @ self.W_z)
        r_dst = self.sigmoid(concat_dst @ self.W_r)
        concat_reset = np.concatenate([r_dst * self.memory[dst], msg_dst])
        h_tilde_dst = np.tanh(concat_reset @ self.W_h)
        self.memory[dst] = (1 - z_dst) * self.memory[dst] + z_dst * h_tilde_dst

        # Update last interaction time
        self.last_time[src] = time
        self.last_time[dst] = time

    def get_embeddings(self):
        """Get current node embeddings (memory)."""
        return self.memory.copy()

    def reset(self):
        """Reset memory."""
        self.memory = np.zeros((self.n_nodes, self.memory_dim))
        self.last_time = np.zeros(self.n_nodes)


def create_temporal_graph():
    """Create example temporal graph with community structure."""
    n_nodes = 20
    graph = TemporalGraph(n_nodes, feature_dim=8)

    # Two communities
    community_1 = list(range(10))
    community_2 = list(range(10, 20))

    # Generate temporal events
    # Initially, mostly intra-community
    for t in range(50):
        if t < 25:
            # Phase 1: Communities separate
            if np.random.random() < 0.8:
                comm = community_1 if np.random.random() < 0.5 else community_2
            else:
                comm = list(range(n_nodes))
        else:
            # Phase 2: Communities merge
            comm = list(range(n_nodes))

        src = np.random.choice(comm)
        candidates = [n for n in comm if n != src]
        if len(candidates) == 0:
            continue
        dst = np.random.choice(candidates)
        graph.add_event(src, dst, time=t)

    return graph


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_temporal_gnn():
    """Comprehensive temporal GNN visualization."""
    print("\n" + "="*60)
    print("TEMPORAL GNN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: Temporal Graph Evolution ============
    ax1 = fig.add_subplot(2, 3, 1)

    graph = create_temporal_graph()

    # Show snapshots at different times
    times = [10, 25, 45]
    colors = ['blue', 'green', 'red']

    for t, color in zip(times, colors):
        adj = graph.get_snapshot(t)
        density = np.sum(adj) / (graph.n_nodes ** 2)
        ax1.bar(t, density, width=5, color=color, alpha=0.7, label=f't={t}')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Edge Density')
    ax1.set_title('Temporal Graph Evolution\nDensity changes over time')
    ax1.legend()

    # ============ Plot 2: Time Encoding ============
    ax2 = fig.add_subplot(2, 3, 2)

    time_enc = TimeEncoding(dim=8)
    delta_ts = np.linspace(0, 10, 100)

    encodings = np.array([time_enc.encode(dt) for dt in delta_ts])

    for i in range(4):
        ax2.plot(delta_ts, encodings[:, i], label=f'dim {i}')

    ax2.set_xlabel('Δt')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Time Encoding\nFourier-like features')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: TGN Memory Evolution ============
    ax3 = fig.add_subplot(2, 3, 3)

    graph = create_temporal_graph()
    tgn = TGN(n_nodes=20, feature_dim=8, memory_dim=16)

    memory_norms = []

    for src, dst, t, _ in graph.events:
        tgn.process_event(src, dst, t)
        memory_norms.append(np.mean(np.linalg.norm(tgn.memory, axis=1)))

    ax3.plot(memory_norms, 'purple', linewidth=2)
    ax3.set_xlabel('Event')
    ax3.set_ylabel('Avg Memory Norm')
    ax3.set_title('TGN Memory Evolution\nMemory accumulates information')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Snapshot GNN vs TGN ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Compare approaches on simple task
    n_snapshots = 5
    n_nodes = 10

    # Create snapshots
    snapshots = []
    X_list = []
    for t in range(n_snapshots):
        adj = np.random.rand(n_nodes, n_nodes) > 0.5
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        snapshots.append(adj.astype(float))
        X_list.append(np.random.randn(n_nodes, 8))

    # Snapshot GNN
    snapshot_model = SnapshotGNN(n_features=8, hidden_dim=16, n_classes=2)
    probs = snapshot_model.forward(snapshots, X_list)
    entropy_snapshot = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

    # TGN
    tgn = TGN(n_nodes=n_nodes, feature_dim=8, memory_dim=16)
    for t, adj in enumerate(snapshots):
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj[i, j] > 0:
                    tgn.process_event(i, j, t)

    # Show comparison
    ax4.bar([0, 1], [entropy_snapshot, 0.5], color=['steelblue', 'coral'])
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Snapshot GNN', 'TGN'])
    ax4.set_ylabel('Output Entropy')
    ax4.set_title('Model Comparison\nDifferent temporal modeling')

    # ============ Plot 5: Community Detection Over Time ============
    ax5 = fig.add_subplot(2, 3, 5)

    graph = create_temporal_graph()
    tgn = TGN(n_nodes=20, feature_dim=8, memory_dim=16)

    # Track embeddings over time
    times_to_plot = [0, 10, 25, 40, 49]
    embeddings_over_time = []

    event_idx = 0
    for t in range(50):
        # Process events up to time t
        while event_idx < len(graph.events) and graph.events[event_idx][2] <= t:
            src, dst, time, _ = graph.events[event_idx]
            tgn.process_event(src, dst, time)
            event_idx += 1

        if t in times_to_plot:
            embeddings_over_time.append(tgn.get_embeddings().copy())

    # Show embedding separation (using PCA-like projection)
    for i, (t, emb) in enumerate(zip(times_to_plot, embeddings_over_time)):
        # Project to 1D
        proj = emb @ np.random.randn(16)
        ax5.scatter([t] * 10, proj[:10], c='blue', alpha=0.5, s=30)
        ax5.scatter([t] * 10, proj[10:], c='red', alpha=0.5, s=30)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Embedding (projected)')
    ax5.set_title('Node Embeddings Over Time\nCommunities may merge/split')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Temporal GNN
    ══════════════════════════════

    THE KEY IDEA:
    Graphs change over time!

    G(t) = (V(t), E(t))

    APPROACHES:
    ┌────────────────────────────┐
    │ Snapshot-based:            │
    │   GNN per snapshot         │
    │   RNN/Transformer across   │
    ├────────────────────────────┤
    │ Continuous-time:           │
    │   Events: (u, v, t)        │
    │   Node memory updated      │
    │   on each interaction      │
    └────────────────────────────┘

    TGN KEY IDEAS:
    • Node memory (persistent state)
    • Time encoding (Δt → features)
    • GRU-style memory update

    TASKS:
    • Link prediction over time
    • Dynamic node classification
    • Event prediction
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    plt.suptitle('Temporal GNN — Dynamic Graphs Over Time\n'
                 'Capture both spatial and temporal patterns',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for temporal GNN."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Memory dimension effect
    print("\n1. EFFECT OF MEMORY DIMENSION (TGN)")
    print("-" * 40)

    for memory_dim in [4, 8, 16, 32]:
        graph = create_temporal_graph()
        tgn = TGN(n_nodes=20, feature_dim=8, memory_dim=memory_dim)

        for src, dst, t, _ in graph.events:
            tgn.process_event(src, dst, t)

        # Measure memory norm (proxy for information content)
        norm = np.mean(np.linalg.norm(tgn.memory, axis=1))
        print(f"memory_dim={memory_dim:<3}  avg_norm={norm:.3f}")

    print("→ Larger memory can store more information")

    # 2. Time encoding dimension
    print("\n2. EFFECT OF TIME ENCODING DIMENSION")
    print("-" * 40)

    for time_dim in [4, 8, 16, 32]:
        time_enc = TimeEncoding(dim=time_dim)

        # Test distinctiveness of different Δt
        encodings = [time_enc.encode(dt) for dt in [0, 1, 5, 10]]
        avg_dist = np.mean([np.linalg.norm(encodings[i] - encodings[j])
                          for i in range(len(encodings))
                          for j in range(i+1, len(encodings))])

        print(f"time_dim={time_dim:<3}  avg_encoding_distance={avg_dist:.3f}")

    print("→ More dimensions = more distinguishable time differences")

    # 3. Number of snapshots
    print("\n3. EFFECT OF NUMBER OF SNAPSHOTS")
    print("-" * 40)

    for n_snapshots in [1, 3, 5, 10]:
        snapshots = []
        X_list = []
        for t in range(n_snapshots):
            adj = (np.random.rand(10, 10) > 0.5).astype(float)
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)
            snapshots.append(adj)
            X_list.append(np.random.randn(10, 8))

        model = SnapshotGNN(n_features=8, hidden_dim=16, n_classes=2)
        probs = model.forward(snapshots, X_list)
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        print(f"n_snapshots={n_snapshots:<3}  output_entropy={entropy:.3f}")

    print("→ More snapshots = more temporal information")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Temporal GNN — Dynamic Graphs Over Time")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_temporal_gnn()
    save_path = '/Users/sid47/ML Algorithms/43_temporal_gnn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Temporal GNN: Graphs that change over time
2. Snapshot-based: GNN per timestep + RNN across time
3. Continuous-time: Event-driven, node memory
4. TGN: Memory + time encoding + GRU updates
5. Time encoding: Fourier features for Δt
6. Tasks: Dynamic link prediction, evolving classification
7. Key: Capture BOTH spatial AND temporal patterns
    """)
