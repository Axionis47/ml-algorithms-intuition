# ML Algorithms — Intuition-First Implementation Guide

## Philosophy
Every algorithm is implemented from scratch with:
1. **Core idea in plain English** — what is it REALLY doing?
2. **Key equations** — the math that makes it work
3. **Inductive bias** — what does it assume? What can't it see?
4. **Ablation experiments** — what happens when you remove/change each component?
5. **Visualizations** — decision boundaries, learning curves, attention patterns

---

## Current Algorithms (00-24)

### Foundational
- `00_datasets.py` — Challenge datasets (moons, circles, XOR, spiral, etc.)
- `01_linear_regression.py` — Paradigm: PROJECTION
- `02_logistic_regression.py` — Paradigm: PROBABILITY
- `03_knn.py` — Paradigm: MEMORY
- `04_naive_bayes.py` — Paradigm: INDEPENDENCE

### Bayesian Methods
- `05_gaussian_process.py` — Paradigm: FUNCTION DISTRIBUTION
- `06_bayesian_linear_reg.py` — Paradigm: POSTERIOR OVER WEIGHTS

### Tree-Based
- `07_decision_tree.py` — Paradigm: AXIS-ALIGNED SPLITS
- `08_svm.py` — Paradigm: MAXIMUM MARGIN
- `09_random_forest.py` — Paradigm: ENSEMBLE OF TREES
- `10_gradient_boosting.py` — Paradigm: SEQUENTIAL CORRECTION
- `11_adaboost.py` — Paradigm: WEIGHTED RESAMPLING

### Neural Networks
- `12_mlp.py` — Paradigm: LEARNED FEATURES
- `13_cnn.py` — Paradigm: SPATIAL HIERARCHY
- `14_rnn_lstm.py` — Paradigm: SEQUENTIAL MEMORY
- `15_transformer.py` — Paradigm: ATTENTION (All-to-All)

### Generative & Latent
- `16_gmm.py` — Paradigm: MIXTURE MODEL
- `17_hmm.py` — Paradigm: HIDDEN STATE SEQUENCE
- `18_vae.py` — Paradigm: LATENT COMPRESSION

### Uncertainty Quantification
- `19_conformal_prediction.py` — Paradigm: COVERAGE GUARANTEE
- `20_mc_dropout.py` — Paradigm: APPROXIMATE BAYESIAN
- `21_calibration.py` — Paradigm: PROBABILITY CORRECTNESS

### Advanced
- `22_online_learning.py` — Paradigm: STREAMING UPDATES
- `23_maml.py` — Paradigm: LEARNING TO LEARN
- `24_arena.py` — Benchmark comparison

---

## EXPANSION: Reinforcement Learning (25-34)

### 25_rl_fundamentals.py — Paradigm: SEQUENTIAL DECISION MAKING
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

An agent interacts with an environment:
    State → Action → Reward → Next State

Goal: Learn a POLICY π(a|s) that maximizes cumulative reward.

THE BELLMAN EQUATION (the key insight):
    V(s) = max_a [R(s,a) + γ × V(s')]

"The value of a state is the best immediate reward plus
 the discounted value of where you end up."

===============================================================
KEY CONCEPTS
===============================================================

1. POLICY π(a|s) — probability of action a in state s
2. VALUE FUNCTION V(s) — expected return from state s
3. Q-FUNCTION Q(s,a) — expected return from (state, action) pair
4. REWARD — immediate signal
5. RETURN — cumulative discounted reward: Σ γᵗ rₜ
6. DISCOUNT γ — how much we care about future (0-1)

===============================================================
EXPLORATION vs EXPLOITATION
===============================================================

The fundamental RL dilemma:
- EXPLOIT: Do what you know works (greedy)
- EXPLORE: Try new things (might find better)

ε-greedy: With probability ε, take random action
UCB: Optimism in the face of uncertainty
Thompson Sampling: Sample from belief, act greedily
```

**Environments to implement:**
- GridWorld (simple navigation)
- CartPole (classic control)
- Multi-armed Bandit

**Ablations:**
- Discount factor γ effect
- Exploration rate ε effect
- Reward shaping impact

---

### 26_q_learning.py — Paradigm: VALUE ITERATION (Tabular)
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Learn Q(s,a) directly from experience, without knowing the environment.

THE Q-LEARNING UPDATE:
    Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
                              \_______TD target_______/

"Adjust Q toward the observed reward + best future estimate"

===============================================================
WHY IT WORKS
===============================================================

1. OFF-POLICY: Can learn from any data (doesn't need π)
2. BOOTSTRAPPING: Uses its own estimates (TD learning)
3. CONVERGENCE: Guaranteed to find optimal Q* (with enough exploration)

===============================================================
INDUCTIVE BIAS
===============================================================

1. Tabular — needs to visit every (s,a) pair
2. Markov assumption — state is sufficient
3. Stationary environment — dynamics don't change
```

**Ablations:**
- Learning rate α sweep
- Discount γ effect on myopic vs farsighted behavior
- ε-greedy vs softmax exploration
- SARSA (on-policy) vs Q-learning (off-policy)

---

### 27_dqn.py — Paradigm: VALUE ITERATION (Deep)
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Replace Q-table with a neural network: Q_θ(s,a)

THE DQN TRICKS (all three are essential):

1. EXPERIENCE REPLAY — store transitions, sample randomly
   WHY: Breaks correlation, reuses data, stabilizes learning

2. TARGET NETWORK — separate network for TD target
   WHY: Moving target problem → use frozen copy

3. FRAME STACKING — stack last k frames as state
   WHY: Single frame is not Markov (velocity is hidden)

===============================================================
THE LOSS FUNCTION
===============================================================

L(θ) = E[(r + γ × max_a' Q_θ'(s',a') - Q_θ(s,a))²]
              \_____target network____/

This is just MSE between prediction and TD target.
```

**Ablations:**
- With/without experience replay
- With/without target network
- Replay buffer size effect
- Target network update frequency
- Double DQN (decouple action selection from evaluation)

---

### 28_policy_gradient.py — Paradigm: DIRECT POLICY OPTIMIZATION
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of learning values, directly optimize the policy.

THE POLICY GRADIENT THEOREM:
    ∇J(θ) = E[∇log π_θ(a|s) × R]

"Increase probability of actions that led to high reward"

REINFORCE: Monte Carlo policy gradient
    θ ← θ + α × ∇log π_θ(aₜ|sₜ) × Gₜ

where Gₜ = Σ γᵏ rₜ₊ₖ (return from time t)

===============================================================
THE VARIANCE PROBLEM
===============================================================

Policy gradients have HIGH VARIANCE:
- Same action can have different returns (environment stochasticity)
- Long episodes = more variance

SOLUTIONS:
1. BASELINE: Subtract baseline b(s) to reduce variance
   ∇log π(a|s) × (R - b(s))

2. ACTOR-CRITIC: Use learned value function as baseline
   ∇log π(a|s) × (R - V(s))

===============================================================
INDUCTIVE BIAS
===============================================================

1. Stochastic policies (naturally explores)
2. On-policy (must use current policy's data)
3. No value bootstrapping in pure REINFORCE
```

**Ablations:**
- With/without baseline
- Return normalization effect
- Entropy regularization
- Batch size impact on variance

---

### 29_actor_critic.py — Paradigm: VALUE + POLICY (A2C)
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Combine best of both:
- ACTOR: Policy network π_θ(a|s)
- CRITIC: Value network V_φ(s)

THE ADVANTAGE FUNCTION:
    A(s,a) = Q(s,a) - V(s)
           = r + γV(s') - V(s)  (TD estimate)

"How much better is this action than average?"

UPDATE RULES:
    Actor:  θ ← θ + α × ∇log π_θ(a|s) × A(s,a)
    Critic: φ ← φ - β × ∇(r + γV(s') - V_φ(s))²

===============================================================
A3C = ASYNCHRONOUS + A2C
===============================================================

Multiple workers collect experience in parallel.
Aggregated gradients update shared network.
WHY: More diverse experience, faster wall-clock time.
```

**Ablations:**
- Actor vs Critic learning rate ratio
- N-step returns (TD(n))
- Generalized Advantage Estimation (GAE) λ
- Number of parallel workers (A3C)

---

### 30_ppo.py — Paradigm: STABLE POLICY OPTIMIZATION
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Policy gradient, but CLIPPED to prevent too-large updates.

THE PPO OBJECTIVE:
    L(θ) = E[min(r_t(θ) × Aₜ, clip(r_t(θ), 1-ε, 1+ε) × Aₜ)]

where r_t(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)

"Move toward better actions, but not too far from old policy"

===============================================================
WHY CLIPPING WORKS
===============================================================

1. If advantage > 0 (good action):
   - Want to increase probability
   - But clip prevents ratio > 1+ε

2. If advantage < 0 (bad action):
   - Want to decrease probability
   - But clip prevents ratio < 1-ε

This creates a TRUST REGION without the complexity of TRPO.

===============================================================
IMPLEMENTATION TRICKS
===============================================================

1. Multiple epochs on same batch (unlike vanilla PG)
2. Value function clipping (optional)
3. Advantage normalization
4. Entropy bonus for exploration
```

**Ablations:**
- Clip ratio ε effect
- Number of update epochs
- GAE lambda
- Value loss coefficient
- Entropy coefficient

---

### 31_sac.py — Paradigm: MAXIMUM ENTROPY RL
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Maximize reward AND entropy (exploration).

THE MAXIMUM ENTROPY OBJECTIVE:
    J(π) = Σ E[r(s,a) + α × H(π(·|s))]

"Get high reward, but also stay random (explore)"

WHY ENTROPY BONUS?
1. Prevents premature convergence to suboptimal policy
2. Better exploration
3. More robust policies
4. Multiple near-optimal solutions are averaged

===============================================================
THE SAC COMPONENTS
===============================================================

1. Soft Q-function: Q(s,a) includes entropy
2. Soft value function: V(s) = E[Q(s,a) - α log π(a|s)]
3. Policy: Gaussian, outputs mean + std
4. Automatic temperature α tuning

===============================================================
REPARAMETERIZATION TRICK
===============================================================

Instead of sampling a ~ π(a|s):
    a = μ(s) + σ(s) × ε,  where ε ~ N(0,1)

This makes the gradient flow through the sample!
```

**Ablations:**
- Fixed vs learned temperature α
- Replay buffer size
- Target update rate (soft vs hard)
- Continuous vs discrete action spaces

---

### 32_model_based_rl.py — Paradigm: WORLD MODEL
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Learn a model of the environment, then plan.

LEARN:
    f_θ(s, a) → s', r  (transition model)

PLAN:
    Use model to simulate trajectories
    Optimize policy in imagination

===============================================================
DYNA-Q: SIMPLE MODEL-BASED
===============================================================

1. Act in real environment, store (s,a,r,s')
2. Update Q with real experience
3. Also update Q with SIMULATED experience from model
4. More planning steps = faster learning (if model is good)

===============================================================
MODEL PREDICTIVE CONTROL (MPC)
===============================================================

At each step:
1. Sample action sequences
2. Simulate with learned model
3. Execute first action of best sequence
4. Repeat

No policy network! Pure planning.

===============================================================
THE MODEL ERROR PROBLEM
===============================================================

Model errors COMPOUND over long horizons.
Small error per step → huge error over trajectory.

SOLUTIONS:
1. Short planning horizons
2. Ensemble of models (uncertainty)
3. Dyna: Mix real and simulated experience
```

**Ablations:**
- Planning horizon length
- Real vs simulated experience ratio
- Model ensemble size
- Model architecture (deterministic vs stochastic)

---

### 33_multi_agent_rl.py — Paradigm: GAME THEORY + RL
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Multiple agents learning simultaneously.
Each agent's optimal policy depends on others.

SETTINGS:
1. COOPERATIVE — shared reward, common goal
2. COMPETITIVE — zero-sum, adversarial
3. MIXED — both cooperation and competition

===============================================================
THE NON-STATIONARITY PROBLEM
===============================================================

From agent A's perspective:
- Environment includes other agents
- Other agents are learning (changing)
- Environment is non-stationary!

Agent A learns → Agent B adapts → Agent A's learning invalid

===============================================================
APPROACHES
===============================================================

1. INDEPENDENT LEARNERS
   Each agent ignores others, treat as environment noise
   Simple but unstable

2. CENTRALIZED TRAINING, DECENTRALIZED EXECUTION (CTDE)
   Train with global info, execute with local
   QMIX, MADDPG, etc.

3. SELF-PLAY
   Agent plays against copies of itself
   How AlphaGo was trained
```

**Environments:**
- Cooperative navigation
- Predator-prey
- Simple adversarial games

**Ablations:**
- Independent vs centralized critic
- Communication channels
- Self-play vs population training

---

### 34_inverse_rl.py — Paradigm: LEARN REWARD FROM BEHAVIOR
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Given expert demonstrations, recover the reward function.

WHY?
- Reward engineering is HARD
- Experts demonstrate WHAT to do, not HOW to reward

THE AMBIGUITY PROBLEM:
Many reward functions explain the same behavior!
- R(s) = 0 explains everything
- Need to find "most likely" reward

===============================================================
MAXIMUM ENTROPY IRL
===============================================================

Principle: Expert is optimal w.r.t. some reward,
          but also maximizes entropy (random tiebreaking)

P(trajectory) ∝ exp(Σ r(s,a))

Learn r(s) such that expert trajectories are most likely.

===============================================================
GENERATIVE ADVERSARIAL IMITATION LEARNING (GAIL)
===============================================================

GAN for imitation:
- Generator: policy π (tries to match expert)
- Discriminator: D (distinguishes expert vs policy)

Policy learns to fool discriminator.
No explicit reward recovery — directly imitates.
```

**Ablations:**
- Number of expert demonstrations
- Expert optimality (noisy vs perfect)
- Feature representation for reward
- GAIL vs behavioral cloning vs IRL

---

## EXPANSION: Graph Learning (35-44)

### 35_graph_fundamentals.py — Paradigm: RELATIONAL DATA
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Data with RELATIONSHIPS between entities.

GRAPH G = (V, E)
- V: nodes (entities)
- E: edges (relationships)

Node features: X ∈ R^(n × d)
Adjacency matrix: A ∈ {0,1}^(n × n)

TASKS:
1. NODE CLASSIFICATION — predict label per node
2. LINK PREDICTION — predict if edge exists
3. GRAPH CLASSIFICATION — predict label per graph

===============================================================
WHY SPECIAL ARCHITECTURE?
===============================================================

Graphs have no fixed size, no fixed order!
- Can't flatten into vector (loses structure)
- Can't use CNN (no grid)
- Need PERMUTATION INVARIANT/EQUIVARIANT operations

KEY INSIGHT: Aggregate information from neighbors
    h_v = f(x_v, AGGREGATE({x_u : u ∈ N(v)}))
```

**Datasets:**
- Zachary's Karate Club
- Cora citation network
- Molecular graphs

---

### 36_gcn.py — Paradigm: SPECTRAL CONVOLUTION
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Convolution on graphs via the GRAPH LAPLACIAN.

GRAPH LAPLACIAN:
    L = D - A  (unnormalized)
    L = I - D^(-1/2) A D^(-1/2)  (normalized)

Eigenvalues of L = graph frequencies
Eigenvectors = Fourier basis for graphs

THE GCN LAYER:
    H' = σ(Ã H W)

where Ã = D^(-1/2)(A + I)D^(-1/2)  (normalized adjacency + self-loop)

WHAT IT DOES:
Each node aggregates features from neighbors (and itself).
This is SMOOTHING — connected nodes become more similar.

===============================================================
WHY ADD SELF-LOOPS?
===============================================================

Without self-loop: node's own features can be washed out
With self-loop: node retains its own information

===============================================================
OVER-SMOOTHING PROBLEM
===============================================================

Deep GCN → all nodes converge to same representation!
WHY? Repeated smoothing = information diffusion
After k layers: node sees k-hop neighborhood
Too deep = see entire graph = everything looks the same
```

**Ablations:**
- Number of layers vs over-smoothing
- With/without self-loops
- Different normalizations
- Spectral vs spatial interpretation

---

### 37_graphsage.py — Paradigm: INDUCTIVE AGGREGATION
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Learn to aggregate neighbor features (not fixed weights per node).

THE GRAPHSAGE FRAMEWORK:
    h_v^(k) = σ(W × CONCAT(h_v^(k-1), AGG({h_u^(k-1) : u ∈ N(v)})))

AGGREGATORS:
1. MEAN: average neighbor features
2. POOL: max-pool after MLP
3. LSTM: treat neighbors as sequence (order shouldn't matter though!)

===============================================================
WHY GRAPHSAGE > GCN?
===============================================================

1. INDUCTIVE: Can generalize to unseen nodes/graphs
   GCN learns weights per node position
   GraphSAGE learns aggregation function

2. SAMPLING: Don't need full neighborhood
   Sample k neighbors → scalable to huge graphs

3. MINIBATCH: Can train on subgraphs
```

**Ablations:**
- Mean vs pool vs LSTM aggregator
- Neighborhood sample size
- Number of aggregation layers
- Supervised vs unsupervised (random walk) loss

---

### 38_gat.py — Paradigm: ATTENTION ON GRAPHS
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Not all neighbors are equally important. Learn attention weights!

THE GAT LAYER:
    h_i' = σ(Σ_j α_ij W h_j)

where attention α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))

WHAT THE ATTENTION DOES:
- [Wh_i || Wh_j]: concatenate transformed features
- a^T [...]: learned attention mechanism
- softmax: normalize over neighbors
- Result: weighted combination of neighbor features

===============================================================
MULTI-HEAD ATTENTION
===============================================================

Like Transformers: multiple attention heads
    h_i' = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)

Different heads can focus on different relationship types.

===============================================================
GAT vs GCN
===============================================================

GCN: Fixed weights based on degree (1/sqrt(d_i × d_j))
GAT: LEARNED weights based on features

GAT is more expressive but has more parameters.
```

**Ablations:**
- Number of attention heads
- Attention dropout
- GCN vs GAT on same tasks
- Visualize attention weights

---

### 39_gin.py — Paradigm: MAXIMALLY EXPRESSIVE (WL Test)
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Make GNNs as powerful as the Weisfeiler-Lehman graph isomorphism test.

THE WL TEST:
Iteratively refine node labels based on neighbor labels.
Two graphs are non-isomorphic if WL gives different labelings.

THE GIN LAYER:
    h_v^(k) = MLP((1 + ε) × h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))

WHY THIS FORM?
Theorem: GIN is as powerful as WL test (provably maximal for GNNs).
The (1 + ε) term distinguishes node's own features from neighbors.

===============================================================
SUM vs MEAN vs MAX
===============================================================

For INJECTIVE aggregation (distinguish different multisets):
- SUM: works! Different multisets → different sums (for most inputs)
- MEAN: loses count information {1,1,1} vs {1}
- MAX: loses all but maximum

GIN uses SUM + MLP for maximal expressiveness.
```

**Ablations:**
- Sum vs mean vs max aggregation
- Learnable ε vs fixed
- Depth vs expressiveness
- Compare on graph isomorphism tasks

---

### 40_mpnn.py — Paradigm: MESSAGE PASSING (Unified Framework)
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

General framework that encompasses GCN, GraphSAGE, GAT, GIN, etc.

THE MESSAGE PASSING FRAMEWORK:
    m_v^(t+1) = Σ_{u∈N(v)} M_t(h_v^t, h_u^t, e_vu)   (message)
    h_v^(t+1) = U_t(h_v^t, m_v^(t+1))                 (update)

M_t: Message function
U_t: Update function
e_vu: Edge features

INSTANTIATIONS:
- GCN: M = h_u / sqrt(d_v × d_u), U = σ(W × sum)
- GAT: M = α_vu × W h_u, U = σ(sum)
- GIN: M = h_u, U = MLP((1+ε)h_v + sum)

===============================================================
EDGE FEATURES
===============================================================

Many graphs have edge attributes (bond type, distance, etc.)
MPNN naturally incorporates: M_t(h_v, h_u, e_vu)

Graph Transformer can also handle: attention = f(q_v, k_u, e_vu)
```

**Ablations:**
- Different message functions
- Different update functions
- With/without edge features
- Gated updates (GRU-style)

---

### 41_graph_pooling.py — Paradigm: HIERARCHICAL GRAPHS
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

How to go from node representations to graph representation?

SIMPLE POOLING:
- SUM: Σ_v h_v (size-sensitive)
- MEAN: (1/|V|) Σ_v h_v (size-invariant)
- MAX: max_v h_v (captures salient features)

HIERARCHICAL POOLING:
Learn to coarsen the graph:
    Graph → Cluster → Smaller Graph → ... → Final embedding

===============================================================
DIFFPOOL
===============================================================

Learn soft cluster assignments:
    S^(l) = softmax(GNN(A^(l), X^(l)))  (assignment matrix)
    X^(l+1) = S^(l)T X^(l)               (cluster features)
    A^(l+1) = S^(l)T A^(l) S^(l)         (cluster adjacency)

End-to-end differentiable graph coarsening!

===============================================================
TOP-K POOLING
===============================================================

Select top-k nodes based on learned scores:
    y = σ(X p / ||p||)  (node scores)
    idx = top-k(y)      (select indices)
    X' = X[idx] ⊙ y[idx] (gate by score)
```

**Ablations:**
- Global sum vs mean vs max
- DiffPool vs TopK vs SAGPool
- Number of pooling layers
- Cluster size ratio

---

### 42_hetero_gnn.py — Paradigm: MULTIPLE NODE/EDGE TYPES
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Real graphs often have DIFFERENT types of nodes and edges.
- Knowledge graphs: (entity, relation, entity)
- Citation networks: (paper, cites, paper), (author, writes, paper)
- Social networks: (user, follows, user), (user, posts, content)

HETEROGENEOUS GRAPH:
    G = (V, E, τ_v, τ_e)
    τ_v: node type function
    τ_e: edge type function

===============================================================
RELATIONAL GCN (R-GCN)
===============================================================

Different weights for different edge types:

    h_v' = σ(Σ_r Σ_{u∈N_r(v)} (1/c_{v,r}) W_r h_u + W_0 h_v)

Where r indexes relation types, N_r(v) = neighbors via relation r.

PARAMETER EXPLOSION PROBLEM:
|relations| × d × d parameters per layer!

SOLUTIONS:
1. Basis decomposition: W_r = Σ_b a_rb B_b
2. Block diagonal: W_r = diag(W_r^1, ..., W_r^B)

===============================================================
HETEROGENEOUS ATTENTION (HAN)
===============================================================

Use attention to weight different relation paths (metapaths).
Different metapaths capture different semantics.
```

**Ablations:**
- Relation-specific vs shared weights
- Metapath selection
- Attention over relations
- Knowledge graph completion task

---

### 43_temporal_gnn.py — Paradigm: DYNAMIC GRAPHS
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Graphs that change over time:
- Edges appear/disappear
- Node features evolve
- New nodes join

APPROACHES:

1. DISCRETE TIME (Snapshots)
   G_1, G_2, ..., G_T
   Process each snapshot, combine with RNN/Transformer

2. CONTINUOUS TIME
   Events: (u, v, t, type)
   Model as temporal point process

===============================================================
TEMPORAL GRAPH NETWORKS (TGN)
===============================================================

Key components:
1. MESSAGE FUNCTION: encode interaction event
2. MEMORY: node-level memory updated at each interaction
3. EMBEDDING: combine memory with neighbor info

Memory update:
    m_i(t) = msg(s_i(t-), s_j(t-), Δt, e_ij)
    s_i(t) = GRU(s_i(t-), m_i(t))

===============================================================
TEMPORAL ATTENTION
===============================================================

Attention over neighbors, weighted by TIME ENCODING:

    α_ij(t) = softmax_j(q_i^T [k_j || Φ(t - t_j)])

Where Φ is time encoding (like positional encoding).
```

**Ablations:**
- Discrete vs continuous time
- Memory vs no memory
- Time encoding schemes
- Link prediction over time

---

### 44_graph_transformer.py — Paradigm: FULL ATTENTION ON GRAPHS
```
===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Apply Transformer attention to graphs.
Unlike GNNs that aggregate from neighbors, attend to ALL nodes.

THE CHALLENGE:
Transformers are permutation-equivariant — need structure!

SOLUTIONS:

1. POSITIONAL ENCODINGS FOR GRAPHS
   - Laplacian eigenvectors (spectral position)
   - Random walk landing probabilities
   - Distance encodings

2. STRUCTURAL ENCODINGS
   - Degree as bias
   - Shortest path distance
   - Substructure counts

===============================================================
ATTENTION BIAS FROM STRUCTURE
===============================================================

Instead of: Attention(Q, K, V) = softmax(QK^T/√d) V

Use: Attention = softmax(QK^T/√d + B) V

Where B_ij encodes structural relationship between nodes i and j.
- B_ij = distance(i, j) embedding
- B_ij = edge features if (i,j) ∈ E

===============================================================
GRAPHORMER
===============================================================

Three types of encodings:
1. Centrality encoding: add degree embedding to node features
2. Spatial encoding: attention bias from shortest path distance
3. Edge encoding: attention bias from edge features along path

Performance: State-of-the-art on molecular property prediction!

===============================================================
GRAPH vs VANILLA TRANSFORMER
===============================================================

Vanilla Transformer:
- O(n²) attention to all positions
- Positional encoding for sequence order

Graph Transformer:
- O(n²) attention to all nodes (expensive for large graphs!)
- Structural encoding for graph topology
- Optional: sparse attention (only neighbors)

TRADE-OFF:
- Full attention: maximum expressiveness, O(n²)
- Sparse (neighbor) attention: scales, but limited receptive field

===============================================================
WHEN TO USE
===============================================================

Graph Transformer excels when:
1. Small-medium graphs (< 5000 nodes)
2. Long-range dependencies matter
3. Rich structural features available

Stick with GNN when:
1. Large graphs (> 100k nodes)
2. Local interactions dominate
3. Computational budget is limited
```

**Ablations:**
- With/without positional encoding
- Laplacian PE vs random walk PE
- Full attention vs sparse (neighbor) attention
- Attention bias types
- Compare to GCN/GAT on same tasks

---

## Implementation Checklist

### For each algorithm:
- [ ] Docstring with PARADIGM, CORE IDEA, EQUATIONS, INDUCTIVE BIAS
- [ ] Clean NumPy implementation (no PyTorch/TensorFlow)
- [ ] `ablation_experiments()` function
- [ ] `benchmark_on_datasets()` function
- [ ] `visualize_*()` function(s)
- [ ] Main block with explanation and summary
- [ ] Save visualizations as PNG

### Testing:
- [ ] Run each file standalone
- [ ] Verify ablations show expected behavior
- [ ] Compare performance across algorithms in arena

---

## File Naming Convention

```
{number}_{algorithm_name}.py
{number}_{algorithm_name}_{visualization_type}.png
```

Examples:
- `25_rl_fundamentals.py`
- `26_q_learning.py`
- `27_dqn.py` → `27_dqn_learning_curve.png`, `27_dqn_replay_ablation.png`
- `36_gcn.py` → `36_gcn_oversmoothing.png`
- `44_graph_transformer.py` → `44_graph_transformer_attention.png`

---

## Environments & Datasets

### RL Environments (implement in 25_rl_fundamentals.py):
```python
class GridWorld:
    """Simple navigation grid with walls and goal."""

class CartPole:
    """Classic control: balance pole on cart."""

class MultiArmedBandit:
    """K-armed bandit for exploration experiments."""
```

### Graph Datasets (implement in 35_graph_fundamentals.py):
```python
def karate_club():
    """Zachary's karate club — classic small graph."""

def cora():
    """Citation network — node classification."""

def molecular_graphs():
    """Small molecules — graph classification."""

def synthetic_graphs():
    """Generated graphs for controlled experiments."""
```

---

## Dependencies

Core (already used):
- numpy
- matplotlib

May need for RL visualizations:
- gymnasium (optional, can implement own envs)

All graph algorithms should work with just numpy + matplotlib.
