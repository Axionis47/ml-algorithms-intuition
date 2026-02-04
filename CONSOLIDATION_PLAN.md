# ML Algorithms — Consolidation Plan

## Overview

Restructure the repository from 48 flat files into **8 cohesive phases**, each with:
- A comprehensive **README.md** that walks through visualizations and concepts
- Grouped algorithm implementations
- Organized ablations and visualizations
- Cross-algorithm comparisons within each phase

---

## Proposed Directory Structure

```
ML-Algorithms/
│
├── README.md                          # Master overview + learning roadmap
├── shared/
│   ├── datasets.py                    # Challenge datasets (from 00_datasets.py)
│   └── utils.py                       # Common utilities
│
├── 01-foundations/
│   ├── README.md                      # The building blocks of ML
│   ├── algorithms/
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   └── naive_bayes.py
│   └── visualizations/
│       ├── linear_regression.png
│       ├── logistic_regression.png
│       ├── knn_boundaries.png
│       ├── knn_k_effect.png
│       └── naive_bayes.png
│
├── 02-bayesian-methods/
│   ├── README.md                      # Thinking in distributions
│   ├── algorithms/
│   │   ├── gaussian_process.py
│   │   └── bayesian_linear_reg.py
│   └── visualizations/
│       ├── gp_boundaries.png
│       ├── gp_lengthscale.png
│       ├── gp_uncertainty.png
│       ├── bayesian_linear_reg.png
│       └── blr_alpha.png
│
├── 03-trees-and-ensembles/
│   ├── README.md                      # From single splits to forests
│   ├── algorithms/
│   │   ├── decision_tree.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   ├── gradient_boosting.py
│   │   └── adaboost.py
│   └── visualizations/
│       └── ...
│
├── 04-neural-networks/
│   ├── README.md                      # Learning representations
│   ├── algorithms/
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── rnn_lstm.py
│   │   └── transformer.py
│   └── visualizations/
│       └── ...
│
├── 05-generative-models/
│   ├── README.md                      # Modeling the data distribution
│   ├── algorithms/
│   │   ├── gmm.py
│   │   ├── hmm.py
│   │   └── vae.py
│   └── visualizations/
│       └── ...
│
├── 06-uncertainty-quantification/
│   ├── README.md                      # Knowing what you don't know
│   ├── algorithms/
│   │   ├── conformal_prediction.py
│   │   ├── mc_dropout.py
│   │   ├── calibration.py
│   │   ├── online_learning.py
│   │   └── maml.py
│   └── visualizations/
│       └── ...
│
├── 07-reinforcement-learning/
│   ├── README.md                      # Learning from interaction
│   ├── algorithms/
│   │   ├── bandits.py
│   │   ├── q_learning.py
│   │   ├── dqn.py
│   │   ├── policy_gradient.py
│   │   ├── actor_critic.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── model_based_rl.py
│   │   ├── multi_agent_rl.py
│   │   └── inverse_rl.py
│   └── visualizations/
│       └── ... (26 visualization files)
│
├── 08-graph-learning/
│   ├── README.md                      # Learning on relational data
│   ├── algorithms/
│   │   ├── graph_fundamentals.py
│   │   ├── gcn.py
│   │   ├── graphsage.py
│   │   ├── gat.py
│   │   ├── gin.py
│   │   ├── mpnn.py
│   │   ├── graph_pooling.py
│   │   ├── hetero_gnn.py
│   │   ├── temporal_gnn.py
│   │   └── graph_transformer.py
│   └── visualizations/
│       └── ... (10 visualization files)
│
└── arena/
    ├── README.md                      # Cross-phase benchmarking
    ├── arena.py
    └── arena.png
```

---

## Phase README Structure

Each phase README will follow this template for **strong grouping with visualization walkthroughs**:

```markdown
# Phase X: [Theme Name]

> **One-line philosophy**: [What unifies these algorithms]

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Algorithm Progression](#algorithm-progression)
3. [Visual Walkthrough](#visual-walkthrough)
4. [Key Equations](#key-equations)
5. [Ablation Insights](#ablation-insights)
6. [When to Use What](#when-to-use-what)
7. [Common Pitfalls](#common-pitfalls)
8. [Hands-on Exercises](#hands-on-exercises)

## The Big Picture
[2-3 paragraphs explaining the paradigm shift this phase represents]

## Algorithm Progression
[Flowchart/diagram showing how algorithms build on each other]

## Visual Walkthrough
### Algorithm 1: [Name]
![Description](visualizations/algo1.png)
**What you're seeing:** [Explanation of the visualization]
**Key insight:** [The "aha" moment]

### Algorithm 2: [Name]
...

## Ablation Insights
### What happens when you change X?
![Ablation](visualizations/algo_ablation.png)
**Finding:** [Concrete insight from ablation]

## When to Use What
| Scenario | Best Choice | Why |
|----------|-------------|-----|
| ... | ... | ... |

## Common Pitfalls
1. **Pitfall**: [Description]
   **Solution**: [Fix]

## Hands-on Exercises
1. [ ] Run algorithm X on dataset Y
2. [ ] Modify hyperparameter Z and observe...
```

---

## Detailed Phase Content

### Phase 1: Foundations
**Theme**: "The Four Paradigms of Learning"

| Algorithm | Paradigm | Core Question |
|-----------|----------|---------------|
| Linear Regression | PROJECTION | "What line best fits?" |
| Logistic Regression | PROBABILITY | "What's the probability of class 1?" |
| KNN | MEMORY | "What are similar examples doing?" |
| Naive Bayes | INDEPENDENCE | "What do features suggest independently?" |

**README Highlights**:
- Visualization: Decision boundaries side-by-side for all 4 on same dataset
- Ablation: KNN's k effect on boundary smoothness
- Comparison table: Assumptions, computational cost, interpretability

---

### Phase 2: Bayesian Methods
**Theme**: "Uncertainty as a First-Class Citizen"

| Algorithm | Paradigm | Core Question |
|-----------|----------|---------------|
| Gaussian Process | FUNCTION DISTRIBUTION | "What functions could explain this data?" |
| Bayesian Linear Reg | POSTERIOR OVER WEIGHTS | "What weights are plausible given data?" |

**README Highlights**:
- Visualization: GP uncertainty bands expanding away from data
- Visualization: Prior → Posterior weight distributions
- Ablation: Lengthscale effect on GP smoothness
- Key insight: "Uncertainty increases where we have no data"

---

### Phase 3: Trees and Ensembles
**Theme**: "From Axis-Aligned Splits to Ensemble Wisdom"

| Algorithm | Paradigm | Builds On |
|-----------|----------|-----------|
| Decision Tree | AXIS-ALIGNED SPLITS | - |
| SVM | MAXIMUM MARGIN | - |
| Random Forest | ENSEMBLE OF TREES | Decision Tree |
| Gradient Boosting | SEQUENTIAL CORRECTION | Decision Tree |
| AdaBoost | WEIGHTED RESAMPLING | Decision Tree |

**README Highlights**:
- Visualization: Single tree vs forest boundaries (dramatic improvement)
- Visualization: Boosting iterations showing residual fitting
- Ablation: Tree depth → overfitting visualization
- Comparison: Bagging (RF) vs Boosting (GB) philosophies

---

### Phase 4: Neural Networks
**Theme**: "Learning Features, Not Engineering Them"

| Algorithm | Paradigm | Specialization |
|-----------|----------|----------------|
| MLP | LEARNED FEATURES | Tabular data |
| CNN | SPATIAL HIERARCHY | Images, grids |
| RNN/LSTM | SEQUENTIAL MEMORY | Sequences |
| Transformer | ATTENTION | Long-range dependencies |

**README Highlights**:
- Visualization: MLP learning XOR (non-linear boundary)
- Visualization: CNN filters and feature maps
- Visualization: Attention patterns in Transformer
- Ablation: Activation functions (ReLU vs sigmoid vs tanh)
- Evolution story: MLP → CNN (translation invariance) → RNN (sequences) → Transformer (parallelism + attention)

---

### Phase 5: Generative Models
**Theme**: "Modeling Where Data Comes From"

| Algorithm | Paradigm | Generates |
|-----------|----------|-----------|
| GMM | MIXTURE MODEL | Cluster assignments + samples |
| HMM | HIDDEN STATE SEQUENCE | State sequences |
| VAE | LATENT COMPRESSION | New samples via latent space |

**README Highlights**:
- Visualization: GMM fitting multi-modal data
- Visualization: HMM state transitions and emissions
- Visualization: VAE latent space interpolation
- Key insight: "Generative models answer P(X), not P(Y|X)"

---

### Phase 6: Uncertainty Quantification
**Theme**: "Knowing What You Don't Know"

| Algorithm | Paradigm | Uncertainty Type |
|-----------|----------|------------------|
| Conformal Prediction | COVERAGE GUARANTEE | Finite-sample valid |
| MC Dropout | APPROXIMATE BAYESIAN | Epistemic |
| Calibration | PROBABILITY CORRECTNESS | Reliability |
| Online Learning | STREAMING UPDATES | Adaptation |
| MAML | LEARNING TO LEARN | Fast adaptation |

**README Highlights**:
- Visualization: Conformal prediction intervals with coverage
- Visualization: MC Dropout uncertainty vs distance from training data
- Visualization: Calibration curves (before vs after)
- Ablation: Number of MC samples vs uncertainty estimate stability
- Practical guide: "Which uncertainty method for your use case?"

---

### Phase 7: Reinforcement Learning
**Theme**: "Learning from Interaction"

**Sub-sections**:

#### 7.1 Exploration & Bandits
| Algorithm | Key Concept |
|-----------|-------------|
| Bandits | Explore vs Exploit |

#### 7.2 Tabular Methods
| Algorithm | Key Concept |
|-----------|-------------|
| Q-Learning | Value iteration without model |

#### 7.3 Deep Value Methods
| Algorithm | Key Concept |
|-----------|-------------|
| DQN | Q-function as neural network |

#### 7.4 Policy Methods
| Algorithm | Key Concept |
|-----------|-------------|
| Policy Gradient | Direct policy optimization |
| Actor-Critic | Value + Policy |
| PPO | Stable policy updates |
| SAC | Maximum entropy |

#### 7.5 Advanced RL
| Algorithm | Key Concept |
|-----------|-------------|
| Model-Based | Learn the world model |
| Multi-Agent | Game theory + RL |
| Inverse RL | Learn reward from demos |

**README Highlights**:
- Visualization: Bandit regret curves (ε-greedy vs UCB vs Thompson)
- Visualization: Q-value heatmaps on GridWorld
- Visualization: DQN learning curve with/without replay
- Visualization: Policy gradient variance reduction with baseline
- Visualization: PPO clipping effect
- Flowchart: "Which RL algorithm should I use?"
- Ablation: γ (discount) effect on myopic vs farsighted behavior

---

### Phase 8: Graph Learning
**Theme**: "When Data Has Structure"

**Sub-sections**:

#### 8.1 Fundamentals
- Graph representations, adjacency matrix, Laplacian

#### 8.2 Spectral Methods
| Algorithm | Key Concept |
|-----------|-------------|
| GCN | Spectral convolution simplified |

#### 8.3 Spatial Methods
| Algorithm | Key Concept |
|-----------|-------------|
| GraphSAGE | Inductive, sampling |
| GAT | Attention on neighbors |
| GIN | Maximally expressive |

#### 8.4 Unified Framework
| Algorithm | Key Concept |
|-----------|-------------|
| MPNN | Message passing abstraction |

#### 8.5 Advanced
| Algorithm | Key Concept |
|-----------|-------------|
| Graph Pooling | Hierarchical graphs |
| Hetero GNN | Multiple node/edge types |
| Temporal GNN | Dynamic graphs |
| Graph Transformer | Full attention on graphs |

**README Highlights**:
- Visualization: GCN over-smoothing with depth
- Visualization: GAT attention weights on Karate Club
- Visualization: GIN vs GCN expressiveness
- Visualization: Graph pooling hierarchy
- Comparison table: GCN vs GraphSAGE vs GAT vs GIN
- Ablation: Number of layers vs over-smoothing

---

## Implementation Plan

### Phase 1: Setup (Creates structure)
```bash
# Create directories
mkdir -p ML-Algorithms/{shared,arena}
mkdir -p ML-Algorithms/0{1..8}-*/algorithms
mkdir -p ML-Algorithms/0{1..8}-*/visualizations
```

### Phase 2: Move files
```bash
# Example for foundations
mv 01_linear_regression.py ML-Algorithms/01-foundations/algorithms/linear_regression.py
mv 01_linear_regression.png ML-Algorithms/01-foundations/visualizations/
# ... repeat for all files
```

### Phase 3: Create READMEs
Write comprehensive README.md for each phase following the template.

### Phase 4: Update imports
Update any cross-file imports to use the new structure.

### Phase 5: Create master README
Write the root README.md with learning roadmap and navigation.

---

## README Writing Guidelines

### For Visualizations
Every visualization in the README should have:

1. **The image** (embedded)
2. **What you're seeing** (1-2 sentences describing the plot)
3. **Key insight** (the "aha" moment)
4. **Code to reproduce** (optional, link to file + function)

Example:
```markdown
### Decision Boundary Evolution with Tree Depth

![Tree depth effect](visualizations/decision_tree_depth.png)

**What you're seeing:** Decision boundaries for depths 1, 3, 5, and unlimited on the moons dataset.

**Key insight:** Shallow trees underfit (linear-ish boundaries), deep trees overfit (jagged boundaries that memorize noise). Depth 3-5 is often the sweet spot.

**Reproduce:** `python algorithms/decision_tree.py --ablation depth`
```

### For Ablations
Each ablation should answer:
1. What parameter/component are we varying?
2. What do we expect to happen?
3. What actually happens? (with visualization)
4. What's the practical takeaway?

### For Comparisons
Use tables with clear criteria:
```markdown
| Algorithm | Handles Non-linear | Interpretable | Scales to 1M samples |
|-----------|-------------------|---------------|---------------------|
| Linear Reg | ❌ | ✅ | ✅ |
| KNN | ✅ | ✅ | ❌ |
| ...
```

---

## Estimated Effort

| Task | Files | Estimated Work |
|------|-------|----------------|
| Create directory structure | - | 10 min |
| Move Python files | 48 | 30 min |
| Move PNG files | 70 | 20 min |
| Update imports | ~10 files | 1 hour |
| Write Phase 1 README | 1 | 1 hour |
| Write Phase 2 README | 1 | 45 min |
| Write Phase 3 README | 1 | 1 hour |
| Write Phase 4 README | 1 | 1 hour |
| Write Phase 5 README | 1 | 45 min |
| Write Phase 6 README | 1 | 1 hour |
| Write Phase 7 README | 1 | 2 hours |
| Write Phase 8 README | 1 | 2 hours |
| Write Master README | 1 | 1 hour |
| Testing & fixes | - | 1 hour |

**Total: ~13 hours**

---

## Success Criteria

After consolidation:

1. ✅ Each phase folder is self-contained and runnable
2. ✅ Each README tells a complete story with visualizations
3. ✅ A newcomer can follow Phase 1 → 8 as a learning path
4. ✅ Cross-algorithm comparisons exist within each phase
5. ✅ Ablations are documented with insights, not just plots
6. ✅ "When to use what" guidance in each phase
7. ✅ All original functionality preserved

---

## Next Steps

1. **Review this plan** — Any phases to merge/split differently?
2. **Approve structure** — Happy with 8 phases?
3. **Start execution** — Phase by phase or all at once?

Ready to proceed when you are.
