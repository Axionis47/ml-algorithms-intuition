"""
INFORMATION THEORY — Paradigm: UNCERTAINTY MEASUREMENT

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Information theory quantifies UNCERTAINTY and INFORMATION.

You already use it everywhere in ML:
- Cross-entropy loss in classification (02_logistic_regression.py)
- KL divergence in t-SNE (60_tsne.py) and VAE (18_vae.py)
- Entropy in decision tree splits (07_decision_tree.py)
- Mutual information in feature selection

This file builds the concepts from scratch so you can REASON
about them, not just use them as a loss function.

THE KEY INSIGHT:
    "Information is the RESOLUTION OF UNCERTAINTY."
    — Claude Shannon, 1948

If you already know the outcome → no information gained.
If the outcome is surprising → lots of information gained.

===============================================================
THE MATHEMATICS
===============================================================

ENTROPY (average uncertainty):
    H(X) = -Σ p(x) log p(x)

    Measures: How many bits do you need to encode a sample?
    Maximum: H = log(K) when all K outcomes are equally likely
    Minimum: H = 0 when outcome is certain

CROSS-ENTROPY (encoding with wrong distribution):
    H(P,Q) = -Σ p(x) log q(x)

    If you use distribution Q to encode samples from P,
    you need H(P,Q) bits. Always ≥ H(P).
    Excess bits = KL divergence.

    THIS IS YOUR CLASSIFICATION LOSS:
        L = -Σ y_i log(ŷ_i)  ← cross-entropy between labels & predictions

KL DIVERGENCE (how different are P and Q?):
    KL(P||Q) = Σ p(x) log(p(x)/q(x))
             = H(P,Q) - H(P)

    Properties:
    - KL ≥ 0 (Gibbs' inequality)
    - KL(P||Q) ≠ KL(Q||P) → NOT symmetric! NOT a metric!
    - KL(P||Q) = 0 ⟺ P = Q

    ASYMMETRY INTUITION:
    KL(P||Q): "Cost of using Q when truth is P"
    KL(Q||P): "Cost of using P when truth is Q"
    When P has mass where Q doesn't → KL(P||Q) = ∞

JENSEN-SHANNON DIVERGENCE (symmetric version):
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Properties:
    - SYMMETRIC: JSD(P||Q) = JSD(Q||P)
    - BOUNDED: 0 ≤ JSD ≤ log(2)
    - sqrt(JSD) is a proper metric!

MUTUAL INFORMATION (shared information):
    I(X;Y) = Σ_{x,y} p(x,y) log(p(x,y) / (p(x)p(y)))
           = KL(P(X,Y) || P(X)P(Y))
           = H(X) + H(Y) - H(X,Y)

    "How much does knowing X tell you about Y?"
    I(X;Y) = 0 when X and Y are independent.

===============================================================
INDUCTIVE BIAS — What Information Theory Assumes
===============================================================

1. DISCRETE DISTRIBUTIONS — continuous requires differential entropy
   (different properties, can be negative)

2. LOG BASE MATTERS — log2 = bits, ln = nats, log10 = bans
   We use natural log (nats) throughout, like ML frameworks

3. 0 log 0 = 0 — by convention (limit as p→0)

4. KL IS NOT A DISTANCE — asymmetric, doesn't satisfy triangle inequality
   If you need a distance, use JSD or Wasserstein

5. ENTROPY ≠ IMPORTANCE — high entropy means uncertain,
   not necessarily useless (uniform prior in Bayesian = maximum ignorance)

===============================================================
WHERE IT SHOWS UP IN THIS REPO
===============================================================

- 60_tsne.py: KL(P||Q) is the OBJECTIVE (match high-D to low-D)
- 18_vae.py: ELBO = reconstruction - KL(q(z|x) || p(z))
- 02_logistic_regression.py: Binary cross-entropy loss
- 16_gmm.py: Log-likelihood = -cross_entropy + const
- 07_decision_tree.py: Information gain = parent_entropy - child_entropy
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CORE FUNCTIONS
# ============================================================

def entropy(p):
    """
    Shannon Entropy: H(X) = -Σ p(x) log p(x)

    Measures the average uncertainty (in nats) of distribution p.

    Args:
        p: Probability distribution (1D array, must sum to 1)

    Returns:
        H: Entropy in nats (use / log(2) for bits)
    """
    p = np.asarray(p, dtype=float)
    # Filter out zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def cross_entropy(p, q):
    """
    Cross-Entropy: H(P,Q) = -Σ p(x) log q(x)

    The expected number of nats needed to encode samples from P
    using a code optimized for Q.

    In classification: p = one-hot labels, q = model predictions.

    Args:
        p: True distribution
        q: Model distribution (must be > 0 where p > 0)

    Returns:
        H(P,Q): Cross-entropy in nats
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Avoid log(0) by clipping q
    q = np.clip(q, 1e-15, 1.0)
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))


def kl_divergence(p, q):
    """
    KL Divergence: KL(P||Q) = Σ p(x) log(p(x)/q(x))

    Measures information LOST when using Q to approximate P.
    NOT symmetric: KL(P||Q) ≠ KL(Q||P).

    Args:
        p: True distribution
        q: Approximate distribution

    Returns:
        KL(P||Q): KL divergence in nats (≥ 0)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    q = np.clip(q, 1e-15, 1.0)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def jsd(p, q):
    """
    Jensen-Shannon Divergence: JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
    where M = 0.5*(P+Q)

    Symmetric, bounded [0, log(2)], sqrt(JSD) is a metric.

    Args:
        p, q: Probability distributions

    Returns:
        JSD(P||Q): Jensen-Shannon divergence in nats
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mutual_information(joint_pxy):
    """
    Mutual Information: I(X;Y) = Σ_{x,y} p(x,y) log(p(x,y) / (p(x)p(y)))

    Measures how much knowing X reduces uncertainty about Y.
    I(X;Y) = 0 iff X and Y are independent.

    Args:
        joint_pxy: Joint probability table P(X,Y) as 2D array
                   shape (|X|, |Y|), must sum to 1

    Returns:
        I(X;Y): Mutual information in nats (≥ 0)
    """
    joint = np.asarray(joint_pxy, dtype=float)
    # Marginals
    px = joint.sum(axis=1)  # P(X)
    py = joint.sum(axis=0)  # P(Y)

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
    return mi


def make_distribution(name, n=10):
    """
    Create named probability distributions for experiments.

    Args:
        name: One of 'uniform', 'peaked', 'bimodal', 'skewed', 'sparse'
        n: Number of bins (support size)

    Returns:
        p: Normalized probability vector of length n
    """
    if name == 'uniform':
        p = np.ones(n)
    elif name == 'peaked':
        # Most mass on one outcome
        p = np.ones(n) * 0.01
        p[n // 2] = 10.0
    elif name == 'bimodal':
        p = np.ones(n) * 0.1
        p[n // 4] = 5.0
        p[3 * n // 4] = 5.0
    elif name == 'skewed':
        # Exponentially decaying
        p = np.exp(-np.linspace(0, 3, n))
    elif name == 'sparse':
        # Only a few outcomes have probability
        p = np.zeros(n)
        p[0] = 3.0
        p[n // 3] = 1.0
        p[2 * n // 3] = 0.5
    else:
        raise ValueError(f"Unknown distribution: {name}")

    return p / p.sum()


def make_joint_distribution(correlation, n=5, random_state=42):
    """
    Create a joint distribution P(X,Y) with controlled correlation.

    Args:
        correlation: Float in [0, 1]. 0 = independent, 1 = perfectly correlated.
        n: Support size for each variable
        random_state: For reproducibility

    Returns:
        joint: (n, n) array summing to 1
    """
    rng = np.random.RandomState(random_state)

    # Independent component: px * py
    px = make_distribution('skewed', n)
    py = make_distribution('skewed', n)
    independent = np.outer(px, py)

    # Correlated component: mass on diagonal
    diagonal = np.zeros((n, n))
    for i in range(n):
        diagonal[i, i] = px[i]
    diagonal /= diagonal.sum()

    # Blend
    joint = (1 - correlation) * independent + correlation * diagonal
    # Add small noise for numerical stability
    joint += rng.rand(n, n) * 1e-6
    joint /= joint.sum()

    return joint


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    5 ablations exploring information theory concepts.
    """
    np.random.seed(42)
    print("=" * 70)
    print("INFORMATION THEORY — ABLATION EXPERIMENTS")
    print("=" * 70)

    # --------------------------------------------------------
    # ABLATION 1: Entropy vs Number of Classes
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 1: Entropy vs Number of Classes (Uniform)")
    print("=" * 60)
    print("\nFor a UNIFORM distribution over K classes:")
    print("H = log(K) — entropy grows logarithmically with K.\n")

    ks = [2, 3, 5, 10, 20, 50, 100]
    for k in ks:
        p_uniform = np.ones(k) / k
        h = entropy(p_uniform)
        h_max = np.log(k)
        print(f"  K={k:3d}:  H = {h:.4f} nats  (theoretical max = {h_max:.4f})")

    print("\n  KEY INSIGHT: Uniform distribution = MAXIMUM entropy = maximum uncertainty.")
    print("  Any other distribution over K classes has LESS entropy (more structured).")

    # Non-uniform comparison
    print("\n  Comparison at K=10:")
    for name in ['uniform', 'peaked', 'bimodal', 'skewed', 'sparse']:
        p = make_distribution(name, 10)
        h = entropy(p)
        print(f"    {name:10s}: H = {h:.4f} nats  (max = {np.log(10):.4f})")

    # --------------------------------------------------------
    # ABLATION 2: KL Divergence Asymmetry
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 2: KL Divergence Asymmetry")
    print("=" * 60)
    print("\nKL(P||Q) vs KL(Q||P) — they can be VERY different!\n")

    # Create P (peaked) and Q (uniform-ish)
    n = 10
    P = make_distribution('peaked', n)
    Q = make_distribution('uniform', n)

    kl_pq = kl_divergence(P, Q)
    kl_qp = kl_divergence(Q, P)
    jsd_val = jsd(P, Q)

    print(f"  P = peaked distribution (most mass in center)")
    print(f"  Q = uniform distribution")
    print(f"  KL(P||Q) = {kl_pq:.4f} nats  (cost of using Q when truth is P)")
    print(f"  KL(Q||P) = {kl_qp:.4f} nats  (cost of using P when truth is Q)")
    print(f"  Ratio:     {kl_pq/kl_qp:.2f}x")
    print(f"  JSD(P,Q) = {jsd_val:.4f} nats  (symmetric!)")

    print("\n  WHY THE ASYMMETRY?")
    print("  KL(P||Q): P puts mass where Q is flat → moderate penalty")
    print("  KL(Q||P): Q puts mass where P is near-zero → HUGE log(q/p) penalty")

    # More pairs
    print("\n  More examples:")
    pairs = [
        ('peaked', 'uniform', "Peaked vs Uniform"),
        ('bimodal', 'uniform', "Bimodal vs Uniform"),
        ('skewed', 'uniform', "Skewed vs Uniform"),
        ('peaked', 'bimodal', "Peaked vs Bimodal"),
    ]
    for name_p, name_q, label in pairs:
        p = make_distribution(name_p, n)
        q = make_distribution(name_q, n)
        print(f"    {label:25s}: KL(P||Q)={kl_divergence(p,q):.4f}  "
              f"KL(Q||P)={kl_divergence(q,p):.4f}  "
              f"JSD={jsd(p,q):.4f}")

    # --------------------------------------------------------
    # ABLATION 3: Cross-Entropy as Loss Function
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 3: Cross-Entropy vs MSE as Loss")
    print("=" * 60)
    print("\nWhen model is CONFIDENT but WRONG:")
    print("Cross-entropy penalizes exponentially harder than MSE.\n")

    # True label: class 0 with probability 1
    p_true = np.array([1.0, 0.0])

    print(f"  True label: [1, 0]  (class 0)")
    print(f"  {'Prediction':20s}  {'CE Loss':>10s}  {'MSE Loss':>10s}  {'CE/MSE':>10s}")
    print(f"  {'-'*55}")

    predictions = [
        [0.9, 0.1],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.01, 0.99],
    ]

    for pred in predictions:
        q_pred = np.array(pred)
        ce = cross_entropy(p_true, q_pred)
        mse = np.mean((p_true - q_pred) ** 2)
        ratio = ce / mse if mse > 0 else float('inf')
        print(f"  [{pred[0]:.2f}, {pred[1]:.2f}]          {ce:10.4f}  {mse:10.4f}  {ratio:10.2f}")

    print("\n  KEY INSIGHT: When model says [0.01, 0.99] but truth is [1, 0]:")
    print("  - MSE = 0.98 (just a number)")
    print("  - CE  = 4.61 (HUGE — -log(0.01) penalty)")
    print("  Cross-entropy PUNISHES confident wrong predictions much harder.")
    print("  This is WHY cross-entropy is the standard classification loss.")

    # --------------------------------------------------------
    # ABLATION 4: Mutual Information vs Correlation
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 4: Mutual Information vs Correlation")
    print("=" * 60)
    print("\nI(X;Y) = 0 when independent, increases with dependence.\n")

    correlations = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"  {'Correlation':>12s}  {'MI (nats)':>10s}  {'H(X)':>8s}  {'H(Y)':>8s}  {'H(X,Y)':>8s}")
    print(f"  {'-'*55}")

    for corr in correlations:
        joint = make_joint_distribution(corr, n=5)
        mi = mutual_information(joint)
        hx = entropy(joint.sum(axis=1))
        hy = entropy(joint.sum(axis=0))
        hxy = entropy(joint.ravel())
        print(f"  {corr:12.1f}  {mi:10.4f}  {hx:8.4f}  {hy:8.4f}  {hxy:8.4f}")

    print("\n  KEY INSIGHT: I(X;Y) = H(X) + H(Y) - H(X,Y)")
    print("  Independent: H(X,Y) = H(X) + H(Y) → MI = 0")
    print("  Perfect correlation: H(X,Y) = H(X) = H(Y) → MI = H(X)")
    print("  MI measures shared information between variables.")

    # --------------------------------------------------------
    # ABLATION 5: JSD vs KL — Symmetry and Boundedness
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 5: JSD vs KL — Symmetry and Boundedness")
    print("=" * 60)

    # Sweep: gradually move Q from P to very different
    n = 20
    P = make_distribution('peaked', n)
    print(f"\n  P = peaked distribution")
    print(f"  Q = blend of peaked (alpha) and uniform (1-alpha)")
    print(f"\n  {'alpha':>6s}  {'KL(P||Q)':>10s}  {'KL(Q||P)':>10s}  {'JSD':>10s}  {'sqrt(JSD)':>10s}")
    print(f"  {'-'*55}")

    Q_uniform = make_distribution('uniform', n)
    for alpha in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
        Q = alpha * P + (1 - alpha) * Q_uniform
        kl_forward = kl_divergence(P, Q)
        kl_backward = kl_divergence(Q, P)
        jsd_val = jsd(P, Q)
        print(f"  {alpha:6.1f}  {kl_forward:10.4f}  {kl_backward:10.4f}  "
              f"{jsd_val:10.4f}  {np.sqrt(jsd_val):10.4f}")

    print(f"\n  Maximum possible JSD = log(2) = {np.log(2):.4f}")
    print(f"  JSD is always in [0, {np.log(2):.4f}] — BOUNDED.")
    print(f"  KL can be unbounded (→ ∞ when Q=0 where P>0).")
    print(f"  sqrt(JSD) satisfies the triangle inequality → proper METRIC.")

    return True


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_entropy_distributions():
    """
    Visualize different distributions and their entropy values.
    Shows that entropy measures the "spread" of a distribution.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Entropy of Different Distributions\n'
                 'Higher entropy = more uncertain = more spread out',
                 fontsize=14, fontweight='bold')

    n = 20
    dist_names = ['uniform', 'peaked', 'bimodal', 'skewed', 'sparse']
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    max_entropy = np.log(n)

    for idx, (name, color) in enumerate(zip(dist_names, colors)):
        ax = axes[idx // 3, idx % 3]
        p = make_distribution(name, n)
        h = entropy(p)

        ax.bar(range(n), p, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name.capitalize()}\nH = {h:.3f} nats ({h/max_entropy*100:.0f}% of max)',
                     fontsize=11)
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, max(p) * 1.2)

    # Last subplot: entropy vs K for uniform
    ax = axes[1, 2]
    ks = np.arange(2, 101)
    entropies = [np.log(k) for k in ks]
    ax.plot(ks, entropies, color='#2196F3', linewidth=2)
    ax.set_title('Max Entropy vs K\n(Uniform over K outcomes)', fontsize=11)
    ax.set_xlabel('Number of outcomes (K)')
    ax.set_ylabel('H = log(K) [nats]')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/68_entropy_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 68_entropy_distributions.png")
    return fig


def visualize_kl_asymmetry():
    """
    Visualize KL divergence asymmetry and JSD comparison.
    The critical insight: KL(P||Q) ≠ KL(Q||P).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('KL Divergence Asymmetry & Jensen-Shannon Divergence',
                 fontsize=14, fontweight='bold')

    n = 20
    pairs = [
        ('peaked', 'uniform'),
        ('bimodal', 'uniform'),
        ('peaked', 'bimodal'),
    ]

    for col, (name_p, name_q) in enumerate(pairs):
        P = make_distribution(name_p, n)
        Q = make_distribution(name_q, n)

        kl_pq = kl_divergence(P, Q)
        kl_qp = kl_divergence(Q, P)
        jsd_val = jsd(P, Q)

        # Top row: distributions overlaid
        ax = axes[0, col]
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width/2, P, width, label='P', color='#2196F3', alpha=0.7)
        ax.bar(x + width/2, Q, width, label='Q', color='#F44336', alpha=0.7)
        ax.set_title(f'P={name_p}, Q={name_q}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylabel('Probability')

        # Bottom row: divergence values
        ax = axes[1, col]
        vals = [kl_pq, kl_qp, jsd_val]
        labels = ['KL(P||Q)', 'KL(Q||P)', 'JSD(P,Q)']
        colors = ['#2196F3', '#F44336', '#4CAF50']
        bars = ax.bar(labels, vals, color=colors, alpha=0.7, edgecolor='black')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Divergence (nats)')
        ax.set_title(f'Ratio: KL(P||Q)/KL(Q||P) = {kl_pq/kl_qp:.2f}x' if kl_qp > 0 else '',
                     fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/68_kl_asymmetry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 68_kl_asymmetry.png")
    return fig


def visualize_mutual_information():
    """
    Visualize mutual information via joint distribution heatmaps.
    Shows how MI increases as X and Y become more correlated.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Mutual Information: How Correlation Affects Shared Information',
                 fontsize=14, fontweight='bold')

    correlations = [0.0, 0.3, 0.7, 1.0]
    n = 8

    for col, corr in enumerate(correlations):
        joint = make_joint_distribution(corr, n=n)
        mi = mutual_information(joint)
        hx = entropy(joint.sum(axis=1))
        hy = entropy(joint.sum(axis=0))

        # Top row: joint distribution heatmap
        ax = axes[0, col]
        im = ax.imshow(joint, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_title(f'corr = {corr:.1f}\nI(X;Y) = {mi:.3f} nats', fontsize=11)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: marginals
        ax = axes[1, col]
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width/2, px, width, label='P(X)', color='#2196F3', alpha=0.7)
        ax.bar(x + width/2, py, width, label='P(Y)', color='#F44336', alpha=0.7)
        ax.set_title(f'H(X)={hx:.2f}, H(Y)={hy:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability')

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/68_mutual_information.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 68_mutual_information.png")
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  INFORMATION THEORY — Paradigm: UNCERTAINTY MEASUREMENT")
    print("  Entropy, KL Divergence, Cross-Entropy, Mutual Information")
    print("=" * 70)

    # Run ablation experiments
    ablation_experiments()

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    visualize_entropy_distributions()
    visualize_kl_asymmetry()
    visualize_mutual_information()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Information Theory in ML")
    print("=" * 70)
    print("""
    CONCEPT          | WHAT IT MEASURES              | WHERE IN ML
    -----------------+-------------------------------+---------------------------
    Entropy H(X)     | Average uncertainty            | Decision tree splits
    Cross-Entropy    | Encoding cost (wrong model)    | Classification loss
    KL Divergence    | Distribution difference (asym) | VAE, t-SNE objective
    JSD              | Distribution difference (sym)  | GAN training (JS-GAN)
    Mutual Info      | Shared information             | Feature selection

    KEY TAKEAWAYS:
    1. Entropy is maximized by uniform distributions (maximum ignorance)
    2. Cross-entropy ≥ entropy — using wrong model always costs more
    3. KL divergence is NOT symmetric — direction matters!
       KL(P||Q): penalizes where P has mass but Q doesn't (mode-seeking)
       KL(Q||P): penalizes where Q has mass but P doesn't (mean-seeking)
    4. JSD is symmetric, bounded, and sqrt(JSD) is a proper metric
    5. MI = 0 for independent variables, increases with dependence
    6. Cross-entropy loss works better than MSE for classification because
       it penalizes confident-wrong predictions exponentially harder
    """)
