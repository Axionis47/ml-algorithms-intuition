"""
FEATURE IMPORTANCE — Paradigm: GLOBAL EXPLANATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Which features does the model RELY ON? Three approaches:

1. PERMUTATION IMPORTANCE (model-agnostic):
   Shuffle one feature → measure performance drop.
   Big drop = important feature.

2. MEAN DECREASE IMPURITY (tree-specific):
   Sum up impurity reduction from each feature across all splits.
   Used internally by random forests.

3. PARTIAL DEPENDENCE (model-agnostic):
   Vary one feature, average prediction over all other features.
   Shows the MARGINAL EFFECT of each feature.

ALL THREE answer: "What features matter?"
But they answer in DIFFERENT WAYS and can DISAGREE.

===============================================================
THE MATHEMATICS
===============================================================

PERMUTATION IMPORTANCE:
    I_j = score(y, f(X)) - score(y, f(X_permuted_j))

    Where X_permuted_j = X with column j randomly shuffled.
    If feature j matters, shuffling it destroys information → score drops.

    Properties:
    - Model-agnostic (works with any model)
    - Accounts for feature interactions (to some extent)
    - Biased when features are CORRELATED (both get low importance)

MEAN DECREASE IMPURITY (MDI):
    I_j = Σ_{nodes using feature j} (n_node / n_total) × Δimpurity

    Where Δimpurity = impurity_parent - weighted_avg(impurity_children)
    For classification: impurity = Gini or entropy
    For regression: impurity = variance

    Properties:
    - Fast (computed during training)
    - Biased toward HIGH-CARDINALITY features
    - Biased toward features with many possible split points

PARTIAL DEPENDENCE:
    f̂_j(x_j) = (1/n) Σ_i f(x_j, x_{-j}^(i))

    "What is the average prediction if we SET feature j to x_j,
     keeping everything else as-is?"

    Properties:
    - Shows DIRECTION of effect (positive/negative)
    - Assumes features are INDEPENDENT (marginal effect)
    - Can miss INTERACTIONS between features

===============================================================
INDUCTIVE BIAS
===============================================================

1. PERMUTATION assumes features are INDEPENDENT
   - Correlated features: importance is split between them
   - Can underestimate importance of redundant features

2. MDI is biased toward high-cardinality
   - Continuous features > categorical (more split points)
   - Random noise features can appear important!

3. PARTIAL DEPENDENCE assumes INDEPENDENCE
   - Shows marginal effect, not conditional
   - Can show impossible feature combinations
   - For interactions: use ICE plots or SHAP

4. ALL methods require a TRAINED MODEL
   - Importance is model-specific (linear vs tree vs NN)
   - Same feature can be important for one model, not another

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SIMPLE MODELS FOR DEMONSTRATION
# ============================================================

class SimpleDecisionTree:
    """
    Minimal decision tree for feature importance demonstration.
    Builds axis-aligned splits, tracks feature usage.
    """

    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.mean(y == c)
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        n = len(y)
        parent_gini = self._gini(y)

        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, j], np.linspace(10, 90, 15))

            for t in thresholds:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) < 2 or len(right) < 2:
                    continue

                gain = parent_gini - (len(left)/n * self._gini(left) +
                                      len(right)/n * self._gini(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': np.mean(y), 'n': len(y)}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'value': np.mean(y), 'n': len(y)}

        # Track feature importance (weighted impurity decrease)
        self.feature_importances_[feature] += gain * len(y)

        left_mask = X[:, feature] <= threshold
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'gain': gain,
            'n': len(y),
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.tree = self._build(X, y, 0)
        # Normalize
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([1 if self._predict_one(x, self.tree) > 0.5 else 0
                        for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class SimpleRandomForest:
    """
    Minimal random forest: ensemble of decision trees with bagging.
    """

    def __init__(self, n_trees=20, max_depth=5, max_features='sqrt',
                 random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n, d = X.shape
        self.trees = []

        n_features = int(np.sqrt(d)) if self.max_features == 'sqrt' else d
        self.feature_importances_ = np.zeros(d)

        for _ in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        self.feature_importances_ /= self.n_trees
        return self

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        return (np.mean(preds, axis=0) > 0.5).astype(int)

    def predict_proba(self, X):
        probas = np.array([t.predict_proba(X) for t in self.trees])
        return np.mean(probas, axis=0)


class SimpleLogisticRegression:
    """Minimal logistic regression for comparison."""

    def __init__(self, lr=0.1, n_iter=500):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            error = pred - y
            self.weights -= self.lr * (X.T @ error) / n
            self.bias -= self.lr * np.mean(error)
        return self

    def predict(self, X):
        z = X @ self.weights + self.bias
        return (1 / (1 + np.exp(-np.clip(z, -500, 500))) > 0.5).astype(int)

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ============================================================
# FEATURE IMPORTANCE METHODS
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def permutation_importance(model, X, y, n_repeats=10, scoring=accuracy,
                           random_state=42):
    """
    Permutation Importance — model-agnostic.

    Shuffle feature j → measure performance drop.

    Args:
        model: Fitted model with predict() method
        X: Test features
        y: Test labels
        n_repeats: Number of shuffles per feature
        scoring: Scoring function (higher = better)

    Returns:
        importances_mean: Mean importance per feature
        importances_std: Std of importance per feature
    """
    np.random.seed(random_state)
    baseline_score = scoring(y, model.predict(X))

    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))

    for r in range(n_repeats):
        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_score = scoring(y, model.predict(X_perm))
            importances[r, j] = baseline_score - perm_score

    return importances.mean(axis=0), importances.std(axis=0)


def partial_dependence(model, X, feature_idx, grid_size=50):
    """
    Partial Dependence Plot — model-agnostic.

    For each value of feature j:
        Set ALL samples' feature j to that value
        Average the predictions

    Args:
        model: Fitted model with predict_proba() method
        X: Data
        feature_idx: Which feature to vary
        grid_size: Number of grid points

    Returns:
        grid_values: Feature values tested
        pdp_values: Average prediction at each grid point
    """
    feature_values = X[:, feature_idx]
    grid_values = np.linspace(feature_values.min(), feature_values.max(), grid_size)

    pdp_values = np.zeros(grid_size)

    for i, val in enumerate(grid_values):
        X_modified = X.copy()
        X_modified[:, feature_idx] = val
        proba = model.predict_proba(X_modified)
        # Handle both 1D (logistic regression) and 2D (RF) predict_proba
        if proba.ndim == 2:
            pdp_values[i] = np.mean(proba[:, 1])
        else:
            pdp_values[i] = np.mean(proba)

    return grid_values, pdp_values


# ============================================================
# DATASETS
# ============================================================

def make_importance_dataset(n_samples=500, random_state=42):
    """
    Dataset where we KNOW which features matter.

    Features:
        0: x1 — IMPORTANT (main signal)
        1: x2 — IMPORTANT (additive with x1)
        2: x3 — NOISE (random)
        3: x4 — NOISE (random)
        4: x5 — CORRELATED WITH x1 (redundant)

    y = 1 if (x1 + x2 > 0) else 0
    """
    np.random.seed(random_state)

    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)  # Pure noise
    x4 = np.random.randn(n_samples)  # Pure noise
    x5 = x1 + np.random.randn(n_samples) * 0.3  # Correlated with x1

    X = np.column_stack([x1, x2, x3, x4, x5])
    y = (x1 + x2 > 0).astype(int)

    feature_names = ['x1 (signal)', 'x2 (signal)', 'x3 (noise)',
                     'x4 (noise)', 'x5 (corr w/ x1)']

    return X, y, feature_names


def make_nonlinear_dataset(n_samples=500, random_state=42):
    """
    Dataset with nonlinear importance.

    y = 1 if x1² + x2² < 1 (circle boundary)
    x3-x5 are noise
    """
    np.random.seed(random_state)

    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.randn(n_samples)
    x4 = np.random.randn(n_samples)
    x5 = np.random.randn(n_samples)

    X = np.column_stack([x1, x2, x3, x4, x5])
    y = (x1**2 + x2**2 < 1.5).astype(int)

    feature_names = ['x1 (circle)', 'x2 (circle)', 'x3 (noise)',
                     'x4 (noise)', 'x5 (noise)']

    return X, y, feature_names


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What each importance method reveals (and hides).
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # -------- Experiment 1: Permutation vs MDI --------
    print("\n1. PERMUTATION vs MDI IMPORTANCE")
    print("-" * 40)
    print("Do they agree? (Spoiler: not always)")

    X, y, names = make_importance_dataset(500)
    # Train/test split
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    rf = SimpleRandomForest(n_trees=30, max_depth=6).fit(X_train, y_train)

    perm_imp, perm_std = permutation_importance(rf, X_test, y_test)

    print(f"   {'Feature':<20} {'MDI':<10} {'Permutation':<12}")
    print(f"   {'-'*42}")
    for i, name in enumerate(names):
        print(f"   {name:<20} {rf.feature_importances_[i]:<10.3f} {perm_imp[i]:<12.3f}")

    print("→ MDI may overrate correlated/high-cardinality features")
    print("→ Permutation importance is more reliable but noisier")

    # -------- Experiment 2: Correlated Features --------
    print("\n2. EFFECT OF FEATURE CORRELATION")
    print("-" * 40)
    print("x5 is correlated with x1. How does this affect importance?")

    # Train with and without x5
    rf_with = SimpleRandomForest(n_trees=30, max_depth=6).fit(X_train, y_train)
    rf_without = SimpleRandomForest(n_trees=30, max_depth=6).fit(
        X_train[:, :4], y_train)

    perm_with, _ = permutation_importance(rf_with, X_test, y_test)
    perm_without, _ = permutation_importance(rf_without, X_test[:, :4], y_test)

    print("   With x5 (correlated):")
    print(f"     x1 importance: {perm_with[0]:.3f}")
    print(f"     x5 importance: {perm_with[4]:.3f}")
    print("   Without x5:")
    print(f"     x1 importance: {perm_without[0]:.3f}")
    print("→ Correlation SPLITS importance between correlated features")
    print("→ Remove redundant features or use grouped permutation")

    # -------- Experiment 3: Linear vs Nonlinear Importance --------
    print("\n3. LINEAR vs NONLINEAR MODEL — Different Importance!")
    print("-" * 40)

    X_nl, y_nl, names_nl = make_nonlinear_dataset(500)
    X_train_nl, X_test_nl = X_nl[:400], X_nl[400:]
    y_train_nl, y_test_nl = y_nl[:400], y_nl[400:]

    lr = SimpleLogisticRegression().fit(X_train_nl, y_train_nl)
    rf_nl = SimpleRandomForest(n_trees=30, max_depth=6).fit(X_train_nl, y_train_nl)

    perm_lr, _ = permutation_importance(lr, X_test_nl, y_test_nl)
    perm_rf, _ = permutation_importance(rf_nl, X_test_nl, y_test_nl)

    print(f"   {'Feature':<20} {'LR perm':<10} {'RF perm':<10}")
    for i, name in enumerate(names_nl):
        print(f"   {name:<20} {perm_lr[i]:<10.3f} {perm_rf[i]:<10.3f}")

    acc_lr = accuracy(y_test_nl, lr.predict(X_test_nl))
    acc_rf = accuracy(y_test_nl, rf_nl.predict(X_test_nl))
    print(f"   LR accuracy: {acc_lr:.3f}, RF accuracy: {acc_rf:.3f}")
    print("→ Importance is MODEL-SPECIFIC, not dataset-specific")
    print("→ LR can't capture circle boundary → features seem unimportant")

    # -------- Experiment 4: Importance Stability --------
    print("\n4. IMPORTANCE STABILITY (Bootstrap)")
    print("-" * 40)

    X, y, names = make_importance_dataset(500)
    n_bootstrap = 10
    all_importances = []

    for b in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_b, y_b = X[idx], y[idx]
        rf_b = SimpleRandomForest(n_trees=20, max_depth=5,
                                  random_state=b).fit(X_b[:400], y_b[:400])
        perm_b, _ = permutation_importance(rf_b, X_b[400:], y_b[400:])
        all_importances.append(perm_b)

    all_importances = np.array(all_importances)
    print(f"   {'Feature':<20} {'Mean':<10} {'Std':<10} {'Stable?'}")
    for i, name in enumerate(names):
        mean_imp = all_importances[:, i].mean()
        std_imp = all_importances[:, i].std()
        stable = "✓" if std_imp < 0.5 * abs(mean_imp + 1e-10) else "✗"
        print(f"   {name:<20} {mean_imp:<10.3f} {std_imp:<10.3f} {stable}")

    print("→ Signal features have stable importance")
    print("→ Noise features fluctuate around zero")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_importance_comparison():
    """
    THE KEY INSIGHT: MDI vs Permutation importance side by side.
    """
    np.random.seed(42)

    X, y, names = make_importance_dataset(500)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    rf = SimpleRandomForest(n_trees=30, max_depth=6).fit(X_train, y_train)
    perm_imp, perm_std = permutation_importance(rf, X_test, y_test, n_repeats=20)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x_pos = np.arange(len(names))

    # MDI
    ax = axes[0]
    colors = ['green' if 'signal' in n else ('orange' if 'corr' in n else 'gray')
              for n in names]
    ax.barh(x_pos, rf.feature_importances_, color=colors, alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title('Mean Decrease Impurity (MDI)\n(Tree-specific, fast)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    # Permutation
    ax = axes[1]
    ax.barh(x_pos, perm_imp, xerr=perm_std, color=colors, alpha=0.7, capsize=3)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance (accuracy drop)')
    ax.set_title('Permutation Importance\n(Model-agnostic, reliable)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    # Ground truth
    ax = axes[2]
    true_importance = [1.0, 1.0, 0.0, 0.0, 0.5]  # Approximate
    ax.barh(x_pos, true_importance, color=colors, alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title('Ground Truth\n(We know the data-generating process)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('FEATURE IMPORTANCE: MDI vs Permutation vs Truth\n'
                 'Green = signal | Orange = correlated | Gray = noise',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_partial_dependence():
    """
    Partial dependence plots: how each feature affects predictions.
    Linear vs Random Forest comparison.
    """
    np.random.seed(42)

    X, y, names = make_importance_dataset(500)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    lr = SimpleLogisticRegression().fit(X_train, y_train)
    rf = SimpleRandomForest(n_trees=30, max_depth=6).fit(X_train, y_train)

    fig, axes = plt.subplots(2, len(names), figsize=(3 * len(names), 8))

    for j, name in enumerate(names):
        # Logistic Regression PDP
        ax = axes[0, j]
        grid_vals, pdp_vals = partial_dependence(lr, X_test, j)
        ax.plot(grid_vals, pdp_vals, 'b-', linewidth=2)
        ax.set_title(f'{name}', fontsize=9)
        ax.set_ylabel('Avg Prediction' if j == 0 else '')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel('LogReg\nAvg Prediction', fontsize=10)

        # Random Forest PDP
        ax = axes[1, j]
        grid_vals, pdp_vals = partial_dependence(rf, X_test, j)
        ax.plot(grid_vals, pdp_vals, 'r-', linewidth=2)
        ax.set_xlabel('Feature Value')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel('RF\nAvg Prediction', fontsize=10)

    plt.suptitle('PARTIAL DEPENDENCE PLOTS\n'
                 'Top: Logistic Regression (linear) | Bottom: Random Forest (nonlinear)\n'
                 'Signal features show clear trends, noise features are flat',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_importance_stability():
    """
    Bootstrap confidence intervals for importance estimates.
    """
    np.random.seed(42)

    X, y, names = make_importance_dataset(600)

    n_bootstrap = 15
    all_importances = []

    for b in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_b, y_b = X[idx], y[idx]
        rf = SimpleRandomForest(n_trees=20, max_depth=5,
                                random_state=b).fit(X_b[:450], y_b[:450])
        perm_imp, _ = permutation_importance(rf, X_b[450:], y_b[450:], n_repeats=5)
        all_importances.append(perm_imp)

    all_importances = np.array(all_importances)

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot([all_importances[:, i] for i in range(len(names))],
                     vert=False, patch_artist=True, tick_labels=names)

    colors = ['green' if 'signal' in n else ('orange' if 'corr' in n else 'gray')
              for n in names]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero importance')
    ax.set_xlabel('Permutation Importance (accuracy drop)', fontsize=11)
    ax.set_title('IMPORTANCE STABILITY: Bootstrap Confidence Intervals\n'
                 'Signal features are consistently positive, noise fluctuates around zero',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("FEATURE IMPORTANCE — Paradigm: GLOBAL EXPLANATION")
    print("="*60)

    print("""
WHAT THIS IS:
    Three ways to ask "which features matter?"
    1. Permutation: shuffle feature, measure damage
    2. MDI: track impurity reduction at tree splits
    3. Partial Dependence: vary feature, average prediction

KEY INSIGHT:
    Different methods give DIFFERENT answers!
    Understand the biases of each method.

INDUCTIVE BIAS:
    - Permutation: assumes feature independence
    - MDI: biased toward high-cardinality features
    - PDP: shows marginal (not conditional) effects
    """)

    # Run ablations
    ablation_experiments()

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_importance_comparison()
    save_path1 = '/Users/sid47/ML Algorithms/66_importance_comparison.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_partial_dependence()
    save_path2 = '/Users/sid47/ML Algorithms/66_partial_dependence.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_importance_stability()
    save_path3 = '/Users/sid47/ML Algorithms/66_importance_stability.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Feature Importance Reveals")
    print("="*60)
    print("""
1. PERMUTATION IMPORTANCE is the most reliable
   → Model-agnostic, measures actual performance drop
   → But: biased with correlated features

2. MDI is fast but BIASED
   → Overestimates high-cardinality features
   → Can rank random noise features highly!

3. PARTIAL DEPENDENCE shows effect direction
   → Linear model: straight lines
   → Nonlinear model: curves and thresholds

4. CORRELATED FEATURES are a trap
   → Importance splits between correlated features
   → Both look less important than they are

5. IMPORTANCE is MODEL-SPECIFIC
   → Same feature can be important for RF, not for LR
   → Because different models use features differently

CONNECTIONS:
    → 07_decision_tree: MDI computed during training
    → 09_random_forest: feature_importances_ attribute
    → 01_linear_regression: coefficients ≈ linear importance

NEXT: 67_local_explanations.py — Why THIS specific prediction?
      (SHAP values from game theory + LIME)
    """)
