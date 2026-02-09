"""
LOCAL EXPLANATIONS — Paradigm: LOCAL EXPLANATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

WHY did the model make THIS specific prediction?

Feature importance (66) tells us GLOBAL patterns.
Local explanations tell us about ONE PREDICTION.

TWO APPROACHES:

1. SHAPLEY VALUES (from game theory)
   Each feature is a "player" in a "game" (the prediction).
   Shapley value = each player's fair share of the payout.

   The ONLY method satisfying all fairness axioms:
   - EFFICIENCY: contributions sum to (prediction - baseline)
   - SYMMETRY: identical features get equal credit
   - NULL PLAYER: irrelevant features get zero credit
   - LINEARITY: additive over models

2. LIME (Local Interpretable Model-agnostic Explanations)
   Fit a simple model (linear) LOCALLY around the prediction.
   The simple model explains the complex one's behavior nearby.

===============================================================
THE MATHEMATICS
===============================================================

SHAPLEY VALUES:
    phi_i = SUM over S subset of N\\{i}:
        (|S|! * (|N|-|S|-1)! / |N|!) * [f(S + {i}) - f(S)]

    In words: Average marginal contribution of feature i
    across ALL possible coalitions (subsets) of other features.

    EXPONENTIAL COST: O(2^n) subsets for n features.
    APPROXIMATION: Sample random coalitions.

    For a prediction f(x) with baseline E[f(X)]:
        f(x) = E[f(X)] + phi_1 + phi_2 + ... + phi_n

LIME:
    For instance x to explain:
    1. Sample neighbors z around x (perturbed versions)
    2. Get model predictions f(z) for each neighbor
    3. Weight neighbors by proximity: pi(z) = exp(-d(x,z)^2 / sigma^2)
    4. Fit weighted linear model: g(z) = w^T z + b
    5. Coefficients w = local feature importance

===============================================================
INDUCTIVE BIAS
===============================================================

1. SHAPLEY: The ONLY fair attribution method (game theory proves it)
   But: exponential cost, needs sampling, assumes feature independence

2. LIME: Assumes locally linear decision boundary
   Simple but: kernel width sigma is critical and hard to choose

3. BACKGROUND DATA matters for Shapley:
   "Feature absent" = replace with background distribution
   Different backgrounds = different explanations!

4. BOTH are model-agnostic: work with ANY black-box model

===============================================================
CONNECTIONS TO OTHER FILES
===============================================================

- 66_feature_importance.py: global importance (this file = local)
- 33_multi_agent_rl.py: Shapley values come from cooperative game theory
- 01_linear_regression.py: LIME fits a local linear model
  (if model IS linear, SHAP recovers exact coefficients)
- 09_random_forest.py: the black box we explain here
- 12_mlp.py: neural network explanations
===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


# ============================================================
# SIMPLE MODELS (self-contained for explanations)
# ============================================================

class SimpleDecisionTree:
    """Minimal decision tree for use as explainee model."""

    def __init__(self, max_depth=5, min_samples=5, random_state=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n, d = X.shape
        if depth >= self.max_depth or n < self.min_samples or len(np.unique(y)) == 1:
            counts = np.bincount(y.astype(int), minlength=self.n_classes)
            return {'leaf': True, 'proba': counts / counts.sum()}

        best_gain, best_feat, best_thr = -1, 0, 0
        parent_gini = 1 - np.sum((np.bincount(y.astype(int), minlength=self.n_classes) / n) ** 2)

        for feat in range(d):
            thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 10))
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) < 2 or len(right) < 2:
                    continue
                gl = 1 - np.sum((np.bincount(left.astype(int), minlength=self.n_classes) / len(left)) ** 2)
                gr = 1 - np.sum((np.bincount(right.astype(int), minlength=self.n_classes) / len(right)) ** 2)
                gain = parent_gini - (len(left) / n) * gl - (len(right) / n) * gr
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr

        if best_gain <= 0:
            counts = np.bincount(y.astype(int), minlength=self.n_classes)
            return {'leaf': True, 'proba': counts / counts.sum()}

        left_mask = X[:, best_feat] <= best_thr
        return {
            'leaf': False, 'feature': best_feat, 'threshold': best_thr,
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['proba']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])


class SimpleRandomForest:
    """Minimal random forest as the black box to explain."""

    def __init__(self, n_trees=20, max_depth=5, max_features='sqrt', random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.trees = []
        self.feature_subsets = []

    def fit(self, X, y):
        n, d = X.shape
        if self.max_features == 'sqrt':
            m = max(1, int(np.sqrt(d)))
        else:
            m = d

        self.trees = []
        self.feature_subsets = []
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, n, replace=True)
            feats = self.rng.choice(d, m, replace=False)
            tree = SimpleDecisionTree(max_depth=self.max_depth, random_state=self.rng.randint(1e6))
            tree.fit(X[np.ix_(idx, feats)], y[idx])
            self.trees.append(tree)
            self.feature_subsets.append(feats)
        return self

    def predict_proba(self, X):
        probas = []
        for tree, feats in zip(self.trees, self.feature_subsets):
            probas.append(tree.predict_proba(X[:, feats]))
        return np.mean(probas, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# DATASETS
# ============================================================

def make_explanation_dataset(n_samples=500, random_state=42):
    """
    Dataset with known feature effects for validating explanations.

    x1: strong positive effect on class 1
    x2: moderate effect (interaction with x1)
    x3: threshold effect (only matters if > 0)
    x4, x5: pure noise
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 5)

    # Non-linear decision boundary
    logits = 2.0 * X[:, 0] + 1.5 * X[:, 1] * (X[:, 0] > 0) + 1.0 * (X[:, 2] > 0)
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n_samples) < prob).astype(int)

    feature_names = ['x1 (strong)', 'x2 (interaction)', 'x3 (threshold)',
                     'x4 (noise)', 'x5 (noise)']
    return X, y, feature_names


def make_simple_linear_dataset(n_samples=300, random_state=42):
    """Simple linear dataset where SHAP should recover exact coefficients."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 3)
    # Explicit linear model: y = 3*x1 - 2*x2 + 0*x3
    y = (3 * X[:, 0] - 2 * X[:, 1] > 0).astype(int)
    feature_names = ['x1 (coeff=3)', 'x2 (coeff=-2)', 'x3 (noise)']
    return X, y, feature_names


# ============================================================
# SHAPLEY VALUES (Sampling-based approximation)
# ============================================================

def shapley_values(model, x_instance, X_background, n_samples=500,
                   random_state=42):
    """
    Compute approximate Shapley values for a single prediction.

    THE ALGORITHM (Monte Carlo sampling):
    For each sample:
        1. Draw random permutation of features
        2. For each feature i in permutation:
           - Compute f(S + {i}) - f(S) where S = features before i
           - This is i's marginal contribution in this permutation
        3. Average over all sampled permutations

    Parameters
    ----------
    model : object with predict_proba method
    x_instance : array of shape (n_features,)
        The instance to explain
    X_background : array of shape (n_background, n_features)
        Background data (reference distribution)
    n_samples : int
        Number of permutations to sample

    Returns
    -------
    shap_values : array of shape (n_features,)
        Shapley value for each feature (for class 1)
    base_value : float
        Expected prediction (baseline)
    """
    rng = np.random.RandomState(random_state)
    n_features = len(x_instance)
    phi = np.zeros(n_features)

    # Base value: expected prediction over background
    base_value = model.predict_proba(X_background)[:, 1].mean()

    for _ in range(n_samples):
        # Random permutation
        perm = rng.permutation(n_features)

        # Random background instance to fill in "absent" features
        bg_idx = rng.randint(len(X_background))
        bg = X_background[bg_idx].copy()

        # Build x_with and x_without for each feature in permutation
        x_current = bg.copy()

        for j, feat in enumerate(perm):
            # Before adding feature: prediction with features so far
            x_without = x_current.copy()
            f_without = model.predict_proba(x_without.reshape(1, -1))[0, 1]

            # Add this feature
            x_current[feat] = x_instance[feat]
            x_with = x_current.copy()
            f_with = model.predict_proba(x_with.reshape(1, -1))[0, 1]

            # Marginal contribution
            phi[feat] += (f_with - f_without)

    phi /= n_samples
    return phi, base_value


def exact_shapley_values(model, x_instance, X_background):
    """
    Compute Shapley values via exhaustive enumeration (exponential cost).

    Enumerates all 2^n subsets. Uses up to 50 background samples
    for expectation over absent features. Useful for validation
    on small problems — near-exact for small background sets.
    """
    n_features = len(x_instance)
    if n_features > 10:
        raise ValueError("Exact Shapley is O(2^n) — too expensive for >10 features")

    phi = np.zeros(n_features)
    n = n_features

    for i in range(n_features):
        others = [j for j in range(n_features) if j != i]

        for size in range(n):
            for S in combinations(others, size):
                S = set(S)
                # Weight: |S|! * (n - |S| - 1)! / n!
                weight = (np.math.factorial(len(S)) *
                          np.math.factorial(n - len(S) - 1) /
                          np.math.factorial(n))

                # f(S + {i}) vs f(S)
                marginals_with = []
                marginals_without = []

                for bg in X_background[:50]:  # Limit for speed
                    x_with = bg.copy()
                    x_without = bg.copy()

                    for j in S:
                        x_with[j] = x_instance[j]
                        x_without[j] = x_instance[j]
                    x_with[i] = x_instance[i]

                    marginals_with.append(model.predict_proba(x_with.reshape(1, -1))[0, 1])
                    marginals_without.append(model.predict_proba(x_without.reshape(1, -1))[0, 1])

                phi[i] += weight * (np.mean(marginals_with) - np.mean(marginals_without))

    base_value = model.predict_proba(X_background[:50])[:, 1].mean()
    return phi, base_value


# ============================================================
# LIME (Local Interpretable Model-agnostic Explanations)
# ============================================================

def lime_explain(model, x_instance, X_train, n_samples=500,
                 kernel_width=0.75, random_state=42):
    """
    LIME explanation for a single prediction.

    THE ALGORITHM:
    1. Generate perturbed samples around x_instance
    2. Get model predictions for each
    3. Weight by proximity (RBF kernel)
    4. Fit weighted linear regression
    5. Return coefficients as local importance

    Parameters
    ----------
    model : object with predict_proba method
    x_instance : array of shape (n_features,)
    X_train : array of shape (n_train, n_features)
        Training data (for feature statistics)
    n_samples : int
        Number of perturbed samples
    kernel_width : float
        RBF kernel width for proximity weighting

    Returns
    -------
    lime_weights : array of shape (n_features,)
        Local linear coefficients (importance for class 1)
    intercept : float
        Linear model intercept
    r_squared : float
        Local model fit quality
    """
    rng = np.random.RandomState(random_state)
    n_features = len(x_instance)

    # Feature statistics from training data
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-8

    # Standardize instance
    x_std = (x_instance - feature_mean) / feature_std

    # Generate perturbed samples (in standardized space)
    perturbations = rng.randn(n_samples, n_features)
    Z_std = x_std + perturbations * 0.5  # perturb around instance

    # Convert back to original space
    Z = Z_std * feature_std + feature_mean

    # Get model predictions
    predictions = model.predict_proba(Z)[:, 1]

    # Compute proximity weights (RBF kernel in standardized space)
    distances = np.sqrt(np.sum((Z_std - x_std) ** 2, axis=1))
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # Weighted linear regression: minimize Σ w_i (y_i - X_i β)²
    # Solution: (X^T W X)^{-1} X^T W y
    X_design = np.column_stack([np.ones(n_samples), Z_std])
    W = np.diag(weights)

    try:
        XtWX = X_design.T @ W @ X_design
        XtWy = X_design.T @ W @ predictions
        beta = np.linalg.solve(XtWX + 1e-6 * np.eye(n_features + 1), XtWy)
    except np.linalg.LinAlgError:
        beta = np.zeros(n_features + 1)

    intercept = beta[0]
    lime_weights = beta[1:]

    # R-squared (weighted)
    y_pred = X_design @ beta
    ss_res = np.sum(weights * (predictions - y_pred) ** 2)
    ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)

    return lime_weights, intercept, r_squared


# ============================================================
# VISUALIZATION: Waterfall Plot
# ============================================================

def waterfall_plot(shap_values, feature_names, base_value, prediction,
                   ax=None, title=None):
    """
    Waterfall plot showing how each feature pushes prediction
    from baseline to final value.

    THE KEY INSIGHT: This is the decomposition
        f(x) = base_value + phi_1 + phi_2 + ... + phi_n
    visualized as a waterfall of contributions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n_features = len(shap_values)

    # Sort by absolute value
    order = np.argsort(np.abs(shap_values))
    sorted_values = shap_values[order]
    sorted_names = [feature_names[i] for i in order]

    # Build waterfall
    cumulative = base_value
    starts = []
    widths = []

    for val in sorted_values:
        starts.append(cumulative)
        widths.append(val)
        cumulative += val

    # Plot horizontal bars
    colors = ['#e74c3c' if w > 0 else '#3498db' for w in widths]
    y_pos = range(n_features)

    for i, (s, w, c) in enumerate(zip(starts, widths, colors)):
        ax.barh(i, w, left=s, color=c, alpha=0.8, height=0.6,
                edgecolor='white', linewidth=0.5)
        # Value label
        label_x = s + w / 2
        ax.text(label_x, i, f'{w:+.3f}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # Base value and prediction lines
    ax.axvline(base_value, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(prediction, color='black', linestyle='-', alpha=0.7, linewidth=1.5)

    ax.text(base_value, -0.8, f'Base\n{base_value:.3f}', ha='center',
            fontsize=8, color='gray')
    ax.text(prediction, n_features + 0.3, f'Prediction\n{prediction:.3f}',
            ha='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Model output (P(class=1))')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    return ax


# ============================================================
# VISUALIZATION: Beeswarm Plot
# ============================================================

def beeswarm_plot(all_shap_values, X, feature_names, ax=None, title=None):
    """
    Beeswarm plot: global view from local explanations.

    Each dot = one instance.
    x-axis = SHAP value (impact on prediction)
    y-axis = feature
    color = feature value (high = red, low = blue)

    THE KEY INSIGHT: Patterns across many local explanations
    reveal global feature effects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    n_features = all_shap_values.shape[1]

    # Order by mean |SHAP|
    mean_abs = np.mean(np.abs(all_shap_values), axis=0)
    order = np.argsort(mean_abs)

    for idx, feat_idx in enumerate(order):
        shap_vals = all_shap_values[:, feat_idx]
        feat_vals = X[:, feat_idx]

        # Normalize feature values to [0, 1] for coloring
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            feat_norm = (feat_vals - fmin) / (fmax - fmin)
        else:
            feat_norm = np.full_like(feat_vals, 0.5)

        # Add jitter to y for beeswarm effect
        jitter = np.random.RandomState(42).uniform(-0.3, 0.3, len(shap_vals))

        # Color: blue (low) to red (high)
        colors = plt.cm.RdBu_r(feat_norm)

        ax.scatter(shap_vals, idx + jitter, c=colors, s=8, alpha=0.5,
                   edgecolors='none')

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SHAP value (impact on model output)')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Feature value\n(normalized)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    return ax


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Number of samples vs Shapley accuracy
    # ---------------------------------------------------------
    print("\n1. SAMPLING CONVERGENCE: How many samples for good SHAP?")
    print("-" * 40)

    X, y, names = make_simple_linear_dataset(n_samples=300)
    model = SimpleRandomForest(n_trees=30, max_depth=4, random_state=42)
    model.fit(X, y)

    x_test = X[0]
    prediction = model.predict_proba(x_test.reshape(1, -1))[0, 1]
    X_bg = X[:50]

    print(f"Explaining instance: {x_test}")
    print(f"Prediction P(class=1): {prediction:.4f}")
    print(f"True model: y = 3*x1 - 2*x2 (x3 is noise)")

    sample_counts = [50, 100, 200, 500, 1000, 2000]
    print(f"\n   {'n_samples':<12} {'phi_x1':<10} {'phi_x2':<10} {'phi_x3':<10} {'sum+base':<10}")
    for n_samp in sample_counts:
        phi, base = shapley_values(model, x_test, X_bg, n_samples=n_samp,
                                   random_state=42)
        total = base + phi.sum()
        print(f"   {n_samp:<12} {phi[0]:<10.4f} {phi[1]:<10.4f} {phi[2]:<10.4f} {total:<10.4f}")

    print(f"\n   Actual prediction: {prediction:.4f}")
    print("-> More samples -> SHAP values converge")
    print("-> phi_x1 should be largest (coefficient=3)")
    print("-> phi_x3 should be near zero (noise)")
    print("-> sum + base should approach the actual prediction (efficiency axiom)")

    # ---------------------------------------------------------
    # 2. SHAP on linear model (should recover coefficients)
    # ---------------------------------------------------------
    print("\n\n2. SHAP ON LINEAR MODEL: Should recover coefficients")
    print("-" * 40)

    X, y, names = make_simple_linear_dataset(n_samples=400)

    # Build a "linear" model via simple thresholds
    class LinearModel:
        def __init__(self, w):
            self.w = np.array(w, dtype=float)

        def predict_proba(self, X):
            logits = X @ self.w
            p = 1 / (1 + np.exp(-logits))
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    linear_model = LinearModel([3.0, -2.0, 0.0])
    x_test = np.array([1.0, -0.5, 0.3])
    pred = linear_model.predict_proba(x_test.reshape(1, -1))[0, 1]

    phi, base = shapley_values(linear_model, x_test, X[:50], n_samples=2000,
                               random_state=42)

    print(f"True coefficients: [3.0, -2.0, 0.0]")
    print(f"Instance: {x_test}")
    print(f"Prediction: {pred:.4f}")
    print(f"Base value: {base:.4f}")
    print(f"\nSHAP values:")
    for i, name in enumerate(names):
        print(f"   {name}: phi = {phi[i]:.4f}")
    print(f"\nSum check: base + sum(phi) = {base + phi.sum():.4f} vs pred = {pred:.4f}")
    print("-> For a linear model, SHAP values are proportional to w_i * (x_i - E[x_i])")
    print("-> phi_x1 should be positive (coeff=3, x1=1.0 is above mean~0)")
    print("-> phi_x2 should be positive (coeff=-2, x2=-0.5 is below mean~0)")
    print("-> phi_x3 should be near zero (noise)")

    # ---------------------------------------------------------
    # 3. SHAP vs LIME agreement/disagreement
    # ---------------------------------------------------------
    print("\n\n3. SHAP vs LIME: Agreement and Disagreement")
    print("-" * 40)

    X, y, names = make_explanation_dataset(n_samples=500)
    model = SimpleRandomForest(n_trees=30, max_depth=5, random_state=42)
    model.fit(X, y)

    # Explain a few instances
    test_indices = [0, 10, 50, 100, 200]
    print(f"\n   {'Instance':<10} {'Method':<8} ", end='')
    for name in names:
        print(f"{name[:6]:<10}", end='')
    print()
    print("   " + "-" * 70)

    for idx in test_indices:
        x_test = X[idx]
        pred = model.predict_proba(x_test.reshape(1, -1))[0, 1]

        # SHAP
        phi, base = shapley_values(model, x_test, X[:50], n_samples=500,
                                   random_state=42)
        print(f"   [{idx:>3}]     SHAP    ", end='')
        for v in phi:
            print(f"{v:<10.4f}", end='')
        print()

        # LIME
        lime_w, intercept, r2 = lime_explain(model, x_test, X,
                                              n_samples=500, random_state=42)
        print(f"           LIME    ", end='')
        for v in lime_w:
            print(f"{v:<10.4f}", end='')
        print(f"  (R²={r2:.3f})")
        print()

    print("-> SHAP and LIME often AGREE on top features")
    print("-> But magnitudes differ: SHAP has theoretical guarantees, LIME depends on kernel_width")
    print("-> LIME R² tells us how well the local linear model fits")

    # ---------------------------------------------------------
    # 4. Explaining correct vs incorrect predictions
    # ---------------------------------------------------------
    print("\n\n4. CORRECT vs INCORRECT PREDICTIONS")
    print("-" * 40)

    preds = model.predict(X)
    correct_mask = (preds == y)
    incorrect_mask = ~correct_mask

    print(f"Accuracy: {correct_mask.mean():.3f}")

    if incorrect_mask.sum() > 0:
        # Pick a correct and incorrect prediction
        correct_idx = np.where(correct_mask)[0][0]
        incorrect_idx = np.where(incorrect_mask)[0][0]

        for label, idx in [("CORRECT", correct_idx), ("INCORRECT", incorrect_idx)]:
            x_test = X[idx]
            pred = model.predict_proba(x_test.reshape(1, -1))[0, 1]
            true_label = y[idx]
            pred_label = preds[idx]

            phi, base = shapley_values(model, x_test, X[:50], n_samples=500,
                                       random_state=42)
            print(f"\n   {label} prediction (index={idx}):")
            print(f"   True={true_label}, Predicted={pred_label}, P(class=1)={pred:.4f}")
            print(f"   Feature contributions:")
            for i, name in enumerate(names):
                direction = "pushes UP" if phi[i] > 0.01 else "pushes DOWN" if phi[i] < -0.01 else "negligible"
                print(f"      {name}: {phi[i]:+.4f} ({direction})")

        print("\n-> For INCORRECT predictions, SHAP shows which features MISLED the model")
        print("-> This helps debug: is the model wrong, or is the data point unusual?")
    else:
        print("All predictions correct — model too accurate for this demo")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_waterfall():
    """
    THE KEY INSIGHT: Waterfall plot decomposes a single prediction
    into base value + feature contributions.

    Shows two instances: one predicted positive, one negative.
    Each bar shows how a feature pushes prediction up or down.
    """
    X, y, names = make_explanation_dataset(n_samples=500)
    model = SimpleRandomForest(n_trees=30, max_depth=5, random_state=42)
    model.fit(X, y)

    # Find clear positive and negative predictions
    probas = model.predict_proba(X)[:, 1]
    pos_idx = np.argmax(probas)
    neg_idx = np.argmin(probas)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, idx, label in [(axes[0], pos_idx, 'HIGH confidence (class 1)'),
                            (axes[1], neg_idx, 'LOW confidence (class 1)')]:
        x_test = X[idx]
        pred = model.predict_proba(x_test.reshape(1, -1))[0, 1]
        phi, base = shapley_values(model, x_test, X[:50], n_samples=1000,
                                   random_state=42)
        waterfall_plot(phi, names, base, pred, ax=ax,
                       title=f'{label}\nP(class=1) = {pred:.3f}')

        # Add feature values
        info = "\n".join([f"{names[i][:8]}: {x_test[i]:.2f}" for i in range(len(names))])
        ax.text(0.98, 0.02, f"Feature values:\n{info}",
                transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('SHAPLEY WATERFALL: How Each Feature Pushes the Prediction\n'
                 'Red = pushes toward class 1 | Blue = pushes toward class 0\n'
                 'Base value = average prediction over background data',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_beeswarm():
    """
    Global view from local explanations.

    Compute SHAP for MANY instances, then plot all together.
    Patterns emerge: which features consistently matter, and
    how feature values relate to their impact.
    """
    X, y, names = make_explanation_dataset(n_samples=300)
    model = SimpleRandomForest(n_trees=30, max_depth=5, random_state=42)
    model.fit(X, y)

    # Compute SHAP for subset of instances
    n_explain = 100
    X_bg = X[:50]
    all_shap = np.zeros((n_explain, X.shape[1]))

    print("Computing SHAP values for beeswarm plot...")
    for i in range(n_explain):
        phi, _ = shapley_values(model, X[i], X_bg, n_samples=200,
                                random_state=42 + i)
        all_shap[i] = phi
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{n_explain} instances explained")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Beeswarm
    beeswarm_plot(all_shap, X[:n_explain], names, ax=axes[0],
                  title='SHAP Beeswarm: Local -> Global Patterns')

    # Mean |SHAP| bar chart (global importance from local explanations)
    mean_abs_shap = np.mean(np.abs(all_shap), axis=0)
    order = np.argsort(mean_abs_shap)
    axes[1].barh(range(len(names)), mean_abs_shap[order],
                 color=['#e74c3c' if mean_abs_shap[i] > 0.05 else '#95a5a6'
                        for i in order],
                 alpha=0.8, edgecolor='white')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([names[i] for i in order], fontsize=9)
    axes[1].set_xlabel('Mean |SHAP value|')
    axes[1].set_title('Global Importance from Local Explanations\n'
                      '(average |SHAP| across instances)', fontsize=11,
                      fontweight='bold')
    axes[1].grid(True, alpha=0.2, axis='x')

    plt.suptitle('FROM LOCAL TO GLOBAL: SHAP Values Across All Instances\n'
                 'Each dot = one instance | Color = feature value | '
                 'x-axis = impact on prediction',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_lime():
    """
    LIME: Local linear approximation of complex model.

    Shows the complex model's decision boundary and LIME's
    local linear approximation around specific points.
    """
    # Simple 2D problem for visualization
    rng = np.random.RandomState(42)
    n = 400
    X = rng.randn(n, 2)
    # Complex non-linear boundary
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5) |
         ((X[:, 0] > 0.5) & (X[:, 1] > 0.5))).astype(int)

    model = SimpleRandomForest(n_trees=30, max_depth=6, random_state=42)
    model.fit(X, y)
    feature_names = ['x1', 'x2']

    # Points to explain
    explain_points = [
        np.array([0.0, 0.0]),   # Inside circle (class 1)
        np.array([1.5, 0.0]),   # Outside circle (class 0)
        np.array([0.8, 0.8]),   # Near corner (class 1)
        np.array([-1.0, -1.0]), # Outside (class 0)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for ax, x_test in zip(axes.ravel(), explain_points):
        # Decision boundary
        xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 100),
                             np.linspace(-2.5, 2.5, 100))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=20, cmap='RdBu_r', alpha=0.3)
        ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)

        # Training data
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.15, s=15)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.15, s=15)

        # LIME explanation
        lime_w, intercept, r2 = lime_explain(model, x_test, X,
                                              n_samples=500,
                                              kernel_width=0.75,
                                              random_state=42)
        pred = model.predict_proba(x_test.reshape(1, -1))[0, 1]

        # Draw the point
        ax.plot(x_test[0], x_test[1], 'k*', markersize=15, zorder=10)

        # Draw LIME's local linear boundary
        # lime predicts: w1*x1_std + w2*x2_std + intercept = 0.5
        # In standardized space
        feat_mean = X.mean(axis=0)
        feat_std = X.std(axis=0) + 1e-8

        if abs(lime_w[1]) > 1e-8:
            x1_range = np.linspace(x_test[0] - 1.5, x_test[0] + 1.5, 100)
            x1_std = (x1_range - feat_mean[0]) / feat_std[0]
            # w1*x1_std + w2*x2_std + intercept = 0.5
            x2_std = (0.5 - intercept - lime_w[0] * x1_std) / lime_w[1]
            x2_range = x2_std * feat_std[1] + feat_mean[1]

            valid = (x2_range > -2.5) & (x2_range < 2.5)
            ax.plot(x1_range[valid], x2_range[valid], 'g--', linewidth=3,
                    alpha=0.8, label=f'LIME boundary (R²={r2:.2f})')

        # Draw arrow showing gradient direction from LIME weights
        arrow_scale = 0.5
        ax.annotate('', xy=(x_test[0] + lime_w[0] * arrow_scale,
                             x_test[1] + lime_w[1] * arrow_scale),
                     xytext=(x_test[0], x_test[1]),
                     arrowprops=dict(arrowstyle='->', color='green',
                                    lw=2.5))

        # Info box
        info = (f'P(class=1) = {pred:.3f}\n'
                f'LIME weights:\n'
                f'  x1: {lime_w[0]:+.3f}\n'
                f'  x2: {lime_w[1]:+.3f}\n'
                f'R² = {r2:.3f}')
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', facecolor='lightgreen',
                                    alpha=0.8))

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f'Point ({x_test[0]:.1f}, {x_test[1]:.1f})',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.suptitle('LIME: Local Linear Approximation of Complex Model\n'
                 'Black boundary = true model | Green dashed = LIME local fit\n'
                 'Green arrow = LIME gradient direction | Star = explained point',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LOCAL EXPLANATIONS -- Paradigm: LOCAL EXPLANATION")
    print("=" * 60)

    print("""
WHAT THIS IS:
    WHY did the model make THIS prediction?
    Two methods that answer this from different angles:

    1. SHAPLEY VALUES (game theory):
       Each feature's "fair share" of the prediction.
       phi_i = average marginal contribution across all coalitions.
       The ONLY method satisfying all fairness axioms.

    2. LIME (local interpretable model):
       Fit a linear model locally around the prediction.
       Simple approximation of complex model's behavior nearby.

WHEN TO USE WHICH:
    SHAP: When you need theoretically grounded attributions
    LIME: When you need a quick interpretable approximation
    BOTH: When you want to cross-validate explanations
    """)

    # Run ablation experiments
    ablation_experiments()

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig1 = visualize_waterfall()
    save_path1 = '/Users/sid47/ML Algorithms/67_shap_waterfall.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_beeswarm()
    save_path2 = '/Users/sid47/ML Algorithms/67_shap_summary.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_lime()
    save_path3 = '/Users/sid47/ML Algorithms/67_lime_local.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: What Local Explanations Reveal")
    print("=" * 60)
    print("""
1. SHAPLEY VALUES are the gold standard
   -> Only method satisfying all fairness axioms
   -> phi_1 + phi_2 + ... + phi_n = prediction - baseline
   -> Cost: O(2^n) exact, but sampling works well with ~500-1000 samples

2. LIME is fast and intuitive
   -> Fits a simple model locally
   -> R² tells you how good the approximation is
   -> kernel_width controls "local" neighborhood size

3. They USUALLY agree on top features
   -> Both identify the most impactful features
   -> Magnitudes differ (different scales)
   -> Disagreement often reveals interesting model behavior

4. INCORRECT predictions are most informative
   -> SHAP shows which features MISLED the model
   -> Helps debug: is the model wrong, or the data unusual?

5. LOCAL -> GLOBAL: Beeswarm plots
   -> Compute SHAP for many instances
   -> Patterns emerge: which features consistently matter
   -> This bridges 66_feature_importance (global) and 67 (local)

THE BIG PICTURE:
   66_feature_importance.py: "What features matter overall?"
   67_local_explanations.py: "Why THIS specific prediction?"
   Together: complete picture of model behavior.

CONNECTIONS:
   -> 66_feature_importance.py: global vs local view
   -> 33_multi_agent_rl.py: Shapley values = cooperative game theory
   -> 01_linear_regression.py: LIME = local linear regression
   -> 09_random_forest.py: the black box we explained here
   -> 12_mlp.py: also works on neural networks (model-agnostic!)
    """)
