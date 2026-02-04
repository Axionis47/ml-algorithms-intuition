"""
RANDOM FOREST — Paradigm: COMMITTEE (Bagging + Random Features)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Decision trees have HIGH VARIANCE — small data changes → very different trees.

SOLUTION: Train MANY trees and AVERAGE their predictions.

But averaging identical trees doesn't help — errors are correlated.

THE KEY: Make trees DIFFERENT through randomization:
    1. BAGGING: Each tree trains on a bootstrap sample (sample with replacement)
    2. FEATURE SUBSAMPLING: Each split considers only √d random features

This DECORRELATES the trees, so their errors cancel out.

===============================================================
WHY IT WORKS — VARIANCE REDUCTION
===============================================================

If you have n i.i.d. random variables with variance σ², their average has variance σ²/n.

But trees from the same dataset are NOT independent — they're correlated.

If correlation is ρ, the variance of the average is:
    Var(average) = ρσ² + (1-ρ)σ²/n

Random Forest reduces ρ through:
    - Bootstrap sampling: each tree sees ~63% of unique samples
    - Feature subsampling: trees split on different features

Lower ρ → lower variance → better generalization!

===============================================================
OUT-OF-BAG (OOB) ERROR
===============================================================

Each tree doesn't see ~37% of samples (not in its bootstrap sample).

These "out-of-bag" samples can be used for validation WITHOUT a held-out set!

OOB error ≈ cross-validation error, but FREE.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Averaging reduces variance but keeps bias the same
2. Feature subsampling decorrelates trees
3. Individual trees can still overfit, but ensemble is robust

WHAT IT CAN DO:
    ✓ Handle high dimensions (feature subsampling selects)
    ✓ Robust to overfitting (ensemble averaging)
    ✓ Provides feature importance
    ✓ Embarrassingly parallel (trees are independent)

WHAT IT CAN'T DO:
    ✗ Extrapolate beyond training data range
    ✗ Capture complex interactions better than deep nets
    ✗ Learn smooth functions (still axis-aligned)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
tree_module = import_module('07_decision_tree')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
plot_decision_boundary = datasets_module.plot_decision_boundary
accuracy = datasets_module.accuracy
DecisionTreeClassifier = tree_module.DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest: Ensemble of decision trees with bagging + feature subsampling.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, random_state=None):
        """
        Parameters:
        -----------
        n_estimators : Number of trees
        max_depth : Maximum tree depth (None = unlimited)
        min_samples_split : Minimum samples to split a node
        max_features : Features per split ('sqrt', 'log2', int, or None for all)
        bootstrap : Whether to use bootstrap sampling
        random_state : Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.oob_indices = []  # For OOB error computation

    def _get_max_features(self, n_features):
        """Compute number of features to consider per split."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features

    def fit(self, X, y):
        """
        Train the random forest.

        For each tree:
            1. Create bootstrap sample (if bootstrap=True)
            2. Train tree with feature subsampling
            3. Track OOB indices
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.trees = []
        self.oob_indices = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]

                # Track OOB (out-of-bag) samples
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[indices] = False
                self.oob_indices.append(np.where(oob_mask)[0])
            else:
                X_boot = X
                y_boot = y
                self.oob_indices.append(np.array([]))

            # Train tree with feature subsampling
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        """
        Average probability predictions from all trees.
        """
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Average (probability of class 1)
        return predictions.mean(axis=0)

    def predict(self, X):
        """
        Majority vote from all trees.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def oob_score(self, X, y):
        """
        Compute out-of-bag score.

        For each sample, use only trees that didn't see it during training.
        """
        n_samples = len(y)
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if len(oob_idx) > 0:
                pred = tree.predict(X[oob_idx])
                oob_predictions[oob_idx] += pred
                oob_counts[oob_idx] += 1

        # Average predictions where we have OOB predictions
        valid = oob_counts > 0
        oob_predictions[valid] /= oob_counts[valid]
        oob_class = (oob_predictions >= 0.5).astype(int)

        return accuracy(y[valid], oob_class[valid])

    def feature_importances(self, X, y):
        """
        Compute feature importance via mean decrease in impurity.
        (Simplified version)
        """
        # This would require tracking impurity decrease during training
        # For now, return uniform importances
        return np.ones(X.shape[1]) / X.shape[1]


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component?
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    # -------- Experiment 1: Number of Trees --------
    print("\n1. EFFECT OF NUMBER OF TREES")
    print("-" * 40)
    for n_trees in [1, 5, 10, 25, 50, 100]:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy(y_test, rf.predict(X_test))
        print(f"n_trees={n_trees:<4} accuracy={acc:.3f}")
    print("→ More trees = lower variance, diminishing returns after ~50")

    # -------- Experiment 2: Single Tree vs Forest --------
    print("\n2. SINGLE TREE vs FOREST (Variance Reduction)")
    print("-" * 40)

    # Run multiple single trees
    single_accs = []
    for i in range(10):
        tree = DecisionTreeClassifier(max_depth=10)
        # Bootstrap sample
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        tree.fit(X_train[idx], y_train[idx])
        single_accs.append(accuracy(y_test, tree.predict(X_test)))

    # Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    forest_acc = accuracy(y_test, rf.predict(X_test))

    print(f"Single tree (10 runs): mean={np.mean(single_accs):.3f}, std={np.std(single_accs):.3f}")
    print(f"Random Forest (100 trees): {forest_acc:.3f}")
    print("→ Forest is more stable AND often more accurate!")

    # -------- Experiment 3: Max Features --------
    print("\n3. EFFECT OF MAX FEATURES (Decorrelation)")
    print("-" * 40)
    n_features = X_train.shape[1]
    for max_feat in [1, 'sqrt', 'log2', None]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   max_features=max_feat, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy(y_test, rf.predict(X_test))
        feat_str = f"{max_feat}" if max_feat else "all"
        print(f"max_features={feat_str:<6} accuracy={acc:.3f}")
    print("→ 'sqrt' is usually a good default (decorrelates trees)")

    # -------- Experiment 4: Bootstrap vs No Bootstrap --------
    print("\n4. BOOTSTRAP vs NO BOOTSTRAP")
    print("-" * 40)
    for bootstrap in [False, True]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   bootstrap=bootstrap, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy(y_test, rf.predict(X_test))
        print(f"bootstrap={str(bootstrap):<6} accuracy={acc:.3f}")
    print("→ Bootstrap provides diversity, usually helps")

    # -------- Experiment 5: OOB Score --------
    print("\n5. OUT-OF-BAG (OOB) SCORE")
    print("-" * 40)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                               bootstrap=True, random_state=42)
    rf.fit(X_train, y_train)
    oob = rf.oob_score(X_train, y_train)
    test_acc = accuracy(y_test, rf.predict(X_test))
    print(f"OOB score:  {oob:.3f}")
    print(f"Test score: {test_acc:.3f}")
    print("→ OOB ≈ cross-validation, but FREE!")

    # -------- Experiment 6: Max Depth --------
    print("\n6. EFFECT OF MAX DEPTH")
    print("-" * 40)
    for depth in [1, 3, 5, 10, None]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy(y_test, rf.predict(X_test))
        depth_str = f"{depth}" if depth else "None"
        print(f"max_depth={depth_str:<5} accuracy={acc:.3f}")
    print("→ Deeper trees can capture more complex patterns")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Random Forest on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Random Forest on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        acc = accuracy(y_te, y_pred)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    datasets = get_2d_datasets()
    plot_datasets = {k: v for k, v in datasets.items() if k != 'clustered'}

    n = len(plot_datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_tr, y_tr)
        acc = accuracy(y_te, rf.predict(X_te))

        plot_decision_boundary(rf.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('RANDOM FOREST: Decision Boundaries\n'
                 '(Ensemble of trees → smoother, more robust boundaries)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_tree_vs_forest():
    """Compare single tree vs forest decision boundaries."""
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Single tree
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(X_train, y_train)
    acc_tree = accuracy(y_test, tree.predict(X_test))
    plot_decision_boundary(tree.predict, X, y, ax=axes[0],
                          title=f'Single Tree (acc={acc_tree:.2f})')

    # Forest with 10 trees
    rf10 = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    rf10.fit(X_train, y_train)
    acc_rf10 = accuracy(y_test, rf10.predict(X_test))
    plot_decision_boundary(rf10.predict, X, y, ax=axes[1],
                          title=f'Forest 10 trees (acc={acc_rf10:.2f})')

    # Forest with 100 trees
    rf100 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf100.fit(X_train, y_train)
    acc_rf100 = accuracy(y_test, rf100.predict(X_test))
    plot_decision_boundary(rf100.predict, X, y, ax=axes[2],
                          title=f'Forest 100 trees (acc={acc_rf100:.2f})')

    plt.suptitle('Tree vs Forest: Boundary Smoothness\n'
                 'More trees = smoother, more robust boundary',
                 fontsize=12)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("RANDOM FOREST — Paradigm: COMMITTEE")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Ensemble of decision trees with:
    1. Bagging (bootstrap sampling)
    2. Feature subsampling (√d features per split)

THE KEY INSIGHT:
    Averaging DECORRELATED predictors reduces variance
    while keeping bias the same.

    Var(average) = ρσ² + (1-ρ)σ²/n

    Lower correlation ρ → lower variance!

HYPERPARAMETERS:
    - n_estimators: more trees = lower variance
    - max_features: lower = more decorrelated trees
    - max_depth: controls individual tree complexity
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/09_random_forest_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path1}")
    plt.close(fig1)

    fig2 = compare_tree_vs_forest()
    save_path2 = '/Users/sid47/ML Algorithms/09_tree_vs_forest.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved tree vs forest comparison to: {save_path2}")
    plt.close(fig2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Random Forest Reveals")
    print("="*60)
    print("""
1. AVERAGING reduces variance but not bias
2. DECORRELATION is key — correlated trees don't help
3. OOB gives free cross-validation estimate
4. More trees always helps (diminishing returns after ~50-100)
5. Still axis-aligned (same limitation as single trees)

KEY INSIGHT:
    Random Forest shows that COMBINING WEAK LEARNERS
    can create a strong learner. The trees are "weak"
    (high variance), but their average is strong.

NEXT: Gradient Boosting — reduce BIAS instead of variance
    """)
