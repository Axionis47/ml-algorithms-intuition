"""
DECISION TREE — Paradigm: PARTITIONING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Don't fit a function. CHOP THE SPACE INTO BOXES.

Recursively:
    1. Find the best feature and threshold to split on
    2. Split the data into two groups
    3. Repeat for each group until stopping criterion

At each leaf: predict the majority class (or mean for regression).

The result is a set of AXIS-ALIGNED RECTANGLES.

===============================================================
THE SPLIT CRITERION
===============================================================

How do we choose the best split? Maximize PURITY after splitting.

GINI IMPURITY:
    Gini(node) = 1 - Σᵢ pᵢ²

    where pᵢ is the probability of class i in the node.

    Gini = 0: pure (all same class)
    Gini = 0.5: maximum impurity (binary, 50/50 split)

INFORMATION GAIN (Entropy):
    Entropy(node) = -Σᵢ pᵢ log(pᵢ)
    Information Gain = Entropy(parent) - weighted avg Entropy(children)

In practice, Gini and Entropy give similar results.
Gini is faster (no log computation).

===============================================================
INDUCTIVE BIAS
===============================================================

1. AXIS-ALIGNED SPLITS — can only cut perpendicular to one axis
   - Diagonal boundaries need many splits (staircase approximation)
   - XOR is TRIVIAL (exactly 2 splits!)
   - Circles are HARD (need many rectangular boxes)

2. GREEDY CONSTRUCTION — may not find globally optimal tree
   - Each split is locally optimal
   - Can miss better solutions that require "bad" initial splits

3. HIGH VARIANCE — small data changes can dramatically change the tree

WHAT IT CAN DO:
    ✓ Handle mixed feature types (continuous, categorical)
    ✓ Naturally handle feature interactions (splits condition on previous)
    ✓ Interpretable (you can visualize the tree)
    ✓ No feature scaling needed

WHAT IT CAN'T DO:
    ✗ Smooth boundaries (always jagged/rectangular)
    ✗ Stable predictions (high variance)
    ✗ Avoid overfitting without pruning

===============================================================
OVERFITTING: THE FUNDAMENTAL TREE PROBLEM
===============================================================

An unpruned tree can memorize the training data:
- Grow until each leaf has one sample
- 100% training accuracy, terrible generalization

Solutions:
    1. MAX DEPTH: limit how deep the tree can grow
    2. MIN SAMPLES: require minimum samples per leaf
    3. PRUNING: grow full tree, then remove nodes that don't help
    4. ENSEMBLES: average many trees (Random Forest)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
plot_decision_boundary = datasets_module.plot_decision_boundary
accuracy = datasets_module.accuracy


class Node:
    """A node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, n_samples=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child (< threshold)
        self.right = right          # Right child (>= threshold)
        self.value = value          # Prediction if leaf (class label or probs)
        self.n_samples = n_samples  # Number of samples at this node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier built from scratch.

    GREEDY ALGORITHM:
        For each node:
            1. If stopping criterion met → create leaf
            2. Else:
                - For each feature and threshold:
                    - Compute impurity reduction
                - Split on best (feature, threshold)
                - Recursively build children
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', max_features=None):
        """
        Parameters:
        -----------
        max_depth : Maximum tree depth (None = unlimited)
        min_samples_split : Minimum samples required to split a node
        min_samples_leaf : Minimum samples required in a leaf
        criterion : 'gini' or 'entropy'
        max_features : Number of features to consider per split (for Random Forest)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
        self.n_classes = None
        self.n_features = None

    def _gini(self, y):
        """
        GINI IMPURITY

        Gini = 1 - Σᵢ pᵢ²

        Interpretation: probability that a randomly chosen sample
        would be incorrectly labeled if labeled randomly according
        to the distribution of labels in the node.

        Gini = 0: pure node (all same class)
        Gini = 0.5: maximum impurity (binary, 50/50)
        """
        if len(y) == 0:
            return 0

        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _entropy(self, y):
        """
        ENTROPY

        H = -Σᵢ pᵢ log₂(pᵢ)

        Measures uncertainty in the distribution.

        Entropy = 0: pure node
        Entropy = 1: maximum uncertainty (binary, 50/50)
        """
        if len(y) == 0:
            return 0

        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y):
        """Compute impurity based on chosen criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)

    def _information_gain(self, y, left_mask, right_mask):
        """
        INFORMATION GAIN

        IG = Impurity(parent) - weighted_avg(Impurity(children))

        We want to MAXIMIZE information gain (reduce impurity most).
        """
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            return 0

        parent_impurity = self._impurity(y)
        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])

        # Weighted average of children impurities
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        return parent_impurity - child_impurity

    def _best_split(self, X, y):
        """
        Find the best (feature, threshold) split.

        GREEDY SEARCH:
            For each feature:
                For each unique value as threshold:
                    Compute information gain
            Return the split with highest gain.

        This is O(n × d × n) = O(n²d) per node — expensive!
        Real implementations use sorted feature values and running sums.
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        # Determine which features to consider
        if self.max_features is not None:
            features = np.random.choice(n_features,
                                       min(self.max_features, n_features),
                                       replace=False)
        else:
            features = range(n_features)

        for feature in features:
            # Get unique values for thresholds
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                # Check minimum samples constraint
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """
        RECURSIVE TREE BUILDING

        Base cases (create leaf):
            1. Pure node (all same class)
            2. Max depth reached
            3. Too few samples to split
            4. No valid split found

        Recursive case:
            1. Find best split
            2. Create left and right subtrees
        """
        n_samples = len(y)

        # Compute class distribution for this node
        counts = np.bincount(y, minlength=self.n_classes)
        most_common = np.argmax(counts)

        # ---- STOPPING CRITERIA ----

        # Pure node (all same class)
        if len(np.unique(y)) == 1:
            return Node(value=most_common, n_samples=n_samples)

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=most_common, n_samples=n_samples)

        # Not enough samples to split
        if n_samples < self.min_samples_split:
            return Node(value=most_common, n_samples=n_samples)

        # ---- FIND BEST SPLIT ----
        feature, threshold, gain = self._best_split(X, y)

        # No valid split found
        if feature is None or gain <= 0:
            return Node(value=most_common, n_samples=n_samples)

        # ---- CREATE SPLIT ----
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
            n_samples=n_samples
        )

    def fit(self, X, y):
        """Build the decision tree."""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y.astype(int))
        return self

    def _predict_sample(self, x, node):
        """Traverse tree for one sample."""
        # Leaf node
        if node.value is not None:
            return node.value

        # Decision node
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """Predict classes for samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def get_depth(self, node=None):
        """Get the depth of the tree."""
        if node is None:
            node = self.root
        if node.value is not None:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def get_n_leaves(self, node=None):
        """Get number of leaves."""
        if node is None:
            node = self.root
        if node.value is not None:
            return 1
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)


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

    # -------- Experiment 1: Effect of Max Depth --------
    print("\n1. EFFECT OF MAX DEPTH")
    print("-" * 40)
    for depth in [1, 2, 3, 5, 10, None]:
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X_train, y_train)
        acc_train = accuracy(y_train, tree.predict(X_train))
        acc_test = accuracy(y_test, tree.predict(X_test))
        actual_depth = tree.get_depth()
        n_leaves = tree.get_n_leaves()
        print(f"max_depth={str(depth):<5} actual={actual_depth:<3} leaves={n_leaves:<4} "
              f"train={acc_train:.3f}  test={acc_test:.3f}")
    print("→ Deep trees overfit (high train, low test)")
    print("→ Shallow trees underfit (low train)")

    # -------- Experiment 2: Gini vs Entropy --------
    print("\n2. GINI vs ENTROPY")
    print("-" * 40)
    for criterion in ['gini', 'entropy']:
        tree = DecisionTreeClassifier(max_depth=5, criterion=criterion)
        tree.fit(X_train, y_train)
        acc = accuracy(y_test, tree.predict(X_test))
        print(f"criterion={criterion:<8}  accuracy={acc:.3f}")
    print("→ Usually very similar results")

    # -------- Experiment 3: XOR is EASY --------
    print("\n3. XOR — TRIVIAL FOR DECISION TREES")
    print("-" * 40)
    X_xor, X_xor_test, y_xor, y_xor_test = datasets['xor']

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_xor, y_xor)
    acc = accuracy(y_xor_test, tree.predict(X_xor_test))
    print(f"XOR accuracy with depth=3: {acc:.3f}")
    print(f"Tree depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")
    print("→ XOR needs only 2 axis-aligned splits — perfect for trees!")

    # -------- Experiment 4: Variance --------
    print("\n4. HIGH VARIANCE (Instability)")
    print("-" * 40)
    print("Train on 5 random subsets, observe tree differences:")
    depths = []
    accs = []
    for i in range(5):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        tree = DecisionTreeClassifier(max_depth=10)
        tree.fit(X_train[idx], y_train[idx])
        depths.append(tree.get_depth())
        accs.append(accuracy(y_test, tree.predict(X_test)))
        print(f"  Subset {i+1}: depth={tree.get_depth()}, leaves={tree.get_n_leaves()}, acc={accs[-1]:.3f}")
    print(f"→ Depth range: {min(depths)}-{max(depths)}, Acc range: {min(accs):.3f}-{max(accs):.3f}")
    print("   Small data changes → different trees!")

    # -------- Experiment 5: Axis-Aligned Limitation --------
    print("\n5. AXIS-ALIGNED LIMITATION (Diagonal Boundary)")
    print("-" * 40)
    # Create diagonal boundary data
    n = 200
    X_diag = np.random.randn(n, 2)
    y_diag = (X_diag[:, 0] + X_diag[:, 1] > 0).astype(int)  # Diagonal boundary

    X_train_d, X_test_d = X_diag[:150], X_diag[150:]
    y_train_d, y_test_d = y_diag[:150], y_diag[150:]

    for depth in [3, 5, 10, 20]:
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X_train_d, y_train_d)
        acc = accuracy(y_test_d, tree.predict(X_test_d))
        print(f"depth={depth:<3} leaves={tree.get_n_leaves():<4} acc={acc:.3f}")
    print("→ Needs many splits to approximate diagonal with axis-aligned cuts")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Decision Tree on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Decision Tree on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = DecisionTreeClassifier(max_depth=10)
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy(y_te, y_pred)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}  (depth={model.get_depth()}, leaves={model.get_n_leaves()})")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    print("\n" + "="*60)
    print("VISUALIZING DECISION BOUNDARIES")
    print("="*60)

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

        model = DecisionTreeClassifier(max_depth=10)
        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('DECISION TREE: Decision Boundaries\n'
                 '(Axis-aligned splits → rectangular regions)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_depth_effect():
    """Visualize how depth affects decision boundary."""
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    depths = [1, 2, 3, 5, 10, None]

    for i, depth in enumerate(depths):
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'depth={depth} (acc={acc:.2f}, leaves={model.get_n_leaves()})')

    plt.suptitle('DECISION TREE: Effect of Max Depth (Moons Dataset)\n'
                 'Shallow=underfit, Deep=overfit with jagged boundaries',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("DECISION TREE — Paradigm: PARTITIONING")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Recursively chop the space into boxes using axis-aligned splits.
    Each leaf predicts the majority class in that region.

THE KEY INSIGHT:
    - XOR is TRIVIAL (2 axis-aligned splits)
    - Diagonal boundaries are HARD (need many staircase splits)
    - Trees are GREEDY — may not find optimal structure

INDUCTIVE BIAS:
    - Axis-aligned decision boundaries
    - Hierarchical feature importance
    - Greedy, locally optimal splits

OVERFITTING:
    Unpruned trees memorize training data.
    Control with: max_depth, min_samples, pruning, or ensembles.
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize decision boundaries
    fig1 = visualize_decision_boundaries()
    save_path1 = '/Users/sid47/ML Algorithms/07_decision_tree_boundaries.png'
    fig1.savefig(save_path1, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path1}")
    plt.close(fig1)

    # Visualize depth effect
    fig2 = visualize_depth_effect()
    save_path2 = '/Users/sid47/ML Algorithms/07_decision_tree_depth.png'
    fig2.savefig(save_path2, dpi=100, bbox_inches='tight')
    print(f"Saved depth effect to: {save_path2}")
    plt.close(fig2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Decision Trees Reveal")
    print("="*60)
    print("""
1. PARTITIONING is a fundamentally different approach than projection
2. Axis-aligned splits = rectangular decision regions
3. XOR is trivial, diagonal is hard (opposite of linear models!)
4. High variance — small data changes alter the tree dramatically
5. Depth controls bias-variance: shallow underfit, deep overfit

KEY INSIGHT:
    Trees are GREEDY and HIGH-VARIANCE, but their simplicity
    makes them perfect building blocks for ENSEMBLES.

NEXT: Random Forest — reduce variance by averaging many trees
    """)
