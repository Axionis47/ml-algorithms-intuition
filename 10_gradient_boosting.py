"""
GRADIENT BOOSTING — Paradigm: COMMITTEE (Boosting Residuals)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of averaging trees (Random Forest), train each tree to
CORRECT THE ERRORS of the previous ensemble.

F₀(x) = initial prediction
F₁(x) = F₀(x) + η × h₁(x)    where h₁ fits residuals of F₀
F₂(x) = F₁(x) + η × h₂(x)    where h₂ fits residuals of F₁
...

Each tree h_m targets what the current ensemble F_{m-1} gets WRONG.

===============================================================
THE KEY INSIGHT: GRADIENT DESCENT IN FUNCTION SPACE
===============================================================

For squared loss: residual = y - F(x) = -∂L/∂F

So fitting to residuals IS fitting to the negative gradient!

This is GRADIENT DESCENT, but instead of updating parameters,
we're adding FUNCTIONS to our model.

For other losses (logistic, etc.), we fit to the pseudo-residuals:
    r = -∂L/∂F(x)

===============================================================
LEARNING RATE (Shrinkage)
===============================================================

η (learning rate) controls how much each tree contributes:

F_m = F_{m-1} + η × h_m

- η = 1: full correction (can overfit quickly)
- η = 0.1: small steps (needs more trees, but generalizes better)

Lower η + more trees = better regularization (usually).

===============================================================
COMPARISON: RANDOM FOREST vs GRADIENT BOOSTING
===============================================================

RANDOM FOREST:
    - Trees are independent (parallel training)
    - Reduces VARIANCE through averaging
    - Robust to overfitting

GRADIENT BOOSTING:
    - Trees are sequential (each fixes previous errors)
    - Reduces BIAS through correction
    - Can overfit if not careful (needs early stopping / learning rate)

In practice: Gradient Boosting often achieves lower error,
but Random Forest is more robust and easier to tune.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Additive model: F(x) = Σ trees
2. Sequential correction: each tree targets current errors
3. Learning rate trades off speed vs generalization

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


class GradientBoostingClassifier:
    """
    Gradient Boosting for binary classification.

    Uses log loss (cross-entropy) and fits trees to pseudo-residuals.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, random_state=None):
        """
        Parameters:
        -----------
        n_estimators : Number of boosting stages (trees)
        learning_rate : Shrinkage parameter (η)
        max_depth : Max depth of each tree (usually small, 3-5)
        min_samples_split : Min samples to split node
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.trees = []
        self.init_pred = None  # Initial prediction (log-odds)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _compute_pseudo_residuals(self, y, F):
        """
        PSEUDO-RESIDUALS for log loss.

        For log loss: L = -[y log(p) + (1-y) log(1-p)]
        where p = sigmoid(F)

        The pseudo-residual (negative gradient) is:
            r = -∂L/∂F = y - p

        This is what we want the next tree to fit!
        """
        p = self._sigmoid(F)
        return y - p  # Same form as logistic regression gradient!

    def fit(self, X, y):
        """
        Train gradient boosting model.

        1. Initialize with log-odds of class prevalence
        2. For each stage:
           a. Compute pseudo-residuals
           b. Fit tree to pseudo-residuals
           c. Update predictions: F += η × tree(X)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(y)

        # Initialize with log-odds
        # If 70% class 1, init = log(0.7/0.3) ≈ 0.85
        p_init = np.mean(y)
        p_init = np.clip(p_init, 0.01, 0.99)
        self.init_pred = np.log(p_init / (1 - p_init))

        # Current predictions (log-odds)
        F = np.full(n_samples, self.init_pred)

        self.trees = []

        for m in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = self._compute_pseudo_residuals(y, F)

            # Fit tree to pseudo-residuals
            # Note: For proper GBM, we'd use a regression tree
            # Here we use our classification tree with a hack
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            update = tree.predict(X)
            F += self.learning_rate * update

        return self

    def predict_proba(self, X):
        """Predict probabilities."""
        F = np.full(X.shape[0], self.init_pred)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return self._sigmoid(F)

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int)


class DecisionTreeRegressor:
    """Simple regression tree for gradient boosting."""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        parent_mse = self._mse(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                child_mse = (n_left * left_mse + n_right * right_mse) / n_samples

                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'value': np.mean(y)}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return {'leaf': True, 'value': np.mean(y)}

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _predict_sample(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] < node['threshold']:
            return self._predict_sample(x, node['left'])
        return self._predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    # -------- Experiment 1: Number of Estimators --------
    print("\n1. EFFECT OF NUMBER OF TREES (n_estimators)")
    print("-" * 40)
    for n_trees in [1, 5, 10, 25, 50, 100]:
        gb = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=0.1,
                                       max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy(y_test, gb.predict(X_test))
        print(f"n_trees={n_trees:<4} accuracy={acc:.3f}")
    print("→ More trees = better fit (until overfitting)")

    # -------- Experiment 2: Learning Rate --------
    print("\n2. EFFECT OF LEARNING RATE (η)")
    print("-" * 40)
    for lr in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=lr,
                                       max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy(y_test, gb.predict(X_test))
        print(f"lr={lr:<5} accuracy={acc:.3f}")
    print("→ Lower lr = slower learning, often better generalization")

    # -------- Experiment 3: Tree Depth --------
    print("\n3. EFFECT OF TREE DEPTH")
    print("-" * 40)
    for depth in [1, 2, 3, 5, 10]:
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                       max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy(y_test, gb.predict(X_test))
        print(f"depth={depth:<3} accuracy={acc:.3f}")
    print("→ Depth 3-5 usually optimal. Deeper = more overfit.")

    # -------- Experiment 4: Learning Rate vs n_estimators Tradeoff --------
    print("\n4. LEARNING RATE vs N_TREES TRADEOFF")
    print("-" * 40)
    settings = [
        (0.5, 20), (0.1, 100), (0.05, 200), (0.01, 500)
    ]
    for lr, n_trees in settings:
        gb = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=lr,
                                       max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy(y_test, gb.predict(X_test))
        print(f"lr={lr:<5} n_trees={n_trees:<4} accuracy={acc:.3f}")
    print("→ Lower lr + more trees often wins (better regularization)")


def benchmark_on_datasets():
    print("\n" + "="*60)
    print("BENCHMARK: Gradient Boosting on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)
        gb.fit(X_tr, y_tr)
        acc = accuracy(y_te, gb.predict(X_te))
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}")

    return results


def visualize_decision_boundaries():
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

        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)
        gb.fit(X_tr, y_tr)
        acc = accuracy(y_te, gb.predict(X_te))

        plot_decision_boundary(gb.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('GRADIENT BOOSTING: Decision Boundaries\n'
                 '(Sequential correction of errors)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("GRADIENT BOOSTING — Gradient Descent in Function Space")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Train trees SEQUENTIALLY, each correcting errors of the previous.
    F_m = F_{m-1} + η × h_m(x)

THE KEY INSIGHT:
    Fitting to residuals = gradient descent in FUNCTION space!
    -∂Loss/∂F(x) = y - p (for log loss)

RANDOM FOREST vs GRADIENT BOOSTING:
    RF: Parallel trees, reduces VARIANCE
    GB: Sequential trees, reduces BIAS

HYPERPARAMETERS:
    - n_estimators: more = better fit (watch for overfit)
    - learning_rate: lower = slower but better regularization
    - max_depth: usually 3-5 (weak learners)
    """)

    ablation_experiments()
    results = benchmark_on_datasets()

    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/10_gradient_boosting.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Each tree targets RESIDUALS of the current ensemble
2. This is gradient descent in FUNCTION space
3. Learning rate trades off speed vs generalization
4. Shallow trees (weak learners) work best
5. Often achieves lower error than RF, but more prone to overfit
    """)
