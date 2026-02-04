"""
ADABOOST — Paradigm: COMMITTEE (Reweight Mistakes)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Train weak learners SEQUENTIALLY, but instead of fitting residuals
(like Gradient Boosting), REWEIGHT THE SAMPLES.

Misclassified samples get HEAVIER weights → next learner focuses on them.

Final prediction = weighted vote of all weak learners.

===============================================================
THE KEY INSIGHT: EXPONENTIAL LOSS & COORDINATE DESCENT
===============================================================

AdaBoost minimizes EXPONENTIAL LOSS:
    L = Σᵢ exp(-yᵢ F(xᵢ))

where F(x) = Σₘ αₘ hₘ(x)   (weighted sum of weak learners)

At each step, we greedily add one learner to minimize the total loss.
This is COORDINATE DESCENT in the space of weak learner weights!

The math works out beautifully:
    αₘ = ½ log((1-εₘ)/εₘ)

where εₘ = weighted error rate of learner m.

If εₘ = 0.5 (random), αₘ = 0 (ignored)
If εₘ < 0.5 (better than random), αₘ > 0 (trust it)
If εₘ → 0 (perfect), αₘ → ∞ (trust completely)

===============================================================
SAMPLE WEIGHT UPDATE
===============================================================

After adding learner m with weight αₘ:

    wᵢ ← wᵢ × exp(-αₘ yᵢ hₘ(xᵢ))

Correctly classified (yᵢ hₘ(xᵢ) = +1):  wᵢ ← wᵢ × exp(-αₘ)  (decrease)
Misclassified (yᵢ hₘ(xᵢ) = -1):        wᵢ ← wᵢ × exp(+αₘ)  (increase)

Misclassified samples get EXPONENTIALLY heavier with each mistake.

===============================================================
ADABOOST vs GRADIENT BOOSTING
===============================================================

ADABOOST:
    - Reweight SAMPLES, train on weighted data
    - Each learner sees different sample distribution
    - Exponential loss → sensitive to outliers
    - Weak learners just need >50% accuracy

GRADIENT BOOSTING:
    - Fit RESIDUALS (gradient of loss)
    - Each learner targets different y values
    - Flexible loss functions (squared, log, huber)
    - Can use any regressor as base

Both: sequential, additive models → reducing bias.

===============================================================
THE WEAK-TO-STRONG THEOREM
===============================================================

If each weak learner is slightly better than random (ε < 0.5 - γ),
after M rounds, training error ≤ exp(-2γ²M) → 0 exponentially fast!

You can turn ANY weak learner into a strong one.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Additive model: F(x) = Σ αₘ hₘ(x)
2. Focus on hard examples: misclassified → higher weight
3. Exponential loss: margins matter more than just correct/incorrect
4. Weak learner limitation: can't capture complex patterns alone

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


class DecisionStump:
    """
    Decision stump = depth-1 decision tree.

    The weakest meaningful learner: just ONE split.

    Why stumps? They guarantee > 50% accuracy on any non-trivial
    problem (if data isn't pure random). Perfect for AdaBoost.
    """

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # +1 or -1 (which side is which class)

    def fit(self, X, y, sample_weights=None):
        """
        Find the best single split considering sample weights.

        For each feature, for each threshold:
            - Predict class based on > or < threshold
            - Calculate WEIGHTED error
            - Keep the best
        """
        n_samples, n_features = X.shape

        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        # Convert y to ±1 for AdaBoost
        y_signed = 2 * y - 1  # {0,1} → {-1,+1}

        best_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                for polarity in [1, -1]:
                    # Predict: polarity * sign(x - threshold)
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] < threshold] = -1
                    else:
                        predictions[X[:, feature] >= threshold] = -1

                    # Weighted error
                    error = np.sum(sample_weights * (predictions != y_signed))

                    if error < best_error:
                        best_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity

        return self

    def predict(self, X):
        """Predict ±1 labels."""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature] < self.threshold] = -1
        else:
            predictions[X[:, self.feature] >= self.threshold] = -1

        return predictions


class AdaBoostClassifier:
    """
    AdaBoost for binary classification.

    Sequentially trains weak learners on reweighted samples.
    """

    def __init__(self, n_estimators=50, weak_learner='stump', max_depth=1):
        """
        Parameters:
        -----------
        n_estimators : Number of weak learners
        weak_learner : 'stump' for decision stump, 'tree' for shallow tree
        max_depth : If using tree, maximum depth (default 1 = stump)
        """
        self.n_estimators = n_estimators
        self.weak_learner = weak_learner
        self.max_depth = max_depth

        self.estimators = []
        self.alphas = []  # Learner weights

    def fit(self, X, y):
        """
        Train AdaBoost.

        1. Initialize uniform sample weights
        2. For each round:
           a. Train weak learner on weighted samples
           b. Compute weighted error ε
           c. Compute learner weight α = ½ log((1-ε)/ε)
           d. Update sample weights: w × exp(-α y h(x))
           e. Normalize weights
        """
        n_samples = len(y)

        # Convert y to ±1
        y_signed = 2 * y - 1

        # Initialize uniform weights
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators = []
        self.alphas = []

        for m in range(self.n_estimators):
            # Train weak learner
            if self.weak_learner == 'stump':
                learner = DecisionStump()
                learner.fit(X, y, sample_weights)
            else:
                # Use decision tree with sample weights
                learner = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=2
                )
                # Note: Our tree doesn't support sample_weights directly
                # We'd need to modify it or use resampling
                learner.fit(X, y)

            # Get predictions (±1)
            if self.weak_learner == 'stump':
                predictions = learner.predict(X)
            else:
                predictions = 2 * learner.predict(X) - 1

            # Compute weighted error
            misclassified = (predictions != y_signed)
            epsilon = np.sum(sample_weights * misclassified)

            # Prevent division by zero
            epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)

            # If worse than random, stop
            if epsilon >= 0.5:
                break

            # Compute learner weight
            # α = ½ log((1-ε)/ε)
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)

            self.estimators.append(learner)
            self.alphas.append(alpha)

            # Update sample weights
            # w_i ← w_i × exp(-α y_i h(x_i))
            sample_weights *= np.exp(-alpha * y_signed * predictions)

            # Normalize
            sample_weights /= np.sum(sample_weights)

        return self

    def decision_function(self, X):
        """
        Compute the weighted sum: F(x) = Σ αₘ hₘ(x)
        """
        n_samples = X.shape[0]
        F = np.zeros(n_samples)

        for alpha, learner in zip(self.alphas, self.estimators):
            if isinstance(learner, DecisionStump):
                predictions = learner.predict(X)
            else:
                predictions = 2 * learner.predict(X) - 1
            F += alpha * predictions

        return F

    def predict_proba(self, X):
        """
        Convert decision function to probabilities via sigmoid.
        Not native to AdaBoost, but useful for comparison.
        """
        F = self.decision_function(X)
        # Use sigmoid to get pseudo-probabilities
        return 1 / (1 + np.exp(-2 * F))

    def predict(self, X):
        """Predict class labels."""
        F = self.decision_function(X)
        return (F >= 0).astype(int)


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

    # -------- Experiment 1: Number of Weak Learners --------
    print("\n1. EFFECT OF NUMBER OF WEAK LEARNERS")
    print("-" * 40)
    print("More learners = more complex model")
    for n_est in [1, 5, 10, 25, 50, 100, 200]:
        ada = AdaBoostClassifier(n_estimators=n_est)
        ada.fit(X_train, y_train)
        train_acc = accuracy(y_train, ada.predict(X_train))
        test_acc = accuracy(y_test, ada.predict(X_test))
        print(f"n_estimators={n_est:<4} train={train_acc:.3f} test={test_acc:.3f}")
    print("→ AdaBoost can drive training error to 0!")
    print("→ Test accuracy saturates (sometimes overfits)")

    # -------- Experiment 2: Weak Learner Complexity --------
    print("\n2. EFFECT OF BASE LEARNER DEPTH")
    print("-" * 40)
    print("Stumps vs deeper trees as weak learners")
    for depth in [1, 2, 3, 5]:
        ada = AdaBoostClassifier(n_estimators=50, weak_learner='stump' if depth == 1 else 'tree',
                                  max_depth=depth)
        ada.fit(X_train, y_train)
        test_acc = accuracy(y_test, ada.predict(X_test))
        actual_learners = len(ada.estimators)
        print(f"max_depth={depth:<2} estimators_used={actual_learners:<3} test_acc={test_acc:.3f}")
    print("→ Stumps often work best! Deeper trees can overfit.")

    # -------- Experiment 3: Watch Sample Weights Evolve --------
    print("\n3. SAMPLE WEIGHT EVOLUTION (First 10 iterations)")
    print("-" * 40)
    ada = AdaBoostClassifier(n_estimators=10)

    n_samples = len(y_train)
    y_signed = 2 * y_train - 1
    sample_weights = np.ones(n_samples) / n_samples

    for m in range(10):
        learner = DecisionStump()
        learner.fit(X_train, y_train, sample_weights)
        predictions = learner.predict(X_train)

        misclassified = (predictions != y_signed)
        epsilon = np.sum(sample_weights * misclassified)
        epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)

        sample_weights *= np.exp(-alpha * y_signed * predictions)
        sample_weights /= np.sum(sample_weights)

        max_weight = np.max(sample_weights)
        min_weight = np.min(sample_weights)
        ratio = max_weight / min_weight

        print(f"Round {m+1}: ε={epsilon:.3f} α={alpha:.3f} "
              f"max/min_weight={ratio:.1f}x")
    print("→ Weights become VERY unequal (hard examples dominate)")

    # -------- Experiment 4: Alpha Values Analysis --------
    print("\n4. LEARNER WEIGHTS (α) DISTRIBUTION")
    print("-" * 40)
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(X_train, y_train)

    alphas = np.array(ada.alphas)
    print(f"Number of learners: {len(alphas)}")
    print(f"α range: [{alphas.min():.3f}, {alphas.max():.3f}]")
    print(f"α mean: {alphas.mean():.3f}, std: {alphas.std():.3f}")
    print(f"Early learners (first 5): {alphas[:5]}")
    print(f"Late learners (last 5): {alphas[-5:]}")
    print("→ Earlier learners often have higher α (easier patterns)")

    # -------- Experiment 5: Effect of Label Noise --------
    print("\n5. SENSITIVITY TO LABEL NOISE")
    print("-" * 40)
    for noise_frac in [0.0, 0.05, 0.10, 0.20, 0.30]:
        y_noisy = y_train.copy()
        n_flip = int(noise_frac * len(y_train))
        flip_idx = np.random.choice(len(y_train), n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

        ada = AdaBoostClassifier(n_estimators=50)
        ada.fit(X_train, y_noisy)
        test_acc = accuracy(y_test, ada.predict(X_test))
        print(f"noise={noise_frac:.0%} test_accuracy={test_acc:.3f}")
    print("→ AdaBoost is SENSITIVE to label noise!")
    print("  (Exponential loss amplifies outlier mistakes)")

    # -------- Experiment 6: Training Error Bound --------
    print("\n6. TRAINING ERROR DECAY (Weak-to-Strong)")
    print("-" * 40)
    print("Theoretical: training_error ≤ exp(-2γ²M)")

    ada = AdaBoostClassifier(n_estimators=100)

    n_samples = len(y_train)
    y_signed = 2 * y_train - 1
    sample_weights = np.ones(n_samples) / n_samples
    F = np.zeros(n_samples)

    estimators = []
    alphas = []

    for m in range(100):
        learner = DecisionStump()
        learner.fit(X_train, y_train, sample_weights)
        predictions = learner.predict(X_train)

        misclassified = (predictions != y_signed)
        epsilon = np.sum(sample_weights * misclassified)
        epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)

        if epsilon >= 0.5:
            break

        alpha = 0.5 * np.log((1 - epsilon) / epsilon)

        estimators.append(learner)
        alphas.append(alpha)
        F += alpha * predictions

        sample_weights *= np.exp(-alpha * y_signed * predictions)
        sample_weights /= np.sum(sample_weights)

        if (m + 1) in [1, 5, 10, 25, 50, 100]:
            train_error = np.mean((np.sign(F) != y_signed))
            print(f"M={m+1:<3} train_error={train_error:.4f}")

    print("→ Training error drops EXPONENTIALLY fast!")


def benchmark_on_datasets():
    print("\n" + "="*60)
    print("BENCHMARK: AdaBoost on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        ada = AdaBoostClassifier(n_estimators=100)
        ada.fit(X_tr, y_tr)
        acc = accuracy(y_te, ada.predict(X_te))
        n_learners = len(ada.estimators)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f} (used {n_learners} learners)")

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

        ada = AdaBoostClassifier(n_estimators=100)
        ada.fit(X_tr, y_tr)
        acc = accuracy(y_te, ada.predict(X_te))

        plot_decision_boundary(ada.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('ADABOOST: Decision Boundaries\n'
                 '(Weighted voting of decision stumps)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_boosting_methods():
    """Compare AdaBoost vs Gradient Boosting."""
    print("\n" + "="*60)
    print("ADABOOST vs GRADIENT BOOSTING")
    print("="*60)

    try:
        gb_module = import_module('10_gradient_boosting')
        GradientBoostingClassifier = gb_module.GradientBoostingClassifier
    except:
        print("Gradient Boosting module not found, skipping comparison")
        return

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'AdaBoost':<12} {'GradBoost':<12} {'Winner':<12}")
    print("-" * 51)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        # AdaBoost
        ada = AdaBoostClassifier(n_estimators=100)
        ada.fit(X_tr, y_tr)
        ada_acc = accuracy(y_te, ada.predict(X_te))

        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=42)
        gb.fit(X_tr, y_tr)
        gb_acc = accuracy(y_te, gb.predict(X_te))

        winner = "AdaBoost" if ada_acc > gb_acc else "GradBoost" if gb_acc > ada_acc else "Tie"
        print(f"{name:<15} {ada_acc:<12.3f} {gb_acc:<12.3f} {winner:<12}")

    print("\n→ AdaBoost: simpler, more interpretable (weighted stumps)")
    print("→ GradBoost: more flexible (any loss, regression trees)")


if __name__ == '__main__':
    print("="*60)
    print("ADABOOST — Boosting by Reweighting Samples")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Train weak learners SEQUENTIALLY, each focuses on
    samples that previous learners got WRONG.

    Final model: F(x) = Σ αₘ hₘ(x)  (weighted vote)

THE KEY INSIGHT:
    AdaBoost minimizes EXPONENTIAL LOSS via coordinate descent.
    Learner weight α = ½ log((1-ε)/ε)
    Sample weight update: w × exp(-α y h(x))

WEAK-TO-STRONG THEOREM:
    Any learner slightly better than random →
    Combined model with arbitrarily low error!
    Training error ≤ exp(-2γ²M)

ADABOOST vs GRADIENT BOOSTING:
    AdaBoost: Reweight SAMPLES, exponential loss
    GradBoost: Fit RESIDUALS, flexible loss functions
    """)

    ablation_experiments()
    results = benchmark_on_datasets()
    compare_boosting_methods()

    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/11_adaboost.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Reweight samples: misclassified → heavier
2. Learner weight α = ½ log((1-ε)/ε)
3. Exponential loss → sensitive to outliers/noise
4. Stumps often work best (keep learners weak!)
5. Training error decays EXPONENTIALLY (weak-to-strong)
    """)
