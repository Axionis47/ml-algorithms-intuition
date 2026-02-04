"""
NAIVE BAYES — Paradigm: PROBABILISTIC (Generative)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of finding a boundary, MODEL HOW EACH CLASS GENERATES DATA.

Generative approach:
    1. Learn p(x|y=c) — "how does class c produce features?"
    2. Learn p(y=c) — "how common is class c?"
    3. Use Bayes rule: p(y=c|x) ∝ p(x|y=c) × p(y=c)

Compare: Which class was most likely to have generated this x?

===============================================================
THE ONE EQUATION (Bayes Rule)
===============================================================

    p(y=c|x) = p(x|y=c) × p(y=c) / p(x)

Since p(x) is the same for all classes, we just need:

    ŷ = argmax_c [ p(x|y=c) × p(y=c) ]
      = argmax_c [ log p(x|y=c) + log p(y=c) ]  (log for numerical stability)

===============================================================
THE "NAIVE" ASSUMPTION
===============================================================

The "naive" part: features are CONDITIONALLY INDEPENDENT given class.

    p(x₁, x₂, ..., xₙ | y) = ∏ᵢ p(xᵢ | y)

This is almost always FALSE in reality!
- Example: height and weight are correlated
- Example: word "machine" and "learning" co-occur

WHY DOES IT WORK?
    1. We only need RANKING, not exact probabilities
    2. Independence violations often cancel out
    3. Massive parameter reduction: O(d) instead of O(d²)
    4. With limited data, the simpler model often wins

===============================================================
GENERATIVE vs DISCRIMINATIVE
===============================================================

GENERATIVE (Naive Bayes, GMM):
    - Model p(x|y) — how data is generated
    - Can generate new samples
    - Makes strong assumptions about data distribution
    - Works well with small data (strong prior)

DISCRIMINATIVE (Logistic, SVM, Neural Nets):
    - Model p(y|x) directly — the boundary
    - Can't generate new samples
    - Fewer assumptions
    - Needs more data, but often better asymptotically

===============================================================
INDUCTIVE BIAS
===============================================================

1. Features are conditionally independent (the "naive" part)
2. Each feature follows assumed distribution (Gaussian, Bernoulli, etc.)
3. Class priors matter (imbalance is handled naturally)

WHAT IT CAN DO:
    ✓ Handle missing features (just skip them)
    ✓ Handle high dimensions (no curse!)
    ✓ Fast training and prediction
    ✓ Provides actual probabilities

WHAT IT CAN'T DO:
    ✗ Complex feature interactions
    ✗ Highly correlated features (violates independence)

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


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes: Assume each feature is Gaussian within each class.

    p(xᵢ | y=c) = N(xᵢ | μᵢc, σ²ᵢc)

    Parameters per class per feature: mean and variance
    Total parameters: 2 × d × C  (very few!)
    """

    def __init__(self, var_smoothing=1e-9):
        """
        var_smoothing: Added to variance for numerical stability
                       (prevents division by zero for constant features)
        """
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_priors = None
        self.means = None       # μᵢc for each class c and feature i
        self.variances = None   # σ²ᵢc for each class c and feature i

    def fit(self, X, y):
        """
        TRAINING: Compute mean and variance per feature per class.

        This is maximum likelihood estimation (MLE) of Gaussian parameters.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameter arrays
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            # Get samples belonging to class c
            X_c = X[y == c]

            # Class prior: p(y=c) = count(y=c) / n
            self.class_priors[idx] = X_c.shape[0] / n_samples

            # Mean of each feature for this class
            self.means[idx] = X_c.mean(axis=0)

            # Variance of each feature for this class (+ smoothing)
            self.variances[idx] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _compute_log_likelihood(self, X):
        """
        Compute log p(x|y=c) for all classes.

        Log of Gaussian PDF:
        log N(x|μ,σ²) = -0.5 × [log(2πσ²) + (x-μ)²/σ²]

        Returns: (n_samples, n_classes) array of log-likelihoods
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihoods = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            # For each feature: log p(xᵢ|y=c)
            # = -0.5 × [log(2πσ²ᵢc) + (xᵢ - μᵢc)²/σ²ᵢc]

            mean = self.means[idx]      # (n_features,)
            var = self.variances[idx]   # (n_features,)

            # (x - μ)² / σ²
            diff_sq = (X - mean) ** 2 / var  # (n_samples, n_features)

            # log(2πσ²)
            log_var_term = np.log(2 * np.pi * var)  # (n_features,)

            # Sum over features (independence assumption!)
            # This is where the "naive" comes in: we MULTIPLY probabilities
            # across features, which becomes SUM in log space
            log_likelihoods[:, idx] = -0.5 * np.sum(log_var_term + diff_sq, axis=1)

        return log_likelihoods

    def predict_log_proba(self, X):
        """
        Compute log p(y=c|x) ∝ log p(x|y=c) + log p(y=c)

        We don't normalize (would need to compute p(x)), but that's fine
        for classification — we just need to compare.
        """
        log_likelihood = self._compute_log_likelihood(X)  # log p(x|y)
        log_prior = np.log(self.class_priors)             # log p(y)

        # log p(y|x) ∝ log p(x|y) + log p(y)
        return log_likelihood + log_prior

    def predict_proba(self, X):
        """
        Convert log probabilities to probabilities using softmax.

        This normalizes so they sum to 1 (approximates true posterior).
        """
        log_proba = self.predict_log_proba(X)
        # Softmax for numerical stability
        log_proba_max = log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba - log_proba_max)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict class with highest posterior probability."""
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes: For binary features (0/1).

    p(xᵢ=1 | y=c) = θᵢc
    p(xᵢ=0 | y=c) = 1 - θᵢc

    Used for text classification with binary term presence.
    """

    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter (additive smoothing)
               Prevents zero probabilities for unseen features.
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_priors = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = X_c.shape[0] / n_samples

            # θᵢc = (count of 1s + α) / (n_c + 2α)
            # Laplace smoothing prevents zero probabilities
            self.feature_probs[idx] = (X_c.sum(axis=0) + self.alpha) / \
                                      (X_c.shape[0] + 2 * self.alpha)

        return self

    def predict_log_proba(self, X):
        # p(x|y=c) = ∏ᵢ θᵢc^xᵢ × (1-θᵢc)^(1-xᵢ)
        # log p(x|y=c) = Σᵢ [xᵢ log θᵢc + (1-xᵢ) log(1-θᵢc)]

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_proba = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            theta = self.feature_probs[idx]
            log_theta = np.log(theta)
            log_1_theta = np.log(1 - theta)

            # For each sample
            log_likelihood = X @ log_theta + (1 - X) @ log_1_theta
            log_proba[:, idx] = log_likelihood + np.log(self.class_priors[idx])

        return log_proba

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


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

    # -------- Experiment 1: Effect of Independence Violation --------
    print("\n1. EFFECT OF FEATURE CORRELATION (Independence Violation)")
    print("-" * 40)

    # Create data with varying correlation
    n = 500
    for correlation in [0.0, 0.5, 0.9, 0.99]:
        # Generate correlated features
        cov = np.array([[1, correlation], [correlation, 1]])
        X_class0 = np.random.multivariate_normal([0, 0], cov, n)
        X_class1 = np.random.multivariate_normal([2, 2], cov, n)
        X = np.vstack([X_class0, X_class1])
        y = np.array([0]*n + [1]*n)

        # Shuffle and split
        idx = np.random.permutation(2*n)
        X, y = X[idx], y[idx]
        X_train, X_test = X[:int(0.8*2*n)], X[int(0.8*2*n):]
        y_train, y_test = y[:int(0.8*2*n)], y[int(0.8*2*n):]

        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        print(f"correlation={correlation:.2f}  accuracy={acc:.3f}")

    print("→ NB still works with correlated features!")
    print("   Only the RANKING matters, not exact probabilities")

    # -------- Experiment 2: Effect of Variance Smoothing --------
    print("\n2. EFFECT OF VARIANCE SMOOTHING")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['linear']

    # Add a constant feature (zero variance)
    X_train_const = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test_const = np.c_[X_test, np.ones(X_test.shape[0])]

    for smoothing in [0, 1e-15, 1e-9, 1e-3, 1.0]:
        model = GaussianNaiveBayes(var_smoothing=smoothing)
        try:
            model.fit(X_train_const, y_train)
            acc = accuracy(y_test, model.predict(X_test_const))
            print(f"smoothing={smoothing:<8}  accuracy={acc:.3f}")
        except Exception as e:
            print(f"smoothing={smoothing:<8}  ERROR: {str(e)[:30]}")

    print("→ Smoothing prevents division by zero for constant features")

    # -------- Experiment 3: Class Imbalance --------
    print("\n3. HANDLING CLASS IMBALANCE")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['imbalanced']

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    print(f"Class priors learned: {dict(zip(model.classes, model.class_priors))}")
    print(f"Test accuracy: {accuracy(y_test, model.predict(X_test)):.3f}")

    # Predictions breakdown
    y_pred = model.predict(X_test)
    print(f"Predicted class distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    print(f"True class distribution:      {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print("→ NB naturally handles imbalance via priors")

    # -------- Experiment 4: High Dimensions --------
    print("\n4. PERFORMANCE IN HIGH DIMENSIONS")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['high_dim']

    # Compare NB vs KNN
    model_nb = GaussianNaiveBayes()
    model_nb.fit(X_train, y_train)
    acc_nb = accuracy(y_test, model_nb.predict(X_test))

    from importlib import import_module
    knn_module = import_module('03_knn')
    model_knn = knn_module.KNN(k=5)
    model_knn.fit(X_train, y_train)
    acc_knn = accuracy(y_test, model_knn.predict(X_test))

    print(f"Naive Bayes:  {acc_nb:.3f}")
    print(f"KNN (k=5):    {acc_knn:.3f}")
    print("→ NB doesn't suffer from curse of dimensionality!")
    print("   Each feature is modeled independently")

    # -------- Experiment 5: Probability Calibration --------
    print("\n5. ARE PROBABILITIES CALIBRATED?")
    print("-" * 40)

    X_train, X_test, y_train, y_test = datasets['moons']
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("Prob range    Predicted    Actual")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if np.sum(mask) > 0:
            predicted_avg = probs[mask].mean()
            actual_avg = y_test[mask].mean()
            print(f"[{lo:.1f}, {hi:.1f})      {predicted_avg:.3f}        {actual_avg:.3f}")
    print("→ NB probabilities are often OVERCONFIDENT (extreme 0 or 1)")
    print("   The independence assumption makes it too certain")

    # -------- Experiment 6: Decision Boundary Shape --------
    print("\n6. DECISION BOUNDARY SHAPE")
    print("-" * 40)
    print("Gaussian NB produces QUADRATIC decision boundaries!")
    print("(When variances differ between classes)")
    print("")
    print("With equal variances → linear boundary (like Logistic)")
    print("With unequal variances → curved boundary")
    print("")
    print("This comes from the log-likelihood difference:")
    print("  log p(x|y=0) - log p(x|y=1)")
    print("  = quadratic terms if σ²₀ ≠ σ²₁")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Naive Bayes on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Naive Bayes on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = GaussianNaiveBayes()
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
        print(f"{name:<15} accuracy: {acc:.3f}")

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

        model = GaussianNaiveBayes()
        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('NAIVE BAYES: Decision Boundaries\n'
                 '(Gaussian assumption → elliptical/quadratic boundaries)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_with_others():
    """Compare Naive Bayes vs other models."""
    print("\n" + "="*60)
    print("COMPARISON: Naive Bayes vs Others")
    print("="*60)

    from importlib import import_module
    logistic_module = import_module('02_logistic_regression')
    knn_module = import_module('03_knn')

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'Logistic':<10} {'KNN':<10} {'NaiveBayes':<10}")
    print("-" * 50)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        # Logistic
        log_model = logistic_module.LogisticRegression(lr=0.1, n_iters=1000)
        log_model.fit(X_tr, y_tr)
        acc_log = accuracy(y_te, log_model.predict(X_te))

        # KNN
        knn_model = knn_module.KNN(k=5)
        knn_model.fit(X_tr, y_tr)
        acc_knn = accuracy(y_te, knn_model.predict(X_te))

        # NB
        nb_model = GaussianNaiveBayes()
        nb_model.fit(X_tr, y_tr)
        acc_nb = accuracy(y_te, nb_model.predict(X_te))

        print(f"{name:<15} {acc_log:<10.3f} {acc_knn:<10.3f} {acc_nb:<10.3f}")

    print("\n→ NB works well on high_dim (no curse of dimensionality)")
    print("→ NB struggles on highly nonlinear (circles, spiral)")
    print("→ NB is FAST — O(nd) training, O(d) prediction")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("NAIVE BAYES — Paradigm: PROBABILISTIC (Generative)")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Model how each class GENERATES data, then invert with Bayes rule.
    "Which class was most likely to have produced this x?"

THE "NAIVE" ASSUMPTION:
    Features are independent given the class.
    Almost always false, but works anyway!

GENERATIVE vs DISCRIMINATIVE:
    Generative: Models p(x|y), can sample new data
    Discriminative: Models p(y|x), just finds boundaries

STRENGTHS:
    ✓ Fast training and prediction
    ✓ Works in high dimensions (no curse!)
    ✓ Handles missing features naturally
    ✓ Interpretable (feature likelihoods per class)

WEAKNESSES:
    ✗ Strong distributional assumptions (Gaussian)
    ✗ Probabilities are often overconfident
    ✗ Struggles with complex nonlinear patterns
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Compare with others
    compare_with_others()

    # Visualize
    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/04_naive_bayes.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path}")
    plt.close(fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Naive Bayes Reveals")
    print("="*60)
    print("""
1. GENERATIVE models think about how data is produced
2. The "naive" independence assumption is wrong but useful
3. NB doesn't suffer from curse of dimensionality (unlike KNN)
4. Probabilities are often overconfident (need calibration)
5. Gaussian NB gives quadratic boundaries (with unequal variances)

KEY INSIGHT:
    Sometimes a "wrong" model with fewer parameters beats
    a "correct" model with too many parameters.
    This is the bias-variance tradeoff in action.

WHEN TO USE:
    - Text classification (bag of words)
    - High dimensions
    - Fast baseline needed
    - Prior knowledge about distributions

NEXT: Gaussian Processes — probabilistic with uncertainty over FUNCTIONS
    """)
