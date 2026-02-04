"""
LOGISTIC REGRESSION — Paradigm: PROJECTION + SIGMOID

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Same as Linear Regression, but squash the output through sigmoid
to get probabilities:

    p(y=1|x) = σ(w'x + b) = 1 / (1 + exp(-(w'x + b)))

THE DECISION BOUNDARY IS STILL A HYPERPLANE!
    σ(w'x + b) = 0.5  ⟺  w'x + b = 0

The sigmoid doesn't change the shape of the boundary — it's still
a straight line (in 2D) or hyperplane (in higher D).

===============================================================
WHY SIGMOID?
===============================================================

1. Squashes any real number to (0, 1) — interpretable as probability
2. Derivative is simple: σ'(x) = σ(x)(1 - σ(x))
3. Log-odds are LINEAR: log(p/(1-p)) = w'x + b
   This is why it's called "logistic" — models the log-odds.

===============================================================
WHY CROSS-ENTROPY, NOT MSE?
===============================================================

With sigmoid output p ∈ (0,1) and MSE loss:
- When p ≈ 0 but y = 1, gradient ∝ p(1-p) ≈ 0  → VANISHING!
- The model is confident and wrong, but gradient is tiny.

Cross-entropy loss: L = -[y log(p) + (1-y) log(1-p)]
- When p → 0 but y = 1: loss → ∞, gradient large
- Forces correction when confidently wrong

The gradient is beautifully simple: dL/dw = (p - y) * x
No vanishing, clean optimization.

===============================================================
INDUCTIVE BIAS (same as Linear Regression)
===============================================================

1. Decision boundary is LINEAR (hyperplane)
2. Classes are separable by a line
3. Features contribute INDEPENDENTLY (no interactions without feature engineering)

WHAT IT CAN'T SEE: Same limitations as Linear Regression!
- Circles, XOR, spirals — anything requiring curved boundaries

The ONLY improvement over Linear Regression:
- Outputs are proper probabilities
- Training dynamics are better (cross-entropy vs MSE)

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


def sigmoid(z):
    """
    THE SIGMOID FUNCTION

    σ(z) = 1 / (1 + e^(-z))

    Properties:
    - σ(0) = 0.5
    - σ(∞) → 1
    - σ(-∞) → 0
    - σ(-z) = 1 - σ(z)  (symmetric around 0.5)
    - σ'(z) = σ(z)(1 - σ(z))  (derivative in terms of itself)

    The last property makes backprop clean.
    """
    # Clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred):
    """
    BINARY CROSS-ENTROPY

    L = -[y log(p) + (1-y) log(1-p)]

    WHY THIS LOSS?
    - Maximum likelihood under Bernoulli assumption
    - No vanishing gradients when confident but wrong
    - Gradient is simply (p - y)

    The ε clipping prevents log(0) = -∞
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class LogisticRegression:
    """
    Logistic Regression via Gradient Descent.

    No closed-form solution (unlike Linear Regression) because
    cross-entropy loss is nonlinear in w. Must use iterative optimization.
    """

    def __init__(self, lr=0.1, n_iters=1000, regularization=0.0):
        """
        Parameters:
        -----------
        lr : learning rate
        n_iters : gradient descent iterations
        regularization : L2 penalty (prevents overfitting, stabilizes)
        """
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        GRADIENT DESCENT FOR LOGISTIC REGRESSION

        The beauty: gradient has the same form as linear regression!
            dL/dw = (1/n) X'(p - y) + λw
            dL/db = (1/n) sum(p - y)

        where p = σ(Xw + b)

        The sigmoid's derivative σ(1-σ) cancels with cross-entropy's
        1/(p(1-p)) term, giving this clean gradient.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iters):
            # Forward: linear → sigmoid
            z = X @ self.weights + self.bias
            p = sigmoid(z)

            # Gradient: the (p - y) term is where all the magic is
            # This is dL/d(z), already simplified from chain rule
            error = p - y

            # Weight gradient = input × error (averaged over samples)
            dw = (1 / n_samples) * (X.T @ error) + self.regularization * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss
            loss = cross_entropy_loss(y, p)
            reg_loss = 0.5 * self.regularization * np.sum(self.weights ** 2)
            self.loss_history.append(loss + reg_loss)

        return self

    def predict_proba(self, X):
        """Return probability of class 1."""
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Return class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


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

    # Generate linearly separable data
    n = 200
    X0 = np.random.randn(n, 2) + np.array([-2, 0])
    X1 = np.random.randn(n, 2) + np.array([2, 0])
    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)
    idx = np.random.permutation(2*n)
    X, y = X[idx], y[idx]

    # Split
    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]

    # -------- Experiment 1: Sigmoid vs Linear Output --------
    print("\n1. SIGMOID vs NO SIGMOID (raw linear)")
    print("-" * 40)
    model = LogisticRegression(lr=0.1, n_iters=500)
    model.fit(X_train, y_train)

    # Raw scores (before sigmoid)
    z = X_test @ model.weights + model.bias
    p = sigmoid(z)

    print(f"Raw scores (z) range: [{z.min():.2f}, {z.max():.2f}]")
    print(f"Probabilities (p) range: [{p.min():.4f}, {p.max():.4f}]")
    print("→ Sigmoid squashes any real number to (0, 1)")

    # -------- Experiment 2: Cross-Entropy vs MSE --------
    print("\n2. CROSS-ENTROPY vs MSE LOSS")
    print("-" * 40)

    # Train with cross-entropy (standard)
    model_ce = LogisticRegression(lr=0.1, n_iters=500)
    model_ce.fit(X_train, y_train)
    acc_ce = accuracy(y_test, model_ce.predict(X_test))

    # Simulate MSE training (manually)
    # MSE gradient: dL/dw = (2/n) X' (p - y) p(1-p)
    # The p(1-p) term causes vanishing gradients
    weights_mse = np.zeros(2)
    bias_mse = 0.0
    lr_mse = 0.1
    for _ in range(500):
        z = X_train @ weights_mse + bias_mse
        p = sigmoid(z)
        error = p - y_train
        # MSE gradient has extra p(1-p) term
        gradient_scale = p * (1 - p)
        dw = (2 / len(y_train)) * (X_train.T @ (error * gradient_scale))
        db = (2 / len(y_train)) * np.sum(error * gradient_scale)
        weights_mse -= lr_mse * dw
        bias_mse -= lr_mse * db

    p_mse = sigmoid(X_test @ weights_mse + bias_mse)
    acc_mse = accuracy(y_test, (p_mse >= 0.5).astype(int))

    print(f"Cross-Entropy accuracy: {acc_ce:.3f}")
    print(f"MSE-style accuracy:     {acc_mse:.3f}")
    print("→ MSE can work but has vanishing gradient issues when p→0 or p→1")

    # -------- Experiment 3: Learning Rate --------
    print("\n3. LEARNING RATE SWEEP")
    print("-" * 40)
    for lr in [0.001, 0.01, 0.1, 1.0, 5.0]:
        model = LogisticRegression(lr=lr, n_iters=200)
        model.fit(X_train, y_train)
        final_loss = model.loss_history[-1]
        acc = accuracy(y_test, model.predict(X_test))
        print(f"lr={lr:<5} final_loss={final_loss:.4f}  acc={acc:.3f}")

    # -------- Experiment 4: Regularization --------
    print("\n4. REGULARIZATION EFFECT")
    print("-" * 40)
    for reg in [0, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(lr=0.1, n_iters=500, regularization=reg)
        model.fit(X_train, y_train)
        weight_norm = np.linalg.norm(model.weights)
        acc = accuracy(y_test, model.predict(X_test))
        print(f"λ={reg:<5} ||w||={weight_norm:.4f}  acc={acc:.3f}")
    print("→ Higher regularization = smaller weights, may hurt accuracy if too strong")

    # -------- Experiment 5: Decision Threshold --------
    print("\n5. DECISION THRESHOLD EFFECT")
    print("-" * 40)
    model = LogisticRegression(lr=0.1, n_iters=500)
    model.fit(X_train, y_train)

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y_pred = model.predict(X_test, threshold=threshold)
        acc = accuracy(y_test, y_pred)
        n_pos = np.sum(y_pred)
        print(f"threshold={threshold}  predicted_positives={n_pos:<3}  acc={acc:.3f}")
    print("→ Threshold moves the decision boundary along the probability axis")

    # -------- Experiment 6: Probability Calibration --------
    print("\n6. ARE PROBABILITIES CALIBRATED?")
    print("-" * 40)
    probs = model.predict_proba(X_test)

    # Bin predictions and check actual fraction
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("Prob range    Predicted    Actual")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if np.sum(mask) > 0:
            predicted_avg = probs[mask].mean()
            actual_avg = y_test[mask].mean()
            print(f"[{lo:.1f}, {hi:.1f})      {predicted_avg:.3f}        {actual_avg:.3f}")
    print("→ Well-calibrated = predicted prob ≈ actual fraction of positives")

    # -------- Experiment 7: Failure on Nonlinear --------
    print("\n7. FAILURE ON NONLINEAR DATA (XOR)")
    print("-" * 40)
    # Create XOR data
    n = 100
    X_xor = np.vstack([
        np.random.randn(n, 2) * 0.3 + np.array([0, 0]),
        np.random.randn(n, 2) * 0.3 + np.array([1, 1]),
        np.random.randn(n, 2) * 0.3 + np.array([0, 1]),
        np.random.randn(n, 2) * 0.3 + np.array([1, 0]),
    ])
    y_xor = np.array([0]*n + [0]*n + [1]*n + [1]*n)

    model_xor = LogisticRegression(lr=0.1, n_iters=1000)
    model_xor.fit(X_xor, y_xor)
    acc_xor = accuracy(y_xor, model_xor.predict(X_xor))
    print(f"XOR accuracy (training set): {acc_xor:.3f}")
    print("→ ~50% = random guessing. Logistic regression CANNOT learn XOR.")
    print("   The decision boundary must be a hyperplane, but XOR needs two lines.")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Logistic Regression on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Logistic Regression on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = LogisticRegression(lr=0.1, n_iters=1000)
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

        model = LogisticRegression(lr=0.1, n_iters=1000)
        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('LOGISTIC REGRESSION: Decision Boundaries\n'
                 '(Still linear! Sigmoid only changes output range, not boundary shape)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_with_linear():
    """Compare Logistic vs Linear Regression side by side."""
    print("\n" + "="*60)
    print("COMPARISON: Logistic vs Linear Regression")
    print("="*60)

    # Import Linear Regression
    from importlib import import_module
    linear_module = import_module('01_linear_regression')
    LinearRegressionClassifier = linear_module.LinearRegressionClassifier

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'Linear':<10} {'Logistic':<10} {'Diff':<10}")
    print("-" * 45)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:
            continue
        if name == 'clustered':
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        # Linear
        lin_model = LinearRegressionClassifier(method='normal')
        lin_model.fit(X_tr, y_tr)
        acc_lin = accuracy(y_te, lin_model.predict(X_te))

        # Logistic
        log_model = LogisticRegression(lr=0.1, n_iters=1000)
        log_model.fit(X_tr, y_tr)
        acc_log = accuracy(y_te, log_model.predict(X_te))

        diff = acc_log - acc_lin
        print(f"{name:<15} {acc_lin:<10.3f} {acc_log:<10.3f} {diff:+.3f}")

    print("\n→ Both have LINEAR decision boundaries, so accuracy is similar.")
    print("   Logistic is better because outputs are proper probabilities.")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("LOGISTIC REGRESSION — Paradigm: PROJECTION + SIGMOID")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Linear regression + sigmoid squashing
    Outputs probabilities, not raw numbers

THE KEY INSIGHT:
    Decision boundary is STILL a hyperplane!
    Sigmoid only changes the output range, not the boundary shape.

IMPROVEMENT OVER LINEAR REGRESSION:
    ✓ Outputs are probabilities in [0, 1]
    ✓ Cross-entropy loss has better gradients
    ✗ Still can't learn nonlinear boundaries

EXPECT SIMILAR FAILURES AS LINEAR REGRESSION:
    - circles, xor, spiral — all need curved boundaries
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Compare with linear
    compare_with_linear()

    # Visualize
    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/02_logistic_regression.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path}")
    plt.close(fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Logistic Regression Reveals")
    print("="*60)
    print("""
1. Sigmoid squashes linear output to probabilities
2. Decision boundary is STILL LINEAR (hyperplane)
3. Cross-entropy loss >> MSE for classification (better gradients)
4. Fails on exactly the same datasets as Linear Regression
5. The improvement is in WHAT it outputs, not WHAT it can learn

KEY TAKEAWAY:
    To learn nonlinear boundaries, you need a fundamentally
    different approach — not just a different output function.

NEXT: KNN (Memory paradigm) — no model at all, just memorize the data
    """)
