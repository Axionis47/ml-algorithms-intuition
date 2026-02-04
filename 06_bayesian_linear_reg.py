"""
BAYESIAN LINEAR REGRESSION — Paradigm: PROBABILISTIC (Linear + Uncertainty)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of finding ONE weight vector w, maintain a DISTRIBUTION over w.

Before data: p(w) = N(0, α⁻¹I)    (prior: weights are probably small)
After data:  p(w|X,y) = N(μ_N, Σ_N)  (posterior: updated beliefs)

At prediction time, you get:
    - Mean: E[y*] = μ_N' x*
    - Variance: Var[y*] = β⁻¹ + x*' Σ_N x*

The variance has TWO parts:
    1. β⁻¹: irreducible noise (data is inherently noisy)
    2. x*' Σ_N x*: epistemic uncertainty (we're unsure about w)

===============================================================
THE KEY EQUATIONS
===============================================================

POSTERIOR (closed form, conjugate Gaussian):
    Σ_N = (αI + βX'X)⁻¹
    μ_N = βΣ_N X'y

where:
    α = prior precision (larger = stronger prior, smaller weights)
    β = noise precision (larger = trust data more)

CONNECTION TO REGULAR LINEAR REGRESSION:
    As α → 0: Bayesian → OLS (no regularization)
    As β → ∞: posterior collapses to point estimate (no uncertainty)

CONNECTION TO RIDGE REGRESSION:
    The MAP (mode of posterior) equals Ridge solution with λ = α/β

CONNECTION TO GAUSSIAN PROCESSES:
    Bayesian linear regression IS a GP with linear kernel!
    GP generalizes this to any kernel (nonlinear).

===============================================================
INDUCTIVE BIAS
===============================================================

1. Weights are Gaussian-distributed (prior belief)
2. Noise is Gaussian and homoscedastic
3. Relationship is LINEAR (like regular linear regression)
4. Prior pulls weights toward zero (regularization)

WHAT IT CAN DO:
    ✓ Uncertainty quantification
    ✓ Natural regularization (no hyperparameter tuning)
    ✓ Works with small data (prior provides information)
    ✓ Principled way to incorporate prior knowledge

WHAT IT CAN'T DO:
    ✗ Nonlinear relationships (same as regular linear regression)
    ✗ Non-Gaussian noise

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


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate Gaussian prior.

    MATH:
        Prior:     p(w) = N(0, α⁻¹I)
        Likelihood: p(y|X,w) = N(Xw, β⁻¹I)
        Posterior: p(w|X,y) = N(μ_N, Σ_N)

    where:
        Σ_N = (αI + βX'X)⁻¹
        μ_N = βΣ_N X'y
    """

    def __init__(self, alpha=1.0, beta=25.0, fit_intercept=True):
        """
        Parameters:
        -----------
        alpha : Prior precision (larger = stronger prior, smaller weights)
        beta : Noise precision = 1/σ² (larger = trust data more)
        fit_intercept : Whether to fit bias term
        """
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.mean = None      # μ_N: posterior mean
        self.cov = None       # Σ_N: posterior covariance
        self.n_features = None

    def fit(self, X, y):
        """
        Compute posterior distribution over weights.

        CLOSED FORM (conjugate prior magic):
            Σ_N = (αI + βX'X)⁻¹
            μ_N = βΣ_N X'y
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Prior precision matrix
        S_0_inv = self.alpha * np.eye(n_features)

        # Posterior precision: S_N⁻¹ = S_0⁻¹ + βX'X
        S_N_inv = S_0_inv + self.beta * (X.T @ X)

        # Posterior covariance: Σ_N = (S_N⁻¹)⁻¹
        self.cov = np.linalg.inv(S_N_inv)

        # Posterior mean: μ_N = βΣ_N X'y
        self.mean = self.beta * (self.cov @ X.T @ y)

        return self

    def predict(self, X, return_std=False):
        """
        Predict with posterior mean, optionally return uncertainty.

        PREDICTIVE DISTRIBUTION:
            p(y*|x*, X, y) = N(y* | μ_N'x*, σ²(x*))

        where:
            mean = μ_N' x*
            σ²(x*) = β⁻¹ + x*' Σ_N x*

        The uncertainty has two parts:
            - β⁻¹: irreducible noise
            - x*' Σ_N x*: epistemic (model) uncertainty
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Mean prediction
        mean = X @ self.mean

        if return_std:
            # Predictive variance for each point
            # σ²(x*) = β⁻¹ + x*' Σ_N x*
            var = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                x = X[i]
                # Epistemic uncertainty: x' Σ_N x
                epistemic = x @ self.cov @ x
                # Total variance = noise + epistemic
                var[i] = 1/self.beta + epistemic

            std = np.sqrt(var)
            return mean, std

        return mean

    def sample_weights(self, n_samples=10):
        """
        Sample weight vectors from the posterior.

        This lets you visualize the "family of lines" the model believes in.
        """
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)


class BayesianLinearClassifier:
    """
    Classification via Bayesian Linear Regression.

    Simple approach: regression on {0, 1} labels, threshold at 0.5.
    For proper Bayesian classification, would need Laplace approximation
    or variational inference.
    """

    def __init__(self, alpha=1.0, beta=25.0):
        self.model = BayesianLinearRegression(alpha=alpha, beta=beta)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return (pred > 0.5).astype(int)

    def predict_proba(self, X):
        mean, std = self.model.predict(X, return_std=True)
        # Clip to [0, 1] (not proper probabilities, but reasonable approximation)
        return np.clip(mean, 0, 1)


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

    # Generate 1D data
    n = 20
    X_train = np.random.uniform(-1, 1, n).reshape(-1, 1)
    y_train = 2 * X_train.squeeze() + 1 + np.random.randn(n) * 0.3
    X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
    y_true = 2 * X_test.squeeze() + 1

    # -------- Experiment 1: Effect of Alpha (Prior Precision) --------
    print("\n1. EFFECT OF ALPHA (Prior Precision)")
    print("-" * 40)
    print("Higher α = stronger prior = smaller weights = more regularization")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for idx, alpha in enumerate(alphas):
        ax = axes[idx // 3, idx % 3]

        model = BayesianLinearRegression(alpha=alpha, beta=25.0)
        model.fit(X_train, y_train)
        mean, std = model.predict(X_test, return_std=True)

        # Sample some lines from posterior
        weights = model.sample_weights(10)
        X_test_bias = np.c_[np.ones(100), X_test]
        for w in weights:
            y_sample = X_test_bias @ w
            ax.plot(X_test, y_sample, 'b-', alpha=0.2, linewidth=0.5)

        ax.fill_between(X_test.squeeze(), mean - 2*std, mean + 2*std,
                       alpha=0.3, color='blue', label='±2σ')
        ax.plot(X_test, mean, 'b-', linewidth=2, label='Mean')
        ax.scatter(X_train, y_train, c='red', s=30, zorder=5)
        ax.set_title(f'α = {alpha}')
        ax.set_ylim(-3, 5)

        mse = np.mean((mean - y_true) ** 2)
        print(f"α={alpha:<8}  weight_norm={np.linalg.norm(model.mean):.4f}  MSE={mse:.4f}")

    plt.suptitle('Bayesian Linear Regression: Effect of Prior Precision (α)\n'
                 'Blue lines = samples from posterior. Higher α = smaller weights.',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig('/Users/sid47/ML Algorithms/06_blr_alpha.png', dpi=100)
    plt.close(fig)
    print("Saved alpha experiment to 06_blr_alpha.png")

    # -------- Experiment 2: Effect of Beta (Noise Precision) --------
    print("\n2. EFFECT OF BETA (Noise Precision)")
    print("-" * 40)
    print("Higher β = trust data more = tighter fit = narrower uncertainty")

    for beta in [1.0, 10.0, 25.0, 100.0]:
        model = BayesianLinearRegression(alpha=1.0, beta=beta)
        model.fit(X_train, y_train)
        mean, std = model.predict(X_test, return_std=True)
        print(f"β={beta:<6}  avg_std={std.mean():.4f}  MSE={np.mean((mean-y_true)**2):.4f}")
    print("→ Higher β = smaller uncertainty")

    # -------- Experiment 3: Uncertainty Away from Data --------
    print("\n3. UNCERTAINTY GROWS AWAY FROM DATA")
    print("-" * 40)

    model = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.fit(X_train, y_train)

    # Check uncertainty at different distances from data
    test_points = np.array([[-1.5], [0.0], [1.5]])
    means, stds = model.predict(test_points, return_std=True)

    print("Distance from data region vs Uncertainty:")
    for i, x in enumerate(test_points.squeeze()):
        in_data = -1 <= x <= 1
        print(f"  x={x:>5.1f}  {'(in data)  ' if in_data else '(outside)  '}  std={stds[i]:.4f}")
    print("→ Uncertainty is higher outside the training data range!")

    # -------- Experiment 4: Connection to Ridge Regression --------
    print("\n4. CONNECTION TO RIDGE REGRESSION")
    print("-" * 40)

    # Ridge regression: minimize ||y - Xw||² + λ||w||²
    # Solution: w = (X'X + λI)⁻¹ X'y
    # BLR MAP (mode): w = β(αI + βX'X)⁻¹ X'y = (α/β I + X'X)⁻¹ X'y
    # So BLR MAP = Ridge with λ = α/β

    alpha, beta = 2.5, 25.0
    lambda_ridge = alpha / beta

    # Bayesian
    model = BayesianLinearRegression(alpha=alpha, beta=beta)
    model.fit(X_train, y_train)
    w_bayesian = model.mean[1:]  # Exclude bias

    # Ridge (manual computation)
    X_b = np.c_[np.ones(n), X_train]
    ridge_reg = lambda_ridge * np.eye(2)
    ridge_reg[0, 0] = 0  # Don't regularize bias
    w_ridge = np.linalg.solve(X_b.T @ X_b + ridge_reg, X_b.T @ y_train)[1:]

    print(f"λ = α/β = {lambda_ridge}")
    print(f"Bayesian MAP weight: {w_bayesian[0]:.4f}")
    print(f"Ridge weight:        {w_ridge[0]:.4f}")
    print("→ MAP estimate of BLR = Ridge solution!")

    # -------- Experiment 5: Small Data vs Large Data --------
    print("\n5. SMALL vs LARGE DATA (Prior Influence)")
    print("-" * 40)

    for n_data in [5, 20, 100, 500]:
        X_n = np.random.uniform(-1, 1, n_data).reshape(-1, 1)
        y_n = 2 * X_n.squeeze() + 1 + np.random.randn(n_data) * 0.3

        model = BayesianLinearRegression(alpha=1.0, beta=25.0)
        model.fit(X_n, y_n)

        # True weight is 2
        w_learned = model.mean[1]
        w_prior = 0  # Prior mean is 0
        print(f"n={n_data:<4}  learned_w={w_learned:.4f}  (true=2.0, prior=0.0)")
    print("→ With more data, posterior shifts from prior toward MLE")

    # -------- Experiment 6: Posterior Visualization --------
    print("\n6. POSTERIOR DISTRIBUTION OVER WEIGHTS")
    print("-" * 40)

    model = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.fit(X_train, y_train)

    print(f"Posterior mean: w0={model.mean[0]:.3f}, w1={model.mean[1]:.3f}")
    print(f"Posterior std:  σ0={np.sqrt(model.cov[0,0]):.3f}, σ1={np.sqrt(model.cov[1,1]):.3f}")
    print(f"Correlation:    ρ={model.cov[0,1]/np.sqrt(model.cov[0,0]*model.cov[1,1]):.3f}")
    print("→ We have a full distribution, not just point estimates!")


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Bayesian Linear Classifier on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Bayesian Linear on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = BayesianLinearClassifier(alpha=1.0, beta=25.0)
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

        model = BayesianLinearClassifier(alpha=1.0, beta=25.0)
        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('BAYESIAN LINEAR REGRESSION: Decision Boundaries\n'
                 '(Linear boundary + uncertainty quantification)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("BAYESIAN LINEAR REGRESSION — Linear with Uncertainty")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Linear regression with a DISTRIBUTION over weights instead of a point estimate.
    Get uncertainty for free!

THE KEY INSIGHT:
    - Prior p(w) encodes regularization (α)
    - Likelihood encodes data fit (β)
    - Posterior balances both

CONNECTIONS:
    - MAP estimate = Ridge regression
    - Gaussian Process with linear kernel = Bayesian linear regression
    - As data → ∞, posterior → MLE

STRENGTHS:
    ✓ Uncertainty quantification
    ✓ Natural regularization
    ✓ Principled prior incorporation

WEAKNESSES:
    ✗ Still LINEAR (same as regular linear regression)
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize
    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/06_bayesian_linear_reg.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path}")
    plt.close(fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Bayesian Linear Regression Reveals")
    print("="*60)
    print("""
1. You can have DISTRIBUTIONS over parameters, not just point estimates
2. Uncertainty grows outside the training data region
3. Prior precision α acts like Ridge regularization
4. More data → posterior shifts from prior toward MLE
5. Still limited to LINEAR decision boundaries

KEY INSIGHT:
    Bayesian Linear Regression is the simplest example of
    "putting distributions on things". GP generalizes this
    to distributions over FUNCTIONS.

NEXT: SVM — Maximum margin classification
    """)
